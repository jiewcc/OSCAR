"""Experimental accumulator-resident-softmax INT2 decode for Hopper.

This file is deliberately not wired into the attention dispatch.  It keeps
the gather + 16-byte packed-load path from
``flashinfer_cutedsl_int2_decode`` but changes the two GEMMs to

    K[64, 128] @ Q.T[128, 8] -> S.T[64, 8]
    V.T[128, 64] @ P.T[64, 8] -> O.T[128, 8]

so MHA/GQA decode no longer pays for a 64-row query tile.  The group-size-1
or group-size-4 query rows occupy the leading columns of the n8 WGMMA
tile.  P is staged through shared memory because Hopper WGMMA only supports a
register-sourced A operand, while the transposed PV formulation needs P as B.

Unlike the shared-score transposed prototype, the m64n8 QK accumulator stays
in registers through max/exp/sum.  Each warp reduces its 16-token fragment in
the eight-lane groups selected by ``lane % 4``; two tiny 4-warp by 8-query
shared arrays combine the four warp partials.  This avoids the full 64x8 FP32
score shared-memory round trip while preserving the same four CTA barriers per
KV tile and the same stage-1 scratch contract.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch

from .flashinfer_cutedsl_int2_decode import (
    MIN_BLOCK_KV,
    _cached_from_dlpack,
    _index_cutlass_dtype,
    _layout_acc_mn,
    can_use_flashinfer_cutedsl_decode,
)

__all__ = [
    "can_use_flashinfer_cutedsl_decode_regsoftmax",
    "flashinfer_cutedsl_decode_attention_fwd_int2_regsoftmax",
]


DECODE_BLOCK_N = 64
_WGMMA_N = 8
_LOG2_E = 1.4426950408889634

_LOADER_THREADS = 128
_NUM_THREADS = 128

_compiled_decode_kernels: Dict[Tuple, object] = {}


@cute.jit
def _gemm_zero_acc(
    tiled_mma, operand_a, operand_b, accumulator
):
    # CuTe 4.5.2's high-level GEMM verifier rejects exact n8 fragments.  The
    # low-level atom call accepts the native rank-1 per-thread fragments.
    atom = cute.make_mma_atom(tiled_mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, False)
    accumulator_atom = accumulator[None, 0, 0]
    for k_block in cutlass.range_constexpr(cute.size(operand_a.shape[2])):
        cute.mma_atom_call(
            atom,
            accumulator_atom,
            operand_a[None, 0, k_block],
            operand_b[None, 0, k_block],
            accumulator_atom,
        )
        atom.set(warpgroup.Field.ACCUMULATE, True)


@cute.jit
def _gemm_accumulate(tiled_mma, operand_a, operand_b, accumulator):
    atom = cute.make_mma_atom(tiled_mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, True)
    accumulator_atom = accumulator[None, 0, 0]
    for k_block in cutlass.range_constexpr(cute.size(operand_a.shape[2])):
        cute.mma_atom_call(
            atom,
            accumulator_atom,
            operand_a[None, 0, k_block],
            operand_b[None, 0, k_block],
            accumulator_atom,
        )


@cute.jit
def _subgroup_max8(value: cutlass.Float32) -> cutlass.Float32:
    """Reduce lanes with the same ``lane % 4`` inside one warp."""

    for stage in cutlass.range_constexpr(3):
        shift: cutlass.Constexpr[int] = 1 << (stage + 2)
        other = cute.arch.shuffle_sync_bfly(
            value, offset=shift, mask=-1, mask_and_clamp=31
        )
        value = cute.arch.fmax(value, other)
    return value


@cute.jit
def _subgroup_sum8(value: cutlass.Float32) -> cutlass.Float32:
    """Sum lanes with the same ``lane % 4`` inside one warp."""

    for stage in cutlass.range_constexpr(3):
        shift: cutlass.Constexpr[int] = 1 << (stage + 2)
        value = value + cute.arch.shuffle_sync_bfly(
            value, offset=shift, mask=-1, mask_and_clamp=31
        )
    return value


def _define_decode_kernel(
    head_dim: cutlass.Constexpr[int],
    block_n: cutlass.Constexpr[int],
    kv_group_num: cutlass.Constexpr[int],
):
    quarter_dim: cutlass.Constexpr[int] = head_dim // 4
    packed_bytes_per_tile: cutlass.Constexpr[int] = block_n * quarter_dim
    packed_bytes_per_loader: cutlass.Constexpr[int] = (
        packed_bytes_per_tile + _LOADER_THREADS - 1
    ) // _LOADER_THREADS
    loader_lanes_per_token: cutlass.Constexpr[int] = _LOADER_THREADS // block_n

    @cute.kernel
    def kernel(
        Q: cute.Tensor,
        K_packed: cute.Tensor,
        V_packed: cute.Tensor,
        K_sz: cute.Tensor,
        V_sz: cute.Tensor,
        kv_indptr: cute.Tensor,
        kv_indices: cute.Tensor,
        num_kv_splits: cute.Tensor,
        att_out: cute.Tensor,
        att_lse: cute.Tensor,
        sm_scale: cutlass.Float32,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        k_smem_layout_staged: cute.ComposedLayout,
        q_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        min_block_kv: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cur_batch, cur_kv_head, split_kv_id = cute.arch.block_idx()

        loader_tid = tidx
        lane_idx = tidx % 32
        warp_idx = tidx // 32
        q_head_base = cur_kv_head * kv_group_num

        kv_start_idx = cutlass.Int32(kv_indptr[cur_batch])
        seq_len = cutlass.Int32(kv_indptr[cur_batch + 1]) - kv_start_idx
        kv_splits = cutlass.Int32(num_kv_splits[cur_batch])
        partition_splits = max(kv_splits, cutlass.Int32(1))
        kv_len_per_split = (
            ((seq_len + partition_splits - 1) // partition_splits + min_block_kv - 1)
            // min_block_kv
            * min_block_kv
        )
        split_start = kv_len_per_split * split_kv_id
        split_stop = min(seq_len, split_start + kv_len_per_split)

        smem = cutlass.utils.SmemAllocator()
        sK_mma = smem.allocate_tensor(
            cutlass.BFloat16,
            k_smem_layout_staged.outer,
            1024,
            swizzle=k_smem_layout_staged.inner,
        )
        sQ_mma = smem.allocate_tensor(
            cutlass.BFloat16,
            q_smem_layout_staged.outer,
            1024,
            swizzle=q_smem_layout_staged.inner,
        )
        sV_lo_mma = smem.allocate_tensor(
            cutlass.BFloat16,
            v_smem_layout_staged.outer,
            1024,
            swizzle=v_smem_layout_staged.inner,
        )
        sV_hi_mma = smem.allocate_tensor(
            cutlass.BFloat16,
            v_smem_layout_staged.outer,
            1024,
            swizzle=v_smem_layout_staged.inner,
        )
        sP_mma = smem.allocate_tensor(
            cutlass.BFloat16,
            p_smem_layout_staged.outer,
            1024,
            swizzle=p_smem_layout_staged.inner,
        )
        # One partial per (warp, query) is enough to combine the four m64
        # token fragments.  Scores themselves remain in the QK accumulator.
        sPartialMax = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((4, _WGMMA_N), stride=(_WGMMA_N, 1)),
            16,
        )
        sPartialSum = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((4, _WGMMA_N), stride=(_WGMMA_N, 1)),
            16,
        )
        sMax = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((_WGMMA_N,)), 16
        )
        sSum = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((_WGMMA_N,)), 16
        )

        qk_thr_mma = qk_tiled_mma.get_slice(tidx)
        pv_thr_mma = pv_tiled_mma.get_slice(tidx)

        tKsK = qk_thr_mma.partition_A(sK_mma)
        tQsQ = qk_thr_mma.partition_B(sQ_mma)
        tKrK = qk_thr_mma.make_fragment_A(tKsK)
        tQrQ = qk_thr_mma.make_fragment_B(tQsQ)
        tVlosV = pv_thr_mma.partition_A(sV_lo_mma)
        tVhisV = pv_thr_mma.partition_A(sV_hi_mma)
        tPsP = pv_thr_mma.partition_B(sP_mma)
        tVlorV = pv_thr_mma.make_fragment_A(tVlosV)
        tVhirV = pv_thr_mma.make_fragment_A(tVhisV)
        tPrP = pv_thr_mma.make_fragment_B(tPsP)

        qk_acc_shape = qk_thr_mma.partition_shape_C((block_n, _WGMMA_N))
        pv_acc_shape = pv_thr_mma.partition_shape_C((64, _WGMMA_N))
        acc_pv_lo = pv_thr_mma.make_fragment_C(pv_acc_shape)
        acc_pv_hi = pv_thr_mma.make_fragment_C(pv_acc_shape)
        acc_pv_lo.fill(cutlass.Float32(0.0))
        acc_pv_hi.fill(cutlass.Float32(0.0))

        cQK = cute.make_identity_tensor((block_n, _WGMMA_N))
        tQcQK = qk_thr_mma.partition_C(cQK)
        tQcQK_mn = cute.make_tensor(
            tQcQK.iterator, _layout_acc_mn(qk_tiled_mma, tQcQK.layout)
        )
        cPV = cute.make_identity_tensor((64, _WGMMA_N))
        tPcPV = pv_thr_mma.partition_C(cPV)
        tPcPV_mn = cute.make_tensor(
            tPcPV.iterator, _layout_acc_mn(pv_tiled_mma, tPcPV.layout)
        )

        # Operand B is logically (query-column, K).  Fill all eight columns;
        # the unused columns stay zero for MHA/GQA4.
        if tidx < head_dim:
            for qid in cutlass.range_constexpr(_WGMMA_N):
                if cutlass.const_expr(qid < kv_group_num):
                    sQ_mma[qid, tidx, 0] = Q[
                        cur_batch, q_head_base + qid, tidx
                    ]
                else:
                    sQ_mma[qid, tidx, 0] = cutlass.BFloat16(0.0)
        if tidx < _WGMMA_N:
            sMax[tidx] = -cutlass.Float32.inf
            sSum[tidx] = cutlass.Float32(0.0)
        # The unused n8 columns are never touched by a softmax warp.  Clear P
        # once so their PV contribution remains zero for every tile.
        for item in cutlass.range_constexpr(
            _WGMMA_N * block_n // _NUM_THREADS
        ):
            linear = tidx + item * _NUM_THREADS
            qid = linear // block_n
            token = linear % block_n
            sP_mma[qid, token, 0] = cutlass.BFloat16(0.0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        tile_start = split_start
        while tile_start < split_stop:
            # Same metadata-hoisted, barrier-free gather and 16-byte vector
            # load as the optimized non-transposed kernel.
            if loader_tid < _LOADER_THREADS:
                row = loader_tid // loader_lanes_per_token
                d4_base = (
                    loader_tid % loader_lanes_per_token
                ) * packed_bytes_per_loader
                logical_kv = tile_start + row
                valid_token = logical_kv < split_stop
                kv_pos = cutlass.Int32(0)
                ks = cutlass.Float32(0.0)
                kz = cutlass.Float32(0.0)
                vs = cutlass.Float32(0.0)
                vz = cutlass.Float32(0.0)
                if valid_token:
                    kv_pos = cutlass.Int32(kv_indices[kv_start_idx + logical_kv])
                    ks = cutlass.Float32(K_sz[kv_pos, cur_kv_head, 0])
                    kz = cutlass.Float32(K_sz[kv_pos, cur_kv_head, 1])
                    vs = cutlass.Float32(V_sz[kv_pos, cur_kv_head, 0])
                    vz = cutlass.Float32(V_sz[kv_pos, cur_kv_head, 1])

                gK_packed = K_packed[kv_pos, cur_kv_head, None]
                gV_packed = V_packed[kv_pos, cur_kv_head, None]
                packed_lane = loader_tid % loader_lanes_per_token
                gK_lane = cute.local_tile(
                    gK_packed, (packed_bytes_per_loader,), (packed_lane,)
                )
                gV_lane = cute.local_tile(
                    gV_packed, (packed_bytes_per_loader,), (packed_lane,)
                )
                rK_packed = cute.make_rmem_tensor(
                    (packed_bytes_per_loader,), cutlass.Uint8
                )
                rV_packed = cute.make_rmem_tensor(
                    (packed_bytes_per_loader,), cutlass.Uint8
                )
                rK_packed.fill(cutlass.Uint8(0))
                rV_packed.fill(cutlass.Uint8(0))
                if valid_token:
                    cute.autovec_copy(gK_lane, rK_packed)
                    cute.autovec_copy(gV_lane, rV_packed)

                for j in cutlass.range_constexpr(packed_bytes_per_loader):
                    d4 = d4_base + j
                    kp_i = cutlass.Int32(rK_packed[j])
                    vp_i = cutlass.Int32(rV_packed[j])

                    k0 = (cutlass.Float32(kp_i & 0x03) - kz) * ks
                    k1 = (cutlass.Float32((kp_i >> 2) & 0x03) - kz) * ks
                    k2 = (cutlass.Float32((kp_i >> 4) & 0x03) - kz) * ks
                    k3 = (cutlass.Float32((kp_i >> 6) & 0x03) - kz) * ks
                    v0 = (cutlass.Float32(vp_i & 0x03) - vz) * vs
                    v1 = (cutlass.Float32((vp_i >> 2) & 0x03) - vz) * vs
                    v2 = (cutlass.Float32((vp_i >> 4) & 0x03) - vz) * vs
                    v3 = (cutlass.Float32((vp_i >> 6) & 0x03) - vz) * vs

                    sK_mma[row, d4, 0] = cutlass.BFloat16(k0)
                    sK_mma[row, d4 + quarter_dim, 0] = cutlass.BFloat16(k1)
                    sK_mma[row, d4 + 2 * quarter_dim, 0] = cutlass.BFloat16(k2)
                    sK_mma[row, d4 + 3 * quarter_dim, 0] = cutlass.BFloat16(k3)
                    sV_lo_mma[d4, row, 0] = cutlass.BFloat16(v0)
                    sV_lo_mma[d4 + quarter_dim, row, 0] = cutlass.BFloat16(v1)
                    sV_hi_mma[d4, row, 0] = cutlass.BFloat16(v2)
                    sV_hi_mma[d4 + quarter_dim, row, 0] = cutlass.BFloat16(v3)
                cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()

            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
            acc_qk.fill(cutlass.Float32(0.0))
            cute.nvgpu.warpgroup.fence()
            _gemm_zero_acc(
                qk_tiled_mma,
                tKrK[(None, None, None, 0)],
                tQrQ[(None, None, None, 0)],
                acc_qk,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            acc_qk_mn = cute.make_tensor(
                acc_qk.iterator, _layout_acc_mn(qk_tiled_mma, acc_qk.layout)
            )
            valid_tokens = split_stop - tile_start
            for row in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[0])):
                for col in cutlass.range_constexpr(
                    cute.size(acc_qk_mn, mode=[1])
                ):
                    if tQcQK_mn[row, col][0] >= valid_tokens:
                        acc_qk_mn[row, col] = -cutlass.Float32.inf

            # Native m64n8 C mapping gives each thread two token rows and two
            # adjacent query columns.  Lanes with equal lane%4 jointly own all
            # 16 tokens for one query in this warp.
            token00 = tQcQK_mn[0, 0][0]
            token10 = tQcQK_mn[1, 0][0]
            token01 = tQcQK_mn[0, 1][0]
            token11 = tQcQK_mn[1, 1][0]
            qid0 = tQcQK_mn[0, 0][1]
            qid1 = tQcQK_mn[0, 1][1]
            score00 = cutlass.Float32(acc_qk_mn[0, 0])
            score10 = cutlass.Float32(acc_qk_mn[1, 0])
            score01 = cutlass.Float32(acc_qk_mn[0, 1])
            score11 = cutlass.Float32(acc_qk_mn[1, 1])

            warp_max0 = _subgroup_max8(cute.arch.fmax(score00, score10))
            warp_max1 = _subgroup_max8(cute.arch.fmax(score01, score11))
            if lane_idx < 4:
                sPartialMax[warp_idx, qid0] = warp_max0
                sPartialMax[warp_idx, qid1] = warp_max1
            cute.arch.barrier()

            tile_max0 = -cutlass.Float32.inf
            tile_max1 = -cutlass.Float32.inf
            for source_warp in cutlass.range_constexpr(4):
                tile_max0 = cute.arch.fmax(
                    tile_max0,
                    cutlass.Float32(sPartialMax[source_warp, qid0]),
                )
                tile_max1 = cute.arch.fmax(
                    tile_max1,
                    cutlass.Float32(sPartialMax[source_warp, qid1]),
                )
            previous_max0 = cutlass.Float32(sMax[qid0])
            previous_max1 = cutlass.Float32(sMax[qid1])
            current_max0 = cute.arch.fmax(previous_max0, tile_max0)
            current_max1 = cute.arch.fmax(previous_max1, tile_max1)
            finite_max0 = current_max0
            finite_max1 = current_max1
            if finite_max0 == -cutlass.Float32.inf:
                finite_max0 = cutlass.Float32(0.0)
            if finite_max1 == -cutlass.Float32.inf:
                finite_max1 = cutlass.Float32(0.0)

            scale_log2 = cutlass.Float32(_LOG2_E) * sm_scale
            p00 = cute.math.exp2(
                (score00 - finite_max0) * scale_log2, fastmath=True
            )
            p10 = cute.math.exp2(
                (score10 - finite_max0) * scale_log2, fastmath=True
            )
            p01 = cute.math.exp2(
                (score01 - finite_max1) * scale_log2, fastmath=True
            )
            p11 = cute.math.exp2(
                (score11 - finite_max1) * scale_log2, fastmath=True
            )
            warp_sum0 = _subgroup_sum8(p00 + p10)
            warp_sum1 = _subgroup_sum8(p01 + p11)
            if lane_idx < 4:
                sPartialSum[warp_idx, qid0] = warp_sum0
                sPartialSum[warp_idx, qid1] = warp_sum1

            # Identity coordinates put the register probabilities directly in
            # the WGMMA-B shared layout; no full FP32 score tensor is needed.
            if qid0 < kv_group_num:
                sP_mma[qid0, token00, 0] = cutlass.BFloat16(p00)
                sP_mma[qid0, token10, 0] = cutlass.BFloat16(p10)
            if qid1 < kv_group_num:
                sP_mma[qid1, token01, 0] = cutlass.BFloat16(p01)
                sP_mma[qid1, token11, 0] = cutlass.BFloat16(p11)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()

            tile_sum0 = cutlass.Float32(0.0)
            tile_sum1 = cutlass.Float32(0.0)
            for source_warp in cutlass.range_constexpr(4):
                tile_sum0 = tile_sum0 + cutlass.Float32(
                    sPartialSum[source_warp, qid0]
                )
                tile_sum1 = tile_sum1 + cutlass.Float32(
                    sPartialSum[source_warp, qid1]
                )
            correction0 = cute.math.exp2(
                (previous_max0 - finite_max0) * scale_log2, fastmath=True
            )
            correction1 = cute.math.exp2(
                (previous_max1 - finite_max1) * scale_log2, fastmath=True
            )
            if warp_idx == 0 and lane_idx < 4:
                if qid0 < kv_group_num:
                    sMax[qid0] = current_max0
                    sSum[qid0] = (
                        cutlass.Float32(sSum[qid0]) * correction0 + tile_sum0
                    )
                if qid1 < kv_group_num:
                    sMax[qid1] = current_max1
                    sSum[qid1] = (
                        cutlass.Float32(sSum[qid1]) * correction1 + tile_sum1
                    )

            # Both WGMMA C layouts have the same n8 value mapping, so the two
            # register corrections map directly to the two PV query columns.
            acc_pv_lo_mn = cute.make_tensor(
                acc_pv_lo.iterator,
                _layout_acc_mn(pv_tiled_mma, acc_pv_lo.layout),
            )
            acc_pv_hi_mn = cute.make_tensor(
                acc_pv_hi.iterator,
                _layout_acc_mn(pv_tiled_mma, acc_pv_hi.layout),
            )
            for row in cutlass.range_constexpr(cute.size(acc_pv_lo_mn, mode=[0])):
                acc_pv_lo_mn[row, 0] = acc_pv_lo_mn[row, 0] * correction0
                acc_pv_hi_mn[row, 0] = acc_pv_hi_mn[row, 0] * correction0
                acc_pv_lo_mn[row, 1] = acc_pv_lo_mn[row, 1] * correction1
                acc_pv_hi_mn[row, 1] = acc_pv_hi_mn[row, 1] * correction1

            if tidx < _NUM_THREADS:
                cute.nvgpu.warpgroup.fence()
                _gemm_accumulate(
                    pv_tiled_mma,
                    tVlorV[(None, None, None, 0)],
                    tPrP[(None, None, None, 0)],
                    acc_pv_lo,
                )
                _gemm_accumulate(
                    pv_tiled_mma,
                    tVhirV[(None, None, None, 0)],
                    tPrP[(None, None, None, 0)],
                    acc_pv_hi,
                )
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

            cute.arch.barrier()
            tile_start = tile_start + block_n

        if split_kv_id < kv_splits:
            acc_pv_lo_mn = cute.make_tensor(
                acc_pv_lo.iterator,
                _layout_acc_mn(pv_tiled_mma, acc_pv_lo.layout),
            )
            acc_pv_hi_mn = cute.make_tensor(
                acc_pv_hi.iterator,
                _layout_acc_mn(pv_tiled_mma, acc_pv_hi.layout),
            )
            cO = cute.make_identity_tensor((64, _WGMMA_N))
            tOcO = pv_thr_mma.partition_C(cO)
            tOcO_mn = cute.make_tensor(
                tOcO.iterator, _layout_acc_mn(pv_tiled_mma, tOcO.layout)
            )
            for row in cutlass.range_constexpr(cute.size(acc_pv_lo_mn, mode=[0])):
                for col in cutlass.range_constexpr(
                    cute.size(acc_pv_lo_mn, mode=[1])
                ):
                    out_dim = tOcO_mn[row, col][0]
                    qid = tOcO_mn[row, col][1]
                    if qid < kv_group_num:
                        total = cutlass.Float32(sSum[qid])
                        inv_total = cutlass.Float32(0.0)
                        lse = -cutlass.Float32.inf
                        if total > cutlass.Float32(0.0):
                            inv_total = cutlass.Float32(1.0) / total
                            lse = cutlass.Float32(sMax[qid]) * sm_scale + cute.log(
                                total
                            )
                        att_out[
                            cur_batch,
                            q_head_base + qid,
                            split_kv_id,
                            out_dim,
                        ] = acc_pv_lo_mn[row, col] * inv_total
                        att_out[
                            cur_batch,
                            q_head_base + qid,
                            split_kv_id,
                            out_dim + 64,
                        ] = acc_pv_hi_mn[row, col] * inv_total
                        if out_dim == 0:
                            att_lse[
                                cur_batch, q_head_base + qid, split_kv_id
                            ] = lse

    return kernel


def _compile_decode(
    *,
    head_dim: int,
    block_n: int,
    kv_group_num: int,
    batch: int,
    q_heads: int,
    kv_heads: int,
    max_splits: int,
    cache_size: int,
    kv_indices_dtype: torch.dtype,
    att_out_shape: tuple,
    att_out_stride: tuple,
    att_lse_shape: tuple,
    att_lse_stride: tuple,
):
    key = (
        head_dim,
        block_n,
        kv_group_num,
        batch,
        q_heads,
        kv_heads,
        max_splits,
        cache_size,
        kv_indices_dtype,
        att_out_shape,
        att_out_stride,
        att_lse_shape,
        att_lse_stride,
    )
    if key in _compiled_decode_kernels:
        return _compiled_decode_kernels[key]

    kernel = _define_decode_kernel(head_dim, block_n, kv_group_num)
    smem_bytes = (
        block_n * head_dim * 2
        + _WGMMA_N * head_dim * 2
        + head_dim * block_n * 2
        + _WGMMA_N * block_n * 2
        + 4 * _WGMMA_N * 2 * 4
        + _WGMMA_N * 2 * 4
        + 8192
    )

    @cute.jit
    def launcher(
        Q: cute.Tensor,
        K_packed: cute.Tensor,
        V_packed: cute.Tensor,
        K_sz: cute.Tensor,
        V_sz: cute.Tensor,
        kv_indptr: cute.Tensor,
        kv_indices: cute.Tensor,
        num_kv_splits: cute.Tensor,
        att_out: cute.Tensor,
        att_lse: cute.Tensor,
        sm_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        row_major = cutlass_utils.LayoutEnum.ROW_MAJOR
        qk_tiler = (block_n, _WGMMA_N, head_dim)
        # Hopper's WGMMA atom has a fixed m64 shape.  Two m64n8 PV
        # accumulators cover the 128 output dimensions.
        pv_tiler = (64, _WGMMA_N, block_n)

        qk_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            row_major.sm90_mma_major_mode(),
            row_major.sm90_mma_major_mode(),
            cutlass.Float32,
            (1, 1, 1),
            qk_tiler[:2],
        )
        qk_tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
        pv_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            row_major.sm90_mma_major_mode(),
            row_major.sm90_mma_major_mode(),
            cutlass.Float32,
            (1, 1, 1),
            pv_tiler[:2],
        )
        pv_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)

        k_smem_layout_staged = sm90_utils.make_smem_layout_a(
            row_major, qk_tiler, cutlass.BFloat16, 1
        )
        q_smem_layout_staged = sm90_utils.make_smem_layout_b(
            row_major, qk_tiler, cutlass.BFloat16, 1
        )
        v_smem_layout_staged = sm90_utils.make_smem_layout_a(
            row_major, pv_tiler, cutlass.BFloat16, 1
        )
        p_smem_layout_staged = sm90_utils.make_smem_layout_b(
            row_major, pv_tiler, cutlass.BFloat16, 1
        )
        kernel(
            Q,
            K_packed,
            V_packed,
            K_sz,
            V_sz,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            att_out,
            att_lse,
            sm_scale,
            qk_tiled_mma,
            pv_tiled_mma,
            k_smem_layout_staged,
            q_smem_layout_staged,
            v_smem_layout_staged,
            p_smem_layout_staged,
            min_block_kv=MIN_BLOCK_KV,
        ).launch(
            grid=(batch, kv_heads, max_splits),
            block=(_NUM_THREADS, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    fake_q = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (batch, q_heads, head_dim),
        (q_heads * head_dim, head_dim, 1),
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )
    packed_shape = (cache_size, kv_heads, head_dim // 4)
    packed_stride = (kv_heads * (head_dim // 4), head_dim // 4, 1)
    fake_k = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        packed_shape,
        packed_stride,
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )
    fake_v = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        packed_shape,
        packed_stride,
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )
    sz_shape = (cache_size, kv_heads, 2)
    sz_stride = (kv_heads * 2, 2, 1)
    fake_ksz = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        sz_shape,
        sz_stride,
        memspace=cute.AddressSpace.gmem,
        assumed_align=8,
    )
    fake_vsz = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        sz_shape,
        sz_stride,
        memspace=cute.AddressSpace.gmem,
        assumed_align=8,
    )
    fake_indptr = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (batch + 1,),
        (1,),
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )
    fake_indices = cute.runtime.make_fake_tensor(
        _index_cutlass_dtype(kv_indices_dtype),
        (cute.sym_int(),),
        (1,),
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )
    fake_splits = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (batch,),
        (1,),
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )
    fake_att = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        att_out_shape,
        att_out_stride,
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )
    fake_lse = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        att_lse_shape,
        att_lse_stride,
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )

    compiled = cute.compile(
        launcher,
        fake_q,
        fake_k,
        fake_v,
        fake_ksz,
        fake_vsz,
        fake_indptr,
        fake_indices,
        fake_splits,
        fake_att,
        fake_lse,
        sm_scale=cutlass.Float32(1.0),
        stream=cuda.CUstream(0),
    )
    _compiled_decode_kernels[key] = compiled
    return compiled


def can_use_flashinfer_cutedsl_decode_regsoftmax(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scales_zeros: torch.Tensor,
    v_scales_zeros: torch.Tensor,
) -> bool:
    return can_use_flashinfer_cutedsl_decode(
        q, k_buffer, v_buffer, k_scales_zeros, v_scales_zeros
    )


def flashinfer_cutedsl_decode_attention_fwd_int2_regsoftmax(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scales_zeros: torch.Tensor,
    v_scales_zeros: torch.Tensor,
    att_out: torch.Tensor,
    att_lse: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
):
    """Run the experimental accumulator-resident-softmax stage-1 kernel."""

    if not can_use_flashinfer_cutedsl_decode_regsoftmax(
        q, k_buffer, v_buffer, k_scales_zeros, v_scales_zeros
    ):
        raise ValueError(
            "register-softmax CuTeDSL INT2 decode requires SM90, head_dim=128, "
            "BF16 Q, packed uint8 KV, and query group size 1 or 4"
        )
    if max_kv_splits <= 0:
        raise ValueError("max_kv_splits must be positive")

    batch, q_heads, head_dim = q.shape
    cache_size, kv_heads, _ = k_buffer.shape
    kv_group_num = q_heads // kv_heads
    if kv_indptr.dtype != torch.int32 or not kv_indptr.is_contiguous():
        raise TypeError("kv_indptr must be contiguous torch.int32")
    if kv_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("kv_indices must be torch.int32 or torch.int64")
    if not kv_indices.is_contiguous():
        raise ValueError("kv_indices must be contiguous")
    if num_kv_splits.dtype != torch.int32 or not num_kv_splits.is_contiguous():
        raise TypeError("num_kv_splits must be contiguous torch.int32")
    if att_out.dtype != torch.float32 or att_lse.dtype != torch.float32:
        raise TypeError("att_out and att_lse must be torch.float32")
    if att_out.shape != (batch, q_heads, max_kv_splits, head_dim):
        raise ValueError("att_out shape does not match the stage-1 contract")
    if att_lse.shape != (batch, q_heads, max_kv_splits):
        raise ValueError("att_lse shape does not match the stage-1 contract")
    if att_out.stride(-1) != 1 or att_lse.stride(-1) != 1:
        raise ValueError("scratch tensors require contiguous innermost dimensions")

    tensors = (
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        att_out,
        att_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
    )
    if any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on the same CUDA device")

    compiled = _compile_decode(
        head_dim=head_dim,
        block_n=DECODE_BLOCK_N,
        kv_group_num=kv_group_num,
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        max_splits=max_kv_splits,
        cache_size=cache_size,
        kv_indices_dtype=kv_indices.dtype,
        att_out_shape=tuple(att_out.shape),
        att_out_stride=tuple(att_out.stride()),
        att_lse_shape=tuple(att_lse.shape),
        att_lse_stride=tuple(att_lse.stride()),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    compiled(
        _cached_from_dlpack(q, 16),
        _cached_from_dlpack(k_buffer, 16),
        _cached_from_dlpack(v_buffer, 16),
        _cached_from_dlpack(k_scales_zeros, 8),
        _cached_from_dlpack(v_scales_zeros, 8),
        _cached_from_dlpack(kv_indptr, 16),
        _cached_from_dlpack(kv_indices, 16),
        _cached_from_dlpack(num_kv_splits, 16),
        _cached_from_dlpack(att_out, 16),
        _cached_from_dlpack(att_lse, 16),
        cutlass.Float32(sm_scale),
        stream,
    )
