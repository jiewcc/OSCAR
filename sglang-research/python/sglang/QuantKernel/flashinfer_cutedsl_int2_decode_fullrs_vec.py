"""Optimized packed-dequant fully register-sourced WGMMA INT2 decode.

This production stage-1 component keeps the gather + 16-byte packed-load path
from ``flashinfer_cutedsl_int2_decode`` but changes the two GEMMs to

    K[64, 128] @ Q.T[128, 8] -> S.T[64, 8]
    V.T[128, 64] @ P.T[64, 8] -> O.T[128, 8]

so MHA/GQA decode no longer pays for a 64-row query tile.  The group-size-1
or group-size-4 query rows occupy the leading columns of the n8 WGMMA
tile.  P is staged through shared memory because Hopper WGMMA only supports a
register-sourced A operand, while the transposed PV formulation needs P as B.

The K and V register fragments use PRMT plus packed BF16 add/FMA to replace
scalar INT-to-FP32 conversion and dequantization.  The public entry point
keeps the same stage-1 scratch contract as the existing kernel and is ready
for attention dispatch selection.
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
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .flashinfer_cutedsl_int2_decode import (
    MIN_BLOCK_KV,
    _cached_from_dlpack,
    _index_cutlass_dtype,
    _layout_acc_mn,
    can_use_flashinfer_cutedsl_decode,
)

__all__ = [
    "can_use_flashinfer_cutedsl_decode_fullrs_vec",
    "flashinfer_cutedsl_decode_attention_fwd_int2_fullrs_vec",
]


DECODE_BLOCK_N = 64
_WGMMA_N = 8
_LOG2_E = 1.4426950408889634

_LOADER_THREADS = 128
_NUM_THREADS = 128

_compiled_decode_kernels: Dict[Tuple, object] = {}


@dsl_user_op
def _add_bf16x2(a, b, *, loc=None, ip=None):
    out = llvm.inline_asm(
        cutlass.Uint32.mlir_type,
        [
            cutlass.Uint32(a).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(b).ir_value(loc=loc, ip=ip),
        ],
        "add.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(out)


@dsl_user_op
def _fma_bf16x2(a, b, c, *, loc=None, ip=None):
    out = llvm.inline_asm(
        cutlass.Uint32.mlir_type,
        [
            cutlass.Uint32(a).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(b).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(c).ir_value(loc=loc, ip=ip),
        ],
        "fma.rn.bf16x2 $0, $1, $2, $3;",
        "=r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(out)


@cute.jit
def _pack_bf16x2(value):
    value_u16 = cutlass.BFloat16(value).bitcast(cutlass.Uint16)
    value_u32 = cutlass.Uint32(value_u16)
    return value_u32 | (value_u32 << 16)


@cute.jit
def _pack_bf16_pair(value_lo, value_hi):
    value_lo_u16 = cutlass.BFloat16(value_lo).bitcast(cutlass.Uint16)
    value_hi_u16 = cutlass.BFloat16(value_hi).bitcast(cutlass.Uint16)
    return cutlass.Uint32(value_lo_u16) | (cutlass.Uint32(value_hi_u16) << 16)


@cute.jit
def _pack_rs_k_words(rK_packed, lane_idx):
    """Warp-transpose one coalesced 16-byte K load into the RS A mapping.

    Each input lane owns one 16-byte half-row.  The Hopper m64k16 register
    fragment instead owns two rows eight apart and two columns in each of the
    [0:8] and [8:16] halves.  Four packed output words retain exactly the 16
    INT2 bytes needed by that lane for all eight k16 instructions.
    """

    rK_words = cute.recast_tensor(rK_packed, cutlass.Uint32)
    rK_rs = cute.make_rmem_tensor((4,), cutlass.Uint32)
    row_in_half = lane_idx // 4
    col_group = lane_idx % 4
    pair_shift = (col_group % 2) * 16

    for packed_half in cutlass.range_constexpr(2):
        src_row0 = 2 * row_in_half + packed_half
        src_row1 = 2 * (row_in_half + 8) + packed_half
        for col_half in cutlass.range_constexpr(2):
            word0_idx = 2 * col_half
            word1_idx = word0_idx + 1
            row0_word0 = cute.arch.shuffle_sync(rK_words[word0_idx], src_row0)
            row0_word1 = cute.arch.shuffle_sync(rK_words[word1_idx], src_row0)
            row1_word0 = cute.arch.shuffle_sync(rK_words[word0_idx], src_row1)
            row1_word1 = cute.arch.shuffle_sync(rK_words[word1_idx], src_row1)
            row0_word = row0_word0
            row1_word = row1_word0
            if col_group >= 2:
                row0_word = row0_word1
                row1_word = row1_word1
            row0_pair = (row0_word >> pair_shift) & cutlass.Uint32(0xFFFF)
            row1_pair = (row1_word >> pair_shift) & cutlass.Uint32(0xFFFF)
            rK_rs[2 * packed_half + col_half] = row0_pair | (row1_pair << 16)
    return rK_rs


@cute.jit
def _gemm_rs_k_dequant(
    tiled_mma,
    rK_rs,
    ks,
    kz,
    lane_idx,
    operand_b,
    accumulator,
):
    """Dequantize one k16 RS fragment at a time and immediately issue WGMMA."""

    row_in_half = lane_idx // 4
    src_row0 = 2 * row_in_half
    src_row1 = 2 * (row_in_half + 8)
    ks0 = cute.arch.shuffle_sync(ks, src_row0)
    kz0 = cute.arch.shuffle_sync(kz, src_row0)
    ks1 = cute.arch.shuffle_sync(ks, src_row1)
    kz1 = cute.arch.shuffle_sync(kz, src_row1)

    atom = cute.make_mma_atom(tiled_mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, False)
    accumulator_atom = accumulator[None, 0, 0]
    scale0 = _pack_bf16x2(ks0)
    scale1 = _pack_bf16x2(ks1)
    bias0 = _pack_bf16x2(-(kz0 * ks0))
    bias1 = _pack_bf16x2(-(kz1 * ks1))
    neg_128 = cutlass.Uint32(0xC300C300)
    bf16_magic = cutlass.Uint32(0x43434343)
    rK_bf16x2 = cute.make_rmem_tensor((4,), cutlass.Uint32)
    for k_block in cutlass.range_constexpr(8):
        packed_half = k_block % 2
        field_shift = (k_block // 2) * 2
        word_lo = rK_rs[2 * packed_half]
        word_hi = rK_rs[2 * packed_half + 1]
        qbytes_lo = (word_lo >> field_shift) & cutlass.Uint32(0x03030303)
        qbytes_hi = (word_hi >> field_shift) & cutlass.Uint32(0x03030303)
        q01_lo = cutlass.Uint32(
            cute.arch.prmt(qbytes_lo, bf16_magic, cutlass.Uint32(0x4140))
        )
        q23_lo = cutlass.Uint32(
            cute.arch.prmt(qbytes_lo, bf16_magic, cutlass.Uint32(0x4342))
        )
        q01_hi = cutlass.Uint32(
            cute.arch.prmt(qbytes_hi, bf16_magic, cutlass.Uint32(0x4140))
        )
        q23_hi = cutlass.Uint32(
            cute.arch.prmt(qbytes_hi, bf16_magic, cutlass.Uint32(0x4342))
        )
        rK_bf16x2[0] = _fma_bf16x2(_add_bf16x2(q01_lo, neg_128), scale0, bias0)
        rK_bf16x2[1] = _fma_bf16x2(_add_bf16x2(q23_lo, neg_128), scale1, bias1)
        rK_bf16x2[2] = _fma_bf16x2(_add_bf16x2(q01_hi, neg_128), scale0, bias0)
        rK_bf16x2[3] = _fma_bf16x2(_add_bf16x2(q23_hi, neg_128), scale1, bias1)
        rK_atom = cute.recast_tensor(rK_bf16x2, cutlass.BFloat16)
        cute.mma_atom_call(
            atom,
            accumulator_atom,
            rK_atom,
            operand_b[None, 0, k_block],
            accumulator_atom,
        )
        atom.set(warpgroup.Field.ACCUMULATE, True)


@cute.jit
def _gemm_rs_v_dequant_both(
    tiled_mma,
    sVRaw,
    sVScale,
    sVBias,
    lane_idx,
    warp_idx,
    operand_b,
    accumulator_lo,
    accumulator_hi,
):
    """Stream packed V through the native m64n8k16 register-A mapping.

    PTX assigns each thread two M rows and four K columns.  The packed V tile
    remains in shared memory in its naturally coalesced [token, d/4] layout;
    each k16 step gathers only the eight BF16 values required by that thread,
    dequantizes them in registers, and immediately issues the low- and
    high-64 output WGMMA operations.  V scale/zero loads are shared by the two
    output halves.
    """

    row0 = 16 * warp_idx + lane_idx // 4
    row1 = row0 + 8
    token_pair = 2 * (lane_idx % 4)
    packed_dim0 = row0 % 32
    packed_dim1 = row1 % 32
    lo_shift = cutlass.Int32(0)
    hi_shift = cutlass.Int32(4)
    if warp_idx >= 2:
        lo_shift = cutlass.Int32(2)
        hi_shift = cutlass.Int32(6)

    atom = cute.make_mma_atom(tiled_mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, True)
    accumulator_lo_atom = accumulator_lo[None, 0, 0]
    accumulator_hi_atom = accumulator_hi[None, 0, 0]
    neg_128 = cutlass.Uint32(0xC300C300)
    bf16_magic = cutlass.Uint32(0x43434343)
    rV_lo_bf16x2 = cute.make_rmem_tensor((4,), cutlass.Uint32)
    rV_hi_bf16x2 = cute.make_rmem_tensor((4,), cutlass.Uint32)

    for k_block in cutlass.range_constexpr(4):
        token0 = 16 * k_block + token_pair
        token1 = token0 + 1
        token8 = token0 + 8
        token9 = token8 + 1

        vs0 = cutlass.BFloat16(sVScale[token0])
        vb0 = cutlass.BFloat16(sVBias[token0])
        vs1 = cutlass.BFloat16(sVScale[token1])
        vb1 = cutlass.BFloat16(sVBias[token1])
        vs8 = cutlass.BFloat16(sVScale[token8])
        vb8 = cutlass.BFloat16(sVBias[token8])
        vs9 = cutlass.BFloat16(sVScale[token9])
        vb9 = cutlass.BFloat16(sVBias[token9])

        # Register order follows Figure 148 of the PTX WGMMA fragment
        # mapping: row0@k[0:2], row1@k[0:2], then k[8:10].
        b00 = cutlass.Uint32(sVRaw[token0, packed_dim0])
        b01 = cutlass.Uint32(sVRaw[token1, packed_dim0])
        b10 = cutlass.Uint32(sVRaw[token0, packed_dim1])
        b11 = cutlass.Uint32(sVRaw[token1, packed_dim1])
        b08 = cutlass.Uint32(sVRaw[token8, packed_dim0])
        b09 = cutlass.Uint32(sVRaw[token9, packed_dim0])
        b18 = cutlass.Uint32(sVRaw[token8, packed_dim1])
        b19 = cutlass.Uint32(sVRaw[token9, packed_dim1])

        raw01 = b00 | (b01 << 8) | (b10 << 16) | (b11 << 24)
        raw89 = b08 | (b09 << 8) | (b18 << 16) | (b19 << 24)
        scale01 = _pack_bf16_pair(vs0, vs1)
        bias01 = _pack_bf16_pair(vb0, vb1)
        scale89 = _pack_bf16_pair(vs8, vs9)
        bias89 = _pack_bf16_pair(vb8, vb9)

        qbytes_lo01 = (raw01 >> lo_shift) & cutlass.Uint32(0x03030303)
        qbytes_lo89 = (raw89 >> lo_shift) & cutlass.Uint32(0x03030303)
        qbytes_hi01 = (raw01 >> hi_shift) & cutlass.Uint32(0x03030303)
        qbytes_hi89 = (raw89 >> hi_shift) & cutlass.Uint32(0x03030303)

        q01_lo = cutlass.Uint32(
            cute.arch.prmt(qbytes_lo01, bf16_magic, cutlass.Uint32(0x4140))
        )
        q23_lo = cutlass.Uint32(
            cute.arch.prmt(qbytes_lo01, bf16_magic, cutlass.Uint32(0x4342))
        )
        q89_lo = cutlass.Uint32(
            cute.arch.prmt(qbytes_lo89, bf16_magic, cutlass.Uint32(0x4140))
        )
        qab_lo = cutlass.Uint32(
            cute.arch.prmt(qbytes_lo89, bf16_magic, cutlass.Uint32(0x4342))
        )
        q01_hi = cutlass.Uint32(
            cute.arch.prmt(qbytes_hi01, bf16_magic, cutlass.Uint32(0x4140))
        )
        q23_hi = cutlass.Uint32(
            cute.arch.prmt(qbytes_hi01, bf16_magic, cutlass.Uint32(0x4342))
        )
        q89_hi = cutlass.Uint32(
            cute.arch.prmt(qbytes_hi89, bf16_magic, cutlass.Uint32(0x4140))
        )
        qab_hi = cutlass.Uint32(
            cute.arch.prmt(qbytes_hi89, bf16_magic, cutlass.Uint32(0x4342))
        )

        rV_lo_bf16x2[0] = _fma_bf16x2(_add_bf16x2(q01_lo, neg_128), scale01, bias01)
        rV_lo_bf16x2[1] = _fma_bf16x2(_add_bf16x2(q23_lo, neg_128), scale01, bias01)
        rV_lo_bf16x2[2] = _fma_bf16x2(_add_bf16x2(q89_lo, neg_128), scale89, bias89)
        rV_lo_bf16x2[3] = _fma_bf16x2(_add_bf16x2(qab_lo, neg_128), scale89, bias89)
        rV_hi_bf16x2[0] = _fma_bf16x2(_add_bf16x2(q01_hi, neg_128), scale01, bias01)
        rV_hi_bf16x2[1] = _fma_bf16x2(_add_bf16x2(q23_hi, neg_128), scale01, bias01)
        rV_hi_bf16x2[2] = _fma_bf16x2(_add_bf16x2(q89_hi, neg_128), scale89, bias89)
        rV_hi_bf16x2[3] = _fma_bf16x2(_add_bf16x2(qab_hi, neg_128), scale89, bias89)
        rV_lo = cute.recast_tensor(rV_lo_bf16x2, cutlass.BFloat16)
        rV_hi = cute.recast_tensor(rV_hi_bf16x2, cutlass.BFloat16)

        cute.mma_atom_call(
            atom,
            accumulator_lo_atom,
            rV_lo,
            operand_b[None, 0, k_block],
            accumulator_lo_atom,
        )
        cute.mma_atom_call(
            atom,
            accumulator_hi_atom,
            rV_hi,
            operand_b[None, 0, k_block],
            accumulator_hi_atom,
        )


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
        q_smem_layout_staged: cute.ComposedLayout,
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
        sQ_mma = smem.allocate_tensor(
            cutlass.BFloat16,
            q_smem_layout_staged.outer,
            1024,
            swizzle=q_smem_layout_staged.inner,
        )
        sVRaw = smem.allocate_tensor(
            cutlass.Uint8,
            cute.make_layout((block_n, quarter_dim), stride=(quarter_dim, 1)),
            1024,
        )
        sVScale = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((block_n,)), 16
        )
        sVBias = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((block_n,)), 16
        )
        sP_mma = smem.allocate_tensor(
            cutlass.BFloat16,
            p_smem_layout_staged.outer,
            1024,
            swizzle=p_smem_layout_staged.inner,
        )
        sScore = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((_WGMMA_N, block_n), stride=(block_n, 1)),
            16,
        )
        sMax = smem.allocate_tensor(cutlass.Float32, cute.make_layout((_WGMMA_N,)), 16)
        sSum = smem.allocate_tensor(cutlass.Float32, cute.make_layout((_WGMMA_N,)), 16)
        sCorrection = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((_WGMMA_N,)), 16
        )

        qk_thr_mma = qk_tiled_mma.get_slice(tidx)
        pv_thr_mma = pv_tiled_mma.get_slice(tidx)

        tQsQ = qk_thr_mma.partition_B(sQ_mma)
        tQrQ = qk_thr_mma.make_fragment_B(tQsQ)
        tPsP = pv_thr_mma.partition_B(sP_mma)
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
                    sQ_mma[qid, tidx, 0] = Q[cur_batch, q_head_base + qid, tidx]
                else:
                    sQ_mma[qid, tidx, 0] = cutlass.BFloat16(0.0)
        if tidx < _WGMMA_N:
            sMax[tidx] = -cutlass.Float32.inf
            sSum[tidx] = cutlass.Float32(0.0)
            sCorrection[tidx] = cutlass.Float32(1.0)
        # The unused n8 columns are never touched by a softmax warp.  Clear P
        # once so their PV contribution remains zero for every tile.
        for item in cutlass.range_constexpr(_WGMMA_N * block_n // _NUM_THREADS):
            linear = tidx + item * _NUM_THREADS
            qid = linear // block_n
            token = linear % block_n
            sP_mma[qid, token, 0] = cutlass.BFloat16(0.0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        tile_start = split_start
        while tile_start < split_stop:
            # Preserve the proven one-LDG.E.128-per-thread V gather, but stage
            # only its 2 KiB packed representation.  Dequantization is delayed
            # until each native PV register fragment is issued.
            row = loader_tid // loader_lanes_per_token
            d4_base = (loader_tid % loader_lanes_per_token) * packed_bytes_per_loader
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

            packed_lane = loader_tid % loader_lanes_per_token
            gV_packed = V_packed[kv_pos, cur_kv_head, None]
            gV_lane = cute.local_tile(
                gV_packed, (packed_bytes_per_loader,), (packed_lane,)
            )
            rV_packed = cute.make_rmem_tensor((packed_bytes_per_loader,), cutlass.Uint8)
            rV_packed.fill(cutlass.Uint8(0))
            if valid_token:
                cute.autovec_copy(gV_lane, rV_packed)
            for j in cutlass.range_constexpr(packed_bytes_per_loader):
                d4 = d4_base + j
                sVRaw[row, d4] = rV_packed[j]
            if packed_lane == 0:
                sVScale[row] = cutlass.BFloat16(vs)
                sVBias[row] = cutlass.BFloat16(-(vz * vs))

            # K is loaded once as one LDG.E.128 per lane, warp-transposed in
            # registers, and streamed through eight native m64n8k16 RS ops.
            gK_packed = K_packed[kv_pos, cur_kv_head, None]
            gK_lane = cute.local_tile(
                gK_packed, (packed_bytes_per_loader,), (packed_lane,)
            )
            rK_packed = cute.make_rmem_tensor((packed_bytes_per_loader,), cutlass.Uint8)
            rK_packed.fill(cutlass.Uint8(0))
            if valid_token:
                cute.autovec_copy(gK_lane, rK_packed)
            rK_rs = _pack_rs_k_words(rK_packed, lane_idx)

            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
            acc_qk.fill(cutlass.Float32(0.0))
            cute.nvgpu.warpgroup.fence()
            _gemm_rs_k_dequant(
                qk_tiled_mma,
                rK_rs,
                ks,
                kz,
                lane_idx,
                tQrQ[(None, None, None, 0)],
                acc_qk,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            if tidx < _NUM_THREADS:
                acc_qk_mn = cute.make_tensor(
                    acc_qk.iterator, _layout_acc_mn(qk_tiled_mma, acc_qk.layout)
                )
                valid_tokens = split_stop - tile_start
                for row in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[0])):
                    for col in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[1])):
                        if tQcQK_mn[row, col][0] >= valid_tokens:
                            acc_qk_mn[row, col] = -cutlass.Float32.inf

                # The m64 token dimension spans all four warps.  Materialize
                # scores by identity coordinate; one warp per real query then
                # owns the complete 64-token softmax reduction.
                for row in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[0])):
                    for col in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[1])):
                        token = tQcQK_mn[row, col][0]
                        qid = tQcQK_mn[row, col][1]
                        sScore[qid, token] = acc_qk_mn[row, col]
            cute.arch.barrier()

            if warp_idx < kv_group_num:
                qid = warp_idx
                score_lo = cutlass.Float32(sScore[qid, lane_idx])
                score_hi = cutlass.Float32(sScore[qid, lane_idx + 32])
                tile_max = cute.arch.fmax(score_lo, score_hi)
                tile_max = cute.arch.warp_reduction_max(tile_max, threads_in_group=32)
                previous_max = cutlass.Float32(sMax[qid])
                current_max = cute.arch.fmax(previous_max, tile_max)
                finite_max = current_max
                if finite_max == -cutlass.Float32.inf:
                    finite_max = cutlass.Float32(0.0)
                scale_log2 = cutlass.Float32(_LOG2_E) * sm_scale
                p_lo = cute.math.exp2(
                    (score_lo - finite_max) * scale_log2, fastmath=True
                )
                p_hi = cute.math.exp2(
                    (score_hi - finite_max) * scale_log2, fastmath=True
                )
                tile_sum = cute.arch.warp_reduction_sum(
                    p_lo + p_hi, threads_in_group=32
                )
                correction = cute.math.exp2(
                    (previous_max - finite_max) * scale_log2, fastmath=True
                )
                if lane_idx == 0:
                    sMax[qid] = current_max
                    sSum[qid] = cutlass.Float32(sSum[qid]) * correction + tile_sum
                    sCorrection[qid] = correction
                sP_mma[qid, lane_idx, 0] = cutlass.BFloat16(p_lo)
                sP_mma[qid, lane_idx + 32, 0] = cutlass.BFloat16(p_hi)
                cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()

            # Apply the online-softmax correction to both 64-d output
            # fragments before adding this tile's transposed PV products.
            acc_pv_lo_mn = cute.make_tensor(
                acc_pv_lo.iterator,
                _layout_acc_mn(pv_tiled_mma, acc_pv_lo.layout),
            )
            acc_pv_hi_mn = cute.make_tensor(
                acc_pv_hi.iterator,
                _layout_acc_mn(pv_tiled_mma, acc_pv_hi.layout),
            )
            for row in cutlass.range_constexpr(cute.size(acc_pv_lo_mn, mode=[0])):
                for col in cutlass.range_constexpr(cute.size(acc_pv_lo_mn, mode=[1])):
                    qid = tPcPV_mn[row, col][1]
                    correction = cutlass.Float32(sCorrection[qid])
                    acc_pv_lo_mn[row, col] = acc_pv_lo_mn[row, col] * correction
                    acc_pv_hi_mn[row, col] = acc_pv_hi_mn[row, col] * correction

            if tidx < _NUM_THREADS:
                cute.nvgpu.warpgroup.fence()
                _gemm_rs_v_dequant_both(
                    pv_tiled_mma,
                    sVRaw,
                    sVScale,
                    sVBias,
                    lane_idx,
                    warp_idx,
                    tPrP[(None, None, None, 0)],
                    acc_pv_lo,
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
                for col in cutlass.range_constexpr(cute.size(acc_pv_lo_mn, mode=[1])):
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
                        ] = (
                            acc_pv_lo_mn[row, col] * inv_total
                        )
                        att_out[
                            cur_batch,
                            q_head_base + qid,
                            split_kv_id,
                            out_dim + 64,
                        ] = (
                            acc_pv_hi_mn[row, col] * inv_total
                        )
                        if out_dim == 0:
                            att_lse[cur_batch, q_head_base + qid, split_kv_id] = lse

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
        _WGMMA_N * head_dim * 2
        + block_n * (head_dim // 4)
        + block_n * 2 * 2
        + _WGMMA_N * block_n * 2
        + _WGMMA_N * block_n * 4
        + _WGMMA_N * 3 * 4
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
            warpgroup.OperandMajorMode.K,
            row_major.sm90_mma_major_mode(),
            cutlass.Float32,
            (1, 1, 1),
            qk_tiler[:2],
            a_source=warpgroup.OperandSource.RMEM,
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
            a_source=warpgroup.OperandSource.RMEM,
        )
        pv_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)

        q_smem_layout_staged = sm90_utils.make_smem_layout_b(
            row_major, qk_tiler, cutlass.BFloat16, 1
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
            q_smem_layout_staged,
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
        assumed_align=4,
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


def can_use_flashinfer_cutedsl_decode_fullrs_vec(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scales_zeros: torch.Tensor,
    v_scales_zeros: torch.Tensor,
) -> bool:
    return can_use_flashinfer_cutedsl_decode(
        q, k_buffer, v_buffer, k_scales_zeros, v_scales_zeros
    )


def flashinfer_cutedsl_decode_attention_fwd_int2_fullrs_vec(
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
    """Run the optimized register-sourced transposed-WGMMA stage-1 kernel."""

    if not can_use_flashinfer_cutedsl_decode_fullrs_vec(
        q, k_buffer, v_buffer, k_scales_zeros, v_scales_zeros
    ):
        raise ValueError(
            "register-sourced CuTeDSL INT2 decode requires SM90, head_dim=128, "
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
    if kv_indptr.numel() < batch + 1 or num_kv_splits.numel() < batch:
        raise ValueError("KV metadata is smaller than the query batch")
    if att_out.dtype != torch.float32 or att_lse.dtype != torch.float32:
        raise TypeError("att_out and att_lse must be torch.float32")
    if att_out.ndim != 4 or att_lse.ndim != 3:
        raise ValueError("att_out must be 4D and att_lse must be 3D")
    if att_out.shape[0] < batch or att_lse.shape[0] < batch:
        raise ValueError("attention scratch batch dimension is too small")
    if att_out.shape[1] != q_heads or att_lse.shape[1] != q_heads:
        raise ValueError("attention scratch head dimension does not match Q")
    if att_out.shape[2] != max_kv_splits or att_lse.shape[2] != max_kv_splits:
        raise ValueError(
            "attention scratch split dimension does not match max_kv_splits"
        )
    if att_out.shape[3] != head_dim:
        raise ValueError("att_out head dimension does not match Q")
    if att_out.stride(-1) != 1 or att_lse.stride(-1) != 1:
        raise ValueError("attention scratch requires a contiguous innermost dimension")

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
        _cached_from_dlpack(att_lse, 4),
        cutlass.Float32(sm_scale),
        stream,
    )
