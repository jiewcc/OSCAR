"""FlashInfer-style CuTe DSL decode attention for an INT2 KV cache.

This module intentionally lives next to, but does not modify, the legacy
``cutedsl_int2_kv`` implementation.  It implements the stage-1 split-KV
contract used by SGLang decode attention:

* gather logical KV tokens through ``kv_indptr`` / ``kv_indices``;
* unpack four INT2 values from every byte and apply per-token/head
  ``(q - zero) * scale`` dequantization in the loader;
* materialize BF16 K/V tiles in WGMMA-compatible shared memory;
* run QK, online softmax, and PV on Hopper tensor cores for every split; and
* write one normalized output and LSE per split for the existing stage-2
  reducer.

The block is warp-group specialized on Hopper: the first warp group gathers
and dequantizes K/V, while the second warp group consumes the BF16 tiles with
SM90 WGMMA.  WGMMA has a fixed 64-row atom, so the one (MHA) or four (GQA)
query rows occupy the leading rows of the tile and padded rows are discarded.

Unlike the legacy kernel, the KV tile loop is a true runtime ``while`` loop.
It is bounded by the actual token range assigned to the split, never by a
compile-time sequence-length heuristic or by ``max_kv_splits``.
"""

from __future__ import annotations

import weakref
from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.runtime import from_dlpack

__all__ = [
    "can_use_flashinfer_cutedsl_decode",
    "flashinfer_cutedsl_decode_attention_fwd_int2",
]


# Match the split rounding used by the Triton INT2 stage-1 path.  This is
# intentionally independent of the attention tile size.
MIN_BLOCK_KV = 32

# A 64-token tile keeps the two dequantized BF16 operands plus softmax scratch
# comfortably below Hopper's per-CTA shared-memory limit.  The runtime loop
# handles arbitrarily many tiles.
DECODE_BLOCK_N = 64

# Hopper WGMMA instructions have a fixed 64-row M dimension.  Only the first
# ``kv_group_num`` rows are meaningful for decode; the remaining rows never
# reach global memory.
_WGMMA_M = 64
_LOG2_E = 1.4426950408889634

_LOADER_THREADS = 128
_CONSUMER_THREADS = 128
_NUM_THREADS = _LOADER_THREADS + _CONSUMER_THREADS

_compiled_decode_kernels: Dict[Tuple, object] = {}

# Creating a CuTe runtime tensor from a PyTorch DLPack capsule is surprisingly
# expensive relative to a decode kernel.  Cache adapters by Python tensor
# identity, but keep only a weak reference to that tensor: the adapter owns a
# detached DLPack view, so retaining it after the caller tensor dies would pin
# CUDA storage indefinitely.  The metadata signature catches in-place
# ``set_``/``resize_`` changes while deliberately ignoring ``Tensor._version``;
# changing the values in a KV cache does not invalidate its pointer/layout.
_dlpack_adapter_cache: Dict[int, tuple] = {}


def _dlpack_signature(tensor: torch.Tensor, assumed_align: int) -> tuple:
    data_ptr = tensor.data_ptr()
    if data_ptr % assumed_align != 0:
        raise ValueError(
            f"tensor data pointer 0x{data_ptr:x} is not {assumed_align}-byte aligned"
        )
    return (
        data_ptr,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        assumed_align,
    )


def _cached_from_dlpack(tensor: torch.Tensor, assumed_align: int):
    """Return a reusable CuTe runtime tensor without extending input lifetime.

    An ``id`` lookup alone is unsafe because CPython can recycle object ids.
    Requiring ``entry_ref() is tensor`` closes that hole, and the callback only
    removes the cache slot if it still contains the same weakref (important if
    metadata changed and the entry was replaced before collection).
    """

    cache_key = id(tensor)
    signature = _dlpack_signature(tensor, assumed_align)
    entry = _dlpack_adapter_cache.get(cache_key)
    if entry is not None:
        tensor_ref, cached_signature, adapter = entry
        if tensor_ref() is tensor and cached_signature == signature:
            return adapter

    def evict(dead_ref, key=cache_key):
        current = _dlpack_adapter_cache.get(key)
        if current is not None and current[0] is dead_ref:
            _dlpack_adapter_cache.pop(key, None)

    tensor_ref = weakref.ref(tensor, evict)
    adapter = from_dlpack(tensor.detach(), assumed_align=assumed_align)
    _dlpack_adapter_cache[cache_key] = (tensor_ref, signature, adapter)
    return adapter


def _layout_separate(thr, src, ref):
    """Split an MMA value layout into logical M and N components.

    This is the same layout transform used by CUTLASS's Hopper ``fmha.py``.
    WGMMA accumulator values are distributed across the warp group; the
    transform exposes a compact per-thread ``(M, N)`` view for softmax and the
    epilogue.
    """

    lt = cute.make_layout(())
    ge = cute.make_layout(())
    for k, value in enumerate(ref):
        if cutlass.const_expr(value < thr):
            lt = cute.append(lt, src[k])
        else:
            ge = cute.append(ge, src[k])
    if cutlass.const_expr(cute.rank(lt) == 1):
        return cute.append(lt, ge)
    return cute.append(cute.append(cute.make_layout(()), lt), ge)


@cute.jit
def _layout_acc_mn(tiled_mma, acc):
    separated = _layout_separate(
        tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1]
    )
    value_m = separated[0]
    value_n = separated[1]
    if cutlass.const_expr(cute.rank(value_m) == 1):
        value_m = cute.append(value_m, acc[1])
    else:
        value_m = cute.append(cute.append(cute.make_layout(()), value_m), acc[1])
    if cutlass.const_expr(cute.rank(value_n) == 1):
        value_n = cute.append(value_n, acc[2])
    else:
        value_n = cute.append(cute.append(cute.make_layout(()), value_n), acc[2])
    if cutlass.const_expr(cute.rank(value_m) == 1):
        return cute.append(value_m, value_n)
    return cute.append(cute.append(cute.make_layout(()), value_m), value_n)


@cute.jit
def _reduction_target_n(tiled_mma):
    separated = _layout_separate(
        tiled_mma.shape_mnk[0],
        cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
        tiled_mma.tv_layout_C.stride[0],
    )
    return separated[1]


@cute.jit
def _gemm_zero_acc(
    tiled_mma_zero, tiled_mma_accumulate, operand_a, operand_b, accumulator
):
    """Issue a multi-K-block WGMMA and overwrite ``accumulator``."""

    cute.gemm(
        tiled_mma_zero,
        accumulator,
        operand_a[None, None, 0],
        operand_b[None, None, 0],
        accumulator,
    )
    for k_block in cutlass.range_constexpr(1, cute.size(operand_a, mode=[2])):
        cute.gemm(
            tiled_mma_accumulate,
            accumulator,
            operand_a[None, None, k_block],
            operand_b[None, None, k_block],
            accumulator,
        )


@cute.jit
def _gemm_accumulate(tiled_mma, operand_a, operand_b, accumulator):
    """Issue a multi-K-block WGMMA and preserve the previous accumulator."""

    for k_block in cutlass.range_constexpr(cute.size(operand_a, mode=[2])):
        cute.gemm(
            tiled_mma,
            accumulator,
            operand_a[None, None, k_block],
            operand_b[None, None, k_block],
            accumulator,
        )


def _convert_c_layout_to_a_layout(c_layout, a_layout):
    return cute.make_layout(
        (
            a_layout,
            c_layout.shape[1],
            (c_layout.shape[2], cute.size(c_layout, mode=[0]) // cute.size(a_layout)),
        ),
        stride=(
            c_layout.stride[0],
            c_layout.stride[1],
            (
                c_layout.stride[2],
                cute.size(a_layout, mode=[2]) * c_layout.stride[0][2],
            ),
        ),
    )


@cute.jit
def _make_acc_into_bf16_operand(accumulator, operand_layout_tv):
    """Reinterpret the QK C fragment as the register-sourced PV A fragment."""

    operand = cute.make_rmem_tensor_like(
        _convert_c_layout_to_a_layout(accumulator.layout, operand_layout_tv.shape[1]),
        cutlass.BFloat16,
    )
    operand_as_acc = cute.make_tensor(operand.iterator, accumulator.layout)
    operand_as_acc.store(accumulator.load().to(cutlass.BFloat16))
    return operand


@cute.jit
def _online_softmax(
    acc_qk,
    qk_tiled_mma,
    s_max,
    a_sum,
    acc_pv,
    pv_tiled_mma,
    scale_softmax_log2,
):
    """Update online-softmax state in the WGMMA accumulator fragments."""

    acc_qk_mn = cute.make_tensor(
        acc_qk.iterator, _layout_acc_mn(qk_tiled_mma, acc_qk.layout)
    )
    acc_pv_mn = cute.make_tensor(
        acc_pv.iterator, _layout_acc_mn(pv_tiled_mma, acc_pv.layout)
    )
    reduction_target = _reduction_target_n(qk_tiled_mma)
    reduction_rank = cute.rank(reduction_target)
    s_max_prev = cute.make_rmem_tensor_like(s_max, s_max._dtype)

    for row in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[0])):
        s_max_prev[row] = s_max[row]
        for col in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[1])):
            s_max[row] = cute.arch.fmax(s_max[row], acc_qk_mn[row, col])
        for reduction in cutlass.range_constexpr(reduction_rank):
            s_max[row] = cute.arch.warp_reduction_max(
                s_max[row], threads_in_group=reduction_target.shape[reduction]
            )

        local_max = s_max[row]
        if s_max[row] == -cutlass.Float32.inf:
            local_max = cutlass.Float32(0.0)
        scale_max = scale_softmax_log2 * local_max
        for col in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[1])):
            acc_qk_mn[row, col] = cute.math.exp2(
                scale_softmax_log2 * acc_qk_mn[row, col] - scale_max,
                fastmath=True,
            )

        current_max = s_max[row]
        if s_max[row] == -cutlass.Float32.inf:
            current_max = cutlass.Float32(0.0)
        correction = cute.math.exp2(
            (s_max_prev[row] - current_max) * scale_softmax_log2,
            fastmath=True,
        )
        a_sum[row] = a_sum[row] * correction
        for col in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[1])):
            acc_pv_mn[row, col] = acc_pv_mn[row, col] * correction

        a_sum[row] = a_sum[row] + acc_qk_mn[row, None].load().reduce(
            cute.ReductionOp.ADD, cutlass.Float32.zero, 0
        )


@cute.jit
def _normalize_softmax(s_max, a_sum, acc_pv, pv_tiled_mma, sm_scale):
    """Finish cross-lane sum reduction, normalize O, and return per-row LSE."""

    reduction_target = _reduction_target_n(pv_tiled_mma)
    for reduction in cutlass.range_constexpr(cute.rank(reduction_target)):
        for row in cutlass.range_constexpr(cute.size(a_sum)):
            a_sum[row] = cute.arch.warp_reduction_sum(
                a_sum[row], threads_in_group=reduction_target.shape[reduction]
            )

    acc_pv_mn = cute.make_tensor(
        acc_pv.iterator, _layout_acc_mn(pv_tiled_mma, acc_pv.layout)
    )
    lse = cute.make_rmem_tensor_like(a_sum, cutlass.Float32)
    for row in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[0])):
        total = a_sum[row]
        inv_total = cutlass.Float32(0.0)
        lse[row] = -cutlass.Float32.inf
        if total > cutlass.Float32(0.0):
            inv_total = cutlass.Float32(1.0) / total
            lse[row] = s_max[row] * sm_scale + cute.log(total)
        for col in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[1])):
            acc_pv_mn[row, col] = acc_pv_mn[row, col] * inv_total
    return lse


def _define_decode_kernel(
    head_dim: cutlass.Constexpr[int],
    block_n: cutlass.Constexpr[int],
    kv_group_num: cutlass.Constexpr[int],
):
    """Build the Hopper stage-1 kernel.

    Grid layout is ``(batch, kv_head, split)``.  A CTA handles every query
    head that shares one KV head, so GQA gathers and dequantizes a K/V tile
    once and reuses it across the group.
    """

    quarter_dim: cutlass.Constexpr[int] = head_dim // 4
    packed_bytes_per_tile: cutlass.Constexpr[int] = block_n * quarter_dim
    packed_bytes_per_loader: cutlass.Constexpr[int] = (
        packed_bytes_per_tile + _LOADER_THREADS - 1
    ) // _LOADER_THREADS
    loader_lanes_per_token: cutlass.Constexpr[int] = _LOADER_THREADS // block_n

    @cute.kernel
    def kernel(
        Q: cute.Tensor,  # [batch, q_heads, 128] bf16
        K_packed: cute.Tensor,  # [cache, kv_heads, 32] uint8
        V_packed: cute.Tensor,  # [cache, kv_heads, 32] uint8
        K_sz: cute.Tensor,  # [cache, kv_heads, 2] fp32
        V_sz: cute.Tensor,  # [cache, kv_heads, 2] fp32
        kv_indptr: cute.Tensor,  # [batch + 1] int32
        kv_indices: cute.Tensor,  # [total_kv] int32/int64
        num_kv_splits: cute.Tensor,  # [batch] int32
        att_out: cute.Tensor,  # [batch, q_heads, max_splits, 128] fp32
        att_lse: cute.Tensor,  # [batch, q_heads, max_splits] fp32
        sm_scale: cutlass.Float32,
        qk_tiled_mma: cute.TiledMma,
        qk_tiled_mma_accumulate: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        pv_tiled_mma_accumulate: cute.TiledMma,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        min_block_kv: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cur_batch, cur_kv_head, split_kv_id = cute.arch.block_idx()

        is_loader = tidx < _LOADER_THREADS
        is_consumer = tidx >= _LOADER_THREADS
        loader_tid = tidx
        consumer_tid = tidx - _LOADER_THREADS
        q_head_base = cur_kv_head * kv_group_num

        kv_start_idx = cutlass.Int32(kv_indptr[cur_batch])
        seq_len = cutlass.Int32(kv_indptr[cur_batch + 1]) - kv_start_idx
        kv_splits = cutlass.Int32(num_kv_splits[cur_batch])
        # Unified attention can legitimately assign zero splits to this tier
        # for a batch row with no INT2 tokens.  Clamp only the partition math;
        # the raw value still gates stores so every scratch LSE remains -inf.
        partition_splits = max(kv_splits, cutlass.Int32(1))

        # Same split partition as Triton:
        # ceil_div(ceil_div(seq_len, splits), MIN_BLOCK_KV) * MIN_BLOCK_KV.
        kv_len_per_split = (
            ((seq_len + partition_splits - 1) // partition_splits + min_block_kv - 1)
            // min_block_kv
            * min_block_kv
        )
        split_start = kv_len_per_split * split_kv_id
        split_stop = min(seq_len, split_start + kv_len_per_split)

        smem = cutlass.utils.SmemAllocator()

        # Q and the loader-to-consumer boundary use the same swizzled layouts
        # as CUTLASS Hopper FMHA.  The trailing dimension is the single stage.
        sQ = smem.allocate_tensor(
            cutlass.BFloat16,
            q_smem_layout_staged.outer,
            1024,
            swizzle=q_smem_layout_staged.inner,
        )
        sK_mma = smem.allocate_tensor(
            cutlass.BFloat16,
            k_smem_layout_staged.outer,
            1024,
            swizzle=k_smem_layout_staged.inner,
        )
        sV_mma = smem.allocate_tensor(
            cutlass.BFloat16,
            v_smem_layout_staged.outer,
            1024,
            swizzle=v_smem_layout_staged.inner,
        )

        # The first half-warp of the producer resolves each token index and
        # scale pair once.  All packed-byte loader lanes reuse these values,
        # mirroring FlashInfer's page-offset prefetch/reuse strategy.
        sKvPos = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((block_n,), stride=(1,)), 16
        )
        sKScale = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((block_n,), stride=(1,)), 16
        )
        sKZero = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((block_n,), stride=(1,)), 16
        )
        sVScale = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((block_n,), stride=(1,)), 16
        )
        sVZero = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((block_n,), stride=(1,)), 16
        )

        # WGMMA descriptors and register fragments.  ``tidx % 128`` gives the
        # lane inside either warp group; only the consumer group issues MMA.
        mma_tid = tidx % _CONSUMER_THREADS
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tid)
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tid)

        tSsQ = qk_thr_mma.partition_A(sQ)
        tSsK = qk_thr_mma.partition_B(sK_mma)
        tSrQ = qk_thr_mma.make_fragment_A(tSsQ)
        tSrK = qk_thr_mma.make_fragment_B(tSsK)
        tOsV = pv_thr_mma.partition_B(sV_mma)
        tOrV = pv_thr_mma.make_fragment_B(tOsV)

        qk_acc_shape = qk_thr_mma.partition_shape_C((_WGMMA_M, block_n))
        pv_acc_shape = pv_thr_mma.partition_shape_C((_WGMMA_M, head_dim))
        acc_pv = pv_thr_mma.make_fragment_C(pv_acc_shape)
        acc_pv.fill(cutlass.Float32(0.0))

        s_max_layout = cute.make_layout(
            cute.size(_layout_acc_mn(pv_tiled_mma, acc_pv.layout), mode=[0])
        )
        s_max = cute.make_rmem_tensor_like(s_max_layout, cutlass.Float32)
        a_sum = cute.make_rmem_tensor_like(s_max_layout, cutlass.Float32)
        s_max.fill(-cutlass.Float32.inf)
        a_sum.fill(cutlass.Float32(0.0))

        cQK = cute.make_identity_tensor((_WGMMA_M, block_n))
        tQcQK = qk_thr_mma.partition_C(cQK)
        tQcQK_mn = cute.make_tensor(
            tQcQK.iterator, _layout_acc_mn(qk_tiled_mma, tQcQK.layout)
        )

        if is_consumer & (consumer_tid < head_dim):
            for qid in cutlass.range_constexpr(kv_group_num):
                sQ[qid, consumer_tid, 0] = Q[cur_batch, q_head_base + qid, consumer_tid]
        # WGMMA always reads an m64 tile.  The producer group is idle during
        # this one-time Q load, so let it clear padded MHA/GQA rows in parallel
        # with the consumer loading the one/four real query rows.
        if is_loader & (loader_tid < head_dim):
            for qid in cutlass.range_constexpr(kv_group_num, _WGMMA_M):
                sQ[qid, loader_tid, 0] = cutlass.BFloat16(0.0)
        # Generic shared stores become visible to the WGMMA async proxy.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        # Runtime-bounded loop: no cache-size cap and no max-split-derived
        # compile-time tile count.
        tile_start = split_start
        while tile_start < split_stop:
            # -------------------------------------------------------------
            # Producer warp group: token gather and scale/zero prefetch.
            # -------------------------------------------------------------
            if is_loader & (loader_tid < block_n):
                logical_kv = tile_start + loader_tid
                valid_token = logical_kv < split_stop
                kv_pos = cutlass.Int32(0)
                k_scale = cutlass.Float32(0.0)
                k_zero = cutlass.Float32(0.0)
                v_scale = cutlass.Float32(0.0)
                v_zero = cutlass.Float32(0.0)
                if valid_token:
                    kv_pos = cutlass.Int32(kv_indices[kv_start_idx + logical_kv])
                    k_scale = cutlass.Float32(K_sz[kv_pos, cur_kv_head, 0])
                    k_zero = cutlass.Float32(K_sz[kv_pos, cur_kv_head, 1])
                    v_scale = cutlass.Float32(V_sz[kv_pos, cur_kv_head, 0])
                    v_zero = cutlass.Float32(V_sz[kv_pos, cur_kv_head, 1])

                sKvPos[loader_tid] = kv_pos
                sKScale[loader_tid] = k_scale
                sKZero[loader_tid] = k_zero
                sVScale[loader_tid] = v_scale
                sVZero[loader_tid] = v_zero
            cute.arch.barrier()

            # Every producer lane owns a fixed group of packed bytes.  Each
            # byte becomes four BF16 values in the quarter-interleaved OSCAR
            # layout: d4, d4+32, d4+64, d4+96.
            if is_loader:
                # BLOCK_N=64 maps exactly two producer lanes to one token;
                # each lane owns a contiguous 16-byte half-row.  Hoist the
                # gathered position and scale/zero metadata out of the byte
                # loop instead of reloading the same five shared values 16x.
                row = loader_tid // loader_lanes_per_token
                d4_base = (
                    loader_tid % loader_lanes_per_token
                ) * packed_bytes_per_loader
                logical_kv = tile_start + row
                valid_token = logical_kv < split_stop
                kv_pos = sKvPos[row]
                ks = sKScale[row]
                kz = sKZero[row]
                vs = sVScale[row]
                vz = sVZero[row]

                for j in cutlass.range_constexpr(packed_bytes_per_loader):
                    d4 = d4_base + j
                    kp = cutlass.Uint8(0)
                    vp = cutlass.Uint8(0)
                    if valid_token:
                        kp = K_packed[kv_pos, cur_kv_head, d4]
                        vp = V_packed[kv_pos, cur_kv_head, d4]

                    kp_i = cutlass.Int32(kp)
                    vp_i = cutlass.Int32(vp)

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
                    # PV is P[M,K=token] * V[K=token,N=dim], hence V is
                    # stored as logical (dim, token) for WGMMA operand B.
                    sV_mma[d4, row, 0] = cutlass.BFloat16(v0)
                    sV_mma[d4 + quarter_dim, row, 0] = cutlass.BFloat16(v1)
                    sV_mma[d4 + 2 * quarter_dim, row, 0] = cutlass.BFloat16(v2)
                    sV_mma[d4 + 3 * quarter_dim, row, 0] = cutlass.BFloat16(v3)
                cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()

            # -------------------------------------------------------------
            # Consumer warp group: WGMMA QK, online softmax, then WGMMA PV.
            # -------------------------------------------------------------
            if is_consumer:
                acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
                cute.nvgpu.warpgroup.fence()
                _gemm_zero_acc(
                    qk_tiled_mma,
                    qk_tiled_mma_accumulate,
                    tSrQ[(None, None, None, 0)],
                    tSrK[(None, None, None, 0)],
                    acc_qk,
                )
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                acc_qk_mn = cute.make_tensor(
                    acc_qk.iterator,
                    _layout_acc_mn(qk_tiled_mma, acc_qk.layout),
                )
                valid_tokens = split_stop - tile_start
                for row in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[0])):
                    for col in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[1])):
                        if tQcQK_mn[row, col][1] >= valid_tokens:
                            acc_qk_mn[row, col] = -cutlass.Float32.inf

                _online_softmax(
                    acc_qk,
                    qk_tiled_mma,
                    s_max,
                    a_sum,
                    acc_pv,
                    pv_tiled_mma,
                    cutlass.Float32(_LOG2_E) * sm_scale,
                )
                p_operand = _make_acc_into_bf16_operand(
                    acc_qk, pv_tiled_mma.tv_layout_A
                )

                cute.nvgpu.warpgroup.fence()
                _gemm_accumulate(
                    pv_tiled_mma_accumulate,
                    p_operand,
                    tOrV[(None, None, None, 0)],
                    acc_pv,
                )
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

            # Producer must not overwrite the shared operands until every
            # consumer has completed PV.
            cute.arch.barrier()
            tile_start = tile_start + block_n

        # Only runtime-active splits write.  Inactive slots retain the caller's
        # ``-inf`` initialization, matching the Triton stage-1 convention.
        if (split_kv_id < kv_splits) & is_consumer:
            lse = _normalize_softmax(s_max, a_sum, acc_pv, pv_tiled_mma, sm_scale)
            acc_pv_mn = cute.make_tensor(
                acc_pv.iterator,
                _layout_acc_mn(pv_tiled_mma, acc_pv.layout),
            )
            cO = cute.make_identity_tensor((_WGMMA_M, head_dim))
            tOcO = pv_thr_mma.partition_C(cO)
            tOcO_mn = cute.make_tensor(
                tOcO.iterator, _layout_acc_mn(pv_tiled_mma, tOcO.layout)
            )
            for row in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[0])):
                qid = tOcO_mn[row, 0][0]
                if qid < kv_group_num:
                    for col in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[1])):
                        out_dim = tOcO_mn[row, col][1]
                        if out_dim < head_dim:
                            att_out[
                                cur_batch,
                                q_head_base + qid,
                                split_kv_id,
                                out_dim,
                            ] = acc_pv_mn[row, col]
                    if tOcO_mn[row, 0][1] == 0:
                        att_lse[cur_batch, q_head_base + qid, split_kv_id] = lse[row]

    return kernel


def _index_cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.int32:
        return cutlass.Int32
    if dtype == torch.int64:
        return cutlass.Int64
    raise TypeError(f"kv_indices must be int32 or int64, got {dtype}")


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

    kernel = _define_decode_kernel(
        head_dim=head_dim,
        block_n=block_n,
        kv_group_num=kv_group_num,
    )

    # Three single-stage WGMMA operands plus gather/dequant metadata.  The
    # 1024-byte allowance covers allocator padding at swizzled-tensor borders.
    smem_bytes = (
        _WGMMA_M * head_dim * 2
        + block_n * head_dim * 2
        + head_dim * block_n * 2
        + block_n * 4
        + 4 * block_n * 4
        + 1024
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
        qk_mma_tiler = (_WGMMA_M, block_n, head_dim)
        pv_mma_tiler = (_WGMMA_M, head_dim, block_n)
        qk_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            row_major.sm90_mma_major_mode(),
            row_major.sm90_mma_major_mode(),
            cutlass.Float32,
            (1, 1, 1),
            qk_mma_tiler[:2],
        )
        qk_tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
        qk_tiled_mma_accumulate = sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            row_major.sm90_mma_major_mode(),
            row_major.sm90_mma_major_mode(),
            cutlass.Float32,
            (1, 1, 1),
            qk_mma_tiler[:2],
        )
        qk_tiled_mma_accumulate.set(warpgroup.Field.ACCUMULATE, True)
        pv_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            warpgroup.OperandMajorMode.K,
            row_major.sm90_mma_major_mode(),
            cutlass.Float32,
            (1, 1, 1),
            pv_mma_tiler[:2],
            warpgroup.OperandSource.RMEM,
        )
        pv_tiled_mma_accumulate = sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            warpgroup.OperandMajorMode.K,
            row_major.sm90_mma_major_mode(),
            cutlass.Float32,
            (1, 1, 1),
            pv_mma_tiler[:2],
            warpgroup.OperandSource.RMEM,
        )
        pv_tiled_mma_accumulate.set(warpgroup.Field.ACCUMULATE, True)
        q_smem_layout_staged = sm90_utils.make_smem_layout_a(
            row_major, qk_mma_tiler, cutlass.BFloat16, 1
        )
        k_smem_layout_staged = sm90_utils.make_smem_layout_b(
            row_major, qk_mma_tiler, cutlass.BFloat16, 1
        )
        v_smem_layout_staged = sm90_utils.make_smem_layout_b(
            row_major, pv_mma_tiler, cutlass.BFloat16, 1
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
            qk_tiled_mma_accumulate,
            pv_tiled_mma,
            pv_tiled_mma_accumulate,
            q_smem_layout_staged,
            k_smem_layout_staged,
            v_smem_layout_staged,
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

    fake_stream = cuda.CUstream(0)
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
        stream=fake_stream,
    )
    _compiled_decode_kernels[key] = compiled
    return compiled


def can_use_flashinfer_cutedsl_decode(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scales_zeros: torch.Tensor,
    v_scales_zeros: torch.Tensor,
) -> bool:
    """Return whether the fused SM90 kernel covers this tensor contract."""

    if not q.is_cuda:
        return False
    try:
        if torch.cuda.get_device_capability(q.device)[0] != 9:
            return False
    except Exception:
        return False

    if q.ndim != 3 or k_buffer.ndim != 3 or v_buffer.ndim != 3:
        return False
    if k_scales_zeros.ndim != 3 or v_scales_zeros.ndim != 3:
        return False
    if q.dtype != torch.bfloat16:
        return False
    if k_buffer.dtype != torch.uint8 or v_buffer.dtype != torch.uint8:
        return False
    if k_scales_zeros.dtype != torch.float32 or v_scales_zeros.dtype != torch.float32:
        return False

    head_dim = k_buffer.shape[-1] * 4
    if head_dim != 128 or q.shape[-1] != head_dim:
        return False
    if v_buffer.shape[-1] * 4 != head_dim:
        return False
    if k_scales_zeros.shape[-1] != 2 or v_scales_zeros.shape[-1] != 2:
        return False
    if k_buffer.shape[:2] != v_buffer.shape[:2]:
        return False
    if k_scales_zeros.shape[:2] != k_buffer.shape[:2]:
        return False
    if v_scales_zeros.shape[:2] != v_buffer.shape[:2]:
        return False

    kv_heads = k_buffer.shape[1]
    if kv_heads <= 0 or q.shape[1] % kv_heads != 0:
        return False
    # Validated targets: MHA and Qwen3-4B-style 32Q/8KV GQA.
    if q.shape[1] // kv_heads not in (1, 4):
        return False

    tensors = (q, k_buffer, v_buffer, k_scales_zeros, v_scales_zeros)
    if any(t.device != q.device for t in tensors):
        return False
    # Compilation bakes in compact input strides.  Output strides remain
    # explicit so sliced unified-stage scratch is supported.
    if any(not t.is_contiguous() for t in tensors):
        return False
    return True


def flashinfer_cutedsl_decode_attention_fwd_int2(
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
    """Run fused INT2-dequant decode attention stage 1.

    The signature and output contract match
    ``cutedsl_decode_attention_fwd_int2``.  Reduction across splits remains in
    SGLang's existing stage-2 kernel.
    """

    if not can_use_flashinfer_cutedsl_decode(
        q, k_buffer, v_buffer, k_scales_zeros, v_scales_zeros
    ):
        raise ValueError(
            "FlashInfer-style CuTeDSL INT2 decode requires SM90, BF16 Q, "
            "contiguous uint8 INT2 K/V with head_dim=128, one FP32 "
            "scale/zero pair, and MHA or GQA group size 4"
        )
    if max_kv_splits <= 0:
        raise ValueError(f"max_kv_splits must be positive, got {max_kv_splits}")

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

    all_tensors = (
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
    if any(t.device != q.device for t in all_tensors):
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
        _cached_from_dlpack(q, assumed_align=16),
        _cached_from_dlpack(k_buffer, assumed_align=16),
        _cached_from_dlpack(v_buffer, assumed_align=16),
        _cached_from_dlpack(k_scales_zeros, assumed_align=8),
        _cached_from_dlpack(v_scales_zeros, assumed_align=8),
        _cached_from_dlpack(kv_indptr, assumed_align=16),
        _cached_from_dlpack(kv_indices, assumed_align=16),
        _cached_from_dlpack(num_kv_splits, assumed_align=16),
        _cached_from_dlpack(att_out, assumed_align=16),
        _cached_from_dlpack(att_lse, assumed_align=16),
        cutlass.Float32(sm_scale),
        stream,
    )
