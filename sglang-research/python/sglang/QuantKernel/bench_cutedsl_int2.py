"""Benchmark INT2 KV cache quantize + complete decode implementations.

Defaults match the validated Qwen3-4B Thinking eval config
(``head_dim=128``, ``group_size==head_dim`` per-row scale/zero).

Decode comparison keeps Triton, the legacy SIMT CuTeDSL kernel, and the
FlashInfer-derived fused-dequant CuTeDSL kernel in the same run. Every timing
includes both stage 1 and split-KV reduction.

Exit code 0 on speedup >= ``--assert-speedup``; non-zero otherwise.
"""

import argparse
import math
import sys
from typing import Dict, Sequence

import torch
import triton

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _decode_softmax_reducev_fwd,
    decode_attention_fwd_grouped_quant_int2,
    decode_attention_fwd_normal_quant_int2,
)
from sglang.srt.mem_cache.kv_quant_kernels import _launch_quantize_int2
from sglang.QuantKernel.cutedsl_int2_kv import (
    _launch_quantize_one,
    cuda_decode_attention_fwd_int2,
    cutedsl_decode_attention_fwd_int2,
    _load_cuda_decode_extension,
    get_cuda_invocation_counters,
    reset_cuda_invocation_counters,
)


TRITON_BACKEND = "triton"
LEGACY_CUTEDSL_BACKEND = "cutedsl"
FLASHINFER_CUTEDSL_BACKEND = "flashinfer-cutedsl"
COMPARE_BACKENDS = (
    TRITON_BACKEND,
    LEGACY_CUTEDSL_BACKEND,
    FLASHINFER_CUTEDSL_BACKEND,
)
ALL_BACKENDS = COMPARE_BACKENDS + ("cuda", "cuda-wgmma")
BACKEND_LABELS = {
    TRITON_BACKEND: "Triton INT2",
    LEGACY_CUTEDSL_BACKEND: "Legacy CuteDSL",
    FLASHINFER_CUTEDSL_BACKEND: "FI-derived CuteDSL",
    "cuda": "CUDA C++ wmma",
    "cuda-wgmma": "CUDA C++ wgmma",
}


# Optional FlashInfer fp16 reference (skip if not importable).
_HAS_FLASHINFER = False
try:
    import flashinfer  # noqa: F401
    _HAS_FLASHINFER = True
except Exception:
    pass


# H100 HBM3 peak DRAM bandwidth used for utilization percentages.
H100_HBM3_PEAK_GBPS = 3350.0

_BENCH_WARMUP = 25
_BENCH_REP = 200


def _bench(fn, *, warmup=None, rep=None):
    warmup = _BENCH_WARMUP if warmup is None else warmup
    rep = _BENCH_REP if rep is None else rep
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def _expand_backend_selection(selection: str) -> tuple[str, ...]:
    """Expand a CLI backend selection into concrete decode implementations.

    A single optimized backend is always paired with Triton so its timing and
    accuracy have a reference from the same run. ``compare`` is the primary
    three-way comparison requested by this benchmark; ``all`` additionally
    includes the two CUDA C++ experimental kernels already supported here.
    """
    if selection == "compare":
        return COMPARE_BACKENDS
    if selection == "all":
        return ALL_BACKENDS
    if selection == TRITON_BACKEND:
        return (TRITON_BACKEND,)
    return (TRITON_BACKEND, selection)


def _get_stage1_fn(backend: str):
    """Resolve a non-Triton stage-1 implementation lazily.

    Lazy imports keep legacy-only benchmarking usable when an optional kernel
    module is unavailable and avoid compiling the CUDA extension unless it was
    explicitly selected.
    """
    if backend == LEGACY_CUTEDSL_BACKEND:
        return cutedsl_decode_attention_fwd_int2
    if backend == FLASHINFER_CUTEDSL_BACKEND:
        from sglang.QuantKernel.flashinfer_cutedsl_int2_decode import (
            flashinfer_cutedsl_decode_attention_fwd_int2,
        )

        return flashinfer_cutedsl_decode_attention_fwd_int2
    if backend == "cuda":
        return cuda_decode_attention_fwd_int2
    if backend == "cuda-wgmma":
        from sglang.QuantKernel.cutedsl_int2_kv import (
            cuda_decode_attention_fwd_int2_wgmma,
        )

        return cuda_decode_attention_fwd_int2_wgmma
    raise ValueError(f"{backend!r} is not a stage-1 backend")


def _allocate_decode_state(
    batch: int,
    q_heads: int,
    head_dim: int,
    max_splits: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    return {
        "out": torch.empty(
            batch, q_heads, head_dim, dtype=torch.float32, device=device
        ),
        "output_lse": torch.empty(
            batch, q_heads, dtype=torch.float32, device=device
        ),
        "split_out": torch.empty(
            batch,
            q_heads,
            max_splits,
            head_dim,
            dtype=torch.float32,
            device=device,
        ),
        "split_lse": torch.full(
            (batch, q_heads, max_splits),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        ),
    }


def _allocate_sliced_decode_state(
    batch: int,
    q_heads: int,
    head_dim: int,
    max_splits: int,
    device: str,
    prefix_splits: int = 8,
) -> Dict[str, torch.Tensor]:
    """Allocate quant scratch as a non-contiguous slice of unified scratch.

    ``prefix_splits=8`` mirrors the default HP tier split reservation.  The
    returned stage-1 views have the production unified-path strides and an
    aligned storage offset, while the backing tensors stay alive in the state.
    """
    total_splits = prefix_splits + max_splits
    combined_out = torch.empty(
        batch,
        q_heads,
        total_splits,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    combined_lse = torch.full(
        (batch, q_heads, total_splits),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    split_out = combined_out[:, :, prefix_splits:, :]
    split_lse = combined_lse[:, :, prefix_splits:]
    if split_out.is_contiguous() or split_lse.is_contiguous():
        raise AssertionError("unified quant scratch views must be non-contiguous")
    return {
        "out": torch.empty(
            batch, q_heads, head_dim, dtype=torch.float32, device=device
        ),
        "output_lse": torch.empty(
            batch, q_heads, dtype=torch.float32, device=device
        ),
        "split_out": split_out,
        "split_lse": split_lse,
        "combined_out": combined_out,
        "combined_lse": combined_lse,
    }


def _make_full_decode_runner(
    backend: str,
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    k_sz: torch.Tensor,
    v_sz: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_splits: int,
    sm_scale: float,
    state: Dict[str, torch.Tensor],
):
    """Create a closure that performs the same complete decode for each backend.

    Triton's public wrappers already perform stage 1 and split reduction. The
    optimized backends expose the stage-1-compatible API, so the exact same
    Triton reduction is appended here. This fixes the historical benchmark in
    which Triton was timed for two stages while CuteDSL was timed for one.
    """
    if k_packed.shape[1] == q.shape[1]:
        triton_fn = decode_attention_fwd_normal_quant_int2
    else:
        triton_fn = decode_attention_fwd_grouped_quant_int2

    if backend == TRITON_BACKEND:

        def run_triton():
            triton_fn(
                q,
                k_packed,
                v_packed,
                k_sz,
                v_sz,
                state["out"],
                kv_indptr,
                kv_indices,
                state["split_out"],
                state["split_lse"],
                num_kv_splits,
                max_splits,
                sm_scale=sm_scale,
                output_lse=state["output_lse"],
            )

        return run_triton

    stage1_fn = _get_stage1_fn(backend)

    def run_optimized():
        stage1_fn(
            q,
            k_packed,
            v_packed,
            k_sz,
            v_sz,
            state["split_out"],
            state["split_lse"],
            kv_indptr,
            kv_indices,
            num_kv_splits,
            max_splits,
            sm_scale,
        )
        _decode_softmax_reducev_fwd(
            state["split_out"],
            state["split_lse"],
            q,
            state["out"],
            v_scale=1.0,
            v_buffer=state["out"],
            kv_indptr=kv_indptr,
            num_kv_splits=num_kv_splits,
            max_kv_splits=max_splits,
            output_lse=state["output_lse"],
        )

    return run_optimized


def bench_quantize(num_tokens: int, num_heads: int, head_dim: int):
    device = "cuda"
    x = torch.randn(num_tokens, num_heads, head_dim,
                    dtype=torch.bfloat16, device=device)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=device)
    cache = torch.empty(num_tokens, num_heads, head_dim // 4,
                        dtype=torch.uint8, device=device)
    sz = torch.empty(num_tokens, num_heads, 2,
                     dtype=torch.float32, device=device)
    x2 = torch.randn_like(x)
    cache2 = torch.empty_like(cache)
    sz2 = torch.empty_like(sz)

    def run_tri():
        _launch_quantize_int2(x, loc, cache, sz, None)
        _launch_quantize_int2(x2, loc, cache2, sz2, None)

    def run_dsl():
        _launch_quantize_one(x, loc, cache, sz, None)
        _launch_quantize_one(x2, loc, cache2, sz2, None)

    run_tri(); run_dsl(); torch.cuda.synchronize()
    tri_ms = _bench(run_tri)
    dsl_ms = _bench(run_dsl)
    bytes_read = x.element_size() * x.numel()
    bytes_written = cache.numel() + sz.element_size() * sz.numel()
    total_bytes = bytes_read + bytes_written
    tri_gbps = total_bytes / (tri_ms * 1e-3) / 1e9
    dsl_gbps = total_bytes / (dsl_ms * 1e-3) / 1e9
    return tri_ms, dsl_ms, tri_gbps, dsl_gbps


def _bench_flashinfer_decode(
    batch: int, q_heads: int, kv_heads: int, head_dim: int,
    seq_len: int,
):
    """Run a fp16 single-query decode through FlashInfer for the same shape.

    Returns (ms, gbps) where gbps uses the fp16 K+V bytes touched."""
    if not _HAS_FLASHINFER:
        return None, None
    try:
        import flashinfer
        device = "cuda"
        dtype = torch.float16

        # Paged KV cache, page_size=1, NHD layout. Indices arrange the
        # ``seq_len`` pages contiguously for each request.
        page_size = 1
        num_pages = batch * seq_len
        kv_cache = torch.randn(
            num_pages, 2, kv_heads, page_size, head_dim,
            dtype=dtype, device=device,
        )
        kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        kv_indptr = torch.arange(
            0, num_pages + seq_len, seq_len, dtype=torch.int32, device=device,
        )[: batch + 1]
        kv_last_page_len = torch.full(
            (batch,), page_size, dtype=torch.int32, device=device,
        )
        q = torch.randn(batch, q_heads, head_dim, dtype=dtype, device=device)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            q_heads, kv_heads, head_dim, page_size,
            q_data_type=dtype, kv_data_type=dtype,
        )

        def run():
            wrapper.run(q, kv_cache)

        run(); torch.cuda.synchronize()
        ms = _bench(run)
        fp16_bytes = (
            batch * seq_len * kv_heads * head_dim * 2 * 2  # fp16 K + V
        )
        gbps = fp16_bytes / (ms * 1e-3) / 1e9
        return ms, gbps
    except Exception:
        # FlashInfer's BatchDecodeWithPagedKVCacheWrapper currently has no
        # dispatch for q_heads=32 / kv_heads=8 (group=4) at head_dim=128 in
        # this build — error is reported as ``Unsupported group_size: 32``.
        # Treat as "no reference available" instead of failing the bench.
        return None, None


def bench_decode(
    batch: int, q_heads: int, kv_heads: int, head_dim: int,
    seq_len: int, max_splits: int,
    backends: Sequence[str] = COMPARE_BACKENDS,
):
    device = "cuda"
    dtype = torch.bfloat16

    # Give every request a disjoint physical KV range.  Repeating one range
    # across a high-batch workload lets later CTAs hit L2 while the bandwidth
    # accounting still multiplies by ``batch``, which substantially overstates
    # production throughput and can distort backend comparisons.
    num_tokens = batch * seq_len
    k = torch.randn(num_tokens, kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(num_tokens, kv_heads, head_dim, dtype=dtype, device=device)
    k_packed = torch.empty(num_tokens, kv_heads, head_dim // 4,
                           dtype=torch.uint8, device=device)
    v_packed = torch.empty_like(k_packed)
    k_sz = torch.empty(num_tokens, kv_heads, 2, dtype=torch.float32, device=device)
    v_sz = torch.empty_like(k_sz)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=device)
    # Populate request-sized chunks so high-batch setup mirrors independent
    # cache insertion and does not make one million-program launch dominate
    # benchmark startup.  These setup launches are outside the timed region.
    for token_start in range(0, num_tokens, seq_len):
        token_stop = token_start + seq_len
        token_locs = loc[token_start:token_stop]
        _launch_quantize_int2(
            k[token_start:token_stop], token_locs, k_packed, k_sz, None
        )
        _launch_quantize_int2(
            v[token_start:token_stop], token_locs, v_packed, v_sz, None
        )

    q = torch.randn(batch, q_heads, head_dim, dtype=dtype, device=device)
    kv_indptr = torch.arange(0, (batch + 1) * seq_len, seq_len,
                              dtype=torch.int32, device=device)
    kv_indices = loc
    num_kv_splits = torch.full((batch,), max_splits, dtype=torch.int32, device=device)
    sm_scale = 1.0 / (head_dim ** 0.5)

    states = {
        backend: _allocate_decode_state(
            batch, q_heads, head_dim, max_splits, device
        )
        for backend in backends
    }
    runners = {
        backend: _make_full_decode_runner(
            backend,
            q,
            k_packed,
            v_packed,
            k_sz,
            v_sz,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            max_splits,
            sm_scale,
            states[backend],
        )
        for backend in backends
    }

    for backend in backends:
        runners[backend]()
    torch.cuda.synchronize()
    elapsed_ms = {backend: _bench(runners[backend]) for backend in backends}

    # Effective INT2 bytes touched (packed K + V + scale/zero for active heads).
    int2_bytes = (
        batch * seq_len * kv_heads * (head_dim // 4) * 2  # packed K+V
        + batch * seq_len * kv_heads * 2 * 4 * 2          # scale/zero (fp32)
    )
    effective_gbps = {
        backend: int2_bytes / (ms * 1e-3) / 1e9
        for backend, ms in elapsed_ms.items()
    }

    fi_ms, fi_gbps = _bench_flashinfer_decode(batch, q_heads, kv_heads,
                                               head_dim, seq_len)
    return elapsed_ms, effective_gbps, fi_ms, fi_gbps


def _decode_shape_list(gqa: bool, quick: bool = False):
    """Return [(batch, q_heads, kv_heads, seq, max_splits), ...].

    GQA shapes mirror Qwen3-4B Thinking production (q=32, kv=8, group=4).
    MHA shapes use q=kv=8 (CuteDSL's historical default).
    Production GPQA serves at batch=32 (MAX_RUNNING) — the high-batch
    shape exercises the GQA-coalesced grid where the kernel has many
    CTAs/SM and amortizes barrier overhead.
    """
    if gqa:
        shapes = [
            (1, 32, 8, 4096, 4),
            (1, 32, 8, 8192, 8),
            (1, 32, 8, 16384, 8),
            (1, 32, 8, 20000, 8),
            (32, 32, 8, 4096, 4),  # production max-batch shape
        ]
    else:
        shapes = [
            (1, 8, 8, 4096, 4),
            (1, 8, 8, 8192, 8),
            (1, 8, 8, 16384, 8),
            (1, 8, 8, 20000, 8),
        ]
    return shapes[:1] if quick else shapes


def _make_correctness_case(
    q_heads: int,
    kv_heads: int,
    seq_lens: Sequence[int],
    runtime_splits: Sequence[int],
    max_splits: int,
    head_dim: int,
    seed: int,
    kv_indices_dtype: torch.dtype = torch.int32,
):
    """Build an indirect, variable-length decode case.

    The physical cache is deliberately larger than the logical KV set and the
    indices are shuffled. The small amount of padding also gives the legacy
    kernel enough compiled tiles when ``runtime_splits < max_splits``; without
    it, that kernel's historical cache-size-based loop bound can truncate the
    test before the backend comparison is reached.
    """
    if len(seq_lens) != len(runtime_splits):
        raise ValueError("seq_lens and runtime_splits must have equal length")
    if not seq_lens or any(seq <= 0 for seq in seq_lens):
        raise ValueError(f"all sequence lengths must be positive, got {seq_lens}")
    if any(split <= 0 or split > max_splits for split in runtime_splits):
        raise ValueError(
            f"runtime splits must be in [1, {max_splits}], got {runtime_splits}"
        )

    device = "cuda"
    dtype = torch.bfloat16
    batch = len(seq_lens)
    total_kv = sum(seq_lens)
    max_tokens_per_split = max(
        math.ceil(seq / split)
        for seq, split in zip(seq_lens, runtime_splits)
    )
    cache_size = max(total_kv + 17, max_splits * max_tokens_per_split)

    torch.manual_seed(seed)
    k = torch.randn(cache_size, kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    k_packed = torch.empty(
        cache_size, kv_heads, head_dim // 4, dtype=torch.uint8, device=device
    )
    v_packed = torch.empty_like(k_packed)
    k_sz = torch.empty(
        cache_size, kv_heads, 2, dtype=torch.float32, device=device
    )
    v_sz = torch.empty_like(k_sz)
    loc = torch.arange(cache_size, dtype=torch.int32, device=device)
    _launch_quantize_int2(k, loc, k_packed, k_sz, None)
    _launch_quantize_int2(v, loc, v_packed, v_sz, None)

    q = torch.randn(batch, q_heads, head_dim, dtype=dtype, device=device)
    indptr_host = [0]
    for seq in seq_lens:
        indptr_host.append(indptr_host[-1] + seq)
    kv_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)
    kv_indices = torch.randperm(cache_size, device=device)[:total_kv]
    kv_indices = kv_indices.to(kv_indices_dtype).contiguous()
    num_kv_splits = torch.tensor(
        runtime_splits, dtype=torch.int32, device=device
    )

    return {
        "q": q,
        "k_packed": k_packed,
        "v_packed": v_packed,
        "k_sz": k_sz,
        "v_sz": v_sz,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "num_kv_splits": num_kv_splits,
        "max_splits": max_splits,
        "sm_scale": 1.0 / math.sqrt(head_dim),
        "label": (
            f"B={batch} Hq={q_heads} Hk={kv_heads} "
            f"S={list(seq_lens)} split={list(runtime_splits)}/{max_splits}"
        ),
    }


def _run_flashinfer_unified_contract_case(
    *,
    head_dim: int,
    out_atol: float,
    out_rtol: float,
    lse_atol: float,
    lse_rtol: float,
):
    """Exercise the new wrapper with production mixed-KV metadata/scratch.

    This intentionally imports and calls the FlashInfer-derived wrapper
    directly.  It cannot pass through the environment dispatch or its Triton
    fallback, so a successful comparison proves that the int64-index and
    non-contiguous unified-scratch specialization itself executed.
    """
    from sglang.QuantKernel.flashinfer_cutedsl_int2_decode import (
        flashinfer_cutedsl_decode_attention_fwd_int2,
    )

    q_heads, kv_heads = 32, 8
    seq_lens, runtime_splits, max_splits = (65, 37), (3, 2), 4
    case = _make_correctness_case(
        q_heads,
        kv_heads,
        seq_lens,
        runtime_splits,
        max_splits,
        head_dim,
        seed=0xC0FFEE + 0x100,
        kv_indices_dtype=torch.int64,
    )
    batch = len(seq_lens)
    ref_state = _allocate_decode_state(
        batch, q_heads, head_dim, max_splits, "cuda"
    )
    opt_state = _allocate_sliced_decode_state(
        batch, q_heads, head_dim, max_splits, "cuda"
    )
    ref_runner = _make_full_decode_runner(
        TRITON_BACKEND,
        case["q"],
        case["k_packed"],
        case["v_packed"],
        case["k_sz"],
        case["v_sz"],
        case["kv_indptr"],
        case["kv_indices"],
        case["num_kv_splits"],
        case["max_splits"],
        case["sm_scale"],
        ref_state,
    )

    def run_direct_wrapper():
        flashinfer_cutedsl_decode_attention_fwd_int2(
            case["q"],
            case["k_packed"],
            case["v_packed"],
            case["k_sz"],
            case["v_sz"],
            opt_state["split_out"],
            opt_state["split_lse"],
            case["kv_indptr"],
            case["kv_indices"],
            case["num_kv_splits"],
            case["max_splits"],
            case["sm_scale"],
        )
        _decode_softmax_reducev_fwd(
            opt_state["split_out"],
            opt_state["split_lse"],
            case["q"],
            opt_state["out"],
            v_scale=1.0,
            v_buffer=opt_state["out"],
            kv_indptr=case["kv_indptr"],
            num_kv_splits=case["num_kv_splits"],
            max_kv_splits=case["max_splits"],
            output_lse=opt_state["output_lse"],
        )

    ref_runner()
    run_direct_wrapper()
    torch.cuda.synchronize()

    ref_out = ref_state["out"].float()
    ref_lse = ref_state["output_lse"].float()
    out = opt_state["out"].float()
    output_lse = opt_state["output_lse"].float()
    cos = torch.nn.functional.cosine_similarity(
        ref_out.flatten(), out.flatten(), dim=0
    ).item()
    out_max_abs = (ref_out - out).abs().max().item()
    lse_max_abs = (ref_lse - output_lse).abs().max().item()
    close_out = _is_close(out, ref_out, atol=out_atol, rtol=out_rtol)
    close_lse = _is_close(
        output_lse, ref_lse, atol=lse_atol, rtol=lse_rtol
    )
    label = case["label"] + " int64+sliced-unified"
    return label, cos, out_max_abs, lse_max_abs, close_out and close_lse


def _is_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> bool:
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return True
    except AssertionError:
        return False


def _test_correctness(
    backends: Sequence[str],
    head_dim: int = 128,
    cos_threshold: float = 0.999,
    out_atol: float = 0.05,
    out_rtol: float = 0.05,
    lse_atol: float = 0.05,
    lse_rtol: float = 0.01,
    quick: bool = False,
):
    """Compare complete decode outputs and final LSE against Triton.

    Cases cover MHA, GQA, irregular sequence lengths, per-request runtime split
    counts smaller than the compiled maximum, and shuffled physical KV slots.
    """
    cases = [
        # Irregular MHA and runtime_splits < max_splits.
        (8, 8, (33,), (2,), 4),
        # Variable-length GQA with different runtime split counts per request.
        (32, 8, (65, 37), (3, 2), 4),
    ]
    if not quick:
        cases.append((32, 8, (4097,), (8,), 8))

    use_counter = any(b in ("cuda", "cuda-wgmma") for b in backends)
    if use_counter:
        reset_cuda_invocation_counters()
        pre_w, pre_g = get_cuda_invocation_counters()
    else:
        pre_w, pre_g = 0, 0

    print("\n=== INT2 complete decode correctness ===")
    print("Backends: " + ", ".join(BACKEND_LABELS[b] for b in backends))
    print(
        f"{'shape':<53s} {'backend':<20s} {'cos':>9s} "
        f"{'out|max|':>10s} {'lse|max|':>10s} {'PASS?':>7s}"
    )

    min_cos = 1.0
    failed = False
    for case_idx, (q_heads, kv_heads, seq_lens, splits, max_splits) in enumerate(
        cases
    ):
        case = _make_correctness_case(
            q_heads,
            kv_heads,
            seq_lens,
            splits,
            max_splits,
            head_dim,
            seed=0xC0FFEE + case_idx,
        )
        batch = len(seq_lens)
        states = {
            backend: _allocate_decode_state(
                batch, q_heads, head_dim, max_splits, "cuda"
            )
            for backend in backends
        }
        runners = {}
        for backend in backends:
            try:
                runners[backend] = _make_full_decode_runner(
                    backend,
                    case["q"],
                    case["k_packed"],
                    case["v_packed"],
                    case["k_sz"],
                    case["v_sz"],
                    case["kv_indptr"],
                    case["kv_indices"],
                    case["num_kv_splits"],
                    case["max_splits"],
                    case["sm_scale"],
                    states[backend],
                )
            except Exception as exc:
                failed = True
                min_cos = min(min_cos, -1.0)
                print(
                    f"{case['label']:<53s} {BACKEND_LABELS[backend]:<20s} "
                    f"setup error: {type(exc).__name__}: {exc}"
                )

        if TRITON_BACKEND not in runners:
            print("FAIL: Triton reference could not be constructed")
            return float("nan"), 1

        completed = []
        for backend in backends:
            if backend not in runners:
                continue
            try:
                runners[backend]()
                torch.cuda.synchronize()
                completed.append(backend)
            except Exception as exc:
                failed = True
                min_cos = min(min_cos, -1.0)
                print(
                    f"{case['label']:<53s} {BACKEND_LABELS[backend]:<20s} "
                    f"run error: {type(exc).__name__}: {exc}"
                )

        if TRITON_BACKEND not in completed:
            print("FAIL: Triton reference execution failed")
            return float("nan"), 1

        ref_out = states[TRITON_BACKEND]["out"].float()
        ref_lse = states[TRITON_BACKEND]["output_lse"].float()
        for backend in completed:
            if backend == TRITON_BACKEND:
                print(
                    f"{case['label']:<53s} {BACKEND_LABELS[backend]:<20s} "
                    f"{1.0:>9.6f} {0.0:>10.3e} {0.0:>10.3e} {'REF':>7s}"
                )
                continue

            out = states[backend]["out"].float()
            output_lse = states[backend]["output_lse"].float()
            cos = torch.nn.functional.cosine_similarity(
                ref_out.flatten(), out.flatten(), dim=0
            ).item()
            out_max_abs = (ref_out - out).abs().max().item()
            lse_max_abs = (ref_lse - output_lse).abs().max().item()
            close_out = _is_close(
                out, ref_out, atol=out_atol, rtol=out_rtol
            )
            close_lse = _is_close(
                output_lse, ref_lse, atol=lse_atol, rtol=lse_rtol
            )
            passed = cos >= cos_threshold and close_out and close_lse
            min_cos = min(min_cos, cos if math.isfinite(cos) else -1.0)
            failed |= not passed
            print(
                f"{case['label']:<53s} {BACKEND_LABELS[backend]:<20s} "
                f"{cos:>9.6f} {out_max_abs:>10.3e} {lse_max_abs:>10.3e} "
                f"{'PASS' if passed else 'FAIL':>7s}"
            )

    if FLASHINFER_CUTEDSL_BACKEND in backends:
        try:
            label, cos, out_max_abs, lse_max_abs, close = (
                _run_flashinfer_unified_contract_case(
                    head_dim=head_dim,
                    out_atol=out_atol,
                    out_rtol=out_rtol,
                    lse_atol=lse_atol,
                    lse_rtol=lse_rtol,
                )
            )
            passed = cos >= cos_threshold and close
            min_cos = min(min_cos, cos if math.isfinite(cos) else -1.0)
            failed |= not passed
            print(
                f"{label:<53s} {BACKEND_LABELS[FLASHINFER_CUTEDSL_BACKEND]:<20s} "
                f"{cos:>9.6f} {out_max_abs:>10.3e} {lse_max_abs:>10.3e} "
                f"{'PASS' if passed else 'FAIL':>7s}"
            )
        except Exception as exc:
            failed = True
            min_cos = min(min_cos, -1.0)
            print(
                f"{'int64+sliced-unified direct wrapper':<53s} "
                f"{BACKEND_LABELS[FLASHINFER_CUTEDSL_BACKEND]:<20s} "
                f"run error: {type(exc).__name__}: {exc}"
            )

    if use_counter:
        post_w, post_g = get_cuda_invocation_counters()
        d_w, d_g = post_w - pre_w, post_g - pre_g
        print("\nCUDA invocation counters delta over this test:")
        print(f"  wmma  launches: {d_w}")
        print(f"  wgmma launches: {d_g}")
        if "cuda" in backends and d_w <= 0:
            print("FAIL: CUDA wmma backend selected but its counter did not advance")
            failed = True
        if "cuda-wgmma" in backends and d_g <= 0:
            print("FAIL: CUDA wgmma backend selected but its counter did not advance")
            failed = True

    if failed:
        print(f"\nFAIL: minimum final-output cosine similarity = {min_cos:.6f}")
        return min_cos, 1
    print(f"\nPASS: minimum final-output cosine similarity = {min_cos:.6f}")
    return min_cos, 0


def main():
    global _BENCH_WARMUP, _BENCH_REP

    p = argparse.ArgumentParser()
    p.add_argument("--assert-speedup", type=float, default=None,
                   help="Fail if the primary selected backend's minimum "
                        "speedup over Triton is below this value.")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--gqa", action="store_true",
                   help="Use Qwen3-4B GQA shapes (q=32, kv=8, group=4).")
    p.add_argument("--quantize-only", action="store_true",
                   help="Skip decode benchmark.")
    p.add_argument(
        "--backend",
        choices=tuple(BACKEND_LABELS) + ("compare", "all"),
        default="compare",
        help=(
            "Single optimized backend (automatically paired with Triton), "
            "'compare' for Triton + legacy CuteDSL + FI-derived CuteDSL, or "
            "'all' to additionally include CUDA wmma/wgmma."
        ),
    )
    p.add_argument("--test-correctness", action="store_true",
                   help="Compare complete output and LSE against Triton for "
                        "the selected backend set (no benchmark).")
    p.add_argument("--dump-cosine-sim", action="store_true",
                   help="Alias for --test-correctness; prints cos_sim line "
                        "the harness reads.")
    p.add_argument("--cos-threshold", type=float, default=0.999,
                   help="Cosine-sim PASS threshold for --test-correctness.")
    p.add_argument("--out-atol", type=float, default=0.05)
    p.add_argument("--out-rtol", type=float, default=0.05)
    p.add_argument("--lse-atol", type=float, default=0.05)
    p.add_argument("--lse-rtol", type=float, default=0.01)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Use the short correctness/performance matrix and fewer benchmark reps.",
    )
    p.add_argument("--warmup", type=int, default=None,
                   help="Benchmark warmup budget in ms (default: 25, quick: 3).")
    p.add_argument("--rep", type=int, default=None,
                   help="Benchmark measurement budget in ms (default: 200, quick: 10).")
    args = p.parse_args()

    _BENCH_WARMUP = args.warmup if args.warmup is not None else (3 if args.quick else 25)
    _BENCH_REP = args.rep if args.rep is not None else (10 if args.quick else 200)
    backends = _expand_backend_selection(args.backend)
    if any(backend in ("cuda", "cuda-wgmma") for backend in backends):
        _load_cuda_decode_extension()

    if args.test_correctness or args.dump_cosine_sim:
        min_cos, rc = _test_correctness(
            backends,
            args.head_dim,
            args.cos_threshold,
            out_atol=args.out_atol,
            out_rtol=args.out_rtol,
            lse_atol=args.lse_atol,
            lse_rtol=args.lse_rtol,
            quick=args.quick,
        )
        # Emit a machine-readable line so an external harness can grep it.
        print(f"\ncosine_sim_min: {min_cos:.6f}")
        sys.exit(rc)

    head_dim = args.head_dim

    print(f"\n=== INT2 quantize benchmark (head_dim={head_dim}) ===")
    print(f"{'shape':<30s} {'Triton ms':>10s} {'CuteDSL ms':>11s} "
          f"{'Tri GB/s':>10s} {'DSL GB/s':>10s} {'speedup':>9s}")
    q_min_speedup = float("inf")
    quant_shapes = [
        (4096, 8), (8192, 8), (16384, 8),
    ]
    if args.quick:
        quant_shapes = quant_shapes[:1]
    for (num_tokens, num_heads) in quant_shapes:
        tri, dsl, tg, dg = bench_quantize(num_tokens, num_heads, head_dim)
        sp = tri / dsl if dsl > 0 else float("inf")
        q_min_speedup = min(q_min_speedup, sp)
        print(f"  T={num_tokens:>5d} H={num_heads:>2d}{'':<14s} "
              f"{tri:>10.4f} {dsl:>11.4f} {tg:>10.1f} {dg:>10.1f} {sp:>8.2f}x")

    if args.quantize_only:
        print(f"\nmin quantize speedup: {q_min_speedup:.4f}x")
        return

    shapes = _decode_shape_list(args.gqa, quick=args.quick)
    tag = "GQA (q=32,kv=8)" if args.gqa else "MHA (q=8,kv=8)"
    print(f"\n=== INT2 decode attention benchmark "
          f"(head_dim={head_dim}, {tag}, complete stage1+stage2) ===")
    print("Backends: " + ", ".join(BACKEND_LABELS[b] for b in backends))

    print(
        f"{'shape':<38s} {'backend':<22s} {'ms':>10s} "
        f"{'eff GB/s':>11s} {'vs Triton':>11s}"
    )
    min_speedup = {
        backend: float("inf")
        for backend in backends
        if backend != TRITON_BACKEND
    }
    last_gbps = None
    last_fi_gbps = None
    last_seq = None
    for (batch, q_heads, kv_heads, seq, splits) in shapes:
        elapsed_ms, effective_gbps, fi_ms, fi_gbps = bench_decode(
            batch, q_heads, kv_heads, head_dim, seq, splits,
            backends=backends,
        )
        tri_ms = elapsed_ms[TRITON_BACKEND]
        shape_label = f"B={batch} Hq={q_heads} Hk={kv_heads} S={seq}"
        for row_idx, backend in enumerate(backends):
            ms = elapsed_ms[backend]
            speedup = tri_ms / ms if ms > 0 else float("inf")
            if backend != TRITON_BACKEND:
                min_speedup[backend] = min(min_speedup[backend], speedup)
            print(
                f"{shape_label if row_idx == 0 else '':<38s} "
                f"{BACKEND_LABELS[backend]:<22s} {ms:>10.4f} "
                f"{effective_gbps[backend]:>11.1f} {speedup:>10.2f}x"
            )
        if fi_ms is not None:
            # This reference moves fp16 bytes, so its effective bandwidth is
            # intentionally labeled separately from the INT2 rows above.
            print(
                f"{'':<38s} {'FlashInfer fp16 ref':<22s} {fi_ms:>10.4f} "
                f"{fi_gbps:>11.1f} {tri_ms / fi_ms:>10.2f}x"
            )
        print()
        last_gbps = effective_gbps
        last_fi_gbps = fi_gbps
        last_seq = seq

    # H100 HBM3 peak: 3350 GB/s. Print utilization for the largest shape.
    print(f"H100 HBM3 peak: {H100_HBM3_PEAK_GBPS:.0f} GB/s. Util at S={last_seq}:")
    for backend in backends:
        gbps = last_gbps[backend]
        print(
            f"  {BACKEND_LABELS[backend]:<22s} {gbps:>7.1f} GB/s "
            f"({100 * gbps / H100_HBM3_PEAK_GBPS:>4.1f}%)"
        )
    if last_fi_gbps is not None:
        print(
            f"  {'FlashInfer fp16 ref':<22s} {last_fi_gbps:>7.1f} GB/s "
            f"({100 * last_fi_gbps / H100_HBM3_PEAK_GBPS:>4.1f}%)"
        )

    print(f"\nminimum complete-decode speedups across {tag} shapes:")
    for backend, speedup in min_speedup.items():
        print(f"  {BACKEND_LABELS[backend]:<22s}: {speedup:.4f}x vs Triton")
    print(f"  legacy quantize minimum : {q_min_speedup:.4f}x")

    if args.assert_speedup is not None:
        if not min_speedup:
            p.error("--assert-speedup requires an optimized backend")
        gate_backend = (
            FLASHINFER_CUTEDSL_BACKEND
            if FLASHINFER_CUTEDSL_BACKEND in min_speedup
            else next(reversed(min_speedup))
        )
        measured = min_speedup[gate_backend]
        gate = args.assert_speedup
        if measured < gate:
            print(
                f"FAIL: {BACKEND_LABELS[gate_backend]} minimum speedup "
                f"{measured:.4f}x < target {gate:.4f}x"
            )
            sys.exit(1)
        print(
            f"PASS: {BACKEND_LABELS[gate_backend]} minimum speedup "
            f"{measured:.4f}x >= target {gate:.4f}x"
        )


if __name__ == "__main__":
    main()
