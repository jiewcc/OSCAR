"""Stage-1-only benchmark and Nsight Compute harness for fused INT2 decode.

The regular benchmark includes split reduction.  This utility isolates the
Triton and FlashInfer-derived stage-1 kernels so kernel changes can be accepted
or rejected without stage-2 noise.  ``--profile`` wraps exactly one warm
launch in an NVTX range named ``profile`` for ``ncu --nvtx`` filtering.
"""

import argparse
import math
from dataclasses import dataclass

import torch
import triton

from sglang.QuantKernel.flashinfer_cutedsl_int2_decode import (
    flashinfer_cutedsl_decode_attention_fwd_int2,
)
from sglang.QuantKernel.flashinfer_cutedsl_int2_decode_transposed import (
    flashinfer_cutedsl_decode_attention_fwd_int2_transposed,
)
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _decode_att_m_fwd_quant_int2,
    _decode_grouped_att_m_fwd_quant_int2,
)


@dataclass
class Stage1Case:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    k_sz: torch.Tensor
    v_sz: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    splits: torch.Tensor
    tri_out: torch.Tensor
    tri_lse: torch.Tensor
    dsl_out: torch.Tensor
    dsl_lse: torch.Tensor
    max_splits: int
    sm_scale: float


def _parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _auto_splits(seq_len: int) -> int:
    return 4 if seq_len <= 4096 else 8


def make_case(
    batch: int, seq_len: int, gqa: bool, max_splits_override: int | None = None
) -> Stage1Case:
    device = "cuda"
    q_heads, kv_heads = (32, 8) if gqa else (8, 8)
    head_dim = 128
    max_splits = max_splits_override or _auto_splits(seq_len)
    num_tokens = batch * seq_len

    torch.manual_seed(batch * 1000003 + seq_len + int(gqa))
    q = torch.randn(batch, q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randint(
        0,
        256,
        (num_tokens, kv_heads, head_dim // 4),
        dtype=torch.uint8,
        device=device,
    )
    v = torch.randint_like(k, 0, 256)
    k_sz = torch.empty(num_tokens, kv_heads, 2, dtype=torch.float32, device=device)
    v_sz = torch.empty_like(k_sz)
    k_sz[..., 0].uniform_(0.01, 0.05)
    k_sz[..., 1].uniform_(0.0, 3.0)
    v_sz[..., 0].uniform_(0.01, 0.05)
    v_sz[..., 1].uniform_(0.0, 3.0)
    kv_indptr = torch.arange(
        0,
        num_tokens + seq_len,
        seq_len,
        dtype=torch.int32,
        device=device,
    )
    kv_indices = torch.arange(num_tokens, dtype=torch.int32, device=device)
    splits = torch.full((batch,), max_splits, dtype=torch.int32, device=device)
    out_shape = (batch, q_heads, max_splits, head_dim)
    lse_shape = (batch, q_heads, max_splits)
    return Stage1Case(
        q=q,
        k=k,
        v=v,
        k_sz=k_sz,
        v_sz=v_sz,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        splits=splits,
        tri_out=torch.empty(out_shape, dtype=torch.float32, device=device),
        tri_lse=torch.full(
            lse_shape, float("-inf"), dtype=torch.float32, device=device
        ),
        dsl_out=torch.empty(out_shape, dtype=torch.float32, device=device),
        dsl_lse=torch.full(
            lse_shape, float("-inf"), dtype=torch.float32, device=device
        ),
        max_splits=max_splits,
        sm_scale=1.0 / math.sqrt(head_dim),
    )


def run_triton(case: Stage1Case, gqa: bool) -> None:
    fn = _decode_grouped_att_m_fwd_quant_int2 if gqa else _decode_att_m_fwd_quant_int2
    fn(
        case.q,
        case.k,
        case.v,
        case.k_sz,
        case.v_sz,
        case.tri_out,
        case.tri_lse,
        case.kv_indptr,
        case.kv_indices,
        case.splits,
        case.max_splits,
        case.sm_scale,
        0.0,
    )


def run_cutedsl(case: Stage1Case) -> None:
    flashinfer_cutedsl_decode_attention_fwd_int2(
        case.q,
        case.k,
        case.v,
        case.k_sz,
        case.v_sz,
        case.dsl_out,
        case.dsl_lse,
        case.kv_indptr,
        case.kv_indices,
        case.splits,
        case.max_splits,
        case.sm_scale,
    )


def run_transposed(case: Stage1Case) -> None:
    flashinfer_cutedsl_decode_attention_fwd_int2_transposed(
        case.q,
        case.k,
        case.v,
        case.k_sz,
        case.v_sz,
        case.dsl_out,
        case.dsl_lse,
        case.kv_indptr,
        case.kv_indices,
        case.splits,
        case.max_splits,
        case.sm_scale,
    )


def bench(fn, warmup: int, rep: int) -> float:
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa", action="store_true")
    parser.add_argument("--batches", type=_parse_int_list, default=[1, 4, 16, 32])
    parser.add_argument(
        "--seq-lens", type=_parse_int_list, default=[1024, 4096, 16384, 32768]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument(
        "--profile", choices=("triton", "cutedsl", "transposed"), default=None
    )
    args = parser.parse_args()

    if args.profile is not None and (len(args.batches) != 1 or len(args.seq_lens) != 1):
        parser.error("--profile requires one batch and one sequence length")

    print(
        "mode,batch,seq,splits,triton_ms,cutedsl_ms,transposed_ms,"
        "tri_over_dsl,tri_over_transposed"
    )
    for batch in args.batches:
        for seq_len in args.seq_lens:
            case = make_case(batch, seq_len, args.gqa, args.splits)
            if args.profile is not None:
                if args.profile == "triton":
                    fn = lambda: run_triton(case, args.gqa)
                elif args.profile == "cutedsl":
                    fn = lambda: run_cutedsl(case)
                else:
                    fn = lambda: run_transposed(case)
                fn()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("profile")
                fn()
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                return

            tri_ms = bench(lambda: run_triton(case, args.gqa), args.warmup, args.rep)
            dsl_ms = bench(lambda: run_cutedsl(case), args.warmup, args.rep)
            transposed_ms = bench(lambda: run_transposed(case), args.warmup, args.rep)
            mode = "gqa" if args.gqa else "mha"
            print(
                f"{mode},{batch},{seq_len},{case.max_splits},"
                f"{tri_ms:.6f},{dsl_ms:.6f},{transposed_ms:.6f},"
                f"{tri_ms / dsl_ms:.4f},{tri_ms / transposed_ms:.4f}"
            )


if __name__ == "__main__":
    main()
