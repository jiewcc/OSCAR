"""Optimized FlashInfer-derived INT2 decode attention for Hopper.

The public names in this module are stable production aliases for the fully
register-sourced WGMMA stage-1 kernel.  The original fused implementation and
the transposed shared-memory implementation remain available as side-by-side
correctness and performance baselines.
"""

from .flashinfer_cutedsl_int2_decode_fullrs_vec import (
    can_use_flashinfer_cutedsl_decode_fullrs_vec,
    flashinfer_cutedsl_decode_attention_fwd_int2_fullrs_vec,
)

__all__ = [
    "can_use_flashinfer_cutedsl_decode_optimized",
    "flashinfer_cutedsl_decode_attention_fwd_int2_optimized",
]


can_use_flashinfer_cutedsl_decode_optimized = (
    can_use_flashinfer_cutedsl_decode_fullrs_vec
)
flashinfer_cutedsl_decode_attention_fwd_int2_optimized = (
    flashinfer_cutedsl_decode_attention_fwd_int2_fullrs_vec
)
