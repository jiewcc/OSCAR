#!/usr/bin/env bash
# Qwen3-8B wrapper for GPQA/AIME25/MATH500/HumanEval/LCBv6.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export TASK="${TASK:-gpqa}"
export KV_MODE="${KV_MODE:-oscar}"
export ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/rotations}"
export TP_SIZE="${TP_SIZE:-1}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"

exec bash "${SCRIPT_DIR}/../eval_benchmark.sh"
