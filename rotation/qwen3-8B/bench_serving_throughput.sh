#!/usr/bin/env bash
# Qwen3-8B wrapper for the Table-9-style serving throughput benchmark.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/rotations}"
export TP_SIZE="${TP_SIZE:-1}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"

exec bash "${SCRIPT_DIR}/../bench_serving_throughput.sh"
