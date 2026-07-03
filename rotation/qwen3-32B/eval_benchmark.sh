#!/usr/bin/env bash
# Qwen3-32B wrapper for GPQA/AIME25/MATH500/HumanEval/LCBv6.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_MODEL="/home/relay/wangzijie/model/Qwen3-32B-FP8"
if [[ ! -f "${DEFAULT_MODEL}/config.json" ]]; then
    DEFAULT_MODEL="Qwen/Qwen3-32B"
fi

export MODEL="${MODEL:-${DEFAULT_MODEL}}"
export TASK="${TASK:-gpqa}"
export KV_MODE="${KV_MODE:-oscar}"
export CALIB_TAG="${CALIB_TAG:-seq30000_prompt120_group128}"
export ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/GPQA/${CALIB_TAG}/rotations}"
export TP_SIZE="${TP_SIZE:-4}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.96}"

exec bash "${SCRIPT_DIR}/../eval_benchmark.sh"
