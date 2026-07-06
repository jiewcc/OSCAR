#!/usr/bin/env bash
# Dump MLA latent-space calibration tensors for GLM-4.7-Flash.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GLM47_DIR="${SCRIPT_DIR}/../GLM-4.7"

export MODEL="${MODEL:-/home/relay/wangzijie/model/GLM-4.7-Flash}"
export DATASET="${DATASET:-GPQA}"
export TP_SIZE="${TP_SIZE:-8}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${SCRIPT_DIR}}"
export DUMP_KVCACHE="${DUMP_KVCACHE:-false}"
export DUMP_MLA_LATENT_KVCACHE="${DUMP_MLA_LATENT_KVCACHE:-true}"
export DUMP_KVCACHE_TOKENS="${DUMP_KVCACHE_TOKENS:-30000}"
export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-} --disable-cuda-graph"

exec bash "${GLM47_DIR}/save_qkv_glm47.sh"
