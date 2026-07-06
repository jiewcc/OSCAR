#!/usr/bin/env bash
# GPQA eval wrapper for GLM-4.7-Flash MLA + OSCAR INT2 latent KV-cache.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-/home/relay/wangzijie/model/GLM-4.7-Flash}"
if [[ -z "${ROT_DIR:-}" ]]; then
    CALIB_DIR="$(ls -1dt "${SCRIPT_DIR}/GPQA"/seq*_prompt*_group*/ 2>/dev/null | head -1 | sed 's:/$::')"
    export ROT_DIR="${CALIB_DIR}/rotations"
else
    export ROT_DIR
fi
export RUN_DIR="${RUN_DIR:-$(dirname "${ROT_DIR}")/_eval_gpqa_oscar_mla}"
export TP_SIZE="${TP_SIZE:-8}"
export GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export GROUP_SIZE="${GROUP_SIZE:-128}"
export K_CLIP="${K_CLIP:-0.96}"
export V_CLIP="${V_CLIP:-0.92}"
export NAME="${NAME:-gpqa_oscar_glm_4_7_flash_mla}"
export K_ROT_FILENAME="${K_ROT_FILENAME:-k_rotation_mla_latent_r_h_pbr.pt}"
export V_ROT_FILENAME="${V_ROT_FILENAME:-v_rotation_mla_latent_r_h_pbr.pt}"
export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---disable-cuda-graph --disable-piecewise-cuda-graph --prefill-attention-backend triton}"

exec bash "${SCRIPT_DIR}/../eval_oscar_gpqa.sh"
