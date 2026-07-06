#!/usr/bin/env bash
# Compute OSCAR rotations for MLA latent KV cache.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_SCRIPT="${SCRIPT_DIR}/../compute_kv_rotation.py"

METHOD="${METHOD:-mla_latent}"
HEAD_DIM="${HEAD_DIM:-512}"
MLA_ROPE_DIM="${MLA_ROPE_DIM:-64}"
COMPOSITION="${COMPOSITION:-r_h_pbr}"
CHUNK_ID="${CHUNK_ID:-all}"
DATASET="${DATASET:-GPQA}"

if [[ -z "${CALIB_DIR:-}" ]]; then
    CALIB_DIR="$(ls -1dt "${SCRIPT_DIR}/${DATASET}"/seq*_prompt*_group*/ 2>/dev/null | head -1 | sed 's:/$::')"
fi
DUMP_PATH="${DUMP_PATH:-${CALIB_DIR}/qkv_dumps/gpqa}"
OUTPUT_DIR="${OUTPUT_DIR:-${CALIB_DIR}/rotations}"

if [[ -z "${PY:-}" ]]; then
    for candidate in \
        ${HOME}/miniconda3/envs/oscar/bin/python3 \
        ${HOME}/anaconda3/envs/oscar/bin/python3 \
        "$(command -v python3 || true)"
    do
        if [[ -x "${candidate}" ]]; then
            PY="${candidate}"
            break
        fi
    done
fi
: "${PY:?no python3 found; set PY=/path/to/python3}"

mkdir -p "${OUTPUT_DIR}"
echo "[compute_mla_rotation] dump_path=${DUMP_PATH}"
echo "[compute_mla_rotation] output_dir=${OUTPUT_DIR}"
echo "[compute_mla_rotation] head_dim=${HEAD_DIM} mla_rope_dim=${MLA_ROPE_DIM}"

"${PY}" "${COMPUTE_SCRIPT}" \
    --dump-path "${DUMP_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --head-dim "${HEAD_DIM}" \
    --mla-rope-dim "${MLA_ROPE_DIM}" \
    --chunk-id "${CHUNK_ID}" \
    --method "${METHOD}" \
    --composition "${COMPOSITION}"

ls -la "${OUTPUT_DIR}" | grep -E "rotation.*\\.pt" || true
