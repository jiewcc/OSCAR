#!/usr/bin/env bash
# BF16/auto KV-cache GPQA baseline for GLM-4.7-Flash.
set -euo pipefail

export HF_HOME="${HF_HOME:-/shared/huggingface}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-/home/relay/wangzijie/model/GLM-4.7-Flash}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/GPQA/baseline_eval}"
SGLANG_RESEARCH_DIR="${SGLANG_RESEARCH_DIR:-${REPO_ROOT}/sglang-research}"
TP_SIZE="${TP_SIZE:-8}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
PORT="${PORT:-31058}"
DIST_PORT="${DIST_PORT:-41058}"
MEM_FRAC="${MEM_FRAC:-0.8}"
MAX_RUNNING="${MAX_RUNNING:-64}"

CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-oscar}"
if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
fi

PY="${PY:-python}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi
export PYTHONPATH="${SGLANG_RESEARCH_DIR}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}"
LOG_SERVER="${RUN_DIR}/server.log"
LOG_RUNNER="${RUN_DIR}/runner.log"
: > "${LOG_SERVER}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        pkill -TERM -P "${SERVER_PID}" 2>/dev/null || true
        sleep 2
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
        pkill -KILL -P "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${GPUS}" \
"${PY}" -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --prefill-attention-backend triton \
    --decode-attention-backend triton \
    --kv-cache-dtype auto \
    --mem-fraction-static "${MEM_FRAC}" \
    --max-running-requests "${MAX_RUNNING}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dist-init-addr "127.0.0.1:${DIST_PORT}" \
    --trust-remote-code \
    ${EXTRA_SERVER_ARGS:---disable-cuda-graph --disable-piecewise-cuda-graph} \
    >> "${LOG_SERVER}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[baseline] server ready"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[baseline] server died"
        tail -100 "${LOG_SERVER}" || true
        exit 1
    fi
    sleep 5
done

RUNNER="${REPO_ROOT}/rotation/_eval_runner/run_simple_eval.py"
"${PY}" "${RUNNER}" \
    --task gpqa \
    --model "${MODEL}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --max-tokens "${MAX_NEW_TOKENS:-32768}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --top-p "${TOP_P:-0.95}" \
    --top-k "${TOP_K:-40}" \
    --n-repeats "${N_REPEATS:-1}" \
    --num-threads "${NUM_WORKERS:-32}" \
    ${NUM_EXAMPLES:+--num-examples ${NUM_EXAMPLES}} \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${LOG_RUNNER}"

grep -iE "gpqa/score|gpqa/chars" "${RUN_DIR}/eval.log" | tail -10 || true
