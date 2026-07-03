#!/usr/bin/env bash
# Generic benchmark driver for OSCAR and BF16/baseline SGLang runs.
set -euo pipefail

export HF_HOME="${HF_HOME:-/shared/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL:?MODEL is required}"

TASK="${TASK:-gpqa}"
KV_MODE="${KV_MODE:-oscar}"  # oscar | baseline
if [[ "${KV_MODE}" == "oscar" ]]; then
    DEFAULT_KV_CACHE_DTYPE="int2"
else
    DEFAULT_KV_CACHE_DTYPE="auto"
fi
TP_SIZE="${TP_SIZE:-1}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
PORT="${PORT:-31057}"
DIST_PORT="${DIST_PORT:-41057}"
MEM_FRAC="${MEM_FRAC:-0.8}"
MAX_RUNNING="${MAX_RUNNING:-64}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
GROUP_SIZE="${GROUP_SIZE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
NUM_WORKERS="${NUM_WORKERS:-32}"
NAME="${NAME:-${TASK}_${KV_MODE}}"

if [[ "${KV_MODE}" == "oscar" ]]; then
    : "${ROT_DIR:?ROT_DIR is required when KV_MODE=oscar}"
    RUN_DIR="${RUN_DIR:-$(dirname "${ROT_DIR}")/_eval_${TASK}_oscar}"
elif [[ "${KV_MODE}" == "baseline" ]]; then
    RUN_DIR="${RUN_DIR:-${REPO_ROOT}/rotation/eval_outputs/${TASK}_baseline}"
else
    echo "[eval-benchmark] KV_MODE must be 'oscar' or 'baseline' (got ${KV_MODE})" >&2
    exit 2
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
elif [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME:-oscar}"
fi

SGLANG_DIR="${SGLANG_DIR:-${REPO_ROOT}/sglang-research}"
if [[ "${KV_MODE}" == "oscar" ]]; then
    export PYTHONPATH="${REPO_ROOT}/rotation/_triton_per_rank:${SGLANG_DIR}/python:${PYTHONPATH:-}"
else
    export PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"
fi
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}"
LOG_SERVER="${RUN_DIR}/server.log"
LOG_RUNNER="${RUN_DIR}/runner.log"
: > "${LOG_SERVER}"

export OSCAR_TRITON_PER_RANK_BASE="${OSCAR_TRITON_PER_RANK_BASE:-${RUN_DIR}/triton_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OSCAR_TRITON_PER_RANK_BASE}/main}"
mkdir -p "${OSCAR_TRITON_PER_RANK_BASE}" "${TRITON_CACHE_DIR}"

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

SERVER_ARGS=(
    --model-path "${MODEL}"
    --tensor-parallel-size "${TP_SIZE}"
    --prefill-attention-backend "${PREFILL_ATTENTION_BACKEND:-fa3}"
    --decode-attention-backend "${DECODE_ATTENTION_BACKEND:-triton}"
    --kv-cache-dtype "${KV_CACHE_DTYPE:-${DEFAULT_KV_CACHE_DTYPE}}"
    --mem-fraction-static "${MEM_FRAC}"
    --max-running-requests "${MAX_RUNNING}"
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
    --host 127.0.0.1
    --port "${PORT}"
    --dist-init-addr "127.0.0.1:${DIST_PORT}"
    --trust-remote-code
)

if [[ "${KV_MODE}" == "oscar" ]]; then
    SERVER_ARGS+=(--kv-cache-quant-group-size "${GROUP_SIZE}" --enable-cache-report)
fi
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    SERVER_ARGS+=(${EXTRA_SERVER_ARGS})
fi

echo "[eval-benchmark] task=${TASK} mode=${KV_MODE} model=${MODEL} tp=${TP_SIZE} gpus=${GPUS} out=${RUN_DIR}"

if [[ "${KV_MODE}" == "oscar" ]]; then
    SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    SGLANG_OSCAR_ABSORB_V_ROTATION=1 \
    SGLANG_MIXED_KV_HP_MAX_SPLITS=8 \
    SGLANG_MIXED_KV_PREFIX_TOKENS=${SGLANG_MIXED_KV_PREFIX_TOKENS:-64} \
    SGLANG_MIXED_KV_RECENT_TOKENS=${SGLANG_MIXED_KV_RECENT_TOKENS:-256} \
    SGLANG_MIXED_KV_HP_DTYPE=bfloat16 \
    SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
    SGLANG_OSCAR_K_ROTATION_PATH="${ROT_DIR}/${K_ROT_FILENAME:-k_rotation_qqt_r_h_pbr.pt}" \
    SGLANG_OSCAR_V_ROTATION_PATH="${ROT_DIR}/${V_ROT_FILENAME:-v_rotation_sst_r_h_pbr.pt}" \
    SGLANG_OSCAR_K_CLIP_RATIO="${K_CLIP:-0.96}" \
    SGLANG_OSCAR_V_CLIP_RATIO="${V_CLIP:-0.92}" \
    SGLANG_LLOYD_MAX="${SGLANG_LLOYD_MAX:-0}" \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    python -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${LOG_SERVER}" 2>&1 &
else
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    python -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${LOG_SERVER}" 2>&1 &
fi
SERVER_PID=$!

for _ in $(seq 1 "${SERVER_READY_POLLS:-240}"); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[eval-benchmark] server ready"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[eval-benchmark] server died"
        tail -100 "${LOG_SERVER}" || true
        exit 1
    fi
    sleep 5
done

if ! curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[eval-benchmark] server not ready"
    tail -100 "${LOG_SERVER}" || true
    exit 1
fi

RUNNER_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then
    RUNNER_ARGS+=(--data-root "${DATA_ROOT}")
fi
if [[ -n "${NUM_EXAMPLES:-}" ]]; then
    RUNNER_ARGS+=(--num-examples "${NUM_EXAMPLES}")
fi

python "${REPO_ROOT}/rotation/_eval_runner/run_benchmark_eval.py" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --max-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --top-p "${TOP_P:-0.95}" \
    --top-k "${TOP_K:-40}" \
    --num-threads "${NUM_WORKERS}" \
    "${RUNNER_ARGS[@]}" \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${LOG_RUNNER}"

echo "[eval-benchmark] done. score:"
grep -iE "${TASK}/score|${TASK}/chars" "${RUN_DIR}/eval.log" | tail -10 || true
