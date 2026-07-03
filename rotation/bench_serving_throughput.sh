#!/usr/bin/env bash
# End-to-end serving throughput benchmark for BF16 baseline, BF16+FA3 decode,
# and OSCAR INT2 KV cache. Defaults follow the OSCAR paper Table 9 workload:
# 32 concurrent requests, 8192-token inputs, 1024-token outputs.
set -euo pipefail

export HF_HOME="${HF_HOME:-/shared/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL:?MODEL is required}"

SGLANG_DIR="${SGLANG_DIR:-${REPO_ROOT}/sglang-research}"
BASE_PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"
export PYTHONPATH="${BASE_PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
elif [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME:-oscar}"
fi

MODES="${MODES:-baseline baseline_fa3 oscar}"
if [[ "${MODES}" == "all" ]]; then
    MODES="baseline baseline_fa3 oscar"
fi

TP_SIZE="${TP_SIZE:-1}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
PORT="${PORT:-31060}"
DIST_PORT="${DIST_PORT:-41060}"
MEM_FRAC="${MEM_FRAC:-0.8}"
CONCURRENCY="${CONCURRENCY:-32}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
WARMUP_TOKENS="${WARMUP_TOKENS:-100000}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.0}"
DATASET_NAME="${DATASET_NAME:-random-ids}"
TOKENIZE_PROMPT="${TOKENIZE_PROMPT:-1}"
FLUSH_CACHE="${FLUSH_CACHE:-1}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
MAX_RUNNING="${MAX_RUNNING:-1}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-1}"
GROUP_SIZE="${GROUP_SIZE:-128}"
SEED="${SEED:-1}"
SERVER_READY_POLLS="${SERVER_READY_POLLS:-240}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

WARMUP_OUTPUT_LEN="${OUTPUT_LEN}"
if (( WARMUP_OUTPUT_LEN > 32 )); then
    WARMUP_OUTPUT_LEN=32
fi
WARMUP_TOKENS_PER_REQUEST=$((INPUT_LEN + WARMUP_OUTPUT_LEN))
if [[ -z "${WARMUP_REQUESTS:-}" ]]; then
    WARMUP_REQUESTS=$(((WARMUP_TOKENS + WARMUP_TOKENS_PER_REQUEST - 1) / WARMUP_TOKENS_PER_REQUEST))
fi

if [[ -n "${ROT_DIR:-}" ]]; then
    RUN_ROOT="${RUN_ROOT:-$(dirname "${ROT_DIR}")/_throughput_e2e_${RUN_ID}}"
else
    RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/rotation/throughput/e2e_${RUN_ID}}"
fi

GPU_LIST="${GPUS//,/ }"
# shellcheck disable=SC2206
GPU_ARRAY=(${GPU_LIST})
GPU_COUNT="${GPU_COUNT:-${#GPU_ARRAY[@]}}"

mkdir -p "${RUN_ROOT}"
SUMMARY_JSONL="${RUN_ROOT}/summary.jsonl"
SUMMARY_TSV="${RUN_ROOT}/summary.tsv"
: > "${SUMMARY_JSONL}"
printf "mode\tcompleted\tduration_s\toutput_tok_s\tG_tok_s_per_gpu\tmedian_U_tok_s_per_user\tmean_U_tok_s_per_user\tmean_ttft_ms\tmedian_e2e_ms\tp99_e2e_ms\tresult_file\n" > "${SUMMARY_TSV}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        pkill -TERM -P "${SERVER_PID}" 2>/dev/null || true
        sleep 2
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
        pkill -KILL -P "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup EXIT INT TERM

mode_needs_rotation() {
    [[ "$1" == "oscar" ]]
}

launch_server() {
    local mode="$1"
    local mode_dir="$2"
    local server_log="$3"
    local decode_backend
    local kv_cache_dtype

    case "${mode}" in
        baseline)
            kv_cache_dtype="${BASELINE_KV_CACHE_DTYPE:-auto}"
            decode_backend="${BASELINE_DECODE_BACKEND:-triton}"
            ;;
        baseline_fa3)
            kv_cache_dtype="${BASELINE_FA3_KV_CACHE_DTYPE:-auto}"
            decode_backend="${BASELINE_FA3_DECODE_BACKEND:-fa3}"
            ;;
        oscar)
            kv_cache_dtype="${OSCAR_KV_CACHE_DTYPE:-int2}"
            decode_backend="${OSCAR_DECODE_BACKEND:-triton}"
            ;;
        *)
            echo "[bench-serving] unknown mode: ${mode}" >&2
            exit 2
            ;;
    esac

    if mode_needs_rotation "${mode}"; then
        : "${ROT_DIR:?ROT_DIR is required when MODES contains oscar}"
    fi

    local -a server_args=(
        --model-path "${MODEL}"
        --tensor-parallel-size "${TP_SIZE}"
        --prefill-attention-backend "${PREFILL_ATTENTION_BACKEND:-fa3}"
        --decode-attention-backend "${decode_backend}"
        --kv-cache-dtype "${kv_cache_dtype}"
        --mem-fraction-static "${MEM_FRAC}"
        --max-running-requests "${MAX_RUNNING}"
        --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
        --host 127.0.0.1
        --port "${PORT}"
        --dist-init-addr "127.0.0.1:${DIST_PORT}"
        --trust-remote-code
    )

    if [[ "${mode}" == "oscar" ]]; then
        server_args+=(--kv-cache-quant-group-size "${GROUP_SIZE}" --enable-cache-report)
    fi
    if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
        server_args+=(--disable-radix-cache)
    fi
    if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        server_args+=(${EXTRA_SERVER_ARGS})
    fi

    printf "%q " python -m sglang.launch_server "${server_args[@]}" > "${mode_dir}/server_args.txt"
    printf "\n" >> "${mode_dir}/server_args.txt"
    : > "${server_log}"

    echo "[bench-serving] launching mode=${mode} decode=${decode_backend} kv=${kv_cache_dtype} out=${mode_dir}"
    if [[ "${mode}" == "oscar" ]]; then
        local oscar_pythonpath="${REPO_ROOT}/rotation/_triton_per_rank:${BASE_PYTHONPATH}"
        local triton_rank_base="${OSCAR_TRITON_PER_RANK_BASE:-${mode_dir}/triton_cache}"
        local triton_cache_dir="${TRITON_CACHE_DIR:-${triton_rank_base}/main}"
        mkdir -p "${triton_rank_base}" "${triton_cache_dir}"
        env \
            PYTHONPATH="${oscar_pythonpath}" \
            SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
            SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
            SGLANG_OSCAR_ABSORB_V_ROTATION=1 \
            SGLANG_MIXED_KV_HP_MAX_SPLITS="${SGLANG_MIXED_KV_HP_MAX_SPLITS:-8}" \
            SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}" \
            SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS="${SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS:-8192}" \
            SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}" \
            SGLANG_MIXED_KV_HP_DTYPE="${SGLANG_MIXED_KV_HP_DTYPE:-bfloat16}" \
            SGLANG_MIXED_KV_SCALE_DTYPE="${SGLANG_MIXED_KV_SCALE_DTYPE:-float32}" \
            SGLANG_OSCAR_K_ROTATION_PATH="${ROT_DIR}/${K_ROT_FILENAME:-k_rotation_qqt_r_h_pbr.pt}" \
            SGLANG_OSCAR_V_ROTATION_PATH="${ROT_DIR}/${V_ROT_FILENAME:-v_rotation_sst_r_h_pbr.pt}" \
            SGLANG_OSCAR_K_CLIP_RATIO="${K_CLIP:-0.96}" \
            SGLANG_OSCAR_V_CLIP_RATIO="${V_CLIP:-0.92}" \
            SGLANG_LLOYD_MAX="${SGLANG_LLOYD_MAX:-0}" \
            OSCAR_TRITON_PER_RANK_BASE="${triton_rank_base}" \
            TRITON_CACHE_DIR="${triton_cache_dir}" \
            CUDA_VISIBLE_DEVICES="${GPUS}" \
            python -m sglang.launch_server "${server_args[@]}" >> "${server_log}" 2>&1 &
    else
        env CUDA_VISIBLE_DEVICES="${GPUS}" \
            python -m sglang.launch_server "${server_args[@]}" >> "${server_log}" 2>&1 &
    fi
    SERVER_PID=$!
}

wait_server_ready() {
    local server_log="$1"
    local waited=0
    for _ in $(seq 1 "${SERVER_READY_POLLS}"); do
        if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "[bench-serving] server ready after ${waited}s"
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[bench-serving] server died while starting" >&2
            tail -100 "${server_log}" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "[bench-serving] server not ready after ${waited}s" >&2
    tail -100 "${server_log}" || true
    return 1
}

summarize_result() {
    local mode="$1"
    local result_jsonl="$2"
    python - "$mode" "$result_jsonl" "$GPU_COUNT" "$SUMMARY_JSONL" "$SUMMARY_TSV" <<'PY'
import json
import math
import statistics
import sys

mode, result_path, gpu_count_s, summary_jsonl, summary_tsv = sys.argv[1:]
gpu_count = max(float(gpu_count_s), 1.0)

rows = []
with open(result_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
if not rows:
    raise SystemExit(f"no benchmark result rows in {result_path}")

row = rows[-1]
output_lens = row.get("output_lens") or []
ttfts = row.get("ttfts") or []
itls = row.get("itls") or []

user_tps = []
for out_len, ttft, per_token in zip(output_lens, ttfts, itls):
    latency = float(ttft or 0.0) + sum(float(x) for x in (per_token or []))
    if out_len and latency > 0:
        user_tps.append(float(out_len) / latency)

def median(xs):
    return statistics.median(xs) if xs else float("nan")

def mean(xs):
    return statistics.mean(xs) if xs else float("nan")

summary = {
    "mode": mode,
    "completed": row.get("completed"),
    "duration_s": row.get("duration"),
    "output_tok_s": row.get("output_throughput"),
    "G_tok_s_per_gpu": (
        row.get("output_throughput") / gpu_count
        if isinstance(row.get("output_throughput"), (int, float))
        else None
    ),
    "median_U_tok_s_per_user": median(user_tps),
    "mean_U_tok_s_per_user": mean(user_tps),
    "mean_ttft_ms": row.get("mean_ttft_ms"),
    "median_e2e_ms": row.get("median_e2e_latency_ms"),
    "p99_e2e_ms": row.get("p99_e2e_latency_ms"),
    "result_file": result_path,
}

with open(summary_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(summary, sort_keys=True) + "\n")

fields = [
    "mode",
    "completed",
    "duration_s",
    "output_tok_s",
    "G_tok_s_per_gpu",
    "median_U_tok_s_per_user",
    "mean_U_tok_s_per_user",
    "mean_ttft_ms",
    "median_e2e_ms",
    "p99_e2e_ms",
    "result_file",
]
with open(summary_tsv, "a", encoding="utf-8") as f:
    f.write("\t".join(str(summary.get(k, "")) for k in fields) + "\n")

print(
    "[bench-serving] summary "
    f"mode={mode} completed={summary['completed']} "
    f"U_median={summary['median_U_tok_s_per_user']:.2f} tok/s/user "
    f"G={summary['G_tok_s_per_gpu']:.2f} tok/s/GPU "
    f"output={summary['output_tok_s']:.2f} tok/s"
)
PY
}

run_mode() {
    local mode="$1"
    local mode_dir="${RUN_ROOT}/${mode}"
    local server_log="${mode_dir}/server.log"
    local runner_log="${mode_dir}/bench_serving.log"
    local result_jsonl="${mode_dir}/bench_serving.jsonl"
    mkdir -p "${mode_dir}"
    : > "${result_jsonl}"

    cleanup
    launch_server "${mode}" "${mode_dir}" "${server_log}"
    wait_server_ready "${server_log}"

    local -a bench_args=(
        --backend sglang
        --base-url "http://127.0.0.1:${PORT}"
        --dataset-name "${DATASET_NAME}"
        --model "${MODEL}"
        --tokenizer "${TOKENIZER:-${MODEL}}"
        --num-prompts "${NUM_PROMPTS}"
        --random-input-len "${INPUT_LEN}"
        --random-output-len "${OUTPUT_LEN}"
        --random-range-ratio "${RANDOM_RANGE_RATIO}"
        --request-rate "${REQUEST_RATE}"
        --max-concurrency "${CONCURRENCY}"
        --warmup-requests "${WARMUP_REQUESTS}"
        --seed "${SEED}"
        --output-file "${result_jsonl}"
        --output-details
    )
    if [[ "${TOKENIZE_PROMPT}" == "1" ]]; then
        bench_args+=(--tokenize-prompt)
    fi
    if [[ "${DISABLE_TQDM:-1}" == "1" ]]; then
        bench_args+=(--disable-tqdm)
    fi
    if [[ "${FLUSH_CACHE}" == "1" ]]; then
        bench_args+=(--flush-cache)
    fi
    if [[ -n "${EXTRA_REQUEST_BODY:-}" ]]; then
        bench_args+=(--extra-request-body "${EXTRA_REQUEST_BODY}")
    fi
    if [[ -n "${BENCH_EXTRA_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        bench_args+=(${BENCH_EXTRA_ARGS})
    fi

    echo "[bench-serving] running workload: prompts=${NUM_PROMPTS} concurrency=${CONCURRENCY} input=${INPUT_LEN} output=${OUTPUT_LEN}"
    echo "[bench-serving] warmup_requests=${WARMUP_REQUESTS} approx_warmup_tokens=$((WARMUP_REQUESTS * WARMUP_TOKENS_PER_REQUEST)) target=${WARMUP_TOKENS}"
    env CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-}" \
        python -m sglang.bench_serving "${bench_args[@]}" 2>&1 | tee "${runner_log}"
    summarize_result "${mode}" "${result_jsonl}"
    cleanup
    sleep "${BETWEEN_MODE_SLEEP:-5}"
}

echo "[bench-serving] model=${MODEL} tp=${TP_SIZE} gpus=${GPUS} gpu_count=${GPU_COUNT}"
echo "[bench-serving] modes=${MODES}"
echo "[bench-serving] run_root=${RUN_ROOT}"

for mode in ${MODES}; do
    run_mode "${mode}"
done

echo "[bench-serving] done"
echo "[bench-serving] summary_tsv=${SUMMARY_TSV}"
echo "[bench-serving] summary_jsonl=${SUMMARY_JSONL}"
