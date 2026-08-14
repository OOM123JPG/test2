#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   METHOD=svd_text MODEL_SIZE=8b TOKENS="64 96 128 160 192" bash text_svd/scripts/sweep_tokens.sh
#   METHOD=text_svd MODEL_SIZE=26b TOKENS="128 160" LIMIT=20 bash text_svd/scripts/sweep_tokens.sh
#   STOP_WAIT=120 METHOD=svd_text MODEL_SIZE=26b TOKENS="64 96 128" bash text_svd/scripts/sweep_tokens.sh
#
# This script starts one vLLM server per token setting, waits until /v1/models is
# ready, runs run_evalscope.sh, stops the server, then moves to the next setting.
# Server stdout/stderr is written to text_svd/outputs/server_logs/*.log.

: "${MODEL_SIZE:=8b}"
: "${METHOD:=svd_text}"
: "${TOKENS:=64 96 128 160 192}"
: "${PORT_BASE:=8100}"
: "${DATASETS:=blink,mvbench}"
: "${API_TIMEOUT:=900}"
: "${STREAM:=1}"
: "${PERF_PROXY:=1}"
: "${STARTUP_TIMEOUT:=900}"
: "${STARTUP_INTERVAL:=5}"
: "${SVD_ALPHA:=0.7}"
: "${TEXT_BETA:=0.3}"
: "${STOP_TIMEOUT:=60}"
: "${STOP_WAIT:=60}"

if [[ -z "${NPROC:-}" ]]; then
  if [[ "${MODEL_SIZE}" == "40b" ]]; then
    NPROC=64
  else
    NPROC=128
  fi
fi

case "${METHOD}" in
  svd_text)
    serve_script="/home/GZGKD001/tmp/yanhong/text_svd/scripts/serve_internvl2_${MODEL_SIZE}_svd_text.sh"
    model_suffix="tsvd"
    ;;
  text_svd)
    serve_script="/home/GZGKD001/tmp/yanhong/text_svd/scripts/serve_internvl2_${MODEL_SIZE}_text_svd.sh"
    model_suffix="text_svd"
    ;;
  *)
    echo "Unsupported METHOD=${METHOD}; use svd_text or text_svd" >&2
    exit 1
    ;;
esac

if [[ ! -x "${serve_script}" ]]; then
  echo "Serve script not found or not executable: ${serve_script}" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    if kill -0 "-${server_pid}" 2>/dev/null; then
      kill "-${server_pid}" 2>/dev/null || true
      local deadline=$((SECONDS + STOP_TIMEOUT))
      while kill -0 "-${server_pid}" 2>/dev/null; do
        if (( SECONDS >= deadline )); then
          echo "[SWEEP] server process group ${server_pid} did not stop; sending SIGKILL" >&2
          kill -9 "-${server_pid}" 2>/dev/null || true
          break
        fi
        sleep 1
      done
    elif kill -0 "${server_pid}" 2>/dev/null; then
      kill "${server_pid}" 2>/dev/null || true
    fi
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_ready() {
  local port="$1"
  local log_file="${2:-}"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  local last_notice=0
  until python - "$port" <<'PY'
import sys
from urllib.request import urlopen

port = sys.argv[1]
try:
    with urlopen(f"http://localhost:{port}/v1/models", timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
  do
    if [[ -n "${server_pid:-}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "[SWEEP] server process exited while waiting on port ${port}" >&2
      if [[ -n "${log_file}" && -f "${log_file}" ]]; then
        echo "[SWEEP] last server log lines from ${log_file}:" >&2
        tail -n 40 "${log_file}" >&2 || true
      fi
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for vLLM on port ${port}" >&2
      if [[ -n "${log_file}" && -f "${log_file}" ]]; then
        echo "[SWEEP] last server log lines from ${log_file}:" >&2
        tail -n 40 "${log_file}" >&2 || true
      fi
      return 1
    fi
    if (( SECONDS - last_notice >= 30 )); then
      echo "[SWEEP] waiting for vLLM on port ${port}; log=${log_file}"
      last_notice="${SECONDS}"
    fi
    sleep "${STARTUP_INTERVAL}"
  done
  echo "[SWEEP] vLLM ready on port ${port}"
}

index=0
for final_tokens in ${TOKENS}; do
  port=$((PORT_BASE + index))
  served_model_name="internvl2_${MODEL_SIZE}_${model_suffix}${final_tokens}_${SVD_ALPHA/./}${TEXT_BETA/./}"
  log_dir="/home/GZGKD001/tmp/yanhong/text_svd/outputs/server_logs"
  log_file="${log_dir}/${served_model_name}.log"
  mkdir -p "${log_dir}"

  echo "[SWEEP] starting METHOD=${METHOD} MODEL_SIZE=${MODEL_SIZE} FINAL_TOKENS=${final_tokens} PORT=${port} MODEL=${served_model_name}"
  echo "[SWEEP] server log: ${log_file}"
  FINAL_TOKENS="${final_tokens}" \
  PORT="${port}" \
  SERVED_MODEL_NAME="${served_model_name}" \
    setsid "${serve_script}" >"${log_file}" 2>&1 &
  server_pid="$!"

  wait_ready "${port}" "${log_file}"

  eval_cmd=(
    /home/GZGKD001/tmp/yanhong/text_svd/scripts/run_evalscope.sh
    --port "${port}"
    --model "${served_model_name}"
    --datasets "${DATASETS}"
    --nproc "${NPROC}"
    --api-timeout "${API_TIMEOUT}"
  )

  if [[ -n "${LIMIT:-}" ]]; then
    eval_cmd+=(--limit "${LIMIT}")
  fi
  if [[ "${STREAM}" == "1" ]]; then
    eval_cmd+=(--stream)
  else
    eval_cmd+=(--no-stream)
  fi
  if [[ "${PERF_PROXY}" == "1" ]]; then
    eval_cmd+=(--perf-proxy)
  fi

  echo "[SWEEP] evaluating MODEL=${served_model_name} DATASETS=${DATASETS}"
  "${eval_cmd[@]}" "$@"
  echo "[SWEEP] finished MODEL=${served_model_name}"

  cleanup
  echo "[SWEEP] stopped server for MODEL=${served_model_name}"
  if [[ "${STOP_WAIT}" != "0" ]]; then
    echo "[SWEEP] waiting ${STOP_WAIT}s for device memory/runtime cleanup"
    sleep "${STOP_WAIT}"
  fi
  unset server_pid
  index=$((index + 1))
done
