#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   METHOD=svd_text MODEL_SIZE=8b FINAL_TOKENS=160 bash text_svd/scripts/sweep_ratios.sh
#
#   METHOD=text_svd MODEL_SIZE=26b FINAL_TOKENS=160 \
#   RATIOS="0.9:0.1 0.7:0.3 0.5:0.5 0.3:0.7" \
#     bash text_svd/scripts/sweep_ratios.sh
#   STOP_WAIT=120 METHOD=svd_text MODEL_SIZE=26b FINAL_TOKENS=160 bash text_svd/scripts/sweep_ratios.sh
#
# This script starts one vLLM server per SVD_ALPHA:TEXT_BETA ratio, waits until
# /v1/models is ready, runs run_evalscope.sh, stops the server, then moves to
# the next ratio. Server stdout/stderr is written to
# text_svd/outputs/server_logs/*.log.

: "${MODEL_SIZE:=8b}"
: "${METHOD:=svd_text}"
: "${FINAL_TOKENS:=160}"
: "${RATIOS:=0.9:0.1 0.8:0.2 0.7:0.3 0.6:0.4 0.5:0.5 0.4:0.6 0.3:0.7 0.2:0.8 0.1:0.9}"
: "${PORT_BASE:=8200}"
: "${DATASETS:=blink,mvbench}"
: "${API_TIMEOUT:=900}"
: "${STREAM:=1}"
: "${PERF_PROXY:=1}"
: "${STARTUP_TIMEOUT:=900}"
: "${STARTUP_INTERVAL:=5}"
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
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
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
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for vLLM on port ${port}" >&2
      return 1
    fi
    sleep "${STARTUP_INTERVAL}"
  done
}

index=0
for ratio in ${RATIOS}; do
  IFS=':,' read -r svd_alpha text_beta extra <<< "${ratio}"
  if [[ -z "${svd_alpha:-}" || -z "${text_beta:-}" || -n "${extra:-}" ]]; then
    echo "Invalid ratio '${ratio}'; use SVD_ALPHA:TEXT_BETA, e.g. 0.7:0.3" >&2
    exit 1
  fi

  port=$((PORT_BASE + index))
  served_model_name="internvl2_${MODEL_SIZE}_${model_suffix}${FINAL_TOKENS}_${svd_alpha/./}${text_beta/./}"
  log_dir="/home/GZGKD001/tmp/yanhong/text_svd/outputs/server_logs"
  log_file="${log_dir}/${served_model_name}.log"
  mkdir -p "${log_dir}"

  echo "[SWEEP] starting METHOD=${METHOD} MODEL_SIZE=${MODEL_SIZE} FINAL_TOKENS=${FINAL_TOKENS} SVD_ALPHA=${svd_alpha} TEXT_BETA=${text_beta} PORT=${port} MODEL=${served_model_name}"
  FINAL_TOKENS="${FINAL_TOKENS}" \
  SVD_ALPHA="${svd_alpha}" \
  TEXT_BETA="${text_beta}" \
  PORT="${port}" \
  SERVED_MODEL_NAME="${served_model_name}" \
    setsid "${serve_script}" >"${log_file}" 2>&1 &
  server_pid="$!"

  wait_ready "${port}"

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

  "${eval_cmd[@]}" "$@"

  cleanup
  if [[ "${STOP_WAIT}" != "0" ]]; then
    echo "[SWEEP] waiting ${STOP_WAIT}s for device memory/runtime cleanup"
    sleep "${STOP_WAIT}"
  fi
  unset server_pid
  index=$((index + 1))
done
