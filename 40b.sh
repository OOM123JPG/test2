#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=/home/GZGKD001/tmp/models/InternVL2-40B}"
: "${SERVED_MODEL_NAME:=internvl2_40b}"
: "${PORT:=8040}"
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${ASCEND_RT_VISIBLE_DEVICES:=8,9}"
: "${HOST:=0.0.0.0}"
: "${DTYPE:=float16}"
if [[ -z "${LIMIT_MM_PER_PROMPT:-}" ]]; then
  LIMIT_MM_PER_PROMPT='{"image": 32}'
fi
: "${GPU_MEMORY_UTILIZATION:=0.9}"
: "${MAX_NUM_SEQS:=512}"
: "${MAX_MODEL_LEN:=24576}"
: "${MAX_NUM_BATCHED_TOKENS:=8192}"
PROJECT_DIR="${PROJECT_DIR:-/home/GZGKD001/tmp/yanhong/text_svd}"
: "${ALLOWED_LOCAL_MEDIA_PATH:=${PROJECT_DIR}/cache/data}"

export ASCEND_RT_VISIBLE_DEVICES

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --dtype "${DTYPE}" \
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --port "${PORT}" \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}" \
  --enforce-eager \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --trust-remote-code \
  "$@"
