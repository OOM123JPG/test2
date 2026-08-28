#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/GZGKD001/tmp/yanhong/text_svd}"
cd "${PROJECT_DIR}"

port=""
model="${MODEL_NAME:-}"
dataset_dir="${DATASET_DIR:-${PROJECT_DIR}/cache/data/mmiu_dataset}"
output_dir="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/mmiu}"
target_tasks="${TARGET_TASKS:-}"
model_path="${MODEL_PATH:-}"
judge_api_base="${JUDGE_API_BASE:-http://localhost:8027/v1}"
judge_model="${JUDGE_MODEL:-qwen3.5-27b-instruct}"
reuse=0

usage() {
  cat <<'EOF'
Usage: scripts/eval_mmiu.sh --port PORT [options]
       scripts/eval_mmiu.sh PORT [options]


scripts/eval_mmiu.sh --port 8000 --reuse --target-tasks next_img_prediction,spot_the_diff
scripts/eval_mmiu.sh --port 8010 --reuse --target-tasks next_img_prediction,spot_the_diff

Options:
  --port PORT                 vLLM server port for the evaluated model
  --model MODEL               served model name; default: read from /v1/models
  --dataset-dir DIR           default: $PROJECT_DIR/cache/data/mmiu_dataset
  --output-dir DIR            default: $PROJECT_DIR/outputs/mmiu
  --target-tasks A,B          run selected MMIU tasks only
  --model-path DIR            optional model dir, used to read config hints
  --judge-api-base URL        default: http://localhost:8027/v1
  --judge-model MODEL         default: qwen3.5-27b-instruct
  --reuse                     skip existing metadata files
  -h, --help                  show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value"; exit 1; }
      port="$2"; shift 2 ;;
    --model)
      [[ $# -ge 2 ]] || { echo "--model requires a value"; exit 1; }
      model="$2"; shift 2 ;;
    --dataset-dir)
      [[ $# -ge 2 ]] || { echo "--dataset-dir requires a value"; exit 1; }
      dataset_dir="$2"; shift 2 ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value"; exit 1; }
      output_dir="$2"; shift 2 ;;
    --target-tasks)
      [[ $# -ge 2 ]] || { echo "--target-tasks requires a value"; exit 1; }
      target_tasks="$2"; shift 2 ;;
    --model-path)
      [[ $# -ge 2 ]] || { echo "--model-path requires a value"; exit 1; }
      model_path="$2"; shift 2 ;;
    --judge-api-base)
      [[ $# -ge 2 ]] || { echo "--judge-api-base requires a value"; exit 1; }
      judge_api_base="$2"; shift 2 ;;
    --judge-model)
      [[ $# -ge 2 ]] || { echo "--judge-model requires a value"; exit 1; }
      judge_model="$2"; shift 2 ;;
    --reuse)
      reuse=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      if [[ -z "$port" && "$1" =~ ^[0-9]+$ ]]; then
        port="$1"; shift
      else
        echo "Unknown argument: $1"
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$port" ]]; then
  echo "Error: --port is required"
  usage
  exit 1
fi

infer_model_name() {
  local models_url="http://localhost:${port}/v1/models"
  python - "$models_url" <<'PY'
import json
import sys
import urllib.request

models_url = sys.argv[1]
try:
    with urllib.request.urlopen(models_url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception as exc:
    raise SystemExit(f"Cannot fetch model name from {models_url}: {exc}")

for item in payload.get("data", []):
    model_id = item.get("id")
    if model_id:
        print(model_id)
        break
else:
    raise SystemExit(f"No model id returned by {models_url}")
PY
}

if [[ -z "$model" ]]; then
  model="$(infer_model_name)"
fi

if [[ ! -d "$dataset_dir" ]]; then
  echo "Error: dataset directory not found: $dataset_dir"
  echo "Put MMIU data under: ${PROJECT_DIR}/cache/data/mmiu_dataset"
  exit 1
fi

mkdir -p "$output_dir"

reuse_flag=()
if [[ "$reuse" -eq 1 ]]; then
  reuse_flag=(--reuse)
fi

echo "############ MMIU ############"
echo "port=${port}"
echo "api_base=http://localhost:${port}/v1"
echo "model=${model}"
echo "dataset_dir=${dataset_dir}"
echo "output_dir=${output_dir}"
echo "target_tasks=${target_tasks:-[all]}"
echo "judge_api_base=${judge_api_base}"
echo "judge_model=${judge_model}"

python "${PROJECT_DIR}/eval/mmiu_infer.py" \
  --api-base "http://localhost:${port}/v1" \
  --model "${model}" \
  --dataset-dir "${dataset_dir}" \
  --output-dir "${output_dir}" \
  --model-path "${model_path}" \
  --target-tasks "${target_tasks}" \
  "${reuse_flag[@]}"

python "${PROJECT_DIR}/eval/mmiu_grade.py" \
  --output-dir "${output_dir}" \
  --model "${model}" \
  --target-tasks "${target_tasks}" \
  --judge-api-base "${judge_api_base}" \
  --judge-model "${judge_model}" \
  "${reuse_flag[@]}"

python "${PROJECT_DIR}/eval/mmiu_score.py" \
  --output-dir "${output_dir}" \
  --target-tasks "${target_tasks}"
