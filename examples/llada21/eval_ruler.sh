#!/usr/bin/env bash
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

model_name_or_path="ML-GSAI/LLaDA2.1-mini"
adapter_model_name_or_path=""
base_model_name_or_path=""
num_gpu=1
max_length=8192
tasks="ruler"
metadata='{"max_seq_lengths":[4096,8192]}'
output_path="./eval_results"
extra_model_args=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --adapter_model_name_or_path)
      adapter_model_name_or_path="$2"; shift 2 ;;
    --base_model_name_or_path)
      base_model_name_or_path="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    --max_length)
      max_length="$2"; shift 2 ;;
    --tasks)
      tasks="$2"; shift 2 ;;
    --metadata)
      metadata="$2"; shift 2 ;;
    --output_path)
      output_path="$2"; shift 2 ;;
    --extra_model_args)
      extra_model_args="$2"; shift 2 ;;
    *)
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

model_args="pretrained=${model_name_or_path},max_length=${max_length}"
if [[ -n "${adapter_model_name_or_path}" ]]; then
  model_args="${model_args},adapter_model_name_or_path=${adapter_model_name_or_path}"
fi
if [[ -n "${base_model_name_or_path}" ]]; then
  model_args="${model_args},base_model_name_or_path=${base_model_name_or_path}"
fi
if [[ -n "${extra_model_args}" ]]; then
  model_args="${model_args},${extra_model_args}"
fi

if [[ "${metadata}" != *"\"tokenizer\""* && "${metadata}" != *"\"pretrained\""* ]]; then
  tokenizer_source="${base_model_name_or_path:-${model_name_or_path}}"
  metadata="$(python - "${metadata}" "${tokenizer_source}" <<'PY'
import json
import sys

raw_metadata = sys.argv[1]
tokenizer_source = sys.argv[2]

try:
    data = json.loads(raw_metadata)
except json.JSONDecodeError as e:
    raise SystemExit(f"Invalid --metadata JSON: {e}")

if not isinstance(data, dict):
    raise SystemExit("--metadata must be a JSON object.")

data.setdefault("pretrained", tokenizer_source)
print(json.dumps(data, separators=(",", ":")))
PY
)"
fi

accelerate launch --num_processes "${num_gpu}" memdlm/eval/llada21.py \
  --tasks "${tasks}" \
  --model llada21 \
  --num_fewshot 0 \
  --model_args "${model_args}" \
  --metadata "${metadata}" \
  --output_path "${output_path}"
