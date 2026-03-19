#!/usr/bin/env bash
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

model_name_or_path="ML-GSAI/LLaDA2.1-mini"
adapter_model_name_or_path=""
base_model_name_or_path=""
num_gpu=1
max_length=4096
max_new_tokens=16
steps=32
block_size=32
metadata=""
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
    --max_new_tokens)
      max_new_tokens="$2"; shift 2 ;;
    --steps)
      steps="$2"; shift 2 ;;
    --block_size)
      block_size="$2"; shift 2 ;;
    --metadata)
      metadata="$2"; shift 2 ;;
    --extra_model_args)
      extra_model_args="$2"; shift 2 ;;
    *)
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

model_args="pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},max_length=${max_length}"
if [[ -n "${adapter_model_name_or_path}" ]]; then
  model_args="${model_args},adapter_model_name_or_path=${adapter_model_name_or_path}"
fi
if [[ -n "${base_model_name_or_path}" ]]; then
  model_args="${model_args},base_model_name_or_path=${base_model_name_or_path}"
fi
if [[ -n "${extra_model_args}" ]]; then
  model_args="${model_args},${extra_model_args}"
fi

metadata_args=""
if [[ -n "${metadata}" ]]; then
  metadata_args="--metadata ${metadata}"
fi

# Usage examples:
#   bash examples/llada21/eval_run.sh
#   bash examples/llada21/eval_run.sh --metadata '{"max_seq_lengths":"2k"}'
#   bash examples/llada21/eval_run.sh --metadata '{"max_seq_lengths":"4k"}' --max_length 8192 --extra_model_args "mem_enabled=True"

accelerate launch --num_processes "${num_gpu}" memdlm/eval/llada21.py \
  --tasks babilong \
  --model llada21 \
  --num_fewshot 2 \
  --model_args "${model_args}" \
  ${metadata_args} \
  --output_path ./eval_results
