#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For datasets with custom code

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling for multi-GPU communication
export NCCL_DEBUG=warn                      # Show NCCL warnings without flooding logs

# ===== Defaults =====
model_name_or_path="inclusionAI/LLaDA-MoE-7B-A1B-Base"
adapter_model_name_or_path=""
base_model_name_or_path=""
num_gpu=1
max_length=8192
max_new_tokens=256
steps=128
block_size=128
cfg=0.0
extra_model_args="mem_enabled=True,mem_num_inner_steps=2,mem_inner_rank=32,mem_inner_alpha=64.0,mem_inner_layer_fraction=0.1,mem_masking_strategy=pmc"
tasks="longbench"   # or longbench_e, longbench_triviaqa, etc.
output_path="./eval_results"

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
    --cfg)
      cfg="$2"; shift 2 ;;
    --tasks)
      tasks="$2"; shift 2 ;;
    --output_path)
      output_path="$2"; shift 2 ;;
    --extra_model_args)
      extra_model_args="$2"; shift 2 ;;
    *)
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

model_args="pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},max_length=${max_length},cfg=${cfg}"
if [[ -n "${adapter_model_name_or_path}" ]]; then
  model_args="${model_args},adapter_model_name_or_path=${adapter_model_name_or_path}"
fi
if [[ -n "${base_model_name_or_path}" ]]; then
  model_args="${model_args},base_model_name_or_path=${base_model_name_or_path}"
fi
if [[ -n "${extra_model_args}" ]]; then
  model_args="${model_args},${extra_model_args}"
fi

accelerate launch --num_processes "${num_gpu}" memdlm/eval/llada.py \
  --tasks "${tasks}" \
  --model llada \
  --num_fewshot 0 \
  --model_args "${model_args}" \
  --output_path "${output_path}"