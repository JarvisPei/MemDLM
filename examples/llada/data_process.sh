MODEL_PATH="inclusionAI/LLaDA-MoE-7B-A1B-Base"

python -c "import memdlm; from dllm.tools.preprocess_sft_dataset import main; main()" -- \
    --model_name_or_path $MODEL_PATH \
    --sft_map_fn_path "dllm.utils.default_sft_map_fn" \
    --dataset_args "Yukang/LongAlpaca-12k" \
    --output_dir "data/sft/llada/longalpaca-12k" \
    --num_proc 64

python dllm/tools/preprocess_sft_dataset.py \
    --model_name_or_path $MODEL_PATH \
    --sft_map_fn_path "dllm.utils.default_sft_map_fn" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "data/sft/llada/tulu-3-sft-mixture" \
    --num_proc 64

python dllm/tools/preprocess_sft_dataset.py \
    --model_name_or_path $MODEL_PATH \
    --sft_map_fn_path "dllm.utils.default_sft_map_fn" \
    --dataset_args "tatsu-lab/alpaca" \
    --output_dir "data/sft/llada/alpaca" \
    --num_proc 64