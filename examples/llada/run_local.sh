export CUDA_VISIBLE_DEVICES=0

accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/llada/sft.py \
        --load_in_4bit True --lora True \
        --model_name_or_path "inclusionAI/LLaDA-MoE-7B-A1B-Base" \
        --dataset_args "tatsu-lab/alpaca" \
        --load_preprocessed_data True \
        --max_length 1024 \
        --num_train_epochs 5 \
        --learning_rate 2e-5 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --output_dir "models/LLaDA-MoE-7B-A1B-Base/alpaca" \
        --save_only_model True \