export CUDA_VISIBLE_DEVICES=0

accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/llada21/sft.py \
    --load_in_4bit True --lora True \
    --model_name_or_path "ML-GSAI/LLaDA2.1-mini" \
    --dataset_args "Yukang/LongAlpaca-12k" \
    --load_preprocessed_data True \
    --max_length 4096 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --output_dir "models/LLaDA2.1-mini/longalpaca-memory" \
    --save_only_model True \
    --mem_enabled True \
    --mem_masking_strategy "pmc" \
    --mem_num_inner_steps 2 \
    --mem_inner_lr 0.1 \
    --mem_inner_grad_clip 1.0 \
    --mem_inner_rank 32 \
    --mem_inner_alpha 64.0 \
    --mem_inner_layer_fraction 0.1 \
    --mem_inner_target_modules "gate_proj,up_proj,down_proj"
