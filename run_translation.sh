#!/bin/bash
MODEL_NAME_OR_PATH="alirezamsh/small100"
# MODEL_NAME_OR_PATH="facebook/nllb-200-distilled-600M"

train_args=(
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --target_langs de
    --eval_langs de
    --output_dir "./out/train-translation"
    --max_source_length 1024
    --max_target_length 256
    --num_beams 5
    --seed 42
    --preprocessing_num_workers 8
    # --debug
    --overwrite_cache

    # --exclude_mt_loss
    --kd_lambda 0.4

    # --resume_from_checkpoint
    --per_device_train_batch_size 8
    --gradient_accumulation_steps 1
    --learning_rate 4e-6
    --max_train_steps 5000
    --lr_scheduler_type "inverse_sqrt"
    --num_warmup_steps 200
    --per_device_eval_batch_size 16
    --eval_steps 100
)
gc_args=(
    --gc_training
    --gc_lambda 1.0

    --gc_loss_dir "original"
    --gc_loss_type "supcon"
    # --gc_anchor_type "original"
    --gc_anchor_type "both"
    --gc_positive_type "both"
    --gc_negative_type "full"
)

set -x
accelerate launch --mixed_precision=fp16 run_translation.py "${train_args[@]}" "${gc_args[@]}"
