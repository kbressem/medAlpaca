#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate

cd /sc-projects/sc-proj-cc06-medbert/medAlpaca/training/step1_supervised_finetuning

# activate conda environment
source /home/bressekk/miniconda3/etc/profile.d/conda.sh
conda activate deepspeed-chat

export HF_HOME="/sc-projects/sc-proj-cc06-medbert/hfcache"

deepspeed --num_gpus 1 main.py \
    --model_name_or_path facebook/opt-1.3b \
    --data_path "medalpaca/stack_exchange" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --deepspeed \
    --max_seq_len 512 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --num_train_epochs 1
    --wandb_run_name 'opt-1.3b' \
    --lr_scheduler_type cosine \
    --zero_stage 3