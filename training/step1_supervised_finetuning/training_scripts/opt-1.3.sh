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
    --gradient_accumulation_steps 2 \
    --lora_dim 128  \
    --deepspeed \
