#!/bin/bash
#SBATCH --job-name=alpaca-33-deepspeed  # Specify job name
#SBATCH --partition=pgpu                # Specify partition name
#SBATCH --mem=0                         # Use entire memory of node
#SBATCH --gres=gpu:8                    # Generic resources; 8 GPU
#SBATCH --exclusive                     # Do not share node
#SBATCH --time=120:00:00                # Set a limit on the total run time
#SBATCH --output=logs_alp-33.o%j        # File name for standard output
#SBATCH --error=errors_alp-33.e%j       # File name for standard error output
#SBATCH --nodes=2                       # Number of nodes


cd /sc-projects/sc-proj-cc06-medbert/medAlpaca/training/step1_supervised_finetuning

# activate conda environment
source /home/bressekk/miniconda3/etc/profile.d/conda.sh
conda activate deepspeed-chat

export HF_HOME="/sc-projects/sc-proj-cc06-medbert/hfcache"

deepspeed --num_gpus 1 main.py \
    --data_path 'medalpaca/medical_meadow_stackexchange' \
    --model_name_or_path 'decapoda-research/llama-30b-hf' \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --deepspeed \
    --max_seq_len 512 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --num_train_epochs 1
    --wandb_run_name 'medalpaca-30b' \
    --lr_scheduler_type cosine \
    --zero_stage 3
   
