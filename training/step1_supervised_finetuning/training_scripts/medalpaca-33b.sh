#!/bin/bash
#SBATCH --job-name=alpaca-33-deepspeed  # Specify job name
#SBATCH --partition=pgpu                # Specify partition name
#SBATCH --mem=0                         # Use entire memory of node
#SBATCH --gres=gpu:8                    # Generic resources; 8 GPU
#SBATCH --exclusive                     # Do not share node
#SBATCH --time=120:00:00                # Set a limit on the total run time
#SBATCH --output=logs_alp-33.o%j        # File name for standard output
#SBATCH --error=errors_alp-33.e%j       # File name for standard error output


cd /sc-projects/sc-proj-cc06-medbert/medAlpaca/training/step1_supervised_finetuning

# activate conda environment
source /home/bressekk/miniconda3/etc/profile.d/conda.sh
conda activate deepspeed-chat

export HF_HOME="/sc-projects/sc-proj-cc06-medbert/hfcache"

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname ${SLURM_JOB_NODELIST} | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=${SLURM_NODEID}
export WORLD_SIZE=${SLURM_JOB_NUM_NODES}

# if you have multiple nodes, create a hostfile from the SLURM_JOB_NODELIST
# it should look like this
# s-sc-dgx01 slots=8
# s-sc-dgx02 slots=8
# 
# then add --hostfile hostfile.txt to the deepspeed args

deepspeed --num_nodes ${SLURM_JOB_NUM_NODES} --num_gpus 8 main.py \
    --data_path "medalpaca/stack_exchange" \
    --model_name_or_path 'decapoda-research/llama-30b-hf' \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --num_warmup_steps 1000 \
    --max_seq_len 512 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --num_train_epochs 10 \
    --wandb_run_name 'medalpaca-30b' \
    --lr_scheduler_type cosine \
    --zero_stage 3 \
    --output_dir "/sc-projects/sc-proj-cc06-medbert/alpaca-zoo/in-training/medalpaca-deepspeed-33b" \
    --gradient_checkpointing True \
    --seed 42 \
    --deepspeed 