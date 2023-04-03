#!/bin/bash
#SBATCH --job-name=alpaca-7        # Specify job name
#SBATCH --partition=pgpu           # Specify partition name
#SBATCH --mem=0                    # Use entire memory of node
#SBATCH --gres=gpu:8               # Generic resources; 8 GPU
#SBATCH --exclusive                # Do not share node
#SBATCH --time=48:00:00            # Set a limit on the total run time
#SBATCH --output=logs_alp-7.o%j    # File name for standard output
#SBATCH --error=errors_alp-7.e%j   # File name for standard error output

cd /path/to/gitrepo

# activate conda environment
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate medalpaca

# recommended to manually set the hf cache dir, as the files are huge
export HF_HOME="/path/to/your/hfcache"

# feel free to adapt the below command, to run the training
# in 8bit with LoRA, fp16 with LoRA or bf16 and fsdp

torchrun --nproc_per_node=8 --master_port=9876 medalpaca/train.py \
    --model 'decapoda-research/llama-7b-hf' \
    --data_path 'medical_meadow_small.json' \
    --output_dir './lora-alpaca-7b' \
    --train_in_8bit False \
    --use_lora False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --gradient_checkpointing True \
    --global_batch_size 256 \
    --per_device_batch_size 4 \
    --wandb_project 'medalpaca' \
    --use_wandb False
