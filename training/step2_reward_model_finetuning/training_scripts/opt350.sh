#!/bin/bash
#SBATCH --job-name=alpaca-7-deepspeed   # Specify job name
#SBATCH --partition=pgpu                # Specify partition name
#SBATCH --mem=0                         # Use entire memory of node
#SBATCH --gres=gpu:4                    # Generic resources; 8 GPU
#SBATCH --exclusive                     # Do not share node
#SBATCH --time=120:00:00                # Set a limit on the total run time
#SBATCH --output=logs_alp-7.o%j         # File name for standard output
#SBATCH --error=errors_alp-7.e%j        # File name for standard error output


cd /sc-projects/sc-proj-cc06-medbert/medAlpaca/training/step2_reward_model_finetuning

# activate conda environment
source /home/bressekk/miniconda3/etc/profile.d/conda.sh
conda activate deepspeed-chat

deepspeed --num_gpus 1 main.py \
    --model_name_or_path facebook/opt-350m \
    --num_padding_at_beginning 1 \
    --gradient_accumulation_steps 2 \
    --deepspeed 