#!/bin/bash
#SBATCH --job-name=alpaca-7-deepspeed   # Specify job name
#SBATCH --partition=pgpu                # Specify partition name
#SBATCH --mem=0                         # Use entire memory of node
#SBATCH --gres=gpu:4                    # Generic resources; 8 GPU
#SBATCH --exclusive                     # Do not share node
#SBATCH --time=120:00:00                # Set a limit on the total run time
#SBATCH --output=logs_alp-7.o%j         # File name for standard output
#SBATCH --error=errors_alp-7.e%j        # File name for standard error output


cd /sc-projects/sc-proj-cc06-medbert/medAlpaca/training/step3_rlhf_finetuning

# activate conda environment
source /home/bressekk/miniconda3/etc/profile.d/conda.sh
conda activate deepspeed-chat

deepspeed main.py \
   --actor_model_name_or_path facebook/opt-1.3b \
   --critic_model_name_or_path  /sc-projects/sc-proj-cc06-medbert/opt-350m \
   --actor_zero_stage 1 \
   --critic_zero_stage 1 \
   --num_padding_at_beginning 1 \
   --gradient_accumulation_steps 2 \
   --deepspeed \
   --enable_hybrid_engine \
   --actor_gradient_checkpointing 
