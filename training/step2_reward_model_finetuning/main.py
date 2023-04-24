#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
from typing import List

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters

import fire
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = True

def main(data_path: List[str] = ['Dahoas/rm-static'],
         data_split: str = '6,2,2',
         data_output_path: str = '/tmp/data_files/',
         model_name_or_path: str = '',
         num_padding_at_beginning: int = 1,
         per_device_train_batch_size: int = 16,
         per_device_eval_batch_size: int = 16,
         max_seq_len: int = 512,
         learning_rate: float = 5e-5,
         weight_decay: float = 0.1,
         num_train_epochs: int = 1,
         gradient_accumulation_steps: int = 1,
         lr_scheduler_type: SchedulerType = "cosine",
         num_warmup_steps: int = 0,
         output_dir: str = "./output",
         seed: int = 1234,
         local_rank: int = -1,
         gradient_checkpointing: bool = False,
         offload: bool = False,
         zero_stage: int = 0,
         lora_dim: int = 0,
         lora_module_name: str = "decoder.layers.",
         only_optimize_lora: bool = False,
         wandb_project_name='medalpca',
         wandb_run_name=None):
    """
    Args:
        data_path (List[str], optional): Path to the training dataset. 
            Accepted format:
                1) a single data path
                2) multiple datasets in the form: dataset1-path dataset2-path ... Defaults to ['Dahoas/rm-static'].
        data_split (str, optional): Comma-separated list of proportions for training
            phase 1, 2, and 3 data. For example the split `2,4,4` will use 60% of data for phase 1, 20% for phase 2
            and 20% for phase 3. Defaults to '6,2,2'.
        data_output_path (str, optional): 
            Where to store the data-related files such as shuffle index. Defaults to '/tmp/data_files/'.
        model_name_or_path (str, optional): 
            Path to pretrained model or model identifier from huggingface.co/models. Defaults to ''.
        num_padding_at_beginning (int, optional): 
            OPT model has a fixed number (1) of padding tokens at the beginning of the input.
            We did not see this in other models but keep it as an option for now. Defaults to 1.
        per_device_train_batch_size (int, optional): 
            Batch size (per device) for the training dataloader. Defaults to 16.
        per_device_eval_batch_size (int, optional): 
            Batch size (per device) for the evaluation dataloader. Defaults to 16.
        max_seq_len (int, optional): 
            The maximum sequence length. Defaults to 512.
        learning_rate (float, optional): 
            Initial learning rate (after the potential warmup period) to use. Defaults to 5e-5.
        weight_decay (float, optional): 
            Weight decay to use. Defaults to 0.1.
        num_train_epochs (int, optional): 
            Total number of training epochs to perform. Defaults to 1.
        gradient_accumulation_steps (int, optional): 
            Number of updates steps to accumulate before performing 
            a backward/update pass. Defaults to 1.
        lr_scheduler_type (SchedulerType, optional): 
            The scheduler type to use. Defaults to "cosine".
        num_warmup_steps (int, optional): 
            Number of steps for the warmup in the lr scheduler. Defaults to 0.
        output_dir (str, optional): 
            Where to store the model. Defaults to None.
        seed (int, optional): 
            A seed for reproducible training. Defaults to 1234.
        local_rank (int, optional): 
            local_rank for distributed training on GPUs. Defaults to -1.
        gradient_checkpointing (bool, optional): 
            Enable HF gradient checkpointing for Actor model. Defaults to False.
        offload (bool, optional): 
            Enable ZeRO Offload techniques. Defaults to False.
        zero_stage (int, optional): 
            ZeRO optimization stage for Actor model (and clones). Defaults to 0.
        lora_dim (int, optional): 
            If > 0, use LoRA for efficient training. Defaults to 0.
        lora_module_name (str, optional): 
            The scope of LoRA. Defaults to "decoder.layers.".
        only_optimize_lora (bool, optional): 
            Only optimize the LoRA parameters. Defaults to False.
        wandb_project_name (str): 
            The project name to use with Weights & Biases for logging.
        wandb_run_name (str): 
            The run name to use with Weights & Biases for logging.
    """

    args = locals()

    if local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    global_rank = torch.distributed.get_rank()

    assert not offload, "zero-offload is not currently supported but coming soon!"

    ds_config = get_train_ds_config(offload=offload,
                                    stage=zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = per_device_train_batch_size
    ds_config[
        'train_batch_size'] = per_device_train_batch_size * torch.distributed.get_world_size(
        ) * gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(seed)
    torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token

    rm_model = create_critic_model(model_name_or_path, tokenizer,
                                   ds_config, num_padding_at_beginning)

    if lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                lora_module_name,
                                                lora_dim)
        if only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)

    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        local_rank, data_path, data_split,
        data_output_path, train_phase, seed, tokenizer,
        max_seq_len)

    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=per_device_eval_batch_size)

    def evaluation_reward(model, eval_dataloader, epoch):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            scores += outputs["chosen_mean_scores"].mean().float()
            if step == 99:  # For faster evaluation and debugging
                break
            acc = correct_predictions / total_predictions
            scores = scores / (step + 1)
        try:
            acc = get_all_reduce_mean(acc).item()
            scores = get_all_reduce_mean(scores).item()
        except:
            pass
        if global_rank == 0:
            wandb.log({"eval/acc": acc, "eval/epoch": epoch + 1})  # Log eval loss to wandb
            wandb.log({"eval/scores": scores, "eval/epoch": epoch + 1})  # Log perplexity to wandb

        return scores, acc

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_epochs * num_update_steps_per_epoch,
    )

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    if global_rank == 0:
        wandb.init(project=wandb_project_name, config=args)  
        wandb.run.name = wandb_run_name or f"{model_name_or_path}_deepspeed_step2"
    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{num_train_epochs} *****",
        global_rank)
    reward_score, acc = evaluation_reward(rm_model, eval_dataloader, -1)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
        global_rank)

    for epoch in range(num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            global_rank)
        rm_model.train()
        mean_loss = 0
        total_steps = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
            
            if global_rank == 0:
                wandb.log({"train/loss": loss, "train/epoch": epoch + 1, "train/step": step + total_steps * epoch})  # Log train loss to wandb           
        print_rank_0(
            f"Epoch {epoch+1}/{num_train_epochs} with loss {mean_loss/(step+1)}",
            global_rank)
        # Evaluate reward_loss on the validation set.
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch+1}/{num_train_epochs} *****",
            global_rank)
        reward_score, acc = evaluation_reward(rm_model, eval_dataloader, epoch)
        print_rank_0(
            f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
            global_rank)
        rm_model.tput_timer.update_epoch_count()

    if output_dir is not None:
        print_rank_0('saving model ...', global_rank)
        rm_model = convert_lora_to_linear_layer(rm_model)

        if global_rank == 0:
            save_hf_format(rm_model, tokenizer, args)
        if zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(rm_model,
                                  global_rank,
                                  output_dir,
                                  zero_stage=zero_stage)
    if global_rank == 0: 
        wandb.finish()  # Finish the wandb run

if __name__ == "__main__":
    fire.Fire(main)
