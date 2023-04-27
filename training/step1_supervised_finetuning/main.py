#!/usr/bin/env python
# Adapted from Copyright (c) Microsoft Corporation.

import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

import fire 
from typing import List

# enhance logging
import wandb  

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model



def main(data_path: List[str] = ['Dahoas/rm-static'],
         data_split='6,2,2',
         sft_only_data_path=[],
         data_output_path='/tmp/data_files/',
         model_name_or_path=None,
         per_device_train_batch_size=16,
         per_device_eval_batch_size=16,
         max_seq_len=512,
         learning_rate=1e-3,
         weight_decay=0.1,
         num_train_epochs=1,
         gradient_accumulation_steps=1,
         lr_scheduler_type="cosine",
         num_warmup_steps=0,
         output_dir="./output",
         seed=1234,
         local_rank=-1,
         gradient_checkpointing=False,
         offload=False,
         zero_stage=0,
         lora_dim=0,
         lora_module_name="decoder.layers.",
         only_optimize_lora=False,
         wandb_project_name='medalpca',
         wandb_run_name=None):
    """    
    Args:
        data_path (list): 
            Path to the training dataset. Accepted format: 
                1) a single data path
                2) multiple datasets in the form: dataset1-path dataset2-path ...
        data_split (str): 
            Comma-separated list of proportions for training phase 1, 2, and 3 data. 
            For example the split `6,2,2` will use 60% of data for phase 1, 20% for phase 2 and 20% for phase 3.
        sft_only_data_path (list): 
            Path to the dataset for only using in SFT phase.
        data_output_path (str): 
            Where to store the data-related files such as shuffle index. 
            This needs to be on a local storage of a node (not on a shared storage)
        model_name_or_path (str): 
            Path to pretrained model or model identifier from huggingface.co/models. Required.
        per_device_train_batch_size (int): 
            Batch size (per device) for the training dataloader.
        per_device_eval_batch_size (int): 
            Batch size (per device) for the evaluation dataloader.
        max_seq_len (int): 
            The maximum sequence length.
        learning_rate (float): 
            Initial learning rate (after the potential warmup period) to use for training.
        weight_decay (float): 
            Weight decay to use during training.
        num_train_epochs (int): 
            Total number of training epochs to perform.
        gradient_accumulation_steps (int): 
            Number of updates steps to accumulate before performing a backward/update pass.
        lr_scheduler_type (str): 
            The learning rate scheduler type. One of "linear", "cosine", "cosine_with_restarts", "polynomial", or "constant".
        num_warmup_steps (int): 
            Number of steps for the learning rate warmup.
        output_dir (str): 
            The output directory where the model checkpoints will be written.
        seed (int): 
            Random seed for initialization.
        local_rank (int): 
            Local rank for distributed training on GPUs. Set to -1 for non-distributed setup.
        gradient_checkpointing (bool): 
            Enable gradient checkpointing to save memory at the cost of slower backward pass.
        offload (bool): 
            Enable offloading optimizer and/or gradients for ZeRO-Offload.
        zero_stage (int): 
            ZeRO stage to use. Can be 0 (no ZeRO), 1, 2, or 3.
        lora_dim (int): 
            Dimension of LoRA (local receptive field) for each layer. Set to 0 to disable LoRA.
        lora_module_name (str): 
            Name of the sub-module that should be converted to LoRA.
        only_optimize_lora (bool): 
            If True, only optimize LoRA parameters and freeze the rest.
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

    ds_config = get_train_ds_config(offload=offload,
                                    stage=zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = per_device_train_batch_size
    ds_config[
        'train_batch_size'] = per_device_train_batch_size * torch.distributed.get_world_size(
        ) * gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(seed)

    assert not offload, "zero-offload is not currently supported but coming soon!"

    torch.distributed.barrier()

    if "llama" in model_name_or_path: 
        GetTokenizer = LlamaTokenizer
        GetModel = LlamaForCausalLM
    else: 
        GetTokenizer = AutoTokenizer
        GetModel = AutoModelForCausalLM
        
    tokenizer = GetTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer="llama" not in model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    if "llama" in model_name_or_path: 
        tokenizer.pad_token_id = 0
    
    model = create_hf_model(GetModel, model_name_or_path,
                            tokenizer, ds_config)

    if lora_dim > 0:
        model = convert_linear_layer_to_lora(model, lora_module_name,
                                             lora_dim)
        if only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    # Prepare the data
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        local_rank,
        [data_path] if isinstance(data_path, str) else data_path,
        data_split,
        data_output_path,
        train_phase,
        seed,
        tokenizer,
        max_seq_len,
        sft_only_data_path=sft_only_data_path)

    # DataLoaders creation:
    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=per_device_eval_batch_size)

    def evaluation(model, eval_dataloader, epoch):
        model.eval()
        losses = 0
        total_steps = len(eval_dataloader)
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        if global_rank == 0:
            wandb.log({"eval/loss": losses, "eval/epoch": epoch + 1})  # Log eval loss to wandb
            wandb.log({"eval/perplexity": perplexity, "eval/epoch": epoch + 1})  # Log perplexity to wandb
        return perplexity

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, weight_decay)

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

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    if global_rank == 0:
        wandb.init(project=wandb_project_name, config=args)  
        wandb.run.name = wandb_run_name or f"{model_name_or_path}_deepspeed_step1"

    for epoch in range(num_train_epochs):
        total_steps = len(train_dataloader)
        model.train()
        
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()
            if global_rank == 0:
                wandb.log({"train/loss": loss, "train/epoch": epoch + 1, "train/step": step + total_steps * epoch})  # Log train loss to wandb

        # Evaluate perplexity on the validation set.
        perplexity = evaluation(model, eval_dataloader, epoch)
        model.tput_timer.update_epoch_count()

    if output_dir is not None:
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
        model = convert_lora_to_linear_layer(model)

        if global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  global_rank,
                                  output_dir,
                                  zero_stage=zero_stage)
    if global_rank == 0: 
        wandb.finish()  # Finish the wandb run

if __name__ == "__main__":
    fire.Fire(main)
