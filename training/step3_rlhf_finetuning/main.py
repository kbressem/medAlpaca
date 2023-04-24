#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""

engine = DeepSpeedRLHFEngine(actor_model_name_or_path=actor_model_name_or_path,
                             critic_model_name_or_path=critic_model_name_or_path,
                             tokenizer=tokenizer,
                             args=args)
trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
    out = trainer.generate_experience(prompt_batch)
    actor_loss, critic_loss = trainer.train_rlhf(out)

"""
import argparse
import os
import random
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
)

import deepspeed

from ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from rlhf_engine import DeepSpeedRLHFEngine

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model
from utils.module.lora import convert_lora_to_linear_layer

import fire


def main(data_path=['Dahoas/rm-static'],
         data_split='6,2,2',
         data_output_path='/tmp/data_files',
         unsupervised_dataset_name=None,
         unsupervised_dataset_config_name=None,
         unsup_coef=27.8,
         actor_model_name_or_path=None,
         critic_model_name_or_path=None,
         num_padding_at_beginning=1,
         per_device_train_batch_size=16,
         per_device_mini_train_batch_size=16,
         generation_batch_numbers=1,
         ppo_epochs=1,
         max_prompt_seq_len=256,
         max_answer_seq_len=256,
         actor_learning_rate=9.65e-6,
         critic_learning_rate=5e-6,
         actor_weight_decay=0.1,
         critic_weight_decay=0.1,
         num_train_epochs=1,
         lr_scheduler_type="cosine",
         gradient_accumulation_steps=1,
         num_warmup_steps=100,
         output_dir="./output",
         seed=None,
         preprocessing_num_workers=None,
         local_rank=-1,
         enable_hybrid_engine=False,
         unpin_actor_parameters=False,
         release_inference_cache=False,
         inference_tp_size=1,
         tp_gather_partition_size=8,
         offload=False,
         offload_reference_model=False,
         actor_zero_stage=0,
         critic_zero_stage=0,
         actor_gradient_checkpointing=False,
         critic_gradient_checkpointing=False,
         actor_lora_dim=0,
         actor_lora_module_name="decoder.layers.",
         critic_lora_dim=0,
         critic_lora_module_name="decoder.layers.",
         only_optimize_lora=False,
         enable_ema=False,
         wandb_project_name='medalpca',
         wandb_run_name=None):
    """
    Args:
        data_path (list of str, optional): 
        Path to the training dataset. 
        Accepted format: 
            1) a single data path, 
                2) multiple datasets in the form: dataset1-path dataset2-path .... Defaults to ['Dahoas/rm-static'].
        data_split (str, optional): 
            Comma-separated list of proportions for training phase 1, 2, and 3 data. 
            For example the split `2,4,4` will use 60% of data for phase 1, 20% for phase 2 and 20% for phase 3. 
            Defaults to '6,2,2'.
        data_output_path (str, optional): 
            Where to store the data-related files such as shuffle index. 
            This needs to be on a local storage of a node (not on a shared storage). 
            Defaults to '/tmp/data_files'.
        unsupervised_dataset_name (str, optional): 
            The name of the dataset to use (via the datasets library). Defaults to None.
        unsupervised_dataset_config_name (str, optional): 
            The configuration name of the dataset to use (via the datasets library). Defaults to None.
        unsup_coef (float, optional): 
            gamma in Equation 2 from InstructGPT paper. Defaults to 27.8.
        actor_model_name_or_path (str, optional): 
            Path to pretrained model or model identifier from huggingface.co/models for the actor model. Defaults to None.
        critic_model_name_or_path (str, optional): 
            Path to pretrained model or model identifier from huggingface.co/models for the critic model. Defaults to None.
        num_padding_at_beginning (int, optional): 
            OPT model has a fixed number (1) of padding tokens at the beginning of the input. 
            We did not see this in other models but keep it as an option for now. Defaults to 1.
        per_device_train_batch_size (int, optional): 
            Batch size (per device) for the training dataloader and generation purpose. Defaults to 16.
        per_device_mini_train_batch_size (int, optional): 
            Mini Batch size (per device) for the training dataloader and training purpose. Defaults to 16.
        generation_batch_numbers (int, optional): 
            Generate x batches to go to training mode. Defaults to 1.
        ppo_epochs (int, optional):
            For generated data, how many PPO training epochs to run. Defaults to 1.
        max_prompt_seq_len (int, optional): 
            The maximum sequence length. Defaults to 256.
        max_answer_seq_len (int, optional): 
            The maximum sequence length. Defaults to 256.
        actor_learning_rate (float, optional): 
            Initial learning rate (after the potential warmup period) to use for the actor model. Defaults to 9.65e-6.
        critic_learning_rate (float, optional): 
            Initial learning rate (after the potential warmup period) to use for the critic model. Defaults to 5e-6.
        actor_weight_decay (float, optional): 
            Weight decay to use for the actor model. Defaults to 0.1.
        critic_weight_decay (float, optional): 
            Weight decay to use for the critic model. Defaults to 0.1.
        num_train_epochs (int, optional): 
            Total number of training epochs to perform. Defaults to 1.
        lr_scheduler_type (str, optional): 
            The scheduler type to use. 
            Choices: ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]. 
            Defaults to "cosine".
        gradient_accumulation_steps (int, optional): 
            Number of steps for the warmup in the lr scheduler. Defaults to 1.
        num_warmup_steps (int, optional): 
            Number of steps for the warmup in the lr scheduler. Defaults to 100.
        output_dir (str, optional): 
            Where to store the model. Defaults to None.
        seed (int, optional): 
            A seed for reproducible training. Defaults to None.
        preprocessing_num_workers (int, optional): 
            The number of processes to use for the preprocessing. Defaults to None.
        local_rank (int, optional): 
            local_rank for distributed training on GPUs. Defaults to -1.
        enable_hybrid_engine (bool, optional): 
            Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed. 
            Defaults to False.
        unpin_actor_parameters (bool, optional): 
            Unpin actor's parameters during generation. 
            This makes generation slower but requires less memory. Defaults to False.
        release_inference_cache (bool, optional): 
            Release the memory cache used for inference. 
            This makes generation preparation slower but might increase e2e throughput by using larger batch size. Defaults to False.
        inference_tp_size (int, optional): 
            Tensor-parallelism degree used for the inference-optimization. 
            Please note hybrid-engine need to be enabled when using this feature. Defaults to 1.
        tp_gather_partition_size (int, optional): 
            Granularity to bring in layers for TP sharding inside the hybrid engine. 
            Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature. Defaults to 8.
        offload (bool, optional): 
            Enable ZeRO Offload techniques. Defaults to False.
        offload_reference_model (bool, optional): 
            Enable ZeRO Offload techniques for reference model. Defaults to False.
        actor_zero_stage (int, optional): 
            ZeRO optimization stage for Actor model (and clones). Defaults to 0.
        critic_zero_stage (int, optional): 
            ZeRO optimization stage for Critic model (and reward). Defaults to 0.
        actor_gradient_checkpointing (bool, optional): 
            Enable HF gradient checkpointing for Actor model. Defaults to False.
        critic_gradient_checkpointing (bool, optional): 
            Enable HF gradient checkpointing for Critic model. Defaults to False.
        actor_lora_dim (int, optional): 
            If > 0, use LoRA for efficient training for the actor model. Defaults to 0.
        actor_lora_module_name (str, optional): 
            The scope of LoRA for the actor model. Defaults to "decoder.layers.".
        critic_lora_dim (int, optional): 
            If > 0, use LoRA for efficient training for the critic model. Defaults to 0.
        critic_lora_module_name (str, optional): 
            The scope of LoRA for the critic model. Defaults to "decoder.layers.".
        only_optimize_lora (bool, optional): 
            Only optimize the LoRA parameters. Defaults to False.
        enable_ema (bool, optional): 
            Enable EMA checkpoint for the model. Defaults to False.
        wandb_project_name (str): 
            The project name to use with Weights & Biases for logging.
        wandb_run_name (str): 
            The run name to use with Weights & Biases for logging.        
    """
    args = locals()
    
    if (actor_gradient_checkpointing
        and actor_lora_dim > 0) or (critic_gradient_checkpointing
                                         and critic_lora_dim > 0):
        assert (
            not only_optimize_lora
        ), "--{actor,critic}_gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time."

    if inference_tp_size > 1:
        assert (
            actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"


    if local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    global_rank = torch.distributed.get_rank()

    assert not offload, "zero-offload is not currently supported but coming soon!"

    unsupervised_training_enabled = unsupervised_dataset_name and unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        gradient_accumulation_steps_actor = gradient_accumulation_steps * 2
    else:
        gradient_accumulation_steps_actor = gradient_accumulation_steps

    args["gradient_accumulation_steps_actor"] = gradient_accumulation_steps_actor

    # If passed along, set the training seed now.
    set_random_seed(seed)
    torch.distributed.barrier()

    # create common tokenizer based on actor model
    tokenizer = AutoTokenizer.from_pretrained(actor_model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # create datasets
    train_phase = 3
    prompt_train_dataset, _ = create_prompt_dataset(
        local_rank, data_path, data_split,
        data_output_path, train_phase, seed, tokenizer,
        max_prompt_seq_len)
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(max_prompt_seq_len,
                                     inference_tp_size)
    if local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=per_device_train_batch_size)
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=per_device_train_batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader

    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (per_device_train_batch_size / per_device_mini_train_batch_size) * \
        ppo_epochs / gradient_accumulation_steps
    num_total_iters = int(num_train_epochs * num_update_steps_per_epoch)

    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=actor_model_name_or_path,
        critic_model_name_or_path=critic_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)

    end_of_conversation_token = "<|endoftext|>"

    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(generation_batch_numbers,
                                   per_device_mini_train_batch_size)
    unsup_mini_dataset = MiniDataset(generation_batch_numbers,
                                     per_device_mini_train_batch_size)

    # Train!
    print_rank_0("***** Running training *****", global_rank)
    if global_rank == 0:
        wandb.init(project=wandb_project_name, config=args)  
        wandb.run.name = wandb_run_name or f"{model_name_or_path}_deepspeed_step3"
    
    total_steps = len(prompt_train_dataloader)
    for epoch in range(num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            global_rank)
        for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)):
            batch_prompt = to_device(batch_prompt, device)
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * per_device_train_batch_size])
            prompts = batch_prompt['prompt']
            length = prompts.size(-1)
            if length > max_prompt_seq_len:
                prompts = prompts[:, length - max_prompt_seq_len:]
                raise ValueError("Prompt length is too long")

            out = trainer.generate_experience(prompts)
            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                critic_loss, actor_loss, unsuper_loss = 0, 0, 0
                average_reward = 0

                if actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()

                for ppo_ep in range(ppo_epochs):
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                        critic_loss += actor_loss.item()
                        actor_loss += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()

                        if unsupervised_training_enabled:
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, unsup_coef)
                            unsuper_loss += unsup_loss.item()

                        inner_iter += 1
                        if enable_ema:
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=actor_zero_stage)

                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)
                if global_rank == 0:
                    wandb.log({"train/act_loss": actor_loss/inner_iter, "train/cri_loss": critic_loss/inner_iter, "train/unsuper_loss": unsuper_loss/inner_iter,  "train/epoch": epoch + 1, "train/step": step + total_steps * epoch})  # Log train loss 
                print_rank_0(
                    f'epoch: {epoch}|step: {step}|ppo_ep: {ppo_ep+1}|act_loss: {actor_loss/inner_iter}|cri_loss: {critic_loss/inner_iter}|unsuper_loss: {unsuper_loss/inner_iter}',
                    global_rank)
                average_reward = get_all_reduce_mean(average_reward).item()
                print_rank_0(
                    f"average reward score: {average_reward/inner_iter}",
                    global_rank)
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    global_rank)

            if actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()

    if output_dir is not None:
        print_rank_0('saving model ...')
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)
        if enable_ema:
            rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                rlhf_engine.actor_ema)

        if torch.distributed.get_rank() == 0:
            save_hf_format(rlhf_engine.actor,
                           tokenizer,
                           args,
                           sub_folder='actor')
            save_hf_format(rlhf_engine.critic,
                           tokenizer,
                           args,
                           sub_folder='critic')
            if enable_ema:
                save_hf_format(rlhf_engine.actor_ema,
                               tokenizer,
                               args,
                               sub_folder='actor_ema')

        if actor_zero_stage == 3:
            save_zero_three_model(rlhf_engine.actor,
                                  global_rank=global_rank,
                                  save_dir=os.path.join(
                                      output_dir, 'actor'),
                                  zero_stage=actor_zero_stage)
            if enable_ema:
                save_zero_three_model(rlhf_engine.actor_ema,
                                      global_rank=global_rank,
                                      save_dir=os.path.join(
                                          output_dir, 'actor_ema'),
                                      zero_stage=actor_zero_stage)
        if critic_zero_stage == 3:
            save_zero_three_model(rlhf_engine.critic,
                                  global_rank=global_rank,
                                  save_dir=os.path.join(
                                      output_dir, 'critic'),
                                  zero_stage=critic_zero_stage)


    if global_rank == 0: 
        wandb.finish()  # Finish the wandb run

if __name__ == "__main__":
    fire.Fire(main)
