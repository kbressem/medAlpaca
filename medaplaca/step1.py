"""
Inspired and adapted from: 
1. https://github.com/huggingface/accelerate/blob/main/examples/by_feature/fsdp_with_peak_mem_tracking.py#L191
"""

import os
import sys
from typing import Tuple, Union, List, Dict, Any
from tqdm import tqdm
import fire
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
    )
from accelerate import Accelerator

from handler import DataHandler    

def init_model_tokenizer(model_name: str): 
    if "llama" in model_name:
        # The LLaMA config on HF is not up to date with the library,
        # leading to errors when using AutoModelForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_name)
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return model, tokenizer

   
def main(
    model: str = "EleutherAI/gpt-neo-1.3B", #"decapoda-research/llama-7b-hf",
    val_set_size: Union[int, float] = 0.1,
    prompt_template: str = "prompts/medalpaca.json",
    model_max_length: int = 256, # should not exceed 2048, as LLaMA is trained with this
    train_on_inputs: bool = True, # if False, masks out inputs in loss
    data_path: str = "../medical_meadow_small.json",
    per_device_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    global_batch_size: int = 128,
    output_dir: str = "./output",
    save_total_limit: int = 3,
    eval_steps: int = 0.5,
    group_by_length: bool = False,
    wandb_run_name: str = "test",
    use_wandb: bool = False,
    wandb_project: str = "medalpaca",
    warmup_steps: int = 100,
    **kwargs
    ):
    """
    Args:
    model (str, optional):
        The model identifier on HuggingFace Model Hub.
    val_set_size (Union[int, float], optional):
        The proportion or number of samples to use for validation. Default is 0.1.
    prompt_template (str, optional):
        The path to the JSON file containing prompt templates. Default is "prompts/medalpaca.json".
    model_max_length (int, optional):
        The maximum length for model inputs. Default is 256.
    train_on_inputs (bool, optional):
        Whether to train on input tokens. Default is True.
    data_path (str, optional):
        The path to the dataset file. Default is "medical_meadow_small.json".
    per_device_batch_size (int, optional):
        The batch size per device. Default is 2.
    num_epochs (int, optional):
        The number of epochs for training. Default is 3.
    learning_rate (float, optional):
        The learning rate for the optimizer. Default is 2e-5.
    global_batch_size (int, optional):
        The number of samples the model needs to see until the weights get updated.
        Default is 128.
    output_dir (str, optional):
        The directory to save the model and outputs. Default is "./output".
    save_total_limit (int, optional):
        The maximum number of saved checkpoints. Default is 3.
    eval_steps (int, optional):
        The number of steps between evaluations. If a float number, it will be treated as the 
        percentage of an epoch. Default is 200.
    wandb_run_name (str, optional):
        The run name for Weights & Biases logging. Default is "test".
    wandb_project (str, optional):
        The Weights & Biases project name. Default is "medalpaca".

    warmup_steps (int, optional):
        The number of steps for warmup. Default is 200.
    """
    model_name = model    
    os.environ["WANDB_PROJECT"] = wandb_project
        
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = global_batch_size // per_device_batch_size // world_size
    
    accelerator = Accelerator(
        device_placement=True,
        log_with="wandb",
        mixed_precision="bf16", 
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    accelerator.init_trackers(
        project_name = wandb_project, 
        config = {
            "learning_rate": learning_rate,
            "batch_size": global_batch_size,
            "warmup_steps": warmup_steps,
            "num_epochs": num_epochs, 
            "model_max_length": model_max_length 
        }
    )    

    model, tokenizer = init_model_tokenizer(model_name)
      
    # prepare the data
    with accelerator.main_process_first():
        # load and tokenize data
        data_handler = DataHandler(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            model_max_length=model_max_length,
            train_on_inputs=train_on_inputs,
        )
        data = load_dataset("json", data_files=data_path)

        if val_set_size > 0:
            data = (
                data["train"]
                .train_test_split(test_size=val_set_size, shuffle=True, seed=42)
                .map(data_handler.generate_and_tokenize_prompt)
            )
        else:
            data = data.shuffle(seed=42).map(data_handler.generate_and_tokenize_prompt)
        data = data.map(remove_columns=["system", "user", "assistant"])

    collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest", pad_to_multiple_of=16)
    
    train_dataloader = DataLoader(
        data["train"], shuffle=True, collate_fn=collate_fn, batch_size=per_device_batch_size
    )
    eval_dataloader = DataLoader(
        data["test"], shuffle=False, collate_fn=collate_fn, batch_size=per_device_batch_size
    )
            
    # For FSDP feature, it is highly recommended and efficient to prepare the model before creating optimizer
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)   
    model = accelerator.prepare(model)
    
    # set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method. Model was already prepared, so we do not need to repeat this
    optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )


    if isinstance(eval_steps, float): 
        eval_steps = int(eval_steps * len(train_dataloader))
        
    # Now we train the model
    overall_step = 0    
    best_eval_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch", disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(epoch_iterator):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            overall_step += 1
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accelerator.print(lr_scheduler.get_lr())

            if overall_step % eval_steps == 0: 
                model.eval()
                eval_loss = 0
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)
                        eval_loss =+ outputs.loss.detach().float()
                mean_eval_loss = eval_loss.item() / len(eval_dataloader)

                # Use accelerator.print to print only on the main process.
                accelerator.print(f"epoch {epoch}:", loss)
                
                if mean_eval_loss < best_eval_loss:
                    accelerator.save_state(output_dir)
                    accelerator.print(f"Saved new best model at epoch {epoch} with loss", loss)

                accelerator.log(
                    {
                        "train_loss": total_loss.item() / step,
                        "eval_loss": mean_eval_loss,
                    },
                    step=epoch,
                )
            
    accelerator.end_training()

if __name__ == "__main__":
    fire.Fire(main)