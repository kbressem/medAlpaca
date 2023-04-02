import os
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from .utils import load_json, generate_prompt
from typing import Union, Dict, List

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

# configs -> move to main-function later
model: str = "decapoda-research/llama-7b-hf"
val_set_size: Union[int, float] = 0.1
prompt_template: Dict = load_json("prompts/medalpaca.json")
model_max_length: int = 128  # should not exceed 2048, as LLaMA is trained with this 
train_on_inputs: bool = True,  # if False, masks out inputs in loss
data_path: str = "../medical_meadow_small.json"
train_in_8bit: bool = True
use_lora: bool = True
lora_r: int = 8,
lora_alpha: int = 16,
lora_dropout: float = 0.05,
lora_target_modules: List[str] = [
    "q_proj",
    "v_proj",
],
per_device_batch_size: int = 2
num_epochs: int = 3
learning_rate: float = 2e-5
gradient_accumulation_steps: int = 128
output_dir: str = "./output"
save_total_limit: int =3,
eval_steps: int = 200
device_map: str = "auto"
group_by_length: bool = False
wandb_run_name: str
use_wandb: bool = True
wandb_project: str = "medalpaca"

# adapt arguments
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // world_size

if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

# create the tokenizer and tokenize function
tokenizer = LlamaTokenizer.from_pretrained(
    model, 
    pad_token_id = 0, 
    padding_side = "left"
    )


def tokenize(prompt: str, add_eos_token: bool=True):
    """
    Tokenize the given prompt and optionally add an end-of-sequence (EOS) token.

    This function tokenizes the input prompt without adding special tokens by default.
    If the `add_eos_token` parameter is True and the tokenized sequence doesn't already
    end with an EOS token, an EOS token will be added to the end of the sequence.

    Args:
        prompt (str): The text to be tokenized.
        add_eos_token (bool, optional): Whether to add an EOS token at the end of
            the tokenized sequence. Defaults to True.

    Returns:
        Dict: A dictionary containing the tokenized data:
            - input_ids: The tokenized input IDs of the prompt.
            - attention_mask: The attention mask for the tokenized input IDs.
            - labels: The labels for the tokenized input IDs (identical to input_ids).
    """
    # Tokenize the prompt without adding special tokens
    tokens = tokenizer.encode_plus(
        prompt,
        truncation=True,
        max_length=model_max_length,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )

    input_ids = tokens["input_ids"]

    # Check if an EOS token should be added
    if (
        input_ids[-1] != tokenizer.eos_token_id
        and len(input_ids) < model_max_length
        and add_eos_token
    ):
        input_ids.append(tokenizer.eos_token_id)

    attention_mask = [1] * len(input_ids)

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.copy(),
    }

    return result


def generate_and_tokenize_prompt(data_point: Dict):
    """
    Generate a prompt based on the given data point and tokenize it.

    This function creates a prompt using the given data point, which consists
    of an instruction, input, and output. It then tokenizes the generated prompt
    and returns the tokenized representation. If the `train_on_inputs` global variable
    is False, the function will create a user prompt without the expected output and
    only tokenize that part, masking the output part in the "labels" field with -100.

    Args:
        data_point (Dict): A dictionary containing the following keys:
            - instruction: The instruction text for the prompt.
            - input: The input text for the prompt.
            - output: The output text for the prompt.

    Returns:
        Dict: A dictionary containing the tokenized prompt and associated data:
            - input_ids: The tokenized input IDs of the generated prompt.
            - attention_mask: The attention mask for the tokenized input IDs.
            - labels: The labels to be used during model training, with the output
              part unmasked and the rest masked with -100 if `train_on_inputs` is False.
    """
    prompt = generate_prompt(
        prompt_template,
        instruction=data_point["instruction"],
        input=data_point["input"],
        output=data_point["output"],
    )
    tokenized_prompt = tokenize(prompt)
    if not train_on_inputs:
        user_prompt = generate_prompt(
            prompt_template,
            instruction=data_point["instruction"], 
            input=data_point["input"]
        )
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        # mask out the inputs 
        tokenized_prompt["labels"] = [-100 if i < user_prompt_len else label for i, label in enumerate(tokenized_prompt["labels"])]
    return tokenized_prompt


# load data and train/val split
data = load_dataset("json", data_files=data_path)
data = data.map(generate_and_tokenize_prompt)

if val_set_size > 0:
    data = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )

    train_data = data["train"].shuffle(seed=123)
    val_data = data["test"]
else: 
    train_data = data.shuffle(seed=123)
    val_data = None

if train_in_8bit:
    model = prepare_model_for_int8_training(model)

if use_lora: 
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  



training_args = TrainingArguments(
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=100,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps" if val_set_size > 0 else "no",
    save_strategy="steps",
    eval_steps=eval_steps if val_set_size > 0 else None,
    save_steps=eval_steps,
    output_dir=output_dir,
    save_total_limit=save_total_limit,
    load_best_model_at_end=True if val_set_size > 0 else False,
    ddp_find_unused_parameters=False if ddp else None,
    group_by_length=group_by_length,
    report_to="wandb" if use_wandb else None,
    run_name=wandb_run_name if use_wandb else None,
    ),


trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train()

model.save_pretrained(output_dir)


# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./output")
tokenizer.save_pretrained("./output")