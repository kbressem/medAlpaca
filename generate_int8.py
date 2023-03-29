import json
from collections import defaultdict
import tqdm
import pandas as pd

import torch
from peft import PeftModel
import transformers

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-13b-hf", device_map={'':2})

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-13b-hf",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map={'':2},
)
model = PeftModel.from_pretrained(
    model, "lora-alpaca", torch_dtype=torch.float16, device_map={'':2},
)


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


model.eval()


def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=2048,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


if __name__ == "__main__":
    json_file_path = 'anki_questions_subset.json'
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())

    d = defaultdict(str)
    for key in contents['INSTRUCTION'].keys():
        value = contents['INSTRUCTION'][key]
        if value[-1] == '?':
            d[key] = value
    answers = []
    keys = []
    instructs = []
    groundtruths = []
    
    counter = 0
    for key in tqdm.tqdm(list(d.keys())):
        if counter == 326:
            continue
        instruct = d[key]
        if '_' not in instruct:
            answer = evaluate(instruct)
            
            keys.append(key)
            instructs.append(instruct)
            answers.append(answer)
            groundtruths.append(contents["RESPONSE"][key])
        counter += 1
    
    df = pd.DataFrame(data={'key': keys, 'instruction': instructs, 
                            'llama answer': answers, 'ground truth': groundtruths})
    df.to_csv('llama.csv', index=False)
