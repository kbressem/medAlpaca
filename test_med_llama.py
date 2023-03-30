import json
import os
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import pandas as pd

import torch
from peft import PeftModel
import transformers

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

############################ load model ############################
def get_model(model_name, device_map, torch_dtype, perf_tuned=False, perf_tuned_path=None):
    tokenizer = LlamaTokenizer.from_pretrained(model_name, 
                                               device_map=device_map,
                                               # use_auth_token=True
                                               )
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True if perf_tuned else False,
        torch_dtype=torch_dtype,
        device_map=device_map,
        # use_auth_token=True,
    )
    if perf_tuned:
        model = PeftModel.from_pretrained(
            model, perf_tuned_path, 
            torch_dtype=torch_dtype, device_map=device_map
        )
    model.eval()
    return model, tokenizer

########################## generate prompt ###########################

def load_trivia_questions(file_path):
    with open(file_path, 'r') as file:
        trivia_data = json.load(file)
    return trivia_data

def generate_question_string(question_data):
    question = question_data['question']
    choices = [f"    {key}. {question_data['options'][key]}\n" if key != list(question_data['options'].keys())[-1] else f"    {key}. {question_data['options'][key]}" for key in question_data['options'].keys()]
    return f"{question}\n{''.join(choices)}"

def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. Only answer the question. Keep token limit low.
### Instruction:
{instruction}
### Response:
"""
def generate_cotprompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. Only answer the question. Keep token limit low.
### Instruction:
{instruction}
### Response:
let's think step by step
"""

############################ query model ############################
def query_model(
        prompt,
        model,
        tokenizer,
        device,
        max_new_tokens=50,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
    ):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            # top_p=top_p,
            # top_k=top_k,
            # num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        response = output.split("### Response:")[1].strip()
        return response.split("### Instruction:")[0].strip()


############################# main #################################

if __name__ == "__main__":

    cot = False
    model_name = "decapoda-research/llama-13b-hf"  # 'GerMedBERT/medalpaca-7b' 
    cuda_id = 2

    model, tokenizer = get_model(model_name, 
                                 device_map={'':cuda_id}, 
                                 torch_dtype=torch.float16, 
                                 perf_tuned=True if model_name.split('/')[0] == 'decapoda-research' else False, 
                                 perf_tuned_path="lora-alpaca-med-13b")
    
    for n in range(1, 4):
        json_file_path = 'data/test/step%d.json' % n
        questions = load_trivia_questions(json_file_path)
        answers = load_trivia_questions(os.path.join('data/test', os.path.basename(json_file_path).split('.')[0] + '_solutions.json'))

        keys = []
        instructs = []
        preditions = []
        groundtruths = []
        
        for i, question_data in enumerate(tqdm(questions)):
            question_string = generate_question_string(question_data)
            if cot:
                prompt = generate_cotprompt(question_string)
            else:
                prompt = generate_prompt(question_string)

            llm_answer = query_model(prompt, model, tokenizer, 
                                    max_new_tokens=300, # roughly 200 words
                                    device = torch.device("cuda:"+ str(cuda_id))
                                )
            keys.append(i+1)
            instructs.append(question_string)
            preditions.append(llm_answer)
            groundtruths.append(answers[str(i+1)])

        df = pd.DataFrame(data={'key': keys, 'instruction': instructs, 
                        'llama answer': preditions, 'ground truth': groundtruths})
        if cot:
            Path("results/chain_of_thought/").mkdir(parents=True, exist_ok=True)
            output_path = 'results/chain_of_thought/%s-cot.csv' % model_name.split('/')[1]
        else:
            dir = 'results/step%d' % n
            Path(dir).mkdir(parents=True, exist_ok=True)
            output_path = dir + '/%s.csv' % model_name.split('/')[1]
        
        df.to_csv(output_path, index=False)