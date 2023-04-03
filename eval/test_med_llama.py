import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch
import transformers
from peft import PeftModel
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, set_seed


############################ load model ############################
def get_model(model_name, device_map, torch_dtype, perf_tuned=False, perf_tuned_path=None):

    device_map = device_map if perf_tuned else 'auto'
    tokenizer = LlamaTokenizer.from_pretrained(model_name, 
                                               device_map=device_map,
                                               use_auth_token=True
                                               )
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True if perf_tuned else False,
        torch_dtype=torch_dtype,
        device_map=device_map,
        use_auth_token=True,
    )
    if perf_tuned:
        model = PeftModel.from_pretrained(
            model, perf_tuned_path, 
            torch_dtype=torch_dtype, 
            device_map=device_map,
            use_auth_token=True,
        )
    model = model.eval()
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
    return f"""The following are multiple choice questions about professional medicine. Write a response that appropriately follows the instruction.
### Instruction:
{instruction} 
### Response:
"""

# few shot cot examples are here :https://github.com/vlievin/medical-reasoning/blob/master/prompts/usmle-medqa-5shot-prompt.txt
def generate_1shotprompt(instruction):
    return f"""The following are multiple choice questions about professional medicine. Write a response that appropriately follows the instruction.
### Instruction:
A 53-year-old man comes to the physician because of a 3-month history of a nonpruritic rash. He has been feeling more tired than usual and occasionally experiences pain in his wrists and ankles. He does not smoke or drink alcohol. His temperature is 37.6°C (99.7°F), pulse is 98/min, respirations are 18/min, and blood pressure is 130/75 mm Hg. Physical examination shows multiple, erythematous, purpuric papules on his trunk and extremities that do not blanch when pressed. The remainder of the examination shows no abnormalities. The patient's hemoglobin is 14 g/dL, leukocyte count is 9,500/mm3, and platelet count is 228,000/mm3. Urinalysis and liver function tests are within normal limits. The test for rheumatoid factor is positive. Serum ANA is negative. Serum complement levels are decreased. Serum protein electrophoresis and immunofixation shows increased gammaglobulins with pronounced polyclonal IgM and IgG bands. Testing for cryoglobulins shows no precipitate after 24 hours. Chest x-ray and ECG show no abnormalities. Which of the following is the most appropriate next step in management?
(A) Rapid plasma reagin test (B) Hepatitis C serology (C) pANCA assay (D) Bence Jones protein test
### Response:
A: Let's think step by step. We refer to Wikipedia articles on medicine for help. The patient has a rash, fatigue, and pain in his wrists and ankles. He has normal hemoglobin (normal range 13.8-17.2 g/dL), normal leukocyte count (normal range 4500-11000 wbc/microliter), and normal platelet count (normal range 150-450 platelets/microliter). His serum complement levels are decreased. His serum protein electrophoresis and immunofixation show increased gammaglobulins with pronounced polyclonal IgM and IgG bands. The cryoglobulin precipitation test is normal, ruling out cryoglobulinemia. Serum ANA is negative, so Lupus is less likely. Physical examination and symptoms are consistent with possible non-cryoglobulinemic vasculitis. The most likely associated diagnosis is Hepatitis C. The answer is (B). 
=== 
### Instruction:
{instruction}
### Response:
"""

############################ query model ############################
def query_model(
        prompt,
        model,
        tokenizer,
        device,
        max_new_tokens=50,
        temperature=1.0,
        do_sample=True,
        top_p=1.0,
        top_k=50,
        num_return_sequences=1,
        **kwargs,
    ):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
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
        output = tokenizer.decode(s, skip_special_tokens=True)
        response = output.split("### Response:")[1].strip()
        return response.split("### Instruction:")[0].strip()


############################# main #################################

if __name__ == "__main__":

    model_name = 'decapoda-research/llama-13b-hf' # 'chavinlo/alpaca-native' # 'GerMedBERT/medalpaca-7b'    
    cuda_id = 1
    set_seed(42)

    model, tokenizer = get_model(model_name, 
                                 device_map={'':cuda_id}, 
                                 torch_dtype=torch.float16, 
                                 perf_tuned=True,
                                 perf_tuned_path='lora-alpaca-13b',
                                 )
    
    for n in range(1, 4):
        json_file_path = 'data/test/step%d.json' % n
        questions = load_trivia_questions(json_file_path)
        answers = load_trivia_questions(os.path.join('data/test', os.path.basename(json_file_path).split('.')[0] + '_solutions.json'))

        keys = []
        instructs = []
        preditions_greedy = []
        predictions_1shot = []
        predictins_sample = defaultdict(list)
        groundtruths = []
        
        for i, question_data in enumerate(tqdm(questions)):
            question_string = generate_question_string(question_data)
            prompt = generate_prompt(question_string)
            prompt_1shot = generate_1shotprompt(question_string)
            
            # get top1 prediction from greedy search
            llm_answer_greedy = query_model(prompt, model, tokenizer, device='cuda',
                                    max_new_tokens=50,
                                    do_sample=False,
                                    temperature=0.1,
                                    )
            preditions_greedy.append(llm_answer_greedy)
            # get 1-shot prediction
            llm_answer_1shot = query_model(prompt_1shot, model, tokenizer, device='cuda',
                                    max_new_tokens=50,
                                    do_sample=False,
                                    temperature=0.1,
                                    )
            predictions_1shot.append(llm_answer_1shot)
            
            # get top5 predictions from sampling
            llamas = []
            for idx in range(5):
                llm_answer = query_model(prompt, model, tokenizer, device='cuda',
                                        max_new_tokens=50, 
                                        do_sample=True,
                                        temperature=0.1*(idx+1),
                                        top_p=0.75, top_k=100,
                                        )
            
                predictins_sample['sample %d' %idx].append(llm_answer)
                llamas.append(llm_answer)

            keys.append('No.%d' %(i+1))
            instructs.append(question_string)
            groundtruths.append(answers[str(i+1)])

        df = pd.DataFrame(data={'key': keys, 
                                'instruction': instructs, 
                                'llama top1': preditions_greedy,
                                'llama top1 1-shot': predictions_1shot,
                                'llama top5 sample 0': predictins_sample['sample 0'],
                                'llama top5 sample 1': predictins_sample['sample 1'],
                                'llama top5 sample 2': predictins_sample['sample 2'],
                                'llama top5 sample 3': predictins_sample['sample 3'],
                                'llama top5 sample 4': predictins_sample['sample 4'], 
                                'ground truth': groundtruths}
                                )

        dir = 'results_updated/step%d' % n
        Path(dir).mkdir(parents=True, exist_ok=True)
        output_path = dir + '/%s.json' % 'llama-lora-13b-naive-ty'
        df = df.set_index('key')
        df.to_json(output_path, orient="index", indent=4)