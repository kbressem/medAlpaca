import json
import torch
import transformers
from peft import PeftModel
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


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
