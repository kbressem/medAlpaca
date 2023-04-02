import json
import fire
from tqdm import tqdm
from inferer import medAlapaca



available_models = [
    "llama-7b-hf",
    "alpaca-lora",
    "medalapca-7b",
    "medalapca-lora-7b-8bit",
    "medalapca-lora-13b-8bit",
    "medalapca-lora-30b-8bit",
]


def eval_step(model, step_no: int, n_samples: int, model_name: str, cot: bool): 
    with open(f"/sc-projects/sc-proj-cc06-medbert/medAlpaca/data/step{step_no}.json") as fp: 
        step = json.load(fp)

    for question in tqdm(step): 
        for i in tqdm(range(n_samples), leave=False):
            question[f"answer{i}"] = model(
                question["question"] + "\n" + "\n".join([f"{k}: {v}" for k,v in question["options"].items()]), 
                instruction= "Answer the  multiple choice question correctly and truthfully." , 
                max_new_tokens=128, 
                verbose=False
            )
        with open(f"{model_name}-step{step_no}{'-cot' if cot else ''}.json", "w+") as fp: 
            json.dump(step, fp)
            
def main(
    model_name: str, 
    prompt_template: str,
    n_samples: int = 1
): 
    assert model_name in available_models, model_name
    
    model = medAlapaca(model_name, prompt_template)
    for step_no in [1, 2, 3]: 
        eval_step(
            model, step_no=step_no, n_samples=n_samples, model_name=model_name, cot="cot" in prompt_template
        )
    
if __name__ == "__main__": 
    fire.Fire(main)