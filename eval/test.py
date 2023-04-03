import os
import fire
import torch
from pathlib import Path
import pandas as pd 
from tqdm import tqdm
from collections import defaultdict

os.environ['HF_HOME'] = "/sc-projects/sc-proj-cc06-medbert/hfcache"
assert torch.cuda.is_available(), "No cuda device detected"

from eval.test_utils import get_model, load_trivia_questions, generate_question_string, generate_prompt, generate_1shotprompt, query_model


class medalapaca: 
    "Basic inference method to access medAlpaca models programmatically"
        
    available_models = {
        "llama-7b-hf": {"peft": False, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-7b-hf", "load_in_8bit": False, "lora_model_id": None},
        "alpaca-lora": {"peft": True, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-7b-hf", "load_in_8bit": True, "lora_model_id":"tloen/alpaca-lora-7b"},
        "medalapca-7b": {"peft": False, "torch_dtype": torch.float16, "base_model": "GerMedBERT/medalpaca-7b", "load_in_8bit": False, "lora_model_id": None},
        "medalapca-13b": {"peft": False, "torch_dtype": torch.float16, "base_model": "GerMedBERT/medalpaca-13b", "load_in_8bit": False, "lora_model_id": None},
        "medalapca-lora-7b-8bit": {"peft": True, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-7b-hf", "load_in_8bit": True, "lora_model_id": "GerMedBERT/medalpaca-lora-7b-8bit"},
        "medalapca-lora-13b-8bit": {"peft": True, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-13b-hf", "load_in_8bit": True, "lora_model_id": "GerMedBERT/medalpaca-lora-13b-8bit" },
        "medalapca-lora-30b-8bit": {"peft": True, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-30b-hf", "load_in_8bit": True, "lora_model_id": "GerMedBERT/medalpaca-lora-30b-8bit" },
        "medalapca-lora-65b-8bit": {"peft": True, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-65b-hf", "load_in_8bit": True, "lora_model_id": "GerMedBERT/medalpaca-lora-65b-8bit" },
    }
    
    def __init__(self, modelname: str, cuda_id=None) -> None:
        if modelname not in self.available_models.keys(): 
            raise ValueError(f"`modelname` should be in {list(self.available_models.keys())}")
        self.model_name = modelname
        self.model_args = self.available_models[modelname]
        if cuda_id is not None: 
            self.device_map = {'': cuda_id}
        else:
            self.device_map = 'auto'
        self.model, self.tokenizer = get_model(self.model_args['base_model'], 
                                               self.device_map,
                                               self.model_args['torch_dtype'],
                                               self.model_args['peft'],
                                               self.model_args['lora_model_id']
                                               )

    def __call__(
        self, 
        json_file_path: str,
        answer_root: str,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        max_new_tokens=128,
        n_samples=5,
        save_json: bool = False,
        save_dir: str = None,
    ) -> str: 
        questions = load_trivia_questions(json_file_path)
        answers = load_trivia_questions(os.path.join(answer_root, 
                                                     os.path.basename(json_file_path).split('.')[0] + '_solutions.json'))
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
            llm_answer_greedy = query_model(prompt, 
                                            self.model, 
                                            self.tokenizer, 
                                            device='cuda',
                                            max_new_tokens=max_new_tokens,
                                            do_sample=False,
                                            temperature=temperature,
                                            )
            preditions_greedy.append(llm_answer_greedy)
            # get 1-shot prediction
            llm_answer_1shot = query_model(prompt_1shot, 
                                           self.model, 
                                           self.tokenizer, 
                                           device='cuda', 
                                           max_new_tokens=max_new_tokens,
                                           do_sample=False,
                                           temperature=temperature,
                                        )
            predictions_1shot.append(llm_answer_1shot)
            
            # get top5 predictions from sampling
            llamas = []
            for idx in range(n_samples):
                llm_answer = query_model(prompt, 
                                         self.model, 
                                         self.tokenizer, 
                                         device='cuda',
                                         max_new_tokens=max_new_tokens, 
                                         do_sample=True,
                                         temperature=0.1*(idx+1),
                                         top_p=top_p, 
                                         top_k=top_k,
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
        if not save_json:
            return df
        else:
            dir = os.path.join(save_dir, os.path.basename(json_file_path).split('.')[0])
            Path(dir).mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(dir, '%s.json' % self.model_name)
            df = df.set_index('key')
            df.to_json(output_path, orient="index", indent=4)


def main(
    model_name: str, 
    cuda_id: int = 0,
    json_file_path: list = ['data/test/step1.json', 'data/test/step2.json', 'data/test/step3.json'],
    anser_root: str = 'data/test',
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    max_new_tokens=128,
    n_samples=5,
    save_json: bool = True,
    save_dir: str = './results',
): 
    
    tester = medalapaca(model_name, cuda_id)
    for file in json_file_path:
        _ = tester(file, anser_root, temperature, top_p, 
                    top_k, max_new_tokens, n_samples,
                    save_json, save_dir)
    
if __name__ == "__main__": 
    fire.Fire(main)