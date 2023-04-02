import os
os.environ['HF_HOME'] = "/sc-projects/sc-proj-cc06-medbert/hfcache"
import sys
import json
import torch
from torch import nn
from peft import PeftModel
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, set_seed
assert torch.cuda.is_available(), "No cuda device detected"


class medAlapaca: 
    "Basic inference method to access medAlpaca models programmatically"
        
    available_models = {
        "llama-7b-hf": {"peft": False, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-7b-hf", "load_in_8bit": False},
        "alpaca-lora": {"peft": True, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-7b-hf", "load_in_8bit": True, "lora_model_id":"tloen/alpaca-lora-7b"},
        "medalapca-7b": {"peft": False, "torch_dtype": torch.float16, "base_model": "GerMedBERT/medalpaca-7b", "load_in_8bit": False},
        "medalapca-13b": {"peft": False, "torch_dtype": torch.float16, "base_model": "GerMedBERT/medalpaca-13b", "load_in_8bit": False},
        "medalapca-lora-7b-8bit": {"peft": True, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-7b-hf", "load_in_8bit": True, "lora_model_id": "GerMedBERT/medalpaca-lora-7b-8bit"},
        "medalapca-lora-13b-8bit": {"peft": True, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-13b-hf", "load_in_8bit": True, "lora_model_id": "GerMedBERT/medalpaca-lora-13b-8bit" },
        "medalapca-lora-30b-8bit": {"peft": True, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-30b-hf", "load_in_8bit": True, "lora_model_id": "GerMedBERT/medalpaca-lora-30b-8bit" },
        "medalapca-lora-65b-8bit": {"peft": True, "torch_dtype": torch.float16, "base_model": "decapoda-research/llama-65b-hf", "load_in_8bit": True, "lora_model_id": "GerMedBERT/medalpaca-lora-65b-8bit" },
    }
    
    def __init__(self, modelname: str, prompt_template: str) -> None:
        if modelname not in self.available_models.keys(): 
            raise ValueError(f"`modelname` should be in {list(self.available_models.keys())}")
        self.model = self._load_model(modelname)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.available_models[modelname]["base_model"])
        with open(prompt_template) as fp: 
            self.prompt_template = json.load(fp)
        
    def _load_model(self, modelname: str) -> nn.Module: 
        load_args = self.available_models[modelname]
        
        model = LlamaForCausalLM.from_pretrained(
            load_args["base_model"], 
            load_in_8bit=load_args["load_in_8bit"],
            torch_dtype=load_args["torch_dtype"], 
            device_map={'':0}
        )
        
        if not load_args["load_in_8bit"]:
            model.half()
        
        if load_args["peft"]: 
            model = PeftModel.from_pretrained(
                model,
                model_id=load_args["lora_model_id"], 
                torch_dtype=load_args["torch_dtype"],
                device_map={'':0}
            )
            
        model.eval()
        
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
            
        return model

    
    def __call__(
        self, 
        input: str, 
        instruction: str = None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        verbose: bool = False,
        **kwargs
    ) -> str: 
        
        prompt = self.prompt_template["prompt_input"].format(
                instruction=instruction, input=input
            )
        if verbose: 
            print(prompt)
        input_tokens = self.tokenizer(prompt, return_tensors="pt")
        input_token_ids = input_tokens["input_ids"].to("cuda")
        
        generation_config = GenerationConfig(
            temperature=temperature or self.temperature,
            top_p=top_p or self.top_p,
            top_k=top_k or self.top_k,
            num_beams=num_beams or self.num_beams,
            **kwargs,
        )
        
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_token_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
            )
        output = self.tokenizer.decode(generation_output.sequences[0])
        response = output.split(self.prompt_template["response_split"])[1].strip()
        return response.split("### Instruction:")[0].strip()