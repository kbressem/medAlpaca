import sys
import json
import torch
from torch import nn
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

from .utils import load_json

assert torch.cuda.is_available(), "No cuda device detected"

class medAlapaca: 
    """
    A basic inference class for accessing medAlpaca models programmatically.

    This class provides methods for loading supported medAlpaca models, tokenizing inputs,
    and generating outputs based on the specified model and configurations.

    Attributes:
        available_models (dict): A dictionary containing the supported models and their configurations.

    Args:
        modelname (str): The name of the medAlpaca model to use for inference.
        prompt_template (str): The path to the JSON file containing the prompt template.

    Raises:
        ValueError: If the specified `modelname` is not in the list of available models.

    Example:

        medalpaca = medAlapaca("medalapca-7b", "prompts/alpaca.json")
        response = medalpaca(input="What is Amoxicillin?")
    """
        
    available_models = load_json("configs/supported_models.json")
    
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
        temperature: float = 0.1,
        top_p: float = 0.75,
        top_k: int = 40,
        num_beams: int = 4,
        max_new_tokens: int = 128,
        verbose: bool = False,
        **kwargs
    ) -> str: 
        """
        Generate a response from the medAlpaca model using the given input and instruction.

        Args:
            input (str): The input text to provide to the model.
            instruction (str, optional): An optional instruction to guide the model's response.
            temperature (float, optional): Sampling temperature. Higher values make the output more random.
            top_p (float, optional): Nucleus sampling probability threshold. Controls diversity of output.
            top_k (int, optional): Top-k sampling. Controls the number of candidates considered for sampling.
            num_beams (int, optional): Number of beams for beam search. 
                Controls the number of alternative sequences considered.
            max_new_tokens (int, optional): Maximum number of new tokens to generate in the response.
            verbose (bool, optional): If True, print the prompt before generating a response.
            **kwargs: Additional keyword arguments to pass to the `GenerationConfig`.

        Returns:
            str: The generated response from the medAlpaca model.
        """

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