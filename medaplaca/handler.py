import json
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def load_json(fn: str):
    with open(fn, "r") as fp:
        d = json.load(fp)
    return d


class DataHandler:
    """Helper class to handle prompt generation and data tokenization.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        prompt_template (str, optional):
            The path to the JSON file containing the prompt template.
            Defaults to "prompts/medalpaca.json".
        model_max_length (int, optional):
            The maximum length of the tokenized sequence.
            Should not exceed 2048, as LLaMA is trained with this. Defaults to 256.
        train_on_inputs (bool, optional):
            If False, masks out users in loss. Defaults to True.

    Methods:
        tokenize(prompt: str, add_eos_token: bool = True) -> Dict:
            Tokenizes the given prompt and optionally adds an end-of-sequence (EOS) token.

        generate_and_tokenize_prompt(data_point: Dict) -> Dict:
            Generates a prompt based on the given data point and tokenizes it.

    """

    def __init__(
        self,
        tokenizer,
        prompt_template: str = "prompts/medalpaca.json",
        model_max_length: int = 256,
        train_on_inputs: bool = True,
    ) -> None:
        if model_max_length > 2048:
            logger.warn(f"{model_max_length} exceeds the max token length LLaMA was trained with.")
        self.prompt_template = load_json(prompt_template)
        self.model_max_length = model_max_length
        self.train_on_inputs = train_on_inputs
        self.tokenizer = tokenizer

    def tokenize(self, prompt: str, add_eos_token: bool = True, return_tensors: str = None, truncation: bool = True) -> Dict[str, list]:
        """
        Tokenize the given prompt and optionally add an end-of-sequence (EOS) token.

        This function tokenizes the user prompt without adding special tokens by default.
        If the `add_eos_token` parameter is True and the tokenized sequence doesn't already
        end with an EOS token, an EOS token will be added to the end of the sequence.

        Args:
            prompt (str): The text to be tokenized.
            add_eos_token (bool, optional): Whether to add an EOS token at the end of
                the tokenized sequence. Defaults to True.
            return_tensors (str, optional): If tensors should be returned (and what type).
            trunctaion (bool, optional); Whether to truncate the user to max_model_length
            

        Returns:
            Dict: A dictionary containing the tokenized data:
                - input_ids: The tokenized user IDs of the prompt.
                - attention_mask: The attention mask for the tokenized user IDs.
                - labels: The labels for the tokenized user IDs (identical to input_ids).
        """
        result: Dict = self.tokenizer(
            prompt,
            truncation=truncation,
            max_length=self.model_max_length,
            padding=False,
            return_tensors=return_tensors,
            add_special_tokens=False,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point: Dict):
        """
        Generate a prompt based on the given data point and tokenize it.

        This function creates a prompt using the given data point, which consists
        of a `system`- prompt, `user`- prompt, and an assistant anser. 
        It then tokenizes the generated prompt and returns the tokenized representation. 
        If the `train_on_inputs` global variable is False, the function will create a user prompt without the
        expected assistant and only tokenize that part, masking the assistant part in the
        "labels" field with -100.

        Args:
            data_point (Dict): A dictionary containing the following keys:
                - system: The system text for the prompt. Used to set up the assistant
                - user: The user text for the prompt.
                - assistant: The assistant text for the prompt.

        Returns:
            Dict: A dictionary containing the tokenized prompt and associated data:
                - input_ids: The tokenized user IDs of the generated prompt.
                - attention_mask: The attention mask for the tokenized user IDs.
                - labels: The labels to be used during model training, with the assistant
                part unmasked and the rest masked with -100 if `train_on_inputs` is False.
        """
        prompt: str = self.generate_prompt(
            system=data_point.get("system", ""),
            user=data_point.get("user", ""),
            assistant=data_point.get("assistant", ""),
        )
        tokenized_prompt: Dict = self.tokenize(prompt)
        if not self.train_on_inputs:
            user_prompt: str = self.generate_prompt(
                system=data_point.get("system", ""), user=data_point.get("user", "")
            )
            tokenized_user_prompt: Dict = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            # mask out the users
            tokenized_prompt["labels"] = [
                -100 if i < user_prompt_len else label
                for i, label in enumerate(tokenized_prompt["labels"])
            ]
        return tokenized_prompt

    def generate_prompt(
        self,
        system: Optional[str] = None,
        user: Optional[str] = None,
        assistant: Optional[str] = None,
    ) -> str:
        """
        Generates a prompt for the given system promot, user prompt and assistant answer using the specified prompt
        template.

        Args:
            system (Optional[str]):
                An optional string representing the system message to be included in the prompt.
            user (Optional[str]):
                An optional string representing the user message to be included in the prompt.
            assistant (Optional[str]):
                An optional string representing the assistant answer to be included in the prompt.

        Returns:
            str: The prompt string created using the specified prompt template.

        Raises:
            ValueError: If none of `system`, `user`, and `assistant` is defined.

        ## Example

        data_handler = DataHandler(tokenizer, "prompt_templates/medalpaca.json")
        prompt = data_hanlder.generate_prompt(
            system = "You are a helpful medical chatbot.",
            user = "What to expect if I have Aortic coarctation  (Outlook/Prognosis)?",
            assistant = (
                "The prognosis of aortic coarctation depends on whether balloon "
                "angioplasty and stenting or the surgery has been done or not."
            )
        )
        print(prompt)
        >>> Below is an system that describes a task, paired with an user that provides
            further context. Write a response that appropriately completes the request.

            ### System:
            You are a helpful medical assistant.

            ### User:
            What to expect if I have Aortic coarctation  (Outlook/Prognosis)?

            ### Assistant:
            The prognosis of aortic coarctation depends on whether balloon angioplasty and
            stenting or the surgery has been done or not.
        """

        if not any([system, user, assistant]):
            raise ValueError("At least one of `system`, `user`, `assistant` should be defined")

        prompt = (
            f'{self.prompt_template["system"]}{system or ""}'
            f'{self.prompt_template["user"]}{user or ""}'
            f'{self.prompt_template["assistant"]}{assistant or ""}'
        )

        return prompt

    def resolve_assistant(self, assistant: str): 
        pass
        