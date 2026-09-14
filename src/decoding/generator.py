from .token_selector import append_token, append_best_token
from src.models import Prompt, FunctionDefinition
from src.llm_sdk import Small_LLM_Model
from torch import Tensor


class Generator:
    def __init__(
        self, llm: Small_LLM_Model,
        vocabs: dict[str, int], funcs_def: list[FunctionDefinition]
            ) -> str:
        self.llm = llm
        self.vocabs = vocabs
        self.funcs_def = funcs_def
    
    def generate(self, prompt: Prompt, tokens_tensor: Tensor) -> str:
        tokens = tokens_tensor.squeeze(0).tolist()
        result = self.generate_prompt(prompt, tokens)
        result, chosen_funct = self.generate_function_name(tokens)
        # result = self.generate_parameters(tokens)
        return result

    def generate_prompt(self, prompt: Prompt, tokens: list[int]) -> str:
        generation_order = ["{", '"prompt":']
        generation_order.extend(self.split_character(prompt.value))
        generation_order.append("\",")

        for character in generation_order:
            append_token(character, tokens, self.vocabs)
        return self.llm.decode(tokens)

    def generate_function_name(
        self, tokens: list[int]
            ) -> tuple[str, FunctionDefinition]:
        generation_order = [
            '"name":', "\"", [
                funct.name for funct in self.funcs_def
                ], "\"", ","
                ]
        for item in generation_order:
            if isinstance(item, str):
                append_token(item, tokens, self.vocabs)
            else:
                append_best_token(self.llm, tokens, item, self.vocabs)
        result = self.llm.decode(tokens)
        funct = next(
            funct for funct in self.funcs_def
            if result.endswith(funct.name + '",'))
        return result, funct
    
    def generate_parameters(
        self, tokens: list[int],
        selected_funct: FunctionDefinition
            ) -> str:
        generation_order = ['"parameters:"', "{"]

    def split_character(self, string: str) -> list[str]:
        words = string.split(" ")
        words[0] = "\"" + words[0]
        result = [words[0]] if words[0] else []
        result += ["Ġ" + w for w in words[1:]]
        return result
