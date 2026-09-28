from .token_selector import append_token, append_best_token
from src.models import Prompt, FunctionDefinition
from src.llm_sdk import Small_LLM_Model
import src.decoding.parameter_generator as param_gen
from torch import Tensor
from typing import Any
import re


class Generator:
    def __init__(
        self, llm: Small_LLM_Model,
        vocabs: dict[str, int], funcs_def: list[FunctionDefinition],
        tokens_sets: dict[str, Any]
            ) -> str:
        self.llm = llm
        self.vocabs = vocabs
        self.funcs_def = funcs_def
        self.tokens_sets = tokens_sets
    
    def generate(
        self, prompt: Prompt, tokens_tensor: Tensor, len_srt: int
            ) -> str:
        tokens = tokens_tensor.squeeze(0).tolist()
        result = self.generate_prompt(prompt, tokens)
        result, selected_funct = self.generate_function_name(tokens)
        result = self.generate_parameters(tokens, prompt, selected_funct, self.tokens_sets)
        print()
        return result

    def generate_prompt(self, prompt: Prompt, tokens: list[int]) -> str:
        generation_order = ["{", '"prompt":']
        generation_order.extend(self.split_character(prompt.value))
        generation_order.append("\",")

        for character in generation_order:
            append_token(self.llm, character, tokens, self.vocabs)
        return self.llm.decode(tokens)

    def generate_function_name(
        self, tokens: list[int],
            ) -> tuple[str, FunctionDefinition]:
        possible_names = [funct.name for funct in self.funcs_def]
        generation_order = [
            '"name":', "\"", possible_names, "\"", ","
                ]
        selected_name = ""
        for item in generation_order:
            if isinstance(item, str):
                append_token(self.llm, item, tokens, self.vocabs)
            else:
                selected_tokens = append_best_token(
                    self.llm, tokens, item, self.vocabs
                )
                selected_name = self.llm.decode(selected_tokens)
        result = self.llm.decode(tokens)
        funct = next(
            funct for funct in self.funcs_def
            if funct.name == selected_name)
        return result, funct
    
    def generate_parameters(
        self, tokens: list[int], prompt: Prompt,
        selected_funct: FunctionDefinition,
        tokens_sets: dict[str, Any]
            ) -> str:
        generation_order = ['"parameters":', "{"]
        i = 0
        while i < len(selected_funct.parameters):
            generation_order.extend(["\"", True, "\"", ":", "\"", False, "\""])
            if i < len(selected_funct.parameters) - 1:
                generation_order.append(",")
            i += 1
        # generation_order.extend(["}", "}"])

        param_names = list(selected_funct.parameters.keys())
        param_index = 0

        for item in generation_order:
            if isinstance(item, str):
                append_token(self.llm, item, tokens, self.vocabs)
            elif item is True:
                append_best_token(self.llm, tokens, param_names, self.vocabs)
            else:  # item is False -> generate the value
                current_parameter = param_names[param_index]
                generation_type = selected_funct.parameters[current_parameter]["type"]
                parameters = [param for param in selected_funct.parameters]
                if selected_funct.parameters[parameters[param_index]]["type"] == "string":
                    param_gen.generate_string(self.llm, tokens, self.vocabs)
                else:
                    param_gen.generate_number(self.llm, tokens, self.vocabs, tokens_sets)
                param_index += 1

        result = self.llm.decode(tokens)
        return result

    def split_character(self, string: str) -> list[str]:
        words = string.split(" ")
        words[0] = "\"" + words[0]
        result = [words[0]] if words[0] else []
        result += ["Ġ" + w for w in words[1:]]
        return result
