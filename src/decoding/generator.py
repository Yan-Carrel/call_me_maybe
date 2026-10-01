from .token_selector import append_token, append_best_token
from src.models import Prompt, FunctionDefinition
from src.llm_sdk import Small_LLM_Model
import src.decoding.parameter_generator as param_gen
from torch import Tensor
from typing import Any
import re
import json


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
        generation_order = [
            "{",
            '"prompt":',
            json.dumps(prompt.value, ensure_ascii=False),
            ",",
        ]

        for text in generation_order:
            append_token(self.llm, text, tokens, self.vocabs)

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
    self,
    tokens: list[int],
    prompt: Prompt,
    selected_funct: FunctionDefinition,
    tokens_sets: dict[str, Any]
) -> str:

        tokens_str = self.llm.decode(tokens)
        starting_idx = len(tokens_str)
        append_token(self.llm, '"parameters":', tokens, self.vocabs)
        append_token(self.llm, '{', tokens, self.vocabs)

        param_names = list(selected_funct.parameters.keys())

        for i, param_name in enumerate(param_names):

            if i > 0 and self.llm.decode(tokens)[-1] != ',':
                append_token(self.llm, ',', tokens, self.vocabs)

            append_token(self.llm, '"', tokens, self.vocabs)
            append_token(self.llm, param_name, tokens, self.vocabs)
            append_token(self.llm, '":', tokens, self.vocabs)

            param_type = selected_funct.parameters[param_name]["type"]

            if param_type == "string" and param_name == "regex":
                tokens_str = self.llm.decode(tokens)
                tokens_str[tokens_str.rindex("{\"prompt\":"):]
                append_token(self.llm, '"', tokens, self.vocabs)
                param_gen.generate_regex(
                    self.llm,
                    tokens,
                    self.vocabs,
                    tokens_sets
                )

            elif param_type == "string":
                append_token(self.llm, '"', tokens, self.vocabs)

                param_gen.generate_string(
                    self.llm,
                    tokens,
                    tokens_sets
                )

            elif param_type == "number":
                param_gen.generate_number(
                    self.llm,
                    tokens,
                    self.vocabs,
                    tokens_sets
                )

        append_token(self.llm, '}}', tokens, self.vocabs)

        return self.llm.decode(tokens)

    def split_character(self, string: str) -> list[str]:
        words = string.split(" ")
        words[0] = "\"" + words[0]
        result = [words[0]] if words[0] else []
        result += ["Ġ" + w for w in words[1:]]
        return result
