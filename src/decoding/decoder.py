from src.models import Prompt, FunctionDefinition
from .generator import Generator
from src.llm_sdk import Small_LLM_Model
from typing import Any
import copy
import re


class Decoder:
    def __init__(
        self, llm: Small_LLM_Model, funcs_def_json: str, funcs_def: list[FunctionDefinition],
        prompts: list[Prompt], vocabs: dict[str, int],
            ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.funcs_def = funcs_def
        self.vocabs = vocabs
        self.tokens_sets: dict[str, Any] = self.build_type_token_sets(self.vocabs)
        self.funcs_def_json = funcs_def_json

    def generation_loop(self) -> None:
        generator = Generator(self.llm, self.vocabs, self.funcs_def, self.tokens_sets)
        available_functs_list = []
        for prompt in self.prompts:
            tokens_str = (
                "System: You are a function calling assistant. "
                "Given a user prompt and a list of available "
                "functions with their parameters and descriptions, "
                "choose the appropriate function and extract its parameters. "
                "For number parameters, output the value as a decimal number, "
                "for example 2.0, 16.0, or 144.0. "
                "Do not repeat or extend the number with extra digits. Make\n"
                "sure to properly close all the brackets. Write a valid, fully closed regular expression:"
                r"example: (\\d+)"
                "Functions:"
            )
            for funct in self.funcs_def:
                funct_str = f"\n- {funct.name}: {funct.description} Parameters: "
                for parameter in funct.parameters:
                    funct_str += f"{parameter} ({funct.parameters[parameter]["type"]})"
                tokens_str += funct_str
            tokens_str += f"\n{prompt.value}\nAssistant: "
            tokens = self.llm.encode(tokens_str)
            result = generator.generate(prompt, tokens, len(tokens_str))
    

    def build_type_token_sets(self, vocabs: dict[str, int]) -> dict:
        digit_tokens, dot_tokens, minus_tokens, quote_tokens, non_quote, closing_brackets = [], [], [], [], [], []
        for tok_str, tok_id in vocabs.items():
            bare = tok_str.replace("Ġ", "")

            if bare and re.fullmatch(r"\d+", bare):
                digit_tokens.append(tok_id)

            elif bare == ".":
                dot_tokens.append(tok_id)

            elif bare == "-":
                minus_tokens.append(tok_id)

            elif bare == '"':
                quote_tokens.append(tok_id)
            
            elif bare in ["}", "]", ")"]:
                closing_brackets.append(tok_id)

            elif '"' in bare:
                pass

            elif "," in bare:
                pass

            else:
                non_quote.append(tok_id)
        return {
            "digit": digit_tokens, "dot": dot_tokens,
            "minus": minus_tokens, "quote": quote_tokens, "non_quote":non_quote, "bracket":closing_brackets
        }
