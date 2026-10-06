from src.models import Prompt, FunctionDefinition
from .generator import Generator
from .llm import Llm
from src.llm_sdk import Small_LLM_Model
from typing import Any
import copy
import re
import json


class Decoder:
    def __init__(
        self, llm_sdk: Small_LLM_Model, funcs_def_json: str,
        funcs_def: list[FunctionDefinition],
        prompts: list[Prompt], vocabs: dict[str, int],
            ) -> None:
        list_vocabs = list(vocabs.keys())
        self.llm = Llm(llm=llm_sdk, vocabs=vocabs, vocabs_str=list_vocabs)
        self.prompts = prompts
        self.funcs_def = funcs_def
        self.vocabs = vocabs
        self.tokens_sets: dict[str, Any] = self.build_type_token_sets(self.vocabs)
        self.funcs_def_json = funcs_def_json

    def _build_prompt(self, funcs_def: list[FunctionDefinition], user_prompt: str) -> str:
        system_content = (
            "System: You are a function calling assistant.\n"

            "Given a user prompt and a list of functions, select the correct function and output its parameters.\n"
            "Rules:\n"
            "- Copy text parameters exactly as they appear in the user prompt.\n"
            "- Do not modify, add, remove, or repeat characters.\n"
            "- Output decimal numbers exactly as written by the user.\n"
            "- Do not add or repeat digits.\n"
            "- Integers must be written as decimals, e.g. 16 -> 16.0.\n"
            "- Return valid JSON with all quotes and brackets correctly closed.\n"

            "Functions:"
        )

        for funct in funcs_def:
            system_content += (
                f"\n- {funct.name}: {funct.description} Parameters: "
            )
            for parameter, definition in funct.parameters.items():
                system_content += f"{parameter} ({definition['type']})"

        return (
            f"<|im_start|>system\n{system_content}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generation_loop(self) -> None:
        small_llm_model = Small_LLM_Model()
        generator = Generator(self.llm, self.vocabs, self.funcs_def, self.tokens_sets)

        for prompt in self.prompts:
            tokens_str = self._build_prompt(self.funcs_def, prompt.value)
            tokens = self.llm.llm_encode(tokens_str)
            result = generator.generate(prompt, tokens, len(tokens))

            start = result.find('{"prompt":')

            generated = result[start:]
            CYAN = "\033[96m"
            RESET = "\033[0m"
            try:
                decoded, _ = json.JSONDecoder().raw_decode(generated)
                print(CYAN + json.dumps(decoded, indent=2) + RESET)
            except json.JSONDecodeError:
                print(generated)

        return


    def build_type_token_sets(self, vocabs: dict[str, int]) -> dict:
        digit_tokens = []
        dot_tokens = []
        minus_tokens = []
        quote_tokens = []
        non_quote = []
        closing_brackets = []
        forbidden_quote_tokens = []

        forbidden_quotes = {"”", "“", "‟"}

        for token_text, token_id in vocabs.items():
            decoded = self.llm.llm_decode([token_id])

            if (
                any(char in token_text for char in forbidden_quotes)
                or any(char in decoded for char in forbidden_quotes)
            ):
                forbidden_quote_tokens.append(token_id)

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
            elif "}" in bare:
                closing_brackets.append(tok_id)
            elif '"' in bare and ',' in bare:
                pass
            elif '"' in bare and '\\' in bare:
                pass
            else:
                non_quote.append(tok_id)

        return {
            "digit": digit_tokens,
            "dot": dot_tokens,
            "minus": minus_tokens,
            "quote": quote_tokens,
            "non_quote": non_quote,
            "bracket": closing_brackets,
            "forbidden_quote": forbidden_quote_tokens,
        }
