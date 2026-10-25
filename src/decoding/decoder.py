"""Decode language model outputs into structured function calls."""
from pydantic import BaseModel, ConfigDict, Field
from src.models import Prompt, FunctionDefinition
from .generator import Generator
from .llm import Llm
from pathlib import Path
from typing import Any
import re
import json


class Decoder(BaseModel):
    """Decode prompts into function calls using constrained generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Llm
    funcs_def_json: str
    funcs_def: list[FunctionDefinition]
    prompts: list[Prompt]
    vocabs: dict[str, int]
    file_name: str

    tokens_sets: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Initialize token groups after model creation."""
        self.tokens_sets = self.type_token_sets(self.vocabs)

    def _build_prompt(
        self, funcs_def: list[FunctionDefinition]
            ) -> str:
        """Build the system and user messages for function calling.

        Args:
            funcs_def: Available function definitions.
            user_prompt: The user's input prompt.

        Returns:
            The formatted conversation string for the language model.
        """
        system_content = (
            "System: You are a function calling assistant.\n"

            "Given a user prompt and a list of functions, select the correct "
            "function and output its parameters.\n"
            "Rules:\n"
            "- Copy text parameters exactly as "
            "they appear in the user prompt.\n"
            "- Do not modify, add, remove, or repeat characters.\n"
            "- Output decimal numbers exactly as written by the user.\n"
            "- Do not add or repeat digits.\n"
            "- End each output with single double quotes.\n"
            "- Integers must be written as decimals, e.g. 16 -> 16.0.\n"
            "Example:\n"
            "prompt: Greet shrek\n"
            '"parameters":{"name":"shrek"}}\n\n'
            "- Return valid JSON with all quotes correctly closed.\n"

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
        )

    def generation_loop(self) -> None:
        """Generate function calls and save valid JSON results.

        Encodes each prompt, generates a response, extracts the JSON
        object, and appends successfully decoded results to the output file.
        """
        generator = Generator(
            llm=self.llm, vocabs=self.vocabs,
            funcs_def=self.funcs_def, tokens_sets=self.tokens_sets)
        results = []
        output_path = Path(self.file_name)
        tokens_str = self._build_prompt(self.funcs_def)
        tokens_prompt = self.llm.llm_encode(tokens_str)
        for prompt in self.prompts:
            additional_prompt = (
                self.llm.llm_encode(
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                    )
            )
            tokens = tokens_prompt[:]
            tokens.extend(additional_prompt)
            result = generator.generate(prompt, tokens, len(tokens))
            start = result.find('{"prompt":')

            generated = result[start:]
            try:
                decoded, _ = json.JSONDecoder().raw_decode(generated)
                results.append(decoded)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.file_name, "w", encoding="utf-8") as json_file:
                    json_file.write(json.dumps(
                        results, indent=4, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"Error, Invalid JSON: {generated}")

        return

    def type_token_sets(self, vocabs: dict[str, int]) -> dict:
        """Group vocabulary token IDs according to their character types.

        Args:
            vocabs: Mapping of token strings to their integer IDs.

        Returns:
            A dictionary containing token IDs grouped into digits,
            decimal points, minus signs, quotes, non-quote tokens,
            closing brackets, and forbidden quotation marks.
        """
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
