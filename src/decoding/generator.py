"""Generate structured function calls using constrained token selection."""
from .token_selector import append_token, append_best_token, mask_logits
from src.models import Prompt, FunctionDefinition
from .llm import Llm
from typing import Any
import json
from .number_generator import generate_number
from .regex_generator import generate_regex
from .string_generator import generate_string
from pydantic import BaseModel, ConfigDict


class Generator(BaseModel):
    """Generate function names and parameters for user prompts.

    Attributes:
        llm: Language model used to encode, decode, and generate tokens.
        vocabs: Mapping of vocabulary tokens to their IDs.
        funcs_def: Available function definitions and their parameters.
        tokens_sets: Groups of token IDs used during constrained generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Llm
    vocabs: dict[str, int]
    funcs_def: list[FunctionDefinition]
    tokens_sets: dict[str, Any]

    def generate(
        self, prompt: Prompt, tokens: list[int], len_srt: int
            ) -> str:
        """Generate a complete function call for a user prompt.

        Args:
            prompt: User prompt to process.
            tokens: Initial token IDs used to build the output.
            len_srt: Length of the initial token sequence.

        Returns:
            The generated function call as a string.
        """
        result = self.generate_prompt(prompt, tokens)
        result, selected_funct = self.generate_function_name(tokens)
        result = self.generate_parameters(
            tokens, prompt, selected_funct, self.tokens_sets)
        print()
        return result

    def generate_prompt(self, prompt: Prompt, tokens: list[int]) -> str:
        """Generate the prompt field in the function-call output.

        Args:
            prompt: User prompt whose value will be encoded.
            tokens: Token sequence to which generated tokens are appended.

        Returns:
            The generated prompt field as a decoded string.
        """
        generation_order = self.llm.llm_encode(
            f"{{\"prompt\":{json.dumps(prompt.prompt, ensure_ascii=False)},"
            )

        result_token: list[int] = []
        for text in generation_order:
            logits = self.llm.llm.get_logits_from_input_ids([0])
            logits = mask_logits(logits, [text])
            best = logits.index(max(logits))
            tokens.append(best)
            result_token.append(best)
            print(self.llm.llm_decode(text), end="", flush=True)

        return self.llm.llm_decode(result_token)

    def generate_function_name(
        self, tokens: list[int],
            ) -> tuple[str, FunctionDefinition]:
        """Select a function and append its name to the output.

        Args:
            tokens: Token sequence being generated.

        Returns:
            A tuple of the decoded output and selected function definition.
        """
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
                selected_name = self.llm.llm_decode(selected_tokens)
        result = self.llm.llm_decode(tokens)
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
        """Generate the parameters required by the selected function.

        Uses the parameter types to select the
        appropriate generation method
        for strings, regular expressions, and numeric values.

        Args:
            tokens: Token sequence being generated.
            prompt: Original user prompt.
            selected_funct: Definition of the selected function.
            tokens_sets: Groups of token IDs used
            during constrained generation.

        Returns:
            The generated function call as a decoded string.
        """
        tokens_str = self.llm.llm_decode(tokens)
        append_token(self.llm, '"parameters":', tokens, self.vocabs)
        append_token(self.llm, '{', tokens, self.vocabs)

        param_names = list(selected_funct.parameters.keys())

        for i, param_name in enumerate(param_names):

            if i > 0 and self.llm.llm_decode(tokens)[-1] != ',':
                append_token(self.llm, ',', tokens, self.vocabs)

            append_token(self.llm, '"', tokens, self.vocabs)
            append_token(self.llm, param_name, tokens, self.vocabs)
            append_token(self.llm, '":', tokens, self.vocabs)

            param_type = selected_funct.parameters[param_name]["type"]

            if param_type == "string" and param_name == "regex":
                tokens_str = self.llm.llm_decode(tokens)
                tokens_str[tokens_str.rindex("{\"prompt\":"):]
                append_token(self.llm, '"', tokens, self.vocabs)
                generate_regex(
                    self.llm, tokens, self.vocabs
                )
            elif param_type == "boolean":
                generate_string(
                    self.llm, tokens, prompt.prompt,
                    self.vocabs, tokens_sets, param_name, param_type
                )
            elif param_type in ("string", "object", "array"):
                signals = {'{': '}', '[': ']', '"': '"'}
                if param_type == "object":
                    signal = '{'
                elif param_type == "array":
                    signal = '['
                else:
                    signal = '"'
                append_token(self.llm, signal, tokens, self.vocabs)

                generate_string(
                    self.llm, tokens, prompt.prompt,
                    self.vocabs, tokens_sets, param_name, param_type
                )
                if not self.llm.llm_decode(tokens).endswith(signals[signal]):
                    append_token(
                        self.llm, signals[signal], tokens, self.vocabs
                        )

            elif param_type in ("number", "integer", "float"):
                generate_number(
                    self.llm, tokens, self.vocabs,
                    prompt.prompt, tokens_sets, param_name, param_type
                )

        append_token(self.llm, '}}', tokens, self.vocabs)

        return self.llm.llm_decode(tokens)

    def split_character(self, string: str) -> list[str]:
        """Split a string into vocabulary-style space-prefixed tokens.

        Args:
            string: Input string to split.

        Returns:
            A list of strings with the first word unprefixed and subsequent
            words prefixed with the vocabulary's space marker.
        """
        words = string.split(" ")
        words[0] = "\"" + words[0]
        result = [words[0]] if words[0] else []
        result += ["Ġ" + w for w in words[1:]]
        return result
