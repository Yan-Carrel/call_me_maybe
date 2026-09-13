from .token_selector import append_token, append_best_token
from src.llm_sdk import Small_LLM_Model


class Generator:
    def __init__(
        self, llm: Small_LLM_Model,
        vocabs: dict[str, int]
            ) -> str:
        self.llm = llm
        self.vocabs = vocabs
    
    def generate(self, prompt: Prompt, tokens: list[int]) -> str:
        result = self.generate_prompt(prompt, tokens)
        return result

    def generate_prompt(self, prompt: Prompt, tokens: list[int]) -> str:
        generation_order = ["{", '"prompt":']
        generation_order.extend(self.split_character(" " + prompt.value))
        generation_order.append(",")
        tokens = tokens.squeeze(0).tolist()

        for character in generation_order:
            append_token(character, tokens, self.vocabs)
        return self.llm.decode(tokens)

    def generate_function_name(self) -> str:
        generation_order = ['"name":', [funct.name for funct in functs], ","]
        for item in generation_order:
            if isinstance(item, str):
                append_token(item, self.vocabs)
            elif isinstance(item, list[str]):
                append_best_token(item)

    def split_character(self, string: str) -> list[str]:
        words = string.split(" ")
        result = [words[0]] if words[0] else []
        result += ["Ġ" + w for w in words[1:]]
        return result