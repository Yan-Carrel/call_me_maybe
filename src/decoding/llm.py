from pydantic import BaseModel, Field, ConfigDict
from src.llm_sdk import Small_LLM_Model
from typing import Optional
import re


class Llm(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    vocabs: dict[str, int]
    vocabs_str: list[str]
    model_name: Optional[str] = Field(default="Qwen/Qwen3-0.6B")

    def llm_encode(self, text: str) -> list[int]:
        text = text.replace(" ", "Ġ").replace("\n", "Ć")
        splitted = re.findall(r"Ġ*[^Ġ]+", text)

        tokens = []

        for item in splitted:
            tokens += self.llm_tokenize(item)

        result = []

        for token in tokens:
            result.append(self.vocabs[token])
        return result

    def llm_decode(self, tokens: list[int] | int) -> str:
        if isinstance(tokens, int):
            return self.vocabs_str[tokens].replace("Ġ", " ").replace("Ć", "\n")
        result = []
        for token in tokens:
            result.append(self.vocabs_str[token])
        return "".join(result).replace("Ġ", " ").replace("Ć", "\n")

    def llm_tokenize(self, text: str) -> list[str]:
        result = []

        while text:
            i = len(text)

            while i > 0 and text[:i] not in self.vocabs:
                i -= 1

            if i <= 0:
                raise ValueError(f"Cannot tokenize: {text!r}")

            token = text[:i]
            result.append(token)
            text = text[i:]

        return result
