from src.models import Prompt, FunctionDefinition
from .generator import Generator
from src.llm_sdk import Small_LLM_Model
import copy


class Decoder:
    def __init__(
        self, llm: Small_LLM_Model, prompts: list[Prompt],
        funcs_def: list[FunctionDefinition], vocabs: dict[str, int]
            ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.funcs_def = funcs_def
        self.vocabs = vocabs

    def generation_loop(self) -> None:
        generator = Generator(self.llm, self.vocabs, self.funcs_def)
        for prompt in self.prompts:
            tokens = self.llm.encode(prompt.value)
            result = generator.generate(prompt, tokens)
            print(f"\n{result[len(prompt.value):]}")