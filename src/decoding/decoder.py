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
        # available_functs = [funct for funct in self.funcs_def]
        available_functs_list = []
        for funct in self.funcs_def:
            s = ""
            s += f"{funct.name} | {funct.description} | parameters:"
            for param in funct.parameters:
                s += f" {param}={funct.parameters[param]["type"]}"
            available_functs_list.append(s)
        available_functs = "\n".join(available_functs_list)
        for prompt in self.prompts:
            tokens_str = prompt.value + "\navailable functions:\n" + available_functs
            tokens = self.llm.encode(tokens_str)
            result = generator.generate(prompt, tokens)
            # print(tokens_str)
            # print(result)
            # import sys
            # sys.exit(0)
            print(f"\n{result[len(tokens_str):]}")