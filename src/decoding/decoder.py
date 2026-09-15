from src.models import Prompt, FunctionDefinition
from .generator import Generator
from src.llm_sdk import Small_LLM_Model
import copy


class Decoder:
    def __init__(
        self, llm: Small_LLM_Model, funcs_def_json: str, funcs_def: list[FunctionDefinition],
        prompts: list[Prompt], vocabs: dict[str, int],
            ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.funcs_def = funcs_def
        self.vocabs = vocabs
        self.funcs_def_json = funcs_def_json

    def generation_loop(self) -> None:
        generator = Generator(self.llm, self.vocabs, self.funcs_def)
        # available_functs = [funct for funct in self.funcs_def]
        available_functs_list = []
        # for funct in self.funcs_def:
        #     s = ""
        #     s += f"{funct.name} | {funct.description} | parameters:"
        #     for param in funct.parameters:
        #         s += f" {param}={funct.parameters[param]['type']}"
        #     available_functs_list.append(s)
        # available_functs = "\n".join(available_functs_list)
        # print(type(self.prompts))
        for prompt in self.prompts:
            # tokens_str = prompt.value + "\navailable functions:\n" + available_functs
            tokens_str = prompt.value + self.funcs_def_json
            tokens = self.llm.encode(tokens_str)
            result = generator.generate(prompt, tokens, len(tokens_str))
