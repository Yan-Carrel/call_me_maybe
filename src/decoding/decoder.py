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
        available_functs_list = []
        for prompt in self.prompts:
            tokens_str = (
                "System: You are a function calling assistant. "
                "Given a user prompt and a list of available "
                "functions with their parameters and descriptions, "
                "choose the appropriate function and extract its parameters.\n"
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
