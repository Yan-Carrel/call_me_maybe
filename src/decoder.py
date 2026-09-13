# from .models import Prompt, FunctionDefinition
# from src.llm_sdk import Small_LLM_Model
# import copy


# class Decoder:
#     def __init__(
#         self, llm: Small_LLM_Model, prompts: list[Prompt],
#         funcs_def: list[FunctionDefinition], vocabs: dict[str, int]
#             ) -> None:
#         self.llm = llm
#         self.prompts = prompts
#         self.funcs_def = funcs_def
#         self.vocabs = vocabs

#     def generation_loop(self) -> None:
#         for prompt in self.prompts:
#             result = ""
#             tokens = self.llm.encode(prompt.value)
#             result = self.generate_prompt(prompt, tokens)[len(prompt.value):]

    # def generate_prompt(self, prompt: Prompt, tokens: list[int]) -> str:
    #     generation_order = ["{", '"prompt":']
    #     generation_order.extend(self.split_character(" " + prompt.value))
    #     generation_order.append(",")
    #     tokens = tokens.squeeze(0).tolist()

    #     for character in generation_order:
    #         self.append_token(character)
    #     return self.llm.decode(tokens)
    
    # def generate_function_name(self, functs: list[FunctionDefinition]) -> str:
    #     generation_order = ['"name":', [funct.name for funct in functs], ","]
    #     for item in generation_order:
    #         if isinstance(item, str):
    #             self.append_token(item)
    #         elif isinstance(item, list[str]):
    #             self.append_best_token(item)
    
    # def append_best_token(self, function_names: list[str]) -> str:
    #     result = ""
    
    # def append_token(self, character: str) -> None:
    #     try:
    #         tokens.append(self.vocabs[character])
    #     except KeyError:
    #         try:
    #             tokens.append(self.vocabs["Ġ" + character])
    #         except KeyError:
    #             self.split_and_append(tokens, character)

    # def split_character(self, string: str) -> list[str]:
    #     words = string.split(" ")
    #     result = [words[0]] if words[0] else []
    #     result += ["Ġ" + w for w in words[1:]]
    #     return result

    # def split_and_append(self, tokens: list[int], character: str) -> None:
    #     i = 0
    #     idx = len(character) - i
    #     first_half = character[:idx]
    #     second_half = character[idx:]
    #     while not self.exist_in_vocabs(first_half):
    #         i += 1
    #         idx = len(character) - i
    #         first_half = character[:idx]
    #         second_half = character[idx:]
    #     try:
    #         if first_half == " ":
    #             tokens.append(self.vocabs["Ġ"])
    #         else:
    #             tokens.append(self.vocabs[first_half])
    #     except KeyError:
    #         tokens.append(self.vocabs["Ġ" + first_half])
    #     if i > 0:
    #         self.split_and_append(tokens, second_half)
    
    # def exist_in_vocabs(self, character: str) -> bool:
    #     try:
    #         if character == " ":
    #             return True
    #         self.vocabs[character]
    #         return True
    #     except KeyError:
    #         try:
    #             self.vocabs["Ġ" + character]
    #             return True
    #         except KeyError:
    #             return False
    #     return False

