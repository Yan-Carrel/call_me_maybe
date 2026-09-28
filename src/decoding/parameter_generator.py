from src.llm_sdk import Small_LLM_Model
from src.models import Prompt, FunctionDefinition
import math
import torch
import numpy as np


def generate_number(llm: Small_LLM_Model, tokens: list[int], vocabs: dict[str, int], type_sets: dict[str, Any]) -> str:
    result_token = []
    used_dot = False
    number_after_dot = False
    i = 0
    while True:
        allowed = []
        allowed += type_sets["digit"]
        if i == 0:
            allowed += type_sets["minus"]
        elif not used_dot:
            allowed += type_sets["dot"]
        if number_after_dot:
            allowed = type_sets["quote"]
        
        logits = llm.get_logits_from_input_ids(tokens)
        logits = mask_logits(logits, allowed)
        best_token = logits.index(max(logits))

        if llm.decode(best_token) == ".":
            used_dot = True
        elif used_dot:
            number_after_dot = True
        elif llm.decode(best_token) != "." and used_dot:
            number_after_dot = True
 
        result_token.append(best_token)
        tokens.append(best_token)
        next_token_str = llm.decode(best_token)
        print(next_token_str, end="", flush=True)

        i += 1
        if "\"" in next_token_str:
            break


def generate_string(llm: Small_LLM_Model, tokens: list[int], vocabs: dict[str, int]) -> None:
    result_token = []
    while True:
        logits = llm.get_logits_from_input_ids(tokens)
        best_token = logits.index(max(logits))
        result_token.append(best_token)
        tokens.append(best_token)
        next_token_str = llm.decode(best_token)
        if "\"" in next_token_str:
            break
        print(next_token_str, end="", flush=True)



def mask_logits(
    logits: list[float],
    allowed: list[int]
) -> list[float]:

    allowed_ids = set(allowed)

    for i in range(len(logits)):
        if i not in allowed_ids:
            logits[i] = -math.inf

    return logits
