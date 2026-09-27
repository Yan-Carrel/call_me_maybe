from src.llm_sdk import Small_LLM_Model
from src.models import Prompt, FunctionDefinition
import math
import torch
import numpy as np


def generate_number(llm: Small_LLM_Model, tokens: list[int], vocabs: dict[str, int], type_sets: dict[str, Any]) -> str:
    pass


def generate_string(llm: Small_LLM_Model, tokens: list[int], vocabs: dict[str, int]) -> None:
    result_token = []
    while True:
        logits = llm.get_logits_from_input_ids(tokens)
        best_token = logits.index(max(logits))
        result_token.append(best_token)
        tokens.append(best_token)
        next_token_str = llm.decode(best_token)
        print(next_token_str, end="", flush=True)

        if "\"" in next_token_str:
            break


def mask_logits(logits: list[float], allowed: list[str], vocabs: dict[str, int]) -> list[float]:
    for i, logit in enumerate(logits):
        if logit not in allowed:
            logits[i] = -math.inf
    return logits
