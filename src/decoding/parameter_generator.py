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
 
        if "\"" in llm.decode(best_token):
            break
            # print("break")
        result_token.append(best_token)
        tokens.append(best_token)
        next_token_str = llm.decode(best_token)
        print(next_token_str, end="", flush=True)

        i += 1


def generate_string(llm: Small_LLM_Model, tokens: list[int], type_sets) -> None:
    result_token = []
    while True:
        allowed = []
        allowed += type_sets["digit"] + type_sets["quote"] + type_sets["non_quote"] + type_sets["dot"] + type_sets["minus"]
        if llm.decode(tokens).endswith('"'):
            type_sets["bracket"]
        logits = llm.get_logits_from_input_ids(tokens)
        for token_id in type_sets["forbidden_quote"]:
            logits[token_id] = float("-inf")
        logits[854] = -math.inf
        logits = mask_logits(logits, allowed)
        best_token = logits.index(max(logits))
        result_token.append(best_token)
        tokens.append(best_token)
        next_token_str = llm.decode(best_token)
        print(next_token_str, end="", flush=True)

        if "\"" in next_token_str:
            break

def generate_regex(llm: Small_LLM_Model, tokens: list[int], vocabs, type_sets) -> None:
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


def bracket_validator(text: str) -> tuple[bool, str | None]:
    brackets: dict[str, str] = {'{': '}', '(': ')', '[': ']'}
    closers = set(brackets.values())
    bracket_list: list[str] = []
    for c in text:
        if c in brackets:
            bracket_list.append(c)
        elif c in closers:
            if not bracket_list or brackets[bracket_list[-1]] != c:
                return False, c

            bracket_list.pop()
        elif c == '"':
            backslashes = 0
            i = text.index(c) - 1

            while i >= 0 and text[i] == "\\":
                backslashes += 1
                i -= 1

            if backslashes % 2 == 0:
                return False, c
    if bracket_list:
        return False, brackets[bracket_list[-1]]
    return True, None


def mask_logits(
    logits: list[float],
    allowed: list[int]
) -> list[float]:

    allowed_ids = set(allowed)

    for i in range(len(logits)):
        if i not in allowed_ids:
            logits[i] = -math.inf

    return logits
