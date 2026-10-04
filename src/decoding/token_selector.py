from src.llm_sdk import Small_LLM_Model
import copy
import torch
import numpy as np


def append_best_token(
    llm: Small_LLM_Model, tokens: list[int],
    possible_tokens: list[str], vocabs: dict[str, int]
        ) -> list[int]:
    if not possible_tokens:
        while self.llm.decode(tokens[-1]) != '\"':
            logits = llm.get_logits_from_input_ids(tokens)

    tokens_list = [
        llm.encode(t).squeeze(0).tolist() for t in possible_tokens
    ]

    i = 0
    result_token = []
    while i < len(max(tokens_list, key=len)):
        tokens_to_choose = []
        for token in tokens_list:
            if valid_token(result_token, token, i):
                tokens_to_choose.append(token[i])
        if tokens_to_choose == []:
            break
        logits = llm.get_logits_from_input_ids(tokens)
        best_token_id = max(tokens_to_choose, key=lambda token_id: logits[token_id])
        tokens.append(best_token_id)
        result_token.append(best_token_id)
        if any(llm.decode(result_token) == funct for funct in possible_tokens):
            break
        i += 1
    print(llm.decode(result_token), end="", flush=True)
    return result_token


def append_token(llm: Small_LLM_Model, character: str, tokens: list[int], vocabs: dict[str, int]) -> None:
    result = llm.encode(character).squeeze(0).tolist()
    tokens.extend(result)
    print(llm.decode(result), end="", flush=True)


def valid_token(result_token: list[int], token: list[int], i: int) -> bool:
    return i < len(token) and token[:i] == result_token


def exist_in_vocabs(character: str, vocabs: dict[str, int]) -> bool:
    try:
        if character == " ":
            return True
        vocabs[character]
        return True
    except KeyError:
        try:
            vocabs["Ġ" + character]
            return True
        except KeyError:
            return False
    return False