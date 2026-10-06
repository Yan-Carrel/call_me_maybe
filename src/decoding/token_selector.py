from .llm import Llm
import copy
import torch
import numpy as np


def append_best_token(
    llm: Llm, tokens: list[int],
    possible_tokens: list[str], vocabs: dict[str, int]
        ) -> list[int]:
    if not possible_tokens:
        while self.llm.llm_decode(tokens[-1]) != '\"':
            logits = llm.llm.get_logits_from_input_ids(tokens)

    tokens_list = [
        llm.llm_encode(t) for t in possible_tokens
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
        logits = llm.llm.get_logits_from_input_ids(tokens)
        best_token_id = max(tokens_to_choose, key=lambda token_id: logits[token_id])
        tokens.append(best_token_id)
        result_token.append(best_token_id)
        if any(llm.llm_decode(result_token) == funct for funct in possible_tokens):
            break
        i += 1
    print(llm.llm_decode(result_token), end="", flush=True)
    return result_token


def append_token(llm: Llm, character: str, tokens: list[int], vocabs: dict[str, int]) -> None:
    result = llm.llm_encode(character)
    tokens.extend(result)
    print(llm.llm_decode(result), end="", flush=True)


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