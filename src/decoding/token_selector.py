from src.llm_sdk import Small_LLM_Model
import copy
import torch


def append_best_token(
    llm: Small_LLM_Model, tokens: list[int],
    possible_tokens: list[str], vocabs: dict[str, int]
        ) -> list[int]:
    if not possible_tokens:
        return []
    
    possible_tokens_list = [
        llm.encode(t).squeeze(0).tolist() for t in possible_tokens
    ]
    possible_tokens_list = [c for c in possible_tokens_list if c]
    if not possible_tokens_list:
        return []
    if len(possible_tokens_list) == 1:
        tokens.extend(possible_tokens_list[0])
        print(llm.decode(possible_tokens_list[0]), end="", flush=True)
        return possible_tokens_list[0]

    chosen: list[int] = []
    live = list(range(len(possible_tokens_list)))
    pos = 0

    while True:
        live = [i for i in live if pos < len(possible_tokens_list[i])]
        if len(live) <= 1:
            break

        logits = llm.get_logits_from_input_ids(tokens + chosen)
        log_probs = torch.log_softmax(torch.tensor(logits), dim=0)

        # only compare the tokens that live candidates actually offer at this position
        next_token_options = {possible_tokens_list[i][pos] for i in live}
        best_tok = max(next_token_options, key=lambda t: log_probs[t].item())

        chosen.append(best_tok)
        live = [i for i in live if possible_tokens_list[i][pos] == best_tok]
        pos += 1

    if len(live) == 1:
        chosen.extend(possible_tokens_list[live[0]][pos:])

    tokens.extend(chosen)
    print(llm.decode(chosen), end="", flush=True)
    return chosen


def append_token(llm: Small_LLM_Model, character: str, tokens: list[int], vocabs: dict[str, int]) -> None:
    try:
        tokens.append(vocabs[character])
        print(llm.decode(vocabs[character]), end="", flush=True)
    except KeyError:
        try:
            tokens.append(vocabs["Ġ" + character])
            print(llm.decode(vocabs["Ġ" + character]), end="", flush=True)
        except KeyError:
            split_and_append(llm, tokens, character, vocabs)

def valid_token(result_token: list[int], token: list[int], i: int) -> bool:
    return i < len(token) and token[:i] == result_token

def split_and_append(
    llm: Small_LLM_Model, tokens: list[int],
    character: str, vocabs: dict[str, int]
        ) -> None:
    i = 0
    idx = len(character) - i
    first_half = character[:idx]
    second_half = character[idx:]
    while not exist_in_vocabs(first_half, vocabs):
        i += 1
        idx = len(character) - i
        first_half = character[:idx]
        second_half = character[idx:]
    try:
        if first_half == " ":
            tokens.append(vocabs["Ġ"])
            print(llm.decode(vocabs["Ġ"]), end="", flush=True)
        else:
            tokens.append(vocabs[first_half])
            print(llm.decode(vocabs[first_half]), end="", flush=True)
    except KeyError:
        tokens.append(vocabs["Ġ" + first_half])
        print(llm.decode(vocabs["Ġ" + first_half]), end="", flush=True)
    if i > 0:
        split_and_append(llm, tokens, second_half, vocabs)

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