from src.llm_sdk import Small_LLM_Model


def append_best_token(
    llm: Small_LLM_Model, tokens: list[int],
    possible_tokens: list[str], vocabs: dict[str, int]
        ) -> str:
    if not possible_tokens:
        return ""

    functions_token = [llm.encode(funct) for funct in possible_tokens]
    tokens_list = [
        encoded.squeeze(0).tolist()
        for encoded in functions_token
        if encoded.numel() > 0
    ]
    if not tokens_list:
        return ""

    i = 0
    result_token = []
    while tokens_list and i < len(max(tokens_list, key=len)):
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
        # print(llm.decode(best_token_id), end="", flush=True)
        if any(result_token == token for token in tokens_list):
            break
        i += 1
    result = llm.decode(result_token)
    print(result, end="", flush=True)
    return result

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