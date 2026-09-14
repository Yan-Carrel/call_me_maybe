from src.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition


def append_best_token(
    llm: Small_LLM_Model, tokens: list[int],
    funcs_def_name: list[str], vocabs: dict[str, int]
        ) -> str:
    functions_token = [llm.encode(funct) for funct in funcs_def_name]
    tokens_list = list(map(lambda x: x.squeeze(0).tolist(), functions_token))
    i = 0
    result_token = []
    while i < len(max(tokens_list, key=len)):
        tokens_to_choose = []
        for token in tokens_list:
            if valid_token(llm, result_token, token, i):
                tokens_to_choose.append(token[i])
        if tokens_to_choose == []:
            break
        logits = llm.get_logits_from_input_ids(tokens)
        best_token_id = max(tokens_to_choose, key=lambda token_id: logits[token_id])
        tokens.append(best_token_id)
        result_token.append(best_token_id)
        if any(llm.decode(result_token) == funct for funct in funcs_def_name):
            break
        i += 1
    result = llm.decode(tokens)
    return result

def append_token(character: str, tokens: list[int], vocabs: dict[str, int]) -> None:
    try:
        tokens.append(vocabs[character])
    except KeyError:
        try:
            tokens.append(vocabs["Ġ" + character])
        except KeyError:
            split_and_append(tokens, character, vocabs)

def valid_token(llm: Small_LLM_Model, result_token: list[int], token: list[int], i: int) -> bool:
    if i < len(token):
        if result_token:
            result_token_str = llm.decode(result_token)
            token_str = llm.decode(token)
            if token_str.startswith(result_token_str):
                return True
            else:
                return False
        else:
            return True
    return False

def split_and_append(
    tokens: list[int],
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
        else:
            tokens.append(vocabs[first_half])
    except KeyError:
        tokens.append(vocabs["Ġ" + first_half])
    if i > 0:
        split_and_append(tokens, second_half, vocabs)

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