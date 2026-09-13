def append_best_token(function_names: list[str]) -> str:
    result = ""

def append_token(character: str, tokens: list[int], vocabs: dict[str, int]) -> None:
    try:
        tokens.append(vocabs[character])
    except KeyError:
        try:
            tokens.append(vocabs["Ġ" + character])
        except KeyError:
            split_and_append(tokens, character, vocabs)

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