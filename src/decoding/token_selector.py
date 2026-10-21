"""Select and append tokens using constrained vocabulary matching."""
from .llm import Llm
import math
from collections.abc import Sequence


def append_best_token(
    llm: Llm, tokens: list[int],
    possible_tokens: Sequence[str], vocabs: dict[str, int]
        ) -> list[int]:
    """Select and append the best matching token sequence.

    Compare the encoded candidate strings, restrict token selection to
    valid candidates, and choose tokens according to the model's logits.

    Args:
        llm: Language model used to encode and decode tokens.
        tokens: Token sequence to which selected tokens are appended.
        possible_tokens: Candidate strings from which to select a match.
        vocabs: Mapping of vocabulary tokens to their IDs.

    Returns:
        The token IDs selected for the candidate string.
    """
    if not possible_tokens:
        while llm.llm_decode(tokens[-1]) != '\"':
            logits = llm.llm.get_logits_from_input_ids(tokens)

    tokens_list = [
        llm.llm_encode(t) for t in possible_tokens
    ]

    i = 0
    result_token: list[int] = []
    while i < len(max(tokens_list, key=len)):
        tokens_to_choose = []
        for token in tokens_list:
            if valid_token(result_token, token, i):
                tokens_to_choose.append(token[i])
        if tokens_to_choose == []:
            break
        if len(tokens_to_choose) == 1:
            logits = llm.llm.get_logits_from_input_ids([0])
        else:
            logits = llm.llm.get_logits_from_input_ids(tokens)
        logits = mask_logits(logits, tokens_to_choose)
        best_token_id = logits.index(max(logits))
        print(llm.llm_decode(best_token_id), end="", flush=True)
        tokens.append(best_token_id)
        result_token.append(best_token_id)
        if any(
            llm.llm_decode(result_token) == funct
            for funct in possible_tokens
                ):
            break
        i += 1
    return result_token


def append_token(
    llm: Llm, character: str, tokens: list[int],
    vocabs: dict[str, int]
        ) -> None:
    """Encode and append a string to the token sequence.

    Args:
        llm: Language model used to encode and decode tokens.
        character: String to encode and append.
        tokens: Token sequence to extend.
        vocabs: Mapping of vocabulary tokens to their IDs.

    Returns:
        None.
    """
    result = llm.llm_encode(character)
    tokens.extend(result)
    print(llm.llm_decode(result), end="", flush=True)


def valid_token(result_token: list[int], token: list[int], i: int) -> bool:
    """Check whether a candidate token sequence matches the current prefix.

    Args:
        result_token: Tokens selected so far.
        token: Encoded candidate token sequence.
        i: Current position in the candidate sequence.

    Returns:
        True if the candidate has a token at position i and its prefix
        matches result_token; otherwise, False.
    """
    return i < len(token) and token[:i] == result_token


def mask_logits(
    logits: list[float],
    allowed: list[int]
        ) -> list[float]:
    """Disable token choices that are not in the allowed set.

    Set the logits of disallowed token IDs to negative infinity so they
    cannot be selected by the generation process.

    Args:
        logits: Model scores indexed by token ID.
        allowed: Token IDs permitted for selection.

    Returns:
        The modified logits list.
    """
    allowed_ids = set(allowed)
    for i in range(len(logits)):
        if i not in allowed_ids:
            logits[i] = -math.inf

    return logits
