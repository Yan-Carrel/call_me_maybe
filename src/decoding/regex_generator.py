"""Generate regular expression parameters using constrained token selection."""
import math
import json
from .llm import Llm
from .token_selector import mask_logits


def generate_regex(
    llm: Llm, tokens: list[int], vocabs: dict[str, int],
        ) -> None:
    """Generate a regular expression for a function parameter.

    Generate tokens while handling backslashes and quotation marks, then
    validate and append the generated expression to the output.

    Args:
        llm: Language model used to generate and decode tokens.
        tokens: Token sequence to which the generated expression is appended.
        vocabs: Mapping of vocabulary tokens to their IDs.

    Returns:
        None.
    """
    result_token: list[int] = []
    tokens_cp = tokens.copy()
    for _ in range(15):
        logits = llm.llm.get_logits_from_input_ids(tokens_cp)
        logits[vocabs["\\n"]] = -math.inf
        best_token = logits.index(max(logits))
        if (
            result_token
            and count_backslash_at_end(
                llm.llm_decode(result_token)
                ) % 2 != 0
                    ):
            best_token = vocabs["\\"]
        next_token_str = llm.llm_decode(best_token)
        if (
            '"' in next_token_str
            and not next_token_str.endswith('"')
            and next_token_str != '\\"'
                ):
            try:
                best_token = vocabs[next_token_str[:len(next_token_str) - 1]]
            except Exception:
                best_token = vocabs['"']
        result_token.append(best_token)
        tokens_cp.append(best_token)

        if "\"" in next_token_str:
            break
    display_tokens(llm, result_token, tokens, vocabs)


def display_tokens(
    llm: Llm, result_token: list[int], tokens: list[int],
    vocabs: dict[str, int]
        ) -> None:
    """Validate and append generated regular expression tokens.

    Validate the generated expression as part of a JSON string. If validation
    fails, append and display a double quote instead.

    Args:
        llm: Language model used to decode and process tokens.
        result_token: Token IDs representing the generated expression.
        tokens: Output token sequence to extend.
        vocabs: Mapping of vocabulary tokens to their IDs.

    Returns:
        None.
    """
    try:
        generated_str = llm.llm_decode(result_token)
        json.loads(f'{{"regex":"{generated_str}}}')

        for token in result_token:
            logits = llm.llm.get_logits_from_input_ids([0])
            logits = mask_logits(logits, [token])
            best = logits.index(max(logits))
            tokens.append(best)
            print(llm.llm_decode([best]), end="", flush=True)

    except json.JSONDecodeError:
        tokens.append(vocabs['"'])
        print('"', end="", flush=True)


def count_backslash_at_end(text: str) -> int:
    """Count consecutive backslashes at the end of a string.

    Args:
        text: String whose trailing backslashes are counted.

    Returns:
        The number of consecutive backslashes at the end of the string.
    """
    count = 0
    i = len(text) - 1
    while i >= 0 and text[i] == "\\":
        i -= 1
        count += 1
    return count
