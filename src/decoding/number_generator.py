"""Generate numeric parameter values using constrained token selection."""
import json
import re
from .llm import Llm
from typing import Any
from .token_selector import mask_logits


def generate_number(
    llm: Llm, tokens: list[int], vocabs: dict[str, int],
    prompt: str, type_sets: dict[str, Any], key: str, param_type: str
        ) -> None:
    """Generate a numeric value for a function parameter.

    Extract numbers from the user prompt and use them to guide token
    generation while restricting the allowed numeric characters.

    Args:
        llm: Language model used to generate and decode tokens.
        tokens: Token sequence to which the generated number is appended.
        vocabs: Mapping of vocabulary tokens to their IDs.
        prompt: Original user prompt.
        type_sets: Groups of token IDs used for constrained generation.
        key: Parameter name or type used to format the number.
        param_type: Parameter type that indicate the type being generated.

    Returns:
        None.
    """
    result_token: list[int] = []
    used_dot = False
    number_after_dot = False
    nbs_in_prompt = re.findall(r"-*\d+\.\d+|-*\d+", prompt)
    formatted_nbs = format_nbs(nbs_in_prompt, param_type)
    tokens_cp = tokens.copy()
    if nbs_in_prompt:
        max_tokens = len(
            max(formatted_nbs, key=lambda x: len(llm.llm_encode(x)))
            )
    else:
        max_tokens = 5
    i = 0
    for i in range(max_tokens):
        allowed = get_allowed_tokens(i, type_sets, used_dot, number_after_dot)
        if (
            result_token and
            (
                any(
                True for nb in formatted_nbs
                if llm.llm_decode(result_token) == nb)
            )
                ):
            break

        logits = llm.llm.get_logits_from_input_ids(tokens_cp)
        logits = mask_logits(logits, allowed)
        best_token = logits.index(max(logits))

        if llm.llm_decode(best_token) == ".":
            used_dot = True
        elif used_dot:
            number_after_dot = True
        elif llm.llm_decode(best_token) != "." and used_dot:
            number_after_dot = True

        if "\"" in llm.llm_decode(best_token) and nbs_in_prompt == []:
            break
        result_token.append(best_token)
        tokens_cp.append(best_token)

        i += 1
        if i >= max_tokens and not nbs_in_prompt:
            break

    display_tokens(llm, result_token, tokens, key, vocabs)


def get_allowed_tokens(
    i: int, type_sets: dict[str, Any],
    used_dot: bool, number_after_dot: bool
        ) -> list[int]:
    """Determine which token IDs are allowed at the current position.

    Args:
        i: Current position in the generated number.
        type_sets: Groups of token IDs for numeric characters and quotes.
        used_dot: Whether a decimal point has already been generated.
        number_after_dot: Whether a digit has been generated after the
            decimal point.

    Returns:
        A list of allowed token IDs.
    """
    allowed = []
    allowed += type_sets["digit"]
    if i == 0:
        allowed += type_sets["minus"]
    elif not used_dot:
        allowed += type_sets["dot"]
    if number_after_dot:
        allowed += type_sets["quote"]
    return allowed


def display_tokens(
    llm: Llm, result_token: list[int], tokens: list[int],
    key: str, vocabs: dict[str, int]
        ) -> None:
    """Validate and append generated numeric tokens to the output.

    Validate the generated value as JSON before appending its tokens.
    If validation fails, append and display a double quote instead.

    Args:
        llm: Language model used to decode and process tokens.
        result_token: Token IDs representing the generated number.
        tokens: Output token sequence to extend.
        key: Parameter name used to validate the JSON value.
        vocabs: Mapping of vocabulary tokens to their IDs.

    Returns:
        None.
    """
    try:
        generated_str = llm.llm_decode(result_token)
        json.loads(f'{{"{key}":{generated_str}}}')

        for token in result_token:
            logits = llm.llm.get_logits_from_input_ids([0])
            logits = mask_logits(logits, [token])
            best = logits.index(max(logits))
            tokens.append(best)
            print(llm.llm_decode([best]), end="", flush=True)

    except json.JSONDecodeError:
        tokens.append(vocabs['0'])
        tokens.append(vocabs['.'])
        tokens.append(vocabs['0'])
        print('0.0', end="", flush=True)


def format_nbs(nbs: list[str], param_type: str) -> list[str]:
    """Format extracted numbers according to the requested parameter type.

    Append a decimal suffix to integer-form numbers when the key is
    "integer", as implemented by the function.

    Args:
        nbs: Numbers extracted from the user prompt.
        key: Parameter name or type that determines number formatting.

    Returns:
        A list of formatted number strings.
    """
    result = []
    for nb in nbs:
        if '.' not in nb and param_type != "integer":
            nb += '.0'
            result.append(nb)
        else:
            result.append(nb)
    return result
