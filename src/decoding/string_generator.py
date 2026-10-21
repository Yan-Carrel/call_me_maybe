"""Generate string, array, and object parameters using token selection."""
import math
import json
from .llm import Llm
from .token_selector import mask_logits, append_best_token


def generate_string(
    llm: Llm, tokens: list[int], prompt: str,
    vocabs: dict[str, int], type_sets: dict[str, list[int]],
    key: str, param_type: str
        ) -> None:
    """Generate a string or delegate generation of an array or object.

    Restrict token selection to permitted token groups and stop when a
    closing quotation mark is generated.

    Args:
        llm: Language model used to generate and decode tokens.
        tokens: Token sequence to which the generated value is appended.
        prompt: Original user prompt used to determine the generation length.
        vocabs: Mapping of vocabulary tokens to their IDs.
        type_sets: Groups of token IDs used during constrained generation.
        key: Name of the parameter being generated.
        param_type: Expected parameter type.

    Returns:
        None.
    """
    if param_type != "string" and param_type != "boolean":
        generate_array_or_object(
            llm, tokens,
            vocabs, type_sets, key, param_type
        )
        return
    elif param_type == "boolean":
        result_token = append_best_token(
            llm, tokens, ["true", "false"], vocabs)
        return
    result_token = []
    max_tokens = len(llm.llm_encode(prompt)) + 1
    tokens_cp = tokens.copy()
    for _ in range(max_tokens):
        allowed = []
        allowed += (
            type_sets["digit"] + type_sets["quote"] + type_sets["non_quote"]
            + type_sets["dot"] + type_sets["minus"]
            )
        logits = llm.llm.get_logits_from_input_ids(tokens_cp)
        for token_id in type_sets["forbidden_quote"]:
            logits[token_id] = float("-inf")
        logits[854] = -math.inf
        logits = mask_logits(logits, allowed)
        best_token = logits.index(max(logits))
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
    display_tokens(llm, result_token, tokens, key, param_type, vocabs)


def generate_array_or_object(
    llm: Llm, tokens: list[int], vocabs: dict[str, int],
    type_sets: dict[str, list[int]], key: str, param_type: str,
    max_tokens: int = 15
        ) -> list[int]:
    """Generate an array or object parameter.

    Generate tokens until the appropriate closing character is encountered
    or the maximum number of tokens is reached.

    Args:
        llm: Language model used to generate and decode tokens.
        tokens: Existing token sequence.
        vocabs: Mapping of vocabulary tokens to their IDs.
        type_sets: Groups of token IDs available for generation.
        key: Name of the parameter being generated.
        param_type: Expected container type, either "array" or "object".
        max_tokens: Maximum number of tokens to generate.

    Returns:
        The token IDs generated for the array or object.
    """
    stop_char = "}" if param_type == "object" else "]"
    result_token: list[int] = []

    for i in range(max_tokens):

        logits = llm.llm.get_logits_from_input_ids(tokens + result_token)
        best_token = logits.index(max(logits))
        next_token_str = llm.llm_decode(best_token)

        if stop_char in next_token_str and vocabs['"'] in result_token:
            result_token.append(vocabs[f'{stop_char}'])
            break
        elif stop_char in next_token_str and result_token[0] == vocabs['"']:
            result_token.append(vocabs[stop_char])
            break

        result_token.append(best_token)
    display_tokens(llm, result_token, tokens, key, param_type, vocabs)
    return result_token


def display_tokens(
    llm: Llm, result_token: list[int], tokens: list[int],
    key: str, param_type: str, vocabs: dict[str, int]
        ) -> None:
    """Validate and append generated string, array, or object tokens.

    Validate the generated value as part of a JSON object before appending
    its tokens. If validation fails, append the appropriate closing character.

    Args:
        llm: Language model used to decode and process tokens.
        result_token: Token IDs representing the generated value.
        tokens: Output token sequence to extend.
        key: Name of the parameter being generated.
        param_type: Expected parameter type.
        vocabs: Mapping of vocabulary tokens to their IDs.

    Returns:
        None.
    """
    try:
        str_to_append = ""
        if param_type == "array":
            str_to_append = '['
        elif param_type == "object":
            str_to_append = '{'
        else:
            str_to_append = '"'
        generated_str = llm.llm_decode(result_token)
        if param_type == "string":
            json.loads(f'{{"{key}":"{generated_str}}}')
        else:
            json.loads('{' + f'"{key}":{str_to_append + generated_str}' + '}')

        for token in result_token:
            logits = llm.llm.get_logits_from_input_ids([0])
            logits = mask_logits(logits, [token])
            best = logits.index(max(logits))
            tokens.append(best)
            print(llm.llm_decode([best]), end="", flush=True)

    except json.JSONDecodeError:
        closes = {'{': '}', '[': ']'}
        if str_to_append == '"':
            print('"', end="", flush=True)
            tokens.append(vocabs['"'])
        else:
            print(closes[str_to_append], end="", flush=True)
            tokens.append(vocabs[closes[str_to_append]])
