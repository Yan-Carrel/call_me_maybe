"""Parse command-line arguments and validate function-calling data."""
import sys
import json
import argparse
from typing import Any
from pydantic import ValidationError
from src.models import Prompt, FunctionDefinition


def parsing() -> tuple[list[Prompt], list[FunctionDefinition], str, str, str]:
    """Parse command-line arguments and load input JSON files.

    Load the function definitions and prompts from the specified files,
    validate their contents, and determine the model name.

    Returns:
        A tuple containing the parsed prompts, function definitions,
        function-definition file path, output file path, and model name.
    """
    parser = argparse.ArgumentParser(
        description="Process function calling data.")
    parser.add_argument('--functions_definition', required=False)
    parser.add_argument('--input', required=False)
    parser.add_argument('--output', required=False)
    parser.add_argument('--model_name', required=False)

    args = parser.parse_args()
    paths = [
        "data/input/functions_definition.json",
        "data/input/function_calling_tests.json",
        "data/output/function_calling_results.json"]
    funcs_def = args.functions_definition or paths[0]
    funcs_call = args.input or paths[1]
    funcs_call_result = args.output or paths[2]

    with open(funcs_def, "r") as funcs_def_file:
        funcs_def_list = []
        try:
            funcs_def_list = json.load(funcs_def_file)
        except Exception:
            print("Error, json file is invalid.")
    with open(funcs_call, "r") as input_file:
        input_list = []
        try:
            input_list = json.load(input_file)
        except Exception:
            print("Error, json file is invalid.")

    if not args.model_name:
        model_name = "Qwen/Qwen3-0.6B"
    else:
        model_name = args.model_name

    return (
        parse_function_calling(input_list),
        parse_functions_definition(funcs_def_list),
        funcs_def,
        funcs_call_result,
        model_name
        )


def parse_function_calling(input_list: list[dict[str, Any]]) -> list[Prompt]:
    """Validate and convert input items into Prompt objects.

    Args:
        input_list: List of dictionaries containing prompt data.

    Returns:
        A list of validated Prompt objects.
    """
    prompts = []
    for item in input_list:
        if len(item) != 1:
            print(f"Error, too many key-value pairs in {item}.")
        try:
            prompts.append(Prompt.model_validate(item))
        except ValidationError as e:
            errors = e.errors()[0]
            print(errors['msg'])
            print(f"Input: {errors['input']}")
            sys.exit(0)
    return prompts


def parse_functions_definition(
    funcs_def_list: list[dict[str, Any]]
        ) -> list[FunctionDefinition]:
    """Validate and convert function definitions into model objects.

    Args:
        funcs_def_list: List of dictionaries containing function definitions.

    Returns:
        A list of validated FunctionDefinition objects.
    """
    funcs_defs = []
    for item in funcs_def_list:
        try:
            funcs_defs.append(FunctionDefinition.model_validate(item))
        except ValidationError as e:
            errors = e.errors()[0]
            print(errors['msg'])
            print(f"Input: {errors['input']}")
            sys.exit(0)
    return funcs_defs
