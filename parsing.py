import sys
import json
import argparse
from typing import Any
from pydantic import ValidationError
from models import Prompt, FunctionDefinition


def parsing() -> tuple[list[Prompt], list[FunctionDefinition]]:
    parser = argparse.ArgumentParser(
        description="Process function calling data.")
    parser.add_argument('--functions_definition', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()
    with open(args.functions_definition, "r") as funcs_def_file:
        funcs_def_list = []
        try:
            funcs_def_list = json.load(funcs_def_file)
        except json.JSONDecodeError:
            print("Error, json file is invalid.")
    with open(args.input, "r") as input_file:
        input_list = []
        try:
            input_list = json.load(input_file)
        except json.JSONDecodeError:
            print("Error, json file is invalid.")

    return (
        parse_function_calling(input_list),
        parse_functions_definition(funcs_def_list))


def parse_function_calling(input_list: list[dict[str, Any]]) -> list[Prompt]:
    prompts = []
    for item in input_list:
        if len(item) != 1:
            print(f"Error, too many key-value pairs in {item}.")
        for key, value in item.items():
            if key != "prompt":
                print(
                    "Error, 'prompt' is the only valid key"
                    " in function_calling.json."
                    )
                sys.exit(0)
            try:
                prompts.append(Prompt(key=key, value=value))
            except ValidationError as e:
                errors = e.errors()[0]
                print(errors['msg'])
                print(f"Input: {errors['input']}")
                sys.exit(0)
    return prompts


def parse_functions_definition(
    funcs_def_list: list[dict[str, Any]]
        ) -> list[FunctionDefinition]:
    funcs_defs = []
    for item in funcs_def_list:
        valid_keys = ["name", "description", "parameters", "returns"]
        for key in valid_keys:
            if key not in item:
                print(
                    "Error, 'name', 'description', 'parameters' "
                    "and 'returns' are mandatory for function definition.")
                sys.exit(0)
        try:
            funcs_defs.append(FunctionDefinition(
                name=item["name"],
                description=item["description"],
                parameters=item["parameters"],
                returns=item["returns"]
            ))
        except ValidationError as e:
            errors = e.errors()[0]
            print(errors['msg'])
            print(f"Input: {errors['input']}")
            sys.exit(0)
    return funcs_defs
