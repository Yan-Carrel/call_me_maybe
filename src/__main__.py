import pydantic
import sys
import json
import argparse
# from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process function calling data.")
    parser.add_argument('--functions_definition', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()
    with open(args.functions_definition, "r") as file:
        funcs_def = json.load(file)
