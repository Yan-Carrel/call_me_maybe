*This project has been created as part of the 42 curriculum by yaandria.*

# call me maybe

Introduction to function calling in LLMs, using **constrained decoding** on a small `Qwen/Qwen3-0.6B` model.

## Description

Large language models are good at understanding natural language but unreliable at producing machine-readable output. A 0.6B-parameter model asked to emit JSON might only succeed a fraction of the time.

This project translates a natural-language request into a structured function call instead of answering it. Given:

```
"What is the sum of 40 and 2?"
```

the program does not answer `42`. It outputs:

```json
{
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {"a": 40.0, "b": 2.0}
}
```

The key point is that the JSON is not "hoped for" from the model. The output is built **token by token**: at each step the model's logits are masked so that only tokens that keep the structure and the schema valid can be selected. The LLM still makes the real decisions (which function, which argument values), but it can never break the format.

### Features

- Constrained decoding with direct manipulation of logits (`-inf` masking of invalid tokens).
- Function selection done by the LLM, restricted to the names found in `functions_definition.json`.
- Type-aware argument generation (`number`, `integer`, `float`, `boolean`, `string`, `array`, `object`), plus a dedicated path for regex parameters.
- **Custom tokenizer** (bonus): `encode` / `decode` are re-implemented from the vocabulary file, so the main code only relies on `get_logits_from_input_ids` and `get_path_to_vocab_file` from the SDK.
- Pydantic validation of both input files.
- Results are written to disk after each prompt, so a crash never loses finished work.

### Project structure

```
call_me_maybe
└── 📁call_me_maybe
    └── 📁data
        └── 📁input
            ├── function_calling_tests.json
            ├── functions_definition.json
    └── 📁llm_sdk
        ├── pyproject.toml
    └── 📁src
        └── 📁decoding
            ├── __init__.py
            ├── decoder.py
            ├── generator.py
            ├── llm.py
            ├── number_generator.py
            ├── regex_generator.py
            ├── string_generator.py
            ├── token_selector.py
        └── 📁llm_sdk
            ├── __init__.py
        ├── __main__.py
        ├── models.py
        ├── parsing.py
    ├── .flake8
    ├── .gitignore
    ├── Makefile
    ├── pyproject.toml
    ├── README.md
    └── uv.lock
```

## Instructions

### Requirements

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/)
- Runtime dependencies: `numpy`, `pydantic`, the Python standard library, and the provided `llm_sdk`.
- Development tools (dev dependency group): [`flake8`](https://flake8.pycqa.org/) for linting and [`mypy`](https://mypy.readthedocs.io/) for static type checking.

Forbidden packages (`transformers`, `torch`, `outlines`, `dspy`, etc.) are not used directly in the project code.

### Installation

Before installing dependencies, configure the environment variables used by `uv` and the Hugging Face libraries.

#### 1. Configure the environment

If you use **Zsh**, open your `~/.zshrc` file:

```bash
nano ~/.zshrc
```

Add the following configuration at the end of the file:

```bash
export UV_CACHE_DIR="/goinfre/user/.cache/uv"
export HF_HOME="/goinfre/user/.huggingface"
export TMPDIR="/goinfre/user/tmp"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TMPDIR"
```
Make sure to replace 'user' with your username.
Save the file, then reload the configuration:

```bash
source ~/.zshrc
```

If you use **Bash**, add the same configuration to `~/.bashrc` and reload it with `source ~/.bashrc`.

These variables configure the virtual environment, package cache, Hugging Face cache, and temporary directory. The configuration needs to be loaded before running `uv` commands.

#### 2. Install dependencies

Run:

```bash
make install
```

Alternatively, use:

```bash
uv sync
```

### Running

Run the program with the default configuration:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

Alternatively, use the Makefile shortcut:

```bash
make run
```

#### Command-line arguments

| Argument | Required | Description |
|---|---|---|
| `--functions_definition` | Yes | Path to the JSON file defining the available functions and their parameters. |
| `--input` | Yes | Path to the JSON file containing the prompts to process. |
| `--output` | Yes | Path where the generated function calls will be saved as JSON. |
| `--model_name` | No | Name or identifier of the language model to use. If omitted, the program uses its configured default model. |

To specify a different model, add `--model_name` to the command, using a model identifier supported by the project:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json \
    --model_name "Qwen/Qwen3-0.6B"
```

#### Supported models

The project reimplements the tokenizer (`src/decoding/llm.py`) from the model's vocabulary file, and builds prompts with the ChatML template (`<|im_start|>` / `<|im_end|>`). A model is therefore supported if it uses:

- a **byte-level BPE** vocabulary (GPT-2 / Qwen style, where a space is `Ġ`), and
- the **ChatML** chat template.

In practice, that means the Qwen family, example: `Qwen/Qwen2.5-0.5B-Instruct`

**Not supported:** models with a different tokenizer or chat template, such as Llama or Gemma (SentencePiece, `▁` marker), or Mistral. They might work but the result might be unexpected.

### Other Makefile rules

| Rule | Action |
|---|---|
| `make install` | Install project dependencies using `uv sync`. |
| `make run` | Run the program with the default arguments configured in the Makefile. |
| `make debug` | Run the program under Python's debugger (`pdb`). |
| `make clean` | Remove generated caches, including `__pycache__` and `.mypy_cache`, where applicable. |
| `make lint` | Run `flake8` and `mypy` with the project's configured checks. |

### Input format

`functions_definition.json`:

```json
[
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {"name": {"type": "string"}},
    "returns": {"type": "string"}
  }
]
```

`function_calling_tests.json`:

```json
[
  {"prompt": "Greet shrek"},
  {"prompt": "What is the sum of 2 and 3?"}
]
```

### Output format

A JSON array in which each object contains exactly `prompt`, `name` and `parameters`:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0}
    },
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": {"name": "shrek"}
    }
]
```

## Algorithm explanation

### 1. Principle

At every generation step the SDK returns one logit per token of the vocabulary (`get_logits_from_input_ids`). Instead of taking the `argmax` over all of them, we:

1. determine the set of tokens that are legal at this exact position of the JSON;
2. set every other logit to `-inf` (`mask_logits`);
3. take the `argmax` of what remains and append that token to the context.

Because illegal tokens have probability zero, the structure is valid by construction.

### 2. Prompt given to the model

For each user prompt, `Decoder._build_prompt` creates a Qwen chat-formatted context (`<|im_start|>system ... <|im_end|>`) containing a short instruction block, one worked example, and the list of functions with their parameter names and types. This context only *helps* the model make good choices. The correctness of the JSON never depends on it.

### 3. Generation order

`Generator.generate` builds the object in a fixed order. Fixed parts of the structure are *forced*; only the values are left to the model:

```
{"prompt": <forced: the user prompt, JSON-escaped>,
 "name": "<LLM chooses among the allowed function names>",
 "parameters": {"<forced key>": <LLM-generated, type-constrained value>, ...}}
```

- **`prompt`**: written deterministically from the input, so it is always an exact copy of the original request.
- **`name`**: each allowed function name is tokenized. At step *i*, the allowed tokens are the *i*-th tokens of the candidates whose first *i* tokens match what has been generated so far. The logits are masked to that set and the best one is taken. Generation stops as soon as a complete function name is formed. The model therefore picks the function, but can only ever pick an existing one. When a single candidate remains, the forward pass is skipped since the choice is forced (a small speed optimization).
- **`parameters`**: keys are taken from the function definition of the selected function and written verbatim. Each value is then generated by a generator specific to its type.

### 4. Type-specific constraints

- **Numbers** (`number_generator.py`): allowed tokens are digit tokens, a minus sign only at the first position, and a single dot. Numbers appearing in the prompt are extracted with a regex and normalized to decimal form (`16` → `16.0`). Generation stops as soon as the generated text equals one of them. The result is checked with `json.loads` before being committed.
- **Strings** (`string_generator.py`): digits, punctuation and text tokens are allowed; typographic quotes (`“ ” ‟`) and some problematic tokens are masked, and tokens that would embed a stray `"` are truncated to the part before it. Generation stops at the closing quote or after a token budget derived from the prompt length. The result is checked with `json.loads`; if it fails, the string is closed safely.
- **Regex strings** (`regex_generator.py`): a dedicated path for parameters named `regex`. Backslash parity is tracked so that escapes remain valid JSON, raw newlines are blocked, and generation is capped at 15 tokens.
- **Booleans** are currently handled through the string path (see *Known limitations*).

### 5. Custom tokenizer

`llm.py` re-implements the tokenizer from the vocabulary file alone (bonus feature):

- the GPT-2 / Qwen **byte-to-unicode** table (space → `Ġ`, newline → `Ċ`, ...);
- a Qwen-style **pre-tokenizer** regular expression written with `re` only;
- **greedy longest-match** tokenization inside each pre-token;
- recognition of **special tokens** (`<|im_start|>`, `<|im_end|>`, ...);
- `llm_decode` maps ids back to bytes and then to UTF-8.

### 6. Output writing

After each prompt, the generated text is parsed with `json.JSONDecoder().raw_decode` and the whole result list is rewritten to the output file. The file is therefore always valid JSON, even if the run is interrupted.

## Design decisions

- **Forced structure, free values.** Everything that can be known in advance (keys, braces, the prompt copy) is injected without asking the model. This is faster and removes whole classes of errors.
- **Prefix-based choice for function names.** Selecting among tokenized candidates guarantees a valid name without needing a full grammar engine.
- **Dedicated generator per type.** Numbers, free text and regexes fail in different ways, so each has its own mask and stop condition.
- **Prompt-derived stop conditions.** Using the numbers present in the prompt to decide when to stop prevents the common small-model failure of repeating digits.
- **Own tokenizer.** It removes the dependency on the SDK's `encode`/`decode` and makes the vocabulary-to-string mapping explicit, which is what constrained decoding relies on.
- **Pydantic everywhere.** Input prompts and function definitions are validated with pydantic models before any generation starts.
- **Incremental output.** Results are saved after each prompt for robustness.

## Performance analysis

- **Accuracy:** `<xx>%` correct function selection and `<xx>%` correct arguments on `<N>` test prompts (target: 90%+).
- **JSON validity:** 100% of generated objects are parseable, since the structure is enforced by masking and every value is validated before being committed.
- **Speed:** `<xx>` seconds for `<N>` prompts on `<hardware>` (target: under 5 minutes). The main cost is one forward pass per generated token; forced tokens and single-candidate choices skip the model call.
- **Reliability:** malformed or missing input files and invalid definitions produce an explicit error message rather than a crash.

### Known limitations

- Argument values still come from a 0.6B model, so ambiguous prompts can yield wrong values even though the format is always valid.
- `boolean` parameters are produced via the string generator rather than a dedicated `true`/`false` constraint.
- Nested object/array arguments are not supported.
- Token budgets (e.g. 15 tokens for regexes) can truncate very long values.

## Challenges faced

- **Tokenization mismatches.** Tokens carry a leading-space marker (`Ġ`) and byte-level encodings, so comparing text to vocabulary entries required re-implementing the byte-to-unicode mapping and the pre-tokenizer.
- **Stray quotes inside tokens.** Some tokens contain `"` combined with other characters and would close a string prematurely. These tokens are truncated or filtered, and typographic quotes are masked.
- **Repeated digits and characters.** Small models tend to loop. Stopping as soon as the output matches a number found in the prompt, and bounding the token budget, fixed most cases.
- **Escaping in regexes.** Backslashes had to be tracked by parity so that the resulting JSON string stays valid.
- **Speed.** Skipping forward passes for forced choices reduced the number of model calls.

## Testing strategy

- Run the program on the provided example files and validate the output file with `json.load` plus a schema check (keys exactly `prompt`, `name`, `parameters`; parameter names and types matching the definition).
- Edge cases: empty strings, large and negative numbers, decimals, special characters and quotes, regex-style prompts, ambiguous prompts, and functions with several parameters.
- Error handling: missing files, invalid JSON, missing keys in function definitions, empty prompts, wrong `prompt` key.
- Static checks: `make lint` (flake8 + mypy).
- Peer review of the logic, following the AI usage rules of the subject.

## Example usage

```bash
# default demo files
make run

# custom files
uv run python -m src \
    --functions_definition my_functions.json \
    --input my_prompts.json \
    --output results/out.json

# debug mode
make debug
```

Example run (the program streams the JSON as it is generated):

```
{"prompt":"Greet shrek","name":"fn_greet","parameters":{"name":"shrek"}}
```

## Resources

### References

- Subject: *call me maybe — Introduction to function calling in LLMs* (`en.subject.pdf`)
- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Byte-level BPE (GPT-2 paper, *Language Models are Unsupervised Multitask Learners*)
- Willard & Louf, *Efficient Guided Generation for LLMs* (2023): background on constrained decoding
- JSON specification: https://www.json.org
- Pydantic documentation: https://docs.pydantic.dev
- uv documentation: https://docs.astral.sh/uv/

### Use of AI

AI assistants were used for: `<e.g. discussing the constrained decoding design, explaining tokenizer internals>`.