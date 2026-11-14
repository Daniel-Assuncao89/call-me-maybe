*This project has been created as part of the 42 curriculum by dassunca.*

# call me maybe

Introduction to function calling in LLMs: a program that turns a natural
language request ("What is the sum of 2 and 3?") into a structured call
(`fn_add_numbers`, `{"a": 2, "b": 3}`) using a small model
(`Qwen/Qwen3-0.6B`) and **constrained decoding**.

## Description

A 0.6B model asked to "answer in JSON" is wrong often. Here the model is
never trusted with the syntax: at every step the logits of the tokens
that would break the JSON schema are masked, so the output is valid by
construction and the model only decides the *meaning* (which function,
which values). Output: a JSON array of `{prompt, name, parameters}`.

## Instructions

Requirements: Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
make install        # uv sync
make run            # uses data/input/*.json, writes data/output/function_calls.json
make run ARGS="--functions_definition F --input P --output O"
make debug          # runs under pdb
make lint           # flake8 + mypy (mandatory flags)
make lint-strict    # flake8 + mypy --strict
make test           # pytest (no model download needed)
make clean
```

Extra options: `--trace` (print every decoding step), `--trace-html FILE`
(HTML visualisation), `--model NAME`, `--use-sdk-tokenizer`,
`--max-string-tokens N`, `--quiet`.

## Algorithm explanation

1. The prompt lists the available functions and ends with
   `Answer: {"name": "`.
2. **Function name**: candidates are the defined names; at each step only
   tokens that keep some name a valid prefix are allowed (the rest get
   `-inf`). A single remaining candidate is written without calling the
   model.
3. **Parameters**: the program itself writes the structure
   (`", "parameters": {"a": `) and only the *values* are generated, with
   one automaton per type:
   - number/integer: JSON number automaton; the model decides when to stop
     by comparing the probability mass of "stop" tokens vs "continue";
   - string: any character except raw control chars, with a valid escape
     automaton; stops when the closing quote wins;
   - boolean / enum / null: literal choice like the function name;
   - array / nested object: recursive; `[` vs `]` and `,` vs `]` are
     decided by the model among structural tokens only.
4. The result is validated against the schema; if decoding dead-ends, the
   prompt is retried in a compact form before the item is reported as
   skipped.

## Design decisions

- Own byte-level BPE tokenizer built from `vocab.json`/`merges.txt`
  (public SDK methods only); checked against the reference tokenizer
  output on a toy vocabulary. `--use-sdk-tokenizer` delegates to the SDK.
- No private attribute of `llm_sdk` is used; no forbidden library
  (`dspy`, `torch`, `transformers`, `outlines`...) is imported in `src/`.
- pydantic v2 validates every input file and every result.
- Errors are explicit exceptions (`src/errors.py`) turned into a clean
  message and exit code 1; nothing is hardcoded about specific functions.

## Performance analysis

The model is queried once per generated token; forced text (structure,
single-candidate names) costs no model call, and logits are cached by
context (LRU) because the SDK has no KV cache. Vocabulary masks are
precomputed with numpy. >90% accurate and <3min execution

## Challenges

- Tokens do not align with JSON boundaries (`","` or `":` are single
  tokens): closing quotes/brackets are queued and merged into the next
  forced text so the context tokenizes naturally.
- Deciding when a number or a string ends without letting the model run
  forever (token budgets + probability mass comparison).
- No KV cache in the SDK: caching and forced shortcuts limit the cost.

## Testing strategy

`tests/` runs without downloading the model: a toy BPE vocabulary plus a
scripted fake LLM that "wants" a given answer. It checks the tokenizer,
the JSON automata, every value type, escapes, unicode, budgets, error
recovery, the cache, I/O errors and a full CLI run. Real-model accuracy
must be measured with `make run`.

## Example usage

```bash
uv run python -m src --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Resources

- Subject: *call me maybe* (42). Qwen3 model card on Hugging Face.
- Byte-pair encoding (Sennrich et al.); GPT-2 byte-level BPE; JSON
  grammar (RFC 8259); pydantic, uv, numpy documentation.
- AI usage: code review, headge cases and to make the readme file
