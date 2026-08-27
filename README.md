*This project has been created as part of the 42 curriculum by \<login\>.*

# Call me maybe

## Description

Given a file of function definitions and a file of natural-language prompts,
this program emits, for every prompt, a JSON object
`{"prompt": ..., "name": ..., "parameters": {...}}` saying **which function to
call and with which arguments**. Not the answer to the question: the call.
At every generation step the model's logits are masked so that only the 
tokens which keep both the JSON syntax and the function's schema valid can 
be chosen.

Default model: `Qwen/Qwen3-0.6B`. A second model, `SmolLM2-360M-Instruct`, is
wired in behind `--model` (see [multiple models](#multiple-models)).

## Instructions

```sh
make install      # uv sync: dependencies + dev tools
make run          # runs on the default paths below
make lint         # flake8 + mypy
make lint-strict  # flake8 + mypy --strict
make test         # pytest, no model and no network
make clean
```

Full form, with every path spelled out:

```sh
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input                data/input/function_calling_tests.json \
    --output               data/output/function_calling_results.json \
    --model                qwen
```

Every flag has the default shown above, so `make run` is the same command.
The output directory is created if it does not exist.

### Multiple models

`--model` takes either a short alias or any HuggingFace `repo_id`:

| alias | repo_id |
|---|---|
| `qwen` (default) | `Qwen/Qwen3-0.6B` |
| `smollm` | `HuggingFaceTB/SmolLM2-360M-Instruct` |

```sh
uv run python -m src --model smollm
uv run python -m src --model Qwen/Qwen3-1.7B
```

A model works here if it satisfies two conditions:

1. **ChatML turns** (`<|im_start|>role ... <|im_end|>`), which is what
   `prompt_generation` writes. Qwen3 and SmolLM2 share this format; the only
   difference is Qwen3's `<think></think>` block, which declares the reasoning
   phase closed and would be meaningless text on a model without it, so it is
   emitted only for Qwen.
2. **A byte-level BPE `vocab.json`**, the only format `load_vocabulary` reads.
   A model that ships only a `tokenizer.json` (SmolLM3) or a SentencePiece
   tokenizer exits with a message, not a traceback.

## Algorithm explanation

### 1. The skeleton is ours, the content is the model's

The program writes the JSON punctuation itself — `{"name": "`, `", "parameters": {`,
`"a": `, the commas, the closing braces — and asks the model **only** for the
content of one slot at a time. That is why `generate_slot` never parses the
model's output: the caller already knows which slot it is filling.

This is also the answer to the obvious peer-review objection, *"then your code
picks the function, not the model"*. No. The mask only switches off tokens that
continue **no** valid function name. Among the names still reachable, the winner
is the argmax of the model's own logits.

### 2. Masking, formally

At step `t`, with `V` the vocabulary and `V_valid ⊆ V` the tokens that keep the
constraint satisfiable, the reference formulation renormalises the model's
distribution. The numerator preserves the model's original preference; the 
denominator is the
probability mass that was originally assigned to the valid tokens, and dividing
by it makes the survivors sum to 1 again.

This project decodes **greedily**, so it works one level earlier, on the raw
logits `l`. Working in log space also means the mask is a plain additive 
`-inf`, which is exactly one numpy line (`src/decode.py`, `costringere`).

The invariant that comes with it: **`V_valid` is never empty**. An empty mask
means the automaton reached a state with no way out, and `l + m` would be `-inf`
everywhere — an argmax over that returns index 0, silently. So an empty mask
raises instead.

### 3. What makes `V_valid` hard: characters vs tokens

The grammar reasons in characters. The model reasons in tokens. The two do not
line up. One token can be `"`, or `":`, or `265`, or `"},`.
So a token is admissible only if **all** of its characters survive:

```
V_valid(s)  =  { x ∈ V : δ*(s, text(x)) ≠ ⊥ }
```

where `s` is the current slot state, `text(x)` is the token's real text, `δ*`
feeds those characters one by one, and `⊥` is the dead state.

### 4. The vocabulary is not text

`vocab.json` does not contain strings, it contains bytes remapped onto printable
characters: byte 32 (space) is stored as `Ġ`, byte 10 (newline) as `Ċ`. 
`load_vocabulary` inverts it.

## Design decisions

I designd it wathing examples on the internet

## Performance analysis

It runs fast, i optimised it.

## Performance analysis

Delirium.

## Testing strategy

Just run it man, what do you expect.

## Example usage

```console
$ make run

Prompt: Replace all vowels in 'PROGRAMMING is fun' with asterisks
Response: {"name": "fn_substitute_string_with_regex", "parameters": 
{"source_string": "PROGRAMMING is fun", "regex": "([aeiouAEIOU])", "replacement": "asterisk"}}

Prompt: Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'
Response: {"name": "fn_substitute_string_with_regex", "parameters": 
{"source_string": "The cat sat on the mat with another cat", "regex": "cat", "replacement": "dog"}}
```

## Resources

Constrained decoding:

- Michael Brenndoerfer, [*Constrained Decoding*](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output)
  - source of the renormalisation formula in [Algorithm explanation]
    (#2-masking-formally),
  and of the framing of token masking as a four-step loop.
- [*Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation*](https://arxiv.org/abs/2403.06988)
- [*Efficient Guided Generation for Large Language Models*](https://arxiv.org/pdf/2307.09702)
- [awesome-llm-constrained-decoding](https://github.com/saibo-creator/awesome-llm-constrained-decoding)

Tokenizers and models:

- GPT-2's `encoder.py` byte-level table, which `_bytes_to_unicode` rebuilds.
- [Qwen function calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)

Use of AI: large language models were used as a reviewer for the project 
and to generate test cases and drafting this README from
the code and the project notes. The design choices were
taken and are defended by the author.
