"""Listato principale per chiamata a funzione con decodifica vincolata."""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from llm_sdk import Small_LLM_Model as Model
from src.decode import (
    costringere,
    get_number_ids,
    get_prefix_ids,
    get_special_ids,
    get_stop_ids,
    get_string_ids,
    load_vocabulary,
)
from src.files_reader import Reader

MODELS = {
    "qwen": "Qwen/Qwen3-0.6B",
    "smollm": "HuggingFaceTB/SmolLM2-360M-Instruct",
}


def prompt_generation(prompt: str, funcs: list[dict[str, Any]],
                      model: str = MODELS["qwen"]) -> str:
    """Costruisce il contesto in formato ChatML.

    Qwen3 e SmolLM2 usano lo stesso formato di turni: cambia solo il
    blocco ``<think></think>``, che su Qwen3 serve a dichiarare chiuso
    il ragionamento e su un modello che non ne ha sarebbe testo a caso.
    """
    tools = "\n".join(json.dumps(f) for f in funcs)
    think = "<think>\n\n</think>\n\n" if "qwen" in model.lower() else ""
    return f"""<|im_start|>system
# Tools
You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and parameters
 within <tool_call></tool_call> XML tags:
<tool_call>""" + """{"name": <function-name>, "parameters": <args-json-object>}
</tool_call><|im_end|>""" + f"""
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{think}<tool_call>
"""


class Vincoli:
    """Sottoinsiemi di id che non dipendono da cosa è già generato."""

    def __init__(self, vocab: dict[int, str]) -> None:
        """Inizializza."""
        self.vocab = vocab
        self.specials = get_special_ids(vocab)
        self.strings = get_string_ids(vocab, exclude=self.specials)
        self.strings_start = [i for i in self.strings
                              if vocab[i] == " "
                              or not vocab[i].startswith(" ")]
        self.quote = get_stop_ids(vocab, '"')
        self.stop_number = get_stop_ids(vocab, ',}" ')


def slot_kind(declared: str) -> str:
    """Traduce il tipo dichiarato nello schema nel kind dello slot."""
    kind = "boolean" if declared == "bool" else declared
    if kind not in ("string", "number", "integer", "boolean"):
        print(f"attenzione: tipo '{declared}' sconosciuto, "
              "trattato come 'string'", file=sys.stderr)
        return "string"
    return kind


def slot_ids(kind: str, current: str, v: Vincoli,
             candidates: list[str] | None = None
             ) -> tuple[list[int], list[int]]:
    """Id che allungano ``current`` e id che chiudono lo slot ``kind``."""
    if kind == "boolean":
        candidates = ["true", "false"]
    if candidates is not None:
        closers = v.quote if kind == "name" else v.stop_number
        return (get_prefix_ids(v.vocab, candidates, current),
                closers if current in candidates else [])
    if kind in ("number", "integer"):
        digits = any(c.isdigit() for c in current)
        return (get_number_ids(v.vocab, current, kind == "integer"),
                v.stop_number if digits else [])
    if not current:
        return v.strings_start, []
    return v.strings, v.quote


def emit(text: str, chunk: str, stream: bool) -> str:
    """Aggiunge ``chunk`` al contesto e, se richiesto, lo mostra a schermo."""
    if stream:
        print(chunk, end='', flush=True)
    return text + chunk


def generate_slot(text: str, kind: str, llm: Model, v: Vincoli,
                  candidates: list[str] | None = None,
                  stream: bool = False) -> str:
    """Genera un solo campo, token per token, restando dentro il vincolo."""
    current = ""
    limit = 42
    for _ in range(limit):
        allowed, stop = slot_ids(kind, current, v, candidates)
        if not allowed:
            return current
        token_ids = llm.encode(text + current)[0].tolist()
        logits = llm.get_logits_from_input_ids(token_ids)
        next_token = costringere(logits, allowed + stop)
        if next_token in stop:
            return current
        current = emit(current, v.vocab[next_token], stream)
    print(f"attenzione: slot '{kind}' troncato a {limit} token: "
          f"{current!r}", file=sys.stderr)
    return current


def coerce(kind: str, raw: str) -> Any:
    """Converte il testo grezzo di uno slot nel tipo dichiarato."""
    if kind == "integer":
        try:
            return int(raw)
        except ValueError:
            return 0
    if kind == "number":
        try:
            value = float(raw)
        except ValueError:
            return 0.0
        return value if math.isfinite(value) else 0.0
    if kind == "boolean":
        return raw.strip().lower() == "true"
    return raw


def choose_function(text: str, funcs_list: list[dict[str, Any]], llm: Model,
                    v: Vincoli, stream: bool) -> tuple[dict[str, Any], str]:
    """Fa scegliere al modello la funzione da chiamare."""
    names: list[str] = [f["name"] for f in funcs_list]
    if not names:
        raise ValueError("nessuna funzione disponibile in "
                         "functions_definition")
    text = emit(text, '{"name": "', stream)
    name = generate_slot(text, "name", llm, v, names, stream=stream)
    if name not in names:
        print(f"attenzione: nome fuori dai candidati ({name!r}), "
              f"uso {names[0]!r}", file=sys.stderr)
        name = names[0]
    text = emit(text + name, '", "parameters": {', stream)
    return next(f for f in funcs_list if f["name"] == name), text


def generate_parameter(text: str, pname: str, spec: dict[str, Any],
                       llm: Model, v: Vincoli,
                       stream: bool) -> tuple[Any, str]:
    """Genera il valore di un parametro e lo aggiunge al contesto."""
    kind = slot_kind(str(spec.get("type", "string")))
    text = emit(text, f'"{pname}": ', stream)
    if kind == "string":
        text = emit(text, '"', stream)
    raw = generate_slot(text, kind, llm, v, stream=stream)
    value = coerce(kind, raw)
    if kind == "string":
        return value, emit(text + raw, '"', stream)
    return value, text + json.dumps(value)


def generate_answer(user_prompt: str, context: str, llm: Model,
                    funcs_list: list[dict[str, Any]], v: Vincoli,
                    stream: bool = False) -> dict[str, Any]:
    """Assembla la chiamata a funzione uno slot alla volta."""
    func, text = choose_function(context, funcs_list, llm, v, stream)
    parameters: dict[str, Any] = {}
    for i, (pname, spec) in enumerate(func["parameters"].items()):
        if i:
            text = emit(text, ", ", stream)
        value, text = generate_parameter(text, pname, spec, llm, v, stream)
        parameters[pname] = value
    emit(text, "}}", stream)
    return {"prompt": user_prompt, "name": func["name"],
            "parameters": parameters}


def parse_args() -> argparse.Namespace:
    """Legge le opzioni da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Function calling con decodifica vincolata (src).")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="file JSON con le definizioni delle funzioni "
             "(default: %(default)s)")
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json",
        help="file JSON con i prompt da elaborare (default: %(default)s)")
    parser.add_argument(
        "--output", default="data/output/function_calling_results.json",
        help="file JSON di output (default: %(default)s)")
    parser.add_argument(
        "--model", default="qwen",
        help=f"alias fra {', '.join(MODELS)}, oppure un repo_id "
             "HuggingFace qualsiasi che esponga un vocab.json "
             "byte-level BPE (default: %(default)s)")
    return parser.parse_args()


def load_model(repo_id: str) -> Model:
    """Carica il modello, o esce con un messaggio su stderr."""
    try:
        return Model(model_name=repo_id)
    except Exception as e:
        print(f"errore nel caricamento del modello '{repo_id}': {e}",
              file=sys.stderr)
        sys.exit(1)


def vocab_path(llm: Model, repo_id: str) -> str:
    """Percorso locale del `vocab.json` del modello."""
    try:
        return llm.get_path_to_vocab_file()
    except EntryNotFoundError:
        return hf_hub_download(repo_id=repo_id, filename="vocab.json")


def load_vocab(llm: Model, repo_id: str) -> dict[int, str]:
    """Scarica e decodifica il vocabolario, o esce su stderr."""
    try:
        return load_vocabulary(vocab_path(llm, repo_id))
    except Exception as e:
        print(f"errore nel caricamento del vocabolario di '{repo_id}': "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


def run_prompts(prompts: list[str], funcs_list: list[dict[str, Any]],
                llm: Model, v: Vincoli,
                model: str = MODELS["qwen"]) -> list[dict[str, Any]]:
    """Elabora i prompt uno per uno e ritorna le chiamate riuscite."""
    answers: list[dict[str, Any]] = []
    for prompt in prompts:
        print(f"\nPrompt: {prompt}\nRisposta: ", end="", flush=True)
        context = prompt_generation(prompt, funcs_list, model)
        try:
            call = generate_answer(prompt, context, llm, funcs_list, v,
                                   stream=True)
        except Exception as e:
            print(f"\nprompt saltato ({prompt!r}): {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        print()
        answers.append(call)
    return answers


def write_output(path_name: str, answers: list[dict[str, Any]]) -> None:
    """Scrive i risultati, creando la cartella se manca."""
    path = Path(path_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(answers, indent=2))
    except OSError as e:
        print(f"errore nella scrittura di '{path}': {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Scritti {len(answers)} risultati in {path}")


def main() -> None:
    """Elabora tutti i prompt del file di input e scrive i risultati."""
    args = parse_args()
    reader = Reader(args.input, args.functions_definition)
    prompts = reader.get_valid_p()
    funcs_list = [{"name": f.name, "description": f.description,
                   "parameters": f.parameters, "returns": f.returns}
                  for f in reader.get_valid_f().funcs]

    repo_id = MODELS.get(args.model, args.model)
    llm = load_model(repo_id)
    v = Vincoli(load_vocab(llm, repo_id))

    answers = run_prompts([p.prompt for p in prompts.tests], funcs_list,
                          llm, v, repo_id)
    write_output(args.output, answers)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("EHHHH VOLEVI!")
        try:
            with open(Path(__file__).with_name("volevi.txt"), 'r') as f:
                print(f.read())
        except OSError:
            pass
    except Exception as e:
        print(f"errore inatteso: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
