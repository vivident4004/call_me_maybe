"""Listato con le funzioni di decodifica a seconda del tipo."""

import json
import re

import numpy as np

STR_FORBIDDEN = set('"\\\n\r\t')
_NUMBER_PREFIX = re.compile(r"-?\d*|-?\d+\.\d*")
_INTEGER_PREFIX = re.compile(r"-?\d*")
_SPECIAL_TOKEN = re.compile(r"<\|.+\|>")


def _bytes_to_unicode() -> dict[int, str]:
    """Tabella byte -> carattere usata dai tokenizer byte-level BPE."""
    table: dict[int, str] = {}
    spare = 256
    for b in range(256):
        char = chr(b)
        if char.isprintable() and not char.isspace():
            table[b] = char
        else:
            table[b] = chr(spare)
            spare += 1
    return table


def load_vocabulary(path: str) -> dict[int, str]:
    """Legge vocab.json e ritorna {token_id: testo reale del token}."""
    with open(path) as f:
        raw: dict[str, int] = json.load(f)

    unicode_to_byte = {v: k for k, v in _bytes_to_unicode().items()}
    vocab: dict[int, str] = {}
    for token, token_id in raw.items():
        try:
            text = bytes(unicode_to_byte[c] for c in token).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue
        vocab[token_id] = text
    return vocab


def _is_number_prefix(s: str, integer_only: bool) -> bool:
    """True se ``s`` è ancora un prefisso valido di un numero JSON."""
    pattern = _INTEGER_PREFIX if integer_only else _NUMBER_PREFIX
    return pattern.fullmatch(s) is not None


def get_number_ids(vocab: dict[int, str], current: str = "",
                   integer_only: bool = False) -> list[int]:
    """Token che estendono ``current`` mantenendolo un numero JSON valido."""
    return [i for i, t in vocab.items()
            if t and _is_number_prefix(current + t, integer_only)]


def get_special_ids(vocab: dict[int, str]) -> set[int]:
    """Id dei token di controllo del modello (``<|...|>``)."""
    return {i for i, t in vocab.items() if _SPECIAL_TOKEN.fullmatch(t)}


def get_string_ids(vocab: dict[int, str],
                   exclude: set[int] | None = None) -> list[int]:
    """Id dei token ammessi dentro una stringa JSON."""
    skip = exclude if exclude is not None else set()
    return [i for i, t in vocab.items()
            if t and not (set(t) & STR_FORBIDDEN) and i not in skip]


def get_stop_ids(vocab: dict[int, str], closers: str) -> list[int]:
    """Token che iniziano con uno dei caratteri di chiusura."""
    return [i for i, t in vocab.items() if t and t[0] in closers]


def get_prefix_ids(vocab: dict[int, str], candidates: list[str],
                   current: str) -> list[int]:
    """Token che mantengono `current` prefisso di almeno un candidato."""
    ids = []
    for token_id, token in vocab.items():
        if not token:
            continue
        if any(c.startswith(current + token) for c in candidates):
            ids.append(token_id)
    return ids


def costringere(logits: list[float], allowed_ids: list[int]) -> int:
    """Azzera tutto tranne allowed_ids e ritorna il token vincitore."""
    scores = np.asarray(logits, dtype=np.float64)
    ids = np.asarray(allowed_ids, dtype=np.intp)
    ids = ids[ids < scores.size]
    if not ids.size:
        raise ValueError("maschera vuota: nessun token ammesso")
    mask = np.full_like(scores, -np.inf)
    mask[ids] = 0.0
    return int(np.argmax(scores + mask))
