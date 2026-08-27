"""Lettura e validazione dei due file JSON di input del programma.

`Reader` è la porta d'ingresso: se un file manca, non è JSON valido o
non passa i modelli pydantic di `src.validators`, il programma esce
con un messaggio chiaro su stderr invece di proseguire su dati rotti.
"""
import json
import sys
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.validators import FuncsValidator, TestsValidator

ModelT = TypeVar("ModelT", bound=BaseModel)


class Reader:
    """Carica e valida i file di prompt e di definizioni di funzione."""

    def __init__(self, path_p: str, path_f: str) -> None:
        """Legge subito entrambi i file JSON (fallisce presto)."""
        self.path_p = path_p
        self.path_f = path_f
        self.prompts_obj = self._read_json(path_p)
        self.funcs_obj = self._read_json(path_f)

    @staticmethod
    def _read_json(path: str) -> Any:
        """Legge un file JSON, o esce con un messaggio chiaro."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Errore nella lettura di '{path}': {e}", file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def _validate(model: type[ModelT], path: str, **data: Any) -> ModelT:
        """Valida `data` contro `model`, o esce nominando `path`."""
        try:
            validated = model(**data)
        except ValidationError as e:
            print(f"Errore di validazione in '{path}': {e}",
                  file=sys.stderr)
            sys.exit(1)
        return validated

    def get_valid_p(self) -> TestsValidator:
        """Ritorna i prompt validati (campo `tests`)."""
        return self._validate(TestsValidator, self.path_p,
                              tests=self.prompts_obj)

    def get_valid_f(self) -> FuncsValidator:
        """Ritorna le funzioni validate (campo `funcs`)."""
        return self._validate(FuncsValidator, self.path_f,
                              funcs=self.funcs_obj)
