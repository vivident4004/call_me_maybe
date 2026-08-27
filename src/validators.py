"""Modelli pydantic per validare i due file JSON di input."""
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


class PromptValidator(BaseModel):
    """Un singolo prompt da elaborare."""

    prompt: str


class TestsValidator(BaseModel):
    """La lista di prompt del file di input (`tests`)."""

    tests: list[PromptValidator]


class FunctionValidator(BaseModel):
    """Definizione di funzione: nome, descrizione, parametri, ritorno."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    returns: dict[str, Any]

    @field_validator("parameters")
    @classmethod
    def _parameters_have_type(
        cls, parameters: dict[str, dict[str, Any]], info: ValidationInfo
    ) -> dict[str, dict[str, Any]]:
        fname = info.data.get("name", "?")
        for pname, spec in parameters.items():
            if not isinstance(spec.get("type"), str):
                raise TypeError(
                    f'funzione "{fname}", parametro "{pname}": '
                    'manca la chiave "type" (stringa)'
                )
        return parameters


class FuncsValidator(BaseModel):
    """La lista di funzioni del file di definizioni (`funcs`)."""

    funcs: list[FunctionValidator]
