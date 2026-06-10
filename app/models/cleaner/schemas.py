"""Esquemas request/response del Modelo 4 (cleaner FLAN-T5)."""
from pydantic import BaseModel


class CleanReq(BaseModel):
    texts: list[str]               # fragmentos sucios a normalizar


class ItemPair(BaseModel):
    input: str
    output: str                    # texto canónico (corregido por el revisor)


class SaveReq(BaseModel):
    filename: str | None = None
    items: list[ItemPair]
