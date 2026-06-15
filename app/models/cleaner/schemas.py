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


class VerifyItem(BaseModel):
    input: str
    nn_output: str                 # FLAN-T5 output


class VerifyReq(BaseModel):
    items: list[VerifyItem]


class HistoryMessage(BaseModel):
    role: str                      # "user" | "assistant"
    content: str


class ChatReq(BaseModel):
    items: list[VerifyItem]        # current cleaned items {input, nn_output}
    message: str
    history: list[HistoryMessage] = []
