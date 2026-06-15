"""Esquemas request/response del Modelo 3 (span classifier: single/multi/broken)."""
from pydantic import BaseModel


class ClassifyReq(BaseModel):
    texts: list[str]               # fragmentos a clasificar


class ItemLabel(BaseModel):
    text: str
    label: int                     # 0=single, 1=multi, 2=broken


class SaveReq(BaseModel):
    filename: str | None = None
    items: list[ItemLabel]


class VerifyItem(BaseModel):
    text: str
    nn_label: str                  # "single" | "multi" | "broken"


class VerifyReq(BaseModel):
    items: list[VerifyItem]


class HistoryMessage(BaseModel):
    role: str                      # "user" | "assistant"
    content: str


class ChatReq(BaseModel):
    items: list[VerifyItem]        # current classified spans
    message: str
    history: list[HistoryMessage] = []
