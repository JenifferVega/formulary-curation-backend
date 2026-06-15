"""Esquemas request/response del Modelo 2 (segmentador BIO)."""
from pydantic import BaseModel


class SegmentReq(BaseModel):
    upload_id: str
    page: int


class Span(BaseModel):
    start: int
    end: int


class SaveReq(BaseModel):
    filename: str
    page: int
    text: str
    spans: list[Span]


class HistoryMessage(BaseModel):
    role: str        # "user" | "assistant"
    content: str


class ChatReq(BaseModel):
    text: str
    spans: list[Span] = []
    message: str
    history: list[HistoryMessage] = []   # previous turns for context resolver


class VerifyReq(BaseModel):
    text: str
    spans: list[Span] = []
