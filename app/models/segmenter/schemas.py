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


class ChatReq(BaseModel):
    """Pista del revisor para corregir la segmentación de una página."""
    text: str                      # texto de la página
    spans: list[Span] = []         # medicamentos detectados actualmente
    message: str                   # instrucción en lenguaje natural


class VerifyReq(BaseModel):
    """Verificar la segmentación de la red contra una relectura del texto (LLM)."""
    text: str
    spans: list[Span] = []         # lo que segmentó la red
