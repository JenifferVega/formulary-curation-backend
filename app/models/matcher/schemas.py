"""Esquemas request/response del Modelo 5 (cross-encoder reranker RxNorm)."""
from pydantic import BaseModel


class CandidatesReq(BaseModel):
    query: str                         # texto del medicamento (idealmente ya limpio)
    top_k: int | None = None


class Pair(BaseModel):
    query: str
    candidate_text: str
    candidate_rxcui: str = ""
    candidate_tty: str = ""
    candidate_score: float | None = None   # score del bi-encoder
    bucket: str = "irrelevant"             # positive | hard_negative | irrelevant
    raw_text: str = ""


class SaveReq(BaseModel):
    filename: str | None = None
    items: list[Pair]
