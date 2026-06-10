"""Esquemas request/response del Modelo 1 (relevancia de página)."""
from pydantic import BaseModel


class ClassifyReq(BaseModel):
    upload_id: str
    threshold: float | None = None


class PageLabel(BaseModel):
    page: int
    text: str = ""
    label: str  # "RELEVANT" | "NOT_RELEVANT"


class SaveReq(BaseModel):
    filename: str
    pages: list[PageLabel]
