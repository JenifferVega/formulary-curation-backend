"""Esquemas request/response del Modelo 3 (span classifier: single/multi/broken)."""
from pydantic import BaseModel


class ClassifyReq(BaseModel):
    texts: list[str]               # fragmentos a clasificar


class ItemLabel(BaseModel):
    text: str
    label: int                     # 0=single, 1=multi, 2=broken


class SaveReq(BaseModel):
    filename: str | None = None    # opcional (los spans pueden venir de varias fuentes)
    items: list[ItemLabel]
