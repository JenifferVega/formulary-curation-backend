"""Gestión compartida de PDFs subidos.

Un PDF se sube una vez y lo reusan todas las secciones. Guardamos el archivo en
disco (para re-renderizar imágenes) y cacheamos el texto por página en memoria.
"""
from __future__ import annotations

import uuid

from app import config
from app.core import pdf

# upload_id -> {"filename", "path", "pages": [{"page","text"}]}
_CACHE: dict[str, dict] = {}


def save_upload(filename: str, data: bytes) -> dict:
    upload_id = uuid.uuid4().hex
    path = config.UPLOAD_DIR / f"{upload_id}.pdf"
    path.write_bytes(data)
    pages = pdf.extract_pages_text(path)
    _CACHE[upload_id] = {"filename": filename, "path": path, "pages": pages}
    return {"upload_id": upload_id, "filename": filename, "page_count": len(pages)}


def get(upload_id: str) -> dict | None:
    """Metadata cacheada; la recupera del disco si el proceso se reinició."""
    entry = _CACHE.get(upload_id)
    if entry:
        return entry
    path = config.UPLOAD_DIR / f"{upload_id}.pdf"
    if path.exists():
        entry = {"filename": path.name, "path": path,
                 "pages": pdf.extract_pages_text(path)}
        _CACHE[upload_id] = entry
        return entry
    return None


def get_pages(upload_id: str) -> list[dict] | None:
    entry = get(upload_id)
    return entry["pages"] if entry else None


def get_path(upload_id: str):
    entry = get(upload_id)
    return entry["path"] if entry else None
