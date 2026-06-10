"""Modelo 1 — endpoints (/api/relevance/*)."""
from fastapi import APIRouter, HTTPException

from app import config
from app.core import pdf, store, uploads
from . import service
from .schemas import ClassifyReq, SaveReq

router = APIRouter(prefix="/api/relevance", tags=["relevance"])


def _stats() -> dict:
    recs = store.read_all(config.RELEVANCE_OUT)
    rel = sum(1 for r in recs if r.get("label") == config.LABEL_RELEVANT)
    return {"dir": str(config.RELEVANCE_OUT), "files": store.file_count(config.RELEVANCE_OUT),
            "total_pages": len(recs), "relevant": rel, "not_relevant": len(recs) - rel}


@router.post("/classify")
def classify(req: ClassifyReq):
    entry = uploads.get(req.upload_id)
    if not entry:
        raise HTTPException(404, "Upload not found (re-upload the PDF).")
    thr = config.DEFAULT_THRESHOLD if req.threshold is None else float(req.threshold)
    try:
        rows = service.classify_pages(entry["pages"], thr)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    for row in rows:
        row["thumbnail"] = pdf.render_page_data_uri(entry["path"], row["page"], config.THUMB_DPI)
    n_rel = sum(1 for r in rows if r["label"] == config.LABEL_RELEVANT)
    return {"upload_id": req.upload_id, "filename": entry["filename"],
            "page_count": len(rows), "threshold": thr,
            "relevant": n_rel, "not_relevant": len(rows) - n_rel, "pages": rows}


@router.post("/relevant-pages")
def relevant_pages(req: ClassifyReq):
    """Respaldo (b) para la cola del segmentador: corre el Modelo 1 y devuelve
    solo los números de página relevantes (sin miniaturas → más liviano)."""
    entry = uploads.get(req.upload_id)
    if not entry:
        raise HTTPException(404, "Upload not found (re-upload the PDF).")
    thr = config.DEFAULT_THRESHOLD if req.threshold is None else float(req.threshold)
    try:
        rows = service.classify_pages(entry["pages"], thr)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    pages = [r["page"] for r in rows if r["label"] == config.LABEL_RELEVANT]
    return {"pages": pages, "threshold": thr, "total": len(entry["pages"])}


@router.post("/save")
def save(req: SaveReq):
    if not req.pages:
        raise HTTPException(400, "No pages to save.")
    records = []
    for p in req.pages:
        if p.label not in (config.LABEL_RELEVANT, config.LABEL_NOT_RELEVANT):
            raise HTTPException(400, f"Invalid label {p.label!r} on page {p.page}.")
        records.append({"page": int(p.page), "label": p.label, "text": p.text})
    stem = store.safe_stem(req.filename)
    path = store.write_jsonl(config.RELEVANCE_OUT, stem, records)
    n_rel = sum(1 for r in records if r["label"] == config.LABEL_RELEVANT)
    return {"saved": True, "filename": f"{stem}.jsonl", "path": str(path),
            "total": len(records), "relevant": n_rel,
            "not_relevant": len(records) - n_rel, "dataset": _stats()}


@router.get("/stats")
def stats():
    return _stats()
