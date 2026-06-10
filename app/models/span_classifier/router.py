"""Modelo 3 — endpoints (/api/span_classifier/*)."""
from fastapi import APIRouter, HTTPException

from app import config
from app.core import store
from . import service
from .schemas import ClassifyReq, SaveReq

router = APIRouter(prefix="/api/span_classifier", tags=["span_classifier"])

_VALID = {0, 1, 2}


def _stats() -> dict:
    recs = store.read_all(config.SPAN_CLASSIFIER_OUT)
    c = {0: 0, 1: 0, 2: 0}
    for r in recs:
        if r.get("label") in c:
            c[r["label"]] += 1
    return {"dir": str(config.SPAN_CLASSIFIER_OUT),
            "files": store.file_count(config.SPAN_CLASSIFIER_OUT),
            "total": len(recs), "single": c[0], "multi": c[1], "broken": c[2]}


@router.post("/classify")
def classify(req: ClassifyReq):
    texts = [t for t in (s.strip() for s in req.texts) if t]
    if not texts:
        raise HTTPException(400, "No spans to classify.")
    try:
        items = service.classify_texts(texts)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"items": items, "count": len(items)}


@router.post("/save")
def save(req: SaveReq):
    """Guarda filas {text, label}. Dedup por texto (re-etiquetar actualiza).
    Acumula en un .jsonl por nombre (default 'curated.jsonl')."""
    if not req.items:
        raise HTTPException(400, "No items to save.")
    for it in req.items:
        if it.label not in _VALID:
            raise HTTPException(400, f"Invalid label {it.label} (must be 0/1/2).")

    stem = store.safe_stem(req.filename) if req.filename else "curated"
    path = config.SPAN_CLASSIFIER_OUT / f"{stem}.jsonl"

    # Merge por texto: lo existente + lo nuevo (nuevo gana).
    merged: dict[str, int] = {}
    for r in store.read_file(path):
        if isinstance(r.get("text"), str) and r.get("label") in _VALID:
            merged[r["text"]] = r["label"]
    for it in req.items:
        merged[it.text] = it.label

    records = [{"text": t, "label": l} for t, l in merged.items()]
    store.write_jsonl(config.SPAN_CLASSIFIER_OUT, stem, records)
    return {"saved": True, "filename": f"{stem}.jsonl", "path": str(path),
            "added": len(req.items), "total_unique": len(records), "dataset": _stats()}


@router.get("/stats")
def stats():
    return _stats()
