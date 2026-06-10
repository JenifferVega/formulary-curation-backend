"""Modelo 4 — endpoints (/api/cleaner/*)."""
from fastapi import APIRouter, HTTPException

from app import config
from app.core import store
from . import service
from .schemas import CleanReq, SaveReq

router = APIRouter(prefix="/api/cleaner", tags=["cleaner"])


def _stats() -> dict:
    recs = store.read_all(config.CLEANER_OUT)
    return {"dir": str(config.CLEANER_OUT),
            "files": store.file_count(config.CLEANER_OUT),
            "total": len(recs)}


@router.post("/clean")
def clean(req: CleanReq):
    texts = [t for t in (s.strip() for s in req.texts) if t]
    if not texts:
        raise HTTPException(400, "No spans to clean.")
    try:
        items = service.clean_texts(texts)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"items": items, "count": len(items)}


@router.post("/save")
def save(req: SaveReq):
    """Guarda pares {input, output}. Dedup por input (re-editar actualiza)."""
    if not req.items:
        raise HTTPException(400, "No items to save.")
    stem = store.safe_stem(req.filename) if req.filename else "curated"
    path = config.CLEANER_OUT / f"{stem}.jsonl"

    merged: dict[str, str] = {}
    for r in store.read_file(path):
        if isinstance(r.get("input"), str) and isinstance(r.get("output"), str):
            merged[r["input"]] = r["output"]
    for it in req.items:
        if it.output.strip():  # no guardar salidas vacías
            merged[it.input] = it.output.strip()

    records = [{"input": i, "output": o} for i, o in merged.items()]
    store.write_jsonl(config.CLEANER_OUT, stem, records)
    return {"saved": True, "filename": f"{stem}.jsonl", "path": str(path),
            "added": len(req.items), "total_unique": len(records), "dataset": _stats()}


@router.get("/stats")
def stats():
    return _stats()
