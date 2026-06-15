"""Modelo 5 — endpoints (/api/matcher/*)."""
from fastapi import APIRouter, HTTPException, Query

from app import config
from app.core import store, trainer
from . import service
from .schemas import CandidatesReq, SaveReq

router = APIRouter(prefix="/api/matcher", tags=["matcher"])

_BUCKETS = {"positive", "hard_negative", "irrelevant"}


def _stats() -> dict:
    recs = store.read_all(config.MATCHER_OUT)
    pos = sum(1 for r in recs if r.get("bucket") == "positive")
    return {"dir": str(config.MATCHER_OUT), "files": store.file_count(config.MATCHER_OUT),
            "total": len(recs), "positive": pos, "negative": len(recs) - pos}


@router.post("/candidates")
def candidates(req: CandidatesReq, trained: bool = Query(False)):
    q = req.query.strip()
    if not q:
        raise HTTPException(400, "Empty query.")
    ctx = trainer.get_trained_model("matcher") if trained else None
    try:
        cands = service.candidates(q, req.top_k, _ctx=ctx)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"query": q, "candidates": cands, "count": len(cands)}


@router.post("/save")
def save(req: SaveReq):
    """Guarda pares en el esquema del trainer:
    {raw_text, query, candidate_text, candidate_rxcui, candidate_tty,
     candidate_score, label, bucket}. label = 1.0 si positive, si no 0.0.
    Dedup por (query, candidate_rxcui|candidate_text)."""
    if not req.items:
        raise HTTPException(400, "No items to save.")
    for it in req.items:
        if it.bucket not in _BUCKETS:
            raise HTTPException(400, f"Invalid bucket {it.bucket!r}.")

    stem = store.safe_stem(req.filename) if req.filename else "curated"
    path = config.MATCHER_OUT / f"{stem}.jsonl"

    def key(rec) -> str:
        return f"{rec.get('query','')}|||{rec.get('candidate_rxcui') or rec.get('candidate_text','')}"

    merged: dict[str, dict] = {}
    for r in store.read_file(path):
        merged[key(r)] = r
    for it in req.items:
        rec = {
            "raw_text": it.raw_text or it.query,
            "query": it.query,
            "candidate_text": it.candidate_text,
            "candidate_rxcui": it.candidate_rxcui,
            "candidate_tty": it.candidate_tty,
            "candidate_score": it.candidate_score,
            "label": 1.0 if it.bucket == "positive" else 0.0,
            "bucket": it.bucket,
        }
        merged[key(rec)] = rec

    records = list(merged.values())
    store.write_jsonl(config.MATCHER_OUT, stem, records)
    return {"saved": True, "filename": f"{stem}.jsonl", "path": str(path),
            "added": len(req.items), "total_unique": len(records), "dataset": _stats()}


@router.get("/stats")
def stats():
    return _stats()
