"""Modelo 5 — lógica ML (cross-encoder reranker BGE).

Dos etapas (portado de app.py: _topk_bi + _score_pairs + match_items):
  1. el índice RxNorm (bi-encoder) trae top-K candidatos para el query;
  2. el cross-encoder (1 logit) re-puntúa cada par (query, candidate).
Devuelve los candidatos con su score bi y su score CE, ordenados por CE.
"""
from __future__ import annotations

from app import config
from app.core import rxnorm_index
from app.core.loader import LazyModel

_model = LazyModel("Model 5", config.PAIRWISE_MODEL, kind="sequence")


def status() -> dict:
    return _model.status()


def candidates(query: str, top_k: int | None = None) -> list[dict]:
    """query → candidatos RxNorm reordenados por el cross-encoder."""
    import torch

    k = top_k or config.MATCH_TOP_K
    hits = rxnorm_index.search(query, k)
    if not hits:
        return []

    tok, model, device = _model.load()
    enc = tok(
        [query] * len(hits),
        [c.get("text", "") for c in hits],
        truncation=True, max_length=config.MATCH_MAX_LEN, padding=True,
        return_tensors="pt",
    )
    enc = {key: val.to(device) for key, val in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(-1).float().cpu().tolist()
    if isinstance(logits, float):  # un solo candidato
        logits = [logits]

    rows: list[dict] = []
    for cand, ce in zip(hits, logits):
        rows.append({
            "candidate_text": cand.get("text", ""),
            "candidate_rxcui": str(cand.get("rxcui", "") or ""),
            "candidate_tty": str(cand.get("tty", "") or ""),
            "bi_score": round(float(cand.get("score") or 0.0), 4),
            "ce_score": round(float(ce), 4),
        })
    # Orden final = el del cross-encoder (mayor score = mejor match).
    rows.sort(key=lambda r: r["ce_score"], reverse=True)
    return rows
