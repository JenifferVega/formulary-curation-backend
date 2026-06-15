"""Modelo 1 — lógica ML (relevancia de página).

DistilBERT (sequence classification). score = softmax(logits)[1];
label = RELEVANT si score > umbral. Portado de ML-parsing-api/app.py.
"""
from __future__ import annotations

from app import config
from app.core.loader import LazyModel

_model = LazyModel("Model 1", config.RELEVANCE_MODEL, kind="sequence")


def status() -> dict:
    return _model.status()


def classify_pages(pages: list[dict], threshold: float, _ctx=None) -> list[dict]:
    """[{page,text}] → [{page,text,score,ai_label,label,too_short}].
    Pass _ctx=(tok, model, device) to use a trained model instead of the default."""
    import torch

    tok, model, device = _ctx if _ctx is not None else _model.load()
    out: list[dict] = []
    with torch.no_grad():
        for p in pages:
            text = (p.get("text") or "").strip()
            if len(text) < 20:  # el trainer también descarta textos muy cortos
                ai = config.LABEL_NOT_RELEVANT
                out.append({"page": p["page"], "text": p.get("text", ""),
                            "score": 0.0, "ai_label": ai, "label": ai, "too_short": True})
                continue
            inp = tok(text, return_tensors="pt", truncation=True, max_length=512)
            inp = {k: v.to(device) for k, v in inp.items()}
            logits = model(**inp).logits[0]
            probs = torch.softmax(logits, dim=-1)
            score = probs[1].item() if probs.shape[0] > 1 else probs[0].item()
            ai = config.LABEL_RELEVANT if score > threshold else config.LABEL_NOT_RELEVANT
            out.append({"page": p["page"], "text": p.get("text", ""),
                        "score": round(score, 4), "ai_label": ai, "label": ai,
                        "too_short": False})
    return out
