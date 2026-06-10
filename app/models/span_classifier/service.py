"""Modelo 3 — lógica ML (span classifier: single / multi / broken).

DistilBERT-cased (sequence classification, 3 clases). Portado de app.py:
classify_spans. softmax sobre 3 logits; marca "possible broken" cuando la clase
top es single pero broken queda dentro del margen.
"""
from __future__ import annotations

from app import config
from app.core.loader import LazyModel

_model = LazyModel("Model 3", config.SPAN_CLASSIFIER_MODEL, kind="sequence")


def status() -> dict:
    return _model.status()


def classify_texts(texts: list[str]) -> list[dict]:
    """[str] → [{text, ai_label, label, conf, p_single, p_multi, p_broken, possible_broken}]."""
    import torch

    tok, model, device = _model.load()
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().tolist()
    preds = logits.argmax(-1).cpu().tolist()

    out: list[dict] = []
    for text, pred, prob in zip(texts, preds, probs):
        top = config.SPAN_LABELS[int(pred)]
        # Soft override: single pero broken pisándole los talones → marcar para revisar.
        possible_broken = (top == "single"
                           and (prob[0] - prob[2]) < config.POSSIBLE_BROKEN_MARGIN)
        out.append({
            "text": text,
            "ai_label": top,
            "label": top,                       # editable por el revisor
            "conf": round(prob[int(pred)], 4),
            "p_single": round(prob[0], 4),
            "p_multi": round(prob[1], 4),
            "p_broken": round(prob[2], 4),
            "possible_broken": possible_broken,
        })
    return out
