"""Modelo 2 — lógica ML (segmentador BIO de medicamentos).

XLM-RoBERTa (token classification, O / B-DRUG / I-DRUG). Inferencia con ventana
deslizante + decodificación BIO → spans de caracteres con confianza. Portado de
ML-parsing-api/app.py: predict_drug_spans.
"""
from __future__ import annotations

import re

from app import config
from app.core.loader import LazyModel

_model = LazyModel("Model 2", config.SEGMENTER_MODEL, kind="token")


def status() -> dict:
    return _model.status()


def predict_spans(text: str, max_len: int = 512, stride: int = 128, _ctx=None) -> list[dict]:
    """Texto de una página → [{start,end,conf,snippet}] de cada medicamento.
    Pass _ctx=(tok, model, device) to use a trained model."""
    import torch

    if _ctx is not None:
        tok, model, device = _ctx
        id2label = {int(k): v for k, v in model.config.id2label.items()}
    else:
        tok, model, device = _model.load()
        id2label = _model.id2label
    enc = tok(text, truncation=True, max_length=max_len, stride=stride,
              return_overflowing_tokens=True, return_offsets_mapping=True,
              padding=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping").tolist()
    enc.pop("overflow_to_sample_mapping", None)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs_all = torch.softmax(logits, dim=-1).cpu().tolist()
    preds_all = logits.argmax(-1).cpu().tolist()

    spans: list[list] = []
    for w_preds, w_offsets, w_probs in zip(preds_all, offsets, probs_all):
        cur: list | None = None
        for p, (s, e), token_probs in zip(w_preds, w_offsets, w_probs):
            if s == e:
                continue
            label = id2label[p]
            pred_prob = token_probs[p]
            if label == "B-DRUG":
                if cur is not None:
                    spans.append(cur)
                cur = [s, e, [pred_prob]]
            elif label == "I-DRUG" and cur is not None:
                cur[1] = e
                cur[2].append(pred_prob)
            else:
                if cur is not None:
                    spans.append(cur)
                    cur = None
        if cur is not None:
            spans.append(cur)

    spans.sort(key=lambda x: x[0])
    merged: list[list] = []
    for s, e, pr in spans:
        if merged and s < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2].extend(pr)
        else:
            merged.append([s, e, list(pr)])

    out: list[dict] = []
    for s, e, pr in merged:
        snippet = re.sub(r"\s+", " ", text[s:e]).strip()
        if snippet:
            out.append({"start": s, "end": e,
                        "conf": round(sum(pr) / len(pr), 4) if pr else 0.0,
                        "snippet": snippet})
    return out


def locate_spans(text: str, drug_strings: list[str]) -> tuple[list[dict], list[str]]:
    """Convierte textos de medicamentos (copiados textualmente por el LLM) en
    spans de caracteres, tolerante a diferencias de espacios/saltos de línea.
    Mismo enfoque que 2_build_bio_dataset. Devuelve (spans, no_encontrados)."""
    spans: list[dict] = []
    unmatched: list[str] = []
    cursor = 0
    for raw in drug_strings:
        target = (raw or "").strip()
        if not target:
            continue
        parts = re.split(r"\s+", target)
        pattern = r"\s+".join(re.escape(p) for p in parts)
        m = re.search(pattern, text[cursor:])
        if m:
            start, end = cursor + m.start(), cursor + m.end()
        else:  # fuera de orden: busca desde el inicio
            m = re.search(pattern, text)
            if not m:
                unmatched.append(target)
                continue
            start, end = m.start(), m.end()
        spans.append({"start": start, "end": end, "conf": None,
                      "snippet": re.sub(r"\s+", " ", text[start:end]).strip()})
        cursor = max(cursor, end)

    spans.sort(key=lambda x: x["start"])
    merged: list[dict] = []
    for sp in spans:
        if merged and sp["start"] < merged[-1]["end"]:
            continue  # descarta solapes
        merged.append(sp)
    return merged, unmatched
