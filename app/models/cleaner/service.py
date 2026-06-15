"""Modelo 4 — lógica ML (cleaner FLAN-T5: normaliza un span sucio).

Seq2seq generativo. Portado de app.py: clean_items. Prefijo de instrucción +
model.generate (beams=2 → da sequences_scores para la confianza). Por lotes.
"""
from __future__ import annotations

from app import config
from app.core.loader import LazyModel

_model = LazyModel("Model 4", config.SEQ2SEQ_MODEL, kind="seq2seq")


def status() -> dict:
    return _model.status()


def clean_texts(texts: list[str], batch_size: int = 8, _ctx=None) -> list[dict]:
    """[str] → [{input, output, conf}]. output = texto canónico generado.
    Pass _ctx=(tok, model, device) to use a trained model."""
    import math
    import torch

    tok, model, device = _ctx if _ctx is not None else _model.load()
    rows: list[dict] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        prompts = [f"{config.SEQ2SEQ_INSTRUCTION} {t}" for t in batch]
        inputs = tok(prompts, return_tensors="pt", truncation=True,
                     max_length=config.SEQ2SEQ_MAX_INPUT_LEN, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=config.SEQ2SEQ_MAX_OUTPUT_LEN,
                num_beams=2,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        decoded = tok.batch_decode(output.sequences, skip_special_tokens=True)
        seq_scores = getattr(output, "sequences_scores", None)
        for i, (src, cleaned) in enumerate(zip(batch, decoded)):
            conf = math.exp(seq_scores[i].item()) if seq_scores is not None else 0.0
            rows.append({"input": src, "output": cleaned.strip(),
                         "conf": round(conf, 3)})
    return rows
