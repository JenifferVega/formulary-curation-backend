"""Modelo 3 — endpoints (/api/span_classifier/*)."""
import json
import re

from fastapi import APIRouter, HTTPException, Query

from app import config
from app.core import llm, store, trainer
from . import service
from .schemas import ClassifyReq, SaveReq, VerifyReq, ChatReq

router = APIRouter(prefix="/api/span_classifier", tags=["span_classifier"])

_VALID = {0, 1, 2}

_VERIFY_SYSTEM = (
    "You classify drug entries from US health insurance formulary pages. "
    "For each span, classify it as exactly one of:\n"
    "- \"single\": one complete drug entry (one drug name + strength + dose form)\n"
    "- \"multi\": two or more drugs combined in one entry that needs splitting\n"
    "- \"broken\": incomplete, corrupted, or non-drug text that should be discarded\n"
    "Respond with ONLY a JSON object: "
    "{\"classifications\": [{\"text\": \"<exact input text>\", "
    "\"label\": \"single\"|\"multi\"|\"broken\"}, ...]}"
)

_CHAT_SYSTEM = (
    "You are the Span Classification Assistant for a formulary curation tool. "
    "Your job is to help a human reviewer correct how drug spans from a US health-insurance "
    "formulary are classified by a DistilBERT neural network.\n\n"
    "CLASSIFICATIONS:\n"
    "- \"single\": one complete drug entry (name + strength + dose form)\n"
    "- \"multi\": two or more drugs merged into one span that needs splitting\n"
    "- \"broken\": incomplete, corrupted, or non-drug text that should be discarded\n\n"
    "WHAT YOU CAN DO:\n"
    "- Change a classification the model got wrong ('that is not broken, it is single')\n"
    "- Explain why a span should be multi vs single\n"
    "- Review all current labels and flag suspicious ones\n"
    "- Answer questions about how to classify edge cases\n\n"
    "WHAT YOU CANNOT DO:\n"
    "- Add or remove spans — that is Model 2 (Segmenter)\n"
    "- Normalize drug names — that is Model 4 (Cleaner)\n"
    "- Save data — the reviewer does that manually\n\n"
    "When the reviewer asks you to fix a label, respond with JSON:\n"
    "{\"changes\": [{\"text\": \"<exact span text>\", \"label\": \"single\"|\"multi\"|\"broken\"}], "
    "\"reply\": \"<one short sentence>\"}\n"
    "When answering a question or reviewing without specific changes:\n"
    "{\"changes\": [], \"reply\": \"<your explanation>\"}"
)


def _parse_json(raw: str) -> dict:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1:
        s = s[start:end + 1]
    return json.loads(s)


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
def classify(req: ClassifyReq, trained: bool = Query(False)):
    texts = [t for t in (s.strip() for s in req.texts) if t]
    if not texts:
        raise HTTPException(400, "No spans to classify.")
    ctx = trainer.get_trained_model("span_classifier") if trained else None
    try:
        items = service.classify_texts(texts, _ctx=ctx)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"items": items, "count": len(items)}


@router.post("/save")
def save(req: SaveReq):
    """Guarda filas {text, label}. Dedup por texto (re-etiquetar actualiza)."""
    if not req.items:
        raise HTTPException(400, "No items to save.")
    for it in req.items:
        if it.label not in _VALID:
            raise HTTPException(400, f"Invalid label {it.label!r} — must be 0, 1, or 2.")
    stem = store.safe_stem(req.filename) if req.filename else "curated"
    path = config.SPAN_CLASSIFIER_OUT / f"{stem}.jsonl"

    existing: dict[str, int] = {}
    for r in store.read_file(path):
        if isinstance(r.get("text"), str) and isinstance(r.get("label"), int):
            existing[r["text"]] = r["label"]
    for it in req.items:
        existing[it.text] = it.label

    records = [{"text": t, "label": l} for t, l in existing.items()]
    store.write_jsonl(config.SPAN_CLASSIFIER_OUT, stem, records)
    return {"saved": True, "filename": f"{stem}.jsonl", "path": str(path),
            "added": len(req.items), "total_unique": len(records), "dataset": _stats()}


@router.post("/verify")
def verify(req: VerifyReq):
    """LLM re-classifies each span independently; diff against NN predictions."""
    if not req.items:
        raise HTTPException(400, "No items to verify.")

    listed = "\n".join(f"{i + 1}. {it.text}" for i, it in enumerate(req.items))
    user = f"Classify each of these drug spans:\n{listed}"

    try:
        raw = llm.complete(_VERIFY_SYSTEM, user, max_tokens=4096, json_mode=True)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    try:
        data = _parse_json(raw)
        classifications = data.get("classifications", [])
    except (json.JSONDecodeError, ValueError, TypeError):
        raise HTTPException(502, "LLM did not return valid JSON.")

    llm_labels: dict[str, str] = {}
    for c in classifications:
        text = str(c.get("text", "")).strip()
        label = str(c.get("label", "")).strip().lower()
        if text and label in ("single", "multi", "broken"):
            llm_labels[text] = label

    matched = 0
    disagreements = []
    for it in req.items:
        llm_label = llm_labels.get(it.text)
        if llm_label is None:
            for k, v in llm_labels.items():
                if it.text in k or k in it.text:
                    llm_label = v
                    break
        if llm_label is None:
            continue
        if llm_label == it.nn_label:
            matched += 1
        else:
            disagreements.append({
                "text": it.text,
                "nn_label": it.nn_label,
                "llm_label": llm_label,
            })

    return {"matched": matched, "disagreements": disagreements, "total": len(req.items)}


@router.post("/chat")
def chat(req: ChatReq):
    """Context-aware chat for span classification corrections."""
    listed = "\n".join(
        f"{i+1}. [{it.nn_label}] {it.text}"
        for i, it in enumerate(req.items)
    ) or "(no spans loaded)"

    first_user = f"CURRENT SPAN CLASSIFICATIONS ({len(req.items)} spans):\n{listed}"

    messages: list[dict] = []
    if req.history:
        messages.append({"role": "user",
                         "content": f"{first_user}\n\nREVIEWER: {req.history[0].content}"})
        for h in req.history[1:]:
            messages.append({"role": h.role, "content": h.content})
        messages.append({"role": "user", "content": req.message})
    else:
        messages.append({"role": "user",
                         "content": f"{first_user}\n\nREVIEWER: {req.message}"})

    try:
        raw = llm.complete_chat(_CHAT_SYSTEM, messages, max_tokens=2048)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    try:
        data = _parse_json(raw)
        changes = data.get("changes", [])
        reply = str(data.get("reply", "")).strip()
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"changes": [], "reply": raw.strip()[:500]}

    return {"changes": changes, "reply": reply}


@router.get("/stats")
def stats():
    return _stats()
