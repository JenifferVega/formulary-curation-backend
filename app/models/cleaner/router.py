"""Modelo 4 — endpoints (/api/cleaner/*)."""
import json
import re

from fastapi import APIRouter, HTTPException, Query

from app import config
from app.core import llm, store, trainer
from . import service
from .schemas import CleanReq, SaveReq, VerifyReq, ChatReq

router = APIRouter(prefix="/api/cleaner", tags=["cleaner"])

_VERIFY_SYSTEM = (
    "You normalize raw drug entries from US health insurance formulary pages. "
    "For each raw span, produce a clean canonical drug name: generic name + strength + dose form. "
    "Remove tier codes, quantity limits (QL), pricing, step therapy notes, and restriction symbols. "
    "Do NOT invent information not present in the input. "
    "Respond with ONLY a JSON object: "
    "{\"cleaned\": [{\"input\": \"<exact input text>\", \"output\": \"<canonical name>\"}, ...]}"
)

_CHAT_SYSTEM = (
    "You are the Drug Name Normalization Assistant for a formulary curation tool. "
    "Your job is to help a human reviewer correct how a FLAN-T5 neural network "
    "normalizes raw drug entries from US health-insurance formulary pages.\n\n"
    "NORMALIZATION RULES:\n"
    "- Output: generic name + strength + dose form (e.g. 'metformin 500 mg tablet')\n"
    "- Remove: tier codes (Tier 1/2/3), quantity limits (QL), pricing, step therapy notes, "
    "restriction symbols (*, †, PA, ST), NDC codes\n"
    "- Do NOT invent information not present in the input\n"
    "- Keep all strengths (e.g. '10 mg, 20 mg' stays as is)\n\n"
    "WHAT YOU CAN DO:\n"
    "- Fix a normalization the model got wrong ('that output is missing the strength')\n"
    "- Explain what should be removed vs kept for a specific entry\n"
    "- Review all current outputs and flag suspicious ones\n"
    "- Answer questions about normalization edge cases\n\n"
    "WHAT YOU CANNOT DO:\n"
    "- Change which spans exist — that is Model 2 (Segmenter)\n"
    "- Classify spans — that is Model 3 (Span Classifier)\n"
    "- Save data — the reviewer does that manually\n\n"
    "When the reviewer asks you to fix an output, respond with JSON:\n"
    "{\"changes\": [{\"input\": \"<exact input text>\", \"output\": \"<corrected canonical name>\"}], "
    "\"reply\": \"<one short sentence>\"}\n"
    "When answering a question or reviewing without specific fixes:\n"
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
    recs = store.read_all(config.CLEANER_OUT)
    return {"dir": str(config.CLEANER_OUT),
            "files": store.file_count(config.CLEANER_OUT),
            "total": len(recs)}


@router.post("/clean")
def clean(req: CleanReq, trained: bool = Query(False)):
    texts = [t for t in (s.strip() for s in req.texts) if t]
    if not texts:
        raise HTTPException(400, "No spans to clean.")
    ctx = trainer.get_trained_model("cleaner") if trained else None
    try:
        items = service.clean_texts(texts, _ctx=ctx)
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
        if it.output.strip():
            merged[it.input] = it.output.strip()

    records = [{"input": i, "output": o} for i, o in merged.items()]
    store.write_jsonl(config.CLEANER_OUT, stem, records)
    return {"saved": True, "filename": f"{stem}.jsonl", "path": str(path),
            "added": len(req.items), "total_unique": len(records), "dataset": _stats()}


@router.post("/verify")
def verify(req: VerifyReq):
    """LLM re-normalizes each span independently; diff against FLAN-T5 outputs."""
    if not req.items:
        raise HTTPException(400, "No items to verify.")

    listed = "\n".join(f"{i + 1}. {it.input}" for i, it in enumerate(req.items))
    user = f"Normalize each of these drug spans:\n{listed}"

    try:
        raw = llm.complete(_VERIFY_SYSTEM, user, max_tokens=4096, json_mode=True)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    try:
        data = _parse_json(raw)
        cleaned = data.get("cleaned", [])
    except (json.JSONDecodeError, ValueError, TypeError):
        raise HTTPException(502, "LLM did not return valid JSON.")

    llm_outputs: dict[str, str] = {}
    for c in cleaned:
        inp = str(c.get("input", "")).strip()
        out = str(c.get("output", "")).strip()
        if inp:
            llm_outputs[inp] = out

    matched = 0
    disagreements = []
    for it in req.items:
        llm_out = llm_outputs.get(it.input)
        if llm_out is None:
            for k, v in llm_outputs.items():
                if it.input in k or k in it.input:
                    llm_out = v
                    break
        if llm_out is None:
            continue
        if llm_out.strip().lower() == it.nn_output.strip().lower():
            matched += 1
        else:
            disagreements.append({
                "input": it.input,
                "nn_output": it.nn_output,
                "llm_output": llm_out,
            })

    return {"matched": matched, "disagreements": disagreements, "total": len(req.items)}


@router.post("/chat")
def chat(req: ChatReq):
    """Context-aware chat for drug name normalization corrections."""
    listed = "\n".join(
        f"{i+1}. IN: {it.input!r}  →  OUT: {it.nn_output!r}"
        for i, it in enumerate(req.items)
    ) or "(no items loaded)"

    first_user = f"CURRENT NORMALIZATIONS ({len(req.items)} items):\n{listed}"

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
