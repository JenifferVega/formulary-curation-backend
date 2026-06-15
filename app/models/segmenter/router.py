"""Modelo 2 — endpoints (/api/segmenter/*)."""
import json
import re

from fastapi import APIRouter, HTTPException, Query

from app import config
from app.core import llm, store, trainer, uploads
from . import service
from .schemas import ChatReq, SaveReq, SegmentReq, VerifyReq

router = APIRouter(prefix="/api/segmenter", tags=["segmenter"])

# ── Personality for Model 2 chat ─────────────────────────────────────────────
_CHAT_SYSTEM = (
    "You are the Drug Segmentation Assistant for a formulary curation tool. "
    "Your job is to help a human reviewer correct which text spans on a US health-insurance "
    "formulary page are drug entries detected by a BIO neural network (XLM-RoBERTa).\n\n"
    "WHAT YOU CAN DO:\n"
    "- Add a drug entry the model missed (reviewer says 'you missed X', 'add Y')\n"
    "- Remove a false positive (reviewer says 'that header is not a drug', 'remove Z')\n"
    "- Fix a truncated or over-extended span\n"
    "- Review the whole page and report what's missing or extra ('what am I missing?', 'check this')\n\n"
    "WHAT YOU CANNOT DO:\n"
    "- Change tabs or work on other models\n"
    "- Classify spans (single/multi/broken) — that is Model 3\n"
    "- Normalize drug names — that is Model 4\n"
    "- Save data — the reviewer does that manually\n\n"
    "RULES:\n"
    "Every drug string you return must be an EXACT verbatim substring of the page text "
    "(same words, same spacing, full row including strength/form/restrictions). "
    "Never paraphrase or invent. Exclude category headers, column headers, footers, "
    "phone numbers, and legends.\n\n"
    "Decide the action:\n"
    "- \"review\": reviewer asks to check/verify in general, no specific drug named to change\n"
    "- \"correct\": reviewer gives a specific instruction to add, remove, or fix an entry\n\n"
    "Respond with ONLY a JSON object: "
    '{"action": "review"|"correct", "drugs": ["<verbatim entry>", ...], '
    '"reply": "<one short sentence>"}'
)

_VERIFY_SYSTEM = (
    "You read US health-insurance formulary pages and list every drug entry. "
    "Given the RAW TEXT of one page, return EVERY drug entry, each copied VERBATIM "
    "(an exact substring of the page text, including its strengths, form and any "
    "restriction/QL on the same row). Do NOT include category headers, column headers, "
    "page footers, phone numbers or legends. "
    'Respond with ONLY a JSON object: {"drugs": ["<verbatim entry>", ...]}'
)


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and a1 > b0


def _diff(text: str, llm_spans: list[dict], current: list[tuple[int, int]]) -> dict:
    missing = [
        ls for ls in llm_spans
        if not any(_overlaps(ls["start"], ls["end"], c0, c1) for c0, c1 in current)
    ]
    extra = []
    for c0, c1 in current:
        if not any(_overlaps(c0, c1, ls["start"], ls["end"]) for ls in llm_spans):
            extra.append({"start": c0, "end": c1,
                          "snippet": re.sub(r"\s+", " ", text[c0:c1]).strip()})
    return {"missing": missing, "extra": extra, "matched": len(current) - len(extra),
            "total_llm": len(llm_spans), "total_current": len(current)}


def _parse_json(raw: str) -> dict:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1:
        s = s[start:end + 1]
    return json.loads(s)


def _page_text(upload_id: str, page: int) -> str | None:
    pages = uploads.get_pages(upload_id)
    if pages is None:
        return None
    for p in pages:
        if p["page"] == page:
            return p.get("text", "")
    return None


def _stats() -> dict:
    recs = store.read_all(config.SEGMENTER_OUT)
    total_spans = sum(len(r.get("spans", [])) for r in recs)
    return {"dir": str(config.SEGMENTER_OUT), "files": store.file_count(config.SEGMENTER_OUT),
            "pages": len(recs), "spans": total_spans}


@router.post("/segment")
def segment(req: SegmentReq, trained: bool = Query(False)):
    text = _page_text(req.upload_id, req.page)
    if text is None:
        raise HTTPException(404, "Upload/page not found (re-upload the PDF).")
    if not text.strip():
        return {"page": req.page, "text": text, "spans": []}
    ctx = trainer.get_trained_model("segmenter") if trained else None
    try:
        spans = service.predict_spans(text, _ctx=ctx)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"page": req.page, "text": text, "spans": spans}


@router.post("/save")
def save(req: SaveReq):
    stem = store.safe_stem(req.filename)
    path = config.SEGMENTER_OUT / f"{stem}.jsonl"
    existing = [r for r in store.read_file(path) if r.get("source_page") != int(req.page)]
    spans = sorted([[int(s.start), int(s.end)] for s in req.spans], key=lambda x: x[0])
    existing.append({"source_file": stem, "source_page": int(req.page),
                     "text": req.text, "spans": spans})
    existing.sort(key=lambda r: r.get("source_page", 0))
    store.write_jsonl(config.SEGMENTER_OUT, stem, existing)
    return {"saved": True, "filename": f"{stem}.jsonl", "path": str(path),
            "page": int(req.page), "spans": len(spans), "dataset": _stats()}


@router.post("/chat")
def chat(req: ChatReq):
    """Context-aware chat: builds multi-turn history so the LLM remembers
    previous corrections in the same session."""
    current = [
        re.sub(r"\s+", " ", req.text[s.start:s.end]).strip()
        for s in req.spans
    ]
    listed = "\n".join(f"{i+1}. {c}" for i, c in enumerate(current)) or "(none detected yet)"

    # First user turn includes the full page context
    first_user = (
        f"PAGE TEXT:\n---\n{req.text}\n---\n"
        f"CURRENTLY DETECTED DRUG ENTRIES ({len(current)}):\n{listed}"
    )

    # Build message list: inject page context into the first user message
    messages: list[dict] = []
    if req.history:
        # Prepend page context to the oldest user message
        first = req.history[0]
        messages.append({"role": "user",
                         "content": f"{first_user}\n\nREVIEWER: {first.content}"})
        for h in req.history[1:]:
            messages.append({"role": h.role, "content": h.content})
        messages.append({"role": "user", "content": req.message})
    else:
        messages.append({"role": "user",
                         "content": f"{first_user}\n\nREVIEWER: {req.message}\n\nReturn the corrected full list as JSON."})

    try:
        raw = llm.complete_chat(_CHAT_SYSTEM, messages, max_tokens=8192)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    try:
        data = _parse_json(raw)
        drugs = [str(d) for d in data.get("drugs", []) if str(d).strip()]
        reply = str(data.get("reply", "")).strip()
        action = str(data.get("action", "correct")).strip().lower()
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"mode": "correct", "reply": raw.strip()[:500],
                "spans": [s.model_dump() for s in req.spans], "unmatched": [], "applied": False}

    llm_spans, unmatched = service.locate_spans(req.text, drugs)

    if action == "review":
        diff = _diff(req.text, llm_spans, [(s.start, s.end) for s in req.spans])
        if not reply:
            reply = f"{diff['matched']} match · {len(diff['missing'])} missing · {len(diff['extra'])} extra."
        return {"mode": "review", "reply": reply, **diff}

    if not reply:
        reply = f"Updated to {len(llm_spans)} drug entries."
    return {"mode": "correct", "reply": reply, "spans": llm_spans,
            "unmatched": unmatched, "applied": True}


@router.post("/verify")
def verify(req: VerifyReq):
    user = f"PAGE TEXT:\n---\n{req.text}\n---\nReturn every drug entry as JSON."
    try:
        raw = llm.complete(_VERIFY_SYSTEM, user, max_tokens=8192, json_mode=True)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    try:
        drugs = [str(d) for d in _parse_json(raw).get("drugs", []) if str(d).strip()]
    except (json.JSONDecodeError, ValueError, TypeError):
        raise HTTPException(502, "El LLM no devolvió una lista válida.")
    llm_spans, _ = service.locate_spans(req.text, drugs)
    return _diff(req.text, llm_spans, [(s.start, s.end) for s in req.spans])


@router.get("/stats")
def stats():
    return _stats()
