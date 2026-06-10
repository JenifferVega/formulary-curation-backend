"""Modelo 2 — endpoints (/api/segmenter/*)."""
import json
import re

from fastapi import APIRouter, HTTPException

from app import config
from app.core import llm, store, uploads
from . import service
from .schemas import ChatReq, SaveReq, SegmentReq, VerifyReq

router = APIRouter(prefix="/api/segmenter", tags=["segmenter"])

_CHAT_SYSTEM = (
    "You help a reviewer curate drug-entry segmentation on US health-insurance "
    "formulary pages. You receive the RAW TEXT of one page, the drug entries currently "
    "detected, and the reviewer's message.\n"
    "First decide the ACTION:\n"
    "- \"review\": the message asks you to CHECK / VERIFY / find what's missing or wrong "
    "in general (e.g. 'what am I missing?', 'review the page', 'check this', "
    "'¿qué me falta?', 'revisa', 'verifica'). No specific drug is named to change.\n"
    "- \"correct\": the message is a specific instruction to CHANGE the list (add, remove, "
    "fix or adjust a particular entry).\n"
    "Then produce \"drugs\": the COMPLETE correct list of drug entries for the page, each "
    "copied VERBATIM (an exact substring of the page text — same words, full row incl. "
    "strengths/form/restrictions; do not paraphrase or invent). For \"correct\", apply the "
    "reviewer's instruction. For \"review\", just list every real drug entry you see. "
    "Exclude category headers (e.g. ANALGESICS), column headers, footers, phone numbers, legends.\n"
    "Respond with ONLY a JSON object: "
    '{"action": "review"|"correct", "drugs": ["<verbatim entry>", ...], '
    '"reply": "<one short sentence>"}'
)

_VERIFY_SYSTEM = (
    "You read US health-insurance formulary pages and list every drug entry. "
    "Given the RAW TEXT of one page, return EVERY drug entry, each copied VERBATIM "
    "(an exact substring of the page text, including its strengths, form and any "
    "restriction/QL on the same row). Do NOT include category headers (e.g. "
    "ANALGESICS), column headers, page footers, phone numbers or legends. "
    'Respond with ONLY a JSON object: {"drugs": ["<verbatim entry>", ...]}'
)


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and a1 > b0


def _diff(text: str, llm_spans: list[dict], current: list[tuple[int, int]]) -> dict:
    """Compara los spans del LLM contra los actuales (por solapamiento)."""
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
    """Extrae el objeto JSON de la respuesta del LLM (tolera ``` y texto extra)."""
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
def segment(req: SegmentReq):
    text = _page_text(req.upload_id, req.page)
    if text is None:
        raise HTTPException(404, "Upload/page not found (re-upload the PDF).")
    if not text.strip():
        return {"page": req.page, "text": text, "spans": []}
    try:
        spans = service.predict_spans(text)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"page": req.page, "text": text, "spans": spans}


@router.post("/save")
def save(req: SaveReq):
    """Guarda (o reemplaza) el registro BIO de UNA página. Un .jsonl por PDF, un
    registro por página; re-guardar la misma página la reemplaza."""
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
    """El revisor da una pista en lenguaje natural; un LLM (Gemini/Anthropic)
    devuelve la lista corregida de medicamentos y el backend recalcula los spans
    localizándolos en el texto. Devuelve {reply, spans, unmatched}."""
    current = [
        re.sub(r"\s+", " ", req.text[s.start:s.end]).strip()
        for s in req.spans
    ]
    listed = "\n".join(f"{i+1}. {c}" for i, c in enumerate(current)) or "(none detected yet)"
    user = (
        f"PAGE TEXT:\n---\n{req.text}\n---\n"
        f"CURRENTLY DETECTED DRUG ENTRIES ({len(current)}):\n{listed}\n\n"
        f"REVIEWER HINT: {req.message}\n\n"
        "Return the corrected full list as JSON."
    )
    try:
        raw = llm.complete(_CHAT_SYSTEM, user, max_tokens=8192, json_mode=True)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    try:
        data = _parse_json(raw)
        drugs = [str(d) for d in data.get("drugs", []) if str(d).strip()]
        reply = str(data.get("reply", "")).strip()
        action = str(data.get("action", "correct")).strip().lower()
    except (json.JSONDecodeError, ValueError, TypeError):
        # El LLM no devolvió JSON válido: deja los spans como están y muestra el texto.
        return {"mode": "correct", "reply": raw.strip()[:500],
                "spans": [s.model_dump() for s in req.spans], "unmatched": [], "applied": False}

    llm_spans, unmatched = service.locate_spans(req.text, drugs)

    if action == "review":
        # No aplica nada: devuelve el diff para que el revisor lo apruebe.
        diff = _diff(req.text, llm_spans, [(s.start, s.end) for s in req.spans])
        if not reply:
            reply = f"{diff['matched']} match · {len(diff['missing'])} missing · {len(diff['extra'])} extra."
        return {"mode": "review", "reply": reply, **diff}

    # action == "correct": aplica la lista corregida.
    if not reply:
        reply = f"Updated to {len(llm_spans)} drug entries."
    return {"mode": "correct", "reply": reply, "spans": llm_spans,
            "unmatched": unmatched, "applied": True}


@router.post("/verify")
def verify(req: VerifyReq):
    """Segunda lectura independiente del texto (LLM): lista todos los medicamentos
    y los compara con lo que segmentó la red. Devuelve los que FALTAN (en el texto
    pero no segmentados) y los que SOBRAN (segmentados pero el LLM no los ve como
    medicamento), por solapamiento de spans."""
    user = (f"PAGE TEXT:\n---\n{req.text}\n---\nReturn every drug entry as JSON.")
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
