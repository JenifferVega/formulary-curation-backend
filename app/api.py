"""Router raíz de la API: endpoints compartidos + monta el router de cada modelo.

Esta es la "carpeta api": el único lugar que reúne todas las URLs. Los endpoints
por modelo viven en su propio paquete (app/models/<x>/router.py) y se montan aquí
automáticamente desde el registry.
"""
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from app import config
from app.core import llm, pdf, rxnorm_index, uploads
from app.registry import MODELS

router = APIRouter()

# Monta el router de cada modelo (registrado en registry.py).
for _m in MODELS:
    router.include_router(_m["router"])


# ── endpoints compartidos ────────────────────────────────────────────────────
@router.get("/api/health")
def health():
    return {
        "status": "ok",
        "hf_token_present": bool(config.HF_TOKEN),
        "device": config.resolve_device(),
        "models": {m["key"]: m["status"]() for m in MODELS},
        "chat": llm.status(),
        "rxnorm_index": rxnorm_index.status(),
    }


@router.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "The file must be a PDF.")
    data = await file.read()
    try:
        return uploads.save_upload(file.filename, data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not read the PDF: {exc}") from exc


@router.get("/api/page-image/{upload_id}/{page}")
def page_image(upload_id: str, page: int):
    path = uploads.get_path(upload_id)
    if not path:
        raise HTTPException(404, "Upload not found (did the session expire?).")
    png = pdf.render_page_png(path, page, config.FULL_DPI)
    if png is None:
        raise HTTPException(404, f"Page {page} is out of range.")
    return Response(content=png, media_type="image/png")
