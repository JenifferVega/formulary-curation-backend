"""Router raíz de la API: endpoints compartidos + monta el router de cada modelo.

Esta es la "carpeta api": el único lugar que reúne todas las URLs. Los endpoints
por modelo viven en su propio paquete (app/models/<x>/router.py) y se montan aquí
automáticamente desde el registry.
"""
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app import config
from app.core import hf_datasets, llm, pdf, rxnorm_index, trainer, uploads
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


@router.get("/api/hf-datasets")
def list_hf_datasets():
    """Estado de todos los datasets de HF (cargados / no cargados / error)."""
    return hf_datasets.status()


@router.get("/api/hf-datasets/{key}")
def get_hf_dataset(
    key: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    q: str = Query(""),
    file: str = Query(""),    # filtra por _source_file
    label: str = Query(""),   # filtra por label (valor exacto, p.ej. "0", "1", "2", "RELEVANT")
):
    """Devuelve registros paginados + lista de archivos del dataset."""
    if key not in hf_datasets.HF_DATASETS:
        raise HTTPException(404, f"Dataset desconocido: '{key}'")
    try:
        data = hf_datasets.get(key)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    records = data["records"]
    if file:
        records = [r for r in records if r.get("_source_file") == file]
    if label != "":
        # Soporta int o string según cómo esté guardado el campo.
        def _label_match(r: dict) -> bool:
            v = r.get("label")
            return str(v) == label
        records = [r for r in records if _label_match(r)]
    if q:
        ql = q.lower()
        records = [r for r in records if ql in str(r).lower()]

    total = len(records)
    return {
        "key": key,
        "total": total,
        "offset": offset,
        "limit": limit,
        "files": data["files"],
        "records": records[offset: offset + limit],
    }


# ── trainer endpoints ────────────────────────────────────────────────────────

class _TrainReq(BaseModel):
    indices: list[int] | None = None   # None = all curated records
    epochs: int = 3


@router.get("/api/trainer/status")
def trainer_status_all():
    """Status + curated count for all 5 models."""
    return trainer.status_all()


@router.get("/api/trainer/status/{key}")
def trainer_status(key: str):
    if key not in trainer.TRAIN_META:
        raise HTTPException(404, f"Unknown model key: '{key}'")
    curated = trainer.curated_records(key)
    # Recompute conflicts every time status is fetched so they're visible
    # before training starts, not only after pressing Train
    conflicts = trainer.detect_conflicts(key, curated)
    trainer._states[key].conflicts = conflicts
    s = trainer.training_status(key)
    return {
        **s,
        "curated_count": len(curated),
        "has_trained": trainer.get_trained_model(key) is not None,
        "min_examples": config.MIN_TRAIN_EXAMPLES,
    }


@router.get("/api/trainer/curated/{key}")
def trainer_curated(
    key: str,
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Curated (locally-saved) records for the given model, paginated."""
    if key not in trainer.TRAIN_META:
        raise HTTPException(404, f"Unknown model key: '{key}'")
    records = trainer.curated_records(key)
    return {
        "key": key,
        "total": len(records),
        "offset": offset,
        "limit": limit,
        "records": records[offset: offset + limit],
        "name": trainer.TRAIN_META[key]["name"],
    }


@router.post("/api/trainer/train/{key}")
def trainer_train(key: str, req: _TrainReq):
    """Start a full-parameter training job (background thread)."""
    if key not in trainer.TRAIN_META:
        raise HTTPException(404, f"Unknown model key: '{key}'")
    try:
        trainer.start_training(key, req.indices, req.epochs)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"started": True, "key": key, "epochs": req.epochs}


@router.post("/api/trainer/push/{key}")
def trainer_push(key: str):
    """Push trained model + merged dataset to HuggingFace Hub (background thread)."""
    if key not in trainer.TRAIN_META:
        raise HTTPException(404, f"Unknown model key: '{key}'")
    try:
        trainer.push_to_hub(key)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"pushing": True, "key": key}


@router.delete("/api/trainer/trained/{key}")
def trainer_discard(key: str):
    """Remove trained model from memory and disk, resetting state to idle."""
    if key not in trainer.TRAIN_META:
        raise HTTPException(404, f"Unknown model key: '{key}'")
    try:
        trainer.discard_trained_model(key)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"discarded": True, "key": key}


# ── page image ───────────────────────────────────────────────────────────────

@router.get("/api/page-image/{upload_id}/{page}")
def page_image(upload_id: str, page: int):
    path = uploads.get_path(upload_id)
    if not path:
        raise HTTPException(404, "Upload not found (did the session expire?).")
    png = pdf.render_page_png(path, page, config.FULL_DPI)
    if png is None:
        raise HTTPException(404, f"Page {page} is out of range.")
    return Response(content=png, media_type="image/png")
