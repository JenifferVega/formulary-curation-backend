"""Índice RxNorm (bi-encoder, Modelo 6) — recuperación de candidatos.

Carga UNA vez un índice txtai `Embeddings` (encoder nomic-embed-text-v1.5) y lo
consulta con SQL-ish `similar()` para traer los top-K candidatos que luego el
cross-encoder (Modelo 5) reordena. Portado de ML-parsing-api/app.py
(get_bi_index / _topk_bi).

El índice se baja del repo dataset `Ethermed/rxnorm-drug-index` (~831 MB) la
primera vez (cacheado), salvo que RXNORM_INDEX_DIR apunte a una carpeta local.
"""
from __future__ import annotations

from app import config

_state: dict = {"emb": None, "error": None}


def status() -> dict:
    return {"index": config.RXNORM_INDEX_DIR or config.RXNORM_INDEX_REPO,
            "loaded": _state["emb"] is not None, "error": _state["error"]}


def get_index():
    if _state["emb"] is not None:
        return _state["emb"]

    # El encoder nomic ship código remoto; pre-aprobar antes de cargar (como app.py).
    import transformers.dynamic_module_utils as _trc
    _trc.resolve_trust_remote_code = lambda *a, **kw: True
    from txtai import Embeddings

    try:
        if config.RXNORM_INDEX_DIR:
            path = config.RXNORM_INDEX_DIR
        else:
            from huggingface_hub import snapshot_download
            path = snapshot_download(
                config.RXNORM_INDEX_REPO, repo_type="dataset", token=config.HF_TOKEN)
        emb = Embeddings()
        emb.load(path)
    except Exception as exc:  # noqa: BLE001
        _state["error"] = f"Could not load RxNorm index ('{config.RXNORM_INDEX_REPO}'): {exc}"
        raise RuntimeError(_state["error"]) from exc

    _state["emb"] = emb
    return emb


def search(query: str, k: int) -> list[dict]:
    """Top-K candidatos RxNorm para el query (orden por similitud del bi-encoder)."""
    emb = get_index()
    safe = query.replace("'", "''")
    return emb.search(
        f"SELECT id, text, score, tty, rxcui, base_name, dose_form, "
        f"brand_name, strength_text FROM txtai WHERE similar('{safe}') LIMIT {k}"
    )
