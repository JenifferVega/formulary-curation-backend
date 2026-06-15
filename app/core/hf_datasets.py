"""Carga lazy + caché de los datasets de entrenamiento en HuggingFace.

Carga archivo por archivo (JSONL o Parquet) y etiqueta cada registro con
`_source_file` para poder filtrar por origen en el explorador.
Caché local en ~/.cache/huggingface; las llamadas siguientes son instantáneas.
El índice RxNorm (vector DB) se excluye intencionalmente.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from app import config

HF_DATASETS: dict[str, dict] = {
    "relevance": {
        "repo": f"{config.HF_ORG}/formulary-page-classifier-data",
        "label": "1 · Pages",
        "schema": "relevance",
    },
    "segmenter": {
        "repo": f"{config.HF_ORG}/drug-segmenter-data",
        "label": "2 · Drug entries",
        "schema": "segmenter",
    },
    "span_classifier": {
        "repo": f"{config.HF_ORG}/drug-span-classifier-data",
        "label": "3 · Single/Multi/Broken",
        "schema": "span_classifier",
    },
    "cleaner": {
        "repo": f"{config.HF_ORG}/drug-span-cleaner-data",
        "label": "4 · Clean / Normalize",
        "schema": "cleaner",
    },
    "matcher": {
        "repo": f"{config.HF_ORG}/drug-matcher-cross-encoder-data",
        "label": "5 · RxNorm match",
        "schema": "matcher",
    },
}

# cache[key] = {"records": [...], "files": [str, ...]}
_cache: dict[str, dict] = {}
_errors: dict[str, str] = {}
_lock = threading.Lock()

_DATA_EXTS = {".jsonl", ".json", ".parquet"}


def _load(key: str) -> dict:
    """Descarga todos los archivos de datos del repo y los parsea con _source_file."""
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = HF_DATASETS[key]["repo"]

    # Lista todos los archivos del repo (filtra solo datos).
    all_files = list(list_repo_files(repo, repo_type="dataset", token=config.HF_TOKEN))
    data_files = [
        f for f in all_files
        if Path(f).suffix in _DATA_EXTS and not f.startswith(".") and "README" not in f
    ]

    if not data_files:
        # Fallback: load_dataset si no hay archivos de datos reconocibles.
        from datasets import load_dataset
        ds = load_dataset(repo, token=config.HF_TOKEN, split="train")
        records = ds.to_list()
        return {"records": records, "files": []}

    records: list[dict] = []
    file_names: list[str] = []

    for fname in sorted(data_files):
        stem = Path(fname).name
        file_names.append(stem)
        local = hf_hub_download(
            repo_id=repo, filename=fname,
            repo_type="dataset", token=config.HF_TOKEN,
        )
        ext = Path(fname).suffix.lower()
        if ext in {".jsonl", ".json"}:
            _parse_jsonl(local, stem, records)
        elif ext == ".parquet":
            _parse_parquet(local, stem, records)

    return {"records": records, "files": sorted(file_names)}


def _parse_jsonl(path: str, stem: str, out: list) -> None:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if not isinstance(r, dict):
                    continue   # ignorar líneas que no sean objetos JSON
                r["_source_file"] = stem
                out.append(r)
            except json.JSONDecodeError:
                pass


def _parse_parquet(path: str, stem: str, out: list) -> None:
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        for r in df.to_dict("records"):
            r["_source_file"] = stem
            out.append(r)
    except ImportError:
        # pandas no instalado — intentar con pyarrow directamente
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        for batch in table.to_batches():
            for r in batch.to_pylist():
                r["_source_file"] = stem
                out.append(r)


def get(key: str) -> dict:
    """Devuelve {"records": [...], "files": [...]}. Carga la primera vez."""
    if key not in HF_DATASETS:
        raise KeyError(f"Dataset desconocido: '{key}'")
    with _lock:
        if key in _cache:
            return _cache[key]
        if key in _errors:
            raise RuntimeError(_errors[key])
        try:
            data = _load(key)
            _cache[key] = data
            _errors.pop(key, None)
            return data
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            hint = ""
            if any(k in msg.lower() for k in ("401", "403", "gated", "authoriz", "private")):
                hint = " — ¿HF_TOKEN configurado con acceso a la org Ethermed?"
            _errors[key] = f"{msg}{hint}"
            raise RuntimeError(_errors[key]) from exc


def status() -> list[dict]:
    result = []
    with _lock:
        for key, meta in HF_DATASETS.items():
            cached = _cache.get(key)
            result.append({
                "key": key,
                "repo": meta["repo"],
                "label": meta["label"],
                "schema": meta["schema"],
                "loaded": cached is not None,
                "total": len(cached["records"]) if cached else None,
                "files": cached["files"] if cached else [],
                "error": _errors.get(key),
            })
    return result
