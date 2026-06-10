"""Helpers compartidos para escribir/leer/contar datasets JSONL.

Cada modelo construye sus propios registros (su esquema vive en su router); aquí
solo va lo común: sanear nombres, escribir y contar.
"""
import json
import re
from pathlib import Path


def safe_stem(filename: str) -> str:
    """Nombre de archivo derivado del PDF (sin extensión, saneado)."""
    stem = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return stem or "documento"


def write_jsonl(out_dir: Path, stem: str, records: list[dict]) -> Path:
    """Sobrescribe out_dir/<stem>.jsonl con los registros dados."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def read_all(out_dir: Path) -> list[dict]:
    """Todos los registros de todos los .jsonl de la carpeta."""
    records: list[dict] = []
    for fp in sorted(Path(out_dir).glob("*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def read_file(path: Path) -> list[dict]:
    """Registros de un .jsonl concreto (vacío si no existe)."""
    out: list[dict] = []
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def file_count(out_dir: Path) -> int:
    return len(list(Path(out_dir).glob("*.jsonl")))
