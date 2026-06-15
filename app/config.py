"""Configuración central de la app (compartida por todos los modelos).

Lee variables de entorno desde:
  1. backend/.env                               ← tu HF_TOKEN va aquí
  2. ../../training-guide/ML-parsing-api/.env   ← respaldo del proyecto real
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# .../curation-app/backend  (la raíz del backend; el paquete vive en backend/app)
BACKEND_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BACKEND_DIR / ".env", override=False)
_ml_api_env = BACKEND_DIR.parent.parent / "training-guide" / "ML-parsing-api" / ".env"
if _ml_api_env.exists():
    load_dotenv(_ml_api_env, override=False)

# ── credenciales / hardware compartidos ──────────────────────────────────────
HF_ORG = os.getenv("HF_ORG", "Ethermed")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
DEVICE = os.getenv("DEVICE")  # None ⇒ autodetect

# ── repos de cada modelo (overridable; repo de HF o carpeta local) ───────────
RELEVANCE_MODEL = os.getenv("RELEVANCE_MODEL", f"{HF_ORG}/formulary-relevance-classifier")
SEGMENTER_MODEL = os.getenv("SEGMENTER_MODEL", f"{HF_ORG}/bio-drug-segmenter")
SPAN_CLASSIFIER_MODEL = os.getenv("SPAN_CLASSIFIER_MODEL", f"{HF_ORG}/drug-span-classifier")
SEQ2SEQ_MODEL = os.getenv("SEQ2SEQ_MODEL", f"{HF_ORG}/drug-span-cleaner-flan-t5")
PAIRWISE_MODEL = os.getenv("PAIRWISE_MODEL", f"{HF_ORG}/drug-matcher-cross-encoder-bge")

# Modelo 5 — índice RxNorm (bi-encoder, Modelo 6) para recuperar candidatos.
RXNORM_INDEX_REPO = os.getenv("RXNORM_INDEX_REPO", f"{HF_ORG}/rxnorm-drug-index")
RXNORM_INDEX_DIR = os.getenv("RXNORM_INDEX_DIR")   # carpeta local opcional (salta descarga)
MATCH_TOP_K = int(os.getenv("MATCH_TOP_K", "25"))  # candidatos recuperados por query
MATCH_MAX_LEN = 192                                # max_len del cross-encoder

# Modelo 3 — etiquetas (orden EXACTO del trainer: 0=single,1=multi,2=broken).
SPAN_LABELS = {0: "single", 1: "multi", 2: "broken"}
# Si la clase top es single pero broken está a <margen, se marca "possible broken".
POSSIBLE_BROKEN_MARGIN = float(os.getenv("POSSIBLE_BROKEN_MARGIN", "0.15"))

# Modelo 4 — cleaner FLAN-T5 (prefijo + longitudes EXACTAS del trainer).
SEQ2SEQ_INSTRUCTION = "Clean and normalize this drug formulary span:"
SEQ2SEQ_MAX_INPUT_LEN = 128
SEQ2SEQ_MAX_OUTPUT_LEN = 128

# ── LLM del chat de corrección (Gemini o Anthropic) ─────────────────────────
# El proveedor se autodetecta por la API key presente; CHAT_PROVIDER puede forzarlo.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
# Acepta CHAT_PROVIDER o LLM_PROVIDER (nombre del proyecto original).
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER") or os.getenv("LLM_PROVIDER")  # "anthropic"|"gemini"|None=auto
# Modelo por proveedor (overridable; acepta nombres alternativos).
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL") or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def chat_provider() -> str | None:
    """Proveedor activo del chat: forzado por CHAT_PROVIDER o autodetectado por key."""
    if CHAT_PROVIDER in ("anthropic", "gemini"):
        return CHAT_PROVIDER
    if GEMINI_API_KEY:
        return "gemini"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    return None


# ── render de PDF ────────────────────────────────────────────────────────────
THUMB_DPI = int(os.getenv("THUMB_DPI", "72"))
FULL_DPI = int(os.getenv("FULL_DPI", "140"))

# ── almacenamiento (fuera del paquete, en la raíz del backend) ───────────────
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BACKEND_DIR / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_DIR = Path(os.getenv("DATASETS_DIR", str(BACKEND_DIR / "datasets")))
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

TRAINED_MODELS_DIR = Path(os.getenv("TRAINED_MODELS_DIR", str(BACKEND_DIR / "trained_models")))
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRAIN_EXAMPLES = int(os.getenv("MIN_TRAIN_EXAMPLES", "50"))

# Subcarpeta de dataset por modelo (la que apuntará cada trainer).
RELEVANCE_OUT = DATASETS_DIR / "relevance"
SEGMENTER_OUT = DATASETS_DIR / "segmenter"
SPAN_CLASSIFIER_OUT = DATASETS_DIR / "span_classifier"
CLEANER_OUT = DATASETS_DIR / "cleaner"          # {input, output}
MATCHER_OUT = DATASETS_DIR / "matcher"          # pares query/candidate + label
for _d in (RELEVANCE_OUT, SEGMENTER_OUT, SPAN_CLASSIFIER_OUT, CLEANER_OUT, MATCHER_OUT):
    _d.mkdir(parents=True, exist_ok=True)

# ── etiquetas canónicas del Modelo 1 (no cambiar: el trainer mapea a 1/0) ────
LABEL_RELEVANT = "RELEVANT"
LABEL_NOT_RELEVANT = "NOT_RELEVANT"
DEFAULT_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))


def resolve_device() -> str:
    """cpu / cuda, resuelto en runtime. Compartido por todos los modelos."""
    if DEVICE:
        return DEVICE
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
