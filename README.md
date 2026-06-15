# Formulary Curation — Backend

FastAPI backend for the formulary curation tool. Serves inference, dataset storage, training, and LLM-assisted review for all five neural network models in the Ethermed formulary pipeline.

---

## Architecture

```
formulary-curation-backend/
├── app/
│   ├── main.py              # FastAPI app entry point, CORS
│   ├── api.py               # Root router — shared endpoints + model routers
│   ├── config.py            # All env vars and path constants
│   ├── registry.py          # Model registry (list of active models)
│   ├── core/
│   │   ├── llm.py           # LLM client (Anthropic / Gemini, single & multi-turn)
│   │   ├── pdf.py           # PDF parsing and page image rendering (PyMuPDF)
│   │   ├── uploads.py       # Upload storage and page text extraction
│   │   ├── store.py         # JSONL dataset read/write helpers
│   │   ├── trainer.py       # Fine-tuning pipeline (background threads, HF push)
│   │   ├── hf_datasets.py   # HuggingFace dataset loader
│   │   └── rxnorm_index.py  # RxNorm bi-encoder index (txtai)
│   └── models/
│       ├── relevance/       # Model 1 — DistilBERT page classifier
│       ├── segmenter/       # Model 2 — XLM-RoBERTa BIO drug segmenter
│       ├── span_classifier/ # Model 3 — DistilBERT span classifier
│       ├── cleaner/         # Model 4 — FLAN-T5 drug name normalizer
│       └── matcher/         # Model 5 — BGE cross-encoder RxNorm matcher
├── datasets/                # Curated JSONL files per model (gitignored)
├── uploads/                 # Uploaded PDFs and extracted page text (gitignored)
├── trained_models/          # Fine-tuned model checkpoints (gitignored)
└── requirements.txt
```

Each model package follows the same structure:

```
models/<name>/
├── __init__.py
├── router.py    # FastAPI endpoints (/api/<name>/*)
├── schemas.py   # Pydantic request/response models
└── service.py   # Model loading and inference logic
```

---

## Models

| Tab | Model | Base | Task |
|-----|-------|------|------|
| 1 · Pages | Relevance Classifier | DistilBERT | `RELEVANT` / `NOT_RELEVANT` per page |
| 2 · Drug entries | BIO Segmenter | XLM-RoBERTa | Token classification — `B-DRUG` / `I-DRUG` / `O` |
| 3 · Span type | Span Classifier | DistilBERT | `single` / `multi` / `broken` per span |
| 4 · Normalize | Cleaner | FLAN-T5 | Raw drug entry → canonical name |
| 5 · RxNorm | Matcher | BGE cross-encoder | Drug span → RxNorm concept |

All models are loaded from private HuggingFace repos under the `Ethermed/` organization. A `HF_TOKEN` with read access is required.

---

## API Endpoints

### Shared
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Server status, model load state, LLM provider, RxNorm index |
| POST | `/api/upload` | Upload a formulary PDF |
| GET | `/api/page-image/{upload_id}/{page}` | Rendered PNG of a page |
| GET | `/api/trainer/status` | Training status for all 5 models |
| GET | `/api/trainer/status/{key}` | Status + curated count for one model |
| GET | `/api/trainer/curated/{key}` | Paginated curated records for one model |
| POST | `/api/trainer/train/{key}` | Start fine-tuning in a background thread |
| POST | `/api/trainer/push/{key}` | Push trained model + dataset to HuggingFace Hub |
| DELETE | `/api/trainer/trained/{key}` | Discard trained model from memory and disk |

### Per model
Each model exposes: `POST /predict-or-equivalent`, `POST /save`, `GET /stats`, `POST /verify`, `POST /chat`

| Model | Prefix | Key endpoints |
|-------|--------|---------------|
| Relevance | `/api/relevance` | `/classify`, `/save`, `/stats` |
| Segmenter | `/api/segmenter` | `/segment`, `/save`, `/verify`, `/chat`, `/stats` |
| Span Classifier | `/api/span_classifier` | `/classify`, `/save`, `/verify`, `/chat`, `/stats` |
| Cleaner | `/api/cleaner` | `/clean`, `/save`, `/verify`, `/chat`, `/stats` |
| Matcher | `/api/matcher` | `/match`, `/save`, `/stats` |

### Verify endpoints (Models 2–4)
Each runs an independent LLM pass over the same content the neural network processed and returns a diff:
- **Segmenter** `/verify` — finds missing/extra drug spans vs NN output
- **Span Classifier** `/verify` — re-classifies each span, returns disagreements
- **Cleaner** `/verify` — re-normalizes each entry, returns disagreements

### Chat endpoints (Models 2–4)
Context-aware multi-turn chat. Each has a distinct personality scoped to its model:
- **Segmenter** — knows about adding/removing spans, review vs correct actions
- **Span Classifier** — knows single/multi/broken classification rules
- **Cleaner** — knows normalization rules (strip tiers, QL, symbols)

Requests include a `history` field (list of `{role, content}`) so the LLM has memory of previous turns in the session.

---

## Setup

### 1. Environment variables

Create `backend/.env`:

```env
# Required — HuggingFace token with read access to Ethermed/* repos
HF_TOKEN=hf_...

# LLM for verify and chat (one or both)
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...

# Optional overrides
CHAT_PROVIDER=gemini          # or "anthropic" (auto-detected from keys if omitted)
ANTHROPIC_CHAT_MODEL=claude-haiku-4-5
GEMINI_CHAT_MODEL=gemini-2.5-flash
DEVICE=cuda                   # auto-detected if omitted
RELEVANCE_THRESHOLD=0.5
MIN_TRAIN_EXAMPLES=50
```

### 2. Install dependencies

```bash
cd formulary-curation-backend
pip install -r requirements.txt
```

### 3. Run

```bash
uvicorn app.main:app --port 8000 --reload
```

The frontend proxies `/api` → `localhost:8000` via Vite.

---

## LLM Client (`app/core/llm.py`)

Supports Anthropic and Gemini. Auto-detects provider from whichever API key is present. Exposes two functions:

- `complete(system, user)` — single-turn call
- `complete_chat(system, messages)` — multi-turn call, accepts `[{role, content}]`

Provider and model are overridable via env vars. Gemini disables the thinking budget by default to avoid burning output tokens.

---

## Training Pipeline (`app/core/trainer.py`)

- Runs in a background thread; status polled via `/api/trainer/status/{key}`
- Mixes HuggingFace base dataset + locally curated JSONL records
- Detects label conflicts in curated data before training
- Saves checkpoint to `trained_models/{key}/`
- Push to Hub merges curated records into the HF dataset and uploads both

Minimum curated examples before training is allowed: `MIN_TRAIN_EXAMPLES` (default 50).

---

## Dataset Storage

Curated records are stored as JSONL files in `datasets/{model_name}/`. Each save call deduplicates by primary key (text / input) so re-editing an entry updates rather than appends.

| Model | File schema |
|-------|-------------|
| Relevance | `{text, label}` — label `"RELEVANT"` or `"NOT_RELEVANT"` |
| Segmenter | `{source_file, source_page, text, spans: [[start, end]]}` |
| Span Classifier | `{text, label}` — label `0/1/2` |
| Cleaner | `{input, output}` |
| Matcher | `{query, candidate, label}` |
