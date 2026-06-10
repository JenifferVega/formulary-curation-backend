# Curation app — formulary ML models

Una sola app (**1 backend FastAPI + 1 frontend React**) para curar los modelos del
pipeline de `ML-parsing-api`. Cada modelo se cura en su propia **pestaña**. Subes un
PDF una vez y lo reusan todas las secciones. Lo que corriges se guarda como dataset
de entrenamiento en el esquema exacto de cada trainer.

```
React (tabs)  ──HTTP /api──►  FastAPI  ──►  modelos de HF (Ethermed/*)
 1 · Pages        /api/relevance/*        Modelo 1  DistilBERT
 2 · Drug entries /api/segmenter/*        Modelo 2  XLM-RoBERTa
 (3,4,5 pendientes)
```

Estado: **Modelos 1-5 implementados ✅** (los 5 modelos curables del pipeline).

---

## Cómo correrla

### 1. Token de Hugging Face (obligatorio, los repos son privados)
En `backend/.env`:
```
HF_TOKEN=hf_tuTokenConAccesoAEthermed
```

### 2. Backend
```powershell
cd curation-app\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --reload
```
> El target es **`app.main:app`** (el backend es un paquete en `backend/app/`).

### 3. Frontend
```powershell
cd curation-app\frontend
npm run dev
```
Abre **http://localhost:5173**. (Proxy Vite `/api`→`:8000`.)

### Uso
1. **Choose PDF → Upload** (una vez; lo usan las dos pestañas).
2. Pestaña **1 · Pages** (Modelo 1): *Classify pages* → la IA marca 💊/📄 → corriges
   las mal clasificadas → *Save for training*.
3. Pestaña **2 · Drug entries** (Modelo 2): eliges una página → *Find drug entries* →
   la IA resalta cada medicamento en el texto → **seleccionas texto para añadir** uno
   que faltó, **clic en un resaltado para quitarlo** → *Save page for training*.
   - **Chat "Fix with AI"** (botón flotante abajo a la derecha): le das una pista en
     lenguaje natural (“te faltó X”, “el 3º está cortado”) y un LLM corrige los resaltados.
     Tú revisas y, cuando estés segura, *Save*. Necesita una API key (Gemini o Anthropic)
     en `backend/.env`: `GEMINI_API_KEY=...` o `ANTHROPIC_API_KEY=...`.
     ⚠️ El aprendizaje NO es en tiempo real: las correcciones se guardan y mejoran el
     modelo al **reentrenar** (batch). El chat usa un LLM aparte (los Modelos 1/2 son
     clasificadores, no chatbots).

---

## Modelos y esquemas (de `training-guide/MODELS.md`)

| Pestaña | Modelo HF | Tipo | Dataset local | Esquema JSONL |
|---|---|---|---|---|
| 1 · Pages | `Ethermed/formulary-relevance-classifier` | DistilBERT | `backend/datasets/relevance/` | `{"page","label","text"}` |
| 2 · Drug entries | `Ethermed/bio-drug-segmenter` | XLM-RoBERTa | `backend/datasets/segmenter/` | `{"source_file","source_page","text","spans":[[ini,fin]]}` |
| 3 · Single/Multi/Broken | `Ethermed/drug-span-classifier` | DistilBERT-cased | `backend/datasets/span_classifier/` | `{"text","label"}` (0=single,1=multi,2=broken) |
| 4 · Clean / Normalize | `Ethermed/drug-span-cleaner-flan-t5` | FLAN-T5-base (seq2seq) | `backend/datasets/cleaner/` | `{"input","output"}` |
| 5 · RxNorm match | `Ethermed/drug-matcher-cross-encoder-bge` + índice `Ethermed/rxnorm-drug-index` | BGE reranker + txtai bi-encoder | `backend/datasets/matcher/` | `{"raw_text","query","candidate_text","candidate_rxcui","candidate_tty","candidate_score","label","bucket"}` |

> **Modelo 5** descarga ~2.4 GB la primera vez (índice RxNorm 831 MB + nomic-embed + bge-reranker) y necesita `txtai` + `einops` (ya en requirements.txt).

`label` del Modelo 1 = `RELEVANT` / `NOT_RELEVANT`. El Modelo 2 guarda **un registro
por página** (re-guardar la misma página la reemplaza); el trainer descarta páginas
con <1 span.

---

## Entrenar con lo que generes (desde cero o reentrenar)

### Modelo 1 — relevancia
```powershell
# apunta el trainer a tu carpeta curada y entrena
hf download Ethermed/formulary-page-classifier-data --repo-type dataset --local-dir classified_out  # opcional: combinar con datos previos
python training-guide\scripts\1_train_relevance.py   # lee ./classified_out, 3 épocas, DistilBERT
hf upload Ethermed/formulary-relevance-classifier formulary_classifier --repo-type model --private
```
> El trainer lee `./classified_out`. Copia/renombra ahí tu `backend/datasets/relevance/`,
> o ajusta la ruta en el script.

### Modelo 2 — segmentador BIO
```powershell
python training-guide\scripts\2_train_bio.py   # XLM-R, 3 épocas, ventana deslizante stride 128
hf upload Ethermed/bio-drug-segmenter bio_drug_segmenter --repo-type model --private --exclude "checkpoint-*/*"
```
> ⚠️ El README avisa: en el Modelo 2 la métrica F1 **esconde errores de límites de span
> en tablas** — por eso esta app te deja revisar los spans a ojo. Esa es su mayor utilidad.

El ciclo completo (generar → curar → entrenar → push) y los comandos de push están en
`training-guide/README.md`, `MODELS.md`, y `phase-2/PUSH_TO_HF.md`.

---

## Arquitectura (módulos verticales — cómo añadir el Modelo 3, 4, 5)

Cada modelo es un **paquete autocontenido** (service + schemas + router) sobre un
**núcleo compartido**. Añadir un modelo = añadir una carpeta + una línea en el registro.

```
backend/app/
  main.py              → crea FastAPI + monta el router de api
  config.py            → token, device, rutas, ids de modelo
  registry.py          → LISTA central de modelos (health + montaje de routers automático)
  api.py               → endpoints compartidos (/api/health, /api/upload, /api/page-image)
                          + incluye el router de cada modelo
  core/                → infraestructura compartida, agnóstica al modelo
    pdf.py             →   extraer texto / renderizar imágenes (PyMuPDF)
    uploads.py         →   subida de PDF compartida (texto por página cacheado)
    store.py           →   helpers JSONL (escribir / leer / contar)
    loader.py          →   cargador HF perezoso genérico (token, errores)
  models/              → un paquete por modelo (módulo vertical)
    relevance/  {service.py, schemas.py, router.py}   → Modelo 1  /api/relevance
    segmenter/  {service.py, schemas.py, router.py}   → Modelo 2  /api/segmenter

frontend/src/
  App.jsx              → shell: cabecera, subida, pestañas (TABS + VIEWS registry)
  api/                 → client.js (base) + relevance.js, segmenter.js + index.js
  views/               → RelevanceView.jsx, SegmenterView.jsx
  components/          → PageCard.jsx, PreviewModal.jsx, SpanEditor.jsx
  index.css
```

**Añadir el Modelo 3** (span classifier), receta exacta:
1. `backend/app/models/span_classifier/` con `service.py` (usa `core.loader.LazyModel`),
   `schemas.py`, `router.py` (`prefix="/api/span_classifier"`).
2. Una entrada en `backend/app/registry.py`.
3. `frontend/src/api/span_classifier.js` + `frontend/src/views/SpanClassifierView.jsx`.
4. Una entrada en `TABS` y en `VIEWS` de `App.jsx`.

El resto (subida, salud, imágenes, JSONL, cargador) se reusa — no se toca.

---

## Notas
- La app **no** hace push ni lanza el trainer (frontera deliberada; se usan los comandos
  `hf upload` de phase-2). Guardar el JSONL es su límite.
- Bug en `training-guide/scripts/1_classify_pages_llm.py:111` (cuenta `"DRUG_LIST"`, que
  nunca existe). Solo afecta un print.
- Nombre de repo inconsistente en `MODELS.md` (`formulary-relevance-classifier` vs
  `drug-formulary-relevance-classifier`). `app.py` y esta app usan el primero.
