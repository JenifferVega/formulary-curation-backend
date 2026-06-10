"""Cargador genérico de modelos de Hugging Face (perezoso + singleton).

Centraliza lo que antes repetía cada modelo: resolver device, autenticar con el
token, cargar tokenizer + modelo una sola vez, y traducir errores de carga
(repo privado, sin token) a un mensaje accionable. Cada `service.py` se queda en
unas pocas líneas.

Uso:
    seg = LazyModel("Model 2", config.SEGMENTER_MODEL, kind="token")
    tok, model, device = seg.load()      # descarga/caché la 1ª vez
    seg.id2label                         # disponible tras load() si kind="token"
    seg.status()                         # {"model","loaded","error"} para /health
"""
from __future__ import annotations

from app import config

# kind -> nombre de la clase Auto de transformers (import perezoso en load()).
_AUTO = {
    "sequence": "AutoModelForSequenceClassification",   # clasifica TODO el texto
    "token": "AutoModelForTokenClassification",         # etiqueta cada token
    "seq2seq": "AutoModelForSeq2SeqLM",                 # genera texto (Modelo 4)
}


class LazyModel:
    def __init__(self, label: str, repo: str, kind: str):
        self.label = label          # "Model 1" — para mensajes de error
        self.repo = repo            # repo de HF o carpeta local
        self.kind = kind            # 'sequence' | 'token' | 'seq2seq'
        self._tok = None
        self._model = None
        self._device = None
        self.id2label: dict | None = None
        self.error: str | None = None

    def status(self) -> dict:
        return {"model": self.repo, "loaded": self._model is not None, "error": self.error}

    def load(self):
        if self._model is not None:
            return self._tok, self._model, self._device

        import transformers
        from transformers import AutoTokenizer
        auto_cls = getattr(transformers, _AUTO[self.kind])

        try:
            device = config.resolve_device()
            tok = AutoTokenizer.from_pretrained(self.repo, token=config.HF_TOKEN)
            model = auto_cls.from_pretrained(self.repo, token=config.HF_TOKEN).to(device).eval()
        except Exception as exc:  # noqa: BLE001 — a mensaje accionable
            msg = str(exc).lower()
            hint = ""
            if any(k in msg for k in ("401", "403", "gated", "authoriz")):
                hint = (f"  ⇒ The repo is private. Set an HF_TOKEN with access to "
                        f"the '{config.HF_ORG}' org in backend/.env")
            elif any(k in msg for k in ("couldn't connect", "resolve", "not a local folder")):
                hint = "  ⇒ Offline? The model is downloaded on first use."
            self.error = f"Could not load {self.label} ('{self.repo}'): {exc}{hint}"
            raise RuntimeError(self.error) from exc

        self._tok, self._model, self._device = tok, model, device
        if self.kind == "token":
            self.id2label = {int(k): v for k, v in model.config.id2label.items()}
        self.error = None
        return tok, model, device
