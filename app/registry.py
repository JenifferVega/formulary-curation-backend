"""Registro central de modelos.

Única lista que hay que tocar para añadir un modelo: `health` y el montaje de
routers iteran sobre ella. Añadir el Modelo 3 = crear app/models/span_classifier/
y agregar una entrada aquí.
"""
from app.models import cleaner, matcher, relevance, segmenter, span_classifier

MODELS = [
    {"key": "relevance", "title": "Relevance classifier (Model 1)",
     "router": relevance.router, "status": relevance.status},
    {"key": "segmenter", "title": "BIO drug segmenter (Model 2)",
     "router": segmenter.router, "status": segmenter.status},
    {"key": "span_classifier", "title": "Span classifier (Model 3)",
     "router": span_classifier.router, "status": span_classifier.status},
    {"key": "cleaner", "title": "Span cleaner FLAN-T5 (Model 4)",
     "router": cleaner.router, "status": cleaner.status},
    {"key": "matcher", "title": "Cross-encoder RxNorm matcher (Model 5)",
     "router": matcher.router, "status": matcher.status},
]
