"""Punto de entrada FastAPI. Arrancar con:  uvicorn app.main:app --port 8000

Crea la app, habilita CORS para el frontend y monta el router raíz de la API
(que a su vez monta los endpoints compartidos + el router de cada modelo).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router

app = FastAPI(title="Curation app — formulary ML models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev; restringir en producción
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)