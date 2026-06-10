"""Cliente LLM para el chat de corrección — soporta Anthropic y Gemini.

El proveedor se elige por `config.chat_provider()` (forzado o autodetectado por
la API key presente). Expone:
  - status(): para /api/health (qué proveedor/modelo y si hay key)
  - complete(system, user): texto de respuesta del LLM

Los SDKs se importan de forma perezosa para no exigirlos si no se usa el chat.
"""
from __future__ import annotations

from app import config


def status() -> dict:
    prov = config.chat_provider()
    model = None
    if prov == "anthropic":
        model = config.ANTHROPIC_CHAT_MODEL
    elif prov == "gemini":
        model = config.GEMINI_CHAT_MODEL
    return {"provider": prov, "model": model, "ready": prov is not None}


def complete(system: str, user: str, max_tokens: int = 8192,
             temperature: float = 0.0, json_mode: bool = False) -> str:
    """Una llamada de chat. Devuelve el texto. Lanza RuntimeError con mensaje
    accionable si no hay proveedor/clave o si el SDK no está instalado.

    json_mode=True fuerza salida JSON (Gemini) — úsalo cuando esperas un objeto."""
    prov = config.chat_provider()
    if prov is None:
        raise RuntimeError(
            "No hay LLM configurado para el chat. Pon GEMINI_API_KEY o "
            "ANTHROPIC_API_KEY en backend/.env (y opcional CHAT_PROVIDER).")

    if prov == "anthropic":
        return _anthropic(system, user, max_tokens, temperature)
    return _gemini(system, user, max_tokens, temperature, json_mode)


def _anthropic(system: str, user: str, max_tokens: int, temperature: float) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Falta el SDK: pip install anthropic") from exc
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.ANTHROPIC_CHAT_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


def _gemini(system: str, user: str, max_tokens: int, temperature: float,
            json_mode: bool = False) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Falta el SDK: pip install google-genai") from exc
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    kwargs = dict(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    # Desactiva el "thinking" de gemini-2.5-* para no gastar tokens de salida
    # (best-effort: algunos modelos no lo soportan).
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:  # noqa: BLE001
        pass

    try:
        resp = client.models.generate_content(
            model=config.GEMINI_CHAT_MODEL, contents=user,
            config=types.GenerateContentConfig(**kwargs))
    except Exception:  # noqa: BLE001 — reintenta sin thinking_config si el modelo lo rechaza
        kwargs.pop("thinking_config", None)
        resp = client.models.generate_content(
            model=config.GEMINI_CHAT_MODEL, contents=user,
            config=types.GenerateContentConfig(**kwargs))
    return resp.text or ""
