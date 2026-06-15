"""Cliente LLM — soporta Anthropic y Gemini.

Expone:
  - status(): proveedor/modelo y si hay key
  - complete(system, user): llamada single-turn
  - complete_chat(system, messages): llamada multi-turn
    messages = [{"role": "user"|"assistant", "content": str}, ...]
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
    """Single-turn convenience wrapper."""
    return complete_chat(system, [{"role": "user", "content": user}],
                         max_tokens=max_tokens, temperature=temperature,
                         json_mode=json_mode)


def complete_chat(system: str, messages: list[dict],
                  max_tokens: int = 8192, temperature: float = 0.0,
                  json_mode: bool = False) -> str:
    """Multi-turn chat. messages = [{role, content}, ...].
    Lanza RuntimeError si no hay proveedor configurado."""
    prov = config.chat_provider()
    if prov is None:
        raise RuntimeError(
            "No hay LLM configurado. Pon GEMINI_API_KEY o ANTHROPIC_API_KEY "
            "en backend/.env (y opcional CHAT_PROVIDER).")
    if prov == "anthropic":
        return _anthropic_chat(system, messages, max_tokens, temperature)
    return _gemini_chat(system, messages, max_tokens, temperature, json_mode)


def _anthropic_chat(system: str, messages: list[dict],
                    max_tokens: int, temperature: float) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Falta el SDK: pip install anthropic") from exc
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    # Anthropic requires alternating user/assistant; roles already match.
    api_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
    resp = client.messages.create(
        model=config.ANTHROPIC_CHAT_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=api_msgs,
    )
    return "".join(block.text for block in resp.content
                   if getattr(block, "type", "") == "text")


def _gemini_chat(system: str, messages: list[dict],
                 max_tokens: int, temperature: float,
                 json_mode: bool = False) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Falta el SDK: pip install google-genai") from exc
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    # Gemini uses "model" role instead of "assistant"
    def _to_gemini_role(r: str) -> str:
        return "model" if r == "assistant" else r

    contents = [
        types.Content(role=_to_gemini_role(m["role"]),
                      parts=[types.Part(text=m["content"])])
        for m in messages
    ]

    kwargs = dict(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:  # noqa: BLE001
        pass

    try:
        resp = client.models.generate_content(
            model=config.GEMINI_CHAT_MODEL, contents=contents,
            config=types.GenerateContentConfig(**kwargs))
    except Exception:  # noqa: BLE001
        kwargs.pop("thinking_config", None)
        resp = client.models.generate_content(
            model=config.GEMINI_CHAT_MODEL, contents=contents,
            config=types.GenerateContentConfig(**kwargs))
    return resp.text or ""
