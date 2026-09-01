"""Thin, provider-agnostic LLM wrapper.

The recap script and the Chinese translation are *written* by an LLM when
configured. If you prefer to hand-write the script / translation (or have no
API key), the pipeline reads them from files instead — see script.py and
translate.py for the fallback paths.

Supports: OpenAI, Anthropic, DeepSeek (OpenAI-compatible), and Ollama.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str


class LLMError(RuntimeError):
    pass


def _client_from(provider: str, model: str):
    """Lazily build a client and return (client, model, needs_key=False)."""
    provider = (provider or "").strip().lower()

    if provider in ("", "none"):
        raise LLMError(
            "No LLM provider configured. Set LLM_PROVIDER (openai/anthropic/"
            "deepseek/ollama) and the matching API key, OR provide a pre-written "
            "script/translation file (see README)."
        )

    if provider == "openai":
        import openai  # type: ignore

        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        model = model or os.environ.get("MODEL_NAME") or "gpt-4o-mini"
        return client, model

    if provider == "deepseek":
        import openai  # type: ignore

        client = openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1",
        )
        model = model or os.environ.get("MODEL_NAME") or "deepseek-chat"
        return client, model

    if provider == "anthropic":
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        model = model or os.environ.get("MODEL_NAME") or "claude-3-5-sonnet-latest"
        return client, model

    if provider == "ollama":
        import openai  # type: ignore

        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        client = openai.OpenAI(base_url=base, api_key="ollama")
        model = model or os.environ.get("MODEL_NAME") or "qwen2.5"
        return client, model

    raise LLMError(f"Unknown LLM provider: {provider!r}")


def complete(provider: str, model: str, system: str, user: str) -> str:
    """Send a single completion (no history). Returns assistant text."""
    client, resolved_model = _client_from(provider, model)

    try:  # OpenAI-compatible + Ollama
        resp = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except AttributeError:
        pass
    except Exception as exc:  # surface a friendly Ollama hint
        if provider == "ollama":
            raise LLMError(
                f"Ollama request failed ({type(exc).__name__}: {exc}). "
                "Make sure Ollama is running (`ollama serve`) and the model is "
                f"pulled (`ollama pull {resolved_model}`), and that "
                "OLLAMA_BASE_URL points at it."
            ) from exc
        raise

    # Anthropic
    resp = client.messages.create(
        model=resolved_model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def provider_configured(provider: str) -> bool:
    """Whether an LLM provider looks usable (has a key we can find)."""
    p = (provider or "").strip().lower()
    if p == "ollama":
        # Ollama needs no key and is configured by default at localhost:11434.
        return True
    if p in ("", "none"):
        return False
    if p == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if p == "deepseek":
        return bool(os.environ.get("DEEPSEEK_API_KEY"))
    if p == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return False
