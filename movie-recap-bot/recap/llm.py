"""Thin, provider-agnostic LLM wrapper.

The recap script and the Chinese translation are *written* by an LLM when
configured. If you prefer to hand-write the script / translation (or have no
API key), the pipeline reads them from files instead — see script.py and
translate.py for the fallback paths.

Supports: OpenAI, Anthropic, DeepSeek (OpenAI-compatible), Ollama, and the
free-tier OpenAI-compatible clouds Groq and Google Gemini (Flash).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str


class LLMError(RuntimeError):
    pass


def _retryable(exc: Exception) -> bool:
    """True for transient faults worth retrying (network, 429, 5xx).

    ``empty message`` is deliberately NOT here: a provider that answered with
    an empty completion spent the tokens already and will very likely answer
    empty again — retrying it up to four times is pure token waste.
    """
    s = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "timeout", "timed out", "connection", "conn reset", "temporarily",
        "rate limit", "ratelimit", "429", "500", "502", "503", "504",
        "overloaded", "unavailable", "apiconnection", "internalserver",
        "getaddrinfo",
    )
    return any(m in s for m in markers)


def _token_log_enabled() -> bool:
    return os.environ.get("RECAP_TOKEN_LOG", "").strip() in ("1", "true", "yes", "on")


def _log_tokens(p: str, model: str, resp) -> None:
    """One-line usage/cost trace, gated behind RECAP_TOKEN_LOG=1."""
    if not _token_log_enabled():
        return
    try:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        pin = int(getattr(u, "prompt_tokens", 0) or 0)
        pout = int(getattr(u, "completion_tokens", 0) or 0)
        total = int(getattr(u, "total_tokens", 0) or (pin + pout))
        print(f"  * [{p}/{model}] ~{pin:,} in + ~{pout:,} out "
              f"(total ~{total:,} tokens)", flush=True)
    except Exception:
        pass


# Per-provider fallback when no model is configured.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "anthropic": "claude-3-5-sonnet-latest",
    "ollama": "qwen2.5",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-3.6-flash",
}

# Providers whose OpenAI-compatible endpoint supports
# response_format={"type": "json_object"}. DeepSeek does; Ollama's shim does
# not accept it reliably across versions, so we only ask where it is safe.
_JSON_MODE_PROVIDERS = {"deepseek", "openai", "groq"}


def _client_from(provider: str, model: str, base_url: str | None = None):
    """Lazily build a client and return (client, model).

    ``base_url`` is the provider endpoint from config; it wins over the env
    var so the control panel's Base URL field is honoured for every provider,
    not just Ollama.
    """
    provider = (provider or "").strip().lower()
    try:
        timeout = float(os.environ.get("LLM_TIMEOUT", "3600"))
    except ValueError:
        timeout = 3600.0

    if provider in ("", "none"):
        raise LLMError(
            "No LLM provider configured. Set LLM_PROVIDER (openai/anthropic/"
            "deepseek/groq/gemini/ollama) and the matching API key, OR provide "
            "a pre-written script/translation file (see README)."
        )

    if provider == "openai":
        import openai  # type: ignore

        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
            timeout=timeout,
        )
        model = model or os.environ.get("MODEL_NAME") or DEFAULT_MODELS["openai"]
        return client, model

    if provider == "deepseek":
        import openai  # type: ignore

        client = openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com/v1",
            timeout=timeout,
        )
        model = model or os.environ.get("MODEL_NAME") or DEFAULT_MODELS["deepseek"]
        return client, model

    if provider == "groq":
        # Free tier (no credit card): console.groq.com -> API Keys.
        # Very fast Llama on custom hardware; ~30 req/min is plenty for a recap
        # (~25 LLM calls per movie).
        import openai  # type: ignore

        client = openai.OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url=base_url
            or os.environ.get("GROQ_BASE_URL")
            or "https://api.groq.com/openai/v1",
            timeout=timeout,
        )
        model = model or os.environ.get("MODEL_NAME") or DEFAULT_MODELS["groq"]
        return client, model

    if provider == "gemini":
        # Google AI Studio free API key (aistudio.google.com -> Get API key).
        # Flash models keep a generous free tier (~1500 requests/day). This is
        # Google's OpenAI-compatible endpoint; model names like
        # gemini-3.6-flash and other current Flash models work here.
        import openai  # type: ignore

        client = openai.OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url=base_url
            or os.environ.get("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=timeout,
        )
        model = model or os.environ.get("MODEL_NAME") or DEFAULT_MODELS["gemini"]
        return client, model

    if provider == "anthropic":
        import anthropic  # type: ignore

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=timeout
        )
        model = model or os.environ.get("MODEL_NAME") or DEFAULT_MODELS["anthropic"]
        return client, model

    if provider == "ollama":
        import openai  # type: ignore

        base = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        client = openai.OpenAI(base_url=base, api_key="ollama", timeout=timeout)
        model = model or os.environ.get("MODEL_NAME") or DEFAULT_MODELS["ollama"]
        return client, model

    raise LLMError(f"Unknown LLM provider: {provider!r}")


def verify_model(cfg_llm: dict) -> None:
    """Fail fast (actionable error) before the long LLM passes.

    For Ollama this asks the server which models are already pulled. Without
    this check, generating on a model that was never pulled makes Ollama
    *silently download the model first* — the #1 cause of "stuck for an hour
    with no output" on a fresh setup.
    """
    provider = (cfg_llm.get("provider") or "").strip().lower()
    model = (cfg_llm.get("model") or "").strip()
    if provider != "ollama" or not model:
        return
    import json
    import urllib.error
    import urllib.request

    base = (
        (cfg_llm.get("base_url") or os.environ.get("OLLAMA_BASE_URL"))
        or "http://localhost:11434/v1"
    ).rstrip("/")
    try:
        with urllib.request.urlopen(base + "/models", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8") or "{}")
    except Exception as exc:
        raise LLMError(
            f"Ollama is not answering at {base} ({type(exc).__name__}).\n"
            "  Start it:  ollama serve   (keep it running)\n"
            f"  Pull the model:  ollama pull {model}"
        ) from exc
    ids = {m.get("id", "") for m in data.get("data", [])}
    # Ollama reports pulled models as "<name>:latest" on the OpenAI-compatible
    # endpoint while the config may say "qwen2.5" — the same model. Compare both
    # the raw id and the id minus a trailing ":latest" tag.
    known = set(ids)
    known |= {i.rsplit(":", 1)[0] for i in ids if i.rsplit(":", 1)[-1] == "latest"}
    if model not in known:
        shown = ", ".join(sorted(ids)) or "(none — first run: ollama pull <model>)"
        raise LLMError(
            f"Ollama is running but does not have model '{model}' yet.\n"
            f"  Models available right now: {shown}\n"
            f"  Run once:  ollama pull {model}\n"
            "(check with: ollama list)"
        )


def complete(
    provider: str,
    model: str,
    system: str,
    user: str,
    base_url: str | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Send a single completion (no history). Returns assistant text.

    ``json_mode`` asks the provider to guarantee syntactically valid JSON
    (DeepSeek / OpenAI / Groq support ``response_format``). The callers still
    parse defensively, so a provider that ignores the hint is harmless.

    Token economy (paid APIs like DeepSeek bill per token):
      * ``max_tokens`` caps the *output*. Callers pass a bound derived from the
        size they actually need (a section script needs ~150 tokens, not the
        provider's 4-8K default), so a model that starts rambling cannot burn
        a large bill. Fallback: env ``LLM_MAX_TOKENS`` or 4096.
      * ``temperature`` defaults to env ``LLM_TEMPERATURE`` (0.7), overridable
        per call; deterministic JSON passes can drop it (e.g. 0.2) which makes
        the model hit the requested shape first try instead of retrying.
      * An *empty* answer is never retried (the tokens were already spent and
        the model will likely answer empty again).

    Transient network / rate-limit failures are retried with backoff — a cloud
    provider hiccup two thirds of the way through a 20-section script pass
    should not throw the whole run away.
    """
    import time

    p = (provider or "").strip().lower()
    client, resolved_model = _client_from(provider, model, base_url)

    try:
        env_max = int(os.environ.get("LLM_MAX_TOKENS", "0") or "0")
    except ValueError:
        env_max = 0
    cap = max_tokens or env_max or 4096

    try:
        temp = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
    except ValueError:
        temp = 0.7
    if temperature is not None:
        temp = float(temperature)

    # Is this client OpenAI-compatible (chat.completions) or Anthropic?
    has_chat = hasattr(client, "chat") and hasattr(getattr(client, "chat", None),
                                                  "completions")
    has_messages = hasattr(client, "messages")

    kwargs: dict = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temp,
        "max_tokens": int(cap),
    }
    # deepseek-reasoner rejects temperature; keep the call bare.
    if "reasoner" in (resolved_model or ""):
        kwargs.pop("temperature", None)
    elif json_mode and p in _JSON_MODE_PROVIDERS:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        attempts = int(os.environ.get("LLM_RETRIES", "4"))
    except ValueError:
        attempts = 4
    attempts = max(1, attempts)

    for attempt in range(attempts):
        try:
            if has_chat:  # OpenAI-compatible (OpenAI / DeepSeek / Ollama / ...)
                resp = client.chat.completions.create(**kwargs)
                _log_tokens(p, resolved_model, resp)
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text
                raise LLMError(
                    "provider returned an empty message (tokens were billed but "
                    "nothing came back) — retry the run, or switch to a "
                    "stronger model if this repeats"
                )
            if not has_messages:
                raise LLMError(
                    "LLM client is neither OpenAI-compatible nor Anthropic — "
                    "cannot call it."
                )
            break  # Anthropic client -> handled below
        except LLMError as exc:
            raise exc  # empty message: do not burn tokens retrying
        except Exception as exc:
            if not _retryable(exc) or attempt >= attempts - 1:
                if p == "ollama":
                    raise LLMError(
                        f"Ollama request failed ({type(exc).__name__}: {exc}). "
                        "Make sure Ollama is running (`ollama serve`) and the model is "
                        f"pulled (`ollama pull {resolved_model}`), and that "
                        "OLLAMA_BASE_URL points at it."
                    ) from exc
                if p == "deepseek":
                    raise LLMError(
                        f"DeepSeek request failed ({type(exc).__name__}: {exc}).\n"
                        "  - check DEEPSEEK_API_KEY is set and has credit "
                        "(platform.deepseek.com -> Usage)\n"
                        f"  - model '{resolved_model}' should be deepseek-chat "
                        "or deepseek-reasoner"
                    ) from exc
                raise
            wait = min(2.0 * (2 ** attempt), 30.0)
            print(f"  * LLM call failed ({type(exc).__name__}); retry "
                  f"{attempt + 1}/{attempts} in {wait:.0f}s ...", flush=True)
            time.sleep(wait)

    # Anthropic
    resp = client.messages.create(
        model=resolved_model,
        max_tokens=int(cap),
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temp,
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
    if p == "groq":
        return bool(os.environ.get("GROQ_API_KEY"))
    if p == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    return False
