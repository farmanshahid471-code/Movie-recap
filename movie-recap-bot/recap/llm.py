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
import re
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str


class LLMError(RuntimeError):
    pass


def _retryable(exc: Exception) -> bool:
    """True for transient faults worth retrying (network, 429, 5xx).

    ``empty message`` is deliberately NOT here: retrying the SAME request
    would spend the tokens again for the same nothing. Empty answers go
    through the dedicated ladder in ``_chat_with_recovery`` instead, which
    changes something real at every step.
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

# Per-provider env var that names the model for THAT provider. `MODEL_NAME` is
# the legacy global and is only honoured when it is actually a model of the
# configured provider — this is the bug in the report:
#   "The supported API model names are deepseek-flash, deepseek-v4-pro, but
#    you passed gemini-3.1-flash-lite."
# A .env that set MODEL_NAME for the vision pass was handed to DeepSeek, the
# call failed 4 times with backoff, and the run died.
_PROVIDER_MODEL_ENV = {
    "openai": "OPENAI_MODEL",
    "deepseek": "DEEPSEEK_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "ollama": "OLLAMA_MODEL",
    "groq": "GROQ_MODEL",
    "gemini": "GEMINI_MODEL",
}

# Model strings that belong to another provider's family. Used to refuse a
# cross-provider name instead of spending retries on a request the endpoint
# can only answer with a 400.
_FAMILIES = {
    "gemini": ("gemini",),
    "openai": ("gpt", "o1", "o3", "o4", "chatgpt"),
    "anthropic": ("claude",),
    "deepseek": ("deepseek",),
    "groq": ("llama", "mixtral", "gemma", "qwen", "groq"),
    "ollama": ("llama", "qwen", "mistral", "gemma", "phi", "deepseek"),
}


def _provider_of_model(name: str) -> str:
    """Which provider a model string belongs to (``""`` when unknown)."""
    low = (name or "").strip().lower()
    if not low:
        return ""
    if "gemini" in low:
        return "gemini"
    if "claude" in low:
        return "anthropic"
    if "deepseek" in low:
        # deepseek-v4-pro served by a generic OpenAI-compatible proxy still
        # belongs to the deepseek family
        return "deepseek"
    if "gpt" in low or low.startswith(("o1", "o3", "o4")):
        return "openai"
    for prov, markers in _FAMILIES.items():
        if any(m in low for m in markers):
            return prov
    return ""


def _model_fits(provider: str, name: str) -> bool:
    """True when ``name`` can plausibly be served by ``provider``."""
    p = (provider or "").strip().lower()
    fam = _provider_of_model(name)
    if not name:
        return False
    if not fam:
        return True                     # unknown family: let the endpoint judge
    if p in ("", "none"):
        return False
    return fam == p or p in ("ollama", "groq", "openai")


# Endpoints that already told us which model names they accept, keyed by the
# base URL. Populated from the provider's error text ("The supported API model
# names are X, Y") so we never waste the retry budget on a name we now know
# the endpoint refuses.
_ENDPOINT_MODELS: dict[str, list[str]] = {}
_NOTED: set[str] = set()

_SUPPORTED_RE = re.compile(
    r"supported\s+(?:api\s+)?model\s*names?\s*(?:are|:)\s*([^\n.]+)", re.I)


def parse_supported_models(message: str) -> list[str]:
    """Model names a provider listed in its own error message.

    Handles the reported wording ("The supported API model names are
    deepseek-flash, deepseek-v4-pro, but you passed ...") as well as the
    common "supported models: a, b" / "available models are ..." variants.
    Returns [] when the message carries no such list.
    """
    try:
        text = str(message or "")
    except Exception:
        return []
    m = _SUPPORTED_RE.search(text)
    if not m:
        m = re.search(r"(?:available|valid|known)\s+models?\s*(?:are|:)\s*([^\n.]+)",
                      text, re.I)
    if not m:
        return []
    chunk = m.group(1)
    # Trim the tail of the sentence ("..., but you passed ...")
    chunk = re.split(r"\bbut\b|\bhowever\b|\byou passed\b", chunk, flags=re.I)[0]
    names: list[str] = []
    for piece in re.split(r"[,\s;|]+", chunk):
        name = piece.strip().strip("'\"`().")
        if not name or name.lower() in ("and", "or"):
            continue
        if re.fullmatch(r"[A-Za-z0-9._:/+-]{2,64}", name):
            names.append(name)
    return names


def remember_endpoint_models(base_url: str | None, names: list[str]) -> None:
    if base_url and names:
        _ENDPOINT_MODELS[str(base_url).rstrip("/")] = list(names)


def _endpoint_models(base_url: str | None) -> list[str]:
    return list(_ENDPOINT_MODELS.get(str(base_url).rstrip("/"), [])) if base_url else []


def endpoint_models(base_url: str | None, api_key: str | None,
                    timeout: float = 10.0) -> list[str]:
    """Ask an OpenAI-compatible endpoint which models it serves (best effort).

    A custom ``base_url`` (a proxy, a self-hosted gateway, a regional router)
    does not necessarily serve the model names this project ships as
    defaults. Querying ``/models`` costs one cheap request and turns a run
    that dies after four retries into a run that picks a name the endpoint
    actually has. Failures are silent: the caller keeps the configured name.
    """
    import json
    import urllib.request

    url = (base_url or "").rstrip("/")
    if not url:
        return []
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    if not url.endswith("/v1") and "/v1/" not in url:
        url = f"{url}/v1"
    url = f"{url}/models"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {(api_key or '').strip()}",
        "User-Agent": "MovieRecap/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return []
    rows = data.get("data") if isinstance(data, dict) else None
    names: list[str] = []
    for row in rows or []:
        if isinstance(row, dict) and row.get("id"):
            names.append(str(row["id"]))
        elif isinstance(row, str):
            names.append(row)
    return names


def resolve_model(provider: str, model: str, base_url: str | None = None,
                  *, quiet: bool = False) -> str:
    """Pick the model string that matches the provider that will serve it.

    Precedence: the configured model if it fits the provider, else the
    provider-specific env var (DEEPSEEK_MODEL, GEMINI_MODEL, ...), else the
    legacy MODEL_NAME when it fits, else the provider default. A mismatched
    name is never silently shipped to an endpoint that cannot serve it — the
    substitution is printed once so the user can fix the config.
    """
    p = (provider or "").strip().lower()
    chosen = ""
    source = ""
    candidates: list[tuple[str, str]] = []
    if model:
        candidates.append((str(model).strip(), "config"))
    env_key = _PROVIDER_MODEL_ENV.get(p)
    if env_key and os.environ.get(env_key):
        candidates.append((os.environ[env_key].strip(), env_key))
    if os.environ.get("MODEL_NAME"):
        candidates.append((os.environ["MODEL_NAME"].strip(), "MODEL_NAME"))
    default = DEFAULT_MODELS.get(p, "")
    if default:
        candidates.append((default, "provider default"))

    for name, src in candidates:
        if not name:
            continue
        if _model_fits(p, name):
            chosen, source = name, src
            break
        if not quiet and f"mismatch:{p}:{name}" not in _NOTED:
            _NOTED.add(f"mismatch:{p}:{name}")
            print(f"  ! [llm] model {name!r} (from {src}) is not a {p} model "
                  f"— ignoring it for provider {p!r}", flush=True)

    if not chosen:
        chosen = default or (model or "").strip()
        source = source or "fallback"

    # A custom endpoint decides for itself which names exist: prefer one it
    # already told us it accepts, and remember the correction.
    known = _endpoint_models(base_url)
    fitted = [n for n in known if _model_fits(p, n)] or known
    if chosen and known and chosen not in known and fitted:
        replacement = _pick_endpoint_model(p, fitted, base_url)
        if not quiet and f"endpoint:{base_url}" not in _NOTED:
            _NOTED.add(f"endpoint:{base_url}")
            print(f"  ! [llm] endpoint {base_url} does not list model "
                  f"{chosen!r}; it supports {', '.join(fitted[:6])} — using "
                  f"{replacement!r}. Make it permanent with llm.model: "
                  f"{replacement} in config.yaml (or MODEL_NAME={replacement} "
                  "in .env).", flush=True)
        chosen = replacement
    return chosen


def _pick_endpoint_model(provider: str, names: list[str],
                         base_url: str | None) -> str:
    """Choose among the endpoint's own names (provider family first)."""
    prefer = _FAMILIES.get((provider or "").strip().lower(), ())
    for name in names:
        low = name.lower()
        if any(m in low for m in prefer):
            return name
    for name in names:
        if "embed" not in name.lower():
            return name
    return names[0]


def _model_error_model(exc: Exception) -> str | None:
    """The replacement model an endpoint's rejection suggests, if any."""
    msg = f"{type(exc).__name__}: {exc}"
    low = msg.lower()
    markers = ("supported api model names", "supported model names",
               "model not found", "does not exist", "unknown model",
               "invalid model", "not a valid model", "no such model")
    if not any(m in low for m in markers):
        return None
    names = parse_supported_models(msg)
    return names[0] if names else None


def _client_from(provider: str, model: str, base_url: str | None = None):
    """Lazily build a client and return (client, model).

    ``base_url`` is the provider endpoint from config; it wins over the env
    var so the control panel's Base URL field is honoured for every provider,
    not just Ollama.
    """
    provider = (provider or "").strip().lower()
    # 180s per chat request: a completion NEVER needs an hour. The old 3600s
    # default turned one dead socket (VPN drop, machine sleep, provider
    # hiccup) into a SILENT multi-hour stall -- x4 retries = 4+ hours of no
    # output, then a raw ConnectionAbortedError killed the run. A dead
    # connection now raises in minutes, gets retried with backoff, and only
    # then fails with a clear message. Override with LLM_TIMEOUT if needed.
    try:
        timeout = float(os.environ.get("LLM_TIMEOUT", "180"))
    except ValueError:
        timeout = 180.0

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
        model = resolve_model("openai", model, base_url)
        return client, model

    if provider == "deepseek":
        import openai  # type: ignore

        client = openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com",
            timeout=timeout,
        )
        model = resolve_model("deepseek", model, base_url)
        return client, model

    if provider == "gemini":
        import openai  # type: ignore

        client = openai.OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url=base_url
            or os.environ.get("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=timeout,
        )
        model = resolve_model("gemini", model, base_url)
        return client, model

    if provider == "anthropic":
        import anthropic  # type: ignore

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=timeout
        )
        model = resolve_model("anthropic", model, base_url)
        return client, model

    if provider == "ollama":
        import openai  # type: ignore

        base = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        client = openai.OpenAI(base_url=base, api_key="ollama", timeout=timeout)
        model = resolve_model("ollama", model, base_url)
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


def _direct_http_completion(
    provider: str,
    base_url: str | None,
    api_key: str | None,
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
    timeout: float = 180.0,
) -> str | None:
    """Zero-dependency HTTP fallback for OpenAI-compatible endpoints.

    Bypasses SDK version incompatibilities or logging/process errors.
    """
    url = (base_url or "").rstrip("/")
    if not url:
        if provider == "deepseek":
            url = "https://api.deepseek.com"
        elif provider == "groq":
            url = "https://api.groq.com/openai/v1"
        elif provider == "openai":
            url = "https://api.openai.com/v1"
        elif provider == "ollama":
            url = "http://localhost:11434/v1"

    if url.endswith("/chat/completions"):
        endpoint = url
    elif url.endswith("/v1"):
        endpoint = f"{url}/chat/completions"
    elif "deepseek.com" in url:
        endpoint = f"{url}/chat/completions"
    else:
        endpoint = f"{url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "MovieRecap/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    payload: dict = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # 1. Try requests library
    try:
        import requests
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices") or []
            if choices:
                return (choices[0].get("message", {}).get("content") or "").strip()
        else:
            print(f"  ! Direct HTTP status {resp.status_code}: {resp.text[:200]}", flush=True)
    except ImportError:
        pass
    except Exception as exc:
        print(f"  ! Direct requests attempt failed ({type(exc).__name__}: {exc})", flush=True)

    # 2. Try standard library urllib
    try:
        import json
        import ssl
        import urllib.request

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                body = json.loads(r.read().decode("utf-8"))
                choices = body.get("choices") or []
                if choices:
                    return (choices[0].get("message", {}).get("content") or "").strip()
        except urllib.error.HTTPError as h_err:
            print(f"  ! Direct urllib HTTPError {h_err.code}: {h_err.read().decode(errors='replace')[:200]}", flush=True)
        except Exception:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                body = json.loads(r.read().decode("utf-8"))
                choices = body.get("choices") or []
                if choices:
                    return (choices[0].get("message", {}).get("content") or "").strip()
    except Exception as exc:
        print(f"  ! Direct urllib fallback failed ({type(exc).__name__}: {exc})", flush=True)

    return None


# ---------------------------------------------------------------------------
# EMPTY-ANSWER RECOVERY
# ---------------------------------------------------------------------------
# The reported run:
#   * [llm] provider=deepseek model='deepseek-flash' base_url=https://api.deepseek.com/v1
#   ! [story] ... (LLMError: provider returned an empty message (tokens were
#     billed but nothing came back)) — falling back to the per-beat writer
#   ! [beats] ... falling back to chunk path
#   ... chunk 1/36 via deepseek-chat ... ERROR: LLMError: provider returned an
#     empty message ...
# Every request came back HTTP 200, billed, and EMPTY, so three passes died in
# a row and the run ended with nothing. An empty 200 has several very different
# causes and each has its own fix, so the wrapper walks a ladder instead of
# raising on the first one:
#   1. a REASONING model spent the whole output budget thinking
#      (finish_reason=length / reasoning_tokens)      -> raise the budget, retry;
#   2. the endpoint ignored response_format=json_object and answered empty
#                                                     -> retry without it;
#   3. the gateway only fills message.content when streaming
#                                                     -> retry with stream=True;
#   4. the SDK transport is at fault                   -> retry the raw HTTP path;
#   5. the MODEL itself answers nothing on this endpoint -> switch to another
#      name the endpoint lists, and remember the broken one for the rest of the
#      run (so the next 36 chunks do not pay for the same discovery).
# Only when all five fail does the error surface — and it now names the model,
# the finish_reason, the token usage and the exact next step.
_EMPTY_STREAK: dict[str, int] = {}
_BAD_MODELS: set[str] = set()
_CAP_FLOOR: dict[str, int] = {}
_EMPTY_NOTED: set[str] = set()
_EMPTY_STREAK_LIMIT = 2          # empties before a model is considered broken
_MAX_OUTPUT_TOKENS = 16384       # ceiling for the escalated output budget


def _get(obj, name: str, default=None):
    """Attribute or key access — responses arrive as objects or plain dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _call_key(provider: str, base_url: str | None, model: str) -> str:
    return f"{provider}|{(base_url or '').rstrip('/')}|{model}"


def _message_text(msg) -> str:
    """Assistant text out of a message, whatever shape the gateway used."""
    content = _get(msg, "content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(_get(item, "text", "") or ""))
        return "".join(parts).strip()
    if content is None:
        alt = _get(msg, "text")
        if isinstance(alt, str):
            return alt.strip()
    return str(content or "").strip()


def _message_reasoning(msg) -> str:
    for attr in ("reasoning_content", "reasoning"):
        val = _get(msg, attr)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _response_facts(resp) -> dict:
    """finish_reason + token usage of a response, extracted defensively."""
    facts = {"finish_reason": "", "completion_tokens": 0, "reasoning_tokens": 0,
             "content_chars": 0, "has_reasoning": False}
    try:
        choices = _get(resp, "choices") or []
        if not choices:
            return facts
        ch = choices[0]
        msg = _get(ch, "message")
        facts["content_chars"] = len(_message_text(msg))
        facts["has_reasoning"] = bool(_message_reasoning(msg))
        facts["finish_reason"] = str(_get(ch, "finish_reason") or "")
        usage = _get(resp, "usage")
        if usage is not None:
            facts["completion_tokens"] = int(_get(usage, "completion_tokens", 0) or 0)
            details = _get(usage, "completion_tokens_details")
            if details is not None:
                facts["reasoning_tokens"] = int(
                    _get(details, "reasoning_tokens", 0) or 0)
    except Exception:
        pass
    return facts


def _empty_note(key: str, message: str) -> None:
    """Print a recovery note once per (kind, model)."""
    if key in _EMPTY_NOTED:
        return
    _EMPTY_NOTED.add(key)
    print(message, flush=True)


def _try_chat(client, kwargs: dict, p: str, model: str) -> tuple[str, dict]:
    resp = client.chat.completions.create(**kwargs)
    _log_tokens(p, model, resp)
    choices = _get(resp, "choices") or []
    msg = _get(choices[0], "message") if choices else None
    return _message_text(msg), _response_facts(resp)


def _stream_text(client, kwargs: dict) -> str:
    """Retry as a stream: some gateways only fill content in streaming mode."""
    try:
        stream = client.chat.completions.create(**{**kwargs, "stream": True})
    except Exception:
        return ""
    parts: list[str] = []
    try:
        for chunk in stream:
            for ch in (_get(chunk, "choices") or []):
                piece = _get(_get(ch, "delta"), "content")
                if isinstance(piece, str) and piece:
                    parts.append(piece)
    except Exception:
        return "".join(parts).strip()
    return "".join(parts).strip()


def _next_model(provider: str, base_url: str | None, current: str) -> str:
    """Another model name the SAME endpoint can serve, or ``""``."""
    p = (provider or "").strip().lower()
    key = os.environ.get(f"{p.upper()}_API_KEY") or os.environ.get("LLM_API_KEY")
    known = [n for n in _endpoint_models(base_url) if n and n != current]
    if not known and base_url:
        # ask the endpoint once, best effort (it may list models even when its
        # chat endpoint answers empty)
        probed = [n for n in endpoint_models(base_url, key) if n and n != current]
        if probed:
            remember_endpoint_models(base_url, [current] + probed)
            known = probed
    if not known and p == "deepseek" and (not base_url or "deepseek.com" in base_url):
        # official endpoint (or a probe that could not answer): the two models
        # it documents, so a name it does not serve cannot end the run
        known = [n for n in ("deepseek-chat", "deepseek-reasoner") if n != current]
    usable = [n for n in known
              if "embed" not in n.lower()
              and _call_key(p, base_url, n) not in _BAD_MODELS]
    if not usable:
        return ""
    # A model that just answered NOTHING is usually an over-loaded little
    # variant ("-flash", "-mini", "-lite", 3B/7B): when a bigger sibling is on
    # the endpoint, try that first.
    weak = ("flash", "mini", "lite", "small", "turbo", "nano", "8b", "7b", "4b", "3b")

    def _rank(name: str) -> int:
        low = name.lower()
        return 1 if any(w in low for w in weak) else 0

    prefer = _FAMILIES.get(p, ())
    family = [n for n in usable if any(m in n.lower() for m in prefer)]
    ordered = sorted(family or usable, key=_rank)
    return ordered[0] if ordered else ""


def _chat_with_recovery(
    client, kwargs: dict, *, p: str, base_url: str | None, api_key: str | None,
    timeout: float, json_mode: bool, model: str, allow_switch: bool = True,
) -> tuple[str, str, dict, int]:
    """One completion plus the empty-answer ladder. -> (text, model, facts, cap).

    ``text`` is ``""`` only when every rung failed; ``facts`` then describes the
    last empty answer so the caller can raise an actionable error.
    """
    tried: list[str] = []
    cap = int(kwargs.get("max_tokens") or 0)

    text, facts = _try_chat(client, kwargs, p, model)
    if text:
        return text, model, facts, cap

    fr = str(facts.get("finish_reason") or "")
    comp = int(facts.get("completion_tokens") or 0)
    reason = int(facts.get("reasoning_tokens") or 0)
    cut_short = (
        fr == "length"
        or (cap and comp >= cap)
        or (reason and reason >= max(cap - 32, 1))
        or facts.get("has_reasoning")
    )
    if cut_short:
        bigger = min(max(cap * 4, 2048), _MAX_OUTPUT_TOKENS)
        if bigger > cap:
            _empty_note(f"escalate:{model}:{cap}",
                        f"  ! [llm] '{model}' came back empty after spending its "
                        f"whole output budget (finish_reason={fr or 'length'}, "
                        f"{reason or comp} tokens of {cap}"
                        f"{', reasoning-only' if reason or facts.get('has_reasoning') else ''}"
                        f") — retrying with max_tokens={bigger}")
            kwargs = {**kwargs, "max_tokens": bigger}
            cap = bigger
            tried.append("a 4x bigger output budget")
            text, facts = _try_chat(client, kwargs, p, model)
            if text:
                return text, model, facts, cap

    if json_mode and "response_format" in kwargs:
        _empty_note(f"nojson:{model}",
                    f"  * [llm] '{model}' answered empty with "
                    "response_format=json_object — retrying without it "
                    "(the callers parse defensively)")
        kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
        tried.append("no response_format")
        text, facts = _try_chat(client, kwargs, p, model)
        if text:
            return text, model, facts, cap

    _empty_note(f"stream:{model}",
                f"  * [llm] '{model}' answered empty — retrying as a stream")
    tried.append("a streamed request")
    streamed = _stream_text(client, kwargs)
    if streamed:
        return streamed, model, facts, cap

    tried.append("the raw HTTP path")
    raw = _direct_http_completion(
        provider=p,
        base_url=base_url or os.environ.get(f"{p.upper()}_BASE_URL"),
        api_key=api_key,
        model=model,
        messages=kwargs.get("messages") or [],
        temperature=kwargs.get("temperature"),
        max_tokens=cap,
        json_mode=False,
        timeout=timeout,
    )
    if raw:
        return raw, model, facts, cap

    if allow_switch:
        alt = _next_model(p, base_url, model)
        if alt:
            _empty_note(f"switch:{model}->{alt}",
                        f"  ! [llm] model '{model}' answered NOTHING "
                        f"(tried {', '.join(tried)}) — switching to {alt!r} for "
                        "the rest of this run. Make it permanent: set "
                        f"llm.model: {alt} in config.yaml "
                        f"(or MODEL_NAME={alt} in .env).")
            _BAD_MODELS.add(_call_key(p, base_url, model))
            alt_kwargs = {**kwargs, "model": alt}
            text2, facts2 = _try_chat(client, alt_kwargs, p, alt)
            if not text2 and "response_format" in alt_kwargs:
                text2, facts2 = _try_chat(
                    client,
                    {k: v for k, v in alt_kwargs.items() if k != "response_format"},
                    p, alt)
            if text2:
                return text2, alt, facts2, cap
    return "", model, facts, cap


def _empty_strike(provider: str, base_url: str | None, model: str) -> int:
    """Count consecutive empty answers for this model/endpoint pair."""
    key = _call_key(provider, base_url, model)
    n = _EMPTY_STREAK.get(key, 0) + 1
    _EMPTY_STREAK[key] = n
    if n >= _EMPTY_STREAK_LIMIT:
        _BAD_MODELS.add(key)
    return n


def _empty_error(provider: str, base_url: str | None, model: str,
                 facts: dict) -> str:
    """The final, actionable error once every rung of the ladder failed."""
    fr = str(facts.get("finish_reason") or "?")
    comp = int(facts.get("completion_tokens") or 0)
    reason = int(facts.get("reasoning_tokens") or 0)
    known = _endpoint_models(base_url)
    lines = [
        f"provider returned an empty message for model '{model}' "
        "(tokens were billed but nothing came back).",
        f"  - last answer: finish_reason={fr}, completion_tokens={comp}"
        + (f" ({reason} of them reasoning tokens)" if reason else ""),
        "  - tried: a 4x bigger output budget, no response_format, a streamed "
        "request, and the raw HTTP path",
    ]
    if known:
        lines.append(f"  - this endpoint lists: {', '.join(known[:8])}")
    alt = _next_model(provider, base_url, model)
    if alt:
        lines.append("  - next step: pin a model that answers — "
                     f"llm.model: {alt} in config.yaml (or MODEL_NAME={alt} in "
                     ".env)")
    else:
        lines.append("  - next step: try another model (llm.model: ...), a "
                     "bigger LLM_MAX_TOKENS, or another provider")
    lines.append("  - a reasoning-first model that runs out of output budget "
                 "is the usual cause: LLM_MAX_TOKENS=8192 (or more) fixes it")
    lines.append("  - an account out of credit can also answer 200 with no "
                 "content: check the provider dashboard")
    return "\n".join(lines)


def _wants_max_completion_tokens(exc: Exception) -> bool:
    """Some (reasoning) models reject ``max_tokens`` and want the newer name."""
    low = f"{type(exc).__name__}: {exc}".lower()
    if "max_completion_tokens" in low:
        return True
    if "max_tokens" not in low:
        return False
    return any(w in low for w in ("unsupported", "not supported", "unknown",
                                  "invalid", "unrecognized"))


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
      * An *empty* answer walks the recovery ladder above (bigger budget,
        no response_format, streaming, raw HTTP, another model on the same
        endpoint) instead of being retried unchanged — see the EMPTY-ANSWER
        RECOVERY block. Only when every rung fails is it fatal.

    Transient network / rate-limit failures are retried with backoff — a cloud
    provider hiccup two thirds of the way through a 20-section script pass
    should not throw the whole run away.
    """
    import time

    p = (provider or "").strip().lower()
    client, resolved_model = _client_from(provider, model, base_url)

    # Custom endpoint (proxy / self-hosted gateway / regional router)? Ask it
    # ONCE which model names it serves. This is the difference between a 400
    # that costs four retries and a run that quietly uses a name the endpoint
    # actually has.
    if base_url and p in (_JSON_MODE_PROVIDERS | {"gemini", "ollama"}):
        if not _endpoint_models(base_url) and f"probe:{base_url}" not in _NOTED:
            _NOTED.add(f"probe:{base_url}")
            _names = endpoint_models(
                base_url,
                os.environ.get(f"{p.upper()}_API_KEY") or os.environ.get("LLM_API_KEY"),
            )
            if _names:
                remember_endpoint_models(base_url, _names)
                _resolved = resolve_model(p, resolved_model, base_url)
                if _resolved and _resolved != resolved_model:
                    resolved_model = _resolved
                    print(f"  * [llm] endpoint {base_url} serves "
                          f"{len(_names)} model(s); using {resolved_model!r}",
                          flush=True)

    if f"diag:{p}:{base_url}:{resolved_model}" not in _NOTED:
        _NOTED.add(f"diag:{p}:{base_url}:{resolved_model}")
        print(f"  * [llm] provider={p or '(none)'} model={resolved_model!r} "
              f"base_url={base_url or '(provider default)'}", flush=True)

    try:
        timeout = float(os.environ.get("LLM_TIMEOUT", "180"))
    except ValueError:
        timeout = 180.0

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

    # A model this endpoint already answered NOTHING with (twice) is not worth
    # another billed call: switch to another name the endpoint serves and say
    # so once. This is what keeps the 36-chunk summarizer from re-discovering
    # the same broken model on every chunk.
    if _call_key(p, base_url, resolved_model) in _BAD_MODELS:
        _alt = _next_model(p, base_url, resolved_model)
        if _alt and _alt != resolved_model:
            _empty_note(f"skipbad:{p}:{base_url}:{resolved_model}",
                        f"  * [llm] skipping {resolved_model!r} (it returned "
                        f"empty answers earlier on this endpoint) — using "
                        f"{_alt!r} instead")
            resolved_model = _alt
            kwargs["model"] = resolved_model

    # A model that needed a bigger output budget earlier in the run keeps it:
    # re-learning that on every chunk would be a wasted (billed) call each time.
    _floor = _CAP_FLOOR.get(_call_key(p, base_url, resolved_model), 0)
    if _floor > int(kwargs.get("max_tokens") or 0):
        kwargs["max_tokens"] = _floor

    try:
        attempts = int(os.environ.get("LLM_RETRIES", "4"))
    except ValueError:
        attempts = 4
    attempts = max(1, attempts)

    for attempt in range(attempts):
        try:
            if has_chat:  # OpenAI-compatible (OpenAI / DeepSeek / Ollama / ...)
                try:
                    text, _used_model, _facts, _cap = _chat_with_recovery(
                        client, kwargs, p=p, base_url=base_url,
                        api_key=(os.environ.get(f"{p.upper()}_API_KEY")
                                 or os.environ.get("LLM_API_KEY")),
                        timeout=timeout, json_mode=json_mode,
                        model=resolved_model,
                    )
                except (TypeError, Exception) as inner_exc:
                    # MODEL-NAME RECOVERY (the reported crash): the endpoint
                    # answered "The supported API model names are X, Y, but you
                    # passed Z". Instead of burning four retries and dying,
                    # remember the names it does support and retry with one.
                    _replacement = _model_error_model(inner_exc)
                    if _replacement:
                        _names = parse_supported_models(str(inner_exc)) or [_replacement]
                        remember_endpoint_models(base_url, _names)
                        _resolution = resolve_model(p, _replacement, base_url)
                        if _resolution and _resolution != resolved_model:
                            print(f"  ! [llm] {p} refused model {resolved_model!r} "
                                  f"— {str(inner_exc)[:200]}", flush=True)
                            print(f"  * [llm] the endpoint supports "
                                  f"{', '.join(_names[:6])}; retrying with "
                                  f"{_resolution!r}. Make it permanent: set "
                                  f"llm.model: {_resolution} in config.yaml "
                                  f"(or MODEL_NAME={_resolution} in .env).",
                                  flush=True)
                            resolved_model = _resolution
                            kwargs["model"] = resolved_model
                            continue
                    if _wants_max_completion_tokens(inner_exc) and "max_tokens" in kwargs:
                        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                        print("  * [llm] this model wants "
                              "'max_completion_tokens' instead of 'max_tokens' "
                              "— retrying.", flush=True)
                        continue
                    if isinstance(inner_exc, TypeError) or not _retryable(inner_exc):
                        print(f"  * Note: OpenAI SDK error ({type(inner_exc).__name__}: {inner_exc}); switching to direct HTTP request...", flush=True)
                        fallback_text = _direct_http_completion(
                            provider=p,
                            base_url=base_url or os.environ.get(f"{p.upper()}_BASE_URL"),
                            api_key=os.environ.get(f"{p.upper()}_API_KEY") or os.environ.get("LLM_API_KEY"),
                            model=resolved_model,
                            messages=kwargs["messages"],
                            temperature=kwargs.get("temperature"),
                            max_tokens=kwargs.get("max_tokens"),
                            json_mode=json_mode and p in _JSON_MODE_PROVIDERS,
                            timeout=timeout,
                        )
                        if fallback_text:
                            return fallback_text
                    raise inner_exc
                if text:
                    resolved_model = _used_model
                    kwargs["model"] = resolved_model
                    if _cap > int(kwargs.get("max_tokens") or 0):
                        # remember the output budget this model needed
                        kwargs["max_tokens"] = _cap
                        _CAP_FLOOR[_call_key(p, base_url, resolved_model)] = _cap
                    _EMPTY_STREAK.pop(_call_key(p, base_url, resolved_model), None)
                    return text
                # 200 with NO content, on every rung of the ladder
                _empty_strike(p, base_url, resolved_model)
                raise LLMError(_empty_error(p, base_url, resolved_model, _facts))
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
                    _known = _endpoint_models(base_url)
                    _hint = (f"  - the endpoint at {base_url or 'api.deepseek.com'} "
                             f"lists: {', '.join(_known[:8])}\n" if _known else "")
                    raise LLMError(
                        f"DeepSeek request failed ({type(exc).__name__}: {exc}).\n"
                        "  - check DEEPSEEK_API_KEY is set and has credit "
                        "(platform.deepseek.com -> Usage)\n"
                        f"  - model '{resolved_model}' should be one the "
                        "endpoint actually serves (deepseek-chat / "
                        "deepseek-reasoner on the official API; whatever the "
                        "proxy lists on a custom base_url)\n" + _hint
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
