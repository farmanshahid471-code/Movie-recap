"""Tests for the empty-answer recovery ladder (recap/llm.py).

The reported run:

    * [llm] provider=deepseek model='deepseek-flash' base_url=https://api.deepseek.com/v1
    ! [story] English story writer failed (LLMError: provider returned an empty
      message (tokens were billed but nothing came back)) — falling back to the
      per-beat writer
    ! [beats] English beat generation failed (LLMError: ...) — falling back to
      chunk path
    * Summarizing 36 English chunks via deepseek/deepseek-chat ...
      ... chunk 1/36 ... ERROR: LLMError: provider returned an empty message ...
    !!! run failed

Every request came back HTTP 200, billed, and EMPTY, so three passes died in a
row and nothing was produced. An empty 200 has several very different causes,
so the wrapper now walks a ladder instead of raising the same error four times:
a bigger output budget (reasoning models), no ``response_format`` (gateways
that ignore JSON mode), a streamed request, the raw HTTP path, and finally
another model name the SAME endpoint serves (remembered for the rest of the
run). Only when every rung fails is it fatal — and the error names the model,
the finish_reason, the usage and the next step.

Run:  python -m pytest tests/test_llm_recovery.py -q
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap import llm  # noqa: E402

GW = "http://127.0.0.1:9/v1"          # refuses instantly: keeps tests offline


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

def _resp(text: str = "", *, finish: str = "stop", ct: int = 0, rt: int = 0,
          reasoning: str | None = None, parts: bool = False) -> dict:
    content = [{"type": "text", "text": text}] if parts else text
    msg: dict = {"content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {
        "choices": [{"message": msg, "finish_reason": finish}],
        "usage": {"completion_tokens": ct,
                  "completion_tokens_details": {"reasoning_tokens": rt}},
    }


class _Completions:
    def __init__(self, handler):
        self._handler = handler
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self._handler(kwargs)


class _Client:
    def __init__(self, handler):
        self._completions = _Completions(handler)
        self.chat = type("_Chat", (), {})()
        self.chat.completions = self._completions

    @property
    def calls(self) -> list[dict]:
        return self._completions.calls


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("_EMPTY_STREAK", "_BAD_MODELS", "_CAP_FLOOR", "_EMPTY_NOTED",
                 "_ENDPOINT_MODELS", "_NOTED"):
        getattr(llm, name).clear()
    monkeypatch.delenv("RECAP_TOKEN_LOG", raising=False)
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    yield


def _install(monkeypatch, handler, model: str = "deepseek-flash") -> _Client:
    client = _Client(handler)
    monkeypatch.setattr(llm, "_client_from",
                        lambda p, m, b=None: (client, model))
    return client


def _call(**kw) -> str:
    with redirect_stdout(io.StringIO()):
        return llm.complete("deepseek", kw.pop("model", "deepseek-flash"),
                            "system", "user", **kw)


# ---------------------------------------------------------------------------
# 1. reading the answer
# ---------------------------------------------------------------------------

def test_content_parts_are_read_not_treated_as_empty(monkeypatch) -> None:
    """Some gateways return content as [{type: text, text: ...}]."""
    _install(monkeypatch, lambda kw: _resp("Hello world.", parts=True))
    assert _call() == "Hello world."
    print("ok: list-of-parts content is read, not mistaken for empty")


# ---------------------------------------------------------------------------
# 2. rung 1 — the reasoning model ate the output budget
# ---------------------------------------------------------------------------

def test_empty_answer_escalates_the_output_budget(monkeypatch) -> None:
    seen: list[int] = []

    def handler(kw):
        seen.append(int(kw.get("max_tokens") or 0))
        if len(seen) == 1:
            # finish_reason=length and every completion token was reasoning:
            # the classic "billed but nothing came back"
            return _resp(finish="length", ct=kw["max_tokens"],
                         rt=kw["max_tokens"], reasoning="thinking " * 200)
        return _resp("The pilot wakes up and the forest is quiet.")

    monkeypatch.setenv("DEEPSEEK_MAX_TOKENS", "0")  # test the escalation ladder
    client = _install(monkeypatch, handler)
    out = _call(max_tokens=512, json_mode=True)
    assert out.startswith("The pilot wakes up")
    assert seen[0] == 512 and seen[1] > 512, seen
    # ...and the bigger budget is remembered for the rest of the run (the
    # summarizer makes dozens of calls; re-learning it per call = billed waste)
    client.calls.clear()
    _call(max_tokens=512, json_mode=True)
    assert int(client.calls[0]["max_tokens"]) > 512, client.calls[0]
    print(f"ok: empty 'length' answer -> max_tokens {seen[0]} -> {seen[1]}, remembered")


def test_reasoning_only_answer_is_treated_as_a_cut_off(monkeypatch) -> None:
    """reasoning_content present + empty content = budget problem, not a stub."""
    def handler(kw):
        if int(kw.get("max_tokens") or 0) <= 2048:
            return _resp(finish="stop", ct=1200, rt=1200, reasoning="hmm " * 50)
        return _resp("She finds the letter.")

    monkeypatch.setenv("DEEPSEEK_MAX_TOKENS", "0")  # test the escalation ladder
    client = _install(monkeypatch, handler)
    assert _call(max_tokens=1024) == "She finds the letter."
    assert len(client.calls) == 2
    print("ok: reasoning-only answer escalates instead of failing")


# ---------------------------------------------------------------------------
# 3. rung 2 — the endpoint ignores response_format
# ---------------------------------------------------------------------------

def test_empty_answer_drops_json_mode(monkeypatch) -> None:
    def handler(kw):
        if "response_format" in kw:
            return _resp()                       # 200, billed, empty
        return _resp('{"sentences": ["She runs."]}')

    client = _install(monkeypatch, handler)
    out = _call(json_mode=True, max_tokens=256)
    assert out == '{"sentences": ["She runs."]}'
    assert "response_format" in client.calls[0]
    assert "response_format" not in client.calls[1], client.calls[1]
    print("ok: JSON-mode empty answer retried without response_format")


# ---------------------------------------------------------------------------
# 4. rung 3 — the gateway only fills content when streaming
# ---------------------------------------------------------------------------

def test_empty_answer_retries_as_a_stream(monkeypatch) -> None:
    def handler(kw):
        if kw.get("stream"):
            return iter([
                {"choices": [{"delta": {"content": "streamed "}}]},
                {"choices": [{"delta": {"content": "text."}}]},
                {"choices": [{"delta": {}}]},
            ])
        return _resp()

    client = _install(monkeypatch, handler)
    assert _call(max_tokens=256) == "streamed text."
    assert any(c.get("stream") for c in client.calls)
    print("ok: empty answer recovered by streaming")


# ---------------------------------------------------------------------------
# 5. rung 5 — the model itself answers nothing: switch, and remember
# ---------------------------------------------------------------------------

def test_empty_model_is_replaced_by_another_endpoint_model(monkeypatch) -> None:
    llm.remember_endpoint_models(GW, ["deepseek-flash", "deepseek-v4-pro"])

    def handler(kw):
        if kw["model"] == "deepseek-flash":
            return _resp()
        return _resp("v4 answers fine.")

    _install(monkeypatch, handler)
    out = _call(base_url=GW, json_mode=True, max_tokens=256)
    assert out == "v4 answers fine."
    bad = llm._call_key("deepseek", GW, "deepseek-flash")
    assert bad in llm._BAD_MODELS, llm._BAD_MODELS
    print("ok: a model that answers nothing is replaced on the same endpoint")


def test_broken_model_is_skipped_on_later_calls(monkeypatch) -> None:
    """The 36-chunk cascade: call 2 must not re-discover the dead model."""
    llm.remember_endpoint_models(GW, ["deepseek-flash", "deepseek-v4-pro"])

    def handler(kw):
        if kw["model"] == "deepseek-flash":
            return _resp()
        return _resp("Good answer.")

    client = _install(monkeypatch, handler)
    assert _call(base_url=GW, max_tokens=256) == "Good answer."       # discovers
    client.calls.clear()
    assert _call(base_url=GW, max_tokens=256) == "Good answer."       # skips it
    assert client.calls[0]["model"] == "deepseek-v4-pro", client.calls[0]
    print("ok: later calls start on the working model (no repeat discovery)")


# ---------------------------------------------------------------------------
# 6. every rung failed -> an actionable error
# ---------------------------------------------------------------------------

def test_every_rung_failed_raises_an_actionable_error(monkeypatch) -> None:
    llm.remember_endpoint_models(GW, ["deepseek-flash", "deepseek-v4-pro"])
    _install(monkeypatch, lambda kw: _resp(finish="stop"))
    with pytest.raises(llm.LLMError) as ei:
        _call(base_url=GW, json_mode=True, max_tokens=256)
    msg = str(ei.value)
    assert "deepseek-flash" in msg
    assert "finish_reason=stop" in msg
    assert "deepseek-v4-pro" in msg, msg          # the next step names a model
    assert "LLM_MAX_TOKENS" in msg
    print("ok: exhausted ladder -> error names model, reason and next step")


def test_error_lists_the_endpoints_supported_models(monkeypatch) -> None:
    llm.remember_endpoint_models(GW, ["deepseek-flash", "deepseek-v4-pro"])
    _install(monkeypatch, lambda kw: _resp())
    with pytest.raises(llm.LLMError) as ei:
        _call(base_url=GW, max_tokens=256)
    assert "this endpoint lists: deepseek-flash, deepseek-v4-pro" in str(ei.value)
    print("ok: the error shows what the endpoint actually serves")


# ---------------------------------------------------------------------------
# 7. a model that rejects max_tokens outright
# ---------------------------------------------------------------------------

def test_model_that_rejects_max_tokens_gets_the_new_name(monkeypatch) -> None:
    def handler(kw):
        if "max_completion_tokens" not in kw:
            raise ValueError(
                "Unsupported parameter: 'max_tokens' is not supported with "
                "this model. Use 'max_completion_tokens' instead.")
        return _resp("ok")

    client = _install(monkeypatch, handler)
    assert _call(max_tokens=256) == "ok"
    assert "max_completion_tokens" in client.calls[-1]
    assert "max_tokens" not in client.calls[-1]
    print("ok: max_tokens -> max_completion_tokens on request")


# ---------------------------------------------------------------------------
# 8. the reported log, replayed
# ---------------------------------------------------------------------------

def test_reported_run_now_completes(monkeypatch) -> None:
    """flash answers empty, v4-pro works: story + beats + summaries all live."""
    llm.remember_endpoint_models("https://api.deepseek.com/v1",
                                 ["deepseek-flash", "deepseek-v4-pro"])
    calls: list[str] = []

    def handler(kw):
        calls.append(kw["model"])
        if kw["model"] == "deepseek-flash":
            return _resp(finish="length", ct=kw["max_tokens"],
                         rt=kw["max_tokens"], reasoning="thinking")
        return _resp('{"sentences": ["He gets out."]}')

    _install(monkeypatch, handler)
    results, per_call = [], []
    for _ in range(6):        # story unit, beats, then 4 summary chunks
        start = len(calls)
        results.append(_call(base_url="https://api.deepseek.com/v1",
                             json_mode=True, max_tokens=1024))
        per_call.append(calls[start:])
    assert all(r for r in results), results
    # the FIRST call discovers the dead model (walking the ladder)...
    assert per_call[0][0] == "deepseek-flash", per_call[0]
    # ...and every later call (beats + 4 summary chunks) goes straight to the
    # model that answers: the flash model is never billed again
    assert all("deepseek-flash" not in c for c in per_call[1:]), per_call[1:]
    assert all(c and c[-1] == "deepseek-v4-pro" for c in per_call[1:]), per_call[1:]
    print(f"ok: replayed run recovers; later calls skip the dead model "
          f"(first call: {per_call[0]}, next: {per_call[1]})")


# ---------------------------------------------------------------------------
# 8b. the most likely cause of the reported run: the official DeepSeek
#     endpoint never served the configured name, and the probe now catches it
#     BEFORE the first call instead of after three dead passes
# ---------------------------------------------------------------------------

def test_official_endpoint_probe_replaces_a_name_it_does_not_serve(
        monkeypatch, capsys) -> None:
    llm.remember_endpoint_models("https://api.deepseek.com/v1",
                                 ["deepseek-chat", "deepseek-reasoner"])
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    got = llm.resolve_model("deepseek", "", "https://api.deepseek.com/v1")
    assert got == "deepseek-chat", got
    out = capsys.readouterr().out
    assert "does not list model 'deepseek-flash'" in out
    assert "deepseek-chat" in out
    print("ok: endpoint listing replaces a model it does not serve, before any call")


def test_first_call_already_uses_the_served_model(monkeypatch) -> None:
    """End-to-end through complete(): the probe runs BEFORE the call is made.

    With ``DEEPSEEK_MODEL=deepseek-flash`` (the reported setting) the very
    first request must already go to a model the endpoint serves — the
    cascade of three dead passes cannot start.
    """
    import types

    billed: list[str] = []

    class _FakeClient:
        def __init__(self):
            self.chat = type("_C", (), {})()
            self.chat.completions = self

        def create(self, **kw):
            billed.append(kw["model"])
            if kw["model"] == "deepseek-flash":
                return {"choices": [{"message": {"content": ""},
                                     "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 3}}
            return {"choices": [{"message": {"content": "Good answer."},
                                 "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 5}}

    fake_mod = types.ModuleType("openai")
    fake_mod.OpenAI = lambda **kw: _FakeClient()
    monkeypatch.setitem(sys.modules, "openai", fake_mod)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setattr(llm, "endpoint_models",
                        lambda base_url, api_key, timeout=10.0:
                        ["deepseek-chat", "deepseek-reasoner"])
    with redirect_stdout(io.StringIO()):
        for _ in range(3):
            assert llm.complete("deepseek", "", "sys", "write a story",
                                base_url="https://api.deepseek.com/v1",
                                json_mode=True, max_tokens=256) == "Good answer."
    assert billed == ["deepseek-chat"] * 3, billed
    print("ok: first call already uses the model the endpoint serves")


def test_official_endpoint_falls_back_to_documented_models(monkeypatch) -> None:
    """Even with no /models listing, a broken name has somewhere to go."""
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    assert llm._next_model("deepseek", "https://api.deepseek.com/v1",
                           "deepseek-flash") == "deepseek-chat"
    assert llm._next_model("deepseek", None, "deepseek-chat") == "deepseek-reasoner"
    print("ok: official DeepSeek endpoint offers its documented models")


# ---------------------------------------------------------------------------
# 9. after a model fails, prefer a stronger sibling — and say where a
#    multi-chunk pass died
# ---------------------------------------------------------------------------

def test_next_model_prefers_a_stronger_sibling() -> None:
    llm.remember_endpoint_models(GW, ["deepseek-flash", "deepseek-mini",
                                      "deepseek-v4-pro"])
    assert llm._next_model("deepseek", GW, "deepseek-flash") == "deepseek-v4-pro"
    # ...and never suggests a model this run already proved broken
    llm._BAD_MODELS.add(llm._call_key("deepseek", GW, "deepseek-v4-pro"))
    assert llm._next_model("deepseek", GW, "deepseek-flash") == "deepseek-mini"
    print("ok: replacement prefers the strong sibling, skips known-broken names")


def test_summary_pass_says_where_it_died(monkeypatch, tmp_path, capsys) -> None:
    from recap import summarize

    calls = {"n": 0}

    def fake_complete(provider, model, system, user, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise llm.LLMError("provider returned an empty message for model "
                               "'deepseek-flash' (tokens were billed but "
                               "nothing came back).")
        return "[00:10] Something happens."

    monkeypatch.setattr(summarize.llm, "complete", fake_complete)
    chunks = [{"text": "hello there", "start": 0.0, "end": 60.0,
               "cues": [{"start": 0.0, "end": 2.0}]} for _ in range(3)]
    out = tmp_path / "summaries.txt"
    with pytest.raises(llm.LLMError):
        summarize.summarize_chunks(chunks,
                                   {"provider": "deepseek",
                                    "model": "deepseek-flash"},
                                   out_partial=out)
    printed = capsys.readouterr().out
    assert "summary chunk 2/3 failed" in printed, printed
    assert "1/3 chunks are already summarized" in printed, printed
    assert "re-run to resume" in printed
    print("ok: a failed summary chunk names its index and the resume point")


if __name__ == "__main__":
    import traceback

    failed = 0
    for name, fn in sorted(list(globals().items())):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            if "monkeypatch" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                from _pytest.monkeypatch import MonkeyPatch
                mp = MonkeyPatch()
                try:
                    fn(mp)
                finally:
                    mp.undo()
            else:
                fn()
        except Exception:
            failed += 1
            traceback.print_exc()
    print("\nall llm recovery tests passed" if not failed else f"\n{failed} failed")
