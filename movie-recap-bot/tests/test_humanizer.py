"""Regression tests for the humanizer pass (recap/script.py).

THE REPORTED BUG (ToyStory5 run, 159-sentence script):

    ! HUMANIZE_FAILED: 69/159 sentences (43%) failed to humanize
    ERROR: RuntimeError: HUMANIZE_FAILED ... (43% > 10% threshold)

after ~1h15m of transcription, vision captioning and writing. The log breaks
down exactly: 51 x "Pass B: sentence tightened length would exceed lock --
keeping original" + 18 x "retry still failed" == 69. So the crash came from
two code defects, not from the movie:

  1. Pass B was DEAD CODE. HUMANIZER_PROMPT_PASS_B existed but was never sent
     to the model, so a good voice rewrite that came back a few words longer
     was thrown away -- even though Pass A is explicitly told "NO length
     constraint in this pass" -- and then counted as a humanizer failure.
  2. "unchanged" was scored as "failed" even for lines with no AI tell left to
     remove, and the echo retry fired one serial API call per sentence with no
     escalation (180 extra round trips on that run).
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recap import script as script_mod  # noqa: E402
from recap.util import count_words  # noqa: E402

CFG = {"provider": "deepseek", "model": "x"}


def _draft_lines(user: str) -> list[str]:
    """The numbered sentences inside a humanizer prompt (never its pattern list)."""
    import re

    body = user
    for marker in ("=== FINISHED SCRIPT", "=== PAIRS"):
        if marker in body:
            body = body.split(marker, 1)[1]
    body = body.split("=== END", 1)[0]
    if "HUMANIZED (" in body:          # Pass B pair block
        return [m.strip() for m in
                re.findall(r"HUMANIZED \(\d+ words\):\s*(.+)", body)]
    return [t for _, t in re.findall(r"^\s*(\d+)\.\s+(.*)$", body, flags=re.M)]


def _patch(fn):
    orig = script_mod.llm.complete
    script_mod.llm.complete = fn
    return lambda: setattr(script_mod.llm, "complete", orig)


# ---------------------------------------------------------------------------
# 1. the local tell scan
# ---------------------------------------------------------------------------

def test_tell_scan_finds_ai_writing_and_leaves_clean_lines_alone() -> None:
    telling = {
        "It's not just a crash site, it's the start of a manhunt.": "not-X-but-Y",
        "Woody freezes — and that changes everything.": "em-dash-or-semicolon",
        "At its core, the toy is afraid.": "deep-saying",
        "Here's the thing about Bonnie's room.": "staged-run-up",
        "Buzz stares at the sky, symbolizing his freedom.": "ing-rider",
        "The escape is a breathtaking sequence.": "sales-language",
        "This is a pivotal moment for Jessie.": "inflated-significance",
        "Forky delves into the trash can.": "ai-vocabulary",
        "The lamp serves as a signal.": "weak-copula",
        "The film cuts to the hallway.": "meta-commentary",
    }
    for line, want in telling.items():
        got = script_mod._ai_tells(line)
        assert want in got, (line, got)

    clean = [
        "Woody stares at the closed door.",
        "Meanwhile, Bonnie packs her backpack for the first day of school.",
        "Forky decides he belongs in the trash again, and Woody talks him out of it.",
        "Jessie keeps the others calm while the van pulls away.",
        "That night, Buzz slips through the window and lands in the flower bed.",
    ]
    for line in clean:
        assert script_mod._ai_tells(line) == [], (line, script_mod._ai_tells(line))
    print("ok: tell scan flags AI writing and leaves clean narration alone")


def test_enumeration_prefix_echo_is_not_mistaken_for_a_rewrite() -> None:
    """A model that copies '12. ' back from the numbered draft used to look
    like a successful rewrite (different string) — it is an echo."""
    assert script_mod._strip_enum_prefix("12. Woody hides.") == "Woody hides."
    assert script_mod._strip_enum_prefix("- Woody hides.") == "Woody hides."
    assert script_mod._strip_enum_prefix("Woody hides.") == "Woody hides."
    assert script_mod._is_humanize_failed("Woody hides.",
                                          script_mod._strip_enum_prefix("12. Woody hides."))
    print("ok: numbered echoes are detected as echoes")


# ---------------------------------------------------------------------------
# 2. a clean script must not fail the build
# ---------------------------------------------------------------------------

def test_clean_script_echoed_back_is_not_a_failure() -> None:
    """THE CRASH: deepseek returned the draft unchanged and every line was
    scored as a failure (100%/43%). Lines with nothing to fix are a PASS."""
    sents = [
        "Woody stares at the closed door.",
        "Bonnie packs her backpack for the first day of school.",
        "Forky decides he belongs in the trash again.",
        "Buzz tries to talk him out of it.",
        "Jessie keeps the others calm while the van pulls away.",
    ] * 32                                   # 160 sentences, 6 windows
    calls = {"n": 0}

    def echo(provider, model, system, user, **kw):
        calls["n"] += 1
        return json.dumps({"sentences": _draft_lines(user)})

    restore = _patch(echo)
    try:
        out = script_mod._humanize_script(CFG, sents, "English")
    finally:
        restore()
    stats = script_mod._humanize_script.last_stats
    assert out == sents, "an echoed clean script is returned untouched"
    assert stats["needed"] == 0, stats
    assert stats["failed"] == 0 and stats["failed_rate"] == 0.0, stats
    # one Pass A call per 30-sentence window, NOT one retry per sentence
    assert calls["n"] == 6, f"{calls['n']} llm calls (was 165 before the fix)"
    print("ok: clean script echoed back -> 0 failures, 6 calls (was 165)")


# ---------------------------------------------------------------------------
# 3. Pass B really runs
# ---------------------------------------------------------------------------

def test_pass_b_tightens_a_long_rewrite_instead_of_dropping_it() -> None:
    """Pass A is told "no length constraint"; a rewrite that comes back longer
    must be TIGHTENED by Pass B (the prompt that used to be dead code), not
    discarded and counted as a failure."""
    sents = ["It's not just a crash site, it's the start of a manhunt."]
    long_rewrite = ("The crash site is where the manhunt begins, and every "
                    "soldier in the valley is already moving toward the smoke.")
    tight = "The crash site is where the manhunt begins."
    seen = {"pass_b": False}

    def fake(provider, model, system, user, **kw):
        if "Timing lock pass" in user:
            seen["pass_b"] = True
            assert _draft_lines(user) == [long_rewrite], _draft_lines(user)
            return json.dumps({"sentences": [tight]})
        return json.dumps({"sentences": [long_rewrite]})

    restore = _patch(fake)
    try:
        out = script_mod._humanize_script(CFG, sents, "English")
    finally:
        restore()
    stats = script_mod._humanize_script.last_stats
    assert seen["pass_b"], "HUMANIZER_PROMPT_PASS_B must actually be sent"
    assert out == [tight], out
    assert stats["failed"] == 0 and stats["timing_rescued"] == 1, stats
    print("ok: Pass B tightens an over-long rewrite back into its film window")


def test_pass_b_local_shrinker_rescues_when_the_model_will_not() -> None:
    """If the model refuses to tighten, a deterministic local shrink
    (contractions, filler adverbs, wordy connectives) gets the line back into
    its window before we give up on it."""
    orig = "The pilot is not just injured, he is hunted through the woods tonight."
    bloated = ("It is quite obvious that the pilot cannot walk, and he is "
               "actually being hunted through the dark woods.")

    def fake(provider, model, system, user, **kw):
        # Pass A rewrites; Pass B hands the same too-long line straight back.
        return json.dumps({"sentences": [bloated]})

    restore = _patch(fake)
    try:
        out = script_mod._humanize_script(CFG, [orig], "English")
    finally:
        restore()
    stats = script_mod._humanize_script.last_stats
    assert out[0] != orig, "the rewrite must survive, tightened"
    assert out[0] != bloated, "and it must be shorter than the bloated version"
    assert script_mod._fits_timing_lock(orig, out[0]), (out[0], count_words(out[0]))
    assert stats["failed"] == 0, stats
    print(f"ok: local shrinker rescued -> {out[0]!r}")


def test_a_rewrite_that_cannot_fit_keeps_the_original_and_stays_in_sync() -> None:
    """The timing lock is still absolute: a rewrite nothing can shrink is
    rejected so the picture never falls behind the narration."""
    orig = "Woody hides."
    huge = ("Woody, who has been listening from behind the bookshelf for a "
            "while now, slips down into the shadows beneath the bed frame and "
            "waits there without making a single sound.")

    restore = _patch(lambda *a, **k: json.dumps({"sentences": [huge]}))
    try:
        out = script_mod._humanize_script(CFG, [orig], "English")
    finally:
        restore()
    assert out == [orig], out
    print("ok: an unshrinkable rewrite is rejected, original kept")


# ---------------------------------------------------------------------------
# 4. echo repair is batched, not one call per sentence
# ---------------------------------------------------------------------------

def test_echoed_tell_lines_are_retried_in_one_batched_call() -> None:
    telling = [f"It's not just problem {i}, it's a disaster." for i in range(10)]
    calls: list[str] = []

    def fake(provider, model, system, user, **kw):
        calls.append(user)
        lines = _draft_lines(user)
        if len(calls) == 1:                       # Pass A: echo everything
            return json.dumps({"sentences": lines})
        # batched retry: real rewrites, same length class
        return json.dumps({"sentences": [f"Problem {i} turns into a disaster."
                                         for i in range(len(lines))]})

    restore = _patch(fake)
    try:
        out = script_mod._humanize_script(CFG, telling, "English")
    finally:
        restore()
    stats = script_mod._humanize_script.last_stats
    assert len(calls) == 2, f"1 Pass A + 1 batched retry, got {len(calls)}"
    assert stats["failed"] == 0 and stats["kept"] == 10, stats
    assert all("not just" not in s for s in out), out
    print("ok: 10 echoed lines repaired with ONE batched retry (was 10 calls)")


def test_single_sentence_retries_are_capped() -> None:
    """A provider that keeps echoing must not turn into hundreds of serial
    round trips (the reported run made 180)."""
    telling = [f"It's not just problem {i}, it's a disaster." for i in range(20)]
    calls = {"n": 0}

    def stubborn(provider, model, system, user, **kw):
        calls["n"] += 1
        return json.dumps({"sentences": _draft_lines(user)})

    restore = _patch(stubborn)
    try:
        script_mod._humanize_script(
            dict(CFG, humanize_max_single_retries=4), telling, "English")
    finally:
        restore()
    # 1 Pass A + 1 batched retry + 4 capped singles
    assert calls["n"] == 6, calls["n"]
    print("ok: stubborn provider costs 6 calls, not 20+")


# ---------------------------------------------------------------------------
# 5. end-to-end: the reported run must not fail the build any more
# ---------------------------------------------------------------------------

def test_the_reported_159_sentence_run_no_longer_fails_the_build() -> None:
    """Replay of the ToyStory5 pathology: Pass A echoes every window and every
    repaired line comes back ~30% too long. Before the fix that was
    'HUMANIZE_FAILED 69/159 (43%)' and a RuntimeError; now Pass B tightens and
    the run ships."""
    pool = [
        "It's not just a toy, it's a friend.",              # tell
        "Woody stares at the closed door.",                 # clean
        "This is a pivotal moment for Jessie.",             # tell
        "Bonnie packs her backpack for school.",            # clean
        "Buzz stares at the sky, showcasing his courage.",  # tell
    ]
    sents = [pool[i % len(pool)] for i in range(159)]

    # What the reported run actually produced: temp-0.9 retries DID remove the
    # tells, but came back ~30% longer, and Pass B then threw them all away.
    longer = {
        pool[0]: ("Woody is a friend to Bonnie, and he has been one since the "
                  "day she found him."),
        pool[2]: ("Jessie decides right here, in the middle of the hallway, "
                  "which side she is on."),
        pool[4]: ("Buzz looks up at the open sky above the roof and does not "
                  "look away for a long moment."),
    }
    tighter = {
        pool[0]: "Woody has been Bonnie's friend since she found him.",
        pool[2]: "Jessie picks her side in the hallway.",
        pool[4]: "Buzz looks up at the open sky and holds there.",
    }

    def fake(provider, model, system, user, **kw):
        lines = _draft_lines(user)
        if "Timing lock pass" in user:
            return json.dumps({"sentences": [
                tighter[k] for s in lines
                for k, v in longer.items() if v == s]})
        if "you already saw these lines" in user or "echoed this line" in user:
            return json.dumps({"sentences": [longer.get(s, s) for s in lines]})
        return json.dumps({"sentences": lines})          # Pass A echoes

    restore = _patch(fake)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            out = script_mod._humanize_script(CFG, sents, "English")
    finally:
        restore()
    stats = script_mod._humanize_script.last_stats
    assert len(out) == 159, "sentence count is the timing lock"
    assert stats["needed"] > 0, stats
    assert stats["failed_rate"] <= 0.10, stats
    assert all(script_mod._fits_timing_lock(a, b) for a, b in zip(sents, out)), \
        "every kept rewrite still fits its film window"
    print(f"ok: the reported run now scores {stats['failed_rate']:.0%} "
          f"(was 43% and a hard RuntimeError)")


def test_full_pipeline_does_not_raise_on_an_echoing_provider() -> None:
    """generate_segmented_script end to end (writer + humanizer both echoing):
    a clean-but-unchanged script must ship, not raise."""
    chunk = {
        "index": 0, "start": 1000.0, "end": 1150.0,
        "summary": "A pilot is shot down and hunted through the woods.",
        "beats": [{"t": 1000.0 + i * 2.4, "text": f"beat {i}"} for i in range(63)],
    }
    written = [
        "The pilot crashes into the pine forest.",
        "An armed stranger inspects the wreck.",
        "The pilot grabs the barrel and both men fall.",
        "He drags himself away and hides in the trees.",
        "By morning the army is tracking his trail.",
        "He crosses a frozen river to lose the dogs.",
        "The chase ends at the border bridge.",
        "Troy opens the gate and lets him through.",
    ]

    def echo(provider, model, system, user, **kw):
        if "FINISHED SCRIPT" in user or "Timing lock pass" in user:
            return json.dumps({"sentences": _draft_lines(user)})
        return json.dumps({"sentences": written})

    restore = _patch(echo)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            out = script_mod.generate_segmented_script(
                [chunk], dict(CFG), 120, words_per_minute=150,
                lang_name="English", sign_off=False, visual_match=True,
                humanize=True,
            )
    finally:
        restore()
    log = buf.getvalue()
    assert len(out) == len(written)
    assert "humanizer pass" in log
    assert "HUMANIZE_FAILED" not in log, log
    print("ok: an echoing provider on a clean script ships instead of crashing")


# ---------------------------------------------------------------------------
# 6. the gate still exists, and it never throws the run away silently
# ---------------------------------------------------------------------------

def test_gate_still_fails_loudly_and_writes_a_report_first(tmp_path=None) -> None:
    import tempfile

    d = Path(tmp_path or tempfile.mkdtemp(prefix="humanize-report-"))
    chunk = {
        "index": 0, "start": 1000.0, "end": 1150.0,
        "summary": "A pilot is shot down.",
        "beats": [{"t": 1000.0 + i * 2.4, "text": f"beat {i}"} for i in range(63)],
    }
    telling = [f"It's not just problem {i}, it's a disaster." for i in range(8)]

    def echo(provider, model, system, user, **kw):
        if "FINISHED SCRIPT" in user or "Timing lock pass" in user:
            return json.dumps({"sentences": _draft_lines(user)})
        return json.dumps({"sentences": telling})

    restore = _patch(echo)
    buf = io.StringIO()
    raised = None
    try:
        with redirect_stdout(buf):
            script_mod.generate_segmented_script(
                [chunk], dict(CFG), 120, words_per_minute=150,
                lang_name="English", sign_off=False, visual_match=True,
                humanize=True, report_dir=d,
            )
    except RuntimeError as exc:
        raised = exc
    finally:
        restore()
    assert raised is not None, "a script full of unfixed tells must still fail"
    assert "HUMANIZE_FAILED" in str(raised)
    report = d / "humanizer_report_english.json"
    assert report.exists(), "the finished script must be saved before failing"
    data = json.loads(report.read_text(encoding="utf-8"))
    assert len(data["sentences"]) == 8
    assert data["sentences"][0]["tells_before"], data["sentences"][0]
    assert "nothing from this run is lost" in buf.getvalue()
    print("ok: the gate still fails loudly, but the script is saved first")


def test_a_couple_of_stubborn_lines_do_not_burn_a_finished_run() -> None:
    """On a 60-line script with 4 flagged lines, 1 unfixable line is 25% --
    a rate, but not a reason to throw an hour of work away."""
    sents = ["Woody stares at the closed door."] * 59 + \
            ["It's not just a toy, it's a friend."]
    chunk = {"index": 0, "start": 0.0, "end": 900.0,
             "summary": "s", "beats": [{"t": i * 10.0, "text": f"b{i}"}
                                       for i in range(90)]}

    def echo(provider, model, system, user, **kw):
        if "FINISHED SCRIPT" in user or "Timing lock pass" in user:
            return json.dumps({"sentences": _draft_lines(user)})
        return json.dumps({"sentences": sents})

    restore = _patch(echo)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            out = script_mod.generate_segmented_script(
                [chunk], dict(CFG), 700, words_per_minute=150,
                lang_name="English", sign_off=False, visual_match=True,
                humanize=True,
            )
    finally:
        restore()
    assert out, "the run must ship"
    assert "under the" in buf.getvalue() and "floor for failing a build" in buf.getvalue()
    print("ok: a single stubborn line no longer kills a finished run")


# ---------------------------------------------------------------------------
# 7. mechanically trimmed sections are mandatory work for the humanizer
# ---------------------------------------------------------------------------

def test_mechanically_trimmed_sentences_are_mandatory_even_without_tells() -> None:
    clean = ["Woody stares at the closed door.",
             "Bonnie packs her backpack.",
             "Buzz follows her outside."]
    calls: list[str] = []

    def fake(provider, model, system, user, **kw):
        calls.append(user)
        lines = _draft_lines(user)
        if len(calls) == 1:
            return json.dumps({"sentences": lines})       # echo
        return json.dumps({"sentences": [f"Rewritten line {i}." for i in range(len(lines))]})

    restore = _patch(fake)
    try:
        out = script_mod._humanize_script(CFG, clean, "English", mandatory=[1])
    finally:
        restore()
    stats = script_mod._humanize_script.last_stats
    assert stats["mandatory"] == 1, stats
    # a trimmed beat gets the retry budget, but it can never fail the build:
    # the tell scan found nothing wrong with it, so "needed" stays 0
    assert stats["needed"] == 0 and stats["failed"] == 0, stats
    assert out[1] != clean[1], "the mechanically trimmed line must be rewritten"
    assert out[0] == clean[0] and out[2] == clean[2], "clean lines untouched"
    assert "mechanically-trimmed" in calls[0], "the prompt names why it must change"
    print("ok: mechanically trimmed beats get the retry budget, never fail the build")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall humanizer tests passed")
