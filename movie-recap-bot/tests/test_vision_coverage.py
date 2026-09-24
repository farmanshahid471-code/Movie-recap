"""Regression tests for the vision pass coverage gate (recap/vision.py).

THE REPORTED BUG (ToyStory5, second run):

    * Vision pass: captioning 594 frames ... in batches of 6
      ... 216/594 frames (221 captioned)      <- should be 222
      ... 336/594 frames (335 captioned)      <- should be 342
      ... 594/594 frames (587 captioned)      <- should be 600
    * Vision pass: 587 on-screen notes — coverage 97.8% (587/600)
    ! VISION_COVERAGE_GATE FAILED: ... below the required 99% (594/600).

Note what is NOT in that log: a single "vision batch N failed" line. Every
request "succeeded" — 13 frames simply came back with no caption line and were
dropped on the floor. They were never added to the failed list, so the final
sweep never ran, and the run died at the gate 20 minutes later with nothing in
the log to explain the missing frames.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recap import vision as vision_mod  # noqa: E402


class _FakeResp:
    def __init__(self, text: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


def _labels(parts) -> list[int]:
    """The frame times a request asked about (parsed back out of its prompt)."""
    import re

    head = parts[0]["text"]
    out = []
    for m in re.finditer(r"-\s*(\d{1,3}):(\d{2})(?::(\d{2}))?", head):
        if m.group(3) is not None:
            out.append(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)))
        else:
            out.append(int(m.group(1)) * 60 + int(m.group(2)))
    return out


def _fake_client(behaviour):
    """behaviour(times) -> str response text (or raises)."""
    class _Completions:
        def create(self, model, messages, **kw):
            times = _labels(messages[1]["content"])
            return _FakeResp(behaviour(times))

    return type("C", (), {"chat": type("Ch", (), {"completions": _Completions()})()})()


def _patch(monkey: dict):
    saved = {k: getattr(vision_mod, k) for k in monkey}
    for k, v in monkey.items():
        setattr(vision_mod, k, v)
    return lambda: [setattr(vision_mod, k, v) for k, v in saved.items()]


def _setup(td: Path, n_frames: int = 24):
    """A fake movie whose frames all extract fine."""
    movie = td / "film.mp4"
    movie.write_bytes(b"\x00" * 32)
    times = list(range(10, 10 + n_frames * 10, 10))

    def fake_extract(mv, ts, outdir, width=512):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        got = {}
        for t in ts:
            f = outdir / f"frame_{t:06d}.jpg"
            f.write_bytes(b"\xff\xd8\xff\xd9")
            got[t] = f
        return got

    return movie, times, fake_extract


# ---------------------------------------------------------------------------
# 1. the reported failure: a "successful" batch that drops frames
# ---------------------------------------------------------------------------

def test_frames_missing_from_a_successful_batch_are_swept_not_dropped() -> None:
    td = Path(tempfile.mkdtemp(prefix="vision-gap-"))
    movie, times, fake_extract = _setup(td, 24)
    state = {"dropped": False}

    def behaviour(ts):
        # Exactly the reported pathology: the request succeeds, but the
        # response comes back empty and the batch's frames are lost. It is a
        # transient glitch -- asking again works.
        if not state["dropped"] and len(ts) > 1 and 70 in ts:
            state["dropped"] = True
            return ""                      # 6 frames silently lost
        return "\n".join(f"{vision_mod._fmt_label(t)} - Woody hides." for t in ts)

    client = _fake_client(behaviour)
    restore = _patch({
        "_scene_times": lambda *a, **k: [],
        "extract_frames": fake_extract,
        "probe_duration": lambda *a, **k: float(times[-1] + 10),
        "pick_times": lambda *a, **k: times,
        "provider_configured": lambda p: True,
        "_client": lambda cfg: (client, "fake-model"),
        "_fallback_client": lambda cfg: (None, None),
    })
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            notes = vision_mod.capture(
                movie, {"provider": "gemini", "frames_per_request": 6,
                        "sweep_pause_seconds": 0.0}, td)
    finally:
        restore()
    log = buf.getvalue()
    assert "returned no caption" in log, log
    assert "final sweep" in log, log
    assert len(notes) == len(times), (
        f"the sweep must recover the silently dropped frames, "
        f"got {len(notes)}/{len(times)}")
    print(f"ok: {len(times)}/{len(times)} captioned — dropped frames are swept, "
          "not lost (was 587/600)")


def test_sweeps_fall_back_to_one_frame_per_request() -> None:
    """A model that mangles labels in big batches still gets every frame in,
    because each sweep halves the batch and the last one asks frame by frame."""
    td = Path(tempfile.mkdtemp(prefix="vision-single-"))
    movie, times, fake_extract = _setup(td, 12)
    sizes: list[int] = []

    def behaviour(ts):
        sizes.append(len(ts))
        if len(ts) > 1:
            return "here are your frames"       # no labels -> unusable
        return f"{vision_mod._fmt_label(ts[0])} - Buzz looks up."

    client = _fake_client(behaviour)
    restore = _patch({
        "_scene_times": lambda *a, **k: [],
        "extract_frames": fake_extract,
        "probe_duration": lambda *a, **k: float(times[-1] + 10),
        "pick_times": lambda *a, **k: times,
        "provider_configured": lambda p: True,
        "_client": lambda cfg: (client, "fake-model"),
        "_fallback_client": lambda cfg: (None, None),
    })
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            notes = vision_mod.capture(
                movie, {"provider": "gemini", "frames_per_request": 6,
                        "sweep_pause_seconds": 0.0, "sweep_attempts": 3}, td)
    finally:
        restore()
    assert 1 in sizes, f"a sweep must reach 1 frame per request, sizes={sizes}"
    assert len(notes) == len(times), f"{len(notes)}/{len(times)}"
    print(f"ok: batch sizes escalated {sorted(set(sizes))} -> all "
          f"{len(times)} frames captioned")


# ---------------------------------------------------------------------------
# 2. shot detection is computed once, not three times
# ---------------------------------------------------------------------------

def test_shot_detection_is_cached_across_calls() -> None:
    """_scene_times is a full decode of the film (~3 min for a 100-min movie).
    It ran once in capture() and AGAIN in check_coverage() right before the
    gate — six wasted minutes per attempt on a deterministic answer."""
    td = Path(tempfile.mkdtemp(prefix="vision-scenes-"))
    movie = td / "film.mp4"
    movie.write_bytes(b"\x00" * 32)
    calls = {"n": 0}

    def slow_scene_times(mv, threshold, duration):
        calls["n"] += 1
        return [12.0, 40.0, 95.0]

    restore = _patch({"_scene_times": slow_scene_times})
    try:
        a = vision_mod._scene_times_cached(movie, 0.35, 300.0, td)
        b = vision_mod._scene_times_cached(movie, 0.35, 300.0, td)
        c = vision_mod._scene_times_cached(movie, 0.35, 300.0, td)
    finally:
        restore()
    assert a == b == c == [12.0, 40.0, 95.0]
    assert calls["n"] == 1, f"detected {calls['n']} times, expected 1"
    assert (td / "scene_times.json").exists()
    print("ok: shot detection runs once per film, then comes from cache")


# ---------------------------------------------------------------------------
# 3. the gate judges blind stretches, not a bare percentage
# ---------------------------------------------------------------------------

def _write_notes(td: Path, movie: Path, cfg: dict, captioned: list[int]) -> None:
    sig = vision_mod._movie_sig(movie, cfg)
    (td / "visual_notes.json").write_text(json.dumps({
        "sig": sig,
        "frames": [{"t": t, "text": "a shot", "confidence": "high",
                    "provider": "primary", "retried": False}
                   for t in captioned]}), encoding="utf-8")


def test_gate_passes_when_the_missing_frames_are_scattered() -> None:
    """The reported run: 13 of 600 frames missing, spread over two hours. The
    beat matcher has a neighbour on either side of every hole — that is not a
    reason to throw away 20 minutes of captioning."""
    td = Path(tempfile.mkdtemp(prefix="vision-gate-ok-"))
    movie = td / "film.mp4"
    movie.write_bytes(b"\x00" * 32)
    times = list(range(10, 6010, 10))                 # 600 frames
    missing = set(times[i] for i in range(13, 600, 45))  # 14 scattered holes
    cfg = {"provider": "gemini", "coverage_threshold": 0.99}
    _write_notes(td, movie, cfg, [t for t in times if t not in missing])

    restore = _patch({
        "_scene_times": lambda *a, **k: [],
        "probe_duration": lambda *a, **k: 6010.0,
        "pick_times": lambda *a, **k: times,
        "provider_configured": lambda p: True,
    })
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            vision_mod.coverage_gate(movie, cfg, td)   # must NOT raise
    finally:
        restore()
    log = buf.getvalue()
    assert "scattered" in log and "continuing" in log, log
    print("ok: 14 scattered holes in 600 frames no longer kill a finished run")


def test_gate_still_fails_on_a_blind_stretch_of_film() -> None:
    """What the gate is actually for: a run of film the timeline cannot see."""
    td = Path(tempfile.mkdtemp(prefix="vision-gate-bad-"))
    movie = td / "film.mp4"
    movie.write_bytes(b"\x00" * 32)
    times = list(range(10, 6010, 10))
    missing = set(times[200:260])                     # 600 consecutive seconds
    cfg = {"provider": "gemini", "coverage_threshold": 0.99}
    _write_notes(td, movie, cfg, [t for t in times if t not in missing])

    restore = _patch({
        "_scene_times": lambda *a, **k: [],
        "probe_duration": lambda *a, **k: 6010.0,
        "pick_times": lambda *a, **k: times,
        "provider_configured": lambda p: True,
    })
    raised = None
    try:
        vision_mod.coverage_gate(movie, cfg, td)
    except vision_mod.VisionError as exc:
        raised = exc
    finally:
        restore()
    assert raised is not None, "a 10-minute blind stretch must still fail"
    assert "no caption" in str(raised).lower(), str(raised)
    print("ok: a 10-minute blind stretch still fails the build, loudly")


def test_blind_stretch_measurement() -> None:
    have = [0, 10, 20, 100, 110]
    assert vision_mod.blind_stretch(have, [], 200.0) == (0.0, 0.0)
    worst, at = vision_mod.blind_stretch(have, [30, 40, 50], 200.0)
    assert worst == 80.0 and at == 20.0, (worst, at)
    print("ok: blind-stretch measurement is correct")


# ---------------------------------------------------------------------------
# 4. caption parsing
# ---------------------------------------------------------------------------

def test_caption_lines_parse_with_bullets_and_dashes() -> None:
    frames = [(65, Path("a.jpg")), (130, Path("b.jpg"))]
    raw = "- 01:05 — Woody hides under the bed.\n* [02:10]: Buzz opens the door."
    out = vision_mod._parse_caption_batch(raw, frames)
    assert out[65] == "Woody hides under the bed.", out
    assert out[130] == "Buzz opens the door.", out
    print("ok: bulleted / em-dashed caption lines parse cleanly")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall vision coverage tests passed")
