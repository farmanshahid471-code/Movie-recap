"""Tests for the Edge-TTS 403 fix (recap/tts.py). No network, no edge-tts.

Run:  python tests/test_tts_edge.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["RECAP_STRIP_SILENCE"] = "0"  # no ffmpeg in the test env

from recap import tts  # noqa: E402


class FakeCommunicate:
    """Fails with the reported 403 twice, then speaks."""

    calls = 0

    def __init__(self, text, voice, **kw):
        FakeCommunicate.calls += 1
        self.n = FakeCommunicate.calls

    async def stream(self):
        if self.n <= 2:
            raise RuntimeError(
                "403, message='Invalid response status', "
                "url='wss://speech.platform.bing.com/...'")
        yield {"type": "WordBoundary", "offset": 0, "duration": 1_000_000,
               "text": "Hello"}
        yield {"type": "audio", "data": b"ID3-fake-audio"}


def _provider(fake_module) -> tts.EdgeTTS:
    sys.modules["edge_tts"] = fake_module
    return tts.EdgeTTS()


def test_old_edge_tts_version_rejected_with_upgrade_hint():
    fake = types.SimpleNamespace(__version__="6.1.19", Communicate=FakeCommunicate)
    try:
        _provider(fake)
        raise AssertionError("old edge-tts must be rejected up front")
    except tts.TTSError as exc:
        assert "pip install -U" in str(exc) and "403" in str(exc)
    print("ok: edge-tts < 7.2 rejected with the upgrade command")


def test_403_is_retryable_and_friendly():
    exc = RuntimeError("403, message='Invalid response status'")
    assert tts._edge_retryable(exc)
    msg = tts._edge_friendly(str(exc))
    assert "edge-tts>=7.2.0" in msg and "EDGE_TTS_PROXY" in msg
    assert "tts_provider" in msg  # the provider-switch escape hatch
    print("ok: 403 is retryable + message lists upgrade/proxy/provider fixes")


def test_beat_retries_403_then_succeeds(tmp: Path):
    FakeCommunicate.calls = 0
    fake = types.SimpleNamespace(__version__="7.3.1", Communicate=FakeCommunicate)
    prov = _provider(fake)
    out = tmp / "beat_001.mp3"
    words, _dur = asyncio.run(prov._sync_one_beat("Hello world.", "v", out))
    assert FakeCommunicate.calls == 3  # 2x 403, then success
    assert out.read_bytes() == b"ID3-fake-audio"
    assert words and words[0][0] == "Hello"
    print("ok: a beat survives transient 403s (fresh Communicate per try)")


def test_synthesize_beats_resumes_finished_beats(tmp: Path):
    FakeCommunicate.calls = 0
    fake = types.SimpleNamespace(__version__="7.3.1", Communicate=FakeCommunicate)
    prov = _provider(fake)
    beats = tmp / "beats"
    beats.mkdir()
    # line 1 already spoken in the crashed run; probe_duration will read
    # 0.0 without ffprobe, so pad via a stubbed probe for this test only
    (beats / "beat_001.mp3").write_bytes(b"already-spoken")
    orig = tts.probe_duration
    tts.probe_duration = lambda p: 2.0
    try:
        cues, files = prov.synthesize_beats(["Done.", "New line."], "v", beats)
    finally:
        tts.probe_duration = orig
    assert len(cues) == len(files) == 2
    spoken = [f.name for f in files]
    assert spoken == ["beat_001.mp3", "beat_002.mp3"]
    print("ok: finished beats are reused, only missing lines are spoken")


def test_resume_rejects_changed_script(tmp: Path):
    FakeCommunicate.calls = 0
    fake = types.SimpleNamespace(__version__="7.3.1", Communicate=FakeCommunicate)
    prov = _provider(fake)
    beats = tmp / "beats"
    beats.mkdir()
    (beats / "beat_001.mp3").write_bytes(b"old-script-audio")
    (beats / ".beats_sig").write_text("different-script", encoding="utf-8")
    orig = tts.probe_duration
    tts.probe_duration = lambda p: 2.0
    try:
        prov.synthesize_beats(["Completely new line."], "v", beats)
    finally:
        tts.probe_duration = orig
    assert (beats / "beat_001.mp3").read_bytes() == b"ID3-fake-audio"
    print("ok: changed script never inherits a crashed run's beat audio")


if __name__ == "__main__":
    import tempfile
    test_old_edge_tts_version_rejected_with_upgrade_hint()
    test_403_is_retryable_and_friendly()
    with tempfile.TemporaryDirectory() as d:
        test_beat_retries_403_then_succeeds(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_synthesize_beats_resumes_finished_beats(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_resume_rejects_changed_script(Path(d))
    print("\nall edge TTS tests passed")
