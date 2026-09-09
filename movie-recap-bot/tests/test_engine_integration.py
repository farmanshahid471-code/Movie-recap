"""Full-engine integration test for ``pipeline.auto_recap`` (Steps A-F).

The heavy/external pieces (whisper, LLM, real embeddings, TTS network calls and
ffmpeg) are stubbed; everything *between* them — chunking, summarization
wiring, JSON script generation, per-language narration resolution, beat
mapping, window building, subtitle + ASS writing and output layout — is the
real production code. This is what catches wiring mistakes without needing a
movie file, an Ollama server or ffmpeg.

Run from the movie-recap-bot folder:

    python tests/test_engine_integration.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap import pipeline  # noqa: E402
from recap.config import load_config  # noqa: E402
from recap.tts import TimedCue  # noqa: E402


class _FakeTTS:
    """Narration with realistic silent gaps between sentences.

    12 sentences x 1.0s speech + 0.2s gaps = 13.2s of speech but a 14.2s file
    (there is a 1.0s tail of silence). The old pipeline sized the video from
    the 13.2s and let `-shortest` cut the rest; the whole point of the fix is
    that the render must cover the full span.
    """

    speech = 1.0
    gap = 0.2
    n_lines = 12
    # last cue ends at (n-1)*1.2 + 1.0, plus a 1.0s tail of silence
    audio_span = (n_lines - 1) * (speech + gap) + speech + 1.0

    def __init__(self):
        self.n = 0

    def synthesize(self, sentences, voice, out_mp3):
        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        out_mp3.write_bytes(b"ID3fakeaudio")
        cues = []
        t = 0.0
        for s in sentences:
            cues.append(TimedCue(s, t, t + self.speech))
            t += self.speech + self.gap
        return cues


# Deterministic narration the "LLM" returns — 12 sentences (the pipeline
# rejects recaps under 10 lines as broken).
SENTENCES = [
    "Narration sentence {i} races forward and surprises everyone.",
    "The detective uncovers the truth behind the closed door.",
    "A sudden twist leaves the whole town in shock.",
    "The hero faces the villain one last time.",
    "Everything explodes in a final dramatic confrontation.",
    "The quiet village finally breathes again.",
    "But the shadow returns when night falls.",
    "Two old friends reconcile at the train station.",
    "A letter from the past changes everything.",
    "The chase cuts through the crowded market.",
    "Nobody notices the stranger in the crowd.",
    "And the story ends with a quiet, haunting close.",
]


def _install_stubs():
    """Replace external calls with deterministic stand-ins. Returns restorer."""
    saved = {}

    def _save(mod, name):
        saved[(mod, name)] = getattr(mod, name)

    _save(pipeline.dialogue, "extract_dialogue")
    _save(pipeline.summarize, "summarize_chunks")
    _save(pipeline.script, "generate_script_json")
    _save(pipeline.script, "generate_segmented_script")
    _save(pipeline.llm, "verify_model")
    _save(pipeline.tts, "make_provider")

    pipeline.llm.verify_model = lambda cfg_llm: None  # no real API call in tests

    _save(pipeline.clip, "build_locked_visual")
    _save(pipeline.video, "burn_and_mux_locked")
    _save(pipeline, "probe_duration")

    pipeline.dialogue.extract_dialogue = lambda *a, **k: [
        {"text": f"dialogue line {i}", "start": i * 12.0, "end": i * 12.0 + 6.0,
         "words": [{"word": "x", "start": i * 12.0, "end": i * 12.0 + 1.0}]}
        for i in range(40)
    ]
    pipeline.summarize.summarize_chunks = lambda chunks, cfg, **k: [
        f"summary of chunk {c['index']}" for c in chunks
    ]
    pipeline.script.generate_script_json = lambda summary, cfg, target, mn, mx: list(SENTENCES)

    def _segmented(chunk_summaries, cfg_llm, target, **k):
        """Spread the fixture sentences over the film chunks, in order."""
        n = max(len(chunk_summaries), 1)
        out = []
        for i, s in enumerate(SENTENCES):
            c = chunk_summaries[min(i * n // len(SENTENCES), n - 1)]
            out.append({"sentence": s, "film_start": float(c["start"]),
                        "film_end": float(c["end"])})
        return out

    pipeline.script.generate_segmented_script = _segmented
    pipeline.tts.make_provider = lambda *a, **k: _FakeTTS()

    # Record what the timeline asked for so the test can assert the A/V lock.
    RECORDED = {}

    def _visual(movie, cuts, workdir, cfg_video, audio_span, mode="reencode"):
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        RECORDED["cuts"] = list(cuts)
        RECORDED["audio_span"] = audio_span
        out = workdir / "visual.mp4"
        out.write_bytes(b"fakebasevideo")
        return out

    pipeline.clip.build_locked_visual = _visual

    def _burn(base, narration, ass, out_mp4, cfg_video, duration=None):
        RECORDED["mux_duration"] = duration
        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(Path(base).read_bytes() + b"muxed")
        return out_mp4

    pipeline.video.burn_and_mux_locked = _burn
    def _probe(p):
        """Realistic per-file durations.

        The mp3 must report the FULL narration span (speech + the silent gaps
        between sentences), because that is precisely the number the old
        pipeline ignored when it sized the visual track.
        """
        name = Path(p).name
        if name.endswith(".mp3"):
            return _FakeTTS.audio_span
        if name.endswith(".mp4") and "e2e_" in name:
            return _FakeTTS.audio_span  # rendered output == narration
        return 600.0  # the source film

    pipeline.probe_duration = _probe
    _install_stubs.recorded = RECORDED

    def _restore():
        for (mod, name), fn in saved.items():
            setattr(mod, name, fn)

    return _restore


def test_full_semantic_flow() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="recap-e2e-"))
    movie = tmp / "film.mp4"
    movie.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # exists for Path checks

    os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-used")
    cfg = load_config()
    cfg["llm"] = {"provider": "deepseek", "model": "deepseek-chat",
                  "base_url": "https://api.deepseek.com/v1"}
    out = tmp / "out"
    out.mkdir(parents=True, exist_ok=True)
    cfg["project"]["_out"] = out
    cfg["project"]["output_dir"] = str(out)
    cfg["project"]["name"] = "e2e"
    cfg["language"]["target_languages"] = ["en"]
    cfg["language"]["_resolved"] = [{"code": "en", "tag": "en"}]
    cfg["narration"]["words_target"] = 200
    cfg["narration"]["words_min"] = 60
    cfg["narration"]["words_max"] = 4200

    restore = _install_stubs()
    try:
        outs = pipeline.auto_recap(cfg, movie)
    finally:
        restore()

    wd = out / "_work"
    assert outs and Path(outs[0]).exists()
    assert (wd / "transcript.json").exists()
    assert (wd / "script" / "script_en.json").exists()
    assert (wd / "script" / "script_en.txt").exists()
    assert (wd / "beats_en.json").exists()
    assert (wd / "en.mp3").exists()
    assert (wd / "en.srt").exists() and (wd / "en.ass").exists()
    assert (wd / "en.timing.json").exists()
    # --- the chronological timeline ran for real (only I/O was stubbed) ---
    beats = json.loads((wd / "beats_en.json").read_text(encoding="utf-8"))
    assert beats, "timeline produced no beats"
    starts = [b["film_start"] for b in beats]
    assert starts == sorted(starts), f"beats must be chronological, got {starts}"

    rec = _install_stubs.recorded
    span = rec["audio_span"]
    cut_total = sum(d for _, d in rec["cuts"])
    assert abs(cut_total - span) < 0.5, (
        f"BUG 1: visual {cut_total:.2f}s must equal narration {span:.2f}s"
    )
    assert abs(rec["mux_duration"] - span) < 1e-6, (
        "mux must be given an explicit duration, never -shortest"
    )
    assert len(rec["cuts"]) >= len(beats), "expected micro-cuts per beat"

    # The lock must cover the gaps AND the trailing silence, not just speech.
    spoken = _FakeTTS.n_lines * _FakeTTS.speech
    assert span > spoken, (
        f"narration span {span} should exceed pure speech {spoken}"
    )
    assert cut_total > spoken, (
        f"BUG 1: video {cut_total:.2f}s only covers the spoken {spoken:.2f}s — "
        "the silent gaps were dropped again and -shortest would truncate."
    )
    sentences = json.loads((wd / "script" / "script_en.json").read_text(encoding="utf-8"))
    assert len(sentences) == len(SENTENCES)
    # .txt sidecar is line-aligned
    txt_lines = [l for l in (wd / "script" / "script_en.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(txt_lines) == len(sentences)
    print("  outputs written:", [p.name for p in Path(out).glob('*.mp4')])
    print(f"  A/V lock: {len(rec['cuts'])} cuts = {cut_total:.2f}s "
          f"== narration {span:.2f}s")
    print("  intermediates: script_en.json, beats_en.json, en.mp3, en.srt, "
          "en.ass, en.timing.json OK")


if __name__ == "__main__":
    test_full_semantic_flow()
    print("ALL ENGINE INTEGRATION TESTS PASSED")
