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
    def __init__(self):
        self.n = 0

    def synthesize(self, sentences, voice, out_mp3):
        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        out_mp3.write_bytes(b"ID3fakeaudio")
        cues = []
        t = 0.0
        for s in sentences:
            cues.append(TimedCue(s, t, t + 1.0))
            t += 1.2
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
    _save(pipeline.match, "map_beats")
    _save(pipeline.match, "Embedder")
    _save(pipeline.tts, "make_provider")

    class _DummyEmbedder:
        dim = 384

        def __init__(self, model_name="", device="cpu"):
            pass

        def encode(self, texts):
            import numpy as np

            return np.zeros((max(len(texts), 1), self.dim), dtype=np.float32)

    pipeline.match.Embedder = _DummyEmbedder
    _save(pipeline.clip, "build_visual_from_windows")
    _save(pipeline.video, "burn_and_mux")
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

    def _map_beats(sentences, cues, embedder, store, **k):
        return [
            {"index": i, "sentence": s, "cue_idx": i, "start": 10.0 + i * 60.0,
             "end": 10.0 + i * 60.0 + 6.0, "score": 0.8,
             "source_text": cues[i]["text"], "fallback": False}
            for i, s in enumerate(sentences)
        ]

    pipeline.match.map_beats = _map_beats
    pipeline.tts.make_provider = lambda *a, **k: _FakeTTS()

    def _visual(movie, windows, workdir, cfg_video, mode="copy"):
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / "visual.mp4"
        out.write_bytes(b"fakebasevideo")
        return out

    pipeline.clip.build_visual_from_windows = _visual

    def _burn(base, narration, ass, out_mp4, cfg_video):
        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(Path(base).read_bytes() + b"muxed")
        return out_mp4

    pipeline.video.burn_and_mux = _burn
    pipeline.probe_duration = lambda p: 600.0  # movie + outputs

    def _restore():
        for (mod, name), fn in saved.items():
            setattr(mod, name, fn)

    return _restore


def test_full_semantic_flow() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="recap-e2e-"))
    movie = tmp / "film.mp4"
    movie.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # exists for Path checks

    cfg = load_config()
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
    assert (wd / "beats.json").exists()
    assert (wd / "en.mp3").exists()
    assert (wd / "en.srt").exists() and (wd / "en.ass").exists()
    assert (wd / "en.timing.json").exists()
    # semantic beat mapping wired the real code path
    beats = json.loads((wd / "beats.json").read_text(encoding="utf-8"))
    assert beats and not beats[0]["fallback"]
    sentences = json.loads((wd / "script" / "script_en.json").read_text(encoding="utf-8"))
    assert len(sentences) == len(SENTENCES)
    # .txt sidecar is line-aligned
    txt_lines = [l for l in (wd / "script" / "script_en.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(txt_lines) == len(sentences)
    print("  outputs written:", [p.name for p in Path(out).glob('*.mp4')])
    print("  intermediates: script_en.json, beats.json, en.mp3, en.srt, en.ass, en.timing.json OK")


if __name__ == "__main__":
    test_full_semantic_flow()
    print("ALL ENGINE INTEGRATION TESTS PASSED")
