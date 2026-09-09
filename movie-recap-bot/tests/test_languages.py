"""Multi-language engine tests (AR / ES native authoring, ZH translation).

Covers:
  * the language catalog + config defaults (voices, Arabic subtitle font)
  * per-language subtitle discovery (<movie>.<code>.srt never steals the
    untagged English .srt)
  * language-aware prompts (chunk summaries + section script writer)
  * a full auto_recap run with TWO authored languages (en + ar) producing
    <name>_en.mp4 and <name>_ar.mp4 with per-language intermediates
  * a ZH-only run that auto-authors the English master and translates it
    (back-compat: zh needs no Chinese subtitle)

Run from the movie-recap-bot folder:

    python tests/test_languages.py
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

from recap import languages, pipeline  # noqa: E402
from recap.config import load_config  # noqa: E402
from recap.tts import TimedCue  # noqa: E402

SENTENCES = [
    f"Fixture narration sentence {i} races forward with energy." for i in range(12)
]


def _cues(lang: str, n: int = 40) -> list[dict]:
    prefix = "ARABIC" if lang == "ar" else ("ESP" if lang == "es" else "EN")
    return [
        {"text": f"{prefix} dialogue line {i}",
         "start": i * 12.0, "end": i * 12.0 + 6.0,
         "words": [{"word": "x", "start": i * 12.0, "end": i * 12.0 + 1.0}]}
        for i in range(n)
    ]


class _FakeTTS:
    speech, gap = 1.0, 0.2

    def synthesize(self, sentences, voice, out_mp3):
        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        out_mp3.write_bytes(b"ID3fakeaudio")
        cues, t = [], 0.0
        for s in sentences:
            cues.append(TimedCue(s, t, t + self.speech))
            t += self.speech + self.gap
        return cues


def _install_stubs():
    """Stub whisper/LLM/ffmpeg like test_engine_integration does."""
    saved = {}

    def _save(mod, name):
        saved[(mod, name)] = getattr(mod, name)

    for modname, names in [
        (pipeline.dialogue, ["extract_dialogue"]),
        (pipeline.summarize, ["summarize_chunks"]),
        (pipeline.script, ["generate_segmented_script"]),
        (pipeline.llm, ["verify_model"]),
        (pipeline.tts, ["make_provider"]),
        (pipeline.clip, ["build_locked_visual"]),
        (pipeline.video, ["burn_and_mux_locked"]),
        (pipeline, ["probe_duration"]),
    ]:
        for n in names:
            _save(modname, n)

    pipeline.llm.verify_model = lambda cfg_llm: None

    def _extract(video, srt=None, **kw):
        lang = kw.get("lang") or "en"
        return _cues(lang)

    pipeline.dialogue.extract_dialogue = _extract

    def _summarize(chunks, cfg, **kw):
        lang = kw.get("lang") or "en"
        return [f"summary {lang} of chunk {c['index']}" for c in chunks]

    pipeline.summarize.summarize_chunks = _summarize

    def _segmented(chunk_summaries, cfg_llm, target, **kw):
        n = max(len(chunk_summaries), 1)
        out = []
        for i, s in enumerate(SENTENCES):
            c = chunk_summaries[min(i * n // len(SENTENCES), n - 1)]
            out.append({"sentence": s, "film_start": float(c["start"]),
                        "film_end": float(c["end"])})
        return out

    pipeline.script.generate_segmented_script = _segmented
    pipeline.tts.make_provider = lambda *a, **k: _FakeTTS()

    def _visual(movie, cuts, workdir, cfg_video, audio_span, mode="reencode"):
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / "visual.mp4"
        out.write_bytes(b"fakebasevideo")
        return out

    pipeline.clip.build_locked_visual = _visual

    def _burn(base, narration, ass, out_mp4, cfg_video, duration=None):
        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(Path(base).read_bytes() + b"muxed")
        return out_mp4

    pipeline.video.burn_and_mux_locked = _burn

    def _probe(p):
        name = Path(p).name
        if name.endswith(".mp3"):
            return 12 * 1.0 + 11 * 0.2 + 1.0   # speech + gaps + tail silence
        if name.endswith(".mp4") and "langs_" in name:
            return 15.2
        return 600.0

    pipeline.probe_duration = _probe

    def _restore():
        for (mod, name), fn in saved.items():
            setattr(mod, name, fn)

    return _restore


def _base_cfg(tmp: Path, langs: list[str]) -> dict:
    os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-used")
    cfg = load_config()
    cfg["llm"] = {"provider": "deepseek", "model": "deepseek-chat",
                  "base_url": "https://api.deepseek.com/v1"}
    cfg["vision"] = {"enabled": False, "provider": "gemini"}
    out = tmp / "out"
    out.mkdir(parents=True, exist_ok=True)
    cfg["project"]["_out"] = out
    cfg["project"]["output_dir"] = str(out)
    cfg["project"]["name"] = "langs"
    cfg["language"]["target_languages"] = list(langs)
    cfg["language"]["_resolved"] = [{"code": c, "tag": c} for c in langs]
    cfg["narration"]["words_target"] = 200
    cfg["narration"]["words_min"] = 60
    cfg["narration"]["words_max"] = 4200
    return cfg


def test_catalog_and_config_defaults() -> None:
    assert languages.validate(["en", "ar", "es", "zh"]) == []
    assert languages.validate(["xx"]) == ["xx"]
    assert languages.name("ar") == "Arabic"
    assert languages.voice_default("ar") == "ar-SA-HamedNeural"
    assert languages.voice_default("es") == "es-MX-JorgeNeural"
    cfg = load_config()
    lv = cfg["narration"]["lang_voice"]
    assert lv["ar"] == "ar-SA-HamedNeural" and lv["es"] == "es-MX-JorgeNeural"
    assert cfg["subtitles"]["lang_font"]["ar"] == "Arial"
    assert languages.font_for("en", cfg["subtitles"]) == cfg["subtitles"]["font"]
    assert languages.font_for("ar", cfg["subtitles"]) == "Arial"
    print("  catalog + config defaults OK")


def test_subtitle_discovery_per_language() -> None:
    from recap import dialogue

    tmp = Path(tempfile.mkdtemp(prefix="langsub-"))
    movie = tmp / "Toy Story 5.mp4"
    movie.write_bytes(b"x")
    (tmp / "Toy Story 5.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n",
                                         encoding="utf-8")
    (tmp / "Toy Story 5.ar.srt").write_text("x", encoding="utf-8")
    (tmp / "Toy Story 5_es.srt").write_text("x", encoding="utf-8")

    assert dialogue.find_subtitle_near(movie, lang="ar").name == "Toy Story 5.ar.srt"
    assert dialogue.find_subtitle_near(movie, lang="es").name == "Toy Story 5_es.srt"
    assert dialogue.find_subtitle_near(movie, lang="zh") is None
    # untagged .srt belongs to the English source, never stolen
    assert dialogue.find_subtitle_near(movie, lang="en").name == "Toy Story 5.srt"
    assert dialogue.find_subtitle_near(movie).name == "Toy Story 5.srt"
    # explicit path wins for any language
    ext = tmp / "elsewhere.srt"
    ext.write_text("y", encoding="utf-8")
    assert dialogue.find_subtitle_near(movie, str(ext), lang="ar") == ext
    print("  per-language subtitle discovery OK")


def test_prompts_are_language_aware() -> None:
    from recap import llm as rllm
    from recap import script, summarize

    calls: list[str] = []

    def fake_complete(provider, model, system, user, **kw):
        calls.append(user)
        return "[00:00:01] Beat line.\n[00:00:05] Another beat."

    rllm.complete = fake_complete
    summarize.summarize_chunks(
        [{"index": 0, "text": "some dialogue", "start": 0.0, "end": 30.0}],
        {"provider": "x", "model": "y"}, lang="ar",
    )
    assert "in Arabic" in calls[-1], calls[-1]

    summarize.summarize_chunks(
        [{"index": 0, "text": "dialogue", "start": 0.0, "end": 30.0}],
        {"provider": "x", "model": "y"},   # default lang en
    )
    assert "in Arabic" not in calls[-1]

    n0 = len(calls)
    script.generate_segmented_script(
        [{"index": 0, "start": 0.0, "end": 30.0, "summary": "s",
          "beats": [{"t": 1.0, "text": "beat"}]}],
        {"provider": "x", "model": "y"}, 400, lang_name="Arabic",
    )
    assert "entirely in Arabic" in calls[n0], calls[n0]   # section writer
    assert "in Arabic" in calls[-1], calls[-1]            # humanizer pass

    n1 = len(calls)
    script.generate_segmented_script(
        [{"index": 0, "start": 0.0, "end": 30.0, "summary": "s",
          "beats": [{"t": 1.0, "text": "beat"}]}],
        {"provider": "x", "model": "y"}, 400, lang_name="Spanish",
    )
    assert "entirely in Spanish" in calls[n1]
    assert "in Spanish" in calls[-1]
    print("  language-aware summarize + script prompts OK")


def _write_srt(path: Path) -> None:
    lines = []
    for i in range(40):
        lines += [str(i + 1),
                  f"00:{i // 60:02d}:{i % 60:02d},000 --> "
                  f"00:{i // 60:02d}:{(i % 60) + 6:02d},000",
                  f"dialogue line {i}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_two_authored_languages_flow() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="langs-e2e-"))
    movie = tmp / "film.mp4"
    movie.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    _write_srt(tmp / "film.ar.srt")     # real file: discovery finds it
    cfg = _base_cfg(tmp, ["en", "ar"])

    restore = _install_stubs()
    try:
        outs = pipeline.auto_recap(cfg, movie)
    finally:
        restore()

    names = sorted(p.name for p in outs)
    assert names == ["langs_ar.mp4", "langs_en.mp4"], names
    wd = tmp / "out" / "_work"
    assert (wd / "transcript.json").exists()          # en keeps legacy names
    assert (wd / "transcript_ar.json").exists()
    assert (wd / "script" / "script_en.segments.json").exists()
    assert (wd / "script" / "script_ar.segments.json").exists()
    assert (wd / "script" / "script_ar.txt").exists()
    assert (wd / "beats_ar.json").exists()
    assert (wd / "ar.mp3").exists() and (wd / "ar.srt").exists()
    for b in json.loads((wd / "beats_ar.json").read_text(encoding="utf-8")):
        pass
    print(f"  en+ar run OK -> {names}")


def test_zh_only_translates_forced_en_master() -> None:
    import recap.llm as rllm

    tmp = Path(tempfile.mkdtemp(prefix="langs-zh-"))
    movie = tmp / "film.mp4"
    movie.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    cfg = _base_cfg(tmp, ["zh"])

    def fake_translate(provider, model, system, user, **kw):
        return "这是翻译后的第一句台词。\n这是第二句。"

    rllm.complete = fake_translate

    restore = _install_stubs()
    try:
        outs = pipeline.auto_recap(cfg, movie)
    finally:
        restore()

    names = sorted(p.name for p in outs)
    assert names == ["langs_zh.mp4"], names
    wd = tmp / "out" / "_work"
    # zh never reads its own subtitle: the English master was authored for it
    assert (wd / "transcript.json").exists()
    assert (wd / "script" / "script_zh.txt").exists()
    # translation reused the EN film windows via the master segments
    beats = json.loads((wd / "beats_zh.json").read_text(encoding="utf-8"))
    assert beats
    starts = [b["film_start"] for b in beats]
    assert starts == sorted(starts)
    print(f"  zh-only run OK -> {names} (en master + translation)")


if __name__ == "__main__":
    test_catalog_and_config_defaults()
    test_subtitle_discovery_per_language()
    test_prompts_are_language_aware()
    test_two_authored_languages_flow()
    test_zh_only_translates_forced_en_master()
    print("\nALL LANGUAGE TESTS PASSED")
