"""End-to-end recap pipeline.

Runs the full flow and writes per-language outputs to the output dir.

Two engines:

* ``run()`` — the classic 5-step engine (script file/LLM -> translate ->
  narrate -> subtitles -> montage assembly). Used by Recap Studio and the
  ``run`` CLI command.
* ``auto_recap()`` — the Step A-F engine: whisper -> contextual chunking ->
  chunk summarization -> section-by-section JSON script -> TTS with timing ->
  chronological, audio-locked timeline -> frame-exact ffmpeg clipping -> final
  assembly. Used by the ``auto`` CLI command.

Interrupted runs resume: transcript, chunk summaries, the generated script,
the narration and the final renders are all gated by content-signature marker
files under ``_work``, so a network or power failure never forces a re-run of
the expensive steps that already finished.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import (align, chunk, clip, dialogue, languages, llm, scenes, script,
               subtitles, summarize, timeline, translate, tts, video, vision)
from .config import out_dir, work_dir
from .dialogue import DialogueError
from .util import count_words, probe_duration


def _sig(*parts: object) -> str:
    """Short content signature over the inputs that shape an expensive step."""
    h = hashlib.sha1()
    for p in parts:
        s = p if isinstance(p, str) else json.dumps(p, ensure_ascii=False, sort_keys=True)
        h.update(s.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:20]


def _marker_ok(marker: Path, sig: str) -> bool:
    try:
        return marker.exists() and json.loads(
            marker.read_text(encoding="utf-8")
        ).get("sig") == sig
    except Exception:
        return False


def _write_marker(marker: Path, sig: str) -> None:
    try:
        marker.write_text(json.dumps({"sig": sig}), encoding="utf-8")
    except OSError:
        pass


def _get_plot(cfg: dict, workdir: Path) -> str:
    """Optional plot summary used to guide LLM scripting (may be empty)."""
    notes_file = workdir / "script" / "plot_notes.txt"
    if notes_file.exists():
        return notes_file.read_text(encoding="utf-8").strip()
    # Fall back to the repo's bundled sample plot notes if present.
    bundled = workdir / "script" / "plot_notes.txt"
    return bundled.read_text(encoding="utf-8").strip() if bundled.exists() else ""


def _auto_script(cfg: dict, workdir: Path, video: Path | None) -> str:
    """Write the EN recap from the movie's dialogue/transcript via the LLM."""
    provider = cfg["llm"].get("provider", "")
    if not llm.provider_configured(provider):
        raise DialogueError(
            f"Auto-recap needs an LLM provider; current provider={provider!r}. "
            "Set LLM_PROVIDER (e.g. ollama) or provide a pre-written script."
        )

    print("  * Extracting dialogue/transcript ...")
    srt = cfg.get("llm", {}).get("srt_path") or cfg.get("dialogue", {}).get("srt_path")
    cues = dialogue.extract_dialogue(
        video,
        srt,
        whisper_model=cfg.get("dialogue", {}).get("whisper_model", "small"),
        # "auto" lets faster-whisper pick the GPU when one is present and fall
        # back to CPU otherwise — a big speed-up on GPU machines, no config needed.
        whisper_device=cfg.get("dialogue", {}).get("whisper_device", "auto"),
        whisper_language=cfg.get("dialogue", {}).get("whisper_language"),
        tmp_dir=workdir / "audio",   # scratch wav stays out of the media folder
    )
    transcript = dialogue.to_transcript_text(cues, max_chars=cfg.get("dialogue", {}).get("max_chars"))
    tdir = workdir / "script"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "transcript.txt").write_text(
        dialogue.transcript_markdown(cues), encoding="utf-8"
    )
    print(f"  * Transcript: {len(cues)} cues, {len(transcript)} chars "
          f"(saved to script/transcript.txt)")

    plot = _get_plot(cfg, workdir)
    print(f"  * Writing EN recap from dialogue via {provider} ...")
    return script.normalize(
        script.generate_from_dialogue(
            transcript,
            plot,
            cfg["llm"],
            cfg["narration"]["words_target"],
            cfg["narration"]["words_min"],
            cfg["narration"]["words_max"],
        ).splitlines()
    )


def _get_script(cfg: dict, workdir: Path, video: Path | None = None) -> str:
    """Return EN script text + source metadata."""
    sdir = workdir / "script"
    sdir.mkdir(parents=True, exist_ok=True)

    script_text: str | None = None

    # 1) Pre-written script file? (most control)
    for name in ("script_en.txt", "recap.txt", "script.txt"):
        p = sdir / name
        if p.exists():
            script_text = script.load_script(p)
            print(f"  * Using pre-written EN script: {p}")
            break

    # 2) Auto-recap from the movie's dialogue (if a video is present and LLM set).
    if script_text is None and video is not None:
        try:
            script_text = _auto_script(cfg, workdir, video)
            print("  * Auto-written EN recap from dialogue.")
        except llm.LLMError as exc:
            # The movie and dialogue were fine — the LLM is unreachable. Do NOT
            # fall through to the plot-summary path (its "provide script_en.txt"
            # error is misleading here); say what to actually do.
            raise llm.LLMError(
                "Auto-recap read your movie's dialogue but could not reach the LLM to "
                f"write the narration ({exc}). If you use Ollama, start it: run "
                "`ollama serve` and `ollama pull qwen2.5`. Otherwise pick a provider and "
                "key in Settings -> LLM."
            ) from exc
        except DialogueError as exc:
            print(f"  ! Auto-recap unavailable ({exc}); falling back.")
            script_text = None

    # 3) LLM from a plot summary.
    if script_text is None:
        provider = cfg["llm"].get("provider", "")
        model = cfg["llm"].get("model", "")
        if llm.provider_configured(provider):
            notes = _get_plot(cfg, workdir)
            if not notes:
                raise FileNotFoundError(
                    "Auto-recap needs a movie/dialogue or a plot summary. Provide "
                    "script_en.txt, a movie to transcribe, or plot_notes.txt."
                )
            print(f"  * Writing EN recap via {provider}/{model} ...")
            script_text = script.normalize(
                script.generate_online(
                    notes,
                    cfg["llm"],
                    cfg["narration"]["words_target"],
                    cfg["narration"]["words_min"],
                    cfg["narration"]["words_max"],
                ).splitlines()
            )
        else:
            raise FileNotFoundError(
                "No EN script found. Provide script_en.txt OR a movie/transcript "
                "with an LLM provider configured."
            )

    text = script.normalize(script_text.splitlines())
    script.write_script_file(text, sdir / "script_en.txt")
    print(f"  * EN script: {count_words(text)} words, {len(text.splitlines())} lines")
    return text


def _get_translation(en_text: str, cfg: dict, workdir: Path) -> str:
    tdir = workdir / "script"
    tr_path = tdir / "script_zh.txt"
    provider = cfg["llm"].get("provider", "")
    if tr_path.exists():
        print(f"  * Using pre-written ZH translation: {tr_path}")
        return translate.load_translation(tr_path)
    if llm.provider_configured(provider):
        print(f"  * Translating to Simplified Chinese via {provider} ...")
        return translate.normalize(
            translate.generate_online(en_text, cfg["llm"]).splitlines()
        )
    raise FileNotFoundError(
        "No ZH translation found. Provide script_zh.txt (line-aligned with EN) "
        "or configure an LLM provider."
    )


def run(cfg: dict, clips: list[Path], storyboard: bool = False) -> list[Path]:
    outd = out_dir(cfg)
    wd = work_dir(cfg)
    outd.mkdir(parents=True, exist_ok=True)
    name = cfg["project"]["name"]

    resolved = cfg["language"]["_resolved"]
    codes = [l["code"] for l in resolved]
    extra = [c for c in codes if c not in ("en", "zh")]
    if extra:
        raise DialogueError(
            f"The classic `run` engine narrates en + zh clips only (requested: "
            f"{extra}). Use `python -m recap.cli auto ...` for "
            "Arabic/Spanish — the semantic engine writes those natively from "
            "their subtitles."
        )
    need_en = "en" in codes
    need_zh = "zh" in codes

    print("== Step 1/5: Script (EN) ==")
    # The base story/transcript is always the English script; used directly for
    # the EN clip and as the source for the ZH translation. If no pre-written
    # script exists, a video + LLM lets us auto-write it from the dialogue.
    source_video = clips[0] if clips else None
    en_text = _get_script(cfg, wd, video=source_video)
    en_lines = en_text.splitlines()
    print(f"  * EN script: {count_words(en_text)} words, {len(en_lines)} lines")

    zh_text: str = ""
    if need_zh:
        print("== Step 2/5: Translate to Simplified Chinese ==")
        zh_text = _get_translation(en_text, cfg, wd)
        zh_lines = zh_text.splitlines()
        print(f"  * ZH script: {len(zh_lines)} lines")
    else:
        zh_lines = []

    print("== Step 3/5: Narrate (TTS) ==")
    provider = tts.make_provider(cfg["narration"].get("tts_provider", "edge"), cfg["narration"])
    lang_voice = cfg["narration"].get("lang_voice", {})
    audios: dict[str, tuple[Path, list[tts.TimedCue]]] = {}
    split_seg = bool(cfg["narration"].get("segment_audio", False))

    if need_en:
        en_voice = lang_voice.get("en", "en-US-ChristopherNeural")
        print(f"  * Narrating EN ({en_voice}) ...")
        audios["en"] = tts.synthesize_language(
            en_lines, {"code": "en", "voice": en_voice}, wd, provider, split_seg
        )
    if need_zh:
        zh_voice = lang_voice.get("zh", "zh-CN-YunjianNeural")
        print(f"  * Narrating ZH ({zh_voice}) ...")
        audios["zh"] = tts.synthesize_language(
            zh_lines, {"code": "zh", "voice": zh_voice}, wd, provider, split_seg
        )

    print("== Step 4/5: Subtitles (SRT + ASS) ==")
    sub_cfg = cfg["subtitles"]
    max_units = int(sub_cfg.get("line_width_units", 30))
    subs_by_lang: dict[str, list[dict]] = {}
    for code, (mp3, cues) in audios.items():
        cue_list = [c.as_dict() for c in cues]
        subs = subtitles.build_cues_for_subtitle(cue_list, max_units)
        subtitles.write_srt(subs, wd / f"{code}.srt")
        subtitles.write_ass(subs, wd / f"{code}.ass", sub_cfg)
        subtitles.write_timed_json(subs, wd / f"{code}.subs.json")
        subs_by_lang[code] = subs

    print("== Step 5/5: Assemble video ==")
    # Each language gets a montage sized to its own narration length, so the
    # audio stays the master clock and subtitles remain in sync.
    results: list[Path] = []
    if storyboard or not clips:
        if not clips:
            print("  * No clips -> generating storyboard placeholder scenes.")
            clips = video.make_storyboard(8, cfg["video"], wd)

    for code, (mp3, cues) in audios.items():
        target = max((c.end for c in cues), default=5.0) + 0.5
        print(f"  * {code}: narration {target:.1f}s over {len(cues)} cues")
        montage = str(cfg["video"].get("montage", "scenes")).lower()
        if montage == "scenes" and len(clips) == 1 and not storyboard:
            # recap-channel style: cut real beats from the film itself
            base = scenes.build_montage(clips[0], target, cfg["video"], wd)
        else:
            base = video.compose_base(clips, target, cfg["video"], wd)
        base = video.add_bgm_if_any(base, cfg["video"].get("bgm", ""), cfg["video"].get("bgm_volume", 0.12), wd)
        ass = wd / f"{code}.ass"
        out_mp4 = outd / f"{name}_{code}.mp4"
        # Explicit duration, never -shortest: the montage is built to `target`
        # but any rounding there must not silently clip the narration.
        video.burn_and_mux_locked(base, mp3, ass, out_mp4, cfg["video"],
                                  duration=probe_duration(mp3))
        results.append(out_mp4)
        print(f"  + {out_mp4}")

    print("\nDone. Outputs:")
    for r in results:
        print(f"   - {r}  ({probe_duration(r):.1f}s)")
    return results


# ===========================================================================
# Step A-F engine (semantic auto-recap) — the specced production flow
# ===========================================================================

def _translation_lines(
    cfg: dict, wd: Path, target_code: str, source_code: str,
    source_sentences: list[str],
) -> list[str]:
    """Line-aligned LLM translation of an authored recap into ``target_code``.

    Every translated sentence is paired 1:1 with its source sentence so it can
    reuse that sentence's film window (the chronological timeline is built from
    the source's windows). Cached per exact source text + provider + model, so
    re-runs reuse the file and stale translations are never paired with the
    wrong film moments.
    """
    provider = cfg["llm"].get("provider", "")
    source_text = "\n".join(source_sentences)
    tdir = wd / "script"
    tdir.mkdir(parents=True, exist_ok=True)
    tr = tdir / f"script_{target_code}.txt"
    marker = tdir / f"script_{target_code}.marker.json"
    sig = _sig(source_text, target_code, source_code,
               cfg["llm"].get("provider"), cfg["llm"].get("model"))
    if tr.exists() and not cfg.get("regenerate_translation") and (
        _marker_ok(marker, sig) or not marker.exists()
    ):
        print(f"  * Using existing {languages.name(target_code)} translation: {tr}")
        return translate.normalize(
            tr.read_text(encoding="utf-8").splitlines()
        ).splitlines()
    if not llm.provider_configured(provider):
        raise FileNotFoundError(
            f"A {languages.name(target_code)} clip was requested but no "
            f"{languages.name(target_code)} subtitle exists and no LLM is "
            "configured to translate the source recap."
        )
    print(f"  * Translating the {languages.name(source_code)} recap to "
          f"{languages.name(target_code)} via {provider} ...")
    raw = translate.generate_translation(
        source_text, cfg["llm"], target=target_code, source=source_code
    )
    lines = translate.normalize(raw.splitlines()).splitlines()
    aligned, why = translate.check_alignment(
        source_sentences, lines,
        target=f"{languages.name(target_code)} translation",
        source=f"{languages.name(source_code)} source",
    )
    if not aligned:
        print(f"  ! WARNING: {why}")
    translate.write_translation_file("\n".join(lines), tr)
    _write_marker(marker, sig)
    return lines


def _transcript_key(movie: Path, source_srt: str | None, dlg: dict) -> dict:
    """Cache identity for one language's dialogue extraction.

    Covers the movie file AND the exact dialogue source (subtitle file bytes or
    the whisper model/device), so swapping a subtitle never reuses another
    language's transcript.
    """
    st = movie.stat()
    if source_srt:
        try:
            ps = Path(source_srt).stat()
            src = {"kind": "srt", "path": str(Path(source_srt).resolve()),
                   "size": ps.st_size, "mtime_ns": ps.st_mtime_ns}
        except OSError:
            src = {"kind": "srt", "path": str(source_srt), "size": 0, "mtime_ns": 0}
    else:
        src = {"kind": "whisper",
               "model": dlg.get("whisper_model", "small"),
               "device": dlg.get("whisper_device", "auto")}
    return {"path": str(movie.resolve()), "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "src": src}


def _cached_transcript(cfg: dict, wd: Path, movie: Path, code: str,
                       source_srt: str | None) -> list[dict] | None:
    """Return the cached cues for one language when inputs are unchanged."""
    name = "transcript" if code == "en" else f"transcript_{code}"
    meta_p = wd / f"{name}.meta.json"
    json_p = wd / f"{name}.json"
    try:
        key = _transcript_key(movie, source_srt, cfg.get("dialogue", {}))
        if not meta_p.exists() or not json_p.exists():
            return None
        stored = json.loads(meta_p.read_text(encoding="utf-8"))
        # Pre-multi-language caches (no "src" field) recorded the movie only —
        # that was the whisper path, so accept it when the movie is unchanged.
        if stored == key:
            return json.loads(json_p.read_text(encoding="utf-8"))
        if "src" not in stored and key.get("src", {}).get("kind") == "whisper" \
                and stored.get("path") == key["path"] \
                and stored.get("size") == key["size"] \
                and stored.get("mtime_ns") == key["mtime_ns"]:
            return json.loads(json_p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _align_narration_for(
    wd: Path,
    code: str,
    mp3: Path,
    lines: list[str],
    cues: list,
    narr_cfg: dict,
    dlg_cfg: dict,
):
    """Whisper-align one narration track to its own audio (see recap/align.py).

    Returns ``(mp3, cues, aligned)``. The timing sidecar is rewritten with the
    refined cues so a resumed run reloads the aligned times without
    re-transcribing.
    """
    audio_span = max(probe_duration(mp3), cues[-1].end if cues else 0.0)
    model_size = narr_cfg.get("whisper_align_model") \
        or dlg_cfg.get("whisper_model", "small")
    cues, aligned = align.align_narration(
        mp3, lines, cues, audio_span, wd, code=code,
        model_size=model_size,
        device=dlg_cfg.get("whisper_device", "auto"),
        language=code,
        enabled=bool(narr_cfg.get("whisper_align", True)),
    )
    if aligned:
        try:
            (wd / f"{code}.timing.json").write_text(
                json.dumps([c.as_dict() for c in cues], ensure_ascii=False,
                           indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
    return mp3, cues, aligned


def auto_recap(cfg: dict, movie: Path) -> list[Path]:
    """Run the full Step A-F flow on a movie file.

    Step A  extract dialogue per authored language (provided subtitle, else
            Whisper on the movie's audio) -> overlapping chunks -> per-chunk
            action summaries in that language
    Step B  section-by-section narration pass -> sentences tagged with the film
            window they describe, sized to hit the word target
    Step C  TTS narration per language with sentence (+word) timestamps;
            the generated audio is then re-measured with faster-whisper
            (narration.whisper_align, default on) so every cue and word is
            locked to what is actually spoken — required for openai/elevenlabs
            voices, which return a bare mp3. Languages without their own
            dialogue are line-aligned translations of the authored master
            recap (their sentences reuse the master's film windows)
    Step D  chronological timeline: beats advance monotonically through the
            film, each locked to its narration cue (no vector search)
    Step E  ffmpeg-clip each micro-shot frame-exactly from the movie
    Step F  concat -> burn .ass subtitles -> mux narration at an explicit
            duration -> <name>_<code>.mp4

    Languages: "en" is always authored (its own subtitles or Whisper).
    "zh"/"ar"/"es" are authored natively from a subtitle tagged for that
    language (<movie>.<lang>.srt, language.sources, or --subtitle-<lang>);
    without one they fall back to translating the English recap (zh's
    long-standing behaviour), which keeps runs usable when only the movie's
    own English dialogue is available.

    Video length always equals narration length; the render is never truncated
    by -shortest and never drifts out of sync.
    """
    movie = Path(movie)
    if not movie.exists():
        raise FileNotFoundError(f"Movie not found: {movie}")

    provider = cfg["llm"].get("provider", "")
    if not llm.provider_configured(provider):
        hint = {
            "deepseek": "Set DEEPSEEK_API_KEY in .env (get a key at "
                        "platform.deepseek.com -> API keys).",
            "openai": "Set OPENAI_API_KEY in .env.",
            "groq": "Set GROQ_API_KEY in .env.",
            "gemini": "Set GEMINI_API_KEY in .env.",
            "anthropic": "Set ANTHROPIC_API_KEY in .env.",
        }.get((provider or "").strip().lower(),
              "Set LLM_PROVIDER=deepseek and DEEPSEEK_API_KEY in .env.")
        raise DialogueError(
            f"Auto-recap needs a working LLM; provider={provider!r} is not "
            f"configured. {hint}"
        )

    outd = out_dir(cfg)
    wd = work_dir(cfg)
    outd.mkdir(parents=True, exist_ok=True)
    name = cfg["project"]["name"]
    resolved = cfg["language"]["_resolved"]
    codes = [l["code"] for l in resolved]

    unsupported = languages.validate(codes)
    if unsupported:
        raise DialogueError(
            f"Unsupported language(s): {unsupported}. Supported: "
            f"{', '.join(languages.SUPPORTED)}."
        )

    movie_dur = probe_duration(movie)
    print(f"Movie: {movie}  ({movie_dur:.1f}s)")
    print(f"Engine: semantic auto-recap  languages={codes}")

    # ---------------------------------------------------- dialogue sources
    dlg = cfg.setdefault("dialogue", {})
    lang_cfg = cfg.setdefault("language", {})
    sources_cfg = {str(k).strip().lower(): str(v).strip()
                   for k, v in (lang_cfg.get("sources") or {}).items()
                   if str(v).strip()}
    bare_srt = (dlg.get("srt_path") or "").strip() or None
    # A single-language, non-English job with one explicit subtitle treats it
    # as that language's source (keeps `--langs ar --subtitle movie.ar.srt` working).
    if bare_srt and "en" not in codes and len(codes) == 1:
        sources_cfg.setdefault(codes[0], bare_srt)
    # Locate each requested language's own subtitle (<movie>.<code>.srt etc.).
    native_sources: dict[str, str] = {}
    for code in codes:
        found = dialogue.find_subtitle_near(movie, sources_cfg.get(code),
                                            lang=code)
        if found:
            native_sources[code] = str(found)

    # Which languages are authored (written from their own dialogue) vs
    # translated (line-aligned from the authored master recap)?
    native = [c for c in codes if c == "en" or c in native_sources]
    translated = [c for c in codes if c not in native]
    if translated:
        if "en" not in native:
            # Any non-native language needs a master recap to translate from;
            # the movie's own English dialogue provides it (as before), so
            # author English too even though no EN clip was requested.
            native.insert(0, "en")
            print("  * Languages requested without their own subtitle will be "
                  "translated from the English recap — reading the movie's "
                  "English dialogue as the translation source ...")
        master_code = "en"
    else:
        master_code = None
    if not native:
        raise DialogueError(
            f"No dialogue source for languages {codes}. Provide a subtitle per "
            "language (name it <movie>.<code>.srt next to the film, e.g. "
            "ToyStory5.ar.srt, or set language.sources in config.yaml / "
            "--subtitle-<code> on the CLI), or include 'en' so Whisper can "
            "read the movie's own audio."
        )

    # ------------------------------------- Step A: transcript per language
    print("== Step A: Plot extraction & chunking ==")
    tdir = wd / "script"
    tdir.mkdir(parents=True, exist_ok=True)
    transcripts: dict[str, list[dict]] = {}
    for code in native:
        source_srt = native_sources.get(code)
        cache_name = "transcript" if code == "en" else f"transcript_{code}"
        cached = _cached_transcript(cfg, wd, movie, code, source_srt)
        if cached is not None:
            transcripts[code] = cached
            print(f"  * Reusing cached {languages.name(code)} transcript for "
                  f"unchanged movie ({len(cached)} cues). Delete "
                  f"{wd / (cache_name + '.json')} to force re-extract.")
            continue
        cues = dialogue.extract_dialogue(
            movie,
            source_srt,
            whisper_model=dlg.get("whisper_model", "small"),
            whisper_device=dlg.get("whisper_device", "auto"),
            whisper_language=dlg.get("whisper_language"),
            word_timestamps=bool(dlg.get("word_timestamps", True)),
            tmp_dir=wd / "audio",
            lang=code,
        )
        transcripts[code] = cues
        dialogue.write_cues_json(cues, wd / f"{cache_name}.json")
        dialogue.write_cues_srt(cues, wd / f"{cache_name}.srt")
        try:
            meta_p = wd / f"{cache_name}.meta.json"
            meta_p.write_text(
                json.dumps(_transcript_key(movie, source_srt, dlg)),
                encoding="utf-8",
            )
        except OSError:
            pass
        _tx = "transcript" if code == "en" else f"transcript_{code}"
        (tdir / f"{_tx}.txt").write_text(
            dialogue.transcript_markdown(cues), encoding="utf-8"
        )
        src_note = f" (subtitle {source_srt})" if source_srt else " (Whisper)"
        print(f"  * {languages.name(code)} transcript: {len(cues)} cues"
              f"{src_note} ({movie_dur:.0f}s of film), "
              f"word-level={bool(dlg.get('word_timestamps', True))} "
              f"(transcript{'' if code == 'en' else '_' + code}.json / .srt)")
    if not transcripts:
        raise DialogueError("No dialogue was extracted for any requested "
                            "language.")
    ck = cfg.setdefault("chunking", {})
    window = float(ck.get("window_seconds", 300.0))
    overlap = float(ck.get("overlap_seconds", 30.0))

    # ------------------------------------------------- Step A (pass 1.5) vision
    # DeepSeek cannot see the film, so silent set-pieces would never be
    # narrated. If a vision provider key is configured (default: GEMINI_API_KEY
    # free tier), caption on-screen action once per movie (cached + resumable)
    # and attach the notes to every language's chunks by film time. No key ->
    # graceful text-only (current behaviour).
    vcfg = cfg.get("vision") or {}
    visual_notes: list[dict] = []
    if vcfg.get("enabled", True):
        try:
            visual_notes = vision.capture(movie, vcfg, wd)
        except Exception as exc:  # never let the visual pass kill a movie run
            print(f"  ! Vision pass failed ({exc}); continuing text-only.",
                  flush=True)
            visual_notes = []
        if visual_notes:
            print(f"  * Vision notes: {len(visual_notes)} on-screen moments "
                  f"merged into each language's beat list.")

    def _attach_visual(chunks: list[dict]) -> None:
        by_t = {int(n.get("t", -1)): (n.get("text") or "").strip()
                for n in visual_notes if n.get("t") is not None}
        if not by_t:
            return
        for c in chunks:
            lo, hi = float(c.get("start", 0.0)), float(c.get("end", 0.0))
            vis = [{"t": t, "text": by_t[t]}
                   for t in sorted(by_t)
                   if lo - 1.0 <= t < hi and by_t[t]]
            if vis:
                c["visual"] = vis

    # Summarization is the slowest LLM step on a CPU-only machine. Fail fast
    # when the model is missing (otherwise Ollama silently downloads it, which
    # looks identical to "stuck"), and use a smaller model if configured.
    summary_cfg = dict(cfg["llm"])
    if ck.get("model"):
        summary_cfg["model"] = ck["model"]          # chunking.model override
    summary_cfg["summary_model"] = summary_cfg.get("model", "")
    llm.verify_model(summary_cfg)

    if len(codes) > 1:
        print("  * NOTE: every language written natively runs its own "
              "summarize + script pass (one LLM pass per language). Languages "
              "without subtitles are translated instead and add only one "
              "translation call each.")
    print(f"  * Contextual chunking: {window:.0f}s windows, "
          f"{overlap:.0f}s overlap")

    # per-language chunked summaries + segmented scripts
    authored: dict[str, dict] = {}        # code -> {segments, sentences}
    for code in native:
        cues = transcripts[code]
        chunks = chunk.chunk_cues(cues, window, overlap)
        _attach_visual(chunks)
        (wd / "chunks").mkdir(parents=True, exist_ok=True)
        for c in chunks:
            _tag = "" if code == "en" else f"{code}_"
            (wd / "chunks" / f"chunk_{_tag}{c['index']:03d}.txt").write_text(
                c["text"], encoding="utf-8"
            )
        lang_name = languages.name(code)
        partial_name = "summaries.txt" if code == "en" else f"summaries_{code}.txt"
        summaries_path = tdir / partial_name
        print(f"  * Summarizing {len(chunks)} {lang_name} chunks via "
              f"{summary_cfg.get('provider')}/{summary_cfg.get('model')} ...")
        summaries = summarize.summarize_chunks(
            chunks,
            summary_cfg,
            parallel=bool(ck.get("parallel", False)),
            out_partial=summaries_path,
            lang=code,
        )
        merged = summarize.merge_summaries(summaries)
        summaries_path.write_text(merged, encoding="utf-8")
        print(f"  * {lang_name} summaries: {len(summaries)} chunks -> "
              f"{len(merged.split())} words (script/{partial_name})")

        # ---------------------------------------------- Step B per language
        print(f"== Step B: {lang_name} script generation "
              "(chronological, length-locked) ==")
        llm.verify_model(cfg["llm"])   # fail fast if the main model is missing
        nar = cfg["narration"]
        target = int(nar.get("words_target", 2000))
        wpm = int(nar.get("words_per_minute", 150))
        # VISUAL MATCH: never ask for more narration than the film can show
        # at 1x. A recap is normally far shorter than its film, so this only
        # fires for extreme targets (e.g. 25 minutes out of a 30-minute
        # movie) -- the picture then stays at full speed instead of drifting
        # into permanent slow motion.
        if nar.get("visual_match", True) and movie_dur > 0:
            film_cap = int(movie_dur / 60.0 * wpm * 0.75)
            if film_cap >= 600 and target > film_cap:
                print(f"  * visual match: target {target} words exceeds what "
                      f"the film can show at 1x ({film_cap} words); clamping "
                      "so the narration never outruns the footage.")
                target = film_cap
        print(f"  * Target: {target} words ≈ "
              f"{target / max(wpm, 1) * 60:.0f}s of speech at {wpm} wpm")

        chunk_summaries = [
            {"index": c["index"], "start": c["start"], "end": c["end"],
             "summary": s, "beats": summarize.parse_beats(s)}
            for c, s in zip(chunks, summaries)
        ]

        b_marker = tdir / f"script_{code}.marker.json"
        b_sig = _sig(merged, cfg["llm"].get("provider"),
                     cfg["llm"].get("model"), target,
                     bool(nar.get("sign_off", True)), "segmented-v12")
        seg_path = tdir / f"script_{code}.segments.json"
        segments = None
        if _marker_ok(b_marker, b_sig) and seg_path.exists():
            try:
                segments = json.loads(seg_path.read_text(encoding="utf-8")) or None
            except Exception:
                segments = None
            if segments:
                print(f"  * Reusing existing {lang_name} script "
                      f"({len(segments)} sentences — matches this "
                      f"movie/summaries/LLM). Delete "
                      f"script_{code}.segments.json + "
                      f"script_{code}.marker.json to force a new script.")

        if segments is None:
            def _prog(done: int, total: int, got: int, budget: int) -> None:
                print(f"    ... section {done}/{total}: {got} words "
                      f"(budget {budget})", flush=True)

            print(f"  * Writing the {lang_name} recap section by section over "
                  f"{len(chunk_summaries)} chunks ...")
            segments = script.generate_segmented_script(
                chunk_summaries, cfg["llm"], target,
                words_per_minute=wpm, progress=_prog, lang_name=lang_name,
                sign_off=bool(nar.get("sign_off", True)),
                visual_match=bool(nar.get("visual_match", True)),
                humanize=bool(nar.get("humanize", True)),
            )
            if len(segments) < 10:
                raise DialogueError(
                    f"LLM returned only {len(segments)} sentences — the recap "
                    "looks broken. Check the API key/credit, or try a larger "
                    "model."
                )
            seg_path.write_text(
                json.dumps(segments, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _write_marker(b_marker, b_sig)

        sentences = [s["sentence"] for s in segments]
        got_words = count_words(" ".join(sentences))
        est = got_words / max(wpm, 1) * 60
        script.write_script_file("\n".join(sentences),
                                 tdir / f"script_{code}.txt")
        (tdir / f"script_{code}.json").write_text(
            json.dumps(sentences, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  * {lang_name} recap: {len(sentences)} sentences, "
              f"{got_words} words ≈ {est:.0f}s of speech (target {target} / "
              f"{target / max(wpm, 1) * 60:.0f}s)")
        if got_words < target * 0.75:
            print(f"  ! WARNING: the script is "
                  f"{got_words / max(target, 1) * 100:.0f}% of the requested "
                  f"length, so the video will be ~{est:.0f}s not "
                  f"{target / max(wpm, 1) * 60:.0f}s. A stronger model "
                  f"(deepseek-chat) usually fixes this.")
        authored[code] = {"segments": segments, "sentences": sentences}

    # --------------------------------------------- translations (if needed)
    # A translated language reuses its master recap's film windows 1:1, so its
    # narration stays in sync with the exact moments the master described.
    master_segments = authored[master_code]["segments"] if master_code else []
    narration: dict[str, list[str]] = {}
    for code in native:
        narration[code] = authored[code]["sentences"]
    for code in translated:
        assert master_code is not None
        lines = _translation_lines(cfg, wd, code, master_code,
                                   authored[master_code]["sentences"])
        narration[code] = lines

    # ------------------------------------------------ Step C (narration)
    print("== Step C: Voiceover (TTS) ==")
    tts_cfg = cfg["narration"]
    prov = tts.make_provider(tts_cfg.get("tts_provider", "edge"), tts_cfg)
    lang_voice = tts_cfg.get("lang_voice", {})
    audios: dict[str, tuple[Path, list[tts.TimedCue]]] = {}
    for code in codes:
        voice = lang_voice.get(code) or languages.voice_default(code)
        lines = narration[code]
        mp3 = wd / f"{code}.mp3"
        tj = wd / f"{code}.timing.json"
        c_marker = wd / f".nar_{code}.marker.json"
        c_sig = _sig(lines, voice, tts_cfg.get("tts_provider", "edge"),
                     tts_cfg.get("rate", "+0%"),
                     tts_cfg.get("pitch", "-0Hz"))
        if _marker_ok(c_marker, c_sig) and mp3.exists() and tj.exists():
            try:
                nar_cues = [
                    tts.TimedCue(
                        d.get("text", ""), float(d.get("start", 0.0)),
                        float(d.get("end", 0.0)),
                        [(w["word"], float(w["start"]), float(w["end"]))
                         for w in d.get("words") or []],
                    )
                    for d in json.loads(tj.read_text(encoding="utf-8"))
                ]
            except Exception:
                nar_cues = []
            if nar_cues:
                audios[code] = (mp3, nar_cues)
                print(f"  * Reusing narration {code} ({voice}) — "
                      f"{len(nar_cues)} lines already narrated. Delete "
                      f"{code}.mp3 to re-narrate.")
                continue
        print(f"  * Narrating {code} ({voice}) — {len(lines)} lines ...")
        mp3, cues_t = tts.synthesize_language(
            lines, {"code": code, "voice": voice}, wd, prov, False
        )
        # --- Whisper alignment (Step C+) -----------------------------------
        # Run the GENERATED narration audio back through faster-whisper and
        # re-anchor every cue (and every word) to what is actually spoken.
        # This is what keeps the visuals glued to the voice for ANY TTS
        # provider: edge-tts already reports word boundaries, but OpenAI /
        # ElevenLabs return a bare mp3 (the old code guessed their cue times
        # proportionally, drifting seconds off). Cached per audio content.
        align_cfg = cfg["narration"]
        if align_cfg.get("whisper_align", True):
            dlg_cfg = cfg.get("dialogue", {})
            mp3, cues_t, _ = _align_narration_for(
                wd, code, mp3, lines, cues_t, align_cfg, dlg_cfg
            )
        audios[code] = (mp3, cues_t)
        _write_marker(c_marker, c_sig)
        print(f"    -> {audios[code][0]}  "
              f"({audios[code][1][-1].end:.1f}s total)")

    # ------------------------------------------------ Step D (chronological map)
    # The timeline is built per language, because each language's narration has
    # its own cue boundaries and the visual must lock to THOSE. See Step E+F.
    print("== Step D+E+F: Chronological timeline, clipping & assembly ==")

    vcfg = cfg["video"]
    tl_cfg = cfg.setdefault("timeline", {})
    sem = cfg.setdefault("semantic", {})
    # clip.mode default is now "reencode": stream-copy snaps every cut to the
    # nearest keyframe, and those errors accumulate into seconds of A/V drift.
    clip_mode = str((sem.get("clip") or {}).get("mode", "reencode")).lower()

    # Optional: the film's real shot boundaries, so every visual cut lands on
    # an actual camera cut (the reference-channel edit feel — no mid-shot
    # drift-ins). One cached PySceneDetect pass per movie.
    snap_bounds: list[float] = []
    if tl_cfg.get("snap_to_scenes", True):
        try:
            snap_bounds, _method = scenes.scene_boundaries(movie, vcfg, wd)
            if snap_bounds:
                print(f"  * {len(snap_bounds)} shot boundaries on file — "
                      "visual cuts will land on the film's real shot changes")
            else:
                print("  * shot boundaries unavailable (optional: "
                      "`pip install scenedetect[opencv]` unlocks "
                      "cut-on-shot-change) — continuing un-snapped")
        except Exception as exc:
            print(f"  ! shot-boundary detection failed ({exc}); "
                  "continuing without snapping.")

    results: list[Path] = []
    for code, (mp3, cues_t) in audios.items():
        out_mp4 = outd / f"{name}_{code}.mp4"

        # The narration span is the master clock: from 0 to the end of the mp3,
        # INCLUDING the silences between sentences. Measuring the real file
        # rather than the last cue's end keeps trailing silence in the render.
        audio_span = max(probe_duration(mp3),
                         cues_t[-1].end if cues_t else 0.0)

        # Chronological, audio-locked beat plan for this language.
        durations = timeline.lock_durations(cues_t, audio_span)
        tl_stats: dict = {}
        if code in authored:
            seg_for_lang = authored[code]["segments"]
        else:
            # Translated lines stay 1:1 with the master, so reuse the master's
            # film windows (clamped so a slightly misaligned translation can
            # never index past the end).
            ms = master_segments
            seg_for_lang = [
                {"sentence": c.text,
                 "film_start": ms[min(i, len(ms) - 1)]["film_start"],
                 "film_end": ms[min(i, len(ms) - 1)]["film_end"],
                 "zone_lo": ms[min(i, len(ms) - 1)].get("zone_lo"),
                 "zone_hi": ms[min(i, len(ms) - 1)].get("zone_hi")}
                for i, c in enumerate(cues_t)
            ]
        # VISUAL MATCH, measured: the script sized every sentence's window
        # from an ESTIMATE of its speech (words / words_per_minute), before
        # any audio existed. The voice speaks at its own real rate
        # (rate "-8%", the voice itself, the language), so when the real
        # narration is slower than the estimate every window comes out
        # smaller than its sentence -- the timeline then slow-moes
        # essentially every section and the narration runs ahead of the
        # picture from the first scene. The REAL durations are now
        # measured; re-size each section's windows to them so every
        # section that fits its footage plays at exactly 1x.
        seg_for_lang = timeline.rewindow_to_speech(
            seg_for_lang, durations, movie_dur)
        beats = timeline.build_timeline(
            seg_for_lang, durations, movie_dur, tl_cfg,
            word_times=[c.words for c in cues_t],
            stats=tl_stats,
            scene_bounds=snap_bounds,
        )
        _write_json(beats, wd / f"beats_{code}.json")
        _report = timeline.timeline_report(
            beats, audio_span, tl_stats.get("word_locked_beats", 0),
            tl_stats.get("snapped_cuts", 0),
            tl_stats.get("slowed_groups", 0),
            tl_stats.get("slowed_seconds", 0.0),
        )
        print(f"  * [{code}] timeline: {_report}")
        # Honest sync prognosis: if sections still had to slow down after
        # the measured re-windowing, say WHY -- the section's narration is
        # genuinely longer than the film behind it (over-budget section or
        # a far slower voice than words_per_minute assumes).
        _slowed = tl_stats.get("slowed_groups", 0)
        if _slowed:
            print(f"  ! [{code}] {_slowed} section(s) still play below 1x: "
                  "their measured narration is longer than the film zone "
                  "behind them. If this is most of the video, lower "
                  "narration.words_target (or raise words_per_minute to "
                  "match the voice's real rate).")
        # Measured narration rate vs configured -- grounds future wpm
        # tuning in reality instead of guesses.
        try:
            _spoken = count_words(" ".join(c.text for c in cues_t))
            if audio_span > 0 and _spoken > 0:
                _real = _spoken / audio_span * 60
                print(f"  * [{code}] measured narration rate: "
                      f"{_real:.0f} wpm (configured words_per_minute: "
                      f"{wpm}; window sizing no longer depends on this "
                      "guess)")
        except Exception:
            pass

        # Resume: if the final render already exists for these exact inputs
        # (narration + beats + movie + subtitle/assembly settings), skip the
        # expensive ffmpeg clipping + burn entirely.
        ef_marker = wd / f".render_{code}.marker.json"
        ef_sig = _sig(
            str(out_mp4), [c.as_dict() for c in cues_t], beats,
            str(movie.resolve()), clip_mode, dict(cfg.get("subtitles", {})),
            vcfg.get("bgm", ""), float(vcfg.get("bgm_volume", 0.12)),
        )
        if _marker_ok(ef_marker, ef_sig) and out_mp4.exists() \
                and out_mp4.stat().st_size > 0:
            results.append(out_mp4)
            print(f"  * Reusing existing render {out_mp4.name}. Delete it to "
                  "re-render.")
            continue

        cuts = timeline.flatten_cuts(beats)
        print(f"  * [{code}] cutting {len(cuts)} shots from the film "
              f"(mode={clip_mode}) ...")
        visual = clip.build_locked_visual(
            movie, cuts, wd / "visual" / code, vcfg, audio_span, mode=clip_mode
        )

        # subtitles synced to THIS language's narration (font aware per lang)
        sub_cfg = dict(cfg["subtitles"])
        sub_cfg["font"] = languages.font_for(code, cfg["subtitles"])
        max_units = int(sub_cfg.get("line_width_units", 30))
        subs = subtitles.build_cues_for_subtitle(
            [c.as_dict() for c in cues_t], max_units
        )
        subtitles.write_srt(subs, wd / f"{code}.srt")
        subtitles.write_ass(subs, wd / f"{code}.ass", sub_cfg)
        base = video.add_bgm_if_any(
            visual, str(vcfg.get("bgm", "")),
            float(vcfg.get("bgm_volume", 0.12)), wd / "visual" / code,
        )
        ass = wd / f"{code}.ass"
        video.burn_and_mux_locked(base, mp3, ass, out_mp4, vcfg,
                                  duration=audio_span)
        _write_marker(ef_marker, ef_sig)
        results.append(out_mp4)
        final = probe_duration(out_mp4)
        drift = abs(final - audio_span)
        flag = "" if drift < 0.5 else f"   ! drift {drift:.2f}s"
        print(f"  + {out_mp4}  ({final:.1f}s vs narration "
              f"{audio_span:.1f}s){flag}")

    print("\nDone. Outputs:")
    for r in results:
        print(f"   - {r}  ({probe_duration(r):.1f}s)")
    return results
