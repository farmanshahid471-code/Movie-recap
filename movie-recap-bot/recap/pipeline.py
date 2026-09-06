"""End-to-end recap pipeline.

Runs the full flow and writes per-language outputs to the output dir.

Two engines:

* ``run()`` — the classic 5-step engine (script file/LLM -> translate ->
  narrate -> subtitles -> montage assembly). Used by Recap Studio and the
  ``run`` CLI command.
* ``auto_recap()`` — the Step A-F engine: whisper -> contextual chunking ->
  chunk summarization -> JSON-array script -> TTS with timing -> semantic
  timestamp mapping (pgvector) -> ffmpeg clipping -> final assembly.
  Used by the ``auto`` CLI command.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import (chunk, clip, dialogue, llm, match, scenes, script, subtitles,
               summarize, translate, tts, video)
from .config import out_dir, work_dir
from .dialogue import DialogueError
from .util import count_words, probe_duration


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
            print(f"  * Auto-written EN recap from dialogue.")
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
        zh_voice = lang_voice.get("zh", "zh-CN-YunxiNeural")
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
        video.burn_and_mux(base, mp3, ass, out_mp4, cfg["video"])
        results.append(out_mp4)
        print(f"  + {out_mp4}")

    print("\nDone. Outputs:")
    for r in results:
        print(f"   - {r}  ({probe_duration(r):.1f}s)")
    return results


# ===========================================================================
# Step A-F engine (semantic auto-recap) — the specced production flow
# ===========================================================================

def _resolve_narration_lines(cfg: dict, code: str, en_sentences: list[str], wd: Path) -> list[str]:
    """Return the narration lines for one language (EN master; ZH translated).

    The English script is the master: anchors (semantic matches) are computed
    for it, and every other language stays line-aligned to reuse the same
    anchors. New languages can be added here later.
    """
    if code == "en":
        return en_sentences
    if code.startswith("zh"):
        provider = cfg["llm"].get("provider", "")
        tr = wd / "script" / "script_zh.txt"
        if tr.exists() and not cfg.get("regenerate_translation"):
            print(f"  * Using existing ZH translation: {tr}")
            return translate.normalize(tr.read_text(encoding="utf-8").splitlines()).splitlines()
        if not llm.provider_configured(provider):
            raise FileNotFoundError(
                "A ZH clip was requested but no translation exists and no LLM "
                "is configured to write one."
            )
        print(f"  * Translating EN recap to Simplified Chinese via {provider} ...")
        en_text = "\n".join(en_sentences)
        zh = translate.normalize(translate.generate_online(en_text, cfg["llm"]).splitlines())
        tr.parent.mkdir(parents=True, exist_ok=True)
        translate.write_translation_file(zh, tr)
        return zh.splitlines()
    raise NotImplementedError(
        f"Language '{code}' is not wired into the semantic engine yet. "
        "EN is primary; zh is supported via line-aligned translation."
    )


def _anchors_for_lines(beats: list[dict], n_lines: int) -> list[dict]:
    """Reindex anchors when a language's line count differs from EN's."""
    if n_lines == len(beats):
        return beats
    out = []
    for i in range(n_lines):
        src = beats[min(int(i * len(beats) / max(n_lines, 1)), len(beats) - 1)]
        out.append({**src, "index": i, "sentence": ""})
    return out


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def auto_recap(cfg: dict, movie: Path) -> list[Path]:
    """Run the full Step A-F flow on a movie file.

    Step A  extract dialogue (faster-whisper / .srt) -> 5-min overlapping chunks
            -> per-chunk action summaries
    Step B  final narrative pass -> JSON array of sentences (EN master)
    Step C  TTS narration per language with sentence (+word) timestamps
    Step D  embed transcript + script, cosine-match via pgvector store -> beats
    Step E  ffmpeg-clip each beat from the movie (stream copy by default)
    Step F  concat -> burn .ass subtitles -> mux narration -> <name>_<code>.mp4
    """
    movie = Path(movie)
    if not movie.exists():
        raise FileNotFoundError(f"Movie not found: {movie}")

    provider = cfg["llm"].get("provider", "")
    if not llm.provider_configured(provider):
        raise DialogueError(
            f"Auto-recap needs an LLM provider; current provider={provider!r}. "
            "Set LLM_PROVIDER (e.g. ollama) in .env / config.yaml."
        )

    outd = out_dir(cfg)
    wd = work_dir(cfg)
    outd.mkdir(parents=True, exist_ok=True)
    name = cfg["project"]["name"]
    resolved = cfg["language"]["_resolved"]
    codes = [l["code"] for l in resolved]

    movie_dur = probe_duration(movie)
    print(f"Movie: {movie}  ({movie_dur:.1f}s)")
    print(f"Engine: semantic auto-recap  languages={codes}")

    # ------------------------------------------------------------- Step A
    print("== Step A: Plot extraction & chunking ==")
    dlg = cfg.setdefault("dialogue", {})
    # Whisper is the slow step; when a studio job runs EN then ZH separately,
    # reuse the extraction of the same (unchanged) movie file.
    meta_p = wd / "transcript.meta.json"
    cached = False
    try:
        st = movie.stat()
        key = {"path": str(movie.resolve()), "size": st.st_size,
               "mtime_ns": st.st_mtime_ns}
        if meta_p.exists():
            cached = json.loads(meta_p.read_text(encoding="utf-8")) == key
    except OSError:
        cached = False

    if cached and (wd / "transcript.json").exists():
        cues = json.loads((wd / "transcript.json").read_text(encoding="utf-8"))
        print(f"  * Reusing cached transcription for unchanged movie "
              f"({len(cues)} cues). Delete {wd / 'transcript.json'} to force re-extract.")
    else:
        cues = dialogue.extract_dialogue(
            movie,
            dlg.get("srt_path"),
            whisper_model=dlg.get("whisper_model", "small"),
            whisper_device=dlg.get("whisper_device", "auto"),
            whisper_language=dlg.get("whisper_language"),
            word_timestamps=bool(dlg.get("word_timestamps", True)),
            tmp_dir=wd / "audio",
        )
        dialogue.write_cues_json(cues, wd / "transcript.json")
        dialogue.write_cues_srt(cues, wd / "transcript.srt")
        try:
            st = movie.stat()
            meta_p.write_text(
                json.dumps({"path": str(movie.resolve()), "size": st.st_size,
                            "mtime_ns": st.st_mtime_ns}),
                encoding="utf-8",
            )
        except OSError:
            pass
    if not cues:
        raise DialogueError(
            "No dialogue was extracted from the movie (no .srt found and "
            "Whisper returned nothing). Drop a .srt next to the movie or check "
            "the audio track."
        )
    tdir = wd / "script"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "transcript.txt").write_text(
        dialogue.transcript_markdown(cues), encoding="utf-8"
    )
    print(f"  * Transcript: {len(cues)} cues ({movie_dur:.0f}s of film), "
          f"word-level={bool(dlg.get('word_timestamps', True))} "
          f"(transcript.json / transcript.srt / script/transcript.txt)")

    ck = cfg.setdefault("chunking", {})
    chunks = chunk.chunk_cues(
        cues,
        float(ck.get("window_seconds", 300.0)),
        float(ck.get("overlap_seconds", 30.0)),
    )
    print(f"  * Contextual chunking: {len(chunks)} blocks of "
          f"{ck.get('window_seconds', 300)}s with "
          f"{ck.get('overlap_seconds', 30)}s overlap")
    for c in chunks:
        (wd / "chunks").mkdir(parents=True, exist_ok=True)
        (wd / "chunks" / f"chunk_{c['index']:03d}.txt").write_text(
            c["text"], encoding="utf-8"
        )

    print(f"  * Summarizing each chunk via {provider}/{cfg['llm'].get('model')} ...")
    summaries = summarize.summarize_chunks(
        chunks,
        cfg["llm"],
        parallel=bool(ck.get("parallel", False)),
    )
    merged = summarize.merge_summaries(summaries)
    (tdir / "summaries.txt").write_text(merged, encoding="utf-8")
    print(f"  * Summaries: {len(summaries)} chunks -> {len(merged.split())} words "
          f"(script/summaries.txt)")

    # ------------------------------------------------------------- Step B
    print("== Step B: Script generation (JSON array of sentences) ==")
    nar = cfg["narration"]
    target = int(nar.get("words_target", 2000))
    mn = int(nar.get("words_min", 600))
    mx = int(nar.get("words_max", 4200))
    sentences = script.generate_script_json(merged, cfg["llm"], target, mn, mx)
    if len(sentences) < 10:
        raise DialogueError(
            f"LLM returned only {len(sentences)} sentences — the recap looks "
            "broken. Retry with a larger model or check the Ollama context."
        )
    script.write_script_file("\n".join(sentences), tdir / "script_en.txt")
    (tdir / "script_en.json").write_text(
        json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  * EN recap: {len(sentences)} sentences, "
          f"{count_words(' '.join(sentences))} words (script_en.json/.txt)")

    # ------------------------------------------------ Step C (narration)
    print("== Step C: Voiceover (TTS) ==")
    tts_cfg = cfg["narration"]
    prov = tts.make_provider(tts_cfg.get("tts_provider", "edge"), tts_cfg)
    lang_voice = tts_cfg.get("lang_voice", {})
    audios: dict[str, tuple[Path, list[tts.TimedCue]]] = {}
    for code in codes:
        voice = lang_voice.get(code) or lang_voice.get("en", "en-US-ChristopherNeural")
        lines = _resolve_narration_lines(cfg, code, sentences, wd)
        print(f"  * Narrating {code} ({voice}) — {len(lines)} lines ...")
        audios[code] = tts.synthesize_language(
            lines, {"code": code, "voice": voice}, wd, prov, False
        )
        print(f"    -> {audios[code][0]}  ({audios[code][1][-1].end:.1f}s total)")

    # ------------------------------------------------ Step D (semantic map)
    print("== Step D: Semantic timestamp mapping ==")
    sem = cfg.setdefault("semantic", {})
    # compute beats once on the master (EN) script
    if not sem.get("enabled", True):
        beats = None
        print("  * semantic matching disabled -> evenly spaced anchors")
    else:
        embedder = match.Embedder(sem.get("embedding_model", "all-MiniLM-L6-v2"))
        store = match.make_store(sem, wd, dim=embedder.dim)
        try:
            beats = match.map_beats(
                sentences,
                cues,
                embedder,
                store,
                top_k=int(sem.get("top_k", 3)),
                min_score=float(sem.get("min_score", 0.10)),
                movie_duration=movie_dur,
            )
        finally:
            store.close()
        _write_json(beats, wd / "beats.json")
        n_ok = sum(1 for b in beats if not b.get("fallback"))
        print(f"  * Beats: {len(beats)} (semantic={n_ok}, fallback={len(beats) - n_ok}); "
              f"saved to beats.json")

    # ------------------------------------------------ Step E + F per language
    print("== Step E+F: Clipping & assembly ==")
    vcfg = cfg["video"]
    sem = cfg.setdefault("semantic", {})
    pre_roll = float(sem.get("pre_roll", 0.5))
    pad = float(sem.get("clip_pad", 0.15))
    min_clip = float(sem.get("min_clip", 0.8))
    max_clip = float(sem.get("max_clip", 10.0))
    clip_mode = str((sem.get("clip") or {}).get("mode", "copy")).lower()

    results: list[Path] = []
    for code, (mp3, cues_t) in audios.items():
        lines = [c.text for c in cues_t]
        anchors = _anchors_for_lines(beats or [], len(lines))
        if not anchors:
            # no transcript + no beats possible: even spacing over the runtime
            anchors = match._even_fallback(lines, movie_dur, [])
        windows: list[tuple[float, float]] = []
        for i, cue in enumerate(cues_t):
            need = max(cue.duration + pad, min_clip)
            need = min(need, max_clip)
            a = anchors[i] if i < len(anchors) else anchors[-1]
            src = float(a.get("start") or (i + 0.5) / max(len(lines), 1) * movie_dur)
            start = max(0.0, src - pre_roll)
            if movie_dur > 0:
                need = min(need, max(movie_dur - start, 0.2))
            windows.append((start, need))
        visual = clip.build_visual_from_windows(
            movie, windows, wd / "visual" / code, vcfg, mode=clip_mode
        )
        # subtitles synced to THIS language's narration
        sub_cfg = cfg["subtitles"]
        max_units = int(sub_cfg.get("line_width_units", 30))
        subs = subtitles.build_cues_for_subtitle(
            [c.as_dict() for c in cues_t], max_units
        )
        subtitles.write_srt(subs, wd / f"{code}.srt")
        subtitles.write_ass(subs, wd / f"{code}.ass", sub_cfg)
        base = video.add_bgm_if_any(
            visual, str(vcfg.get("bgm", "")), float(vcfg.get("bgm_volume", 0.12)),
            wd / "visual" / code,
        )
        ass = wd / f"{code}.ass"
        out_mp4 = outd / f"{name}_{code}.mp4"
        video.burn_and_mux(base, mp3, ass, out_mp4, vcfg)
        results.append(out_mp4)
        print(f"  + {out_mp4}  ({probe_duration(out_mp4):.1f}s)")

    print("\nDone. Outputs:")
    for r in results:
        print(f"   - {r}  ({probe_duration(r):.1f}s)")
    return results
