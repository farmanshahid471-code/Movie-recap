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

Interrupted runs resume: transcript, chunk summaries, the generated script,
the narration and the final renders are all gated by content-signature marker
files under ``_work``, so a network or power failure never forces a re-run of
the expensive steps that already finished.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import (chunk, clip, dialogue, llm, match, scenes, script, subtitles,
               summarize, timeline, translate, tts, video)
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


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def auto_recap(cfg: dict, movie: Path) -> list[Path]:
    """Run the full Step A-F flow on a movie file.

    Step A  extract dialogue (faster-whisper / .srt) -> 5-min overlapping chunks
            -> per-chunk action summaries
    Step B  section-by-section narration pass -> sentences tagged with the film
            window they describe (EN master), sized to hit the word target
    Step C  TTS narration per language with sentence (+word) timestamps
    Step D  chronological timeline: beats advance monotonically through the
            film, each locked to its narration cue (no vector search)
    Step E  ffmpeg-clip each micro-shot frame-exactly from the movie
    Step F  concat -> burn .ass subtitles -> mux narration at an explicit
            duration -> <name>_<code>.mp4

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

    # Summarization is the slowest LLM step on a CPU-only machine. Fail fast
    # when the model is missing (otherwise Ollama silently downloads it, which
    # looks identical to "stuck"), and use a smaller model if configured.
    summary_cfg = dict(cfg["llm"])
    if ck.get("model"):
        summary_cfg["model"] = ck["model"]          # chunking.model override
    summary_cfg["summary_model"] = summary_cfg.get("model", "")
    llm.verify_model(summary_cfg)

    if len(chunks) > 6:
        print("  * NOTE: this LLM summarization pass is the slow step on CPU-only "
              "machines. It prints per-chunk progress below and writes "
              "script/summaries.txt as it goes. To speed it up:\n"
              "      - set chunking.model to a smaller local model "
              "(e.g. qwen2.5:3b) in config.yaml, or\n"
              "      - use a cloud provider (LLM_PROVIDER=deepseek + key) for "
              "near-instant summaries,\n"
              "      - or raise chunking.window_seconds (e.g. 600) for fewer chunks.")
    summaries_path = tdir / "summaries.txt"
    # NOTE: do NOT truncate summaries.txt here — summarize_chunks() resumes
    # from whatever is already in it, so an interrupted run can continue from
    # the last finished chunk instead of redoing the slow CPU pass. Delete the
    # file to force a full re-run.
    print(f"  * Summarizing each chunk via "
          f"{summary_cfg.get('provider')}/{summary_cfg.get('model')} "
          f"({len(chunks)} chunks) ...")
    summaries = summarize.summarize_chunks(
        chunks,
        summary_cfg,
        parallel=bool(ck.get("parallel", False)),
        out_partial=summaries_path,
    )
    merged = summarize.merge_summaries(summaries)
    summaries_path.write_text(merged, encoding="utf-8")
    print(f"  * Summaries: {len(summaries)} chunks -> {len(merged.split())} words "
          f"(script/summaries.txt)")

    # ------------------------------------------------------------- Step B
    print("== Step B: Script generation (chronological, length-locked) ==")
    llm.verify_model(cfg["llm"])   # fail fast if the main model is missing
    nar = cfg["narration"]
    target = int(nar.get("words_target", 2000))
    mn = int(nar.get("words_min", 600))
    mx = int(nar.get("words_max", 4200))
    wpm = int(nar.get("words_per_minute", 150))
    print(f"  * Target: {target} words ≈ {target / max(wpm, 1) * 60:.0f}s of speech "
          f"at {wpm} wpm")

    # Pair each chunk with its summary so every sentence keeps the film window
    # it was written from — that is what makes Step D chronological.
    chunk_summaries = [
        {"index": c["index"], "start": c["start"], "end": c["end"],
         "summary": s}
        for c, s in zip(chunks, summaries)
    ]

    # Resume: reuse the existing script when the summaries + LLM settings are
    # unchanged (e.g. the previous run failed later at TTS). The signature
    # covers the LLM provider/model too, so switching models regenerates.
    b_marker = tdir / "script_en.marker.json"
    b_sig = _sig(merged, cfg["llm"].get("provider"), cfg["llm"].get("model"),
                 target, "segmented-v2")
    seg_path = tdir / "script_en.segments.json"
    segments = None
    if _marker_ok(b_marker, b_sig) and seg_path.exists():
        try:
            segments = json.loads(seg_path.read_text(encoding="utf-8")) or None
        except Exception:
            segments = None
        if segments:
            print(f"  * Reusing existing EN script ({len(segments)} sentences — "
                  f"matches this movie/summaries/LLM). Delete "
                  f"script_en.segments.json + script_en.marker.json to force a "
                  f"new script.")

    if segments is None:
        def _prog(done: int, total: int, got: int, budget: int) -> None:
            print(f"    ... section {done}/{total}: {got} words "
                  f"(budget {budget})", flush=True)

        print(f"  * Writing the recap section by section over {len(chunk_summaries)} "
              f"chunks (keeps every LLM call small and hits the length target) ...")
        segments = script.generate_segmented_script(
            chunk_summaries, cfg["llm"], target,
            words_per_minute=wpm, progress=_prog,
        )
        if len(segments) < 10:
            raise DialogueError(
                f"LLM returned only {len(segments)} sentences — the recap looks "
                "broken. Check the API key/credit, or try a larger model."
            )
        seg_path.write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_marker(b_marker, b_sig)

    sentences = [s["sentence"] for s in segments]
    got_words = count_words(" ".join(sentences))
    est = got_words / max(wpm, 1) * 60
    script.write_script_file("\n".join(sentences), tdir / "script_en.txt")
    (tdir / "script_en.json").write_text(
        json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  * EN recap: {len(sentences)} sentences, {got_words} words "
          f"≈ {est:.0f}s of speech (target {target} / "
          f"{target / max(wpm, 1) * 60:.0f}s)")
    if got_words < target * 0.75:
        print(f"  ! WARNING: the script is {got_words / max(target, 1) * 100:.0f}% of "
              f"the requested length, so the video will be ~{est:.0f}s not "
              f"{target / max(wpm, 1) * 60:.0f}s. A stronger model "
              f"(deepseek-chat) usually fixes this.")

    # ------------------------------------------------ Step C (narration)
    print("== Step C: Voiceover (TTS) ==")
    tts_cfg = cfg["narration"]
    prov = tts.make_provider(tts_cfg.get("tts_provider", "edge"), tts_cfg)
    lang_voice = tts_cfg.get("lang_voice", {})
    audios: dict[str, tuple[Path, list[tts.TimedCue]]] = {}
    for code in codes:
        voice = lang_voice.get(code) or lang_voice.get("en", "en-US-ChristopherNeural")
        lines = _resolve_narration_lines(cfg, code, sentences, wd)
        mp3 = wd / f"{code}.mp3"
        tj = wd / f"{code}.timing.json"
        # Resume: skip re-narrating when the mp3 + timing exist for these exact
        # lines/voice/provider (e.g. a run that failed later in clipping).
        c_marker = wd / f".nar_{code}.marker.json"
        c_sig = _sig(lines, voice, tts_cfg.get("tts_provider", "edge"),
                     tts_cfg.get("rate", "+0%"))
        if _marker_ok(c_marker, c_sig) and mp3.exists() and tj.exists():
            # NOTE: this MUST NOT be called `cues` — that name holds the movie
            # transcript and Step D matches against it. Shadowing it here made
            # a resumed run build its timeline from the narration instead of
            # the film, which silently produced nonsense visuals.
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
                print(f"  * Reusing narration {code} ({voice}) — {len(nar_cues)} lines "
                      f"already narrated. Delete {code}.mp3 to re-narrate.")
                continue
        print(f"  * Narrating {code} ({voice}) — {len(lines)} lines ...")
        audios[code] = tts.synthesize_language(
            lines, {"code": code, "voice": voice}, wd, prov, False
        )
        _write_marker(c_marker, c_sig)
        print(f"    -> {audios[code][0]}  ({audios[code][1][-1].end:.1f}s total)")

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

    results: list[Path] = []
    for code, (mp3, cues_t) in audios.items():
        out_mp4 = outd / f"{name}_{code}.mp4"

        # The narration span is the master clock: from 0 to the end of the mp3,
        # INCLUDING the silences between sentences. Measuring the real file
        # rather than the last cue's end keeps trailing silence in the render.
        audio_span = max(probe_duration(mp3), cues_t[-1].end if cues_t else 0.0)

        # Chronological, audio-locked beat plan for this language.
        durations = timeline.lock_durations(cues_t, audio_span)
        if code == "en":
            seg_for_lang = segments
        else:
            # Translations stay line-aligned, so reuse the EN film windows.
            seg_for_lang = [
                {"sentence": c.text,
                 "film_start": segments[min(i, len(segments) - 1)]["film_start"],
                 "film_end": segments[min(i, len(segments) - 1)]["film_end"]}
                for i, c in enumerate(cues_t)
            ]
        beats = timeline.build_timeline(seg_for_lang, durations, movie_dur, tl_cfg)
        _write_json(beats, wd / f"beats_{code}.json")
        print(f"  * [{code}] timeline: {timeline.timeline_report(beats, audio_span)}")

        # Resume: if the final render already exists for these exact inputs
        # (narration + beats + movie + subtitle/assembly settings), skip the
        # expensive ffmpeg clipping + burn entirely.
        ef_marker = wd / f".render_{code}.marker.json"
        ef_sig = _sig(
            str(out_mp4), [c.as_dict() for c in cues_t], beats,
            str(movie.resolve()), clip_mode, dict(cfg.get("subtitles", {})),
            vcfg.get("bgm", ""), float(vcfg.get("bgm_volume", 0.12)),
        )
        if _marker_ok(ef_marker, ef_sig) and out_mp4.exists() and out_mp4.stat().st_size > 0:
            results.append(out_mp4)
            print(f"  * Reusing existing render {out_mp4.name}. Delete it to re-render.")
            continue

        cuts = timeline.flatten_cuts(beats)
        print(f"  * [{code}] cutting {len(cuts)} shots from the film "
              f"(mode={clip_mode}) ...")
        visual = clip.build_locked_visual(
            movie, cuts, wd / "visual" / code, vcfg, audio_span, mode=clip_mode
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
        video.burn_and_mux_locked(base, mp3, ass, out_mp4, vcfg, duration=audio_span)
        _write_marker(ef_marker, ef_sig)
        results.append(out_mp4)
        final = probe_duration(out_mp4)
        drift = abs(final - audio_span)
        flag = "" if drift < 0.5 else f"   ! drift {drift:.2f}s"
        print(f"  + {out_mp4}  ({final:.1f}s vs narration {audio_span:.1f}s){flag}")

    print("\nDone. Outputs:")
    for r in results:
        print(f"   - {r}  ({probe_duration(r):.1f}s)")
    return results
