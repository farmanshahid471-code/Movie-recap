"""End-to-end recap pipeline.

Runs the full flow and writes per-language outputs to the output dir.
"""
from __future__ import annotations

from pathlib import Path

from . import dialogue, llm, scenes, script, subtitles, translate, tts, video
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
        whisper_device=cfg.get("dialogue", {}).get("whisper_device", "cpu"),
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
        except (DialogueError, llm.LLMError) as exc:
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
