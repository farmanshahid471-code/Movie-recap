"""Command-line interface.

Examples:
    # Full pipeline for a movie you own:
    python -m recap.cli run --movie path/to/source.mp4

    # Multiple clips: extra --movie args, or a comma-separated list
    python -m recap.cli run --movie clip1.mp4,clip2.mp4,clip3.mp4

    # Skip needing real footage (placeholder scenes) for testing:
    python -m recap.cli run --movie clip1.mp4 --storyboard

    # Just write the recap script + Chinese translation (no video):
    python -m recap.cli script --plot plot_notes.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from . import llm, pipeline, script, translate
from .config import BASE_DIR, load_config, out_dir, work_dir


def _load_raw_yaml(path: str | None) -> dict:
    p = Path(path) if path else BASE_DIR / "config.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _resolve_clips(clips: list[str]) -> list[Path]:
    out: list[Path] = []
    for c in clips:
        for piece in c.split(","):
            piece = piece.strip()
            if piece and Path(piece).exists():
                out.append(Path(piece))
            elif piece:
                raise FileNotFoundError(f"Clip not found: {piece}")
    return out


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    # allow overrides from CLI args
    cfg["project"]["name"] = args.name or cfg["project"]["name"]
    if args.langs:
        cfg["language"]["target_languages"] = args.langs.split(",")
        # re-resolve
        langs = []
        for lang in cfg["language"]["target_languages"]:
            tag = lang
            if lang.startswith("zh"):
                tag = cfg["language"].get("zh_variant", "zh-CN")
            langs.append({"code": lang, "tag": tag})
        cfg["language"]["_resolved"] = langs
    if args.script_file:
        sfile = Path(args.script_file)
        wd = work_dir(cfg)
        sdir = wd / "script"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "script_en.txt").write_text(sfile.read_text(encoding="utf-8"), encoding="utf-8")
    if args.zh_file:
        zfile = Path(args.zh_file)
        wd = work_dir(cfg)
        sdir = wd / "script"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "script_zh.txt").write_text(zfile.read_text(encoding="utf-8"), encoding="utf-8")

    clips = _resolve_clips(args.movie or [])
    pipeline.run(cfg, clips, storyboard=args.storyboard)


def cmd_auto(args: argparse.Namespace) -> None:
    """Auto-recap a movie with the Step A-F engine (whisper -> chunks ->
    summaries -> JSON script -> TTS -> semantic match -> clips)."""
    cfg = load_config(args.config)
    cfg["project"]["name"] = args.name or cfg["project"]["name"]
    if args.langs:
        cfg["language"]["target_languages"] = args.langs.split(",")
        langs = []
        for lang in cfg["language"]["target_languages"]:
            tag = lang
            if lang.startswith("zh"):
                tag = cfg["language"].get("zh_variant", "zh-CN")
            langs.append({"code": lang, "tag": tag})
        cfg["language"]["_resolved"] = langs

    # Optional explicit subtitle path for dialogue extraction.
    if args.subtitle:
        cfg.setdefault("dialogue", {})["srt_path"] = args.subtitle
    # Whisper tuning via CLI.
    if args.whisper_model:
        cfg.setdefault("dialogue", {})["whisper_model"] = args.whisper_model
    if args.whisper_device:
        cfg.setdefault("dialogue", {})["whisper_device"] = args.whisper_device
    # Target narration length (full-length ~10-16 min by default).
    if args.minutes:
        mins = max(1, int(args.minutes))
        nar = cfg.setdefault("narration", {})
        words = int(mins * 150)
        words = max(words, int(nar.get("words_min", 600)))
        words = min(words, int(nar.get("words_max", 4200)))
        nar["words_target"] = words
        print(f"  * Target narration: ~{mins} min -> {words} words")

    movie = Path(args.movie)
    if not movie.exists():
        raise SystemExit(f"Movie not found: {movie}")

    # The semantic engine regenerates the script from the film's dialogue every
    # run — never reuse a stale staged script.
    sdir = work_dir(cfg) / "script"
    (sdir / "script_en.txt").unlink(missing_ok=True)
    (sdir / "script_zh.txt").unlink(missing_ok=True)

    pipeline.auto_recap(cfg, movie)


def cmd_script(args: argparse.Namespace) -> None:
    """Generate the EN recap script (and optionally the ZH translation)."""
    cfg = load_config(args.config)
    provider = cfg["llm"].get("provider", "")
    if not llm.provider_configured(provider):
        raise SystemExit(
            "An LLM provider is required to *write* the recap script. "
            "Set LLM_PROVIDER + key in .env, or provide a pre-written "
            "script_en.txt and use `run --script-file`."
        )
    notes_file = args.plot
    if not notes_file or not Path(notes_file).exists():
        raise SystemExit(f"Plot notes file not found: {notes_file}")
    notes = Path(notes_file).read_text(encoding="utf-8")
    cfg["llm"]["provider"] = provider
    en = script.generate_online(
        notes,
        cfg["llm"],
        cfg["narration"]["words_target"],
        cfg["narration"]["words_min"],
        cfg["narration"]["words_max"],
    )
    en = script.normalize(en.splitlines())
    wd = work_dir(cfg)
    sdir = wd / "script"
    sdir.mkdir(parents=True, exist_ok=True)
    script.write_script_file(en, sdir / "script_en.txt")
    print(f"\nEnglish recap written to {sdir/'script_en.txt'}")
    print(f"({len(en.splitlines())} lines)")

    if args.translate:
        zh = translate.generate_online(en, cfg["llm"])
        zh = translate.normalize(zh.splitlines())
        translate.write_translation_file(zh, sdir / "script_zh.txt")
        print(f"Chinese translation written to {sdir/'script_zh.txt'}")
        print(f"({len(zh.splitlines())} lines)")


def cmd_translate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    en = Path(args.en).read_text(encoding="utf-8")
    zh = translate.normalize(translate.generate_online(en, cfg["llm"]).splitlines())
    out = Path(args.out) if args.out else (work_dir(cfg) / "script" / "script_zh.txt")
    translate.write_translation_file(zh, out)
    print(f"Chinese translation written to {out}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="recap",
        description="Movie-Recaps-style recap bot (EN + Simplified Chinese)",
    )
    ap.add_argument("--config", default=None, help="path to an alternate config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline")
    p_run.add_argument("--movie", action="append", help="source clips (comma-separated)")
    p_run.add_argument("--storyboard", action="store_true", help="placeholder scenes if no clips")
    p_run.add_argument("--langs", default=None, help="comma list, e.g. en,zh")
    p_run.add_argument("--name", default=None, help="output name")
    p_run.add_argument("--script-file", default=None, help="pre-written EN script")
    p_run.add_argument("--zh-file", default=None, help="pre-written ZH translation")
    p_run.set_defaults(func=cmd_run)

    p_auto = sub.add_parser(
        "auto",
        help="auto-recap a movie: extract dialogue -> LLM writes recap -> clips",
    )
    p_auto.add_argument("--movie", required=True, help="movie file path")
    p_auto.add_argument("--subtitle", default=None, help="explicit dialogue subtitle (.srt)")
    p_auto.add_argument("--whisper-model", default=None, help="whisper model size")
    p_auto.add_argument("--whisper-device", default=None,
                        help="whisper device: auto | cpu | cuda | cuda:0")
    p_auto.add_argument("--minutes", default=None, type=int,
                        help="target narration length in minutes (~150 wpm)")
    p_auto.add_argument("--langs", default=None, help="comma list, e.g. en,zh")
    p_auto.add_argument("--name", default=None, help="output name")
    p_auto.set_defaults(func=cmd_auto)

    p_scr = sub.add_parser("script", help="write the EN recap script (+ZH)")
    p_scr.add_argument("--plot", required=True, help="plot notes text file")
    p_scr.add_argument("--translate", action="store_true", help="also translate to ZH")
    p_scr.set_defaults(func=cmd_script)

    p_tr = sub.add_parser("translate", help="translate an EN script to ZH")
    p_tr.add_argument("--en", required=True, help="EN script file")
    p_tr.add_argument("--out", default=None, help="output ZH file")
    p_tr.set_defaults(func=cmd_translate)

    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
