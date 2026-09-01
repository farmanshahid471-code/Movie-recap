"""Build burned-in subtitle files (SRT + ASS) from timed narration cues.

The pipeline burns an ASS subtitle into each language's video, styled to look
like a clean, always-on recap channel subtitle (slightly animated fade, clear
Outline/Shadow, bottom placement).

We also export SRT for accessibility / re-use.
"""
from __future__ import annotations

import json
from pathlib import Path

import pysubs2  # type: ignore

from .util import fmt_ts, fmt_ts_ass

LINE_DELAY = 0.12       # gap between cue end and next cue start, seconds
PAD_START = 0.05
PAD_END = 0.05


def _is_wide(ch: str) -> bool:
    """CJK/wide glyphs take a full character cell; Latin is half-width."""
    o = ord(ch)
    return (
        0x1100 <= o <= 0x115F
        or 0x2E80 <= o <= 0xA4CF
        or 0xAC00 <= o <= 0xD7A3
        or 0xF900 <= o <= 0xFAFF
        or 0xFF00 <= o <= 0xFF60
        or 0xFFE0 <= o <= 0xFFE6
        or 0x20000 <= o <= 0x2FFFD
        or 0x30000 <= o <= 0x3FFFD
    )


def _units(text: str) -> float:
    return sum(1.0 if _is_wide(ch) else 0.5 for ch in text)


def _wrap_long(text: str, max_units: int) -> list[str]:
    """Width-aware wrap: Chinese full-width glyphs count double, so a line of
    中文 fits fewer characters than a line of English. Prefers natural breaks."""
    if _units(text) <= max_units:
        return [text]
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if _units(trial) <= max_units:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def build_cues_for_subtitle(cues: list[dict], max_units: int) -> list[dict]:
    """Return subtitle events with `lines` (width-aware wrapped) + start/end."""
    subs: list[dict] = []
    for cue in cues:
        wrapped = _wrap_long(cue["text"], max_units)
        subs.append({"start": cue["start"], "end": cue["end"], "lines": wrapped})
    return subs


def write_srt(subs: list[dict], path: Path) -> Path:
    events = [
        pysubs2.SSAEvent(
            start=int(s["start"] * 1000),
            end=int(s["end"] * 1000),
            text="\\N".join(s["lines"]),
        )
        for s in subs
    ]
    sub = pysubs2.SSAFile()
    sub.events.extend(events)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.save(str(path), format_="srt", encoding="utf-8")
    return path


def _ass_font_style(font: str, fontsize: int, outline: int, shadow: int, margin_x: int) -> str:
    return (
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Top,{font},{fontsize},&H00FFFFFF,&H00FFFFFF,&H00101010,"
        f"&H96000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_x},{margin_x},0,1\n"
        f"Style: Bot,{font},{fontsize},&H00FFFFFF,&H00FFFFFF,&H00101010,"
        f"&H96000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_x},{margin_x},0,1\n"
    )


def write_ass(subs: list[dict], path: Path, cfg_sub: dict) -> Path:
    """Write a .ass file with a soft pop-in/fade-out fade used by recap channels."""
    font = cfg_sub.get("font", "Noto Serif CJK SC")
    fontsize = int(cfg_sub.get("fontsize", 56))
    outline = int(cfg_sub.get("outline", 3))
    shadow = int(cfg_sub.get("shadow", 1))
    margin_v = int(cfg_sub.get("margin_v", 96))
    margin_x = int(cfg_sub.get("margin_x", 40))

    lines: list[str] = []
    lines.append(
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "ScaledBorderAndShadow: yes\n"
        "WrapStyle: 0\n"          # smart auto-wrap so no line overflows the frame
    )
    lines.append(_ass_font_style(font, fontsize, outline, shadow, margin_x))
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    for s in subs:
        start = fmt_ts_ass(s["start"])
        end = fmt_ts_ass(s["end"])
        text = "\\N".join(s["lines"])
        # \fad(180,180) + \pos bottom via style Bot (Alignment=2 bottom-center)
        effect = r"\fad(150,150)"
        text = "{" + effect + "}" + text
        lines.append(
            f"Dialogue: 0,{start},{end},Bot,,0,0,{margin_v},,{text}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_timed_json(subs: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subs, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
