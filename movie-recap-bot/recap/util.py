"""Shared helpers: ffmpeg/ffprobe resolution, subprocess runners, timing utils."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

try:
    import static_ffmpeg  # type: ignore

    static_ffmpeg.add_paths()
except Exception:  # pragma: no cover - static_ffmpeg may be absent
    pass


def which_ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    raise RuntimeError(
        "ffmpeg not found on PATH. Install system ffmpeg or `pip install static-ffmpeg`."
    )


def which_ffprobe() -> str:
    p = shutil.which("ffprobe")
    if p:
        return p
    raise RuntimeError(
        "ffprobe not found on PATH. Install system ffmpeg or `pip install static-ffmpeg`."
    )


def run(
    cmd: list[str],
    check: bool = True,
    capture: bool = True,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, echoing the call for debuggability.

    ``cwd`` matters for filtergraph arguments: a Windows path like
    ``D:\\recap\\_work\\en.ass`` contains a colon, and ffmpeg's filter parser
    reads ``:`` as an option separator, so absolute paths in filters break.
    Callers pass the file's folder here and use a bare relative filename.
    """
    pretty = " ".join(str(c) for c in cmd)
    print(f"  $ {pretty}")
    res = subprocess.run(
        [str(c) for c in cmd],
        capture_output=capture,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    if check and res.returncode != 0:
        raise RuntimeError(
            f"Command failed ({res.returncode}): {pretty}\n"
            f"STDOUT:\n{res.stdout[-3000:]}\nSTDERR:\n{res.stderr[-3000:]}"
        )
    return res


def probe_duration(path: str | Path) -> float:
    """Return media duration in seconds via ffprobe."""
    p = Path(path)
    res = run(
        [
            which_ffprobe(),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(p),
        ]
    )
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0


def probe_streams(path: str | Path) -> list[dict]:
    """Return a compact list of media streams (codec_type, width, height, etc.)."""
    p = Path(path)
    res = run(
        [
            which_ffprobe(),
            "-v", "error",
            "-show_streams",
            "-of", "json",
            str(p),
        ]
    )
    import json

    data = json.loads(res.stdout or "{}")
    streams = data.get("streams", [])
    out = []
    for s in streams:
        out.append(
            {
                "index": s.get("index"),
                "codec_type": s.get("codec_type"),
                "codec_name": s.get("codec_name"),
                "width": s.get("width"),
                "height": s.get("height"),
                "duration": s.get("duration"),
            }
        )
    return out


def fmt_ts(seconds: float) -> str:
    """Format seconds -> 'HH:MM:SS.mmm' (for SRT timestamps)."""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_ts_ass(seconds: float) -> str:
    """Format seconds -> 'H:MM:SS.cc' (ASS timestamps)."""
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 3600_00)
    m, rem = divmod(rem, 60_00)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))
