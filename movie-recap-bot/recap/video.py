"""Assemble the final recap video with ffmpeg.

Inputs (per language):
    * source movie clip(s) you own   -> background montage
    * narration mp3 + timing cues    -> audio track + subtitle timing
    * an .ass subtitle file          -> burned into the frames

Produced output (per language):
    output/<name>_<lang>.mp4

The montage is built to be *at least* as long as the narration; `-shortest`
trims it exactly to the narration so audio stays the master clock. If the
clips are shorter than the narration, they are looped seamlessy.

If no real clips are provided, `make_storyboard()` fabricates colored scene
clips so the whole pipeline is runnable end-to-end (useful for testing/demo).
"""
from __future__ import annotations

from pathlib import Path

from .util import probe_duration, run, which_ffmpeg

SCALE_FILL = (
    "scale=1920:1080:force_original_aspect_ratio=increase,"
    "crop=1920:1080,setsar=1"
)


def normalize_clip(src: Path, out: Path, cfg_video: dict, trim: float | None = None) -> Path:
    """Convert a source clip to a uniform 1920x1080@fps, silent, h264 fragment."""
    fps = int(cfg_video.get("fps", 30))
    vf = f"{SCALE_FILL},fps={fps},setpts=PTS-STARTPTS"
    cmd = [
        which_ffmpeg(), "-y",
        "-i", str(src),
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        "-video_track_timescale", "90000",
    ]
    if trim:
        cmd += ["-t", f"{trim:.3f}"]
    cmd += [str(out)]
    run(cmd)
    return out


def _concat(normalized: list[Path], workdir: Path) -> Path:
    listfile = workdir / "concat.txt"
    lines = []
    for p in normalized:
        lines.append(f"file '{p.as_posix()}'")
    listfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = workdir / "concat.mp4"
    run(
        [
            which_ffmpeg(), "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c", "copy",
            str(out),
        ]
    )
    return out


def compose_base(clips: list[Path], target_duration: float, cfg_video: dict, workdir: Path) -> Path:
    """Build a silent base video of at least `target_duration` seconds."""
    if not clips:
        raise ValueError("No clips provided. Provide film clips or run with --storyboard.")

    normdir = workdir / "norm"
    normdir.mkdir(parents=True, exist_ok=True)
    normalized = []
    for i, c in enumerate(clips):
        out = normdir / f"seg_{i:03d}.mp4"
        if out.exists():
            # cache reuse across the per-language runs
            normalized.append(out)
        else:
            normalized.append(normalize_clip(c, out, cfg_video))
    concat = _concat(normalized, workdir)
    total = probe_duration(concat)

    base = workdir / "base.mp4"
    cmd = [which_ffmpeg(), "-y"]
    if total >= target_duration:
        cmd += ["-i", str(concat)]
    else:
        # loop the montage until it covers the narration
        cmd += ["-stream_loop", "-1", "-i", str(concat)]
    cmd += [
        "-t", f"{target_duration:.3f}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(int(cfg_video.get("fps", 30))),
        "-an",
        str(base),
    ]
    run(cmd)
    return base


def add_bgm_if_any(base: Path, bgm: str, volume: float, workdir: Path) -> Path:
    if not bgm:
        return base
    out = workdir / "base_bgm.mp4"
    run(
        [
            which_ffmpeg(), "-y",
            "-i", str(base),
            "-i", bgm,
            "-filter_complex", f"[1:a]volume={volume}[bg];[bg]aloop=loop=-1:size=2e9[lo]",
            "-map", "0:v", "-map", "[lo]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(out),
        ],
        check=False,
    )
    return out


def burn_and_mux(
    base: Path,
    narration_mp3: Path,
    ass_path: Path,
    out_mp4: Path,
    cfg_video: dict,
) -> Path:
    """Burn subtitles (ASS) and add narration as the audio track."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            which_ffmpeg(), "-y",
            "-i", str(base),
            "-i", str(narration_mp3),
            "-vf", f"ass='{ass_path.as_posix()}'",
            "-map", "0:v", "-map", "1:a",
            "-c:v", cfg_video.get("codec", "libx264"),
            "-c:a", cfg_video.get("audio_codec", "aac"),
            "-shortest",
            "-movflags", "+faststart",
            str(out_mp4),
        ]
    )
    return out_mp4


# --------------------------------------------------------------------------
# Storyboard fallback (colored scene clips) so the pipeline runs without clips
# --------------------------------------------------------------------------
def make_storyboard(count: int, cfg_video: dict, workdir: Path, duration: float = 3.0) -> list[Path]:
    """Generate `count` short silent color clips to stand in for movie footage."""
    fps = int(cfg_video.get("fps", 30))
    colors = ["0x20304a", "0x2a1f3a", "0x301a1a", "0x1f2a35", "0x332a1f"]
    sbdir = workdir / "storyboard"
    sbdir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for i in range(count):
        c = colors[i % len(colors)]
        out = sbdir / f"scene_{i:03d}.mp4"
        run(
            [
                which_ffmpeg(), "-y",
                "-f", "lavfi",
                "-i", f"color=c={c}:s=1920x1080:d={duration}:r={fps}",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-an",
                str(out),
            ],
            check=False,
        )
        if out.exists():
            clips.append(out)
    return clips
