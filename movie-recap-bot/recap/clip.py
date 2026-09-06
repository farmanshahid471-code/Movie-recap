"""Step E & F — automated clipping and final assembly.

Given the array of matched movie timestamps (one per narration line), cut the
source film into segments with ffmpeg, concat them into one silent visual
track, then burn the .ass subtitles and mux the narration over it.

Speed vs accuracy:
    mode "copy"     -> ``-ss <start> -i movie -t <dur> -c copy``
                       (stream copy: fast, no re-encode; cut lands on the
                       nearest keyframe — usually within a frame or two)
    mode "reencode" -> frame-exact re-encode of every cut (slower)

The narration stays the master clock: every segment's *length* is the length
of its narration line (plus padding), so the concatenated visual is exactly as
long as the audio and subtitles never drift. The matched timestamp only
decides *where* in the film the beat's visual comes from.
"""
from __future__ import annotations

from pathlib import Path

from .util import probe_duration, run, which_ffmpeg


def cut_segment(
    movie: Path,
    out: Path,
    start: float,
    duration: float,
    cfg_video: dict,
    mode: str = "copy",
) -> Path:
    """Cut one segment of the source film.

    ``mode="copy"`` uses stream copy for speed; ``mode="reencode"`` re-encodes
    for frame-exact cuts and normalized output.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    start = max(float(start), 0.0)
    duration = max(float(duration), 0.2)

    cmd = [which_ffmpeg(), "-y", "-ss", f"{start:.3f}", "-i", str(movie),
           "-t", f"{duration:.3f}"]
    if mode == "copy":
        cmd += [
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(out),
        ]
    else:
        fps = int(cfg_video.get("fps", 30))
        vf = (
            "scale=1920:1080:force_original_aspect_ratio=increase,"
            "crop=1920:1080,setsar=1,"
            f"fps={fps},setpts=PTS-STARTPTS"
        )
        cmd += [
            "-vf", vf,
            "-r", str(fps),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-an",
            "-video_track_timescale", "90000",
            str(out),
        ]
    run(cmd)
    return out


def concat_segments(segments: list[Path], out: Path, workdir: Path) -> Path:
    """Join the segments with the concat demuxer (Step F, no re-encode)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    listfile = Path(workdir) / "concat_segments.txt"
    listfile.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in segments) + "\n",
        encoding="utf-8",
    )
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


def _cover_target(visual: Path, out: Path, target: float, cfg_video: dict) -> Path:
    """Guarantee the visual is >= target seconds (loop, then trim)."""
    total = probe_duration(visual)
    if total >= target:
        run(
            [
                which_ffmpeg(), "-y",
                "-i", str(visual),
                "-t", f"{target:.3f}",
                "-c:v", cfg_video.get("codec", "libx264"),
                "-pix_fmt", "yuv420p",
                "-an",
                str(out),
            ]
        )
        return out
    run(
        [
            which_ffmpeg(), "-y",
            "-stream_loop", "-1",
            "-i", str(visual),
            "-t", f"{target:.3f}",
            "-c:v", cfg_video.get("codec", "libx264"),
            "-pix_fmt", "yuv420p",
            "-an",
            str(out),
        ]
    )
    return out


def build_visual_from_windows(
    movie: Path,
    windows: list[tuple[float, float]],
    workdir: Path,
    cfg_video: dict,
    mode: str = "copy",
) -> Path:
    """Cut `windows` (start, duration) pairs from the movie, concat them.

    Returns the silent visual track sized exactly to the narration target
    (sum of the windows' durations).
    """
    mode = (mode or "copy").strip().lower()
    if mode not in ("copy", "reencode"):
        mode = "copy"

    cutdir = Path(workdir) / "beats"
    cutdir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []
    for i, (start, dur) in enumerate(windows):
        out = cutdir / f"seg_{i:03d}.mp4"
        cut_segment(movie, out, start, dur, cfg_video, mode=mode)
        segments.append(out)

    raw = Path(workdir) / "visual_raw.mp4"
    concat_segments(segments, raw, workdir)

    target = sum(max(d, 0.2) for _, d in windows)
    base = Path(workdir) / "visual.mp4"
    return _cover_target(raw, base, target, cfg_video)
