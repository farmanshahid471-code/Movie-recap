"""Cut a recap-style montage out of the actual movie.

Recap channels don't play the film straight through: they cut a fast sequence of
short beats from across the whole runtime and narrate over them. This module
produces that sequence.

    detect_scenes()  -> where the cuts are
    build_montage()  -> the actual clip sequence, laid end to end

Two ways to find the cuts:

  1. PySceneDetect, when installed - real content-aware detection
     (`pip install scenedetect[opencv]`).
  2. Even beats - no extra dependency. The film is sliced into fixed-length
     segments, which already looks far more like a recap than one continuous
     shot, and is what runs out of the box.

Either way the narration stays the master clock: the montage is cut to be at
least as long as the narration and `-shortest` trims it exactly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .util import probe_duration, run, which_ffmpeg

DEFAULT_BEAT = 6.0      # seconds per beat when there is no scene detector
MIN_BEAT = 1.5
MAX_BEAT = 20.0


def _clamp_beat(seconds: float) -> float:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        s = DEFAULT_BEAT
    return min(max(s, MIN_BEAT), MAX_BEAT)


def even_beats(duration: float, beat: float) -> list[tuple[float, float]]:
    """Slice a runtime into fixed-length beats (dependency-free fallback)."""
    beat = _clamp_beat(beat)
    if duration <= 0:
        return []
    out: list[tuple[float, float]] = []
    t = 0.0
    while t < duration - MIN_BEAT:
        out.append((t, min(t + beat, duration)))
        t += beat
    return out


def _scenedetect(video: Path, threshold: float) -> list[tuple[float, float]] | None:
    """Content-aware scene list via PySceneDetect, or None if unavailable."""
    try:
        from scenedetect import SceneManager, open_video  # type: ignore
        from scenedetect.detectors import ContentDetector  # type: ignore
    except Exception:
        return None
    try:
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold))
        sm.detect_scenes(open_video(str(video)), show_progress=False)
        pairs = [(s.get_seconds(), e.get_seconds()) for s, e in sm.get_scene_list()]
        return pairs or None
    except Exception as exc:
        print(f"  ! PySceneDetect failed ({type(exc).__name__}: {exc}); using even beats.")
        return None


def detect_scenes(
    video: Path,
    cfg_video: dict,
    workdir: Path,
    duration: float | None = None,
) -> tuple[list[tuple[float, float]], str]:
    """Return the scene list for a movie plus which method produced it.

    Cached in ``<workdir>/scenes.json`` so the per-language runs reuse it.
    """
    cache = Path(workdir) / "scenes.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            pairs = [(float(a), float(b)) for a, b in data.get("scenes", [])]
            if pairs:
                return pairs, data.get("method", "cache")
        except Exception:
            pass

    if duration is None:
        duration = probe_duration(video)

    method = "even-beats"
    scenes = _scenedetect(
        Path(video), float(cfg_video.get("scene_threshold", 27.0))
    )
    if scenes:
        method = "scenedetect"
        # keep beats inside the configured length band so no shot drags
        lo = _clamp_beat(cfg_video.get("scene_min_len", 2.0))
        hi = _clamp_beat(cfg_video.get("scene_max_len", MAX_BEAT))
        scenes = [(a, b) for a, b in scenes if (b - a) >= lo]
        scenes = [(a, min(b, a + hi)) for a, b in scenes]
    if not scenes:
        scenes = even_beats(duration, cfg_video.get("scene_len", DEFAULT_BEAT))

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"method": method, "scenes": scenes}, ensure_ascii=False),
        encoding="utf-8",
    )
    return scenes, method


def _cut(
    video: Path,
    start: float,
    end: float,
    out: Path,
    cfg_video: dict,
) -> Path:
    """Cut one beat and normalize it to the project's size/fps."""
    from .video import SCALE_FILL

    fps = int(cfg_video.get("fps", 30))
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            which_ffmpeg(), "-y",
            "-ss", f"{max(start, 0.0):.3f}",
            "-i", str(video),
            "-t", f"{max(end - start, 0.2):.3f}",
            "-vf", f"{SCALE_FILL},fps={fps},setpts=PTS-STARTPTS",
            "-r", str(fps),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-an",
            "-video_track_timescale", "90000",
            str(out),
        ]
    )
    return out


def build_montage(
    video: Path,
    target_duration: float,
    cfg_video: dict,
    workdir: Path,
    duration: float | None = None,
) -> Path:
    """Cut beats from `video` and lay them end to end for `target_duration`.

    Beats are taken in story order and cycle from the top if the narration runs
    longer than the film's worth of beats, so the montage never ends early.
    """
    workdir = Path(workdir)
    scenes, method = detect_scenes(video, cfg_video, workdir, duration)
    if not scenes:
        raise ValueError(f"Could not find any scenes in {video}")

    # A recap montage must cover the WHOLE film, not just its opening. Take only
    # as many beats as the narration needs, spread evenly across the timeline so
    # the footage walks from beginning through middle to end.
    avg = sum(max(b - a, 0.2) for a, b in scenes) / len(scenes)
    n_needed = max(1, math.ceil(target_duration / avg))
    picked: list[tuple[float, float]] = []
    if n_needed >= len(scenes):
        picked = list(scenes)
    else:
        prev = -1
        for j in range(n_needed):
            idx = round(j * (len(scenes) - 1) / (n_needed - 1)) if n_needed > 1 else 0
            if idx != prev:
                picked.append(scenes[idx])
                prev = idx

    cutdir = workdir / "cuts"
    cutdir.mkdir(parents=True, exist_ok=True)
    print(f"  * montage: {len(picked)} beats spread across the whole movie ({method})")

    pieces: list[Path] = []
    covered = 0.0
    i = 0
    while covered < target_duration and i < len(picked) * 4:
        start, end = picked[i % len(picked)]
        # include the start time in the name so cuts from an older selection
        # strategy are never mistaken for the right beat
        out = cutdir / f"beat_{i:04d}_{int(start * 1000):07d}.mp4"
        if not out.exists():
            _cut(video, start, end, out, cfg_video)
        pieces.append(out)
        covered += max(end - start, 0.2)
        i += 1

    listfile = workdir / "montage.txt"
    listfile.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in pieces) + "\n", encoding="utf-8"
    )
    joined = workdir / "montage_raw.mp4"
    run(
        [
            which_ffmpeg(), "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c", "copy",
            str(joined),
        ]
    )

    base = workdir / "base.mp4"
    run(
        [
            which_ffmpeg(), "-y",
            "-i", str(joined),
            "-t", f"{target_duration:.3f}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(int(cfg_video.get("fps", 30))),
            "-an",
            str(base),
        ]
    )
    print(f"  * montage: {len(pieces)} cuts, {covered:.1f}s of footage for "
          f"{target_duration:.1f}s of narration")
    return base
