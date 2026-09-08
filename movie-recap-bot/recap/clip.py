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

import json
from pathlib import Path

from .util import ffmpeg_timeout, probe_duration, run, which_ffmpeg


def cut_segment(
    movie: Path,
    out: Path,
    start: float,
    duration: float,
    cfg_video: dict,
    mode: str = "copy",
    exact: bool = False,
    freeze: float = 0.0,
) -> Path:
    """Cut one segment of the source film.

    ``mode="copy"`` uses stream copy for speed; ``mode="reencode"`` re-encodes
    for frame-exact cuts and normalized output.

    ``exact=True`` (used by the audio-locked timeline) puts ``-ss`` *before*
    the input for a fast seek but re-states ``-t`` on the output so the written
    file is exactly ``duration`` long regardless of keyframe placement.

    ``freeze>0`` renders the timeline's freeze-hold cuts: only the first
    ``duration - freeze`` seconds are read from the film, then the final frame
    is cloned for the remaining ``freeze`` seconds (``tpad``), so the output is
    still exactly ``duration`` long — the picture holds the shot instead of
    running ahead of the narration. Stream copy cannot freeze, so a freeze cut
    is always re-encoded.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    start = max(float(start), 0.0)
    duration = max(float(duration), 0.05 if exact else 0.2)
    freeze = min(max(float(freeze or 0.0), 0.0), max(duration - 0.05, 0.0))

    cmd = [which_ffmpeg(), "-y", "-ss", f"{start:.3f}"]
    if freeze > 0:
        # read only the moving part of the shot from the source
        cmd += ["-t", f"{max(duration - freeze, 0.05):.3f}"]
    cmd += ["-i", str(movie), "-t", f"{duration:.3f}"]
    if mode == "copy" and freeze <= 0:
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
            f"fps={fps}"
        )
        if freeze > 0:
            vf += f",tpad=stop_mode=clone:stop_duration={freeze:.3f}"
        vf += ",setpts=PTS-STARTPTS"
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
    run(cmd, timeout=ffmpeg_timeout(duration + 1.0, minimum=240.0))
    return out


def concat_segments(
    segments: list[Path],
    out: Path,
    workdir: Path,
    reencode: bool = False,
    media_seconds: float | None = None,
) -> Path:
    """Join the segments with the concat demuxer (Step F).

    With ``reencode=True`` the join is re-encoded. The segments are already
    normalized to one size/fps/timebase, but a stream-copy concat still
    inherits each part's container timestamps, and a single bad edit there can
    shift everything after it. Re-encoding the join makes the output duration
    the exact sum of the parts, which is what the audio lock depends on.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    listfile = Path(workdir) / "concat_segments.txt"
    listfile.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in segments) + "\n",
        encoding="utf-8",
    )
    cmd = [
        which_ffmpeg(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(listfile),
    ]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-fps_mode", "cfr", "-video_track_timescale", "90000", "-an"]
    else:
        cmd += ["-c", "copy"]
    cmd += [str(out)]
    if media_seconds:
        run(cmd, timeout=ffmpeg_timeout(media_seconds, minimum=600.0))
    else:
        run(cmd, timeout=ffmpeg_timeout(0.0, minimum=600.0))
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


def _visual_plan_id(movie: Path, cuts: list[tuple[float, float]],
                    mode: str, cfg_video: dict) -> str:
    """Deterministic id of one visual-assembly plan.

    Includes the movie's identity (path + size + mtime) and the full ordered
    cut list, so re-cutting only happens when the source, the beat plan or the
    encode settings actually changed.
    """
    try:
        st = movie.stat()
        movie_id = [str(movie.resolve()), st.st_size, st.st_mtime_ns]
    except OSError:
        movie_id = [str(movie.resolve()), 0, 0]
    return json.dumps(
        {
            "movie": movie_id,
            "cuts": [
                [round(float(cut[0]), 4), round(float(cut[1]), 4),
                 round(float(cut[2]), 4) if len(cut) > 2 else 0.0]
                for cut in cuts
            ],
            "mode": mode,
            "fps": int(cfg_video.get("fps", 30)),
            "codec": cfg_video.get("codec", "libx264"),
        },
        sort_keys=True,
    )


def build_locked_visual(
    movie: Path,
    cuts: list[tuple[float, float]],
    workdir: Path,
    cfg_video: dict,
    audio_span: float,
    mode: str = "reencode",
) -> Path:
    """Build a visual track whose length is LOCKED to the narration.

    This is the frame-accurate replacement for ``build_visual_from_windows``.

    Two hard rules the old path broke:

    1. **Every cut is re-encoded to an exact duration.** ``-c copy`` snaps the
       cut to the nearest keyframe, so a clip asked for 4.20s could come back
       3.6s or 5.1s. Summed over 150 beats that is tens of seconds of drift and
       the audio slides off the picture. We re-encode by default (``mode``
       still accepts ``"copy"`` for a fast preview) and force the exact length
       with ``-t`` on the *output* plus ``fps`` normalisation.

    2. **The final track is padded/trimmed to the narration span**, so
       ``-shortest`` in the mux can never truncate the video. Previously the
       visual was only as long as the sum of the *spoken* clip lengths, while
       the mp3 also contained the silent gaps between sentences — which is why
       a 900s request rendered ~360-650s.
    """
    mode = (mode or "reencode").strip().lower()
    if mode not in ("copy", "reencode"):
        mode = "reencode"

    # cuts are (film_start, duration, freeze?) — legacy 2-tuples still accepted
    norm_cuts: list[tuple[float, float, float]] = []
    for cut in cuts:
        _a, _b = float(cut[0]), float(cut[1])
        _f = float(cut[2]) if len(cut) > 2 else 0.0
        norm_cuts.append((_a, _b, _f))

    cutdir = Path(workdir) / "beats"
    cutdir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = [cutdir / f"seg_{i:04d}.mp4" for i in range(len(cuts))]

    if not segments:
        raise ValueError("no cuts to assemble")

    # Resume guard: a rerun after a crash must not re-cut (and re-bill hours
    # of ffmpeg) work that already finished. The plan id covers the movie file,
    # the exact cut list and the encode settings, so any real change re-cuts.
    plan = _visual_plan_id(movie, cuts, mode, cfg_video)
    stamp = cutdir / ".cuts.json"
    plan_ok = stamp.exists() and stamp.read_text(encoding="utf-8").strip() == plan
    have_segs = all(p.exists() and p.stat().st_size > 0 for p in segments)

    if plan_ok and have_segs:
        print(f"  * [{mode}] reusing {len(segments)} existing film clips "
              f"(unchanged cut plan)", flush=True)
    else:
        # A changed plan leaves stale seg_*.mp4 files that would corrupt the
        # concat (same filename, different moment). Clear them first.
        if stamp.exists() or any(p.exists() for p in segments):
            removed = 0
            for p in cutdir.glob("seg_*.mp4"):
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
            if removed:
                print(f"  * [{mode}] cleared {removed} stale clip segment(s)", flush=True)
        for i, (start, dur, freeze) in enumerate(norm_cuts):
            cut_segment(movie, segments[i], start, dur, cfg_video,
                        mode=mode, exact=True, freeze=freeze)
        stamp.write_text(plan, encoding="utf-8")

    raw = Path(workdir) / "visual_raw.mp4"
    raw_stamp = Path(workdir) / ".raw.json"
    total = sum(max(d, 0.0) for _, d, _f in norm_cuts)
    if raw_stamp.exists() and raw_stamp.read_text(encoding="utf-8").strip() == plan \
            and raw.exists() and raw.stat().st_size > 0:
        print(f"  * reusing concatenated visual {raw.name} "
              f"({probe_duration(raw):.1f}s of film clips)", flush=True)
    else:
        print(f"  * joining {len(segments)} clips into {raw.name} ...", flush=True)
        try:
            # Every clip was just re-encoded to identical size/fps/timebase, so
            # a stream-copy join is exact and takes seconds, not a full second
            # 1080p encode. _lock_to_audio() below re-encodes once and trims or
            # pads to the exact narration span, which is where A/V sync is set.
            concat_segments(segments, raw, workdir, reencode=False,
                            media_seconds=total)
        except RuntimeError:
            # The demuxer rejected a piece (corrupt or foreign codec): fall
            # back to the slow re-encoding join rather than failing the run.
            print("  ! stream-copy join failed - retrying with a full "
                  "re-encode (slow; only needed for non-uniform clips)",
                  flush=True)
            concat_segments(segments, raw, workdir, reencode=True,
                            media_seconds=total)
        raw_stamp.write_text(plan, encoding="utf-8")

    # Lock the final length to the narration, to the millisecond.
    base = Path(workdir) / "visual.mp4"
    base_stamp = Path(workdir) / ".visual.json"
    base_plan = json.dumps(
        {"plan": json.loads(plan), "audio_span": round(float(audio_span), 3)},
        sort_keys=True,
    )
    if base_stamp.exists() and base_stamp.read_text(encoding="utf-8").strip() == base_plan \
            and base.exists() and base.stat().st_size > 0:
        print(f"  * reusing audio-locked visual {base.name} "
              f"({probe_duration(base):.1f}s == narration "
              f"{float(audio_span):.1f}s)", flush=True)
        return base
    print(f"  * final visual pass: locking {total:.1f}s of clips to "
          f"{float(audio_span):.1f}s of narration "
          f"(single encode; slowest local step on older PCs)", flush=True)
    base = _lock_to_audio(raw, base, audio_span, cfg_video)
    base_stamp.write_text(base_plan, encoding="utf-8")
    return base


def _lock_to_audio(src: Path, out: Path, span: float, cfg_video: dict) -> Path:
    """Trim or freeze-extend ``src`` so it lasts exactly ``span`` seconds."""
    have = probe_duration(src)
    fps = int(cfg_video.get("fps", 30))
    codec = cfg_video.get("codec", "libx264")
    cmd = [which_ffmpeg(), "-y"]
    if have + 0.05 < span:
        # Slightly short (rounding across many cuts): hold the last frame.
        # tpad is frame-exact and avoids a visible loop-jump back to scene 1.
        cmd += ["-i", str(src), "-vf",
                f"tpad=stop_mode=clone:stop_duration={max(span - have, 0):.3f},fps={fps}"]
    else:
        cmd += ["-i", str(src), "-vf", f"fps={fps}"]
    cmd += [
        "-t", f"{span:.3f}",
        "-c:v", codec, "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-video_track_timescale", "90000", "-an", str(out),
    ]
    run(cmd, timeout=ffmpeg_timeout(span, minimum=600.0))
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
