"""Assemble the final recap video with ffmpeg.

Inputs (per language):
    * source movie clip(s) you own   -> background montage
    * narration mp3 + timing cues    -> audio track + subtitle timing
    * an .ass subtitle file          -> burned into the frames

Produced output (per language):
    output/<name>_<lang>.mp4

The montage is built to be *at least* as long as the narration and muxed with
an explicit duration (``burn_and_mux_locked``) so the audio stays the master
clock and the render can never be truncated. If the clips are shorter than the
narration, they are looped seamlessly.

If no real clips are provided, `make_storyboard()` fabricates colored scene
clips so the whole pipeline is runnable end-to-end (useful for testing/demo).
"""
from __future__ import annotations

from pathlib import Path

from .util import ffmpeg_timeout, probe_duration, run, which_ffmpeg

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


def _filter_arg(name: str) -> str:
    """Escape a filename for use inside an ffmpeg filtergraph argument.

    Inside a filter arg, ``\\``, ``'``, ``:``, ``,``, ``;``, ``[`` and ``]`` are
    special. Each needs a double backslash: one level for the filtergraph
    parser, one for the filter-argument parser.
    """
    out = []
    for ch in name:
        if ch in "\\':,;[]":
            out.append("\\\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def burn_and_mux_locked(
    base: Path,
    narration_mp3: Path,
    ass_path: Path,
    out_mp4: Path,
    cfg_video: dict,
    duration: float | None = None,
) -> Path:
    """Burn subtitles + mux narration with an EXPLICIT duration.

    ``burn_and_mux`` uses ``-shortest``, which silently truncates the render to
    whichever stream ends first. When the visual track was built only from the
    spoken clip lengths (ignoring the pauses between sentences) that meant a
    900-second narration came out as a ~360-second video with the back half of
    the story missing. Here the video is already audio-locked, so we state the
    duration outright and never let ffmpeg pick.
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ass_path = Path(ass_path)
    if duration is None:
        duration = probe_duration(narration_mp3)
    cmd = [
        which_ffmpeg(), "-y",
        "-i", str(base),
        "-i", str(narration_mp3),
        "-vf", f"ass={_filter_arg(ass_path.name)}",
        "-map", "0:v", "-map", "1:a",
        "-c:v", cfg_video.get("codec", "libx264"),
        "-preset", cfg_video.get("preset", "medium"),
        "-crf", str(cfg_video.get("crf", 20)),
        "-c:a", cfg_video.get("audio_codec", "aac"),
        "-b:a", cfg_video.get("audio_bitrate", "192k"),
        "-pix_fmt", "yuv420p",
        "-t", f"{float(duration):.3f}",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    print(f"  * burning subtitles + muxing narration "
          f"({float(duration):.1f}s; final encode pass)", flush=True)
    run(cmd, cwd=ass_path.parent,
        timeout=ffmpeg_timeout(float(duration), minimum=600.0))
    return out_mp4


def burn_and_mux(
    base: Path,
    narration_mp3: Path,
    ass_path: Path,
    out_mp4: Path,
    cfg_video: dict,
) -> Path:
    """Burn subtitles (ASS) and add narration as the audio track.

    The subtitle path is passed as a bare filename with ffmpeg's working
    directory set to the file's folder. An absolute Windows path
    (``D:\\recap\\_work\\en.ass``) would be split at the colon by ffmpeg's
    filter parser and fail with "Unable to parse option value ... as image
    size" / "Error applying option 'original_size' to filter 'ass'".
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ass_path = Path(ass_path)
    run(
        [
            which_ffmpeg(), "-y",
            "-i", str(base),
            "-i", str(narration_mp3),
            "-vf", f"ass={_filter_arg(ass_path.name)}",
            "-map", "0:v", "-map", "1:a",
            "-c:v", cfg_video.get("codec", "libx264"),
            "-c:a", cfg_video.get("audio_codec", "aac"),
            "-shortest",
            "-movflags", "+faststart",
            str(out_mp4),
        ],
        cwd=ass_path.parent,
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


# --------------------------------------------------------------------------
# Dynamic Audio & Video Retiming (atempo, freeze-framing, silence padding)
# --------------------------------------------------------------------------
def retime_audio(
    audio_in: Path,
    out_audio: Path,
    target_duration: float,
    max_atempo: float = 1.15,
) -> float:
    """Retime a TTS audio clip to match a target video beat duration.

    Fallback logic:
    - Audio Stretching (atempo): If audio is slightly longer than the video beat
      (target_duration < audio_dur <= target_duration * 1.15), speeds up audio by
      up to 1.15x using FFmpeg's atempo filter.
    - Video Freeze-Framing overflow: If audio heavily overruns (audio_dur > target_duration * 1.15),
      speeds up audio by max_atempo (1.15x); caller freezes the video frame for the rest.
    - Silence Padding: If video beat is longer than audio (target_duration > audio_dur),
      inserts silence at the end using apad so next narration line snaps exactly
      to the start of the next visual scene.
    """
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    a_dur = probe_duration(audio_in)
    t_dur = float(target_duration)

    if a_dur <= 0.0:
        # Mock/invalid audio data (e.g. unit tests): preserve file and return target
        try:
            out_audio.write_bytes(audio_in.read_bytes() if audio_in.exists() else b"")
        except Exception:
            pass
        return t_dur

    if t_dur <= 0.05 or abs(a_dur - t_dur) < 0.01:
        run([which_ffmpeg(), "-y", "-i", str(audio_in), "-c", "copy", str(out_audio)], check=False)
        if not out_audio.exists():
            out_audio.write_bytes(audio_in.read_bytes())
        return a_dur

    if t_dur < a_dur <= t_dur * max_atempo:
        speed = min(max(a_dur / t_dur, 1.0), max_atempo)
        cmd = [
            which_ffmpeg(), "-y",
            "-i", str(audio_in),
            "-filter:a", f"atempo={speed:.4f}",
            "-t", f"{t_dur:.3f}",
            str(out_audio),
        ]
        run(cmd, check=False)
        return probe_duration(out_audio) if out_audio.exists() else t_dur

    if a_dur > t_dur * max_atempo:
        cmd = [
            which_ffmpeg(), "-y",
            "-i", str(audio_in),
            "-filter:a", f"atempo={max_atempo:.4f}",
            str(out_audio),
        ]
        run(cmd, check=False)
        return probe_duration(out_audio) if out_audio.exists() else (a_dur / max_atempo)

    pad_dur = max(t_dur - a_dur, 0.0)
    cmd = [
        which_ffmpeg(), "-y",
        "-i", str(audio_in),
        "-af", f"apad=pad_dur={pad_dur:.3f}",
        "-t", f"{t_dur:.3f}",
        str(out_audio),
    ]
    run(cmd, check=False)
    return probe_duration(out_audio) if out_audio.exists() else t_dur


def freeze_frame_video(
    video_in: Path,
    out_video: Path,
    freeze_duration: float,
    cfg_video: dict | None = None,
) -> Path:
    """Freeze the final frame of video_in for freeze_duration seconds using FFmpeg tpad."""
    out_video.parent.mkdir(parents=True, exist_ok=True)
    cfg = cfg_video or {}
    fps = int(cfg.get("fps", 30))
    if freeze_duration <= 0.01:
        run([which_ffmpeg(), "-y", "-i", str(video_in), "-c", "copy", str(out_video)], check=False)
        if not out_video.exists():
            out_video.write_bytes(video_in.read_bytes())
        return out_video

    vf = f"tpad=stop_mode=clone:stop_duration={freeze_duration:.3f}"
    cmd = [
        which_ffmpeg(), "-y",
        "-i", str(video_in),
        "-vf", vf,
        "-r", str(fps),
        "-c:v", cfg.get("codec", "libx264"),
        "-pix_fmt", "yuv420p",
        "-an",
        str(out_video),
    ]
    run(cmd, check=False)
    return out_video


def retime_beat_clip(
    video_in: Path,
    audio_in: Path,
    target_dur: float,
    out_video: Path,
    out_audio: Path,
    max_atempo: float = 1.15,
    cfg_video: dict | None = None,
) -> tuple[Path, Path, float]:
    """Retime a single (video, audio) beat pair with stretching, freezing, or padding."""
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_audio.parent.mkdir(parents=True, exist_ok=True)

    a_dur = probe_duration(audio_in)
    v_dur = probe_duration(video_in)
    if target_dur <= 0:
        target_dur = v_dur

    if v_dur < a_dur <= v_dur * max_atempo:
        final_dur = retime_audio(audio_in, out_audio, v_dur, max_atempo=max_atempo)
        freeze_frame_video(video_in, out_video, 0.0, cfg_video)
        return out_video, out_audio, final_dur

    if a_dur > v_dur * max_atempo:
        final_dur = retime_audio(audio_in, out_audio, v_dur, max_atempo=max_atempo)
        freeze_dur = max(final_dur - v_dur, 0.0)
        freeze_frame_video(video_in, out_video, freeze_dur, cfg_video)
        return out_video, out_audio, final_dur

    final_dur = retime_audio(audio_in, out_audio, v_dur, max_atempo=max_atempo)
    freeze_frame_video(video_in, out_video, 0.0, cfg_video)
    return out_video, out_audio, final_dur


def assemble_retimed_narration_track(
    beat_audios: list[Path],
    visual_durations: list[float],
    out_mp3: Path,
    workdir: Path,
    max_atempo: float = 1.15,
) -> tuple[Path, list, list[float]]:
    """Retime each beat audio file to snap directly to the visual scene boundary.

    Prevents compound drift across the entire montage. Returns (out_mp3, cues, final_durations).
    """
    from .tts import TimedCue

    workdir = Path(workdir)
    retimed_dir = workdir / "retimed_audio"
    retimed_dir.mkdir(parents=True, exist_ok=True)

    retimed_files: list[Path] = []
    cues: list[TimedCue] = []
    final_durs: list[float] = []
    cum = 0.0

    n = min(len(beat_audios), len(visual_durations))
    for i in range(n):
        src_audio = beat_audios[i]
        target = float(visual_durations[i])
        out_beat = retimed_dir / f"beat_{i+1:03d}_retimed.mp3"
        dur = retime_audio(src_audio, out_beat, target, max_atempo=max_atempo)
        retimed_files.append(out_beat)
        final_durs.append(dur)
        cues.append(TimedCue(f"Beat {i+1}", cum, cum + dur))
        cum += dur

    listfile = workdir / "concat_retimed_audio.txt"
    listfile.write_text("\n".join(f"file '{p.as_posix()}'" for p in retimed_files) + "\n", encoding="utf-8")
    run([which_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out_mp3)], check=False)
    if not out_mp3.exists() or out_mp3.stat().st_size == 0:
        with open(out_mp3, "wb") as o:
            for rf in retimed_files:
                if rf.exists():
                    o.write(rf.read_bytes())

    return out_mp3, cues, final_durs
