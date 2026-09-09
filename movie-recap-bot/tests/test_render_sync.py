"""Render-sync regression tests: the cut/render path must not stretch the
schedule relative to the narration audio.

The reported bug (twice):
  "the narration goes much faster than the visuals, even when the visuals
   are in slow motion -- the narration is way ahead describing the next
   scene"

Root cause (found by auditing the whole sync path -- tts.py cues, align.py
whisper lock, timeline.py schedule, clip.py render, video.py mux):
  The SCHEDULE is correct: lock_durations makes beat i occupy exactly
  [cue_i.start, cue_i+1.start) of the output, and build_timeline places
  beat i's cuts there. The RENDER is not: every clip is written by ffmpeg
  with an output -t + a fixed fps, and frame quantization is one-sided
  (ceil) -- each clip comes out a frame or part-frame LONG, never short.
  Over ~1000 clips that accumulates 10-20 SECONDS of stretch. The audio
  keeps its own clock, the picture schedule slides late, and by the second
  half of the video the narrator is describing the next scene while an
  older one is still on screen. The final trim to the audio span hides the
  TOTAL, so every end-to-end duration check passes while the interior is
  out of sync -- which is exactly why this survived every earlier test.

The fix: drift-compensated cutting in clip.build_locked_visual -- after
each clip is rendered, ffprobe measures its real duration and the
accumulated error is subtracted from the NEXT clip's requested duration,
so the cumulative schedule never leaves a ~1-frame corridor.

Run:  python tests/test_render_sync.py
(no third-party deps, no ffmpeg needed -- the renderer is simulated with
ffmpeg's exact quantization rule)
"""
from __future__ import annotations

import math
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap import clip  # noqa: E402

FPS = 30.0


def _ffmpeg_render(requested: float) -> float:
    """What ffmpeg's output ``-t`` + ``fps`` filter actually produce.

    Frames land on the 1/fps grid and the container reports the last
    frame's full display time, so the written clip is CEIL-quantized:
    never shorter than requested, usually a touch longer. This is the
    one-sided rounding that used to accumulate into seconds of drift.
    """
    return math.ceil(round(requested * FPS, 6)) / FPS


def test_uncompensated_rounding_stretches_schedule() -> None:
    """The bug, in isolation: rendering each planned duration through
    ffmpeg's frame quantization makes the schedule LONGER than the audio --
    by seconds, not milliseconds."""
    random.seed(7)
    durs = [round(random.uniform(1.5, 6.0), 3) for _ in range(950)]
    rendered = [_ffmpeg_render(d) for d in durs]
    # every clip is long, never short (one-sided rounding)
    assert all(r >= d - 1e-9 for r, d in zip(rendered, durs))
    drift = sum(rendered) - sum(durs)
    assert drift > 5.0, f"expected seconds of drift, got {drift:.2f}s"
    print(f"ok: uncompensated frame rounding stretches a "
          f"{sum(durs) / 60:.0f}-min schedule by {drift:.1f}s "
          "(the reported bug)")


def test_drift_compensated_cutting_holds_av_lock() -> None:
    """The fix: with ffprobe feedback after every clip, the cumulative
    rendered schedule stays within ~1 frame of the planned (audio)
    schedule for the WHOLE video -- and a rerun reuses the clips."""
    random.seed(7)
    durs = [round(random.uniform(1.5, 6.0), 3) for _ in range(400)]
    cuts = [(100.0 + 7.0 * i, d, 0.0, 1.0) for i, d in enumerate(durs)]
    audio_span = sum(durs)

    rendered: dict[str, float] = {}
    requested: list[float] = []

    def fake_cut_segment(movie, out, start, duration, cfg_video,
                         mode="copy", exact=False, freeze=0.0, speed=1.0):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"mp4")
        requested.append(float(duration))
        rendered[str(out)] = _ffmpeg_render(float(duration))

    def fake_probe(path):
        return rendered.get(str(path), 0.0)

    def fake_run(cmd, **kw):
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")

    orig = (clip.cut_segment, clip.probe_duration, clip.run,
            clip.which_ffmpeg)
    clip.cut_segment = fake_cut_segment
    clip.probe_duration = fake_probe
    clip.run = fake_run
    clip.which_ffmpeg = lambda: "ffmpeg"
    try:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            movie = tdp / "movie.mp4"
            movie.write_bytes(b"film")
            visual = clip.build_locked_visual(
                movie, cuts, tdp / "work", {"fps": 30}, audio_span,
                mode="reencode",
            )
            assert visual.exists() and visual.stat().st_size > 0
            # second call with identical plan: reuses the clips, cuts nothing
            n_cut = len(requested)
            visual2 = clip.build_locked_visual(
                movie, cuts, tdp / "work", {"fps": 30}, audio_span,
                mode="reencode",
            )
            assert len(requested) == n_cut, "resume must not re-cut"
            assert visual2 == visual
    finally:
        (clip.cut_segment, clip.probe_duration, clip.run,
         clip.which_ffmpeg) = orig

    assert len(requested) == len(durs)
    # THE assertion: the rendered schedule tracks the audio schedule
    planned = actual = worst = 0.0
    for want, d in zip(requested, durs):
        planned += d
        actual += _ffmpeg_render(want)
        worst = max(worst, abs(actual - planned))
    assert worst <= 1.5 / FPS, \
        f"schedule drifted {worst * 1000:.0f} ms from the narration"
    assert abs(actual - audio_span) <= 1.0 / FPS, \
        "total render lands on the narration span within one frame"
    print(f"ok: {len(durs)} drift-compensated cuts -- cumulative A/V "
          f"schedule held to {worst * 1000:.0f} ms (uncompensated: "
          f"{(sum(_ffmpeg_render(d) for d in durs) - audio_span):.1f}s)")


def test_plan_id_versioned_for_recut() -> None:
    """The cut-plan signature carries the cut-engine version, so segments
    rendered by the old (uncompensated) code are re-cut once, never
    silently reused."""
    with tempfile.TemporaryDirectory() as td:
        movie = Path(td) / "movie.mp4"
        movie.write_bytes(b"film")
        plan = clip._visual_plan_id(
            movie, [(0.0, 2.5, 0.0, 1.0)], "reencode", {"fps": 30})
        assert '"cut_engine": 2' in plan
    print("ok: cut-plan signature is versioned (stale segments re-cut)")


if __name__ == "__main__":
    test_uncompensated_rounding_stretches_schedule()
    test_drift_compensated_cutting_holds_av_lock()
    test_plan_id_versioned_for_recut()
    print("\nALL RENDER-SYNC TESTS PASSED")
