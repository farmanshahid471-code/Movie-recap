"""Atomic clip-plan synchronisation (ported from keithhb33/AI-Movie-Shorts).

The reference project's sync model, and the one this module implements
----------------------------------------------------------------------
In AI-Movie-Shorts (``src/generator.c``) narration and picture are never
created separately and then *matched*. One LLM call returns an atomic
**clip plan**::

    {"clips": [{"start": 120, "end": 145, "narration": "..."}, ...]}

so every narration text is born already bound to the exact film window it
describes. Then, per clip, in order:

1. TTS is generated for *that clip's* narration only;
2. the real narration duration is measured with ffprobe;
3. ``ffmpeg_make_adjusted_clip`` fits the *whole chosen window* to that
   duration: ``speed = source_duration / narration_duration`` rendered as
   ``setpts=PTS/speed``; and
4. if that would speed the picture up beyond ``MAX_VIDEO_SPEEDUP`` (1.75x),
   the window is first centre-trimmed to ``narration_duration * 1.75``
   seconds, so the viewer still sees the middle -- the most important part
   -- of the moment being narrated.

The picture therefore cannot describe a different moment from the voice:
the window is never truncated to its first N seconds, never allowed to
walk forward into the next scene when the voice runs long (it is slowed
instead), and never shared with another narration item.

How that maps onto this project
-------------------------------
This pipeline already produces the equivalent of a clip plan: the
authored ``segments`` (``{"sentence", "film_start", "film_end", "anchor",
"zone_lo", "zone_hi"}``). What it did *not* do was treat them atomically
at render time -- ``timeline.cluster_narrative_beats`` grouped 2-4
sentences under one continuous shot, and ``build_timeline`` (audio-first)
started every clip at ``max(anchor, film_playhead)`` and consumed exactly
``narration_duration`` seconds of film at 1x. A long sentence therefore
ate the following scene, every later clip was pushed off its own moment
by the accumulated playhead, and only the first ``duration`` seconds of
a described window were ever shown.

:func:`build_sync_beats` replaces that render step (only that step):

* **one narration sentence -> exactly one clip** (no clustering, no
  ``continue_shot``, no micro-cuts borrowed from another beat);
* the clip shows the sentence's *own* window ``[film_start, film_end]``
  (confined to its ``zone_lo``/``zone_hi`` so it can never bleed into the
  neighbouring sentence's footage);
* the whole window is time-fitted to the *measured* narration duration
  with the reference rule above (centre-trim above ``max_speedup``,
  otherwise ``speed = window / narration`` in either direction);
* windows stay chronological and non-overlapping -- but instead of
  pushing a late clip into the *next* scene (the old playhead rule), an
  overlap is resolved by trimming the clip inside its own zone.

Output beats use the exact shape the rest of the pipeline already
consumes (``timeline.flatten_cuts`` / ``clip.build_locked_visual``):
each beat's single cut is ``(film_start, narration_duration, 0.0, speed)``
-- ``clip.cut_segment`` reads ``(duration - freeze) * speed`` seconds of
film and plays them over ``duration`` seconds, so the consumed footage
is precisely the fitted window and the A/V total is unchanged.
"""
from __future__ import annotations

import json
from typing import Sequence

# Reference constants (AI-Movie-Shorts src/generator.c).
MAX_VIDEO_SPEEDUP = 1.75
MIN_CLIP_SECONDS = 0.05


# ---------------------------------------------------------------------------
# The atomic plan itself (reference JSON shape)
# ---------------------------------------------------------------------------
def parse_clip_plan_json(json_text: str) -> list[dict]:
    """Parse ``{"clips": [{start, end, narration}, ...]}`` strictly.

    Invalid items are dropped (never repaired into a different moment),
    exactly like ``parse_clip_plan_json`` in the reference: an item whose
    window or narration is unusable cannot be synchronised, so it must
    not silently acquire footage.
    """
    try:
        root = json.loads(json_text or "")
    except (TypeError, ValueError):
        return []
    clips = root.get("clips") if isinstance(root, dict) else None
    if not isinstance(clips, list):
        return []
    out: list[dict] = []
    for obj in clips:
        if not isinstance(obj, dict):
            continue
        try:
            start, end = float(obj["start"]), float(obj["end"])
        except (KeyError, TypeError, ValueError):
            continue
        narration = obj.get("narration")
        if end <= start or not isinstance(narration, str) or not narration.strip():
            continue
        out.append({"start": start, "end": end,
                    "narration": narration.strip()})
    return validate_clip_plan(out)


def validate_clip_plan(plan: Sequence[dict]) -> list[dict]:
    """Enforce the reference plan invariants: chronological, disjoint.

    Clips must be increasing and non-overlapping (the reference prompt
    demands it; this makes it structural). An overlapping clip is trimmed
    to start where the previous one ended; if nothing is left, it is
    dropped rather than shown under the wrong narration.
    """
    out: list[dict] = []
    prev_end: float | None = None
    for item in sorted((dict(p) for p in plan), key=lambda p: float(p["start"])):
        start, end = float(item["start"]), float(item["end"])
        if prev_end is not None and start < prev_end:
            start = prev_end
        if end - start < MIN_CLIP_SECONDS:
            continue
        item["start"], item["end"] = round(start, 3), round(end, 3)
        out.append(item)
        prev_end = end
    return out


def segments_from_clip_plan(plan: Sequence[dict]) -> list[dict]:
    """Adapt an atomic clip plan to this pipeline's segment dicts.

    One plan item -> one segment, narration unsplit. Splitting a plan
    item's narration into sentences and spreading them evenly over
    sub-windows (what the old beat fallback did) destroys the atomic
    binding, so it is deliberately not done here.
    """
    return [
        {
            "sentence": str(p["narration"]).strip(),
            "film_start": float(p["start"]),
            "film_end": float(p["end"]),
            "anchor": round((float(p["start"]) + float(p["end"])) / 2.0, 3),
            "zone_lo": float(p["start"]),
            "zone_hi": float(p["end"]),
        }
        for p in validate_clip_plan(plan)
        if str(p.get("narration") or "").strip()
    ]


def plan_from_segments(segments: Sequence[dict]) -> list[dict]:
    """View authored segments as a clip plan (for inspection/tests)."""
    plan = []
    for s in segments or []:
        text = str(s.get("sentence") or "").strip()
        if not text:
            continue
        plan.append({
            "start": float(s.get("film_start", 0.0) or 0.0),
            "end": float(s.get("film_end", 0.0) or 0.0),
            "narration": text,
        })
    return validate_clip_plan(plan)


# ---------------------------------------------------------------------------
# The reference duration fit
# ---------------------------------------------------------------------------
def fit_window_to_narration(
    start: float,
    end: float,
    narration_dur: float,
    *,
    max_speedup: float = MAX_VIDEO_SPEEDUP,
    bound_lo: float | None = None,
    bound_hi: float | None = None,
) -> tuple[float, float, float]:
    """Fit one film window to one measured narration duration.

    Returns ``(use_start, use_end, speed)`` where playing
    ``use_end - use_start`` seconds of film at ``setpts=PTS/speed``
    lasts exactly ``narration_dur`` seconds.

    Port of ``ffmpeg_make_adjusted_clip``: ``speed = source/narration``;
    only when ``speed > max_speedup`` is the window centre-trimmed to
    ``narration_dur * max_speedup`` (re-centred, then clamped inside
    ``[start, end]`` and inside the optional zone bounds). A window
    shorter than the narration is slowed (``speed < 1``), never extended
    into neighbouring footage.
    """
    start, end = float(start), float(end)
    narration_dur = max(float(narration_dur), MIN_CLIP_SECONDS)
    if end <= start:
        end = start + narration_dur
    if bound_lo is not None:
        start = max(start, float(bound_lo))
    if bound_hi is not None:
        end = min(end, float(bound_hi))
    if end <= start:  # zone collapsed: fall back to a narration-sized window
        mid = (start + end) / 2.0
        start, end = mid, mid + narration_dur

    source_dur = end - start
    speed = source_dur / narration_dur
    if speed > max_speedup:
        desired = min(narration_dur * max_speedup, source_dur)
        desired = max(desired, MIN_CLIP_SECONDS)
        centre = (start + end) / 2.0
        use_start, use_end = centre - desired / 2.0, centre + desired / 2.0
        if use_start < start:
            use_start, use_end = start, start + desired
        if use_end > end:
            use_start, use_end = end - desired, end
        speed = (use_end - use_start) / narration_dur
        if speed > max_speedup:  # integer-second rounding in the reference
            speed = max_speedup
        return round(use_start, 3), round(use_end, 3), round(speed, 4)
    return round(start, 3), round(end, 3), round(speed, 4)


# ---------------------------------------------------------------------------
# Render-time synchronisation: segments + measured durations -> beats
# ---------------------------------------------------------------------------
def build_sync_beats(
    segments: Sequence[dict],
    durations: Sequence[float],
    movie_dur: float = 0.0,
    cfg_timeline: dict | None = None,
    stats: dict | None = None,
) -> list[dict]:
    """Build audio-locked beats with the reference (atomic) sync rule.

    ``segments`` -- authored segments in narration order; ``durations``
    -- the measured, audio-locked duration of each sentence
    (``timeline.lock_durations`` output). Guarantees:

    * exactly one clip per (non-stub) sentence, in narration order;
    * clip *i*'s footage lies inside sentence *i*'s own zone
      (``zone_lo``..``zone_hi``, else its film window) -- it may be
      centre-trimmed or slowed, but never shows the next sentence's scene;
    * clips are chronological and non-overlapping in film time;
    * ``sum(cut durations) == sum(durations)`` -- the existing A/V lock
      (subtitles, total length) is preserved bit-for-bit;
    * no freeze frames except when the film itself runs out.
    """
    cfg = cfg_timeline or {}
    max_speedup = float(cfg.get("max_video_speedup", MAX_VIDEO_SPEEDUP)
                        or MAX_VIDEO_SPEEDUP)
    max_speedup = max(max_speedup, 1.0)
    pre_roll = max(float(cfg.get("pre_roll", 0.0) or 0.0), 0.0)

    n = min(len(segments), len(durations))
    beats: list[dict] = []
    prev_end = 0.0          # end of film consumed by the previous clip
    fitted = trimmed = slowed = sped = 0

    for i in range(n):
        dur = max(float(durations[i]), 0.0)
        if dur <= 0.001:
            continue  # stub folded into a neighbour by lock_durations
        seg = segments[i] or {}
        fs = float(seg.get("film_start", 0.0) or 0.0)
        fe = float(seg.get("film_end", fs) or fs)
        zone_lo = seg.get("zone_lo")
        zone_hi = seg.get("zone_hi")
        lo = float(zone_lo) if zone_lo is not None else fs
        hi = float(zone_hi) if zone_hi is not None else fe
        if hi <= lo:
            anchor = seg.get("anchor")
            mid = float(anchor) if anchor is not None else fs
            lo, hi = mid - dur / 2.0, mid + dur / 2.0
        # The sentence's own described window, opened slightly early
        # (pre-roll) but never before its zone or the previous clip.
        win_lo = max(fs - pre_roll, lo)
        win_hi = fe
        if win_hi <= win_lo:
            win_lo, win_hi = lo, hi

        use_start, use_end, speed = fit_window_to_narration(
            win_lo, win_hi, dur, max_speedup=max_speedup,
            bound_lo=lo, bound_hi=hi,
        )
        if speed > 1.001:
            sped += 1
        elif speed < 0.999:
            slowed += 1
        if (use_end - use_start) < (win_hi - win_lo) - 1e-6:
            trimmed += 1

        # Non-overlap WITHOUT leaving the sentence's own scene: shift the
        # fitted window forward inside its zone, else trim its head. The
        # old playhead rule instead kept walking into later footage, which
        # is how visuals ended up one scene ahead of the narration.
        if use_start < prev_end:
            shift = prev_end - use_start
            if use_end + shift <= hi + 1e-9:
                use_start += shift
                use_end += shift
            else:
                use_start = prev_end
            if use_end - use_start < MIN_CLIP_SECONDS:
                use_end = min(use_start + MIN_CLIP_SECONDS, hi)
            speed = round((use_end - use_start) / dur, 4) if dur else 1.0
            fitted += 1

        freeze = 0.0
        if movie_dur > 0:
            use_start = min(use_start, movie_dur)
            use_end = min(use_end, movie_dur)
            consumed = max((use_end - use_start), 0.0)
            if consumed <= 0.0:      # film exhausted: hold the last frame
                freeze, speed = dur, 1.0
                use_start = use_end = movie_dur
            else:
                speed = round(consumed / dur, 4)
        prev_end = max(prev_end, use_end)

        beats.append({
            "index": i,
            "sentence": seg.get("sentence", ""),
            "film_start": round(use_start, 3),
            "film_end": round(use_end, 3),
            "duration": round(dur, 3),
            "cuts": [(round(use_start, 3), round(dur, 3),
                      round(freeze, 3), round(speed, 4))],
            "sync_mode": "clip_plan",
        })

    if stats is not None:
        stats.update({
            "word_locked_beats": 0,
            "snapped_cuts": 0,
            "pushed_cuts": fitted,
            "held_shots": sum(1 for b in beats if b["cuts"][0][2] > 0),
            "slowed_groups": slowed,
            "slowed_seconds": 0.0,
            "padded_shots": 0,
            "padded_seconds": 0.0,
            "borrowed_groups": 0,
            "borrowed_seconds": 0.0,
            "clip_plan_clips": len(beats),
            "clip_plan_trimmed": trimmed,
            "clip_plan_sped_up": sped,
        })
    return beats
