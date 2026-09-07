"""Strict chronological timeline builder (replaces semantic vector matching).

Why this module exists
----------------------
The previous Step D embedded every narration sentence and every transcript cue
and picked the best cosine match per line. That destroys narrative order: the
words "he runs" appear in act 1, act 2 and act 3, so the montage jumped
erratically between the beginning, middle and end of the film. Recap channels
are *strictly chronological* — beat N of the story must show footage that comes
at or after beat N-1.

This module instead builds a **monotonically advancing playhead** through the
film. Each narration sentence carries the film window of the transcript chunk
it was written from (see ``script.generate_segmented_script``), so chronology
is guaranteed *by construction* rather than hoped for.

Two further properties the old code lacked:

* **Audio-locked durations.** Every beat's visual duration is taken from the
  narration cue *boundaries* (including the silent gap that follows the line),
  not from the spoken length alone. ``sum(beat durations) == narration span``
  exactly, so the video can never be truncated by ``-shortest`` and subtitles
  cannot drift.

* **Micro-cuts.** A 6-second narration beat is split into 2-3 short shots taken
  from successive points inside that beat's film window, the way real recaps
  cut ("wakes up" -> "plane burning" -> "grabs gun") instead of holding one
  static clip for the whole sentence.
"""
from __future__ import annotations

import math
from typing import Sequence

# A beat is:
#   {"index", "sentence", "film_start", "film_end", "duration", "cuts": [(start, dur)]}


def lock_durations(cues: Sequence, total_audio: float | None = None) -> list[float]:
    """Return per-beat visual durations locked to the narration timeline.

    ``cues`` are the TTS cues (objects with ``.start`` / ``.end``). The visual
    for beat *i* must stay on screen until beat *i+1* starts, otherwise the
    silent gap between two spoken sentences has no picture and every later cut
    slides earlier — the cumulative drift the old pipeline suffered from.

    Boundaries are therefore::

        [0, cue[1].start, cue[2].start, ..., cue[n-1].start, audio_end]

    and the durations are the differences, which sum to ``audio_end`` exactly.
    """
    n = len(cues)
    if n == 0:
        return []
    end = float(total_audio if total_audio else cues[-1].end)
    bounds = [0.0]
    for i in range(1, n):
        # clamp so a mis-ordered cue can never produce a negative duration
        bounds.append(max(float(cues[i].start), bounds[-1]))
    bounds.append(max(end, bounds[-1] + 0.2))
    return [bounds[i + 1] - bounds[i] for i in range(n)]


def _micro_cuts(
    film_pos: float,
    film_span: float,
    duration: float,
    movie_dur: float,
    *,
    micro_target: float,
    max_cuts: int,
    min_cut: float,
    pre_roll: float,
) -> list[tuple[float, float]]:
    """Split one narration beat into 1..max_cuts short shots.

    The shots walk forward through ``[film_pos, film_pos + film_span]`` so even
    a single sentence shows visual progression instead of one frozen clip. The
    returned durations always sum to exactly ``duration``.
    """
    duration = max(float(duration), 0.05)
    n = int(round(duration / max(micro_target, 0.5))) or 1
    n = max(1, min(n, int(max_cuts)))
    # never create shots shorter than min_cut
    while n > 1 and duration / n < min_cut:
        n -= 1

    per = duration / n
    cuts: list[tuple[float, float]] = []
    for k in range(n):
        # spread the shot start points across the beat's film window
        frac = (k / n) if n > 1 else 0.0
        start = film_pos + frac * max(film_span, 0.0) - pre_roll
        start = max(0.0, start)
        if movie_dur > 0:
            # keep the whole shot inside the film
            start = min(start, max(movie_dur - per, 0.0))
        cuts.append((start, per))
    return cuts


def build_timeline(
    sentences: list[dict],
    durations: list[float],
    movie_dur: float,
    cfg_timeline: dict | None = None,
) -> list[dict]:
    """Build the chronological, audio-locked beat list.

    ``sentences`` — ``[{"sentence": str, "film_start": float, "film_end": float}]``
    in story order (produced by the segmented script generator). ``durations``
    — the audio-locked visual length of each beat from :func:`lock_durations`.

    Guarantees:
      * beat starts never move backwards (strict chronology),
      * ``sum(cut durations) == sum(durations)`` (frame-accurate A/V lock),
      * the playhead walks the whole film from start to end.
    """
    cfg = cfg_timeline or {}
    micro_target = float(cfg.get("micro_cut_seconds", 3.0))
    max_cuts = int(cfg.get("max_cuts_per_beat", 3))
    min_cut = float(cfg.get("min_cut_seconds", 1.2))
    pre_roll = float(cfg.get("pre_roll", 0.4))

    n = min(len(sentences), len(durations))
    if n == 0:
        return []

    # ---- group the sentences by the film window they were written from ----
    groups: list[dict] = []
    for i in range(n):
        s = sentences[i]
        fs = float(s.get("film_start", 0.0) or 0.0)
        fe = float(s.get("film_end", fs) or fs)
        if groups and abs(groups[-1]["film_start"] - fs) < 1e-6 and abs(groups[-1]["film_end"] - fe) < 1e-6:
            groups[-1]["idx"].append(i)
        else:
            groups.append({"film_start": fs, "film_end": fe, "idx": [i]})

    # If the script carried no film windows at all, spread the beats evenly over
    # the whole runtime so the montage still walks the film front to back.
    if len(groups) == 1 and groups[0]["film_end"] <= groups[0]["film_start"]:
        groups[0]["film_start"] = 0.0
        groups[0]["film_end"] = movie_dur

    beats: list[dict] = []
    playhead = 0.0  # enforces global monotonicity across groups

    for g in groups:
        f0 = max(float(g["film_start"]), playhead)
        f1 = float(g["film_end"])
        if movie_dur > 0:
            f0 = min(f0, movie_dur)
            f1 = min(max(f1, f0), movie_dur)
        if f1 <= f0:
            f1 = min(f0 + 1.0, movie_dur or (f0 + 1.0))

        idxs = g["idx"]
        total_nar = sum(max(durations[i], 0.05) for i in idxs) or 1.0
        span = f1 - f0

        acc = 0.0
        for i in idxs:
            d = max(durations[i], 0.05)
            # position inside this group's film window, proportional to how far
            # through the group's narration we are -> monotone within the group
            film_pos = f0 + (acc / total_nar) * span
            film_span = (d / total_nar) * span
            film_pos = max(film_pos, playhead)

            cuts = _micro_cuts(
                film_pos, film_span, d, movie_dur,
                micro_target=micro_target, max_cuts=max_cuts,
                min_cut=min_cut, pre_roll=pre_roll,
            )

            beats.append(
                {
                    "index": i,
                    "sentence": sentences[i].get("sentence", ""),
                    "film_start": round(film_pos, 3),
                    "film_end": round(film_pos + film_span, 3),
                    "duration": round(d, 3),
                    "cuts": [(round(a, 3), round(b, 3)) for a, b in cuts],
                }
            )
            acc += d
            playhead = max(playhead, film_pos)

    beats.sort(key=lambda b: b["index"])
    return beats


def flatten_cuts(beats: list[dict]) -> list[tuple[float, float]]:
    """All micro-cuts of every beat, in play order."""
    out: list[tuple[float, float]] = []
    for b in beats:
        out.extend(b.get("cuts") or [])
    return out


def timeline_report(beats: list[dict], audio_span: float) -> str:
    """One-line human summary used in the run log."""
    cuts = flatten_cuts(beats)
    total = sum(d for _, d in cuts)
    starts = [b["film_start"] for b in beats]
    monotone = all(starts[i] <= starts[i + 1] + 1e-6 for i in range(len(starts) - 1))
    return (
        f"{len(beats)} beats / {len(cuts)} cuts, "
        f"video {total:.1f}s vs narration {audio_span:.1f}s "
        f"(drift {abs(total - audio_span) * 1000:.0f}ms), "
        f"chronological={'yes' if monotone else 'NO'}, "
        f"film coverage {min(starts, default=0):.0f}s -> {max(starts, default=0):.0f}s"
    )
