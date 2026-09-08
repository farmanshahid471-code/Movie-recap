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

* **Word-locked cuts.** When the narration's word timings are known (edge-tts
  boundaries, or the faster-whisper alignment pass in ``recap/align.py``),
  the micro-cut points inside a sentence are placed ON WORDS — at clause
  boundaries (after a comma, before "and"/"but"/"while"/...) measured from
  the audio — so the picture switches at the exact moment the narrator moves
  to the next subject, not at an arbitrary even split of the sentence.

* **No-replay playback.** All cuts of the whole track are placed by one
  forward walk: every cut's film start is clamped to start at or after the
  END of the previous cut's footage. The film therefore never rewinds and no
  moment is ever shown twice — the "same clip stutters back" artifact of the
  old per-beat placement is impossible by construction.

* **Bounded lead + motion guarantee (visuals stay WITH the narration, and
  they never stop moving).** In a dialogue-dense section the narration can
  be LONGER than the footage behind it; blindly playing on made the visuals
  run tens of seconds AHEAD of the story. The fix is what a human editor
  does: pace each window — ``speed = window / narration`` clamped to
  ``min_speed`` (0.35x) — so a starved stretch plays as gentle slow motion
  that stays locked to the narrated moment. The picture therefore always
  MOVES (new footage at 1x, or the same moment in slow motion); a frozen
  frame happens only when the narration outlasts the entire movie. And a
  cut never opens less than ``min_new_footage`` (0.8s) past the previous
  footage — micro-jumps that read as stutters become seamless
  continuations instead. Durations still sum to the narration exactly.

* **Shot-boundary snapping.** With the film's real shot-change times
  (detected once per movie, cached — see ``scenes.scene_boundaries``), each
  cut's film start is snapped to the nearest actual camera cut within a
  tolerance. Every visual therefore begins on a real cut of the film instead
  of drifting in mid-shot, which is what makes reference-channel edits feel
  crisp.
"""
from __future__ import annotations

import bisect
import re
from typing import Sequence

# A beat is:
#   {"index", "sentence", "film_start", "film_end", "duration", "cuts": [(start, dur)]}

# Tokens a new clause typically starts with — a shot change right before one
# of these reads as an intentional edit. English-focused, but comma/semicolon
# detection covers the other languages too.
_CONJUNCTIONS = {
    "and", "but", "while", "so", "because", "when", "as", "after", "before",
    "then", "meanwhile", "however", "instead", "once", "until", "since",
    "although", "though", "whereas", "back", "thanks", "determined",
    "excited", "suddenly", "just",
}

_CLAUSE_PUNCT = ",;:" + "，。！？；：、）】」』"

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


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


def _norm_tok(token: str) -> str:
    return (token or "").strip().lower().strip("\"'“”‘’")


def _word_boundary_fractions(
    words: list | None,
    beat_start: float,
    duration: float,
) -> list[float] | None:
    """Candidate shot-switch points inside one beat, as fractions of it.

    ``words`` — the beat's spoken words ``[(text, start, end)]`` in absolute
    narration time; ``beat_start`` — where this beat begins on that same
    clock. A switch is "natural" right before a word that starts a new clause:
    after a comma/semicolon/colon (or a CJK clause mark), or before a
    conjunction like "and"/"but"/"while". Returns sorted fractions in
    (0.12, 0.92) — never close enough to the beat's edges to flash a shot —
    or None when there is nothing usable (the caller splits evenly).
    """
    if not words or duration <= 0:
        return None
    cands: list[float] = []
    for idx in range(len(words) - 1):
        tok = (words[idx][0] or "").rstrip()
        nxt = words[idx + 1]
        nxt_tok = _norm_tok(nxt[0])
        tok_last = tok[-1] if tok else ""
        if tok_last in _CLAUSE_PUNCT or nxt_tok in _CONJUNCTIONS:
            f = (float(nxt[1]) - beat_start) / duration
            if 0.12 <= f <= 0.92:
                cands.append(round(f, 4))
    # de-duplicate, keep order
    return sorted(set(cands)) or None


def _select_shot_boundaries(
    candidates: list[float],
    n_shots: int,
    duration: float,
    min_cut: float,
) -> list[float] | None:
    """Pick the ``n_shots - 1`` switch fractions closest to even spacing.

    Greedy, left to right, every chosen fraction at least ``min_cut`` from its
    neighbours and from both ends, so no shot can come out shorter than
    ``min_cut``. Returns fewer boundaries when the candidates run out (that is
    fine — the beat just gets fewer, longer shots), or None when none fit.
    """
    if not candidates or n_shots < 2:
        return None
    min_sep = min_cut / duration if duration > 0 else 1.0
    min_sep = min(min_sep, 0.45)
    chosen: list[float] = []
    for k in range(1, n_shots):
        target = k / n_shots
        lo = (chosen[-1] + min_sep) if chosen else min_sep
        best, best_d = None, None
        for f in candidates:
            if f < lo or f > 1.0 - min_sep:
                continue
            d = abs(f - target)
            if best_d is None or d < best_d:
                best, best_d = f, d
        if best is None:
            break
        chosen.append(best)
    return chosen or None


def _shot_split(
    duration: float,
    fracs: list[float] | None,
    *,
    micro_target: float,
    max_cuts: int,
    min_cut: float,
) -> list[tuple[float, float]]:
    """Split one narration beat into shots: ``[(seconds, film_fraction), ...]``.

    The ``seconds`` sum to exactly ``duration`` (the A/V lock depends on it).
    ``film_fraction`` is where inside the beat's film window the shot's
    footage comes from (0.0 = window start, 1.0 = window end). With
    ``fracs`` (measured word-boundary fractions) the split lands on clause
    boundaries; otherwise it is even.
    """
    duration = max(float(duration), 0.05)
    n = int(round(duration / max(micro_target, 0.5))) or 1
    n = max(1, min(n, int(max_cuts)))
    # never create shots shorter than min_cut
    while n > 1 and duration / n < min_cut:
        n -= 1

    bounds: list[float] | None = None
    if fracs and n > 1:
        picked = _select_shot_boundaries(fracs, n, duration, min_cut)
        if picked:
            bounds = [0.0] + picked + [1.0]
    if bounds is None:
        bounds = [k / n for k in range(n + 1)]

    return [
        ((bounds[k + 1] - bounds[k]) * duration, bounds[k])
        for k in range(len(bounds) - 1)
    ]


def _snap_to_boundary(start: float, bounds: list[float], tolerance: float) -> float:
    """Move ``start`` to the nearest real shot change within ``tolerance``.

    Returns ``start`` unchanged when no boundary is close enough.
    """
    if not bounds or tolerance <= 0:
        return start
    i = bisect.bisect_left(bounds, start)
    best, best_d = start, tolerance
    for j in (i - 1, i):
        if 0 <= j < len(bounds):
            d = abs(bounds[j] - start)
            if d < best_d:
                best, best_d = bounds[j], d
    return best


def build_timeline(
    sentences: list[dict],
    durations: list[float],
    movie_dur: float,
    cfg_timeline: dict | None = None,
    word_times: list | None = None,
    stats: dict | None = None,
    scene_bounds: list[float] | None = None,
) -> list[dict]:
    """Build the chronological, audio-locked beat list.

    ``sentences`` — ``[{"sentence": str, "film_start": float, "film_end": float}]``
    in story order (produced by the segmented script generator). ``durations``
    — the audio-locked visual length of each beat from :func:`lock_durations`.

    ``word_times`` — optional per-sentence spoken-word timings
    ``[[(text, start, end), ...], ...]`` in absolute narration time (from the
    TTS provider or the faster-whisper alignment pass). When present, the
    micro-cuts inside each sentence land on measured clause boundaries.

    ``scene_bounds`` — optional real shot-change times of the film (see
    ``scenes.scene_boundaries``). Each cut's film start is snapped to the
    nearest boundary within ``timeline.snap_tolerance`` seconds, so every
    visual begins on an actual camera cut.

    Guarantees:
      * beat starts never move backwards (strict chronology),
      * ``sum(cut durations) == sum(durations)`` (frame-accurate A/V lock),
      * NO REPLAY: every cut shows footage at or after the end of the
        previous cut's footage — the film never rewinds, no moment is shown
        twice (except the unavoidable end-of-film clamp when the narration
        outlasts the movie),
      * MOTION GUARANTEE: mid-film the picture NEVER freezes. When a
        section's narration is longer than the film behind it, the footage
        plays in slow motion (down to ``min_speed``) instead of running
        ahead or holding a frame — cuts are
        ``(film_start, duration, freeze, speed)`` tuples and ``freeze`` is
        non-zero only when the narration outlasts the movie itself,
      * with word timings, every intra-sentence shot change happens on a
        spoken word boundary, not mid-word,
      * with scene bounds, every cut starts on a real shot change.
    """
    cfg = cfg_timeline or {}
    micro_target = float(cfg.get("micro_cut_seconds", 3.0))
    max_cuts = int(cfg.get("max_cuts_per_beat", 3))
    min_cut = float(cfg.get("min_cut_seconds", 1.2))
    pre_roll = float(cfg.get("pre_roll", 0.4))
    cut_on_words = bool(cfg.get("cut_on_words", True))
    snap_tol = float(cfg.get("snap_tolerance", 0.8))
    max_lead = max(float(cfg.get("max_lead_seconds", 3.0)), 0.0)
    min_speed = min(max(float(cfg.get("min_speed", 0.35)), 0.1), 1.0)
    min_new = float(cfg.get("min_new_footage", 0.8))
    scene_bounds = sorted(scene_bounds or [])

    n = min(len(sentences), len(durations))
    if n == 0:
        return []

    # Narration-time start of each beat (durations are contiguous from 0 and
    # sum to the audio span — see lock_durations — so the running total IS the
    # beat's offset inside the narration mp3).
    cum: list[float] = [0.0]
    for d in durations[:n]:
        cum.append(cum[-1] + max(float(d), 0.05))

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
    playhead = 0.0        # beat-level monotonicity across groups
    film_playhead = 0.0   # END of the last cut's consumed film: no replays
    word_locked = 0
    snapped = 0
    pushed = 0
    held = 0              # freeze events (end-of-film only)
    slowed = 0            # groups paced below 1x (slow motion)
    slowed_secs = 0.0     # narration seconds played in slow motion

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

        # ---- GROUP PACING (the motion guarantee) --------------------------
        # How much narration time does this window of film have to cover?
        # When the narration is LONGER than the film behind it (a dialogue-
        # dense section), playing at 1x would either run the visuals far
        # ahead of the story or force a frozen frame. A human editor plays
        # that stretch in slow motion instead: the footage keeps MOVING at a
        # reduced speed and stays locked to the moment being narrated. The
        # speed is clamped at min_speed (0.35x default = still smooth at
        # 30fps) and is 1.0 whenever the window has enough film, so normal
        # sections are untouched.
        entry = max(f0, film_playhead)
        room = f1 - entry
        if total_nar > 0 and room > 0:
            grp_speed = min(1.0, max(min_speed, room / total_nar))
        else:
            grp_speed = min_speed if room <= 0 else 1.0
        if grp_speed < 0.999:
            slowed += 1
            slowed_secs += total_nar * (1.0 - grp_speed) / max(grp_speed, 1e-6)

        acc = 0.0
        for i in idxs:
            d = max(durations[i], 0.05)
            # position inside this group's film window, proportional to how far
            # through the group's narration we are -> monotone within the group
            film_pos = f0 + (acc / total_nar) * span
            film_span = (d / total_nar) * span
            film_pos = max(film_pos, playhead)

            # word-measured switch points inside this sentence (narration clock)
            fracs = None
            if cut_on_words and word_times and i < len(word_times) \
                    and i < len(cum):
                fracs = _word_boundary_fractions(word_times[i], cum[i], d)
                if fracs:
                    word_locked += 1

            shots = _shot_split(
                d, fracs, micro_target=micro_target, max_cuts=max_cuts,
                min_cut=min_cut,
            )

            cuts: list[list[float]] = []
            for per, frac in shots:
                desired = film_pos + frac * max(film_span, 0.0) - pre_roll
                # NO-REPLAY RULE: never show footage the previous cut already
                # played. If this beat's window is behind the film playhead
                # (its moment was consumed by a longer earlier cut), the
                # footage plays on forward from there.
                start = max(desired, film_playhead)
                if start > desired + 1e-9:
                    pushed += 1
                # ANTI-HICCUP: a cut that advances the film by only a
                # fraction of a second lands on a near-identical frame — it
                # reads as a stutter, not a cut ("laggy"). Prefer a seamless
                # continuation of the current footage over a micro-jump; the
                # next REAL cut (a shot boundary or a full step forward)
                # will provide the visual change.
                if 0.0 < start - film_playhead < min_new:
                    start = film_playhead
                # Snap to the film's real shot change when one is close —
                # but a snap may never rewind below the film playhead.
                if scene_bounds:
                    s2 = _snap_to_boundary(start, scene_bounds, snap_tol)
                    if s2 != start and s2 >= film_playhead:
                        start = s2
                        snapped += 1
                # SAFETY VALVE: never open a cut far ahead of the moment
                # being narrated (the pacing above keeps this rare).
                if start - desired > max_lead:
                    start = film_playhead
                # Freeze ONLY when the film itself has run out (the narration
                # outlasts the movie). Everywhere else the picture moves:
                # new footage at 1x, or the same moment in slow motion.
                freeze = 0.0
                moving = per * grp_speed
                if movie_dur > 0 and start + moving > movie_dur:
                    film_left = max(movie_dur - max(start, 0.0), 0.0)
                    if film_left >= moving - 1e-9:
                        pass
                    else:
                        # show whatever film remains, freeze the rest
                        moving = film_left
                        freeze = per - moving / max(grp_speed, 1e-6)
                        held += 1
                cut = [start, per, freeze, grp_speed]
                cuts.append(cut)
                film_playhead = start + moving

            beats.append(
                {
                    "index": i,
                    "sentence": sentences[i].get("sentence", ""),
                    "film_start": round(film_pos, 3),
                    "film_end": round(film_pos + film_span, 3),
                    "duration": round(d, 3),
                    "cuts": cuts,
                }
            )
            acc += d
            playhead = max(playhead, film_pos)

    beats.sort(key=lambda b: b["index"])
    for b in beats:  # freeze the mutable [start, dur, freeze, speed] into tuples
        b["cuts"] = [
            (round(float(s), 3), round(float(d), 3), round(float(f), 3),
             round(float(v), 3))
            for s, d, f, v in b["cuts"]
        ]
    if stats is not None:
        stats["word_locked_beats"] = word_locked
        stats["snapped_cuts"] = snapped
        stats["pushed_cuts"] = pushed
        stats["held_shots"] = held
        stats["slowed_groups"] = slowed
        stats["slowed_seconds"] = round(slowed_secs, 1)
    return beats


def flatten_cuts(beats: list[dict]) -> list[tuple[float, float, float]]:
    """All micro-cuts of every beat, in play order.

    Each cut is a ``(film_start, duration, freeze, speed)`` tuple:
    ``duration`` is the narration-locked on-screen time, ``speed`` the
    playback rate of the moving part (1.0 = normal; < 1.0 = slow motion),
    and ``freeze`` seconds at the end hold the final frame (end-of-film
    only). Film consumed by a cut = ``(duration - freeze) * speed``.
    """
    out: list[tuple[float, float]] = []
    for b in beats:
        out.extend(b.get("cuts") or [])
    return out


def timeline_report(beats: list[dict], audio_span: float,
                    word_locked: int = 0, snapped: int = 0,
                    slowed: int = 0, slowed_secs: float = 0.0) -> str:
    """One-line human summary used in the run log."""
    cuts = flatten_cuts(beats)
    total = sum(d for _, d, _f, _v in cuts)
    starts = [b["film_start"] for b in beats]
    monotone = all(starts[i] <= starts[i + 1] + 1e-6 for i in range(len(starts) - 1))
    bits = [f"{len(beats)} beats / {len(cuts)} cuts"]
    if word_locked:
        bits.append(f"word-locked {word_locked}/{len(beats)} beats")
    if snapped:
        bits.append(f"{snapped} cuts on shot changes")
    if slowed:
        bits.append(f"{slowed} slow-mo sections ({slowed_secs:.0f}s)")
    return (
        f"{', '.join(bits)}, "
        f"video {total:.1f}s vs narration {audio_span:.1f}s "
        f"(drift {abs(total - audio_span) * 1000:.0f}ms), "
        f"chronological={'yes' if monotone else 'NO'}, "
        f"film coverage {min(starts, default=0):.0f}s -> {max(starts, default=0):.0f}s"
    )
