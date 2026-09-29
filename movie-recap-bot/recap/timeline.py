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
  ``min_speed`` (0.6x) — so a starved stretch plays as gentle slow motion
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
    durs = [bounds[i + 1] - bounds[i] for i in range(n)]
    return _fold_stub_beats(durs)


# A real spoken sentence (including its share of the pause after it) is never
# shorter than this. A duration below it is a cue-timing artifact (a whisper
# stub, a duplicate cue start) -- rendered naively it becomes a 1-2 frame
# flicker clip that reads as a stutter ("laggy"), like the two back-to-back
# 0.050s clips in the user's run log.
MIN_BEAT_SECONDS = 0.2


def _fold_stub_beats(durs: list[float]) -> list[float]:
    """Fold sub-``MIN_BEAT_SECONDS`` durations into a neighbour beat.

    The stub's speech time is real (it is part of the audio span), so it is
    ADDED to the nearest previous beat (the next one when only stubs precede
    it) and the stub itself becomes 0.0. The sum is preserved exactly -- the
    A/V lock depends on it -- and build_timeline then emits no clip for a
    0.0 beat, so the previous beat's (now longer) cut covers the stub's
    speech instead of flashing two frames of it.
    """
    out = list(durs)
    for i, d in enumerate(out):
        if d >= MIN_BEAT_SECONDS:
            continue
        target = None
        for j in range(i - 1, -1, -1):
            if out[j] > 0.0:
                target = j
                break
        if target is None:
            for j in range(i + 1, len(out)):
                if out[j] > 0.0:
                    target = j
                    break
        if target is None:
            break  # every beat is a stub: leave for the downstream floors
        out[target] += d
        out[i] = 0.0
    return out


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
    max_shot: float = 7.0,
) -> list[tuple[float, float]]:
    """Split one narration beat into shots: ``[(seconds, film_fraction), ...]``.

    The ``seconds`` sum to exactly ``duration`` (the A/V lock depends on it).
    ``film_fraction`` is where inside the beat's film window the shot's
    footage comes from (0.0 = window start, 1.0 = window end). With
    ``fracs`` (measured word-boundary fractions) the split lands on clause
    boundaries; otherwise it is even.

    ``max_shot`` — a shot may never hold for more than this many seconds of
    screen time. Word-locked splits can leave a long final shot when clause
    boundaries are sparse (one 10s+ hold under a fast sentence is what the
    "longest shot is 10.1s -- one visual outlasting several sentences"
    warning was about), so any segment over the cap gets an extra midpoint
    cut. 0 disables the cap.
    """
    import math

    duration = max(float(duration), 0.05)
    n = int(round(duration / max(micro_target, 0.5))) or 1
    n = max(1, min(n, int(max_cuts)))
    # The shot cap can demand MORE cuts than the recap pace does (a 20s
    # sentence with max_shot=7 needs 3+, which the pace already gives — but
    # a 30s one needs 5 against max_cuts=4): honour the cap.
    if max_shot > 0:
        n = max(n, min(max(1, math.ceil(duration / max(max_shot, min_cut, 0.1))),
                       max(int(max_cuts), 1) + 4))
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
    # Word-locked picks can still leave a segment over the cap (the greedy
    # picker may drop candidates that are too close to the neighbours). Cap
    # the longest offender at its midpoint, a few extra cuts at most.
    if max_shot > 0:
        n_hard = max(int(max_cuts), n) + 4
        changed = True
        while changed and (len(bounds) - 1) < n_hard:
            changed = False
            for k in range(len(bounds) - 1):
                if (bounds[k + 1] - bounds[k]) * duration > max_shot + 1e-9:
                    bounds.insert(k + 1, (bounds[k] + bounds[k + 1]) / 2.0)
                    changed = True
                    break

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
    audio_durations: list[float] | None = None,
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
      * MOTION POLICY (``min_speed`` + ``freeze_when_starved``): a section
        whose narration is longer than the film it owns first spends the
        un-narrated film between its window and the next section's (B-roll —
        beats the target-budget selection skipped), then eases into ONE mild
        slow-down that never goes below ``min_speed``, and only then holds
        its final frame (``tpad`` clone in ``clip.cut_segment``) for the
        remainder. A held frame reads as an editing choice; a 0.3x crawl
        reads as a broken render. Cuts are
        ``(film_start, duration, freeze, speed)`` tuples. Set
        ``freeze_when_starved: false`` for the old behaviour (keep slowing /
        walking forward, never freeze mid-film),
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
    min_speed = min(max(float(cfg.get("min_speed", 0.85)), 0.1), 1.0)
    min_new = float(cfg.get("min_new_footage", 0.8))
    max_shot = max(float(cfg.get("max_shot_seconds", 7.0)), 0.0)
    # A section whose narration is longer than the film it owns (even after
    # borrowing the film nobody else narrates) HOLDS its last shot's final
    # frame for the remainder instead of stretching the picture into an ever
    # slower crawl. Set false for the old "keep slowing down / run ahead"
    # behaviour.
    freeze_when_starved = bool(cfg.get("freeze_when_starved", True))
    # When a section is starved, which comes first: the mild slow-down down to
    # min_speed, or the held frame? True = floor speed then freeze (default:
    # the picture still moves a little). False = freeze-only padding: the
    # footage plays at 1x and the shot simply holds its final frame for the
    # overrun, which is the closest thing to "cut, then pad the shot".
    slow_mo_before_freeze = bool(cfg.get("slow_mo_before_freeze", True))
    scene_bounds = sorted(scene_bounds or [])
    # AUDIO-FIRST CUTTING (default): every sentence becomes ONE clip that
    # starts at the sentence's film anchor and runs for EXACTLY the measured
    # audio duration at 1x, playing straight through the film's own shot
    # changes. No slow motion (setpts), no held frames (tpad), no B-roll
    # borrowing: the picture is never stretched to fit the voice. The film
    # playhead still never rewinds (strict monotonic progression); a held
    # frame survives only at the very end of the movie, where no film is left.
    audio_first = bool(cfg.get("audio_first", True))
    broll = bool(cfg.get("broll", False)) and not audio_first

    n = min(len(sentences), len(durations))
    if n == 0:
        return []

    # Narration-time start of each beat (durations are contiguous from 0 and
    # sum to the audio span — see lock_durations — so the running total IS the
    # beat's offset inside the narration mp3).
    cum: list[float] = [0.0]
    for d in durations[:n]:
        # stub beats (folded to 0.0 by lock_durations) contribute their true
        # 0.0 so the narration offsets stay exact
        cum.append(cum[-1] + max(float(d), 0.0))

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

    # Section film windows, in order: the next section's start is the ceiling
    # for how far the current one may walk (see the B-roll note below).
    group_starts = [float(g.get("film_start", 0.0) or 0.0) for g in groups]

    beats: list[dict] = []
    playhead = 0.0        # beat-level monotonicity across groups
    film_playhead = 0.0   # END of the last cut's consumed film: no replays
    word_locked = 0
    snapped = 0
    pushed = 0
    held = 0              # freeze events (end-of-film clamps)
    padded = 0            # cuts that HOLD their last frame because the
                          # section's narration outlasts the film it owns
    padded_secs = 0.0     # narration seconds carried by a held frame
    slowed = 0            # groups paced below 1x (slow motion)
    slowed_secs = 0.0     # narration seconds played in slow motion
    borrowed_groups = 0   # groups that used unused film from a gap
    borrowed_secs = 0.0

    for gi, g in enumerate(groups):
        f0 = max(float(g["film_start"]), playhead)
        f1 = float(g["film_end"])
        if movie_dur > 0:
            f0 = min(f0, movie_dur)
            f1 = min(max(f1, f0), movie_dur)
        if f1 <= f0:
            f1 = min(f0 + 1.0, movie_dur or (f0 + 1.0))

        idxs = g["idx"]
        total_nar = sum(max(float(durations[i]), 0.0) for i in idxs) or 1.0
        span = f1 - f0

        # ---- FILM BUDGET: own window + the B-roll nobody narrates ---------
        # Beats are SELECTED (select_beats_for_target), so the film between
        # one section's window and the next section's window is footage no
        # sentence in the script ever plays over. A section whose narration
        # is longer than its own window may borrow that gap instead of
        # sliding into slow motion: free footage, and the next section still
        # starts on its own window untouched. The last group may run to the
        # end of the film (the tail below the final zone is B-roll too).
        if gi + 1 < len(groups):
            ceiling = group_starts[gi + 1]
            if ceiling < f1:
                ceiling = f1          # windows overlap: no borrowing
        elif movie_dur > 0:
            ceiling = movie_dur
        else:
            ceiling = f1
        if movie_dur > 0:
            ceiling = min(max(ceiling, f1), movie_dur)

        # ---- GROUP PACING (safe motion, then an honest freeze) ------------
        # How much narration time does the available film have to cover?
        #  * enough film  -> 1x, untouched;
        #  * a little short -> one mild slow-down, never below min_speed;
        #  * a lot short -> min_speed AND the final frame of the section is
        #    held (tpad clone) for the remainder: a still frame reads as an
        #    editing choice, a 0.3x crawl reads as a broken render.
        entry = max(f0, film_playhead)
        room = ceiling - entry
        if not broll:
            # B-ROLL DISABLED: a section may not wander into film nobody
            # narrates to pad itself out.
            ceiling = f1
            room = ceiling - entry
        if broll and ceiling > f1 + 1e-6 and total_nar > 0 and room > 0:
            # how much of the gap this section actually SPENT (own window
            # first, then the un-narrated film beyond it)
            _film_used = min(room, total_nar * min(1.0, max(room / total_nar, 0.0)))
            _own = max(f1 - entry, 0.0)
            _borrow = _film_used - _own
            if _borrow > 0.5:
                borrowed_groups += 1
                borrowed_secs += _borrow
        if total_nar > 0 and room > 0:
            ideal = room / total_nar
        else:
            ideal = 0.0
        starved = ideal < min_speed and not audio_first
        if audio_first or room <= 0 or ideal >= 1.0:
            grp_speed = 1.0
        elif starved and not slow_mo_before_freeze:
            grp_speed = 1.0        # freeze-only padding: never slow the picture
        else:
            grp_speed = max(min_speed, ideal)
        if grp_speed < 0.999:
            slowed += 1
            slowed_secs += total_nar * (1.0 - grp_speed) / max(grp_speed, 1e-6)

        acc = 0.0
        for i in idxs:
            # stub beats emit no clip: their speech time was folded into the
            # previous beat by lock_durations, so this beat would only flash
            # 1-2 frames -- a stutter, not a shot
            if float(durations[i]) <= 0.001:
                continue
            d = max(durations[i], 0.05)
            # position inside this group's film window, proportional to how far
            # through the group's narration we are -> monotone within the group
            film_pos = f0 + (acc / total_nar) * span
            film_span = (d / total_nar) * span
            film_pos = max(film_pos, playhead)

            if audio_first:
                # Start at the sentence's absolute anchor (the transcript
                # timestamp the writer tagged it with), never before the
                # film playhead; run exactly ``d`` seconds of film at 1x.
                a = sentences[i].get("anchor")
                if a is not None:
                    desired = float(a) - pre_roll
                else:
                    desired = float(sentences[i].get("film_start", film_pos)
                                    or film_pos)
                zl = sentences[i].get("zone_lo")
                if zl is not None:
                    desired = max(desired, float(zl))
                desired = max(desired, 0.0)
                start = max(desired, film_playhead)
                if start > desired + 1e-9:
                    pushed += 1
                if beats and 0.0 < start - film_playhead < min_new:
                    start = film_playhead   # continue instead of a micro-jump
                if scene_bounds:
                    s2 = _snap_to_boundary(start, scene_bounds, snap_tol)
                    if s2 != start and s2 >= film_playhead:
                        start = s2
                        snapped += 1
                moving = d
                if movie_dur > 0:
                    start = min(start, movie_dur)
                    if start + moving > movie_dur:
                        moving = max(movie_dur - start, 0.0)
                freeze = max(d - moving, 0.0)
                if freeze > 1e-6:
                    held += 1         # end-of-film only
                film_playhead = start + moving
                beats.append(
                    {
                        "index": i,
                        "sentence": sentences[i].get("sentence", ""),
                        "film_start": round(start, 3),
                        "film_end": round(start + moving, 3),
                        "duration": round(d, 3),
                        "cuts": [[start, d, freeze, 1.0]],
                    }
                )
                acc += d
                playhead = max(playhead, start)
                continue

            # word-measured switch points inside this sentence (narration clock)
            fracs = None
            if cut_on_words and word_times and i < len(word_times) \
                    and i < len(cum):
                fracs = _word_boundary_fractions(word_times[i], cum[i], d)
                if fracs:
                    word_locked += 1

            shots = _shot_split(
                d, fracs, micro_target=micro_target, max_cuts=max_cuts,
                min_cut=min_cut, max_shot=max_shot,
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
                # FILM BUDGET per cut. A cut plays ``per`` seconds of
                # narration; at ``grp_speed`` it needs ``per * speed`` seconds
                # of film. When the section's film budget runs out first —
                # the honest over-budget case — the shot HOLDS its final
                # frame (clip.cut_segment renders it with tpad
                # stop_mode=clone) instead of being stretched further.
                # Freezing is also the only legal move at the very end of the
                # movie, where there is no film left at all.
                freeze = 0.0
                moving = per * grp_speed
                if starved and freeze_when_starved:
                    film_left = max(ceiling - film_playhead, 0.0)
                    if moving > film_left:
                        moving = max(film_left, 0.0)
                if movie_dur > 0 and film_playhead + moving > movie_dur:
                    moving = max(movie_dur - film_playhead, 0.0)
                if moving < per * grp_speed - 1e-9:
                    freeze = per - moving / max(grp_speed, 1e-6)
                    if starved and freeze_when_starved:
                        padded += 1
                        padded_secs += freeze
                    else:
                        held += 1
                cut = [start, per, freeze, grp_speed]
                cuts.append(cut)
                film_playhead = max(film_playhead, start) + moving

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
    if audio_durations:
        max_atempo = float((cfg_timeline or {}).get("max_atempo", 1.15))
        apply_beat_retiming_to_timeline(beats, audio_durations, max_atempo=max_atempo)

    if stats is not None:
        stats["word_locked_beats"] = word_locked
        stats["snapped_cuts"] = snapped
        stats["pushed_cuts"] = pushed
        stats["held_shots"] = held
        stats["slowed_groups"] = slowed
        stats["slowed_seconds"] = round(slowed_secs, 1)
        stats["padded_shots"] = padded
        stats["padded_seconds"] = round(padded_secs, 1)
        stats["borrowed_groups"] = borrowed_groups
        stats["borrowed_seconds"] = round(borrowed_secs, 1)
    return beats


def compute_dynamic_retiming(
    audio_dur: float,
    video_dur: float,
    max_atempo: float = 1.15,
) -> dict:
    """Calculate fallback retiming parameters for one visual/audio beat pair.

    1. Audio Stretching (atempo): If audio is slightly longer than video
       (video_dur < audio_dur <= video_dur * 1.15), speed up audio by up to
       1.15x so audio fits the video beat.
    2. Video Freeze-Framing: If narration heavily overruns the visual scene
       (audio_dur > video_dur * 1.15), freeze the final frame of the video
       beat until the audio finishes, rather than letting audio bleed into
       the next scene.
    3. Silence Padding: If video beat is longer than audio (video_dur > audio_dur),
       insert silence at the end of the TTS file so the next narration line
       snaps exactly to the start of the next visual scene.
    """
    a_dur = max(float(audio_dur), 0.05)
    v_dur = max(float(video_dur), 0.05)

    if abs(a_dur - v_dur) < 0.01:
        return {
            "action": "exact",
            "audio_dur": round(a_dur, 3),
            "video_dur": round(v_dur, 3),
            "atempo": 1.0,
            "freeze": 0.0,
            "silence_pad": 0.0,
            "final_dur": round(v_dur, 3),
        }

    if v_dur < a_dur <= v_dur * max_atempo:
        speed = a_dur / v_dur
        return {
            "action": "atempo",
            "audio_dur": round(a_dur, 3),
            "video_dur": round(v_dur, 3),
            "atempo": round(speed, 4),
            "freeze": 0.0,
            "silence_pad": 0.0,
            "final_dur": round(v_dur, 3),
        }

    if a_dur > v_dur * max_atempo:
        eff_a_dur = a_dur / max_atempo
        freeze = eff_a_dur - v_dur
        return {
            "action": "freeze",
            "audio_dur": round(a_dur, 3),
            "video_dur": round(v_dur, 3),
            "atempo": round(max_atempo, 4),
            "freeze": round(freeze, 3),
            "silence_pad": 0.0,
            "final_dur": round(eff_a_dur, 3),
        }

    silence_pad = v_dur - a_dur
    return {
        "action": "silence_padding",
        "audio_dur": round(a_dur, 3),
        "video_dur": round(v_dur, 3),
        "atempo": 1.0,
        "freeze": 0.0,
        "silence_pad": round(silence_pad, 3),
        "final_dur": round(v_dur, 3),
    }


def apply_beat_retiming_to_timeline(
    beats: list[dict],
    audio_durations: list[float],
    max_atempo: float = 1.15,
) -> list[dict]:
    """Apply dynamic retiming fallback logic across the timeline beats.

    - Freezes the video beat's final frame when narration heavily overruns.
    - Speeds up audio up to 1.15x when audio is slightly longer.
    - Pads silence when video is longer, snapping the next line to the next scene.
    """
    for i, b in enumerate(beats):
        if i >= len(audio_durations):
            break
        a_dur = float(audio_durations[i])
        v_dur = float(b.get("duration", 0.0))
        retiming = compute_dynamic_retiming(a_dur, v_dur, max_atempo=max_atempo)
        b["retiming"] = retiming

        if retiming["action"] == "freeze" and retiming["freeze"] > 0:
            cuts = list(b.get("cuts") or [])
            if cuts:
                s, d, f, v = cuts[-1]
                cuts[-1] = (
                    round(float(s), 3),
                    round(float(d) + retiming["freeze"], 3),
                    round(float(f) + retiming["freeze"], 3),
                    round(float(v), 3),
                )
                b["cuts"] = cuts
            b["duration"] = round(retiming["final_dur"], 3)
    return beats


def rewindow_to_speech(
    segments: list[dict],
    durations: list[float],
    movie_dur: float = 0.0,
) -> list[dict]:
    """Re-size every sentence's film window from its MEASURED speech
    duration -- the structural fix for "the narration runs ahead of the
    visuals / the visuals move slowly".

    Why this exists: the script sizes each sentence's window from an
    ESTIMATE of its speech length (``words / words_per_minute``), computed
    before any audio exists. If the real voice speaks slower than the
    configured wpm (voice choice, ``rate: "-8%"``, longer pauses, another
    language), every window comes out smaller than its sentence's real
    duration -- and the timeline then paces essentially every section
    below 1x: permanent slow motion, the picture falling behind the
    narration from the first scene. No script-side fix can cure that,
    because the script only ever sees the estimate.

    After TTS (+ whisper alignment) the pipeline knows the REAL duration of
    every sentence. This pass re-sizes each sentence's window to that
    measured duration -- and keeps it ON THE BEAT the sentence narrates.
    Each segment carries the film moment it was written from (``anchor``,
    set by ``script.generate_segmented_script``; old cached segments carry
    no anchor, and the midpoint of their pre-resize window is used
    instead). The new window is that anchor, centered, exactly as long as
    the sentence's real speech, clamped to the section's zone
    (``zone_lo``/``zone_hi``); a forward walk resolves any overlap so the
    windows never rewind and the picture always lands on the moment the
    voice is describing.

    Why the anchor matters (the reported "narration and visuals do not
    match at all"): the earlier version of this pass re-tiled each zone
    PROPORTIONALLY TO SPEECH TIME, throwing the anchors away. Beats are not
    spread evenly through a zone -- a zone holds one busy scene and a quiet
    stretch -- so proportional tiling made every sentence show footage 15-40
    seconds off the beat it narrates, from the first second of the video,
    and stretched a 5-second sentence's three micro-cuts across a 50-second
    window (the picture racing ~9x ahead of the story inside one sentence).
    Anchored windows fix both: the median sentence now sits within a couple
    of seconds of its beat, and a sentence's shots stay inside that
    sentence's own beat.

    Because the section budgets capped the narration at ~40% of the zone's
    film time, the measured windows normally fit the zone with room to
    spare and the timeline plays EVERY cut at 1x. A section whose measured
    narration genuinely exceeds its film zone (the rare, honest case)
    keeps a contiguous walk scaled to the zone, and the timeline's
    slow-motion safety net paces it.

    Sentences without zone information (the sign-off outro on an old
    cache, pre-zone segments) keep their existing windows untouched.
    """
    n = min(len(segments), len(durations))
    if n <= 0:
        return segments
    out = [dict(s) for s in segments[:n]]
    i = 0
    resized = 0
    while i < n:
        zl = segments[i].get("zone_lo")
        zh = segments[i].get("zone_hi")
        if zl is None or zh is None or float(zh) <= float(zl) + 1e-6:
            i += 1
            continue
        j = i
        while (j < n and segments[j].get("zone_lo") == zl
               and segments[j].get("zone_hi") == zh):
            j += 1
        # sentences i..j-1 share this film zone
        durs = [max(float(durations[k]), 0.05) for k in range(i, j)]
        total = sum(durs)
        lo0 = max(float(zl), 0.0)
        hi0 = float(zh)
        if movie_dur > 0:
            hi0 = min(hi0, movie_dur)
        zone = max(hi0 - lo0, 0.0)
        if zone <= 0 or total <= 0:
            i = j
            continue
        if total <= zone:
            # ANCHOR-TRUE PLACEMENT: the section fits its footage, so every
            # sentence keeps the beat it was written from: window = anchor
            # (centered), sized to the MEASURED speech, clamped to the zone.
            # A forward walk resolves overlaps (monotone, no rewind); when
            # anchors are well spaced (the paced case) the walk never binds
            # and every window is exactly centered on its beat.
            anchors = []
            for rel in range(i, j):
                s = out[rel]
                a = s.get("anchor")
                if a is None:
                    a = (float(s.get("film_start", lo0))
                         + float(s.get("film_end", lo0))) / 2.0
                anchors.append(min(max(float(a), lo0), hi0))
            walk = lo0
            for k, rel in enumerate(range(i, j)):
                d = durs[k]
                a = anchors[k]
                lo = min(max(a - d / 2.0, walk), max(hi0 - d, walk))
                hi = min(lo + d, hi0)
                if hi < lo:
                    hi = lo
                out[rel]["film_start"] = round(lo, 3)
                # 0.8s floor so a whisper-stub sentence still gets a real
                # shot -- but the floor may never push a window past the
                # zone end (degenerate zone-end case: keep it inside).
                out[rel]["film_end"] = round(max(hi, min(lo + 0.8, hi0)), 3)
                walk = hi
                resized += 1
        else:
            # Genuinely over-budget section (measured narration longer than
            # the film zone): the zone cannot hold the speech, so walk the
            # sentences contiguously through it at zone scale; the timeline
            # then paces the section (B-roll borrowing first, then min_speed,
            # then a held final frame -- rare and honest, not the norm).
            scale = zone / total
            walk = 0.0
            for rel in range(i, j):
                lo = lo0 + walk
                walk += durs[rel - i] * scale
                hi = min(lo0 + walk, hi0)
                if hi < lo:
                    hi = lo
                out[rel]["film_start"] = round(lo, 3)
                out[rel]["film_end"] = round(max(hi, min(lo + 0.8, hi0)), 3)
                resized += 1
        i = j
    if resized:
        print(f"  * visual match: {resized} sentence windows re-sized to "
              "the MEASURED narration and kept on the beat each sentence "
              "narrates -- every section can now play at 1x in sync",
              flush=True)
    return out


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


def cut_order_violations(cuts: list[tuple], tol: float = 0.02) -> list[int]:
    """Indices of cuts whose film PTS replays footage the previous cut showed.

    Problem 2 of the bug report — "visuals shown before the relevant scene /
    ordering chaos" — is exactly this: a clip whose source PTS is behind the
    clip that plays before it. ``build_timeline`` makes it structurally
    impossible (the film playhead never rewinds), so this is the assertion
    that keeps it that way and names the offending shots when something
    upstream hands the assembler a shuffled plan.
    """
    bad: list[int] = []
    playhead = None
    for i, cut in enumerate(cuts or []):
        try:
            start, dur, freeze, speed = cut
        except (TypeError, ValueError):
            continue
        start = float(start)
        moving = max((float(dur) - float(freeze)) * float(speed), 0.0)
        if playhead is not None and start < playhead - tol:
            bad.append(i)
        playhead = max(playhead or 0.0, start + moving)
    return bad


def repair_cut_order(cuts: list[tuple], tol: float = 0.02) -> tuple[list[tuple], int]:
    """Push a rewind-causing cut forward to the playhead (never reorder).

    Sorting the shots by PTS would put them under the WRONG sentences — the
    cut order IS the narration timeline — so a violation is repaired by
    starting the offending shot where the previous one ended. Returns
    ``(cuts, fixed_count)``.
    """
    out: list[tuple] = []
    playhead: float | None = None
    fixed = 0
    for cut in cuts or []:
        try:
            start, dur, freeze, speed = cut
        except (TypeError, ValueError):
            continue
        start = float(start)
        if playhead is not None and start < playhead - tol:
            start = playhead
            fixed += 1
        moving = max((float(dur) - float(freeze)) * float(speed), 0.0)
        playhead = max(playhead or 0.0, start + moving)
        out.append((round(start, 3), float(dur), float(freeze), float(speed)))
    return out, fixed


def timeline_report(beats: list[dict], audio_span: float,
                    word_locked: int = 0, snapped: int = 0,
                    slowed: int = 0, slowed_secs: float = 0.0,
                    padded: int = 0, padded_secs: float = 0.0,
                    borrowed: int = 0) -> str:
    """One-line human summary used in the run log."""
    cuts = flatten_cuts(beats)
    total = sum(d for _, d, _f, _v in cuts)
    starts = [b["film_start"] for b in beats]
    monotone = all(starts[i] <= starts[i + 1] + 1e-6 for i in range(len(starts) - 1))
    bits = [f"{len(beats)} beats / {len(cuts)} cuts"]
    # The direct answer to "a single visual remained for 15-20 sentences":
    # a cut cannot outlive its own sentence, so a small longest-shot number
    # makes that pathology structurally impossible in this render.
    if cuts:
        bits.append(f"longest shot {max(d for _, d, _f, _v in cuts):.1f}s")
    if word_locked:
        bits.append(f"word-locked {word_locked}/{len(beats)} beats")
    if snapped:
        bits.append(f"{snapped} cuts on shot changes")
    if slowed:
        bits.append(f"{slowed} slow-mo sections ({slowed_secs:.0f}s)")
    if padded:
        bits.append(f"{padded} held-frame shots ({padded_secs:.0f}s)")
    if borrowed:
        bits.append(f"{borrowed} sections on B-roll")
    out_of_order = cut_order_violations(cuts)
    if out_of_order:
        bits.append(f"!! {len(out_of_order)} cut(s) out of PTS order")
    return (
        f"{', '.join(bits)}, "
        f"video {total:.1f}s vs narration {audio_span:.1f}s "
        f"(drift {abs(total - audio_span) * 1000:.0f}ms), "
        f"chronological={'yes' if monotone else 'NO'}, "
        f"film coverage {min(starts, default=0):.0f}s -> {max(starts, default=0):.0f}s"
    )
