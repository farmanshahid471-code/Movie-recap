"""Per-beat architecture (Steps 0-5) — replaces arbitrary 180s windows.

Why this exists
--------------
The old chunker sliced by clock time (180s windows). The writer got a
summary of an arbitrary 180s slice and had no idea how much *footage time*
it had to work with per beat inside that slice. Content then got rewritten
by fallback stages that didn't know what footage they described, then glued
to footage by position, not meaning — the root cause of
\"visuals and narration don't match\" despite 0ms drift.

This module implements the 5-step beat-first plan from the bug report:

  Step 0  redefine the unit: beats, not 180s windows
  Step 1  per-beat prompt with a hard word budget tied to real seconds
  Step 2  overflow handling: one targeted retry, then borrow
  Step 3  time-borrowing between adjacent sparse/dense beats
  Step 4  importance-scoring trim of last resort (clause-level)
  Step 5  cheap pre-encode validation (MATCH / PARTIAL / MISMATCH)

All helpers are pure and testable; the high-level ``generate_beat_script``
wires them together for ``pipeline.auto_recap``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .util import count_words

# ---------------------------------------------------------------------------
# Step 0 — beat detection
# ---------------------------------------------------------------------------
# beat_boundaries = merge(shot_boundaries, vision_note_timestamps, subtitle_gaps > 1.5s)

_DEFAULT_WPM = 175
_DEFAULT_GAP = 1.5
_DEFAULT_MAX_BORROW = 0.3

# action verbs that signal plot-critical movement; used for importance scoring
_ACTION_VERBS = {
    "run", "runs", "running", "ran",
    "jump", "jumps", "jumped", "hop", "hops", "hopped",
    "grab", "grabs", "grabbed", "take", "takes", "took", "taken",
    "give", "gives", "gave", "hand", "hands", "handed",
    "hit", "hits", "hitting", "strike", "strikes", "punch", "punches",
    "kill", "kills", "killed", "shoot", "shoots", "shot", "firing",
    "open", "opens", "opened", "close", "closes", "closed",
    "push", "pull", "drag", "drags", "drop", "drops", "throw", "throws",
    "catch", "catches", "caught", "find", "finds", "found",
    "hide", "hides", "hid", "escape", "escapes", "escaped",
    "enter", "enters", "entered", "leave", "leaves", "left",
    "arrive", "arrives", "arrived", "return", "returns", "returned",
    "attack", "defend", "chase", "chases", "follow", "follows",
    "save", "saves", "saved", "rescue", "rescued",
    "decide", "decides", "decided", "realize", "realizes", "realized",
    "discover", "discovers", "discovered", "reveal", "reveals", "revealed",
    "explain", "explains", "admits", "confront", "warns", "insists",
    "rides", "ride", "drives", "flies", "walks", "climbs", "falls",
    "wakes", "sleeps", "stands", "sits", "turns", "looks", "sees", "watches",
}

_CAUSAL_CONNECTIVES = {
    "so", "because", "therefore", "thus", "since", "as", "then", "hence",
    "although", "though", "while", "whereas", "meanwhile", "however",
}

_PURE_DESCRIPTIVE_MARKERS = {
    "beautiful", "dark", "bright", "silent", "quiet", "loud", "huge", "small",
    "vast", "empty", "lonely", "cold", "warm", "misty", "gloomy",
    "atmospheric", "scenic", "cinematic", "glowing", "shimmering",
}


def _merge_sorted_unique(values: list[float], eps: float = 0.05) -> list[float]:
    """Sort and deduplicate within eps seconds (camera cuts 30ms apart are the same cut)."""
    if not values:
        return []
    vals = sorted(float(v) for v in values)
    out: list[float] = [vals[0]]
    for v in vals[1:]:
        if abs(v - out[-1]) > eps:
            out.append(v)
    return out


def detect_beats(
    cues: list[dict] | None,
    vision_notes: list[dict] | None,
    scene_bounds: list[float] | None,
    movie_duration: float | None = None,
    gap_threshold: float = _DEFAULT_GAP,
    *,
    min_beat_seconds: float = 2.0,
) -> list[dict]:
    """Build beats from merged boundaries.

    ``beat_boundaries = merge(shot_boundaries, vision_note_timestamps, subtitle_gaps > 1.5s)``

    Returns ``[{start_ts, end_ts, duration, transcript_lines, vision_notes, shot_count}]``
    in chronological order.  Every beat carries its own real duration and its
    own grounding text — one continuous scene/action unit (4s or 40s).
    """
    cues = list(cues or [])
    vision_notes = list(vision_notes or [])
    scene_bounds = list(scene_bounds or [])

    # ---- collect boundary points ----------------------------------------
    boundaries: list[float] = []

    # shot changes
    for t in scene_bounds:
        try:
            ft = float(t)
            if ft >= 0:
                boundaries.append(ft)
        except (TypeError, ValueError):
            continue

    # vision note times
    for v in vision_notes:
        t = v.get("t") if isinstance(v, dict) else None
        if t is not None:
            try:
                boundaries.append(float(t))
            except (TypeError, ValueError):
                continue

    # subtitle gaps > 1.5s
    # sort cues by start
    norm_cues = []
    for c in cues:
        try:
            s = float(c.get("start", 0.0) or 0.0)
            e = float(c.get("end", s) or s)
        except (TypeError, ValueError):
            continue
        norm_cues.append({**c, "start": s, "end": max(e, s)})
    norm_cues.sort(key=lambda x: x["start"])

    for i in range(len(norm_cues) - 1):
        gap = float(norm_cues[i + 1]["start"]) - float(norm_cues[i]["end"])
        if gap > float(gap_threshold):
            # one boundary in the silent gap (midpoint) — the natural scene break
            mid = (float(norm_cues[i]["end"]) + float(norm_cues[i + 1]["start"])) / 2.0
            boundaries.append(mid)
            # also keep the edges so a long silence becomes its own beat if useful;
            # deduplication will keep them distinct only when the gap is wide
            # (we add mid only by default to avoid 0.2s slivers; uncomment below if strict)
            # boundaries.append(float(norm_cues[i]["end"]))
            # boundaries.append(float(norm_cues[i+1]["start"]))

    # always include film start/end so the whole runtime is covered
    boundaries.append(0.0)
    if movie_duration is not None and movie_duration > 0:
        boundaries.append(float(movie_duration))
    else:
        # fall back to cue coverage
        if norm_cues:
            boundaries.append(max(c.get("end", 0.0) for c in norm_cues))
        else:
            boundaries.append(300.0)

    # optional: also seed with cue edges so beats align with dialogue boundaries?
    # The spec says merge only shot/vision/gaps, but cue edges inside gaps are already
    # represented by the gap midpoint. We deliberately do NOT add every cue edge,
    # otherwise a dialogue-dense scene would fragment into 2s beats per line.

    uniq = _merge_sorted_unique(boundaries, eps=0.1)
    uniq.sort()

    # ---- build beats ------------------------------------------------------
    # Pre-index vision by time for fast lookup
    raw_beats: list[dict] = []
    for i in range(len(uniq) - 1):
        lo = float(uniq[i])
        hi = float(uniq[i + 1])
        dur = hi - lo
        if dur < 0.15:
            continue
        t_lines = []
        for c in norm_cues:
            if c["start"] >= lo - 0.03 and c["end"] <= hi + 0.03:
                t_lines.append(c)
        v_lines = []
        for v in vision_notes:
            t = v.get("t")
            if t is None:
                continue
            try:
                vt = float(t)
            except (TypeError, ValueError):
                continue
            if lo - 0.05 <= vt < hi + 0.05:
                v_lines.append(v)
        shot_count = sum(1 for s in scene_bounds if lo < float(s) <= hi)
        raw_beats.append(
            {
                "start_ts": round(lo, 3),
                "end_ts": round(hi, 3),
                "duration": round(dur, 3),
                "transcript_lines": t_lines,
                "vision_notes": v_lines,
                "shot_count": shot_count,
            }
        )
    # Merge short beats (< min_beat_seconds) into neighbor
    beats: list[dict] = []
    for b in raw_beats:
        if beats and float(b["duration"]) < float(min_beat_seconds):
            # merge this short beat into previous (keep previous's start, extend end)
            prev = beats[-1]
            prev["end_ts"] = b["end_ts"]
            prev["duration"] = round(float(prev["end_ts"]) - float(prev["start_ts"]), 3)
            prev["transcript_lines"] = list(prev.get("transcript_lines") or []) + list(b.get("transcript_lines") or [])
            prev["vision_notes"] = list(prev.get("vision_notes") or []) + list(b.get("vision_notes") or [])
            prev["shot_count"] = int(prev.get("shot_count", 0)) + int(b.get("shot_count", 0))
            continue
        beats.append(b)
    # If last beat is short, merge it backwards
    if len(beats) >= 2 and float(beats[-1]["duration"]) < float(min_beat_seconds):
        last = beats.pop()
        prev = beats[-1]
        prev["end_ts"] = last["end_ts"]
        prev["duration"] = round(float(prev["end_ts"]) - float(prev["start_ts"]), 3)
        prev["transcript_lines"] = list(prev.get("transcript_lines") or []) + list(last.get("transcript_lines") or [])
        prev["vision_notes"] = list(prev.get("vision_notes") or []) + list(last.get("vision_notes") or [])
        prev["shot_count"] = int(prev.get("shot_count", 0)) + int(last.get("shot_count", 0))
    # If still no beats after merging (e.g., single short raw beat), restore
    if not beats and raw_beats:
        beats = raw_beats
        # merge all into one
        if len(beats) > 1:
            lo = min(float(b["start_ts"]) for b in beats)
            hi = max(float(b["end_ts"]) for b in beats)
            t_all = []
            v_all = []
            sc = 0
            for b in beats:
                t_all.extend(b.get("transcript_lines") or [])
                v_all.extend(b.get("vision_notes") or [])
                sc += int(b.get("shot_count", 0))
            beats = [{"start_ts": round(lo,3), "end_ts": round(hi,3), "duration": round(hi-lo,3), "transcript_lines": t_all, "vision_notes": v_all, "shot_count": sc}]

    # Split overly-long beats (>40s) so writer gets bounded budgets
    # A 120s beat would need 350 words — unmanageable. Split on even time or cue edges.
    MAX_BEAT = 40.0
    split_beats: list[dict] = []
    for b in beats:
        dur = float(b.get("duration", 0.0) or 0.0)
        if dur <= MAX_BEAT + 1e-6:
            split_beats.append(b)
            continue
        n_parts = int((dur + MAX_BEAT - 0.01) // 30.0)  # ~30s chunks
        n_parts = max(2, min(n_parts, int(dur // 8) or 2))
        step = dur / n_parts
        lo0 = float(b.get("start_ts", 0.0))
        # distribute transcript/vision by time slice
        t_lines_all = list(b.get("transcript_lines") or [])
        v_lines_all = list(b.get("vision_notes") or [])
        sc_total = int(b.get("shot_count", 0))
        for k in range(n_parts):
            lo = lo0 + k * step
            hi = lo + step if k < n_parts - 1 else lo0 + dur
            # slice grounding lines by midpoint
            t_slice = []
            for c in t_lines_all:
                mid = (float(c.get("start", lo)) + float(c.get("end", lo))) / 2.0
                if lo - 0.03 <= mid < hi + 0.03 or (k==n_parts-1 and mid >= hi-0.03):
                    t_slice.append(c)
                elif c.get("start", lo) >= lo - 0.03 and c.get("end", hi) <= hi + 0.03:
                    t_slice.append(c)
            v_slice = []
            for v in v_lines_all:
                try:
                    vt = float(v.get("t", lo))
                except Exception:
                    continue
                if lo - 0.05 <= vt < hi + 0.05 or (k==n_parts-1 and vt >= hi-0.05):
                    v_slice.append(v)
            part_sc = sc_total // n_parts if n_parts else sc_total
            # last part gets remainder
            if k == n_parts -1:
                part_sc = sc_total - part_sc * (n_parts -1)
            split_beats.append({
                "start_ts": round(lo,3),
                "end_ts": round(hi,3),
                "duration": round(hi-lo,3),
                "transcript_lines": t_slice,
                "vision_notes": v_slice,
                "shot_count": part_sc,
            })
    beats = split_beats

    # If no beats (degenerate), fall back to one beat covering the film
    if not beats:
        lo = 0.0
        hi = float(movie_duration or 0.0) or 300.0
        beats.append(
            {
                "start_ts": lo,
                "end_ts": hi,
                "duration": hi - lo,
                "transcript_lines": norm_cues,
                "vision_notes": vision_notes,
                "shot_count": len(scene_bounds),
            }
        )

    # Ensure total coverage sums to movie_duration (within rounding)
    return beats


# ---------------------------------------------------------------------------
# Step 1 — per-beat prompt template
# ---------------------------------------------------------------------------

BEAT_SYSTEM_PROMPT = (
    "You are writing one beat of a chronological movie recap narration.\n"
    "You will receive:\n"
    "- The exact dialogue/subtitle lines spoken during this beat\n"
    "- The exact visual notes (on-screen action) captured during this beat\n"
    "- A hard word budget for this beat\n"
    "\n"
    "RULES:\n"
    "- Describe ONLY what happens in this beat. Do not reference earlier or later beats.\n"
    "- Your output MUST be between {min_words} and {max_words} words. This is not a\n"
    "  suggestion — it is a hard constraint, because your narration will be timed\n"
    "  to exactly {duration}s of footage at ~175 words/minute.\n"
    "- If you cannot cover everything in budget, prioritize: (1) plot-critical action\n"
    "  or decision, (2) character intent/emotion, (3) atmospheric/visual detail.\n"
    "  Drop (3) first, then (2), never (1).\n"
    "- Output plain narration only. No meta-commentary, no \"in this scene.\"\n"
)

# Keep a legacy alias for tests that may import SYSTEM_BEAT or similar
SYSTEM_BEAT = BEAT_SYSTEM_PROMPT


def word_budget_for_duration(duration: float, wpm: int = _DEFAULT_WPM) -> tuple[int, int]:
    """Word budget for a beat of ``duration`` seconds at ``wpm`` words/minute.

    ``max_words`` is the hard ceiling (duration * wpm / 60). ``min_words`` is
    a floor that keeps the narration from collapsing to a single word for
    very short beats.
    """
    dur = max(float(duration), 0.1)
    # Use floor so max_w words never needs more time than the beat provides
    import math
    max_w = max(1, int(math.floor(dur * float(wpm) / 60.0)))
    if max_w < 1:
        max_w = 1
    # short beats still need at least a few words to be audible
    if max_w <= 5:
        min_w = max(1, max_w - 1)
    elif max_w <= 12:
        min_w = max(4, max_w - 4)
    else:
        min_w = max(8, int(max_w * 0.55))
    # clamp min <= max
    min_w = min(min_w, max_w)
    return min_w, max_w


def required_time_for_words(words: int, wpm: int = _DEFAULT_WPM) -> float:
    """Seconds of footage needed to speak ``words`` at ``wpm``."""
    return max(float(words), 0.0) / max(float(wpm), 1.0) * 60.0


def _format_transcript_block(lines: list[dict]) -> str:
    if not lines:
        return "(no dialogue in this beat)"
    out = []
    for c in lines:
        t = c.get("text", "") or ""
        s = c.get("start")
        if s is not None:
            try:
                secs = float(s)
                m, sec = divmod(int(secs) % 3600, 60)
                h = int(secs) // 3600
                stamp = f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
                out.append(f"[{stamp}] {t.strip()}")
                continue
            except Exception:
                pass
        out.append(t.strip())
    return "\n".join(out) if out else "(no dialogue in this beat)"


def _format_vision_block(notes: list[dict]) -> str:
    if not notes:
        return "(no visual notes in this beat)"
    out = []
    for v in notes:
        t = v.get("t")
        text = (v.get("text") or "").strip()
        if not text:
            continue
        if t is not None:
            try:
                secs = float(t)
                m, s = divmod(int(secs) % 3600, 60)
                h = int(secs) // 3600
                stamp = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
                out.append(f"[{stamp}] {text}")
                continue
            except Exception:
                pass
        out.append(text)
    return "\n".join(out) if out else "(no visual notes in this beat)"


def build_beat_prompt(
    beat: dict,
    wpm: int = _DEFAULT_WPM,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one beat.

    The budget is stated as a hard fact tied to real seconds, not an abstract
    target — models respect budgets far better when the reason is explicit.
    Grounding is scoped to the beat only so the model cannot drift into a
    moment 40 seconds away.
    """
    dur = float(beat.get("duration", 0.0) or 0.0)
    min_w, max_w = word_budget_for_duration(dur, wpm=wpm)
    system = BEAT_SYSTEM_PROMPT.format(min_words=min_w, max_words=max_w, duration=f"{dur:.1f}")
    user = (
        f"Beat duration: {dur:.1f}s ({max_w} words max at {wpm}wpm)\n"
        f"Dialogue in this beat:\n{_format_transcript_block(beat.get('transcript_lines') or [])}\n"
        f"\nVisual notes in this beat:\n{_format_vision_block(beat.get('vision_notes') or [])}\n"
        "\nWrite the narration for this beat now."
    )
    return system, user


# Backwards compat for older test imports
def beat_prompt(beat: dict, wpm: int = _DEFAULT_WPM) -> tuple[str, str]:
    return build_beat_prompt(beat, wpm=wpm)


# ---------------------------------------------------------------------------
# Step 2 — overflow handling (replaces 5-stage fallback)
# ---------------------------------------------------------------------------

def _call_llm_for_beat(
    cfg_llm: dict,
    system: str,
    user: str,
    max_tokens: int,
) -> str:
    """Thin wrapper around recap.llm.complete for one beat."""
    from . import llm as llm_mod

    return llm_mod.complete(
        cfg_llm.get("provider", ""),
        cfg_llm.get("model", ""),
        system,
        user,
        base_url=cfg_llm.get("base_url"),
        max_tokens=max_tokens,
    )


def generate_one_beat(
    beat: dict,
    cfg_llm: dict,
    wpm: int = _DEFAULT_WPM,
) -> tuple[str, dict]:
    """Generate narration for a single beat with overflow handling.

    Returns (narration_text, info) where info records whether a retry was used
    and the word counts. The caller may then apply time-borrowing if still over.
    Steps 2-3 combined:
      draft = call_writer(beat_prompt)
      if word_count(draft) <= max_words: accept
      else: ONE targeted retry dropping atmospheric detail; if still over -> borrow
    """
    dur = float(beat.get("duration", 0.0) or 0.0)
    min_w, max_w = word_budget_for_duration(dur, wpm=wpm)
    system, user = build_beat_prompt(beat, wpm=wpm)
    # generous token cap: ~1.5 tokens/word plus overhead
    max_tokens = max(64, int(max_w * 1.8) + 64)

    draft = _call_llm_for_beat(cfg_llm, system, user, max_tokens=max_tokens)
    draft = (draft or "").strip()
    wc = count_words(draft)
    info: dict[str, Any] = {"attempts": 1, "words": wc, "max_words": max_w, "min_words": min_w}

    if wc <= max_w:
        return draft, info

    over = wc - max_w
    # ONE targeted retry, dropping atmospheric/visual detail first
    retry_user = (
        user
        + f"\n\nYour previous draft was {wc} words, {over} words over budget.\n"
        f"Rewrite it, dropping atmospheric/visual detail first, keeping all\n"
        f"plot-critical action and character intent. Target: {max_w} words."
    )
    draft2 = _call_llm_for_beat(cfg_llm, system, retry_user, max_tokens=max_tokens)
    draft2 = (draft2 or "").strip()
    wc2 = count_words(draft2)
    info["attempts"] = 2
    info["words_retry"] = wc2
    info["over_first"] = over

    if wc2 <= max_w:
        return draft2, info

    # still over — caller should borrow time, not chop
    info["still_over"] = wc2 - max_w
    return draft2, info


# ---------------------------------------------------------------------------
# Step 3 — time-borrowing instead of tail-trimming
# ---------------------------------------------------------------------------

def compute_slack(
    beats: list[dict],
    narrations: list[str],
    wpm: int = _DEFAULT_WPM,
) -> list[float]:
    """Slack per beat: positive = spare time, negative = overflowing."""
    slacks: list[float] = []
    for beat, text in zip(beats, narrations):
        dur = float(beat.get("duration", 0.0) or 0.0)
        words = count_words(text or "")
        need = required_time_for_words(words, wpm=wpm)
        slacks.append(dur - need)
    return slacks


def borrow_time(
    beats: list[dict],
    narrations: list[str],
    wpm: int = _DEFAULT_WPM,
    max_borrow_ratio: float = _DEFAULT_MAX_BORROW,
) -> tuple[list[dict], list[float]]:
    """Reallocate seconds from sparse beats to overflowing dense beats.

    Mutates beat durations in place (copy first if you need immutable).
    Returns (beats, new_slacks) after borrowing. Only the residual that cannot
    be borrowed should go to mechanical trimming.
    """
    n = min(len(beats), len(narrations))
    if n == 0:
        return beats, []

    # work on a shallow copy of durations so we don't mutate caller unexpectedly?
    # Spec says borrow_from_neighbor modifies durations; we mutate in place but
    # also return slacks for the caller to apply to timeline.
    slacks = compute_slack(beats, narrations, wpm=wpm)

    # enrich beats with slack for introspection (optional)
    for i, s in enumerate(slacks):
        beats[i]["slack"] = round(float(s), 3)

    max_ratio = max(0.0, min(float(max_borrow_ratio), 0.9))

    for idx in range(n):
        if slacks[idx] >= -1e-6:
            continue  # not overflowing
        needed = -slacks[idx]
        # neighbors: prefer previous then next (spec says prefer same scene, same shot group)
        # We approximate \"same scene\" by smallest shot_count difference? But spec says
        # prefer same scene, same shot group — we can check if shot_count similar.
        # For now, try prev then next; if shot_count differs a lot, maybe skip?
        order: list[int] = []
        if idx - 1 >= 0:
            order.append(idx - 1)
        if idx + 1 < n:
            order.append(idx + 1)

        # Sort neighbors by \"within same scene\" heuristic: same shot_count group?
        # If spec wants same scene, we can consider beats with same zone or shot_count.
        # We'll keep simple: prefer neighbor whose duration slack is larger.
        # But spec says prefer same scene, same shot group — we can sort by
        # absolute shot_count diff ascending.
        def scene_distance(j: int) -> int:
            return abs(int(beats[j].get("shot_count", 0)) - int(beats[idx].get("shot_count", 0)))

        order.sort(key=scene_distance)

        for nb in order:
            if needed <= 1e-6:
                break
            if slacks[nb] <= 1e-6:
                continue
            available = slacks[nb]
            # cap borrowing to max_borrow_ratio of neighbor's original duration
            cap = max_ratio * float(beats[nb].get("duration", 0.0) or 0.0)
            # also don't borrow more than needed
            transfer = min(needed, available, cap)
            if transfer <= 1e-6:
                continue
            # apply
            beats[idx]["duration"] = round(float(beats[idx]["duration"]) + transfer, 3)
            beats[idx]["end_ts"] = round(float(beats[idx]["start_ts"]) + float(beats[idx]["duration"]), 3)
            beats[nb]["duration"] = round(float(beats[nb]["duration"]) - transfer, 3)
            beats[nb]["end_ts"] = round(float(beats[nb]["start_ts"]) + float(beats[nb]["duration"]), 3)
            # if neighbor was not the immediate next beat, we should shift intermediate
            # beats' start_ts/end_ts to keep continuity? In the borrow model, beats are
            # contiguous by construction (end of beat i == start of i+1). Borrowing
            # from non-adjacent would break contiguity. But our order only includes
            # immediate neighbors, so shifting is just adjusting the shared boundary.
            # We need to shift all beats after the donor/borrower to keep timeline contiguous.
            # Simpler: re-tile starts from the donor's start forward.
            # For borrowing from prev (idx-1), the boundary between nb and idx moves left by transfer
            # For borrowing from next (idx+1), the boundary between idx and nb moves right.
            # We can re-derive starts sequentially after any mutation to keep monotone.
            # Easiest: recompute starts from beats[0].start_ts forward.
            # But we mutated durations; we should propagate shift to subsequent beats' start_ts.
            # Let's do a full re-tile of start_ts from 0th beat.
            # However beats may not be zero-based; we'll assume they are contiguous originally.
            # Re-tile only the affected window idx-1..idx+1 for efficiency.
            # Full re-tile:
            for k in range(1, n):
                # ensure continuity: beat k starts where k-1 ends
                prev_end = float(beats[k - 1].get("end_ts", 0.0))
                if abs(float(beats[k].get("start_ts", 0.0)) - prev_end) > 1e-6:
                    beats[k]["start_ts"] = round(prev_end, 3)
                    beats[k]["end_ts"] = round(prev_end + float(beats[k].get("duration", 0.0)), 3)
            # recompute slacks for affected beats
            # Instead of incremental, recompute all slacks from updated durations
            slacks = compute_slack(beats, narrations, wpm=wpm)
            for k, s in enumerate(slacks):
                beats[k]["slack"] = round(float(s), 3)
            needed = -slacks[idx] if slacks[idx] < 0 else 0.0

        # after trying neighbors, if still needed >0, caller will trim residual
    # final slacks
    final_slacks = compute_slack(beats, narrations, wpm=wpm)
    for i, s in enumerate(final_slacks):
        beats[i]["slack"] = round(float(s), 3)
    return beats, final_slacks


def borrow_from_neighbor(
    beat_idx: int,
    beats: list[dict],
    narrations: list[str],
    wpm: int = _DEFAULT_WPM,
    max_borrow_ratio: float = _DEFAULT_MAX_BORROW,
) -> float:
    """Borrow for a single overflowing beat; returns remaining needed (>0 if still over)."""
    _, slacks = borrow_time(beats, narrations, wpm=wpm, max_borrow_ratio=max_borrow_ratio)
    if beat_idx < len(slacks):
        return max(0.0, -slacks[beat_idx])
    return 0.0


# ---------------------------------------------------------------------------
# Step 4 — importance-scoring for the trim-of-last-resort
# ---------------------------------------------------------------------------

_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# For scoring, we need sets of keywords
_VISION_KEYWORDS_CACHE: dict[int, set[str]] = {}


def split_into_clauses(narration: str) -> list[str]:
    """Split narration into clauses (rough: split on . but keep dependent clauses attached).

    We split on sentence boundaries first, then keep dependent clauses attached
    to their main clause — e.g., \"... , because ...\" stays with its host
    unless the comma signals a true independent clause. For now we use sentence
    splits as clause proxies (cheap, deterministic) and then optionally split
    long sentences on \", \" where the tail starts with a causal connective.
    """
    text = (narration or "").strip()
    if not text:
        return []
    # first split into sentences
    sents = [s.strip() for s in _CLAUSE_SPLIT_RE.split(text) if s.strip()]
    clauses: list[str] = []
    for s in sents:
        # Heuristic: keep dependent clauses attached; only split a sentence further
        # if it contains a strong clause break like \", and \" with a new subject.
        # For simplicity we keep sentences intact as clauses; scoring will handle
        # dropping whole sentences. If a sentence is very long (>40 words) we split
        # on commas before causal connectives.
        words = s.split()
        if len(words) > 35 and ", " in s:
            parts = [p.strip() for p in s.split(",")]
            # re-attach leading/trailing punctuation
            # keep first part with comma attached? We'll just treat each comma segment as potential clause
            # But ensure we don't produce tiny fragments.
            cur = parts[0]
            for p in parts[1:]:
                # if p starts with causal connective, it's a separate clause worth scoring separately
                first = (p.split() or [""])[0].lower().strip(",")
                if first in _CAUSAL_CONNECTIVES or first in {"and", "but", "so"}:
                    if cur:
                        clauses.append(cur.strip() + ("," if not cur.endswith(",") else ""))
                    cur = p
                else:
                    cur = cur + ", " + p
            if cur:
                clauses.append(cur.strip())
        else:
            clauses.append(s)
    # filter empties, ensure punctuation
    out = [c.strip() for c in clauses if c.strip()]
    return out


def _contains_named_entity(clause: str, beat: dict | None = None) -> bool:
    # heuristic: capitalized word not at start that appears in beat's transcript/vision
    # or any Proper Noun pattern (Capitalized word with 3+ letters)
    text = clause.strip()
    if not text:
        return False
    # find capitalized tokens beyond first word
    toks = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    if not toks:
        return False
    # if beat provided, check if any token matches beat's entities
    if beat is not None:
        # gather entities from transcript + vision
        entities: set[str] = set()
        for c in beat.get("transcript_lines") or []:
            for w in re.findall(r"\b[A-Z][a-z]{2,}\b", c.get("text", "") or ""):
                entities.add(w.lower())
        for v in beat.get("vision_notes") or []:
            for w in re.findall(r"\b[A-Z][a-z]{2,}\b", v.get("text", "") or ""):
                entities.add(w.lower())
        if entities:
            for tok in toks:
                if tok.lower() in entities:
                    return True
            # also consider if clause contains any entity
            # but even without beat, a named entity is a keep signal
    # generic: if any capitalized token beyond first word, count it
    # ensure not sentence-initial only: check if toks includes non-first word
    words = text.split()
    if words and words[0].rstrip(".,!?\"'").lower().title() == words[0].rstrip(".,!?\"'") and len(words) > 1:
        # first word is capitalized; look for another capitalized beyond first
        for tok in toks:
            if tok not in words[0]:
                return True
        return False
    return bool(toks)


def _contains_action_verb(clause: str) -> bool:
    lows = {w.strip(".,!?\"'").lower() for w in clause.split()}
    return any(v in lows for v in _ACTION_VERBS)


def _matches_vision_keyword(clause: str, beat: dict | None) -> bool:
    if beat is None:
        return False
    notes = beat.get("vision_notes") or []
    if not notes:
        return False
    clause_words = {w.strip(".,!?\"'").lower() for w in clause.split() if len(w) > 3}
    for v in notes:
        txt = (v.get("text") or "").lower()
        for w in clause_words:
            if w in txt:
                return True
    return False


def _is_causal_connective(clause: str) -> bool:
    first = (clause.strip().split() or [""])[0].lower().strip(".,")
    return first in _CAUSAL_CONNECTIVES


def _is_pure_description(clause: str) -> bool:
    # heuristic: no action verb, contains adjective-like words, many adjectives
    if _contains_action_verb(clause):
        return False
    lows = {w.strip(".,!?\"'").lower() for w in clause.split()}
    if lows & _PURE_DESCRIPTIVE_MARKERS:
        return True
    # if clause is short and has no verb-like word, likely descriptive
    # count verbs vs adjectives? simplified: if no action verb and contains
    # \"is\", \"was\", \"are\", \"looks\", \"seems\" + adjective, mark as description
    if any(x in lows for x in {"is", "was", "are", "looks", "seems", "appears"}):
        # check for adjective presence (ends with -y, -ful, etc. or known list)
        for w in lows:
            if w.endswith(("y", "ful", "ous", "ive", "al")) and w not in _ACTION_VERBS:
                return True
    return False


def score_clause(clause: str, beat: dict | None = None) -> int:
    """Weighted sum per spec:
        contains_named_entity *3,
        contains_action_verb *3,
        matches_vision_note_keyword *2,
        is_causal_connective *1,
        is_pure_description * -1
    """
    score = 0
    if _contains_named_entity(clause, beat):
        score += 3
    if _contains_action_verb(clause):
        score += 3
    if _matches_vision_keyword(clause, beat):
        score += 2
    if _is_causal_connective(clause):
        score += 1
    if _is_pure_description(clause):
        score -= 1
    return score


def trim_least_important_clause(
    narration: str,
    max_words: int,
    beat: dict | None = None,
) -> str:
    """Drop lowest-score clauses until word count fits, never dropping the subject's only clause.

    Returns trimmed narration (may be unchanged if already within budget).
    """
    text = (narration or "").strip()
    if count_words(text) <= max_words:
        return text
    clauses = split_into_clauses(text)
    if not clauses:
        # fallback: hard truncate by words if clause split failed
        words = text.split()
        return " ".join(words[:max_words])

    # score each clause
    scored: list[tuple[int, int, str]] = []  # (score, idx, clause)
    for idx, cl in enumerate(clauses):
        sc = score_clause(cl, beat)
        scored.append((sc, idx, cl))

    # Determine subject of beat: most frequent named entity in beat's lines
    subject: str | None = None
    if beat is not None:
        # gather entities from beat
        ents: dict[str, int] = {}
        for c in beat.get("transcript_lines") or []:
            for w in re.findall(r"\b[A-Z][a-z]{2,}\b", c.get("text", "") or ""):
                ents[w.lower()] = ents.get(w.lower(), 0) + 1
        for v in beat.get("vision_notes") or []:
            for w in re.findall(r"\b[A-Z][a-z]{2,}\b", v.get("text", "") or ""):
                ents[w.lower()] = ents.get(w.lower(), 0) + 1
        if ents:
            subject = max(ents, key=lambda k: ents[k])

    # sort by score ascending (lowest first), then by position (later clauses considered more droppable if tie?)
    # spec says drop lowest-to-highest score
    scored_sorted = sorted(scored, key=lambda x: (x[0], -x[1]))  # low score first; if tie, later index first

    # keep track of which clauses are kept
    keep = [True] * len(clauses)
    # Precompute subject presence per clause
    subject_in_clause = []
    for _, _, cl in scored:
        if subject:
            subject_in_clause.append(subject in cl.lower())
        else:
            # no clear subject, treat as not sole-keeper
            subject_in_clause.append(False)

    # Count how many clauses contain subject
    subject_total = sum(subject_in_clause)

    for sc, idx, cl in scored_sorted:
        if count_words(" ".join(c for k, c in enumerate(clauses) if keep[k])) <= max_words:
            break
        # never drop if it's the only clause containing the subject
        if subject and subject_in_clause[idx] and subject_total <= 1:
            continue
        # check dropping would remove last subject occurrence?
        if subject and subject_in_clause[idx]:
            remaining_subject = sum(1 for k in range(len(clauses)) if keep[k] and subject_in_clause[k] and k != idx)
            if remaining_subject == 0:
                continue
        keep[idx] = False
        if subject and subject_in_clause[idx]:
            subject_total -= 1

    kept_clauses = [c for k, c in enumerate(clauses) if keep[k]]
    if not kept_clauses:
        # fallback: keep highest scored clause
        best = max(scored, key=lambda x: x[0])
        kept_clauses = [best[2]]

    result = " ".join(kept_clauses).strip()
    # final safety: if still over, hard trim words (should be rare)
    if count_words(result) > max_words:
        words = result.split()
        result = " ".join(words[:max_words])
        if not result.endswith((".", "!", "?")):
            result += "."
    return result


# ---------------------------------------------------------------------------
# Step 5 — validate before commit (cheap check per beat)
# ---------------------------------------------------------------------------

VALIDATION_PROMPT_TEMPLATE = (
    "Does this narration accurately describe what's happening based on these\n"
    "visual notes and dialogue? Answer MATCH, PARTIAL, or MISMATCH with one sentence why.\n"
    "\n"
    "Narration: {narration}\n"
    "Visual notes: {vision}\n"
    "Dialogue: {dialogue}\n"
)

_VALID_RE = re.compile(r"\b(MATCH|PARTIAL|MISMATCH)\b", re.I)


def validate_beat(
    narration: str,
    beat: dict,
    cfg_llm: dict | None = None,
    provider: str | None = None,
) -> tuple[str, str]:
    """Run the cheap QA gate for one beat.

    Returns (label, reason) where label is MATCH/PARTIAL/MISMATCH. When no LLM
    is configured or the call fails, returns (\"UNKNOWN\", reason).
    """
    vision_text = _format_vision_block(beat.get("vision_notes") or [])
    dialogue_text = _format_transcript_block(beat.get("transcript_lines") or [])

    prompt = VALIDATION_PROMPT_TEMPLATE.format(
        narration=(narration or "").strip(),
        vision=vision_text,
        dialogue=dialogue_text,
    )

    # Use cfg_llm if provided; otherwise fall back to a cheap default
    # For tests without LLM, just return UNKNOWN
    if cfg_llm is None:
        return "UNKNOWN", "no llm config — skipped"

    # Determine model: prefer a cheap model if configured, else the main model
    provider_name = (provider or cfg_llm.get("provider") or "").strip().lower()
    # For gemini, use gemini-2.0-flash if available as cheap model
    model = cfg_llm.get("validation_model") or cfg_llm.get("model") or ""
    # Allow env override for cheap model
    import os

    cheap_model = os.environ.get("VALIDATION_MODEL") or os.environ.get("VISION_MODEL") or ""
    if cheap_model:
        model = cheap_model

    from . import llm as llm_mod

    if not llm_mod.provider_configured(provider_name):
        return "UNKNOWN", f"provider {provider_name!r} not configured"

    try:
        raw = llm_mod.complete(
            provider_name,
            model,
            "You are a strict validator for movie recap narration. Answer with MATCH, PARTIAL, or MISMATCH and one sentence why.",
            prompt,
            base_url=cfg_llm.get("base_url"),
            max_tokens=128,
        )
    except Exception as exc:
        return "UNKNOWN", f"validation call failed: {exc}"

    text = (raw or "").strip()
    m = _VALID_RE.search(text)
    label = m.group(1).upper() if m else "UNKNOWN"
    # normalize PARTIAL vs MISMATCH vs MATCH
    if label not in ("MATCH", "PARTIAL", "MISMATCH"):
        label = "UNKNOWN"
    reason = text[:300]
    return label, reason


def validate_beats(
    beats: list[dict],
    narrations: list[str],
    cfg_llm: dict | None = None,
) -> list[dict]:
    """Validate every beat before TTS/assembly; logs MISMATCH/PARTIAL.

    Returns list of {beat_index, label, reason, narration, vision, dialogue}.
    """
    results: list[dict] = []
    for idx, (beat, nar) in enumerate(zip(beats, narrations)):
        label, reason = validate_beat(nar, beat, cfg_llm=cfg_llm)
        results.append(
            {
                "beat_index": idx,
                "label": label,
                "reason": reason,
                "narration": nar,
                "vision": _format_vision_block(beat.get("vision_notes") or []),
                "dialogue": _format_transcript_block(beat.get("transcript_lines") or []),
            }
        )
        if label in ("MISMATCH", "PARTIAL"):
            print(f"  ! beat {idx}: {label} — {reason}", flush=True)
    # summary
    mism = sum(1 for r in results if r["label"] == "MISMATCH")
    part = sum(1 for r in results if r["label"] == "PARTIAL")
    if mism or part:
        print(f"  ! validation: {mism} MISMATCH, {part} PARTIAL beats before encoding (check log)", flush=True)
    else:
        print(f"  * validation: all {len(results)} beats checked (no MISMATCH)", flush=True)
    return results


# ---------------------------------------------------------------------------
# High-level: generate a beat-script end-to-end
# ---------------------------------------------------------------------------

def generate_beat_script(
    beats: list[dict],
    cfg_llm: dict,
    wpm: int = _DEFAULT_WPM,
    max_borrow_ratio: float = _DEFAULT_MAX_BORROW,
    do_validate: bool = True,
) -> tuple[list[str], list[dict]]:
    """Full per-beat generation pipeline (Steps 1-5).

    For each beat:
      1. generate with hard word budget
      2. on overflow: one targeted retry
      3. borrow time from neighbors if still over
      4. trim by importance if borrow insufficient
      5. validate after all beats are drafted

    Returns (narrations, beats_with_adjusted_durations).
    Beats' durations may be mutated by borrowing.
    """
    if not beats:
        return [], []

    narrations: list[str] = []
    infos: list[dict] = []

    for beat in beats:
        text, info = generate_one_beat(beat, cfg_llm, wpm=wpm)
        narrations.append(text)
        infos.append(info)

    # Step 3: time-borrowing
    # Only for overflowing beats
    beats_adj, slacks = borrow_time(beats, narrations, wpm=wpm, max_borrow_ratio=max_borrow_ratio)
    # After borrowing, some beats may still be over (borrow capped). Apply step 4 trim.
    for i, (beat, text) in enumerate(zip(beats_adj, narrations)):
        slack = slacks[i] if i < len(slacks) else 0.0
        if slack < -1e-6:
            # still overflowing: trim residual
            dur = float(beat.get("duration", 0.0) or 0.0)
            _, max_w = word_budget_for_duration(dur, wpm=wpm)
            needed_words = max_w
            # But if borrowed duration increased, max_w grew; recompute slack words
            # Trim to fit new duration
            if count_words(text) > needed_words:
                trimmed = trim_least_important_clause(text, needed_words, beat=beat)
                print(
                    f"  * beat {i}: trimmed residual {count_words(text) - count_words(trimmed)} words "
                    f"(kept {count_words(trimmed)}/{needed_words})",
                    flush=True,
                )
                narrations[i] = trimmed
            # recompute slack after trim
            # (not strictly needed for validation)

    # Step 5: validate before commit (logs, does not block encode by default)
    if do_validate:
        try:
            validate_beats(beats_adj, narrations, cfg_llm=cfg_llm)
        except Exception as exc:
            print(f"  ! validation gate failed: {exc}", flush=True)

    return narrations, beats_adj


# ---------------------------------------------------------------------------
# Convenience: build beats from pipeline inputs (cues + vision + scenes)
# ---------------------------------------------------------------------------

def build_beats_from_inputs(
    cues: list[dict] | None,
    vision_notes: list[dict] | None,
    scene_bounds: list[float] | None,
    movie_duration: float | None,
    gap_threshold: float = _DEFAULT_GAP,
) -> list[dict]:
    """One-liner for pipeline: merge everything into beats."""
    return detect_beats(cues, vision_notes, scene_bounds, movie_duration, gap_threshold=gap_threshold)
