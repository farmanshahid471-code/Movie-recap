"""Step A — contextual chunking of a long movie transcript.

A 90-minute film produces a transcript far larger than any LLM context window,
so the first pass is broken into **logical, overlapping blocks**: every window
is ``window_seconds`` long (default 5 minutes) and consecutive windows overlap
by ``overlap_seconds`` (default 30s). The overlap carries the plot across the
seams so the model never loses track of what happened right before the cut.

Windows are *contextual* rather than blindly fixed: a cue that straddles a
window boundary is kept whole in both neighbours (it belongs to both), so no
sentence of dialogue is ever split or dropped.

    cues: [{"text": "...", "start": 0.0, "end": 2.5}, ...]   (seconds)
    chunks: [{"index": 0, "start": 0.0, "end": 300.0,
              "cues": [...], "text": "[00:00] ...\\n[00:02] ..."}, ...]
"""
from __future__ import annotations

from .dialogue import _fmt, to_transcript_text


def chunk_cues(
    cues: list[dict],
    window_seconds: float = 300.0,
    overlap_seconds: float = 30.0,
) -> list[dict]:
    """Group timed cues into overlapping blocks of ``window_seconds``.

    Windows advance by ``window_seconds - overlap_seconds``. A cue belongs to a
    window when it overlaps that window's time range (kept whole — never cut in
    half), which is what makes the chunking contextual instead of mechanical.
    """
    if not cues:
        return []
    window = float(window_seconds)
    step = max(window - float(overlap_seconds), 1.0)

    # Normalise cue keys and coerce to floats once.
    norm = []
    for c in cues:
        start = float(c.get("start", 0.0) or 0.0)
        end = float(c.get("end", start) or start)
        if start < 0:
            start, end = 0.0, max(end - start, 0.0)
        norm.append({**c, "start": start, "end": max(end, start)})

    first = min(c["start"] for c in norm)
    last = max(c["end"] for c in norm)

    chunks: list[dict] = []
    lo = first
    idx = 0
    while lo < last:
        hi = lo + window
        inside = [
            c for c in norm
            if c["end"] > lo and c["start"] < hi
        ]
        if inside:
            chunks.append(
                {
                    "index": idx,
                    "start": lo,
                    "end": hi,
                    "cues": inside,
                    "text": to_transcript_text(inside),
                }
            )
            idx += 1
        lo += step
        # safety valve for degenerate inputs (e.g. all cues identical time)
        if idx > len(norm) * 4 + 64:
            break
    return chunks


def chunk_summary_budget(chunks: list[dict], per_chunk_chars: int = 4500) -> int:
    """Rough character budget the summarizer should stay under per chunk.

    ``per_chunk_chars`` keeps each individual summarization call small enough
    for small local models while still capturing the whole action of a block.
    """
    return max(800, min(int(per_chunk_chars), 8000))
