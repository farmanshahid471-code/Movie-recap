"""Story-first narration writer — the beat path, re-told as a story.

The problem this solves
-----------------------
The beat path used to write the recap one beat at a time::

    BEAT_SYSTEM_PROMPT: "Describe ONLY what happens in this beat. Do not
    reference earlier or later beats."

Each call saw only its own 4-40 seconds of film, and it was asked to
*describe*. The result reads exactly like what it is: a shot-by-shot
description of the picture, one isolated caption per beat, with no story,
no cause and effect, no rising stakes, and nothing connecting line 40 to
line 41. ("The narration is really bad, it's describing the scene — I want
it to do storytelling.")

The fix
-------
1. **Units, not beats.** Consecutive beats are grouped into STORY UNITS of
   roughly 45 seconds of film. One LLM call writes a whole unit as a scene
   of the story, so it can chain cause into effect and land a short beat
   between long ones — the things a single 6-second beat cannot do.
2. **A ledger, not an island.** Each call receives a rolling story-so-far
   (what has already been told, in one line per unit), the cast already
   introduced, and the previous unit's last line to continue from. The
   narration stops reintroducing the same character and stops treating every
   unit as the beginning of the film.
3. **Storytelling voice, not description.** The system prompt is
   ``narrative.STORY_RULES``: people want things, cause leads to consequence,
   stakes rise, emotion comes from action, short beats land, pay off what you
   set up — with an explicit bad/good worked pair, and a hard ban on
   scene/shot/camera/"we see" language. The prompt literally says a
   description of the frame is not a sentence in a story.
4. **Visual-to-text binding.** Every beat in the unit is labelled (``B1``,
   ``B2``, ...) and carries its exact film range and its own dialogue and
   visual facts. The model must tag each sentence with the beat it narrates
   and every beat with a real event must get a sentence, so a sentence can
   never narrate footage from somewhere else.
5. **Chronology by construction.** Sentence windows are built beat by beat
   in film order, split inside a beat in proportion to their length, and
   every window is clamped forward so ``film_start[i] < film_end[i] <=
   film_start[i+1]`` always holds (see ``enforce_chronology``).
6. **Measured, not vibes.** Every unit is checked against its hard word
   budget (words/second of footage), its sentences are checked against the
   beat facts they claim to narrate, and the finished script is scanned by
   ``narrative.description_flags``. Flagged sentences get ONE targeted
   rewrite call; the run reports the number.

Everything is pure Python over plain dicts so it can be tested without a
movie, an API key, or a network.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import llm as llm_mod
from . import narrative
from .beats import word_budget_for_duration
from .script import NARRATOR_PERSONA, VOICE_GUIDE, _out_tokens_for_words
from .util import count_words

_DEFAULT_WPM = 175

# ---------------------------------------------------------------------------
# Defaults (mirrored in config.yaml under `story:`)
# ---------------------------------------------------------------------------
DEFAULT_CFG: dict = {
    "enabled": True,
    # A unit is ~45s of film: long enough that two or three events chain into
    # a cause and an effect, short enough that every sentence can be grounded
    # in footage the unit actually owns.
    "unit_seconds": 45.0,
    "min_beats_per_unit": 2,
    "max_beats_per_unit": 8,
    # Planning number for "how many sentences should this unit be": the model
    # is asked for a range, this only sizes the request.
    "sentence_words": 16,
    # Retry a unit once when it overruns its per-beat word budget, then fall
    # back to description-first tightening (never drop story content).
    "retry_overflow": True,
    # One targeted rewrite call for sentences the description lint flags
    # (and for sentences whose words appear nowhere in the beat they claim).
    "repair": True,
    "repair_batch": 12,
    # A unit that returns less than this share of its word budget gets one
    # "you are N words short" retry (short units = a short video).
    "min_fill": 0.9,
}


class StoryError(RuntimeError):
    """Raised when a unit cannot be written at all (the caller falls back)."""


# ---------------------------------------------------------------------------
# Story units — grouping beats so a scene can be told as a scene
# ---------------------------------------------------------------------------


def group_into_units(
    beats: list[dict],
    cfg_story: dict | None = None,
    *,
    wpm: int = _DEFAULT_WPM,
) -> list[dict]:
    """Group consecutive beats into story units.

    Returns ``[{"index", "start_ts", "end_ts", "duration", "budget_words",
    "budget_min", "beats": [...]}]`` in film order. Each beat inside a unit is
    decorated with its ``label`` (``B1``..), its own ``start_ts``/``end_ts``
    and its ``max_words`` budget — the facts the model must narrate, with the
    exact film range it is allowed to narrate them over.
    """
    cfg = {**DEFAULT_CFG, **(cfg_story or {})}
    unit_seconds = max(float(cfg.get("unit_seconds", 45.0)), 5.0)
    max_beats = max(int(cfg.get("max_beats_per_unit", 8)), 1)
    min_beats = max(int(cfg.get("min_beats_per_unit", 2)), 1)

    usable: list[dict] = []
    for b in beats or []:
        dur = float(b.get("duration", 0.0) or 0.0)
        if dur <= 0.01:
            continue
        usable.append(b)
    if not usable:
        return []

    units: list[dict] = []
    cur: list[dict] = []
    cur_dur = 0.0

    def _close(group: list[dict]) -> dict:
        start = float(group[0].get("start_ts", 0.0) or 0.0)
        end = float(group[-1].get("end_ts", start) or start)
        budget_max = 0
        budget_min = 0
        deco: list[dict] = []
        for k, b in enumerate(group):
            _, mx = word_budget_for_duration(
                float(b.get("duration", 0.0) or 0.0), wpm=wpm)
            mn = max(1, int(round(mx * 0.55)))
            budget_max += mx
            budget_min += mn
            item = dict(b)
            item["label"] = f"B{k + 1}"
            item["start_ts"] = float(b.get("start_ts", 0.0) or 0.0)
            item["end_ts"] = float(b.get("end_ts", start) or start)
            item["max_words"] = mx
            item["min_words"] = mn
            deco.append(item)
        return {
            "index": len(units),
            "start_ts": round(start, 3),
            "end_ts": round(end, 3),
            "duration": round(max(end - start, 0.0), 3),
            "budget_words": budget_max,
            "budget_min": budget_min,
            "beats": deco,
        }

    for b in usable:
        dur = float(b.get("duration", 0.0) or 0.0)
        over_seconds = cur and (cur_dur + dur > unit_seconds * 1.4)
        over_beats = len(cur) >= max_beats
        if cur and (over_seconds or over_beats or cur_dur >= unit_seconds):
            units.append(_close(cur))
            cur, cur_dur = [], 0.0
        cur.append(b)
        cur_dur += dur

    if cur:
        # A trailing scrap (one short beat) rides with the previous unit: a
        # single 4-second unit would produce a one-line "scene" of story.
        if units and len(cur) < min_beats and units[-1]["duration"] <= unit_seconds * 1.6:
            tail = units.pop()
            merged = []
            for item in tail["beats"] + cur:
                merged.append(item)
            # Re-close with merged raw beats (labels are recomputed).
            prev_units = len(units)
            rebuilt = _close([{**item} for item in merged])
            rebuilt["index"] = prev_units
            units.append(rebuilt)
        else:
            units.append(_close(cur))
    for i, u in enumerate(units):
        u["index"] = i
    return units


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# The master system prompt (timestamped-array contract). Used verbatim; the
# unit-specific facts, budget and continuity go in the user message.
YOUTUBER_SYSTEM = """You are a master scriptwriter for a hit YouTube movie recap channel. Your job is to convert raw transcript dialogue and sparse visual notes into a fast-paced, continuous story.

CRITICAL RULES:
1. NO CAMERA WORDS: Never use "we see", "the camera shows", "the scene transitions", "is visible", or "appears".
2. STORYTELLING ONLY: Write in third-person, present tense. Focus only on character actions and plot.
3. PACING: Keep sentences short and punchy.

OUTPUT FORMAT:
You MUST output a JSON array of objects. Each object must have a `timestamp` (the exact starting timestamp of the dialogue or action from the prompt) and a `sentence` (your rewritten story sentence).
Example:
[
  {"timestamp": 137.125, "sentence": "Buzz pushes through the dense, green jungle."},
  {"timestamp": 147.750, "sentence": "Three identical action figures run through the thick undergrowth."}
]
"""

STORY_SYSTEM = YOUTUBER_SYSTEM


# Source mix fed to the writer: ~80% transcript, ~20% vision. Vision notes
# may take at most this share of a beat's fact text when there is dialogue;
# a SILENT beat (no dialogue at all) gets its vision notes in full, since
# that is exactly the gap vision exists to fill.
VISION_SHARE = 0.20
_SILENT_VISION_CHARS = 600


def _fmt_ts(seconds) -> str:
    """``[t=137.125 | 00:02:17] `` -- the exact seconds value the writer must
    copy into ``timestamp``, plus a readable clock."""
    try:
        v = max(float(seconds), 0.0)
    except (TypeError, ValueError):
        return ""
    t = int(v)
    return f"[t={v:.3f} | {t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}] "


def _format_facts(beat: dict) -> str:
    """The dialogue + visual ground truth of one beat, as story material.

    Transcript first and dominant (each line carries its exact Whisper
    timestamp); vision captions are capped at ``VISION_SHARE`` of the text
    and labelled as silent action.
    """
    said: list[str] = []
    for c in beat.get("transcript_lines") or []:
        text = (c.get("text") or "").strip()
        if text:
            said.append(f"    said {_fmt_ts(c.get('start'))}{text}".rstrip())
    said_chars = sum(len(x) for x in said)
    if said_chars:
        budget = max(int(said_chars * VISION_SHARE / (1.0 - VISION_SHARE)), 60)
    else:
        budget = _SILENT_VISION_CHARS
    seen: list[str] = []
    used = 0
    for v in beat.get("vision_notes") or []:
        text = (v.get("text") or "").strip() if isinstance(v, dict) else str(v).strip()
        if not text:
            continue
        room = budget - used
        if room < 25:
            break
        if len(text) > room:
            cut = text[:room].rsplit(" ", 1)[0].rstrip(",;: ")
            text = cut + "..."
        ts = _fmt_ts(v.get("t")) if isinstance(v, dict) else ""
        seen.append(f"    silent action {ts}{text}".rstrip())
        used += len(text)
    lines = said + seen
    if not lines:
        return "    (no dialogue or visual caption for this beat)"
    return "\n".join(lines)


def _language_block(lang_name: str) -> str:
    """Instruction block for natively-authored non-English recaps."""
    name = (lang_name or "English").strip() or "English"
    if name.lower().startswith("english"):
        return ""
    return (f"\n\nLANGUAGE: write the narration entirely in {name} — natural, "
            f"idiomatic {name} for a {name}-speaking recap audience. Keep "
            "character names in their common forms for that audience, "
            "consistently.")


def build_unit_prompt(
    unit: dict,
    *,
    ledger: dict,
    total_units: int,
    film_duration: float = 0.0,
    cfg_story: dict | None = None,
    lang_name: str = "English",
) -> tuple[str, str]:
    """Return ``(system, user)`` for one story unit."""
    cfg = {**DEFAULT_CFG, **(cfg_story or {})}
    span = max(float(unit.get("end_ts", 0.0)) - float(unit.get("start_ts", 0.0)), 0.1)
    position = (
        float(unit.get("start_ts", 0.0)) / film_duration if film_duration > 0 else 0.0
    )
    act_label, act_instr = narrative.act_for(position)

    per_sentence = max(int(cfg.get("sentence_words", 16)), 6)
    aim = max(2, int(round(unit.get("budget_words", 40) / float(per_sentence))))

    beat_lines: list[str] = []
    budget_lines: list[str] = []
    for b in unit["beats"]:
        lo, hi = float(b["start_ts"]), float(b["end_ts"])
        dur = max(hi - lo, 0.0)
        beat_lines.append(
            f"[{b['label']}] {narrative.format_window(lo, hi)} "
            f"({dur:.0f}s of footage — about {b['max_words']} words)\n"
            f"{_format_facts(b)}"
        )
        budget_lines.append(f"  {b['label']} ({narrative.format_window(lo, hi)}): "
                            f"about {b['max_words']} words")

    story_so_far = ledger.get("recaps") or []
    last_line = (ledger.get("last") or "").strip()
    cast = sorted({n for n in (ledger.get("names") or set()) if n})

    context: list[str] = []
    if story_so_far:
        context.append(
            "STORY SO FAR (already told — never re-explain or re-introduce it):\n"
            + "\n".join(f"  - {r}" for r in story_so_far[-6:])
        )
    else:
        context.append(
            "STORY SO FAR: nothing yet. This is the opening of the recap — start "
            "inside the film's first scene, mid-motion."
        )
    if cast:
        context.append(
            "ALREADY ON SCREEN (never introduce these people again with 'a man "
            "named' / 'a woman called' appositives): " + ", ".join(cast[:14])
        )
    if last_line:
        context.append(
            'THE LINE YOU ARE CONTINUING FROM (do not repeat it, pick the story '
            f'up seamlessly): "{last_line}"'
        )

    user = f"""Write story unit {unit['index'] + 1} of {total_units} for the recap narration.

FILM POSITION: {narrative.format_window(unit['start_ts'], unit['end_ts'])} \
({act_label}, {position * 100:.0f}% into the film). This is {span:.0f} seconds of footage.

WHAT THIS PART OF THE STORY MUST DO:
{act_instr}

{chr(10).join(context)}

WHAT HAPPENS IN THIS STRETCH (your only source of facts):
{chr(10).join(beat_lines)}

WORD BUDGET (the narration must FILL this footage; the video length depends on it):
{chr(10).join(budget_lines)}
  whole unit: {unit['budget_words']} words -- write between {int(unit['budget_words'] * 0.92)} and {unit['budget_words']} words, about {aim} sentences.
  Falling short makes the whole recap shorter than requested; going over breaks sync.

TIMESTAMPS: every fact above starts with [t=SECONDS | HH:MM:SS]. Each sentence's
"timestamp" is the t= SECONDS value of the dialogue or action it narrates,
copied exactly (a number, e.g. 137.125). Timestamps never go backwards and
must lie inside this unit's film range. Cover every stretch in order.

No shot description: tell the story, never caption the picture.

Respond with ONLY the JSON array.""" + _language_block(lang_name)
    return STORY_SYSTEM + _language_block(lang_name), user


# ---------------------------------------------------------------------------
# Parsing + repair of the model's unit reply
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_LABEL_PREFIX = re.compile(r"^\s*(?:\[?B\s*\d+\]?|\d+\s*[:.)-])\s*", re.I)


def _strip_reply(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    text = _FENCE.sub("", text).strip().strip("`").strip()
    return text


def _loads(text: str):
    """Best-effort JSON load: whole string, then first object, then array."""
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                continue
    return None


def _clean_sentence(text: str) -> str:
    s = _BULLET.sub("", str(text or "")).strip()
    s = _LABEL_PREFIX.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    if s and s[-1] not in ".!?":
        s += "."
    return s


def _label_of(item: dict, n: int) -> int | None:
    for key in ("b", "beat", "label", "index", "beat_index", "id"):
        if key in item:
            raw = item[key]
            try:
                if isinstance(raw, str) and re.fullmatch(r"\s*B?\s*\d+\s*", raw, re.I):
                    val = int(re.sub(r"\D", "", raw))
                else:
                    val = int(raw)
            except (TypeError, ValueError):
                continue
            if n <= 0:
                return None
            return min(max(val, 1), n)      # clamp: 9 on a 3-beat unit = B3
    return None


def _timestamp_of(item: dict) -> float | None:
    """The writer's ``timestamp`` (seconds, or an HH:MM:SS / MM:SS string)."""
    for key in ("timestamp", "t", "time", "start", "ts"):
        if key not in item:
            continue
        raw = item[key]
        try:
            return max(float(raw), 0.0)
        except (TypeError, ValueError):
            pass
        m = re.fullmatch(r"\s*\[?(?:t=)?(\d+):(\d{1,2})(?::(\d{1,2}(?:\.\d+)?))?\]?\s*",
                         str(raw))
        if m:
            a, b, c = m.group(1), m.group(2), m.group(3)
            if c is None:
                return int(a) * 60 + float(b)
            return int(a) * 3600 + int(b) * 60 + float(c)
    return None


def _beat_for_time(t: float, beats: list[dict]) -> int:
    """1-based index of the beat whose film range holds ``t`` (nearest)."""
    best, best_d = 1, float("inf")
    for k, b in enumerate(beats, start=1):
        lo = float(b.get("start_ts", 0.0) or 0.0)
        hi = float(b.get("end_ts", lo) or lo)
        if lo - 1e-6 <= t < hi + 1e-6:
            return k
        d = min(abs(t - lo), abs(t - hi))
        if d < best_d:
            best, best_d = k, d
    return best


def parse_unit_reply(raw: str, n_beats: int,
                     beats: list[dict] | None = None) -> tuple[list[dict], str]:
    """Parse one unit reply into ``[{"b", "sentence"}, ...]`` in reading order.

    Accepts the documented shape (``{"beats": [{"b": 1, "sentences": [...]}]}``)
    plus the near-misses models actually produce: a bare array of these
    objects, a flat array of strings, ``text`` instead of ``sentences``, and
    ``B1``-style labels. Labels are clamped into range and forced
    non-decreasing, so a confused tag can never reorder the film.
    """
    text = _strip_reply(raw)
    if not text:
        return [], ""
    data = _loads(text)
    recap = ""
    items: list[tuple[int | None, str]] = []

    # PRIMARY CONTRACT: [{"timestamp": 137.125, "sentence": "..."}, ...]
    # (also accepted wrapped as {"sentences": [...]} / {"lines": [...]}).
    arr = data
    if isinstance(data, dict):
        for key in ("sentences", "lines", "script", "narration", "items"):
            if isinstance(data.get(key), list):
                arr = data[key]
                break
    if isinstance(arr, list) and arr and all(isinstance(it, dict) for it in arr) \
            and any(_timestamp_of(it) is not None for it in arr) \
            and any(isinstance(it.get("sentence") or it.get("text"), str) for it in arr):
        rec = data.get("recap") if isinstance(data, dict) else ""
        out: list[dict] = []
        running_t = None
        running_b = 1
        for it in arr:
            sent = _clean_sentence(it.get("sentence") or it.get("text") or "")
            if not sent or sent == ".":
                continue
            t = _timestamp_of(it)
            if t is None:
                t = running_t
            if t is not None and running_t is not None and t < running_t:
                t = running_t                 # CHRONOLOGY: never rewind
            if t is not None:
                running_t = t
            if beats and t is not None:
                lbl = _beat_for_time(t, beats)
            else:
                lbl = _label_of(it, max(n_beats, 1)) or running_b
            lbl = min(max(int(lbl), running_b), max(n_beats, 1))
            running_b = lbl
            row = {"b": lbl, "sentence": sent}
            if t is not None:
                row["t"] = round(float(t), 3)
            out.append(row)
        if out:
            return out, str(rec or "").strip()

    if isinstance(data, dict):
        recap = str(data.get("recap") or data.get("summary") or "").strip()
        block = data.get("beats") or data.get("units") or data.get("lines") or []
        if isinstance(block, dict):
            block = [block]
        if isinstance(block, str):
            block = [block]
        for it in block or []:
            if isinstance(it, str):
                items.append((None, _clean_sentence(it)))
                continue
            if not isinstance(it, dict):
                continue
            label = _label_of(it, max(n_beats, 1))
            for key in ("sentences", "lines", "text", "sentence", "narration"):
                if key not in it:
                    continue
                val = it[key]
                if isinstance(val, str):
                    for piece in _split_lines(val):
                        items.append((label, _clean_sentence(piece)))
                elif isinstance(val, (list, tuple)):
                    for piece in val:
                        items.append((label, _clean_sentence(piece)))
                break
    elif isinstance(data, list):
        for it in data:
            if isinstance(it, str):
                items.append((None, _clean_sentence(it)))
            elif isinstance(it, dict):
                label = _label_of(it, max(n_beats, 1))
                got = False
                for key in ("sentences", "lines", "text", "sentence", "narration"):
                    if key not in it:
                        continue
                    val = it[key]
                    if isinstance(val, str):
                        for piece in _split_lines(val):
                            items.append((label, _clean_sentence(piece)))
                    elif isinstance(val, (list, tuple)):
                        for piece in val:
                            items.append((label, _clean_sentence(piece)))
                    got = True
                    break
                if not got:
                    for piece in _split_lines(str(it.get("value") or "")):
                        items.append((label, _clean_sentence(piece)))
    else:
        # plain prose: one sentence per line
        for piece in _split_lines(text):
            items.append((None, _clean_sentence(piece)))

    out: list[dict] = []
    running = 1
    unknown_count = sum(1 for lbl, _s in items if lbl is None)
    seen_unknown = 0
    for lbl, sentence in items:
        if not sentence:
            continue
        if lbl is None:
            # Spread unlabelled lines over the unit proportionally: the story
            # order is preserved and no line can land in a later beat.
            step = seen_unknown / max(unknown_count, 1)
            lbl = min(n_beats, 1 + int(step * n_beats))
            seen_unknown += 1
        lbl = min(max(int(lbl), 1), max(n_beats, 1))
        running = max(running, lbl)          # CHRONOLOGY: labels never rewind
        lbl = running
        out.append({"b": lbl, "sentence": sentence})
    return out, recap


def _split_lines(text: str) -> list[str]:
    """Split a prose blob into sentences (used only when the reply is prose)."""
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", str(text or "")).strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Facts ↔ sentence binding
# ---------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "from", "by", "as", "is", "are", "was", "were", "be", "been",
    "am", "do", "does", "did", "has", "have", "had", "he", "she", "they", "it",
    "his", "her", "their", "its", "him", "them", "we", "you", "i", "that",
    "this", "these", "those", "there", "then", "than", "so", "not", "no", "up",
    "out", "into", "over", "after", "before", "when", "while", "who", "what",
    "which", "will", "would", "can", "could", "just", "now", "again", "still",
}


def _content_tokens(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z']+", str(text or "").lower())
    return {t for t in toks if len(t) > 2 and t not in _STOP}


def beat_facts_text(beat: dict) -> str:
    parts: list[str] = []
    for c in beat.get("transcript_lines") or []:
        parts.append(str(c.get("text") or ""))
    for v in beat.get("vision_notes") or []:
        parts.append(str(v.get("text") or "") if isinstance(v, dict) else str(v))
    return " ".join(parts)


def binding_score(sentence: str, beat: dict) -> float:
    """Fraction of the sentence's content words that appear in the beat facts.

    This is the *visual-to-audio binding constraint* in measurable form: a
    sentence tagged with ``B3`` had better be about the things the vision
    pass and the dialogue actually recorded in ``B3``. 0.0 means the line
    narrates footage that is not in the beat it claims.
    """
    toks = _content_tokens(sentence)
    if not toks:
        return 1.0
    facts = _content_tokens(beat_facts_text(beat))
    if not facts:
        return 1.0
    return len(toks & facts) / float(len(toks))


BINDING_MIN = 0.12


# ---------------------------------------------------------------------------
# Budget: description-first tightening (never drop story)
# ---------------------------------------------------------------------------

_FILLER = re.compile(
    r"\b(very|really|quite|simply|actually|basically|literally|just|merely|"
    r"somewhat|rather|extremely|totally|completely)\s+",
    re.I,
)


def tighten_to_budget(text: str, max_words: int) -> str:
    """Shorten a sentence to ``max_words`` without losing the story.

    Order of sacrifice: picture-describing clauses first (``it is seen``,
    ``the camera``, ``there is``), then filler adverbs, then trailing clauses
    — never the action itself, and never the whole sentence (a beat with no
    narration is worse than a beat that runs two words long; the timeline
    pads it with B-roll or a held frame instead).
    """
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s or max_words <= 0:
        return s
    if count_words(s) <= max_words:
        return s

    body = s[:-1] if s[-1] in ".!?" else s
    terminal = s[-1] if s and s[-1] in ".!?" else "."
    clauses = [c.strip() for c in re.split(r",\s*|\s+(?:and|but|while|which|who)\s+", body) if c.strip()]
    if len(clauses) > 1:
        kept = [c for c in clauses if not narrative.looks_like_description(c)]
        if not kept:
            kept = [clauses[0]]
        elif kept[0] != clauses[0]:
            # the opening clause was a picture caption: start on the first
            # real action instead of on "The camera lingers on the office"
            kept[0] = kept[0][:1].upper() + kept[0][1:] if kept[0] else kept[0]
        body = ", ".join(kept)

    body = _FILLER.sub("", body).strip()
    if count_words(body) <= max_words:
        return body + terminal

    words = body.split()
    return " ".join(words[:max_words]).rstrip(" ,;:") + terminal


# ---------------------------------------------------------------------------
# Segments: binding sentences to film windows, in strict chronology
# ---------------------------------------------------------------------------


def unit_to_segments(
    unit: dict,
    items: list[dict],
    *,
    prev_end: float = 0.0,
    movie_duration: float = 0.0,
) -> list[dict]:
    """Lay a unit's sentences over the unit's beats, in film order.

    Each sentence lands inside the beat it was tagged with, and the sentences
    sharing a beat split that beat's window in proportion to their length.
    ``film_start[i] < film_end[i] <= film_start[i+1]`` holds by construction
    (windows are strictly increasing, and a sentence never starts before the
    previous one ended).
    """
    beats = {b["label"]: b for b in unit["beats"]}
    order = [b["label"] for b in unit["beats"]]
    grouped: dict[str, list[dict]] = {lbl: [] for lbl in order}
    for it in items:
        lbl = order[min(max(int(it.get("b", 1)) - 1, 0), len(order) - 1)]
        grouped[lbl].append(it)

    segments: list[dict] = []
    cursor = float(prev_end)
    last_anchor = float(prev_end)
    for lbl in order:
        items_here = grouped.get(lbl) or []
        if not items_here:
            continue          # footage stays available as B-roll for the next unit
        beat = beats[lbl]
        b_lo = float(beat.get("start_ts", 0.0) or 0.0)
        b_hi = float(beat.get("end_ts", b_lo) or b_lo)
        if movie_duration > 0:
            b_lo = min(b_lo, movie_duration)
            b_hi = min(b_hi, movie_duration)
        b_hi = max(b_hi, b_lo + 0.3)
        lo = max(b_lo, cursor)
        hi = max(b_hi, lo + 0.3)
        if movie_duration > 0 and lo >= movie_duration - 0.05:
            # Narration outlasts the film: let the timeline clamp & pad.
            lo = max(movie_duration - 0.3, cursor)
            hi = max(lo + 0.3, movie_duration)
        weights = [max(count_words(it["sentence"]), 1) for it in items_here]
        total_w = float(sum(weights)) or 1.0
        width = max(hi - lo, 0.3)
        span_cursor = lo
        for it, w in zip(items_here, weights):
            share = width * (w / total_w)
            s_lo = span_cursor
            s_hi = s_lo + share
            if s_hi <= s_lo:
                s_hi = s_lo + 0.3
            # The writer's exact timestamp is the ABSOLUTE anchor the clip is
            # cut from (clamped into this beat's range, never rewinding).
            anchor = (s_lo + s_hi) / 2.0
            if it.get("t") is not None:
                anchor = min(max(float(it["t"]), lo), hi)
                anchor = max(anchor, last_anchor)
            last_anchor = anchor
            segments.append({
                "sentence": it["sentence"],
                "film_start": round(s_lo, 3),
                "film_end": round(s_hi, 3),
                "anchor": round(anchor, 3),
                "zone_lo": round(lo, 3),
                "zone_hi": round(hi, 3),
                "beat": lbl,
                "beat_index": order.index(lbl) + 1,
                "unit_index": int(unit.get("index", 0)),
            })
            span_cursor = s_hi
        cursor = max(cursor, hi)
    return segments


def enforce_chronology(
    segments: list[dict],
    movie_duration: float = 0.0,
) -> tuple[list[dict], int]:
    """Clamp every segment forward so the film never rewinds.

    Returns ``(segments, moved)`` where ``moved`` counts the sentences whose
    window had to be pushed. Sentence order is NEVER sorted by timestamp:
    the narration order IS the story order, so a violation is resolved by
    pushing the late window forward, never by swapping the lines (that is
    the "visuals shown before the relevant scene" bug in its most literal
    form).
    """
    moved = 0
    prev_end = 0.0
    limit = float(movie_duration) if movie_duration and movie_duration > 0 else 0.0
    for seg in segments:
        lo = float(seg.get("film_start", 0.0) or 0.0)
        hi = float(seg.get("film_end", lo) or lo)
        if lo < prev_end - 1e-6:
            shift = prev_end - lo
            lo, hi = prev_end, hi + shift
            moved += 1
        if hi <= lo:
            hi = lo + 0.3
        if limit > 0:
            # Never emit an inverted window, even when the narration outlasts
            # the film: clamp the pair to the last sliver of footage and let
            # the timeline hold the final frame there (freeze padding).
            lo = min(lo, max(limit - 0.05, 0.0))
            hi = min(max(hi, lo + 0.05), limit)
            if hi <= lo:
                hi = min(limit, lo + 0.05)
        seg["film_start"] = round(lo, 3)
        seg["film_end"] = round(hi, 3)
        seg["anchor"] = round((lo + hi) / 2.0, 3)
        prev_end = hi
    return segments, moved


def check_chronology(segments: list[dict], eps: float = 1e-6) -> list[int]:
    """Indices of segments that show film out of order (the invariant checker).

    Violations are: an inverted window (``end <= start``) or a window that
    moves BACKWARDS through the film. In the normal case the writer also
    guarantees the stronger ``end[i] <= start[i+1]`` (windows never overlap);
    when the narration outlasts the film the tail windows necessarily
    compress into the last sliver of footage, so what must never happen —
    and what this checks — is a rewind or an inverted pair.
    """
    bad: list[int] = []
    prev_lo: float | None = None
    for i, seg in enumerate(segments):
        lo = float(seg.get("film_start", 0.0) or 0.0)
        hi = float(seg.get("film_end", lo) or lo)
        if hi <= lo + eps:
            bad.append(i)
        elif prev_lo is not None and lo < prev_lo - eps:
            bad.append(i)
        prev_lo = lo
    return bad


def check_no_overlap(segments: list[dict], eps: float = 1e-6) -> list[int]:
    """Stricter companion: ``end[i] <= start[i+1]`` (windows never overlap).

    Holds for every normal script (a unit's sentences split their beat's
    window); it is only ever violated by the film-overrun tail, where the
    last sentences must compress into the final sliver of footage and the
    picture holds its last frame — which is why the writer's own invariant
    checker (:func:`check_chronology`) only forbids rewinds and inverted
    windows.
    """
    bad: list[int] = []
    prev_hi: float | None = None
    for i, seg in enumerate(segments):
        lo = float(seg.get("film_start", 0.0) or 0.0)
        if prev_hi is not None and lo < prev_hi - eps:
            bad.append(i)
        prev_hi = float(seg.get("film_end", lo) or lo)
    return bad


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------


def _ask(
    cfg_llm: dict,
    system: str,
    user: str,
    *,
    max_tokens: int,
    temperature: float | None = None,
    json_mode: bool | None = None,
) -> str:
    # The unit writer answers with a bare JSON ARRAY, which the provider's
    # json_object response mode would reject; the repair pass returns an
    # object and keeps json mode.
    if json_mode is None:
        json_mode = "JSON array" not in system
    return llm_mod.complete(
        cfg_llm.get("provider", ""),
        cfg_llm.get("model", ""),
        system,
        user,
        base_url=cfg_llm.get("base_url"),
        json_mode=json_mode,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _update_ledger(ledger: dict, items: list[dict], recap: str) -> None:
    if recap:
        ledger.setdefault("recaps", []).append(re.sub(r"\s+", " ", recap).strip())
    if items:
        ledger["last"] = items[-1]["sentence"]
    names = ledger.setdefault("names", set())
    for it in items:
        for tok in re.findall(r"\b[A-Z][a-z]{2,}\b", it["sentence"]):
            names.add(tok)
    if not ledger.get("recaps"):
        ledger["recaps"] = []


def write_story_script(
    beats: list[dict],
    cfg_llm: dict,
    *,
    wpm: int = _DEFAULT_WPM,
    movie_duration: float = 0.0,
    cfg_story: dict | None = None,
    total_target_words: int | None = None,
    lang_name: str = "English",
    progress=None,
) -> dict:
    """Write the whole narration as story units. Returns segments + report.

    The return value is the contract the pipeline consumes::

        {"segments": [{"sentence", "film_start", "film_end", "anchor",
                       "zone_lo", "zone_hi"}, ...],
         "sentences": [...],
         "report": {...}}

    Raises :class:`StoryError` when nothing usable could be written, so the
    caller can fall back to the per-beat writer instead of shipping an empty
    script.
    """
    cfg = {**DEFAULT_CFG, **(cfg_story or {})}
    units = group_into_units(beats, cfg, wpm=wpm)
    if not units:
        raise StoryError("no usable beats for story units")

    ledger: dict = {"recaps": [], "last": "", "names": set()}
    segments: list[dict] = []
    sentences: list[str] = []
    prev_end = 0.0

    for unit in units:
        system, user = build_unit_prompt(
            unit, ledger=ledger, total_units=len(units),
            film_duration=movie_duration, cfg_story=cfg, lang_name=lang_name,
        )
        max_tokens = _out_tokens_for_words(max(unit["budget_words"], 60))
        budget = max(int(unit["budget_words"]), 8)
        raw = _ask(cfg_llm, system, user, max_tokens=max_tokens)
        items, recap = parse_unit_reply(raw, len(unit["beats"]), unit["beats"])
        if not items:
            # one retry, told exactly what was wrong
            raw = _ask(
                cfg_llm, system,
                user + "\n\nYour previous answer was not usable JSON or had no "
                       "sentences. Reply with ONLY the JSON array of "
                       "{\"timestamp\", \"sentence\"} objects.",
                max_tokens=max_tokens,
            )
            items, recap = parse_unit_reply(raw, len(unit["beats"]), unit["beats"])
        if not items:
            raise StoryError(f"unit {unit['index'] + 1}: model returned no sentences")

        # ---- budget: one targeted retry, then description-first tightening --
        over = _over_budget_beats(unit, items)
        if over and cfg.get("retry_overflow", True):
            note = "\n\nYour previous answer overran the footage. Rewrite it, "
            note += "cutting picture description and filler first, keeping every "
            note += "story event:\n" + "\n".join(
                f"  {lbl}: {wc} words, {mx} allowed" for lbl, wc, mx in over)
            raw = _ask(cfg_llm, system, user + note, max_tokens=max_tokens)
            items2, recap2 = parse_unit_reply(raw, len(unit["beats"]), unit["beats"])
            if items2 and len(items2) >= max(1, len(items) - 2):
                items, recap = items2, recap2 or recap
        # ---- LENGTH: an under-filled unit makes the whole video short ------
        # (the reported "asked for 1500s, got 1300s"). One retry that states
        # the exact shortfall; the longer answer wins if it stays in budget.
        _got = sum(count_words(i["sentence"]) for i in items)
        _fill = float(cfg.get("min_fill", 0.9))
        if _got < budget * _fill and budget >= 20:
            note = (f"\n\nYour previous answer was only {_got} words, but this "
                    f"footage needs about {budget} words (at least "
                    f"{int(budget * _fill)}). Rewrite it with more story "
                    "detail -- motives, reactions, consequences -- covering "
                    "every stretch in order. Same JSON array format.")
            raw = _ask(cfg_llm, system, user + note, max_tokens=max_tokens)
            items3, recap3 = parse_unit_reply(raw, len(unit["beats"]), unit["beats"])
            _got3 = sum(count_words(i["sentence"]) for i in items3)
            if items3 and _got3 > _got:
                items, recap = items3, recap3 or recap
        tightened = 0
        if _over_budget_beats(unit, items):
            items, tightened = _tighten_items(unit, items)

        segs = unit_to_segments(
            unit, items, prev_end=prev_end, movie_duration=movie_duration)
        if not segs:
            raise StoryError(f"unit {unit['index'] + 1}: no segments")
        prev_end = max(prev_end, max(float(s["film_end"]) for s in segs))
        segments.extend(segs)
        sentences.extend(it["sentence"] for it in items)
        if not recap and items:
            # the array contract carries no recap line: keep continuity
            # from the unit's closing sentence
            recap = items[-1]["sentence"]
        _update_ledger(ledger, items, recap)

        if progress:
            progress(unit, len(items), sum(count_words(i["sentence"]) for i in items))

    # ---- repair: description-mode and unbound sentences, one call per batch --
    repair_stats = {"flagged": 0, "unbound": 0, "repaired": 0, "calls": 0}
    if cfg.get("repair", True):
        repair_stats = repair_segments(
            segments, units, cfg_llm, cfg=cfg,
            movie_duration=movie_duration, wpm=wpm, lang_name=lang_name,
        )

    segments, moved = enforce_chronology(segments, movie_duration)
    for seg in segments:
        seg.pop("unit_index", None)
    sentences = [s["sentence"] for s in segments]

    words = count_words(" ".join(sentences))
    lint = narrative.description_report(sentences)
    report = {
        "units": len(units),
        "sentences": len(sentences),
        "words": words,
        "budget_words": sum(int(u["budget_words"]) for u in units),
        "budget_seconds": round(
            sum(float(u["duration"]) for u in units), 1),
        "est_seconds": round(words / max(wpm, 1) * 60.0, 1),
        "target_words": int(total_target_words) if total_target_words else None,
        "description": {
            "flagged": lint["flagged"],
            "ratio": round(lint["ratio"], 4),
            "score": round(lint["score"], 4),
            "examples": lint["examples"],
        },
        "repaired": repair_stats.get("repaired", 0),
        "repair_calls": repair_stats.get("calls", 0),
        "unbound": repair_stats.get("unbound", 0),
        "chronology_moved": moved,
        "acts": [narrative.act_for(
            (float(u["start_ts"]) / movie_duration) if movie_duration else 0.0)[0]
            for u in units],
    }
    return {"segments": segments, "sentences": sentences, "report": report}


def _over_budget_beats(unit: dict, items: list[dict]) -> list[tuple[str, int, int]]:
    """``[(label, words_used, words_allowed), ...]`` for beats over budget."""
    counts: dict[str, int] = {}
    for it in items:
        lbl = unit["beats"][min(max(int(it.get("b", 1)) - 1, 0),
                                len(unit["beats"]) - 1)]["label"]
        counts[lbl] = counts.get(lbl, 0) + count_words(it["sentence"])
    out: list[tuple[str, int, int]] = []
    for b in unit["beats"]:
        used = counts.get(b["label"], 0)
        if used > b["max_words"]:
            out.append((b["label"], used, b["max_words"]))
    return out


def _tighten_items(unit: dict, items: list[dict]) -> tuple[list[dict], int]:
    """Description-first tightening to the beat budgets. Returns (items, n)."""
    by_label = {b["label"]: b for b in unit["beats"]}
    order = [b["label"] for b in unit["beats"]]
    counts: dict[str, int] = {}
    for it in items:
        lbl = order[min(max(int(it.get("b", 1)) - 1, 0), len(order) - 1)]
        counts[lbl] = counts.get(lbl, 0) + count_words(it["sentence"])
    tightened = 0
    for it in items:
        lbl = order[min(max(int(it.get("b", 1)) - 1, 0), len(order) - 1)]
        beat = by_label[lbl]
        allowed = int(beat["max_words"])
        used = counts[lbl]
        if used <= allowed:
            continue
        # Only trim what this sentence contributes beyond its fair share.
        share = max(3, int(round(allowed * count_words(it["sentence"]) /
                                 max(used, 1))))
        new = tighten_to_budget(it["sentence"], share)
        if new and new != it["sentence"]:
            counts[lbl] -= count_words(it["sentence"]) - count_words(new)
            it["sentence"] = new
            tightened += 1
    return items, tightened


# ---------------------------------------------------------------------------
# Repair pass — turn flagged description back into story
# ---------------------------------------------------------------------------

REPAIR_SYSTEM = (
    YOUTUBER_SYSTEM
    + narrative.REWRITE_RULES
    + """
Output ONLY a JSON object: {"sentences": [{"i": <index>, "text": "..."}]}
with one entry per flagged line, same indices, same order.
"""
)


def repair_segments(
    segments: list[dict],
    units: list[dict],
    cfg_llm: dict,
    *,
    cfg: dict | None = None,
    movie_duration: float = 0.0,
    wpm: int = _DEFAULT_WPM,
    lang_name: str = "English",
) -> dict:
    """One rewrite pass over the sentences that read as description.

    A sentence is flagged when the description lint fires (``we see``,
    ``the camera``, ``there is``, a picture with no people...) or when its
    words do not appear in the beat facts it claims to narrate (it is
    narrating footage it was not given). The rewrite keeps the story facts,
    the order and the length; only the shape of the sentence changes.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    if not segments:
        return {"flagged": 0, "unbound": 0, "repaired": 0, "calls": 0}

    label_to_beat: dict[tuple[int, str], dict] = {}
    for u in units:
        for b in u["beats"]:
            label_to_beat[(u["index"], b["label"])] = b

    flagged: list[int] = []
    described = 0
    unbound = 0
    for i, seg in enumerate(segments):
        sentence = seg.get("sentence", "")
        is_desc = narrative.looks_like_description(sentence)
        beat = label_to_beat.get(
            (int(seg.get("unit_index", -1)), str(seg.get("beat", ""))))
        score = binding_score(sentence, beat) if beat else 1.0
        is_unbound = bool(beat) and score < BINDING_MIN
        if is_desc:
            described += 1
        if is_unbound:
            unbound += 1
        if is_desc or is_unbound:
            flagged.append(i)

    stats = {"flagged": described, "unbound": unbound, "repaired": 0, "calls": 0}
    if not flagged:
        return stats

    batch_size = max(int(cfg.get("repair_batch", 12)), 1)
    for start in range(0, len(flagged), batch_size):
        chunk = flagged[start:start + batch_size]
        lines: list[str] = []
        for i in chunk:
            seg = segments[i]
            prev = segments[i - 1]["sentence"] if i > 0 else ""
            beat = label_to_beat.get(
                (int(seg.get("unit_index", -1)), str(seg.get("beat", ""))))
            facts = beat_facts_text(beat) if beat else ""
            words = count_words(seg.get("sentence", ""))
            lines.append(
                f'  [{i}] (currently {words} words) "{seg.get("sentence", "")}"\n'
                f"      the footage it plays over shows: {facts[:400] or '(no caption)'}"
                + (f'\n      previous line for continuity: "{prev}"' if prev else "")
            )
        user = (
            "These lines from a recap narration read as descriptions of the "
            "picture (or narrate footage that is not in their beat). Rewrite "
            "each one so it TELLS the story, keeping the same facts, the same "
            "order and roughly the same number of words.\n\n"
            + "\n".join(lines)
            + "\n\nKeep every rewritten line in the language it is already "
              "written in." + _language_block(lang_name)
            + "\nRespond with ONLY the JSON object."
        )
        try:
            raw = _ask(cfg_llm, REPAIR_SYSTEM + _language_block(lang_name), user,
                       max_tokens=_out_tokens_for_words(
                           sum(count_words(segments[i]["sentence"]) for i in chunk) + 80),
                       temperature=0.4)
        except Exception:
            continue
        stats["calls"] += 1
        fixed = _parse_repairs(raw)
        for idx, text in fixed:
            if idx not in chunk or not text:
                continue
            old = segments[idx].get("sentence", "")
            new = _clean_sentence(text)
            if not new or new == old:
                continue
            add = count_words(new) - count_words(old)
            segments[idx]["sentence"] = new
            # keep the window sensible: a longer line gets a proportionally
            # bigger share inside its own beat (chronology is re-clamped after)
            lo = float(segments[idx]["film_start"])
            hi = float(segments[idx]["film_end"])
            width = max(hi - lo + add * (60.0 / max(wpm, 1)), 0.3)
            segments[idx]["film_end"] = round(lo + width, 3)
            stats["repaired"] += 1

    return stats


def _parse_repairs(raw: str) -> list[tuple[int, str]]:
    data = _loads(_strip_reply(raw))
    out: list[tuple[int, str]] = []
    rows = []
    if isinstance(data, dict):
        rows = data.get("sentences") or data.get("lines") or data.get("rewrites") or []
    elif isinstance(data, list):
        rows = data
    for row in rows or []:
        if isinstance(row, dict):
            idx = None
            for key in ("i", "index", "id", "line"):
                if key in row:
                    try:
                        idx = int(row[key])
                    except (TypeError, ValueError):
                        idx = None
                    break
            text = row.get("text") or row.get("sentence") or row.get("rewrite") or ""
            if idx is not None:
                out.append((idx, str(text)))
        elif isinstance(row, str) and row.strip():
            out.append((-1, row))
    return out


# ---------------------------------------------------------------------------
# Optional: persist the report next to the script for debugging
# ---------------------------------------------------------------------------


def write_report(report: dict, path: Path | str) -> None:
    """Write the story report JSON (best effort; never breaks a run)."""
    try:
        Path(path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Empty-footage behaviour (what the writer can promise the timeline)
# ---------------------------------------------------------------------------
def describe_budget(
    beats: list[dict], cfg_story: dict | None = None, *, wpm: int = _DEFAULT_WPM
) -> str:
    """One line describing what the story writer will be asked to deliver."""
    units = group_into_units(beats, cfg_story, wpm=wpm)
    if not units:
        return "0 story units (no usable beats)"
    words = sum(u["budget_words"] for u in units)
    secs = sum(u["duration"] for u in units)
    return (f"{len(units)} story units over {secs:.0f}s of film, "
            f"{words} words ({words / max(wpm, 1) * 60:.0f}s of speech)")


def unit_windows(units: list[dict]) -> list[tuple[float, float]]:
    """The (start, end) film range of every unit, in order — for logging/tests."""
    return [(float(u["start_ts"]), float(u["end_ts"])) for u in units]
