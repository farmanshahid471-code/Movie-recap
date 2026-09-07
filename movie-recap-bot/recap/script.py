"""Write (or load) the English recap script.

Two modes:
  1. LLM mode — provide a plot summary / notes; the model writes the recap in
     the *Movie Recaps* narration style, one sentence per line.
  2. File mode — provide a ready-made script; it is used verbatim.

Output a text file with ONE SENTENCE PER LINE, so each line maps cleanly to a
subtitle cue later.
"""
from __future__ import annotations

from pathlib import Path

from . import llm
from .util import count_words

STYLE_PRESET = """You are a narration writer for a "movie recap" channel in the style of the
popular YouTube channel *Movie Recaps*.

Write a single English narration script that recounts the entire movie as a
fast, engaging, present-tense story. Rules:

- One complete sentence per line. Never merge multiple ideas into one line.
- Keep sentences moderate length (roughly 8 to 22 words). Short, punchy beats.
- Use present tense throughout ("James loses both his parents...").
- Describe the plot beat by beat in clear chronological order. Include key
  reveals and twists, but do NOT include the very ending/climax if it would
  spoil; the channel style is to tease.
- Conversational but cinematic. Avoid heavy analysis; you are retelling, not
  reviewing. No "this movie", no "in conclusion", no commentary about the
  film itself.
- No dialogue-heavy quoting; summarize what happens in plain narration.
- Use character names consistently. Keep the tone consistent with an English
  movie recap.
- Total roughly {target} words (between {mn} and {mx}).
- Do not write any heading, title, or trailing notes. Only the narration lines.
"""

# Step B — the exact system prompt the channel workflow uses for the final
# narrative pass over the summarized chunks.
SYSTEM_RECAP_WRITER = (
    "You are a professional YouTube movie recap scriptwriter. "
    "Do not mention that you are an AI. Do not quote dialogue. "
    "Write entirely in the third person, focusing on character actions, "
    "tension, and plot progression."
)

PROMPT_SCRIPT_JSON = """You are writing the narration for a full-length movie recap video (~{minutes} minutes of speech, roughly {target} words).

Below is a chronological SUMMARY of the movie, built from the action beats of its dialogue.

Write the complete recap script as a **JSON array of sentence strings** — one sentence per element, in chronological story order. The array is parsed by a machine, so this format is mandatory:

["First sentence.", "Second sentence.", ...]

Rules for the sentences:
- Third person, present tense. Every sentence is something that HAPPENS on screen ("Rita wakes up tied to a chair.", "The van races toward the airport.").
- Never quote dialogue. Never say "the movie", "the film", "we see", "the scene shows". No commentary or analysis.
- Tell the ENTIRE story from opening to ending, strictly chronological, including the ending.
- One self-contained action beat per sentence, ~8 to 22 words. Short, punchy, cinematic pacing.
- Use character names consistently so the viewer can follow.
- In total: about {target} words (between {mn} and {mx}) across the whole array.

Respond with ONLY the JSON array. No markdown fences, no headings, no trailing notes.

=== STORY SUMMARY ===
{summary}
=== END OF SUMMARY ===
"""


def _out_tokens_for_words(words: int) -> int:
    """Sane output cap for a prose/JSON answer of ``words``.

    English runs ~1.4-2 tokens/word; a 25% headroom plus structure padding is
    enough to never truncate a healthy answer while capping a rambling one to
    roughly what the job needs (DeepSeek's own output limit is 8192).
    """
    return max(512, min(8192, int(words * 2.2) + 512))


def render_script_json_prompt(
    summary: str,
    target: int,
    mn: int,
    mx: int,
) -> str:
    minutes = max(1, round(target / 150))  # ~150 words per minute of speech
    return PROMPT_SCRIPT_JSON.format(
        summary=summary, minutes=minutes, target=target, mn=mn, mx=mx
    )


def parse_sentences_json(raw: str) -> list[str]:
    """Robustly parse a JSON-array-of-sentences response from an LLM.

    Falls back through progressively looser strategies so a slightly sloppy
    model (fences, trailing commas, prose preamble) still yields the sentence
    list the pipeline needs.
    """
    import json
    import re

    if not raw:
        return []

    text = raw.strip()
    # 1) strip markdown fences if the model wrapped the array
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fences:
        text = fences[-1].strip()

    # 2) keep only the outermost [...] region (drops any preamble/afterword)
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return _clean_sentences(data)
        except Exception:
            pass
        # 3) tolerant JSON: strip trailing commas, then retry
        try:
            fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
            data = json.loads(fixed)
            if isinstance(data, list):
                return _clean_sentences(data)
        except Exception:
            pass
        # 4) extract every quoted string token inside the brackets
        tokens = re.findall(r'"((?:[^"\\]|\\.)*)"', candidate)
        if tokens:
            return _clean_sentences(tokens)

    # 5) last resort: treat it as loose text, one sentence per line
    return [s for s in normalize(raw.splitlines()).splitlines() if s.strip()]


def _clean_sentences(data: list) -> list[str]:
    out: list[str] = []
    for item in data:
        if isinstance(item, dict) and "sentence" in item:  # be forgiving
            item = item["sentence"]
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s:
            continue
        # sentence-end punctuation is required for clean subtitle cues
        if not s.endswith((".", "!", "?", "。", "！", "？", "…")):
            s += "."
        out.append(s)
    return out


def generate_script_json(
    summary: str,
    cfg_llm: dict,
    target: int,
    mn: int,
    mx: int,
) -> list[str]:
    """Step B — final narrative pass. Returns the script as a list of sentences."""
    user = render_script_json_prompt(summary, target, mn, mx)
    raw = llm.complete(
        cfg_llm.get("provider", ""),
        cfg_llm.get("model", ""),
        SYSTEM_RECAP_WRITER,
        user,
        base_url=cfg_llm.get("base_url"),
        json_mode=True,
        max_tokens=_out_tokens_for_words(mx),
    )
    return parse_sentences_json(raw)


# ===========================================================================
# Segmented script generation (chronological + length-accurate)
# ===========================================================================
#
# Why not one big call?
# --------------------
# Asking a model for "2250 words in one JSON array" reliably under-delivers —
# a 900-word answer becomes a 6-minute video when 15 were requested. That is
# exactly the reported bug. Instead we walk the film's own chunks in order and
# ask for a *word budget per chunk*, which:
#   * keeps every call small (models hit small targets accurately),
#   * yields the total length by construction (sum of budgets == target),
#   * tags every sentence with the film window it came from, which is what
#     makes the visual timeline strictly chronological (see recap/timeline.py).

SYSTEM_RECAP_BEATS = (
    "You are the head writer for a top-tier YouTube movie-recap channel "
    "(the 'Movie Recaps' style). You write narration that is spoken over the "
    "film's own footage. Never mention being an AI. Never quote dialogue. "
    "Never say 'the movie', 'the film', 'the scene', 'we see' or 'the camera'. "
    "Third person, present tense, active verbs, character names. "
    "Deadpan, propulsive, lightly witty. Every sentence is a VISIBLE action. "
    "Your source beats are a factual record: describe them accurately and "
    "specifically — keep every character name and proper noun they contain — "
    "and never invent events that are not in the list."
)

PROMPT_SEGMENT_JSON = """You are writing ONE SECTION of a full movie recap narration.

This section covers the part of the film from {t0} to {t1}. Below, in exact
order, are the action beats for that stretch — the complete factual record of
what happens there. Write the narration FROM those beats; never invent events.

Write EXACTLY about {budget} words of narration for this section — the audio
timing depends on it. That is roughly {nsent} sentences.

Rules:
- Third person, PRESENT tense. Every sentence is something a viewer can SEE
  happen ("Troy kicks the door open.", "The van slams into the barricade.").
- Walk the section in order from its first beat to its last. Never skip an
  entire scene and never jump backwards; where the budget cannot fit every
  minor beat, drop only the least visual sub-steps and keep one sentence per
  distinct scene, with the scene's key detail intact.
- Be SPECIFIC like a top recap channel: keep the character names and proper
  nouns from the beats ("Jessie hops onto Bullseye and rides to the twins'
  house", not "she goes to help"). About 10 to 20 words per sentence.
- Never quote dialogue. Never say "the movie", "the film", "the scene shows",
  "we see", "the camera", or comment on the filmmaking.
- Use character names consistently.
- Do not write an intro, outro, heading or summary. Only the action narration.
- {continuity}

Respond with ONLY a JSON object in this exact shape, no markdown fences:
{{"sentences": ["First sentence.", "Second sentence."]}}

=== ACTION BEATS FOR THIS SECTION ===
{beats}
=== END ===
"""


def _fmt_clock(seconds: float) -> str:
    s = max(int(seconds), 0)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _parse_segment(raw: str) -> list[str]:
    """Parse {"sentences":[...]}, tolerating the usual model sloppiness."""
    import json
    import re

    if not raw:
        return []
    text = raw.strip()
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fences:
        text = fences[-1].strip()
    # object form
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        try:
            data = json.loads(text[a : b + 1])
            if isinstance(data, dict):
                for key in ("sentences", "narration", "lines", "script"):
                    if isinstance(data.get(key), list):
                        return _clean_sentences(data[key])
        except Exception:
            pass
    # bare array form
    return parse_sentences_json(text)


def generate_segmented_script(
    chunk_summaries: list[dict],
    cfg_llm: dict,
    target_words: int,
    *,
    words_per_minute: int = 150,
    progress=None,
) -> list[dict]:
    """Write the recap chunk-by-chunk, in film order, hitting the word target.

    ``chunk_summaries`` — ``[{"index", "start", "end", "summary"}]`` in order.

    Returns ``[{"sentence", "film_start", "film_end"}]``: every sentence knows
    which stretch of film it describes, so the visual timeline is chronological
    by construction and needs no vector search at all.
    """
    usable = [c for c in chunk_summaries if (c.get("summary") or "").strip()]
    if not usable:
        return []

    # Distribute the word budget across chunks by how much story each holds,
    # so a dense 5 minutes gets more narration than a quiet one.
    weights = [max(len((c.get("summary") or "").split()), 20) for c in usable]
    wsum = float(sum(weights)) or 1.0

    out: list[dict] = []
    tail = ""  # last sentence of the previous section, for continuity
    for pos, (c, w) in enumerate(zip(usable, weights)):
        budget = max(40, int(round(target_words * (w / wsum))))
        nsent = max(3, int(round(budget / 15)))
        t0, t1 = float(c.get("start", 0.0)), float(c.get("end", 0.0))

        continuity = (
            f'This section continues directly from: "{tail}" — pick up from there '
            "without repeating it."
            if tail
            else "This is the OPENING of the recap. Start with the very first thing "
            "that happens on screen. Do not write a title or a hook."
        )

        user = PROMPT_SEGMENT_JSON.format(
            t0=_fmt_clock(t0), t1=_fmt_clock(t1), budget=budget, nsent=nsent,
            continuity=continuity, beats=(c.get("summary") or "").strip(),
        )
        raw = llm.complete(
            cfg_llm.get("provider", ""),
            cfg_llm.get("model", ""),
            SYSTEM_RECAP_BEATS,
            user,
            base_url=cfg_llm.get("base_url"),
            json_mode=True,
            max_tokens=_out_tokens_for_words(budget),
        )
        sents = _parse_segment(raw)

        # One retry if the model badly under-delivered on this section.
        got = count_words(" ".join(sents))
        if sents and got < budget * 0.55:
            more = llm.complete(
                cfg_llm.get("provider", ""),
                cfg_llm.get("model", ""),
                SYSTEM_RECAP_BEATS,
                user + (
                    f"\n\nIMPORTANT: your previous attempt was only {got} words. "
                    f"Write the FULL {budget} words this time — expand the action "
                    "into more distinct visual beats. Same JSON shape."
                ),
                base_url=cfg_llm.get("base_url"),
                json_mode=True,
                max_tokens=_out_tokens_for_words(budget),
            )
            retry = _parse_segment(more)
            if count_words(" ".join(retry)) > got:
                sents = retry

        for s in sents:
            out.append({"sentence": s, "film_start": t0, "film_end": t1})
        if sents:
            tail = sents[-1]

        if progress:
            progress(pos + 1, len(usable), count_words(" ".join(sents)), budget)

    return out


def render_prompt(notes: str, target: int, mn: int, mx: int) -> str:
    instructions = STYLE_PRESET.format(target=target, mn=mn, mx=mx)
    return f"{instructions}\n\n=== PLOT NOTES / SUMMARY TO RECAP ===\n\n{notes}\n\n=== END PLOT NOTES ===\n"


DIALOGUE_PRESET = """You are the narrator of a fast-paced YouTube "movie recap" channel.
Your job is to NARRATE the movie — tell the story as it happens, quickly — the way
recap channels talk over a montage of clips. You are NOT reviewing, explaining, or
describing the film.

Below is the TIMESTAMPED DIALOGUE / TRANSCRIPT of a film (what the characters say,
with timecodes). Use it, plus any plot summary, to write a single English narration
that retells the ENTIRE story from opening scene to ending, as one continuous,
fast-moving, present-tense tale.

Rules:
- NARRATE events, don't describe them. Every line is something that HAPPENS:
  "Woody shoves Buzz off the bed." "The van speeds toward the airport."
- Move fast. Chain actions back to back so the story races forward. No filler,
  no scene-setting paragraphs, no lingering.
- Present tense throughout ("Andy's room comes alive...").
- Strict chronological order, covering the whole plot including the ending.
- NEVER say "the movie", "the film", "the show", "we see", "the scene shows",
  and never analyze themes or comment on the story. Just tell it.
- Retell the dialogue as action; do not quote it verbatim.
- One complete sentence per line (roughly 8 to 20 words). Short, punchy beats.
- Use consistent character names.
- Total roughly {target} words (between {mn} and {mx}).
- No heading, title, or trailing notes. Only the narration lines.
"""


def render_dialogue_prompt(transcript: str, plot: str, target: int, mn: int, mx: int) -> str:
    instructions = DIALOGUE_PRESET.format(target=target, mn=mn, mx=mx)
    plot_block = f"=== PLOT SUMMARY (optional, may be empty) ===\n{plot or '(none)'}\n" if plot else ""
    return (
        f"{instructions}\n\n"
        f"{plot_block}\n"
        f"=== TIMESTAMPED DIALOGUE / TRANSCRIPT ===\n{transcript}\n"
        f"=== END TRANSCRIPT ===\n"
    )


def generate_from_dialogue(
    transcript: str,
    plot: str,
    cfg_llm: dict,
    target: int,
    mn: int,
    mx: int,
    model: str | None = None,
) -> str:
    system = "You write engaging, present-tense movie recap narration."
    user = render_dialogue_prompt(transcript, plot, target, mn, mx)
    return llm.complete(
        cfg_llm.get("provider", ""),
        model or cfg_llm.get("model", ""),
        system,
        user,
        base_url=cfg_llm.get("base_url"),
        max_tokens=_out_tokens_for_words(mx),
    )


def generate_online(notes: str, cfg_llm: dict, target: int, mn: int, mx: int) -> str:
    system = "You write engaging, present-tense movie recap narration."
    user = render_prompt(notes, target, mn, mx)
    return llm.complete(
        cfg_llm.get("provider", ""),
        cfg_llm.get("model", ""),
        system,
        user,
        base_url=cfg_llm.get("base_url"),
        max_tokens=_out_tokens_for_words(mx),
    )


def normalize(lines: list[str]) -> str:
    """Join raw LLM lines into clean, one-sentence-per-line output."""
    out: list[str] = []
    for raw in lines:
        text = raw.strip().lstrip("-*0123456789. ").strip()
        if not text:
            continue
        # Split combined sentences to keep cue granularity.
        for part in _split_sentences(text):
            part = part.strip()
            if part:
                out.append(_ensure_sentence_end(part))
    return "\n".join(out)


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _ensure_sentence_end(s: str) -> str:
    """Append an ending punctuation mark if missing, without doubling CJK ones."""
    if s.endswith((".", "!", "?", "。", "！", "？", "…")):
        return s
    # Chinese sentences often end with 。; keep it consistent for zh text.
    if any("\u4e00" <= ch <= "\u9fff" for ch in s):
        return s + "。"
    return s + "."


def load_script(path: Path | str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Script file not found: {p}")
    return normalize(p.read_text(encoding="utf-8").splitlines())


def write_script_file(text: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text.strip() + "\n", encoding="utf-8")
    return dest
