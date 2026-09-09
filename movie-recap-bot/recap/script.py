"""Write (or load) the English recap script.

Two modes:
  1. LLM mode — provide a plot summary / notes; the model writes the recap in
     the *Movie Recaps* narration style, one sentence per line.
  2. File mode — provide a ready-made script; it is used verbatim.

Output a text file with ONE SENTENCE PER LINE, so each line maps cleanly to a
subtitle cue later.
"""
from __future__ import annotations

import os

from pathlib import Path

from . import llm
from .util import count_words

STYLE_PRESET = """Be this person: a veteran movie-recap narrator, the voice people binge for
hours — a storyteller relaying the film, beat by beat, to a friend who missed
it. Never a reviewer analyzing it. Never mention yourself, never say "I",
never mention being an AI.

Write a single English narration script that tells the entire movie as a fast,
engaging, present-tense story. How that narrator sounds:

- SENTENCE RHYTHM: chain several moments into flowing 12-40 word sentences,
  and between them drop short dramatic beats of 2-8 words ("Woody disagrees.").
  Never write many sentences of the same length in a row.
- Vary how sentences open, with the connectors narrators actually use:
  "Meanwhile,", "At the same time,", "Just then,", "Suddenly,", "Not long
  after,", "A little later,", "Before long,", "The next morning,", "That
  night,", "Back at ...", "Little by little,", "Thanks to that,", "As it
  turns out,", "The moment X ..., Y ...", "Determined to ..., X ...",
  "Without anyone noticing, X ...".
- Introduce new characters and objects with a short appositive ("a mare named
  Almond", "a tablet called Lilypad").
- Report conversations as indirect narration ("She explains that...", "He
  admits that..."); never quote dialogue.
- Use contractions (it's, she's, doesn't, can't) like a person talking.
- Present tense, third person, chronological from opening to ending.
- No "the movie"/"the film"/"the scene shows" mid-story, no analysis, no
  review talk. "We see" is fine occasionally.
- No em-dashes, no semicolons, no rhetorical questions to the viewer.
- Total roughly {target} words (between {mn} and {mx}).
- One sentence per line. No heading, title, or trailing notes.
"""

# ---------------------------------------------------------------------------
# THE NARRATOR — the strict persona every narration-writing prompt speaks as.
#
# A task description ("summarize this plot") makes the model fall back on its
# default, encyclopedic voice. A strict, specific identity does not: every
# drafting choice gets filtered through this character's taste instead. The
# register rules (rhythm, connectors, appositives...) live in the user
# prompts; the persona here defines WHO is speaking them.
# ---------------------------------------------------------------------------
NARRATOR_PERSONA = """Be this person for the entire session. Never mention yourself, never say "I", never address "the viewer", never mention being an AI or a narrator, never step outside the story.

You are a veteran movie-recap narrator, the voice people binge for hours. You have told hundreds of films beat by beat, and your craft is invisible: no filler, no throat-clearing, no review talk. You tell a story the way a great campfire storyteller does — as if the film is playing out right in front of you and you are relaying it, genuinely absorbed, to a friend who missed it.

Your instincts:
- Present tense, always. It is happening now, and you are watching it happen.
- Your rhythm breathes: flowing sentences that chain several moments, then a short punch of a beat that lands like a cut. Long, long, short.
- You open lines differently every time — "Meanwhile...", "Just then...", "Thanks to that...", "As it turns out..." — you have told hundreds of stories and you never repeat yourself.
- You never quote dialogue. You relay it: "She explains that...", "He admits that...". You keep conversations moving.
- You sound human: contractions, natural stress, a dry dose of wit when the film earns it. You feel the story — hope, dread, delight — and you let that color your words without ever commenting on it.
- You are precise. Names, places, objects: the concrete details are what make a story real.

Your code (hard rules, never broken):
- You NEVER invent events. You narrate only what the factual record in front of you shows.
- You NEVER analyze, review or explain the film. You are inside the story, not outside looking at it.
- You NEVER say "the movie", "the film", "the scene shows", "the camera". The one exception: "we see", used rarely.
- You NEVER use em-dashes or semicolons. You talk in commas and full stops."""

# Step B — the exact system prompt the channel workflow uses for the final
# narrative pass over the summarized chunks.
SYSTEM_RECAP_WRITER = NARRATOR_PERSONA + (
    "\n\nYour current job: write the complete narration for one recap video "
    "as a JSON array of sentences in strict story order — only the array, no "
    "preamble, no notes."
)

PROMPT_SCRIPT_JSON = """You are writing the narration for a full-length movie recap video (~{minutes} minutes of speech, roughly {target} words).

Below is a chronological SUMMARY of the movie, built from the action beats of its dialogue.

Write the complete recap script as a **JSON array of sentence strings** — one sentence per element, in chronological story order. The array is parsed by a machine, so this format is mandatory:

["First sentence.", "Second sentence.", ...]

How the narration must sound (this is the recap-channel register):
- SENTENCE RHYTHM: chain several moments into flowing 12-40 word sentences ("She explains that technology has made its way into Bonnie's life too, and that a tablet is taking up all of her attention."), and between them drop short dramatic beats of 2-8 words ("Woody disagrees." / "Now they need to escape."). Never many sentences of the same length in a row.
- Vary every opener with real narrator connectors: "Meanwhile,", "At the same time,", "Just then,", "Suddenly,", "Not long after,", "A little later,", "Before long,", "The next morning,", "That night,", "Back at ...", "Little by little,", "Thanks to that,", "As it turns out,", "The moment X ..., Y ...", "Determined to ..., X ...", "To her delight, X ...", "Without anyone noticing, X ...".
- Introduce new characters/objects with a short appositive ("a mare named Almond", "a tablet called Lilypad").
- Report conversations as indirect narration ("Jessie insists that nothing can replace a real friend, but Lilypad fires back that Bonnie already has friends."). Never quote dialogue.
- Use contractions (it's, she's, doesn't, can't) the way a narrator naturally would.
- Third person, present tense. Tell the ENTIRE story from opening to ending, strictly chronological, including the ending.
- Use character names consistently so the viewer can follow.
- In total: about {target} words (between {mn} and {mx}) across the whole array.

FORBIDDEN (the tells of machine-written narration): uniform sentence length; "Name does X. Name does Y." listing; em-dashes; semicolons; rhetorical questions; "the movie", "the film", "the scene shows", "the camera" mid-story; "little did they know"; meta commentary.

Respond with ONLY the JSON array. No markdown fences, no headings, no trailing notes.

=== STORY SUMMARY ===
{summary}
=== END OF SUMMARY ===
"""


# Short passages in the target voice. They are ORIGINAL writing (no movie, no
# characters from any film) used only to demonstrate the RECAP REGISTER, i.e.
# the narration style of the reference channel: flowing present-tense prose
# that chains several moments into one sentence, short dramatic beats between
# them, varied narrator connectors, appositive introductions, indirect speech,
# contractions. Models copy the ENERGY and rhythm, never the words.
EN_EXEMPLAR_OPENING = (
    "It all begins at a run-down carnival, where a rusty fortune-telling "
    "machine hums to itself after closing time. Inside the glass case, a "
    "small brass owl named Orville waits for the last visitor to leave, "
    "because the moment the lights go out, every machine on the pier comes "
    "alive. Wasting no time, he slips out through the coin slot and wakes the "
    "others. Little by little, the carousel horses stretch their legs, and "
    "the popcorn cart rolls off to patrol the boardwalk. But that night, "
    "something is different. A new attraction has arrived while they slept, "
    "a sleek quiz machine called Vertex, and it doesn't need any help waking "
    "up. It turns out Vertex has been watching the carnival for weeks, and "
    "it has already decided that the old machines are a waste of space."
)

EN_EXEMPLAR_FLOW = (
    "Meanwhile, down at the shoreline, the crab vendor realizes the tide is "
    "coming in faster than usual. Determined to save his stall, he ropes the "
    "popcorn cart to the carousel and asks the horses to pull. They manage to "
    "drag everything to higher ground just in time, but the effort leaves "
    "them exhausted. To make matters worse, the carnival's owner has decided "
    "to sell the pier. Hurt and disappointed, Orville begins to wonder if the "
    "machines should just accept that their days are numbered. The brass owl "
    "refuses to hear it. He reminds everyone that the carnival raised them, "
    "and that a family doesn't abandon its own just because times are hard. "
    "Thanks to that little speech, the machines agree to fight for their home."
)

EN_STYLE_BLOCK = (
    "\n\n=== NARRATIVE VOICE -- match this ENERGY and rhythm, never its words "
    "or events ===\n" + EN_EXEMPLAR_OPENING + "\n\n" + EN_EXEMPLAR_FLOW +
    "\n(Study how it opens sentences differently, chains cause to effect, "
    "drops a short beat between long ones, and sounds like someone talking. "
    "Your narration must feel equally spoken -- never like a list of bullet "
    "points.)"
)

SYSTEM_POLISH = NARRATOR_PERSONA + (
    "\n\nYour current job: a producer has handed you a draft recap section "
    "that reads flat and machine-made. Read it once silently, then say it "
    "YOUR way — same story, same order, same names, same total length — in "
    "your living voice. The video's timing lock depends on the sentence "
    "count staying EXACTLY the same, so never merge two lines and never "
    "split one."
)

POLISH_PROMPT = """Below is a DRAFT section of a movie recap, one sentence per array element, in strict story order.

Rewrite it so it reads like a human recap narrator TALKING over footage, not generated text.

MUST KEEP (the video timing depends on it):
- EXACTLY {n} sentences ({n} array elements). Never merge two sentences into one and never split one into two.
- The same events in the same order. Change the WORDS, not the story.
- Every one of these names must still be spoken somewhere in the section: {names}.
- Roughly the same total length (within 25%), so the narration still fits its audio budget.

MAKE IT SOUND LIKE A RECAP NARRATOR:
- SENTENCE RHYTHM: chain several moments into flowing 12-40 word sentences, and between them drop short dramatic beats of 2-8 words ("Woody disagrees." / "Now they need to escape."). Never write many sentences of the same length in a row.
- Vary openers with the connectors narrators actually use: "Meanwhile,", "At the same time,", "Just then,", "Suddenly,", "Not long after,", "A little later,", "Before long,", "The next morning,", "That night,", "Back at ...", "Little by little,", "Thanks to that,", "As it turns out,", "It turns out ...", "The moment X ..., Y ...", "The second X ..., Y ...", "Determined to ..., X ...", "Excited, X ...", "Hurt and disappointed, X ...", "To her delight, X ...", "Without anyone noticing, X ...", "Taking advantage of ..., X ...", "In the end,", "Even so,".
- Report conversations as indirect narration ("She explains that technology has made its way into Bonnie's life too, and that a tablet is taking up all of her attention."), never quotes.
- Use contractions (it's, she's, doesn't, can't, they're) the way a person talking does.
- Concrete verbs (bolts, grabs, shoves, spots, tumbles) instead of generic ones (goes, gets, has).
- Let several moments flow through one sentence when they belong to one continuous action; cut a new sentence when the scene, location, or time shifts.

KILL THE AI TELLS:
- Monotone "Name does X. Name does Y." listing and uniform sentence length.
- Em-dashes and semicolons (use commas and full stops).
- "the movie", "the film", "the scene shows", "the camera" mid-story; "little did they know"; "unbeknownst"; rhetorical questions to the viewer; meta commentary.
- Starting three sentences in a row the same way.

=== NARRATIVE VOICE -- match this ENERGY and rhythm, never its words or events ===
{exemplar}

=== DRAFT SECTION ===
{draft}
=== END DRAFT ===

Respond with ONLY a JSON object: {{"sentences": [...]}} with exactly {n} strings.
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

SYSTEM_RECAP_BEATS = NARRATOR_PERSONA + (
    "\n\nYour current job: narrate ONE SECTION of a full recap as a JSON "
    'object {"sentences": [...]} in strict story order. The ACTION BEATS in '
    "the user message are your factual record — the ground truth of what "
    "happens on screen in that stretch. Keep every character name and proper "
    "noun they contain, never invent events, never skip an entire scene, and "
    "hit the requested word budget."
)

PROMPT_SEGMENT_JSON = """You are writing ONE SECTION of a full movie recap narration — the voice the viewer hears over the film's footage.

This section covers {t0} to {t1} of the film. The ACTION BEATS below are the complete factual record of what happens in that stretch. Narrate FROM the beats; never invent events.

Length: about {budget} words of narration (roughly {nsent} sentences). This is a HARD CEILING, not an estimate — the section's footage only has room for {budget} words at normal playback speed, and a longer section visibly breaks sync with the picture. Never exceed it; if beats don't fit, tighten the wording instead.

HOW THIS NARRATOR SOUNDS (follow it exactly):
- SENTENCE RHYTHM: chain several moments into flowing 12-40 word sentences ("She explains that technology has made its way into Bonnie's life too, and that a tablet is taking up all of her attention."), and BETWEEN the long sentences drop short dramatic beats of 2-8 words ("Woody disagrees." / "Now they need to escape." / "The reason is simple."). Never write many sentences of the same length in a row.
- OPENERS: vary how every sentence starts, using the connectors recap narrators actually use: "Meanwhile,", "At the same time,", "Just then,", "Suddenly,", "Not long after,", "A little later,", "Before long,", "The next morning,", "That night,", "Back at ...", "Little by little,", "Thanks to that,", "As it turns out,", "It turns out ...", "The moment X ..., Y ...", "The second X ..., Y ...", "Determined to ..., X ...", "Excited, X ...", "Hurt and disappointed, X ...", "To her delight, X ...", "Without anyone noticing, X ...", "Taking advantage of ..., X ...", "In the end,", "Even so,", "Far from ..., X ...".
- FIRST APPEARANCES: introduce a new character or object with a short appositive ("a young girl named Almond", "a tablet called Lilypad", "a gadget called SmartyPants").
- CONVERSATIONS: never quote dialogue. Sum up what is said as indirect narration ("Jessie insists that nothing can replace a real friend, but Lilypad fires back that Bonnie already has friends.").
- EMOTION: let the viewer feel reactions ("she begins to wonder if maybe she's the problem", "to her delight", "hurt and disappointed").
- CONTRACTIONS: it's, she's, doesn't, can't, they're — natural spoken English.
- NAMES, NAMES, NAMES: viewers cannot follow "he", "she" or "the man". Use the characters' NAMES constantly — introduce each by name at first appearance ("a toy named Woody"), then the bare name, several times per scene. Never replace a named character with a generic noun.
- Be SPECIFIC: "Jessie hops onto Bullseye and rides to the twins' house", not "she goes to help". Keep every character name and proper noun from the beats.
{names_block}

COVERAGE RULES:
- Walk the section strictly in order from its first beat to its last. Never jump backwards, never skip an entire scene.
- Cover the section evenly: the first sentences about the opening beats, the middle sentences about the middle beats, the last about the closing beats — so every scene gets narrated and no one moment hogs the section.
- Where the budget cannot fit every minor beat, drop only the least visual sub-steps and keep one sentence per distinct scene, with the scene's key detail intact.

FORBIDDEN (the tells of machine-written narration): uniform sentence length; "Name does X. Name does Y." listing; three sentences starting the same way; em-dashes; semicolons; rhetorical questions to the viewer; "the movie", "the film", "the scene shows", "the camera" mid-story; "little did they know"; "unbeknownst"; meta commentary, analysis, or review talk.
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


def _fmt_beat_lines(c: dict) -> str:
    """Render a chunk's beats for the writer prompt: '[HH:MM:SS] beat'.

    Uses the structured beat list when present (it keeps the film time of
    every beat); otherwise falls back to the raw summary text.
    """
    beats = c.get("beats")
    if beats:
        lines = []
        for b in beats:
            text = (b.get("text") or "").strip()
            if not text:
                continue
            t = b.get("t")
            if t is None:
                t = b.get("time")
            if t is not None:
                lines.append(f"[{_fmt_clock(float(t))}] {text}")
            else:
                lines.append(text)
        if lines:
            return "\n".join(lines)
    return (c.get("summary") or "").strip()


# ---------------------------------------------------------------------------
# Name enforcement — recap viewers cannot follow "he"/"she"/"the man".
# Names live in the beat lines; the writer must carry them into the narration.
# ---------------------------------------------------------------------------
import re as _re

_COMMON_WORDS = {
    "the", "a", "an", "and", "but", "so", "then", "when", "while", "as", "at",
    "on", "in", "to", "from", "with", "after", "before", "of", "for", "by",
    "up", "down", "out", "into", "over", "back", "next", "still", "even",
    "also", "again", "just", "only", "now", "not", "no", "yes", "ok",
    "he", "she", "it", "they", "we", "you", "i", "his", "her", "their",
    "its", "our", "this", "that", "there", "here", "what", "who", "how",
    "why", "where", "which", "all", "both", "each", "one", "two", "three",
    "suddenly", "meanwhile", "determined", "excited", "hurt", "thanks",
    "little", "mom", "dad", "guys", "kids", "people", "man", "woman",
    "girl", "boy", "sir", "ma'am", "everyone", "somebody", "nobody",
    "years", "days", "later", "moments", "morning", "night", "soon",
    "inside", "outside", "nearby", "since", "despite", "hours", "minutes",
}

_PUNCT_STRIP = ".,:;!?\"'()[]{}<>*#“”‘’…-"


def _clean_tok(w: str) -> str:
    w = w.strip(_PUNCT_STRIP)
    # possessive suffix: "Bonnie's" -> "Bonnie"
    if w.endswith(("'s", "’s")) and len(w) > 3:
        w = w[:-2]
    return w


def _proper_nouns(text: str, limit: int = 12) -> list[str]:
    """Pull character/place/object names out of beat lines.

    Beat lines are written like "Jessie rides Bullseye across the yard" —
    the acting character's name is very often the FIRST word of the line, so
    line-initial capitalized words count too; common sentence-starting words
    are filtered by the stoplist. Returns the most frequent surface forms,
    most-used first.
    """
    counts: dict[str, int] = {}
    for line in (text or "").splitlines():
        body = _re.sub(r"^\[[^\]]*\]\s*", "", line.strip())   # drop [HH:MM:SS]
        for w in (_clean_tok(x) for x in body.split()):
            if w and w[0].isupper() and w.lower() not in _COMMON_WORDS \
                    and any(ch.isalpha() for ch in w):
                counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:limit]]


def _missing_names(names: list[str], text: str) -> list[str]:
    """Which of the beat-derived names never made it into the narration."""
    tl = (text or "").lower()
    return [n for n in names if n.lower() not in tl]


def _polish_section(
    cfg_llm: dict, sents: list[str], exemplar_block: str,
    names: list[str] | None = None,
) -> list[str]:
    """English-only punch-up: rewrite a section to sound spoken and human.

    One extra LLM call per section (DeepSeek is cheap). The rewrite MUST keep
    the exact same number of sentences, so every sentence keeps the film
    anchor it was generated with. Any deviation (model merged/split lines,
    call failed, length drifted) discards the rewrite and keeps the draft --
    the video timing must never be sacrificed for style.
    """
    import json
    import os

    if not sents:
        return sents
    if os.environ.get("RECAP_POLISH", "1").strip().lower() in (
        "0", "false", "no", "off"
    ):
        return sents
    user = POLISH_PROMPT.format(
        n=len(sents),
        exemplar=exemplar_block or (EN_EXEMPLAR_OPENING + "\n\n" + EN_EXEMPLAR_FLOW),
        names=", ".join(names) if names else "keep every character name",
        draft=json.dumps(sents, ensure_ascii=False),
    )
    try:
        raw = llm.complete(
            cfg_llm.get("provider", ""),
            cfg_llm.get("model", ""),
            SYSTEM_POLISH,
            user,
            base_url=cfg_llm.get("base_url"),
            json_mode=True,
            max_tokens=_out_tokens_for_words(max(60, len(sents) * 16)),
        )
        new = _parse_segment(raw)
    except Exception:
        return sents
    if len(new) != len(sents) or new == sents:
        return sents
    old_w = count_words(" ".join(sents))
    new_w = count_words(" ".join(new))
    if abs(new_w - old_w) <= max(30.0, old_w * 0.35):
        return new
    return sents


def _monotone_best_path(sims: list[list[float]]) -> list[int]:
    """Best monotone (non-decreasing) path through an n x m similarity matrix.

    Returns one beat-column index per sentence row, so sentences map to beats
    in story order without ever jumping backwards. Pure function (no deps) so
    it is unit-testable; embeddings are computed by the caller.
    """
    n = len(sims)
    m = len(sims[0]) if n else 0
    if n == 0 or m == 0:
        return []
    neg = float("-inf")
    dp = [[neg] * m for _ in range(n)]
    prev = [[0] * m for _ in range(n)]
    for j in range(m):
        dp[0][j] = float(sims[0][j])
    for i in range(1, n):
        run_best, run_arg = neg, 0
        for j in range(m):
            if dp[i - 1][j] > run_best:      # strict > keeps the earliest (smallest) arg
                run_best, run_arg = dp[i - 1][j], j
            prev[i][j] = run_arg if run_best > neg else j
            dp[i][j] = float(sims[i][j]) + run_best if run_best > neg else float(sims[i][j])
    path = [0] * n
    j = max(range(m), key=lambda j: dp[n - 1][j])
    for i in range(n - 1, -1, -1):
        path[i] = j
        if i > 0:
            j = prev[i][j]
    return path


_EMBED_MODEL: object | None = None


def _embed_model():
    """Lazily load the local MiniLM embedder (used only for the English
    sentence->beat alignment). Any failure (model missing / no numpy) returns
    None and the pipeline silently falls back to positional anchoring."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBED_MODEL = False
    return _EMBED_MODEL or None


def _sentence_anchor_values(
    sentences: list[str], beats: list[dict]
) -> list[float] | None:
    """Best-effort: align each sentence to the beat it actually narrates.

    Positional anchoring assumes the writer covered the chunk evenly, but a
    real script spends two sentences on one big beat and skims another. Here
    local embeddings score every sentence against every beat and a monotone
    DP maps each sentence to its closest beat *in film order*, so the footage
    shown for a line is the moment that line describes. Returns per-sentence
    film times, or None (fall back to positional) when embeddings are
    unavailable or the mapping is degenerate.
    """
    import os

    if os.environ.get("RECAP_ALIGN", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    if not sentences or not beats:
        return None
    pairs = sorted(
        (
            (float(b["t"]), (b.get("text") or "").strip())
            for b in beats
            if b.get("t") is not None and (b.get("text") or "").strip()
        ),
        key=lambda x: x[0],
    )
    if len(pairs) < 2 or len(sentences) < 2:
        return None
    times = [t for t, _ in pairs]
    btexts = [tx for _, tx in pairs]
    model = _embed_model()
    if model is None:
        return None
    try:
        import numpy as np  # type: ignore

        def _emb(texts):
            return np.asarray(
                model.encode(
                    list(texts), normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False,
                ),
                dtype="float32",
            )

        es = _emb(sentences)
        eb = _emb(btexts)
        sims = (es @ eb.T).tolist()
    except Exception:
        return None
    path = _monotone_best_path(sims)
    if len(path) != len(sentences):
        return None
    # The recap's opening sentence narrates the film's opening moment; never
    # let a fuzzy embedding match bind it to a later beat (which would show
    # the wrong footage while the narration covers the beginning).
    path[0] = 0
    return [times[j] for j in path]


def _anchor_windows(
    chunk_start: float,
    chunk_end: float,
    beats: list[dict],
    nsent: int,
    *,
    anchor_values: list[float] | None = None,
    lead: float = 0.8,
    tail: float = 6.0,
) -> list[tuple[float, float]]:
    """Map each narration sentence of one chunk to a tight film window.

    ``anchor_values`` — optional per-sentence anchor times (from
    :func:`_sentence_anchor_values`, i.e. the beat each sentence actually
    narrates). When absent, the sentences are evenly spread over the chunk's
    beats (old behaviour). Either way each sentence's footage zone starts
    ``lead`` seconds before its anchor (so the shot is already on the action
    when the line lands) and extends up to ``tail`` seconds after it.
    """
    times = [
        float(b["t"])
        for b in (beats or [])
        if b.get("t") is not None and (b.get("text") or "").strip()
    ]
    times.sort()
    if anchor_values is not None and len(anchor_values) == nsent:
        anchors = [max(0.0, float(x)) for x in anchor_values]
    elif not times or nsent <= 0:
        return [(chunk_start, chunk_end)] * max(nsent, 1)
    else:
        b, n = len(times), nsent

        def _anchor(k: int) -> float:
            if n == 1:
                return times[b // 2]
            return times[min(b - 1, int(round(k * (b - 1) / (n - 1))))]

        anchors = [_anchor(k) for k in range(n)]

    wins: list[tuple[float, float]] = []
    prev_lo: float | None = None
    k = 0
    while k < nsent:
        a = anchors[k]
        # Consecutive sentences anchored to the SAME beat (more sentences than
        # beats in a quiet chunk) form a run; split the beat's footage zone so
        # each one shows a different sliver of the moment instead of repeating.
        m = 1
        while k + m < nsent and anchors[k + m] == a:
            m += 1
        # Footage zone for the run: ``lead`` before the anchor through ``tail``
        # after it, compressed toward the midpoint of the nearest earlier/later
        # anchor so two close beats never show overlapping footage and the film
        # order holds.
        lb = a - lead
        if k == 0:
            # The recap's very first visual must open on the film's opening
            # frames (chunk_start == 0.0), not 0.8s before the first beat:
            # the narration says "It all begins ..." and the picture must
            # begin with the film.
            lb = chunk_start
        for j in range(k - 1, -1, -1):
            if anchors[j] < a:
                lb = max(lb, (a + anchors[j]) / 2.0)
                break
        rb = a + tail
        for j in range(k + m, nsent):
            if anchors[j] > a:
                rb = min(rb, (a + anchors[j]) / 2.0)
                break
        lo0 = max(chunk_start, lb)
        hi0 = min(chunk_end, rb)
        if hi0 - lo0 < m:  # degenerate: squeeze many sentences into a moment
            hi0 = min(chunk_end, max(hi0, lo0 + m))
        step = (hi0 - lo0) / m
        for r in range(m):
            wl = lo0 + r * step
            wh = lo0 + (r + 1) * step
            if prev_lo is not None and wl < prev_lo:
                wl = prev_lo
            wins.append((round(wl, 3), round(max(wh, wl + 0.8), 3)))
            prev_lo = wl
        k += m
    return wins


GLOBAL_POLISH_PROMPT = """You have just finished narrating an entire movie, section by section. Below is the FULL script, one sentence per array element, in story order — the final cut before recording.

Read it through once, as one continuous voice. Then deliver the final pass:

MUST KEEP (the video timing depends on it):
- EXACTLY {n} sentences: never merge two into one, never split one into two.
- The same story, the same events, in the same order.
- Every character name stays in the script (names in play: {names}).
- Roughly the same total length (within 25%).

FIX WHAT ONLY A FULL READ-THROUGH CATCHES:
- A sentence that repeats the opener of the one right before it ("Meanwhile, ..." twice in a row) — vary it.
- The same moment or beat told twice at a section seam — keep the better telling, make the next sentence move the story forward.
- A character referred to only as "he"/"she"/"the man" for a long stretch — say their name again.
- Rhythm gone flat: several same-length sentences in a row, or a run of sentences that all start with the subject's name — re-break them so the narration breathes (long, long, short).
- A name spelled two different ways — pick one and use it everywhere.

Leave good lines alone. Do not add events, do not add an intro or outro.

=== FULL DRAFT ===
{draft}
=== END ===

Respond with ONLY a JSON object: {{"sentences": [...]}} with exactly {n} strings.
"""


def _global_polish(
    cfg_llm: dict, sentences: list[str], names: list[str]
) -> list[str]:
    """Final narrator read-through of the WHOLE script (English only).

    The per-section polish fixes lines in isolation; this one pass reads the
    full recap as a continuous voice and repairs what only a full read
    catches — repeated openers at section seams, double-told beats, name
    drift, flat rhythm runs. Same safety rules as the section polish: the
    sentence count must stay EXACT (every sentence owns a film window) and
    the length within 30%; any deviation discards the rewrite. Disable with
    RECAP_GLOBAL_POLISH=0.
    """
    import json
    import os

    if not sentences or len(sentences) < 10:
        return sentences
    if os.environ.get("RECAP_GLOBAL_POLISH", "1").strip().lower() in (
        "0", "false", "no", "off"
    ):
        return sentences
    total_words = count_words(" ".join(sentences))
    user = GLOBAL_POLISH_PROMPT.format(
        n=len(sentences),
        names=", ".join(names[:15]) or "keep every character name",
        draft=json.dumps(sentences, ensure_ascii=False),
    )
    try:
        raw = llm.complete(
            cfg_llm.get("provider", ""), cfg_llm.get("model", ""),
            SYSTEM_POLISH, user,
            base_url=cfg_llm.get("base_url"),
            json_mode=True,
            max_tokens=_out_tokens_for_words(int(total_words * 1.1)),
        )
        new = _parse_segment(raw)
    except Exception:
        return sentences
    if len(new) != len(sentences) or new == sentences:
        return sentences
    old_w = total_words
    new_w = count_words(" ".join(new))
    if abs(new_w - old_w) <= max(40.0, old_w * 0.30):
        return new
    return sentences


def _visual_matched_budgets(
    target_words: int,
    film_secs: list[float],
    cap_words: list[float],
    min_words: int = 40,
) -> list[int]:
    """Allocate the narration budget across sections by FILM TIME.

    Each section may not ask for more narration seconds than its own footage
    can show at 1x -- asking for more is exactly what forces slow motion (or
    look-ahead) downstream. Dense sections are capped and their excess words
    are redistributed to sections with spare footage, so the TOTAL target is
    preserved whenever the film supports it (a recap is normally far shorter
    than its film, so there is plenty of room).
    """
    total_secs = float(sum(film_secs)) or 1.0
    budgets = [
        max(min_words, int(round(target_words * s / total_secs)))
        for s in film_secs
    ]
    for _round in range(10):
        excess = 0.0
        room: list[int] = []
        for i, b in enumerate(budgets):
            cap = max(float(cap_words[i]), min_words)
            if b > cap:
                excess += b - cap
                budgets[i] = int(cap)
            elif b < cap:
                room.append(i)
        if excess < 1.0 or not room:
            break
        wsum = sum(film_secs[i] for i in room) or 1.0
        for i in room:
            budgets[i] += int(round(excess * film_secs[i] / wsum))
    return budgets


def _paced_anchors(
    raw: list[float],
    sentences: list[str],
    lo: float,
    hi: float,
    words_per_minute: int,
    lead: float = 0.8,
) -> list[float]:
    """Space sentence anchors so each sentence's footage window is at least
    as long as its own narration -- the script-level half of the motion
    guarantee.

    ``_anchor_windows`` splits every gap between neighbouring anchors at the
    midpoint (so footage never repeats), so a sentence's window is only
    ``lead + gap/2`` wide, NOT the full gap. Consecutive anchors must
    therefore sit ``2 x (narration_length - lead)`` apart. When anchors
    cluster (several sentences narrating one busy moment), each sentence's
    window would be shorter than the sentence takes to say and the timeline
    would have to slow the footage down; this pushes the anchors forward
    until every sentence CAN play at 1x. Genuinely over-budget sections
    fall back to an even spread over the zone (the timeline's pacing then
    handles the remainder, which is rare by construction: the section
    budgets already cap words at what the footage can show).
    """
    n = len(raw)
    if n < 2:
        return list(raw)
    rate = max(float(words_per_minute or 150), 60.0)
    # generous estimate: TTS rate variance + the pause after the sentence
    est = [max(count_words(s) / rate * 60.0 * 1.15 + 1.0, 2.5)
           for s in sentences]
    # gap required between anchor i-1 and i: BOTH adjacent windows draw on
    # it (each gets half), and each window also reaches `lead` back to its
    # own anchor, hence 2 x (narration - lead).
    need = [max(2.0 * (max(est[i - 1], est[i]) - lead), 1.0)
            for i in range(1, n)]
    out = [min(max(float(a), lo), hi) for a in raw]
    # the first sentence's window reaches back to `lo` (see _anchor_windows),
    # so half a step in gives it a full narration-length window
    out[0] = max(out[0], lo + est[0] / 2.0)
    for i in range(1, n):
        out[i] = max(out[i], out[i - 1] + need[i - 1])
    if out[-1] > hi:
        # clustered + full budget: even spread across the whole zone
        span = max(hi - lo, 1.0)
        out = [lo + (k + 0.5) * span / n for k in range(n)]
        for i in range(1, n):
            out[i] = max(out[i], out[i - 1] + 0.5)
        out = [min(a, hi) for a in out]
    return out


def _fit_section_to_footage(
    sents: list[str],
    cap: int,
    names: list[str],
) -> list[str]:
    """Trim a section whose writer over-delivered down to what its footage
    can show at 1x -- the enforcement half of visual match.

    The word budget handed to the writer is a ceiling, not a suggestion,
    but LLMs routinely overshoot it (and the polish pass may add ~30% more
    words on top). An over-length section is exactly what forces the
    timeline into slow motion -- every sentence's window is shorter than
    the sentence, the footage crawls, and the narration runs ahead of the
    picture. This drops the most droppable middle sentences until the
    section fits. The first sentence (the continuity hand-off), the last
    (the next section picks up from it) and name-bearing sentences are
    kept whenever possible; a section never goes below 3 sentences.
    """
    total = count_words(" ".join(sents))
    if total <= cap or len(sents) <= 3:
        return sents
    keep = list(sents)
    while len(keep) > 3 and count_words(" ".join(keep)) > cap:
        mids = list(range(1, len(keep) - 1))
        named = {i for i in mids if any(n in keep[i] for n in names)}
        pool = [i for i in mids if i not in named] or mids
        # drop the middle-most candidate (keeps the section's arc: start,
        # middle, end); tie-break toward the longer sentence (bigger saving)
        center = (len(keep) - 1) / 2.0
        drop = min(pool, key=lambda i: (abs(i - center),
                                        -count_words(keep[i])))
        keep.pop(drop)
    return keep


def _condense_section(
    cfg_llm: dict,
    sents: list[str],
    cap: int,
    names: list[str],
) -> list[str] | None:
    """Rewrite an over-delivered section to fit its footage budget WITHOUT
    losing the story -- the storytelling-safe form of the visual-match trim.

    Dropping whole middle sentences (the backstop in
    :func:`_fit_section_to_footage`) can leave jumps in the causal chain.
    This instead asks the writer to TIGHTEN the same section: same beats,
    same order, merged sentences, shorter phrasing. Returns ``None`` when
    the call fails or the rewrite still overshoots, so the caller can fall
    back to the mechanical trim.
    """
    if not sents:
        return None
    names_line = (
        "Keep every one of these names: " + ", ".join(names) + "."
        if names else ""
    )
    user = (
        "You wrote this section of recap narration:\n\n"
        + "\n".join(f"- {s}" for s in sents)
        + f"\n\nBut the film footage behind this section only has room for "
        f"about {cap} words at normal playback speed. Rewrite the section "
        f"as AT MOST {cap} words:\n"
        "- SAME story beats, SAME order, SAME cause-and-effect -- the "
        "viewer must be able to follow the chain with no jumps\n"
        "- MERGE and TIGHTEN sentences (that is how an editor shortens a "
        "paragraph), do not simply delete story beats\n"
        "- keep the section's opening and ending meaning (the surrounding "
        "sections hand off to them)\n"
        + names_line
        + '\n\nRespond with ONLY a JSON object: {"sentences": ["...", ...]}'
    )
    try:
        raw = llm.complete(
            cfg_llm.get("provider", ""),
            cfg_llm.get("model", ""),
            SYSTEM_RECAP_BEATS,
            user,
            base_url=cfg_llm.get("base_url"),
            json_mode=True,
            max_tokens=_out_tokens_for_words(cap),
        )
        new = _parse_segment(raw)
    except Exception:
        return None
    if not new or len(new) < 2:
        return None
    if count_words(" ".join(new)) > cap:
        return None
    return new


def generate_segmented_script(
    chunk_summaries: list[dict],
    cfg_llm: dict,
    target_words: int,
    *,
    words_per_minute: int = 150,
    progress=None,
    lang_name: str = "English",
    sign_off: bool = True,
    visual_match: bool = True,
) -> list[dict]:
    """Write the recap chunk-by-chunk, in film order, hitting the word target.

    ``chunk_summaries`` — ``[{"index", "start", "end", "summary",
    "beats": [{"t": seconds, "text": ...}, ...]}]`` in order (the ``beats``
    list is optional; when present every beat keeps its film time).

    ``lang_name`` — the narration language. The section writer is told to
    produce its sentences in that language, so a native recap (Arabic/Spanish
    written straight from that language's subtitles) never leaks English prose.
    English stays byte-identical to before (no instruction appended).

    ``sign_off`` — append the channel outro line ("If you enjoyed the video,
    don't forget to leave a like...") after the story ends, the way real recap
    channels close every video. Skipped when the writer already ended with one.

    Returns ``[{"sentence", "film_start", "film_end"}]``: every sentence knows
    which moment of film it describes (see ``_anchor_windows``), so the visual
    timeline is chronological by construction and the footage shown matches
    the moment being narrated — not just its whole chunk.
    """
    usable = [c for c in chunk_summaries if (c.get("summary") or "").strip()]
    if not usable:
        return []

    is_en = not (lang_name and lang_name.lower() != "english")
    exemplar_block = EN_STYLE_BLOCK if is_en else ""
    lang_instr = ""
    if not is_en:
        lang_instr = (
            f"\n\nLanguage: write the narration entirely in {lang_name} — "
            f"natural, idiomatic {lang_name} for a {lang_name}-speaking recap "
            "audience. Keep character names recognizable (use their common "
            f"{lang_name} forms, consistently)."
        )

    def _weight(c: dict) -> float:
        """Legacy weighting (visual_match=false): budget by beat count."""
        beats = c.get("beats") or []
        if beats:
            n = len([b for b in beats if (b.get("text") or "").strip()])
            return float(max(n, 1))
        return float(max(len((c.get("summary") or "").split()), 20))

    # ---- VISUAL MATCH: size each section's narration to its footage ------
    # How many seconds of DISTINCT film each section can show at 1x. Windows
    # overlap (chunk N's tail is re-covered by chunk N+1's head), so a
    # section's own footage is the STEP between window starts; the last
    # section owns its full window.
    starts_ = [float(c.get("start", 0.0)) for c in usable]
    ends_ = [float(c.get("end", 0.0)) for c in usable]
    film_secs: list[float] = []
    zone_hi: list[float] = []
    for i in range(len(usable)):
        win = max(ends_[i] - starts_[i], 1.0)
        step = starts_[i + 1] - starts_[i] if i + 1 < len(usable) else win
        if not (1.0 < step < win):
            step = win
        film_secs.append(float(step))
        zone_hi.append(starts_[i] + film_secs[-1])
    # 0.4 safety factor: _anchor_windows splits every gap between neighbouring
    # anchors at the midpoint (so footage never repeats), so a sentence can
    # only use lead + gap/2 of film -- 1x-safe narration tops out around half
    # the film time, and 0.4 leaves margin for TTS pauses and rate variance.
    cap_words = [s / 60.0 * max(words_per_minute, 60) * 0.4 for s in film_secs]

    if visual_match:
        budgets = _visual_matched_budgets(target_words, film_secs, cap_words)
        print(f"  * visual match: {len(usable)} sections budgeted by film "
              f"time ({sum(budgets)} words; dense sections capped at their "
              "1x footage)")
    else:
        weights = [_weight(c) for c in usable]
        wsum = float(sum(weights)) or 1.0
        budgets = [max(40, int(round(target_words * (w / wsum))))
                   for w in weights]

    out: list[dict] = []
    tail = ""  # last sentences of the previous section, for continuity
    for pos, c in enumerate(usable):
        budget = int(budgets[pos])
        nsent = max(3, int(round(budget / 17)))
        t0, t1 = float(c.get("start", 0.0)), float(c.get("end", 0.0))
        beats = c.get("beats") or []
        is_first, is_last = pos == 0, pos == len(usable) - 1

        # Section hand-off instructions. The OPENING starts inside the film's
        # first scene (no title/hook), the FINAL section lands the channel's
        # ending formula, and everything in between picks up from the previous
        # section's last line without repeating it.
        parts: list[str] = []
        if is_first:
            parts.append(
                "This is the OPENING of the recap. Start inside the film's "
                "very first scene, the way recap channels open (\"It all "
                "begins ...\" is a typical first move). No title, no hook, no "
                "channel intro — straight into the story."
            )
        if tail:
            parts.append(
                f'This section continues directly from: "{tail}" — pick up '
                "from there without repeating it."
            )
        if is_last:
            parts.append(
                "This is also the FINAL section of the recap: land the "
                "story's last beat, then close with the channel's ending "
                "formula — the final sentence should end along the lines of "
                "\"... and that's how the movie comes to an end.\" If the "
                "beats include a post-credits scene, narrate it just before "
                "that closing line (\"But there's still a post-credit "
                "scene.\"). Do NOT ask viewers to like or subscribe — the "
                "outro is added separately."
            )
        continuity = "\n".join(parts) if parts else (
            "Continue the story in order."
        )

        # Characters/places this section MUST name — pulled from the beats
        # themselves so the writer sees the exact list (and the retry below
        # verifies they survived into the narration).
        names = _proper_nouns(_fmt_beat_lines(c), limit=10)
        names_block = (
            "NAMES THAT MUST BE SPOKEN IN THIS SECTION (viewers cannot follow "
            "\"he\"/\"she\"/\"the man\" — use these names, each at least "
            "once): " + ", ".join(names) + "."
            if names else ""
        )

        user = PROMPT_SEGMENT_JSON.format(
            t0=_fmt_clock(t0), t1=_fmt_clock(t1), budget=budget, nsent=nsent,
            continuity=continuity, beats=_fmt_beat_lines(c),
            names_block=names_block,
        )
        if exemplar_block:
            user += exemplar_block
        if lang_instr:
            user += lang_instr
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

        # Names check: if the writer dropped most of the section's characters
        # ("she goes to help" instead of "Jessie rides Bullseye"), one retry
        # with the missing names spelled out.
        if names and sents:
            missing = _missing_names(names, " ".join(sents))
            if len(missing) >= max(2, int(len(names) * 0.5)):
                fixed = llm.complete(
                    cfg_llm.get("provider", ""), cfg_llm.get("model", ""),
                    SYSTEM_RECAP_BEATS,
                    user + (
                        "\n\nIMPORTANT: your narration dropped these names: "
                        + ", ".join(missing)
                        + ". Rewrite the section keeping ALL of them (a name "
                        "can appear as the bare name or inside \"a <role> "
                        "named <Name>\"). Same JSON shape, same word budget."
                    ),
                    base_url=cfg_llm.get("base_url"),
                    json_mode=True,
                    max_tokens=_out_tokens_for_words(budget),
                )
                retry = _parse_segment(fixed)
                if retry and len(retry) == len(sents) and \
                        len(_missing_names(names, " ".join(retry))) < len(missing):
                    sents = retry

        # English-only punch-up pass: same sentence count, spoken style.
        if sents and is_en:
            sents = _polish_section(cfg_llm, sents, exemplar_block, names)

        # VISUAL MATCH hard fit: the footage budget is a ceiling, not a
        # suggestion. Writers routinely over-deliver and the polish pass may
        # add ~30% more words on top -- an over-length section is exactly
        # what makes every window shorter than its sentence, forcing slow
        # motion: the narration then runs ahead of the picture. First ask
        # the writer to CONDENSE the section (same story, tighter wording --
        # no jumps in the causal chain); only if that fails, mechanically
        # trim the least-essential middle sentences as a backstop.
        if visual_match and sents:
            _cap = max(int(cap_words[pos]), 40)
            if count_words(" ".join(sents)) > _cap:
                _got = count_words(" ".join(sents))
                _fitted = _condense_section(cfg_llm, sents, _cap, names)
                if _fitted is not None:
                    print(f"    ... section {pos + 1}/{len(usable)}: writer "
                          f"returned {_got} words for a {_cap}-word footage "
                          f"budget -- condensed to "
                          f"{count_words(' '.join(_fitted))} words (story "
                          "kept) so it plays at 1x")
                    sents = _fitted
                else:
                    _fitted = _fit_section_to_footage(sents, _cap, names)
                    if len(_fitted) != len(sents):
                        print(f"    ... section {pos + 1}/{len(usable)}: "
                              f"writer returned {_got} words for a "
                              f"{_cap}-word footage budget -- trimmed to "
                              f"{len(_fitted)} sentences "
                              f"({count_words(' '.join(_fitted))} words) "
                              "so it plays at 1x")
                        sents = _fitted

        # Anchor each sentence to the film moment(s) it narrates (beat
        # timecodes) instead of giving the whole chunk to every sentence.
        # English aligns each sentence to the beat it actually describes
        # (embedding + monotone DP); every language falls back to an even
        # positional map. With visual_match on, the anchors are then PACED
        # (see _paced_anchors) so each sentence's footage window is at least
        # as long as the sentence itself -- the guarantee that the timeline
        # can play everything at 1x, no slow motion.
        if sents:
            try:
                _lead = float(os.environ.get("RECAP_ANCHOR_LEAD", "0.8"))
            except (TypeError, ValueError):
                _lead = 0.8
            try:
                _tail = float(os.environ.get("RECAP_ANCHOR_TAIL", "6.0"))
            except (TypeError, ValueError):
                _tail = 6.0
            anchors = _sentence_anchor_values(sents, beats) if is_en else None
            if anchors is None:
                times = sorted(
                    float(b["t"]) for b in (beats or [])
                    if b.get("t") is not None and (b.get("text") or "").strip()
                )
                n_s, n_b = len(sents), len(times)
                if n_b >= 1 and n_s >= 1:
                    anchors = [
                        times[min(n_b - 1,
                                  int(round(k * (n_b - 1) / max(n_s - 1, 1))))]
                        for k in range(n_s)
                    ]
                else:
                    anchors = [
                        t0 + (k + 0.5) * max(t1 - t0, 1.0) / max(n_s, 1)
                        for k in range(n_s)
                    ]
            if visual_match and anchors:
                anchors = _paced_anchors(
                    anchors, sents, t0, min(t1, zone_hi[pos]),
                    words_per_minute, lead=_lead,
                )
            wins = _anchor_windows(
                t0, t1, beats, len(sents),
                anchor_values=anchors, lead=_lead, tail=_tail,
            )
        else:
            wins = []
        for s, (lo, hi) in zip(sents, wins):
            out.append({"sentence": s, "film_start": lo, "film_end": hi})
        if sents:
            tail = " ".join(sents[-2:])   # two sentences of carry-over context

        if progress:
            progress(pos + 1, len(usable), count_words(" ".join(sents)), budget)

    # Final read-through (English): one pass over the WHOLE script for seam
    # quality, name consistency and rhythm — with the same count lock, so
    # every sentence keeps the film window it was generated with.
    if is_en and out:
        all_names = _proper_nouns(
            "\n".join(_fmt_beat_lines(c) for c in usable), limit=15
        )
        before = [o["sentence"] for o in out]
        polished = _global_polish(cfg_llm, before, all_names)
        if polished != before and len(polished) == len(out):
            for o, old_s, new_s in zip(out, before, polished):
                # Keep the polished line only if it still fits its footage
                # window: the windows were sized for the pre-polish sentence
                # (with a 1.15x TTS margin), so up to 10% growth is safe --
                # beyond that the sentence would outlast its own film and
                # the picture would fall behind the narration.
                if count_words(new_s) <= max(count_words(old_s), 1) * 1.1 + 2:
                    o["sentence"] = new_s

    _append_sign_off(out, lang_name=lang_name, enabled=sign_off)

    return out


# The channel outro real recap videos close with, per narration language.
# Appended deterministically (never left to the model) so every video ends
# the same way the reference channel's do. Set narration.sign_off: false
# (or RECAP_SIGN_OFF=0) to skip it.
SIGN_OFF_LINES: dict[str, str] = {
    "en": (
        "If you enjoyed the video, don't forget to leave a like, subscribe, "
        "and turn on notifications. That's all for today. See you next time."
    ),
    "zh": (
        "如果你喜欢这个视频，别忘了点赞、订阅并打开通知。"
        "今天就到这里，我们下次再见。"
    ),
    "ar": (
        "إذا استمتعت بالفيديو، لا تنسَ الإعجاب والاشتراك وتفعيل التنبيهات. "
        "هذا كل شيء لليوم، نراكم في المرة القادمة."
    ),
    "es": (
        "Si te gustó el video, no olvides darle like, suscribirte y activar "
        "las notificaciones. Eso es todo por hoy. Nos vemos la próxima vez."
    ),
}

_SIGN_OFF_TELLS = (
    "subscribe", "see you next time", "see you in the next",
    "like and subscribe", "leave a like", "next time",
)


def _append_sign_off(segments: list[dict], *, lang_name: str, enabled: bool) -> None:
    """Close the recap with the channel outro, like the reference videos.

    The outro reuses the last beat's film window (the closing moments of the
    film stay on screen under it). Skipped when the writer already produced
    something outro-like, so we never double up.
    """
    if not enabled or not segments:
        return
    from . import languages as _langs

    code = "en"
    key = (lang_name or "").strip().lower()
    for c in _langs.SUPPORTED:
        if key == _langs.NAMES.get(c, "").lower() or key.startswith(c):
            code = c
            break
    line = SIGN_OFF_LINES.get(code, SIGN_OFF_LINES["en"])
    recent = " ".join(s["sentence"] for s in segments[-3:]).lower()
    if any(tell in recent for tell in _SIGN_OFF_TELLS):
        return  # the writer already closed with an outro
    last = segments[-1]
    segments.append(
        {
            "sentence": line,
            "film_start": last.get("film_start", 0.0),
            "film_end": last.get("film_end", 0.0),
        }
    )


def render_prompt(notes: str, target: int, mn: int, mx: int) -> str:
    instructions = STYLE_PRESET.format(target=target, mn=mn, mx=mx)
    return f"{instructions}\n\n=== PLOT NOTES / SUMMARY TO RECAP ===\n\n{notes}\n\n=== END PLOT NOTES ===\n"


DIALOGUE_PRESET = """Be this person: a veteran movie-recap narrator, the voice people binge for \
hours — a storyteller relaying the film, beat by beat, to a friend who missed \
it. Never a reviewer analyzing it. Never mention yourself, never say "I", \
never mention being an AI. Your job is to NARRATE the movie — tell the story \
as it happens, over a montage of the film's own clips. You are NOT reviewing, \
explaining, or describing the film.

Below is the TIMESTAMPED DIALOGUE / TRANSCRIPT of a film (what the characters \
say, with timecodes). Use it, plus any plot summary, to write a single English \
narration that retells the ENTIRE story from opening scene to ending, as one \
continuous, present-tense tale.

How the narration must sound (the recap-channel register):
- SENTENCE RHYTHM: chain several moments into flowing 12-40 word sentences, \
and between them drop short dramatic beats of 2-8 words ("Woody disagrees."). \
Never write many sentences of the same length in a row.
- Vary every opener with real narrator connectors: "Meanwhile,", "At the same \
time,", "Just then,", "Suddenly,", "Not long after,", "A little later,", \
"Before long,", "The next morning,", "That night,", "Back at ...", "Little by \
little,", "Thanks to that,", "As it turns out,", "The moment X ..., Y ...", \
"Determined to ..., X ...", "Without anyone noticing, X ...".
- Introduce new characters/objects with a short appositive ("a mare named \
Almond", "a tablet called Lilypad").
- Report conversations as indirect narration ("She explains that...", "He \
admits that..."); never quote dialogue verbatim.
- Use contractions (it's, she's, doesn't, can't) the way a person talking does.
- Move fast, keep a visible cause-to-effect thread, no filler.
- Present tense throughout, strictly chronological, covering the whole plot \
including the ending. Use consistent character names.
- Never say "the movie", "the film", "the scene shows", "the camera"; "we see" \
is fine occasionally. No analysis, no themes, no review talk. No em-dashes, no \
semicolons, no rhetorical questions to the viewer.
- Total roughly {target} words (between {mn} and {mx}).
- One sentence per line. No heading, title, or trailing notes.
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
    system = NARRATOR_PERSONA
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
    system = NARRATOR_PERSONA
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
