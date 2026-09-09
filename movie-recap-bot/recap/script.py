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


# A short passage in the target voice. It is ORIGINAL writing (no movie, no
# characters from any film) used only to demonstrate the RECAP REGISTER:
# flowing, present-tense prose that chains what happens into full sentences,
# opens differently each time, sums up conversations as narration, and lands a
# light emotional beat. Models copy the ENERGY and rhythm, never the words.
EN_VOICE_EXEMPLAR = (
    "The signal reaches the lighthouse at midnight, a single blinking dot on "
    "an old map. Inside, a young keeper realizes it is coming from the sea. "
    "Wasting no time, she launches the rescue boat and heads out into the "
    "storm. Little by little, the waves grow taller, and her mast lamp becomes "
    "the only light for miles. When she finally reaches the source, she finds "
    "a small raft carrying a family of lost travelers, cold but alive. The "
    "moment they step aboard, the oldest one freezes -- he recognizes the "
    "keeper, the daughter he left behind twenty years ago. Back on shore, the "
    "reunion barely has time to sink in before a second dot blinks to life on "
    "the map, and the night begins all over again."
)

EN_STYLE_BLOCK = (
    "\n\n=== NARRATIVE VOICE -- match this ENERGY and rhythm, never its words "
    "or events ===\n" + EN_VOICE_EXEMPLAR +
    "\n(Study how it opens sentences differently, chains cause to effect, and "
    "sounds like someone talking. Your narration must feel equally spoken -- "
    "never like a list of bullet points.)"
)

SYSTEM_POLISH = (
    "You are the dialogue editor for a movie-recap YouTube channel. You take a "
    "draft narration and rewrite it so it sounds like a person TALKING over "
    "footage -- natural, propulsive, varied -- without changing what happens, "
    "the order of events, or the number of sentences. Never mention being an AI."
)

POLISH_PROMPT = """Below is a DRAFT section of a movie recap, one sentence per array element, in strict story order.

Rewrite it so it reads as natural spoken narration, not generated text:
- Keep EXACTLY {n} sentences ({n} array elements). Never merge two sentences into one and never split one into two -- the video timing depends on it.
- Keep the same events in the same order, and keep every character name and proper noun. Change the WORDS, not the story.
- Sound like a recap narrator TALKING over footage: sentences run about 10 to 30 words and must vary in length; never start three in a row the same way.
- Kill robotic patterns: repeated "<Name> does X. <Name> does Y." listing, generic verbs (goes, gets, has) -> concrete ones (bolts, grabs, shoves, spots).
- Let several moments flow through one sentence when they belong to one continuous action ("She explains that technology has made its way into Bonnie's life too, and that a tablet is taking up all of her attention."); cut a new sentence when the scene or time shifts.
- Vary openers with real-narrator connectors: Meanwhile, At the same time, Just then, Not long after, Back at, Little by little, Determined to, Excited, As it turns out, The moment..., Seeing this.
- Present tense, third person. Never quote dialogue directly -- summarize what characters say as indirect narration ("She explains that ...", "He admits ..."). Never say "the movie", "the film", "the scene shows", "the camera". "We see" is fine occasionally.

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

SYSTEM_RECAP_BEATS = (
    "You are the head writer for a top-tier YouTube movie-recap channel "
    "(the 'Movie Recaps' style). You write narration that is spoken over the "
    "film's own footage. Never mention being an AI. Never quote dialogue directly -- turn what "
    "characters say into narration ('She explains that...', 'He admits...'). "
    "Never say 'the movie', 'the film', 'the scene shows' or 'the camera'. "
    "'We see' is fine once in a while: real recap narrators use it. "
    "Third person, present tense, active verbs, character names. "
    "Deadpan, propulsive, lightly witty. Every sentence is a VISIBLE action. "
    "Sound like a confident narrator TALKING over the footage, not a person "
    "reading bullet points: vary sentence length and the way sentences open, "
    "and let each line flow out of the one before it. Your source beats are a "
    "factual record: describe them accurately and specifically — keep every "
    "character name and proper noun they contain — and never invent events "
    "that are not in the list."
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
- COVER THE SECTION EVENLY: spread your ~{nsent} sentences across the whole
  {t0}→{t1} stretch — sentence 1 about its opening beats, the middle sentences
  about the middle beats, the last about the closing beats — so every scene
  gets narrated and no one moment hogs the section.
- WRITE LIKE A RECAP NARRATOR TALKS, NOT LIKE A SUMMARY LIST. Real channels
  cover several moments inside one flowing sentence: "Jessie asks Woody if
  things are really as bad as they seem. She explains that technology has made
  its way into Bonnie's life too, and that a tablet is taking up all of her
  attention." Chain what belongs together; cut a new sentence when the scene,
  location, or time shifts.
- Sentences should run anywhere from about 10 to 30 words -- long enough to
  flow, short enough to say in one breath -- and MUST vary in length. Never
  write three sentences in a row with the same opener.
- Vary how sentences open, using the connectors real recap narrators use:
  "Wasting no time, ...", "Little by little, ...", "After ..., X ...",
  "Determined to ..., X ...", "Meanwhile, ...", "At the same time, ...",
  "Just then, ...", "Not long after, ...", "Back at ..., ...", "The moment
  X happens, Y ...", "Excited, she ...", "As it turns out, ...", "Seeing
  this, ...", "Without anyone noticing, ...".
- Keep a visible cause -> effect thread: "So they climb onto the roof ... and
  Jessie is shocked to discover ...". Avoid "and then / next" repetition.
- When characters talk, sum up what they SAY as indirect narration ("She
  explains that ...", "He admits that ...", "Woody points out that ...").
  Never quote dialogue directly.
- Be SPECIFIC: keep every character name and proper noun from the beats
  ("Jessie hops onto Bullseye and rides to the twins' house", not "she goes
  to help"). One or two proper nouns per sentence keeps it vivid.
- Present tense, third person. Never say "the movie", "the film", "the scene
  shows", or "the camera". "We see" is acceptable occasionally.
- Strictly chronological; keep every character name consistent; never invent
  events that are not in the beat list.
- No dialogue quotes.
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


def _polish_section(
    cfg_llm: dict, sents: list[str], exemplar_block: str
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
        n=len(sents), exemplar=exemplar_block or EN_VOICE_EXEMPLAR,
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


def generate_segmented_script(
    chunk_summaries: list[dict],
    cfg_llm: dict,
    target_words: int,
    *,
    words_per_minute: int = 150,
    progress=None,
    lang_name: str = "English",
) -> list[dict]:
    """Write the recap chunk-by-chunk, in film order, hitting the word target.

    ``chunk_summaries`` — ``[{"index", "start", "end", "summary",
    "beats": [{"t": seconds, "text": ...}, ...]}]`` in order (the ``beats``
    list is optional; when present every beat keeps its film time).

    ``lang_name`` — the narration language. The section writer is told to
    produce its sentences in that language, so a native recap (Arabic/Spanish
    written straight from that language's subtitles) never leaks English prose.
    English stays byte-identical to before (no instruction appended).

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

    # Distribute the word budget across chunks by how much story each holds:
    # beat count when available (a 20-beat chunk gets more narration room than
    # a 5-beat one even if their summaries are similar in length), otherwise
    # the summary's own word count.
    def _weight(c: dict) -> float:
        beats = c.get("beats") or []
        if beats:
            n = len([b for b in beats if (b.get("text") or "").strip()])
            return float(max(n, 1))
        return float(max(len((c.get("summary") or "").split()), 20))

    weights = [_weight(c) for c in usable]
    wsum = float(sum(weights)) or 1.0

    out: list[dict] = []
    tail = ""  # last sentence of the previous section, for continuity
    for pos, (c, w) in enumerate(zip(usable, weights)):
        budget = max(40, int(round(target_words * (w / wsum))))
        nsent = max(3, int(round(budget / 17)))
        t0, t1 = float(c.get("start", 0.0)), float(c.get("end", 0.0))
        beats = c.get("beats") or []

        continuity = (
            f'This section continues directly from: "{tail}" — pick up from there '
            "without repeating it."
            if tail
            else "This is the OPENING of the recap. Start with the very first thing "
            "that happens on screen. Do not write a title or a hook."
        )

        user = PROMPT_SEGMENT_JSON.format(
            t0=_fmt_clock(t0), t1=_fmt_clock(t1), budget=budget, nsent=nsent,
            continuity=continuity, beats=_fmt_beat_lines(c),
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

        # English-only punch-up pass: same sentence count, spoken style.
        if sents and is_en:
            sents = _polish_section(cfg_llm, sents, exemplar_block)

        # Anchor each sentence to the film moment(s) it narrates (beat
        # timecodes) instead of giving the whole chunk to every sentence. For
        # English, first align each sentence to the beat it actually describes
        # (embedding + monotone DP); other languages keep the positional map.
        if sents:
            anchors = _sentence_anchor_values(sents, beats) if is_en else None
            try:
                _lead = float(os.environ.get("RECAP_ANCHOR_LEAD", "0.8"))
            except (TypeError, ValueError):
                _lead = 0.8
            try:
                _tail = float(os.environ.get("RECAP_ANCHOR_TAIL", "6.0"))
            except (TypeError, ValueError):
                _tail = 6.0
            wins = _anchor_windows(
                t0, t1, beats, len(sents),
                anchor_values=anchors, lead=_lead, tail=_tail,
            )
        else:
            wins = []
        for s, (lo, hi) in zip(sents, wins):
            out.append({"sentence": s, "film_start": lo, "film_end": hi})
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
