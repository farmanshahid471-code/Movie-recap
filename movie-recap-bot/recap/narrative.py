"""The storytelling layer: WHO is speaking and WHAT telling a story means.

Why this module exists
----------------------
Every writer prompt in this project used to ask the model, directly or
indirectly, to *describe what happens in this beat*::

    "Describe ONLY what happens in this beat. Do not reference earlier or
     later beats."                         (beats.BEAT_SYSTEM_PROMPT)

A model told to describe a beat describes the picture: "A man in a suit walks
down a hallway, holding a file. The camera follows him." The narration comes
out as a shot-by-shot report — every sentence true, every sentence lifeless,
and nothing chaining into anything. That is the "it's describing the scene"
complaint, and it is a *prompt* bug, not a model bug: the prompt asked for a
description and got one.

This module holds the missing half of the craft, shared by every writer in
the pipeline (``story.py`` for the beat path, ``script.py`` for the chunk
path):

* ``STORY_RULES``       — what "tell this as a story" means, with a worked
                          bad/good pair, so the instruction cannot be read as
                          "describe it more vividly".
* ``act_for``           — where a unit sits in the film (setup → climax →
                          resolution) and what the story must be DOING there,
                          so tension rises instead of the narration being a
                          flat list of equally-weighted events.
* ``description_flags`` — a cheap, deterministic detector for sentences that
                          describe the frame instead of telling the story,
                          used to gate a targeted rewrite and to report a
                          measured "how much of this is description" number.
* ``STORY_RHYTHM``      — the sentence-rhythm half (long chain, short punch).

Nothing here talks to an LLM; it is pure text and pure functions so it can be
unit-tested and reused by both writer paths.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# The craft: telling the story, not describing the picture
# ---------------------------------------------------------------------------

# A worked pair is worth more than any number of adjectives. The BAD example
# is deliberately the register the pipeline used to produce (frame-by-frame,
# presentational, no want, no consequence); the GOOD example keeps every fact
# of the BAD one and turns it into story.
BAD_GOOD_PAIR = """BAD (a description of shots — this is what we must never write):
"A man in a suit walks down a hallway. He is holding a file. The camera follows him. He enters an office. A woman is sitting at the desk."

GOOD (the same facts, told as story):
"The lawyer slips past reception with the file that could bury his own firm, and walks straight into the office of the woman who hired him."
"""

STORY_RULES = """HOW A RECAP NARRATOR TELLS A STORY (this is the whole job)

You are not captioning the picture. You are telling a friend what happens in
this film, in order, and why it matters. If the audience closed their eyes,
they should still follow the story.

1. PEOPLE WANT THINGS, AND THINGS GET IN THE WAY.
   Every sentence needs a person doing something, wanting something, or
   learning something. "Marco walks into the diner" is half a sentence.
   "Marco walks into the diner to collect the money he is owed, and the
   waitress tells him his brother already took it" is the story.
2. CAUSE LEADS TO CONSEQUENCE.
   Chain the moments: each event happens BECAUSE of the one before it. Reach
   for the connectors a storyteller actually uses — "So", "That's when",
   "Because of that", "The moment she does", "Before long", "By morning",
   "Which is exactly when", "That promise comes back to haunt him".
3. NAME THE WANT AND THE OBSTACLE.
   What is this person after in this stretch of the film, and who or what
   stands in the way? An audience leans in when they know what winning and
   losing mean.
4. RAISE THE STAKES.
   Trouble grows. Track what is risked, lost, or revealed, and let the story
   tighten toward its climax. Early events can be lighter and quicker; the
   back half should feel like the walls closing in.
5. EMOTION COMES FROM ACTION, NEVER FROM LABELS.
   Never write "this is sad" or "it is tense". Show the detail that makes it
   so: "He keeps the second helmet in the truck, where she used to sit."
6. LET A SHORT BEAT LAND.
   After a long, flowing sentence, drop a two-to-eight word punch that hits:
   "It doesn't work.", "She's already gone.", "Nobody comes for him."
7. PAY OFF WHAT YOU SET UP.
   When something returns, call it back by name: "the same knife", "the
   promise he made at the gas station". That is what makes a recap feel like
   one story and not twelve summaries stapled together.
8. STAY INSIDE THE STORY.
   No "the movie", no "the film", no "the scene", no "the camera", no "next
   up", no review talk, no explaining what the film is doing. Never address
   the viewer, never mention yourself.
9. NEVER INVENT.
   Use only the events in the facts you are given. You may infer a feeling or
   a motive from what a character plainly does on screen; you may NOT invent
   an event, a line of dialogue, a relationship, or an outcome.

""" + BAD_GOOD_PAIR + """
A DESCRIPTION OF THE FRAME IS NOT A SENTENCE IN A STORY. If a line only says
what a shot looks like — "A dark hallway.", "We see a table covered in
papers.", "The camera pans across the city." — either cut it or fold it into
an action: "He searches the empty office for the file nobody wanted him to
find." Never write "we see", "is seen", "is shown", "there is", or "the scene
shows".
"""

# The same rules, tightened for prompts that already carry a lot of context
# (the chunk path spends tokens on beats, names and continuity blocks).
STORY_RULES_COMPACT = """TELL THE STORY, NEVER DESCRIBE THE PICTURE.
- People want things and things get in the way: every sentence has someone
  doing, wanting or learning something, and the consequence matters.
- Chain cause into effect ("So", "That's when", "Because of that", "The
  moment she does", "Before long", "By morning"). Each event happens BECAUSE
  of the one before it.
- Raise the stakes as the story runs: what is risked, lost or revealed.
- Emotion comes from action, never from labels — show the detail that makes
  it sad or tense instead of saying it is.
- After long sentences, drop a short 2-8 word beat that lands.
- Pay off what was set up earlier by calling it back ("the same knife", "the
  promise he made").
- NEVER write a sentence that only says what a shot looks like ("A dark
  hallway.", "We see a table covered in papers.", "The camera pans across
  the city."). Fold it into an action or cut it. No "we see", "is seen",
  "there is", "the scene shows", "the camera", "the film".
- Never invent events. Infer a motive or feeling from what a character
  plainly does; invent nothing.
"""

STORY_RHYTHM = """SENTENCE RHYTHM (a narrator breathes; a report does not)
- Chain several moments into one flowing 12-40 word sentence, then drop a
  2-8 word beat. Long, long, short. Never write many sentences of the same
  length in a row.
- Open each line differently — "Meanwhile,", "Just then,", "That night,",
  "Thanks to that,", "As it turns out,", "The moment X, Y", "Determined to X,
  she Y", "Before long," — never the same opener twice in a row, and never
  the same sentence shape twice in a row.
- Report conversations as indirect speech ("She admits that the file is
  forged.", "He promises to come back for her.") — never quote dialogue.
- Use contractions, present tense, third person, and the concrete verb
  (bolts, shoves, slips, catches), never the generic one (goes, gets, has).
"""

# What the narration must NOT do, phrased for a rewrite pass. Short and
# concrete: the repair call gets flagged sentences, not the whole script.
REWRITE_RULES = """Rewrite the flagged lines so they TELL THE STORY instead of describing the
shot. Keep every fact, keep the same order, keep the length close to the
original, and keep the speaker's voice. What changes is the shape of the
sentence:

- give it a person who WANTS something, and put the consequence in it;
- delete picture-only filler ("we see", "the camera", "a dark room", "there
  is", "is seen") or fold it into the action;
- connect it to the line before it with a real storyteller connector
  ("So", "That's when", "Because of that", "The moment she does");
- keep the short punch beats short — do not inflate them into long sentences.
"""


# ---------------------------------------------------------------------------
# Where we are in the film: what the story must be DOING here
# ---------------------------------------------------------------------------

# (upper bound of position, label, what this stretch of narration must do)
_ACTS: tuple[tuple[float, str, str], ...] = (
    (0.04, "opening",
     "OPENING. Drop the viewer into the first scene of the film, mid-motion, "
     "the way a recap channel opens. Establish who we are following and what "
     "their ordinary world looks like in a handful of sentences. No title, no "
     "hook line, no 'in this video'."),
    (0.18, "setup",
     "SETUP. The world, the people, and the flaw or want that will drive "
     "everything. Plant what the story will later pay off (a promise, an "
     "object, a warning). Keep momentum: the inciting problem should already "
     "be visible by the end of this stretch."),
    (0.40, "rising",
     "RISING ACTION. The problem grows teeth. Every beat should cost someone "
     "something or close a door. Show the character choosing, failing, and "
     "choosing again."),
    (0.58, "midpoint",
     "MIDPOINT. A revelation or reversal changes what the story is about — "
     "what they thought they knew is wrong, or the danger turns personal. "
     "Let this land hard, then immediately raise the price."),
    (0.76, "complication",
     "COMPLICATION. The worst stretch: allies lost, plans broken, the truth "
     "out. Tighten the screw here — short beats, fast cuts in the telling, no "
     "breathing room."),
    (0.93, "climax",
     "CLIMAX. Everything that was set up collides. Narration should be the "
     "tightest of the film: short sentences, hard verbs, no new information "
     "except what is needed to land the confrontation."),
    (1.01, "resolution",
     "RESOLUTION. Aftermath and the final emotional beat — what changed, who "
     "survived, what it cost. Slow down, land on a concrete detail rather "
     "than a moral, then close the story."),
)


def act_for(position: float) -> tuple[str, str]:
    """Return ``(label, instruction)`` for a 0-1 position through the film."""
    try:
        p = float(position)
    except (TypeError, ValueError):
        p = 0.0
    p = min(max(p, 0.0), 1.0)
    for limit, label, instr in _ACTS:
        if p < limit:
            return label, instr
    return _ACTS[-1][1], _ACTS[-1][2]


# ---------------------------------------------------------------------------
# Description lint: catch a sentence that describes the frame, not the story
# ---------------------------------------------------------------------------

# Hard "this is the film talking" tells. A narrator never says these.
_SCENE_TALK = re.compile(
    r"\b(the|this|that)\s+(scene|shot|frame|sequence|montage|image|footage|"
    r"movie|film|camera|screen|audience|viewer|director|plot|storyline)\b"
    r"|\b(we|you)\s+(see|watch|follow|cut|pan|hear)\b"
    r"|\b(camera|cinematography|score|soundtrack)\b"
    r"|\bnext\s+(up|scene)\b"
    r"|\bcuts?\s+to\b",
    re.I,
)

# Passive presentational description: the frame is the subject of the
# sentence, not a person with a goal.
_PASSIVE_FRAME = re.compile(
    r"\b(is|are|was|were)\s+(seen|shown|revealed|displayed|visible|pictured)\b"
    r"|\bcan\s+be\s+seen\b"
    r"|\bthere\s+(is|are)\b",
    re.I,
)

# "A man in a suit is standing in a hallway." — subject has no agency.
_STATIVE_OPEN = re.compile(
    r"^\s*(?:a|an|the)\s+(?:[a-z]+\s+){0,3}[a-z]+\s+"
    r"(?:is|are|was|were)\s+"
    r"(?:standing|sitting|lying|waiting|walking|holding|wearing|carrying|"
    r"looking|staring|watching|dressed|surrounded|located|positioned)\b",
    re.I,
)

# A sentence with no finite verb doing work at all ("The empty hallway.")
_PICTURE_NOUN_OPEN = re.compile(
    r"^\s*(?:a|an|the|two|three|several|many|rows? of)\b[^.!?]{0,60}[.!?]\s*$",
    re.I,
)

# Visual-only adjectives used as the point of the sentence ("a beautiful,
# atmospheric shot of the harbour").
_VISUAL_ONLY = re.compile(
    r"\b(beautiful|stunning|gorgeous|breathtaking|atmospheric|cinematic|"
    r"moody|glowing|shimmering|scenic|aerial|wide shot|close-?up|vista)\b",
    re.I,
)

_FLAG_NAMES = {
    "scene_talk": _SCENE_TALK,
    "passive_frame": _PASSIVE_FRAME,
    "stative_open": _STATIVE_OPEN,
    "picture_only": _PICTURE_NOUN_OPEN,
    "visual_only": _VISUAL_ONLY,
}


def description_flags(sentence: str) -> list[str]:
    """Names of the description-mode patterns a sentence trips (possibly none).

    Deterministic and cheap on purpose: callers use it to count how much of a
    script is description, to pick the sentences worth a rewrite call, and to
    fail a test when the prompt regresses. A sentence can be flagged more
    than once; the list is de-duplicated but order is stable.
    """
    text = (sentence or "").strip()
    if not text:
        return []
    return [name for name, rx in _FLAG_NAMES.items() if rx.search(text)]


def looks_like_description(sentence: str) -> bool:
    """True when a sentence is (mainly) describing the picture."""
    return bool(description_flags(sentence))


def story_score(sentences: list[str]) -> float:
    """Fraction of sentences that read as story rather than description.

    1.0 = nothing trips the lint. Reported per run so the "narration is just
    describing the scene" regression becomes a number in the log instead of a
    feeling after watching the video.
    """
    items = [s for s in (sentences or []) if (s or "").strip()]
    if not items:
        return 1.0
    clean = sum(0 if looks_like_description(s) else 1 for s in items)
    return clean / float(len(items))


def description_report(sentences: list[str], worst: int = 5) -> dict:
    """``{"sentences", "flagged", "ratio", "score", "examples"}`` for logging."""
    items = [s.strip() for s in (sentences or []) if (s or "").strip()]
    flagged = [s for s in items if looks_like_description(s)]
    return {
        "sentences": len(items),
        "flagged": len(flagged),
        "ratio": (len(flagged) / len(items)) if items else 0.0,
        "score": story_score(items),
        "examples": flagged[:worst],
    }


# ---------------------------------------------------------------------------
# Small formatting helpers shared by the writer prompts
# ---------------------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    """``83.4 -> "01:23"`` (hours only when the film is long enough)."""
    try:
        secs = max(float(seconds), 0.0)
    except (TypeError, ValueError):
        secs = 0.0
    h = int(secs) // 3600
    m, s = divmod(int(secs) % 3600, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_window(start: float, end: float) -> str:
    """``(83.4, 121.0) -> "01:23-02:01"`` — the range a unit's footage covers."""
    return f"{format_timestamp(start)}-{format_timestamp(end)}"


def ordinal(n: int) -> str:
    """``3 -> "3rd"`` for "the 3rd of 17 story units" style context."""
    try:
        i = int(n)
    except (TypeError, ValueError):
        return str(n)
    if 10 <= i % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suffix}"
