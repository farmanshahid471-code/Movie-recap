"""Tests for the narration-sync + recap-register additions.

Covers:
  * recap/align.py    — whisper word -> sentence matching, cue refinement,
                        cue building for providers that return none
  * recap/timeline.py — micro-cuts landing on measured word boundaries
  * recap/script.py   — the channel sign-off outro

Run:  python tests/test_narration_sync.py
(no third-party deps needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap import align, script, timeline  # noqa: E402
from recap.tts import TimedCue  # noqa: E402


# --------------------------------------------------------------------------
# align.py — word-to-sentence matching
# --------------------------------------------------------------------------
def test_map_words_to_sentences_perfect_match() -> None:
    sentences = [
        "It all begins on a deserted island.",
        "Wasting no time, he starts activating them.",
    ]
    words = [
        ("It", 0.00, 0.10), ("all", 0.12, 0.20), ("begins", 0.22, 0.45),
        ("on", 0.47, 0.52), ("a", 0.54, 0.56), ("deserted", 0.58, 0.95),
        ("island.", 1.00, 1.40),
        ("Wasting", 2.00, 2.40), ("no", 2.42, 2.50), ("time,", 2.52, 2.80),
        ("he", 2.82, 2.90), ("starts", 2.95, 3.30), ("activating", 3.35, 3.90),
        ("them.", 3.95, 4.20),
    ]
    groups = align.map_words_to_sentences(words, sentences)
    assert groups is not None, "perfect transcription must match"
    assert [w[0] for w in groups[0]] == [
        "It", "all", "begins", "on", "a", "deserted", "island."
    ]
    assert [w[0] for w in groups[1]] == [
        "Wasting", "no", "time,", "he", "starts", "activating", "them."
    ]
    print("ok: perfect word -> sentence mapping")


def test_map_words_to_sentences_tolerant_match() -> None:
    """Whisper noise (number forms, merged tokens) must not break the match."""
    sentences = [
        "The bus reaches the airport at 5 in the morning.",
        "Woody disagrees.",
    ]
    words = [  # whisper said "five" for "5" and dropped "the" once
        ("The", 0.0, 0.1), ("bus", 0.12, 0.3), ("reaches", 0.32, 0.6),
        ("the", 0.62, 0.7), ("airport", 0.72, 1.1), ("at", 1.12, 1.2),
        ("five", 1.22, 1.5), ("in", 1.52, 1.6), ("morning.", 1.62, 2.0),
        ("Woody", 2.4, 2.8), ("disagrees.", 2.85, 3.4),
    ]
    groups = align.map_words_to_sentences(words, sentences)
    assert groups is not None
    assert groups[1] and groups[1][0][0] == "Woody" \
        and groups[1][-1][0] == "disagrees."
    print("ok: tolerant mapping (number form + dropped word)")


def test_map_words_to_sentences_rejects_garbage() -> None:
    words = [("blah", float(i), float(i) + 0.1) for i in range(6)]
    groups = align.map_words_to_sentences(
        words, ["Completely unrelated sentence here."]
    )
    assert groups is None, "an unrelated transcript must be rejected"
    print("ok: garbage transcript rejected")


def test_refine_cues_uses_measured_times() -> None:
    """A provider's sloppy cue times must be pulled onto the measured words."""
    sentences = ["First sentence here.", "Second one."]
    provider_cues = [
        TimedCue(sentences[0], 0.0, 5.0),   # guessed: way too long
        TimedCue(sentences[1], 5.0, 8.0),   # guessed: starts way too late
    ]
    words = [
        ("First", 0.20, 0.45), ("sentence", 0.50, 0.90), ("here.", 0.95, 1.30),
        ("Second", 2.00, 2.35), ("one.", 2.40, 2.70),
    ]
    refined = align.refine_cues(provider_cues, sentences, words)
    assert refined[0].start <= 0.20 + 1e-9          # measured start (pre-roll only)
    assert abs(refined[0].end - 1.30) < 1e-9        # measured end
    assert abs(refined[1].start - (2.00 - align.DEFAULT_AUDIO_PRE_ROLL)) < 1e-9
    assert abs(refined[1].end - 2.70) < 1e-9
    assert refined[0].words and refined[1].words    # word timings attached
    print("ok: cue times refined onto measured words")


def test_cues_from_words_replaces_proportional_guess() -> None:
    """openai/elevenlabs providers: cues built straight from measurements."""
    sentences = ["Alpha line.", "Beta line.", "Gamma line."]
    words = [
        ("Alpha", 0.10, 0.40), ("line.", 0.45, 0.70),
        ("Beta", 1.00, 1.30), ("line.", 1.35, 1.60),
        ("Gamma", 2.00, 2.35), ("line.", 2.40, 2.65),
    ]
    cues = align.cues_from_words(sentences, words, audio_span=3.0)
    assert cues is not None and len(cues) == 3
    assert abs(cues[1].start - (1.00 - align.DEFAULT_AUDIO_PRE_ROLL)) < 1e-9
    assert all(cues[i].end <= cues[i + 1].start + 1e-9 for i in range(2))
    print("ok: cues built from words for timing-less providers")


# --------------------------------------------------------------------------
# timeline.py — word-locked micro-cuts
# --------------------------------------------------------------------------
def test_micro_cuts_land_on_word_boundaries() -> None:
    """Intra-sentence shot changes must happen at clause boundaries."""
    text = ("Meanwhile, Jessie climbs onto the roof and looks out, "
            "but the kids are gone.")
    # spoken: "Meanwhile," ends 0.85; "and" starts 3.10; "but" starts 4.40
    words = [
        ("Meanwhile,", 0.10, 0.85), ("Jessie", 0.90, 1.30),
        ("climbs", 1.35, 1.80), ("onto", 1.85, 2.10), ("the", 2.15, 2.30),
        ("roof", 2.35, 2.70), ("and", 3.10, 3.30), ("looks", 3.35, 3.70),
        ("out,", 3.75, 4.00), ("but", 4.40, 4.60), ("the", 4.65, 4.80),
        ("kids", 4.85, 5.20), ("are", 5.25, 5.40), ("gone.", 5.45, 5.90),
    ]
    dur = 7.2
    sents = [{"sentence": text, "film_start": 100.0, "film_end": 150.0}]
    stats: dict = {}
    beats = timeline.build_timeline(
        sents, [dur], 600.0,
        {"micro_cut_seconds": 2.4, "max_cuts_per_beat": 4,
         "min_cut_seconds": 1.2, "pre_roll": 0.4},
        [words], stats,
    )
    assert stats.get("word_locked_beats") == 1
    cuts = beats[0]["cuts"]
    assert len(cuts) == 3
    # narration-time cut points: 0 -> 3.10 -> 4.40 -> 7.2
    assert abs(cuts[0][1] - 3.10) < 0.05, cuts
    assert abs(cuts[1][1] - (4.40 - 3.10)) < 0.05, cuts
    assert abs(sum(d for _, d, _f, _v in cuts) - dur) < 1e-6
    print(f"ok: micro-cuts on word boundaries ({[round(d,2) for _, d, _f, _v in cuts]})")


def test_micro_cuts_even_without_words() -> None:
    """No word timings -> the old even split, still summing to the duration."""
    sents = [{"sentence": "A plain sentence.", "film_start": 0.0,
              "film_end": 30.0}]
    beats = timeline.build_timeline(
        sents, [6.0], 600.0,
        {"micro_cut_seconds": 2.4, "max_cuts_per_beat": 4,
         "min_cut_seconds": 1.2, "pre_roll": 0.4},
        None, {},
    )
    cuts = beats[0]["cuts"]
    assert abs(sum(d for _, d, _f, _v in cuts) - 6.0) < 1e-6
    assert all(abs(d - cuts[0][1]) < 1e-9 for _, d, _f, _v in cuts)  # even
    print("ok: even micro-cuts without word timings")


# --------------------------------------------------------------------------
# script.py — channel sign-off
# --------------------------------------------------------------------------
def test_sign_off_appended_once() -> None:
    segs = [{"sentence": "And that's how the movie comes to an end.",
             "film_start": 500.0, "film_end": 520.0}]
    script._append_sign_off(segs, lang_name="English", enabled=True)
    assert len(segs) == 2
    assert "subscribe" in segs[-1]["sentence"].lower()
    assert segs[-1]["film_start"] == 500.0  # reuses the last beat's window
    # appended twice more must never duplicate
    script._append_sign_off(segs, lang_name="English", enabled=True)
    assert len(segs) == 2
    print("ok: sign-off appended exactly once, reusing the final film window")


def test_sign_off_skipped_when_writer_already_closed() -> None:
    segs = [{"sentence": "The two girls become best friends.",
             "film_start": 0.0, "film_end": 10.0},
            {"sentence": "If you enjoyed the video, don't forget to "
                         "subscribe and turn on notifications.",
             "film_start": 10.0, "film_end": 20.0}]
    script._append_sign_off(segs, lang_name="English", enabled=True)
    assert len(segs) == 2, "no double outro"
    print("ok: no duplicate outro when the writer already closed with one")


def test_sign_off_disabled() -> None:
    segs = [{"sentence": "The end.", "film_start": 0.0, "film_end": 5.0}]
    script._append_sign_off(segs, lang_name="English", enabled=False)
    assert len(segs) == 1
    print("ok: sign-off can be disabled")


def test_prompts_render() -> None:
    """All rewritten prompts still render with their format fields."""
    user = script.PROMPT_SEGMENT_JSON.format(
        t0="00:00:00", t1="00:03:00", budget=90, nsent=5,
        continuity="This is the OPENING of the recap.", beats="[00:00:05] beat.",
        names_block="",
    )
    assert "SENTENCE RHYTHM" in user and "{beats}" not in user
    pol = script.POLISH_PROMPT.format(n=4, exemplar="x", names="",
                                      draft='["a"]')
    assert "MUST KEEP" in pol
    js = script.render_script_json_prompt("summary", 2000, 1500, 2500)
    assert "JSON array" in js
    print("ok: all writer prompts render")


def test_strict_personas() -> None:
    """Every LLM role must speak as a strict persona, not a task description."""
    from recap import summarize

    # The narrator persona is one identity shared by every narration writer.
    assert "veteran movie-recap narrator" in script.NARRATOR_PERSONA
    assert "Never mention yourself" in script.NARRATOR_PERSONA
    for sys_prompt in (script.SYSTEM_RECAP_WRITER, script.SYSTEM_RECAP_BEATS,
                       script.SYSTEM_POLISH):
        assert sys_prompt.startswith(script.NARRATOR_PERSONA), \
            "every narration system prompt must BE the narrator persona"
    # The persona carries the hard rules, not just the flavor.
    low = script.NARRATOR_PERSONA.lower()
    for rule in ("never invent events", "never use em-dashes",
                 "never say \"the movie\""):
        assert rule in low
    # The beat extractor has its own strict persona (the log is ground truth).
    sup = summarize.SYSTEM_SUMMARY.lower()
    assert "script supervisor" in sup and "source of truth" in sup
    assert "never quote dialogue" in sup or "you never quote dialogue" in sup
    # Legacy writer presets open with the persona framing too.
    assert "veteran movie-recap narrator" in script.STYLE_PRESET
    assert "veteran movie-recap narrator" in script.DIALOGUE_PRESET
    print("ok: strict personas on every LLM role (narrator + script supervisor)")


def test_first_chunk_covers_film_start() -> None:
    """A dialogue-free opening must not push the first film window late."""
    from recap import chunk as chunk_mod

    cues = [{"text": f"line {i}", "start": 45.0 + i * 5.0,
             "end": 49.0 + i * 5.0} for i in range(40)]
    chunks = chunk_mod.chunk_cues(cues, window_seconds=180, overlap_seconds=30)
    assert chunks, "chunks must exist"
    assert chunks[0]["start"] == 0.0, (
        "first chunk must start at the film's start, not the first cue "
        f"(got {chunks[0]['start']})"
    )
    print("ok: first chunk window starts at 0:00 even with a silent opening")


def test_first_sentence_window_reaches_film_start() -> None:
    """The recap's first visual opens on the film's first frames."""
    beats = [{"t": 60.0, "text": "The hero arrives"},   # first beat is LATE
             {"t": 120.0, "text": "The hero fights"},
             {"t": 180.0, "text": "The hero wins"}]
    wins = script._anchor_windows(0.0, 200.0, beats, 3, lead=0.8, tail=6.0)
    assert wins[0][0] == 0.0, (
        f"first window must start at the film start, got {wins[0][0]}"
    )
    assert all(w[0] < w[1] for w in wins)
    print("ok: first sentence's footage window starts at the film's start")


def test_opening_sentence_anchors_to_first_beat() -> None:
    """The embedding path must bind sentence 0 to the first beat."""
    import numpy as np

    model = script._embed_model()
    if model is None:
        print("  (skip: embeddings unavailable in this environment)")
        return
    beats = [{"t": 10.0, "text": "A soldier wakes up on an island"},
             {"t": 100.0, "text": "The soldier builds a raft"},
             {"t": 200.0, "text": "The soldier sails home"}]
    # sentence 0 deliberately phrased to be closer to beat 1 than beat 0
    sents = ["It all begins with a raft and the open sea.",
             "The soldier builds a raft and sails away."]
    vals = script._sentence_anchor_values(sents, beats)
    if vals is None:
        print("  (skip: alignment unavailable)")
        return
    assert vals[0] == 10.0, f"sentence 0 must anchor to the first beat, got {vals[0]}"
    print("ok: opening sentence is pinned to the film's first beat")


def test_proper_noun_extraction() -> None:
    beats = """[00:00:12] Jessie rides Bullseye across the yard.
[00:01:40] Bonnie tells Lilypad that she already has friends.
[00:02:05] Jessie warns Bonnie that the man grabs his gun.
[00:03:00] Years have passed and everyone plays alone."""
    names = script._proper_nouns(beats)
    for expected in ("Jessie", "Bonnie", "Bullseye", "Lilypad"):
        assert expected in names, f"{expected} missing from {names}"
    assert "Years" not in names and "everyone" not in names
    assert script._missing_names(["Jessie"], "Bonnie runs.") == ["Jessie"]
    print(f"ok: names extracted and checked ({names})")


def test_section_prompt_carries_names_block() -> None:
    user = script.PROMPT_SEGMENT_JSON.format(
        t0="00:00:00", t1="00:03:00", budget=90, nsent=5,
        continuity="This is the OPENING of the recap.",
        beats="[00:00:05] Jessie rides Bullseye.",
        names_block="NAMES THAT MUST BE SPOKEN IN THIS SECTION: Jessie, Bullseye.",
    )
    assert "NAMES THAT MUST BE SPOKEN" in user and "Jessie" in user
    pol = script.POLISH_PROMPT.format(
        n=3, exemplar="x", names="Jessie, Bullseye", draft='["a"]'
    )
    assert "Jessie" in pol, "polish prompt must carry the names"
    print("ok: writer + polish prompts carry the must-use names")


def test_global_polish_count_lock() -> None:
    """The full-script pass never changes the sentence count (or is dropped)."""
    import json as _json

    sentences = [f"Sentence number {i} of the recap." for i in range(12)]
    calls = {"n": 0}

    def fake_complete(provider, model, system, user, **kw):
        calls["n"] += 1
        # bad answer: wrong count -> must be discarded
        if calls["n"] == 1:
            return _json.dumps({"sentences": sentences[:9]})
        # good answer: same count, reworded, ONE sentence per element
        # (multi-sentence elements are split at parse time, which would
        # change the count and correctly discard the rewrite)
        return _json.dumps({"sentences": [
            f"Reworded sentence number {i} of the recap."
            for i in range(12)]})

    original = script.llm.complete
    script.llm.complete = fake_complete
    try:
        out1 = script._global_polish({"provider": "x", "model": "y"},
                                     sentences, ["Jessie"])
        assert out1 == sentences, "wrong-count rewrite must be discarded"
        out2 = script._global_polish({"provider": "x", "model": "y"},
                                     sentences, ["Jessie"])
        assert len(out2) == 12 \
            and out2[0] == "Reworded sentence number 0 of the recap."
    finally:
        script.llm.complete = original
    print("ok: global polish keeps the count lock (bad rewrites discarded)")


if __name__ == "__main__":
    test_map_words_to_sentences_perfect_match()
    test_map_words_to_sentences_tolerant_match()
    test_map_words_to_sentences_rejects_garbage()
    test_refine_cues_uses_measured_times()
    test_cues_from_words_replaces_proportional_guess()
    test_micro_cuts_land_on_word_boundaries()
    test_micro_cuts_even_without_words()
    test_sign_off_appended_once()
    test_sign_off_skipped_when_writer_already_closed()
    test_sign_off_disabled()
    test_prompts_render()
    test_strict_personas()
    test_first_chunk_covers_film_start()
    test_first_sentence_window_reaches_film_start()
    test_opening_sentence_anchors_to_first_beat()
    test_proper_noun_extraction()
    test_section_prompt_carries_names_block()
    test_global_polish_count_lock()
    print("\nALL NARRATION-SYNC TESTS PASSED")
