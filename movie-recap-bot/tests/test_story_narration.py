"""Tests for the storytelling narration layer.

The reported problem: the recap "is really bad, it's describing the scene"
— the writer was told to *describe what happens in this beat*, so it produced
a shot-by-shot description with no story, no cause and effect, and no
continuity between beats.

These tests pin the fix:
  * the unit prompt asks for a story and carries the story so far, the cast
    already introduced and the line to continue from (continuity),
  * every sentence is bound to the beat whose facts it narrates and to that
    beat's film range (the visual-to-text binding constraint),
  * sentence windows are strictly monotone through the film
    (``start[i] < end[i] <= start[i+1]``) so nothing can be shown early,
  * a description lint measures how much of the script reads as description
    and drives ONE targeted rewrite pass,
  * a unit that overruns its footage budget is tightened description-first,
    never by dropping story,
  * a provider that refuses a model name is recovered from instead of
    retried into failure (the reported "supported API model names are ..."
    crash).

Run:  python -m pytest tests/test_story_narration.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap import narrative, story  # noqa: E402
from recap.script import count_words  # noqa: E402


def _beats(n: int, *, dur: float = 12.0, start: float = 100.0) -> list[dict]:
    """``n`` consecutive, well-formed beats (start/end/duration + facts)."""
    out = []
    t = start
    for i in range(n):
        out.append({
            "start_ts": t,
            "end_ts": t + dur,
            "duration": dur,
            "transcript_lines": [{"text": f"Line {i} of dialogue.",
                                  "start": t, "end": t + 3}],
            "vision_notes": [{"t": t + 1, "text": f"Action {i} happens."}],
            "shot_count": 2,
        })
        t += dur
    return out


# ---------------------------------------------------------------------------
# 1. the prompt tells a story (and carries continuity)
# ---------------------------------------------------------------------------


def test_unit_prompt_asks_for_story_not_description() -> None:
    units = story.group_into_units(_beats(4), {"unit_seconds": 45.0}, wpm=175)
    assert len(units) == 1
    system, user = story.build_unit_prompt(
        units[0], ledger={"recaps": [], "last": "", "names": set()},
        total_units=1, film_duration=600.0,
    )
    # the storytelling voice is present...
    assert system.startswith("You are a master scriptwriter for a hit YouTube movie recap channel.")
    assert "NO CAMERA WORDS" in system and '"we see"' in system
    assert "third-person, present tense" in system
    assert "JSON array of objects" in system and "`timestamp`" in system
    # ...and so is the anti-description contract in the user prompt
    assert "shot description" in user.lower()
    # the facts arrive labelled with their EXACT film ranges and budgets
    assert "[B1]" in user and "[B4]" in user
    assert narrative.format_window(units[0]["beats"][0]["start_ts"],
                                   units[0]["beats"][0]["end_ts"]) in user
    assert "about" in user and "words" in user  # per-beat word budget
    print("ok: unit prompt asks for a story, with labelled film ranges")


def test_unit_prompt_carries_the_story_so_far() -> None:
    units = story.group_into_units(_beats(3, start=500.0), {"unit_seconds": 40.0})
    ledger = {"recaps": ["Marco took the file."], "last": "He runs for the door.",
              "names": {"Marco"}}
    _system, user = story.build_unit_prompt(
        units[0], ledger=ledger, total_units=2, film_duration=1000.0,
    )
    assert "Marco took the file." in user, "story-so-far must reach the writer"
    assert "He runs for the door." in user, "the previous line must reach the writer"
    assert "ALREADY ON SCREEN" in user and "Marco" in user
    assert "do not repeat it" in user.lower()
    # ...and the act guidance tracks the film position (500/1000 = midpoint-ish)
    assert "STORY" in user and ("MIDPOINT" in user or "RISING" in user
                                or "COMPLICATION" in user)
    print("ok: continuity ledger (so far, cast, last line) reaches the writer")


# ---------------------------------------------------------------------------
# 2. units group consecutive beats and keep their budgets
# ---------------------------------------------------------------------------


def test_units_are_contiguous_and_budgeted() -> None:
    beats = _beats(10, dur=12.0)          # 120s of film
    units = story.group_into_units(beats, {"unit_seconds": 45.0,
                                           "max_beats_per_unit": 4})
    assert len(units) >= 3
    seen: list[float] = []
    for u in units:
        labels = [b["label"] for b in u["beats"]]
        assert labels == [f"B{i + 1}" for i in range(len(labels))]
        assert u["budget_words"] == sum(b["max_words"] for b in u["beats"])
        for b in u["beats"]:
            seen.append(b["start_ts"])
    # every beat appears exactly once, in film order
    assert seen == sorted(seen)
    assert len(seen) == len(beats)
    print(f"ok: 10 beats -> {len(units)} story units, contiguous, budgeted")


# ---------------------------------------------------------------------------
# 3. parsing: labels are clamped, never reordered
# ---------------------------------------------------------------------------


def test_parse_reply_clamps_labels_and_order() -> None:
    raw = """
    ```json
    {"beats": [
        {"b": 2, "sentences": ["First.", "Second."]},
        {"b": 9, "sentences": ["Third."]},
        {"beat": "B1", "text": "Fourth, out of order."}
    ], "recap": "Marco finds the file."}
    ```
    """
    items, recap = story.parse_unit_reply(raw, 3)
    assert [i["b"] for i in items] == [2, 2, 3, 3], "labels must never rewind"
    assert all(1 <= i["b"] <= 3 for i in items)
    assert items[-1]["sentence"].endswith(".")
    assert recap == "Marco finds the file."
    print("ok: out-of-order/out-of-range beat labels are clamped monotone")


def test_parse_reply_accepts_flat_string_array() -> None:
    items, _ = story.parse_unit_reply(
        '["The pilot wakes up.", "The forest is silent.", "She runs."]', 2)
    labels = [i["b"] for i in items]
    assert labels == sorted(labels) and labels[0] == 1 and labels[-1] == 2
    assert len(items) == 3
    print("ok: flat sentence arrays still map onto beats, in order")


# ---------------------------------------------------------------------------
# 4. windows: strict chronology, in beat order
# ---------------------------------------------------------------------------


def test_segments_never_show_film_early() -> None:
    beats = _beats(2, dur=10.0, start=200.0)
    units = story.group_into_units(beats, {"unit_seconds": 30.0})
    items = [{"b": 1, "sentence": "He slips inside and locks the door."},
             {"b": 1, "sentence": "The guard never sees him."},
             {"b": 2, "sentence": "Morning comes and the office is empty."}]
    segs = story.unit_to_segments(units[0], items, prev_end=0.0)
    assert [s["beat"] for s in segs] == ["B1", "B1", "B2"]
    assert all(200.0 <= float(s["film_start"]) for s in segs)
    assert story.check_chronology(segs) == []
    for a, b in zip(segs, segs[1:]):
        assert float(a["film_end"]) <= float(b["film_start"]) + 1e-6
        assert float(a["film_start"]) < float(a["film_end"])
    print("ok: sentence windows are strictly ordered inside their beats")


def test_enforce_chronology_pushes_never_sorts() -> None:
    segs = [
        {"sentence": "First line.", "film_start": 100.0, "film_end": 110.0},
        {"sentence": "Second line.", "film_start": 50.0, "film_end": 60.0},
    ]
    out, moved = story.enforce_chronology(segs, movie_duration=1000.0)
    assert moved == 1
    assert [s["sentence"] for s in out] == ["First line.", "Second line."], \
        "the narration order IS the story order — never sort it"
    assert float(out[1]["film_start"]) >= float(out[0]["film_end"]) - 1e-6
    assert story.check_chronology(out) == []
    print("ok: a late window is pushed forward, never swapped")


def test_chronology_survives_a_narration_longer_than_the_film() -> None:
    """Narration outlasting the film: windows never invert or rewind."""
    segs = [
        {"sentence": f"Line {i}.", "film_start": 90.0 + i * 5.0,
         "film_end": 95.0 + i * 5.0}
        for i in range(6)
    ]
    out, _moved = story.enforce_chronology(segs, movie_duration=100.0)
    assert story.check_chronology(out) == [], "no rewind, no inverted window"
    for s in out:
        assert float(s["film_start"]) < float(s["film_end"])
        assert float(s["film_end"]) <= 100.0 + 1e-6
    # the tail necessarily compresses (the picture will hold its last frame)
    assert story.check_no_overlap(out), "the film-overrun tail overlaps by design"
    print("ok: film-overrun windows compress into the tail without inverting")


# ---------------------------------------------------------------------------
# 5. description lint + description-first tightening
# ---------------------------------------------------------------------------


def test_description_lint_catches_frame_talk() -> None:
    bad = [
        "The camera pans across the city skyline.",
        "We see a table covered in papers.",
        "There is a dark hallway.",
        "A man in a suit is standing near the door.",
    ]
    for line in bad:
        assert narrative.looks_like_description(line), line
    good = [
        "Marco grabs the file and runs before the guard turns around.",
        "She admits that the letter is a forgery.",
        "By morning the whole office knows what he did.",
    ]
    for line in good:
        assert not narrative.looks_like_description(line), line
    report = narrative.description_report(bad + good)
    assert report["flagged"] == 4 and report["sentences"] == 7
    assert abs(report["score"] - 3 / 7) < 1e-6
    print("ok: description lint separates captions from story")


def test_tighten_drops_description_before_story() -> None:
    long = ("The camera lingers on the empty office, there is dust on the "
            "desk, and Marco finds the missing file under a broken lamp.")
    short = story.tighten_to_budget(long, 14)
    assert count_words(short) <= 14
    assert "Marco" in short and "file" in short, "the story must survive the trim"
    assert "camera" not in short.lower() and "there is" not in short.lower()
    print(f"ok: description-first trim -> {short!r}")


def test_over_budget_beats_are_seen_by_the_writer() -> None:
    units = story.group_into_units(_beats(2, dur=8.0), {"unit_seconds": 30.0})
    unit = units[0]
    beat = unit["beats"][0]
    items = [{"b": 1, "sentence": " ".join(["word"] * (beat["max_words"] + 5))}]
    over = story._over_budget_beats(unit, items)
    assert over and over[0][0] == "B1" and over[0][1] > over[0][2]
    print("ok: per-beat overruns are detected against the footage budget")


# ---------------------------------------------------------------------------
# 6. end-to-end with a stubbed model
# ---------------------------------------------------------------------------


def test_write_story_script_end_to_end(monkeypatch) -> None:
    beats = _beats(6, dur=12.0, start=100.0)      # 72s of film
    calls: list[str] = []

    def fake_ask(cfg_llm, system, user, *, max_tokens, temperature=None):
        calls.append(user)
        assert "NO CAMERA WORDS" in system
        # answer in the documented shape, one beat per beat of the unit
        unit_beats = [b for b in user.split("[B") if b]
        n = len([ln for ln in user.splitlines() if ln.startswith("[B")])
        rows = ",".join(
            '{"b": %d, "sentences": ["Event %d turns the story.", '
            '"He decides to act."]}' % (i + 1, i + 1) for i in range(n)
        )
        return '{"beats": [%s], "recap": "Something changes in beat %d."}' % (rows, n)

    monkeypatch.setattr(story, "_ask", fake_ask)
    result = story.write_story_script(
        beats, {"provider": "deepseek", "model": "deepseek-chat"},
        wpm=175, movie_duration=600.0, progress=None,
    )
    segs = result["segments"]
    assert len(segs) >= 6
    assert story.check_chronology(segs) == []
    assert all(100.0 <= float(s["film_start"]) for s in segs)
    assert result["report"]["units"] >= 1
    assert result["report"]["words"] > 0
    assert result["report"]["description"]["flagged"] == 0
    assert result["report"]["chronology_moved"] == 0
    # the SECOND unit's prompt must carry the continuity ledger from the first
    unit2 = [c for c in calls if c.startswith("Write story unit 2 ")]
    if unit2:
        assert "STORY SO FAR" in unit2[0]
        assert "Something changes in beat" in unit2[0], \
            "the previous unit's recap must reach the next unit"
    print(f"ok: story script {len(segs)} sentences, "
          f"{result['report']['words']} words, strictly chronological")


def test_repair_pass_rewrites_description(monkeypatch) -> None:
    beats = _beats(2, dur=12.0, start=300.0)
    units = story.group_into_units(beats, {"unit_seconds": 30.0})
    segs = story.unit_to_segments(
        units[0],
        [{"b": 1, "sentence": "The camera pans over the empty office."},
         {"b": 2, "sentence": "Action 1 happens again at dawn."}],
        prev_end=0.0,
    )

    def fake_ask(cfg_llm, system, user, *, max_tokens, temperature=None):
        assert "TELL THE STORY" in system
        return '{"sentences": [{"i": 0, "text": "Marco slips into the empty ' \
               'office to hide the file."}]}'

    monkeypatch.setattr(story, "_ask", fake_ask)
    stats = story.repair_segments(segs, units, {"provider": "deepseek"})
    assert stats["flagged"] >= 1 and stats["repaired"] == 1
    assert "camera" not in segs[0]["sentence"].lower()
    assert "office" in segs[0]["sentence"].lower()
    print("ok: description-mode line rewritten into story, facts kept")


# ---------------------------------------------------------------------------
# 6b. the pipeline's own progress callback contract + report file
# ---------------------------------------------------------------------------


def test_pipeline_progress_contract_and_report_file(monkeypatch, tmp_path) -> None:
    """``story.write_report`` + the progress callback the pipeline passes."""
    from recap import narrative as narrative_mod, pipeline, story as story_mod

    beats = _beats(4, dur=12.0, start=0.0)
    seen: list[str] = []

    def fake_ask(cfg_llm, system, user, *, max_tokens, temperature=None):
        n = len([ln for ln in user.splitlines() if ln.startswith("[B")])
        rows = ",".join('{"b": %d, "sentences": ["Something happens here."]}'
                        % (i + 1) for i in range(n))
        return '{"beats": [%s], "recap": "It begins."}' % rows

    monkeypatch.setattr(story_mod, "_ask", fake_ask)

    # exactly the lambda pipeline.py passes to write_story_script
    _units_total = len(story_mod.group_into_units(beats, None))

    def _story_progress(_u, _n, _w, _t=_units_total):
        seen.append(f"{_u['index'] + 1}/{_t} "
                    f"{narrative_mod.format_window(_u['start_ts'], _u['end_ts'])} "
                    f"-> {_n} sentences, {_w} words")

    result = story_mod.write_story_script(
        beats, {"provider": "deepseek", "model": "deepseek-chat"},
        wpm=175, movie_duration=600.0, progress=_story_progress,
    )
    assert seen and seen[0].startswith("1/"), seen
    out = tmp_path / "story_en.json"
    story_mod.write_report(result["report"], out)
    import json as _json
    data = _json.loads(out.read_text(encoding="utf-8"))
    assert data["units"] == len(seen) and data["words"] > 0
    assert "description" in data and data["description"]["score"] <= 1.0
    # the pipeline's story block wires this function
    src = __import__("inspect").getsource(pipeline)
    assert "story.write_story_script" in src
    assert "story.write_report" in src
    assert '"story-v1"' in src, "the cache tag must invalidate pre-story scripts"
    print("ok: pipeline contract (progress shape + report json + cache tag)")


# ---------------------------------------------------------------------------
# 7. model-name recovery (the reported crash)
# ---------------------------------------------------------------------------


def test_supported_model_names_parsed_from_the_reported_error() -> None:
    from recap import llm

    msg = ("Error code: 400 - {'error': {'message': 'The supported API model "
           "names are deepseek-flash, deepseek-v4-pro, but you passed "
           "gemini-3.1-flash-lite.', 'type': 'invalid_request_error'}}")
    names = llm.parse_supported_models(msg)
    assert names == ["deepseek-flash", "deepseek-v4-pro"]
    # ...and a name that does NOT belong to the provider never ships
    assert llm.resolve_model("deepseek", "gemini-3.1-flash-lite") == "deepseek-chat"
    # a custom endpoint that listed its models wins over the stale default
    llm.remember_endpoint_models("https://proxy.example/v1", names)
    assert llm.resolve_model("deepseek", "deepseek-chat",
                             "https://proxy.example/v1") == "deepseek-flash"
    assert llm.resolve_model("deepseek", "",
                             "https://proxy.example/v1") == "deepseek-flash"
    # an honest model passes through untouched
    assert llm.resolve_model("deepseek", "deepseek-reasoner") == "deepseek-reasoner"
    print("ok: cross-provider model names refused; endpoint's own names used")


def test_model_error_detection_only_fires_on_model_errors() -> None:
    from recap import llm

    ok = llm._model_error_model(
        Exception("The supported API model names are tiny-a, tiny-b, "
                  "but you passed tiny-c"))
    assert ok == "tiny-a"
    assert llm._model_error_model(
        Exception("Connection reset by peer")) is None
    assert llm._model_error_model(Exception("429 rate limit")) is None
    print("ok: model-name recovery fires only on model-name errors")


# ---------------------------------------------------------------------------
# 7b. the chunk path carries the same storytelling rules
# ---------------------------------------------------------------------------


def test_legacy_beat_prompt_also_tells_a_story() -> None:
    """The per-beat FALLBACK writer must not go back to describing shots."""
    from recap import beats as beats_mod

    prompt = beats_mod.BEAT_SYSTEM_PROMPT
    assert "Describe ONLY" not in prompt
    assert "do not describe the picture" in prompt.lower()
    assert "we see" in prompt.lower(), "the shot-description ban must be explicit"
    # the hard budget contract stays (the timeline times every word to footage)
    rendered = prompt.format(min_words=8, max_words=20, duration="7.0")
    assert "8" in rendered and "20" in rendered and "7.0" in rendered
    print("ok: fallback beat prompt tells the story, budget intact")


def test_chunk_writer_carries_the_story_rules() -> None:
    """Both writers must tell a story, not just the new beat path."""
    from recap import script as script_mod

    assert "TELL THE STORY" in script_mod.SYSTEM_RECAP_BEATS, \
        "the section system prompt must carry the storytelling rules"
    assert "STORY DESCRIPTION" not in script_mod.PROMPT_SEGMENT_JSON
    # the story block that rides in the user message starts with the compact
    # rules and is non-trivial
    block = script_mod.STORY_BLOCK_SEGMENT
    assert "TELL THE STORY" in block and len(block) > 400
    # the description ban is part of the writer's forbidden list
    assert "SHOT DESCRIPTION" in script_mod.PROMPT_SEGMENT_JSON
    # ...and the polish pass is told to fix description, not preserve it
    assert "SHOT DESCRIPTION" in script_mod.POLISH_PROMPT or \
        "only says what the picture looks like" in script_mod.POLISH_PROMPT
    print("ok: chunk writer + polish prompt carry the storytelling contract")


# ---------------------------------------------------------------------------
# 8. the story script drives the real timeline at 1x
# ---------------------------------------------------------------------------


def test_story_segments_drive_a_1x_timeline(monkeypatch) -> None:
    """End-to-end chain: story units -> segments -> timeline -> all 1x.

    The sentences are budgeted by the same words-per-second rule the timeline
    paces by, so the measured narration fits the film the script claims and
    every cut plays at 1x with no held frames and no rewinds.
    """
    from recap import timeline

    beats = _beats(8, dur=15.0, start=200.0)     # 120s of film, 8 beats

    def fake_ask(cfg_llm, system, user, *, max_tokens, temperature=None):
        n = len([ln for ln in user.splitlines() if ln.startswith("[B")])
        rows = ",".join(
            '{"b": %d, "sentences": ["Evan takes the keys and drives.", '
            '"The gate is already open."]}' % (i + 1) for i in range(n))
        return '{"beats": [%s], "recap": "Evan gets inside."}' % rows

    monkeypatch.setattr(story, "_ask", fake_ask)
    result = story.write_story_script(
        beats, {"provider": "deepseek", "model": "deepseek-chat"},
        wpm=175, movie_duration=600.0,
    )
    segs = result["segments"]
    assert story.check_chronology(segs) == []

    # the narration the writer produced, measured at the SAME 175 wpm budget
    durs = [count_words(s["sentence"]) / 175.0 * 60.0 for s in segs]
    stats: dict = {}
    btl = timeline.build_timeline(segs, durs, 600.0, {"min_speed": 0.85},
                                  stats=stats)
    speeds = [sp for _s, _d, _f, sp in
              (c for b in btl for c in b["cuts"])]
    assert speeds and all(abs(sp - 1.0) < 1e-9 for sp in speeds), \
        f"story script must play at 1x, got {sorted(set(speeds))}"
    assert stats.get("padded_shots", 0) == 0
    assert timeline.cut_order_violations(timeline.flatten_cuts(btl)) == []
    print(f"ok: {len(segs)} story sentences -> timeline at 1x, "
          f"nothing frozen, no replays")


def test_timestamp_array_contract_sets_anchors() -> None:
    beats = _beats(3, dur=12.0, start=100.0)
    units = story.group_into_units(beats, {"unit_seconds": 60.0}, wpm=175)
    unit = units[0]
    raw = ('[{"timestamp": 101.5, "sentence": "Buzz pushes through the jungle."},'
           ' {"timestamp": 118.25, "sentence": "Three figures run."},'
           ' {"timestamp": 110.0, "sentence": "Rewind is clamped."}]')
    items, _ = story.parse_unit_reply(raw, len(unit["beats"]), unit["beats"])
    assert [i["t"] for i in items] == [101.5, 118.25, 118.25]
    assert [i["b"] for i in items] == [1, 2, 2]
    segs = story.unit_to_segments(unit, items, prev_end=100.0, movie_duration=600.0)
    assert abs(segs[0]["anchor"] - 101.5) < 1e-6
    assert abs(segs[1]["anchor"] - 118.25) < 1e-6
