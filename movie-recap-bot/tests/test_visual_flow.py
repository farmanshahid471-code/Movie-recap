"""Tests for the visual-flow fixes (no replays + shot-boundary snapping).

The reported bugs:
  1. "Scenes repeat while the narration is going on" — consecutive cuts
     pointed at film times that overlapped or even rewound, so the same
     footage replayed with a stutter.
  2. "Scene changes feel laggy" — cuts landed mid-shot at arbitrary computed
     times instead of on the film's real shot changes.

Run:  python tests/test_visual_flow.py
(no third-party deps needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap import timeline  # noqa: E402
from recap.script import count_words  # noqa: E402
from recap.tts import TimedCue  # noqa: E402

CFG = {"micro_cut_seconds": 2.4, "max_cuts_per_beat": 4, "min_cut_seconds": 1.2}


def _cuts_in_order(beats):
    """Every cut as (film_start, duration, freeze) in play order."""
    for b in beats:
        yield from b["cuts"]


def test_no_replay_same_window() -> None:
    """Many sentences anchored to ONE tight film moment must never rewind."""
    cues = [TimedCue(f"S{i}.", i * 4.0, i * 4.0 + 3.2) for i in range(3)]
    durs = timeline.lock_durations(cues, 12.0)
    sents = [{"sentence": c.text, "film_start": 100.0, "film_end": 103.0}
             for c in cues]
    beats = timeline.build_timeline(sents, durs, 6000.0, CFG)
    prev_end = -1.0
    for s, d, f, sp in _cuts_in_order(beats):
        assert s >= prev_end - 1e-6, (
            f"cut at {s:.2f}s replays footage that ended at {prev_end:.2f}s"
        )
        # a freeze-hold does not consume film: track the moving footage end
        prev_end = max(prev_end, s + (d - f) * sp)
    print("ok: tight same-window beats never replay (was 5 rewinds before)")


def test_no_replay_overlapping_windows() -> None:
    """Beats whose anchor windows overlap by design still never rewind."""
    cues = [TimedCue(f"S{i}.", i * 4.0, i * 4.0 + 3.2) for i in range(4)]
    durs = timeline.lock_durations(cues, 16.0)
    # each window starts only 1s after the previous one, extends 6s
    sents = [{"sentence": c.text,
              "film_start": 200.0 + i * 1.0,
              "film_end": 206.0 + i * 1.0}
             for i, c in enumerate(cues)]
    beats = timeline.build_timeline(sents, durs, 6000.0, CFG)
    prev_end = -1.0
    for s, d, f, sp in _cuts_in_order(beats):
        assert s >= prev_end - 1e-6, f"rewind: {s:.2f} < {prev_end:.2f}"
        prev_end = max(prev_end, s + (d - f) * sp)
    # length lock still exact
    total = sum(d for _, d, _f, _v in _cuts_in_order(beats))
    assert abs(total - 16.0) < 1e-6
    print("ok: overlapping anchor windows never rewind, length locked")


def test_no_replay_long_run_random_windows() -> None:
    """A 150-sentence montage over jittery windows: strictly forward film."""
    cues = [TimedCue(f"S{i}.", i * 6.0, i * 6.0 + 4.0) for i in range(150)]
    span = 149 * 6.0 + 4.0
    durs = timeline.lock_durations(cues, span)
    sents = []
    t = 0.0
    for i, c in enumerate(cues):
        w = 20.0 + (i % 7) * 13.0        # windows of varying width...
        t += 15.0 + (i % 5) * 9.0        # ...at jittery, increasing positions
        sents.append({"sentence": c.text, "film_start": t,
                      "film_end": min(t + w, 5900.0)})
    beats = timeline.build_timeline(sents, durs, 6000.0, CFG)
    prev_end = -1.0
    for s, d, f, sp in _cuts_in_order(beats):
        assert s >= prev_end - 1e-6, (
            f"beat rewound the film: {s:.2f}s after {prev_end:.2f}s"
        )
        prev_end = max(prev_end, s + (d - f) * sp)
    total = sum(d for _, d, _f, _v in _cuts_in_order(beats))
    assert abs(total - span) < 0.5
    print(f"ok: 150-sentence montage strictly forward, {total:.0f}s locked")


def test_snap_to_boundary() -> None:
    """Cuts move onto the nearest real shot change within tolerance."""
    assert timeline._snap_to_boundary(100.4, [100.0, 112.5, 130.0], 0.8) == 100.0
    assert timeline._snap_to_boundary(112.9, [100.0, 112.5, 130.0], 0.8) == 112.5
    assert timeline._snap_to_boundary(120.0, [100.0, 112.5, 130.0], 0.8) == 120.0
    assert timeline._snap_to_boundary(129.7, [100.0, 112.5, 130.0], 0.8) == 130.0
    assert timeline._snap_to_boundary(50.0, [], 0.8) == 50.0
    print("ok: snapping picks the nearest boundary within tolerance only")


def test_timeline_uses_scene_bounds() -> None:
    """build_timeline snaps cut starts onto supplied shot boundaries and
    still never rewinds."""
    cues = [TimedCue(f"S{i}.", i * 6.0, i * 6.0 + 4.5) for i in range(6)]
    durs = timeline.lock_durations(cues, 34.0)
    sents = [{"sentence": c.text, "film_start": 40.0 + i * 25.0,
              "film_end": 65.0 + i * 25.0} for i, c in enumerate(cues)]
    bounds = [40.0, 52.37, 65.11, 90.0, 102.44, 115.3, 130.0, 142.9, 160.0,
              172.15, 190.0]
    stats = {}
    beats = timeline.build_timeline(
        sents, durs, 600.0, CFG, stats=stats, scene_bounds=bounds,
    )
    assert stats.get("snapped_cuts", 0) > 0, "expected some cuts to snap"
    prev_end = -1.0
    on_boundary = 0
    total_cuts = 0
    for s, d, f, sp in _cuts_in_order(beats):
        assert s >= prev_end - 1e-6, "snap caused a rewind"
        if any(abs(s - b) < 1e-6 for b in bounds):
            on_boundary += 1
        total_cuts += 1
        prev_end = max(prev_end, s + (d - f) * sp)
    total = sum(d for _, d, _f, _v in _cuts_in_order(beats))
    assert abs(total - 34.0) < 1e-6
    print(f"ok: {on_boundary}/{total_cuts} cuts land on real shot boundaries, "
          f"no rewind, length locked")


def test_snap_never_breaks_monotonic_playhead() -> None:
    """A boundary just behind the film playhead must be ignored."""
    cues = [TimedCue("A.", 0.0, 4.0), TimedCue("B.", 5.0, 9.0)]
    durs = timeline.lock_durations(cues, 9.5)
    # window 0: film [100,110] -> first cut plays [99.6, 104.1]
    # boundary at 101 would snap cut2's desired 101.8 backwards -> must NOT.
    sents = [{"sentence": "A.", "film_start": 100.0, "film_end": 110.0},
             {"sentence": "B.", "film_start": 100.0, "film_end": 110.0}]
    bounds = [101.0, 104.09, 108.0]
    beats = timeline.build_timeline(sents, durs, 600.0, CFG, scene_bounds=bounds)
    prev_end = -1.0
    for s, d, f, sp in _cuts_in_order(beats):
        assert s >= prev_end - 1e-6, "snap rewound below the film playhead"
        prev_end = max(prev_end, s + (d - f) * sp)
    print("ok: snapping respects the film playhead (no forced rewinds)")


def test_end_of_film_stays_in_bounds() -> None:
    """Narration longer than the film: cuts clamp inside the movie."""
    cues = [TimedCue(f"S{i}.", i * 9.0, i * 9.0 + 8.0) for i in range(40)]
    span = 39 * 9.0 + 8.0
    durs = timeline.lock_durations(cues, span)
    sents = [{"sentence": c.text, "film_start": 0.0, "film_end": 300.0}
             for c in cues]
    beats = timeline.build_timeline(sents, durs, 300.0, CFG)
    for s, d, _f, _v in _cuts_in_order(beats):
        assert 0.0 <= s <= 300.0
    total = sum(d for _, d, _f, _v in _cuts_in_order(beats))
    assert abs(total - span) < 0.5
    print("ok: end-of-film clamp keeps every cut inside the movie")


def test_visuals_never_run_ahead_of_the_narration() -> None:
    """Dense-beat section: the narration is LONGER than its footage. Before
    the pacing fix the forward walk made the visuals run up to ~86s of film
    ahead of the story. Now the section plays in SLOW MOTION: the picture
    never stops, never repeats, and stays with the narrated moment."""
    # 25 sentences, 6s narration each (149s) over beats only 2.4s apart.
    cues = [TimedCue(f"S{i}.", i * 6.0, i * 6.0 + 5.0) for i in range(25)]
    span = 24 * 6.0 + 5.0
    durs = timeline.lock_durations(cues, span)
    sents = [{"sentence": c.text,
              "film_start": 1000.0 + i * 2.4,
              "film_end": 1002.4 + i * 2.4}
             for i, c in enumerate(cues)]
    cfg = dict(CFG)
    cfg["min_speed"] = 0.35
    stats: dict = {}
    beats = timeline.build_timeline(sents, durs, 6000.0, cfg, stats=stats)

    max_lead_seen = 0.0
    prev_end = -1.0
    for b in beats:
        for s, d, f, sp in b["cuts"]:
            lead = s - b["film_start"]
            max_lead_seen = max(max_lead_seen, lead)
            assert s >= prev_end - 1e-6, "no-replay must still hold"
            prev_end = max(prev_end, s + (d - f) * sp)
    assert max_lead_seen <= 8.0, (
        f"visuals ran {max_lead_seen:.1f}s ahead of the narration "
        "(was ~86s before pacing)"
    )
    # THE MOTION GUARANTEE: not a single frozen frame mid-film
    frozen = sum(f for _, _d, f, _v in _cuts_in_order(beats))
    assert frozen == 0.0, "mid-film freezes are forbidden (slow-mo instead)"
    assert stats.get("held_shots", 0) == 0
    # the dense section was paced in slow motion, never below min_speed
    assert stats.get("slowed_groups", 0) > 0
    for _s, _d, _f, sp in _cuts_in_order(beats):
        assert sp >= 0.35 - 1e-6, "speed must respect min_speed"
    # durations still sum to the narration exactly
    total = sum(d for _, d, _f, _v in _cuts_in_order(beats))
    assert abs(total - span) < 1e-6
    print(f"ok: dense section -> lead {max_lead_seen:.1f}s (was ~86s), "
          f"{stats['slowed_groups']} slow-mo groups, 0 frozen seconds, "
          f"{total:.0f}s locked")


def test_holds_keep_av_lock_in_sparse_sections() -> None:
    """Normal pacing (windows wider than narration) must not hold at all."""
    cues = [TimedCue(f"S{i}.", i * 5.0, i * 5.0 + 4.0) for i in range(10)]
    span = 9 * 5.0 + 4.0
    durs = timeline.lock_durations(cues, span)
    sents = [{"sentence": c.text,
              "film_start": i * 60.0, "film_end": i * 60.0 + 40.0}
             for i, c in enumerate(cues)]   # wide windows, plenty of footage
    stats: dict = {}
    beats = timeline.build_timeline(sents, durs, 6000.0, dict(CFG), stats=stats)
    assert stats.get("held_shots", 0) == 0, "no holds expected when windows are wide"
    assert stats.get("slowed_groups", 0) == 0, "no slow-mo when film is ample"
    for _s, _d, f, sp in _cuts_in_order(beats):
        assert f == 0.0 and abs(sp - 1.0) < 1e-9
    total = sum(d for _, d, _f, _v in _cuts_in_order(beats))
    assert abs(total - span) < 1e-6
    print("ok: wide-window sections play straight through at 1x speed")


def test_slow_motion_cut_command() -> None:
    """A paced (slow-mo) cut reads only duration*speed of film and slows it
    with setpts so the motion is continuous (no freeze, no fps drop)."""
    from recap import clip

    captured: list[list[str]] = []
    original_run, original_which = clip.run, clip.which_ffmpeg

    def fake_run(cmd, **kw):
        captured.append(list(cmd))

    clip.run = fake_run
    clip.which_ffmpeg = lambda: "ffmpeg"
    try:
        # 8s of screen time at 0.5x = 4s of film
        clip.cut_segment(Path("movie.mp4"), Path("seg.mp4"), 100.0, 8.0,
                         {"fps": 30}, mode="reencode", exact=True,
                         freeze=0.0, speed=0.5)
    finally:
        clip.run = original_run
        clip.which_ffmpeg = original_which

    cmd = captured[0]
    assert cmd[cmd.index("-t") + 1] == "4.000", \
        f"input must read 8*0.5=4s of film: {cmd}"
    assert cmd[cmd.index("-t", cmd.index("-i")) + 1] == "8.000", \
        "output must be the full 8s"
    joined = " ".join(cmd)
    assert "setpts=PTS/0.5000" in joined, "slowdown via setpts"
    assert "tpad" not in joined, "no freeze in a pure slow-mo cut"
    print("ok: slow-mo cut = 4s of film stretched to a full 8s of motion")


def test_freeze_cut_command() -> None:
    """A freeze-hold cut reads only the moving part of the shot and clones
    the last frame for the rest (no ffmpeg binary needed: we capture the
    command the pipeline would run)."""
    from recap import clip

    captured: list[list[str]] = []
    original_run, original_which = clip.run, clip.which_ffmpeg

    def fake_run(cmd, **kw):
        captured.append(list(cmd))

    clip.run = fake_run
    clip.which_ffmpeg = lambda: "ffmpeg"
    try:
        clip.cut_segment(Path("movie.mp4"), Path("seg.mp4"), 100.0, 8.0,
                         {"fps": 30}, mode="reencode", exact=True, freeze=5.0)
    finally:
        clip.run = original_run
        clip.which_ffmpeg = original_which

    assert captured, "cut_segment must invoke ffmpeg"
    cmd = captured[0]
    joined = " ".join(cmd)
    # input read limit = duration - freeze ...
    assert cmd[cmd.index("-t") + 1] == "3.000", \
        f"moving part must be 8-5=3s: {cmd}"
    i_ss, i_t = cmd.index("-ss"), cmd.index("-t")
    assert i_ss < i_t < cmd.index("-i"), "input limit must precede -i"
    # ... output still locked to the full 8s ...
    assert cmd[cmd.index("-t", i_t + 1) + 1] == "8.000", "output -t must be the full duration"
    # ... via a tpad clone of the final frame
    assert "tpad=stop_mode=clone:stop_duration=5.000" in joined
    print("ok: freeze cut = 3s of film + 5s frame-hold, output exactly 8s")


def test_visual_matched_budgets() -> None:
    """Section budgets are sized by FILM TIME and capped at 1x footage."""
    from recap import script as script_mod

    # 10 sections x 150s of distinct film; cap = 150/60*150*0.8 = 300 words
    film_secs = [150.0] * 10
    caps = [s / 60 * 150 * 0.8 for s in film_secs]
    b = script_mod._visual_matched_budgets(2250, film_secs, caps)
    assert len(b) == 10 and all(x <= 300 for x in b)
    assert abs(sum(b) - 2250) <= 10, "normal target preserved"

    # extreme target: caps bind but nothing exceeds its footage capacity
    b2 = script_mod._visual_matched_budgets(6000, film_secs, caps)
    assert all(x <= 300 for x in b2), "no section may outrun its footage"

    # one dialogue-dense (tiny) section: capped, the others absorb the excess
    film_secs2 = [150.0, 150.0, 20.0, 150.0]
    caps2 = [s / 60 * 150 * 0.8 for s in film_secs2]
    b3 = script_mod._visual_matched_budgets(600, film_secs2, caps2)
    assert b3[2] <= caps2[2] + 0.5, "dense section capped at 1x footage"
    assert abs(sum(b3) - 600) <= 20, "total preserved (rounding/floor slack)"
    print("ok: budgets sized by film time, dense sections capped, total kept")


def test_paced_anchors_spread_clusters() -> None:
    """Clustered anchors are spread one narration-length apart; well-spread
    anchors are left alone."""
    from recap import script as script_mod

    sents = ["The pilot wakes up in a forest full of tall dark trees."] * 4
    # four sentences all anchored onto one busy 2-second moment
    raw = [1001.0, 1001.4, 1001.8, 1002.2]
    out = script_mod._paced_anchors(raw, sents, 1000.0, 1150.0, 150)
    assert all(out[i] > out[i - 1] for i in range(1, 4)), "must be monotone"
    # each gap must fit BOTH adjacent windows: _anchor_windows splits every
    # gap at the midpoint and each window reaches `lead` back to its anchor,
    # so a gap must be 2 x (narration - lead) wide
    est = 12 / 150 * 60 * 1.15 + 1.0
    need = 2 * (est - 0.8)
    assert all(out[i] - out[i - 1] >= need - 1e-6 for i in range(1, 4))
    assert out[-1] <= 1150.0

    # already-spread anchors stay where they are (except the half-step for
    # the first window, by design)
    raw2 = [1000.0, 1100.0, 1200.0]
    out2 = script_mod._paced_anchors(raw2, sents[:3], 1000.0, 1300.0, 150)
    assert out2[1:] == raw2[1:] and out2[0] >= raw2[0]

    # impossible clustering (no room) -> even spread, still in bounds
    out3 = script_mod._paced_anchors([1148.0] * 4, sents, 1000.0, 1150.0, 150)
    assert all(1000.0 <= a <= 1150.0 for a in out3)
    assert all(out3[i] > out3[i - 1] for i in range(1, 4))
    print("ok: clustered anchors paced apart; sparse anchors untouched")


def test_matched_script_plays_at_1x() -> None:
    """THE PAYOFF: a script budgeted by film time, with paced anchors, plays
    every section at NORMAL SPEED -- even over beats packed 2.4s apart. No
    slow motion, no frozen frames, exact A/V lock."""
    from recap import script as script_mod

    # 63 beats, 2.4s apart, in the film window [1000, 1150)
    beats = [{"t": 1000.0 + i * 2.4, "text": f"beat {i}"} for i in range(63)]
    # film-time budget -> only ~5 sentences for this window (not 63)
    sents = ["The pilot wakes up in a forest full of tall dark trees.",
             "He crawls toward the burning wreck of his own plane.",
             "An armed stranger inspects the smoke and finds him there.",
             "The stranger raises his gun and shouts a warning at him.",
             "The pilot grabs the barrel and both men fall down hard."]
    # writer anchored the 5 sentences onto the FIRST five (clustered) beats
    raw = [b["t"] for b in beats[:5]]
    paced = script_mod._paced_anchors(raw, sents, 1000.0, 1150.0, 150)
    wins = script_mod._anchor_windows(
        1000.0, 1150.0, beats, len(sents),
        anchor_values=paced, lead=0.8, tail=6.0,
    )
    segs = [{"sentence": s, "film_start": lo, "film_end": hi}
            for s, (lo, hi) in zip(sents, wins)]

    # narration: 4.8s speech + 0.4s pause per sentence
    cues = [TimedCue(s, i * 5.2, i * 5.2 + 4.8) for i, s in enumerate(sents)]
    span = 4 * 5.2 + 4.8
    durs = timeline.lock_durations(cues, span)
    stats: dict = {}
    beats_tl = timeline.build_timeline(segs, durs, 6000.0, dict(CFG), stats=stats)

    assert stats.get("slowed_groups", 0) == 0, "no slow motion anywhere"
    assert stats.get("held_shots", 0) == 0, "no frozen frames"
    for _s, _d, f, sp in _cuts_in_order(beats_tl):
        assert abs(sp - 1.0) < 1e-9, f"expected 1x, got {sp}x"
        assert f == 0.0
    total = sum(d for _, d, _f, _v in _cuts_in_order(beats_tl))
    assert abs(total - span) < 1e-6, "A/V lock exact"
    prev_end = -1.0
    for s, d, f, sp in _cuts_in_order(beats_tl):
        assert s >= prev_end - 1e-6, "no replay"
        prev_end = max(prev_end, s + (d - f) * sp)
    print(f"ok: matched script over 2.4s-packed beats -> all 1x, "
          f"{total:.1f}s locked, no slow-mo, no freeze")


def test_fit_section_to_footage() -> None:
    """The per-section budget is a CEILING: over-delivered sections are
    trimmed to what their footage can show at 1x."""
    from recap import script as script_mod

    s = "The pilot wakes up in a forest full of tall dark trees."
    assert count_words(s) == 12
    sents = [s] * 20  # 240 words for a 150-word section
    out = script_mod._fit_section_to_footage(sents, 150, [])
    assert len(out) < 20 and count_words(" ".join(out)) <= 150
    assert len(out) >= 3
    assert out[0] == sents[0] and out[-1] == sents[-1], "hand-off ends kept"

    # name-bearing middle sentences survive the trim
    named = [s] * 20
    named[7] = "Troy grabs his gun and aims it at the armed stranger."
    out2 = script_mod._fit_section_to_footage(named, 150, ["Troy"])
    assert any("Troy" in x for x in out2), "name-bearing sentence kept"

    # within budget -> untouched; tiny section -> floor of 3 kept
    assert script_mod._fit_section_to_footage([s] * 5, 150, []) == [s] * 5
    out3 = script_mod._fit_section_to_footage([s] * 3, 10, [])
    assert len(out3) == 3, "never trim below 3 sentences"
    print("ok: over-delivered sections trimmed to their footage; "
          "names + hand-off ends kept")


def test_overdelivered_section_still_plays_at_1x() -> None:
    """THE USER'S BUG, reproduced and fixed: the writer over-delivers (240
    words for a 150-word section). Untrimmed, the section cannot pace its
    anchors 1x-safe, falls back to an even spread, every window is shorter
    than its sentence -> the timeline slow-moes the whole section and the
    narration runs AHEAD of the picture. After the hard fit, the same
    over-delivery plays every cut at exactly 1x."""
    from recap import script as script_mod

    beats = [{"t": 1000.0 + i * 2.4, "text": f"beat {i}"} for i in range(63)]
    s = "The pilot wakes up in a forest full of tall dark trees."
    over = [s] * 20                      # 240 words; writer ignored the budget
    cap = 150                            # 150s of film x 1 word/s (150 wpm, 0.4)

    # --- before the fix: over-budget section drifts into slow motion ------
    raw = [b["t"] for b in beats[:20]]   # clustered anchors on busy beats
    paced = script_mod._paced_anchors(raw, over, 1000.0, 1150.0, 150)
    wins = script_mod._anchor_windows(
        1000.0, 1150.0, beats, len(over),
        anchor_values=paced, lead=0.8, tail=6.0,
    )
    segs = [{"sentence": x, "film_start": lo, "film_end": hi}
            for x, (lo, hi) in zip(over, wins)]
    cues = [TimedCue(x, i * 5.2, i * 5.2 + 4.8) for i, x in enumerate(over)]
    stats: dict = {}
    timeline.build_timeline(segs, timeline.lock_durations(cues, 20 * 5.2),
                            6000.0, dict(CFG), stats=stats)
    assert stats.get("slowed_groups", 0) > 0, \
        "over-budget section must be the slow-mo case (bug reproduction)"

    # --- after the fix: trimmed to the footage -> all 1x, nothing frozen --
    fitted = script_mod._fit_section_to_footage(over, cap, [])
    assert count_words(" ".join(fitted)) <= cap
    n = len(fitted)
    paced2 = script_mod._paced_anchors(
        [b["t"] for b in beats[:n]], fitted, 1000.0, 1150.0, 150)
    wins2 = script_mod._anchor_windows(
        1000.0, 1150.0, beats, n,
        anchor_values=paced2, lead=0.8, tail=6.0,
    )
    segs2 = [{"sentence": x, "film_start": lo, "film_end": hi}
             for x, (lo, hi) in zip(fitted, wins2)]
    span = (n - 1) * 5.2 + 4.8
    cues2 = [TimedCue(x, i * 5.2, i * 5.2 + 4.8) for i, x in enumerate(fitted)]
    durs = timeline.lock_durations(cues2, span)
    stats2: dict = {}
    btl = timeline.build_timeline(segs2, durs, 6000.0, dict(CFG), stats=stats2)

    assert stats2.get("slowed_groups", 0) == 0, "no slow motion after the fit"
    assert stats2.get("held_shots", 0) == 0, "no frozen frames after the fit"
    speeds = [sp for _, _d, _f, sp in _cuts_in_order(btl)]
    assert speeds and all(abs(sp - 1.0) < 1e-9 for sp in speeds), \
        f"expected all 1x, got {sorted(set(round(x, 3) for x in speeds))}"
    total = sum(d for _, d, _f, _v in _cuts_in_order(btl))
    assert abs(total - span) < 1e-6, "A/V lock exact"
    prev_end = -1.0
    for st, d, f, sp in _cuts_in_order(btl):
        assert st >= prev_end - 1e-6, "no replay"
        prev_end = max(prev_end, st + (d - f) * sp)
    print(f"ok: 240-word over-delivery -> {count_words(' '.join(fitted))} "
          f"words/{n} sentences -> all 1x, {total:.1f}s exact lock "
          f"(untrimmed: {stats.get('slowed_groups')} slowed groups)")


def test_condense_section_tightens_story() -> None:
    """Over-delivered sections are CONDENSED (same story, tighter wording),
    not sentence-dropped -- dropping middles is only the backstop."""
    import json
    from recap import script as script_mod

    s = "The pilot wakes up in a forest full of tall dark trees."
    over = [s] * 20  # 240 words
    condensed = ["The pilot wakes in pain in a forest.",
                 "His crashed plane burns behind him.",
                 "An armed stranger finds the wreck.",
                 "The pilot grabs his gun and passes out."]  # 33 words
    calls = []

    def fake_complete(provider, model, system, user, **kw):
        calls.append(user)
        return json.dumps({"sentences": condensed})

    orig = script_mod.llm.complete
    script_mod.llm.complete = fake_complete
    try:
        out = script_mod._condense_section(
            {"provider": "deepseek", "model": "x"}, over, 150, ["Troy"])
    finally:
        script_mod.llm.complete = orig
    assert out == condensed, "in-budget rewrite accepted verbatim"
    assert "AT MOST 150 words" in calls[0]
    assert "SAME order" in calls[0], "must demand the causal chain be kept"
    assert "Troy" in calls[0], "names must be required in the rewrite"

    # rewrite that still overshoots -> None (caller falls back to the trim)
    script_mod.llm.complete = lambda *a, **k: json.dumps({"sentences": over})
    try:
        assert script_mod._condense_section(
            {"provider": "deepseek", "model": "x"}, over, 150, []) is None
    finally:
        script_mod.llm.complete = orig

    # provider blow-up -> None, never an exception into the pipeline
    def boom(*a, **k):
        raise RuntimeError("api down")
    script_mod.llm.complete = boom
    try:
        assert script_mod._condense_section(
            {"provider": "deepseek", "model": "x"}, over, 150, []) is None
    finally:
        script_mod.llm.complete = orig
    print("ok: condense pass accepts in-budget rewrites, rejects "
          "overshoots, survives api failures")


def test_overdelivery_is_condensed_not_dropped() -> None:
    """Full loop: the writer returns 240 words for a 150-word footage
    budget; the pipeline makes ONE condense call (not a mechanical drop),
    and the condensed section -- story kept -- is what reaches the
    timeline, windows attached."""
    import json
    from recap import script as script_mod

    s = "The pilot wakes up in a forest full of tall dark trees."
    over = [s] * 20                                     # 240 words
    condensed = [                                       # 120 words, ordered
        "The pilot wakes up hurt in a dark forest near his burning plane.",
        "A stranger with a gun inspects the wreck and finds him.",
        "The pilot grabs the barrel and both men fall hard.",
        "He drags himself away and hides in the trees.",
        "By morning the whole army is tracking his trail.",
        "He crosses a frozen river to throw the dogs off.",
        "A village family hides him inside their barn.",
        "The soldiers search the village house by house.",
        "He slips out at night and steals a truck.",
        "The chase ends at the border bridge at dawn.",
    ]
    assert count_words(" ".join(condensed)) == 96
    calls = []

    def fake_complete(provider, model, system, user, **kw):
        calls.append(user)
        if len(calls) == 1:
            return json.dumps({"sentences": over})
        return json.dumps({"sentences": condensed})

    chunk = {
        "index": 0, "start": 1000.0, "end": 1150.0,
        "summary": "A pilot is shot down and hunted through the woods.",
        "beats": [{"t": 1000.0 + i * 2.4, "text": f"beat {i}"}
                  for i in range(63)],
    }
    orig = script_mod.llm.complete
    script_mod.llm.complete = fake_complete
    try:
        out = script_mod.generate_segmented_script(
            [chunk], {"provider": "deepseek", "model": "x"}, 200,
            words_per_minute=150, lang_name="Spanish",
            sign_off=False, visual_match=True, humanize=False,
        )
    finally:
        script_mod.llm.complete = orig

    assert len(calls) == 2, ("one write + one condense, no retries: "
                             f"got {len(calls)} calls")
    assert "AT MOST 150 words" in calls[1], "second call is the condense"
    assert [o["sentence"] for o in out] == condensed, \
        "the condensed story (not the dropped-middles version) is the script"
    assert all("film_start" in o and "film_end" in o for o in out)
    assert all(o["film_end"] > o["film_start"] for o in out)
    prev = -1.0
    for o in out:
        assert o["film_start"] >= prev - 1e-9, "windows stay forward"
        prev = o["film_start"]
    print(f"ok: 240-word over-delivery -> one condense call -> "
          f"{count_words(' '.join(condensed))}-word ordered story with "
          "windows attached")


def test_humanize_script_pass() -> None:
    """The humanizer pass (adapted from blader/humanizer, MIT): one call,
    exact sentence count, AI-tell patterns in the prompt, graceful
    failure."""
    import json
    from recap import script as script_mod

    s = "The pilot wakes up in a forest full of tall dark trees."
    sents = [s] * 8
    ok = ["The pilot comes to in a dark forest, every breath hurting."] * 8
    calls = []

    def fake_complete(provider, model, system, user, **kw):
        calls.append((system, user))
        return json.dumps({"sentences": ok})

    orig = script_mod.llm.complete
    script_mod.llm.complete = fake_complete
    try:
        out = script_mod._humanize_script(
            {"provider": "deepseek", "model": "x"}, sents, "English")
    finally:
        script_mod.llm.complete = orig
    assert out == ok
    system, user = calls[0]
    assert system is script_mod.SYSTEM_HUMANIZER
    # the pattern pack + timing constraints are all in the prompt
    for needle in ("not just X, it's Y", "showcase", "EXACTLY 8",
                   "10% longer", "English", "FINISHED SCRIPT"):
        assert needle in user, needle
    assert "1. The pilot wakes up" in user, "sentences are numbered 1:1"

    # wrong sentence count -> None (caller keeps the original script)
    script_mod.llm.complete = lambda *a, **k: json.dumps({"sentences": ok[:5]})
    try:
        assert script_mod._humanize_script(
            {"provider": "deepseek", "model": "x"}, sents, "English") is None
    finally:
        script_mod.llm.complete = orig

    # api blow-up -> None, never an exception into the pipeline
    def boom(*a, **k):
        raise RuntimeError("api down")
    script_mod.llm.complete = boom
    try:
        assert script_mod._humanize_script(
            {"provider": "deepseek", "model": "x"}, sents, "English") is None
    finally:
        script_mod.llm.complete = orig
    print("ok: humanizer call carries the pattern pack + timing locks; "
          "bad counts and api failures fall back to the original")


def test_humanizer_full_loop_keeps_timing() -> None:
    """Full loop with humanize=True: the final pass rewrites AI-telling
    lines, but a rewrite that would outrun its film window (over +10% +2
    words) is rejected per-sentence and the original line kept."""
    import json
    from recap import script as script_mod

    written = [
        "It is not just a crash site, it is the start of a manhunt.",
        "The pilot wakes up in a forest full of tall dark trees.",
        "An armed stranger inspects the wreck and finds him.",
        "The pilot grabs the barrel and both men fall down hard.",
        "He drags himself away and hides deep inside the trees.",
        "By morning the whole army is tracking his trail.",
        "He crosses a frozen river to throw the dogs off.",
        "The chase finally ends at the border bridge.",
    ]
    bloated = ("What happens next is that the pilot, who is injured and "
               "exhausted and lying in thick snow, slowly and painfully "
               "begins to crawl toward the wreck while the armed stranger "
               "watches him very closely indeed.")
    humanized = [
        "The crash site becomes the start of a manhunt.",   # shorter: ok
        "The pilot comes to in a dark forest, hurting.",    # shorter: ok
        bloated,                                            # way over: reject
        written[3],                                         # identical: keep
        "He drags himself into the trees to hide.",         # shorter: ok
        "By morning the army is tracking his trail.",       # shorter: ok
        "He crosses a frozen river to lose the dogs.",      # shorter: ok
        "The chase ends at the border bridge.",             # shorter: ok
    ]

    def fake_complete(provider, model, system, user, **kw):
        if "FINISHED SCRIPT" in user:
            return json.dumps({"sentences": humanized})
        return json.dumps({"sentences": written})

    chunk = {
        "index": 0, "start": 1000.0, "end": 1150.0,
        "summary": "A pilot is shot down and hunted through the woods.",
        "beats": [{"t": 1000.0 + i * 2.4, "text": f"beat {i}"}
                  for i in range(63)],
    }
    orig = script_mod.llm.complete
    script_mod.llm.complete = fake_complete
    try:
        out = script_mod.generate_segmented_script(
            [chunk], {"provider": "deepseek", "model": "x"}, 200,
            words_per_minute=150, lang_name="Spanish",
            sign_off=False, visual_match=True, humanize=True,
        )
    finally:
        script_mod.llm.complete = orig

    assert len(out) == len(written), "sentence count is the timing lock"
    assert out[0]["sentence"] == humanized[0], "fitting rewrite accepted"
    assert out[1]["sentence"] == humanized[1]
    assert out[2]["sentence"] == written[2], \
        "over-length rewrite rejected, original kept"
    assert out[3]["sentence"] == written[3], "identical line untouched"
    assert out[7]["sentence"] == humanized[7]
    assert all(o["film_end"] > o["film_start"] for o in out), \
        "every sentence keeps its film window"
    print("ok: humanizer rewrites flow through; a rewrite that would "
          "outrun its footage is rejected and the original kept")


def test_rewindow_to_speech_guarantees_1x() -> None:
    """THE REPORTED BUG: 'the narration gets ahead from the start, the
    visuals move slowly.' The script sized every window from an ESTIMATE
    (words / words_per_minute); the real voice speaks ~35% slower than
    that, so every window is smaller than its sentence and the timeline
    slow-moes the whole video. rewindow_to_speech() re-sizes the windows
    from the MEASURED durations -> everything plays at 1x."""
    # 3 sections x 270s zones; 5 sentences each, est-sized windows (~6.5s,
    # what the 150-wpm estimate produces for a 12-word sentence)
    s = "The pilot wakes up in a forest full of tall dark trees."
    segs: list[dict] = []
    for sec in range(3):
        zl, zh = sec * 270.0, sec * 270.0 + 270.0
        for k in range(5):
            a = zl + 20.0 + k * 16.0
            segs.append({"sentence": s, "film_start": a - 2.6,
                         "film_end": a + 3.9,
                         "zone_lo": zl, "zone_hi": zh})
    # the REAL audio: the voice speaks 35% slower than the estimate ->
    # every cue spans 9.0s, not 6.5s
    n = len(segs)
    cues = [TimedCue(s, i * 9.0, i * 9.0 + 8.4) for i in range(n)]
    span = (n - 1) * 9.0 + 8.4
    durs = timeline.lock_durations(cues, span)

    # --- before the fix: every section slow-moes --------------------------
    stats_bug: dict = {}
    timeline.build_timeline(segs, durs, 6000.0, dict(CFG), stats=stats_bug)
    assert stats_bug.get("slowed_groups", 0) >= 3, \
        "estimate-sized windows + slower voice must slow-mo (bug repro)"

    # --- after the fix: windows re-sized to the measured speech -----------
    rewin = timeline.rewindow_to_speech(segs, durs, 6000.0)
    for i, o in enumerate(rewin):
        zl, zh = segs[i]["zone_lo"], segs[i]["zone_hi"]
        assert zl - 1e-6 <= o["film_start"] < o["film_end"] <= zh + 1e-6, \
            "window stays inside its section's zone"
        assert o["film_end"] - o["film_start"] >= durs[i] - 1e-6, \
            "window is at least the sentence's real duration -> 1x fits"
        if i and segs[i]["zone_lo"] == segs[i - 1]["zone_lo"]:
            assert abs(o["film_start"] - rewin[i - 1]["film_end"]) <= 0.01, \
                "windows walk the zone contiguously"
    stats: dict = {}
    btl = timeline.build_timeline(rewin, durs, 6000.0, dict(CFG), stats=stats)
    assert stats.get("slowed_groups", 0) == 0, "no slow motion anywhere"
    assert stats.get("held_shots", 0) == 0, "no frozen frames"
    for _s0, _d0, f0, sp0 in _cuts_in_order(btl):
        assert abs(sp0 - 1.0) < 1e-9, f"expected 1x, got {sp0}x"
        assert f0 == 0.0
    total = sum(d for _, d, _f, _v in _cuts_in_order(btl))
    assert abs(total - span) < 1e-6, "A/V lock exact"
    prev_end = -1.0
    for st0, d0, f0, sp0 in _cuts_in_order(btl):
        assert st0 >= prev_end - 1e-6, "no replay"
        prev_end = max(prev_end, st0 + (d0 - f0) * sp0)
    # the picture stays WITH the narration's zone (no look-ahead)
    for b, o in zip(btl, rewin):
        assert o["zone_lo"] - 1e-6 <= b["film_start"] <= o["zone_hi"] + 1e-6
    print(f"ok: 35%-slower voice -> windows re-sized from measured speech "
          f"-> all 1x, {total:.0f}s exact lock (before: "
          f"{stats_bug.get('slowed_groups')} slowed sections)")


def test_rewindow_overbudget_walks_contiguously() -> None:
    """A section whose measured narration genuinely exceeds its film zone
    keeps a contiguous, monotone walk (the slow-mo net then handles it --
    rare and honest)."""
    s = "The pilot wakes up in a forest full of tall dark trees."
    segs = [{"sentence": s, "film_start": 10.0 + i, "film_end": 10.5 + i,
             "zone_lo": 0.0, "zone_hi": 100.0} for i in range(15)]
    durs = [9.0] * 15                      # 135s of speech, 100s of film
    rewin = timeline.rewindow_to_speech(segs, durs, 6000.0)
    prev_hi = 0.0
    for i, o in enumerate(rewin):
        assert o["film_start"] >= prev_hi - 0.01, "contiguous, monotone"
        assert o["film_end"] <= 100.0 + 1e-6, "never leaves the zone"
        prev_hi = o["film_end"]
    assert abs(prev_hi - 100.0) <= 0.01, "the walk fills the zone exactly"


def test_rewindow_preserves_unzoned_sentences() -> None:
    """Sentences without zone info (the sign-off outro, old cached
    segments) keep their windows untouched."""
    segs = [{"sentence": "Thanks for watching.", "film_start": 400.0,
             "film_end": 406.0}]
    out = timeline.rewindow_to_speech(segs, [8.0], 6000.0)
    assert out[0]["film_start"] == 400.0 and out[0]["film_end"] == 406.0


if __name__ == "__main__":
    test_no_replay_same_window()
    test_no_replay_overlapping_windows()
    test_no_replay_long_run_random_windows()
    test_snap_to_boundary()
    test_timeline_uses_scene_bounds()
    test_snap_never_breaks_monotonic_playhead()
    test_end_of_film_stays_in_bounds()
    test_visuals_never_run_ahead_of_the_narration()
    test_holds_keep_av_lock_in_sparse_sections()
    test_slow_motion_cut_command()
    test_freeze_cut_command()
    test_visual_matched_budgets()
    test_paced_anchors_spread_clusters()
    test_matched_script_plays_at_1x()
    test_fit_section_to_footage()
    test_overdelivered_section_still_plays_at_1x()
    test_condense_section_tightens_story()
    test_overdelivery_is_condensed_not_dropped()
    test_humanize_script_pass()
    test_humanizer_full_loop_keeps_timing()
    test_rewindow_to_speech_guarantees_1x()
    test_rewindow_overbudget_walks_contiguously()
    test_rewindow_preserves_unzoned_sentences()
    print("\nALL VISUAL-FLOW TESTS PASSED")
