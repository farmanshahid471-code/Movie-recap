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
    print("\nALL VISUAL-FLOW TESTS PASSED")
