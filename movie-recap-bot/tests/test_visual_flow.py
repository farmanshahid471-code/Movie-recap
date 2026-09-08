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
from recap.tts import TimedCue  # noqa: E402

CFG = {"micro_cut_seconds": 2.4, "max_cuts_per_beat": 4, "min_cut_seconds": 1.2}


def _cuts_in_order(beats):
    for b in beats:
        for s, d in b["cuts"]:
            yield s, d


def test_no_replay_same_window() -> None:
    """Many sentences anchored to ONE tight film moment must never rewind."""
    cues = [TimedCue(f"S{i}.", i * 4.0, i * 4.0 + 3.2) for i in range(3)]
    durs = timeline.lock_durations(cues, 12.0)
    sents = [{"sentence": c.text, "film_start": 100.0, "film_end": 103.0}
             for c in cues]
    beats = timeline.build_timeline(sents, durs, 6000.0, CFG)
    prev_end = -1.0
    for s, d in _cuts_in_order(beats):
        assert s >= prev_end - 1e-6, (
            f"cut at {s:.2f}s replays footage that ended at {prev_end:.2f}s"
        )
        prev_end = max(prev_end, s + d)
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
    for s, d in _cuts_in_order(beats):
        assert s >= prev_end - 1e-6, f"rewind: {s:.2f} < {prev_end:.2f}"
        prev_end = max(prev_end, s + d)
    # length lock still exact
    total = sum(d for _, d in _cuts_in_order(beats))
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
    for s, d in _cuts_in_order(beats):
        assert s >= prev_end - 1e-6, (
            f"beat rewound the film: {s:.2f}s after {prev_end:.2f}s"
        )
        prev_end = max(prev_end, s + d)
    total = sum(d for _, d in _cuts_in_order(beats))
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
    for s, d in _cuts_in_order(beats):
        assert s >= prev_end - 1e-6, "snap caused a rewind"
        if any(abs(s - b) < 1e-6 for b in bounds):
            on_boundary += 1
        total_cuts += 1
        prev_end = max(prev_end, s + d)
    total = sum(d for _, d in _cuts_in_order(beats))
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
    for s, d in _cuts_in_order(beats):
        assert s >= prev_end - 1e-6, "snap rewound below the film playhead"
        prev_end = max(prev_end, s + d)
    print("ok: snapping respects the film playhead (no forced rewinds)")


def test_end_of_film_stays_in_bounds() -> None:
    """Narration longer than the film: cuts clamp inside the movie."""
    cues = [TimedCue(f"S{i}.", i * 9.0, i * 9.0 + 8.0) for i in range(40)]
    span = 39 * 9.0 + 8.0
    durs = timeline.lock_durations(cues, span)
    sents = [{"sentence": c.text, "film_start": 0.0, "film_end": 300.0}
             for c in cues]
    beats = timeline.build_timeline(sents, durs, 300.0, CFG)
    for s, d in _cuts_in_order(beats):
        assert 0.0 <= s <= 300.0
    total = sum(d for _, d in _cuts_in_order(beats))
    assert abs(total - span) < 0.5
    print("ok: end-of-film clamp keeps every cut inside the movie")


if __name__ == "__main__":
    test_no_replay_same_window()
    test_no_replay_overlapping_windows()
    test_no_replay_long_run_random_windows()
    test_snap_to_boundary()
    test_timeline_uses_scene_bounds()
    test_snap_never_breaks_monotonic_playhead()
    test_end_of_film_stays_in_bounds()
    print("\nALL VISUAL-FLOW TESTS PASSED")
