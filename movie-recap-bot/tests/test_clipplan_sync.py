"""Tests for the AI-Movie-Shorts clip-plan sync (recap/clipplan.py).

Run:  python tests/test_clipplan_sync.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap import clipplan, timeline  # noqa: E402


def seg(text, lo, hi, zone=None):
    return {"sentence": text, "film_start": lo, "film_end": hi,
            "anchor": (lo + hi) / 2,
            "zone_lo": (zone or (lo, hi))[0], "zone_hi": (zone or (lo, hi))[1]}


def test_parse_plan_drops_unusable_items():
    plan = clipplan.parse_clip_plan_json(
        '{"clips":[{"start":10,"end":20,"narration":"A happens."},'
        '{"start":15,"end":12,"narration":"bad"},'
        '{"start":30,"end":40,"narration":""}]}')
    assert len(plan) == 1 and plan[0]["narration"] == "A happens."
    print("ok: plan parsing drops unusable items")


def test_validate_plan_non_overlapping_chronological():
    plan = clipplan.validate_clip_plan([
        {"start": 20, "end": 30, "narration": "b"},
        {"start": 10, "end": 25, "narration": "a"},
    ])
    assert [p["narration"] for p in plan] == ["a", "b"]
    assert plan[1]["start"] == 25  # trimmed, not replayed
    print("ok: plan is chronological + non-overlapping")


def test_fit_speedup_within_cap_uses_whole_window():
    s, e, v = clipplan.fit_window_to_narration(100, 114, 8.0)  # 14s over 8s
    assert (s, e) == (100, 114) and abs(v - 1.75) < 1e-6
    print("ok: whole window fitted at exactly the cap")


def test_fit_speedup_over_cap_centre_trims():
    # Reference case: 30s window, 8s narration -> would be 3.75x, so the
    # middle 14s (8*1.75) is shown at 1.75x, not the first 14s.
    s, e, v = clipplan.fit_window_to_narration(100, 130, 8.0)
    assert (s, e) == (108.0, 122.0) and abs(v - 1.75) < 1e-6
    print("ok: over-cap window is centre-trimmed (reference rule)")


def test_fit_long_narration_slows_same_scene_never_extends():
    s, e, v = clipplan.fit_window_to_narration(100, 110, 20.0)
    assert (s, e) == (100, 110) and abs(v - 0.5) < 1e-6
    print("ok: long narration slows its own scene, never borrows the next")


def test_one_clip_per_sentence_inside_own_zone():
    segs = [seg("One.", 0, 10), seg("Two.", 10, 20), seg("Three.", 20, 50)]
    durs = [5.0, 5.0, 10.0]
    beats = clipplan.build_sync_beats(segs, durs, 100.0, {})
    assert len(beats) == 3 and all(len(b["cuts"]) == 1 for b in beats)
    for b, s in zip(beats, segs):
        st, d, f, v = b["cuts"][0]
        assert s["zone_lo"] <= st and st + d * v <= s["zone_hi"] + 1e-6, b
    # consumed footage never overlaps / rewinds
    ends = [b["cuts"][0][0] + b["cuts"][0][1] * b["cuts"][0][3] for b in beats]
    starts = [b["cuts"][0][0] for b in beats]
    assert all(starts[i] >= ends[i - 1] - 1e-6 for i in (1, 2))
    assert timeline.cut_order_violations(timeline.flatten_cuts(beats)) == []
    print("ok: one clip per sentence, confined to its own zone, no replay")


def test_total_duration_is_audio_lock_unchanged():
    segs = [seg("One.", 0, 30), seg("Two.", 30, 40)]
    durs = [4.5, 9.0]
    beats = clipplan.build_sync_beats(segs, durs, 100.0, {})
    total = sum(d for _, d, _f, _v in timeline.flatten_cuts(beats))
    assert abs(total - sum(durs)) < 1e-6
    print("ok: sum(cuts) == sum(narration durations) exactly")


def test_long_sentence_does_not_push_next_clip_off_its_scene():
    # Sentence 1 narrates far longer than its window: the old playhead rule
    # pushed sentence 2's clip past its own moment. Clip-plan sync slows
    # scene 1 instead, so clip 2 still starts inside scene 2.
    segs = [seg("Long one.", 0, 6), seg("Next scene.", 100, 110)]
    beats = clipplan.build_sync_beats(segs, [20.0, 5.0], 200.0, {})
    assert 100.0 <= beats[1]["cuts"][0][0] < 110.0  # still its own scene
    assert beats[1]["film_end"] <= 110.0             # (centre-trimmed inside it)
    assert beats[0]["cuts"][0][3] < 1.0  # slowed, not walked forward
    print("ok: an over-long sentence cannot displace the next clip")


def test_segments_from_plan_never_splits_narration():
    segs = clipplan.segments_from_clip_plan(
        [{"start": 10, "end": 22, "narration": "First. Second."}])
    assert len(segs) == 1 and segs[0]["sentence"] == "First. Second."
    print("ok: a plan item stays atomic (narration never re-split)")


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_"):
            fn()
    print("\nall clip-plan sync tests passed")
