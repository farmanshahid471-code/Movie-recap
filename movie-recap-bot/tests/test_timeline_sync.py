"""Regression tests for the two reported production bugs.

BUG 1 — "I asked for 900s and got 360s."
    Two independent causes:
      a) the word target was clamped / the LLM under-delivered with no check;
      b) the visual track was built from the sum of *spoken* clip lengths while
         the mp3 also contains the pauses between sentences, so `-shortest`
         truncated the render to the shorter (visual) stream.

BUG 2 — "The visuals don't match the narration at all."
    Semantic cosine matching picked the best-scoring transcript cue per line,
    which jumps between act 1, 2 and 3 because similar words recur. Beats must
    advance monotonically through the film instead.

Run:  python tests/test_timeline_sync.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap import timeline  # noqa: E402
from recap.tts import TimedCue  # noqa: E402


def _fake_narration(n: int, speech: float = 4.2, gap: float = 1.8):
    """TTS cues with realistic silent gaps between sentences."""
    cues, t = [], 0.0
    for i in range(n):
        cues.append(TimedCue(f"Sentence {i}.", t, t + speech))
        t += speech + gap
    return cues, t - gap  # last gap is not part of the audio


def test_durations_cover_the_whole_audio_span() -> None:
    """BUG 1b: beat durations must sum to the FULL narration, gaps included."""
    cues, span = _fake_narration(150)
    durations = timeline.lock_durations(cues, span)

    spoken_only = sum(c.duration for c in cues)
    total = sum(durations)

    assert abs(total - span) < 1e-6, f"visual {total} != narration {span}"
    # The old code produced only the spoken total, which is why 900s -> ~650s.
    assert total > spoken_only + 200, (
        f"durations must include inter-sentence gaps: {total} vs {spoken_only}"
    )
    assert all(d > 0 for d in durations), "no zero/negative beat"
    print(f"  audio span {span:.1f}s, spoken {spoken_only:.1f}s, "
          f"visual {total:.1f}s -> covered")


def test_no_truncation_for_a_900s_request() -> None:
    """BUG 1: a 900-second narration must yield a 900-second video plan."""
    cues, span = _fake_narration(150)
    assert 890 < span < 910, f"fixture should be ~900s, got {span}"

    durations = timeline.lock_durations(cues, span)
    sentences = [
        {"sentence": c.text,
         "film_start": (i // 10) * 400.0,
         "film_end": (i // 10) * 400.0 + 400.0}
        for i, c in enumerate(cues)
    ]
    beats = timeline.build_timeline(sentences, durations, movie_dur=6000.0)
    cuts = timeline.flatten_cuts(beats)
    video_len = sum(d for _, d in cuts)

    assert abs(video_len - span) < 0.5, (
        f"video {video_len:.1f}s must equal narration {span:.1f}s "
        f"(old pipeline gave ~{sum(c.duration for c in cues):.0f}s)"
    )
    print(f"  900s request -> {video_len:.1f}s of video across {len(cuts)} cuts")


def test_beats_are_strictly_chronological() -> None:
    """BUG 2: the playhead must never jump backwards through the film."""
    cues, span = _fake_narration(60)
    durations = timeline.lock_durations(cues, span)
    sentences = [
        {"sentence": c.text,
         "film_start": (i // 6) * 500.0,
         "film_end": (i // 6) * 500.0 + 500.0}
        for i, c in enumerate(cues)
    ]
    beats = timeline.build_timeline(sentences, durations, movie_dur=5000.0)

    starts = [b["film_start"] for b in beats]
    for i in range(len(starts) - 1):
        assert starts[i] <= starts[i + 1] + 1e-6, (
            f"beat {i + 1} jumps backwards: {starts[i]:.1f}s -> {starts[i + 1]:.1f}s"
        )
    # and it should traverse the film, not sit at the start
    assert starts[-1] > starts[0] + 1000, "montage must walk the whole film"
    print(f"  chronological: {starts[0]:.0f}s -> {starts[-1]:.0f}s, monotonic")


def test_micro_cuts_break_up_long_beats() -> None:
    """A long sentence becomes 2-3 shots, not one static clip."""
    cues, span = _fake_narration(12, speech=7.0, gap=0.5)
    durations = timeline.lock_durations(cues, span)
    sentences = [
        {"sentence": c.text, "film_start": i * 100.0, "film_end": i * 100.0 + 100.0}
        for i, c in enumerate(cues)
    ]
    beats = timeline.build_timeline(
        sentences, durations, 2000.0,
        {"micro_cut_seconds": 3.0, "max_cuts_per_beat": 3, "min_cut_seconds": 1.2},
    )

    multi = [b for b in beats if len(b["cuts"]) > 1]
    assert multi, "long beats must be split into micro-cuts"
    for b in beats:
        assert len(b["cuts"]) <= 3
        assert abs(sum(d for _, d in b["cuts"]) - b["duration"]) < 1e-6, (
            "micro-cuts must exactly fill their beat"
        )
        starts = [s for s, _ in b["cuts"]]
        assert starts == sorted(starts), "shots inside a beat advance forward"
    print(f"  {len(multi)}/{len(beats)} beats split into multiple shots")


def test_cuts_stay_inside_the_film() -> None:
    """Never seek past the end of a short film."""
    cues, span = _fake_narration(40)
    durations = timeline.lock_durations(cues, span)
    movie_dur = 300.0  # narration is longer than the film
    sentences = [
        {"sentence": c.text, "film_start": 0.0, "film_end": movie_dur} for c in cues
    ]
    beats = timeline.build_timeline(sentences, durations, movie_dur)
    for b in beats:
        for start, dur in b["cuts"]:
            assert start >= 0.0, "negative seek"
            assert start <= movie_dur, f"seek {start} past end {movie_dur}"
    # length lock still holds
    assert abs(sum(d for _, d in timeline.flatten_cuts(beats)) - span) < 0.5
    print(f"  all cuts within a {movie_dur:.0f}s film, length still locked")


def test_empty_and_degenerate_inputs() -> None:
    assert timeline.lock_durations([]) == []
    assert timeline.build_timeline([], [], 100.0) == []
    cues, span = _fake_narration(1)
    d = timeline.lock_durations(cues, span)
    b = timeline.build_timeline(
        [{"sentence": "only", "film_start": 0.0, "film_end": 0.0}], d, 0.0
    )
    assert len(b) == 1 and b[0]["cuts"], "single beat must still produce a cut"
    print("  degenerate inputs handled")


if __name__ == "__main__":
    test_durations_cover_the_whole_audio_span()
    print("ok: audio span coverage (BUG 1b)")
    test_no_truncation_for_a_900s_request()
    print("ok: 900s request -> 900s video (BUG 1)")
    test_beats_are_strictly_chronological()
    print("ok: strict chronology (BUG 2)")
    test_micro_cuts_break_up_long_beats()
    print("ok: micro-cuts")
    test_cuts_stay_inside_the_film()
    print("ok: bounds")
    test_empty_and_degenerate_inputs()
    print("ok: edge cases")
    print("\nALL TIMELINE SYNC TESTS PASSED")
