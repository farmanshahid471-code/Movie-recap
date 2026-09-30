from recap import timeline, beats, match


def test_cluster_groups_2_to_4_and_shot_continues():
    sents = [{"sentence": f"s{i}", "anchor": 10.0 + 3 * i, "film_start": 10.0 + 3 * i,
              "film_end": 13.0 + 3 * i} for i in range(9)]
    durs = [3.0] * 9
    cl = timeline.cluster_narrative_beats(sents, durs)
    groups = {}
    for s in cl:
        groups.setdefault(s["beat_group"], []).append(s)
    assert all(2 <= len(g) <= 4 for g in list(groups.values())[:-1])
    tl = timeline.build_timeline(cl, durs, 200.0, {"audio_first": True})
    assert len(tl) == 9                     # nothing dropped
    for a, b in zip(tl, tl[1:]):
        if b["continue_shot"]:
            assert abs(b["film_start"] - a["film_end"]) < 1e-3


def test_jump_starts_new_beat():
    sents = [{"anchor": 0.0}, {"anchor": 100.0}]
    cl = timeline.cluster_narrative_beats(sents, [3.0, 3.0])
    assert [s["continue_shot"] for s in cl] == [False, False]


def test_select_key_beats_merges_not_drops():
    raw = [{"start_ts": i, "end_ts": i + 1, "duration": 1, "transcript_lines": [i],
            "vision_notes": []} for i in range(200)]
    out = beats.select_key_beats(raw, target_beat_count=66)
    assert out[0]["start_ts"] == 0 and out[-1]["end_ts"] == 200
    assert sum(len(b["transcript_lines"]) for b in out) == 200


def test_fallback_never_drops():
    class S:
        def search(self, v, k, min_score, min_start):
            return [{"start": 50.0, "end": 52.0, "score": 0.2}]
    r = match.find_visual_match_with_fallback(S(), None, 40.0)
    assert r["fallback"] and r["start"] == 40.0 and r["end"] == 45.0


def test_compile_chronological_timeline_continuous():
    tl = timeline.compile_chronological_timeline(
        [{"film_start": 0}, {"film_start": 10}, {"film_start": 5}], 30.0)
    assert [t["start"] for t in tl] == [0, 10, 10]
    assert tl[-1]["end"] == 30.0
