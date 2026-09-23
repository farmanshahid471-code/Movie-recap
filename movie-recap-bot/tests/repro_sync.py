"""Repro harness: drive the REAL pipeline pieces (chunking, zone budgeting,
segmented-script windows, rewindow_to_speech, build_timeline) with realistic
inputs and audit the resulting visual track for:
  1. footage that does not match the narrated section (lead/lag vs zone),
  2. 'bursts': narration stretches where the visual is a single (frozen)
     frame while many sentences are spoken.

Run:  ../.venv/bin/python tests/repro_sync.py [scenario]
scenarios:
  full2h        2h movie, dialogue across the whole film (baseline sanity)
  short-sample  90s sample movie + default 2520-word target (Studio default)
  mid-sample    240s sample movie + default target
  partial-srt   2h movie but subtitle only covers the first 10 min
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap import chunk as chunk_mod          # noqa: E402
from recap import pipeline, summarize, timeline  # noqa: E402
from recap import script as script_mod        # noqa: E402
from recap.tts import TimedCue                # noqa: E402

CFG_TL = {  # config.yaml defaults
    "micro_cut_seconds": 2.4,
    "max_cuts_per_beat": 4,
    "min_cut_seconds": 1.2,
    "pre_roll": 0.4,
    "cut_on_words": True,
    "snap_to_scenes": True,
    "snap_tolerance": 0.8,
    "max_lead_seconds": 3.0,
    "min_speed": 0.6,
    "max_shot_seconds": 7.0,
    "min_new_footage": 0.8,
}
WPM = 180
TARGET = 2520  # Studio default 840s at 180wpm


# ---------------------------------------------------------------------------
# fake LLM
# ---------------------------------------------------------------------------
class FakeLLM:
    """Returns budget-sized sentence lists; counts calls."""

    def __init__(self):
        self.calls = 0

    def complete(self, provider, model, system, user, **kw):
        self.calls += 1
        if "DRAFT SECTION" in user or "FINISHED SCRIPT" in user or "FULL DRAFT" in user:
            # polish/humanize: echo back the same sentences (extract from prompt)
            sents = json.loads(user.split("=== DRAFT SECTION ===")[1].split("=== END DRAFT ===")[0])
            return json.dumps({"sentences": sents})
        if "=== FULL DRAFT ===" in user:
            sents = json.loads(user.split("=== FULL DRAFT ===")[1].split("=== END ===")[0])
            return json.dumps({"sentences": sents})
        if "HARD CEILING" in user:
            budget = int(user.split("about ")[1].split(" words")[0])
            nsent = max(3, round(budget / 17))
            sents = [f"Narration sentence {i} describing the on screen action for this section."
                     for i in range(nsent)]
            return json.dumps({"sentences": sents})
        return json.dumps({"sentences": ["fallback"]})


def fake_transcript(movie_dur: float, coverage: float) -> list[dict]:
    """Realistic dialogue cues: a line every ~12s, ~3s long, over `coverage`
    seconds of the film (coverage may be < movie_dur for a partial srt)."""
    cues = []
    t = 20.0
    i = 0
    while t < coverage - 3:
        cues.append({"text": f"Line {i} of dialogue said by a character.",
                     "start": t, "end": t + 3.0})
        t += 12.0
        i += 1
    return cues


def run_scenario(movie_dur: float, coverage: float, target: int,
                 label: str) -> None:
    print(f"\n{'=' * 74}\nSCENARIO {label}: movie {movie_dur:.0f}s, "
          f"dialogue covers {coverage:.0f}s, target {target} words\n{'=' * 74}")

    cues = fake_transcript(movie_dur, coverage)
    if not cues:
        print("  (no cues -- nothing to recap)")
        return

    # Step A: chunking (real code)
    chunks = chunk_mod.chunk_cues(cues, 180, 30)
    # pipeline clamps each chunk's film territory to the film's own length
    # (a short clip's last window must not claim footage that does not exist)
    for c in chunks:
        if c["end"] > movie_dur:
            c["end"] = movie_dur
    # fake summaries: a beat every ~15s inside the chunk (real times)
    summaries = []
    for c in chunks:
        lines = []
        t = c["start"] + 5
        while t < min(c["end"], coverage):
            h, rem = divmod(int(t), 3600)
            m, s = divmod(rem, 60)
            lines.append(f"[{h:02d}:{m:02d}:{s:02d}] A character does something "
                         f"visible at {h:02d}:{m:02d}:{s:02d}.")
            t += 15.0
        summaries.append("\n".join(lines) or f"[{int(c['start'])}] quiet.")
    chunk_summaries = [
        {"index": c["index"], "start": c["start"], "end": c["end"],
         "summary": s, "beats": summarize.parse_beats(s)}
        for c, s in zip(chunks, summaries)
    ]

    # pipeline-level target clamp (the visual match guard in pipeline.py):
    # never ask for more narration than the footage (transcript coverage,
    # bounded by the film's length) can show at 1x
    coverage = max((float(c.get("end", 0.0)) for c in chunks), default=0.0)
    narratable = min(movie_dur, max(coverage, 0.0)) or movie_dur
    film_cap = int(narratable / 60.0 * WPM * 0.75)
    eff_target = target
    if target > film_cap:
        eff_target = film_cap
        print(f"  * visual match: target clamped {target} -> {film_cap} "
              f"(footage can show {film_cap} words at 1x)")
    print(f"  * effective target: {eff_target} words (~{eff_target / WPM * 60:.0f}s "
          f"of speech)")

    # Step B: segmented script (real code, fake LLM)
    fake = FakeLLM()
    script_mod.llm.complete = fake.complete
    segments = script_mod.generate_segmented_script(
        chunk_summaries, {"provider": "deepseek", "model": "x"}, eff_target,
        words_per_minute=WPM, progress=None, lang_name="English",
        sign_off=True, visual_match=True, humanize=False,
    )
    sentences = [s["sentence"] for s in segments]
    n_words = sum(len(s.split()) for s in sentences)
    print(f"  * script: {len(sentences)} sentences, {n_words} words "
          f"(~{n_words / WPM * 60:.0f}s of speech), LLM calls: {fake.calls}")

    # Step C: TTS cues — one per sentence, ~150 wpm actual (rate -8%),
    # 0.6s silence between sentences (edge-tts pauses)
    cues_t = []
    t = 0.3
    for s in sentences:
        dur = max(len(s.split()) / 150.0 * 60.0, 0.8)
        cues_t.append(TimedCue(s, t, t + dur))
        t += dur + 0.6
    audio_span = t - 0.6
    print(f"  * narration audio: {audio_span:.1f}s")

    # Step D: durations + zones + rewindow + timeline (all real code)
    durations = timeline.lock_durations(cues_t, audio_span)
    seg_for_lang = pipeline._extend_final_zone(segments, movie_dur)
    seg_for_lang = timeline.rewindow_to_speech(seg_for_lang, durations, movie_dur)
    stats: dict = {}
    beats = timeline.build_timeline(
        seg_for_lang, durations, movie_dur, CFG_TL,
        word_times=[c.words for c in cues_t], stats=stats,
    )
    cuts = timeline.flatten_cuts(beats)
    report = timeline.timeline_report(
        beats, audio_span, stats.get("word_locked_beats", 0),
        stats.get("snapped_cuts", 0), stats.get("slowed_groups", 0),
        stats.get("slowed_seconds", 0.0))
    print(f"  * timeline: {report}")

    # ------------------------------------------------------------------
    # AUDIT 1: where does the footage stand while each sentence is spoken?
    # Walk cuts in play order; each cut covers narration [cur, cur+dur) and
    # footage [start, start + (dur-freeze)*speed).
    # ------------------------------------------------------------------
    film_pos_at = []  # (narration_t, sentence_idx, footage_start)
    cur = 0.0
    bi = 0
    for b in beats:
        for (fs, d, frz, spd) in b["cuts"]:
            film_pos_at.append((cur, b["index"], fs))
            cur += d
    # sample at each sentence start
    cum = [0.0]
    for d in durations:
        cum.append(cum[-1] + d)

    mism = 0
    for i, s in enumerate(sentences):
        if i >= len(cum) - 1:
            break
        t0 = cum[i]
        # footage at sentence start
        fps_at = None
        for (nt, sidx, fs) in film_pos_at:
            if nt >= t0 - 1e-6:
                fps_at = fs
                break
        zone_lo = seg_for_lang[i].get("zone_lo")
        zone_hi = seg_for_lang[i].get("zone_hi")
        if fps_at is None or zone_lo is None:
            continue
        # how far outside the sentence's own section zone is the footage?
        if fps_at < zone_lo - 5 or fps_at > zone_hi + 5:
            mism += 1
    print(f"  AUDIT1: {mism}/{len(sentences)} sentences show footage outside "
          f"their own section's zone (>=5s off)")
    # first mismatch
    for i, s in enumerate(sentences[:8]):
        t0 = cum[i]
        fps_at = next((fs for (nt, sidx, fs) in film_pos_at if nt >= t0 - 1e-6), None)
        zl, zh = seg_for_lang[i].get("zone_lo"), seg_for_lang[i].get("zone_hi")
        print(f"    sent {i:3d} @ {t0:6.1f}s: zone [{zl if zl is not None else float('nan'):7.1f}, "
              f"{zh if zh is not None else float('nan'):7.1f}]  footage @ "
              f"{fps_at if fps_at is not None else float('nan'):.1f}s")

    # ------------------------------------------------------------------
    # AUDIT 2: the 'burst' — longest stretch of narration during which the
    # visible footage does NOT move (same frame / <1s of film consumed).
    # ------------------------------------------------------------------
    runs = []
    cur = 0.0
    prev_movie_t = None
    run_start = 0.0
    run_sents = 0
    for (fs, d, frz, spd) in cuts:
        film_t = fs
        if prev_movie_t is None or film_t - prev_movie_t >= 1.0:
            if cur - run_start > 0:
                runs.append((run_start, cur, run_sents,
                             prev_movie_t if prev_movie_t is not None else -1))
            run_start, run_sents = cur, 0
        else:
            run_sents = max(run_sents, 0)
        # count sentences that start inside this cut
        n_sents_in = sum(1 for i in range(len(cum) - 1)
                         if cum[i] >= cur - 1e-6 and cum[i] < cur + d - 1e-6)
        run_sents = max(run_sents, n_sents_in) if film_t - (prev_movie_t or 0) < 1.0 else run_sents
        prev_movie_t = film_t + (d - frz) * spd if film_t is not None else None
        cur += d
    if cur - run_start > 0:
        runs.append((run_start, cur, run_sents, prev_movie_t or -1))

    # simpler, robust detector: freeze seconds per cut
    frozen = [(fs, d, frz, spd) for (fs, d, frz, spd) in cuts if frz > 0.05]
    total_frozen = sum(f for _, _, f, _ in frozen)
    n_frozen_sents = 0
    t = 0.0
    for (fs, d, frz, spd) in cuts:
        for i in range(len(cum) - 1):
            if cum[i] >= t - 1e-6 and cum[i] < t + d - 1e-6:
                if frz > 0.05:
                    n_frozen_sents += 1
        t += d
    print(f"  AUDIT2: frozen-frame seconds: {total_frozen:.1f}s of "
          f"{audio_span:.1f}s; sentences spoken over frozen footage: "
          f"{n_frozen_sents}/{len(sentences)}")
    if frozen:
        first = frozen[0]
        at_narr_t = 0.0
        for (fs2, d2, f2, s2) in cuts:
            if (fs2, d2, f2, s2) is first:
                break
            at_narr_t += d2
        print(f"    first freeze starts at ~{at_narr_t:.1f}s into the video, "
              f"footage frozen at film time {first[0]:.1f}s (movie is "
              f"{movie_dur:.0f}s)")

    # where does the film playhead end up?
    play = 0.0
    for (fs, d, frz, spd) in cuts:
        play = max(play, fs + (d - frz) * spd)
    print(f"  AUDIT3: film footage consumed: {play:.0f}s of {movie_dur:.0f}s "
          f"({play / max(movie_dur, 1) * 100:.0f}%)")


if __name__ == "__main__":
    scen = sys.argv[1] if len(sys.argv) > 1 else "full2h"
    if scen == "full2h":
        run_scenario(7200, 7200, TARGET, "full2h")
    elif scen == "short-sample":
        run_scenario(90, 90, TARGET, "short 90s sample + 2520-word target")
    elif scen == "mid-sample":
        run_scenario(240, 240, TARGET, "mid 240s sample + 2520-word target")
    elif scen == "partial-srt":
        run_scenario(7200, 600, TARGET, "2h movie, srt covers first 10min")
