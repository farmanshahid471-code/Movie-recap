"""Audit: does rewindow_to_speech keep each sentence's footage on the beat
the sentence was written from (its anchor), or does it slide the window to a
time-proportional slice of the zone?"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recap import chunk as chunk_mod, pipeline, summarize, timeline
from recap import script as script_mod
from recap.tts import TimedCue

CFG_TL = {"micro_cut_seconds": 2.4, "max_cuts_per_beat": 4, "min_cut_seconds": 1.2,
          "pre_roll": 0.4, "cut_on_words": True, "snap_to_scenes": True,
          "snap_tolerance": 0.8, "max_lead_seconds": 3.0, "min_speed": 0.35,
          "min_new_footage": 0.8}
WPM = 180

# 2h movie. Beats are CLUSTERED at the start of each 150s zone (a long
# dialogue scene, then a quiet stretch) -- realistic: dialogue happens in
# scenes, not uniformly.
def transcript():
    cues = []
    t = 10.0
    i = 0
    while t < 7200 - 3:
        # a 'scene' of dialogue every 30-60s, 15s long, clustered early in
        # each 150s zone: beat times 15, 25, 35, 45 within the zone
        cues.append({"text": f"line {i}", "start": t, "end": t + 2.0})
        t += 10.0 if (int(t) % 150) < 45 else 25.0
        i += 1
    return cues

cues = transcript()
chunks = chunk_mod.chunk_cues(cues, 180, 30)
summaries = []
for c in chunks:
    ts = sorted({q["start"] for q in c["cues"]})
    lines = []
    for t in ts:
        h, rem = divmod(int(t), 3600); m, s = divmod(rem, 60)
        lines.append(f"[{h:02d}:{m:02d}:{s:02d}] A character says something at the scene.")
    summaries.append("\n".join(lines) or "[00:00:01] quiet.")
chunk_summaries = [{"index": c["index"], "start": c["start"], "end": c["end"],
                    "summary": s, "beats": summarize.parse_beats(s)}
                   for c, s in zip(chunks, summaries)]

class FakeLLM:
    def complete(self, provider, model, system, user, **kw):
        if "HARD CEILING" in user:
            budget = int(user.split("about ")[1].split(" words")[0])
            nsent = max(3, round(budget / 17))
            # 3 sentences huddle on the FIRST beat of the zone (real writers
            # spend several sentences on one busy scene)
            first = user.split("=== ACTION BEATS FOR THIS SECTION ===")[1].splitlines()
            t0 = first[0].strip()[:9] if first else "00:00:15"
            sents = [f"{t0[:1]} The scene at {t0} unfolds with the characters talking about their plan."
                     for _ in range(nsent)]
            import json
            return json.dumps({"sentences": sents})
        import json
        return json.dumps({"sentences": ["x"]})

script_mod.llm.complete = FakeLLM().complete
segments = script_mod.generate_segmented_script(
    chunk_summaries, {"provider": "d", "model": "x"}, 2520,
    words_per_minute=WPM, lang_name="English", sign_off=True,
    visual_match=True, humanize=False)
sentences = [s["sentence"] for s in segments]
print(f"script: {len(sentences)} sentences")

# TTS: 138 wpm real, 0.6s pauses
cues_t, t = [], 0.3
for s in sentences:
    dur = max(len(s.split()) / 138.0 * 60.0, 0.8)
    cues_t.append(TimedCue(s, t, t + dur)); t += dur + 0.6
audio_span = t - 0.6
durations = timeline.lock_durations(cues_t, audio_span)

# ORIGINAL anchors (what the sentence describes) vs REWINDOWED windows
orig = [(s["film_start"], s["film_end"]) for s in segments]
seg2 = pipeline._extend_final_zone([dict(s) for s in segments], 7200.0)
seg2 = timeline.rewindow_to_speech(seg2, durations, 7200.0)

err = []
for i, s in enumerate(seg2):
    o_lo, o_hi = orig[i]
    # the sentence's beat anchor ~ midpoint of its original tight window
    anchor = (o_lo + o_hi) / 2.0
    new_mid = (s["film_start"] + s["film_end"]) / 2.0
    err.append((i, anchor, s["film_start"], s["film_end"], new_mid - anchor))
big = [e for e in err if abs(e[4]) > 10.0]
print(f"sentences whose rewindowed window midpoint is >10s off the anchor beat: {len(big)}/{len(err)}")
print("sample (idx, anchor, new_lo, new_hi, drift):")
for e in err[:10]:
    print(f"  {e[0]:4d} anchor {e[1]:7.1f}  window [{e[2]:7.1f}, {e[3]:7.1f}]  drift {e[4]:+7.1f}s")
import statistics
dr = [abs(e[4]) for e in err if e[0] < len(err) - 1]
print(f"median |drift| = {statistics.median(dr):.1f}s, mean = {statistics.mean(dr):.1f}s, max = {max(dr):.1f}s")

# now: what footage is actually ON SCREEN while each sentence is spoken?
stats = {}
beats = timeline.build_timeline(seg2, durations, 7200.0, CFG_TL,
                                word_times=[c.words for c in cues_t], stats=stats)
cuts = timeline.flatten_cuts(beats)
cum = [0.0]
for d in durations: cum.append(cum[-1] + d)
cur = 0.0
on_screen = []  # (narr_t, footage_start)
for (fs, d, frz, spd) in cuts:
    on_screen.append((cur, fs)); cur += d
mismatch = 0
for i in range(len(sentences)):
    t0 = cum[i]
    fs = next((f for (nt, f) in on_screen if nt >= t0 - 1e-6), None)
    anchor = (orig[i][0] + orig[i][1]) / 2.0
    if fs is not None and abs(fs - anchor) > 30.0:
        mismatch += 1
print(f"sentences where ON-SCREEN footage is >30s off the narrated beat: {mismatch}/{len(sentences)}")
print(f"stats: slowed={stats.get('slowed_groups')} held={stats.get('held_shots')}")
frozen = sum(f for _, _, f, _ in cuts)
print(f"frozen seconds: {frozen:.1f}")
