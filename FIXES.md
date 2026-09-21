# Fixing the two bugs: short videos and mismatched visuals

You reported:

1. **"I asked for 900 seconds and got 360."**
2. **"The visual and the narration do not match at all."**

Both are now fixed. This document explains what was actually wrong (it was four
separate defects, not two), what changed, and how to run it.

---

## Bug 1 — the video was far shorter than requested

### Cause A: the script was never checked against the target

`900 words ÷ 150 words-per-minute = exactly 360 seconds.`

That is not a coincidence — it is the whole bug. The old `generate_script_json`
made **one** LLM call containing the words *"roughly 2250 words"*, and whatever
came back was accepted. Language models are bad at long word targets and
routinely return a third of what you ask for. Nothing ever compared the result
to the request, so a 900-second job quietly became a 360-second one.

**Fix:** the script is now written **section by section** through the film
(`script.generate_segmented_script`). Each of the ~20 chunks gets its own small
word budget, and the budgets sum to the target. Small targets are hit
accurately, so the total lands. Each section is re-requested once if it comes
back under 55% of its budget, and the pipeline now prints the delivered length
and warns loudly if it is under 75% of what you asked for.

### Cause B: the visual track ignored the silence between sentences

This one truncated the render even when the script *was* long enough.

`clip.py` sized the video as `sum(spoken clip lengths)`. But an mp3 of narration
is not just speech — it contains the pauses between sentences. For a typical
150-sentence recap:

| | |
|---|---|
| speech only | ~652 s |
| actual mp3 (speech + gaps) | ~898 s |

The video was built to 652 s, the audio was 898 s, and `-shortest` in
`video.py` threw away the difference. The last quarter of your story simply did
not exist in the output file.

**Fix:** `timeline.lock_durations()` derives beat durations from the **cue
boundaries**, so they tile the entire audio span, gaps and trailing silence
included. `clip.build_locked_visual()` then pads or trims the finished track to
that exact span, and `video.burn_and_mux_locked()` muxes with an explicit `-t`
instead of `-shortest`. The video can no longer be cut short.

### Cause C: the requested length was silently clamped

`runner.py` did `min(secs * 2.5, words_max)` with `words_max = 4200`, and the
CLI clamped the same way. Long requests were reduced without saying so. Both now
raise the ceiling to match the request and echo the real target.

---

## Bug 2 — visuals did not match the narration

### Cause D: the montage was not in chronological order

Beats were chosen by **cosine similarity** between each narration sentence and
the film's dialogue. Movie recaps are strictly chronological, but similarity
search is not: the words "he runs" match act 1, act 2 and act 3 about equally
well, so sentence 4 could land at 1:52:00 and sentence 5 back at 0:03:00. The
picture ping-ponged across the film while the voice told a linear story.

**Fix:** vector matching is gone from the beat path. Every narration sentence
now already knows which stretch of film it was written from (Step B carries
`film_start` / `film_end` per sentence), and `timeline.build_timeline()` walks a
**monotonic playhead** — it is structurally incapable of going backwards.

There was also a latent bug that made this much worse on **resumed** runs: the
Step C resume branch reassigned the variable `cues`, which until then held the
*film transcript*, to the *narration* cues. Step D then matched the narration
against itself, scored ~1.0 on everything, and used narration timestamps as film
timestamps. Renamed to `nar_cues`.

### Cause E: one static clip per sentence

A 6-second sentence got one 6-second locked-off shot, which reads as a slideshow
rather than a recap.

**Fix:** `timeline` splits each beat into **1–3 micro-cuts** (a new shot roughly
every 3 s, never shorter than 1.2 s), and the shots inside a beat advance
forward through the scene. Tunable under `timeline:` in `config.yaml`.

Cuts are also now **re-encoded** rather than stream-copied. `-c copy` snaps every
cut to the nearest keyframe, so a 4.20 s request could return 3.6 s or 5.1 s;
across 150 beats that is tens of seconds of accumulated drift. Frame-exact cuts
cost some CPU and buy you sync.

---

## What changed

| File | Change |
|---|---|
| `recap/timeline.py` | **New.** Chronological, audio-locked beat planner with micro-cuts. |
| `recap/script.py` | `generate_segmented_script()` — per-section budgets, film windows, short-answer retry. |
| `recap/llm.py` | DeepSeek-first, JSON mode, retry with backoff on transient errors, clearer failures. |
| `recap/clip.py` | `build_locked_visual()`, frame-exact `cut_segment(exact=True)`, re-encoded concat. |
| `recap/video.py` | `burn_and_mux_locked()` — explicit duration, no `-shortest`. |
| `recap/pipeline.py` | Step B segmented; Step D/E/F rebuilt on the timeline; `cues` shadowing fixed. |
| `recap/config.py` | DeepSeek default, `timeline:` block, `words_per_minute`, per-provider base URLs. |
| `recap/cli.py` | `--seconds`, no silent clamping. |
| `recap-studio/*` | DeepSeek default, 900 s default, correct word math. |
| `tests/test_timeline_sync.py` | **New.** Six regression tests covering both bugs. |

A note on `config.py`: `OLLAMA_BASE_URL` was previously applied to **every**
provider, so a leftover env var would have pointed DeepSeek at `localhost:11434`.
Base URLs are now per-provider.

---

## Verification

All three suites pass:

```
tests/test_timeline_sync.py       ALL TIMELINE SYNC TESTS PASSED
tests/test_engine_integration.py  ALL ENGINE INTEGRATION TESTS PASSED
tests/test_semantic_engine.py     ALL SEMANTIC ENGINE SMOKE TESTS PASSED
```

Beyond the unit tests, the pipeline was run through **real ffmpeg** on a
synthetic 300-second film with a 35.03-second narration containing only 15
seconds of actual speech — the exact shape that used to trigger the truncation:

```
film = 300.00s   narration = 35.03s   pure speech = 15.00s
timeline: 10 beats / 10 cuts, video 35.0s vs narration 35.0s (drift 0ms),
          chronological=yes, film coverage 0s -> 270s

FINAL MP4 = 35.030s
NARRATION = 35.030s
DELTA     = +0.000s   -> LOCKED
```

Sampling one frame per beat from the rendered file, the film's on-screen counter
reads **1 → 3 → 6 → 9 → 12 → 15 → 18 → 21 → 24 → 27**: strictly increasing and
evenly spread. Under the old code this sequence jumped around. The old behaviour
would have produced a ~15-second file here.

---

## Running it

1. Get a key at [platform.deepseek.com](https://platform.deepseek.com) → API keys.
2. Put it in `movie-recap-bot/.env`:
   ```
   LLM_PROVIDER=deepseek
   MODEL_NAME=deepseek-chat
   DEEPSEEK_API_KEY=sk-...
   ```
3. Ask for a length in seconds:
   ```
   python -m recap.cli auto "D:\movies\Film.mkv" --seconds 900
   ```
   or set **Duration = 900** in Recap Studio.

Watch for these lines — they are the ones that would have caught the original bug:

```
* Target: 2250 words ≈ 900s of speech at 150 wpm
* EN recap: 148 sentences, 2231 words ≈ 892s of speech (target 2250 / 900s)
* [en] timeline: 148 beats / 291 cuts, video 921.4s vs narration 921.4s
       (drift 0ms), chronological=yes, film coverage 61s -> 6803s
+ output\Film_en.mp4  (921.4s vs narration 921.4s)
```

`drift 0ms`, `chronological=yes`, and matching video/narration figures mean both
bugs stayed fixed.

### Tuning

```yaml
timeline:
  micro_cut_seconds: 3.0   # lower = snappier cutting
  max_cuts_per_beat: 3
  min_cut_seconds: 1.2
  pre_roll: 0.4
semantic:
  clip:
    mode: reencode         # "copy" is faster but reintroduces drift
```

### One honest caveat

Visual accuracy now depends on Step A knowing *when* things happen. With a
subtitle file (`.srt` next to the movie) or Whisper timestamps, beats land on the
right scenes. For a film with almost no dialogue there is little to anchor to, so
beats fall back to even spacing across each chunk — still chronological and still
perfectly in sync, but scene-accurate only to within a chunk. Shot-level accuracy
would need vision on the actual frames, which is a larger change.

---
---

# Fixing "the narration and the visuals just don't match at all from the
# start"

You reported:

1. **"the narration and visual just doesn't match at all from the start"**
2. **"few seconds in at a single point it just burst out 100 of sentences on
   a single visual"**

Both trace back to one defect (plus one guard that was too quiet). This
document explains what was actually wrong — proven at runtime, not just read
out of the code — what changed, and how it was verified.

---

## Root cause — the measured-narration fix threw the beats away

The pipeline had already fixed *slow motion* with `rewindow_to_speech()`:
after TTS, each sentence's film window is re-sized to the sentence's
*measured* speech duration, so a slow voice can never make a window smaller
than its sentence. But the first version of that pass re-tiled each section's
zone **proportionally to speech time** — sentence *k* got the *k*-th slice of
the zone, no matter which moment of the film it was written from. **The
per-sentence anchor beat was discarded.**

Beats are not spread evenly through a zone — a zone holds one busy scene and
a quiet stretch. Re-running the real chain on a 2-hour film with clustered
beats (`tests/repro_anchor.py`) measured the damage:

| | old proportional tiling | anchored (now) |
|---|---|---|
| sentences >10 s off their beat | **97 / 146** | **0 / 146** |
| median drift | **20.7 s** | **2.6 s** |
| max drift | 41.1 s | 5.3 s |
| on-screen footage >30 s off the narrated beat | from sentence 2 onward | **0 / 146** |

Every symptom followed from this one:

* **"doesn't match at all from the start"** — the visual track walked each
  150-s zone at a constant ~9x story pace, so from the very first sentence
  the picture was 15-40 s off the moment being described (sentence 2:
  anchor 152.6 s -> footage at 100-125 s; sentence 5: anchor 302.6 s ->
  footage at 274.6 s).
* **micro-cuts jumping 20-50 s of film mid-sentence** — a 5.3-s sentence got
  a ~50-s window; its three shots each consumed ~1.8 s of *screen* time while
  leaping ~25 s of *film* time.
* **the partial-subtitle crawl** — zones stop where the transcript ends, so a
  10-minute `.srt` of a 2-hour film made the whole video crawl the film's
  opening minutes at 0.7x, and the sign-off outro got the *entire* remaining
  tail as its window: 2200-s jump-cuts every ~2.5 s during
  "don't forget to subscribe".
* **"100 sentences on a single visual"** — the only literal
  one-visual-for-many-sentences path in the engine is the freeze that happens
  when the narration outlasts the film (e.g. a short clip recap'd with the
  2520-word Studio default, or a target far beyond the footage). Nothing in
  the code said anything when that happened; it just shipped.

---

## What changed

### 1. Anchor-true re-windowing (`recap/timeline.py` -> `rewindow_to_speech`)

When a section's measured narration fits its film zone (the normal case — the
section budgets cap narration at ~40% of the zone's film time), each
sentence's window is now:

```
window = [anchor - measured_dur/2, anchor + measured_dur/2]   (clamped to the zone)
```

i.e. **centered on the beat the sentence narrates, exactly as long as the
sentence takes to say**, with a forward walk that resolves any overlap so
windows stay monotone and the film never rewinds. A section whose measured
narration genuinely exceeds its zone keeps the old contiguous walk — the
slow-motion net handles it (rare and honest, not the norm).

### 2. Segments now carry their anchor (`recap/script.py`, `recap/pipeline.py`)

`generate_segmented_script` stores `"anchor"` on every sentence (the beat it
was written from — embedding-aligned for English, positional for the other
languages), the sign-off carries the last story sentence's anchor, and
translated languages inherit the master's anchors 1:1. Old cached
`script_*.segments.json` files (no anchor field) still work: the re-windowing
pass falls back to the midpoint of the sentence's original window.

### 3. Chunk territory cannot exceed the film (`recap/pipeline.py`)

A chunk's window is `window_seconds` long; on a **short clip** the last
window claimed footage that does not exist (a 90-s sample with the 300-s
default window claimed `[0, 300]`). The anchors spread across that phantom
zone piled the ending sentences onto the final frame — the picture froze
under the last stretch of narration (measured: 18.5 s frozen on the 90-s
fixture). Chunk ends are now clamped to the movie's length, and the word
target is clamped to what the *footage* (transcript coverage, bounded by the
film's length) can show at 1x — so a short clip gets a short recap instead of
a 14-minute one that has nowhere left to play.

### 4. The freeze is no longer silent (`recap/pipeline.py`)

The run log now prints, per language:

```
! [en] WARNING: the picture is FROZEN for Ns of this video -- the narration
  runs past the end of the film ...
* [en] visual sync: footage within 2.6s (median) / 5.3s (90th pct) of the
  beat each sentence narrates
```

The first line fires when the narration outlives the footage (the "burst on
a single visual" guard); the second is the measured sync number — 20-40 s
before this fix, a couple of seconds after.

| File | Change |
|---|---|
| `recap/timeline.py` | `rewindow_to_speech`: anchor-centered windows sized to measured speech (over-budget zones keep the contiguous walk). |
| `recap/script.py` | segments carry `"anchor"`; sign-off inherits the last story anchor. |
| `recap/pipeline.py` | chunk ends clamped to the movie length; word target clamped to transcript coverage; translated anchors carried 1:1; loud freeze WARNING + measured visual-sync line in the run log. |
| `tests/test_visual_flow.py` | re-windowing contract updated to anchor-true placement; new `test_rewindow_keeps_windows_on_their_anchor` (clustered beats: old tiling drifts 40+ s, anchored windows sit on the beat) and `test_rewindow_legacy_segments_fall_back_to_midpoint`; full-script loop test asserts every segment carries an in-zone, ordered anchor. |
| `tests/repro_anchor.py` | **new** runtime audit: full chunk->zones->script->lock_durations->rewindow->build_timeline chain on a 2-h film with clustered beats; measures per-sentence window drift + on-screen-vs-narrated-beat mismatches. |
| `tests/repro_sync.py` | scenarios + the pipeline's chunk/target clamps; audits zone violations, frozen seconds, footage consumed. |

---

## Verification

All seven suites pass:

```
tests/test_visual_flow.py        ALL VISUAL-FLOW TESTS PASSED
tests/test_narration_sync.py     ALL NARRATION-SYNC TESTS PASSED
tests/test_timeline_sync.py      ALL TIMELINE SYNC TESTS PASSED
tests/test_render_sync.py        ALL RENDER-SYNC TESTS PASSED
tests/test_semantic_engine.py    ALL SEMANTIC ENGINE SMOKE TESTS PASSED
tests/test_languages.py          ALL LANGUAGE TESTS PASSED
tests/test_engine_integration.py ALL ENGINE INTEGRATION TESTS PASSED
```

Runtime audits (real code, fake LLM/TTS):

```
tests/repro_anchor.py   (2 h film, clustered beats)
  windows >10 s off their anchor beat:   0/146   (was 97/146)
  on-screen footage >30 s off the beat:  0/146   (was: from sentence 2)
  median |drift| 2.6 s (was 20.7 s)   max 5.3 s (was 41.1 s)   frozen 0.0 s

tests/repro_sync.py     (4 scenarios)
  full 2 h         0/145 zone violations   0.0 s frozen   100% film consumed
  90 s clip        0/7   zone violations   0.0 s frozen   (was 8.5 s;
                                                         18.5 s mid-fix) 96%
  240 s clip       0/18  zone violations   0.0 s frozen   (was 8.5 s)   98%
  2 h + 10-min srt 0/47  zone violations   0.0 s frozen   zones stay in the
                                                        transcript's coverage
```

The remaining sub-6 s drifts in the anchor audit are the honest case: several
sentences narrating one busy beat share its zone and are walked forward so
the film never rewinds. If a future run ever shows `visual sync: footage
within >10s` in the log, `tests/repro_anchor.py` is the way to reproduce it.
