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

---
---

# Round 3 — full-film vision coverage, no more chopped sections, louder QC

You reported the picture still did not stay with the voice, with four more
concrete complaints. Two of the four were exactly as diagnosed; one was
already handled by the engine (documented below so it can be trusted); the
last was a real gap, now closed.

---

## 1. Vision frames stopped at ~57% of the film — CONFIRMED, FIXED

`vision.pick_times()` kept **every** detected scene change ("they are
story-critical") and then truncated the sorted list to `max_frames`. Scene
changes cluster in the early scenes (fast intros, more cuts), so the frame
budget was eaten before reaching the back half: your 2h film (740 cuts,
400-frame cap) got its **last frame at 3515s** — the final 43% had zero
visual notes, and the timeline had nothing to anchor on there.

**Fix:** stratified sampling. The film is divided into time bins (one frame
per bin, never finer than every 5s); each bin contributes the real shot
change nearest the bin's middle when one exists, else the bin's middle.
Every stretch of the film is captioned no matter how the cuts are
distributed. `vision.max_frames` default raised 400 -> 600 (~1 caption per
10s of a 100-min film, ~150 Gemini calls at 4 frames/request).

Verified (`test_vision_pick_times_covers_full_film`): 6207s film with 740
cuts denser in the first half -> 600 frames, **last at 6202s**, every 5%
time-slice covered, 579/600 on real cuts; a 90s clip -> 18 frames covering
it end to end.

## 2. "Step A invents plot without a subtitle" — ALREADY FALSE, by design

No code change needed: when no `.srt` is provided, Step A **already runs
faster-whisper on the movie's own audio track** (`dialogue.extract_dialogue`
with `srt=None`; the run log says `(Whisper)` vs `(subtitle ...)`), then
chunks + summarizes exactly as with a subtitle. Your run was already on this
path — that is also why removing the `.srt` changed nothing.

The one valid knob in your note: the ASR model. `dialogue.whisper_model`
(default `"small"`) is the accuracy/speed trade-off for Step A — set it to
`"medium"` or `"large-v3"` in `config.yaml` if dialogue accuracy matters
more than extraction time on your machine (`--whisper-model` on the CLI).

## 3. Every section over budget and chopped — REAL, FIXED (mostly)

Your log: 36/36 sections 2-5x over budget. The prompt already states the
budget as a hard ceiling ("This is a HARD CEILING ... Never exceed it") and
the first enforcement pass (condense) handles most of it — but when
condense *also* failed, the only backstop was the mechanical trim, which
drops middle sentences (a section that trims hard has its tail under no
narration). The trim was doing too much work.

**Fix:** between condense and trim there is now one **writer regenerate
with explicit budget feedback** — the writer that saw the section's beats is
asked again: "your previous attempt was N words, this footage only has room
for M — rewrite covering the window from its FIRST beat to its LAST; do not
stop early, do not pad." That in-budget full-window rewrite is what reaches
the timeline; the mechanical trim is the absolute last resort only (and a
section that still cannot fit after all three passes ships with a loud
`!` warning, as before). Verified by
`test_overdelivery_regenerates_before_trimming`: over-delivery + failed
condense -> one regenerate -> in-budget rewrite shipped, trim never fired.

On the budget math: the per-section ceiling is 0.4x the section's film time
by design (the 1x pacing constraint — narration longer than the footage is
exactly the slow-motion disease). 3600 words across a 104-min film is
already ~40% of the film's narration capacity; for a denser recap raise
`narration.words_target` (the budgets scale with it), don't expect more out
of the same footage at 1x.

## 4. Humanizer "0/143 rewritten" — the pass was never detection-gated, but
    the no-op was SILENT — NOW LOUD

For the record: `_humanize_script` already sends ALL sentences to the model
in one call (no phrase-detection gate), and keeps every rewrite that passes
the timing lock (exact count, +10%/+2 words per line). A 0/N result means
the model echoed the draft back (or every rewrite tripped the lock) — and a
failed call returned `None` with **no log line at all**, which reads as
"humanized, nothing to change". That silence is fixed:

* `humanizer pass: 0/N` now raises
  `! WARNING: humanizer changed 0 sentences — the model echoed the script
  back ... the AI feel you are hearing is the WRITER's voice: try a stronger
  LLM model for narration, or disable the pass (RECAP_HUMANIZE=0)`.
* A failed rewrite call now prints
  `! humanizer pass: the rewrite call failed ... the polished script is
  kept AS-IS (not humanized)`.

Verified by `test_humanizer_zero_changes_warns`. The structural fix for the
"AI feel" is upstream of the humanizer anyway: the section writer carries
the in-prompt voice exemplars, the strict narrator persona, the names
enforcement and the global read-through — if the writer's voice is the
problem, a stronger model (deepseek-chat over a small local one) is the
lever.

## 5. Longest shot 10.1s + min_speed 0.35 — TUNED

* **`timeline.max_shot_seconds: 7.0` (new).** Word-locked splits can leave a
  long final shot when clause boundaries are sparse (one 10.8s hold under a
  12s sentence). `_shot_split` now forces an extra midpoint cut whenever a
  segment would exceed the cap (a few extra cuts at most). Verified by
  `test_shot_split_caps_long_shots`: 12s beat -> max shot 5.4s (cap off:
  10.8s); repro runs now report `longest shot 2.6s` in every scenario.
* **`timeline.min_speed: 0.6` (was 0.35).** With anchor-true re-windowing
  and enforced section budgets, genuinely starved sections are rare, so the
  slow-motion floor is now a mild, barely-perceptible 0.6x instead of an
  obvious 0.35x crawl when it does fire.

| File | Change |
|---|---|
| `recap/vision.py` | `pick_times`: stratified bin sampling with full-runtime coverage (replaces keep-all-cuts-then-truncate); `max_frames` default 400 -> 600. |
| `recap/script.py` | over-budget sections: writer regenerate with budget feedback between condense and the mechanical trim (trim = last resort); humanizer 0-kept WARNING and failed-call log line (no more silent no-ops). |
| `recap/timeline.py` | `_shot_split`: `max_shot` cap forces mid-shot re-cuts; `min_speed` default 0.6. |
| `recap/config.py`, `config.yaml` | new defaults: `vision.max_frames: 600`, `timeline.min_speed: 0.6`, `timeline.max_shot_seconds: 7.0`. |
| `tests/test_visual_flow.py` | new: `test_vision_pick_times_covers_full_film`, `test_shot_split_caps_long_shots`, `test_overdelivery_regenerates_before_trimming`, `test_humanizer_zero_changes_warns`. |

---

## Verification

All seven suites pass, plus the runtime audits:

```
tests/repro_anchor.py   (2 h film, clustered beats)
  windows >10 s off their anchor beat:   0/146   median |drift| 2.6 s
  on-screen footage >30 s off the beat:  0/146   frozen 0.0 s

tests/repro_sync.py     (4 scenarios, all drift 0ms, chronological=yes)
  full 2 h         0/145 zone violations   0.0 s frozen   100% consumed
  90 s clip        0/7   zone violations   0.0 s frozen   96% consumed
  240 s clip       0/18  zone violations   0.0 s frozen   98% consumed
  2 h + 10-min srt 0/47  zone violations   0.0 s frozen
  longest shot 2.6s in every scenario (was 10.1s+ in your run)
```

**On your manual spot-check:** with the vision fix, `visual_notes.json`
now reaches the film's end (confirm: the last `t` in the file is within a
bin of the movie's duration). For the 5 random sentences from the back
third: their footage is anchored to the Whisper transcript beats (Step A
always transcribes the full film's audio when no `.srt` is given), with the
vision notes added on top — if any spot-check still shows >10s of
off-beat footage, the new run-log line
`visual sync: footage within Xs (median) / Ys (90th pct) of the beat each
sentence narrates` will show it, and `tests/repro_anchor.py` reproduces it.

---

## Round 4 (2026-09-23): surviving a Gemini "503 high demand" storm

**Symptom (your Toy Story 5 log, after the round-3 fix):** frame extraction now ran to the
real end (last frame 6202 s — the round-3 sampling fix is working), but EVERY vision batch
returned `Error code: 503 — the model is under high demand` on `gemini-3.1-flash-lite`.
The old retry policy (5/10/20/40 s, ~75 s total per batch) gave each batch up, logged one
line, and moved on — the run finished with zero on-screen notes and no loud signal, so you
had to stop it by hand.

**Root cause:** free-tier demand throttling is a minutes-long condition, not a sub-minute
one. A retry ladder that gives up in ~75 s loses every batch inside a demand window; the
per-batch cache then made a "successful" run that was actually a silent text-only pass.

**Fixes:**
1. **Patient per-batch retries** (`recap/vision.py`): waits are now 15s → 30s → 60s → 120s
   (~4 min of patience per batch instead of 75 s), and "high demand" is an explicit
   retryable keyword. Progress is logged (`waiting 120s (try 4/4)`) so the Studio log shows
   the bot waiting, not hanging.
2. **Final sweep** (`recap/vision.py`): after the main pass, any frames still failed are
   re-captured in one more pass after a configurable pause (`vision.sweep_pause_seconds`,
   default 60 — the storm usually cools by then). The sweep reuses the same cache.
3. **Loud gap accounting** (`recap/vision.py`): the pass now ends with either
   `Vision pass: N on-screen notes` or a `!` line stating exactly how many of how many
   frames could NOT be captioned, with the exact cheap recovery: *re-run the SAME movie —
   cached frames are reused and only the missing ones are re-captured* — plus the two
   escape hatches (switch to the default `gemini-3.6-flash` model, which has more
   free-tier headroom than the `-lite` tier, or `vision.enabled: false` for a deliberate
   text-only run). No more silent text-only passes.
4. **`vision.sweep_pause_seconds`** (config.yaml + defaults): the pre-sweep pause.

**Why re-running is cheap:** every successful batch already persisted its frames to
`workdir/visual_notes.json` during the run, and re-capture skips frames with a cached note
— so after a storm, one re-run of the same movie re-captures only the frames the storm
actually killed.

**Verification:** new regression `test_vision_capture_survives_a_503_storm`:
(a) first 3 batch calls 503 → the sweep recovers all 12/12 frames, log shows "final
sweep", `visual_notes.json` written; (b) storm never ends → notes empty, log contains the
loud "could NOT be captioned" + "Re-run the SAME movie" guidance. All 7 suites green.

---

# Round 5 (2026-09-24): `HUMANIZE_FAILED` killed a finished run at the last step

You reported, after ~1h15m on a 6206.8s film (vision 599/600 frames captioned,
36 chunks summarized, 3584 words written):

```
  ! HUMANIZE_FAILED: 69/159 sentences (43%) failed to humanize
ERROR: RuntimeError: HUMANIZE_FAILED: 69/159 sentences failed to humanize (43% > 10% threshold).
```

and asked for a real fix, not `--allow-unhumanized`.

## The log already contained the whole diagnosis

Your run printed, per window:

* `rewrite identical (sim>0.9)` for nearly **every** sentence of Pass A,
* **51** × `humanizer Pass B: sentence tightened length would exceed lock — keeping original`,
* **18** × `retry still failed for sentence N`,

and `51 + 18 = 69` — exactly the reported failure count. The movie was never
the problem; three code defects were.

### Defect 1 — Pass B was dead code (51 of the 69 failures)

`HUMANIZER_PROMPT_PASS_B` existed in `recap/script.py` and was **never sent to
the model**. Meanwhile Pass A's prompt says, in capitals, *"NO length
constraint in this pass — Length will be handled in Pass B's timing lock."*
So the model was invited to write longer, and the code then silently **threw
every longer rewrite away** (`count_words(rew) <= words(orig)*1.1 + 2` or
discard) and reverted to the original — which the scorer immediately counted
as a humanizer failure. Good rewrites were being produced, paid for, and
deleted.

**Fix:** Pass B is now really a pass.

1. The over-long rewrites are collected and sent **in one batched call** with
   the original beside each one ("tighten this to at most 110% of the
   original's word count, keep the human voice, never restore the tells").
2. Whatever the model still leaves too long goes through a **deterministic
   local shrinker** (`_tighten_sentence`): contractions → wordy connectives
   (`in order to` → `to`, `is able to` → `can`) → filler adverbs, applied in
   escalating order and stopped the moment the line fits, so a sentence that
   needed one contraction keeps all its other words.
3. Only a line that *still* does not fit falls back to the original, and a
   "tightened" line that smuggles an AI tell back in is rejected
   (`_tells_regressed`) — Pass B can never undo Pass A's work.

The +10% +2-word film window itself is unchanged: sync is still absolute.

### Defect 2 — "unchanged" was scored as "failed", even with nothing to fix

24 of your 36 sections ended in a mechanical trim, which leaves short, plain,
already-human prose. The humanizer asked the model to rewrite those lines
anyway, the model correctly returned them unchanged, and the scorer counted
every one as a failure. On a clean script that is a 100% failure rate.

**Fix:** a free, offline **tell scan** (`_ai_tells`, the blader pattern pack as
regexes) now runs first. It decides which lines actually carry AI writing, and:

* those line numbers and their tell names are listed in the prompt
  (`=== TELL SCAN (automatic) ===`), so the model knows what to change instead
  of echoing the draft back;
* **`failed_rate` is measured over the flagged lines only** — a clean line
  returned untouched is the right answer, not a failure;
* a line that was rewritten and lost at least one of its tells is a pass; one
  that improved but kept a minor tell is reported as `residual_tells`, not a
  failure.

### Defect 3 — the retry storm (18 of the 69) and its cost

The echo retry fired **one serial API call per sentence** with a single
strategy (same prompt, temp 0.9) and no escalation — ~180 extra round trips on
your run, each one able to stall like your chunks 15 and 18 did.

**Fix:** retries are now staged — **one batched call** for *all* echoed lines
("you already saw these lines and echoed them back; the tell is still there"),
then at most `narration.humanize_max_single_retries` (default **8**) per-line
escalations at a higher temperature. A stubborn provider now costs ~6 calls
per window instead of 30+.

### Defect 4 — the prompts contradicted each other

`SYSTEM_HUMANIZER` said *"never let a line grow"* while Pass A said *"NO length
constraint"*. At `LLM_TEMPERATURE=0.7`, echoing the input is the safest way to
satisfy both. The system prompt now owns only the sentence-count lock and
defers length to Pass B.

### Defect 5 — a finished run could be thrown away

The gate raised **after** the script was written and **before**
`pipeline.py` persisted it, so an hour of transcription, vision and writing
died with the exception.

**Fix:** `generate_segmented_script(..., report_dir=...)` (wired to
`_work/script/`) writes, *before* raising:

* `humanizer_report_<lang>.json` — every sentence with its film window,
  before/after text, tells before/after, and which lines failed;
* `script_<lang>.unhumanized.txt` — the plain script, one sentence per line.

The error message points at them. Re-running with `--allow-unhumanized` ships
that text as written.

### Defect 6 — a percentage with no floor

On a nearly clean script "1 of 4 flagged lines" is 25% and would still burn the
run. The gate now needs a real number behind the rate:
`narration.humanize_min_failures` (default **3**).

### Mechanically trimmed sections are now actually used

`c["_mechanically_trimmed"]` was set and only ever printed. Those sentences are
now passed to the humanizer as `mandatory` work: they are named in the prompt
and get the retry budget — but they are **not** part of the failure rate, since
a trimmed line is often already plain and human.

## New knobs

| Key | Default | What it does |
|---|---|---|
| `narration.humanize_threshold` | `0.1` | share of *flagged* lines that may stay unfixed (CLI `--humanize-threshold`, env `RECAP_HUMANIZE_THRESHOLD`, accepts `0.25` or `25`) |
| `narration.humanize_min_failures` | `3` | minimum failed lines before a rate can fail the build |
| `narration.humanize_max_single_retries` | `8` | per-sentence retry cap per 30-line window |

## Verification

New suite `movie-recap-bot/tests/test_humanizer.py` (13 tests), all green with
the existing 79:

* `test_clean_script_echoed_back_is_not_a_failure` — 160 clean sentences, a
  provider that echoes everything: **0 failures and 6 LLM calls** (the old code
  made **165** and failed 100%).
* `test_the_reported_159_sentence_run_no_longer_fails_the_build` — replay of
  your run (Pass A echoes every window, repairs come back ~30% too long):
  **0%** failure rate, was 43% + `RuntimeError`.
* `test_pass_b_tightens_a_long_rewrite_instead_of_dropping_it` — asserts the
  Pass B prompt is actually sent and the rewrite survives.
* `test_pass_b_local_shrinker_rescues_when_the_model_will_not`,
  `test_a_rewrite_that_cannot_fit_keeps_the_original_and_stays_in_sync` — the
  local shrinker works and the timing lock is still absolute.
* `test_echoed_tell_lines_are_retried_in_one_batched_call`,
  `test_single_sentence_retries_are_capped` — 10 echoed lines cost 1 retry
  call; a stubborn provider costs 6 calls, not 20+.
* `test_gate_still_fails_loudly_and_writes_a_report_first` — a script that is
  genuinely all tells **still fails**, and the report exists on disk first.
* `test_a_couple_of_stubborn_lines_do_not_burn_a_finished_run`,
  `test_mechanically_trimmed_sentences_are_mandatory_even_without_tells`,
  `test_tell_scan_finds_ai_writing_and_leaves_clean_lines_alone`,
  `test_enumeration_prefix_echo_is_not_mistaken_for_a_rewrite`.

Also fixed while in here: `tests/test_languages.py` replaced `recap.llm.complete`
globally without restoring it, which made `test_visual_flow.py::test_network_
failure_never_stalls_for_hours` fail whenever the whole suite ran (it passed in
isolation). **92 tests, all passing.**
