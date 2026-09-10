# Movie Recap Bot (EN first — semantic Step A-F engine)

A pipeline that turns **a movie you own** into a full-length recap video in the
style of the *Movie Recaps* YouTube channel (~10–16 minutes of continuously
narrated, present-tense storytelling over the film's own footage) with
burned-in subtitles and a full voiceover.

It reproduces the format of the reference channel videos:
* fast, **present-tense**, beat-by-beat narration over a background montage
* a single voiceover as the "dub" on top of the visual
* **one sentence per subtitle cue**, timed to the narration
* `title + thumbnail + description` recipe for each upload

> **Status:** Languages: **en · zh (简体中文) · ar (العربية) · es (Español)**.
> English is always written from the movie's own dialogue. Chinese, Arabic and
> Spanish are written **natively in that language** when you provide a subtitle
> for it (`movie.ar.srt` next to the film, a `language.sources` path, or
> `--subtitle-ar`); without one they fall back to line-aligned translation of
> the English recap. Pick the clips you want per run with `--langs` (CLI) or
> the language bar in Recap Studio.

---

## 🎞️ The Step A-F engine (recommended)

The `auto` command implements the production workflow in six steps. The
transcript is chunked so a 2-hour film never overflows the LLM context window,
each section's narration is written with a per-section word budget, and every
narration beat is locked to its **chronological moment in the film** (no
embeddings, no vector store). The video length always equals the narration
length — gaps between sentences included — so renders are never cut short.

```
movie.mp4
 ├─ A. ffmpeg audio -> faster-whisper/.srt -> timestamped transcript
 │      -> 5-min chunks w/ 30s overlap  -> per-chunk "action" summaries (LLM)
 ├─ B. summaries -> DeepSeek, section by section -> narration sentences, each
 │      tagged with the film window it describes + sized to hit the word target
 ├─ C. sentences -> TTS (edge) -> en.mp3 + sentence (+word) timestamps
 ├─ D. chronological beat timeline: a monotonic playhead through the film,
 │      every beat's visual duration locked to its narration cue
 ├─ E. ffmpeg frame-exact cuts (re-encode) -> per-beat micro-shots
 └─ F. concat -> burn .ass subtitles -> mux narration at an explicit duration
        -> <name>_<lang>.mp4
```

> **Vision pass (optional — the "see the movie" tier).** Step A above reads the
> *dialogue*, so scenes that tell their story visually (montages, chases,
> sight gags) are invisible to the narrator. When a vision-provider key is
> configured (default **Google Gemini, free tier**), the pipeline captions
> frames of the actual film — real shot changes + every ~20 s — and merges
> those on-screen notes into each chunk's beat list, so silent scenes get
> narrated too. Setup: get a free key at aistudio.google.com → Get API key,
> put `GEMINI_API_KEY=...` in `movie-recap-bot/.env` (separate from your
> DeepSeek key; read automatically). No key = the pass is skipped with a
> warning and the pipeline runs text-only. Tunables live in `config.yaml` →
> `vision:` (`cadence_seconds`, `max_frames`, `width`, `frames_per_request`);
> override with `VISION_ENABLED=0` / `VISION_MODEL=...`.
>
> **Vision cost:** it never bills DeepSeek. Image tokens are consumed on the
> vision provider — free on the Gemini free tier (rate-limited); a paid key
> bills ~cents per movie at 512px. The captions add a few thousand text tokens
> to the DeepSeek summary prompts per movie. Frames ≈ movie length / 20 s
> (≈300 for a 100-min film ≈ 75 API calls at 4 frames/request), cached per
> movie so re-runs reuse them.

### Run it locally (no Docker needed)

```bash
cd movie-recap-bot
pip install -r requirements.txt            # + faster-whisper (auto-recap needs it)
# put DEEPSEEK_API_KEY=sk-... in .env (platform.deepseek.com -> API keys)

python -m recap.cli auto --movie "C:\Movies\my_movie.mp4" --minutes 14 --name my-recap
```

**Windows one-click alternative:** open Recap Studio instead — double-click
`..\setup_ui.bat` (it installs missing deps, fetches ffmpeg, and opens the
panel at http://localhost:8080), paste the DeepSeek key under **Settings →
LLM**, set the movie path, and press **Generate**. Everything stays on your
machine; no Docker, no containers.

> Slow PC tip (e.g. older laptops/desktops): the only heavy local step is
> Whisper transcription. Drop a `.srt` next to the movie (or use
> `--subtitle movie.srt`) and Whisper is skipped entirely — the rest of the
> pipeline only needs internet for DeepSeek + the free edge-tts voice servers.

Intermediates land in `output/<name>/_work/`: `transcript.json`/`.srt`,
`chunks/`, `script/summaries.txt`, `script/script_en.json` (the sentence
array), `script/script_en.txt`, `en.mp3`, `en.timing.json` (word-level when
edge-tts provides it), `beats_en.json` (the chronological beat→film-window
map) and `visual/<lang>/beats/seg_NNNN.mp4` (the raw film cuts).

Useful flags: `--minutes` / `--seconds` (target length), `--langs en,zh`,
`--subtitle movie.srt` (skip Whisper), `--whisper-model`,
`--whisper-device auto|cpu|cuda`.

> The extraction step is cached per movie file (`transcript.json` + a file
> fingerprint) so EN/ZH runs don't transcribe twice. Delete it to force a
> re-extract. LLM steps resume too: change nothing and a re-run reuses the
> summaries / script / narration / final render, so nothing is billed twice.

---

## How the flow works (classic 5-step engine)

The legacy `run` command (select "legacy recap" in Recap Studio, or use the
`run` CLI) works like this:

```
  ┌────────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  1. Script     │   │ 2. Translate │   │  3. Narrate  │   │ 4. Subtitles │   │ 5. Assemble  │
  │  (EN recap)    │──▶│ (简体中文)   │──▶│ (TTS + clock)│──▶│ (SRT + ASS)  │──▶│ (ffmpeg mp4)  │
  └────────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
        │                     │                  │                 │                   │
   LLM or your file      LLM or your file     edge-tts free    burned-in on frames  1080p + AAC
   (script_en.txt)       (script_zh.txt)        (EN/ZH)        (CJK-capable font)   per language
```

The **audio (narration) is the master clock.** The montage is made at least as
long as the narration; subtitles are placed exactly at each spoken sentence's
time; the video is trimmed to the narration so nothing drifts.

---

## Quick start

### 1. Install

```bash
cd movie-recap-bot
pip install -r requirements.txt
```

> ffmpeg/ffprobe are pulled in automatically via `static-ffmpeg`. If you
> already have a system `ffmpeg`, that's used instead.

### 2. Know what you need

* **A movie you own/are licensed to use** — supply clip(s). See “Supply your
  own footage” below.
* The **plot** — either your own `plot_notes.txt` (with an LLM) **or** you
  hand-write the recap + translation (no LLM/API needed — that's the default
  path used in this repo).

### 3. Run the full pipeline using the bundled sample scripts

Two sample recap scripts for the reference film are in `inputs/text/`:

```bash
# EN recap, 44 lines (one sentence per line)
inputs/text/script_en.txt
# Line-aligned Simplified Chinese translation
inputs/text/script_zh.txt
```

Run without any real footage (uses placeholder colored scenes) to see the whole
flow:

```bash
python -m recap.cli run \
  --storyboard \
  --langs en,zh \
  --script-file inputs/text/script_en.txt \
  --zh-file inputs/text/script_zh.txt \
  --name pinocchio-recap
```

Output (in `output/`):

| File | Language | Subtitle |
|------|----------|----------|
| `pinocchio-recap_en.mp4` | English | burned-in EN |
| `pinocchio-recap_zh.mp4` | Simplified Chinese | burned-in 简体中文 |

The pipeline also writes sidecar `.srt`, `.ass`, `.timing.json` and per-sentence
audio under `output/_work/` so you can re-edit easily.

---

## 🎞️ Fully automatic: recap a movie from its own dialogue

Drop a movie in and let the bot write the narration itself — no manual script.

```bash
python -m recap.cli auto --movie /path/to/movie.mp4
```

What it does:
1. **Extract dialogue** — using an existing `.srt` next to the movie (best), or Whisper
   ASR on the audio if none exists. (`--whisper-model` / `--subtitle` to override.)
2. **LLM writes the recap** — DeepSeek (`deepseek-chat`) reads each chunk of the
   timestamped transcript and writes the Movie-Recaps-style English narration
   (present tense, beat by beat, section by section with a word budget).
3. **Translate to Simplified Chinese** automatically (when `--langs en,zh`).
4. **Narrate + burn subtitles + assemble** both clips (EN + 简体中文).

Outputs: `output/<name>_<lang>.mp4` for every language in `--langs`
(`_en.mp4`, `_zh.mp4`, `_ar.mp4`, `_es.mp4`).

### Making Arabic / Spanish clips (native, from your subtitles)

Provide a subtitle in the language of the clip you want — the recap is then
written in that language **from that subtitle** (no Whisper needed for it):

```bash
# next to the movie file:  "Toy Story 5.ar.srt"  /  "Toy Story 5.es.srt"
python -m recap.cli auto --movie "D:\Movies\Toy Story 5.mp4" \
    --langs en,ar,es \
    --subtitle-ar "D:\Movies\Toy Story 5.ar.srt" \
    --subtitle-es "D:\Movies\Toy Story 5.es.srt"
# -> output/<name>_en.mp4, _ar.mp4 (Arabic narration from the Arabic subs),
#    and _es.mp4 (Spanish narration from the Spanish subs)
```

Rules of thumb:
* **Name it `<movie>.<code>.srt`** (e.g. `Toy Story 5.ar.srt`) and no `--subtitle-*`
  flag is needed — the pipeline finds it next to the film. The untagged
  `Toy Story 5.srt` stays the English source.
* Default narrators: **ar = `ar-SA-HamedNeural`** (Modern Standard Arabic),
  **es = `es-MX-JorgeNeural`** (Latin American Spanish) — override under
  `narration.lang_voice`.

### Sounding closer to a top recap channel (natural narration + matching cuts)

Three layers now work together to close the gap to channels like *Fantastic
Recaps* — and each layer is independently tunable so you can hear/see what
moves the needle.

**1. Narration that reads as speech, not generated text.** The English section
writer now gets an in-prompt *voice exemplar* (an original passage written in
the target rhythm: varied sentence openers, cause → effect chaining, short
breath-long beats) and is told to match its energy, never its words. On top of
that, an automatic **punch-up pass** (`RECAP_POLISH`, default on for English)
sends each finished section back to the LLM once with an editorial brief —
"kill robotic patterns, never start three lines the same way, replace generic
verbs with concrete ones" — and only keeps the rewrite if it preserves the
exact sentence count, so no film anchor ever shifts. Each English section
therefore costs one extra DeepSeek call (~+$0.01 per movie), cached afterwards.
The recap is plain text before it is voiced, so you can still hand-edit any
line: `_work/script/script_en.txt` (and `_zh/_ar/_es`) holds one sentence per
line — edit, re-run, and only voice + render repeat.

**1b. Humanizer pass — the final de-AI-ing sweep (`narration.humanize`,
default on; `RECAP_HUMANIZE=0` disables).** After the whole script is written,
polished and fitted to its footage, one last pass rewrites the lines that
still *sound* machine-made. It is adapted from
[blader/humanizer](https://github.com/blader/humanizer) (MIT) — the pattern
pack behind the popular Claude skill — condensed to what applies to spoken
recap narration, strongest first: "not just X, it's Y" contrasts, one-line
closers that only restate the previous sentence, sayings that sound deep,
staged run-ups ("Here's what you need to know"), forced triads, repeated
openers, inflated significance ("a moment that changes everything"),
interpretive -ing riders (", symbolizing his freedom"), sales language, and
the stock AI vocabulary (delve, showcase, testament, pivotal, tapestry,
vibrant, underscore, …). The recap register is protected with explicit
guards: short dramatic beats that *add* a fact ("Woody disagrees.") and
time/scene connectors ("Meanwhile,", "That night,") are the narrator's style
and stay; character names always survive. And the timing locks hold: the
pass must return the exact same sentence count (every sentence owns a film
window) and every accepted line stays within +10% +2 words of its original,
so a rewrite can never outrun its footage. Works for every narration
language (structural patterns are universal; the word list is
English-only). One extra LLM call per movie, cached with the script.

**2. Footage that shows the moment each line talks about.** Every narration
sentence is anchored to a beat with its own film timecode. For English the
script now *aligns* each sentence to the beat it actually narrates — local
embeddings score every sentence against every beat, and a monotone (never
backwards) path picks each line's own moment — instead of assuming the writer
covered the film evenly. The camera then cuts inside a tight window around
that beat (`RECAP_ANCHOR_LEAD=0.8` s of lead-in so the shot is already on the
action when the line lands, `RECAP_ANCHOR_TAIL=6.0` s of follow-through), and
micro-cuts walk forward within it. Chronology is guaranteed: beat N always
shows footage at or after beat N−1.

**3. A voice that doesn't sound like a robot.** Swap `narration.lang_voice.<code>`
(or the Studio per-language voice field) and re-run — only TTS + render repeat
because upstream steps are cached, so auditioning costs minutes, not a full run:

| Voice (en) | Feel |
|---|---|
| `en-US-ChristopherNeural` (default) | deep, warm, storytelling |
| `en-US-AndrewNeural` | natural, younger male |
| `en-US-GuyNeural` | energetic, announcer-style |
| `en-GB-RyanNeural` | deeper, British gravitas |
| `en-US-EricNeural` | friendly, lighter |

`rate: "-8%"` (slower = calmer) and `pitch: "-8Hz"` (deeper) tune the same
line in `config.yaml` (`narration:`). Free edge voices are pleasant but still
synthetic — recap channels you admire usually use a paid neural narrator. The
single biggest voice upgrade is `tts_provider: elevenlabs` with an
`ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID` in `.env` (pick a deep male voice
in the ElevenLabs voice library; check elevenlabs.io for current pricing).
ElevenLabs free tier is enough to A/B a few voices on one recap before you
decide.

**What still separates you from the reference channel** (honest list): those
channels use a professional narrator, a subtle music bed under the voice, and
often a human who wrote or heavily edited the script. Two of those three you
can now reach cheaply: hand-edit `script_en.txt` once for a hero video, and
drop a music track into `config.yaml` → `video.bgm` (mp3 path + `bgm_volume`,
e.g. 0.10–0.15) for the under-bed. The last one — a truly human-grade voice —
is ElevenLabs or a local narrator; no free TTS matches a paid neural voice.

**English naturalness knobs (in `.env`)**

| Variable | Default | Meaning |
|---|---|---|
| `RECAP_POLISH` | `1` | spoken-style punch-up rewrite of each English section |
| `RECAP_ALIGN` | `1` | align each English line to the beat it narrates |
| `RECAP_ANCHOR_LEAD` | `0.8` | seconds of footage before a line's beat (raise to 1.5–2 if cuts feel too abrupt) |
| `RECAP_ANCHOR_TAIL` | `6.0` | max follow-through footage per line |
| `RECAP_WHISPER_ALIGN` | `1` | whisper-measure the narration audio; see below |
| `RECAP_SIGN_OFF` | `1` | end with the channel outro line |

### 🎙️ The reference-channel register (script style)

The narration prompts were rewritten against the *actual* transcript style of
the reference video (Fantastic Recaps): **flowing sentences that chain several
moments (12–40 words), punctuated by short dramatic beats of 2–8 words**
("Woody disagrees."), the narrator's real connectors ("Meanwhile,", "Just
then,", "Thanks to that,", "As it turns out,", "Not long after,", "A little
later,"), appositive introductions for new characters/objects ("a mare named
Almond", "a tablet called Lilypad"), conversations reported as indirect
speech ("She explains that..."), and contractions. Two original voice
exemplars in that exact rhythm sit inside the writer and polish prompts so the
model copies the *energy*, never the words. Explicitly banned (the tells of
machine narration): uniform sentence length, "Name does X. Name does Y."
listing, em-dashes, semicolons, "little did they know", rhetorical questions,
and meta commentary.

Every LLM role also speaks as a **strict persona** instead of a task
description (a persona's taste filters every drafting choice; a task
description falls back to the model's default encyclopedic voice):

* all narration writing (section writer, one-shot writer, punch-up, legacy
  modes) speaks as **the same veteran recap narrator** — a storyteller
  relaying the film to a friend who missed it, with instincts (rhythm that
  breathes, varied openers, indirect speech, contractions) and a hard "code"
  (never invent events, never analyze/review, never say "the movie", no
  em-dashes/semicolons, never mention being an AI);
* the beat-extraction step speaks as a **script supervisor** whose log is the
  single source of truth — complete, timecoded, neutral, nothing merged or
  invented — which is what keeps the factual record underneath the narration
  trustworthy.

**Names enforcement (viewers cannot follow "he/she/the man").** The names are
extracted from each section's beat lines and injected into the writer prompt
as an explicit must-use list; if the draft drops most of them, one retry is
issued with the missing names spelled out. The polish passes get the same
list, and the beat extractor itself is instructed to always log who acts by
name. (Chunk signatures change with this update, so summaries + scripts
regenerate on the next run — the new name behavior applies from scratch.)

**Opening accuracy.** The first chunk now always starts at 0:00 (previously it
started at the first *spoken* cue, so a dialogue-free cold open was narrated
but never shown), the recap's first sentence is pinned to the film's first
beat, and its footage window reaches back to the film's very first frames —
"It all begins..." now plays over the film's actual opening.

**Global read-through** (`RECAP_GLOBAL_POLISH=1`, default on): after all
sections are written, one final narrator pass reads the WHOLE script and
fixes what only a full read catches — a sentence repeating the previous
sentence's opener at a section seam, the same beat told twice, a name spelled
two ways, a run of flat same-length sentences. Sentence count is locked
exactly (every sentence keeps its film window), any deviating rewrite is
discarded. Costs about as much as two extra sections.

Structural touches that match how those videos open and close:

* the first section starts *inside the film's first scene* ("It all begins
  ..."), never with a title or a channel hook;
* the final section lands the ending formula ("... and that's how the movie
  comes to an end."), narrating a post-credits scene just before it;
* a deterministic **channel outro** ("If you enjoyed the video, don't forget
  to leave a like, subscribe, and turn on notifications. That's all for
  today. See you next time.") is appended in every language — never left to
  the model, never duplicated. Turn it off with `narration.sign_off: false`
  or `RECAP_SIGN_OFF=0`.

### 🔄 Narration ↔ visual sync (whisper alignment + word-locked cuts)

Two mechanisms keep the picture glued to the voice, exactly like the reference
edits:

**1. `narration.whisper_align` (default on).** After TTS, the *generated*
narration audio is transcribed once more with faster-whisper and every
sentence cue (and every word) is re-anchored to what is actually spoken.
edge-tts already reports word boundaries; this makes cue times exact for **any**
provider — OpenAI and ElevenLabs return a bare mp3, and the old fallback
*guessed* their cue times proportionally (seconds of drift). The pass is
cached per audio content hash (`_work/<lang>.whisper.json`), so re-renders are
free. Set `narration.whisper_align_model` (e.g. `"base"`) for a faster pass,
or disable with `RECAP_WHISPER_ALIGN=0`.

**2. `timeline.cut_on_words` (default on).** When word timings exist, the
micro-shots inside one sentence are no longer evenly spaced: cut points are
placed **on measured word boundaries** — after commas, before "and / but /
while / meanwhile" — so the picture switches at the exact moment the narrator
moves to the next clause, the way a human editor cuts to the beat of the
voice. Shot lengths still respect `min_cut_seconds`, durations still sum to
the audio span exactly, and without word timings it silently falls back to the
even split.

**3. No-replay playback + pacing (always on).** Every cut of the whole track
is placed by one forward walk: a cut's film position is clamped to start at or
after the **end of the previous cut's consumed footage**. The film therefore
never rewinds and no moment is ever shown twice — the "same clip stutters back
mid-sentence" artifact is impossible by construction.

**Visual match — the script is sized to the footage (`narration.visual_match`,
default on).** The recap's *script* is now written to the movie, not the other
way round. The pipeline measures how much distinct film time each section
covers (from the beat map) and gives the LLM a per-section word budget of what
that footage can show at 1x speed — so a section can never be given more
narration than its own film time. On top of that, each sentence's anchor is
*paced*: consecutive sentences are pushed apart until every sentence's footage
window (the film between its moment and the next sentence's moment) is at
least as long as the sentence takes to say. The result: the recap plays at
normal speed from the first cut to the last — the narration and the picture
advance in lockstep, and neither slow motion nor a held frame is ever *needed*
to keep them together. And because estimates can be wrong, the lock is
**measured, not guessed**: after the voice is synthesized, every sentence's
window is re-sized from its *measured* duration (`rewindow_to_speech`), so a
slower voice, a longer pause or a different language can never make a window
smaller than its sentence — the picture walks each section's film in step
with the real narration (a slice per sentence, the reference-channel edit
shape), and the run log prints the voice's true words-per-minute against the
configured one. And the budget is **enforced against what the section was
actually allotted, not the raw footage ceiling**: if the writer over-delivers
(LLMs routinely return 1.5–2× their word budget, and the polish pass can add
~30% more), the section is first sent back for one **condense pass** —
rewrite the same story beats, same order, tighter wording, the way an editor
shortens a paragraph (no jumps in the causal chain) — and only if that fails
is it mechanically trimmed to fit (least-essential middle sentences;
continuity ends and name-bearing lines kept). Every adjustment is
logged (`writer returned 260 words for a 88-word section budget — condensed
to 84 words (story kept)`), and a delivered script more than 35% over the
requested length raises a loud WARNING — that used to be silent, and a 2×
script is exactly what forced near-permanent slow motion. This is what keeps the narration from running
ahead of
the picture on dialogue-dense sections. (For a typical movie this is far from
binding — a 17-minute recap of a 2-hour film uses ~15% of the film time — it
matters for dense films or long targets, where the budget now trims the
script to what the footage can carry at 1x.) Set `narration.visual_match:
false` to go back to the old beat-count budgeting.

**Motion guarantee — the picture never stops.** When a section's narration is
longer than the film behind it (a dialogue-dense stretch), the timeline paces
that window — `speed = window / narration`, clamped to `timeline.min_speed`
(0.35x) — so the footage plays as gentle slow motion that stays locked to the
moment being narrated. Exactly what a human editor does. Mid-film the picture
is therefore ALWAYS moving: new footage at 1x, or the same moment in slow
motion. A frozen frame now occurs only if the narration outlasts the entire
movie. In the stress fixture (25 sentences over 2.4s-apart beats) this took
the visual lead from ~86s of look-ahead down to under a second — with zero
frozen frames and the durations still summing to the narration exactly.
With `visual_match` on this is a *safety net* — the script budgets already
keep almost every section at 1x — but it stays on so no input can ever
produce a still frame.

**Smooth, non-laggy transitions.** Three rules give the reference-channel cut
feel: (1) every cut opens on the film's REAL shot change when scenedetect is
installed (`pip install scenedetect[opencv]`); (2) a cut must show at least
`timeline.min_new_footage` (0.8s) of genuinely new film — smaller advances
continue the current footage seamlessly instead of a stuttering micro-jump;
(3) every cut is re-encoded to its exact frame duration, so there is no
keyframe snapping or timing jitter. Like the reference channel, transitions
are hard cuts — clean and instant, never crossfades.

**Frame-exact render lock — the picture cannot slide off the voice.** The
schedule says beat *i* occupies exactly `[cue_i.start, cue_i+1.start)` of the
output; the render must obey it to the frame. But ffmpeg quantizes every
written clip to whole frames (output `-t` + fixed fps), and that rounding is
one-sided — each clip comes out a frame or part-frame **long, never short**.
Over ~1000 clips that used to accumulate **10–20 seconds of stretch**: the
audio kept its own clock, the picture schedule slid later and later, and by
the second half of the video the narrator was describing the next scene
while an older one was still on screen — even in slow-motion sections,
because the drift is in the render, not the pacing. (The final trim to the
audio span hid the *total*, which is why every end-to-end duration check
kept passing while the interior was out of sync.) The fix is
**drift-compensated cutting**: after each clip is rendered, ffprobe measures
its real duration and the accumulated error is subtracted from the *next*
clip's requested duration — the cumulative schedule never leaves a ~1-frame
corridor for the whole video (the run log prints
`drift-compensated cuts: A/V schedule held within 33 ms …`). Applies to both
`reencode` (frame-exact) and `copy` (fast preview) modes; the cut-plan cache
is versioned so clips rendered by the old code are re-cut once.

**4. `timeline.snap_to_scenes` (default on).** The film's real shot-change
times are detected once per movie (PySceneDetect, cached in
`_work/shot_boundaries.json`) and every cut's film position is snapped to the
nearest actual camera cut within `snap_tolerance` (0.8s). Each visual then
begins on a real cut of the film instead of drifting in mid-shot — the crisp
edit feel of the reference channels. Optional dependency: `pip install
scenedetect[opencv]`; without it the run simply skips snapping (and says so in
the log).

* Subtitles burned on the clip use `subtitles.lang_font.ar` (default `Arial`,
  shaped Arabic) — swap to any installed Arabic font you prefer.
* A language **without** its own subtitle is still rendered: it is translated
  from the English recap (fully dubbed + subtitled), so a missing `.ar.srt`
  never blocks the run.

> Auto-recap needs a configured LLM — DeepSeek is the shipped default
> (`LLM_PROVIDER=deepseek`, `MODEL_NAME=deepseek-chat`, key in `.env`). Ollama
> and other OpenAI-compatible providers are supported too, but local models
> under-write the target length; DeepSeek is recommended when length matters.
> See `--langs`, `--name`, `--subtitle`.

## Supply your own footage

Pass your clips to `--movie` (comma-separated, or repeat the flag). The bot
normalizes each to 1920×1080@30fps and lo-plays them seamlessly to cover the
length of the narration.

```bash
python -m recap.cli run \
  --movie "part_01.mp4,part_02.mp4,part_03.mp4" \
  --script-file inputs/text/script_en.txt \
  --zh-file inputs/text/script_zh.txt \
  --name my-recap
```

Tips:
* Cut your own clips into short segments (~3–10s each) for a punchier montage,
  the way recap channels do.
* `norm/scene folders cache the normalized clips between runs.

---

## Writing the script: two ways

### A) Hand-written (no API key — what's in this repo)

Put a text file with **one sentence per line** at
`output/_work/script/script_en.txt` and `script_zh.txt` (or pass
`--script-file` / `--zh-file`). The Chinese file must be **line-aligned** with
the English (same number of lines, same order). The repo ships a matching pair.

### B) LLM-assisted (auto-write) — DeepSeek (recommended)

DeepSeek is the shipped default: cheap, fast, reliable JSON output, and it hits
long word targets that small local models routinely miss.

1. Get a key at [platform.deepseek.com](https://platform.deepseek.com) → API keys.

2. Put it in `.env`:

   ```env
   LLM_PROVIDER=deepseek
   MODEL_NAME=deepseek-chat          # or deepseek-reasoner for tangled plots
   DEEPSEEK_API_KEY=sk-...
   ```

   (Nothing else needs changing — `config.yaml` already ships this provider.)

3. Write your plot summary and generate the script (and translation):

   ```bash
   python -m recap.cli script --plot inputs/text/plot_notes.txt --translate
   ```

> Free/local alternative: Ollama + Qwen (`ollama serve`, `ollama pull qwen2.5`,
> then `LLM_PROVIDER=ollama MODEL_NAME=qwen2.5`). Expect shorter-than-requested
> scripts from small local models — DeepSeek is recommended when length matters.
> Token use is bounded per call (each section only gets tokens for its word
> budget); set `RECAP_TOKEN_LOG=1` in `.env` to see per-call in/out counts.

The bot uses a prompt tuned to the *Movie Recaps* voice: present-tense, one
sentence per line, moderate length, no film commentary, and a target word
count. The translation prompt produces idiomatic 简体中文 in the same
line-aligned structure.

---

## Configuration

Edit `config.yaml` (template: `config.example.yaml`). Key knobs:

| Setting | Purpose |
|---------|---------|
| `narration.lang_voice.en` | English narrator (`edge` voice) |
| `narration.lang_voice.zh` | Chinese narrator (`edge` voice) |
| `narration.lang_voice.ar` | Arabic narrator (default `ar-SA-HamedNeural`) |
| `narration.lang_voice.es` | Spanish narrator (default `es-MX-JorgeNeural`) |
| `narration.rate` | speaking rate, e.g. `+5%` |
| `narration.visual_match` | size each section's script to the film time it covers, and pace sentence anchors so the recap plays at 1x end to end (default `true`; `false` = old beat-count budgets) |
| `narration.humanize` | final pass that removes AI-writing tells from the finished script (adapted from blader/humanizer, MIT) while keeping every sentence inside its film window (default `true`) |
| `narration.words_target` | desired narration length |
| `subtitles.font` | must include CJK glyphs for 中文 (default `Noto Serif CJK SC`) |
| `subtitles.lang_font.ar` | Arabic subtitle font (default `Arial`, shaped) |
| `subtitles.line_width_units` | wrap width (中文 glyphs count double) |
| `subtitles.fontsize` | subtitle text size |
| `video.bgm` | optional background-music path |
| `video.bgm_volume` | background music level |

You can also swap the **TTS provider** via `TTS_PROVIDER`/`.env`:
`edge` (free, default) · `elevenlabs` · `openai`. `eleven_multilingual_v2`
and OpenAI `tts-1` both support Chinese.

### 📁 Storage — keep everything off the C: drive

Two settings control where **every** byte goes; both accept absolute paths, so
nothing has to live on C::

| Setting (config.yaml / .env) | Holds | Example |
|---|---|---|
| `project.output_dir` / `OUTPUT_DIR` | final videos + `_work/` | `D:\recap\output` |
| `project.cache_dir` / `CACHE_DIR` | model weights (Whisper/HF), static-ffmpeg, temp | `D:\recap\cache` |

Leave `cache_dir: ""` and the cache auto-places itself next to `output_dir`
(same drive). At startup the engine also re-points `HF_HOME`,
`WHISPER_CACHE_DIR`, `STATIC_FFMPEG_CACHE_DIR` and the Python `TEMP` at the
cache root, so the heavy downloads never touch `C:\Users\...`. See
`.env.example` and `SETUP_GUIDE.md → PART 0`.

---

## Outputs & re-use

`output/<name>_<lang>.mp4` is your deliverable (`output` = `project.output_dir`).
For maximum flexibility the pipeline keeps intermediates in `output/_work/`:

* `en.mp3`, `zh.mp3` — narration audio
* `en.srt`, `zh.srt` — standard subtitles
* `en.ass`, `zh.ass` — styled, burned-in subtitle definitions
* `en.timing.json`, `en.subs.json` — per-sentence timings
* `script/script_en.txt`, `script/script_zh.txt` — your editable scripts
* `assemble/<lang>/NNNN.mp3` — individual sentence takes (optional; enable
  with `narration.segment_audio: true`)

---

## Legal / safety notes

Please only use footage you own or are licensed to use (public-domain works,
your own productions, licensed stock, studio-provided material). Recap
channels typically operate on:
* **transformative narration** over **short** excerpts, and/or
* footage you have a **license** for.

This tool is for producing your own recaps from content you have the right to
use. It does **not** help circumvent copyright. Make sure your channel complies
with YouTube's policies and any license terms before publishing.

---

## Project layout

```
movie-recap-bot/
├── config.yaml                 # active config
├── config.example.yaml         # documented template
├── requirements.txt
├── .env.example                # secrets template (copy to .env)
├── migrations/
│   └── 001_pgvector.sql        # LEGACY Supabase schema (retired vector matcher)
├── inputs/text/
│   ├── plot_notes.txt          # sample plot summary (for LLM mode)
│   ├── script_en.txt           # sample EN recap (one sentence/line)
│   └── script_zh.txt           # sample 简体中文 translation (aligned)
├── tests/
│   ├── test_timeline_sync.py   # length-lock + chronology regression suite
│   ├── test_semantic_engine.py # chunking / JSON parsing / legacy matcher
│   └── test_engine_integration.py  # full A-F orchestration (stubbed externals)
└── recap/
    ├── cli.py                  # `python -m recap.cli` entry point
    ├── config.py               # YAML + env config loader
    ├── dialogue.py             # audio rip + faster-whisper / .srt dialogue
    ├── chunk.py                # Step A: contextual 5-min chunking (30s overlap)
    ├── summarize.py            # Step A: per-chunk "action" summaries
    ├── script.py               # Step B: JSON-array recap script (or load file)
    ├── translate.py            # EN → 简体中文 (line-aligned)
    ├── tts.py                  # Step C: narration + sentence/word timestamps
    ├── timeline.py             # Step D: chronological, audio-locked beat plan
    ├── match.py                # LEGACY vector matcher (no longer in the flow)
    ├── clip.py                 # Step E/F: ffmpeg beat clipping + concat
    ├── subtitles.py            # SRT + ASS generation, CJK-aware wrap
    ├── video.py                # bgm, subtitle burn, mux (shared assembly)
    ├── pipeline.py             # orchestrates run() + auto_recap() (Steps A-F)
    └── util.py                 # ffmpeg/ffprobe helpers
```

---

## Common tasks

**Only build the videos (skip re-narrating)?**
Re-run with the same `--script-file`/`--movie`; narration mp3s are overwritten,
so the pipeline is idempotent. To tweak just subtitles, edit the `.ass`/`.srt`
and re-run `recap/video.py`.

**Use a different voice.** Pick an `edge-tts` voice (e.g. `zh-CN-XiaoxiaoNeural`
for a female Mandarin narrator) and set it in `config.yaml`.

**Make it a repeating bot.** Wrap the CLI in a loop that watches a folder of new
movies (or a queue), e.g.:

```bash
for movie in $(ls ./incoming/*.mkv); do
  python -m recap.cli run --movie "$movie" --name "$(basename "$movie" .mkv)"
done
```

---

## 🐳 Docker (optional — not needed to run locally)

Everything above runs on a normal local Python install; containers are only
for people who prefer Docker. The repo ships `Dockerfile` (CLI),
`Dockerfile.studio` (web control panel) and `docker-compose.yml` (Recap Studio
+ headless CLI container), all wired to the **DeepSeek API**:

```bash
export DEEPSEEK_API_KEY=sk-...            # required by docker compose
docker compose up --build          # Recap Studio on http://localhost:8080
# headless CLI on demand:
docker compose run --rm recap auto --movie /movies/my_movie.mp4 --minutes 14
```

Volumes: put movies you own in `./movies`, rendered clips land in `./output`,
model weights cache in `./cache` (Whisper / static-ffmpeg).

## 🧪 Testing (no movie, no network, no ffmpeg needed)

```bash
python tests/test_timeline_sync.py       # length lock + chronology regressions
python tests/test_semantic_engine.py     # chunking, JSON parsing (legacy matcher)
python tests/test_engine_integration.py  # full Steps A-F orchestration
python tests/test_narration_sync.py      # whisper alignment, word-locked cuts, outro
python tests/test_visual_flow.py        # no-replay cuts + shot-boundary snapping
```

## 🎛️ Tuning the timeline (Step D)

After a run, inspect `output/_work/beats_<lang>.json`: every beat carries its
`film_start`/`film_end` window, its audio-locked `duration` and the 1–3
`cuts` (micro-shots) it was split into. Adjust in `config.yaml` under
`timeline:`:

* `micro_cut_seconds` — aim for a new shot every ~N seconds (default 3.0).
* `max_cuts_per_beat` — how many micro-shots one long sentence may split into.
* `min_cut_seconds` — never flash a shot shorter than this.
* `pre_roll` — start each shot slightly before its narration moment.
* `cut_on_words` — switch shots on measured word boundaries inside a sentence
  (needs `narration.whisper_align`, or edge-tts word boundaries); `false`
  gives the old even split.
* `snap_to_scenes` / `snap_tolerance` — land cuts on the film's real shot
  changes (PySceneDetect, cached; `pip install scenedetect[opencv]`).
* `min_speed` (default 0.35) — the slow-motion floor for dialogue-dense
  sections. Lower (0.25) = tighter narration sync but heavier slow-mo; higher
  (0.6) = milder slow-mo with a slightly larger visual lead; 1.0 = never slow
  down (sections then run ahead of the narration instead).
* `min_new_footage` (default 0.8) — minimum of genuinely new film a cut must
  show before it counts as a cut; smaller steps continue the shot seamlessly.
* `max_lead_seconds` (default 3.0) — safety valve on how far the visuals may
  run ahead of the narrated moment.
* No-replay playback is always on: cuts never re-show footage, so the montage
  walks the film strictly forward.
* `semantic.clip.mode` — `reencode` (frame-exact, default) vs `copy` (fast
  preview; snaps to keyframes and reintroduces drift).
