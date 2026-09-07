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

> **Status:** English is the primary output. Simplified Chinese (简体中文) is
> supported through line-aligned translation (`--langs en,zh`); more languages
> plug into `recap/pipeline.py::_resolve_narration_lines`.

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

### Run it (DeepSeek)

```bash
cd movie-recap-bot
pip install -r requirements.txt            # + faster-whisper (auto-recap needs it)
# put DEEPSEEK_API_KEY=sk-... in .env (platform.deepseek.com -> API keys)

python -m recap.cli auto --movie "C:\Movies\my_movie.mp4" --minutes 14 --name my-recap
```

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

Outputs: `output/<name>_en.mp4` and `output/<name>_zh.mp4`.

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
| `narration.rate` | speaking rate, e.g. `+5%` |
| `narration.words_target` | desired narration length |
| `subtitles.font` | must include CJK glyphs for 中文 (default `Noto Serif CJK SC`) |
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

## 🐳 Docker (package the whole codebase)

The repo root ships `Dockerfile` (CLI), `Dockerfile.studio` (web control panel)
and `docker-compose.yml` (Recap Studio + headless CLI container), all wired to
the **DeepSeek API**:

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
* `semantic.clip.mode` — `reencode` (frame-exact, default) vs `copy` (fast
  preview; snaps to keyframes and reintroduces drift).
