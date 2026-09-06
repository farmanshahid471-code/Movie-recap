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

## 🎞️ The Step A-F semantic engine (recommended)

The `auto` command implements the production workflow in six steps. It never
lets the movie transcript overflow the LLM context window, and it chooses each
visual beat of the final video by **semantic similarity** to the dialogue that
inspired it.

```
movie.mp4
 ├─ A. ffmpeg audio -> faster-whisper -> timestamped transcript (transcript.json/.srt)
 │      -> 5-min chunks w/ 30s overlap  -> per-chunk "action" summaries (LLM)
 ├─ B. summaries -> Qwen 2.5 -> STRICT JSON array of narration sentences
 ├─ C. sentences -> TTS (edge) -> en.mp3 + sentence (+word) timestamps
 ├─ D. embed transcript + script (all-MiniLM-L6-v2) -> pgvector store
 │      -> cosine match per sentence -> the film moment for that story beat
 ├─ E. ffmpeg loop: -ss/-to -> seg_NNN.mp4 (stream copy = fast, no re-encode)
 └─ F. concat demuxer -> burn .ass subtitles -> mux narration -> <name>_en.mp4
```

### Run it

```bash
cd movie-recap-bot
pip install -r requirements.txt            # + faster-whisper, sentence-transformers
ollama serve                               # keep running
ollama pull qwen2.5

python -m recap.cli auto --movie "C:\Movies\my_movie.mp4" --minutes 14 --name my-recap
```

Step D vector store:

| Store | When | Setup |
|-------|------|-------|
| **local** (SQLite fallback) | no Supabase credentials yet | nothing — exact same search logic |
| **supabase** (pgvector) | `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` in `.env` | run `migrations/001_pgvector.sql` once in the SQL editor (or set `SUPABASE_DB_URL` and the pipeline bootstraps it) |

Intermediates land in `output/<name>/_work/`: `transcript.json`/`.srt`,
`chunks/`, `script/summaries.txt`, `script/script_en.json` (the sentence
array), `script/script_en.txt`, `en.mp3`, `en.timing.json` (word-level when
edge-tts provides it), `beats.json` (the semantic line→timestamp map) and
`beats/seg_NNN.mp4` (the raw film cuts).

Useful flags: `--minutes` (target length), `--langs en,zh`,
`--subtitle movie.srt` (skip Whisper), `--whisper-model`,
`--whisper-device auto|cpu|cuda`.

> The extraction step is cached per movie file (`transcript.json` + a file
> fingerprint) so EN/ZH runs don't transcribe twice. Delete it to force a
> re-extract.

---

## How the flow works (classic 5-step engine)

The legacy `run` command (used by Recap Studio) works like this:

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
2. **LLM writes the recap** — Ollama/Qwen reads the timestamped transcript and produces
   the Movie-Recaps-style English narration (present tense, beat by beat).
3. **Translate to Simplified Chinese** automatically.
4. **Narrate + burn subtitles + assemble** both clips (EN + 简体中文).

Outputs: `output/<name>_en.mp4` and `output/<name>_zh.mp4`.

> Auto-recap runs end-to-end with the **Ollama + Qwen** default (free, local, no key).
> If no LLM is configured it falls back to a pre-written script (or fails with a clear
> message). See `--langs`, `--name`, `--subtitle`.

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

### B) LLM-assisted (auto-write) — **no API key with Ollama + Qwen**

The default is **Ollama + Qwen**, which is **free and fully local** (no key).

1. Install Ollama and pull the model:

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve          # start the local server
   ollama pull qwen2.5   # the model used by this config
   ```

2. `config.yaml` is already set to `provider: ollama`, `model: qwen2.5`,
   `base_url: http://localhost:11434/v1`. Nothing else to configure.

3. Write your plot summary and generate the script (and translation):

   ```bash
   python -m recap.cli script --plot inputs/text/plot_notes.txt --translate
   ```

> To use a cloud LLM instead (Zhipu GLM, OpenAI, Claude, DeepSeek), set the
> provider + model + key in `.env` / `config.yaml`. Ollama is the default because
> it's free, key-free, and strong at both English and Chinese.

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

---

## Outputs & re-use

`output/<name>_<lang>.mp4` is your deliverable. For maximum flexibility the
pipeline keeps intermediates in `output/_work/`:

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
│   └── 001_pgvector.sql        # Supabase schema for Step D (pgvector)
├── inputs/text/
│   ├── plot_notes.txt          # sample plot summary (for LLM mode)
│   ├── script_en.txt           # sample EN recap (one sentence/line)
│   └── script_zh.txt           # sample 简体中文 translation (aligned)
├── tests/
│   ├── test_semantic_engine.py # chunking / JSON script / semantic matcher
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
    ├── match.py                # Step D: embeddings + pgvector/local cosine match
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
and `docker-compose.yml` (Ollama + panel + optional CLI container):

```bash
docker compose up --build          # Ollama + Recap Studio on http://localhost:8080
# headless CLI on demand:
docker compose run --rm recap auto --movie /movies/my_movie.mp4 --minutes 14
```

Volumes: put movies you own in `./movies`, rendered clips land in `./output`,
model weights cache in `./cache`. Pass `SUPABASE_URL` / `SUPABASE_SERVICE_KEY`
as environment variables when your Supabase pgvector project is ready — until
then the pipeline uses the built-in local vector store.

## 🧪 Testing (no movie, no network, no ffmpeg needed)

```bash
python tests/test_semantic_engine.py     # chunking, JSON parsing, matching
python tests/test_engine_integration.py  # full Steps A-F orchestration
```

## 🎛️ Tuning the semantic mapping (Step D)

After a run, inspect `output/_work/beats.json`: every narration line lists the
matched dialogue cue (`cue_idx`), the film timestamps to cut, the cosine
`score`, and whether it fell back to even spacing (`fallback: true`). Adjust in
`config.yaml`:

* `semantic.min_score` — raise it if lines are matched to unrelated moments,
  lower it if too many lines fall back.
* `semantic.pre_roll` — seconds of footage before the matched cue (0.3–0.8s).
* `semantic.clip.max_clip` — cap per-beat length so one shot never drags.
* `semantic.clip.mode` — `copy` (fast, keyframe cuts) vs `reencode`
  (frame-exact, slower).
* `semantic.store` — `supabase` once your project is deployed (with
  `migrations/001_pgvector.sql` applied once in the SQL editor).
