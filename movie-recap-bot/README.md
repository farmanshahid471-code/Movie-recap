# Movie Recap Bot (EN + 简体中文)

A pipeline that turns **a movie you own** into a short, continuously-narrated
recap video in the style of the *Movie Recaps* YouTube channel — with
burned-in subtitles and full narration voiceover, in **English** and
**Simplified Chinese**.

It reproduces the format of the reference video:
* fast, **present-tense**, beat-by-beat narration over a background montage
* a single voiceover as the "dub" on top of the visual
* **one sentence per subtitle cue**, timed to the narration
* `title + thumbnail + description` recipe for each upload

---

## How the flow works

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
├── inputs/text/
│   ├── plot_notes.txt          # sample plot summary (for LLM mode)
│   ├── script_en.txt           # sample EN recap (one sentence/line)
│   └── script_zh.txt           # sample 简体中文 translation (aligned)
└── recap/
    ├── cli.py                  # `python -m recap.cli` entry point
    ├── config.py               # YAML + env config loader
    ├── script.py               # write/load the EN recap
    ├── translate.py            # EN → 简体中文
    ├── tts.py                  # narration + per-sentence timing
    ├── subtitles.py            # SRT + ASS generation, CJK-aware wrap
    ├── video.py                # montage, bgm, subtitle burn, mux
    ├── pipeline.py             # orchestrates the 5 steps
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
