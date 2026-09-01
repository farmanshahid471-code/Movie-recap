# 🎬 Recap Studio 🇬🇧/🇨🇳 — Control Panel

A self-contained web control panel that produces **two recap clips per movie**:

| Clip | Dubbing | Burned-in subtitles |
|------|---------|---------------------|
| `recap_en.mp4` | English | English |
| `recap_zh.mp4` | Mandarin 普通话 | Simplified Chinese 简体中文 |

It drives the `../movie-recap-bot/recap` pipeline (edge-tts narration + word-synced
burned-in subtitles + ffmpeg assembly). Written in **pure Python stdlib** for the server
and **inline HTML/CSS/JS** for the UI — no npm, no build step, no extra web deps.

---

## What it looks like

Dashboard-style panel:
- Top stats row: **Clips Made**, **Runs**, **Storage**, **Tracks**.
- Tabs for **EN clip**, **ZH clip**, **Script editor**, **Settings**.
- Per-tab Run buttons, a **Generate both clips** button, a **Stop run** button and a
  **Close Studio** button (shuts the server down from the browser).
- Per-clip output table with **open** / **download** links (the server streams the
  mp4 with HTTP Range support, so it also plays inline).
- A live **Console** streaming the real pipeline log.

---

## Install / run

### Windows — one click

| File | Does |
|------|------|
| `setup_ui.bat` | **Opens** Recap Studio: finds Python, installs missing deps, checks ffmpeg, starts the panel and opens it in your browser. If it is already running it just opens the browser again. |
| `stop_ui.bat` | **Closes** it: asks the panel to shut down (cancelling any render), waits for the port to be released, then closes its own window. |

You can also close it from the panel (**Close Studio**) or with `Ctrl+C` in the
launcher window. Both batch files honour a `PORT` environment variable
(default `8080`).

### Any OS — from the command line

```bash
pip install PyYAML pysubs2 edge-tts static-ffmpeg openai

python recap-studio/app.py                      # listens on 0.0.0.0:8080
python recap-studio/app.py --port 9000 --open-browser
```

Open <http://localhost:8080>.

> The server itself is stdlib-only, so nothing to install for the UI. `ffmpeg` is
> resolved automatically through `static-ffmpeg` (it downloads its binaries on
> first use). `recap-studio/runner.py` still works as an entry point and simply
> forwards to the same server.

---

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Control panel UI |
| `/api/status` | GET | config + job state + outputs + env health + script paths |
| `/api/logs?n=200` | GET | recent pipeline log lines |
| `/api/scripts?lang=en\|zh` | GET | narration script for one language |
| `/output/<file>.mp4` | GET | stream a rendered clip (supports `Range`) |
| `/healthz` | GET | liveness check |
| `/api/config` | POST | save/update persisted settings (JSON body) |
| `/api/run` | POST | start a background run, body `{"langs":["en","zh"]}` |
| `/api/generate` | POST | the **Generate** button: run both clips |
| `/api/render` | POST | save edited scripts, then re-render (never calls the LLM) |
| `/api/stop_run` | POST | cancel the running job; the server stays up |
| `/api/stop` | POST | cancel any run and shut the server down |

A run in progress makes `/api/run`, `/api/generate` and `/api/render` return
`409`; `/api/render` still saves your scripts in that case.

### Examples

```bash
curl -X POST localhost:8080/api/config -H "Content-Type: application/json" \
  -d '{"movie_path":"C:\\Movies\\my_movie.mp4","storyboard":false}'

curl -X POST localhost:8080/api/run -H "Content-Type: application/json" \
  -d '{"langs":["en","zh"]}'

curl "localhost:8080/api/scripts?lang=en"
curl -X POST localhost:8080/api/stop          # shut the panel down
```

---

## 🎞️ Auto-recap (drop a movie, it writes the narration)

Tick **Auto-recap** in Settings and set a movie. On Run the tool:
1. Extracts dialogue — from the movie's `.srt` if present, else via Whisper ASR,
2. Asks the LLM (Ollama + Qwen) to write the English recap from the transcript,
3. Translates it to Simplified Chinese,
4. Narrates + burns subtitles + assembles both clips.

> Needs a movie file **and** a running Ollama + `qwen2.5`. If no LLM is available it
> falls back to the bundled sample script, so the clip still renders. (Verified
> end-to-end in the sandbox via an OpenAI-compatible endpoint.)

---

## Config

Stored in `config.json` (next to the code). Key fields:

| Field | Meaning |
|-------|---------|
| `movie_path` | path to an owned movie file (empty → placeholder scenes) |
| `storyboard` | use placeholder scenes when no movie is set |
| `duration` | target clip length in seconds (shapes the LLM prompt in auto mode) |
| `auto` | write the narration from the movie's dialogue (needs movie + LLM) |
| `auto_subtitle` | optional explicit `.srt`; blank = look next to the movie, else Whisper |
| `whisper_model` | Whisper model size used when no subtitle exists (default `small`) |
| `voice_en` / `voice_zh` | narrator voices (edge-tts) |
| `subtitle_lang_en` / `subtitle_lang_zh` | subtitle languages (default `en` / `zh`) |
| `llm_provider` / `llm_base_url` / `llm_api_key` / `llm_model` | LLM used for auto scripting (default Ollama + Qwen, no key) |

Everything in this table is applied to the pipeline by `runner.recap_cfg()` —
including the LLM fields, which are pushed into both the recap config and the
environment (`OLLAMA_BASE_URL`, `OPENAI_API_KEY`, …) that `recap/llm.py` reads.


---

## How the two clips are produced

Everything is driven by a single `lang` per run. The pipeline generates the English
story script, then (for the ZH clip) the Chinese translation, narrates each with its own
voice, burns subtitles in the same language as the dubbing, and renders to
1920×1080 H.264 + AAC:

- **EN clip:** `mn`-style narration in `en`, subtitles in `en`.
- **ZH clip:** narration in `zh-CN` Mandarin, subtitles in `简体中文`.

---

## LLM — Ollama + Qwen (free, local, no key)

The recap script (EN) and the Chinese translation are written by a local **Ollama**
server running **Qwen** — free, offline, no API key.

On your machine (one-time setup):

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start the server + pull the model
ollama serve          # keep this running (or run it as a service)
ollama pull qwen2.5
```

Config is already set to `provider: ollama`, `model: qwen2.5`,
`base_url: http://localhost:11434/v1`. No key to paste.

The pipeline uses Ollama's **OpenAI-compatible endpoint** (`/v1/chat/completions`),
so it also works with any OpenAI-compatible local/cloud endpoint by changing the
base URL and model.

> To use a cloud LLM instead (e.g. Zhipu GLM, OpenAI, Claude), set those in the
> Settings tab / `config.yaml` and supply an API key. Ollama is the default because
> it's free and key-free.

---

## Notes / limits

- This is a **review-first** tool. For maximum quality, install the **PySceneDetect**
  video-splitting and **WhisperX** alignment extras and edit the narration before publishing.
- If the Ollama server isn't running, the pipeline falls back to the bundled sample
  scripts (`../movie-recap-bot/inputs/text/`) so the clip still renders.
- Use only footage you own / are licensed to use. Edge-TTS is for personal/non-commercial
  testing; for a monetized channel switch TTS backend (see `movie-narrator` docs).
