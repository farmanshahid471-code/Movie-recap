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

Dashboard-style panel (modeled on the reference screenshot):
- Top stats row: **Clips Made**, **Runs**, **Storage**, **Accounts**.
- Tabs for **EN clip**, **ZH clip**, **Settings**.
- Per-tab Run buttons; a **Settings** panel for movie path, voice/subtitle language,
  auto-recap toggle, and the **Ollama + Qwen** LLM config.
- A live **Console** streaming pipeline logs.

---

## Install / run

Dependencies for the pipeline (installed via pip):

```bash
pip install PyYAML pysubs2 edge-tts static-ffmpeg
```

Start the control panel:

```bash
cd /home/user/recap-studio
python3 app.py          # listens on 0.0.0.0:8080
```

Open <http://localhost:8080>.

> Server uses only the stdlib, so nothing to install for the UI itself. `ffmpeg` is
> resolved automatically through `static-ffmpeg`.

---

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Control panel UI |
| `/api/status` | GET | config + job state + outputs + env health |
| `/api/config` | POST | save/update persisted settings (JSON body) |
| `/api/run` | POST | start background run, body `{"langs":["en","zh"]}` |
| `/api/logs` | GET | recent pipeline log lines (`?n=200`) |
| `/healthz` | GET | liveness check |

### Examples

```bash
curl -X POST localhost:8080/api/config -H "Content-Type: application/json" \
  -d '{"movie_path":"/home/user/my_movie.mp4","storyboard":false}'

curl -X POST localhost:8080/api/run -H "Content-Type: application/json" \
  -d '{"langs":["en","zh"]}'
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
| `auto` | write the narration from the movie's dialogue (needs movie + LLM) |
| `auto_subtitle` | optional explicit `.srt`; blank = look next to the movie, else Whisper |
| `whisper_model` | Whisper model size used when no subtitle exists (default `small`) |
| `voice_en` / `voice_zh` | narrator voices (edge-tts) |
| `subtitle_lang_en` / `subtitle_lang_zh` | subtitle languages (default `en` / `zh`) |
| `llm_base_url` / `llm_api_key` / `llm_model` | LLM (default Ollama + Qwen, no key) |

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
