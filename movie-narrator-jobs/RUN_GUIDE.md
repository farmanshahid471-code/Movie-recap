# Movie-Narrator — 2-clip EN + 简体中文 setup (Ollama + Qwen, free/no key)

Tool: [zcbacxc/movie-narrator](https://github.com/zcbacxc/movie-narrator) (pip `movie-narrator`, v1.2.1).
Two job configs are ready: `job_en.yaml` (English) and `job_zh.yaml` (简体中文).

The recap script (EN) + Chinese translation are written by a **local Ollama** server
running **Qwen** — **free, fully offline, no API key**. `movie-narrator` *defaults* to
Ollama, so there's nothing to pay for and nothing to paste.

---

## The 2 clips

| Clip | `lang` | voice | dubbing | burned-in subtitles |
|------|--------|-------|---------|---------------------|
| **Clip 1 — English** | `en` | `en-US-ChristopherNeural` | English | English |
| **Clip 2 — Chinese** | `zh` | `zh-CN-YunxiNeural` | Mandarin 普通话 | Simplified Chinese |

Each is a separate `mn create` run.

---

## One-time setup on your machine

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start the server + pull the model
ollama serve          # keep running (or install as a service)
ollama pull qwen2.5
```

The config is already set. `~/.movie-narrator/.env`:

```
MN_LLM_BASE_URL=http://localhost:11434/v1
MN_LLM_API_KEY=ollama
MN_LLM_MODEL=qwen2.5
MN_TTS_PROVIDER=edge
MN_DEFAULT_VOICE=zh-CN-YunxiNeural
```

---

## Install the tool + extras

```bash
pip install movie-narrator
# optional, higher quality (scene cut + word-level alignment); needs PyTorch
pip install "movie-narrator[media]" "movie-narrator[ml]"
```

---

## Point it at your movie

Set `video:` in **both** job files to your owned movie's path (or pass `--video`):

```bash
mn create --config /home/user/movie-narrator-jobs/job_en.yaml
mn create --config /home/user/movie-narrator-jobs/job_zh.yaml
```

Outputs land in `output/<movie>/` — `final.mp4` (your clip), `narration.mp3`
(dubbing), `subtitle.srt` (burned-in), `script.md`, etc.

---

## Options

- **Length/pacing:** `duration` + `narration_preset` (`douyin-fast`/`mainstream-dry`/`bilibili-long`).
- **Word-level subtitle timing:** set `steps.align: true` (with the `[ml]` extra).
- **Bilingual (EN+中文 together):** `steps.translate: true`, `subtitle_lang: en`, `subtitle_mode: bilingual`.
- **Vertical 9:16 Shorts:** `video_format: "9:16"`.

---

## Compliance notes

- Default TTS `edge` is a reverse-engineered interface — **personal/non-commercial only**.
  For a monetized channel set `MN_TTS_PROVIDER=openai` (or `mimo`).
- **AGPL-3.0**: fine for personal use; if you host it as a service, publish your source.
- Use only footage you own / are licensed to use.
