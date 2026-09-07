# Movie-Recap — Complete Project Walkthrough

A file-by-file, step-by-step explanation of what this bot is and how it works.
Written after reading every file in the repository.

> ⚠️ **Partly outdated.** Several sections below describe the *original* Step-D
> vector matcher (embeddings / pgvector / Supabase) and Ollama+Qwen as the
> default LLM. Both have since changed: beat selection is now a chronological
> audio-locked timeline (`recap/timeline.py`) and the shipped default LLM is
> **DeepSeek** (`deepseek-chat`). Where this file contradicts `FIXES.md` or the
> bot README, the latter describe the current code.

---

## 1. What the bot actually is

It turns **a movie file you own** into a **fully narrated recap video** in the style of
the *Movie Recaps* YouTube channel:

* ~10–16 minutes of continuous, present-tense, beat-by-beat narration,
* spoken by a TTS voice (the "dub"),
* laid over **the film's own footage**, cut to the moments the narration is describing,
* with **burned-in subtitles**, one sentence per cue,
* in **English (master)** and **Simplified Chinese** (line-aligned translation).

Final deliverable: `output/<name>_en.mp4` and `output/<name>_zh.mp4` (1920×1080, H.264 + AAC).

It is designed to run **free and fully offline**: Ollama + Qwen for the LLM, edge-tts for
voice, faster-whisper for transcription, all-MiniLM-L6-v2 for embeddings, ffmpeg for video.
Cloud providers (DeepSeek, Groq, Gemini, OpenAI, Anthropic, ElevenLabs, Supabase) are all
optional swap-ins.

---

## 2. Repository map

```
Movie-recap/
├── movie-recap-bot/          ← THE ENGINE (all the real logic)
│   ├── recap/                ← the Python package, 16 modules
│   ├── config.yaml           ← active config (+ config.example.yaml template)
│   ├── .env.example          ← secrets template (keys, storage roots)
│   ├── requirements.txt
│   ├── migrations/001_pgvector.sql   ← Supabase schema for Step D
│   ├── inputs/text/          ← bundled sample EN/ZH scripts + plot notes
│   ├── scripts/verify_supabase.py    ← standalone Supabase checker
│   └── tests/                ← 2 dependency-light test suites
│
├── recap-studio/             ← THE WEB UI (control panel)
│   ├── app.py                ← stdlib-only HTTP server + JSON API
│   ├── runner.py             ← glue: UI settings → engine config, jobs, logs
│   ├── static/index.html     ← the entire UI (inline CSS+JS, no build step)
│   ├── config.json           ← persisted panel settings (local)
│   ├── config.docker.json    ← same, container defaults
│   └── tools/                ← ensure_ffmpeg / portcheck / shutdown helpers
│
├── setup_ui.bat / stop_ui.bat   ← Windows one-click open/close
├── Dockerfile (CLI) / Dockerfile.studio (panel) / docker-compose.yml  ← optional (container users only)
├── SETUP_GUIDE.md            ← storage policy + legacy Supabase/Docker notes, troubleshooting
```

**Important:** Earlier snapshots of this repo shipped `.movie-narrator/` and
`movie-narrator-jobs/` (configs for an unrelated 3rd-party pip package) plus
stale demo artifacts (`demo/`, `uploads/`, `sdtest/`, `output/Sample Movie/`,
`recap-studio/autovideo/`). None of them were used by any code in this repo
and they have been removed.

---

## 3. The two engines

The bot ships **two** pipelines. Both live in `recap/pipeline.py`.

| | `auto_recap()` — **semantic, Step A–F** | `run()` — **legacy 5-step** |
|---|---|---|
| CLI | `python -m recap.cli auto --movie film.mp4` | `python -m recap.cli run --movie ...` |
| Script source | LLM writes it from the film's own dialogue | your text file, or LLM from plot notes |
| Footage choice | **semantically matched** per narration line | scene-detected / evenly-spaced montage |
| Needs | movie + LLM + whisper/srt + embeddings | script (or LLM); movie optional |
| Studio setting | `engine: semantic` (default) | `engine: recap` |

The semantic engine is the product. The legacy engine is the fallback for hand-written
narration and for running without a real movie (placeholder colored "storyboard" scenes).

---

## 4. The Step A–F semantic flow, in detail

`pipeline.auto_recap(cfg, movie)` — this is the heart of the bot.

```
movie.mp4
 ├─ A. ffmpeg rips audio → faster-whisper → timestamped transcript
 │     → 5-min chunks w/ 30s overlap → per-chunk "action" summaries (LLM)
 ├─ B. summaries → LLM → STRICT JSON array of narration sentences
 ├─ C. sentences → edge-tts → en.mp3 + sentence & word timestamps
 ├─ D. embed transcript + script (MiniLM) → vector store → cosine match
 │     → the exact film moment for each story beat
 ├─ E. ffmpeg cuts seg_NNN.mp4 per beat (stream copy = fast)
 └─ F. concat → burn .ass subtitles → mux narration → <name>_en.mp4
```

### Step A — dialogue extraction & contextual chunking

`recap/dialogue.py`

1. **Subtitle first.** `find_subtitle_near()` looks for `movie.srt/.ass/.vtt/.sub` next to
   the film; if the folder has exactly one subtitle it uses that (releases rarely name
   subs identically). Parsed with `pysubs2` (utf-8, then latin-1 fallback). This is
   instant and skips the whole Whisper download.
2. **Else Whisper ASR.** `_extract_audio()` runs
   `ffmpeg -vn -ac 1 -ar 16000 -c:a pcm_s16le` into the workdir (never next to the
   source movie — that folder may be read-only). Then it tries, in order:
   `faster_whisper` (int8, `beam_size=1`, prints progress every 25 segments) →
   `openai-whisper` → `whisperx`. Result: cues of `{text, start, end, words[]}`.

**Transcript caching:** `transcript.meta.json` stores the movie's
`{path, size, mtime_ns}`. If unchanged, `transcript.json` is reloaded instead of
re-transcribing — so an EN run followed by a ZH run never transcribes twice.

Outputs: `_work/transcript.json`, `_work/transcript.srt`, `_work/script/transcript.txt`.

`recap/chunk.py` — **contextual chunking.** A 2-hour film's transcript will not fit any
LLM context window. `chunk_cues()` slides a 300s window forward by
`300 − 30 = 270s`, so consecutive blocks share a 30-second overlap that carries the plot
across the seam. A cue that straddles a boundary is kept **whole in both** neighbours —
never split. Each chunk is written to `_work/chunks/chunk_NNN.txt`.

`recap/summarize.py` — **per-chunk action summaries.** For every block the LLM is asked
(system: *"You are a movie plot analyst… you never quote dialogue, you say what the
characters do"*) to infer the on-screen **action** from the dialogue, as short
present-tense beats. Notable engineering:

* Output budget scales with chunk size: `max(500, min(2200, len(text) * 0.22))`.
* Each finished summary is **appended immediately** to `script/summaries.txt`.
* **Resume**: `_read_partial()` parses `--- Chunk N ---` markers and a `.sig` file holds
  a SHA-1 of the chunk set. Same signature → skip the contiguous prefix already done.
  Different signature (new film / re-extracted transcript / changed window) → the stale
  file is wiped. This is what makes an interrupted 40-minute CPU pass survivable.
* `chunking.model` can point at a smaller/faster model (e.g. `qwen2.5:3b`) just for this
  pass, and `chunking.parallel` enables a `ThreadPoolExecutor` for remote LLMs.

### Step B — the narration script as JSON

`recap/script.py::generate_script_json()`

The merged summaries go to the LLM with `SYSTEM_RECAP_WRITER` ("professional YouTube movie
recap scriptwriter… third person, no dialogue quoting") and `PROMPT_SCRIPT_JSON`, which
demands **only** a JSON array of sentence strings. Rules baked into the prompt: present
tense, one self-contained action beat per element, 8–22 words, never say "the movie / we
see / the scene shows", cover the whole story **including** the ending, ~`target` words.

`parse_sentences_json()` is deliberately paranoid — five fallback layers:
1. strip ```` ```json ```` fences → 2. keep only the outermost `[...]` → 3. strip trailing
commas and retry → 4. regex out every quoted token → 5. treat it as loose text, one
sentence per line. `_clean_sentences()` also appends missing terminal punctuation
(CJK-aware) so every sentence becomes a clean subtitle cue.

Guard: fewer than 10 sentences ⇒ `DialogueError` ("the recap looks broken").
Outputs `script/script_en.json` + `script/script_en.txt`.

### Step C — voiceover with timing

`recap/tts.py`

Default provider **EdgeTTS** (free Microsoft neural voices, `en-US-ChristopherNeural` /
`zh-CN-YunxiNeural`). All sentences are joined with `\n` and streamed in one call so the
pauses are natural. From the stream it collects:

* `audio` chunks → written to `<lang>.mp3`
* `SentenceBoundary` events → sentence cues (offsets are 100-ns ticks ÷ 10,000,000)
* `WordBoundary` events → `_attach_words()` assigns each word to its containing sentence
  (60 ms tolerance)

`_build_cues()` maps boundaries onto the original sentences by index; if the counts
diverge it falls back to length-weighted interpolation of the total duration.
Network resilience: `_edge_retryable()` detects DNS/connection faults and retries
(`TTS_RETRIES`, default 3, with backoff); `_edge_friendly()` converts the raw error into
actionable advice.

Alternatives: `elevenlabs` and `openai` providers exist but are **incomplete** — they
synthesize audio yet return empty cue lists, so timing/subtitles would break. edge is the
supported path.

Outputs: `<lang>.mp3`, `<lang>.timing.json`, optional per-sentence
`assemble/<lang>/NNNN.mp3`.

### Step D — semantic timestamp mapping (the clever bit)

`recap/match.py`

This is what makes it a *recap* rather than a voiceover over random footage.

1. `Embedder` lazily loads `sentence-transformers` `all-MiniLM-L6-v2` (384-dim, CPU).
2. Every **transcript cue** is embedded and pushed into a vector store.
3. Every **narration sentence** is embedded.
4. `cosine_similarity_matrix()` computes the full (sentences × cues) similarity matrix
   with numpy.
5. Greedy matching with **de-duplication**: for each sentence, walk the top-`k` candidates
   in descending score; skip cues already `used`; stop at `min_score`. The winning cue's
   `start`/`end` become the film timestamps for that story beat.
6. Losers get `fallback: true`, then `_fill_fallback_anchors()` places them at the
   **nearest real dialogue cue** to their proportional position in the runtime (so the
   visual still shows speech, not silence). With no transcript at all, `_even_fallback()`
   spaces anchors uniformly.

Result written to `_work/beats.json` — the tuning surface: each line shows `cue_idx`,
`start`/`end`, cosine `score`, `source_text`, `fallback`.

**Two pluggable stores, identical semantics:**

* `LocalVectorStore` — SQLite file (`_work/vectors.db`) + numpy. Exact cosine over the
  whole transcript (< ~5000 cues, no index needed). Zero setup, the default.
* `SupabaseVectorStore` — hosted Postgres + pgvector. Talks PostgREST over
  `urllib` (no SDK). `add_cues()` DELETEs then bulk-INSERTs rows with embeddings as plain
  JSON arrays; `search()` calls the `match_cues` RPC.

`make_store()` resolves `semantic.store`: `local` | `supabase` | `auto` (Supabase when
`SUPABASE_URL`+`SUPABASE_SERVICE_KEY` or `SUPABASE_DB_URL` exist, else local).

`migrations/001_pgvector.sql` creates the `vector` extension, a `transcript_cues` table
(`idx, text, start_ms, end_ms, session, embedding vector(384)`) with an **HNSW cosine
index**, and the `match_cues(query_embedding, match_count, match_threshold, session_name)`
SQL function returning `1 - (embedding <=> query)` as similarity. The file documents *why*
there are no `json → vector` casts (hosted Supabase blocks casts on built-in types with
`ERROR 42501`, and PostgREST accepts JSON arrays for vector columns natively).

### Steps E + F — clipping and assembly

`recap/clip.py`

**The narration is the master clock.** For each narration cue *i*:

```python
need  = clamp(cue.duration + clip_pad, min_clip, max_clip)   # 0.8 … 10.0 s
start = max(0, beat_start − pre_roll)                        # 0.5 s of lead-in
```

So the matched timestamp decides **where** in the film the shot comes from; the narration
line decides **how long** it plays. The concatenated visual is therefore exactly as long
as the audio and subtitles can never drift.

* `cut_segment()` — `mode: copy` uses `-ss/-i/-t -c copy` (fast, keyframe-accurate);
  `mode: reencode` scales/crops to 1920×1080, sets fps and re-encodes frame-exact.
* `concat_segments()` — the ffmpeg **concat demuxer** with a generated list file.
* `_cover_target()` — trims (or `-stream_loop -1` loops) to the exact target length.

`recap/subtitles.py` builds the cues: `_wrap_long()` is **width-aware** — CJK glyphs count
1.0 unit, Latin 0.5, so `line_width_units: 30` means ~30 Chinese or ~60 Latin characters
per line. Writes `.srt` (via pysubs2) and a hand-built `.ass` with a `[V4+ Styles]`
block (white text, dark outline, bottom-centre, `PlayRes 1920×1080`) and a
`\fad(150,150)` soft fade on every event.

`recap/video.py::burn_and_mux()` does the final render:
`-vf ass=<file> -map 0:v -map 1:a -c:v libx264 -c:a aac -shortest -movflags +faststart`.

Two subtle ffmpeg fixes worth noting:
* The `.ass` is passed as a **bare filename with `cwd` set to its folder**, because an
  absolute Windows path (`D:\recap\_work\en.ass`) contains a colon and ffmpeg's filter
  parser splits on `:`.
* `_filter_arg()` double-escapes `\ ' : , ; [ ]` for the filtergraph.

`add_bgm_if_any()` optionally mixes looped background music at `bgm_volume`.

---

## 5. Resume / idempotency system

`auto_recap` is gated by **content-signature marker files**, so a crash, power cut or
network failure never redoes finished expensive work:

| Step | Marker | Signature covers |
|---|---|---|
| A (transcript) | `transcript.meta.json` | movie path + size + mtime |
| A (summaries) | `summaries.txt.sig` | SHA-1 of all chunk texts |
| B (script) | `script_en.marker.json` | summaries + LLM provider/model + word target |
| C (narration) | `.nar_<lang>.marker.json` | lines + voice + provider + rate |
| E/F (render) | `.render_<lang>.marker.json` | cues + beats + movie + clip mode + subtitle cfg + bgm |

Change the model, the voice or the film and the signature changes → that step re-runs.
Everything upstream is reused. Delete the marker (or the artifact) to force a redo.

---

## 6. LLM layer

`recap/llm.py` — one thin provider-agnostic wrapper. Everything except Anthropic goes
through the **OpenAI-compatible** `chat.completions` path:

| Provider | Key | Base URL | Default model |
|---|---|---|---|
| `ollama` (default) | none | `http://localhost:11434/v1` | `qwen2.5` |
| `deepseek` | `DEEPSEEK_API_KEY` | `api.deepseek.com/v1` | `deepseek-chat` |
| `groq` (free tier) | `GROQ_API_KEY` | `api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| `gemini` (free tier) | `GEMINI_API_KEY` | Google OpenAI-compat endpoint | `gemini-3.6-flash` |
| `openai` | `OPENAI_API_KEY` | `api.openai.com/v1` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | SDK-fixed | `claude-3-5-sonnet-latest` |

`verify_model()` is a genuinely useful piece of UX: before the long passes it queries
Ollama's `/models` and fails fast with instructions if the model was never pulled —
otherwise Ollama **silently downloads it**, which is indistinguishable from "hung".
It also normalises the `name` vs `name:latest` mismatch. `LLM_TIMEOUT` defaults to
3600 s because CPU generation of a full script is slow.

`recap/translate.py` — EN → 简体中文. The prompt is itself written in Chinese and demands
strict **line-for-line alignment** (no merging/splitting), colloquial 电影解说 narration
style, present tense, consistent character names. That alignment is what lets the ZH clip
reuse the EN semantic anchors.

---

## 7. Configuration & storage policy

`recap/config.py` — layered: `_DEFAULTS` dict → `config.yaml` (deep-merged) → env
overrides (`LLM_PROVIDER`, `MODEL_NAME`, `OLLAMA_BASE_URL`, `TTS_PROVIDER`, `OUTPUT_DIR`,
`CACHE_DIR`) → `.env` loaded via `os.environ.setdefault` (real env always wins).
Resolves `project._out` and `project._cache` to absolute paths, then calls
`storage.bootstrap()`.

`recap/storage.py` — a deliberate **"nothing on the C: drive"** policy, for users with a
small system disk. Two roots (`output_dir`, `cache_dir`) drive everything, and bootstrap
re-points the libraries' cache env vars before they ever download anything:

```
HF_HOME, HF_HUB_CACHE, SENTENCE_TRANSFORMERS_HOME, TORCH_HOME,
WHISPER_CACHE_DIR, STATIC_FFMPEG_CACHE_DIR, TMP/TEMP/TMPDIR
```

It only sets keys the user didn't export themselves, understands Windows drive letters on
any OS (`_DRIVE_RE`), and is invoked from `recap/__init__.py` at import time when
`OUTPUT_DIR`/`CACHE_DIR` are present. `SETUP_GUIDE.md → PART 0` explains the three things
the code *can't* move (Python itself, `OLLAMA_MODELS`, a system ffmpeg).

`recap/util.py` — ffmpeg plumbing: `which_ffmpeg/ffprobe` prefer a system binary, else
lazily unpack `static-ffmpeg` into the cache root (with the correct per-platform
subfolder). `run()` echoes every command and raises with trimmed stdout/stderr.
`probe_duration()` uses ffprobe, and if ffprobe is missing (common on Windows) falls back
to regexing `Duration:` out of ffmpeg's stderr banner.

---

## 8. CLI

`recap/cli.py` — `python -m recap.cli <cmd>`:

| Command | Does |
|---|---|
| `auto --movie F [--minutes N] [--langs en,zh] [--subtitle S] [--whisper-model/-device] [--name]` | the Step A–F semantic recap |
| `run [--movie a,b] [--storyboard] [--script-file] [--zh-file] [--langs] [--name]` | legacy 5-step engine |
| `script --plot notes.txt [--translate]` | just write the EN script (+ZH) |
| `translate --en file [--out]` | just translate |

`--minutes` converts to words at ~150 wpm, clamped to `words_min/words_max`.
`auto` deliberately **deletes any staged `script_en.txt`/`script_zh.txt`** first, because
the semantic engine must regenerate the narration from this film — a leftover script from
a previous movie would silently produce a recap of the wrong film.

---

## 9. Recap Studio (the web UI)

### `app.py` — server
Pure stdlib `ThreadingHTTPServer`. Routes:

| Endpoint | Purpose |
|---|---|
| `GET /` | the panel HTML |
| `GET /api/status` | config + job state + outputs + readiness + env health |
| `GET /api/logs?n=` | tail of the log ring buffer |
| `GET /api/scripts?lang=` | narration script for one language |
| `GET /api/browse?kind=&path=` | **server-side file browser** |
| `GET /output/<f>.mp4` | stream a clip, HTTP **Range** aware, `?dl=1` forces download |
| `GET /healthz` | liveness |
| `POST /api/config` | persist settings |
| `POST /api/run` \| `/api/generate` | start a background run |
| `POST /api/render` | save edited scripts + re-render (legacy engine only; 400 on semantic) |
| `POST /api/stop_run` | cancel the job, server stays up |
| `POST /api/stop` | cancel + shut the server down |

Nice touches: `handle_error()` swallows the expected `ConnectionAborted/Reset/BrokenPipe`
from the 1.2 s log poll (the stdlib default prints a full traceback for each);
`_stream_clip()` sanitises the filename (`Path(name).name` + regex) against traversal;
`/api/browse` exists because **browsers refuse to reveal a chosen file's absolute path**,
so the server lists its own filesystem (drive letters on Windows, `/` elsewhere).

### `runner.py` — glue
* Adds `../movie-recap-bot` to `sys.path` and imports the real pipeline.
* `_Tee` wraps `sys.stdout` during a run so the pipeline's `print()` output lands in the
  ring buffer (600 lines) **and** the real console **and** `studio.log`.
* `LOCK` is an **RLock** — a plain Lock self-deadlocked when the worker logged while
  holding it (the comment records the bug).
* `check_movie()` is unusually thorough: rejects folders (naming the videos inside),
  resolves **Windows' hidden file extensions** (`MyFilm.2026.HDRip` → `.mp4`), catches
  0-byte files, and only logs each resolution once (the status poll runs every 3 s).
* `output_dir()` write-tests the configured folder, detects a Windows path used on Linux,
  and falls back to `recap-studio/output` with a warning rather than dying mid-render.
* `ensure_whisper()` / `ensure_embeddings()` **pip-install faster-whisper and
  sentence-transformers on demand**, streaming pip's output (splitting on `\r`) into the
  Console so the multi-minute torch download doesn't look frozen. Adds a specific hint if
  Python 3.13 has no matching wheel.
* `_apply_llm()` pushes the panel's LLM fields into both the recap config **and** the
  environment (`OLLAMA_BASE_URL`, `DEEPSEEK_API_KEY`, …) so UI and CLI can't drift.
* `readiness()` produces the plain-English blocking list the UI shows before Run.
* `start_run()` spawns a daemon worker: semantic engine = **one** `auto_recap` call for
  all languages (whisper/chunks/summaries/beats computed once); legacy = one
  `pipeline.run` per language.

### `static/index.html` — UI
~750 lines, no framework. Tabs (EN clip / ZH clip / Script editor / Settings), a stats
row, per-clip output tables with open/download/copy-URL, a live Console polling
`/api/logs`, a modal file browser, environment "pills" and a readiness box. Careful
state handling: it won't re-render the panel while you're typing, auto-saves unsaved
settings on tab switch and before any Run, and warns on `beforeunload` with a dirty script.
Switching LLM provider auto-fills that provider's model + base URL.

---

## 10. Deployment

* **Windows one-click** — `setup_ui.bat` sets `RECAP_DATA=D:\recap-data` (falls back to a
  project-local folder without a D: drive), routes `TEMP/TMP/PIP_CACHE_DIR/CACHE_DIR/
  HF_HOME/WHISPER_CACHE_DIR/STATIC_FFMPEG_CACHE_DIR/RECAP_LOG_DIR` there, finds Python,
  installs missing deps, verifies ffmpeg, kills a stale instance on the port
  (`portcheck.py` + `shutdown.py`), then launches the panel with `--open-browser`.
  `stop_ui.bat` POSTs `/api/stop` and waits for the port to release.
  `.gitattributes` marks `*.bat -text` so CRLF survives — LF-only batch files
  flash-close in cmd.exe.
* **Docker** — `Dockerfile` (python:3.11-slim + ffmpeg, entrypoint `python -m recap.cli`)
  and `Dockerfile.studio` (same + the panel on 8080). `docker-compose.yml` wires
  **ollama** (healthchecked, model volume) + **studio**, with the CLI container behind a
  `cli` profile. Mounts `./movies:ro`, `./output`, a `model-cache` volume, and binds
  `config.docker.json` so panel settings persist on the host. Supabase vars pass through.

---

## 11. Tests

Both are runnable with **no movie, no network, no ffmpeg**. I ran both — they pass.

* `tests/test_semantic_engine.py` — chunk overlap arithmetic (second window must start at
  270 s and share cues across the seam), the messy-JSON parser (fences + trailing comma +
  prose preamble), and the local store: correct nearest hit, no cue reused across
  sentences, and hopeless lines falling back in story order.
* `tests/test_engine_integration.py` — stubs whisper/LLM/embeddings/TTS/ffmpeg and runs
  the **real** `auto_recap` orchestration, then asserts every artifact exists
  (`transcript.json`, `script_en.json/.txt`, `beats.json`, `en.mp3`, `en.srt`, `en.ass`,
  `en.timing.json`, the output mp4) and that the `.txt` sidecar is line-aligned with the
  JSON. This is the one that catches wiring mistakes.

`scripts/verify_supabase.py` is a separate live check: loads `.env`, asserts the store
resolves to Supabase, inserts two deterministic 384-dim rows over REST, runs `match_cues`,
verifies the right row wins, then empties the table.

---

## 12. Observations worth knowing

Things I found while reading that affect behaviour:

1. **Variable shadowing in Step C → Step D (real bug).** In `pipeline.py` the transcript
   cue list is held in `cues` (Step A, line ~471). Inside the Step C narration-resume
   branch (line ~545) `cues` is **reassigned** to the list of narration `TimedCue`s.
   Step D then passes that same `cues` into `match.map_beats(sentences, cues, …)`.
   So on a *resumed* run where narration is reused, the matcher embeds the **narration**
   instead of the movie transcript — every sentence matches itself (score ≈ 1.0) and the
   `start`/`end` become narration times, not film times. Beats would be wrong.
   On a fresh run the resume branch never fires, so it's invisible until you re-run.
   Fix: rename the loop-local to e.g. `nar_cues`.

2. **The pgvector RPC is never used during a real run.** `map_beats()` calls
   `store.add_cues()` and then does the similarity itself in numpy — `store.search()` is
   only exercised by `verify_supabase.py`. So Supabase currently acts as a *write-only
   mirror* of the embeddings; switching stores does not change matching results (which is
   why the local fallback is genuinely equivalent). The HNSW index and `match_cues()`
   are ready but unused by the pipeline.

3. **ElevenLabs / OpenAI TTS are stubs.** Both return an empty cue list
   (`_ElevenLabs` also appends every utterance to one file in `"ab"` mode). Timing and
   subtitles would break. Only `edge` is production-ready.

4. **`_get_plot()` checks the same path twice** — the "bundled fallback" branch reads the
   identical `workdir/script/plot_notes.txt`, so there is no real fallback.

5. **`recap-studio/config.json` ships `"output_dir": "D:\\recap"`** — on Linux/macOS
   `output_dir()` detects the Windows path and falls back to `recap-studio/output` with a
   log warning, so it degrades gracefully.

6. Minor: `dialogue.extract_dialogue()` has an unused `whitelist` parameter path,
   `tts._ensure_audio_ext()` is a no-op, `chunk_summary_budget()` is defined but unused
   (summarize computes its own budget), and `config.py`'s `_DEFAULTS` has no
   `narration.segment_audio` key although `pipeline.run()` reads it (`.get()` default
   `False`, so harmless).

7. **Legal posture** is stated repeatedly in the docs: intended for footage you own or are
   licensed to use; edge-tts is a reverse-engineered endpoint and is flagged
   personal/non-commercial, with `openai`/`elevenlabs` suggested for monetised channels.

---

## 13. The shortest possible summary

> Rip the film's audio → transcribe it → slice the transcript into overlapping 5-minute
> blocks so the LLM never overflows → have the LLM say *what happens* in each block →
> have it write the whole recap as a JSON array of one-sentence beats → speak it with
> edge-tts and capture per-sentence timings → embed both the transcript and the script and
> cosine-match each narration line to the dialogue moment it describes → cut that moment
> out of the film for exactly as long as the line takes to say → concat, burn subtitles,
> mux the voiceover. Repeat for Chinese using a line-aligned translation over the same
> beats.
