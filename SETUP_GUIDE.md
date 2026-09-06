# Setup Guide — Supabase (pgvector) + Docker

This guide walks you through the two remaining pieces end to end:

1. **Part 1 — Supabase pgvector** (the Step-D vector database that maps each
   narration line to the exact film moment).
2. **Part 2 — Docker** (packaging the whole codebase so you can run it
   containerized).

Everything is copy‑paste ready. Do Part 1 first if you run on your Windows PC
locally, or follow Part 2 if you want the Docker flow — you can also do both
(Part 1's credentials feed straight into Part 2's containers).

---

## Preflight checklist

| Requirement | For Part 1 (Supabase) | For Part 2 (Docker) |
|---|---|---|
| Windows 10/11 64-bit, virtualization enabled in BIOS | ✅ recommended | ✅ **required** |
| A Supabase account (free tier is enough) | ✅ required | only if you use Supabase |
| Docker Desktop 4.x+ (WSL2 backend) | — | ✅ required |
| Your movie file(s) — owned/licensed | for the final recap run | for the final recap run |
| `ollama serve` running locally | for the final recap run | not needed (compose runs Ollama) |

> The two heavy things the pipeline auto‑installs on first run
> (faster‑whisper, sentence‑transformers) happen on whatever machine runs the
> recap: your PC locally, or inside the Docker container.

---

# PART 1 — Supabase pgvector

## 1.1 Create the Supabase project

1. Go to <https://supabase.com> and sign in (GitHub or email).
2. Click **New project**.
3. Fill in:
   - **Organization** — pick or create one.
   - **Name** — e.g. `movie-recap`.
   - **Database password** — click *Generate a password*, **save it somewhere
     safe** (you need it for the optional direct‑connection route and for the
     Supabase SQL editor is not enough — you'll use the dashboard).
   - **Region** — the one closest to you.
4. Click **Create new project** and wait ~2 minutes for provisioning.

## 1.2 Collect your credentials (two values, plus one optional)

In the project dashboard:

1. Go to **Project Settings** (gear icon, bottom left) → **API**.
2. Copy:
   - **Project URL** → this is `SUPABASE_URL`
     (looks like `https://abcdefghijklm.supabase.co`).
   - **`service_role` secret** → this is `SUPABASE_SERVICE_KEY`
     (click *Reveal* → *Copy new*. It starts with `eyJ...`).
     > Use the **service_role** key, **not** the anon/public key. The pipeline
     > clears and re-inserts the matching table each run, which only the
     > service role may do.
3. **Optional** — only for the "auto‑migration" route (1.3 B):
   **Project Settings → Database → Connection string → URI**.
   - Copy the *Direct connection* string and replace `[YOUR-PASSWORD]` with the
     database password from step 1.1.
   - It looks like:
     `postgresql://postgres.abcdefghijklm:[YOUR-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:5432/postgres`
   - If your network blocks non‑pooled connections, enable **IPv4** for the
     project, or simply use Route A below and skip the DB URL.

## 1.3 Create the schema (pgvector table + match_cues RPC)

Open **SQL Editor** (left sidebar) → **New query**, paste the **entire**
contents of this file from the repo, then press **Run**:

```
movie-recap-bot/migrations/001_pgvector.sql
```

Expected result: `Success. No rows returned`.

That file creates, in one shot:

- the `vector` extension,
- the `transcript_cues` table (`idx`, `text`, `start_ms`, `end_ms`,
  `session`, `embedding vector(384)`) with an HNSW cosine index,
- the `match_cues(query_embedding, match_count, match_threshold,
  session_name)` RPC the pipeline calls for every narration line.

> **Why no `json -> vector` casts?** Newer hosted Supabase blocks custom casts
> on built-in types (`ERROR 42501: must be owner of type json or type
> vector`), and they are not needed: the pipeline inserts embeddings as plain
> JSON arrays through the PostgREST API, which accepts them directly for
> vector columns/arguments — the same mechanism supabase-js uses in the
> official Supabase vector docs.

> **Re-running is safe.** It drops and recreates the table. The pipeline
> clears the table at the start of each movie run anyway, so there is never
> data you need to preserve.

### Optional Route B — let the pipeline run the migration itself

If you configured `SUPABASE_DB_URL` (the direct connection string), the code
runs the same `001_pgvector.sql` automatically the first time a Supabase store
is created. That path needs one extra package:

```bat
pip install psycopg2-binary
```

Route A (SQL Editor) is the recommended one and needs **no** extra package.

## 1.4 Point the pipeline at Supabase

In the repo, create the bot's environment file from the template:

```bat
cd Movie-recap\movie-recap-bot
copy .env.example .env
```

Edit `.env` and fill the Supabase block (keep everything else as-is):

```dotenv
# --- Step D: Supabase (pgvector) semantic matching ---
SUPABASE_URL=https://YOURPROJECT.supabase.co
SUPABASE_SERVICE_KEY=eyJ...your-service-role-key...
# Optional: direct connection string -> auto-runs the migration (needs psycopg2-binary)
# SUPABASE_DB_URL=postgresql://postgres.abcdefghijklm:YOURPASSWORD@aws-0-us-east-1.pooler.supabase.com:5432/postgres
```

Notes:

- `.env` is now **git-ignored** — your key will never be committed.
- `semantic.store: auto` in `movie-recap-bot/config.yaml` means:
  Supabase is used automatically **when credentials are present**, otherwise
  the built-in local store is used. You don't have to change the config —
  but if you ever want to force one or the other, set `store: supabase` or
  `store: local`.

## 1.5 Verify the connection (one command)

```bat
cd Movie-recap\movie-recap-bot
pip install numpy
python scripts\verify_supabase.py
```

What it does: loads `.env`, resolves the store (must say Supabase), inserts two
fake rows through the REST API, runs `match_cues()` and checks the right row
comes back, then empties the table again.

Expected output ends with:

```
[ok] RPC matched the right row: idx=1 score=1.000 text='everyone sings at the wedding'
SUPABASE VERIFICATION PASSED — the pipeline will now log
'* Vector store: Supabase (pgvector).' on the next auto-recap run.
```

If it fails, see [Supabase troubleshooting](#supabase-troubleshooting) below.

## 1.6 Run a real recap and confirm it uses Supabase

```bat
python -m recap.cli auto --movie "D:\Movies\my_movie.mp4" --minutes 14 --name my-recap
```

During **Step D** you should see in the log:

```
* Vector store: Supabase (pgvector).
* Embedding NNNN transcript cues ...
* Matching NNN narration lines to the film ...
```

Afterwards, inspect the mapping:

```
output\my-recap\_work\beats.json
```

## 1.7 Tuning the semantic matching (Step D)

The mapping lives in `output\_work\beats.json` after every run — each line
shows the matched cue, timestamps, cosine `score`, and `fallback` flag.

| Knob | Where | Effect |
|---|---|---|
| `semantic.min_score` | `config.yaml` | Lower → more lines get a match (risk: worse matches). Raise → only confident matches are used, others fall back to even spacing. Start 0.10–0.15. |
| `semantic.top_k` | `config.yaml` | Candidates considered per line (de-duplicated). |
| `semantic.pre_roll` / `clip_pad` | `config.yaml` | Footage shown before/after the spoken beat. |
| `semantic.clip.mode` | `config.yaml` | `copy` (fast) vs `reencode` (frame-exact). |
| `semantic.clip.max_clip` | `config.yaml` | Cap per-beat length. |

Useful SQL to inspect what is in the table (SQL Editor):

```sql
-- rows from the last run
select idx, text, start_ms, end_ms, round((embedding <=> (select embedding from transcript_cues limit 1))::numeric, 3)
from transcript_cues order by idx limit 20;

-- total rows / sessions ever
select session, count(*), min(start_ms)/1000.0 as starts_at_s from transcript_cues group by session;
```

---

# PART 2 — Docker

## 2.1 Install Docker Desktop (one time)

1. Download **Docker Desktop for Windows** from <https://www.docker.com/products/docker-desktop/>.
2. Run the installer; keep **"Use WSL 2 instead of Hyper-V"** ticked.
3. If WSL isn't installed yet, the installer will ask — or run in PowerShell
   (admin): `wsl --install`, then reboot.
4. Start Docker Desktop and wait until the whale icon says
   **"Docker Desktop is running"**.
5. Verify in PowerShell:

```powershell
docker --version
docker compose version
```

## 2.2 Prepare the project folders

In the repo root (`Movie-recap\`):

```bat
mkdir movies
```

- Copy the movie you own into `movies\` (e.g. `movies\my_movie.mp4`).
- If you have an `.srt`/`.ass` subtitle for it, drop it next to the movie with
  the same base name (e.g. `movies\my_movie.srt`) — it makes the dialogue step
  instant and skips the Whisper download.
- `output\` and the model cache are created automatically.

> Folders are mounted **read-only** from the host into the containers. You
> edit/add movies on the host; the container reads them at `/movies/...`.

## 2.3 Pre-pull the LLM model into the Ollama container

```powershell
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5
```

You only do this once (the model lives in the `ollama-models` volume). Skipping
it is the #1 cause of "Ollama request failed: model not found".

## 2.4 Build and start everything

```powershell
docker compose up --build -d
docker compose ps          # all three should be Up / healthy
```

First build takes a while (installs Python deps, downloads ffmpeg into the
image). Watch the studio logs if you want to follow along:

```powershell
docker compose logs -f studio
```

## 2.5 Use the web control panel (Recap Studio)

Open <http://localhost:8080>.

The panel runs **inside** the container, so paths are *container* paths:

1. **Settings tab**
   - **Engine** = `semantic` (default).
   - **Movie file path** = `/movies/my_movie.mp4` (use **Browse…** to click
     through the container's file system — it starts at `/`).
   - **Dialogue subtitle** = `/movies/my_movie.srt` (leave blank to auto-find
     next to the movie, else Whisper will transcribe).
   - **Output folder** = leave **blank** (writes into the container's
     `recap-studio/output`, which is mounted to your host `output\` folder) —
     or set it explicitly to `/srv/recap-studio/output`.
   - **Target recap length** = `840` seconds (~14 min).
2. Click **Save settings**.
3. Back on the **EN clip** tab click **Recap movie — EN**.

First run auto-installs faster-whisper + sentence-transformers inside the
container (watch the **Console**; it can take several minutes). Ollama at
`http://ollama:11434` is already wired via the compose environment.

Results on your host:

```
output\recap_en.mp4        output\recap_zh.mp4
output\_work\beats.json    output\_work\script\script_en.json   ...
```

## 2.6 Headless CLI (faster for iterating on one movie)

```powershell
docker compose --profile cli run --rm recap auto --movie /movies/my_movie.mp4 --minutes 14 --name my-recap
```

Output: `output\my-recap_en.mp4`. The `recap` service shares the same `movies`,
`output` and model-cache volumes and points at the compose Ollama.

## 2.7 Supabase from Docker

Compose reads a `.env` file sitting **next to `docker-compose.yml`** (the repo
root). Create `Movie-recap\.env` with:

```dotenv
SUPABASE_URL=https://YOURPROJECT.supabase.co
SUPABASE_SERVICE_KEY=eyJ...your-service-role-key...
# optional: SUPABASE_DB_URL=postgresql://postgres.abcdefghijklm:...@...:5432/postgres
```

Then recreate the containers so they pick up the new environment (no rebuild):

```powershell
docker compose up -d
```

The studio and CLI containers pass these straight to the pipeline, which logs
`* Vector store: Supabase (pgvector).` on the next run.

> Two different `.env` files:
> - `Movie-recap\.env` → read by **docker compose** (repo root).
> - `Movie-recap\movie-recap-bot\.env` → read by **local (non-Docker)** runs.
> Both are git-ignored. Fill the one for how you run.

## 2.8 Panel settings persistence

Settings you save in the panel are written to
`recap-studio\config.docker.json` on your host (the compose file bind-mounts
it into the container). That file ships with container-friendly defaults
(`output_dir` blank, `engine: semantic`, Ollama at `http://ollama:11434/v1`).
You can also edit it by hand while the studio container is stopped.

## 2.9 After code updates

```powershell
docker compose up --build -d
```

## 2.10 Stopping / cleaning

```powershell
docker compose stop        # pause, keep everything
docker compose down        # stop + remove containers, keep volumes (models)
docker compose down -v     # ALSO delete Ollama model + cache volumes (re-download next time)
```

---

## Supabase troubleshooting

| Symptom | Fix |
|---|---|
| `[X] No Supabase credentials found` (verify script) | Fill `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` in `movie-recap-bot\.env` and re-run. |
| `PGRST202 ... could not find function public.match_cues` | You didn't run the migration, or it errored midway. Re-paste `001_pgvector.sql` in the SQL Editor and Run. |
| `column "embedding" is of type vector but expression is of type jsonb` | You are inserting over REST with a non-array JSON value. The pipeline sends real arrays (fine); this error usually means the migration never ran or you are testing with a hand-built request. Re-run `001_pgvector.sql`. |
| `401/403` on insert or RPC | Wrong key. Use the **service_role** key (not anon), copy the whole `eyJ...` value with no spaces. |
| `invalid input syntax for type vector` / dimension errors | All rows must be exactly 384 floats. If you changed `embedding_model` in config, recreate the table at the new dimension (edit the `vector(384)` in the migration to match). |
| RPC returns nothing even though rows exist | Check the `session` filter — the pipeline inserts with `session='default'` and queries the same. If you ever add session tagging, keep them in sync. |
| Direct `SUPABASE_DB_URL` connection fails from home | Supabase "direct" connections can be blocked on some ISPs. Use **Route A** (URL + service key only) — no DB URL needed. |

## Docker troubleshooting

| Symptom | Fix |
|---|---|
| `docker: command not found` | Docker Desktop isn't installed/running. |
| WSL errors at startup | PowerShell (admin): `wsl --install`, reboot, then start Docker Desktop. |
| Port 8080 or 11434 already in use | In `docker-compose.yml` change `"8080:8080"` → `"8090:8080"` (and/or the Ollama port), then `docker compose up -d`. |
| `Movie not found: /movies/...` | The path is case-sensitive and must match the file *exactly including the extension* (Windows Explorer hides extensions — the file may really be `my_movie.mp4` while you typed `my_movie`). |
| `Ollama request failed ... model not found` | Run `docker compose exec ollama ollama pull qwen2.5` (Part 2.3). |
| Run hangs on "Installing faster-whisper / sentence-transformers" | First-run downloads inside the container need internet; give it a few minutes and watch the Console. |
| Out of memory / container killed | Docker Desktop → Settings → Resources → give it 8 GB+ RAM (whisper + a 7B LLM are hungry). |
| Edge TTS produces no audio | edge-tts calls Microsoft's servers — allow outbound HTTPS from the container/host. |
| Panel settings reset | They persist in `recap-studio\config.docker.json` (bound mount). If you used `docker compose down` + an old image, re-save settings after `up`. |

---

## Quick reference — where everything lands

| Artifact | Local run | Docker run (host path) |
|---|---|---|
| Final videos | `output\<name>_<lang>.mp4` | `output\<name>_<lang>.mp4` |
| Semantic mapping | `output\_work\beats.json` | `output\_work\beats.json` |
| Recap sentence array | `output\_work\script\script_en.json` | same |
| Transcript | `output\_work\transcript.json` / `.srt` | same |
| Supabase rows | `public.transcript_cues` | same database |
