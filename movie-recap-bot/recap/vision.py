"""Step A (pass 1.5) — optional VISUAL pass: caption what is ON SCREEN.

DeepSeek's chat API is text-only, so the narration writer only ever hears
about moments the *dialogue* implies. Silent set-pieces — montages, chases,
sight gags, a character staring at a glowing tablet — are invisible to it.
This pass adds a second, vision-capable model (default: Google Gemini, which
has a free tier and is multimodal) that looks at actual frames of the film and
captions what is visible. The captions are merged into each chunk's beat list
downstream, so visual-only moments get narrated instead of skipped.

Quota / cost notes (see README "Vision pass" for exact numbers)
--------------------------------------------------------------
* Text: captions add a modest number of tokens to the DeepSeek summary
  prompts (a few thousand per movie). No new text cost beyond that.
* Vision: image tokens are billed/limited by the VISION provider, never by
  DeepSeek. Gemini's free tier does not charge for images (rate-limited).
  A paid OpenAI/Gemini key bills per image — cents, at 512px.
* Requests per movie ~= (frames / frames_per_request): a 100-min film at a
  20s cadence is ~300 frames ≈ 75 requests at 4 frames/request.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

from .util import ffmpeg_timeout, probe_duration, run, which_ffmpeg

PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",  # has no vision model; kept for the table
}

PROVIDER_BASE_DEFAULT = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}

DEFAULT_VISION_MODEL = {
    "gemini": "gemini-3.6-flash",   # Gemini Flash models are multimodal
    "openai": "gpt-4o-mini",
    "groq": "llama-3.2-11b-vision-preview",
}


class VisionError(RuntimeError):
    """A captioning request failed (network, quota, model refused the image)."""


def provider_configured(provider: str) -> bool:
    p = (provider or "").strip().lower()
    if p not in PROVIDER_KEY_ENV:
        return False
    return bool(os.environ.get(PROVIDER_KEY_ENV[p]))


def _scene_times(movie: Path, threshold: float, movie_duration: float) -> list[float]:
    """Real shot-change boundaries via ffmpeg's scene filter.

    Returns seconds where a frame differs from the previous by more than
    ``threshold`` (0..1; ~0.35-0.45 is a good "real cut" value). Best effort:
    any failure yields [] and the caller falls back to pure cadence sampling.
    """
    try:
        res = run(
            [
                which_ffmpeg(), "-hide_banner",
                "-i", str(movie),
                "-vf", f"select='gt(scene,{threshold})',showinfo",
                "-an", "-f", "null", "-",
            ],
            check=False,
            timeout=ffmpeg_timeout(movie_duration, minimum=600.0),
        )
    except RuntimeError:
        return []
    out = []
    for m in re.finditer(r"pts_time:([0-9.]+)", res.stderr or ""):
        try:
            t = float(m.group(1))
        except ValueError:
            continue
        if 0.5 < t < movie_duration:
            out.append(t)
    return out


def pick_times(
    movie_duration: float,
    cadence: float = 20.0,
    scenes: list[float] | None = None,
    max_frames: int = 400,
) -> list[int]:
    """Choose which seconds of the film to caption.

    Guarantees real shot changes are kept (they are where a visible story
    beat starts) and fills the gaps on a ``cadence`` grid so a quiet stretch
    never goes uncaptioned. Deterministic; returns whole-second ints, sorted.
    """
    dur = max(float(movie_duration), 0.0)
    if dur <= 0:
        return []
    cadence = max(float(cadence), 5.0)
    max_frames = max(int(max_frames), 10)

    scene_set = {int(round(s)) for s in (scenes or []) if 0.5 < s < dur}
    grid = set()
    t = cadence
    while t < dur:
        grid.add(int(round(t)))
        t += cadence
    # Scene changes are story-critical: always keep them.
    keep = sorted(scene_set | {int(round(dur / 2))})
    # Interior grid samples fill between them.
    rest = sorted(grid - scene_set)
    n_rest = max(0, max_frames - len(keep))
    if len(rest) > n_rest and n_rest > 0:
        step = len(rest) / float(n_rest)
        rest = [rest[int(i * step)] for i in range(n_rest)]
    elif len(rest) > n_rest and n_rest <= 0:
        rest = []
    out = sorted(set(keep) | set(rest))
    # Clamp to the film (and never caption the very first frame of a 2h movie
    # if it is studio logos/black).
    out = [t for t in out if 0 <= t <= max(dur - 1.0, 0.0)]
    return out[:max_frames]


def _fmt_label(seconds: float) -> str:
    s = max(int(seconds), 0)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def extract_frames(
    movie: Path,
    times: list[int],
    outdir: Path,
    width: int = 512,
) -> dict[int, Path]:
    """Extract one JPEG per time (skip ones already on disk). Returns t->path."""
    outdir.mkdir(parents=True, exist_ok=True)
    found: dict[int, Path] = {}
    for t in times:
        out = outdir / f"frame_{t:06d}.jpg"
        if out.exists() and out.stat().st_size > 0:
            found[t] = out
            continue
        run(
            [
                which_ffmpeg(), "-y", "-ss", str(float(t)),
                "-i", str(movie), "-frames:v", "1",
                "-vf", f"scale={int(width)}:-2",
                "-q:v", "4",
                str(out),
            ],
            check=False,
            timeout=ffmpeg_timeout(0.0, minimum=120.0),
        )
        if out.exists() and out.stat().st_size > 0:
            found[t] = out
    return found


def _client(cfg_vision: dict):
    """OpenAI-compatible chat client for the configured vision provider.

    Gemini is reached through Google's OpenAI-compatible endpoint (same as the
    text path in llm.py) so one free key drives vision too, with no new
    dependency. Any provider on that endpoint (OpenAI, Groq, ...) works.
    """
    import openai  # type: ignore

    provider = (cfg_vision.get("provider") or "gemini").strip().lower()
    if provider not in PROVIDER_KEY_ENV:
        raise VisionError(
            f"Unknown vision provider {provider!r}. Use gemini (default, free), "
            "openai or groq."
        )
    key_env = PROVIDER_KEY_ENV[provider]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise VisionError(
            f"Vision provider {provider} needs a key: set {key_env} in "
            "movie-recap-bot/.env (Gemini free key: aistudio.google.com -> "
            "Get API key). Set vision.enabled: false to skip the visual pass."
        )
    base = (
        cfg_vision.get("base_url")
        or os.environ.get("GEMINI_BASE_URL" if provider == "gemini" else "")
        or PROVIDER_BASE_DEFAULT.get(provider)
    )
    model = (
        cfg_vision.get("model")
        or os.environ.get("VISION_MODEL")
        or DEFAULT_VISION_MODEL[provider]
    )
    client = openai.OpenAI(api_key=api_key, base_url=base, timeout=600)
    return client, model


def _parse_caption_batch(
    raw: str, frames: list[tuple[int, Path]]
) -> dict[int, str]:
    """Tolerantly map the model's labelled lines back onto frame times."""
    out: dict[int, str] = {t: "" for t, _ in frames}
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        return out

    # Label matches first: "[MM:SS] text", "MM:SS - text", "HH:MM:SS text"...
    matched: dict[int, str] = {}
    unmatched: list[str] = []
    for ln in lines:
        tl = _label_time(ln)
        body = ""
        if tl is not None:
            body = re.sub(
                r"^\[?\d{1,3}:\d{2}(?::\d{2})?\]?\s*[\):.\-]?\s*", "", ln, count=1
            ).strip()
        if tl is not None and body and any(abs(tl - ft) <= 1 for ft, _ in frames):
            matched[tl] = body
        else:
            unmatched.append(ln)

    if len(matched) == len(frames):
        return {t: matched.get(t, "") for t, _ in frames}

    # Some lines lacked/renumbered labels: assign the leftovers positionally
    # to the frames that did not match.
    remaining = [t for t, _ in frames if t not in matched]
    idx = 0
    for t in remaining:
        while idx < len(unmatched) and not unmatched[idx].strip():
            idx += 1
        if idx < len(unmatched):
            out[t] = unmatched[idx].strip()
            idx += 1
    for t, text in matched.items():
        out[t] = text
    return out


def _label_time(line: str) -> int | None:
    """Parse a caption line's leading label into seconds.

    Two forms are accepted, mirroring _fmt_label(): 'MM:SS' (under an hour)
    and 'HH:MM:SS'. A third colon group means hours; otherwise the label is
    minutes:seconds (never hours:minutes).
    """
    m = re.match(r"^\[?(\d{1,3}):(\d{2})(?::(\d{2}))?\]?", line.strip())
    if not m:
        return None
    try:
        if m.group(3) is not None:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        return int(m.group(1)) * 60 + int(m.group(2))
    except ValueError:
        return None


CAPTION_SYSTEM = (
    "You are captioning frames of a movie for a recap narrator who cannot see "
    "the film. For each labelled frame, describe what is VISIBLE and worth "
    "narrating: which characters are on screen (use their names if you can "
    "tell), what they are doing, the setting, and any notable object or "
    "on-screen text. Present tense, third person, one line per frame, under "
    "25 words. Do not invent dialogue. Never mention the frame or the image."
)


def _caption_batch(
    client, model: str, frames: list[tuple[int, Path]]
) -> dict[int, str]:
    """Send one batch of frames in a single request; return t -> caption."""
    parts: list[dict] = [
        {
            "type": "text",
            "text": "Here are movie frames with their film-time labels:\n"
            + "\n".join(f"- {_fmt_label(t)}" for t, _ in frames),
        }
    ]
    for t, path in frames:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    parts.append(
        {
            "type": "text",
            "text": "Now describe each frame. Reply with one line per frame, "
            "each starting with its label like '" + _fmt_label(frames[0][0]) + " - text'.",
        }
    )
    # Free tiers throttle aggressively (e.g. Gemini ~15 req/min); a batch
    # that hits the rate cap should wait and retry, not die or leave a hole in
    # the caption list. 429 / 5xx / network blips get a few backoff attempts.
    import time

    resp = None
    last: Exception | None = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": CAPTION_SYSTEM},
                          {"role": "user", "content": parts}],
                max_tokens=2048,
            )
            break
        except Exception as exc:  # network / quota / model refusal
            last = exc
            msg = str(exc).lower()
            retryable = any(k in msg for k in (
                "429", "rate", "quota", "rpm", "too many", "resource exhausted",
                "502", "503", "504", "500", "timeout", "timed out", "connection",
            ))
            if not retryable or attempt == 4:
                break
            wait = 5.0 * (2 ** attempt)          # 5s, 10s, 20s, 40s
            print(f"    ... vision batch rate-limited ({type(exc).__name__}) — "
                  f"waiting {wait:.0f}s and retrying ...", flush=True)
            time.sleep(wait)
    if resp is None:
        raise VisionError(
            f"vision request failed ({model}): {last}"
        ) from last
    raw = ""
    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""
    return _parse_caption_batch(raw, frames)


def _movie_sig(movie: Path, cfg: dict) -> str:
    try:
        st = movie.stat()
        ident = [str(movie.resolve()), st.st_size, st.st_mtime_ns]
    except OSError:
        ident = [str(movie.resolve()), 0, 0]
    return json.dumps(
        {
            "movie": ident,
            "cadence": cfg.get("cadence_seconds", 20.0),
            "scene_threshold": cfg.get("scene_threshold", 0.35),
            "max_frames": cfg.get("max_frames", 400),
            "width": cfg.get("width", 512),
            "provider": cfg.get("provider", "gemini"),
            "model": cfg.get("model") or os.environ.get("VISION_MODEL") or "",
        },
        sort_keys=True,
    )


def capture(
    movie: Path,
    cfg_vision: dict,
    workdir: Path,
    progress=None,
) -> list[dict]:
    """Caption the film's on-screen action. Returns [{"t": int, "text": str}].

    Fully resumable: every finished batch is written to ``visual_notes.json``
    next to the extracted frames, keyed by the movie + settings signature, so a
    crash or quota pause never re-captions finished frames. Best effort — any
    batch that fails logs a warning and the run continues with the frames it
    has (a partial caption list is far better than a dead pipeline).
    """
    movie = Path(movie)
    cfg = dict(cfg_vision or {})
    provider = (cfg.get("provider") or "gemini").strip().lower()
    if not provider_configured(provider):
        print(
            "  * Vision pass SKIPPED: no key for provider "
            f"{provider!r} (set {PROVIDER_KEY_ENV.get(provider, '?')} in "
            "movie-recap-bot/.env to enable). Continuing text-only.",
            flush=True,
        )
        return []

    duration = probe_duration(movie)
    if duration <= 0:
        return []

    print(f"  * Vision pass: sampling on-screen action (provider {provider}) ...",
          flush=True)
    try:
        scenes = _scene_times(
            movie, float(cfg.get("scene_threshold", 0.35)), duration
        )
        if scenes:
            print(f"    ... detected {len(scenes)} shot changes "
                  f"({len(scenes) / max(duration, 1) * 60:.1f}/min)", flush=True)
    except RuntimeError:
        scenes = []

    times = pick_times(
        duration,
        cadence=float(cfg.get("cadence_seconds", 20.0)),
        scenes=scenes,
        max_frames=int(cfg.get("max_frames", 400)),
    )
    if not times:
        return []

    frames_dir = Path(workdir) / "visual_frames"
    cache_path = Path(workdir) / "visual_notes.json"
    sig = _movie_sig(movie, cfg)

    cached: dict[int, str] = {}
    cache_ok = False
    try:
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("sig") == sig:
                cache_ok = True
                for fr in data.get("frames", []):
                    t, text = fr.get("t"), (fr.get("text") or "").strip()
                    if t is not None and text:
                        cached[int(t)] = text
    except Exception:
        cached = {}

    missing = [t for t in times if t not in cached]
    if cache_ok and not missing:
        print(f"  * Vision pass: reusing {len(cached)} cached captions "
              f"({cache_path.name})", flush=True)
        return [{"t": t, "text": cached[t]} for t in sorted(cached)]

    frames = extract_frames(
        movie, missing, frames_dir, width=int(cfg.get("width", 512))
    )
    if not frames:
        print("  ! Vision pass: no frames could be extracted - continuing "
              "text-only.", flush=True)
        return [{"t": t, "text": cached[t]} for t in sorted(cached)]

    try:
        client, model = _client(cfg)
    except VisionError as exc:
        print(f"  ! Vision pass SKIPPED: {exc}", flush=True)
        return [{"t": t, "text": cached[t]} for t in sorted(cached)]

    todo = [t for t in missing if t in frames]
    batch_size = max(1, int(cfg.get("frames_per_request", 4)))
    print(f"  * Vision pass: captioning {len(todo)} frames via "
          f"{provider}/{model} in batches of {batch_size} ...", flush=True)
    started = time.time()
    done_count = len(cached)
    for i in range(0, len(todo), batch_size):
        batch = [(t, frames[t]) for t in todo[i : i + batch_size]]
        try:
            res = _caption_batch(client, model, batch)
        except VisionError as exc:
            print(f"    ! vision batch {i // batch_size + 1} failed: {exc} — "
                  "keeping what succeeded so far", flush=True)
            continue
        for t, text in res.items():
            if text:
                cached[t] = text
                done_count += 1
        # Persist after every batch: a quota pause resumes without rework.
        try:
            cache_path.write_text(
                json.dumps(
                    {"sig": sig, "frames": [
                        {"t": t, "text": cached[t]} for t in sorted(cached)]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        if progress:
            progress(min(i + batch_size, len(todo)), len(todo), done_count)
        elif (i // batch_size) % 5 == 0 or i + batch_size >= len(todo):
            print(f"    ... {min(i + batch_size, len(todo))}/{len(todo)} frames "
                  f"({done_count} captioned, {(time.time() - started) / 60:.1f} min)",
                  flush=True)

    result = [{"t": t, "text": cached[t]} for t in sorted(cached)]
    print(f"  * Vision pass: {len(result)} on-screen notes "
          f"(visual_notes.json)", flush=True)
    return result
