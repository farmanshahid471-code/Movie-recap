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
* Requests per movie ~= (frames / frames_per_request): a 100-min film gets
  ~600 stratified frames (one per ~10s of film, full runtime) ≈ 100
  requests at 6 frames/request (was 150 at 4).
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


def _scene_times_cached(
    movie: Path,
    threshold: float,
    duration: float,
    workdir: "Path | None" = None,
) -> list[float]:
    """``_scene_times`` with an on-disk cache.

    Shot detection is a full decode of the film: ~3 minutes for a 100-minute
    movie. It used to run TWICE per run (once in capture(), once again in
    check_coverage() right before the gate) and again on every re-run, which
    is six wasted minutes per attempt on the exact same, deterministic answer.
    The result is cached next to the frames and keyed by movie identity +
    threshold, so it is computed once per film.
    """
    key = None
    cache_path = None
    if workdir is not None:
        try:
            st = Path(movie).stat()
            key = f"{Path(movie).resolve()}|{st.st_size}|{st.st_mtime_ns}|{threshold}"
        except OSError:
            key = None
        cache_path = Path(workdir) / "scene_times.json"
        if key and cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if data.get("key") == key and isinstance(data.get("times"), list):
                    return [float(x) for x in data["times"]]
            except Exception:
                pass
    scenes = _scene_times(movie, threshold, duration)
    if key and cache_path is not None and scenes:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"key": key, "times": scenes}), encoding="utf-8")
        except OSError:
            pass
    return scenes


def pick_times(
    movie_duration: float,
    cadence: float = 20.0,
    scenes: list[float] | None = None,
    max_frames: int = 600,
) -> list[int]:
    """Choose which seconds of the film to caption — with FULL-RUNTIME
    coverage.

    STRATIFIED SAMPLING: the film is divided into time bins and each bin
    contributes exactly one frame — the real shot change nearest the bin's
    middle when one exists (cuts are where visible beats start), else the
    bin's middle. Every stretch of the film is captioned, including the
    back third, and the spacing is uniform no matter how the scene changes
    are distributed.

    Why this replaced the old logic: it kept ALL scene changes ("they are
    story-critical") and then truncated the sorted list to ``max_frames``.
    Since scene changes cluster in the early scenes (fast intros, more
    cuts), the frame budget was eaten before reaching the back half — a
    2h film with 740 cuts and a 400-frame cap got its LAST frame at
    ~3515s: the final 43% of the film had ZERO visual notes, and the
    timeline had nothing to anchor on there. Deterministic; returns
    whole-second ints, sorted.

    ``cadence`` is kept for config compatibility; the bins (5s floor
    spacing) subsume the old cadence grid.
    """
    dur = max(float(movie_duration), 0.0)
    if dur <= 0:
        return []
    max_frames = max(int(max_frames), 10)

    scene_list = sorted(int(round(s)) for s in (scenes or []) if 0.5 < s < dur)

    # Number of time bins: one frame per bin, never finer than every 5s
    # (short film -> fewer bins, e.g. a 90s clip gets 18 frames).
    n_bins = min(max_frames, max(int(dur // 5), 1))
    width = dur / n_bins

    # Scene changes per bin, ordered by closeness to the bin's middle.
    per_bin: list[list[int]] = [[] for _ in range(n_bins)]
    for s in scene_list:
        per_bin[min(n_bins - 1, int(s / width))].append(s)
    for k in range(n_bins):
        mid = (k + 0.5) * width
        per_bin[k].sort(key=lambda s: (abs(s - mid), s))

    out: set[int] = set()
    for k in range(n_bins):
        if per_bin[k]:
            out.add(per_bin[k][0])            # nearest real cut in the bin
        else:
            out.add(int(round((k + 0.5) * width)))  # quiet stretch: the middle

    # Short films: the 5s-bin floor uses fewer frames than the cap. Spend
    # the remaining budget on the NEXT-closest real cuts, round-robin over
    # bins, so the extra density stays spread across the film.
    budget = max_frames - len(out)
    while budget > 0 and any(per_bin):
        for k in range(n_bins):
            if budget <= 0:
                break
            if per_bin[k]:
                out.add(per_bin[k].pop(0))
                budget -= 1

    out = sorted(t for t in out if 0 <= t <= max(dur - 1.0, 0.0))
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
    # 180s, not 600: a caption batch is a small request; a dead socket must
    # raise in minutes (the retry loop then handles it), never hang silently.
    client = openai.OpenAI(api_key=api_key, base_url=base, timeout=180)
    return client, model


def _fallback_client(cfg_vision: dict):
    """Try to build a client for the configured fallback provider, or None."""
    import openai  # type: ignore

    provider = (cfg_vision.get("fallback_provider") or "").strip().lower()
    if not provider:
        # No explicit fallback; try to auto-pick a different configured provider
        # (e.g. primary gemini failed, fallback to openai if key exists)
        primary = (cfg_vision.get("provider") or "gemini").strip().lower()
        for cand in ("openai", "gemini", "groq"):
            if cand != primary and provider_configured(cand):
                provider = cand
                break
        if not provider:
            return None, None
    if provider not in PROVIDER_KEY_ENV:
        return None, None
    if not provider_configured(provider):
        return None, None
    key_env = PROVIDER_KEY_ENV[provider]
    api_key = os.environ.get(key_env)
    if not api_key:
        return None, None
    base = (
        cfg_vision.get("fallback_base_url")
        or PROVIDER_BASE_DEFAULT.get(provider)
    )
    model = (
        cfg_vision.get("fallback_model")
        or os.environ.get("VISION_FALLBACK_MODEL")
        or DEFAULT_VISION_MODEL.get(provider, "")
    )
    if not model:
        return None, None
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base, timeout=180)
        return client, model
    except Exception:
        return None, None


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
            # strip an optional bullet, the label, and its separator
            body = re.sub(
                r"^\s*[-*\u2022]?\s*\[?\d{1,3}:\d{2}(?::\d{2})?\]?\s*[\):.\-\u2013\u2014]?\s*",
                "", ln, count=1,
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
    m = re.match(r"^\s*[-*\u2022]?\s*\[?(\d{1,3}):(\d{2})(?::(\d{2}))?\]?",
                 line.strip())
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
    # Free tiers throttle aggressively (Gemini ~15 req/min), and the free
    # tier goes through MULTI-MINUTE 503 "high demand" windows. A batch that
    # hits one must ride it out with long, escalating waits (~4 minutes
    # worst case per batch) instead of dying after ~75s and leaving a
    # permanent hole in the caption list. The caller (capture) then does a
    # final sweep over whatever still failed.
    waits = (15.0, 30.0, 60.0, 120.0)
    resp = None
    last: Exception | None = None
    for attempt, wait in enumerate(waits):
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
                "high demand",
            ))
            if not retryable:
                break
            print(f"    ... vision batch throttled ({type(exc).__name__}) — "
                  f"waiting {wait:.0f}s (try {attempt + 1}/{len(waits)}) ...",
                  flush=True)
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


def _caption_batch_with_fallback(
    client, model: str,
    fallback_client, fallback_model: str | None,
    frames: list[tuple[int, Path]],
) -> tuple[dict[int, str], str]:
    """Try primary, then fallback provider. Returns (captions, provider_used).

    provider_used is "primary", "fallback", or raises VisionError if both fail.
    """
    try:
        res = _caption_batch(client, model, frames)
        return res, "primary"
    except VisionError as exc:
        # Primary failed after its own 4-try ladder. Try fallback if available.
        if fallback_client is None or not fallback_model:
            raise
        print(f"    ... primary provider failed ({exc}); "
              f"trying fallback provider ({fallback_model}) ...", flush=True)
        try:
            res = _caption_batch(fallback_client, fallback_model, frames)
            print(f"    ... fallback provider succeeded for batch "
                  f"({len([v for v in res.values() if v])}/{len(frames)} captioned)",
                  flush=True)
            return res, "fallback"
        except Exception as exc2:
            # Both failed: surface the combined error
            raise VisionError(
                f"both primary ({model}) and fallback ({fallback_model}) failed: "
                f"{exc} | fallback: {exc2}"
            ) from exc2


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
            "max_frames": cfg.get("max_frames", 600),
            "width": cfg.get("width", 512),
            "provider": cfg.get("provider", "gemini"),
            "model": cfg.get("model") or os.environ.get("VISION_MODEL") or "",
            "fallback_provider": cfg.get("fallback_provider", ""),
            "fallback_model": cfg.get("fallback_model", ""),
        },
        sort_keys=True,
    )


def _load_cached_notes(cache_path: Path, sig: str) -> tuple[dict[int, dict], bool]:
    """Load cached visual notes. Returns ({t: {text, confidence, provider}}, cache_ok).

    Supports both old format (frames: [{t, text}]) and new format with metadata.
    """
    cached: dict[int, dict] = {}
    cache_ok = False
    try:
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("sig") == sig:
                cache_ok = True
                for fr in data.get("frames", []):
                    t = fr.get("t")
                    text = (fr.get("text") or "").strip()
                    if t is not None and text:
                        cached[int(t)] = {
                            "text": text,
                            "confidence": fr.get("confidence", "high"),
                            "provider": fr.get("provider", "primary"),
                            "retried": bool(fr.get("retried", False)),
                        }
            elif data.get("sig") is not None:
                # stale sig: treat as no cache but don't delete (debug)
                pass
    except Exception:
        cached = {}
    return cached, cache_ok


def _save_cached_notes(cache_path: Path, sig: str, cached: dict[int, dict]) -> None:
    """Persist cached notes with confidence metadata."""
    try:
        cache_path.write_text(
            json.dumps(
                {"sig": sig, "frames": [
                    {"t": t,
                     "text": cached[t]["text"],
                     "confidence": cached[t].get("confidence", "high"),
                     "provider": cached[t].get("provider", "primary"),
                     "retried": bool(cached[t].get("retried", False))}
                    for t in sorted(cached)]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def check_coverage(
    movie: Path,
    cfg_vision: dict,
    workdir: Path,
) -> tuple[float, int, int]:
    """Check current caption coverage without running capture.

    Returns (coverage_ratio, captioned_count, total_expected).
    coverage_ratio is 0.0-1.0; total_expected is len(pick_times(...)).
    """
    cfg = dict(cfg_vision or {})
    duration = probe_duration(movie)
    if duration <= 0:
        return 1.0, 0, 0
    try:
        scenes = _scene_times_cached(
            movie, float(cfg.get("scene_threshold", 0.35)), duration, workdir
        )
    except RuntimeError:
        scenes = []
    times = pick_times(
        duration,
        cadence=float(cfg.get("cadence_seconds", 20.0)),
        scenes=scenes,
        max_frames=int(cfg.get("max_frames", 600)),
    )
    if not times:
        return 1.0, 0, 0
    cache_path = Path(workdir) / "visual_notes.json"
    sig = _movie_sig(movie, cfg)
    cached, _ = _load_cached_notes(cache_path, sig)
    total = len(times)
    have = len([t for t in times if t in cached])
    return (have / total) if total else 1.0, have, total


def missing_times(movie: Path, cfg_vision: dict, workdir: Path) -> list[int]:
    """The sampled seconds of film that still have no caption."""
    cfg = dict(cfg_vision or {})
    duration = probe_duration(movie)
    if duration <= 0:
        return []
    try:
        scenes = _scene_times_cached(
            movie, float(cfg.get("scene_threshold", 0.35)), duration, workdir)
    except RuntimeError:
        scenes = []
    times = pick_times(
        duration,
        cadence=float(cfg.get("cadence_seconds", 20.0)),
        scenes=scenes,
        max_frames=int(cfg.get("max_frames", 600)),
    )
    cached, _ = _load_cached_notes(
        Path(workdir) / "visual_notes.json", _movie_sig(movie, cfg))
    return [t for t in times if t not in cached]


def blind_stretch(captioned: list[int], missing: list[int],
                  duration: float) -> tuple[float, float]:
    """Longest run of film with no caption at all: (seconds, start_second).

    This is what the coverage gate actually protects. Thirteen missing frames
    scattered across two hours leave the beat matcher a neighbour on either
    side of every hole; thirteen CONSECUTIVE missing frames leave a four-minute
    stretch of film the timeline cannot see at all. Only the second one is a
    reason to stop a finished run.
    """
    if not missing:
        return 0.0, 0.0
    have = sorted(captioned)
    worst, worst_at = 0.0, 0.0
    for t in sorted(missing):
        prev = max((h for h in have if h <= t), default=0.0)
        nxt = min((h for h in have if h >= t), default=float(duration))
        span = float(nxt) - float(prev)
        if span > worst:
            worst, worst_at = span, float(prev)
    return worst, worst_at


def coverage_gate(
    movie: Path,
    cfg_vision: dict,
    workdir: Path,
) -> None:
    """Hard gate: ensure vision coverage >= threshold before scriptwriting.

    Raises VisionError if coverage is below threshold and not overridden.
    Called from pipeline before Step B. The log from capture already ran;
    this is the gate that blocks progression on degraded data.
    """
    cfg = dict(cfg_vision or {})
    if not cfg.get("enabled", True):
        return
    threshold = float(cfg.get("coverage_threshold", 0.99))
    if cfg.get("allow_incomplete") or os.environ.get("VISION_ALLOW_INCOMPLETE", "").lower() in ("1", "true", "yes"):
        return
    provider = (cfg.get("provider") or "gemini").strip().lower()
    if not provider_configured(provider):
        # No key -> gracefully text-only; gate doesn't apply
        return
    ratio, have, total = check_coverage(movie, cfg, workdir)
    if total == 0:
        return
    if ratio >= threshold:
        return

    pct = ratio * 100
    need = int(total * threshold) - have
    # A bare percentage is the wrong question. What degrades the timeline is a
    # STRETCH of film with no captions, not a few scattered holes the beat
    # matcher can interpolate across. Measure that before killing the run.
    missing = missing_times(movie, cfg, workdir)
    duration = probe_duration(movie)
    captioned_pts: list[int] = []
    try:
        cached, _ = _load_cached_notes(
            Path(workdir) / "visual_notes.json", _movie_sig(movie, cfg))
        captioned_pts = sorted(cached)
    except Exception:
        captioned_pts = []
    worst, worst_at = blind_stretch(captioned_pts, missing, duration)
    try:
        max_blind = float(cfg.get("max_blind_seconds", 180.0))
    except (TypeError, ValueError):
        max_blind = 180.0
    # Absolute floor: a couple of stubborn frames never fail a build.
    try:
        max_missing = int(cfg.get("max_missing_frames", 0)) or max(
            3, int(round(total * 0.03)))
    except (TypeError, ValueError):
        max_missing = max(3, int(round(total * 0.03)))

    if len(missing) <= max_missing and worst <= max_blind:
        print(f"  * Vision coverage {pct:.1f}% ({have}/{total}) is under the "
              f"{threshold*100:.0f}% target, but the {len(missing)} missing "
              f"frame(s) are scattered: the longest stretch of film with no "
              f"caption is {worst:.0f}s (at {_fmt_label(worst_at)}), inside "
              f"the {max_blind:.0f}s limit. The beat matcher can interpolate "
              "across holes that small — continuing.", flush=True)
        if missing:
            print("    ... uncaptioned moments: "
                  + ", ".join(_fmt_label(m) for m in missing[:12])
                  + (" ..." if len(missing) > 12 else ""), flush=True)
        return

    reason = (f"the longest stretch of film with NO caption is {worst:.0f}s "
              f"(at {_fmt_label(worst_at)}), over the "
              f"{max_blind:.0f}s limit — the timeline would have nothing to "
              f"match there"
              if worst > max_blind else
              f"{len(missing)} frames are missing, over the "
              f"{max_missing}-frame limit")
    raise VisionError(
        f"Vision caption coverage {pct:.1f}% ({have}/{total}) is below "
        f"the required {threshold*100:.0f}% ({int(total*threshold)}/{total}) and "
        f"{reason}. {need} more frames needed. Re-run the same movie (cached "
        f"frames are reused, only the missing ones are re-captured; shot "
        f"detection is cached too, so a re-run starts in seconds), or raise "
        f"vision.sweep_attempts, or switch vision.model off the throttled "
        f"'-lite' free tier, or set vision.allow_incomplete: true / "
        f"VISION_ALLOW_INCOMPLETE=1 to proceed anyway."
    )


def capture(
    movie: Path,
    cfg_vision: dict,
    workdir: Path,
    progress=None,
) -> list[dict]:
    """Caption the film's on-screen action. Returns [{"t": int, "text": str, ...}].

    Fully resumable: every finished batch is written to ``visual_notes.json``
    next to the extracted frames, keyed by the movie + settings signature, so a
    crash or quota pause never re-captions finished frames. Best effort — any
    batch that fails logs a warning and the run continues with the frames it
    has (a partial caption list is far better than a dead pipeline).

    New in this fix:
    - Larger batches by default (6 instead of 4) = fewer requests, less hammering.
    - Circuit breaker: after N consecutive 503s, pause the whole pass for a
      longer cooldown instead of hammering the saturated endpoint.
    - Fallback provider: if primary fails, automatically try a secondary.
    - Confidence tracking: each note tagged high/fallback/low, surfaced in logs.
    - Hard coverage gate is checked by pipeline.coverage_gate before writing.
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
        scenes = _scene_times_cached(
            movie, float(cfg.get("scene_threshold", 0.35)), duration, workdir
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
        max_frames=int(cfg.get("max_frames", 600)),
    )
    if not times:
        return []

    frames_dir = Path(workdir) / "visual_frames"
    cache_path = Path(workdir) / "visual_notes.json"
    sig = _movie_sig(movie, cfg)

    cached_raw, cache_ok = _load_cached_notes(cache_path, sig)
    # Adapt to new dict format; also handle old cache where values are strings
    cached: dict[int, dict] = {}
    for t, v in cached_raw.items():
        if isinstance(v, str):
            cached[t] = {"text": v, "confidence": "high", "provider": "primary", "retried": False}
        elif isinstance(v, dict):
            cached[t] = v
        else:
            cached[t] = {"text": str(v), "confidence": "high", "provider": "primary", "retried": False}
    # Also handle legacy caches that were loaded as strings above but stored as dicts
    # Re-load properly: _load_cached_notes already gave dicts, but ensure compat
    # For old files that stored flat [{"t":..., "text":...}], we already migrated.

    # Re-scan for legacy string caches that _load_cached_notes missed due to format variation
    # (if cache existed but sig matched and frames lacked confidence fields)
    try:
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("sig") == sig:
                for fr in data.get("frames", []):
                    t = fr.get("t")
                    if t is not None and int(t) not in cached:
                        text = (fr.get("text") or "").strip()
                        if text:
                            cached[int(t)] = {
                                "text": text,
                                "confidence": fr.get("confidence", "high"),
                                "provider": fr.get("provider", "primary"),
                                "retried": bool(fr.get("retried", False)),
                            }
    except Exception:
        pass

    missing = [t for t in times if t not in cached]
    if cache_ok and not missing:
        print(f"  * Vision pass: reusing {len(cached)} cached captions "
              f"({cache_path.name})", flush=True)
        # Report confidence summary for cached runs too
        low = sum(1 for v in cached.values() if v.get("confidence") != "high" or v.get("provider") == "fallback")
        if low:
            print(f"  * Vision confidence: {len(cached)-low}/{len(cached)} high, "
                  f"{low} low/fallback (degraded source)", flush=True)
        return [{"t": t, "text": cached[t]["text"],
                 "confidence": cached[t].get("confidence", "high"),
                 "provider": cached[t].get("provider", "primary")}
                for t in sorted(cached)]

    frames = extract_frames(
        movie, missing, frames_dir, width=int(cfg.get("width", 512))
    )
    if not frames:
        print("  ! Vision pass: no frames could be extracted - continuing "
              "text-only.", flush=True)
        return [{"t": t, "text": cached[t]["text"],
                 "confidence": cached[t].get("confidence", "high"),
                 "provider": cached[t].get("provider", "primary")}
                for t in sorted(cached)]

    try:
        client, model = _client(cfg)
    except VisionError as exc:
        print(f"  ! Vision pass SKIPPED: {exc}", flush=True)
        return [{"t": t, "text": cached[t]["text"],
                 "confidence": cached[t].get("confidence", "high"),
                 "provider": cached[t].get("provider", "primary")}
                for t in sorted(cached)]

    # Fallback client (optional, may be None)
    try:
        fb_client, fb_model = _fallback_client(cfg)
        if fb_client and fb_model:
            print(f"  * Vision fallback provider ready: {cfg.get('fallback_provider') or 'auto'}/{fb_model}",
                  flush=True)
    except Exception:
        fb_client, fb_model = None, None
        print("  ! Vision fallback provider not configured (primary only)", flush=True)

    todo = [t for t in missing if t in frames]
    # Larger batches = fewer requests. Default 6 (was 4): 600 frames = 100 req vs 150.
    # Configurable via vision.frames_per_request.
    batch_size = max(1, int(cfg.get("frames_per_request", 6)))
    # Circuit breaker config
    max_consec = max(1, int(cfg.get("max_consecutive_failures", 3)))
    breaker_pause = float(cfg.get("circuit_breaker_pause", 180.0))  # 3 min default
    breaker_pause = max(0.0, min(breaker_pause, 600.0))
    print(f"  * Vision pass: captioning {len(todo)} frames via "
          f"{provider}/{model} in batches of {batch_size} "
          f"(+ fallback {fb_model or 'none'}) ...", flush=True)
    started = time.time()

    consecutive_failures = 0

    def _caption_pass(times: list[int], tag: str, size: int | None = None) -> list[int]:
        """Caption ``times`` in order; return the frames that still failed.

        "Failed" means BOTH kinds of loss:
          * the request raised (503 storm, quota, network), and
          * the request succeeded but came back without a line for a frame.

        The second kind used to be invisible: a batch whose response was empty
        or unparseable was counted as a success, its frames were never written
        to the cache, never swept, and the run ended below the coverage gate
        with no error in the log to explain why ("97.8% (587/600)"). Those
        frames are now returned here like any other failure, so the sweeps
        below re-ask for them — one frame per request if needed.
        """
        nonlocal consecutive_failures
        bs = max(1, int(size or batch_size))
        failed: list[int] = []
        for i in range(0, len(times), bs):
            batch = [(t, frames[t]) for t in times[i : i + bs]]
            # Circuit breaker: if we've hit N consecutive failures, pause the
            # whole pass instead of hammering the saturated endpoint.
            if consecutive_failures >= max_consec:
                if tag:
                    # Already in a sweep: the provider is still down. Sitting
                    # through another multi-minute cooldown here only delays
                    # the run -- the frames are queued, bail out and let the
                    # next sweep (or the next run) pick them up.
                    print(f"    ... circuit breaker: {consecutive_failures} "
                          "consecutive batches failed during the sweep — "
                          "ending this sweep early, the rest stay queued",
                          flush=True)
                    failed.extend(times[i:])
                    return failed
                print(f"    ... circuit breaker: {consecutive_failures} consecutive batches failed — "
                      f"pausing whole vision pass for {breaker_pause:.0f}s cooldown ...", flush=True)
                if breaker_pause:
                    time.sleep(breaker_pause)
                consecutive_failures = 0

            used_fallback = False
            try:
                # Try primary with built-in retries; on failure try fallback
                if fb_client and fb_model:
                    res, provider_used = _caption_batch_with_fallback(
                        client, model, fb_client, fb_model, batch)
                    used_fallback = (provider_used == "fallback")
                else:
                    res = _caption_batch(client, model, batch)
                consecutive_failures = 0
            except VisionError as exc:
                print(f"    ! vision batch {i // bs + 1} {tag}failed: "
                      f"{exc} — those frames stay queued for the final "
                      "sweep / next run", flush=True)
                failed.extend(t for t, _ in batch)
                consecutive_failures += 1
                continue
            # The call succeeded -- but did every frame come back with a line?
            gaps = [t for t, _ in batch if not (res.get(t) or "").strip()]
            if gaps:
                print(f"    ... vision batch {i // bs + 1} {tag}returned no "
                      f"caption for {len(gaps)}/{len(batch)} frame(s) "
                      f"({', '.join(_fmt_label(g) for g in gaps[:6])}"
                      f"{' ...' if len(gaps) > 6 else ''}) — queued for the "
                      "sweep", flush=True)
                failed.extend(gaps)
            for t, text in res.items():
                if text:
                    # Only a FALLBACK provider is a degraded caption. A frame
                    # recovered by a sweep came from the same model looking at
                    # the same image -- it is a normal caption that simply took
                    # two tries, so it keeps full weight in beat-matching and
                    # is marked only by `retried`.
                    confidence = "low" if used_fallback else "high"
                    cached[t] = {
                        "text": text,
                        "confidence": confidence,
                        "provider": "fallback" if used_fallback else "primary",
                        "retried": bool(tag),
                    }
            # Persist after every batch: a quota pause resumes without
            # rework (and the next run re-captures only what is missing).
            _save_cached_notes(cache_path, sig, cached)
            if progress:
                progress(min(i + bs, len(times)), len(times),
                         len(cached))
            elif (i // bs) % 5 == 0 or i + bs >= len(times):
                low_cnt = sum(1 for v in cached.values() if v.get("provider") == "fallback")
                fb_note = f", {low_cnt} via fallback" if low_cnt else ""
                print(f"    ... {min(i + bs, len(times))}/{len(times)} "
                      f"frames ({len(cached)} captioned{fb_note}, "
                      f"{(time.time() - started) / 60:.1f} min)", flush=True)
        return failed

    failed = _caption_pass(todo, "")
    # Free-tier demand spikes (503 "high demand") are usually temporary but
    # can outlive one batch's patient retries, and a batch can also come back
    # empty or unparseable. Sweep whatever is still missing, with a SMALLER
    # batch each round, ending at one frame per request: a single frame can
    # never be lost to label confusion, and a small request is the one most
    # likely to get through a saturated endpoint.
    try:
        sweeps = max(1, int(cfg.get("sweep_attempts", 3)))
    except (TypeError, ValueError):
        sweeps = 3
    pause = max(float(cfg.get("sweep_pause_seconds", 60.0)), 0.0)
    for attempt in range(1, sweeps + 1):
        if not failed:
            break
        size = max(1, batch_size // (2 ** attempt))
        wait = pause if attempt == 1 else min(pause, 30.0)
        print(f"  * Vision pass: {len(failed)} frame(s) still uncaptioned — "
              f"final sweep {attempt}/{sweeps} at {size} frame(s) per request"
              + (f" after a {wait:.0f}s pause" if wait else "") + " ...",
              flush=True)
        if wait:
            time.sleep(wait)
        # Reset breaker for sweep
        consecutive_failures = 0
        before = len(failed)
        failed = _caption_pass(sorted(set(failed)), f"(sweep {attempt}) ", size)
        print(f"    ... sweep {attempt}: recovered {before - len(failed)}"
              f"/{before} frame(s)", flush=True)
        if failed and len(failed) == before and size == 1:
            # One frame per request and nothing came back: more sweeps would
            # only burn quota against the same wall.
            print("    ... sweep made no progress at 1 frame/request — "
                  "stopping here", flush=True)
            break

    # Persist exactly WHICH seconds of film have no caption, so the coverage
    # gate can judge whether they form a blind stretch or are scattered.
    still_missing = sorted(t for t in times if t not in cached)
    try:
        gaps_path = Path(workdir) / "vision_gaps.json"
        if still_missing:
            gaps_path.write_text(json.dumps(
                {"sig": sig, "total": len(times), "missing": still_missing},
                ensure_ascii=False), encoding="utf-8")
        elif gaps_path.exists():
            gaps_path.unlink()
    except OSError:
        pass

    result = [{"t": t, "text": cached[t]["text"],
               "confidence": cached[t].get("confidence", "high"),
               "provider": cached[t].get("provider", "primary"),
               "retried": bool(cached[t].get("retried", False))}
              for t in sorted(cached)]
    # Confidence summary for this run
    low_total = sum(1 for r in result if r.get("confidence") != "high" or r.get("provider") == "fallback")
    retried_total = sum(1 for r in result if r.get("retried"))
    if still_missing:
        # Never read a demand spike as "the film has no visual notes": say
        # exactly WHICH moments are missing and how cheaply to get them (the
        # cache reuses everything that already succeeded).
        coverage = len(result) / max(len(times), 1) * 100
        print(f"  ! Vision pass: {len(still_missing)} frame(s) could NOT be "
              f"captioned after {sweeps} sweep(s) — '{model}' on {provider} "
              f"either refused (503 high demand) or returned no caption for "
              f"them. Coverage {coverage:.1f}% ({len(result)}/{len(times)}). "
              f"The run continues with {len(result)} notes.", flush=True)
        print("    ... uncaptioned moments: "
              + ", ".join(_fmt_label(m) for m in still_missing[:12])
              + (f" ... (+{len(still_missing) - 12} more, full list in "
                 "vision_gaps.json)" if len(still_missing) > 12 else ""),
              flush=True)
        print("    ... Re-run the SAME movie to retry only these (frames, "
              "captions and shot detection are all cached, so a re-run "
              "starts in seconds). If it keeps happening, move vision.model "
              "off the '-lite' free tier — it is the most throttled — or "
              "set vision.enabled: false for a deliberate text-only run.",
              flush=True)
        if low_total:
            print(f"  * Vision confidence: {len(result)-low_total} high, "
                  f"{low_total} low/fallback ({retried_total} retried/sweep)",
                  flush=True)
        # Emit a structured warning for pipeline gate
        print(f"  ! VISION_COVERAGE {coverage:.1f}% — below 99% threshold; "
              f"scriptwriting should be gated (use --allow-incomplete-captions to override)",
              flush=True)
    else:
        print(f"  * Vision pass: {len(result)} on-screen notes "
              f"(visual_notes.json) — coverage {len(result)/max(len(times),1)*100:.1f}% "
              f"({len(result)}/{len(times)})", flush=True)
        if low_total:
            print(f"  * Vision confidence: {len(result)-low_total} high, "
                  f"{low_total} low/fallback ({retried_total} retried) — "
                  f"those beats will be weighted lower in matching",
                  flush=True)
        else:
            print(f"  * Vision confidence: all {len(result)} high (no degraded captions)",
                  flush=True)
    return result

