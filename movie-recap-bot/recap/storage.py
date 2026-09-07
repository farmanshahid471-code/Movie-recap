"""Storage policy: every byte this app persists goes to configurable folders —
never the OS user-profile / C: drive by default.

Two roots matter (both may be absolute paths, e.g. ``D:\\recap\\output``):

  project.output_dir  (env OUTPUT_DIR)   final .mp4s + per-run _work files
  project.cache_dir   (env CACHE_DIR)    model weights, ffmpeg binaries, scratch

``bootstrap()`` re-points the well-known cache locations used by the optional
heavy dependencies (Whisper model weights, HuggingFace / sentence-transformers
models, static-ffmpeg binaries, Python temp files) at the cache root. Libraries
read these env vars lazily when they first download/use something, so setting
them at startup is enough — nothing needs to run on C:.

Relative roots resolve against the bot folder (the project lives wherever the
user unpacked it — keep that off C: too for a fully C:-free setup).
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

# Root of the bot package (movie-recap-bot/).  Computed locally to avoid a
# circular import with recap.config.
BASE_DIR = Path(__file__).resolve().parent.parent

# Keys this module has injected (as opposed to values the user exported).
_WE_SET: set[str] = set()

_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def is_abs(p: str | Path) -> bool:
    """Absolute-path test that also understands Windows drive letters on any OS."""
    s = str(p)
    return Path(p).is_absolute() or bool(_DRIVE_RE.match(s))


def _apply(envs: dict[str, str]) -> None:
    for key, val in envs.items():
        val = str(val)
        # Only touch keys the user did not export themselves (or keys we set
        # earlier ourselves — config may legitimately change between loads).
        if key not in os.environ or key in _WE_SET:
            os.environ[key] = val
            _WE_SET.add(key)


def env_cache_root() -> Path | None:
    """Cache root explicitly requested via CACHE_DIR / RECAP_CACHE_DIR env."""
    raw = os.environ.get("CACHE_DIR") or os.environ.get("RECAP_CACHE_DIR")
    if raw:
        p = Path(raw).expanduser()
        return p if is_abs(p) else BASE_DIR / p
    return None


def env_output_root() -> Path | None:
    """Output root explicitly requested via OUTPUT_DIR."""
    raw = os.environ.get("OUTPUT_DIR")
    if raw:
        p = Path(raw).expanduser()
        return p if is_abs(p) else BASE_DIR / p
    return None


def cache_envs(cache_root: Path) -> dict[str, str]:
    """Env vars that redirect each library's cache under ``cache_root``."""
    return {
        # HuggingFace hub (sentence-transformers / faster-whisper / whisperx).
        "HF_HOME": str(cache_root / "huggingface"),
        "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
        "SENTENCE_TRANSFORMERS_HOME": str(cache_root / "sentence-transformers"),
        # PyTorch hub.
        "TORCH_HOME": str(cache_root / "torch"),
        # openai-whisper model weights (also read by recap.dialogue directly).
        "WHISPER_CACHE_DIR": str(cache_root / "whisper"),
        # static-ffmpeg downloaded binaries (read by recap.util lazily).
        "STATIC_FFMPEG_CACHE_DIR": str(cache_root / "static-ffmpeg"),
    }


def bootstrap(cache_root: str | Path | None = None) -> Path:
    """Point every cache + scratch location at ``cache_root``.

    With no argument the env vars CACHE_DIR / RECAP_CACHE_DIR decide; otherwise
    the default ``<bot folder>/cache`` is used. Returns the effective root.
    """
    if cache_root is None:
        cache_root = env_cache_root() or BASE_DIR / "cache"
    cache = Path(cache_root).expanduser()
    if not is_abs(cache):
        cache = BASE_DIR / cache
    cache = cache.resolve()

    try:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "tmp").mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # e.g. a locked-down or missing drive
        print(f"! storage: cache root {cache} not usable ({exc}); "
              f"libraries will fall back to their defaults", file=sys.stderr)
        return cache

    _apply(cache_envs(cache))

    # Python scratch (tempfile, subprocess TMP for ffmpeg etc.) off C: too.
    tmp = cache / "tmp"
    _apply({"TMP": str(tmp), "TEMP": str(tmp), "TMPDIR": str(tmp)})
    try:
        tempfile.tempdir = None  # re-resolve next gettempdir() against new TMP
    except Exception:
        pass
    return cache
