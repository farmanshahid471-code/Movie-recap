"""Engine glue for the Recap Studio control panel.

This module is *not* a server — it drives the self-contained ``recap`` pipeline
(edge-tts narration + burned-in subtitles + ffmpeg assembly) and keeps the state
the UI needs: persisted settings, a log ring buffer, job status and the output
list. The HTTP layer lives in ``app.py`` (``python recap-studio/app.py``).

Design notes:
  * The pipeline prints progress with ``print()``. We tee stdout into a ring
    buffer so the panel's Console shows the real pipeline log, live.
  * Everything the panel edits (movie path, voices, LLM, scripts) is applied to
    the recap config here — one place, so the UI and the CLI can't drift.
  * The recap pipeline lives in ../movie-recap-bot/recap; we add it to sys.path.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import shutil
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent.parent / "movie-recap-bot"
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

STUDIO_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = STUDIO_DIR / "output"
# Kept as the fallback location. Use output_dir() — it honours the configured
# folder and falls back here if that folder cannot be created/written.
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
INPUTS = BOT_DIR / "inputs" / "text"

VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".wmv",
    ".flv", ".ts", ".m2ts", ".mpg", ".mpeg", ".mxf",
}


class MoviePathError(RuntimeError):
    """The configured movie path cannot be used (missing, a folder, unreadable)."""


# Captured before any redirection so the tee can still echo to the real console.
_REAL_STDOUT = sys.stdout

# --- config that persists between page loads ---
CONFIG_PATH = STUDIO_DIR / "config.json"
DEFAULT_CONFIG = {
    # "semantic" = Step A-F engine (whisper -> chunks -> summaries -> JSON script
    #              -> TTS -> semantic timestamp matching -> real-film clipping).
    # "recap"    = legacy 5-step engine (script file/LLM -> montage -> mux).
    "engine": "semantic",
    "movie_path": "",                       # required for the semantic engine
    "output_dir": r"D:\recap",              # where the clips + _work go; "" = recap-studio/output
    "storyboard": True,                     # legacy engine only: placeholder scenes
    "duration": 840,                        # target recap length in seconds (~14 min full-length)
    "montage": "scenes",                    # legacy engine only
    "scene_len": 6.0,                       # seconds per beat
    "voice_en": "en-US-ChristopherNeural",
    "voice_zh": "zh-CN-YunxiNeural",
    "subtitle_lang_en": "en",
    "subtitle_lang_zh": "zh",
    # LLM — Ollama + Qwen (free, local, no key). Used to auto-write the recap
    # script from your movie's dialogue once a key-free local server is up.
    "llm_provider": "ollama",
    "llm_base_url": "http://localhost:11434/v1",
    "llm_api_key": "",               # not needed for Ollama
    "llm_model": "qwen2.5",
    # Auto-recap flags
    "auto": False,                   # write the recap from the movie's dialogue (needs LLM)
    "auto_subtitle": "",             # optional explicit .srt for dialogue; blank = auto-detect
    "whisper_model": "small",        # used if no subtitle is found
    "whisper_device": "auto",        # auto = GPU when available, else CPU
}

LOCK = threading.Lock()
LOG_BUFFER: list[str] = []
MAX_LOG_LINES = 600
JOB_STATE: dict = {
    "running": False,
    "langs": [],
    "started": 0,
    "finished": 0,
    "error": "",
    "cancel": False,
    "runs": 0,
}


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
# Everything is ALSO written to this file so a run can be inspected even when
# the browser Console or the launcher window is unhelpful.
STUDIO_LOG = STUDIO_DIR / "studio.log"


def _log(msg: str) -> None:
    """Append one line to the shared ring buffer the UI streams (+ a log file)."""
    with LOCK:
        LOG_BUFFER.append(msg)
        if len(LOG_BUFFER) > MAX_LOG_LINES:
            del LOG_BUFFER[: len(LOG_BUFFER) - MAX_LOG_LINES]
    try:
        with open(STUDIO_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except OSError:
        pass


class _Tee(io.TextIOBase):
    """File-like that echoes to the real console *and* the UI log buffer.

    The pipeline logs with ``print()``, so wrapping stdout with this is what
    makes the Console panel show real pipeline output instead of nothing.
    """

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]
        self._partial = ""

    def write(self, s: str) -> int:
        for st in self._streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        self._partial += s
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            _log(line.rstrip())
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False


def log_tail(n: int = 200) -> list[str]:
    with LOCK:
        return LOG_BUFFER[-n:]


# --------------------------------------------------------------------------
# Environment probes
# --------------------------------------------------------------------------
_WHICH_CACHE: dict[str, tuple[float, str | None]] = {}
_WHICH_NEGATIVE_TTL = 30.0  # re-probe a missing binary at most every 30s


def which(name: str) -> str | None:
    """Resolve an executable, including static-ffmpeg's bundled binaries.

    ``static_ffmpeg.add_paths()`` downloads its binaries on first use, so the
    panel must not call it on every status poll — a found path is cached for
    good, a missing one for 30s (long enough to stop hammering the network,
    short enough to notice a fresh install).
    """
    hit = _WHICH_CACHE.get(name)
    if hit:
        when, path = hit
        if path or (time.time() - when) < _WHICH_NEGATIVE_TTL:
            return path

    p = shutil.which(name)
    if not p:
        try:
            import static_ffmpeg  # type: ignore

            static_ffmpeg.add_paths()
            p = shutil.which(name)
        except Exception:
            p = None
    _WHICH_CACHE[name] = (time.time(), p)
    return p


def have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


# --------------------------------------------------------------------------
# Persisted settings
# --------------------------------------------------------------------------
def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception as exc:
            _log(f"! config.json unreadable ({exc}); using defaults")
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def overall_cfg() -> dict:
    return load_config()


_OUTPUT_CACHE: dict[str, Path] = {}

# Paths whose hidden-extension resolution we have already announced, so the
# status poll doesn't reprint the same line every couple of seconds.
_RESOLVE_LOGGED: set[str] = set()


def output_dir(cfg: dict | None = None) -> Path:
    """Folder that holds the rendered clips and the ``_work`` intermediates.

    Configurable from the panel (`output_dir`, e.g. ``D:\\recap``). It is created
    on demand and write-tested once per distinct value; if it cannot be used the
    run falls back to ``recap-studio/output`` with a warning in the log rather
    than dying halfway through a render.
    """
    if cfg is None:
        cfg = load_config()
    raw = (cfg.get("output_dir") or "").strip().strip('"')
    cached = _OUTPUT_CACHE.get(raw)
    if cached is not None:
        return cached

    resolved = DEFAULT_OUTPUT_DIR
    if raw:
        # "D:\recap" is a perfectly legal *filename* on Linux/macOS, so without
        # this check it silently creates a folder named "D:\recap" instead of
        # telling you the path makes no sense on this OS.
        if re.match(r"^[A-Za-z]:[\\/]", raw) and os.name != "nt":
            _log(
                f"! output folder {raw} is a Windows path but this is not Windows "
                f"({sys.platform}); falling back to {DEFAULT_OUTPUT_DIR}"
            )
            raw = ""
    if raw:
        target = Path(raw).expanduser()
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".recap_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            resolved = target
        except Exception as exc:
            _log(
                f"! output folder {target} is not usable ({type(exc).__name__}: {exc}); "
                f"falling back to {DEFAULT_OUTPUT_DIR}"
            )
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except Exception:
        resolved = DEFAULT_OUTPUT_DIR
        resolved.mkdir(parents=True, exist_ok=True)
    _OUTPUT_CACHE[raw] = resolved
    return resolved


def _resolve_hidden_extension(p: Path) -> Path | None:
    """Resolve a path typed without its (hidden) file extension.

    Windows Explorer hides known extensions by default, so a copy-pasted name
    like ``ToyStory5.2026.HDRip`` is really ``ToyStory5.2026.HDRip.mp4``. Try the
    typed name plus each video extension before giving up.
    """
    parent = p.parent
    if not parent.is_dir():
        return None
    for ext in sorted(VIDEO_EXTS):
        cand = parent / (p.name + ext)
        if cand.is_file():
            return cand
    # also match files whose full name starts with the typed name plus a dot
    for cand in sorted(parent.iterdir()):
        if (
            cand.is_file()
            and cand.suffix.lower() in VIDEO_EXTS
            and cand.name.startswith(p.name + ".")
        ):
            return cand
    return None


def _not_found_message(raw: str, p: Path) -> str:
    parent = p.parent
    hint = ""
    if parent.is_dir():
        vids = sorted(
            x.name for x in parent.iterdir()
            if x.is_file() and x.suffix.lower() in VIDEO_EXTS
        )
        if vids:
            hint = (
                f" The folder {parent} holds these video files: "
                f"{', '.join(vids[:6])}{' ...' if len(vids) > 6 else ''}."
            )
        else:
            hint = (
                " Note: Windows hides file extensions by default — if the file "
                "is really there, its true name may end in .mp4/.mkv/.avi."
            )
    return f"Movie not found: {raw}.{hint}"


def check_movie(path: str) -> tuple[Path | None, str]:
    """Validate the movie path. Returns ``(resolved_file, error_message)``.

    A folder is a very common mistake — ffmpeg reports it as a confusing
    "Permission denied", so we catch it here and say what to do instead. If the
    folder holds exactly one video file we use that; if it holds several we list
    them and ask you to pick.
    """
    raw = (path or "").strip().strip('"')
    if not raw:
        return None, ""

    p = Path(raw).expanduser()
    if not p.exists():
        resolved = _resolve_hidden_extension(p)
        if resolved is not None:
            # The panel polls /api/status (which validates the movie) every couple
            # of seconds; log the resolution once, not once per poll.
            if raw not in _RESOLVE_LOGGED:
                _RESOLVE_LOGGED.add(raw)
                _log(f"    {raw} has no extension as typed -> using {resolved.name}")
            p = resolved
        else:
            return None, _not_found_message(raw, p)

    if p.is_dir():
        vids = sorted(
            x for x in p.iterdir()
            if x.is_file() and x.suffix.lower() in VIDEO_EXTS
        )
        if not vids:
            return None, (
                f"{raw} is a FOLDER, and it contains no video files. "
                f"Set Movie file path to a video FILE such as "
                f"{raw}{os.sep}movie.mp4"
            )
        if len(vids) > 1:
            shown = ", ".join(v.name for v in vids[:6])
            return None, (
                f"{raw} is a FOLDER containing {len(vids)} video files "
                f"({shown}{' ...' if len(vids) > 6 else ''}). Recap one film at a "
                f"time: set Movie file path to {raw}{os.sep}{vids[0].name}"
            )
        _log(f"    {raw} is a folder -> using {vids[0].name}")
        p = vids[0]

    if not p.is_file():
        return None, f"Not a readable file: {raw}"
    try:
        size = p.stat().st_size
    except OSError as exc:
        return None, f"Cannot read {raw}: {exc}"
    if size == 0:
        return None, f"{raw} is empty (0 bytes)"
    return p, ""


def llm_ready(cfg: dict) -> bool:
    """Ollama is key-free and configured; other providers need a key."""
    prov = (cfg.get("llm_provider") or "none").lower()
    if prov == "ollama":
        return True  # no key required; reachability is checked separately
    if prov == "none":
        return False
    return bool(cfg.get("llm_api_key"))


_OLLAMA_PROBE: dict[str, tuple[float, bool]] = {}


def ollama_up(cfg: dict) -> bool:
    """True when an Ollama server actually answers at the configured base URL.

    ``llm_ready`` only says Ollama is *configured* (it needs no key); this checks
    it is *running*, so the panel can warn before a run dies on a connection
    error. Probing localhost is instant when up and refuses instantly when down;
    the result is cached a few seconds so the status poll stays cheap.
    """
    if (cfg.get("llm_provider") or "").lower() != "ollama":
        return True  # other providers need a key, not a local server
    base = (cfg.get("llm_base_url") or "http://localhost:11434/v1").rstrip("/")
    now = time.time()
    cached = _OLLAMA_PROBE.get(base)
    if cached and now - cached[0] < 5:
        return cached[1]
    ok = False
    try:
        import urllib.request

        with urllib.request.urlopen(base + "/models", timeout=1.5) as resp:
            ok = 200 <= resp.status < 300
    except Exception:
        ok = False
    _OLLAMA_PROBE[base] = (now, ok)
    return ok


# --------------------------------------------------------------------------
# Recap config assembly (the single place UI settings reach the pipeline)
# --------------------------------------------------------------------------
def engine_name(cfg: dict | None = None) -> str:
    """Which pipeline engine the panel is set to drive: semantic | recap."""
    cfg = cfg if cfg is not None else load_config()
    return str(cfg.get("engine") or "semantic").strip().lower() or "semantic"


def _base_recap(cfg: dict | None = None) -> dict:
    """Load movie-recap-bot/config.yaml and point its output at the studio folder."""
    from recap.config import load_config as recap_load_config

    rc = recap_load_config()
    out = output_dir(cfg)
    rc["project"]["_out"] = out
    rc["project"]["name"] = "recap"
    rc["project"]["output_dir"] = str(out)
    return rc


def _apply_common(rc: dict, cfg: dict, lang: str | None = None) -> None:
    """Push the panel settings that both engines share into the recap config."""
    rc["narration"]["lang_voice"]["en"] = cfg.get("voice_en") or "en-US-ChristopherNeural"
    rc["narration"]["lang_voice"]["zh"] = cfg.get("voice_zh") or "zh-CN-YunxiNeural"
    rc["narration"]["rate"] = "+0%"

    # Target clip length only shapes the narration length; edge-tts narrates
    # roughly 2.5 words/second.
    try:
        secs = int(cfg.get("duration") or 0)
    except (TypeError, ValueError):
        secs = 0
    if secs > 0:
        words = min(max(int(secs * 2.5), 120), int(rc["narration"].get("words_max", 4200)))
        rc["narration"]["words_target"] = words

    if lang is not None:
        rc["subtitles"]["display_lang"] = cfg.get(f"subtitle_lang_{lang}", lang) or lang

    # Legacy montage knobs (the semantic engine ignores these; it cuts real
    # beats via semantic matching).
    montage = str(cfg.get("montage") or "scenes").strip().lower()
    rc["video"]["montage"] = montage if montage in ("scenes", "continuous") else "scenes"
    try:
        rc["video"]["scene_len"] = float(cfg.get("scene_len") or 6.0)
    except (TypeError, ValueError):
        rc["video"]["scene_len"] = 6.0

    _apply_llm(rc, cfg)


def recap_cfg(cfg: dict, lang: str) -> dict:
    """Legacy engine: build the recap config for a single-language run."""
    lang = "zh" if lang.startswith("zh") else "en"
    rc = _base_recap(cfg)

    rc["language"]["target_languages"] = [lang]
    tag = rc["language"].get("zh_variant", "zh-CN") if lang.startswith("zh") else lang
    rc["language"]["_resolved"] = [{"code": lang, "tag": tag}]

    _apply_common(rc, cfg, lang)
    return rc


def semantic_recap_cfg(cfg: dict, langs: list[str]) -> dict:
    """Step A-F engine: full recap config for the requested language set.

    One run covers every requested language (EN master, ZH line-aligned), so
    the whisper extraction, chunking and semantic matching happen exactly once.
    """
    langs = [l for l in langs if l in ("en", "zh")] or ["en"]
    rc = _base_recap(cfg)

    rc["language"]["target_languages"] = list(langs)
    resolved = []
    for code in langs:
        tag = rc["language"].get("zh_variant", "zh-CN") if code.startswith("zh") else code
        resolved.append({"code": code, "tag": tag})
    rc["language"]["_resolved"] = resolved

    _apply_common(rc, cfg, langs[0])
    rc["semantic"]["enabled"] = True
    return rc


_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Sensible model when the provider changes and nothing better is configured.
PROVIDER_PRESETS = {
    "ollama": {"model": "qwen2.5", "base_url": "http://localhost:11434/v1", "key": False},
    "deepseek": {"model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1", "key": True},
    "openai": {"model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1", "key": True},
    "anthropic": {"model": "claude-3-5-sonnet-latest", "base_url": "", "key": True},
    "none": {"model": "", "base_url": "", "key": False},
}


def _apply_llm(rc: dict, cfg: dict) -> None:
    """Push the panel's LLM settings into the recap config + environment.

    recap.llm reads provider/base-url/model from the config *and* from env vars
    (OLLAMA_BASE_URL, DEEPSEEK_API_KEY, ...), so both have to be set here or the
    Settings tab would be a no-op.
    """
    provider = (cfg.get("llm_provider") or "ollama").strip().lower()
    model = cfg.get("llm_model") or PROVIDER_PRESETS.get(provider, {}).get("model", "")
    base = cfg.get("llm_base_url") or ""
    key = (cfg.get("llm_api_key") or "").strip()

    rc.setdefault("llm", {})
    rc["llm"]["provider"] = provider
    rc["llm"]["model"] = model
    if base:
        rc["llm"]["base_url"] = base

    os.environ["LLM_PROVIDER"] = provider
    if provider == "ollama":
        os.environ["OLLAMA_BASE_URL"] = base or "http://localhost:11434/v1"
    elif base:
        # recap.llm now takes base_url from the config, but keep the env var in
        # sync so a direct `python -m recap.cli` run behaves the same.
        os.environ[f"{provider.upper()}_BASE_URL"] = base
    if model:
        os.environ["MODEL_NAME"] = model
    if key:
        os.environ["LLM_API_KEY"] = key
        var = _KEY_ENV.get(provider)
        if var:
            os.environ[var] = key  # authoritative for the selected provider


# --------------------------------------------------------------------------
# Scripts (what the Script editor reads/writes)
# --------------------------------------------------------------------------
def script_dir() -> Path:
    """<output>/_work/script — where the pipeline reads its narration scripts."""
    from recap.config import work_dir

    return work_dir(_base_recap()) / "script"


def script_path(lang: str) -> Path:
    return script_dir() / ("script_zh.txt" if lang.startswith("zh") else "script_en.txt")


def read_script(lang: str) -> str:
    p = script_path(lang)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _ensure_scripts() -> None:
    """Copy the bundled EN + ZH sample scripts in if the workdir has none."""
    sdir = script_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    for name in ("script_en.txt", "script_zh.txt"):
        dest = sdir / name
        src = INPUTS / name
        if not dest.exists() and src.exists():
            dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            _log(f"    staged the BUNDLED SAMPLE script ({name}) into {dest}")


def sample_script_in_use() -> bool:
    """True when the staged EN script is still the bundled sample.

    The pipeline prefers any script_en.txt it finds, so a leftover sample will
    quietly produce a recap of the *sample film* instead of yours. The panel
    uses this to warn before you render.
    """
    staged = script_path("en")
    src = INPUTS / "script_en.txt"
    if not staged.exists() or not src.exists():
        return False
    try:
        return staged.read_text(encoding="utf-8").strip() == src.read_text(encoding="utf-8").strip()
    except OSError:
        return False


def whisper_available() -> bool:
    """Any Whisper implementation the dialogue extractor can use."""
    return any(have(m) for m in ("faster_whisper", "whisper", "whisperx"))


def ensure_whisper() -> tuple[bool, str]:
    """Make a Whisper implementation available, installing it if missing.

    Auto-recap needs the movie's dialogue; when there is no ``.srt`` that comes
    from Whisper. Rather than telling the user to run pip by hand, we check and,
    if nothing is installed, download & install ``faster-whisper`` into the same
    interpreter that is running the panel. One-time; later runs find it already
    present. Returns ``(ok, message)``.
    """
    if whisper_available():
        return True, "already installed"

    _log("    Whisper not installed -> downloading & installing faster-whisper "
         "(one-time; this can take a couple of minutes) ...")
    import subprocess

    cmd = [sys.executable, "-m", "pip", "install", "faster-whisper"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        return False, "pip install faster-whisper timed out"
    except Exception as exc:
        return False, f"could not start pip: {exc}"

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, f"pip install faster-whisper failed: {tail[-1] if tail else 'unknown error'}"

    if whisper_available():
        _log("    faster-whisper installed.")
        return True, "installed"
    return False, "faster-whisper was installed but could not be imported"


def ensure_embeddings() -> tuple[bool, str]:
    """Make sentence-transformers available, installing it if missing.

    The Step D semantic matcher embeds the transcript + narration with a local
    model (all-MiniLM-L6-v2). Like Whisper, rather than sending the user to
    pip we install it into the interpreter that runs the panel. It pulls
    PyTorch (CPU) so it is a large download — done once, on the first run that
    needs it. Returns (ok, message).
    """
    if have("sentence_transformers"):
        return True, "already installed"

    _log("    sentence-transformers not installed -> installing (one-time, "
         "large download: it includes a CPU PyTorch build) ...")
    import subprocess

    cmd = [sys.executable, "-m", "pip", "install", "sentence-transformers"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    except subprocess.TimeoutExpired:
        return False, "pip install sentence-transformers timed out"
    except Exception as exc:
        return False, f"could not start pip: {exc}"

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, f"pip install sentence-transformers failed: {tail[-1] if tail else 'unknown error'}"

    if have("sentence_transformers"):
        _log("    sentence-transformers installed.")
        return True, "installed"
    return False, "sentence-transformers was installed but could not be imported"


def subtitle_for(movie: Path | None, explicit: str = "") -> Path | None:
    """The .srt/.ass/.vtt that auto-recap would read, if any."""
    if movie is None:
        return None
    try:
        from recap.dialogue import find_subtitle_near

        return find_subtitle_near(Path(movie), explicit or None)
    except Exception:
        return None


def readiness(cfg: dict | None = None) -> dict:
    """What the chosen engine needs to run — the panel shows this before Run."""
    cfg = cfg if cfg is not None else load_config()
    eng = engine_name(cfg)
    movie, movie_err = check_movie(cfg.get("movie_path", ""))
    wants_auto = bool(cfg.get("auto"))          # legacy engine only
    llm_ok = llm_ready(cfg)
    srt = subtitle_for(movie, cfg.get("auto_subtitle", ""))
    whisper = whisper_available()
    sample = sample_script_in_use()
    embeddings = have("sentence_transformers")

    blocking = []
    llm_block = "no LLM configured (pick one in Settings -> LLM)"
    if not llm_ok:
        blocking.append(llm_block)
    elif not ollama_up(cfg):
        blocking.append(
            "Ollama is not running — open a terminal and run: ollama serve, then "
            "ollama pull qwen2.5 (or switch provider in Settings -> LLM)"
        )

    if eng == "semantic":
        # The Step A-F engine *always* reads the movie: it needs a real file,
        # an LLM to write the recap, dialogue (subtitle or auto-installed
        # Whisper) and the local embedding model.
        if movie is None:
            blocking.append(movie_err or "no movie file set (semantic engine recaps a real movie)")
        if not srt and not whisper:
            blocking.append(
                "no .srt next to the movie and Whisper not installed — it will be "
                "downloaded & installed automatically on run"
            )
        if not embeddings:
            blocking.append(
                "sentence-transformers not installed — it will be downloaded & "
                "installed automatically on first run (large download)"
            )
        auto_ready = movie is not None and not blocking
    else:
        if wants_auto:
            if movie is None:
                blocking.append(movie_err or "no movie file set")
            # A missing Whisper is no longer a blocker: when the run starts and
            # no .srt is found, ensure_whisper() downloads & installs it.
        auto_ready = (wants_auto or not sample) and movie is not None and not blocking

    whisper_will_install = (
        (eng == "semantic" or wants_auto)
        and movie is not None and not srt and not whisper
    )

    return {
        "engine": eng,
        "whisper_will_install": whisper_will_install,
        "embeddings": embeddings,
        "embeddings_will_install": eng == "semantic" and not embeddings,
        "llm_reachable": ollama_up(cfg),
        "auto": wants_auto,
        "auto_ready": auto_ready,
        "blocking": blocking,
        "movie_ok": movie is not None,
        "movie_resolved": str(movie) if movie else "",
        "subtitle": str(srt) if srt else "",
        "whisper": whisper,
        "llm_ready": llm_ok,
        "llm_provider": cfg.get("llm_provider", "none"),
        "sample_script": sample,
        "script_source": (
            "bundled sample" if sample else ("workdir script" if script_path("en").exists() else "none")
        ),
    }


def write_scripts(en_script: str = "", zh_script: str = "") -> dict:
    """Write edited scripts into the workdir so the next run narrates them."""
    sdir = script_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    written = {}
    if en_script.strip():
        p = sdir / "script_en.txt"
        p.write_text(en_script.strip() + "\n", encoding="utf-8")
        written["en"] = str(p)
    if zh_script.strip():
        p = sdir / "script_zh.txt"
        p.write_text(zh_script.strip() + "\n", encoding="utf-8")
        written["zh"] = str(p)
    return written


def clear_staged_scripts() -> None:
    """Drop staged scripts so auto-recap regenerates them from the dialogue."""
    for name in ("script_en.txt", "script_zh.txt"):
        (script_dir() / name).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------
def _resolve_clips(cfg: dict) -> list[Path]:
    """Return the movie file to recap, or [] to use placeholder scenes.

    Raises MoviePathError with a human-readable reason when the configured path
    is missing, a folder, or unreadable — instead of letting ffmpeg fail later
    with a cryptic "Permission denied".
    """
    resolved, err = check_movie(cfg.get("movie_path", ""))
    if err:
        raise MoviePathError(err)
    return [resolved] if resolved else []


def run_language(lang: str, cfg: dict | None = None) -> Path | None:
    """Run the recap pipeline for one language, returning the output mp4."""
    from recap import pipeline

    cfg = cfg if cfg is not None else load_config()
    lang = "zh" if lang.startswith("zh") else "en"
    _log(f">>> Run clip [{lang}] — dubbing={lang}, subtitles={lang}")

    try:
        clips = _resolve_clips(cfg)
    except MoviePathError as exc:
        _log(f"    ERROR: {exc}")
        _log("    Fix it in Settings -> Movie file path, then run again.")
        raise

    rc = recap_cfg(cfg, lang)
    storyboard = not clips
    wants_auto = bool(cfg.get("auto"))

    # Auto-recap means "read this movie, then write the narration". Without a
    # movie there is nothing to read, and the old behaviour was to quietly fall
    # back to the bundled sample script and render placeholder frames — i.e. a
    # finished-looking video of the wrong film. Refuse instead.
    if wants_auto and not clips:
        _, why = check_movie(cfg.get("movie_path", ""))
        msg = (
            "Auto-recap is ON but no usable movie file is set, so there is nothing "
            "to read. "
            + (f"{why}. " if why else "")
            + "Set Movie file path to a video FILE (not a folder) in Settings, or "
            "turn Auto-recap OFF to narrate the script in the Script editor."
        )
        _log(f"    ERROR: {msg}")
        raise MoviePathError(msg)

    auto = wants_auto and bool(clips)

    if auto:
        # Auto-recap: the LLM writes the EN recap from the movie's dialogue.
        # Only the EN pass clears staged scripts — the ZH pass reuses the EN
        # script that pass produced as its translation source (otherwise the
        # LLM would write the recap twice).
        if lang == "en":
            clear_staged_scripts()
            _log("    auto-recap: writing EN recap from movie dialogue (LLM)...")
        else:
            _log("    auto-recap: translating the EN recap to Simplified Chinese...")
        rc.setdefault("dialogue", {})
        rc["dialogue"]["srt_path"] = cfg.get("auto_subtitle", "") or None
        rc["dialogue"]["whisper_model"] = cfg.get("whisper_model", "small")
        rc["dialogue"]["whisper_device"] = cfg.get("whisper_device", "auto")

        # The dialogue comes from an .srt when one exists; otherwise Whisper must
        # transcribe the movie's audio. If nothing is installed, fetch it now.
        if subtitle_for(clips[0], cfg.get("auto_subtitle", "")) is None:
            ok, why = ensure_whisper()
            if not ok:
                msg = (
                    "Auto-recap needs the movie's dialogue, but Whisper could not be "
                    f"made available ({why}). Drop an .srt next to the movie, or fix "
                    "your network/pip and run again."
                )
                _log(f"    ERROR: {msg}")
                raise RuntimeError(msg)
    else:
        _ensure_scripts()
        if storyboard:
            _log("    no movie file -> using placeholder storyboard scenes")
        else:
            _log(f"    using movie: {clips[0]}")
        if sample_script_in_use():
            _log(
                "    WARNING: the narration is the BUNDLED SAMPLE script, not your "
                "movie. Turn on Auto-recap with a movie file, or paste your own "
                "script in the Script editor tab."
            )

    try:
        with redirect_stdout(_Tee(_REAL_STDOUT)):
            out_mp4s = pipeline.run(rc, clips, storyboard=storyboard)
    except Exception as exc:  # surface errors to the panel log stream
        _log(f"    ERROR: {type(exc).__name__}: {exc}")
        raise
    if out_mp4s:
        _log(f">>> Done [{lang}] -> {out_mp4s[0]}")
        return Path(out_mp4s[0])
    return None


def run_semantic(langs: list[str], cfg: dict | None = None) -> list[Path]:
    """Run the Step A-F engine once for every requested language.

    One ``pipeline.auto_recap`` call covers all languages (the expensive steps —
    whisper extraction, chunking, summaries, semantic matching — run once and
    every language reuses the result), so this returns a list of output mp4s.
    """
    cfg = cfg if cfg is not None else load_config()
    langs = [l for l in langs if l in ("en", "zh")] or ["en"]
    _log(f">>> Semantic recap [{'+'.join(langs)}]")

    from recap import pipeline

    try:
        clips = _resolve_clips(cfg)
    except MoviePathError as exc:
        _log(f"    ERROR: {exc}")
        _log("    Fix it in Settings -> Movie file path, then run again.")
        raise
    if not clips:
        _, why = check_movie(cfg.get("movie_path", ""))
        msg = (
            "The semantic (Step A-F) engine recaps a REAL movie — it reads the "
            "film's audio/dialogue and cuts its actual footage. "
            + (f"{why}. " if why else "")
            + "Set Movie file path to a video FILE in Settings, or switch Engine "
            "to 'legacy recap' in Settings to narrate a script over placeholder scenes."
        )
        _log(f"    ERROR: {msg}")
        raise MoviePathError(msg)

    movie = clips[0]
    rc = semantic_recap_cfg(cfg, langs)

    # FIX: carry the UI's subtitle choice + whisper knobs into the pipeline
    # config, otherwise the engine can't see the browsed .srt and silently
    # falls back to Whisper-transcribing the whole film.
    rc.setdefault("dialogue", {})
    sub = (cfg.get("auto_subtitle") or "").strip() or None
    if sub:
        rc["dialogue"]["srt_path"] = sub
        _log(f"    dialogue subtitle: {sub}")
    else:
        rc["dialogue"]["srt_path"] = None
    rc["dialogue"]["whisper_model"] = cfg.get("whisper_model", "small")
    rc["dialogue"]["whisper_device"] = cfg.get("whisper_device", "auto")

    # Dialogue source: prefer a .srt next to the movie, else Whisper (install
    # on demand if nothing is present).
    if subtitle_for(movie, cfg.get("auto_subtitle", "")) is None:
        ok, why = ensure_whisper()
        if not ok:
            msg = (
                "Auto-recap needs the movie's dialogue, but Whisper could not be "
                f"made available ({why}). Drop an .srt next to the movie, or fix "
                "your network/pip and run again."
            )
            _log(f"    ERROR: {msg}")
            raise RuntimeError(msg)

    # Step D needs the local embedding model (install on demand, large).
    ok, why = ensure_embeddings()
    if not ok:
        msg = (
            "Semantic timestamp mapping needs sentence-transformers, but it "
            f"could not be installed ({why}). Fix your network/pip and run again."
        )
        _log(f"    ERROR: {msg}")
        raise RuntimeError(msg)

    try:
        with redirect_stdout(_Tee(_REAL_STDOUT)):
            outs = pipeline.auto_recap(rc, movie)
    except Exception as exc:  # surface errors to the panel log stream
        _log(f"    ERROR: {type(exc).__name__}: {exc}")
        raise
    if outs:
        _log(f">>> Done [{'+'.join(langs)}] -> {', '.join(str(p) for p in outs)}")
    return [Path(p) for p in outs]


def list_outputs() -> list[dict]:
    res = []
    out = output_dir()
    if out.exists():
        for p in sorted(out.glob("*.mp4")):
            res.append(
                {
                    "path": str(p),
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / 1e6, 2),
                    "url": "/output/" + p.name,
                    "lang": "zh" if "_zh" in p.name else "en",
                }
            )
    return res


def start_run(langs: list[str], cfg: dict | None = None) -> bool:
    """Start a background run. Returns False if one is already running."""
    cfg = cfg if cfg is not None else load_config()
    langs = [l for l in langs if l in ("en", "zh")] or ["en", "zh"]
    with LOCK:
        if JOB_STATE["running"]:
            return False
        JOB_STATE.update(
            running=True,
            langs=list(langs),
            started=time.time(),
            finished=0,
            error="",
            cancel=False,
        )

    def worker():
        try:
            if engine_name(cfg) == "semantic":
                # One pass covers every language (shared whisper/chunks/beats).
                with LOCK:
                    if not JOB_STATE["cancel"]:
                        run_semantic(langs, cfg)
            else:
                for lang in langs:
                    with LOCK:
                        if JOB_STATE["cancel"]:
                            break
                    run_language(lang, cfg)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            with LOCK:
                JOB_STATE["error"] = err
            _log(f"!!! run failed: {err}")
        finally:
            with LOCK:
                cancelled = JOB_STATE["cancel"]
                JOB_STATE["running"] = False
                JOB_STATE["cancel"] = False
                JOB_STATE["finished"] = time.time()
                JOB_STATE["runs"] += 1
            _log(">>> Run cancelled." if cancelled else ">>> All runs finished.")

    threading.Thread(target=worker, daemon=True, name="recap-run").start()
    return True


def cancel_run() -> bool:
    """Ask a running job to stop before its next language."""
    with LOCK:
        if not JOB_STATE["running"]:
            return False
        JOB_STATE["cancel"] = True
    _log(">>> Stop requested — finishing the current step, then halting.")
    return True


def status() -> dict:
    with LOCK:
        return dict(JOB_STATE)


def serve(host: str = "0.0.0.0", port: int = 8080, open_browser: bool = False) -> None:
    """Start the control panel (kept here so old entry points still work)."""
    import app  # local import: app.py imports this module

    app.main(host=host, port=port, open_browser=open_browser)


if __name__ == "__main__":
    # Backwards-compatible entry point: `python recap-studio/runner.py`.
    if str(STUDIO_DIR) not in sys.path:
        sys.path.insert(0, str(STUDIO_DIR))
    import app

    app.main(*app._parse_args())
