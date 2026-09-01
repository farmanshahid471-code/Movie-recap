"""Run the recap pipeline for one language clip, capturing logs.

This is the "engine glue" the control panel calls. It drives the self-contained
``recap`` pipeline (edge-tts narration + burned-in subtitles + ffmpeg assembly)
so it works end-to-end with no API key (scripting is provided via files) and no
external service.

The pipeline lives in ../movie-recap-bot/recap; we add it to sys.path.
All logs go to a shared ring buffer so the UI can stream them.
"""
from __future__ import annotations

import io
import os
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import urlparse
import json

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOT_DIR = Path(__file__).resolve().parent.parent / "movie-recap-bot"
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

STUDIO_DIR = Path(__file__).resolve().parent
INPUTS = BOT_DIR / "inputs" / "text"

# --- config that persists between page loads ---
CONFIG_PATH = STUDIO_DIR / "config.json"
DEFAULT_CONFIG = {
    "engine": "recap",
    "movie_path": "",                       # optional owned movie file
    "storyboard": True,                     # use placeholder scenes when no movie
    "duration": 60,
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
}

LOCK = threading.Lock()
LOG_BUFFER: list[str] = []
MAX_LOG_LINES = 600
JOB_STATE: dict = {"running": False, "langs": [], "started": 0, "finished": 0, "error": ""}


def _log(msg: str) -> None:
    with LOCK:
        LOG_BUFFER.append(msg)
        if len(LOG_BUFFER) > MAX_LOG_LINES:
            del LOG_BUFFER[: len(LOG_BUFFER) - MAX_LOG_LINES]


def _which(name: str) -> str | None:
    import shutil

    p = shutil.which(name)
    if p:
        return p
    # static-ffmpeg binaries (used by the recap pipeline) aren't on PATH; resolve them.
    try:
        import static_ffmpeg  # type: ignore

        static_ffmpeg.add_paths()
        p = shutil.which(name)
        if p:
            return p
    except Exception:
        pass
    return None


def _have(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


def load_config() -> dict:
    if CONFIG_PATH.exists():
        import json

        cfg = dict(DEFAULT_CONFIG)
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except Exception:
            pass
        return cfg
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    import json

    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def log_tail(n: int = 200) -> list[str]:
    with LOCK:
        return LOG_BUFFER[-n:]


def _ensure_scripts(workdir: Path) -> tuple[Path, Path]:
    """Copy the bundled EN + ZH scripts into the pipeline workdir.

    workdir is the pipeline's per-run working folder (output/_work). Because the
    recap pipeline reads its narration scripts from <workdir>/script/, we stage
    them here so both language runs have what they need.
    """
    script_dir = workdir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    en = script_dir / "script_en.txt"
    zh = script_dir / "script_zh.txt"
    if not en.exists() and (INPUTS / "script_en.txt").exists():
        en.write_text((INPUTS / "script_en.txt").read_text(encoding="utf-8"), encoding="utf-8")
    if not zh.exists() and (INPUTS / "script_zh.txt").exists():
        zh.write_text((INPUTS / "script_zh.txt").read_text(encoding="utf-8"), encoding="utf-8")
    return en, zh


def _load_recap(
    cfg: dict,
    lang: str,
) -> dict:
    """Build a recap config for a single-language run."""
    from recap.config import load_config

    rc = load_config()  # reads movie-recap-bot/config.yaml
    rc["language"]["target_languages"] = [lang]
    # re-resolve language tags
    resolved = []
    for l in rc["language"]["target_languages"]:
        tag = l
        if l.startswith("zh"):
            tag = rc["language"].get("zh_variant", "zh-CN")
        resolved.append({"code": l, "tag": tag})
    rc["language"]["_resolved"] = resolved

    name = "recap"
    if lang == "zh":
        rc["narration"]["lang_voice"]["zh"] = cfg.get("voice_zh", "zh-CN-YunxiNeural")
        rc["narration"]["lang_voice"]["en"] = cfg.get("voice_en", "en-US-ChristopherNeural")
    else:
        rc["narration"]["lang_voice"]["en"] = cfg.get("voice_en", "en-US-ChristopherNeural")
        rc["narration"]["lang_voice"]["zh"] = cfg.get("voice_zh", "zh-CN-YunxiNeural")
    rc["narration"]["rate"] = "+0%"

    # Output into the studio output dir so the panel can find it.
    out = STUDIO_DIR / "output"
    out.mkdir(parents=True, exist_ok=True)
    rc["project"]["_out"] = out
    rc["project"]["name"] = name
    rc["project"]["output_dir"] = str(out)
    rc["subtitles"]["display_lang"] = lang
    return rc


def _resolve_clips(cfg: dict) -> list[Path]:
    mp = cfg.get("movie_path", "").strip()
    if mp and Path(mp).exists():
        return [Path(mp)]
    return []


def run_language(lang: str, cfg: dict | None = None) -> Path | None:
    """Run the recap pipeline for one language, returning the output mp4."""
    from recap import pipeline

    cfg = cfg or load_config()
    lang = "zh" if lang.startswith("zh") else "en"
    _log(f">>> Run clip [{lang}] — dubbing={lang}, subtitles={lang}")

    rc = _load_recap(cfg, lang)
    from recap.config import work_dir as _work_dir

    clips = _resolve_clips(cfg)
    storyboard = not clips
    auto = bool(cfg.get("auto")) and bool(clips)

    if auto:
        # Auto-recap: let the LLM write the EN recap from the movie's dialogue.
        # Clear any staged sample script so dialogue drives it.
        wd = _work_dir(rc)
        sdir = wd / "script"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "script_en.txt").unlink(missing_ok=True)
        (sdir / "script_zh.txt").unlink(missing_ok=True)
        _log("    auto-recap: writing EN recap from movie dialogue (LLM)...")
        rc.setdefault("dialogue", {})
        rc["dialogue"]["srt_path"] = cfg.get("auto_subtitle", "") or None
        rc["dialogue"]["whisper_model"] = cfg.get("whisper_model", "small")
    else:
        _ensure_scripts(_work_dir(rc))
        if storyboard:
            _log("    no movie file -> using placeholder storyboard scenes")
        else:
            _log(f"    using movie: {clips[0]}")

    try:
        out_mp4s = pipeline.run(rc, clips, storyboard=storyboard)
    except Exception as exc:  # surface errors to the panel log stream
        _log(f"    ERROR: {type(exc).__name__}: {exc}")
        raise
    if out_mp4s:
        _log(f">>> Done [{lang}] -> {out_mp4s[0]}")
        return Path(out_mp4s[0])
    return None


def list_outputs() -> list[dict]:
    out = STUDIO_DIR / "output"
    res = []
    if out.exists():
        for p in sorted(out.glob("*.mp4")):
            res.append(
                {
                    "path": str(p),
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / 1e6, 2),
                }
            )
    return res


# --------------------------------------------------------------------------
# Helper: write edited scripts into the pipeline workdir
# --------------------------------------------------------------------------
def _write_scripts(en_script: str, zh_script: str) -> None:
    """Write the edited EN + ZH scripts into the pipeline workdir script/ folder.

    The recap pipeline reads its narration scripts from <workdir>/script/script_en.txt
    and script_zh.txt, so we place them there so the next pipeline run uses them.
    """
    from recap.config import work_dir as _work_dir

    wd = _work_dir({})
    sdir = wd / "script"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "script_en.txt").write_text(en_script, encoding="utf-8")
    (sdir / "script_zh.txt").write_text(zh_script, encoding="utf-8")


# --------------------------------------------------------------------------
# Helper: read a script file for the given language
# --------------------------------------------------------------------------
def _scripts_endpoint(lang: str) -> dict:
    """Read the script file for the given language from the pipeline workdir."""
    from recap.config import work_dir as _work_dir
    wd = _work_dir({})
    sdir = wd / "script"
    if lang == "en":
        p = sdir / "script_en.txt"
    else:
        p = sdir / "script_zh.txt"
    if p.exists():
        return {"script": p.read_text(encoding="utf-8")}
    return {"script": ""}


# --------------------------------------------------------------------------
# Background run manager (non-blocking so the UI stays responsive)
# --------------------------------------------------------------------------
def start_run(langs: list[str], cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    with LOCK:
        if JOB_STATE["running"]:
            return False
        JOB_STATE["running"] = True
        JOB_STATE["langs"] = list(langs)
        JOB_STATE["started"] = time.time()
        JOB_STATE["finished"] = 0
        JOB_STATE["error"] = ""

    def worker():
        try:
            for lang in langs:
                run_language(lang, cfg)
        except Exception as exc:
            with LOCK:
                JOB_STATE["error"] = f"{type(exc).__name__}: {exc}"
            _log(f"!!! run failed: {JOB_STATE['error']}")
        finally:
            with LOCK:
                JOB_STATE["running"] = False
                JOB_STATE["finished"] = time.time()
            _log(">>> All runs finished.")

    threading.Thread(target=worker, daemon=True).start()
    return True


def status() -> dict:
    with LOCK:
        return dict(JOB_STATE)


def overall_cfg() -> dict:
    return load_config()


# --------------------------------------------------------------------------
# HTTP control panel server (Python stdlib only)
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    """HTTP request handler for the Recap Studio control panel."""

    def log_message(self, fmt, *args):  # silence default request logging
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    # ---- routes -----------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._index()
        if path == "/api/status":
            return self._json(200, self._status())
        if path == "/api/logs":
            n = int(self._query("n", "200"))
            return self._json(200, {"logs": log_tail(n)})
        if path == "/healthz":
            return self._send(200, b"ok", "text/plain")
        if path == "/api/scripts":
            lang = self._query("lang", "en")
            return self._json(200, _scripts_endpoint(lang))
        if path == "/api/generate":
            ok = start_run(["en", "zh"])
            return self._json(200 if ok else 409, {"ok": ok})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/config":
            cfg = self._read_body()
            merged = load_config()
            merged.update(cfg)
            save_config(merged)
            # make Zhipu key available to the pipeline environment
            if merged.get("llm_api_key"):
                os.environ.setdefault("LLM_API_KEY", merged["llm_api_key"])
            return self._json(200, {"ok": True, "config": merged})
        if path == "/api/run":
            body = self._read_body()
            langs = [l for l in body.get("langs", ["en", "zh"]) if l in ("en", "zh")]
            ok = start_run(langs, load_config())
            return self._json(ok and 200 or 409, {"ok": ok, "running": ok})
        if path == "/api/render":
            body = self._read_body()
            en_script = body.get("en_script", "")
            zh_script = body.get("zh_script", "")
            _write_scripts(en_script, zh_script)
            # Start render run with auto=false so the pipeline uses the existing scripts
            start_run(["en", "zh"], cfg={"auto": False})
            return self._json(200, {"ok": True})
        if path == "/api/scripts":
            # POST to update scripts (optional; GET /api/scripts?lang=en reads them)
            body = self._read_body()
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "not found"})

    def _query(self, key, default):
        q = urlparse(self.path).query
        for pair in q.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                if k == key:
                    return v
        return default

    def _index(self):
        html = (STUDIO_DIR / "static" / "index.html").read_text(encoding="utf-8")
        self._send(200, html.encode(), "text/html; charset=utf-8")

    def _status(self) -> dict:
        cfg = load_config()
        clips = []
        mp = cfg.get("movie_path", "").strip()
        if mp:
            clips.append({"path": mp, "exists": os.path.exists(mp), "size_mb": _mb(mp)})
        return {
            "engine": cfg.get("engine", "recap"),
            "config": cfg,
            "job": status(),
            "outputs": list_outputs(),
            "clips": clips,
            "env": {
                "ffmpeg": bool(_which("ffmpeg")),
                "scene_detect": _have("scenedetect"),
                "llm_provider": cfg.get("llm_provider", "none"),
                "llm_key_set": bool(cfg.get("llm_api_key")),
                "llm_ready": _llm_ready(cfg),
                "movie_set": bool(mp),
                "movie_exists": bool(mp and os.path.exists(mp)),
                "edge_tts": bool(_have("edge_tts")),
            }
        }

    @staticmethod
    def _mb(path: str) -> float:
        try:
            return round(os.path.getsize(path) / 1e6, 2)
        except Exception:
            return 0.0

    def _llm_ready(self, cfg: dict) -> bool:
        """Whether the LLM is usable: Ollama is key-free + configured; others need a key."""
        prov = (cfg.get("llm_provider") or "none").lower()
        if prov == "ollama":
            return True  # no key required; reachability is checked at run time
        if prov == "none":
            return False
        return bool(cfg.get("llm_api_key"))


def main(host: str = "0.0.0.0", port: int = 8080):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Recap Studio control panel listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    main(port=port)