"""Self-contained control panel server (Python stdlib only).

Serves the UI and a small JSON API that drives the recap pipeline in the
background. No pip deps beyond the pipeline itself, so it always boots.

Endpoints:
    GET  /                    -> the control panel HTML
    GET  /api/status          -> config + job state + outputs + env health
    POST /api/config          -> update persisted config (JSON body)
    POST /api/run             -> start background run; body {"langs": ["en","zh"]}
    GET  /api/logs            -> recent log lines (n=200)
    GET  /healthz             -> 200 ok
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

STUDIO_DIR = Path(__file__).resolve().parent
STATIC = STUDIO_DIR / "static"
sys.path.insert(0, str(STUDIO_DIR))

import runner  # noqa: E402


class Handler(BaseHTTPRequestHandler):
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
            return self._json(200, {"logs": runner.log_tail(n)})
        if path == "/healthz":
            return self._send(200, b"ok", "text/plain")
        return self._json(404, {"error": "not found"})

    def _query(self, key, default):
        q = urlparse(self.path).query
        for pair in q.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                if k == key:
                    return v
        return default

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/config":
            cfg = self._read_body()
            merged = runner.load_config()
            merged.update(cfg)
            runner.save_config(merged)
            # make Zhipu key available to the pipeline environment
            if merged.get("llm_api_key"):
                os.environ.setdefault("LLM_API_KEY", merged["llm_api_key"])
            return self._json(200, {"ok": True, "config": merged})
        if path == "/api/run":
            body = self._read_body()
            langs = [l for l in body.get("langs", ["en", "zh"]) if l in ("en", "zh")]
            ok = runner.start_run(langs, runner.load_config())
            return self._json(ok and 200 or 409, {"ok": ok, "running": ok})
        return self._json(404, {"error": "not found"})

    def _index(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        self._send(200, html.encode(), "text/html; charset=utf-8")

    def _status(self) -> dict:
        cfg = runner.load_config()
        clips = []
        mp = cfg.get("movie_path", "").strip()
        if mp:
            clips.append({"path": mp, "exists": os.path.exists(mp), "size_mb": _mb(mp)})
        return {
            "engine": cfg.get("engine", "recap"),
            "config": cfg,
            "job": runner.status(),
            "outputs": runner.list_outputs(),
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
            },
        }


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


def _mb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / 1e6, 2)
    except Exception:
        return 0.0


def _llm_ready(cfg: dict) -> bool:
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
