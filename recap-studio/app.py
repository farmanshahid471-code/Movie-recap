"""Recap Studio control panel server (Python stdlib only).

Serves the UI plus a small JSON API that drives the recap pipeline in the
background. No pip deps beyond the pipeline itself, so it always boots.

Run it with:  python recap-studio/app.py        (or: double-click setup_ui.bat)

Endpoints:
    GET  /                      -> the control panel HTML
    GET  /api/status            -> config + job state + outputs + env health
    GET  /api/logs?n=200        -> recent pipeline log lines
    GET  /api/scripts?lang=en   -> narration script for one language
    GET  /output/<file>.mp4     -> stream a rendered clip (Range supported)
    GET  /healthz               -> 200 ok
    POST /api/config            -> update persisted settings (JSON body)
    POST /api/run               -> start background run; body {"langs":["en","zh"]}
    POST /api/generate          -> run both clips (Generate button)
    POST /api/render            -> save edited scripts, then re-render both clips
    POST /api/stop_run          -> cancel the running job (server stays up)
    POST /api/stop              -> cancel any run and shut the server down
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STUDIO_DIR = Path(__file__).resolve().parent
STATIC = STUDIO_DIR / "static"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import runner  # noqa: E402

_SERVER: ThreadingHTTPServer | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "RecapStudio/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence default request logging
        pass

    # ---- helpers ----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":  # HEAD must not carry a body
            self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _query(self, key: str, default: str = "") -> str:
        return (parse_qs(urlparse(self.path).query).get(key) or [default])[0]

    def _guard(self, fn):
        """Run a route, turning unexpected errors into a 500 the UI can show."""
        try:
            fn()
        except BrokenPipeError:
            pass
        except Exception as exc:
            runner._log(f"!!! panel error: {type(exc).__name__}: {exc}")
            try:
                self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    # ---- routes -----------------------------------------------------------
    def do_GET(self):
        self._guard(self._route_get)

    def _route_get(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            return self._index()
        if path == "/api/status":
            return self._json(200, self._status())
        if path == "/api/logs":
            try:
                n = int(self._query("n", "200"))
            except ValueError:
                n = 200
            return self._json(200, {"logs": runner.log_tail(max(1, min(n, runner.MAX_LOG_LINES)))})
        if path == "/api/scripts":
            lang = "zh" if self._query("lang", "en").startswith("zh") else "en"
            return self._json(200, {"lang": lang, "script": runner.read_script(lang), "path": str(runner.script_path(lang))})
        if path.startswith("/output/"):
            return self._stream_clip(path[len("/output/"):])
        if path == "/api/browse":
            return self._browse(parse_qs(parsed.query))
        if path == "/healthz":
            return self._send(200, b"ok", "text/plain; charset=utf-8")
        return self._json(404, {"ok": False, "error": "not found"})

    def _browse(self, qs):
        """Server-side file browser so the panel can offer a Browse button.

        Browsers refuse to reveal a chosen file's full path, so instead we list
        the machine's folders here (it is the same machine) and let the UI pick.
        """
        kind = qs.get("kind", ["video"])[0]
        raw = (qs.get("path", [""])[0]).strip().strip('"')
        exts = runner.VIDEO_EXTS if kind == "video" else (".srt", ".ass", ".vtt", ".sub", ".txt")

        p = Path(raw).expanduser() if raw else None
        if p is not None and p.is_file():
            p = p.parent
        if p is None or not p.is_dir():
            return self._json(200, {
                "ok": True, "path": "", "parent": "", "dirs": [], "files": [],
                "roots": self._roots(),
            })

        try:
            entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except OSError as exc:
            return self._json(200, {"ok": False, "error": f"cannot open {p}: {exc}"})

        dirs = [d.name for d in entries if d.is_dir() and not d.name.startswith(".")]
        files = [f.name for f in entries if f.is_file() and f.suffix.lower() in exts]
        parent = "" if p == p.parent else str(p.parent)
        return self._json(200, {
            "ok": True, "path": str(p), "parent": parent, "dirs": dirs, "files": files,
            "roots": self._roots(),
        })

    @staticmethod
    def _roots():
        if os.name == "nt":
            import string
            return [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        return ["/"]

    def do_POST(self):
        self._guard(self._route_post)

    def _route_post(self):
        path = urlparse(self.path).path
        if path == "/api/config":
            merged = runner.load_config()
            merged.update(self._read_body())
            runner.save_config(merged)
            runner._apply_llm(runner._base_recap(), merged)
            return self._json(200, {"ok": True, "config": merged})

        if path == "/api/run":
            body = self._read_body()
            langs = [l for l in body.get("langs", ["en", "zh"]) if l in ("en", "zh")]
            ok = runner.start_run(langs or ["en", "zh"], runner.load_config())
            return self._json(200 if ok else 409, {"ok": ok, "running": ok, "error": "" if ok else "a run is already in progress"})

        if path == "/api/generate":
            body = self._read_body()
            langs = [l for l in body.get("langs", ["en", "zh"]) if l in ("en", "zh")]
            ok = runner.start_run(langs or ["en", "zh"], runner.load_config())
            return self._json(200 if ok else 409, {"ok": ok, "running": ok})

        if path == "/api/render":
            body = self._read_body()
            written = runner.write_scripts(body.get("en_script", ""), body.get("zh_script", ""))
            if not written:
                return self._json(400, {"ok": False, "error": "nothing to render: both scripts are empty"})
            langs = [l for l in body.get("langs", ["en", "zh"]) if l in ("en", "zh")]
            cfg = runner.load_config()
            cfg["auto"] = False  # render the edited scripts; never re-ask the LLM
            ok = runner.start_run(langs or ["en", "zh"], cfg)
            return self._json(200 if ok else 409, {"ok": ok, "written": written})

        if path == "/api/stop_run":
            ok = runner.cancel_run()
            return self._json(200, {"ok": ok, "error": "" if ok else "no run in progress"})

        if path == "/api/stop":
            runner.cancel_run()
            runner._log(">>> Studio closing — this window/browser tab can be closed.")
            threading.Thread(target=_shutdown, daemon=True).start()
            return self._json(200, {"ok": True})

        return self._json(404, {"ok": False, "error": "not found"})

    # ---- bodies -----------------------------------------------------------
    def _index(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

    def _status(self) -> dict:
        cfg = runner.load_config()
        clips = []
        mp = (cfg.get("movie_path") or "").strip().strip('"')
        movie, movie_err = runner.check_movie(mp)
        if mp:
            clips.append(
                {
                    "path": mp,
                    "resolved": str(movie) if movie else "",
                    "exists": movie is not None,
                    "size_mb": _mb(str(movie)) if movie else 0.0,
                    "error": movie_err,
                }
            )
        out = runner.output_dir(cfg)
        return {
            "engine": cfg.get("engine", "recap"),
            "config": cfg,
            "job": runner.status(),
            "outputs": runner.list_outputs(),
            "clips": clips,
            "ready": runner.readiness(cfg),
            "scripts": {
                "en": str(runner.script_path("en")),
                "zh": str(runner.script_path("zh")),
            },
            "env": {
                "ffmpeg": bool(runner.which("ffmpeg")),
                "scene_detect": runner.have("scenedetect"),
                "llm_provider": cfg.get("llm_provider", "none"),
                "llm_key_set": bool(cfg.get("llm_api_key")),
                "llm_ready": runner.llm_ready(cfg),
                "movie_set": bool(mp),
                "movie_exists": movie is not None,
                "movie_error": movie_err,
                "movie_resolved": str(movie) if movie else "",
                "edge_tts": runner.have("edge_tts"),
                "pysubs2": runner.have("pysubs2"),
                "whisper": runner.whisper_available(),
                "yaml": runner.have("yaml"),
                "output_dir": str(out),
                "output_writable": os.access(out, os.W_OK),
                "python": sys.version.split()[0],
            },
        }

    def do_HEAD(self):
        self._guard(self._route_get)

    def _stream_clip(self, name: str) -> None:
        """Serve one rendered clip from the studio output dir (Range-aware).

        Add ?dl=1 for a real download (Content-Disposition: attachment) instead
        of inline playback — that is what makes a bare link a download link.
        """
        safe = Path(name).name  # no traversal, no subfolders
        if not safe.lower().endswith(".mp4") or not re.fullmatch(r"[\w.\-]+", safe):
            return self._json(404, {"ok": False, "error": "not found"})
        fp = runner.output_dir() / safe
        if not fp.is_file():
            return self._json(404, {"ok": False, "error": "not found"})

        size = fp.stat().st_size
        ctype = mimetypes.guess_type(safe)[0] or "video/mp4"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
                if start >= size:
                    start, end = 0, size - 1
                partial = True

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if self._query("dl", "") in ("1", "true", "yes"):
            self.send_header("Content-Disposition", f'attachment; filename="{safe}"')
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(fp, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(64 * 1024, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def _mb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / 1e6, 2)
    except Exception:
        return 0.0


def _shutdown() -> None:
    """Give the /api/stop response time to flush, then close the listener."""
    import time

    time.sleep(0.4)
    if _SERVER is not None:
        threading.Thread(target=_SERVER.shutdown, daemon=True).start()


def _open_browser_when_ready(url: str, timeout: float = 20.0) -> None:
    """Open the panel in the default browser as soon as the socket answers."""
    import socket
    import time
    import webbrowser
    from urllib.parse import urlparse

    host = urlparse(url).hostname or "127.0.0.1"
    port = urlparse(url).port or 8080
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                webbrowser.open(url)
                print(f"Opened {url} in your browser.")
                return
        except OSError:
            time.sleep(0.3)
    print(f"Could not reach {url} to open a browser — open it manually.")


def main(host: str = "0.0.0.0", port: int = 8080, open_browser: bool = False):
    global _SERVER
    _SERVER = ThreadingHTTPServer((host, port), Handler)
    _SERVER.daemon_threads = True
    local = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    url = f"http://{local}:{port}"
    print(f"Recap Studio control panel listening on http://{host}:{port}")
    print(f"Open {url} in your browser. Press Ctrl+C (or click Close Studio) to stop.")
    if open_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    try:
        _SERVER.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _SERVER.server_close()
        print("Recap Studio stopped.")


def _parse_args(argv: list[str] | None = None) -> tuple[str, int, bool]:
    import argparse

    ap = argparse.ArgumentParser(description="Recap Studio control panel")
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    ap.add_argument(
        "--open-browser",
        action="store_true",
        default=os.environ.get("RECAP_OPEN_BROWSER", "") == "1",
        help="open the panel in the default browser once it is listening",
    )
    a = ap.parse_args(argv)
    return a.host, a.port, a.open_browser


if __name__ == "__main__":
    main(*_parse_args())
