"""Ensure ffmpeg/ffprobe are available without ever writing to the C: drive.

Uses static-ffmpeg (already a project dependency) with its binaries stored in
STATIC_FFMPEG_CACHE_DIR when that env var is set (setup_ui.bat points it at the
configured data root on D:). Prefers a system ffmpeg on PATH when one exists.

Usage:
    python recap-studio/tools/ensure_ffmpeg.py
Exits 0 when ffmpeg + ffprobe resolve, 1 otherwise.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path


def _platform_key() -> str:
    machine = platform.machine().lower()
    is_arm = machine in ("arm64", "aarch64")
    if sys.platform == "win32":
        return "win32"
    if sys.platform == "darwin":
        return "darwin_arm64" if is_arm else "darwin"
    if sys.platform.startswith("linux"):
        return "linux_arm64" if is_arm else "linux"
    return sys.platform


def main() -> int:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        print(f"  [OK] ffmpeg found on PATH: {shutil.which('ffmpeg')}")
        print(f"  [OK] ffprobe found on PATH: {shutil.which('ffprobe')}")
        return 0

    try:
        import static_ffmpeg  # type: ignore
    except Exception:
        print("  [X] static-ffmpeg not installed — run:  python -m pip install static-ffmpeg")
        return 1

    cache = os.environ.get("STATIC_FFMPEG_CACHE_DIR")
    if cache:
        target = Path(cache) / _platform_key()
        target.mkdir(parents=True, exist_ok=True)
        static_ffmpeg.add_paths(download_dir=str(target))
        print(f"  [..] static-ffmpeg binaries cached in: {cache}")
    else:
        static_ffmpeg.add_paths()

    ff = shutil.which("ffmpeg")
    fp = shutil.which("ffprobe")
    if ff and fp:
        print(f"  [OK] ffmpeg : {ff}")
        print(f"  [OK] ffprobe: {fp}")
        return 0
    print("  [!] ffmpeg/ffprobe still not resolvable — check the network and retry.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
