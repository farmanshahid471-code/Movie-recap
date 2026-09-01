"""Exit 0 when something is listening on a TCP port, 1 otherwise.

Used by setup_ui.bat / stop_ui.bat so they can tell "already running" from
"not running" without parsing `netstat` output.

    python recap-studio/tools/portcheck.py 8080
"""
from __future__ import annotations

import socket
import sys


def is_listening(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main(argv: list[str]) -> int:
    try:
        port = int(argv[1])
    except (IndexError, ValueError):
        port = 8080
    return 0 if is_listening(port) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
