"""Ask a running Recap Studio panel to shut down, then confirm it is gone.

    python recap-studio/tools/shutdown.py 8080

Exit codes: 0 = the server stopped (or was never running), 1 = still up.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from portcheck import is_listening  # noqa: E402


def request_stop(port: int, timeout: float = 5.0) -> bool:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/stop",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "{}")
            return bool(payload.get("ok"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main(argv: list[str]) -> int:
    try:
        port = int(argv[1])
    except (IndexError, ValueError):
        port = 8080

    if not is_listening(port):
        print(f"  [i] Nothing listening on port {port}.")
        return 0

    print("  [..] Asking the panel to shut down ...")
    request_stop(port)

    for _ in range(20):  # up to ~5s for the listener to close
        time.sleep(0.25)
        if not is_listening(port):
            print(f"  [OK] Port {port} released.")
            return 0

    print(f"  [X] Port {port} is still listening after 5s.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
