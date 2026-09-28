"""Connect to a running Vast.ai GPU instance with SSH and port forwarding.

Handles older Windows systems where OpenSSH 'ssh' is not in PATH by:
1. Auto-discovering Git OpenSSH or System32 OpenSSH.
2. Providing a pure-Python Paramiko SSH tunnel + interactive shell as fallback.
"""
from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def find_system_ssh() -> str | None:
    """Search for ssh.exe on Windows across standard locations."""
    # 1. PATH
    try:
        res = subprocess.run(["where", "ssh"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
            if lines:
                return lines[0]
    except Exception:
        pass

    # 2. Common Windows paths
    candidates = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "OpenSSH" / "ssh.exe",
        Path(r"C:\Program Files\Git\usr\bin\ssh.exe"),
        Path(r"C:\Program Files (x86)\Git\usr\bin\ssh.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Git" / "usr" / "bin" / "ssh.exe",
        Path(r"C:\Program Files\OpenSSH\ssh.exe"),
        Path(r"C:\Program Files (x86)\OpenSSH\ssh.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def get_active_vast_instance() -> dict | None:
    """Retrieve the first running Vast.ai instance from the CLI."""
    from recap.vast import run_vast_cmd

    res = run_vast_cmd(["show", "instances", "--raw"])
    if res.returncode == 0:
        try:
            insts = json.loads(res.stdout)
            for ins in insts:
                if ins.get("ssh_host") and ins.get("ssh_port"):
                    return ins
        except Exception:
            pass
    return None


def get_private_key_path() -> Path | None:
    """Find the user's private SSH key corresponding to the registered public key."""
    ssh_dir = Path.home() / ".ssh"
    for name in ("id_ed25519", "id_rsa"):
        p = ssh_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def forward_tunnel(local_port: int, remote_host: str, remote_port: int, transport: paramiko.Transport) -> None:
    """Forward a local TCP port through an active Paramiko transport."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("127.0.0.1", local_port))
        server.listen(5)
    except Exception as exc:
        print(f"  [!] Port forward warning (127.0.0.1:{local_port}): {exc}")
        return

    def handler():
        while True:
            try:
                client_sock, _ = server.accept()
            except Exception:
                break
            try:
                chan = transport.open_channel(
                    "direct-tcpip", (remote_host, remote_port), client_sock.getpeername()
                )
            except Exception as e:
                client_sock.close()
                continue
            if chan is None:
                client_sock.close()
                continue

            def forward(src, dst):
                try:
                    while True:
                        data = src.recv(4096)
                        if not data:
                            break
                        dst.sendall(data)
                except Exception:
                    pass
                finally:
                    src.close()
                    dst.close()

            t1 = threading.Thread(target=forward, args=(client_sock, chan), daemon=True)
            t2 = threading.Thread(target=forward, args=(chan, client_sock), daemon=True)
            t1.start()
            t2.start()

    th = threading.Thread(target=handler, daemon=True)
    th.start()


def run_paramiko_client(host: str, port: int, key_path: Path, local_port: int = 8080) -> None:
    """Connect using pure-Python Paramiko with interactive shell and port forwarding."""
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"  Connecting to root@{host}:{port} with key {key_path.name}...")
    try:
        try:
            k = paramiko.Ed25519Key.from_private_key_file(str(key_path))
        except Exception:
            k = paramiko.RSAKey.from_private_key_file(str(key_path))
        client.connect(hostname=host, port=port, username="root", pkey=k, timeout=20)
    except Exception as exc:
        print(f"  [X] SSH connection failed: {exc}")
        print("  Check that your instance is finished loading in your Vast.ai console.")
        return

    print("  [OK] Connected!")
    # Start port forward 8080:localhost:8080
    transport = client.get_transport()
    forward_tunnel(local_port, "127.0.0.1", local_port, transport)
    print(f"  [OK] Port tunnel active: http://localhost:{local_port} -> GPU instance port {local_port}")
    print("  " + "=" * 60)
    print("  Opening remote interactive shell. Type 'exit' to disconnect.")
    print("  " + "=" * 60)

    channel = client.invoke_shell()
    channel.settimeout(0.0)

    # Windows interactive loop
    stop_event = threading.Event()

    def receive_loop():
        while not stop_event.is_set():
            try:
                data = channel.recv(4096)
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            except Exception:
                time.sleep(0.05)

    th = threading.Thread(target=receive_loop, daemon=True)
    th.start()

    try:
        while not channel.exit_status_ready() and not stop_event.is_set():
            line = sys.stdin.readline()
            if not line:
                break
            channel.send(line)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop_event.set()
        client.close()
        print("\n  Disconnected from Vast.ai instance.")


def main():
    print("=" * 60)
    print("  Vast.ai GPU Instance Connector")
    print("=" * 60)

    # 1. Determine instance host and port
    host = ""
    port = 0
    if len(sys.argv) >= 3:
        host = sys.argv[1]
        try:
            port = int(sys.argv[2])
        except ValueError:
            port = 0

    if not host or not port:
        print("  Querying your running Vast.ai instances...")
        ins = get_active_vast_instance()
        if ins:
            host = ins.get("ssh_host") or ""
            port = int(ins.get("ssh_port") or 0)
            gpu = ins.get("gpu_name", "GPU")
            i_id = ins.get("id")
            print(f"  Found running instance {i_id}: {gpu} at {host}:{port}")
        else:
            print("  [!] No running instances found via API.")
            try:
                host = input("  Enter SSH Host (e.g. ssh7.vast.ai): ").strip()
                p_str = input("  Enter SSH Port (e.g. 15200): ").strip()
                port = int(p_str)
            except (EOFError, KeyboardInterrupt, ValueError):
                pass

    if not host or not port:
        print("  [X] Missing SSH Host or Port.")
        return

    # 2. Check if a native ssh.exe is available anywhere
    ssh_exe = find_system_ssh()
    if ssh_exe:
        print(f"  Using OpenSSH binary found at: {ssh_exe}")
        cmd = [
            ssh_exe,
            "-p", str(port),
            f"root@{host}",
            "-L", "8080:localhost:8080",
            "-o", "StrictHostKeyChecking=no",
        ]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        return

    # 3. Fallback: Pure Python SSH client with Paramiko
    print("  Note: Native ssh.exe was not found on your Windows installation.")
    print("  Using built-in Python SSH connector with port forwarding (8080)...")

    try:
        import paramiko
    except ImportError:
        print("  Installing paramiko for Python SSH support...")
        subprocess.run([sys.executable, "-m", "pip", "install", "paramiko"], check=True)
        import paramiko

    key_path = get_private_key_path()
    if not key_path:
        print("  [X] No private SSH key found in ~/.ssh (id_ed25519 or id_rsa).")
        print("  Run setup_vast.bat first to generate your keys.")
        return

    run_paramiko_client(host, port, key_path, local_port=8080)


if __name__ == "__main__":
    main()
