"""Upload a local movie file from PC (e.g. D: drive) to the active Vast.ai GPU instance."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("  Vast.ai Movie File Uploader")
    print("=" * 60)

    # 1. Get file path from command line arg or user input
    file_path = ""
    if len(sys.argv) > 1:
        file_path = " ".join(sys.argv[1:]).strip().strip('"')

    if not file_path:
        file_path = input("  Enter the path of your movie file (e.g. D:\\movie.mp4): ").strip().strip('"')

    if not file_path or not os.path.exists(file_path):
        print(f"  [X] File not found: {file_path}")
        return

    src = Path(file_path)
    file_size_mb = src.stat().st_size / (1024 * 1024)
    print(f"  File to upload: {src.name} ({file_size_mb:.1f} MB)")

    # 2. Find active Vast instance
    from recap.vast_connect import get_active_vast_instance, get_private_key_path, find_system_ssh

    print("  Querying active Vast.ai GPU instance...")
    ins = get_active_vast_instance()
    if not ins:
        print("  [X] No active Vast.ai instance found. Run setup_vast.bat to check instances.")
        return

    host = ins.get("ssh_host")
    port = ins.get("ssh_port")
    i_id = ins.get("id")
    print(f"  Target Instance: {i_id} ({host}:{port})")

    # 3. Create remote uploads dir
    key_path = get_private_key_path()
    remote_dest_dir = "/root/Movie-recap/uploads"
    remote_path = f"{remote_dest_dir}/{src.name}"

    # Try SCP if available
    ssh_exe = find_system_ssh()
    scp_exe = None
    if ssh_exe:
        candidate_scp = Path(ssh_exe).parent / "scp.exe"
        if candidate_scp.exists():
            scp_exe = str(candidate_scp)

    if scp_exe and key_path:
        print(f"  Uploading via OpenSSH SCP: {src.name} -> {remote_path}...")
        cmd = [
            scp_exe,
            "-P", str(port),
            "-i", str(key_path),
            "-o", "StrictHostKeyChecking=no",
            str(src),
            f"root@{host}:{remote_path}",
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print("\n  " + "=" * 60)
            print("  [OK] Upload Complete!")
            print(f"  Paste this path into Recap Studio 'Movie file path':")
            print(f"    {remote_path}")
            print("  " + "=" * 60)
            return

    # Fallback to Paramiko SFTP
    print("  Uploading via Python SFTP...")
    try:
        import paramiko
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "paramiko"], check=True)
        import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        try:
            k = paramiko.Ed25519Key.from_private_key_file(str(key_path))
        except Exception:
            k = paramiko.RSAKey.from_private_key_file(str(key_path))
        client.connect(hostname=host, port=int(port), username="root", pkey=k, timeout=20)
    except Exception as exc:
        print(f"  [X] Failed to connect: {exc}")
        return

    sftp = client.open_sftp()
    try:
        sftp.mkdir(remote_dest_dir)
    except Exception:
        pass

    import time
    last_print = time.time()

    def progress_callback(transferred, total):
        nonlocal last_print
        now = time.time()
        if now - last_print >= 1.0 or transferred == total:
            last_print = now
            pct = (transferred / total) * 100 if total else 0
            mb_done = transferred / 1e6
            mb_tot = total / 1e6
            print(f"\r  Uploading: {pct:.1f}% ({mb_done:.1f}/{mb_tot:.1f} MB)", end="", flush=True)

    try:
        sftp.put(str(src), remote_path, callback=progress_callback)
        print("\n  [OK] Upload Complete!")
        print("\n  " + "=" * 60)
        print(f"  Paste this path into Recap Studio 'Movie file path':")
        print(f"    {remote_path}")
        print("  " + "=" * 60)
    except Exception as exc:
        print(f"\n  [X] Upload failed: {exc}")
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()
