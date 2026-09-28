"""Vast.ai GPU cloud integration helper.

Simplifies authentication, SSH key configuration, instance searching, and remote
recap workflow so users with Vast.ai credits can launch GPU instances effortlessly.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_ssh_key_path() -> Path:
    """Return path to user's SSH public key, creating one if needed."""
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    pub_key = ssh_dir / "id_rsa.pub"
    priv_key = ssh_dir / "id_rsa"
    if not pub_key.exists():
        # Generate an RSA key without passphrase
        try:
            subprocess.run(
                ["ssh-keygen", "-t", "rsa", "-b", "4096", "-N", "", "-f", str(priv_key)],
                check=True,
                capture_output=True,
            )
        except Exception as exc:
            print(f"  ! Note: ssh-keygen returned ({exc}).")
    return pub_key


def configure_api_key(api_key: str) -> bool:
    """Set the Vast.ai API key using the vastai CLI."""
    api_key = api_key.strip()
    if not api_key:
        print("  [X] API key cannot be empty.")
        return False

    try:
        res = subprocess.run(
            [sys.executable, "-m", "vastai", "set", "api-key", api_key],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            print("  [OK] Vast.ai API key saved successfully.")
            return True
        print(f"  [X] Failed to set Vast.ai API key: {res.stderr or res.stdout}")
        return False
    except Exception as exc:
        print(f"  [X] Error running vastai CLI: {exc}")
        return False


def attach_ssh_key() -> bool:
    """Upload user's public SSH key to their Vast.ai account."""
    pub_key_path = get_ssh_key_path()
    if not pub_key_path.exists():
        print(f"  [!] No SSH public key found at {pub_key_path}.")
        return False

    pub_text = pub_key_path.read_text(encoding="utf-8").strip()
    try:
        # Check existing ssh keys
        res = subprocess.run(
            [sys.executable, "-m", "vastai", "show", "ssh-keys", "--raw"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and pub_text in res.stdout:
            print("  [OK] SSH key is already registered with Vast.ai.")
            return True

        # Attach new key
        add_res = subprocess.run(
            [sys.executable, "-m", "vastai", "create", "ssh-key", pub_text],
            capture_output=True,
            text=True,
            check=False,
        )
        if add_res.returncode == 0:
            print("  [OK] SSH key registered with Vast.ai successfully.")
            return True
        print(f"  [!] Notice registering SSH key: {add_res.stderr or add_res.stdout}")
        return False
    except Exception as exc:
        print(f"  [!] Could not register SSH key: {exc}")
        return False


def check_account() -> dict | None:
    """Check Vast.ai account credits and status."""
    try:
        res = subprocess.run(
            [sys.executable, "-m", "vastai", "show", "user", "--raw"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            data = json.loads(res.stdout)
            credit = data.get("credit", 0.0)
            email = data.get("email", "")
            print(f"  [OK] Authenticated as {email} | Account Balance: ${credit:.2f}")
            return data
        print(f"  [X] Vast.ai authentication check failed. Make sure your API key is correct.")
        return None
    except Exception as exc:
        print(f"  [X] Error checking Vast.ai account: {exc}")
        return None


def list_recommended_offers(limit: int = 5) -> list[dict]:
    """Find cheap, reliable NVIDIA GPU instances for Movie Recap processing."""
    query = "gpu_name in [RTX_3060, RTX_3070, RTX_3080, RTX_3090, RTX_4090, RTX_A4000] num_gpus=1 inet_down>100 reliability>0.95"
    try:
        res = subprocess.run(
            [sys.executable, "-m", "vastai", "search", "offers", query, "--raw"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            offers = json.loads(res.stdout)
            return offers[:limit]
        return []
    except Exception:
        return []


def main():
    """Interactive command-line setup for Vast.ai."""
    print("=" * 60)
    print("  Vast.ai GPU Cloud Setup for Movie Recap Bot")
    print("=" * 60)
    print("  Get your API key at: https://cloud.vast.ai/account/ -> API Keys")
    print("")

    api_key = ""
    if len(sys.argv) > 1:
        if sys.argv[1] == "configure" and len(sys.argv) > 2:
            api_key = sys.argv[2]
        elif sys.argv[1] == "--api-key" and len(sys.argv) > 2:
            api_key = sys.argv[2]
        elif sys.argv[1] in ("--status", "status"):
            print("  Checking existing Vast.ai account status...")
            check_account()
            return

    if not api_key:
        try:
            api_key = input("  Enter your Vast.ai API Key: ").strip()
        except EOFError:
            api_key = ""

    if not api_key:
        print("  Usage: python -m recap.vast configure <YOUR_VAST_API_KEY>")
        return

    print("\n  1. Configuring API Key...")
    if not configure_api_key(api_key):
        return

    print("\n  2. Checking Account...")
    user = check_account()
    if not user:
        return

    print("\n  3. Setting up SSH Key...")
    attach_ssh_key()

    print("\n  4. Searching for top recommended GPU deals...")
    offers = list_recommended_offers(5)
    if offers:
        print(f"  {'ID':<8} {'GPU':<18} {'VRAM':<8} {'$/Hour':<10} {'DL Speed':<12}")
        print("  " + "-" * 56)
        for o in offers:
            o_id = o.get("id")
            gpu = o.get("gpu_name")
            vram = f"{o.get('gpu_ram', 0)/1024:.0f}GB"
            dph = f"${o.get('dph_total', 0.0):.3f}"
            speed = f"{o.get('inet_down', 0.0):.0f} Mbps"
            print(f"  {o_id:<8} {gpu:<18} {vram:<8} {dph:<10} {speed:<12}")
        print("\n  To launch the top GPU instance automatically, run:")
        print(f"    vastai create instance {offers[0].get('id')} --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime --disk 40")
    else:
        print("  No active deals found matching query, check vastai web dashboard.")

    print("\n  Setup Complete! You can now use Vast.ai GPU instances with Movie Recap Bot.")


if __name__ == "__main__":
    main()
