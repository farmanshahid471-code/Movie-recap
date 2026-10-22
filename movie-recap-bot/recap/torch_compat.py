"""Guard against a mismatched torch/torchaudio install (Windows DLL crash).

The symptom (user screenshot)::

    python.exe - Entry Point Not Found
    The procedure entry point ?def@Library@torch@@... could not be located
    in ...\\torchaudio\\lib\\libtorchaudio.pyd

``torch`` and ``torchaudio`` are released as a matched pair: torchaudio's
native extension calls symbols that only exist in the *same* torch
release. Installing them at different versions (easy to do: whisperx pins
``torch~=2.8.0``, coqui-tts and plain ``pip install torch`` pull other
releases, and each install overwrites the other) produces a pair whose
DLLs disagree. Windows then fails to load ``libtorchaudio.pyd`` *at
import time* and shows that dialog -- it is an OS loader error, not a
Python exception, so no ``try/except ImportError`` around
``import whisperx`` / ``from TTS.api import TTS`` can ever catch it. The
process simply dies.

The only defence is to never attempt the import when the pair is known
to be mismatched. This module checks the *installed distribution
metadata* (``importlib.metadata``) -- reading version numbers loads no
native code, so the check itself can never trigger the crash.

Used before every optional import that drags in torchaudio
(``whisperx`` in align/dialogue, ``TTS`` in tts). On a mismatch those
features degrade to their existing fallbacks (faster-whisper / edge-tts)
with an actionable message instead of killing the whole run.
"""
from __future__ import annotations

from importlib import metadata

#: The matched pair whisperx currently pins (torch~=2.8.0 / torchaudio~=2.8.0).
KNOWN_GOOD_PAIR = ("2.8.0", "2.8.0")

FIX_INSTRUCTIONS = (
    "Fix it (one-time, in the project folder):\n"
    "  fix_torchaudio.bat            <- double-click, reinstalls a matched pair\n"
    "or manually:\n"
    "  pip uninstall -y torch torchaudio torchvision\n"
    "  pip install torch==2.8.0 torchaudio==2.8.0 "
    "--index-url https://download.pytorch.org/whl/cpu\n"
    "Or simply remove the optional engines -- the default pipeline "
    "(edge-tts + faster-whisper) does not need torch at all:\n"
    "  pip uninstall -y whisperx coqui-tts torch torchaudio torchvision"
)


def installed_version(dist: str) -> str | None:
    """Installed version of a distribution, or None. Loads NO native code."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    try:
        parts = [int(p) for p in version.replace("+", ".").split(".") if p.isdigit()]
    except ValueError:
        return None
    return (parts[0], parts[1]) if len(parts) >= 2 else None


def torch_stack_problem(versions: dict | None = None) -> str | None:
    """Describe a torch/torchaudio mismatch, or None when safe to import.

    ``versions`` (tests) maps distribution name -> version string; by
    default the real installed metadata is read.
    """
    def ver(name: str) -> str | None:
        if versions is not None:
            return versions.get(name)
        return installed_version(name)

    torch_v, audio_v = ver("torch"), ver("torchaudio")
    if audio_v and not torch_v:
        return (f"torchaudio {audio_v} is installed but torch is missing; "
                "torchaudio cannot load without its matching torch.")
    if torch_v and audio_v:
        if _major_minor(torch_v) != _major_minor(audio_v):
            return (f"torch {torch_v} and torchaudio {audio_v} are different "
                    "releases. torchaudio's DLL calls symbols that only "
                    "exist in its own torch release, so importing it "
                    "crashes python.exe with 'Entry Point Not Found' "
                    "(libtorchaudio.pyd).")
    return None


def torch_stack_ok() -> bool:
    return torch_stack_problem() is None


def guard(what: str) -> bool:
    """True when it is safe to import a torchaudio-dependent package.

    On a mismatch, prints why ``what`` is being skipped plus the fix, and
    returns False so the caller takes its fallback path.
    """
    problem = torch_stack_problem()
    if problem is None:
        return True
    print(f"  ! Skipping {what}: broken torch install. {problem}\n"
          f"{FIX_INSTRUCTIONS}", flush=True)
    return False
