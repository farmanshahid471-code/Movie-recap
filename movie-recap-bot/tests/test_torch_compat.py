"""Tests for the torch/torchaudio mismatch guard (recap/torch_compat.py).

The real crash is a Windows DLL loader error and cannot be reproduced
(or caught) in a test -- what CAN be tested is the metadata-only check
that decides whether whisperx/TTS may be imported at all.

Run:  python tests/test_torch_compat.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap import torch_compat  # noqa: E402


def test_matched_pair_is_ok():
    assert torch_compat.torch_stack_problem(
        {"torch": "2.8.0", "torchaudio": "2.8.0"}) is None
    # local-build suffixes still count as the same release
    assert torch_compat.torch_stack_problem(
        {"torch": "2.8.0+cpu", "torchaudio": "2.8.0"}) is None
    print("ok: matched torch/torchaudio pair passes")


def test_mismatched_pair_is_flagged():
    problem = torch_compat.torch_stack_problem(
        {"torch": "2.11.0", "torchaudio": "2.8.0"})
    assert problem and "Entry Point Not Found" in problem
    assert "2.11.0" in problem and "2.8.0" in problem
    print("ok: mismatched pair (the screenshot) is flagged before import")


def test_absent_stack_is_ok():
    assert torch_compat.torch_stack_problem({}) is None
    assert torch_compat.torch_stack_problem({"torch": "2.8.0"}) is None
    print("ok: no torch at all is fine (default pipeline needs none)")


def test_torchaudio_without_torch_is_flagged():
    assert torch_compat.torch_stack_problem({"torchaudio": "2.8.0"})
    print("ok: torchaudio without torch is flagged")


def test_fix_instructions_cover_both_paths():
    text = torch_compat.FIX_INSTRUCTIONS
    assert "fix_torchaudio.bat" in text          # one-click repair
    assert "torch==2.8.0" in text and "torchaudio==2.8.0" in text
    assert "pip uninstall" in text               # removal escape hatch
    print("ok: fix instructions offer reinstall + removal")


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_"):
            fn()
    print("\nall torch-compat tests passed")
