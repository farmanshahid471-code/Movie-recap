"""Check the torch/torchaudio pair WITHOUT importing them.

Importing a mismatched torchaudio crashes python.exe with a Windows
"Entry Point Not Found" dialog (OS loader error -- uncatchable), so this
check only reads installed version metadata. Exit 0 = safe / absent,
1 = mismatched pair (run fix_torchaudio.bat).

Usage:  python recap-studio/tools/torch_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "movie-recap-bot"))

from recap import torch_compat  # noqa: E402


def main() -> int:
    torch_v = torch_compat.installed_version("torch")
    audio_v = torch_compat.installed_version("torchaudio")
    if not torch_v and not audio_v:
        print("  [OK] torch/torchaudio not installed (not needed for the "
              "default edge-tts + faster-whisper pipeline).")
        return 0
    print(f"       torch={torch_v or '-'}  torchaudio={audio_v or '-'}")
    problem = torch_compat.torch_stack_problem()
    if problem:
        print(f"  [X] BROKEN torch install: {problem}")
        print(torch_compat.FIX_INSTRUCTIONS)
        return 1
    print("  [OK] torch/torchaudio are a matched pair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
