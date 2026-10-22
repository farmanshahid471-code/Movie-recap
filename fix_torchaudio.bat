@echo off
setlocal
title Fix torch / torchaudio mismatch (Entry Point Not Found)
:: =========================================================
::  Repairs the crash in the screenshot:
::    python.exe - Entry Point Not Found ... libtorchaudio.pyd
::  Cause: torch and torchaudio were installed at DIFFERENT versions.
::  Fix:   remove both (+ torchvision / the optional engines that pin
::         them), then reinstall ONE matched pair (2.8.0, the pair
::         whisperx pins) from the official PyTorch CPU index.
::  The default pipeline (edge-tts + faster-whisper) does not need
::  torch at all -- choose option 2 if you only want the crash gone.
:: =========================================================

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY ( where python >nul 2>&1 && set "PY=python" )
if not defined PY (
    echo  [X] Python not found on PATH. Re-run setup_ui.bat first.
    pause
    exit /b 1
)

:: Prefer the F: drive venv created by setup_ui.bat, so the repair
:: touches the packages the app actually uses (on F:, not C:).
if exist "F:\recap-data\venv\Scripts\python.exe" (
    set "PY=F:\recap-data\venv\Scripts\python.exe"
    echo  Using F: venv Python: %PY%
)

echo.
echo  [1] Reinstall a matched torch 2.8.0 + torchaudio 2.8.0 pair (CPU)
echo  [2] Remove torch/torchaudio/whisperx/coqui-tts entirely (fastest;
echo      default edge-tts + faster-whisper pipeline keeps working)
echo.
choice /C 12 /N /M "Choose 1 or 2: "
if errorlevel 2 goto :remove

echo.
echo  Removing mismatched packages...
%PY% -m pip uninstall -y torch torchaudio torchvision whisperx coqui-tts
echo  Installing matched pair (large download, please wait)...
%PY% -m pip install --force-reinstall torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo  [X] Install failed. Check your internet connection and re-run.
    pause
    exit /b 1
)
%PY% -m pip install "whisperx>=3.8"
goto :verify

:remove
echo.
echo  Removing torch stack and the optional engines that need it...
%PY% -m pip uninstall -y torch torchaudio torchvision whisperx coqui-tts

:verify
echo.
%PY% recap-studio\tools\torch_check.py
echo.
echo  Done. You can now run setup_ui.bat again.
pause
exit /b 0
