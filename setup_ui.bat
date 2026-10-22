@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Recap Studio - Setup and Open

:: =========================================================
::  Recap Studio - One Click Setup and Open
:: =========================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

if "%PORT%"=="" set "PORT=8080"
set "URL=http://localhost:%PORT%"

echo.
echo  ==========================================
echo   Recap Studio - Control Panel Setup
echo  ==========================================
echo.

:: ---------------------------------------------------------
:: 0. Storage: EVERYTHING lives on the F: drive (no C: space)
::    data + caches + temp + venv -> F:\recap-data
::    rendered videos + _work     -> F:\recap
::    F: is REQUIRED: setup refuses to silently fall back to
::    another drive (the old D:/project fallback could land
::    on C:). Override only via RECAP_DRIVE, e.g. RECAP_DRIVE=E:
:: ---------------------------------------------------------
if not defined RECAP_DRIVE set "RECAP_DRIVE=F:"
if not exist "%RECAP_DRIVE%\" (
    echo  [X] Drive %RECAP_DRIVE% not found. This setup keeps EVERYTHING
    echo      off the C: drive, on %RECAP_DRIVE% -- plug it in / create it,
    echo      or set RECAP_DRIVE to another big drive and re-run.
    pause
    exit /b 1
)
set "RECAP_DATA=%RECAP_DRIVE%\recap-data"
if /i "%ROOT:~0,2%"=="C:" (
    echo  [!] NOTE: this project folder itself is on C: : %ROOT%
    echo      Only the small code files live there -- packages, models,
    echo      caches and videos all go to %RECAP_DRIVE% below. To move
    echo      even the code, copy this whole folder to %RECAP_DRIVE% and
    echo      run setup_ui.bat from there.
    echo.
)

set "RECAP_OUTPUT=%RECAP_DRIVE%\recap"

mkdir "%RECAP_DATA%" 2>nul
mkdir "%RECAP_DATA%\cache" 2>nul
mkdir "%RECAP_DATA%\tmp" 2>nul
mkdir "%RECAP_DATA%\pip-cache" 2>nul
mkdir "%RECAP_OUTPUT%" 2>nul

set "TEMP=%RECAP_DATA%\tmp"
set "TMP=%RECAP_DATA%\tmp"
set "PIP_CACHE_DIR=%RECAP_DATA%\pip-cache"
set "CACHE_DIR=%RECAP_DATA%\cache"
set "STATIC_FFMPEG_CACHE_DIR=%RECAP_DATA%\cache\static-ffmpeg"
set "HF_HOME=%RECAP_DATA%\cache\huggingface"
set "WHISPER_CACHE_DIR=%RECAP_DATA%\cache\whisper"
set "RECAP_LOG_DIR=%RECAP_DATA%"
set "OUTPUT_DIR=%RECAP_OUTPUT%"
set "XDG_CACHE_HOME=%RECAP_DATA%\cache\xdg"
set "TORCH_HOME=%RECAP_DATA%\cache\torch"
set "PYTHONPYCACHEPREFIX=%RECAP_DATA%\cache\pycache"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo  Data Directory   : %RECAP_DATA%
echo  Output Directory : %RECAP_OUTPUT%
echo  (Packages, caches, temp, models and videos ALL stay on %RECAP_DRIVE% -- nothing is written to C:)
echo.

:: ---------------------------------------------------------
:: 1. Find Python interpreter
:: ---------------------------------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)

if not defined PY (
    for %%D in (F: C: D:) do (
        for %%V in (Python313 Python312 Python311 Python310 Python39 Python) do (
            if exist "%%D\%%V\python.exe" set "PY=%%D\%%V\python.exe"
            if exist "%%D\Program Files\%%V\python.exe" set "PY=%%D\Program Files\%%V\python.exe"
        )
    )
)
if not defined PY (
    if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
)

if not defined PY (
    echo  [X] Python was not found on PATH or standard directories.
    echo      Please install Python 3.10+ from https://www.python.org/downloads/
    echo      IMPORTANT: Check the box "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

%PY% -c "import sys;print('  [OK] Python %%d.%%d.%%d detected at: ' %% sys.version_info[:3], sys.executable)"
if errorlevel 1 (
    echo  [X] Python failed to run.
    pause
    exit /b 1
)
echo.

:: ---------------------------------------------------------
:: 1b. Virtualenv ON F: -- the base Python install stays where
::     Windows put it (its .exe is small), but EVERY installed
::     package (faster-whisper, torch, ... gigabytes) goes into
::     this venv on F:, never into C:\...\Lib\site-packages.
:: ---------------------------------------------------------
set "VENV=%RECAP_DATA%\venv"
if not exist "%VENV%\Scripts\python.exe" (
    echo  Creating virtual environment on %RECAP_DRIVE% : %VENV%
    %PY% -m venv "%VENV%"
    if errorlevel 1 (
        echo  [X] Could not create the virtual environment on %RECAP_DRIVE%
        pause
        exit /b 1
    )
    "%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
)
set "PY=%VENV%\Scripts\python.exe"
echo  [OK] Using venv Python (ALL packages install on %RECAP_DRIVE%): %PY%


:: ---------------------------------------------------------
:: 2. Check and install core dependencies
:: ---------------------------------------------------------
echo  Checking Python dependencies...
set "MISSING="
%PY% -c "import yaml"           >nul 2>&1 || set "MISSING=!MISSING! PyYAML"
%PY% -c "import pysubs2"        >nul 2>&1 || set "MISSING=!MISSING! pysubs2"
%PY% -c "import edge_tts"       >nul 2>&1 || set "MISSING=!MISSING! edge-tts"
%PY% -c "import static_ffmpeg"  >nul 2>&1 || set "MISSING=!MISSING! static-ffmpeg"
%PY% -c "import requests"       >nul 2>&1 || set "MISSING=!MISSING! requests"
%PY% -c "import openai"         >nul 2>&1 || set "MISSING=!MISSING! openai"
%PY% -c "import scenedetect"    >nul 2>&1 || set "MISSING=!MISSING! scenedetect[opencv]"
%PY% -c "import faster_whisper" >nul 2>&1 || set "MISSING=!MISSING! faster-whisper"
%PY% -c "import vastai"         >nul 2>&1 || set "MISSING=!MISSING! vastai"
%PY% -c "import numpy"          >nul 2>&1 || set "MISSING=!MISSING! numpy"
%PY% -c "import moviepy"        >nul 2>&1 || set "MISSING=!MISSING! moviepy"

if defined MISSING (
    echo  Installing missing dependencies: !MISSING!
    %PY% -m pip install !MISSING!
    if errorlevel 1 (
        echo  Retrying installation with --no-warn-script-location...
        %PY% -m pip install --no-warn-script-location !MISSING!
    )
) else (
    echo  [OK] Core dependencies installed: PyYAML, pysubs2, edge-tts, static-ffmpeg, openai, faster-whisper, scenedetect, vastai, numpy, moviepy
)
echo.

:: WhisperX / Coqui XTTS are OPTIONAL (alignment + an alternate voice).
:: They are NOT auto-installed or imported here any more: both drag in
:: torch/torchaudio, and a version-mismatched pair crashes python.exe
:: with "Entry Point Not Found (libtorchaudio.pyd)" -- an OS loader
:: error that happens *during import*, so even this check used to kill
:: the setup. The default pipeline (edge-tts + faster-whisper) needs
:: neither. We only read version metadata (loads no DLLs) and warn.
echo  Checking torch stack (WhisperX / XTTS are optional)...
%PY% recap-studio\tools\torch_check.py
if errorlevel 1 (
    echo  [!] The optional WhisperX/XTTS engines are disabled until this
    echo      is fixed -- run fix_torchaudio.bat. Everything else works.
)
echo  [NOTE] To enable WhisperX alignment later: fix_torchaudio.bat (option 1).
echo.

:: ---------------------------------------------------------
:: 3. Verify ffmpeg / ffprobe
:: ---------------------------------------------------------
%PY% recap-studio\tools\ensure_ffmpeg.py
if errorlevel 1 (
    echo  [!] ffmpeg setup notice - if needed, static-ffmpeg will complete download on first run.
)
echo.

:: ---------------------------------------------------------
:: 4. Verify project layout
:: ---------------------------------------------------------
if not exist "recap-studio\app.py" (
    echo  [X] recap-studio\app.py not found. Please run this batch file from the repository root.
    pause
    exit /b 1
)

:: ---------------------------------------------------------
:: 5. Port check - restart if already running
:: ---------------------------------------------------------
%PY% recap-studio\tools\portcheck.py %PORT% >nul 2>&1
if not errorlevel 1 (
    echo  [..] An instance is already running on port %PORT%.
    echo       Restarting instance with updated code...
    %PY% recap-studio\tools\shutdown.py %PORT% >nul 2>&1
)

:: ---------------------------------------------------------
:: 6. Launch Recap Studio
:: ---------------------------------------------------------
echo.
echo  Starting Recap Studio at %URL%
echo  Keep this console window open while using the panel.
echo  Press Ctrl+C here or click "Close Studio" in the web panel to stop.
echo  -------------------------------------------------------------
echo.

%PY% recap-studio\app.py --port %PORT% --open-browser
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo  Recap Studio has closed normally.
) else (
    echo  Recap Studio exited with return code: %RC%
)
echo.
pause
exit /b 0
