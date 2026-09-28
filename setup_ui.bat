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
:: 0. Storage drive configuration -- use F: drive if available
:: ---------------------------------------------------------
set "TARGET_DRIVE="
if exist "F:\" (
    set "TARGET_DRIVE=F:"
) else if exist "D:\" (
    set "TARGET_DRIVE=D:"
) else (
    set "TARGET_DRIVE=%ROOT%"
)

if "%TARGET_DRIVE%"=="%ROOT%" (
    set "RECAP_DATA=%ROOT%\.recap-data"
    set "RECAP_OUTPUT=%ROOT%\output"
) else (
    set "RECAP_DATA=%TARGET_DRIVE%\recap-data"
    set "RECAP_OUTPUT=%TARGET_DRIVE%\recap"
)

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

echo  Data Directory   : %RECAP_DATA%
echo  Output Directory : %RECAP_OUTPUT%
echo  (All caches, temporary files, and video outputs stay off the C: drive)
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
        for %%V in (Python312 Python311 Python310 Python39 Python) do (
            if exist "%%D\%%V\python.exe" set "PY=%%D\%%V\python.exe"
            if exist "%%D\Program Files\%%V\python.exe" set "PY=%%D\Program Files\%%V\python.exe"
        )
    )
)
if not defined PY (
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
:: 2. Check and install core dependencies
:: ---------------------------------------------------------
echo  Checking Python dependencies...
set "MISSING="
%PY% -c "import yaml"           >nul 2>&1 || set "MISSING=!MISSING! PyYAML"
%PY% -c "import pysubs2"        >nul 2>&1 || set "MISSING=!MISSING! pysubs2"
%PY% -c "import edge_tts"       >nul 2>&1 || set "MISSING=!MISSING! edge-tts"
%PY% -c "import static_ffmpeg"  >nul 2>&1 || set "MISSING=!MISSING! static-ffmpeg"
%PY% -c "import openai"         >nul 2>&1 || set "MISSING=!MISSING! openai"
%PY% -c "import scenedetect"    >nul 2>&1 || set "MISSING=!MISSING! scenedetect[opencv]"
%PY% -c "import vastai"         >nul 2>&1 || set "MISSING=!MISSING! vastai"

if defined MISSING (
    echo  Installing missing dependencies: !MISSING!
    %PY% -m pip install --upgrade pip >nul 2>&1
    %PY% -m pip install !MISSING!
    if errorlevel 1 (
        echo  Retrying installation with --no-warn-script-location...
        %PY% -m pip install --no-warn-script-location !MISSING!
    )
) else (
    echo  [OK] Core dependencies installed: PyYAML, pysubs2, edge-tts, static-ffmpeg, openai, scenedetect, vastai
)
echo.

echo  Checking forced alignment engine - WhisperX...
%PY% -c "import whisperx" >nul 2>&1
if errorlevel 1 (
    echo  Installing whisperx for frame-accurate phoneme alignment...
    %PY% -m pip install whisperx>=3.1.0
    if errorlevel 1 (
        echo  [NOTE] whisperx install had issues. The pipeline will automatically use faster-whisper fallback.
    ) else (
        echo  [OK] whisperx installed successfully.
    )
) else (
    echo  [OK] whisperx ready.
)
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
