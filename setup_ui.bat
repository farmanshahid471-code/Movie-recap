@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Recap Studio - Setup and Open

:: =========================================================
::  Recap Studio - one click to OPEN.
::
::  * finds Python and installs any missing dependencies
::  * checks ffmpeg and the project layout
::  * starts the control panel and opens it in your browser
::
::  To CLOSE: click "Close Studio" in the panel, press Ctrl+C
::  in this window, or double-click stop_ui.bat.
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
:: 0. Data location -- keep EVERY byte off the C: drive
:: ---------------------------------------------------------
:: All outputs, model weights, ffmpeg binaries, pip cache and temp files go
:: under RECAP_DATA. Defaults to D:\recap-data when a D: drive exists; edit
:: RECAP_DATA below if your big drive has another letter.
set "RECAP_DATA=D:\recap-data"
if not exist "D:\nul" set "RECAP_DATA=%ROOT%\.recap-data"
mkdir "%RECAP_DATA%" 2>nul
if not exist "%RECAP_DATA%\" set "RECAP_DATA=%ROOT%\.recap-data"
mkdir "%RECAP_DATA%" 2>nul
mkdir "%RECAP_DATA%\cache" 2>nul
mkdir "%RECAP_DATA%\tmp"   2>nul
set "TEMP=%RECAP_DATA%\tmp"
set "TMP=%RECAP_DATA%\tmp"
set "PIP_CACHE_DIR=%RECAP_DATA%\pip-cache"
set "CACHE_DIR=%RECAP_DATA%\cache"
set "STATIC_FFMPEG_CACHE_DIR=%RECAP_DATA%\cache\static-ffmpeg"
set "HF_HOME=%RECAP_DATA%\cache\huggingface"
set "WHISPER_CACHE_DIR=%RECAP_DATA%\cache\whisper"
set "RECAP_LOG_DIR=%RECAP_DATA%"
echo  Data root : %RECAP_DATA%   (nothing is stored on the C: drive)

:: ---------------------------------------------------------
:: 1. Find a Python interpreter (py launcher first, then python)
:: ---------------------------------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo  [X] Python was not found on PATH.
    echo      Install Python 3.10+ from https://www.python.org/downloads/
    echo      and tick "Add python.exe to PATH" during setup.
    goto :fail
)

%PY% -c "import sys;print('  [OK] Python %%d.%%d.%%d' %% sys.version_info[:3])"
if errorlevel 1 (
    echo  [X] Python could not run. Reinstall it, or fix your PATH.
    goto :fail
)
echo.

:: ---------------------------------------------------------
:: 2. Install / verify the pipeline dependencies
:: ---------------------------------------------------------
echo  Checking Python dependencies...
set "MISSING="
%PY% -c "import yaml"           >nul 2>&1 || set "MISSING=!MISSING! PyYAML"
%PY% -c "import pysubs2"        >nul 2>&1 || set "MISSING=!MISSING! pysubs2"
%PY% -c "import edge_tts"       >nul 2>&1 || set "MISSING=!MISSING! edge-tts"
%PY% -c "import static_ffmpeg"  >nul 2>&1 || set "MISSING=!MISSING! static-ffmpeg"
%PY% -c "import openai"         >nul 2>&1 || set "MISSING=!MISSING! openai"

if not defined MISSING (
    echo  [OK] PyYAML, pysubs2, edge-tts, static-ffmpeg, openai
) else (
    echo  [..] Installing:!MISSING!
    %PY% -m pip install --upgrade pip >nul 2>&1
    %PY% -m pip install!MISSING!
    if errorlevel 1 (
        echo.
        echo  [X] pip could not install the dependencies. Try this by hand:
        echo      %PY% -m pip install PyYAML pysubs2 edge-tts static-ffmpeg openai
        goto :fail
    )
    echo  [OK] Dependencies installed.
)
echo.

:: ---------------------------------------------------------
:: 3. Verify ffmpeg / ffprobe (static-ffmpeg fetches them into the data root)
:: ---------------------------------------------------------
%PY% recap-studio\tools\ensure_ffmpeg.py
if not errorlevel 1 (
    echo  [OK] ffmpeg + ffprobe ready (binaries kept in %RECAP_DATA%, not C:).
) else (
    echo  [!] ffmpeg unavailable right now - setup continues; the first render
    echo      will fetch it (keep an internet connection available).
)
echo.

:: ---------------------------------------------------------
:: 4. Verify the project layout
:: ---------------------------------------------------------
if not exist "recap-studio\app.py" (
    echo  [X] recap-studio\app.py not found.
    echo      Run this file from the Movie-recap folder it shipped in.
    goto :fail
)
if not exist "recap-studio\static\index.html" (
    echo  [X] recap-studio\static\index.html not found.
    goto :fail
)
echo  [OK] Project structure looks good.
echo.

:: ---------------------------------------------------------
:: 5. Already running on this port? Then just open it.
:: ---------------------------------------------------------
%PY% recap-studio\tools\portcheck.py %PORT% >nul 2>&1
if not errorlevel 1 (
    echo  [OK] Recap Studio is already running - opening %URL%
    start "" "%URL%"
    goto :end
)

:: ---------------------------------------------------------
:: 6. Start the panel; it opens the browser itself when ready
:: ---------------------------------------------------------
echo  Starting Recap Studio on %URL%
echo  Keep this window open while you work.
echo  To stop: press Ctrl+C here, click "Close Studio" in the panel,
echo  or double-click stop_ui.bat.
echo  ------------------------------------------
echo.

%PY% recap-studio\app.py --port %PORT% --open-browser
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo  Recap Studio has been stopped. You can close this window.
) else (
    echo  Recap Studio exited with code %RC% - see the messages above.
    echo  Common fix: %PY% -m pip install PyYAML pysubs2 edge-tts static-ffmpeg openai
)
goto :end

:fail
echo.
echo  Setup did not finish - see the message above.

:end
echo.
endlocal
pause
exit /b 0
