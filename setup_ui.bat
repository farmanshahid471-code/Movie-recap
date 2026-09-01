@echo off
title Recap Studio Setup & UI Launcher
echo.
echo ==========================================
echo Recap Studio Control Panel Setup & Launch
echo ==========================================
echo.

:: -------------------------------------------------------
:: Step 1: Install/verify Python dependencies
:: -------------------------------------------------------
echo Checking Python dependencies...

:: PyYAML (required by the pipeline)
if not python3 -c "import yaml" 2>&1 | find "no module" >nul 2>&1 then
    echo Installing PyYAML...
    python3 -m pip install PyYAML -q 2>nul || echo WARNING: PyYAML could not be installed automatically.
else
    echo PyYAML already available.
fi

:: edge-tts (free TTS, English + Chinese)
if not python3 -c "import edge_tts" 2>&1 | find "no module" >nul 2>&1 then
    echo Installing edge-tts...
    python3 -m pip install edge-tts -q 2>nul || echo WARNING: edge-tts could not be installed automatically.
else
    echo edge-tts already available.
fi

:: static-ffmpeg (bundled ffmpeg/ffprobe)
if not python3 -c "import static_ffmpeg" 2>&1 | find "no module" >nul 2>&1 then
    echo Installing static-ffmpeg...
    python3 -m pip install static-ffmpeg -q 2>nul || echo WARNING: static-ffmpeg could not be installed automatically.
else
    echo static-ffmpeg already available.
fi

:: pysubs2 (subtitle processing)
if not python3 -c "import pysubs2" 2>&1 | find "no module" >nul 2>&1 then
    echo Installing pysubs2...
    python3 -m pip install pysubs2 -q 2>nul || echo WARNING: pysubs2 could not be installed automatically.
else
    echo pysubs2 already available.
fi
echo.

:: -------------------------------------------------------
:: Step 2: Check system ffmpeg/ffprobe
:: -------------------------------------------------------
echo.
echo Checking for ffmpeg/ffprobe...
if where ffmpeg >nul 2>&1 (
    echo ffmpeg found on system PATH.
) else (
    echo ffmpeg not found on system PATH.
    echo static-ffmpeg may provide internal ffmpeg.
)

:: Try to resolve ffmpeg via static_ffmpeg
python3 -c "
import shutil, os, sys
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass
p = shutil.which('ffmpeg')
if p:
    print('ffmpeg resolved via static_ffmpeg: ' + p)
else:
    print('ffmpeg not found via static_ffmpeg')
" 2>&1 | find "ffmpeg" && echo ffmpeg available || echo.

echo.

:: -------------------------------------------------------
:: Step 3: Verify project structure
:: -------------------------------------------------------
echo.
echo Checking project structure...
if not exist "recap-studio\runner.py" (
    echo ERROR: recap-studio\runner.py not found!
    echo Please run this batch from the Movie-recap directory.
    pause
    exit /b 1
)

if not exist "recap-studio\static\index.html" (
    echo ERROR: recap-studio\static\index.html not found!
    pause
    exit /b 1
)

echo Project structure looks good.
echo.

:: -------------------------------------------------------
:: Step 4: Start the UI server
:: -------------------------------------------------------
echo.
echo Starting Recap Studio Control Panel...
echo The server will start on http://localhost:8080
echo.
echo To stop: return to this window and press Ctrl+C
echo.

cd /d "%~dp0"
python recap-studio\runner.py

echo.
echo Recap Studio has been stopped.
echo.
echo Thank you for using Recap Studio!
pause