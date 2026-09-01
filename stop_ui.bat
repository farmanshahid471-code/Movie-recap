@echo off
setlocal EnableExtensions
title Recap Studio - Close

:: =========================================================
::  Recap Studio - one click to CLOSE.
::
::  Asks the running panel to shut itself down (which also
::  cancels a render in progress), then waits for the port to
::  be released. No pause at the end: the window closes itself.
:: =========================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

if "%PORT%"=="" set "PORT=8080"

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)

echo.
echo  Closing Recap Studio on port %PORT% ...
echo.

if not defined PY (
    echo  [X] Python was not found - cannot ask the server to stop.
    echo      Close the Recap Studio window manually.
    goto :end
)

%PY% recap-studio\tools\portcheck.py %PORT% >nul 2>&1
if errorlevel 1 (
    echo  [i] Nothing is listening on port %PORT% - Recap Studio is not running.
    goto :end
)

%PY% recap-studio\tools\shutdown.py %PORT%
if errorlevel 1 (
    echo  [!] The panel did not finish shutting down.
    echo      Close the Recap Studio window manually, or end python.exe
    echo      in Task Manager.
) else (
    echo  [OK] Recap Studio has been stopped. You can close its window now.
)

:end
echo.
endlocal
timeout /t 3 /nobreak >nul
exit /b 0
