@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Vast.ai Movie File Uploader

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

echo.
echo  ======================================================
echo    Upload Movie to Vast.ai GPU Instance
echo  ======================================================
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
    echo  [X] Python was not found.
    pause
    exit /b 1
)

:: ---------------------------------------------------------
:: 2. Run Python Uploader
:: ---------------------------------------------------------
set "PYTHONPATH=%ROOT%\movie-recap-bot;%PYTHONPATH%"
%PY% -m recap.vast_upload %*

echo.
pause
exit /b 0
