@echo off
REM PathDSP-X launcher: checks the setup, installs anything missing, verifies, then opens the app.
REM Double-click it. Safe to run every time: when nothing changed since the last full check it starts in seconds.
REM   start.bat full     run the full check again (GPU, PyTorch, every package import)
setlocal EnableExtensions
cd /d "%~dp0"
title PathDSP-X
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PIP_DISABLE_PIP_VERSION_CHECK=1
set "PY=.venv\Scripts\python.exe"

echo.
echo  ==============================================================
echo    PathDSP-X  ^|  Explainable drug sensitivity prediction
echo  ==============================================================
echo.
echo  [1/4] Python environment
if exist "%PY%" (
  echo   [ OK ] virtual environment     .venv
  goto :packages
)
set "SYS_PY="
py -3 --version >nul 2>&1 && set "SYS_PY=py -3"
if not defined SYS_PY python --version >nul 2>&1 && set "SYS_PY=python"
if not defined SYS_PY (
  echo   [MISS] Python 3.10 or newer was not found - installing it quietly with winget ...
  winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
  echo.
  echo   Python was installed. Close this window and double-click start.bat again.
  pause
  exit /b 1
)
echo   [....] creating virtual environment .venv ...
%SYS_PY% -m venv .venv || goto :fail
"%PY%" -m pip install -q --upgrade pip >nul 2>&1
echo   [ OK ] virtual environment     .venv created

:packages
"%PY%" tools\setup_env.py %*
if errorlevel 1 goto :fail

echo.
echo  --------------------------------------------------------------
echo    All checks passed - opening PathDSP-X in your browser ...
echo    Keep this window open while you use the app. Close it to stop.
echo  --------------------------------------------------------------
echo.
"%PY%" -m streamlit run app.py
exit /b 0

:fail
echo.
echo  Setup did not finish. Copy the messages above and send them.
pause
exit /b 1
