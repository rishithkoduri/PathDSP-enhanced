@echo off
REM Final validation plan (v6.0): steps 1 -> 2 -> 3 -> 4 -> finalise -> train-on-everything models.
REM Safe to close and start again: it continues where it stopped.
REM   final_plan.bat            run / continue the plan
REM   final_plan.bat --plan     show every run's settings (trains nothing)
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist .venv\Scripts\python.exe (
  echo Run start.bat first (it sets everything up).
  pause
  exit /b 1
)
.venv\Scripts\python -u tools\final_plan.py %*
pause
