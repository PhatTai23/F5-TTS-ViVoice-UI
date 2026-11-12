@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

if not exist ".venv\Scripts\activate.bat" (
  echo Chua co .venv. Chay web\setup_windows.bat truoc.
  exit /b 1
)

call ".venv\Scripts\activate.bat"

REM Force CUDA when available; override: set F5TTS_DEVICE=cpu
if "%F5TTS_DEVICE%"=="" set F5TTS_DEVICE=cuda:0
if "%F5TTS_HOST%"=="" set F5TTS_HOST=127.0.0.1
if "%F5TTS_PORT%"=="" set F5TTS_PORT=7860

echo Starting F5-TTS Web UI on http://%F5TTS_HOST%:%F5TTS_PORT%  device=%F5TTS_DEVICE%
python main.py
endlocal

