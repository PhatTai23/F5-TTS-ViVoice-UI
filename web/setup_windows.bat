@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

echo === F5-TTS Web UI setup (Windows / Python 3.10 + CUDA) ===
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Khong tim thay python trong PATH. Cai Python 3.10 roi thu lai.
  exit /b 1
)

python -c "import sys; assert sys.version_info[:2]==(3,10), f'Can Python 3.10, hien tai {sys.version}'" 2>nul
if errorlevel 1 (
  echo [WARN] Khuyen nghi Python 3.10. Tiep tuc voi python hien tai...
  python -V
)

if not exist ".venv\Scripts\python.exe" (
  echo Tao venv .venv ...
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"

python -m pip install -U pip setuptools wheel

echo.
echo Cai PyTorch CUDA. Neu GPU cua ban dung CUDA 12.4:
echo   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
echo Neu CUDA 12.1:
echo   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.
set /p CUDA_CHOICE="Chon CUDA [124/121/cpu] (mac dinh 124): "
if "%CUDA_CHOICE%"=="" set CUDA_CHOICE=124
if /I "%CUDA_CHOICE%"=="cpu" (
  pip install torch torchaudio
) else if /I "%CUDA_CHOICE%"=="121" (
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
)

pip install -e .
pip install -r web\requirements.txt
pip install "numpy>=2.0,<2.3" "pyarrow>=24"

echo.
echo Tai model ViVoice vao model\ ...
python web\download_model.py

echo.
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

echo.
echo Xong. Chay: web\start_windows.bat
endlocal
