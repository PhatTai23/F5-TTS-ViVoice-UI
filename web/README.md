# F5-TTS Vietnamese Custom Web UI

Giao diện HTML + FastAPI để chạy model custom (ViVoice), cắt `gen_text` theo xuống dòng, chèn khoảng nghỉ giữa các chunk, xem audio/phổ âm và tải WAV.

Hỗ trợ:

- **macOS** (MPS/CPU) — Python 3.10/3.11
- **Windows** — **Python 3.10 + CUDA** (khuyến nghị)

## Yêu cầu chung

- `ffmpeg` trong PATH ([Windows build](https://www.gyan.dev/ffmpeg/builds/) hoặc `winget install ffmpeg`)
- Checkpoint đặt trong `model/` (ví dụ `model/F5-TTS-Vietnamese-ViVoice/`)
- Model: [hynt/F5-TTS-Vietnamese-ViVoice](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice)

## Windows (Python 3.10 + CUDA) — nhanh

Trong **Command Prompt** / PowerShell tại thư mục gốc repo:

```bat
web\setup_windows.bat
web\start_windows.bat
```

`setup_windows.bat` sẽ:

1. Tạo `.venv`
2. Cài **PyTorch CUDA** (`cu124` mặc định, hoặc `121` / `cpu`)
3. `pip install -e .` + deps web
4. Tải ViVoice vào `model/`

Mở: http://127.0.0.1:7860

### Setup Windows thủ công

```bat
cd path\to\f5-tts
py -3.10 -m venv .venv
.venv\Scripts\activate

python -m pip install -U pip setuptools wheel

REM CUDA 12.4 (đổi cu121 nếu driver cũ hơn)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -e .
pip install -r web\requirements.txt
python web\download_model.py

set F5TTS_DEVICE=cuda:0
python main.py
REM hoặc: .venv\Scripts\python.exe main.py
```



Kiểm tra CUDA:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Nếu `cuda=False` dù có GPU: đang cài nhầm bản CPU — gỡ rồi cài lại bằng `--index-url` CUDA ở trên.

## macOS / Linux

```bash
cd /path/to/f5-tts
python3.10 -m venv .venv   # hoặc python3.11
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install torch torchaudio
pip install -e .
pip install -r web/requirements.txt
python web/download_model.py

# Chạy — không cần activate .venv
python main.py
```

`main.py` sẽ:

1. Đọc file `.env` ở root (nếu có)
2. Tự chuyển sang Python trong `.venv` (không cần `activate`)

Tạo config nhanh: `cp .env.example .env`

## Biến môi trường

Có thể set trong shell **hoặc** trong `.env`:

| Biến | Ý nghĩa | Ví dụ |
|------|---------|--------|
| `F5TTS_DEVICE` | Ép device | `cuda:0`, `cpu`, `mps` |
| `F5TTS_CUDA_INDEX` | GPU index nếu không set DEVICE | `0` |
| `F5TTS_HOST` | Bind host | `127.0.0.1` hoặc `0.0.0.0` |
| `F5TTS_PORT` | Port | `7860` |

Trên Windows script start mặc định `F5TTS_DEVICE=cuda:0`.


## Cách dùng UI

1. Đặt checkpoint/vocab vào `model/` → UI tự quét thành **select**
2. **Run Service** → load model (CUDA/MPS/CPU tự nhận)
3. Upload ref audio + ref text (+ có thể **Lưu mẫu** vào localStorage để tái sử dụng)
4. Nhập `gen_text` (mỗi dòng = 1 chunk)
5. Chỉnh speed / nghỉ / NFE → **Synthesize** → nghe + phổ âm + tải WAV


## API

- `GET /api/status` — có thêm `device_info` (GPU name, CUDA version, platform)
- `GET /api/models` — liệt kê ckpt/vocab trong `model/`
- `POST /api/service/start`
- `POST /api/service/stop`
- `POST /api/infer`

Output: `web/outputs/`

## Ghi chú Windows + CUDA

- Luôn chạy với `--workers 1` (tránh load model CUDA nhiều process)
- Path trong select dùng dạng `model/...` (posix) — server chấp nhận cả `\` và `/`
- Cần driver NVIDIA tương thích bản CUDA của PyTorch wheel
- Nếu OOM: giảm `nfe_step`, rút ngắn ref audio, hoặc `set F5TTS_DEVICE=cpu`
