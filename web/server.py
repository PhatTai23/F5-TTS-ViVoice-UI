"""Custom F5-TTS Vietnamese web UI backend.

Loads a custom checkpoint/vocab, splits gen_text by newlines into chunks,
inserts pause silence between chunks, and returns WAV + spectrogram.
"""

from __future__ import annotations

import gc
import os
import platform
import sys
import threading
import traceback
import uuid
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

# Windows + CUDA friendly defaults
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if sys.platform.startswith("win"):
    # Avoid OpenMP / MKL contention quirks on Windows
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = WEB_DIR / "outputs"
CACHE_DIR = ROOT / ".cache"
NUMBA_CACHE = CACHE_DIR / "numba"
MPL_CACHE = CACHE_DIR / "matplotlib"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
MPL_CACHE.mkdir(parents=True, exist_ok=True)

# librosa/numba need a writable cache; home ~/.cache may be blocked in sandboxes
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

CKPT_EXTS = {".pt", ".pth", ".ckpt", ".safetensors"}
VOCAB_NAMES = {"vocab.txt", "vocab.json", "config.json"}
VOCAB_EXTS = {".txt", ".json"}

DEFAULT_CKPT = str(MODEL_DIR / "F5-TTS-Vietnamese-ViVoice" / "model_last.pt")
DEFAULT_VOCAB = str(MODEL_DIR / "F5-TTS-Vietnamese-ViVoice" / "vocab.txt")

app = FastAPI(title="F5-TTS Vietnamese Custom Web UI")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

_state_lock = threading.Lock()
_infer_lock = threading.Lock()

_state: dict[str, Any] = {
    "loaded": False,
    "loading": False,
    "error": None,
    "device": None,
    "model_name": "F5TTS_Base",
    "ckpt_file": DEFAULT_CKPT,
    "vocab_file": DEFAULT_VOCAB,
    "vocoder_name": "vocos",
    "ema_model": None,
    "vocoder": None,
    "progress": {
        "status": "idle",
        "current": 0,
        "total": 0,
        "message": "",
    },
}


class ServiceStartRequest(BaseModel):
    model_name: str = "F5TTS_Base"
    ckpt_file: str = Field(default=DEFAULT_CKPT)
    vocab_file: str = Field(default=DEFAULT_VOCAB)
    vocoder_name: str = "vocos"


def _detect_device(*, strict: bool = False) -> str:
    """Prefer CUDA on Windows/Linux; allow override via F5TTS_DEVICE."""
    override = (os.environ.get("F5TTS_DEVICE") or "").strip().lower()
    if override:
        if override.startswith("cuda") and not torch.cuda.is_available():
            msg = (
                f"F5TTS_DEVICE={override} nhưng CUDA không khả dụng. "
                "Cài PyTorch CUDA build (xem web/README.md)."
            )
            if strict:
                raise RuntimeError(msg)
            print(f"[warn] {msg} → fallback auto device")
        else:
            return override

    if torch.cuda.is_available():
        # Explicit cuda:0 is clearer for multi-GPU Windows boxes
        idx = int(os.environ.get("F5TTS_CUDA_INDEX", "0"))
        if idx < 0 or idx >= torch.cuda.device_count():
            idx = 0
        return f"cuda:{idx}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device_info(device: str | None = None) -> dict[str, Any]:
    device = device or _detect_device()
    info: dict[str, Any] = {
        "device": device,
        "platform": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            idx = int(device.split(":")[1]) if ":" in device else 0
        except ValueError:
            idx = 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 2)
    return info


def _configure_torch_backend(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # TF32 helps Ampere+ throughput for TTS-ish workloads
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True


def _set_progress(status: str, current: int = 0, total: int = 0, message: str = "") -> None:
    _state["progress"] = {
        "status": status,
        "current": current,
        "total": total,
        "message": message,
    }


def _resolve_path(path: str) -> Path:
    # Accept both model/foo.pt and model\foo.pt from Windows UI/API
    cleaned = (path or "").strip().replace("\\", "/")
    p = Path(cleaned).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _rel_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _scan_model_files() -> dict[str, list[dict[str, str]]]:
    """Scan ROOT/model for checkpoint and vocab files."""
    ckpts: list[dict[str, str]] = []
    vocabs: list[dict[str, str]] = []
    if not MODEL_DIR.is_dir():
        return {"ckpt_files": ckpts, "vocab_files": vocabs, "model_dir": str(MODEL_DIR)}

    for path in sorted(MODEL_DIR.rglob("*")):
        if not path.is_file():
            continue
        # skip huggingface cache / incomplete downloads
        parts = {p.lower() for p in path.parts}
        if ".cache" in parts or path.name.endswith(".incomplete") or path.name.endswith(".lock"):
            continue
        if path.name.startswith("."):
            continue

        rel = _rel_to_root(path)
        suffix = path.suffix.lower()
        if suffix in CKPT_EXTS:
            ckpts.append({"path": rel, "label": rel, "name": path.name})
        elif path.name.lower() in VOCAB_NAMES or (suffix in VOCAB_EXTS and "vocab" in path.name.lower()):
            vocabs.append({"path": rel, "label": rel, "name": path.name})

    return {
        "ckpt_files": ckpts,
        "vocab_files": vocabs,
        "model_dir": _rel_to_root(MODEL_DIR),
    }


def _unload_models() -> None:
    _state["ema_model"] = None
    _state["vocoder"] = None
    _state["loaded"] = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _load_service(
    model_name: str,
    ckpt_file: str,
    vocab_file: str,
    vocoder_name: str,
) -> None:
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT, UNetT  # noqa: F401

    ckpt_path = _resolve_path(ckpt_file)
    vocab_path = _resolve_path(vocab_file)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    device = _detect_device(strict=True)
    _configure_torch_backend(device)
    cfg_path = files("f5_tts").joinpath(f"configs/{model_name}.yaml")
    model_cfg = OmegaConf.load(str(cfg_path)).model
    model_cls = {"DiT": DiT, "UNetT": UNetT}[model_cfg.backbone]

    vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, device=device)
    ema_model = load_model(
        model_cls,
        model_cfg.arch,
        str(ckpt_path),
        mel_spec_type=vocoder_name,
        vocab_file=str(vocab_path),
        device=device,
    )

    _state.update(
        {
            "loaded": True,
            "loading": False,
            "error": None,
            "device": device,
            "model_name": model_name,
            # store resolved absolute path; compare via resolve() later
            "ckpt_file": str(ckpt_path),
            "vocab_file": str(vocab_path),
            "vocoder_name": vocoder_name,
            "ema_model": ema_model,
            "vocoder": vocoder,
        }
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    scanned = _scan_model_files()
    defaults = {
        "ckpt_file": _rel_to_root(Path(DEFAULT_CKPT)) if Path(DEFAULT_CKPT).exists() else (
            scanned["ckpt_files"][0]["path"] if scanned["ckpt_files"] else ""
        ),
        "vocab_file": _rel_to_root(Path(DEFAULT_VOCAB)) if Path(DEFAULT_VOCAB).exists() else (
            scanned["vocab_files"][0]["path"] if scanned["vocab_files"] else ""
        ),
    }
    return {**scanned, "defaults": defaults}


@app.get("/api/status")
def status() -> dict[str, Any]:
    models = list_models()
    device = _state["device"] or _detect_device()
    return {
        "loaded": _state["loaded"],
        "loading": _state["loading"],
        "error": _state["error"],
        "device": device,
        "device_info": _device_info(device),
        "model_name": _state["model_name"],
        "ckpt_file": _state["ckpt_file"],
        "vocab_file": _state["vocab_file"],
        "vocoder_name": _state["vocoder_name"],
        "defaults": models["defaults"],
        "ckpt_files": models["ckpt_files"],
        "vocab_files": models["vocab_files"],
        "model_dir": models["model_dir"],
        "progress": _state["progress"],
        "root": str(ROOT),
    }


@app.post("/api/service/start")
def service_start(req: ServiceStartRequest) -> dict[str, Any]:
    with _state_lock:
        if _state["loading"]:
            return {"ok": True, "message": "Service is already loading.", **status()}

        same = (
            _state["loaded"]
            and _state["ckpt_file"] == str(_resolve_path(req.ckpt_file))
            and _state["vocab_file"] == str(_resolve_path(req.vocab_file))
            and _state["model_name"] == req.model_name
            and _state["vocoder_name"] == req.vocoder_name
        )
        if same:
            return {"ok": True, "message": "Service already running with these paths.", **status()}

        _state["loading"] = True
        _state["error"] = None
        _set_progress("loading", 0, 1, "Loading model and vocoder...")

    try:
        with _infer_lock:
            _unload_models()
            _load_service(req.model_name, req.ckpt_file, req.vocab_file, req.vocoder_name)
        _set_progress("idle", 0, 0, "Model ready")
        return {"ok": True, "message": "Service started.", **status()}
    except Exception as exc:
        with _state_lock:
            _state["loading"] = False
            _state["loaded"] = False
            _state["error"] = str(exc)
        _set_progress("error", 0, 0, str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        with _state_lock:
            _state["loading"] = False


@app.post("/api/service/stop")
def service_stop() -> dict[str, Any]:
    with _infer_lock:
        _unload_models()
    _set_progress("idle", 0, 0, "Service stopped")
    return {"ok": True, "message": "Service stopped.", **status()}


def _split_lines(gen_text: str) -> list[str]:
    return [line.strip() for line in gen_text.splitlines() if line.strip()]


@app.post("/api/infer")
async def infer(
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(""),
    gen_text: str = Form(...),
    speed: float = Form(1.0),
    pause_sec: float = Form(0.3),
    nfe_step: int = Form(32),
    cross_fade_duration: float = Form(0.15),
    remove_silence: str = Form("false"),
    ckpt_file: str = Form(DEFAULT_CKPT),
    vocab_file: str = Form(DEFAULT_VOCAB),
    model_name: str = Form("F5TTS_Base"),
    vocoder_name: str = Form("vocos"),
) -> dict[str, Any]:
    remove_silence_flag = str(remove_silence).strip().lower() in {"1", "true", "yes", "on"}
    lines = _split_lines(gen_text)
    if not lines:
        raise HTTPException(status_code=400, detail="gen_text is empty (after splitting by newlines).")

    if not _state["loaded"]:
        # Auto-start with provided paths for convenience
        service_start(
            ServiceStartRequest(
                model_name=model_name,
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
                vocoder_name=vocoder_name,
            )
        )
    elif (
        str(_resolve_path(ckpt_file)) != _state["ckpt_file"]
        or str(_resolve_path(vocab_file)) != _state["vocab_file"]
        or model_name != _state["model_name"]
        or vocoder_name != _state["vocoder_name"]
    ):
        service_start(
            ServiceStartRequest(
                model_name=model_name,
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
                vocoder_name=vocoder_name,
            )
        )

    if not _infer_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another inference job is running.")

    tmp_ref = OUTPUT_DIR / f"ref_{uuid.uuid4().hex}{Path(ref_audio.filename or 'ref.wav').suffix or '.wav'}"
    try:
        from f5_tts.infer.utils_infer import (
            infer_process,
            preprocess_ref_audio_text,
            remove_silence_for_generated_wav,
            save_spectrogram,
            target_sample_rate,
        )

        content = await ref_audio.read()
        tmp_ref.write_bytes(content)

        _set_progress("running", 0, len(lines), "Preprocessing reference audio...")
        ref_path, ref_text_out = preprocess_ref_audio_text(str(tmp_ref), ref_text, show_info=print)

        waves: list[np.ndarray] = []
        specs: list[np.ndarray] = []
        sample_rate = target_sample_rate

        for idx, line in enumerate(lines, start=1):
            _set_progress("running", idx, len(lines), f"Generating chunk {idx}/{len(lines)}: {line[:60]}")
            wave, sample_rate, spec = infer_process(
                ref_path,
                ref_text_out,
                line,
                _state["ema_model"],
                _state["vocoder"],
                mel_spec_type=_state["vocoder_name"],
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                speed=speed,
                show_info=print,
            )
            if wave is None:
                raise RuntimeError(f"Failed to generate audio for chunk {idx}")
            waves.append(np.asarray(wave).reshape(-1))
            if spec is not None:
                specs.append(np.asarray(spec))

        pause_samples = max(0, int(float(pause_sec) * sample_rate))
        silence = np.zeros(pause_samples, dtype=np.float32) if pause_samples else None
        merged_parts: list[np.ndarray] = []
        for i, wave in enumerate(waves):
            merged_parts.append(wave.astype(np.float32, copy=False))
            if silence is not None and i < len(waves) - 1:
                merged_parts.append(silence)
        final_wave = np.concatenate(merged_parts) if merged_parts else np.zeros(0, dtype=np.float32)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = uuid.uuid4().hex[:8]
        wav_name = f"out_{stamp}_{job_id}.wav"
        png_name = f"out_{stamp}_{job_id}.png"
        wav_path = OUTPUT_DIR / wav_name
        png_path = OUTPUT_DIR / png_name

        sf.write(str(wav_path), final_wave, sample_rate)
        if remove_silence_flag:
            remove_silence_for_generated_wav(str(wav_path))
            final_wave, sample_rate = sf.read(str(wav_path))

        if specs:
            combined_spec = np.concatenate(specs, axis=1)
            save_spectrogram(combined_spec, str(png_path))
            spectrogram_url = f"/outputs/{png_name}"
        else:
            spectrogram_url = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _set_progress("done", len(lines), len(lines), "Done")
        return {
            "ok": True,
            "chunks": len(lines),
            "sample_rate": sample_rate,
            "duration_sec": float(len(final_wave) / sample_rate) if sample_rate else 0.0,
            "ref_text": ref_text_out,
            "audio_url": f"/outputs/{wav_name}",
            "spectrogram_url": spectrogram_url,
            "download_url": f"/outputs/{wav_name}",
            "device": _state.get("device"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        _set_progress("error", 0, 0, str(exc))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            if tmp_ref.exists():
                tmp_ref.unlink()
        except OSError:
            pass
        _infer_lock.release()


if __name__ == "__main__":
    # Prefer: python main.py  (from repo root)
    from main import main

    main()

