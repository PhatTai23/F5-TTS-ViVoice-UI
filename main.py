"""F5-TTS Vietnamese Web UI entry point.

Usage (no activate / no manual .env export needed):
    python main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _venv_python() -> Path | None:
    if sys.platform.startswith("win"):
        candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _in_project_venv() -> bool:
    """True when this process is already the repo .venv interpreter."""
    try:
        return Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    except OSError:
        return False


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE from .env into os.environ (does not override existing)."""
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Warning: cannot read {path}: {exc}", file=sys.stderr)
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ[key] = value


def _ensure_venv() -> None:
    """Re-exec under repo .venv. Do not resolve() the binary — that breaks venv."""
    venv_py = _venv_python()
    if venv_py is None:
        print(
            "Missing .venv. Create it first (see web/README.md), e.g.\n"
            "  python3.11 -m venv .venv && .venv/bin/pip install -e . -r web/requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)
    if _in_project_venv():
        return
    os.environ["VIRTUAL_ENV"] = str(ROOT / ".venv")
    print(f"Using venv: {venv_py}", flush=True)
    # Keep the .venv path in argv[0] so Python finds pyvenv.cfg
    os.execv(str(venv_py), [str(venv_py), str(ROOT / "main.py"), *sys.argv[1:]])


def main() -> None:
    import uvicorn

    host = os.environ.get("F5TTS_HOST", "127.0.0.1")
    port = int(os.environ.get("F5TTS_PORT", "7860"))
    device = os.environ.get("F5TTS_DEVICE") or "(auto)"
    print(f"Starting F5-TTS Web UI on http://{host}:{port}")
    print(f"Python: {sys.executable}")
    print(f"Device: {device}")
    # workers=1: keep CUDA/MPS model in a single process
    uvicorn.run("web.server:app", host=host, port=port, reload=False, workers=1)


if __name__ == "__main__":
    _load_dotenv(ROOT / ".env")
    _ensure_venv()
    main()
