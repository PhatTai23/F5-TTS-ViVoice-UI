#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv. Create it first (see web/README.md)."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

export F5TTS_HOST="${F5TTS_HOST:-127.0.0.1}"
export F5TTS_PORT="${F5TTS_PORT:-7860}"
# Auto device in server; set F5TTS_DEVICE=cuda:0 on Linux CUDA boxes if needed
exec python main.py

