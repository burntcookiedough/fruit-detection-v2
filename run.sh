#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_MODE="${FRUIT_RUN_MODE:-image}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python was not found. Set PYTHON_BIN or install Python 3.10+." >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

required = ["torch", "torchvision", "PIL", "timm"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("ERROR: Missing Python packages: " + ", ".join(missing), file=sys.stderr)
    print("Install dependencies with: python -m pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)
PY

case "$RUN_MODE" in
  image|inference)
    exec "$PYTHON_BIN" run_inference.py "$@"
    ;;
  webcam)
    exec "$PYTHON_BIN" webcam_inference.py "$@"
    ;;
  train)
    exec "$PYTHON_BIN" train.py "$@"
    ;;
  verify)
    exec "$PYTHON_BIN" verify.py "$@"
    ;;
  export)
    exec "$PYTHON_BIN" export.py "$@"
    ;;
  *)
    echo "ERROR: Unsupported FRUIT_RUN_MODE='$RUN_MODE'." >&2
    echo "Supported modes: image, webcam, train, verify, export." >&2
    exit 1
    ;;
esac
