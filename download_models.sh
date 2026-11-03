#!/usr/bin/env bash
# CLI downloader (no GUI). Pass-through to download_models.py, see its --help.
#   ./download_models.sh --dry-run --group core4 --dest /Volumes/MODELS/hf_models
#   HF_TOKEN=hf_xxx ./download_models.sh --group core4 --dest /Volumes/MODELS/hf_models
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Note: no `cd` here, so a relative --dest is resolved against YOUR current directory.
if [[ -x "$HERE/.venv/bin/python" ]]; then
  PYTHON_BIN="$HERE/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if ! "$PYTHON_BIN" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "huggingface_hub is not installed. Run ./start_gui_linux.sh (or start_gui_mac.command) once to create .venv, or:" >&2
  echo "  $PYTHON_BIN -m pip install -r \"$HERE/requirements.txt\"   (inside a virtualenv)" >&2
  exit 3
fi
exec "$PYTHON_BIN" "$HERE/download_models.py" "$@"
