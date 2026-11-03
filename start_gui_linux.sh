#!/usr/bin/env bash
# HF model downloader GUI - Linux launcher.
# Creates .venv on first run, installs requirements, starts the server and opens the browser.
#
#   ./start_gui_linux.sh                              normal start (opens a browser if a desktop is present)
#   ./start_gui_linux.sh --no-browser                 never open a browser
#   ./start_gui_linux.sh --port 9000 --default-dest /data/hf_models
#
# On a server without a desktop (for example a remote GPU server over SSH) no browser is opened.
# Forward the port from your own PC and open the printed URL there:
#   ssh -L 8765:127.0.0.1:8765 user@host
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
VENV="$HERE/.venv"

find_python() {
  local c
  for c in "${PYTHON_BIN:-}" python3.13 python3.12 python3.11 python3.10 python3 python; do
    [ -n "$c" ] || continue
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' >/dev/null 2>&1; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

venv_python() {
  if [ -x "$VENV/bin/python" ]; then echo "$VENV/bin/python"; else echo "$VENV/Scripts/python.exe"; fi
}

echo "== HF model downloader GUI (Linux) =="

VPY="$(venv_python)"
if [ -x "$VPY" ] && ! "$VPY" -c 'import sys' >/dev/null 2>&1; then
  echo "Existing .venv is not usable on this machine (copied from another OS?), recreating it ..."
  rm -rf "$VENV"
fi

VPY="$(venv_python)"
if [ ! -x "$VPY" ]; then
  if ! PY="$(find_python)"; then
    echo "ERROR: Python 3.10 or newer was not found." >&2
    echo "  Debian/Ubuntu: sudo apt install python3 python3-venv python3-pip" >&2
    echo "  RHEL/Fedora  : sudo dnf install python3 python3-pip" >&2
    exit 1
  fi
  echo "Creating virtualenv with: $PY ($("$PY" --version 2>&1))"
  if ! "$PY" -m venv "$VENV"; then
    echo "ERROR: could not create the virtualenv." >&2
    echo "  On Debian/Ubuntu install the venv module first: sudo apt install python3-venv" >&2
    rm -rf "$VENV"
    exit 1
  fi
  VPY="$(venv_python)"
fi

if ! "$VPY" -c 'import huggingface_hub' >/dev/null 2>&1; then
  echo "Installing requirements (internet access to PyPI is needed) ..."
  "$VPY" -m pip install -q -r "$HERE/requirements.txt"
fi

echo "Starting the GUI. Press Ctrl+C to stop."
echo
exec "$VPY" "$HERE/gui_server.py" "$@"
