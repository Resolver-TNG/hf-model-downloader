#!/usr/bin/env bash
# HF model downloader GUI - macOS launcher (double-click in Finder, or run in Terminal).
# Creates .venv on first run, installs requirements, starts the server and opens the browser.
#
# First time after downloading the zip, macOS may block the double-click ("cannot be opened").
# Either right-click the file > Open, or run once in Terminal:
#     xattr -dr com.apple.quarantine <this folder>
#     chmod +x start_gui_mac.command
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
# Finder-launched shells can miss Homebrew's directories
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
VENV="$HERE/.venv"

echo "== HF model downloader GUI (macOS) =="

fail() {
  echo
  echo "ERROR: $*" >&2
  echo
  read -r -p "Press Enter to close this window ... " _ || true
  exit 1
}

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

# remove the quarantine flag so later runs are not blocked (harmless if it is not set)
xattr -dr com.apple.quarantine "$HERE" 2>/dev/null || true

VPY="$(venv_python)"
if [ -x "$VPY" ] && ! "$VPY" -c 'import sys' >/dev/null 2>&1; then
  echo "Existing .venv is not usable on this machine (copied from another OS?), recreating it ..."
  rm -rf "$VENV"
fi

VPY="$(venv_python)"
if [ ! -x "$VPY" ]; then
  PY="$(find_python)" || fail "Python 3.10 or newer was not found. Install it with 'brew install python@3.12' or from https://www.python.org/downloads/macos/ , then run this file again. (The python3 that ships with the Xcode command line tools is too old.)"
  echo "Creating virtualenv with: $PY ($("$PY" --version 2>&1))"
  "$PY" -m venv "$VENV" || fail "could not create the virtualenv."
  VPY="$(venv_python)"
fi

if ! "$VPY" -c 'import huggingface_hub' >/dev/null 2>&1; then
  echo "Installing requirements (internet access to PyPI is needed) ..."
  "$VPY" -m pip install -q -r "$HERE/requirements.txt" || fail "pip install failed."
fi

echo "Starting the GUI. The browser opens automatically. Press Ctrl+C or close this window to stop."
echo
"$VPY" "$HERE/gui_server.py" "$@"
rc=$?
if [ $rc -ne 0 ]; then
  fail "the GUI server exited with code $rc."
fi
