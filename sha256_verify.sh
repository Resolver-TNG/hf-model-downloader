#!/usr/bin/env bash
# Verify <root> against its sha256sums.txt. Reports missing / mismatched / extra files.
# On a machine without Python you can still run:  cd <root> && sha256sum -c sha256sums.txt
#   ./sha256_verify.sh /Volumes/MODELS/hf_models [sums.txt]
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-./hf_models}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
[[ -x "$HERE/.venv/bin/python" ]] && PYTHON_BIN="$HERE/.venv/bin/python"
if [[ $# -ge 2 ]]; then
  exec "$PYTHON_BIN" "$HERE/make_checksums.py" verify "$ROOT" --sums "$2"
fi
exec "$PYTHON_BIN" "$HERE/make_checksums.py" verify "$ROOT"
