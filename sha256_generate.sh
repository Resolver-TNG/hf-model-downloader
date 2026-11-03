#!/usr/bin/env bash
# Write <root>/sha256sums.txt (compatible with `sha256sum -c`). Thin wrapper over make_checksums.py,
# which hashes in parallel and skips the .cache resume folders at every depth.
#   ./sha256_generate.sh /Volumes/MODELS/hf_models [output.txt]
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-./hf_models}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
[[ -x "$HERE/.venv/bin/python" ]] && PYTHON_BIN="$HERE/.venv/bin/python"
if [[ $# -ge 2 ]]; then
  exec "$PYTHON_BIN" "$HERE/make_checksums.py" generate "$ROOT" --sums "$2"
fi
exec "$PYTHON_BIN" "$HERE/make_checksums.py" generate "$ROOT"
