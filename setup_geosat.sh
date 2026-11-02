#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

command -v python3 >/dev/null || {
  echo "Python 3.10+ is required." >&2
  exit 1
}
command -v npm >/dev/null || {
  echo "Node.js 18+ and npm are required." >&2
  exit 1
}

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt -r api/requirements.txt
(cd frontend && npm ci)

echo
echo "GEOSAT setup complete. Run ./run_geosat.sh or build with ./build_geosat.sh."
