#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
rm -rf "$ROOT/dist/geosat" "$ROOT/dist/geosat.exe"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Local environment not found. Create it with: python3 -m venv .venv" >&2
  exit 1
fi
command -v npm >/dev/null || {
  echo "Node.js and npm are required to build the dashboard." >&2
  exit 1
}

.venv/bin/python -m pip install -r build-requirements.txt
(cd frontend && npm ci && npm run build)
.venv/bin/python -m PyInstaller --noconfirm --clean geosat.spec

echo
echo "Build complete: dist/geosat/geosat"
echo "Run it with: ./dist/geosat/geosat"
