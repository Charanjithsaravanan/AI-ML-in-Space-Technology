#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTABLE="$ROOT/dist/geosat/geosat"

if [[ ! -x "$EXECUTABLE" ]]; then
  echo "Executable not found. Build it first with: ./build_geosat.sh" >&2
  exit 1
fi

exec "$EXECUTABLE"
