#!/usr/bin/env bash
# Rendert eine Welt vollständig (fortsetzbar) und erzeugt danach das MP4.
# Nutzung: ./render_world.sh <world_script.py> <exr_out_dir> <post_params.py> <mp4> [extra args]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${NIDO_PY:-python3}"
WORLD="$1"; OUT="$2"; PARAMS="$3"; MP4="$4"; shift 4
"$PY" "$HERE/$WORLD" --out "$OUT" "$@"
"$PY" "$HERE/post.py" "$OUT" "$MP4" --params "$PARAMS" --fps "${NIDO_FPS:-24}"
