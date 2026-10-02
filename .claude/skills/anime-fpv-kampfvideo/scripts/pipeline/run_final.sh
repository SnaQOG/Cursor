#!/bin/bash
# Final-Kette (fortsetzbar): Szene bauen, falls noch kein blend existiert, dann rendern + encodieren.
#   WORK=... CODE=.../drohnen_3d_welten MODUL=world_xyz PARAMS=.../params/xyz.py \
#   RES=BxH SPP=n SIZE=BxH NAME=Video ./run_final.sh
# Python mit bpy im PATH (venv aktivieren). Logs: $WORK/final.log, $WORK/build.log
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
: "${WORK:?}" "${CODE:?}" "${MODUL:?}" "${PARAMS:?}" "${RES:?}" "${SPP:?}" "${SIZE:?}" "${NAME:?}"
mkdir -p "$WORK"
if [ ! -f "$WORK/$NAME.blend" ]; then
  echo "BUILD $(date -u +%FT%TZ)" >> "$WORK/final.log"
  python "$HERE/build_scene.py" "$CODE" "$MODUL" "$WORK/exr" "$WORK/$NAME.blend" "$RES" "$SPP" >> "$WORK/build.log" 2>&1
fi
echo "START $(date -u +%FT%TZ)" >> "$WORK/final.log"
python "$HERE/render_final.py" "$CODE" "$WORK/$NAME.blend" "$PARAMS" "$WORK/exr" "$WORK/$NAME.mp4" "$RES" "$SPP" "$SIZE" \
    >> "$WORK/final.log" 2>&1
echo "END $(date -u +%FT%TZ)" >> "$WORK/final.log"
