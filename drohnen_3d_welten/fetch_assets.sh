#!/usr/bin/env bash
# Lädt die aus dem Internet verwendeten Vorlagen (three.js-Beispielassets, MIT/CC) nach ./assets
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets
B=https://raw.githubusercontent.com/mrdoob/three.js/dev/examples
for f in textures/hardwood2_diffuse.jpg textures/hardwood2_bump.jpg textures/hardwood2_roughness.jpg \
         textures/terrain/grasslight-big.jpg \
         models/gltf/LeePerrySmith/LeePerrySmith.glb; do
  curl -fsSL -o "assets/$(basename "$f")" "$B/$f"
done
echo "Assets geladen."
