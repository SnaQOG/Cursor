#!/usr/bin/env bash
# Lädt die aus dem Internet verwendeten Vorlagen nach ./assets:
#  - three.js-Beispielassets (MIT; Kopf-Scan Lee Perry-Smith: Infinite-Realities, CC BY 3.0)
#  - BabylonJS/Assets (CC BY 4.0)
#  - ambientCG-PBR-Sets (CC0) über das Repo ubyjvovk/asciicity
#  - Godot-TPS-Demo, Nietenplatten (J. Linietsky, F. M. Calabró, CC BY 3.0)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets/bab assets/cc0 assets/tps

B=https://raw.githubusercontent.com/mrdoob/three.js/dev/examples
for f in textures/hardwood2_diffuse.jpg textures/hardwood2_bump.jpg textures/hardwood2_roughness.jpg \
         textures/terrain/grasslight-big.jpg \
         models/gltf/LeePerrySmith/LeePerrySmith.glb; do
  curl -fsSL -o "assets/$(basename "$f")" "$B/$f"
done

B=https://raw.githubusercontent.com/BabylonJS/Assets/master
for f in textures/grass.jpg textures/ground.jpg textures/dirt.jpg textures/rockyGround_basecolor.png; do
  curl -fsSL -o "assets/bab/${f//\//_}" "$B/$f"
done

B=https://raw.githubusercontent.com/ubyjvovk/asciicity/HEAD/public/textures/cc0
for m in plaster paving metal; do
  for k in color normal rough; do
    curl -fsSL -o "assets/cc0/${m}_$k.jpg" "$B/$m/$k.jpg"
  done
done

B=https://raw.githubusercontent.com/godotengine/tps-demo/master/level/textures/structure
for f in tile_rivet_panels_normal.png tile_rivet_panels_orm.png; do
  curl -fsSL -o "assets/tps/$f" "$B/$f"
done
echo "Assets geladen."
