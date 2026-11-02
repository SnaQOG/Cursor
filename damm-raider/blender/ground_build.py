# Bodenpflanzen als Atlas aus CC0-Modellen von Poly Haven. Jede Pflanze wird einmal von der Seite gerendert,
# das Spiel zeichnet sie als zwei gekreuzte Bildflächen in großen Meshes (Batcher.addCover).
#   python3 -I blender/ground_fetch.py --out <downloads>
#   blender -b --factory-startup -P blender/ground_build.py -- --dl <downloads> --out <atlas.png>
# Das Skript gibt die Tabelle für src/groundcover.js aus (Spalte, Zeile, Kantenlänge in Metern).
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
import simready_build as sb  # noqa: E402
from nature_build import load_node  # noqa: E402

COLS, ROWS, CELL = 4, 3, 320
# id im Spiel, Quelle, Knoten
PLANTS = [
    ('gross1', 'grass_medium_01', 'grass_medium_01_large_a_LOD0'), ('gross2', 'grass_medium_01', 'grass_medium_01_large_c_LOD0'),
    ('mittel', 'grass_medium_01', 'grass_medium_01_mid_a_LOD0'), ('klein', 'grass_medium_01', 'grass_medium_01_small_a_LOD0'),
    ('halme', 'grass_medium_01', 'grass_medium_01_tall_a_LOD0'), ('trocken', 'grass_medium_02', 'grass_medium_02_b'),
    ('loewenzahn', 'dandelion_01', 'dandelion_01_a_LOD0'), ('gelb', 'flower_ursinia', 'flower_ursinia_b_LOD0'),
    ('rosa', 'shrub_sorrel_01', 'shrub_sorrel_01_d'), ('nessel', 'nettle_plant', 'nettle_plant_small_a_LOD0'),
    ('kraut', 'weed_plant_02', 'weed_plant_02_a_LOD0'), ('farn', 'fern_02', 'fern_02_b'),
]


def render(src, node, dl, tmp):
    ob, lo, hi = load_node(dl, src, node)
    s = bpy.context.scene
    sb.setup_cycles()
    s.cycles.samples = 64
    s.cycles.max_bounces = 4
    s.cycles.transparent_max_bounces = 16
    s.render.film_transparent = True
    s.render.resolution_x = s.render.resolution_y = CELL
    s.view_settings.view_transform = 'Standard'
    s.view_settings.look = 'None'
    s.render.image_settings.file_format = 'PNG'
    s.render.image_settings.color_mode = 'RGBA'
    s.world.use_nodes = True
    bg = next(n for n in s.world.node_tree.nodes if n.type == 'BACKGROUND')
    bg.inputs['Color'].default_value = (1, 1, 1, 1)
    bg.inputs['Strength'].default_value = 0.85
    sun = bpy.data.lights.new('Sun', 'SUN')
    sun.energy = 1.0
    so = bpy.data.objects.new('Sun', sun)
    so.rotation_euler = (math.radians(35), 0, math.radians(25))
    s.collection.objects.link(so)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    extent = float(max(hi[2] - lo[2], math.hypot(hi[0] - lo[0], hi[1] - lo[1]))) * 1.03
    cz = float(lo[2]) + extent / 2  # Unterkante des Bildes liegt auf dem Boden
    cam = bpy.data.cameras.new('Cam')
    cam.type = 'ORTHO'
    cam.ortho_scale = extent
    co = bpy.data.objects.new('Cam', cam)
    s.collection.objects.link(co)
    s.camera = co
    co.location = (cx, cy - extent * 2, cz)
    co.rotation_euler = (Vector((cx, cy, cz)) - Vector(co.location)).to_track_quat('-Z', 'Y').to_euler()
    s.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(tmp)
    px = np.empty(CELL * CELL * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(CELL, CELL, 4), extent


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--dl', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    atlas = np.zeros((ROWS * CELL, COLS * CELL, 4), np.float32)
    rows = []
    for k, (aid, src, node) in enumerate(PLANTS):
        px, extent = render(src, node, args.dl, args.out + f'.{aid}.png')
        cx, cy = k % COLS, ROWS - 1 - k // COLS
        # Farbe über den Rand ziehen, sonst entstehen helle Säume beim Ausschneiden
        px[..., :3] = sb.fill_misses(px[..., :3], px[..., 3] > 0.3, steps=16)
        atlas[cy * CELL:(cy + 1) * CELL, cx * CELL:(cx + 1) * CELL] = px
        rows.append(f"  {aid}: [{k % COLS}, {k // COLS}, {extent:.2f}],")
        print(aid, round(extent, 2), flush=True)
    out = bpy.data.images.new('cover', COLS * CELL, ROWS * CELL, alpha=True)
    out.pixels.foreach_set(atlas.ravel())
    out.filepath_raw = args.out
    out.file_format = 'PNG'
    out.save()
    print('TABELLE\n' + '\n'.join(rows), flush=True)


if __name__ == '__main__':
    main()
