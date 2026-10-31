# Baut alle 3D-Assets von Damm Raider in Blender und exportiert sie.
#
#   python blender/build.py            (mit pip-Paket "bpy" 4.5)
#   blender -b -P blender/build.py     (mit installiertem Blender 4.2+)
#
# Ergebnis:
#   assets/models.glb   alle Modelle (Waffen, Behälter, ARC, Props, Vegetation ...)
#   assets/icons.webp   Icon-Atlas aller Gegenstände (128 px pro Feld)
#   assets/icons.json   Zuordnung Gegenstand -> Feld im Atlas
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import kit  # noqa: E402
import m_arc  # noqa: E402
import m_items  # noqa: E402
import m_weapons  # noqa: E402
import m_world  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'assets')
CELL = 128
COLS = 8


def item_ids():
    src = open(os.path.join(ROOT, 'src', 'items.js'), encoding='utf-8').read()
    block = src[src.index('export const ITEMS'):]
    block = block[:block.index('\n};')]
    return re.findall(r'^\s{2}(\w+): \{ name:', block, re.M)


def tri_count(prefix):
    n = 0
    for ob in bpy.data.objects:
        if ob.type == 'MESH' and ob.name.startswith(prefix):
            for p in ob.data.polygons:
                n += len(p.vertices) - 2
    return n


def build_models():
    kit.reset()
    col = kit.new_collection('assets')
    builders = (m_weapons.BUILDERS + m_items.CONTAINERS + m_arc.BUILDERS + m_world.PROPS + m_world.KIT
                + m_world.LANDMARKS + m_world.VEG + m_world.CHARS)
    total = 0
    for b in builders:
        a = b(col)
        t = tri_count(a.name + '__')
        total += t
        print(f'  {a.name:<16} {t:>6} Dreiecke')
    print(f'  gesamt          {total:>6} Dreiecke')
    os.makedirs(OUT, exist_ok=True)
    kit.export_glb(os.path.join(OUT, 'models.glb'))


def pose_for(item):
    # (Neigung zur Kamera, Drehung um Z) in Grad
    if item.startswith('w_'):
        return 12, -90 + 18
    if item in ('vinyl', 'arc_circuit', 'electronics', 'adv_elec', 'metal'):
        return 55, -25
    if item in ('shield_light', 'shield_medium', 'shield_heavy'):
        return 50, -20
    return 22, -32


def build_icons(ids):
    kit.reset()
    col = kit.new_collection('icons')
    wb = {b.__name__: b for b in m_weapons.BUILDERS}
    rows = math.ceil(len(ids) / COLS)
    S = 1.0
    for i, item in enumerate(ids):
        if item.startswith('w_'):
            a = wb[item[2:]](col)
        else:
            a = kit.Asset('icon_' + item, col)
            fn = m_items.ICON_MODELS.get(item)
            if fn is None:
                print('  kein Icon-Modell für', item)
                continue
            a.add(fn(a))
        root = a.root
        tilt, spin = pose_for(item)
        root.rotation_mode = 'ZXY'
        root.rotation_euler = (math.radians(tilt), 0, math.radians(spin))
        bpy.context.view_layer.update()
        objs = [o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith(a.name + '__')]
        lo, hi = kit.mesh_bounds(objs)
        c = (lo + hi) / 2
        # nur die in der Bildebene sichtbare Ausdehnung (X und Z) zählt
        size = max(hi.x - lo.x, hi.z - lo.z, 1e-4)
        s = 0.82 * S / size
        cx = (i % COLS + 0.5) * S
        cz = -(i // COLS + 0.5) * S
        root.scale = (s, s, s)
        root.location = (cx - c.x * s, -c.y * s, cz - c.z * s)
    kit.studio_lights(col)
    W, H = COLS * S, rows * S
    kit.camera_look(col, (W / 2, -20, -H / 2), (W / 2, 0, -H / 2), ortho=max(W, H))
    kit.setup_render(COLS * CELL, rows * CELL, samples=40, transparent=True)
    kit.render_to(os.path.join(OUT, 'icons.webp'), fmt='WEBP')
    meta = {'cell': CELL, 'cols': COLS, 'rows': rows, 'items': {item: i for i, item in enumerate(ids)}}
    with open(os.path.join(OUT, 'icons.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)


if __name__ == '__main__':
    only = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ids = item_ids()
    print(f'{len(ids)} Gegenstände')
    if not only or 'models' in only:
        print('Modelle …')
        build_models()
    if not only or 'icons' in only:
        print('Icons …')
        build_icons(ids)
    print('fertig:', OUT)
