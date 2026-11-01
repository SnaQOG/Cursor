# Vorschaubilder ausgewählter Modelle (Cycles, Studiolicht) zum Prüfen nach Änderungen.
#   .venv-blender/bin/python blender/preview.py -- moebel veg      # Gruppen auswählen
#   .venv-blender/bin/python blender/preview.py                    # alle Gruppen
# Bilder landen in blender/preview/ (nicht im Repo).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import kit  # noqa: E402
import m_city as c  # noqa: E402
import m_world as w  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preview')
GROUPS = {
    'gesichter': ([c.f_seefahrer, c.f_schatten, c.f_erfinder, c.f_waechter], 24, (0.35, -1.6, 0.35)),
    'fassade': ([c.kc_window, c.kc_shutter, c.kc_door, c.kc_chimney, c.kc_dormer, c.kc_sign, c.kc_awning], 2.5, (0.6, -1.5, 0.5)),
    'fassade2': ([c.kc_flowerbox, c.kc_beamhead, c.kc_window], 1.4, (0.6, -1.5, 0.4)),
    'moebel': ([c.kc_table, c.kc_chair, c.kc_eckbank, c.kc_bed, c.kc_stove, c.kc_cuckoo], 2.4, (0.7, -1.4, 0.8)),
    'behaelter': ([c.kc_shelf, c.c_truhe, c.c_schrank, c.c_kommode], 1.8, (0.7, -1.4, 0.6)),
    'strasse': ([c.kc_lantern, c.kc_bench, c.kc_stall, c.kc_cart, c.kc_well, c.kc_tank], 8, (0.8, -1.4, 0.7)),
    'landmarken': ([c.l_spire, c.v_giant, c.kc_fountain, c.kc_maypole, c.l_clock], 30, (0.7, -1.5, 0.45)),
    'veg': ([c.v_flowers, c.v_meadow, c.v_weed, c.v_linde, c.v_birke], 4, (0.6, -1.5, 0.5)),
    'autos': ([w.p_car_a, w.p_car_b], 5.6, (1.0, 1.3, 0.55)),
    'baeume': ([w.v_tree, w.v_pine, w.v_bush, w.v_grass], 5.0, (0.6, -1.5, 0.35)),
}

if __name__ == '__main__':
    which = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else list(GROUPS)
    os.makedirs(OUT, exist_ok=True)
    for g in which:
        fns, spacing, d = GROUPS[g]
        kit.reset()
        col = kit.new_collection('p')
        for i, fn in enumerate(fns):
            a = fn(col)
            a.root.location.x = (i - (len(fns) - 1) / 2) * spacing
        bpy.context.view_layer.update()
        objs = [o for o in col.objects if o.type == 'MESH']
        kit.preview(os.path.join(OUT, g + '.png'), objs, direction=d, res=(1100, 520), samples=20)
        print('ok', g)
