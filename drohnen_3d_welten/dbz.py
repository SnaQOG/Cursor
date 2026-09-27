"""Son Goku (Super-Saiyajin, Namek-Saga) und Freezer (Endform) auf dem Gelenk-Rig (figures.Figure).

Goku: goldenes, nach oben stehendes Stachelhaar, orangefarbener Gi mit blauem Unterhemd (Ärmel und V am Hals),
blauer Gürtel, blaue Armbänder, dunkelblaue Stiefel. Freezer: weißer, glänzender Körper, lila Panzerteile
(Schädelkuppel, Schultern, Unterarme, Schienbeine, Brust), langer weißer Schwanz, schlank und klein (1,53 m).
"""
import math

import numpy as np
from mathutils import Vector

import fpv
from figures import POSES, Figure
from ninja import cloth_material, ellipsoid, hair_material, skin_material, spikes, tube


def goku(loc, yaw_deg, pose="stand"):
    ORANGE, BLUE = (0.90, 0.24, 0.03), (0.03, 0.07, 0.30)
    m = {
        "gi": cloth_material("GokuGi", ORANGE, zones=[(0.93, 1.0, BLUE)]),
        "under": cloth_material("GokuUnder", BLUE),
        "pants": cloth_material("GokuPants", ORANGE),
        "boot": cloth_material("GokuBoot", (0.02, 0.04, 0.16), rough=0.6, sheen=0.1),
        "skin": skin_material("GokuSkin", (0.72, 0.43, 0.28)),
        "hair": hair_material("GokuSSJHair", (1.0, 0.78, 0.12), sheen=(1.0, 1.0, 0.7)),
    }
    # SSJ-Haar leuchtet leicht
    nt = m["hair"].node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.8, 0.2, 1)
    bsdf.inputs["Emission Strength"].default_value = 0.6
    fig = Figure("Goku", 1.75, loc, yaw_deg)
    fig.body({"torso": m["gi"], "pelvis": m["pants"], "upper_arm": m["under"], "forearm": m["skin"],
              "hand": m["skin"], "thigh": m["pants"], "lower": m["boot"], "foot": m["boot"], "skin": m["skin"]},
             bulk=1.12, arm_bulk=1.18, fore_bulk=1.15, leg_bulk=1.08, pant_to=0.28, sleeve_to=0.4, feet="boot",
             head=(0.098, 0.106, 0.116))
    # blaues V am Hals, Gürtel mit Knoten, Armbänder
    fig.seg("UnderV", [(0, 0.06, 1.30), (0, 0.075, 1.40)], [(0.06, 0.04), (0.05, 0.035)], m["under"], "spine")
    belt = [(math.cos(a) * 0.172, math.sin(a) * 0.128, 0.965) for a in np.linspace(0, 2 * math.pi, 49)]
    fig.attach(tube("GokuBelt", [tuple(Vector(p) * fig.s) for p in belt], 0.03 * fig.s, m["under"]), "root")
    for sd, x in (("R", 1), ("L", -1)):
        el, wr = Vector((x * 0.21, 0, 1.10)), Vector((x * 0.22, 0, 0.86))
        fig.seg(f"Band{sd}", [el.lerp(wr, 0.7), el.lerp(wr, 0.97)], [(0.05, 0.05), (0.052, 0.052)], m["under"],
                f"elbow.{sd}")
    # Super-Saiyajin-Haar: große, nach oben/hinten stehende goldene Stacheln
    H = Vector((0, 0.01, 1.57)) * fig.s
    head = [ellipsoid("GokuHairCap", H + Vector((0, -0.01, 0.04)) * fig.s, (0.105 * fig.s, 0.114 * fig.s, 0.1 * fig.s),
                      m["hair"])]
    rng = np.random.default_rng(11)
    dirs, lens, brs = [], [], []
    while len(dirs) < 30:
        az = rng.uniform(0, 2 * math.pi)
        el = rng.uniform(0.25, 1.45)
        if math.sin(az) > 0.55 and el < 0.7:
            continue
        dirs.append((math.cos(az) * math.cos(el) * 0.8, math.sin(az) * math.cos(el) - 0.1, math.sin(el) + 0.45))
        lens.append(rng.uniform(0.22, 0.34) * fig.s)
        brs.append(rng.uniform(0.055, 0.075) * fig.s)
    head.append(spikes("GokuSpikes", H + Vector((0, -0.01, 0.07)) * fig.s, dirs, lens, brs, m["hair"]))
    bang = [(0.15 * k, 1.0, -0.6) for k in (-1, 0, 1)]
    head.append(spikes("GokuBang", H + Vector((0, 0.06, 0.1)) * fig.s, bang, [0.1 * fig.s] * 3, [0.03 * fig.s] * 3,
                       m["hair"]))
    for sx in (-1, 1):
        head.append(ellipsoid(f"GokuEar{sx}", H + Vector((sx * 0.097, -0.005, 0.0)) * fig.s,
                              (0.014 * fig.s, 0.022 * fig.s, 0.03 * fig.s), m["skin"]))
    for o in head:
        fig.attach(o, "head")
    fig.pose(1, POSES.get(pose, {}))
    return fig


def _gloss(name, color, rough=0.18, coat=0.6):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    n = nb.noise(co, scale=20.0, detail=2)
    col = nb.mix(nb.math("MULTIPLY", nb.out(n, "Fac"), 0.12), color, [c * 0.85 for c in color])
    p = fpv.principled(nb, Base_Color=col, Roughness=rough)
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Subsurface Weight"].default_value = 0.1
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def freezer(loc, yaw_deg, pose="stand"):
    WHITE, PURPLE = (0.82, 0.80, 0.84), (0.20, 0.025, 0.33)
    m = {"white": _gloss("FreezerWhite", WHITE, rough=0.3, coat=0.35), "gem": _gloss("FreezerGem", PURPLE, 0.12, 0.9),
         "lips": fpv.simple_mat("FreezerLips", (0.06, 0.01, 0.08), rough=0.3),
         "eye": fpv.simple_mat("FreezerEye", (0.5, 0.02, 0.02), rough=0.2)}
    fig = Figure("Freezer", 1.53, loc, yaw_deg)
    w = m["white"]
    fig.body({"torso": w, "pelvis": w, "upper_arm": w, "forearm": w, "hand": w, "thigh": w, "lower": w, "foot": w,
              "skin": w, "head": w}, bulk=0.86, arm_bulk=0.85, fore_bulk=0.95, leg_bulk=0.9, pant_to=0.30,
             feet="boot", head=(0.104, 0.112, 0.128),
             torso_r=[(0.14, 0.105), (0.125, 0.095), (0.15, 0.105), (0.13, 0.095), (0.06, 0.06)])
    s = fig.s
    # lila Panzerteile
    fig.ell("GemHead", (0, -0.005, 1.63), (0.092, 0.098, 0.075), m["gem"], "head")
    fig.ell("GemChest", (0, 0.085, 1.26), (0.09, 0.03, 0.07), m["gem"], "spine")
    for sd, x in (("R", 1), ("L", -1)):
        fig.ell(f"GemShoulder{sd}", (x * 0.2, 0, 1.39), (0.06, 0.06, 0.045), m["gem"], f"shoulder.{sd}")
        fig.ell(f"GemForearm{sd}", (x * 0.228, 0.01, 0.99), (0.028, 0.04, 0.07), m["gem"], f"elbow.{sd}")
        fig.ell(f"GemShin{sd}", (x * 0.097, 0.05, 0.30), (0.04, 0.03, 0.09), m["gem"], f"knee.{sd}")
        fig.ell(f"Eye{sd}", (x * 0.04, 0.105, 1.58), (0.018, 0.008, 0.012), m["eye"], "head")
    fig.ell("Lips", (0, 0.108, 1.51), (0.03, 0.01, 0.008), m["lips"], "head")
    # langer Schwanz aus dem unteren Rücken (verjüngt, geschwungen)
    pts = []
    for t in np.linspace(0, 1, 18):
        pts.append((0.35 * math.sin(t * 3.4) * t, -0.12 - 0.55 * t, 0.9 - 0.55 * t + 0.35 * t * t))
    tail = tube("FreezerTail", [tuple(Vector(p) * s) for p in pts], 0.055 * s, w,
                taper=[1.0 - 0.8 * t for t in np.linspace(0, 1, 18)])
    fig.attach(tail, "root")
    fig.pose(1, POSES.get(pose, {}))
    return fig
