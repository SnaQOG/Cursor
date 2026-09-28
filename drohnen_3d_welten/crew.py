"""Die Strohhutbande (nach der Zeitsprung-Optik) auf dem Gelenk-Rig (figures.Figure) für die Thousand Sunny.

Jede Figur ist auf ihre Wiedererkennungsmerkmale aus 10–25 m Entfernung reduziert: Silhouette, Größe,
Haar und Kleidungsfarben. Kanon-Größen: Ruffy 1,74 m, Zorro 1,81, Nami 1,70, Lysop 1,76, Sanji 1,80,
Chopper 0,90, Robin 1,88, Franky 2,40, Brook 2,77, Jinbei 3,01 m.
Koordinaten der Figuren: Blick +Y, +Z oben, +X = rechte Körperseite (siehe figures.py).
"""
import math

import bmesh
import numpy as np
from mathutils import Euler, Matrix, Vector

import fpv
from figures import POSES, REST, Figure
from ninja import cloth_material, ellipsoid, hair_material, skin_material, spikes, tube

SKIN = (0.72, 0.45, 0.30)
REST_ROOT = REST["root"][1][2]

# zusätzliche Posen für die Crew
POSES.update({
    # auf dem Boden sitzen, Beine ausgestreckt, Arme verschränkt, Kopf gesenkt (Zorro schläft)
    "nap": {"spine": (12, 0, 0), "head": (-28, 0, 8), "hip.R": (88, 0, 4), "hip.L": (88, 0, -4),
            "knee.R": (-8, 0, 0), "knee.L": (-20, 0, 0), "shoulder.R": (45, 0, 30), "elbow.R": (125, 0, 0),
            "shoulder.L": (45, 0, -30), "elbow.L": (125, 0, 0)},
    # Franky: SUPER! (beide Arme hoch, Unterarme über dem Kopf zusammen)
    "super": {"spine": (6, 0, 0), "head": (8, 0, 0), "shoulder.R": (0, -168, 0), "elbow.R": (0, 62, 0),
              "shoulder.L": (0, 168, 0), "elbow.L": (0, -62, 0), "hip.R": (0, -8, 0), "hip.L": (0, 8, 0)},
    # Brook: Geige spielen (links am Kinn, rechts streicht)
    "violin": {"head": (-8, 0, -18), "shoulder.L": (75, 35, -35), "elbow.L": (115, 0, 0),
               "shoulder.R": (55, -30, 25), "elbow.R": (80, 0, 0)},
    "violin_b": {"head": (-8, 0, -18), "shoulder.L": (75, 35, -35), "elbow.L": (115, 0, 0),
                 "shoulder.R": (35, -50, 10), "elbow.R": (35, 0, 0)},
    # Jinbei am Steuerrad
    "helm": {"spine": (-6, 0, 0), "shoulder.R": (62, -12, 8), "elbow.R": (48, 0, 0), "shoulder.L": (62, 12, -8),
             "elbow.L": (48, 0, 0)},
    # Robin liest (sitzend, Buch vor der Brust)
    "read": {"spine": (-4, 0, 0), "head": (-18, 0, 0), "hip.R": (90, 0, 0), "knee.R": (-90, 0, 0),
             "hip.L": (90, 0, 0), "knee.L": (-90, 0, 0), "shoulder.R": (38, -8, 25), "elbow.R": (95, 0, 0),
             "shoulder.L": (38, 8, -25), "elbow.L": (95, 0, 0)},
    "wave_R2": {"shoulder.R": (10, -135, 0), "elbow.R": (65, 0, 0), "shoulder.L": (0, 8, 0), "elbow.L": (8, 0, 0)},
    "wave_both": {"shoulder.R": (10, -150, 0), "elbow.R": (30, 0, 0), "shoulder.L": (10, 150, 0),
                  "elbow.L": (30, 0, 0)},
    "wave_both2": {"shoulder.R": (10, -125, 0), "elbow.R": (70, 0, 0), "shoulder.L": (10, 125, 0),
                   "elbow.L": (70, 0, 0)},
    # zur Drohne hinaufschauen und winken
    "wave_up": {"spine": (8, 0, 0), "head": (22, 0, 0), "shoulder.R": (10, -155, 0), "elbow.R": (25, 0, 0),
                "shoulder.L": (10, 155, 0), "elbow.L": (25, 0, 0)},
    "wave_up2": {"spine": (8, 0, 0), "head": (22, 0, 0), "shoulder.R": (10, -128, 0), "elbow.R": (70, 0, 0),
                 "shoulder.L": (10, 128, 0), "elbow.L": (70, 0, 0)},
    "sit_wave": {"hip.R": (90, 0, 0), "knee.R": (-90, 0, 0), "hip.L": (90, 0, 0), "knee.L": (-90, 0, 0),
                 "shoulder.R": (10, -150, 0), "elbow.R": (30, 0, 0), "shoulder.L": (20, 10, 0), "elbow.L": (40, 0, 0)},
    "sit_wave2": {"hip.R": (90, 0, 0), "knee.R": (-90, 0, 0), "hip.L": (90, 0, 0), "knee.L": (-90, 0, 0),
                  "shoulder.R": (10, -130, 0), "elbow.R": (70, 0, 0), "shoulder.L": (20, 10, 0), "elbow.L": (40, 0, 0)},
    "pockets": {"shoulder.R": (-12, -14, 0), "elbow.R": (35, 0, -10), "shoulder.L": (-12, 14, 0),
                "elbow.L": (35, 0, 10), "head": (6, 0, 12)},
})


def _mats(tag, **cols):
    return {k: cloth_material(f"{tag}{k.capitalize()}", c) for k, c in cols.items()}


def _hair_cap(fig, tag, mat, r=(0.106, 0.116, 0.108), dz=0.035, dy=-0.01):
    H = Vector((0, 0.01, 1.57))
    return fig.attach(ellipsoid(f"{tag}HairCap", (H + Vector((0, dy, dz))) * fig.s, tuple(v * fig.s for v in r), mat),
                      "head")


def _spiky(fig, tag, mat, n, lmin, lmax, bmin, bmax, seed, el=(-0.2, 1.3), face=0.45, up=0.2, back=-0.15):
    H = Vector((0, 0.01, 1.57)) * fig.s
    rng = np.random.default_rng(seed)
    dirs, lens, brs = [], [], []
    while len(dirs) < n:
        az = rng.uniform(0, 2 * math.pi)
        e = rng.uniform(*el)
        if math.sin(az) > face and e < 0.55:
            continue
        dirs.append((math.cos(az) * math.cos(e), math.sin(az) * math.cos(e) + back, math.sin(e) + up))
        lens.append(rng.uniform(lmin, lmax) * fig.s)
        brs.append(rng.uniform(bmin, bmax) * fig.s)
    return fig.attach(spikes(f"{tag}Spikes", H + Vector((0, -0.01, 0.05)) * fig.s, dirs, lens, brs, mat), "head")


def _cyl(name, p0, p1, r0, r1, mat, verts=24):
    bm = bmesh.new()
    d = Vector(p1) - Vector(p0)
    bmesh.ops.create_cone(bm, cap_ends=True, segments=verts, radius1=r0, radius2=r1, depth=d.length)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    ob.location = (Vector(p0) + Vector(p1)) / 2
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = d.normalized().to_track_quat("Z", "Y")
    return ob


def _belt(fig, name, mat, z=0.965, rx=0.172, ry=0.128, r=0.03):
    pts = [(math.cos(a) * rx * fig.s, math.sin(a) * ry * fig.s, z * fig.s) for a in np.linspace(0, 2 * math.pi, 41)]
    return fig.attach(tube(name, pts, r * fig.s, mat), "root")


def _hat(fig, tag, crown_mat, band_mat, brim_mat, brim_r=0.25, crown_r=0.12, crown_h=0.1, z=1.66, tilt=0.0):
    s = fig.s
    obs = [_cyl(f"{tag}Brim", (0, 0.01, z), (0, 0.01, z + 0.012), brim_r, brim_r * 0.98, brim_mat, 40),
           _cyl(f"{tag}Crown", (0, 0.0, z), (0, 0.0, z + crown_h), crown_r, crown_r * 0.9, crown_mat, 32),
           _cyl(f"{tag}Band", (0, 0.0, z + 0.005), (0, 0.0, z + 0.04), crown_r * 1.02, crown_r * 1.0, band_mat, 32)]
    for ob in obs:
        ob.location = Vector(ob.location) * s
        ob.scale = (s, s, s)
        if tilt:
            ob.rotation_mode = "XYZ"
            ob.rotation_euler = (math.radians(tilt), 0, 0)
        fig.attach(ob, "head")
    return obs


# --------------------------------------------------------------------------------------------------- Figuren
def luffy(loc, yaw):
    """Ruffy: Strohhut mit rotem Band, offene rote Weste, blaue Shorts mit weißem Saum, gelbe Schärpe, Sandalen."""
    m = _mats("Luffy", vest=(0.62, 0.035, 0.03), shorts=(0.05, 0.14, 0.42), sash=(0.85, 0.62, 0.05),
              cuff=(0.85, 0.83, 0.78))
    skin = skin_material("LuffySkin", SKIN)
    straw = cloth_material("LuffyStraw", (0.80, 0.58, 0.20), rough=0.7, sheen=0.1)
    band = cloth_material("LuffyHatBand", (0.60, 0.02, 0.02))
    hair = hair_material("LuffyHair", (0.012, 0.012, 0.014))
    fig = Figure("Luffy", 1.74, loc, yaw)
    fig.body({"torso": m["vest"], "pelvis": m["shorts"], "upper_arm": m["vest"], "forearm": skin, "hand": skin,
              "thigh": m["shorts"], "lower": skin, "foot": straw, "skin": skin}, bulk=0.95, sleeve_to=0.12,
             pant_to=0.47)
    fig.ell("Chest", (0, 0.095, 1.22), (0.085, 0.035, 0.16), skin, "spine")         # offene Weste
    fig.ell("Scar", (0, 0.128, 1.27), (0.045, 0.006, 0.045), cloth_material("LuffyScar", (0.45, 0.16, 0.10)), "spine")
    _belt(fig, "LuffySash", m["sash"], z=0.99, r=0.035)
    for sd, x in (("R", 1), ("L", -1)):
        fig.seg(f"Cuff{sd}", [(x * 0.093, 0, 0.53), (x * 0.093, 0, 0.47)], [(0.075, 0.075), (0.072, 0.072)],
                m["cuff"], f"knee.{sd}")
    _hair_cap(fig, "Luffy", hair)
    _spiky(fig, "Luffy", hair, 26, 0.14, 0.2, 0.05, 0.07, seed=5, el=(-0.3, 0.9), up=0.0)
    _hat(fig, "LuffyHat", straw, band, straw, brim_r=0.27, crown_r=0.125, crown_h=0.1, z=1.655)
    fig.pose(1, POSES["stand"])
    return fig


def zoro(loc, yaw):
    """Zorro: grüne kurze Haare, dunkelgrüner Mantel mit rotem Haramaki, drei Schwerter links an der Hüfte."""
    m = _mats("Zoro", coat=(0.05, 0.16, 0.07), sash=(0.50, 0.03, 0.03), pants=(0.02, 0.02, 0.025),
              boot=(0.03, 0.025, 0.02))
    skin = skin_material("ZoroSkin", (0.70, 0.43, 0.27))
    hair = hair_material("ZoroHair", (0.10, 0.36, 0.10))
    fig = Figure("Zoro", 1.81, loc, yaw)
    fig.body({"torso": m["coat"], "pelvis": m["coat"], "upper_arm": m["coat"], "forearm": m["coat"], "hand": skin,
              "thigh": m["coat"], "lower": m["pants"], "foot": m["boot"], "skin": skin}, bulk=1.08, arm_bulk=1.1,
             pant_to=0.2, feet="boot")
    _belt(fig, "ZoroSash", m["sash"], z=0.99, rx=0.182, ry=0.138, r=0.05)
    white = cloth_material("ZoroWado", (0.85, 0.84, 0.80), rough=0.3)
    black = cloth_material("ZoroEnma", (0.05, 0.03, 0.06), rough=0.3)
    red = cloth_material("ZoroKitetsu", (0.45, 0.03, 0.03), rough=0.3)
    for k, (mat, dz) in enumerate(((white, 0.03), (black, 0.0), (red, -0.03))):
        a, b = Vector((-0.2, 0.25, 0.98 + dz)), Vector((-0.23, -0.55, 0.72 + dz))
        fig.attach(_cyl(f"ZoroSword{k}", a * fig.s, b * fig.s, 0.018 * fig.s, 0.016 * fig.s, mat, 10), "root")
    _hair_cap(fig, "Zoro", hair, r=(0.108, 0.118, 0.1))
    _spiky(fig, "Zoro", hair, 30, 0.12, 0.15, 0.04, 0.055, seed=7, el=(0.0, 1.3), up=0.25)
    fig.pose(1, POSES["stand"])
    return fig


def nami(loc, yaw):
    """Nami: lange orange Haare, blau-weißes Bikinioberteil, Jeans, Sandaletten."""
    skin = skin_material("NamiSkin", (0.74, 0.47, 0.32))
    jeans = cloth_material("NamiJeans", (0.10, 0.19, 0.38), rough=0.75)
    top = cloth_material("NamiTop", (0.10, 0.30, 0.70))
    hair = hair_material("NamiHair", (1.0, 0.36, 0.05))
    fig = Figure("Nami", 1.70, loc, yaw)
    fig.body({"torso": skin, "pelvis": jeans, "upper_arm": skin, "forearm": skin, "hand": skin, "thigh": jeans,
              "lower": jeans, "foot": jeans, "skin": skin}, bulk=0.9, arm_bulk=0.88, fore_bulk=0.9, leg_bulk=0.95,
             pant_to=0.4, torso_r=[(0.135, 0.1), (0.12, 0.09), (0.15, 0.12), (0.13, 0.1), (0.06, 0.06)])
    for sd, x in (("R", 1), ("L", -1)):
        fig.ell(f"Top{sd}", (x * 0.058, 0.07, 1.25), (0.07, 0.06, 0.06), top, "spine")
    _belt(fig, "NamiBelt", cloth_material("NamiBeltM", (0.35, 0.18, 0.06)), z=0.96, rx=0.152, ry=0.115, r=0.02)
    _hair_cap(fig, "Nami", hair, r=(0.11, 0.12, 0.112))
    H = Vector((0, 0.0, 1.6))
    rng = np.random.default_rng(4)
    for k in range(9):          # lange Strähnen bis zur Rückenmitte
        a = math.pi * (0.1 + 0.8 * k / 8)
        start = H + Vector((math.cos(a) * 0.1, -0.03 - math.sin(a) * 0.05, 0.02))
        pts = [tuple((start + Vector((math.cos(a) * 0.05 * t, -0.06 * t - 0.02 * math.sin(t * 3 + k), -0.55 * t))) * fig.s)
               for t in np.linspace(0, 1, 10)]
        fig.attach(tube(f"NamiStrand{k}", pts, rng.uniform(0.045, 0.06) * fig.s, hair,
                        taper=[1.0 - 0.6 * t for t in np.linspace(0, 1, 10)]), "head")
    fig.pose(1, POSES["stand"])
    return fig


def usopp(loc, yaw):
    """Lysop: dunkle Haut, lange Nase, dichte schwarze Locken mit Stirnband, gelbe Schutzbrille, braune Latzhose."""
    skin = skin_material("UsoppSkin", (0.22, 0.10, 0.045))
    overall = cloth_material("UsoppOverall", (0.46, 0.34, 0.12))
    hair = hair_material("UsoppHair", (0.02, 0.018, 0.016))
    fig = Figure("Usopp", 1.76, loc, yaw)
    fig.body({"torso": overall, "pelvis": overall, "upper_arm": skin, "forearm": skin, "hand": skin,
              "thigh": overall, "lower": overall, "foot": cloth_material("UsoppBoot", (0.10, 0.06, 0.03)),
              "skin": skin}, bulk=0.95, arm_bulk=0.9, pant_to=0.25, feet="boot")
    fig.attach(_cyl("UsoppNose", Vector((0, 0.1, 1.575)) * fig.s, Vector((0, 0.24, 1.585)) * fig.s, 0.018 * fig.s,
                    0.012 * fig.s, skin, 12), "head")
    H = Vector((0, -0.02, 1.6))
    rng = np.random.default_rng(9)
    for k in range(16):         # Lockenmähne
        d = Vector((rng.uniform(-1, 1), rng.uniform(-1.2, 0.2), rng.uniform(-0.6, 0.9))).normalized()
        fig.ell(f"Curl{k}", tuple(H + d * 0.1), (0.07, 0.07, 0.07), hair, "head")
    band = [(math.cos(a) * 0.112, 0.01 + math.sin(a) * 0.122, 1.65) for a in np.linspace(0, 2 * math.pi, 33)]
    fig.attach(tube("UsoppBand", [tuple(Vector(p) * fig.s) for p in band], 0.02 * fig.s,
                    cloth_material("UsoppBandM", (0.75, 0.62, 0.30))), "head")
    goggle = cloth_material("UsoppGoggle", (0.95, 0.60, 0.05), rough=0.25)
    for x in (-0.045, 0.045):
        fig.ell(f"Goggle{x}", (x, 0.11, 1.675), (0.032, 0.02, 0.028), goggle, "head")
    fig.pose(1, POSES["stand"])
    return fig


def sanji(loc, yaw):
    """Sanji: schwarzer Anzug, blaues Hemd, blonde Haare über dem linken Auge, Zigarette."""
    suit = cloth_material("SanjiSuit", (0.018, 0.018, 0.024), rough=0.55, sheen=0.3)
    shirt = cloth_material("SanjiShirt", (0.18, 0.35, 0.70))
    skin = skin_material("SanjiSkin", (0.74, 0.48, 0.33))
    hair = hair_material("SanjiHair", (0.95, 0.72, 0.28))
    fig = Figure("Sanji", 1.80, loc, yaw)
    fig.body({"torso": suit, "pelvis": suit, "upper_arm": suit, "forearm": suit, "hand": skin, "thigh": suit,
              "lower": suit, "foot": cloth_material("SanjiShoe", (0.01, 0.01, 0.012), rough=0.3), "skin": skin},
             bulk=0.95, leg_bulk=0.92, pant_to=0.02, feet="boot")
    fig.ell("Shirt", (0, 0.098, 1.33), (0.055, 0.025, 0.1), shirt, "spine")
    _hair_cap(fig, "Sanji", hair, r=(0.11, 0.12, 0.11))
    fig.ell("Bang", (0.045, 0.08, 1.575), (0.065, 0.045, 0.1), hair, "head")       # über dem linken Auge (+X)
    fig.ell("HairBackS", (0, -0.05, 1.53), (0.1, 0.07, 0.1), hair, "head")
    fig.attach(_cyl("Cigarette", Vector((-0.02, 0.105, 1.515)) * fig.s, Vector((-0.05, 0.17, 1.51)) * fig.s,
                    0.005 * fig.s, 0.005 * fig.s, cloth_material("CigPaper", (0.9, 0.88, 0.84)), 8), "head")
    fig.pose(1, POSES["stand"])
    return fig


def chopper(loc, yaw):
    """Chopper (Brain Point): braunes Fell, rosa Zylinder mit weißem Kreuz, blaue Shorts, blaue Nase, Geweih."""
    fur = cloth_material("ChopperFur", (0.17, 0.07, 0.025), rough=0.9, sheen=0.8)
    shorts = cloth_material("ChopperShorts", (0.10, 0.25, 0.60))
    pink = cloth_material("ChopperHat", (0.90, 0.28, 0.45))
    white = cloth_material("ChopperX", (0.9, 0.9, 0.88))
    fig = Figure("Chopper", 0.90, loc, yaw)
    fig.body({"torso": fur, "pelvis": shorts, "upper_arm": fur, "forearm": fur, "hand": fur, "thigh": shorts,
              "lower": fur, "foot": fur, "skin": fur, "head": fur}, bulk=1.45, arm_bulk=1.1, leg_bulk=1.25,
             head=(0.16, 0.16, 0.16), pant_to=0.45, feet="boot", neck=False)
    fig.ell("Nose", (0, 0.165, 1.54), (0.035, 0.03, 0.028), cloth_material("ChopperNose", (0.1, 0.2, 0.7), rough=0.3),
            "head")
    s = fig.s
    for ob in (_cyl("ChopperHatCrown", Vector((0, 0, 1.67)) * s, Vector((0, 0, 1.9)) * s, 0.15 * s, 0.14 * s, pink),
               _cyl("ChopperHatBrim", Vector((0, 0, 1.67)) * s, Vector((0, 0, 1.69)) * s, 0.24 * s, 0.24 * s, pink)):
        fig.attach(ob, "head")
    for k, a in enumerate((0.785, -0.785)):
        c = Vector((0, 0.13, 1.8))
        d = Vector((math.cos(a), 0, math.sin(a))) * 0.1
        fig.attach(_cyl(f"ChopperX{k}", (c - d) * s, (c + d) * s, 0.022 * s, 0.022 * s, white, 8), "head")
    horn = cloth_material("ChopperHorn", (0.60, 0.42, 0.22))
    for x in (-1, 1):
        pts = [tuple(Vector((x * (0.1 + 0.12 * t), -0.02, 1.72 + 0.14 * t - 0.05 * t * t)) * s)
               for t in np.linspace(0, 1, 6)]
        fig.attach(tube(f"ChopperAntler{x}", pts, 0.022 * s, horn), "head")
    fig.pose(1, POSES["stand"])
    return fig


def robin(loc, yaw):
    """Robin: groß und schlank, lange glatte schwarze Haare, dunkellila Oberteil, schwarze Hose, Stiefel, Buch."""
    top = cloth_material("RobinTop", (0.20, 0.05, 0.26))
    pants = cloth_material("RobinPants", (0.03, 0.025, 0.04))
    skin = skin_material("RobinSkin", (0.68, 0.42, 0.28))
    hair = hair_material("RobinHair", (0.01, 0.01, 0.015))
    fig = Figure("Robin", 1.88, loc, yaw)
    fig.body({"torso": top, "pelvis": pants, "upper_arm": top, "forearm": top, "hand": skin, "thigh": pants,
              "lower": pants, "foot": pants, "skin": skin}, bulk=0.88, arm_bulk=0.9, leg_bulk=0.9, pant_to=0.05,
             feet="boot", torso_r=[(0.135, 0.1), (0.12, 0.09), (0.15, 0.115), (0.13, 0.1), (0.06, 0.06)])
    _hair_cap(fig, "Robin", hair, r=(0.112, 0.12, 0.114))
    fig.ell("HairBack", (0, -0.07, 1.36), (0.12, 0.05, 0.3), hair, "head")
    fig.ell("Book", (0, 0.24, 1.2), (0.09, 0.02, 0.12), cloth_material("RobinBook", (0.35, 0.10, 0.06)), "spine")
    fig.pose(1, POSES["stand"])
    return fig


def franky(loc, yaw):
    """Franky: 2,40 m, blaue Tolle, Sonnenbrille, offenes Hawaiihemd, blaue Badehose, riesige Unterarme."""
    skin = skin_material("FrankySkin", (0.72, 0.44, 0.28))
    shirt = cloth_material("FrankyShirt", (0.55, 0.035, 0.03), zones=[(1.3 * 1.446, 1.36 * 1.446, (0.9, 0.7, 0.1)),
                                                                     (1.1 * 1.446, 1.16 * 1.446, (0.9, 0.7, 0.1))])
    trunks = cloth_material("FrankyTrunks", (0.04, 0.10, 0.55))
    hair = hair_material("FrankyHair", (0.10, 0.45, 0.95))
    metal = fpv.simple_mat("FrankyMetal", (0.55, 0.57, 0.6), rough=0.45, metal=0.6)
    fig = Figure("Franky", 2.40, loc, yaw)
    fig.body({"torso": shirt, "pelvis": trunks, "upper_arm": shirt, "forearm": skin, "hand": metal, "thigh": skin,
              "lower": skin, "foot": cloth_material("FrankySandal", (0.2, 0.1, 0.05)), "skin": skin},
             bulk=1.35, arm_bulk=1.1, fore_bulk=1.9, leg_bulk=0.95, sleeve_to=0.3, pant_to=0.3)
    star = cloth_material("FrankyStar", (0.05, 0.15, 0.7))
    for sd, x in (("R", 1), ("L", -1)):
        fig.ell(f"Star{sd}", (x * 0.235, 0.075, 0.98), (0.04, 0.012, 0.04), star, f"elbow.{sd}")
    fig.ell("Chest", (0, 0.13, 1.2), (0.11, 0.05, 0.17), skin, "spine")
    fig.ell("Quiff", (0, 0.05, 1.7), (0.1, 0.17, 0.08), hair, "head")
    _hair_cap(fig, "Franky", hair, r=(0.1, 0.11, 0.1))
    shades = fpv.simple_mat("FrankyShades", (0.01, 0.01, 0.01), rough=0.1)
    for x in (-0.042, 0.042):
        fig.ell(f"Shade{x}", (x, 0.104, 1.585), (0.034, 0.012, 0.022), shades, "head")
    fig.pose(1, POSES["stand"])
    return fig


def brook(loc, yaw):
    """Brook: 2,77-m-Skelett, riesiger schwarzer Afro mit Zylinder, schwarzer Anzug, Geige."""
    suit = cloth_material("BrookSuit", (0.012, 0.012, 0.016), rough=0.5, sheen=0.3)
    bone = fpv.simple_mat("BrookBone", (0.82, 0.79, 0.70), rough=0.45)
    afro = hair_material("BrookAfro", (0.008, 0.008, 0.01))
    fig = Figure("Brook", 2.77, loc, yaw)
    fig.body({"torso": suit, "pelvis": suit, "upper_arm": suit, "forearm": suit, "hand": bone, "thigh": suit,
              "lower": suit, "foot": suit, "skin": bone, "head": bone}, bulk=0.62, arm_bulk=0.7, fore_bulk=0.7,
             leg_bulk=0.62, pant_to=0.05, feet="boot", head=(0.09, 0.1, 0.105))
    for x in (-0.035, 0.035):
        fig.ell(f"Socket{x}", (x, 0.085, 1.585), (0.022, 0.012, 0.02), fpv.simple_mat("BrookSocket", (0.01, 0.01, 0.01)),
                "head")
    a = ellipsoid("BrookAfroBall", Vector((0, -0.01, 1.7)) * fig.s, (0.2 * fig.s, 0.19 * fig.s, 0.17 * fig.s), afro)
    fpv.displace_obj(a, "CLOUDS", size=0.05, strength=0.03 * fig.s, depth=2, subdiv=1, name="BrookAfro_d")
    fig.attach(a, "head")
    hat = cloth_material("BrookHat", (0.01, 0.01, 0.012), rough=0.4)
    s = fig.s
    for ob in (_cyl("BrookHatCrown", Vector((0.02, 0, 1.83)) * s, Vector((0.02, 0, 1.99)) * s, 0.09 * s, 0.09 * s, hat),
               _cyl("BrookHatBrim", Vector((0.02, 0, 1.83)) * s, Vector((0.02, 0, 1.845)) * s, 0.15 * s, 0.15 * s,
                    hat)):
        fig.attach(ob, "head")
    wood = fpv.simple_mat("Violin", (0.30, 0.10, 0.03), rough=0.3)
    fig.ell("Violin", (-0.1, 0.17, 1.37), (0.05, 0.15, 0.024), wood, "neck")         # unter dem Kinn, links
    fig.attach(_cyl("Bow", Vector((0.22, 0.02, 0.8)) * s, Vector((0.22, 0.62, 0.8)) * s, 0.005 * s, 0.005 * s, wood, 6),
               "wrist.R")
    fig.pose(1, POSES["stand"])
    return fig


def jinbe(loc, yaw):
    """Jinbei: 3,01 m, blauhäutiger Fischmensch, schwarzes Haar mit Knoten, orangeroter Kimono, dunkle Hakama."""
    skin = skin_material("JinbeSkin", (0.12, 0.28, 0.52))
    kimono = cloth_material("JinbeKimono", (0.78, 0.20, 0.06))
    hakama = cloth_material("JinbeHakama", (0.06, 0.06, 0.09))
    hair = hair_material("JinbeHair", (0.01, 0.01, 0.012))
    fig = Figure("Jinbe", 3.01, loc, yaw)
    fig.body({"torso": kimono, "pelvis": hakama, "upper_arm": kimono, "forearm": skin, "hand": skin,
              "thigh": hakama, "lower": hakama, "foot": cloth_material("JinbeGeta", (0.3, 0.2, 0.1)), "skin": skin},
             bulk=1.55, arm_bulk=1.1, fore_bulk=1.2, leg_bulk=1.2, sleeve_to=0.8, pant_to=0.05,
             head=(0.12, 0.12, 0.12))
    _belt(fig, "JinbeObi", cloth_material("JinbeObiM", (0.9, 0.85, 0.7)), z=0.97, rx=0.24, ry=0.18, r=0.04)
    _hair_cap(fig, "Jinbe", hair, r=(0.11, 0.115, 0.1), dz=0.05, dy=-0.03)
    fig.ell("Topknot", (0, -0.06, 1.72), (0.045, 0.06, 0.04), hair, "head")
    fig.pose(1, POSES["stand"])
    return fig


# --------------------------------------------------------------------------------------------------- Aufstellung
def _loop(fig, poses, f0, f1, period, loc=None, yaw=None, ground=None):
    """Zwei Posen im Wechsel (Winken, Geigen) über den ganzen Clip."""
    f = f0
    k = 0
    while f <= f1:
        fig.pose(int(f), POSES[poses[k % len(poses)]], loc=loc, yaw=yaw, ground=ground)
        f += period / len(poses)
        k += 1


# ------------------------------------------------------------------------------------ Schauspiel (Acting)
FPS = 24
LAG = {"elbow": 0.07, "wrist": 0.12, "neck": 0.03, "head": 0.06}   # Nachziehen (Follow-through) in s

POSES.update({
    "hands_hips_r": {"shoulder.R": (-10, -35, 0), "elbow.R": (95, 0, -40), "shoulder.L": (-10, 35, 0),
                     "elbow.L": (95, 0, 40), "root": (0, 0, 4), "spine": (0, 0, -3)},
    # Sanji führt die Zigarette zum Mund
    "smoke": {"shoulder.R": (62, -18, 28), "elbow.R": (138, 0, 0), "shoulder.L": (-12, 14, 0),
              "elbow.L": (35, 0, 10), "head": (4, 0, 8)},
    # Robin blättert um
    "read_turn": {"spine": (-4, 0, 6), "head": (-14, 0, 10), "hip.R": (90, 0, 0), "knee.R": (-90, 0, 0),
                  "hip.L": (90, 0, 0), "knee.L": (-90, 0, 0), "shoulder.R": (40, -34, 50), "elbow.R": (68, 0, 0),
                  "shoulder.L": (38, 8, -25), "elbow.L": (95, 0, 0)},
    # Franky holt aus (tief in die Knie, Arme nach hinten unten) für SUPER!
    "super_prep": {"spine": (-16, 0, 0), "head": (-6, 0, 0), "shoulder.R": (-28, -22, 0), "elbow.R": (55, 0, 0),
                   "shoulder.L": (-28, 22, 0), "elbow.L": (55, 0, 0), "hip.R": (42, -6, 0), "knee.R": (-72, 0, 0),
                   "ankle.R": (28, 0, 0), "hip.L": (42, 6, 0), "knee.L": (-72, 0, 0), "ankle.L": (28, 0, 0)},
    # Chopper: Hocke vor dem Sprung, gezogene Beine in der Luft
    "squat": {"spine": (-22, 0, 0), "head": (14, 0, 0), "shoulder.R": (-25, -15, 0), "elbow.R": (40, 0, 0),
              "shoulder.L": (-25, 15, 0), "elbow.L": (40, 0, 0), "hip.R": (55, 0, 0), "knee.R": (-95, 0, 0),
              "ankle.R": (40, 0, 0), "hip.L": (55, 0, 0), "knee.L": (-95, 0, 0), "ankle.L": (40, 0, 0)},
    "tuck": {"spine": (10, 0, 0), "head": (20, 0, 0), "shoulder.R": (10, -160, 0), "elbow.R": (20, 0, 0),
             "shoulder.L": (10, 160, 0), "elbow.L": (20, 0, 0), "hip.R": (65, 0, 0), "knee.R": (-105, 0, 0),
             "hip.L": (65, 0, 0), "knee.L": (-105, 0, 0)},
})


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _prog(u, a, o):
    """Fortschritt einer Posenänderung (0 -> 1) über u = 0..1: erst Ausholen um a (Anteil der Strecke in
    Gegenrichtung), dann schneller Schwung bis 1 + o (Überschwingen), dann Einschwingen auf 1."""
    u = np.clip(u, 0.0, 1.0)
    if a <= 0 and o <= 0:
        return _ss(u)
    la = 0.3 if a > 0 else 0.0
    lo = 0.28 if o > 0 else 0.0
    lm = 1.0 - la - lo
    p = np.full_like(u, 1.0)
    if la:
        m = u < la
        p[m] = -a * _ss(u[m] / la)
    m = (u >= la) & (u < la + lm)
    x = (u[m] - la) / lm
    p[m] = -a + (1.0 + o + a) * (1.0 - (1.0 - x) ** 3) * _ss(x * 3.0)
    if lo:
        m = u >= la + lm
        p[m] = 1.0 + o - o * _ss((u[m] - la - lm) / lo)
    return p


def _pose_vec(pose, j):
    return np.array((POSES[pose] if isinstance(pose, str) else pose).get(j, (0, 0, 0)), dtype=float)


def _eval_schedule(sched, j, t):
    """sched = [(t0, pose, dauer, ausholen, überschwingen), ...]; erster Eintrag = Startpose.
    Ergebnis (n, 3) Grad für Gelenk j zu den Zeiten t."""
    prev = _pose_vec(sched[0][1], j)
    val = np.tile(prev, (len(t), 1))
    for (t0, pose, dur, a, o) in sched[1:]:
        b = _pose_vec(pose, j)
        val += (b - prev)[None, :] * _prog((t - t0) / dur, a, o)[:, None]
        prev = b
    return val


def wave_loop(t0, t1, pa, pb, half, o=0.08):
    """Winken als Folge kurzer Posenwechsel mit leichtem Überschwingen."""
    out, k, t = [], 0, t0
    while t < t1:
        out.append((t, pb if k % 2 == 0 else pa, half, 0.0, o))
        t += half
        k += 1
    return out


def _fcurves_of(ob):
    ad = ob.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            for cb in strip.channelbags:
                out += list(cb.fcurves)
    return out


def _bake(ob, path, arr):
    """arr (n, 3) pro Frame 0..n-1 schnell als F-Kurven schreiben."""
    ob.keyframe_insert(path, frame=0)
    n = len(arr)
    fr = np.arange(n, dtype=np.float32)
    for fc in _fcurves_of(ob):
        if fc.data_path != path:
            continue
        kp = fc.keyframe_points
        kp.add(n - len(kp))
        co = np.empty((n, 2), dtype=np.float32)
        co[:, 0] = fr
        co[:, 1] = arr[:, fc.array_index]
        kp.foreach_set("co", co.ravel())
        fc.update()


def _matrix_track(ob, n):
    """Weltmatrix eines Objekts mit animiertem Ort/Rotation (auch über Eltern) für Frames 0..n-1."""
    chain = []
    o = ob
    while o is not None:
        chain.append(o)
        o = o.parent
    fcs = {id(o): {(fc.data_path, fc.array_index): fc for fc in _fcurves_of(o)} for o in chain}
    out = []
    for f in range(n):
        M = Matrix.Identity(4)
        for o in reversed(chain):
            d = fcs[id(o)]
            loc = [d[("location", k)].evaluate(f) if ("location", k) in d else o.location[k] for k in range(3)]
            rot = [d[("rotation_euler", k)].evaluate(f) if ("rotation_euler", k) in d else o.rotation_euler[k]
                   for k in range(3)]
            M = M @ Matrix.Translation(loc) @ Euler(rot, "XYZ").to_matrix().to_4x4()
        out.append(M)
    return out


def _zero_phase(x, tau, fps=FPS):
    """Exponentielle Glättung vor- und rückwärts (ohne Verzögerung)."""
    a = 1.0 - math.exp(-1.0 / (tau * fps))
    y = x.copy()
    for i in range(1, len(y)):
        y[i] = y[i - 1] + a * (y[i] - y[i - 1])
    for i in range(len(y) - 2, -1, -1):
        y[i] = y[i + 1] + a * (y[i] - y[i + 1])
    return y


def act(fig, sched, n, loc, yaw, body_M=None, cam=None, ground=None, seat_z=None, look=1.0, breathe=1.0,
        sway=1.0, seed=0, jump=None, shake=None):
    """Ganze Figur backen: Posenfolge mit Ausholen/Überschwingen, Nachziehen von Unterarm/Hand/Kopf,
    Atmen, Gewichtsverlagerung, Kopf-Mikrobewegung, Blick zur Kamera (Hals/Kopf) und Bodenkontakt.
    jump = (t0, t1, höhe): Parabel auf die Basis; shake = (t0, t1, grad): Zittern (Kraftpose halten)."""
    rng = np.random.default_rng(seed)
    t = (np.arange(n) - 1.0) / FPS
    R = {}
    for j in fig.J:
        lag = LAG.get(j.split(".")[0], 0.0)
        R[j] = _eval_schedule(sched, j, t - lag)
    # Atmen (Rumpf, Schultern), Gewichtsverlagerung, Kopf-Mikrobewegung
    fb = rng.uniform(0.2, 0.3)
    ph = rng.uniform(0, 2 * math.pi, 6)
    br = np.sin(2 * math.pi * fb * t + ph[0])
    R["spine"][:, 0] += 1.1 * breathe * br
    R["shoulder.R"][:, 1] += 0.7 * breathe * br
    R["shoulder.L"][:, 1] -= 0.7 * breathe * br
    fs = rng.uniform(0.06, 0.1)
    sw = np.sin(2 * math.pi * fs * t + ph[1])
    R["root"][:, 2] += 1.4 * sway * sw
    R["spine"][:, 2] -= 0.9 * sway * sw
    R["root"][:, 1] += 0.8 * sway * np.sin(2 * math.pi * fs * t + ph[2])
    R["head"][:, 0] += sway * (1.2 * np.sin(2 * math.pi * 0.37 * t + ph[3]) + 0.6 * np.sin(2 * math.pi * 0.83 * t + ph[4]))
    R["head"][:, 2] += sway * (1.6 * np.sin(2 * math.pi * 0.23 * t + ph[5]) + 0.5 * np.sin(2 * math.pi * 0.71 * t))
    if shake:
        s0, s1, amp = shake
        env = _ss((t - s0) / 0.15) * (1 - _ss((t - s1) / 0.2))
        for j, k in (("shoulder.R", 1), ("shoulder.L", 1), ("spine", 0)):
            R[j][:, k] += amp * env * np.sin(2 * math.pi * 9.0 * t + rng.uniform(0, 6))
    # Blick zur Kamera
    if look > 0 and body_M is not None and cam is not None:
        neck = fig.rest["neck"]
        yaw_r = math.radians(yaw)
        Mb = Matrix.Translation(loc) @ Euler((0, 0, yaw_r), "XYZ").to_matrix().to_4x4()
        ly = np.zeros(n)
        lp = np.zeros(n)
        w = np.zeros(n)
        for f in range(n):
            M = body_M[f] @ Mb
            hp = M @ neck
            d = M.to_3x3().inverted() @ (Vector(cam[f]) - hp)
            ly[f] = math.degrees(math.atan2(-d.x, d.y))
            lp[f] = math.degrees(math.atan2(d.z, math.hypot(d.x, d.y)))
            dist = d.length
            w[f] = float(_ss((30.0 - dist) / 12.0) * _ss((125.0 - abs(ly[f])) / 35.0))
        pose_yaw = R["root"][:, 2] + R["spine"][:, 2] + R["head"][:, 2]
        pose_pit = R["spine"][:, 0] + R["head"][:, 0]
        ay = np.clip(ly - pose_yaw, -65, 65) * w * look
        ap = np.clip(lp - pose_pit, -25, 40) * w * look
        ay = _zero_phase(ay, 0.14)
        ap = _zero_phase(ap, 0.14)
        R["neck"][:, 2] += 0.6 * ay
        R["head"][:, 2] += 0.4 * ay
        R["neck"][:, 0] += 0.5 * ap
        R["head"][:, 0] += 0.5 * ap
    for j, e in fig.J.items():
        e.rotation_mode = "XYZ"
        _bake(e, "rotation_euler", np.radians(R[j]))
    # Basis: Ort (Bodenkontakt pro Frame über Vorwärtskinematik), Yaw
    L = np.tile(np.array(loc, dtype=float), (n, 1))
    if seat_z is not None:
        L[:, 2] = seat_z
    elif ground is not None:
        for f in range(n):
            L[f, 2] = ground - fig.foot_drop({j: tuple(R[j][f]) for j in ("root", "hip.R", "knee.R", "ankle.R",
                                                                          "hip.L", "knee.L", "ankle.L")})
    if jump:
        j0, j1, hgt = jump
        u = np.clip((t - j0) / (j1 - j0), 0, 1)
        L[:, 2] += np.where((t > j0) & (t < j1), 4 * hgt * u * (1 - u), 0.0)
    _bake(fig.base, "location", L)
    fig.base.rotation_euler = (0, 0, math.radians(yaw))


def place_crew(body, frames, sunny, cam_pos=None):
    """Crew an Bord: an 'body' (Schiffskörper mit Stampfen/Rollen) gehängt, Koordinaten im Schiffssystem
    (+X Bug, +Y Backbord). Blickrichtungen überwiegend nach Steuerbord (-Y), wo die Kamera vorbeifliegt.
    Figuren-Yaw: 0 = Blick +Y, -90 = Blick +X (Bug), 180 = Blick -Y (Steuerbord).
    Animation (Schritt 3.4): pro Figur eine Posenfolge mit Ausholen/Überschwingen, versetzte Phasen,
    Atmen/Schwanken, Blick zur vorbeifliegenden Kamera (cam_pos je Frame, Weltkoordinaten)."""
    Z_DECK, Z_FORE, Z_ROOF = sunny.Z_DECK, sunny.Z_FORE, sunny.Z_ROOF
    SB, BOW = 180.0, -90.0
    n = frames + 2
    body_M = _matrix_track(body, n) if cam_pos is not None else None
    crew = []

    def put(fig, loc, yaw, sched, ground=None, seat=None, **kw):
        fig.base.parent = body
        seat_z = None
        if seat is not None:            # sitzend: Gesäß (≈ Wurzel − 9 cm) auf Sitzhöhe
            seat_z = seat - (REST_ROOT - 0.09) * fig.s
        loc3 = (loc[0], loc[1], seat_z if seat_z is not None else (ground or 0.0))
        act(fig, sched, n, loc3, yaw, body_M=body_M, cam=cam_pos, ground=ground, seat_z=seat_z, **kw)
        crew.append(fig)
        return fig

    def hold(pose):
        return [(0.0, pose, 1.0, 0, 0)]

    # Ruffy sitzt auf der Steuerbord-Reling am Bug und winkt die ganze Zeit (Phase versetzt)
    put(luffy((0, 0, 0), 0), (7.9, -4.3, 0), SB,
        [(0.0, "sit_wave", 1, 0, 0)] + wave_loop(0.1, 20.5, "sit_wave", "sit_wave2", 0.3, o=0.1), seat=9.5,
        seed=1, sway=0.6)
    # Jinbei am Steuer: kleine Korrekturen am Rad, Blick zur Kamera
    put(jinbe((0, 0, 0), 0), (6.75, 0.0, 0), BOW,
        [(0.0, "helm", 1, 0, 0), (6.0, {**POSES["helm"], "shoulder.R": (68, -12, 8), "shoulder.L": (56, 12, -8),
                                        "spine": (-6, 0, -5)}, 1.2, 0.05, 0.08),
         (12.5, "helm", 1.2, 0.05, 0.08)], ground=Z_FORE, seed=2, look=0.8)
    put(brook((0, 0, 0), 0), (8.9, -1.5, 0), SB - 25,
        [(0.0, "violin", 1, 0, 0)] + wave_loop(0.2, 20.5, "violin", "violin_b", 0.34, o=0.05),
        ground=Z_FORE, seed=3, look=0.5)
    # Lysop und Chopper auf dem Rasen: stehen, holen aus und winken, sobald die Drohne kommt
    put(usopp((0, 0, 0), 0), (-2.0, -1.0, 0), SB,
        [(0.0, "stand", 1, 0, 0), (9.3, "wave_up", 0.55, 0.15, 0.12)]
        + wave_loop(9.9, 15.0, "wave_up", "wave_up2", 0.3, o=0.1) + [(15.1, "stand", 0.7, 0.05, 0.08)],
        ground=Z_DECK, seed=4)
    put(chopper((0, 0, 0), 0), (-0.7, -0.8, 0), SB + 10,
        [(0.0, "stand", 1, 0, 0), (9.1, "wave_up", 0.5, 0.15, 0.12)]
        + wave_loop(9.65, 10.35, "wave_up", "wave_up2", 0.25, o=0.1)
        + [(10.35, "squat", 0.27, 0.0, 0.05), (10.62, "stand", 0.08, 0.0, 0.0), (10.72, "tuck", 0.12, 0.0, 0.0),
           (11.18, "squat", 0.14, 0.0, 0.1), (11.4, "wave_up", 0.35, 0.0, 0.12)]
        + wave_loop(11.8, 15.0, "wave_up", "wave_up2", 0.25, o=0.1) + [(15.1, "stand", 0.6, 0.05, 0.08)],
        ground=Z_DECK, seed=5, jump=(10.66, 11.22, 0.45))
    put(nami((0, 0, 0), 0), (0.6, 0.6, 0), SB - 25,
        [(0.0, "hands_hips", 1, 0, 0), (9.6, "wave_R", 0.5, 0.12, 0.12)]
        + wave_loop(10.15, 14.2, "wave_R", "wave_R2", 0.36, o=0.1) + [(14.3, "hands_hips_r", 0.6, 0.0, 0.08)],
        ground=Z_DECK, seed=6)
    # Zorro schläft: tiefes Atmen, kein Blick
    put(zoro((0, 0, 0), 0), (2.95, -0.62, 0), SB, hold("nap"), ground=Z_DECK, seed=7, look=0.0, breathe=2.4,
        sway=0.15)
    # Sanji: Hände in den Taschen, führt die Zigarette zum Mund, schaut der Drohne nach
    put(sanji((0, 0, 0), 0), (-3.2, 0.6, 0), SB - 40,
        [(0.0, "pockets", 1, 0, 0), (11.4, "smoke", 0.5, 0.1, 0.08), (13.2, "pockets", 0.6, 0.05, 0.06)],
        ground=Z_DECK, seed=8)
    # Robin liest, blättert um und blickt kurz auf
    put(robin((0, 0, 0), 0), (-2.95, 2.3, 0), SB + 30,
        [(0.0, "read", 1, 0, 0), (12.0, "read_turn", 0.35, 0.08, 0.1), (12.45, "read", 0.4, 0.0, 0.06)],
        seat=Z_DECK + 0.45, seed=9, look=0.6, sway=0.4)
    chair = fpv.simple_mat("DeckChair", (0.75, 0.70, 0.62), rough=0.6)
    for nm, a, b in (("ChairSeat", (-3.35, 1.9, Z_DECK + 0.33), (-2.55, 2.7, Z_DECK + 0.45)),
                     ("ChairBack", (-3.5, 2.5, Z_DECK + 0.4), (-2.4, 2.75, Z_DECK + 1.15))):
        ob = sunny.box(nm, a, b, chair, bevel=0.02)
        ob.parent = body
    # Franky auf dem Kastelldach: holt aus und reißt genau beim Vorbeiflug die SUPER-Pose hoch
    put(franky((0, 0, 0), 0), (-6.2, -2.6, 0), SB - 15,
        [(0.0, "hands_hips", 1, 0, 0), (12.95, "super_prep", 0.35, 0.0, 0.0), (13.3, "super", 0.4, 0.0, 0.15),
         (16.2, "hands_hips", 0.7, 0.05, 0.08)],
        ground=Z_ROOF + 0.18, seed=10, look=0.7, shake=(13.7, 15.6, 1.2))
    return crew
