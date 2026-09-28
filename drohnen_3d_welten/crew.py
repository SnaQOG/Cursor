"""Die Strohhutbande (nach der Zeitsprung-Optik) auf dem Gelenk-Rig (figures.Figure) für die Thousand Sunny.

Jede Figur ist auf ihre Wiedererkennungsmerkmale aus 10–25 m Entfernung reduziert: Silhouette, Größe,
Haar und Kleidungsfarben. Kanon-Größen: Ruffy 1,74 m, Zorro 1,81, Nami 1,70, Lysop 1,76, Sanji 1,80,
Chopper 0,90, Robin 1,88, Franky 2,40, Brook 2,77, Jinbei 3,01 m.
Koordinaten der Figuren: Blick +Y, +Z oben, +X = rechte Körperseite (siehe figures.py).
"""
import math

import bmesh
import numpy as np
from mathutils import Vector

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


def place_crew(body, frames, sunny):
    """Crew an Bord: an 'body' (Schiffskörper mit Stampfen/Rollen) gehängt, Koordinaten im Schiffssystem
    (+X Bug, +Y Backbord). Blickrichtungen überwiegend nach Steuerbord (-Y), wo die Kamera vorbeifliegt.
    Figuren-Yaw: 0 = Blick +Y, -90 = Blick +X (Bug), 180 = Blick -Y (Steuerbord)."""
    Z_DECK, Z_FORE, Z_ROOF = sunny.Z_DECK, sunny.Z_FORE, sunny.Z_ROOF
    SB, BOW = 180.0, -90.0
    crew = []

    def put(fig, loc, yaw, pose=None, ground=None, seat=None, anim=None, period=18):
        fig.base.parent = body
        if seat is not None:            # sitzend: Gesäß (≈ Wurzel − 9 cm) auf Sitzhöhe
            loc = (loc[0], loc[1], seat - (REST_ROOT - 0.09) * fig.s)
        if anim:
            _loop(fig, anim, 1, frames + 2, period, loc=loc, yaw=yaw, ground=ground)
        else:
            fig.pose(1, POSES[pose], loc=loc, yaw=yaw, ground=ground)
        crew.append(fig)
        return fig

    # Ruffy sitzt auf der Steuerbord-Reling am Bug, Beine außenbords (Mähne würde ihn auf dem Löwenkopf verdecken)
    put(luffy((0, 0, 0), 0), (7.9, -4.3, 0), SB, seat=9.5, anim=["sit_wave", "sit_wave2"], period=14)
    put(jinbe((0, 0, 0), 0), (6.75, 0.0, 0), BOW, "helm", ground=Z_FORE)
    put(brook((0, 0, 0), 0), (8.9, -1.5, 0), SB - 25, ground=Z_FORE, anim=["violin", "violin_b"], period=16)
    # Lysop und Chopper auf dem Rasen an der Steuerbord-Bordwand, genau wo die Drohne übers Deck fliegt
    put(usopp((0, 0, 0), 0), (-2.0, -1.0, 0), SB, ground=Z_DECK, anim=["wave_up", "wave_up2"], period=12)
    put(chopper((0, 0, 0), 0), (-0.7, -0.8, 0), SB + 10, ground=Z_DECK, anim=["wave_up", "wave_up2"], period=10)
    put(nami((0, 0, 0), 0), (0.6, 0.6, 0), SB - 25, ground=Z_DECK, anim=["wave_R", "wave_R2"], period=16)
    put(zoro((0, 0, 0), 0), (2.95, -0.62, 0), SB, "nap", ground=Z_DECK)
    put(sanji((0, 0, 0), 0), (-3.2, 0.6, 0), SB - 40, "pockets", ground=Z_DECK)
    put(robin((0, 0, 0), 0), (-2.95, 2.3, 0), SB + 30, "read", seat=Z_DECK + 0.45)
    chair = fpv.simple_mat("DeckChair", (0.75, 0.70, 0.62), rough=0.6)
    for nm, a, b in (("ChairSeat", (-3.35, 1.9, Z_DECK + 0.33), (-2.55, 2.7, Z_DECK + 0.45)),
                     ("ChairBack", (-3.5, 2.5, Z_DECK + 0.4), (-2.4, 2.75, Z_DECK + 1.15))):
        ob = sunny.box(nm, a, b, chair, bevel=0.02)
        ob.parent = body
    put(franky((0, 0, 0), 0), (-6.2, -2.6, 0), SB - 15, "super", ground=Z_ROOF + 0.18)
    return crew
