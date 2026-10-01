"""Die Strohhutbande aus Tripo-GLB-Modellen (Mixamo-Rig, eingebettete Textur) für die Thousand Sunny.

Ersatz für die selbst gebauten Figuren aus crew.py mit gleicher Aufstellung, gleichen Posen und gleichem
Schauspiel: crew.place_crew(..., chars=tripo_crew.chars()). Jede Figur läuft über tripo_chars.rig_model
(Armature in der Modellpose, Copy Transforms von den Gelenk-Empties, Gewichte über Oberflächenabstände).
Franky kam ohne Rig: seine Gelenke sind aus Vorder- und Seitenansicht vermessen (FRANKY_JOINTS).
Requisiten, die die Modelle nicht haben (Robins Buch), hängen an Gelenken. Einige Posen sind je Figur angepasst
(POSE_MAP), z. B. winkt Brook mit seinen kurzen Spielzeug-Armen, statt Geige zu spielen.

Modelle (nicht im Repo, assets/models/onepiece/<figur>/...):
  Ruffy luffycharacter3dmodel · Zorro zorofigure3dmodel_rig_repariert (unvollständiger Download, repariert)
  Nami animegirlfigure3dmodel · Lysop piratecharacter3dmodel · Sanji sanjifigure3dmodel
  Chopper choppertoy3dmodel · Robin femalecharacter3dmodel · Franky muscularactionfigure3dmodel (ohne Rig)
  Brook skeletonclown3dmodel (Spielzeug-Proportionen) · Jinbei fantasyogre3dmodel
"""
import math
import os

import bmesh

from mathutils import Vector

import crew  # noqa: F401  (registriert die Crew-Posen)
import fpv
import tripo_chars as TC
from figures import POSES

MODELS = os.path.join(fpv.ASSETS, "models", "onepiece")
LOOK = os.environ.get("NIDO_CREW_LOOK", "toon")       # "toon" (Cel-Shading + Kontur) oder "real"
TEX = 2048                                             # Texturen verkleinern (aus 10-25 m reicht 2K)

# ---- Franky (ohne Rig): Gelenke im importierten Modell (Höhe 0,978, Blick -Y, rechte Seite -X).
#      Schultergelenk in der Mitte der BF-37-Kugel, damit sie beim Armheben mitdreht.
FRANKY_JOINTS = {"root": (0.0, -0.05, 0.56), "spine": (0.0, -0.05, 0.64), "neck": (0.0, -0.055, 0.85),
                 "head": (0.0, -0.055, 0.875), "head_top": (0.0, -0.05, 0.975)}
for _s, _g in (("R", -1), ("L", 1)):
    FRANKY_JOINTS.update({
        f"shoulder.{_s}": (_g * 0.25, 0.055, 0.805), f"elbow.{_s}": (_g * 0.275, 0.05, 0.64),
        f"wrist.{_s}": (_g * 0.28, 0.05, 0.40), f"hand.{_s}": (_g * 0.29, 0.06, 0.265),
        f"hip.{_s}": (_g * 0.085, -0.045, 0.53), f"knee.{_s}": (_g * 0.095, -0.045, 0.335),
        f"ankle.{_s}": (_g * 0.115, 0.0, 0.075), f"toe.{_s}": (_g * 0.13, -0.14, 0.012)})

# Figur: (Name, Datei, Höhe in m, Gelenke oder None = Mixamo, Material-Feinheiten)
SPEC = {
    "luffy": ("Luffy", "luffy/luffycharacter3dmodel.glb", 1.74, None, {}),
    "zoro": ("Zoro", "zoro/zorofigure3dmodel_rig_repariert.glb", 1.81, None, {}),
    "nami": ("Nami", "nami/animegirlfigure3dmodel.glb", 1.70, None, {}),
    "usopp": ("Usopp", "usopp/piratecharacter3dmodel.glb", 1.76, None, {}),
    "sanji": ("Sanji", "sanji/sanjifigure3dmodel.glb", 1.80, None, {}),
    "chopper": ("Chopper", "chopper/choppertoy3dmodel.glb", 0.90, None, {}),
    "robin": ("Robin", "robin/femalecharacter3dmodel.glb", 1.88, None, {}),
    "franky": ("Franky", "franky/muscularactionfigure3dmodel.glb", 2.40, FRANKY_JOINTS, {}),
    # Spielzeug-Proportionen (Kopf mit Afro und Zylinder fast halb so hoch wie die Figur): kleiner als der
    # Kanon (2,77 m), sonst wird der Kopf riesig
    "brook": ("Brook", "brook/skeletonclown3dmodel.glb", 2.30, None, {}),
    "jinbe": ("Jinbe", "jinbe/fantasyogre3dmodel.glb", 3.01, None, {}),
}

# Rig-Feinheiten je Figur (weite Kimono-Ärmel und Umhang: stärker geglättete Gewichte)
RIG = {"jinbe": dict(smooth=10)}

# Posen, die bei den Modellen anders aussehen müssen: Lysop hält links die Kabuto (winkt nur rechts), Zorro
# lässt die Arme beim Schlafen locker auf den Oberschenkeln (Schärpe und Schwerter hängen an der Hand),
# Chopper winkt seitlich
POSES.update({
    "wave_up_R": {"spine": (8, 0, 0), "head": (22, 0, 0), "shoulder.R": (10, -155, 0), "elbow.R": (25, 0, 0),
                  "shoulder.L": (0, 8, 0), "elbow.L": (8, 0, 0)},
    "wave_up_R2": {"spine": (8, 0, 0), "head": (22, 0, 0), "shoulder.R": (10, -128, 0), "elbow.R": (70, 0, 0),
                   "shoulder.L": (0, 8, 0), "elbow.L": (8, 0, 0)},
    "nap_loose": {"spine": (12, 0, 0), "head": (-28, 0, 8), "hip.R": (88, 0, 4), "hip.L": (88, 0, -4),
                  "knee.R": (-8, 0, 0), "knee.L": (-20, 0, 0), "shoulder.R": (30, -6, 0), "elbow.R": (40, 0, 0),
                  "shoulder.L": (30, 6, 0), "elbow.L": (40, 0, 0)},
    # Chopper (Spielzeug-Proportionen, kurze Arme): seitlich winken, sonst verschwinden die Hände hinter dem Hut
    "wave_side": {"spine": (6, 0, 0), "head": (10, 0, 0), "shoulder.R": (20, -112, 0), "elbow.R": (20, 0, 0),
                  "shoulder.L": (20, 112, 0), "elbow.L": (20, 0, 0)},
    "wave_side2": {"spine": (6, 0, 0), "head": (10, 0, 0), "shoulder.R": (20, -88, 0), "elbow.R": (55, 0, 0),
                   "shoulder.L": (20, 88, 0), "elbow.L": (55, 0, 0)},
    "tuck_side": {"spine": (10, 0, 0), "head": (12, 0, 0), "shoulder.R": (10, -125, 0), "elbow.R": (20, 0, 0),
                  "shoulder.L": (10, 125, 0), "elbow.L": (20, 0, 0), "hip.R": (65, 0, 0), "knee.R": (-105, 0, 0),
                  "hip.L": (65, 0, 0), "knee.L": (-105, 0, 0)},
    # Jinbei: die weiten Ärmel vertragen nur kleine Armbewegungen -> Hände tief vor dem Bauch am Rad
    "helm_low": {"spine": (-4, 0, 0), "shoulder.R": (30, -6, 6), "elbow.R": (40, 0, 0), "shoulder.L": (30, 6, -6),
                 "elbow.L": (40, 0, 0)},
    "helm_low_turn": {"spine": (-4, 0, -5), "shoulder.R": (36, -6, 6), "elbow.R": (40, 0, 0), "shoulder.L": (24, 6, -6),
                      "elbow.L": (40, 0, 0)},
    # Brook (Spielzeug-Proportionen, kurze Arme): Geigenhaltung ist nicht lesbar -> winkt mit rechts und lacht
    # (Yohohoho, Kopf zurück)
    "brook_wave": {"head": (14, 0, -8), "shoulder.R": (10, -150, 0), "elbow.R": (30, 0, 0), "shoulder.L": (0, 10, 0),
                   "elbow.L": (10, 0, 0)},
    "brook_wave2": {"head": (8, 0, 8), "shoulder.R": (10, -128, 0), "elbow.R": (65, 0, 0), "shoulder.L": (0, 10, 0),
                    "elbow.L": (10, 0, 0)},
})
POSE_MAP = {"usopp": {"wave_up": "wave_up_R", "wave_up2": "wave_up_R2"}, "zoro": {"nap": "nap_loose"},
            "chopper": {"wave_up": "wave_side", "wave_up2": "wave_side2", "tuck": "tuck_side"},
            "brook": {"violin": "brook_wave", "violin_b": "brook_wave2"},
            "jinbe": {"helm": "helm_low", "helm_turn": "helm_low_turn"}}

TOON = dict(sat=1.08, shade=(0.56, 0.54, 0.68), rim=(0.95, 0.95, 1.0), rim_w=0.3, diffuse_mix=0.2)
REAL = dict(rough=0.55, sat=1.05)


def model(key, loc=(0, 0, 0), yaw=0.0, look=None):
    """Gerigte Tripo-Figur key (siehe SPEC) an loc/yaw."""
    name, rel, h, joints, kw = SPEC[key]
    look = look or LOOK
    mk = {**(TOON if look == "toon" else REAL), **kw}
    fig = TC.rig_model(name, os.path.join(MODELS, rel), h, joints=joints, mat_kw=mk, outline=0.004, look=look,
                       tex_size=TEX, **RIG.get(key, {}))
    fig.base.location = loc
    fig.base.rotation_euler = (0, 0, math.radians(yaw))
    fig.key = key
    fig.pose_map = POSE_MAP.get(key, {})
    PROPS.get(key, lambda f: None)(fig)
    return fig


# ------------------------------------------------------------------------------------------------ Requisiten
def _robin_props(fig):
    """Aufgeschlagenes Buch zwischen den Händen der Lesepose (Lage relativ zum Rumpfgelenk gemessen), zu ihr
    gekippt: Einband dunkelrot, Seiten hell."""
    s = fig.s / (1.88 / 1.66)
    cover = fpv.simple_mat("RobinBook", (0.32, 0.06, 0.04), rough=0.6)
    pages = fpv.simple_mat("RobinPages", (0.85, 0.82, 0.72), rough=0.8)
    c = fig.rest["spine"] + Vector((0.0, 0.36, 0.25)) * s
    for nm, size, mat, dy in (("RobinBookCover", (0.24, 0.018, 0.17), cover, 0.0),
                              ("RobinBookPages", (0.22, 0.02, 0.155), pages, -0.008)):
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=tuple(v * s for v in size), verts=bm.verts)
        ob = fpv.mesh_from_bmesh(bm, nm, mat, smooth=False)
        ob.location = c + Vector((0, dy * s, 0))
        ob.rotation_euler = (math.radians(40), 0, 0)
        fig.attach(ob, "spine")


PROPS = {"robin": _robin_props}


def chars(look=None):
    """Konstruktoren im Format von crew.py: {key: fn(loc, yaw) -> Figure}."""
    return {k: (lambda loc, yaw, k=k: model(k, loc, yaw, look)) for k in SPEC}
