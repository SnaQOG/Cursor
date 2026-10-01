"""Gelenk-Rig für Figuren: Körpersegmente hängen an einer Hierarchie aus Empties und lassen sich
per Keyframe posieren und animieren (Kampfchoreografien, winkende Crew).

Ruhelage: Blick nach +Y, +Z oben, Füße bei z = 0, Arme hängend. +X ist die RECHTE Körperseite.
Rotationen der Gelenke (Grad, Euler XYZ im Elternraum):
  Schulter/Hüfte X+  = Glied schwingt nach vorn (+Y)      Ellbogen X+ = Unterarm beugt nach vorn
  Knie X-            = Unterschenkel beugt nach hinten    Rumpf/Kopf X- = nach vorn neigen
  rechte Schulter Y- = Arm seitlich hoch (links: Y+)      Wurzel Z   = Drehung um die Hochachse
"""
import math

import bpy
from mathutils import Euler, Matrix, Vector

import fpv
from ninja import ellipsoid, skin_part

# Ruhelage für 1,66 m (Naruto-Maßstab): Gelenk -> (Elterngelenk, Position)
REST = {
    "root": (None, (0, 0, 0.93)),
    "spine": ("root", (0, 0, 0.98)),
    "neck": ("spine", (0, 0, 1.43)),
    "head": ("neck", (0, 0, 1.50)),
}
for _s, _x in (("R", 1), ("L", -1)):
    REST.update({
        f"shoulder.{_s}": ("spine", (_x * 0.19, 0, 1.365)),
        f"elbow.{_s}": (f"shoulder.{_s}", (_x * 0.21, 0, 1.10)),
        f"wrist.{_s}": (f"elbow.{_s}", (_x * 0.22, 0, 0.86)),
        f"hip.{_s}": ("root", (_x * 0.088, 0, 0.88)),
        f"knee.{_s}": (f"hip.{_s}", (_x * 0.093, 0, 0.50)),
        f"ankle.{_s}": (f"knee.{_s}", (_x * 0.10, 0, 0.10)),
    })
ORDER = ["root", "spine", "neck", "head"] + [f"{j}.{s}" for s in ("R", "L")
                                              for j in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")]


class Figure:
    """Figur mit Gelenk-Empties. s = Höhe/1,66 m skaliert alle Positionen."""

    def __init__(self, name, height=1.66, loc=(0, 0, 0), yaw=0.0):
        self.name = name
        self.s = height / 1.66
        self.base = bpy.data.objects.new(name, None)          # Weltplatzierung (Ort/Yaw, animierbar)
        fpv.link(self.base)
        self.base.location = loc
        self.base.rotation_euler = (0, 0, math.radians(yaw))
        self.J = {}
        self.rest = {}
        for j in ORDER:
            parent, p = REST[j]
            P = Vector(p) * self.s
            self.rest[j] = P
            e = bpy.data.objects.new(f"{name}_{j}", None)
            fpv.link(e)
            e.empty_display_size = 0.05
            e.rotation_mode = "XYZ"
            if parent is None:
                e.parent = self.base
                e.location = P
            else:
                e.parent = self.J[parent]
                e.location = P - self.rest[parent]
            self.J[j] = e

    # ---------------------------------------------------------------- Aufbau
    def P(self, x, y, z):
        """Punkt der Ruhelage (1,66-m-Maßstab) in Figurenkoordinaten."""
        return Vector((x, y, z)) * self.s

    def attach(self, ob, joint):
        """Objekt, dessen Geometrie in Ruhelage-Figurenkoordinaten gebaut wurde, an ein Gelenk hängen."""
        ob.parent = self.J[joint]
        ob.location = Vector(ob.location) - self.rest[joint]
        return ob

    def seg(self, name, pts, radii, mat, joint, subdiv=2):
        """Körpersegment als Skin-Kette (Punkte im 1,66-m-Maßstab, Radien werden mitskaliert)."""
        s = self.s
        verts = [Vector(p) * s for p in pts]
        rr = [(r[0] * s, r[1] * s) if isinstance(r, (tuple, list)) else (r * s, r * s) for r in radii]
        ob = skin_part(f"{self.name}_{name}", verts, [(i, i + 1) for i in range(len(verts) - 1)], rr, mat,
                       subdiv=subdiv)
        return self.attach(ob, joint)

    def ell(self, name, center, r, mat, joint):
        s = self.s
        ob = ellipsoid(f"{self.name}_{name}", Vector(center) * s, tuple(v * s for v in r), mat)
        return self.attach(ob, joint)

    def body(self, m, bulk=1.0, arm_bulk=1.0, fore_bulk=1.0, leg_bulk=1.0, head=(0.098, 0.108, 0.118),
             pant_to=0.30, sleeve_to=None, torso_r=None, hands=True, feet="sandal", neck=True):
        """Standardkörper. m: Materialien 'torso','pelvis','upper_arm','forearm','hand','thigh','shin','lower',
        'foot','skin'. sleeve_to: Ärmelende am Oberarm (0..1) -> darunter Haut ('skin')."""
        b, ab, fb, lb = bulk, bulk * arm_bulk, bulk * fore_bulk, bulk * leg_bulk
        tr = torso_r or [(0.155, 0.115), (0.15, 0.11), (0.165, 0.115), (0.14, 0.1), (0.065, 0.065)]
        tr = [(r[0] * b, r[1] * b) for r in tr]
        self.seg("Torso", [(0, 0, 0.92), (0, 0.01, 1.05), (0, 0, 1.22), (0, -0.005, 1.37), (0, 0, 1.45)], tr,
                 m["torso"], "spine")
        self.ell("Pelvis", (0, 0, 0.9), (0.152 * b, 0.112 * b, 0.12), m["pelvis"], "root")
        self.ell("Head", (0, 0.01, 1.57), head, m.get("head", m["skin"]), "head")
        if neck:
            self.seg("Neck", [(0, 0, 1.41), (0, 0.005, 1.52)], [(0.052 * b, 0.05 * b), (0.048 * b, 0.046 * b)],
                     m["skin"], "neck")
        for sd, x in (("R", 1), ("L", -1)):
            sh, el, wr = (Vector(REST[f"{j}.{sd}"][1]) for j in ("shoulder", "elbow", "wrist"))
            inner = Vector((x * 0.08, 0, 1.33))
            if sleeve_to is None:
                self.seg(f"UpperArm{sd}", [inner, sh, el], [(0.06 * ab,) * 2, (0.058 * ab,) * 2, (0.05 * ab,) * 2],
                         m["upper_arm"], f"shoulder.{sd}")
            else:
                cut = sh.lerp(el, sleeve_to)
                self.seg(f"Sleeve{sd}", [inner, sh, cut], [(0.066 * ab,) * 2, (0.064 * ab,) * 2, (0.06 * ab,) * 2],
                         m["upper_arm"], f"shoulder.{sd}")
                self.seg(f"UpperArm{sd}", [sh, el], [(0.05 * ab,) * 2, (0.043 * ab,) * 2], m["skin"],
                         f"shoulder.{sd}")
            self.ell(f"Elbow{sd}", el, (0.047 * fb, 0.047 * fb, 0.047 * fb), m["forearm"], f"elbow.{sd}")
            self.seg(f"Forearm{sd}", [el, wr], [(0.049 * fb,) * 2, (0.041 * fb,) * 2], m["forearm"], f"elbow.{sd}")
            if hands:
                self.ell(f"Hand{sd}", wr + Vector((0, 0.005, -0.055)), (0.04 * fb, 0.034 * fb, 0.058), m["hand"],
                         f"wrist.{sd}")
            hp, kn, an = (Vector(REST[f"{j}.{sd}"][1]) for j in ("hip", "knee", "ankle"))
            self.seg(f"Thigh{sd}", [Vector((x * 0.07, 0, 1.0)), hp, hp.lerp(kn, 0.5), kn],
                     [(0.08 * lb,) * 2, (0.09 * lb,) * 2, (0.079 * lb,) * 2, (0.064 * lb,) * 2], m["thigh"],
                     f"hip.{sd}")
            self.ell(f"Knee{sd}", kn, (0.062 * lb,) * 3, m["thigh"], f"knee.{sd}")
            cut = kn.lerp(an, (0.50 - pant_to) / 0.40)
            self.seg(f"ShinTop{sd}", [kn, cut], [(0.062 * lb,) * 2, (0.057 * lb,) * 2], m["thigh"], f"knee.{sd}")
            self.seg(f"Shin{sd}", [cut + Vector((0, 0, 0.03)), an + Vector((0, 0, 0.03))],
                     [(0.054 * lb,) * 2, (0.046 * lb,) * 2], m["lower"], f"knee.{sd}")
            self.foot(sd, x, m, feet)

    def foot(self, sd, x, m, kind):
        an = Vector(REST[f"ankle.{sd}"][1])
        if kind == "boot":
            self.seg(f"Foot{sd}", [an + Vector((0, -0.02, -0.03)), Vector((x * 0.10, 0.13, 0.04))],
                     [(0.05, 0.045), (0.046, 0.032)], m["foot"], f"ankle.{sd}")
            return
        self.seg(f"Foot{sd}", [an + Vector((0, -0.02, -0.035)), Vector((x * 0.10, 0.12, 0.032))],
                 [(0.046, 0.04), (0.042, 0.026)], m["skin"], f"ankle.{sd}")
        s = self.s
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(0.095 * s, 0.27 * s, 0.025 * s), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(x * 0.10 * s, 0.05 * s, 0.0125 * s), verts=bm.verts)
        sole = fpv.mesh_from_bmesh(bm, f"{self.name}_Sole{sd}", m["foot"], smooth=False)
        bv = sole.modifiers.new("bv", "BEVEL")
        bv.width = 0.01 * s
        bv.segments = 2
        self.attach(sole, f"ankle.{sd}")
        self.seg(f"Strap{sd}", [an + Vector((0, -0.015, -0.01)), an + Vector((0, -0.015, 0.09))],
                 [(0.052, 0.05), (0.05, 0.048)], m["foot"], f"ankle.{sd}")

    # ---------------------------------------------------------------- Animation
    def foot_drop(self, rot):
        """Vorwärtskinematik: wie weit die tiefste Sohle in dieser Pose über/unter z = 0 liegt (Figurenraum)."""
        def R(j):
            return Euler(tuple(math.radians(v) for v in rot.get(j, (0, 0, 0))), "XYZ").to_matrix().to_4x4()
        low = 1e9
        # Ferse/Ballen relativ zum Sprunggelenk (Tripo-Figuren setzen foot_offs aus dem Modell)
        offs = getattr(self, "foot_offs", None) or [Vector(o) * self.s for o in ((0, -0.05, -0.1), (0, 0.12, -0.08))]
        for sd in ("R", "L"):
            M = Matrix.Translation(self.rest["root"]) @ R("root")
            prev = "root"
            for j in (f"hip.{sd}", f"knee.{sd}", f"ankle.{sd}"):
                M = M @ Matrix.Translation(self.rest[j] - self.rest[prev]) @ R(j)
                prev = j
            for off in offs:
                low = min(low, (M @ off).z)
        return low

    def pose(self, frame, rot=None, loc=None, yaw=None, base_rot=None, ground=None):
        """Ganzkörperpose keyframen: rot = {gelenk: (x, y, z) Grad}; nicht genannte Gelenke -> 0.
        loc/yaw = Weltplatzierung der Figur (optional), base_rot = (x, y, z) Grad für ganze Figur (Flug).
        ground = Bodenhöhe: loc.z wird so gesetzt, dass die tiefste Sohle darauf steht."""
        rot = rot or {}
        if ground is not None and loc is not None:
            loc = (loc[0], loc[1], ground - self.foot_drop(rot))
        for j, e in self.J.items():
            r = rot.get(j, (0, 0, 0))
            e.rotation_euler = tuple(math.radians(v) for v in r)
            e.keyframe_insert("rotation_euler", frame=frame)
        if loc is not None:
            self.base.location = loc
            self.base.keyframe_insert("location", frame=frame)
        if yaw is not None or base_rot is not None:
            br = base_rot or (0, 0, 0)
            y = yaw if yaw is not None else math.degrees(self.base.rotation_euler.z)
            self.base.rotation_euler = (math.radians(br[0]), math.radians(br[1]), math.radians(y + br[2]))
            self.base.keyframe_insert("rotation_euler", frame=frame)


# ---------------------------------------------------------------- Posenbibliothek
def mirror(p):
    """Pose spiegeln (rechts <-> links)."""
    out = {}
    for j, (x, y, z) in p.items():
        if j.endswith(".R"):
            out[j[:-1] + "L"] = (x, -y, -z)
        elif j.endswith(".L"):
            out[j[:-1] + "R"] = (x, -y, -z)
        else:
            out[j] = (x, -y, -z)
    return out


POSES = {
    "stand": {"shoulder.R": (0, -8, 0), "shoulder.L": (0, 8, 0), "elbow.R": (8, 0, 0), "elbow.L": (8, 0, 0)},
    "guard": {"spine": (-10, 0, 0), "shoulder.R": (60, -20, 20), "elbow.R": (110, 0, 0), "shoulder.L": (60, 20, -20),
              "elbow.L": (110, 0, 0), "hip.R": (25, 0, 0), "knee.R": (-40, 0, 0), "hip.L": (-5, 0, 0),
              "knee.L": (-25, 0, 0), "ankle.R": (15, 0, 0)},
    "run_a": {"spine": (-18, 0, 0), "shoulder.R": (-50, -10, 0), "elbow.R": (60, 0, 0), "shoulder.L": (45, 10, 0),
              "elbow.L": (80, 0, 0), "hip.R": (-30, 0, 0), "knee.R": (-70, 0, 0), "hip.L": (55, 0, 0),
              "knee.L": (-20, 0, 0)},
    "jump": {"spine": (-15, 0, 0), "shoulder.R": (30, -40, 0), "elbow.R": (70, 0, 0), "shoulder.L": (30, 40, 0),
             "elbow.L": (70, 0, 0), "hip.R": (80, 0, 0), "knee.R": (-110, 0, 0), "hip.L": (60, 0, 0),
             "knee.L": (-120, 0, 0), "ankle.R": (30, 0, 0), "ankle.L": (30, 0, 0)},
    "punch_R": {"spine": (-12, 0, 25), "shoulder.R": (88, -5, 0), "elbow.R": (5, 0, 0), "shoulder.L": (40, 30, -30),
                "elbow.L": (110, 0, 0), "hip.R": (-20, 0, 0), "knee.R": (-15, 0, 0), "hip.L": (40, 0, 0),
                "knee.L": (-45, 0, 0)},
    "kick_R": {"spine": (15, 0, -10), "shoulder.R": (20, -50, 0), "elbow.R": (60, 0, 0), "shoulder.L": (30, 45, 0),
               "elbow.L": (70, 0, 0), "hip.R": (95, 0, 0), "knee.R": (-8, 0, 0), "hip.L": (-5, 0, 0),
               "knee.L": (-20, 0, 0)},
    "crouch_charge_R": {"spine": (-25, 0, 10), "head": (15, 0, 0), "shoulder.R": (25, -25, 0), "elbow.R": (40, 0, 0),
                        "shoulder.L": (55, 15, 0), "elbow.L": (70, 0, 0), "hip.R": (70, 0, 0), "knee.R": (-110, 0, 0),
                        "ankle.R": (35, 0, 0), "hip.L": (20, 0, 0), "knee.L": (-100, 0, 0), "ankle.L": (60, 0, 0)},
    "dash_thrust_R": {"spine": (-35, 0, 15), "head": (25, 0, 0), "shoulder.R": (95, -5, 0), "elbow.R": (10, 0, 0),
                      "shoulder.L": (-40, 20, 0), "elbow.L": (40, 0, 0), "hip.R": (50, 0, 0), "knee.R": (-60, 0, 0),
                      "hip.L": (-45, 0, 0), "knee.L": (-50, 0, 0)},
    "recoil": {"spine": (20, 0, 0), "head": (-15, 0, 0), "shoulder.R": (60, -45, 0), "elbow.R": (30, 0, 0),
               "shoulder.L": (60, 45, 0), "elbow.L": (30, 0, 0), "hip.R": (35, 0, 0), "knee.R": (-70, 0, 0),
               "hip.L": (10, 0, 0), "knee.L": (-60, 0, 0), "ankle.R": (20, 0, 0)},
    "land": {"spine": (-30, 0, 0), "head": (20, 0, 0), "shoulder.R": (20, -40, 0), "elbow.R": (30, 0, 0),
             "shoulder.L": (20, 40, 0), "elbow.L": (30, 0, 0), "hip.R": (80, 0, 0), "knee.R": (-120, 0, 0),
             "ankle.R": (40, 0, 0), "hip.L": (50, 0, 0), "knee.L": (-110, 0, 0), "ankle.L": (60, 0, 0)},
    "wave_R": {"shoulder.R": (10, -150, 0), "elbow.R": (30, 0, 0), "shoulder.L": (0, 8, 0), "elbow.L": (8, 0, 0)},
    "arms_crossed": {"shoulder.R": (45, 0, 30), "elbow.R": (125, 0, 0), "shoulder.L": (45, 0, -30),
                     "elbow.L": (125, 0, 0)},
    "hands_hips": {"shoulder.R": (-10, -35, 0), "elbow.R": (95, 0, -40), "shoulder.L": (-10, 35, 0),
                   "elbow.L": (95, 0, 40)},
    "sit": {"spine": (-5, 0, 0), "hip.R": (90, 0, 0), "knee.R": (-90, 0, 0), "hip.L": (90, 0, 0),
            "knee.L": (-90, 0, 0), "shoulder.R": (20, -10, 0), "elbow.R": (40, 0, 0), "shoulder.L": (20, 10, 0),
            "elbow.L": (40, 0, 0)},
    # Flugposen (DBZ): Körper wird über base_rot nach vorn gekippt
    "fly": {"spine": (5, 0, 0), "head": (-35, 0, 0), "shoulder.R": (-20, -15, 0), "elbow.R": (40, 0, 0),
            "shoulder.L": (-20, 15, 0), "elbow.L": (40, 0, 0), "hip.R": (-5, 0, 0), "knee.R": (-25, 0, 0),
            "hip.L": (5, 0, 0), "knee.L": (-45, 0, 0)},
    "kame_charge": {"spine": (-5, 0, -35), "shoulder.R": (20, -30, 20), "elbow.R": (100, 0, 0),
                    "shoulder.L": (35, 0, -45), "elbow.L": (95, 0, 0), "hip.R": (30, 0, 0), "knee.R": (-50, 0, 0),
                    "hip.L": (-10, 0, 0), "knee.L": (-30, 0, 0)},
    "kame_fire": {"spine": (-12, 0, 0), "shoulder.R": (88, 0, 12), "elbow.R": (5, 0, 0), "shoulder.L": (88, 0, -12),
                  "elbow.L": (5, 0, 0), "wrist.R": (-70, 0, 0), "wrist.L": (-70, 0, 0), "hip.R": (35, 0, 0),
                  "knee.R": (-45, 0, 0), "hip.L": (-15, 0, 0), "knee.L": (-20, 0, 0)},
    "point_R": {"spine": (-5, 0, 10), "shoulder.R": (90, -5, 0), "elbow.R": (0, 0, 0), "shoulder.L": (-10, 25, 0),
                "elbow.L": (20, 0, 0)},
}
POSES["run_b"] = mirror(POSES["run_a"])
POSES["punch_L"] = mirror(POSES["punch_R"])
POSES["kick_L"] = mirror(POSES["kick_R"])
POSES["crouch_charge_L"] = mirror(POSES["crouch_charge_R"])
POSES["dash_thrust_L"] = mirror(POSES["dash_thrust_R"])
POSES["point_L"] = mirror(POSES["point_R"])
POSES["wave_L"] = mirror(POSES["wave_R"])
