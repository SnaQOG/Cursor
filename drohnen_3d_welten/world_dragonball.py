"""Welt 3 – Dragon Ball: Planet Namek.

Tiefe Hauptsonne über grünlich-türkisem Meer, Felsnadeln, Tafelberg mit Steilwand und Überhängen, blaues Gras,
Namekianer-Kuppelhäuser, Ajisa-Bäume, Dragon Balls und Freezers Raumschiff.

Eine durchgehende FPV-Aufnahme (20 s, 24 fps, 9:16, 28 mm), Speed-Ramp über zeitgestempelte Wegpunkte:
  0–2,7 s    Hook: hell, tief über dem Meer (~27 m/s), Slalom links/rechts an zwei Felsnadeln vorbei
  2,7–5,2 s  über offenes Wasser auf den Tafelberg zu (~30 m/s), ferne Zusammenstöße über der Kante
  5,2–7,8 s  Hochziehen und Steigflug dicht an der Steilwand (Schichtbänke, Überhang), über die Felskante:
             Reveal des Plateaus, dabei auf ~40 % abgebremst
  7,8–10 s   tief über blaues Gras, an Kuppelhäusern und den Dragon Balls vorbei (~27 m/s)
  10–20 s    Kampf Goku gegen Freezer, Kamera 3–10 m/s:
             10,0 Zusammenprall in der Luft · 10,5 / 10,95 / 11,4 Schlagabtausch · 11,95 Doppelfaust von oben,
             12,15 Freezer schlägt in den Boden (Krater, Bruchstücke) · 13,15 / 13,4 Todesstrahlen (glühende Krater)
             · 13,6 Freezer weicht übers Meer aus, Goku am Nordrand · 14,3 Aufladen · 15,0 Strahlenduell
             · 15,8 Durchbruch: Explosion über dem Meer (Klimax) · 16,5–20 Nachglühen, Rauchsäule,
             Kamera zieht langsam zurück und steigt
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import bmesh  # noqa: E402,I100
import numpy as np  # noqa: E402

import anime_chars as AC  # noqa: E402
import choreo  # noqa: E402
import dbz_fx  # noqa: E402
import tripo_chars as TC  # noqa: E402
import fpv  # noqa: E402
import namek  # noqa: E402
import nature  # noqa: E402
from world_onepiece import foam_builder  # noqa: E402
import ocean  # noqa: E402
import vfx  # noqa: E402
from figures import POSES, mirror  # noqa: E402
from mathutils import Vector  # noqa: E402

FPS = 24
SECONDS = 20
LENS_MM = 28.0
# Licht (Schritt 3.2): tiefe Hauptsonne links hinten (Felsen/Figuren von vorn-seitlich beleuchtet, heller Hook),
# Aufhellsonne hinten rechts, dritte Sonne tief rechts vorn (sichtbar am Horizont, Gegenlicht)
SUN_ELEV, SUN_AZIM = 19.0, 225.0
SUN_KELVIN, SUN_STRENGTH = 4300, 5.0
SUNS2 = [(36.0, 140.0, 1.3, 5600), (7.0, 16.0, 0.7, 4700)]          # (Höhe, Azimut, W/m², Kelvin)
RIM = dict(elev=18, azim=10, strength=2.6, kelvin=7800, angle=3)
ATMO_DENSITY = float(os.environ.get("NIDO_ATMO", "0"))

MESA_C, MESA_R = (0.0, 250.0), 74.0
DB_C = (1.0, 232.0)
SHIP_C = (42.0, 262.0)
GOLD, KAME, DEATH, KI = (1.0, 0.62, 0.10), (0.10, 0.36, 1.0), (0.85, 0.25, 1.0), (1.0, 0.80, 0.28)

SPIRES = [
    # (x, y, r, h, seed, Neigung x°, Neigung y°)
    (-15, 52, 7.0, 46, 1, 4, -3), (16, 72, 8.0, 58, 2, -3, 5), (-48, 20, 6.0, 30, 3, 6, 2), (42, 18, 5.0, 24, 4, -5, -4),
    (-42, 110, 10.0, 52, 5, 2, 6), (55, 128, 9.0, 40, 6, -4, 3), (-95, 170, 14.0, 70, 7, 3, -2),
    (110, 200, 12.0, 60, 8, -2, 4), (-150, 60, 16.0, 55, 9, 5, 1), (140, 40, 10.0, 48, 10, -6, -2),
    (25, 120, 4.0, 14, 11, 8, -5), (-22, 150, 5.0, 20, 12, -7, 4),
]


# ------------------------------------------------------------------------------------------------ Gelände
def mesa_edge(ang):
    ca, sa = np.cos(ang), np.sin(ang)
    # nur großräumige Buchten/Vorsprünge: feine Rinnen im Umriss würden an der senkrechten Wand zu
    # regelmäßigen Säulen (Orgelpfeifen); die Wand bekommt ihre Unregelmäßigkeit in namek.ring_wall
    return MESA_R * (1 + 0.08 * np.sin(ang * 3 + 1) + 0.05 * np.sin(ang * 7 + 2)
                     + 0.06 * fpv.fbm2(ca * 3, sa * 3, 3, seed=21))


def plateau(X, Y):
    return 39.0 + 2.0 * fpv.fbm2(X / 40, Y / 40, 4, seed=22) + 0.4 * fpv.fbm2(X / 6, Y / 6, 3, seed=23)


def mesa_height(X, Y):
    """Plateau innen; 2,5–6,5 m hinter der Kante fällt das Heightfield steil ab (verdeckt von der Wand und ihrer
    einrollenden Felskante), außen ein Schuttfuß an der Wand."""
    r = np.sqrt((X - MESA_C[0]) ** 2 + (Y - MESA_C[1]) ** 2)
    ang = np.arctan2(Y - MESA_C[1], X - MESA_C[0])
    edge = mesa_edge(ang)
    s = edge - r
    u = np.clip((s - 2.5) / 4.0, 0, 1)
    u = u * u * (3 - 2 * u)
    foot = np.clip((edge + 18 - r) / 18.0, 0, 1) ** 2 * 7 - 3
    return foot + u * (plateau(X, Y) - foot)


def gz(x, y):
    return float(mesa_height(np.array([float(x)]), np.array([float(y)]))[0])


def over(x, y, dz):
    """Punkt dz m über dem Plateau."""
    return (x, y, float(plateau(np.array([float(x)]), np.array([float(y)]))[0]) + dz)


# ------------------------------------------------------------------------------------------------ Schritt 3.1
# Kamera: Wegpunkte mit Uhrzeit (Speed-Ramp aus Abstand/Zeit, monoton-kubisch geglättet)
CAM_KEYS = [
    (0.00, (-2.0, 0.0, 3.4)),        # Hook: tief über dem Meer, Sonnenglitzern, zwei Felsnadeln voraus
    (1.00, (-3.0, 26.0, 2.8)),
    (1.90, (-4.2, 50.0, 2.6)),       # links an Felsnadel 1 vorbei
    (2.65, (3.4, 70.0, 3.0)),        # rechts an Felsnadel 2 vorbei
    (3.50, (2.0, 95.0, 3.4)),
    (4.45, (-2.0, 124.0, 4.0)),
    (5.15, (-3.0, 143.0, 5.0)),
    (5.70, (-3.5, 155.5, 12.5)),     # Hochziehen
    (6.25, (-3.5, 162.5, 26.5)),     # dicht an der Wand hinauf (Wand ~11 m voraus, Überhang oben)
    (6.80, (-3.5, 166.5, 37.5)),
    (7.30, over(-3.3, 173.0, 5.2)),  # über die Felskante: Reveal, auf ~35 % abgebremst
    (7.85, over(-3.0, 179.5, 4.4)),
    (8.45, over(0.0, 195.5, 3.4)),
    (9.05, over(3.5, 213.0, 3.2)),
    (9.55, over(4.6, 229.0, 3.3)),   # Dragon Balls links, 3,5 m
    (10.00, over(5.5, 242.5, 3.6)),  # erster Zusammenprall ~40 m voraus, Kamera bremst
    (10.60, over(6.5, 256.5, 4.2)),  # Schlagabtausch 12–17 m voraus
    (11.30, over(7.5, 263.5, 4.8)),
    (12.10, over(8.0, 266.5, 5.2)),  # Krater 13 m voraus
    (12.90, over(9.5, 266.0, 6.0)),   # etwas zurück: Freezer, Goku und Einschlag im Bild
    (13.60, over(7.0, 281.0, 6.2)),  # folgt den Kämpfern nach Norden
    (14.40, over(4.8, 301.0, 6.8)),
    (15.10, over(2.8, 314.0, 7.2)),  # hinter Goku am Nordrand, Freezer über dem Meer
    (15.80, over(2.5, 316.8, 7.4)),  # Klimax
    (16.40, over(3.0, 315.2, 7.9)),  # von der Druckwelle zurückgedrückt
    (20.05, over(6.0, 305.5, 12.8)), # langsam zurück und hoch: Nachglühen, Rauchsäule
]

T_CLIMAX = 15.8
CH = 1.2                               # Brust über dem Figurenursprung (m)
P_CRATER = (3.0, 278.5)
G_RIM = Vector((-0.5, 323.2, 43.9))    # Goku am Nordrand (Brust)
Z_SEA = Vector((0.5, 369.0, 36.0))     # Freezer über dem Meer (Brust)

# ---- Kampf-Zeitplan (Dragon-Ball-Tempo: Vorstöße 40–130 m/s, Schläge in 2 Frames, Hit-Stops)
TEASERS = [(2.3, (-6, 262, 70), (1, 0.2, 0)), (4.6, (6, 280, 66), (-1, 0.3, 0.1)),
           (7.6, (-2, 300, 62), (1, -0.2, -0.1)), (8.8, (4, 294, 58), (1, -0.15, 0.1))]
RG, RZ = Vector((-9.0, 282.0, 53.0)), Vector((13.5, 284.0, 52.0))   # Start des großen Vorstoßes (Brust)
T_RUSH, T_CLASH = 9.62, 10.0
C0 = Vector((1.0, 273.0, 45.5))        # Zusammenprall
C1 = Vector((1.6, 275.2, 45.8))        # Mitte am Ende des Schlaghagels
FLURRY = [(10.28, "G", "d_punch_R"), (10.42, "Z", "d_punch_R"), (10.56, "G", "d_kick_R"),
          (10.70, "Z", "d_kick_L"), (10.84, "G", "d_punch_L"), (10.98, "Z", "d_punch_L")]
T_WHIP = 11.11                         # Freezers Schwanzhieb (Mitte der 360°-Drehung)
T_KNEE = 11.47                         # Gokus Knie in den Magen
ZAN1 = (11.53, 11.75)                  # Zanzoken (Zeit der Kämpfer, mit Hit-Stop): weg / wieder da über Freezer
T_AXE, T_SLAM = 11.95, 12.15
ZAN2 = (13.15, 13.27)                  # Zanzoken: Todesstrahl 1 trifft nur Gokus Nachbild
# Todesstrahlen senkrecht gestaffelt (Hochformat): Freezer schräg über Goku, der Strahl fährt steil an Goku vorbei
# bzw. durch sein Nachbild in den Boden direkt hinter ihm – Freezer, Goku und Einschlag passen zusammen ins Bild
GB1, GB2 = Vector((-2.5, 272.5, 42.3)), Vector((-6.0, 274.5, 45.2))  # Goku vor / nach dem Ausweichen (Brust)
A1, A2 = Vector((-1.3, 276.0, 48.8)), Vector((-1.0, 276.3, 49.2))    # Freezer beim Feuern (Brust)
GOLD_TRAIL, PURPLE_TRAIL = (1.0, 0.78, 0.25), (0.78, 0.32, 1.0)

POSES.update({
    "axe_up": {"spine": (18, 0, 0), "head": (-10, 0, 0), "shoulder.R": (165, -8, 18), "elbow.R": (15, 0, 0),
               "shoulder.L": (165, 8, -18), "elbow.L": (15, 0, 0), "hip.R": (35, 0, 0), "knee.R": (-70, 0, 0),
               "hip.L": (5, 0, 0), "knee.L": (-50, 0, 0)},
    "axe_down": {"spine": (-32, 0, 0), "head": (18, 0, 0), "shoulder.R": (55, -6, 14), "elbow.R": (5, 0, 0),
                 "shoulder.L": (55, 6, -14), "elbow.L": (5, 0, 0), "hip.R": (50, 0, 0), "knee.R": (-40, 0, 0),
                 "hip.L": (20, 0, 0), "knee.L": (-60, 0, 0)},
    "hover": {"spine": (-3, 0, 0), "shoulder.R": (8, -18, 0), "elbow.R": (28, 0, 0), "shoulder.L": (8, 18, 0),
              "elbow.L": (28, 0, 0), "hip.R": (12, 0, 0), "knee.R": (-28, 0, 0), "hip.L": (-4, 0, 0),
              "knee.L": (-14, 0, 0), "ankle.R": (25, 0, 0), "ankle.L": (20, 0, 0)},
    "slam": {"spine": (35, 0, 0), "head": (-25, 0, 0), "shoulder.R": (120, -60, 0), "elbow.R": (20, 0, 0),
             "shoulder.L": (120, 60, 0), "elbow.L": (20, 0, 0), "hip.R": (-20, 0, 0), "knee.R": (-30, 0, 0),
             "hip.L": (10, 0, 0), "knee.L": (-50, 0, 0)},
    # Nahkampf mit Körpergewicht: Ausholen dreht Hüfte (Wurzel) und Rumpf gegen die Schlagrichtung, der Schlag
    # dreht sie durch; der Kopf dreht gegen, damit der Blick am Gegner bleibt
    "d_wind_R": {"root": (0, 0, -14), "spine": (-6, 0, -22), "head": (0, 0, 22), "shoulder.R": (-35, -25, 0),
                 "elbow.R": (115, 0, 0), "shoulder.L": (60, 20, -20), "elbow.L": (100, 0, 0), "hip.R": (10, 0, 0),
                 "knee.R": (-40, 0, 0), "hip.L": (25, 0, 0), "knee.L": (-50, 0, 0)},
    "d_punch_R": {"root": (0, 0, 16), "spine": (-12, 0, 24), "head": (6, 0, -26), "shoulder.R": (88, -6, 0),
                  "elbow.R": (4, 0, 0), "shoulder.L": (35, 30, -35), "elbow.L": (118, 0, 0), "hip.R": (-18, 0, 0),
                  "knee.R": (-22, 0, 0), "hip.L": (45, 0, 0), "knee.L": (-60, 0, 0)},
    "d_kick_wind_R": {"root": (0, 0, -22), "spine": (-4, 0, 14), "head": (0, 0, 10), "hip.R": (62, 0, 0),
                      "knee.R": (-115, 0, 0), "ankle.R": (25, 0, 0), "hip.L": (0, 0, 0), "knee.L": (-30, 0, 0),
                      "shoulder.R": (30, -40, 0), "elbow.R": (85, 0, 0), "shoulder.L": (55, 30, 0),
                      "elbow.L": (100, 0, 0)},
    "d_kick_R": {"root": (0, 0, 38), "spine": (16, 0, -22), "head": (-8, 0, -18), "hip.R": (70, -50, 0),
                 "knee.R": (-8, 0, 0), "ankle.R": (30, 0, 0), "hip.L": (-12, 0, 0), "knee.L": (-28, 0, 0),
                 "shoulder.R": (15, -65, 0), "elbow.R": (45, 0, 0), "shoulder.L": (45, 40, 0), "elbow.L": (95, 0, 0)},
    "block": {"spine": (8, 0, 0), "head": (-6, 0, 0), "shoulder.R": (78, 0, 36), "elbow.R": (122, 0, 0),
              "shoulder.L": (78, 0, -36), "elbow.L": (122, 0, 0), "hip.R": (32, 0, 0), "knee.R": (-58, 0, 0),
              "hip.L": (10, 0, 0), "knee.L": (-36, 0, 0)},
    "dash": {"spine": (-22, 0, -10), "head": (22, 0, 8), "shoulder.R": (-45, -15, 0), "elbow.R": (105, 0, 0),
             "shoulder.L": (55, 15, -15), "elbow.L": (95, 0, 0), "hip.R": (22, 0, 0), "knee.R": (-72, 0, 0),
             "hip.L": (-25, 0, 0), "knee.L": (-45, 0, 0)},
    "knee_R": {"root": (0, 0, 8), "spine": (-26, 0, 8), "head": (18, 0, 0), "hip.R": (112, 0, 0),
               "knee.R": (-125, 0, 0), "ankle.R": (30, 0, 0), "hip.L": (-18, 0, 0), "knee.L": (-12, 0, 0),
               "shoulder.R": (72, -12, 18), "elbow.R": (92, 0, 0), "shoulder.L": (72, 12, -18),
               "elbow.L": (92, 0, 0)},
    "gut_hit": {"spine": (-42, 0, 0), "neck": (12, 0, 0), "head": (22, 0, 0), "shoulder.R": (45, -35, 0),
                "elbow.R": (45, 0, 0), "shoulder.L": (45, 35, 0), "elbow.L": (45, 0, 0), "hip.R": (62, 0, 0),
                "knee.R": (-85, 0, 0), "hip.L": (50, 0, 0), "knee.L": (-95, 0, 0)},
    "tail_whip": {"spine": (8, 0, 0), "shoulder.R": (10, -72, 0), "elbow.R": (20, 0, 0), "shoulder.L": (10, 72, 0),
                  "elbow.L": (20, 0, 0), "hip.R": (22, 0, 0), "knee.R": (-42, 0, 0), "hip.L": (-10, 0, 0),
                  "knee.L": (-30, 0, 0)},
})
for _n in ("d_wind", "d_punch", "d_kick_wind", "d_kick", "knee"):
    POSES[_n + "_L"] = mirror(POSES[_n + "_R"])


def _key(t, pose, chest, look, lean=0.0, roll=0.0, rot=None, air=True, ease=None, spin=0.0, style=None):
    c = Vector(chest)
    o = {"lean": lean, "roll": roll, "rot": rot}
    if ease:
        o["ease"] = ease
    if spin:
        o["spin"] = spin
    if style:
        o["style"] = style
    return (t, pose, c - Vector((0, 0, CH)) if air else c, Vector(look), air, o)


def _aim_rot(frm, to, both=False):
    d = Vector(to) - Vector(frm)
    aim = math.degrees(math.atan2(d.z, Vector((d.x, d.y)).length))
    if both:
        return {"shoulder.R": (88 + aim, 0, 12), "shoulder.L": (88 + aim, 0, -12)}
    return {"shoulder.R": (90 + aim, -5, 0)}


def _ax(u):
    """Kampfachse Goku -> Freezer während des Schlaghagels (u = 0..1): das Paar dreht sich umeinander."""
    a = math.radians(8.5 + 41.5 * u)
    return Vector((math.cos(a), math.sin(a), 0.1 * math.sin(u * 7.0))).normalized()


def ground_hit(a, through):
    """Einschlag eines Strahls von a durch den Punkt `through` im Boden (Plateau)."""
    a, d = Vector(a), Vector(through) - Vector(a)
    lo, hi = 1.0, 6.0
    for _ in range(60):
        m = (lo + hi) / 2
        p = a + d * m
        if p.z > gz(p.x, p.y):
            lo = m
        else:
            hi = m
    p = a + d * hi
    return (p.x, p.y, gz(p.x, p.y))


def fight_plan():
    """Blocking Goku (G) / Freezer (Z): Listen (t, Pose, Ort, Blickziel, in der Luft, opts) in Welt-Metern.
    Zusätzlich die Ereignisse für die Effekte: Treffer, Ki-Spuren, Zanzoken."""
    G, Z = [], []
    ev = {"hits": [], "trails": [], "zan": []}
    up = Vector((0, 0, 1))
    # ---- ferne Zusammenstöße hoch über dem Plateau (Teaser während des Anflugs): aus 25 m aufeinander zu,
    # Kontakt, Rückprall – aus der Ferne zwei Lichtspuren, die sich treffen
    for k, (t, c, ax) in enumerate(TEASERS):
        c, ax = Vector(c), Vector(ax).normalized()
        perp = ax.cross(up).normalized() * (3.0 if k % 2 else -3.0)
        G += [_key(t - 0.32, "dash", c - ax * 12 + perp, c, lean=-50),
              _key(t, "d_punch_R", c - ax * 0.5, c + ax, ease="lin")]
        Z += [_key(t - 0.32, "dash", c + ax * 12 - perp, c, lean=-45),
              _key(t, "d_punch_L", c + ax * 0.5, c - ax, ease="lin")]
        if k < len(TEASERS) - 1:
            G += [_key(t + 0.22, "recoil", c - ax * 6.0 + up * 1.0, c, lean=15, ease="out"),
                  _key(t + 0.55, "hover", c - ax * 6.8 + up * 1.2, c)]
            Z += [_key(t + 0.22, "recoil", c + ax * 6.5 - up * 0.8, c, lean=15, ease="out"),
                  _key(t + 0.55, "hover", c + ax * 7.2 - up * 0.9, c)]
        else:                          # letzter Teaser: Rückprall zu den Startpunkten des großen Vorstoßes
            G += [_key(t + 0.22, "recoil", c + Vector((-7.0, -3.0, -1.5)), c, lean=15, ease="out"),
                  _key(9.45, "hover", RG + Vector((0.8, 1.5, 0.4)), C0),
                  _key(T_RUSH, "dash", RG, C0, lean=-30)]
            Z += [_key(t + 0.22, "recoil", c + Vector((7.5, -2.0, -2.0)), c, lean=15, ease="out"),
                  _key(9.45, "hover", RZ + Vector((-0.8, 1.2, 0.3)), C0),
                  _key(T_RUSH, "dash", RZ, C0, lean=-30)]
        ev["trails"] += [("G", t - 0.33, t + 0.25, 0.45), ("Z", t - 0.33, t + 0.25, 0.45)]
    # ---- großer Vorstoß (~45 m/s) und Zusammenprall Faust auf Faust
    ax0 = _ax(0.0)
    G += [_key(T_CLASH, "d_punch_R", C0 - ax0 * 0.5, C0 + ax0, lean=-10, ease="lin"),
          _key(T_CLASH + 0.1, "recoil", C0 - ax0 * 1.5 + up * 0.2, C0 + ax0, lean=12, ease="out")]
    Z += [_key(T_CLASH, "d_punch_L", C0 + ax0 * 0.5, C0 - ax0, lean=-10, ease="lin"),
          _key(T_CLASH + 0.1, "recoil", C0 + ax0 * 1.6 - up * 0.1, C0 - ax0, lean=12, ease="out")]
    ev["trails"] += [("G", T_RUSH - 0.02, T_CLASH + 0.02, 0.22), ("Z", T_RUSH - 0.02, T_CLASH + 0.02, 0.22)]
    ev["hits"].append((T_CLASH, tuple(C0), "clash", tuple(ax0)))
    # ---- Schlaghagel: 6 Wechsel in 0,84 s, das Paar dreht sich umeinander und driftet nach Norden
    for i, (t, who, pose) in enumerate(FLURRY):
        u = (i + 1) / len(FLURRY)
        c, ax = C0.lerp(C1, u), _ax(u)
        wind = pose.replace("d_kick", "d_kick_wind") if "kick" in pose else pose.replace("d_punch", "d_wind")
        reach = 0.85 if "kick" in pose else 0.5
        if who == "G":
            G += [_key(t - 0.07, wind, c - ax * (reach + 0.25), c + ax),
                  _key(t, pose, c - ax * reach, c + ax, ease="lin")]
            Z += [_key(t, "block", c + ax * 0.62, c - ax, lean=8)]
            hit = c + ax * 0.22
        else:
            Z += [_key(t - 0.07, wind, c + ax * (reach + 0.25), c - ax),
                  _key(t, pose, c + ax * reach, c - ax, ease="lin")]
            G += [_key(t, "block", c - ax * 0.62, c + ax, lean=8)]
            hit = c - ax * 0.22
        ev["hits"].append((t, tuple(hit + up * 0.28), "flurry", tuple(ax)))
    # ---- Freezer dreht sich einmal um sich selbst, der Schwanz peitscht Goku weg
    c6, ax6 = C1, _ax(1.0)
    zw = c6 + ax6 * 0.55 + up * 0.1
    Z += [_key(11.02, "tail_whip", zw, c6 - ax6),
          _key(11.20, "tail_whip", zw + up * 0.1, c6 - ax6, spin=360.0),
          _key(11.34, "arms_crossed", zw + up * 0.15, c6 - ax6 * 4)]
    G += [_key(T_WHIP, "recoil", c6 - ax6 * 0.7, c6 + ax6, lean=10),
          _key(11.27, "recoil", c6 - ax6 * 4.2 - up * 0.5, c6 + ax6, lean=28, ease="out")]
    ev["hits"].append((T_WHIP, tuple(c6 - ax6 * 0.45 - up * 0.15), "whip", tuple(ax6)))
    ev["trails"] += [("G", T_WHIP + 0.01, 11.32, 0.2)]
    # ---- Goku schießt zurück: Knie in den Magen, Freezer fliegt weg
    gk = zw - ax6 * 0.55 + up * 0.25
    ZK = zw + ax6 * 3.6 + up * 1.4                   # Freezer nach dem Rückstoß (Brust)
    G += [_key(11.33, "dash", c6 - ax6 * 4.1 - up * 0.45, zw, lean=-25),
          _key(T_KNEE, "knee_R", gk, zw + ax6, ease="lin")]
    Z += [_key(T_KNEE, "gut_hit", zw + up * 0.15, gk),
          _key(11.62, "gut_hit", ZK, gk, lean=30, ease="out")]
    ev["hits"].append((T_KNEE, tuple(zw + up * 0.05 - ax6 * 0.12), "knee", tuple(ax6)))
    ev["trails"] += [("G", 11.3, 11.49, 0.2), ("Z", 11.57, 11.72, 0.2)]
    # ---- Zanzoken: Goku verschwindet (Nachbild bleibt), taucht über Freezer auf -> Doppelfaust in den Boden
    top = ZK - ax6 * 0.35 + up * 2.3
    G += [_key(ZAN1[0], "knee_R", gk + ax6 * 0.1, zw + ax6),
          _key(ZAN1[0] + 0.06, "axe_up", top, ZK, ease="lin"),           # unsichtbar umgesetzt
          _key(T_AXE - 0.08, "axe_up", top + up * 0.1, ZK),
          _key(T_AXE, "axe_down", top - up * 0.9 + ax6 * 0.15, ZK - up)]
    ev["zan"].append(("G", ZAN1, tuple(gk), tuple(top)))
    g0 = gz(*P_CRATER)
    Z += [_key(11.78, "hover", ZK + up * 0.1, gk),
          _key(11.87, "guard", ZK + up * 0.05, top, rot={"neck": (18, 0, 0), "head": (22, 0, 0)}),
          _key(T_AXE, "recoil", ZK - up * 0.3, top, lean=40),
          _key(T_SLAM, "slam", (P_CRATER[0], P_CRATER[1], g0 + 0.3), (0, 273, 48), lean=75, ease="in"),
          _key(12.3, "land", (P_CRATER[0], P_CRATER[1], g0 - 1.1), (0, 273, 46), air=False),
          _key(12.78, "land", (P_CRATER[0] + 0.1, P_CRATER[1], g0 - 1.1), (0, 273, 46), air=False),
          _key(12.97, "fly", A1 - up * 0.4, GB1, lean=10, ease="lin"),
          _key(13.08, "point_R", A1, GB1)]
    ev["hits"].append((T_AXE, tuple(ZK + up * 0.35), "axe", (0.0, 0.0, -1.0)))
    ev["trails"] += [("Z", 12.05, 12.17, 0.24), ("Z", 12.8, 13.0, 0.26)]
    G += [_key(12.25, "hover", top - up * 1.2 - ax6 * 0.8, (P_CRATER[0], P_CRATER[1], g0)),
          _key(12.8, "guard", (-1.5, 273.8, 46.2), (3, 279, g0 + 2)),
          _key(13.08, "guard", GB1, A1)]
    # ---- Todesstrahlen: Strahl 1 durchschlägt Gokus Nachbild (Zanzoken), Strahl 2 knapp unter ihm durch
    shots = [(13.15, tuple(A1), ground_hit(A1, GB1)), (13.4, tuple(A2), ground_hit(A2, GB2 - up * 2.2))]
    for (t, a, p) in shots:
        Z += [_key(t - 0.05, "point_R", a, p, rot=_aim_rot(a, p)),
              _key(t + 0.08, "point_R", a, p, rot=_aim_rot(a, (p[0], p[1], p[2] + 3)))]
    G += [_key(ZAN2[0] - 0.01, "guard", GB1 + up * 0.02, A1),
          _key(ZAN2[0] + 0.04, "guard", GB2, A2, ease="lin"),             # unsichtbar umgesetzt
          _key(13.36, "guard", GB2 + up * 0.05, A2),
          _key(13.46, "fly", GB2 + Vector((0.5, 1.2, 1.8)), A2, roll=40, ease="out"),
          _key(13.56, "fly", (-5.0, 276.5, 50.2), (4, 284, 53.4), lean=-15, roll=-20)]
    ev["zan"].append(("G", ZAN2, tuple(GB1), tuple(GB2)))
    ev["trails"] += [("G", 13.36, 13.5, 0.18)]
    # ---- Trennung: Freezer übers Meer, Goku an den Nordrand (70–130 m/s)
    Z += [_key(13.6, "fly", (3.0, 290.0, 53.0), (-5, 276.5, 50), lean=-10),
          _key(13.95, "fly", (3.0, 335.0, 45.0), G_RIM, lean=20, ease="lin"),
          _key(14.3, "point_R", Z_SEA, G_RIM, rot=_aim_rot(Z_SEA, G_RIM), ease="out")]
    G += [_key(13.9, "fly", (-2.5, 300.0, 47.0), Z_SEA, lean=-65, ease="lin"),
          _key(14.25, "hover", G_RIM + Vector((0.2, -0.6, 0.4)), Z_SEA, ease="out"),
          _key(14.45, "kame_charge", G_RIM, Z_SEA),
          _key(14.95, "kame_charge", G_RIM + Vector((0, 0.05, -0.05)), Z_SEA)]
    ev["trails"] += [("Z", 13.58, 14.34, 0.28), ("G", 13.54, 14.3, 0.24)]
    fire = _aim_rot(G_RIM, Z_SEA, both=True)
    G += [_key(15.05, "kame_fire", G_RIM, Z_SEA, rot=fire),
          _key(T_CLIMAX + 0.5, "kame_fire", G_RIM + Vector((0, -0.35, 0.05)), Z_SEA, rot=fire),
          _key(16.9, "hover", G_RIM + Vector((0, -0.2, 0.3)), Z_SEA),
          _key(20.2, "hover", G_RIM + Vector((0, -0.25, 0.45)), Z_SEA + Vector((0, 0, 6)))]
    Z += [_key(15.0, "point_R", Z_SEA, G_RIM, rot=_aim_rot(Z_SEA, G_RIM)),
          _key(T_CLIMAX - 0.1, "point_R", Z_SEA + Vector((0, 0.8, 0)), G_RIM, rot=_aim_rot(Z_SEA, G_RIM)),
          _key(T_CLIMAX + 0.05, "recoil", Z_SEA + Vector((0, 1.5, 0.3)), G_RIM, lean=40)]
    return G, Z, shots, ev


HITS = [(T_CLASH, 3)] + [(t, 1) for t, _, _ in FLURRY] + [(T_WHIP, 2), (T_KNEE, 3), (T_AXE, 3), (T_SLAM, 4),
                                                          (T_CLIMAX, 4)]
SHAKES = [(T_CLASH, 1.2, 6)] + [(t, 0.35, 3) for t, _, _ in FLURRY] + [
    (T_WHIP, 0.7, 5), (T_KNEE, 1.1, 6), (T_AXE, 1.0, 6), (T_SLAM, 2.4, 9), (13.2, 1.0, 6), (13.45, 1.0, 6),
    (15.05, 0.7, 8), (T_CLIMAX, 3.6, 12), (16.35, 1.3, 10)]
CAM_HOLDS = [(T_SLAM, 3), (T_CLIMAX, 3)]


def camera_positions(frames):
    return choreo.keyed_path(CAM_KEYS, FPS, frames)


def hit_warp(t):
    return choreo.time_warp(np.asarray(t, float), HITS, FPS)


def frame_at(tau):
    """Erster Frame, in dem die Kämpfer-Zeit (mit Hit-Stop) tau erreicht."""
    f = np.arange(1, FPS * (SECONDS + 1))
    return int(f[np.argmax(hit_warp((f - 1.0) / FPS) >= tau - 1e-6)])


def camera_path(frames, G_keys, Z_keys, shots):
    """Positionen (Speed-Ramp) + Blickführung: Flug geradeaus (leicht zum Tafelberg), ab dem Anflug auf das
    Kämpferpaar (Kämpferbahnen geglättet: die Kamera folgt schnellen Haken und Teleports ohne Reißen), beim
    Einschlag auf den Krater, bei den Todesstrahlen auf Goku und die Einschläge, beim Strahlenduell zwischen
    Goku und Freezer, am Ende auf Goku und die Rauchsäule."""
    t, pos, v = camera_positions(frames)
    tw = choreo.time_warp(t, CAM_HOLDS, FPS)
    pos = np.stack([np.interp(tw, t, pos[:, k]) for k in range(3)], axis=1)
    a = choreo.track(G_keys, hit_warp(t)) + np.array([0, 0, CH])
    b = choreo.track(Z_keys, hit_warp(t)) + np.array([0, 0, CH])
    a = np.stack([fpv._gauss_smooth(a[:, k], 0.12 * FPS) for k in range(3)], axis=1)
    b = np.stack([fpv._gauss_smooth(b[:, k], 0.12 * FPS) for k in range(3)], axis=1)
    sm = choreo.smooth

    def between(p, q, wq):
        """Blickziel zwischen zwei Punkten nach Winkel (nicht nach Abstand): gewichtete Richtungen."""
        dp = p - pos
        dq = q - pos
        u = dp / np.linalg.norm(dp, axis=1)[:, None] * (1 - wq) + dq / np.linalg.norm(dq, axis=1)[:, None] * wq
        return pos + u / np.linalg.norm(u, axis=1)[:, None] * 12.0
    pair = between(a, b, 0.5)
    crater = np.array([P_CRATER[0], P_CRATER[1], gz(*P_CRATER) + 1.0])
    tgt = pair.copy()
    wc = sm(12.0, 12.25, t) * (1 - sm(12.8, 13.0, t))                        # Einschlag: auf den Krater
    tgt = tgt * (1 - wc[:, None]) + between(a, np.tile(crater, (len(t), 1)), 0.62) * wc[:, None]
    hits = np.array([p for (_, _, p) in shots]).mean(axis=0)
    wb = sm(12.85, 13.05, t) * (1 - sm(13.5, 13.75, t))              # Todesstrahlen: Freezer, Goku, Einschläge
    tgt = tgt * (1 - wb[:, None]) + between(between(a, b, 0.42), np.tile(hits, (len(t), 1)), 0.3) * wb[:, None]
    wd = sm(13.6, 14.3, t)                                                    # Duell: Goku vorn links, Freezer
    tgt = tgt * (1 - wd[:, None]) + between(a, b, 0.5) * wd[:, None]
    smoke = np.tile(np.array([Z_SEA.x, Z_SEA.y, Z_SEA.z + 7.0]), (len(t), 1))
    we = sm(16.2, 17.6, t)                                                    # Ende: Goku + Rauchsäule
    tgt = tgt * (1 - we[:, None]) + between(a, smoke, 0.45) * we[:, None]
    keys_w = [(0.0, 0.0), (5.0, 0.0), (7.3, 0.0), (8.4, 0.25), (9.4, 0.5), (10.0, 0.85), (10.4, 1.0), (21.0, 1.0)]
    w = choreo.pchip([k for k, _ in keys_w], [x for _, x in keys_w], np.clip(t, 0, 21.0))
    d = tgt - pos
    look_yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    look_pit = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
    # Hook: Blick 5° höher (mehr Felsnadeln/Himmel, weniger dunkles Nahwasser); Steigflug: 12° höher (Kante, Himmel)
    pos_f, quats, info = fpv.fpv_orient(pos, FPS, look_pitch=-3.0, pitch_follow=0.45, bank_gain=1.0, max_bank=32,
                                        micro=1.0, seed=13, look=(w, look_yaw, look_pit),
                                        pitch_overrides=[(-1.5, 2.2, 5.0), (5.4, 7.5, 12.0)])
    choreo.camera_shakes(pos_f, quats, t, SHAKES, FPS, seed=78)
    info.update(v=v, t=t, look_w=w, tgt=tgt, G=a, Z=b)
    return pos_f, quats, info


def sound_markers(info):
    t, pos = info["t"], info["pos"]

    def when(cond):
        idx = np.where(cond)[0]
        return float(t[idx[0]]) if len(idx) else None
    m = [("AMBIENCE", 0.0), ("WHOOSH", when(pos[:, 1] > 50.0)), ("WHOOSH", when(pos[:, 1] > 70.0)),
         ("WHOOSH", 5.7), ("WHOOSH", when(pos[:, 2] > 41.0)), ("WHOOSH", when(pos[:, 1] > 231.0))]
    m += [("WHOOSH", T_RUSH), ("ZANZOKEN", (frame_at(ZAN1[0]) - 1) / FPS), ("ZANZOKEN", ZAN2[0]),
          ("WHOOSH", 13.6)]
    m += [("IMPACT", h) for h, _ in HITS[:-1]] + [("IMPACT", 13.15), ("IMPACT", 13.4)]
    m += [("BEAT_DROP", T_CLIMAX), ("AMBIENCE", 17.0)]
    return sorted([(nm, tt) for nm, tt in m if tt is not None], key=lambda x: x[1])


# ------------------------------------------------------------------------------------------------ Schritt 3.4
def _expr(fig, seq):
    """Mimik-Zeitplan [(t, {Shape Key: Wert})]: jeder Eintrag setzt alle Ausdrücke der Figur (übrige = 0)."""
    names = sorted({k for _, d in seq for k in d})
    AC.set_expression(fig, [(F(t), {n: d.get(n, 0.0) for n in names}) for t, d in seq])


def stage_fight(G_keys, Z_keys, n, cam_pos):
    """Goku (SSJ) und Freezer aus den Tripo-Modellen (tripo_chars, Toon-Look mit Kontur; NIDO_CHARS=anime nimmt
    die selbst gebauten Figuren aus anime_chars), gebacken im Dragon-Ball-Timing; Freezers Schwanz schwingt
    nach und peitscht beim Hieb; Mimik (nur anime_chars) passend zum Kampfverlauf."""
    AC.set_light(fpv.sun_dir(SUN_ELEV, SUN_AZIM), fpv.sun_dir(RIM["elev"], RIM["azim"]))
    if os.environ.get("NIDO_CHARS", "tripo") == "anime":
        G, Z = AC.goku(), AC.freezer()
    else:
        G, Z = TC.goku(), TC.freezer()
    kw = dict(style=choreo.dbz_style, lag_scale=0.35, lean_tau=0.04)
    choreo.bake_fighter(G, G_keys, n, FPS, HITS, cam_pos=cam_pos, look_win=(17.0, 21.0), seed=3, **kw)
    choreo.bake_fighter(Z, Z_keys, n, FPS, HITS, seed=4, **kw)
    AC.tail_follow(Z, n, FPS, whips=[(T_WHIP, 60.0)], warp=hit_warp)
    _expr(G, [(9.2, {}), (9.45, {"Clench": 1}), (11.42, {"Clench": 1}), (11.47, {"Shout": 0.8}),
              (11.6, {"Clench": 1}), (11.88, {"Clench": 1}), (11.93, {"Shout": 1}), (12.1, {"Shout": 1}),
              (12.25, {"Clench": 1}), (14.95, {"Clench": 1}), (15.03, {"Shout": 1}), (16.25, {"Shout": 1}),
              (16.7, {"Calm": 1})])
    _expr(Z, [(9.2, {"Smirk": 1}), (9.95, {"Smirk": 1}), (10.02, {"Angry": 1}), (11.0, {"Angry": 1}),
              (11.07, {"Smirk": 1}), (11.44, {"Smirk": 1}), (11.49, {"Shock": 1}), (11.72, {"Shock": 0.7}),
              (11.8, {"Angry": 0.6, "Shock": 0.4}), (11.86, {"Shock": 1}), (12.3, {"Shock": 1}),
              (12.55, {"Angry": 1}), (15.55, {"Angry": 1}), (15.62, {"Shock": 1})])
    G.render_objs = dbz_fx.render_objects(G.base)
    Z.render_objs = dbz_fx.render_objects(Z.base)
    # Freezer verschwindet im Feuerball
    for f, s in ((F(T_CLIMAX) + 1, 1.0), (F(T_CLIMAX) + 2, 0.001)):
        Z.base.scale = (s, s, s)
        Z.base.keyframe_insert("scale", frame=f)
    return G, Z


def F(t):
    return int(round(t * FPS)) + 1


# ------------------------------------------------------------------------------------------------ Schritt 3.5 – FX
HIT_FX = {  # Art: (Radius, Licht W, Funken, Luftring-Radius)
    "clash": (3.2, 9000.0, 110, 0.0), "flurry": (0.9, 1500.0, 40, 1.6), "whip": (1.4, 3000.0, 60, 2.2),
    "knee": (1.6, 6000.0, 110, 4.0), "axe": (1.6, 6000.0, 110, 3.0)}


def motion_fx(G, Z, ev):
    """Ki-Spuren hinter allen schnellen Vorstößen, Zanzoken (Verschwinden, flackerndes Nachbild, Auftauchen)."""
    figs = {"G": (G, GOLD_TRAIL), "Z": (Z, PURPLE_TRAIL)}
    for k, (who, t0, t1, r) in enumerate(ev["trails"]):
        fig, col = figs[who]
        dbz_fx.ki_trail(f"KiTrail{who}{k}", fig, t0, t1, col, radius=r, fps=FPS)
    spans = []
    for k, (who, (tau_out, tau_in), p_out, p_in) in enumerate(ev["zan"]):
        fig, col = figs[who]
        f_out, f_in = frame_at(tau_out), frame_at(tau_in)
        dbz_fx.afterimage(f"Zanzoken{k}", fig.render_objs, f_out - 1, f_out, dbz_fx.FLICKER)
        dbz_fx.pop_ring(f"ZanOut{k}", p_out, f_out, axis=(0, 0, 1), r_max=1.4, color=col)
        dbz_fx.pop_ring(f"ZanIn{k}", p_in, f_in, axis=(0, 0, 1), r_max=1.8, color=col)
        spans.append((f_out, f_in))
        print(f"Zanzoken {k}: weg Frame {f_out}, wieder da Frame {f_in}", flush=True)
    dbz_fx.vanish(dbz_fx.render_objects(G.base), spans)


def fight_fx(G, Z, shots, rock, ground_mat, mesa, ev):
    """Treffer, Ki-Spuren, Zanzoken, Krater mit Bruchstücken, Todesstrahlen mit glühenden Kratern,
    Kamehameha-Duell, Explosion."""
    sc = bpy.context.scene
    vfx.aura("GokuAura", G.base, GOLD, F(9.4), F(17.8), height=1.95, width=0.95, light_w=250.0, opacity=0.25,
             edge_w=0.1, strength=0.6)                                   # nach dem Durchbruch: Aura erlischt
    motion_fx(G, Z, ev)                                                   # (Aura verschwindet beim Zanzoken mit)
    # Teaser-Blitze (fern, Glühkern gut sichtbar)
    for k, (t, c, _) in enumerate(TEASERS):
        vfx.burst(f"Teaser{k}", c, F(t) - 1, (1.0, 0.8, 0.45), r_max=11.0, dur=12, light_w=90000.0, ring=False,
                  bolts=False, seed=30 + k, core_s=25.0, glow_s=6.0, glow_alpha=0.35, core_color=(1.0, 0.95, 0.85))
    # Nahkampf-Treffer: Blitz, Funken, Luftring quer zur Schlagachse
    for k, (t, c, kind, ax) in enumerate(ev["hits"]):
        r, lw, ns, ring = HIT_FX[kind]
        vfx.burst(f"Hit{k}", c, F(t), (1.0, 0.78, 0.35), r_max=r, dur=9 if r > 1 else 6, light_w=lw, bolts=False,
                  ring=r > 1, seed=20 + k, core_s=8.0, glow_s=2.0, glow_alpha=0.25, ring_s=2.5,
                  core_color=(1.0, 0.95, 0.8), core_k=1.0 if kind == "clash" else 0.5)
        vfx.sparks_gn(f"HitSparks{k}", c, t, n=ns, speed=(6, 14), life=(0.15, 0.45), color=(1.0, 0.8, 0.45),
                      strength=60.0, seed=40 + k)
        if ring:
            dbz_fx.pop_ring(f"HitRing{k}", c, F(t) + 1, axis=ax, r_max=ring, color=(1.0, 0.92, 0.75))
    vfx.shockwave("ClashWave0", tuple(C0), F(T_CLASH) + 1, r_max=9.0, dur=10, color=(1.0, 0.9, 0.7), thick=0.12,
                  glow=2.0)
    # ---- Einschlag im Boden (12,15 s): Krater, Bruchstücke (Voronoi + Rigid Body), Staub, Druckwelle
    t_c = T_SLAM
    cx, cy = P_CRATER
    g0 = gz(cx, cy)
    craters = [(cx, cy, 3.2, t_c, 71)]
    for k, (t, a, p) in enumerate(shots):
        craters.append((p[0], p[1], 2.1, t + 0.1, 72 + k))
    holes = []
    for k, (x, y, r, t, seed) in enumerate(craters):
        hm = namek.heat_variant(ground_mat, f"CraterMat{k}", F(t), fps=FPS, cool_s=3.2)
        namek.crater(f"Crater{k}", x, y, r, mesa_height, hm, F(t), fps=FPS, seed=seed)
        holes.append((x, y, r * 1.5))
    namek.punch_hole(mesa, holes)
    vfx.burst("SlamBurst", (cx, cy, g0 + 0.8), F(t_c), (1.0, 0.7, 0.4), r_max=4.5, dur=14, light_w=40000.0, bolts=False,
              seed=61, ring_dz=-0.6, core_s=12.0, glow_s=3.0, ring_s=3.0, core_color=(1.0, 0.9, 0.75))
    vfx.shockwave("SlamWave", (cx, cy, g0 + 0.35), F(t_c) + 1, r_max=16.0, dur=12, color=(0.95, 0.85, 0.7), thick=0.18,
                  glow=2.0)
    vfx.dust_gn("SlamDust", (cx, cy, g0 + 0.2), t_c, n=1400, r_max=9.0, rise=2.2, life=2.2, color=(0.38, 0.36, 0.33),
                size=(0.03, 0.09), seed=62)
    vfx.sparks_gn("SlamSparks", (cx, cy, g0 + 0.5), t_c, n=200, speed=(8, 18), life=(0.3, 0.8), color=(1.0, 0.7, 0.35),
                  strength=50.0, seed=63)
    # Freezer bricht aus dem Krater: Staubstoß und Luftring
    vfx.dust_gn("ExitDust", (cx, cy, g0 + 0.2), 12.8, n=500, r_max=5.0, rise=3.0, life=1.4, color=(0.4, 0.38, 0.35),
                size=(0.03, 0.08), seed=65)
    dbz_fx.pop_ring("ExitRing", (cx, cy, g0 + 0.6), F(12.8), r_max=4.0, color=PURPLE_TRAIL)
    # Bruchstücke: Bodenplatte über dem Krater zerlegt, von unten weggesprengt
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=2.3, radius2=2.2, depth=0.45)
    for v in bm.verts:
        v.co.x += cx
        v.co.y += cy
        v.co.z += g0 - 0.2
    rng = np.random.default_rng(64)
    seeds = [(cx + rng.uniform(-2, 2), cy + rng.uniform(-2, 2), g0 - 0.2 + rng.uniform(-0.2, 0.2)) for _ in range(16)]
    pieces = vfx.voronoi_fracture("SlamChunk", bm, seeds, rock)
    bm.free()
    floor = fpv.grid_mesh("DebrisFloor", 60, 60, 2, 2, lambda X, Y: X * 0 + g0 - 0.05, None, origin=(cx, cy))
    vfx.rigid_sim(pieces, [floor], F(t_c), F(t_c) + 60, [((cx, cy, g0 - 1.2), 38000.0, F(t_c), F(t_c) + 1)], fps=FPS)
    bpy.data.objects.remove(floor, do_unlink=True)
    for ob in pieces:
        for f, hid in ((1, True), (F(t_c) - 1, True), (F(t_c), False)):
            ob.hide_render = hid
            ob.keyframe_insert("hide_render", frame=f)
    # ---- Todesstrahlen: Strahl, Einschlag, glühender Krater (s. o.), Staub, Brocken
    for k, (t, a, p) in enumerate(shots):
        a, p = Vector(a), Vector(p)
        sc.frame_set(F(t))
        o = Z.J["wrist.R"].matrix_world.translation.copy()
        vfx.energy_ball(f"DeathTip{k}", Z.J["wrist.R"], (0, 0, -0.17), DEATH, 0.09, F(t) - 5, F(t) - 1, F(t) + 5,
                        light_w=150.0, swirl=False, spin=False)
        vfx.beam(f"DeathBeam{k}", o, p - o, (p - o).length, 0.18, DEATH, F(t), F(t) + 2, F(t) + 7, light_w=4000.0,
                 core_s=5.0, whiten=0.3, glow_s=3.0)
        vfx.burst(f"BeamImpact{k}", p + Vector((0, 0, 0.8)), F(t) + 2, (1.0, 0.55, 0.85), r_max=4.0, dur=16,
                  light_w=35000.0, bolts=False, seed=80 + k, ring_dz=-0.7, core_s=12.0, glow_s=4.0, ring_s=3.0,
                  core_color=(1.0, 0.9, 0.95))
        vfx.dust_gn(f"BeamDust{k}", tuple(p + Vector((0, 0, 0.2))), t + 0.1, n=900, r_max=7.0, rise=2.5, life=2.0,
                    color=(0.36, 0.34, 0.31), size=(0.03, 0.08), seed=85 + k)
        vfx.debris(f"BeamDebris{k}_", p + Vector((0, 0, 0.3)), F(t) + 2, rock, n=12, speed=12.0, size=0.22,
                   seed=88 + k, ground=p.z - 0.1)
    # ---- Kamehameha gegen Todesstrahl
    vfx.energy_ball("KameCharge", G.J["wrist.R"], (-0.07, 0.06, -0.13), KAME, 0.34, F(14.45), F(14.95), F(15.1),
                    light_w=600.0)
    vfx.energy_ball("DeathCharge", Z.J["wrist.R"], (0, 0, -0.17), DEATH, 0.22, F(14.4), F(14.95), F(15.05),
                    light_w=500.0, swirl=False)
    sc.frame_set(F(15.05))
    o = (G.J["wrist.R"].matrix_world.translation + G.J["wrist.L"].matrix_world.translation) / 2
    oz = Z.J["wrist.R"].matrix_world.translation.copy()
    L = (oz - o).length
    dk = (oz - o).normalized()
    o = o + dk * 0.2
    Lc = 0.55 * L
    t0 = 15.05
    wob = [(0.0, 0.01), (0.25, Lc), (0.45, Lc - 3.0), (0.6, Lc + 2.0), (0.75, Lc - 1.5), (T_CLIMAX - t0, L)]
    kl = [(F(t0 + dt), v) for dt, v in wob]
    zl = [(F(t0), 0.01)] + [(F(t0 + dt), L - v) for dt, v in wob[1:-1]] + [(F(T_CLIMAX) - 1, 0.3)]
    vfx.beam("Kamehameha", o, dk, kl, 0.95, KAME, F(t0), F(t0 + 0.25), F(T_CLIMAX + 0.5), light_w=5000.0,
             wobble=(F(t0 + 0.25), F(T_CLIMAX + 0.4), 0.12), core_s=5.0, whiten=0.35, glow_s=3.0, core_r=0.3)
    vfx.beam("DeathBeamDuel", oz, -dk, zl, 0.55, DEATH, F(t0), F(t0 + 0.25), F(T_CLIMAX), light_w=12000.0,
             wobble=(F(t0 + 0.25), F(T_CLIMAX - 0.1), 0.15), core_s=5.0, whiten=0.35, glow_s=2.6)
    mid = bpy.data.objects.new("DuelPoint", None)
    fpv.link(mid)
    for f, v in kl[1:]:
        mid.location = o + dk * v
        mid.keyframe_insert("location", frame=f)
    vfx.energy_ball("DuelBall", mid, (0, 0, 0), (0.62, 0.55, 1.0), 2.0, F(t0 + 0.2), F(t0 + 0.3), F(T_CLIMAX),
                    light_w=30000.0)
    vfx.lightning("DuelArcs", mid, (0, 0, 0), (0.75, 0.7, 1.0), F(t0 + 0.25), F(T_CLIMAX), radius=4.5, n_bolts=10,
                  variants=6, seed=12, light_w=0.0, thickness=0.06)
    b = vfx.burst("DuelMeet", o + dk * Lc, F(t0 + 0.25), (0.6, 0.55, 1.0), r_max=6.0, dur=18, light_w=90000.0,
                  bolts=False, seed=13, core_s=12.0, glow_s=3.0, ring_s=3.5)
    b.rotation_mode = "QUATERNION"
    b.rotation_quaternion = dk.to_track_quat("Z", "Y")
    # ---- Klimax 15,8 s: Explosion über dem Meer
    explosion(Z_SEA)
    return craters


def explosion(c):
    """Durchbruch: weißer Blitz, Feuerball + Rauchsäule (prozedurales Volumen, keine Fluid-Simulation – Mantaflow
    bricht in diesem bpy-Build ab), Druckwelle in der Luft und auf dem Wasser, Gischt, Funken; Licht warm -> kühl."""
    f0 = F(T_CLIMAX)
    vfx.burst("FinalFlash", c, f0, (1.0, 0.75, 0.4), r_max=7.0, dur=10, light_w=0.0, seed=14, core_s=20.0,
              glow_s=4.0, glow_alpha=0.4, ring_s=4.0, core_color=(1.0, 0.95, 0.85), bolts=False)
    vfx.volume_blast("FinalBlast", c, f0, r_fire=8.0, r_smoke=9.0, rise=14.0, dur=int(4.4 * FPS), fps=FPS, seed=7,
                     fire=14.0)
    lt = vfx.point_light("FinalLight", (1.0, 0.62, 0.3), 0.0, 5.0)
    lt.location = c
    vfx.key_energy(lt, [(f0 - 1, 0.0), (f0, 900000.0), (f0 + 4, 600000.0), (f0 + 14, 220000.0), (f0 + 40, 60000.0),
                        (F(20.2), 20000.0)])
    for f, col in ((f0, (1.0, 0.62, 0.3)), (f0 + 18, (1.0, 0.7, 0.45)), (f0 + 44, (0.72, 0.8, 1.0))):
        lt.data.color = col
        lt.data.keyframe_insert("color", frame=f)
    vfx.shockwave("FinalWave", c, f0 + 1, r_max=60.0, dur=20, color=(1.0, 0.85, 0.65), thick=0.5, glow=2.5)
    vfx.shockwave("SeaWave", (c.x, c.y, 0.3), f0 + 6, r_max=70.0, dur=26, color=(0.9, 0.95, 1.0), thick=0.15, glow=0.25)
    vfx.sparks_gn("FinalSparks", c, T_CLIMAX, n=500, speed=(15, 35), life=(0.5, 1.4), color=(1.0, 0.7, 0.35),
                  strength=60.0, radius=(0.03, 0.08), seed=15)
    vfx.dust_gn("Spray", (c.x, c.y, 0.2), T_CLIMAX + 0.3, n=2500, r_max=30.0, rise=9.0, life=3.2,
                color=(0.78, 0.85, 0.84), size=(0.05, 0.16), k_drag=2.0, seed=16)


# ------------------------------------------------------------------------------------------------ Welt
def namek_sky(nb, d):
    """Himmel (Schritt 3.2): grüner Verlauf, gegenüber der Anime-Vorlage um ~30 % entsättigt: Horizont blass
    gelbgrün, Zenit tiefes Blaugrün; Leuchten um Hauptsonne und dritte Sonne. Diffuses Himmelslicht kommt
    entsättigt an (sonst färbt es blaues Gras und beigen Fels türkis)."""
    x, y, z = nb.sep(d)
    zz = nb.math("MAXIMUM", z, 0.0)
    t = nb.math("POWER", zz, 0.5)
    col = nb.ramp(t, [(0.0, (0.80, 0.89, 0.56)), (0.07, (0.50, 0.75, 0.33)), (0.30, (0.19, 0.50, 0.19)),
                      (1.0, (0.06, 0.30, 0.12))])
    below = nb.math("LESS_THAN", z, 0.0)
    col = nb.mix(below, col, (0.52, 0.64, 0.50))
    for (e, a, s8, s64) in ((SUN_ELEV, SUN_AZIM, 0.55, 1.6), (SUNS2[1][0], SUNS2[1][1], 0.45, 1.6)):
        sd = fpv.sun_dir(e, a)
        dp = nb.math("MAXIMUM", nb.vmath("DOT_PRODUCT", d, tuple(sd)), 0.0)
        glow = nb.math("ADD", nb.math("MULTIPLY", nb.math("POWER", dp, 8.0), s8),
                       nb.math("MULTIPLY", nb.math("POWER", dp, 64.0), s64))
        col = nb.vmath("ADD", col, nb.vmath("SCALE", (1.0, 0.93, 0.78), scale=glow))
    lp = nb.node("ShaderNodeLightPath")
    vivid = nb.math("MAXIMUM", lp.outputs["Is Camera Ray"], lp.outputs["Is Glossy Ray"])
    lum = nb.vmath("DOT_PRODUCT", col, (0.2126, 0.7152, 0.0722))
    neutral = nb.mix(0.6, col, nb.comb(lum, lum, lum))
    return nb.mix(vivid, neutral, col)


def namek_rock():
    """Fels (Schritt 3.3): warmes Beige bis Rostbraun, Schichtfugen, Regen-/Sinterstreifen, dunkle Nässe- und
    Algenzone an der Wasserlinie, Moos in Nischen, Streuung pro Objekt."""
    return nature.rock_material("NamekRock", c1=(0.15, 0.10, 0.10), c2=(0.58, 0.40, 0.28), c3=(0.44, 0.31, 0.25),
                                wet_line=1.6, algae=(0.03, 0.06, 0.05), moss=(0.02, 0.10, 0.28), moss_amount=0.22,
                                strata_scale=2.2, bump=1.0, crack_w=0.15, scale=1.5, lichen=0.12,
                                moss_tex=os.path.join(fpv.ASSETS, "grasslight-big.jpg"), moss_tex_scale=4.0,
                                variation=0.6, bedding=0.45, bed_h=2.2, streak=0.5)


def namek_cliff():
    """Steilwand und Felskante des Tafelbergs: wie der Fels der Nadeln, aber kaum Moos (die flache Oberseite der
    Felskante würde sonst blau)."""
    return nature.rock_material("NamekCliff", c1=(0.15, 0.10, 0.10), c2=(0.60, 0.42, 0.29), c3=(0.45, 0.32, 0.25),
                                wet_line=1.6, algae=(0.03, 0.06, 0.05), moss=(0.03, 0.08, 0.16), moss_amount=0.04,
                                strata_scale=2.2, bump=1.0, crack_w=0.15, scale=1.5, lichen=0.1, variation=0.4,
                                bedding=0.35, bed_h=2.2, streak=0.3)


def plateau_ground():
    """Boden unter dem Gras: dunkle, humose Erde mit blaugrünen Moosflecken und Sand."""
    mat, nb, out = fpv.new_material("NamekGround")
    co = nb.coords("Object")
    n1 = nb.noise(co, scale=0.18, detail=4)
    n2 = nb.noise(co, scale=2.5, detail=4)
    soil = nb.mix(nb.out(n2, "Fac"), (0.08, 0.065, 0.05), (0.16, 0.13, 0.10))
    col = nb.mix(nb.math("GREATER_THAN", nb.out(n1, "Fac"), 0.55), soil, (0.03, 0.08, 0.16))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.85)
    nb.link(nb.bump(nb.out(n2, "Fac"), strength=0.3, distance=0.05), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def build(args):
    sc = fpv.reset()
    frames = FPS * SECONDS
    fpv.setup_render(args.out, res=args.res, fps=FPS, seconds=SECONDS, samples=args.samples,
                     motion_blur=not args.no_mblur, mist_depth=6000.0)
    extra = [(SUN_ELEV, SUN_AZIM, 0.9, 900.0, (1.0, 0.95, 0.85))]
    for (e, a, s, k) in SUNS2:
        extra.append((e, a, 0.6, 350.0 * s, (1.0, 0.96, 0.9)))
    fpv.build_world(sun_elev=SUN_ELEV, sun_azim=SUN_AZIM, sky_strength=0.8, clouds=True, cloud_cover=0.22,
                    cloud_ref=1.1, custom_sky=namek_sky, extra_suns=extra, cloud_color=(1.0, 1.0, 0.86),
                    cloud_scale=1.3)
    key = fpv.add_sun(SUN_ELEV, SUN_AZIM, strength=SUN_STRENGTH, color=(1, 1, 1), name="Sun1", angle_deg=1.2)
    key.data.use_temperature, key.data.temperature = True, SUN_KELVIN
    for i, (e, a, s, k) in enumerate(SUNS2):
        sn = fpv.add_sun(e, a, strength=s, color=(1, 1, 1), name=f"Sun{i + 2}", angle_deg=(2.5, 1.5)[i])
        sn.data.use_temperature, sn.data.temperature = True, k
    if ATMO_DENSITY > 0:
        fpv.add_atmosphere(ATMO_DENSITY, size=(900.0, 900.0, 200.0), center=(0.0, 250.0), color=(0.9, 1.0, 0.94))
    rng = np.random.default_rng(31)
    top_z = gz

    rock = namek_rock()
    ground = plateau_ground()
    # Tafelberg: Plateau als Heightfield + umlaufende Steilwand als eigenes Mesh
    cliff = namek_cliff()
    mesa = fpv.grid_mesh("Mesa", 2 * MESA_R + 60, 2 * MESA_R + 60, 400, 400, mesa_height, ground, origin=MESA_C)
    mesa.data.materials.append(cliff)            # Schuttfuß an der Wand: Fels statt Plateau-Erde
    zc = np.zeros(len(mesa.data.polygons) * 3, dtype=np.float32)
    mesa.data.polygons.foreach_get("center", zc)
    mesa.data.polygons.foreach_set("material_index", (zc[2::3] < 30.0).astype(np.int32))
    wall = namek.ring_wall("MesaWall", MESA_C, mesa_edge, plateau, cliff, n_ang=1500, n_z=150, seed=5)
    # Felsnadeln: Schichtbänke, Karstrinnen, Kopf breiter (Pilzform), leicht geneigt, Höhe variiert
    spires, far_trees = [], []
    for (x, y, r, h, seed, tx, ty) in SPIRES:
        ob = fpv.rock_mesh(f"Spire{seed}", radius=r, height=h + 4, seed=seed + 50, detail=6, taper=-0.35, mat=rock,
                           base_z=-4.0, lumpy=0.22, strata=0.6, flute=0.06, ledges=0.03, bed=(1.8, 4.0))
        ob.location = (x, y, 0)
        ob.rotation_euler = (math.radians(tx), math.radians(ty), seed * 2.1)
        spires.append(ob)
    for k, (x, y, r, h) in enumerate(((-160, 470, 26, 95), (120, 520, 20, 70), (-60, 640, 34, 120),
                                      (260, 700, 30, 85), (-330, 820, 45, 140), (40, 980, 55, 150),
                                      (420, 1050, 50, 120), (-620, 1150, 70, 170))):
        ob = fpv.rock_mesh(f"FarSpire{k}", radius=r, height=h + 4, seed=200 + k, detail=4, taper=-0.3, mat=rock,
                           base_z=-4.0, lumpy=0.22, strata=0.6, ledges=0.025)
        ob.location = (x, y, 0)
        for j in range(int(r * 0.5)):
            a = rng.uniform(0, 2 * math.pi)
            rr = r * 1.1 * math.sqrt(rng.random())
            far_trees.append((x + math.cos(a) * rr, y + math.sin(a) * rr, h - 0.6))
    for k, (x, y, r, h) in enumerate(((-900, 1400, 260, 150), (800, 1700, 300, 170))):
        def hf(X, Y, x=x, y=y, r=r, h=h, k=k):
            rr = np.sqrt((X - x) ** 2 + (Y - y) ** 2) / r
            e = 1 + 0.12 * fpv.fbm2(X / r * 2, Y / r * 2, 3, seed=40 + k)
            tt = np.clip((e - rr) / 0.12, 0, 1)
            return tt * tt * (3 - 2 * tt) * h * (1 + 0.05 * fpv.fbm2(X / 30, Y / 30, 3, seed=50 + k)) - 3
        fpv.grid_mesh(f"Island{k}", r * 2.6, r * 2.6, 180, 180, hf, rock, origin=(x, y))
    bpy.context.view_layer.update()

    # Meer (Schritt 3.2/3.3): dunkleres Blaugrün, Fresnel (Principled, IOR 1,333), Flachwasser/Brandung an Felsen
    shore_img, shore_map = ocean.shore_distance_image(spires + [wall], -200, -100, 200, 420, cell=0.5,
                                                      name="NamekShore")
    wmat = ocean.water_material("NamekSea", deep=(0.004, 0.032, 0.030), shallow=(0.025, 0.14, 0.11),
                                wake_fn=foam_builder(shore_img, shore_map), foam_amount=0.4, view_dark=0.5, micro=0.6)
    frames_all = frames
    ocean.animate_time_value(wmat, FPS, frames_all)
    tile = 100.0
    x0, y0, nx, ny = -140.0, -40.0, 3, 5
    ocean.make_ocean(x0, y0, nx, ny, tile=tile, res=13, wind=6.5, wave_scale=0.7, chop=1.1, fps=FPS, frames=frames,
                     mat=wmat, direction_deg=60, alignment=0.3, foam_coverage=0.1)
    far = ocean.water_material("NamekFarSea", deep=(0.004, 0.032, 0.030), shallow=(0.025, 0.14, 0.11), far=True)
    ocean.far_plane(x0 - tile / 2 + 1, x0 - tile / 2 + tile * nx - 1, y0 - tile / 2 + 1, y0 - tile / 2 + tile * ny - 1,
                    mat=far, z=-0.05)

    mats = {
        "clay": namek.clay_material("NamekClay"),
        "clay_dark": namek.clay_material("NamekClayDark", color=(0.62, 0.64, 0.58)),
        "glass": fpv.simple_mat("DomeGlass", (0.015, 0.03, 0.03), rough=0.06),
        "door": fpv.simple_mat("DomeDoor", (0.03, 0.05, 0.045), rough=0.4),
        "hull": namek.ship_hull_material("FriezaHull"),
        "hull_dark": namek.ship_hull_material("FriezaHullDark", color=(0.30, 0.29, 0.28)),
        "rim": namek.hazard_rim_material(),
        "shipwin": fpv.simple_mat("ShipWin", (0.4, 0.05, 0.05), rough=0.1, emission=(1.0, 0.3, 0.2), estrength=1.2),
    }
    houses = [(-20, 204, 6.5), (-36, 224, 8.0), (-22, 252, 5.5), (-46, 252, 6.0), (-28, 276, 7.5), (26, 204, 6.0),
              (30, 228, 5.0), (-17, 300, 6.0), (-44, 290, 5.0), (27, 300, 6.5)]
    for i, (x, y, r) in enumerate(houses):
        namek.house(f"House{i}", x, y, top_z(x, y), r, mats, rng)
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring_z = mesa_height(DB_C[0] + 1.7 * np.cos(ang), DB_C[1] + 1.7 * np.sin(ang))
    plinth_top = top_z(*DB_C) + 0.4
    base = float(ring_z.min()) - 0.3
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=1.7, radius2=1.5, depth=plinth_top - base)
    slab = fpv.mesh_from_bmesh(bm, "DBPlinth", rock)
    slab.location = (DB_C[0], DB_C[1], (plinth_top + base) / 2)
    namek.dragon_balls(DB_C, mesa_height, rng, radius=0.2, ring=0.8, lift=plinth_top)
    namek.spaceship(mats, SHIP_C[0], SHIP_C[1], top_z(*SHIP_C) + 1.5)

    # Kamera + Kämpfer (Schritt 3.1/3.4)
    G_keys, Z_keys, shots, ev = fight_plan()
    cam_t, cam_p, _ = camera_positions(frames)
    G, Z = stage_fight(G_keys, Z_keys, frames + 2, cam_p)
    fpv.rim_light([G.base, Z.base], RIM["elev"], RIM["azim"], strength=RIM["strength"], kelvin=RIM["kelvin"],
                  angle=RIM["angle"])
    pos, quats, info = camera_path(frames, G_keys, Z_keys, shots)
    info["pos"] = pos
    path_xy = pos[:, :2]

    # Ajisa-Bäume
    leaf = nature.leaf_material("AjisaLeaf", c1=(0.004, 0.045, 0.30), c2=(0.03, 0.12, 0.50), trans=0.3)
    bark = nature.bark_material("AjisaBark", c=(0.42, 0.33, 0.18))
    variants = [namek.ajisa_variant(f"Ajisa{i}", leaf, bark, height=h, crown_r=cr, leaves=int(900 * cr * cr / 4),
                                    seed=70 + i, leaf_size=0.36, turns=tw)
                for i, (h, cr, tw) in enumerate(((12.0, 2.8, 0.8), (15.0, 3.3, 1.0), (9.0, 2.3, 0.6), (17.0, 3.6, 1.1)))]
    _, subs = nature.make_tree_collection("AjisaTrees", variants)
    pts, scl = [], []
    avoid = [(x, y, r + 3) for (x, y, r) in houses] + [(DB_C[0], DB_C[1], 5), (SHIP_C[0], SHIP_C[1], 22)]
    avoid += [(P_CRATER[0], P_CRATER[1], 12)] + [(p[0], p[1], 8) for (_, _, p) in shots]
    groves = [(-56, 205, 9, 8), (50, 210, 8, 6), (-6, 270, 6, 4), (-52, 312, 10, 8), (38, 322, 9, 7),
              (62, 240, 7, 5), (-66, 264, 8, 6), (20, 186, 6, 4), (-28, 184, 6, 5), (-4, 334, 6, 4),
              (24, 250, 5, 3), (-16, 226, 4, 3), (26, 286, 6, 4), (-24, 312, 5, 4), (18, 316, 4, 3)]
    cand = []
    for (gx, gy, gr, gn) in groves:
        for _ in range(gn * 3):
            a = rng.uniform(0, 2 * math.pi)
            rr = gr * math.sqrt(rng.random())
            cand.append((gx + math.cos(a) * rr, gy + math.sin(a) * rr))
    edge_s = lambda x, y: mesa_edge(np.array([math.atan2(y - MESA_C[1], x - MESA_C[0])]))[0] - math.hypot(x - MESA_C[0], y - MESA_C[1])
    for (x, y) in cand:
        if any(math.hypot(x - ax, y - ay) < ar for (ax, ay, ar) in avoid):
            continue
        if np.min(np.hypot(path_xy[:, 0] - x, path_xy[:, 1] - y)) < 9:
            continue
        if edge_s(x, y) < 9:
            continue
        pts.append((x, y, top_z(x, y) - 0.2))
        scl.append(rng.uniform(0.7, 1.25))
    for p in far_trees:
        pts.append(p)
        scl.append(rng.uniform(1.2, 2.2))
    for ob, (x, y, r, h, seed, tx, ty) in zip(spires, SPIRES):
        M = ob.matrix_world
        zt = max((M @ Vector(v.co)).z for v in ob.data.vertices[-400:])
        for k in range(int(3 + r * 0.8)):
            a = rng.uniform(0, 2 * math.pi)
            rr = r * 0.9 * math.sqrt(rng.random())
            pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr, zt - 1.2))
            scl.append(rng.uniform(0.45, 0.9))
    nature.scatter_instances("Ajisa", subs, pts, scales=scl, seed=4)
    # Findlinge
    boulders = []
    for k in range(45):
        a = rng.uniform(0, 2 * math.pi)
        rr = MESA_R * 0.85 * math.sqrt(rng.random())
        x, y = MESA_C[0] + math.cos(a) * rr, MESA_C[1] + math.sin(a) * rr
        if np.min(np.hypot(path_xy[:, 0] - x, path_xy[:, 1] - y)) < 5 or any(
                math.hypot(x - ax, y - ay) < ar for (ax, ay, ar) in avoid) or edge_s(x, y) < 8:
            continue
        br = rng.uniform(0.6, 2.4)
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=4, radius=br)
        ob = fpv.mesh_from_bmesh(bm, f"Boulder{k}", rock)
        ob.scale = (rng.uniform(0.9, 1.4), rng.uniform(0.8, 1.2), rng.uniform(0.5, 0.8))
        ob.rotation_euler = (0, 0, rng.uniform(0, 6.3))
        fpv.displace_obj(ob, "CLOUDS", size=br * 0.6, strength=br * 0.35, depth=3, name=f"Boulder{k}_d")
        ob.location = (x, y, top_z(x, y) - br * 0.15)
        boulders.append((x, y, br * 1.2))
    # Sandflecken (kahle Stellen) und rote Pilzgruppen
    prng = np.random.default_rng(58)
    patches = []
    while len(patches) < 30:
        a = prng.uniform(0, 2 * math.pi)
        rr = MESA_R * 0.8 * math.sqrt(prng.random())
        x, y = MESA_C[0] + math.cos(a) * rr, MESA_C[1] + math.sin(a) * rr
        rx = prng.uniform(1.5, 6.0)
        if edge_s(x, y) < rx + 8 or any(math.hypot(x - ax, y - ay) < ar + 2 for (ax, ay, ar) in avoid):
            continue
        patches.append((x, y, rx, rx * prng.uniform(0.45, 0.9), prng.uniform(0, math.pi)))
    namek.sand_patches("SandPatches", patches, mesa_height, namek.sand_material())
    mclusters = []
    while len(mclusters) < 50:
        k = int(prng.integers(0, len(path_xy)))
        p = path_xy[k]
        if not (190 < p[1] < 318):
            continue
        tng = path_xy[min(k + 1, len(path_xy) - 1)] - path_xy[max(k - 1, 0)]
        tng = tng / (np.linalg.norm(tng) + 1e-9)
        x, y = p + np.array([-tng[1], tng[0]]) * prng.uniform(2.5, 14) * prng.choice([-1, 1])
        if any(math.hypot(x - ax, y - ay) < ar for (ax, ay, ar) in avoid) or edge_s(x, y) < 8:
            continue
        mclusters.append((float(x), float(y)))
    namek.mushrooms("Mushroom", mclusters, mesa_height, prng)

    # Blaues Gras (Schritt 3.3): Büschel, kahle Stellen, Variation; Wind + Druckwellen + Krater (3.5)
    excl = [(x, y, r * 0.95) for (x, y, r) in houses] + [(DB_C[0], DB_C[1], 1.7), (SHIP_C[0], SHIP_C[1], 17.5)]
    excl += boulders + [(x, y, min(rx, ry) * 0.7) for (x, y, rx, ry, _) in patches]

    def on_top(X, Y):
        r = np.hypot(X - MESA_C[0], Y - MESA_C[1])
        return mesa_edge(np.arctan2(Y - MESA_C[1], X - MESA_C[0])) - r > 7.5
    grass = namek.grass_field("NamekGrass", mesa_height, path_xy, namek.grass_blade_material(), rng, ymin=176,
                              ymax=330, exclude=excl, mask_fn=on_top, clump=0.6, max_blades=2_200_000,
                              height=(0.1, 0.42))
    craters = fight_fx(G, Z, shots, rock, ground, mesa, ev)
    shocks = [(C0.x, C0.y, T_CLASH, 45.0, 0.35, 1.5), (P_CRATER[0], P_CRATER[1], 12.15, 38.0, 0.9, 2.0)]
    shocks += [(x, y, t, 30.0, 0.55, 1.5) for (x, y, r, t, _) in craters[1:]]
    shocks += [(Z_SEA.x, Z_SEA.y, T_CLIMAX, 75.0, 1.0, 4.0)]
    namek.grass_motion(grass, shocks=shocks, burns=[(x, y, r * 1.15, t) for (x, y, r, t, _) in craters])

    cam = fpv.make_camera(pos, quats, fov_deg=60.0)
    cam.data.sensor_fit = "VERTICAL"
    cam.data.sensor_height = 36.0
    cam.data.lens_unit = "MILLIMETERS"
    cam.data.lens = LENS_MM
    marks = sound_markers(info)
    for name, tt in marks:
        sc.timeline_markers.new(name, frame=int(round(tt * FPS)) + 1)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "sound_markers.csv"), "w") as fh:
        fh.write("marker,seconds,frame\n")
        for name, tt in marks:
            fh.write(f"{name},{tt:.2f},{int(round(tt * FPS)) + 1}\n")
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.volume_step_rate = 2.0          # Explosionsvolumen: gröbere Schritte (Kosten), Detail reicht auf 50 m
    sc.cycles.volume_max_steps = 128
    print(f"Tempo {info['v'].min():.1f}–{info['v'].max():.1f} m/s")
    return sc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/nido_db")
    ap.add_argument("--res", type=lambda s: tuple(int(v) for v in s.split("x")), default=(1080, 1920))
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--no-mblur", action="store_true")
    ap.add_argument("--save-blend", default=None)
    args = ap.parse_args()
    build(args)
    fpv.try_gpu()
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
    frames = None
    if args.frames:
        if "-" in args.frames:
            a, b = args.frames.split("-")
            frames = range(int(a), int(b) + 1)
        else:
            frames = [int(v) for v in args.frames.split(",")]
    fpv.render_frames(args.out, frames)
