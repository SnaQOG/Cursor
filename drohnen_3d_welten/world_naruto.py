"""Welt 2 – Naruto: Konohagakure mit Hokage-Felsen.

Flug (20 s, eine durchgehende Aufnahme, Speed-Ramp 1–33 m/s, fest 35 mm):
  0–2 s     Hook: Blick durchs Tor aufs Dorf, die Residenz und die Gesichter (ferne Blitze auf dem Dach), Durchflug
  2–7 s     Hauptstraße tief (3,5–5 m) unter Laternenkabeln und Wäsche, Passanten und Stände
  7–9 s     links am alten Baum vorbei, schnell über den Platz
  9–10,9 s  Steigflug über die Brüstung aufs Dach der Hokage-Residenz – der Kampf läuft dort bereits
  10,9–17 s Dachkampf: Luftzusammenprall, Tritt/Block, Schlag/Block, Sprungtritt, Konter gegen ein Horn,
            Rasengan gegen Chidori (Klimax 15,0 s)
  17–20 s   ruhiges Schlussbild: Naruto in Untersicht, dahinter die fünf Hokage-Gesichter
Referenzen: Anime-/Spiel-Standbilder von Konoha (Pastellfassaden, bunte Dächer, Wassertanks,
ockerfarbener Sandstein-Felsen mit Laufspuren, Treppen, Kuppelbauten), Model Sheets Naruto/Sasuke.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import bmesh  # noqa: E402,I100
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import choreo  # noqa: E402
import fpv  # noqa: E402
import konoha  # noqa: E402
import konoha_life  # noqa: E402
import nature  # noqa: E402
import ninja  # noqa: E402
import vfx  # noqa: E402
from figures import POSES  # noqa: E402
import sunny  # noqa: E402
import textures  # noqa: E402

FPS = 24
SECONDS = 20
SPEED = 19.0
# Licht (Schritt 3.2): tiefe, warme Nachmittagssonne aus Westsüdwest – Streiflicht auf den Gesichtern, lange
# Schatten quer über die Straße; kühles Randlicht aus Nordost nur auf Naruto und Sasuke
SUN_ELEV, SUN_AZIM = 18.0, 250.0
SUN_KELVIN = 4300.0
SUN_STRENGTH = 5.0
RIM = dict(elev=22.0, azim=60.0, strength=2.5, kelvin=7800.0, angle=3.0)
ATMO_DENSITY = float(os.environ.get("NIDO_ATMO", "0.0"))

WALL_C, WALL_R = (0.0, 190.0), 190.0
STREET_HW = 12.0
TREE_POS = (4.0, 118.0)
RES_POS = (-20.0, 232.0)
CLIFF_Y = 360.0
HEAD_S = 10.0
HEAD_Z = 92.0
# Zickzack wie in den Referenzen: 1., 3., 5. oben, 2. und 4. tiefer (x, Höhenversatz, Haar)
HEADS = [(-110, 10.0, "hashirama"), (-55, -13.0, "tobirama"), (0, 14.0, "hiruzen"), (56, -13.0, "minato"),
         (112, 10.0, "tsunade")]

ROUTE = [
    (0, -60, 4.4), (0, -30, 4.8), (0, 0, 5.5), (0, 35, 7.6), (-2, 75, 10.0), (-7, 112, 11.0),
    (-3, 145, 14.0), (-4, 168, 23.0), (-10, 190, 33.5), (-16, 211, 39.3), (-19, 232, 39.4),
    (-19, 251, 39.6), (-17, 270, 49.0), (-14, 290, 62.0), (-11, 308, 76.0), (-9, 322, 86.0),
]


def terrain_height(X, Y):
    r = np.sqrt((X - WALL_C[0]) ** 2 + (Y - WALL_C[1]) ** 2)
    ring = np.clip((r - WALL_R - 8) / 70.0, 0, 1)
    ring = ring * ring * (3 - 2 * ring)
    hills = ring * (6 + 34 * (0.5 + fpv.fbm2(X / 220, Y / 220, 5, seed=3)))
    # Tal für die Zufahrtsstraße zum Tor
    valley = np.where(Y < 30, np.clip((np.abs(X) - 22) / 70.0, 0, 1), 1.0)
    hills = hills * valley ** 1.5
    north = np.clip((Y - (CLIFF_Y + 14)) / 16.0, 0, 1)
    plateau = north * (168 + 22 * fpv.fbm2(X / 160, Y / 160, 5, seed=4))
    # Bergspitze links hinter dem Felsen (Referenz)
    peak = 250 * np.exp(-((X + 190) ** 2 + (Y - CLIFF_Y - 260) ** 2) / (2 * 150.0 ** 2))
    return np.maximum(hills, plateau + north * peak)


def cliff_details(cliff_ob, mats, rng):
    """Zickzack-Treppen mit Geländer links der Gesichter, Wachhütten im Fels, Kuppelbauten auf der Kante
    (Referenzbilder). Positionen per Raycast auf die Felswand."""
    def hit(x, z):
        ok, loc, nrm, _ = cliff_ob.ray_cast(Vector((x, CLIFF_Y - 160, z)), Vector((0, 1, 0)))
        if not ok:
            return None, None
        n = Vector((nrm.x, nrm.y, 0))
        if n.length < 0.2 or n.y > 0:
            n = Vector((0, -1, 0))
        return loc, n.normalized()

    bm = bmesh.new()

    def add_box(center, ex, ey, size):
        res = bmesh.ops.create_cube(bm, size=1.0)
        M = Matrix((ex, ey, Vector((0, 0, 1)))).transposed().to_4x4()
        M.translation = center
        bmesh.ops.transform(bm, verts=res["verts"], matrix=M @ Matrix.Diagonal((*size, 1)))

    xa, xb, z = -238.0, -186.0, 3.0
    for k in range(6):
        x0, x1 = (xa, xb) if k % 2 == 0 else (xb, xa)
        z1 = z + 25.0
        n_s = int(abs(x1 - x0) / 0.5)
        rail = []
        for i in range(n_s):
            t = (i + 0.5) / n_s
            x, zz = x0 + (x1 - x0) * t, z + (z1 - z) * t
            p, n = hit(x, zz)
            if p is None:
                continue
            ex = Vector((math.copysign(1.0, x1 - x0), 0, 0))
            ex = (ex - n * ex.dot(n)).normalized()
            add_box(p + n * 0.8 + Vector((0, 0, -0.2)), ex, n, (0.56, 1.7, 0.4))
            if i % 4 == 0:
                add_box(p + n * 1.62 + Vector((0, 0, 0.5)), ex, n, (0.08, 0.08, 1.0))
            rail.append(tuple(p + n * 1.62 + Vector((0, 0, 1.0))))
        p, n = hit(x1, z1)
        if p is not None:
            add_box(p + n * 1.0 + Vector((0, 0, -0.2)), Vector((1, 0, 0)), n, (3.0, 2.1, 0.4))
        if len(rail) > 2:
            konoha.curve_tube(f"StairRail{k}", rail, 0.04, mats["rail"], res=0)
        z = z1
    # Wachhütten im Fels (unter den tieferen Köpfen und seitlich)
    for (x, zz) in ((-55, 34.0), (56, 32.0), (-160, 20.0), (165, 52.0)):
        p, n = hit(x, zz)
        if p is None:
            continue
        ex = Vector((-n.y, n.x, 0)).normalized()
        c = p + n * 1.9
        add_box(c + Vector((0, 0, -0.3)), ex, n, (7.0, 4.6, 0.5))
        rot = math.atan2(ex.y, ex.x)
        konoha.box(f"Hut{x}", c.x, c.y, zz, 5.0, 3.6, 3.6, mats["parapet"], rot=rot, bevel=0.05)
        konoha.hip_roof(f"HutRoof{x}", c.x, c.y, zz + 3.6, 4.2, 6.0, 1.6, mats["roofs"][0], rot=rot, overhang=0.5)
    me = bpy.data.meshes.new("CliffStairs")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("CliffStairs", me)
    me.materials.append(mats["stone"])
    fpv.link(ob)
    # Kuppelbauten auf der Felskante (weiße Rundbauten mit rotbraunen Kuppeln)
    for (x, dy, r) in ((-84, 44, 11.0), (-22, 62, 9.0), (34, 46, 10.0), (98, 70, 12.0)):
        y = CLIFF_Y + dy
        zg = float(terrain_height(np.array([x]), np.array([y]))[0]) - 1.0
        konoha.cylinder(f"Dome{x}", x, y, zg, r, 8.5, mats["horn"], seg=48)
        konoha.cylinder(f"DomeWin{x}", x, y, zg + 4.2, r + 0.05, 1.4, mats["glass"], seg=48, cap=False)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=r * 1.05, location=(x, y, zg + 8.5))
        d = bpy.context.object
        d.scale = (1, 1, 0.5)
        d.data.materials.append(mats["roofs"][0])
        for pp in d.data.polygons:
            pp.use_smooth = True


# ------------------------------------------------------------------------------------------------------------------
# Schritt 3.1 – Kamera & Timing: Hook am Tor, Verfolgung auf der Straße, Sprung über den alten Baum,
# Anflug auf die Residenz, kurzer Kampf auf dem Dach, ruhiges Schlussbild mit den Hokage-Gesichtern.
# ------------------------------------------------------------------------------------------------------------------
LENS_MM = 35.0
DECK_Z = 31.0 + 0.65                     # Flachdach der Residenz (konoha.residence: Z_TOP + Belag)
C_ROOF = (RES_POS[0], RES_POS[1] - 1.0)  # Dachmitte (Spitze bei (0, 0) relativ)
BEAM2_TOP = 17.2                         # Oberkante des unteren Tor-Querbalkens (frei sichtbar, unter dem Tordach)
P_CLASH = (4.75, 6.5, 1.0)               # Zusammenprall relativ zur Dachmitte (östlich der Mittelspitze)
T_CLASH = 15.0


def roof(x, y, z=0.0):
    """Dachkoordinaten (relativ zur Dachmitte, z über dem Dachbelag) -> Welt."""
    return Vector((C_ROOF[0] + x, C_ROOF[1] + y, DECK_Z + z))


# Kamera: Wegpunkte mit Uhrzeit (Speed-Ramp ergibt sich aus Abstand/Zeit, monoton-kubisch geglättet)
CAM_KEYS = [
    (0.00, (0.0, -32.0, 9.0)),       # Waldweg, Blick durchs Tor aufs Dorf und die Gesichter (Hook)
    (1.80, (0.0, 0.0, 7.0)),         # Durchflug unter dem unteren Querbalken (16 m)
    (2.70, (0.5, 20.0, 5.0)),
    (3.20, (1.5, 31.0, 3.6)),        # unter Laternenkabel 1 (y 33)
    (4.00, (-1.5, 50.0, 4.2)),       # unter der Wäscheleine
    (4.80, (-2.0, 69.0, 3.6)),       # Kabel 2 (y 71)
    (5.70, (-1.5, 90.0, 4.0)),
    (6.40, (-4.5, 104.0, 3.6)),      # Kabel 3 (y 109)
    (7.10, (-7.5, 116.0, 4.5)),      # links am alten Baum vorbei
    (7.80, (-6.5, 128.0, 5.5)),
    (8.40, (-5.0, 142.0, 7.5)),
    (9.00, (-5.5, 157.0, 10.5)),     # über Kabel 4 (y 147) hinweg
    (9.55, (-10.0, 174.0, 16.5)),    # schneller Steigflug über den Platz
    (10.10, (-15.5, 192.0, 24.5)),
    (10.55, (-19.5, 205.5, 31.0)),
    (10.85, tuple(roof(0.0, -18.8, 3.8))),   # über die Brüstung: der Kampf ist schon im Gange
    (11.30, tuple(roof(-2.5, -12.5, 3.0))),  # scharf abgebremst, langsamer Bogen um die beiden
    (12.20, tuple(roof(0.5, -12.0, 2.7))),
    (13.20, tuple(roof(3.4, -11.0, 2.6))),
    (14.45, tuple(roof(4.75, -10.4, 2.4))),  # Aufladen: beide (6,4 m Abstand) ganz im Bild
    (15.00, tuple(roof(4.75, -2.0, 2.0))),   # Vorstoß mit dem Ansturm, Zusammenprall 8,5 m voraus
    (15.90, tuple(roof(6.4, -3.0, 2.8))),    # von der Druckwelle zurück- und hochgedrückt
    (17.10, tuple(roof(6.3, -4.6, 0.35))),   # ganz tief auf dem Dach
    (20.05, tuple(roof(6.7, -0.6, 0.75))),   # Schlussbild: Naruto in Untersicht (8° nach oben), Gesichter darüber
]


_pchip = choreo.pchip


def camera_positions(frames):
    """Kameraorte pro Frame 0..frames+1 entlang CAM_KEYS mit Speed-Ramp. Rückgabe: t, pos, Tempo (m/s)."""
    return choreo.keyed_path(CAM_KEYS, FPS, frames)


def fight_plan(cam_t, cam_pos):
    """Blocking der Kämpfer (nur auf dem Dach der Residenz): Liste (t, Pose, Ort, Blickziel, in der Luft) je Figur,
    in Welt-Metern. Bei 1,3 und 5,3 s prallen sie fern über dem Dach zusammen (Teaser-Blitze); wenn die Kamera um
    10,85 s über die Brüstung steigt, prallen sie gerade in der Luft zusammen."""
    N, S = [], []
    R = roof
    P = Vector(P_CLASH)
    Y = 1.8   # Linie für Schlagabtausch und Konter (1,8 m neben der Mittelspitze)
    for tt in (1.3, 5.3):   # ferne Teaser-Zusammenstöße (Blitz über dem Dach), danach zurück in Deckung
        N += [(tt - 0.45, "guard", R(7.0, 4.5), R(0, 5.5), False),
              (tt - 0.25, "jump", R(5.6, 4.8, 1.8), R(1.5, 5.2, 3.5), True),
              (tt, "kick_R", R(3.9, 5.0, 3.3), R(2.6, 5.0, 3.4), True),
              (tt + 0.2, "recoil", R(5.8, 4.6, 2.6), R(1, 5, 2), True),
              (tt + 0.45, "land", R(7.0, 4.5), R(0, 5.5), False),
              (tt + 0.9, "guard", R(7.0, 4.5), R(0, 5.5), False)]
        S += [(tt - 0.45, "guard", R(0.0, 5.5), R(7, 4.5), False),
              (tt - 0.25, "jump", R(1.4, 5.3, 1.8), R(5.5, 4.8, 3.5), True),
              (tt, "punch_L", R(2.6, 5.0, 3.3), R(3.9, 5.0, 3.4), True),
              (tt + 0.2, "recoil", R(0.8, 5.3, 2.6), R(7, 4, 2), True),
              (tt + 0.45, "land", R(0.0, 5.5), R(7, 4.5), False),
              (tt + 0.9, "guard", R(0.0, 5.5), R(7, 4.5), False)]
    N += [(10.4, "guard", R(7.0, 4.5), R(0, 5.5), False),
          (10.7, "jump", R(5.6, 4.8, 1.8), R(1.5, 5.2, 3.5), True),
          (10.95, "kick_R", R(3.9, 5.0, 3.3), R(2.6, 5.0, 3.4), True),       # Zusammenprall in der Luft
          (11.15, "recoil", R(5.8, 4.6, 2.6), R(1, 5, 2), True),
          (11.35, "land", R(7.0, 2.6), R(2, Y), False),
          (11.5, "guard", R(5.2, Y), R(3, Y), False),                         # blockt Sasukes Tritt
          (11.7, "run_a", R(5.0, Y), R(3, Y), False),
          (11.95, "punch_R", R(4.3, Y), R(3.0, Y, 1.0), False),               # Schlag, Sasuke blockt
          (12.1, "recoil", R(5.6, Y, 0.3), R(3, Y), True),
          (12.2, "jump", R(6.2, Y, 0.9), R(3, Y), True),
          (12.35, "kick_R", R(4.7, Y, 0.8), R(3.2, Y, 1.0), True),            # Sprungtritt, geblockt
          (12.5, "recoil", R(5.6, Y, 0.8), R(3, Y), True),
          (12.67, "recoil", R(7.4, Y - 1.0, 1.3), R(3, Y), True),             # Konter trifft
          (12.87, "recoil", R(15.5, -6.3, 1.1), R(3, Y), True),               # gegen das Horn im Südosten (12,9 s)
          (13.1, "land", R(15.1, -6.0), R(0, 6.5), False),
          (13.35, "jump", R(11.2, 1.5, 1.0), R(0, 6.5), True),
          (13.6, "crouch_charge_R", R(7.95, 6.5), R(1.55, 6.5), False),       # Rasengan
          (14.5, "crouch_charge_R", R(7.9, 6.5), R(1.55, 6.5), False),
          (14.65, "run_b", R(6.9, 6.5), R(1.55, 6.5), False),
          (14.82, "dash_thrust_R", R(6.0, 6.5, 0.6), R(4, 6.5, 1), True),
          (15.0, "dash_thrust_R", R(5.37, 6.5, 1.0), R(4, 6.5, 1), True),     # Zusammenprall
          (15.35, "recoil", R(8.0, 5.4, 2.0), R(0, 6.5, 1), True),
          (15.8, "land", R(8.9, 4.3), R(0, 6.5), False),
          (16.4, "stand", R(8.7, 3.9), R(6.3, -4.6, 1.0), False),             # steht auf, dreht sich zur Kamera
          (17.0, "hero", R(8.4, 3.6), R(6.6, -0.9, 1.2), False),               # Faust zur Kamera
          (20.1, "hero", R(8.4, 3.6), R(6.6, -0.9, 1.2), False)]
    S += [(10.4, "guard", R(0.0, 5.5), R(7, 4.5), False),
          (10.7, "jump", R(1.4, 5.3, 1.8), R(5.5, 4.8, 3.5), True),
          (10.95, "punch_L", R(2.6, 5.0, 3.3), R(3.9, 5.0, 3.4), True),
          (11.15, "recoil", R(0.8, 5.3, 2.6), R(7, 4, 2), True),
          (11.3, "run_a", R(1.8, 3.0), R(6, Y), False),
          (11.5, "kick_L", R(3.3, Y), R(5.5, Y), False),                      # Tritt, Naruto blockt
          (11.75, "guard", R(3.1, Y), R(5, Y), False),
          (11.95, "guard", R(3.2, Y), R(4.5, Y), False),                      # blockt Narutos Schlag
          (12.35, "guard", R(3.3, Y), R(6, Y), False),                        # blockt den Sprungtritt
          (12.65, "kick_L", R(3.6, Y), R(7, Y), False),                       # Drehtritt als Konter
          (12.95, "guard", R(3.0, Y + 0.4), R(15, -6), False),
          (13.45, "crouch_charge_L", R(1.55, 6.5), R(7.95, 6.5), False),     # Chidori
          (14.5, "crouch_charge_L", R(1.6, 6.5), R(7.95, 6.5), False),
          (14.65, "run_a", R(2.6, 6.5), R(7.95, 6.5), False),
          (14.82, "dash_thrust_L", R(3.5, 6.5, 0.6), R(6, 6.5, 1), True),
          (15.0, "dash_thrust_L", R(4.13, 6.5, 1.0), R(6, 6.5, 1), True),
          (15.35, "recoil", R(1.0, 7.2, 2.2), R(9, 6.5, 1), True),
          (15.8, "land", R(-5.5, 8.6), R(9, 5, 0), False),
          (16.15, "jump", R(-9.8, 12.0, 3.0), R(-20, 20, 0), True),           # Absprung über die Brüstung
          (16.7, "jump", R(-16.5, 17.5, -4.0), R(-25, 25, -10), True),
          (17.1, "jump", R(-20.0, 21.0, -14.0), R(-25, 25, -20), True)]
    return N, S, P


HITS = [(10.95, 3), (11.5, 3), (11.95, 3), (12.35, 3), (12.9, 3), (T_CLASH, 4)]   # (Zeit, Halte-Frames)
ATTACKS = choreo.ATTACKS
POSES.update({
    # Schlussbild: rechte Faust nach vorn zur Kamera gestreckt (Versprechen), links locker, leicht eingedreht
    "hero": {"spine": (4, 0, 7), "head": (4, 0, -5), "shoulder.R": (84, 4, 0), "elbow.R": (8, 0, 0),
             "wrist.R": (-10, 0, 0), "shoulder.L": (-6, 14, 0), "elbow.L": (18, 0, 0), "hip.R": (0, -5, 0),
             "hip.L": (0, 7, 0), "root": (0, 0, 2)},
})


_style = choreo.default_style


def hit_warp(t):
    """Zeitverzerrung für Treffer-Halte-Frames (Hit-Stop) an HITS."""
    return choreo.time_warp(t, HITS, FPS)


def bake_fighter(fig, keys, n, cam_pos=None, look_win=None, seed=0):
    return choreo.bake_fighter(fig, keys, n, FPS, HITS, _style, cam_pos=cam_pos, look_win=look_win, seed=seed)


def stage_fight(N_keys, S_keys, n=None, cam_pos=None):
    """Figuren anlegen und das Blocking als Animation backen (Schritt 3.4)."""
    nar = ninja.naruto((0, 0, 0), 0.0)
    sas = ninja.sasuke((0, 0, 0), 0.0)
    for fig in (nar, sas):          # Modell-Keyframes der Konstruktion entfernen
        for o in [fig.base] + list(fig.J.values()):
            o.animation_data_clear()
    n = n or FPS * SECONDS + 2
    bake_fighter(nar, N_keys, n, cam_pos=cam_pos, look_win=(17.0, 21.0), seed=1)
    bake_fighter(sas, S_keys, n, seed=2)
    return nar, sas


# ------------------------------------------------------------------------------------------------ Schritt 3.5 – FX
# Kamera: Stöße (Zeit, Stärke in Grad, Frames) und Halte-Frames nur auf dem Dach (Kamera dort langsam)
SHAKES = [(10.95, 1.0, 6), (11.5, 0.8, 6), (11.95, 0.8, 6), (12.35, 1.2, 7), (12.9, 1.6, 8),
          (T_CLASH, 3.2, 10)]
CAM_HOLDS = [(12.35, 2), (12.9, 2), (T_CLASH, 4)]
BLUE = (0.12, 0.42, 1.0)


def F(t):
    return int(round(t * FPS)) + 1


def warp(t, holds):
    return choreo.time_warp(t, holds, FPS)


def impact_small(name, loc, t, color=(1.0, 0.75, 0.4), light_w=4000.0, ring=2.5, sparks=90, seed=1):
    """Kleines Impact-Paket (Kunai/Schlag): Funken, Lichtspitze 0 -> hoch -> 0 über 8 Frames, kleine Druckwelle.
    Halte-Frames, Kamerastoß und Blitz-Frame kommen aus Figuren-Warp, Kamera und Post."""
    f0 = F(t)
    vfx.sparks_gn(name + "Sparks", loc, t, n=sparks, speed=(5, 12), life=(0.15, 0.45), color=color, seed=seed)
    lt = vfx.point_light(name + "Light", color, 0.0, 0.15)
    lt.location = loc
    vfx.key_energy(lt, [(f0 - 1, 0.0), (f0, light_w), (f0 + 2, light_w * 0.5), (f0 + 8, 0.0)])
    vfx.shockwave(name + "Wave", loc, f0 + 1, r_max=ring, dur=8, color=(0.9, 0.85, 0.75), thick=0.1, glow=1.5)


def break_fx(mats, deck_z):
    """Echte Bruchstücke: (1) das Horn im Südosten bricht beim Konter-Treffer (13,15 s) oberhalb 1,6 m in 9 Stücke,
    (2) beim Zusammenprall (15,0 s) werden Dachplanken um den Klimaxpunkt in 14 Stücke gesprengt. Voronoi-Bruch +
    Bullet-Rigid-Body gegen Dachbelag und Brüstung, gebacken als Keyframes."""
    rng = np.random.default_rng(31)
    dg = bpy.context.evaluated_depsgraph_get()
    horn = bpy.data.objects["Horn7"]
    me = bpy.data.meshes.new_from_object(horn.evaluated_get(dg))
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.transform(horn.matrix_world)
    z_break = deck_z + 1.6
    lower, upper = bm.copy(), bm.copy()
    bm.free()
    for part, keep_up in ((lower, False), (upper, True)):
        res = bmesh.ops.bisect_plane(part, geom=part.verts[:] + part.edges[:] + part.faces[:],
                                     plane_co=(0, 0, z_break), plane_no=(0, 0, -1 if keep_up else 1), clear_outer=True)
        cut = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
        if cut:
            bmesh.ops.holes_fill(part, edges=cut, sides=0)
    sme = bpy.data.meshes.new("HornStump")
    lower.to_mesh(sme)
    lower.free()
    sme.materials.append(mats["horn"])
    fpv.link(bpy.data.objects.new("HornStump", sme))
    ctr = [v.co.copy() for v in upper.verts]
    zs = np.array([c.z for c in ctr])
    seeds = []
    for zz in np.linspace(zs.min() + 0.3, zs.max() - 0.2, 9):
        near = [c for c in ctr if abs(c.z - zz) < 0.3]
        c = sum(near, Vector()) / max(len(near), 1)
        seeds.append(c + Vector(rng.uniform(-0.15, 0.15, 3)))
    horn_bits = vfx.voronoi_fracture("HornBit", upper, seeds, mats["horn"])
    upper.free()
    bpy.data.objects.remove(horn, do_unlink=True)
    # Dachplanken um den Klimaxpunkt (erst im Blitz sichtbar)
    Pc = roof(P_CLASH[0], P_CLASH[1], 0.0)
    sb = bmesh.new()
    bmesh.ops.create_cube(sb, size=1.0)
    bmesh.ops.scale(sb, vec=(2.6, 2.6, 0.12), verts=sb.verts)
    bmesh.ops.translate(sb, vec=(Pc.x, Pc.y, deck_z + 0.062), verts=sb.verts)     # auf dem Belag (bis zum Blitz unsichtbar)
    seeds = [Pc + Vector((rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), 0.062))
             for _ in range(14)]
    deck_bits = vfx.voronoi_fracture("DeckBit", sb, seeds, mats["roofdeck"])
    sb.free()
    f_blast = F(T_CLASH) + 4
    colliders = [bpy.data.objects[n] for n in ("ResRoofDeck", "ResParapet", "ResCap")]
    hit = roof(14.6, -6.0, 1.9)          # Naruto trifft das Horn von innen -> Stücke fliegen nach außen
    f_horn = F(12.9) + 3
    # Kraftfeld-Stärken kalibriert (Test: 40 000 -> 1,4 m, 150 000 -> 10 m Steighöhe der Dachplatten)
    vfx.rigid_sim(horn_bits, colliders, f_horn, F(17.5), [(hit, 40000.0, f_horn, f_horn + 1)], mass_density=500.0)
    vfx.rigid_sim(deck_bits, colliders, f_blast, F(19.0), [(Pc - Vector((0, 0, 0.5)), 34000.0, f_blast, f_blast + 1)],
                  mass_density=450.0)
    for ob in deck_bits:        # erst nach der Simulation (rigid_sim ersetzt die Animationsdaten)
        for f, h in ((1, True), (f_blast - 1, True), (f_blast, False)):
            ob.hide_render = h
            ob.keyframe_insert("hide_render", frame=f)
    import crew
    hi = max(max(fc.evaluate(f) for f in range(f_blast, f_blast + 30)) for ob in deck_bits
             for fc in crew._fcurves_of(ob) if fc.data_path == "location" and fc.array_index == 2)
    print(f"Trümmer: Horn {len(horn_bits)} Stücke, Dach {len(deck_bits)} Stücke, max. Höhe Dachstücke {hi - deck_z:.1f} m")
    # Brandfleck im Dach nach der Explosion
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, segments=40, radius=1.6)
    for v in bm.verts:
        v.co.x *= 1.0 + 0.25 * math.sin(3 * math.atan2(v.co.y, v.co.x))
    sm, snb, sout = fpv.new_material("Scorch")
    n = snb.noise(snb.coords("Object"), scale=2.0, detail=4)
    r = snb.vmath("LENGTH", snb.coords("Object"))
    col = snb.mix(snb.math("MULTIPLY", snb.out(n, "Fac"), 0.6), (0.02, 0.016, 0.012), (0.10, 0.07, 0.05))
    pr = fpv.principled(snb, Base_Color=col, Roughness=0.9)
    tr = snb.node("ShaderNodeBsdfTransparent")
    mx = snb.node("ShaderNodeMixShader")
    rm = snb.node("ShaderNodeMapRange", clamp=True)
    snb.link(snb.math("ADD", r, snb.math("MULTIPLY", snb.out(n, "Fac"), 0.6)), rm.inputs["Value"])
    rm.inputs["From Min"].default_value = 1.5
    rm.inputs["From Max"].default_value = 1.0
    snb.link(rm.outputs[0], mx.inputs[0])
    snb.link(tr.outputs[0], mx.inputs[1])
    snb.link(pr.outputs[0], mx.inputs[2])
    snb.link(mx.outputs[0], sout.inputs[0])
    scorch = fpv.mesh_from_bmesh(bm, "Scorch", sm)
    scorch.location = (Pc.x, Pc.y, deck_z + 0.006)
    for f, h in ((1, True), (f_blast - 1, True), (f_blast, False)):
        scorch.hide_render = h
        scorch.keyframe_insert("hide_render", frame=f)


def fight_fx(nar, sas):
    """Rasengan/Chidori, Impact-Pakete an allen Treffern, volles Paket beim Zusammenprall."""
    vfx.energy_ball("Rasengan", nar.J["wrist.R"], (0, 0.03, -0.21), BLUE, 0.34, F(13.55), F(14.2), F(15.05),
                    light_w=160.0)
    vfx.lightning("Chidori", sas.J["wrist.L"], (0, 0.0, -0.12), (0.35, 0.6, 1.0), F(13.4), F(15.05), radius=0.85,
                  n_bolts=10, seed=4, light_w=260.0)
    V = Vector
    # Teaser: zwei ferne Blitze auf dem Dach, während die Kamera durchs Dorf fliegt (der Kampf läuft schon)
    for k, t in enumerate((1.3, 5.3)):
        # sichtbarer Glühkern (Ø bis 4,5 m, Glühhülle 6,5 m: aus 150–250 m ca. 30–60 px) + Lichtspitze 30 kW
        vfx.burst(f"Teaser{k}", roof(3.0, 5.0, 3.5), F(t) - 2, (0.55, 0.75, 1.0), r_max=6.5, dur=12, light_w=30000.0,
                  ring=False, bolts=False, seed=30 + k, core_s=25.0, glow_s=6.0, glow_alpha=0.35,
                  core_color=(0.85, 0.92, 1.0))
        vfx.sparks_gn(f"TeaserSparks{k}", roof(3.0, 5.0, 3.5), t, n=120, speed=(6, 14), life=(0.2, 0.5),
                      color=(0.6, 0.8, 1.0), strength=80.0, seed=30 + k)
    impact_small("HitAir", roof(3.25, 5.0, 4.4), 10.95, color=(0.8, 0.88, 1.0), light_w=5000.0, ring=3.0, seed=2)
    impact_small("HitKick", roof(4.3, 1.8, 1.3), 11.5, light_w=1000.0, ring=2.2, seed=3)
    impact_small("HitPunch", roof(3.8, 1.8, 1.4), 11.95, light_w=1000.0, ring=2.2, seed=6)
    impact_small("HitStrike", roof(4.0, 1.8, 1.9), 12.35, color=(0.85, 0.9, 1.0), light_w=1600.0, ring=3.0, seed=4)
    impact_small("HitHorn", roof(15.4, -6.2, 1.9), 12.9, color=(1.0, 0.8, 0.5), light_w=8000.0, ring=3.5, sparks=140,
                 seed=5)
    vfx.dust_gn("HornDust", roof(15.8, -6.4, 0.05), 12.9 + 3 / FPS, n=260, r_max=3.0, rise=1.2, life=1.2,
                color=(0.62, 0.58, 0.52), size=(0.012, 0.04), seed=5)
    # --- Klimax: Rasengan gegen Chidori
    P = roof(*P_CLASH) + Vector((0, 0, 1.1))
    f0 = F(T_CLASH)
    hold = bpy.data.objects.new("ClashHold", None)
    fpv.link(hold)
    hold.location = P
    vfx.energy_ball("ClashSphere", hold, (0, 0, 0), (0.3, 0.55, 1.0), 0.45, f0 - 1, f0, f0 + 5, light_w=600.0)
    lt = vfx.point_light("ClashFlash", (0.75, 0.85, 1.0), 0.0, 0.4)
    lt.location = P
    vfx.key_energy(lt, [(f0 - 1, 0.0), (f0, 8000.0), (f0 + 1, 14000.0), (f0 + 3, 4000.0), (f0 + 7, 0.0)])
    vfx.shockwave("ClashWave", roof(P_CLASH[0], P_CLASH[1], 0.4), f0 + 4, r_max=15.0, dur=12, color=(0.55, 0.75, 1.0),
                  thick=0.2, glow=2.5)
    vfx.sparks_gn("ClashSparks", P, T_CLASH + 4 / FPS, n=260, speed=(8, 20), life=(0.25, 0.7),
                  color=(0.7, 0.85, 1.0), strength=60.0, seed=11)
    vfx.lightning("ClashArcs", hold, (0, 0, 0), (0.45, 0.7, 1.0), f0, f0 + 10, radius=1.9, n_bolts=10, seed=9,
                  light_w=0.0)
    vfx.dust_gn("ClashDust", roof(P_CLASH[0], P_CLASH[1], 0.05), T_CLASH + 4 / FPS, n=1600, r_max=8.0, rise=1.8,
                life=1.6, color=(0.40, 0.33, 0.27), size=(0.015, 0.05), seed=12)


track = choreo.track


def camera_path(frames, N_keys, S_keys):
    """Positionen (Speed-Ramp) + Blickführung: vorwiegend auf das Kämpferpaar (Mitte, Brusthöhe), zwischendurch
    in Flugrichtung (Tor-Durchflug), am Ende auf Naruto mit den Gesichtern dahinter."""
    t, pos, v = camera_positions(frames)
    n = len(t)
    tw = warp(t, CAM_HOLDS)
    pos = np.stack([np.interp(tw, t, pos[:, k]) for k in range(3)], axis=1)
    a, b = track(N_keys, hit_warp(t)), track(S_keys, hit_warp(t))
    pair = (a + b) / 2 + np.array([0, 0, 1.0])
    nar = a + np.array([0, 0, 1.2])
    nar_end = a + np.array([0, 0, 1.3])      # Schlussbild: Blick 8° nach oben, Gesichter über Naruto
    tgt = pair.copy()
    # Konter: Naruto fliegt gegen das Horn -> Blick folgt ihm
    wN = _smooth(12.55, 12.75, t) * (1 - _smooth(13.05, 13.4, t))
    # nach dem Zusammenprall: zwischen Naruto und Klimaxpunkt, ab 16,6 s nur Naruto
    wE = _smooth(15.12, 15.5, t)
    tgt = tgt * (1 - wN[:, None]) + nar * wN[:, None]
    tgt = tgt * (1 - wE[:, None]) + nar * wE[:, None]
    wZ = _smooth(16.8, 17.8, t)
    tgt = tgt * (1 - wZ[:, None]) + nar_end * wZ[:, None]
    # Gewicht Blickziel (Dach der Residenz bzw. die Kämpfer) gegen Flugrichtung: Tor und Baum geradeaus,
    # Straße leicht zur Residenz und den Gesichtern, Anflug zunehmend aufs Dach, auf dem Dach ganz
    keys_w = [(0.0, 0.0), (1.8, 0.0), (2.6, 0.25), (6.0, 0.25), (6.5, 0.0), (7.8, 0.0), (8.6, 0.45), (10.3, 0.6),
              (10.85, 1.0), (21.0, 1.0)]
    w = _pchip([k for k, _ in keys_w], [x for _, x in keys_w], np.clip(t, 0, 21.0))
    d = tgt - pos
    look_yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    look_pit = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
    pos_f, quats, info = fpv.fpv_orient(pos, FPS, look_pitch=-2.0, pitch_follow=0.45, bank_gain=1.0, max_bank=30,
                                        micro=1.0, seed=9, look=(w, look_yaw, look_pit))
    # Kamerastöße: kurzes, stark verrauschtes, abklingendes Rütteln (Rollen/Nicken/Gieren + 3 cm Versatz)
    choreo.camera_shakes(pos_f, quats, t, SHAKES, FPS, seed=77)
    info.update(v=v, t=t, look_w=w, tgt=tgt, N=a, S=b)
    return pos_f, quats, info


_smooth = choreo.smooth


def sound_markers(info):
    """Zeitmarken fürs Sounddesign: Tor-Durchflug und Laternenkabel aus der tatsächlichen Bahn, Treffer aus HITS."""
    t, pos = info["t"], info["pos"]

    def when(cond):
        idx = np.where(cond)[0]
        return float(t[idx[0]]) if len(idx) else None
    m = [("AMBIENCE", 0.0), ("WHOOSH", when(pos[:, 1] > 0.0))]
    m += [("WHOOSH", when(pos[:, 1] > y)) for y in (33.0, 71.0, 109.0, 147.0)]
    m += [("WHOOSH", 10.85)] + [("IMPACT", h) for h, _ in HITS[:-1]] + [("BEAT_DROP", T_CLASH), ("AMBIENCE", 17.0)]
    return [(nm, tt) for nm, tt in m if tt is not None]


def build(args):
    sc = fpv.reset()
    frames = FPS * SECONDS
    fpv.setup_render(args.out, res=args.res, fps=FPS, seconds=SECONDS, samples=args.samples,
                     motion_blur=not args.no_mblur, mist_depth=6000.0)
    fpv.build_world(sun_elev=SUN_ELEV, sun_azim=SUN_AZIM, sky_strength=0.1, clouds=True, cloud_cover=0.4,
                    cloud_ref=9.0, aerosol=0.9, ozone=1.6, cloud_color=(1.0, 0.98, 0.95))
    key = fpv.add_sun(SUN_ELEV, SUN_AZIM, strength=SUN_STRENGTH, color=(1.0, 1.0, 1.0), angle_deg=0.5)
    key.data.use_temperature, key.data.temperature = True, SUN_KELVIN
    if ATMO_DENSITY > 0:
        fpv.add_atmosphere(ATMO_DENSITY, size=(700.0, 700.0, 160.0), center=(0.0, 180.0), color=(1.0, 0.97, 0.93))
    rng = np.random.default_rng(12)

    mats = {
        "plaster": konoha.plaster_material("Plaster"),
        "parapet": konoha.plaster_material("Parapet", window=False),
        "rooftop": fpv.simple_mat("RoofTop", (0.22, 0.2, 0.18), rough=0.9),
        # bunte Dächer wie in den Referenzen: rote/orange Ziegel, blaue, grüne, violette, türkise Bleche
        "roofs": [konoha.roof_material("RoofRed", (0.42, 0.08, 0.04)),
                  konoha.roof_material("RoofOrange", (0.55, 0.20, 0.06)),
                  konoha.roof_material("RoofGreen", (0.10, 0.28, 0.13)),
                  konoha.roof_material("RoofBlue", (0.07, 0.19, 0.46)),
                  konoha.roof_material("RoofPurple", (0.25, 0.11, 0.36)),
                  konoha.roof_material("RoofTeal", (0.05, 0.28, 0.29))],
        "awnings": [fpv.simple_mat("AwnRed", (0.45, 0.06, 0.04), rough=0.7),
                    fpv.simple_mat("AwnBlue", (0.08, 0.16, 0.40), rough=0.7),
                    fpv.simple_mat("AwnOrange", (0.70, 0.25, 0.04), rough=0.7),
                    fpv.simple_mat("AwnGreen", (0.10, 0.26, 0.14), rough=0.7)],
        "gate": konoha.roof_material("GateWood", (0.34, 0.10, 0.05), var=0.1),
        "gate_roof": konoha.roof_material("GateRoof", (0.12, 0.10, 0.09), var=0.1),
        "stone": nature.rock_material("GateStone", c1=(0.2, 0.19, 0.17), c2=(0.36, 0.34, 0.3), c3=(0.3, 0.28, 0.25),
                                      wet=False, bump=0.4, crack_w=0.3),
        "red": konoha.roof_material("ResRed", (0.58, 0.12, 0.045), var=0.08),
        "res_tile": konoha.terracotta_material("ResTile", (0.62, 0.30, 0.06)),
        "horn": sunny.paint_material("HornWhite", (0.80, 0.78, 0.72), rough=0.4),
        "cable": fpv.simple_mat("ResCable", (0.07, 0.05, 0.09), rough=0.5),
        "red_dark": konoha.roof_material("ResRedDark", (0.30, 0.05, 0.03), var=0.1),
        "red_roof": konoha.roof_material("ResRoof", (0.46, 0.07, 0.035), var=0.15),
        "glass": fpv.simple_mat("ResGlass", (0.02, 0.025, 0.03), rough=0.1),
        "tank": konoha.rust_metal_material("TankMetal"),
        "plaster_street": konoha.plaster_pbr_material("PlasterStreet"),
        "frame": fpv.simple_mat("WinFrame", (0.09, 0.055, 0.03), rough=0.55),
        "sill": fpv.simple_mat("Sill", (0.42, 0.40, 0.37), rough=0.8),
        "shutter": sunny.paint_material("Shutter", (0.10, 0.22, 0.15), rough=0.5),
        "beam": sunny.wood_material("BeamWood", dark=0.75, board=5.0),
        "roofdeck": sunny.wood_material("RoofDeckWood", dark=0.9, axis="X", board=0.2),
        "pipe": konoha.rust_metal_material("PipeMetal", tint=(0.6, 0.6, 0.6)),
        "ac": sunny.paint_material("ACUnit", (0.62, 0.62, 0.60), rough=0.45),
        "tiles": [konoha.terracotta_material("TileRed", (0.40, 0.07, 0.035)),
                  konoha.terracotta_material("TileOrange", (0.50, 0.17, 0.05)),
                  konoha.terracotta_material("TileBlue", (0.08, 0.16, 0.34)),
                  konoha.terracotta_material("TileGreen", (0.10, 0.24, 0.12))],
        "iron": fpv.simple_mat("Iron", (0.06, 0.055, 0.05), rough=0.5, metal=0.7),
        "wood": nature.bark_material("PoleWood", c=(0.16, 0.11, 0.07)),
        "wire": fpv.simple_mat("Wire", (0.01, 0.01, 0.01), rough=0.4),
        "rail": fpv.simple_mat("Rail", (0.12, 0.08, 0.05), rough=0.6),
        "lantern": fpv.simple_mat("Lantern", (0.42, 0.045, 0.025), rough=0.6, emission=(1.0, 0.3, 0.1), estrength=0.12),
        "cloth": [fpv.simple_mat("ClothR", (0.40, 0.05, 0.03), rough=0.8), fpv.simple_mat("ClothW", (0.62, 0.58, 0.50), rough=0.8),
                  fpv.simple_mat("ClothB", (0.06, 0.12, 0.26), rough=0.8)],
    }
    # Schritt 3.3: Ziegeldächer (Reihen, Farbstreuung, Moos, Patina, Regenstreifen), Fensterglas mit Innenraum,
    # Verwitterung (Schmutz von unten, Kantenabrieb, Laufspuren, Farbton pro Objekt), Dachbelag der Residenz
    roof_cols = [(0.42, 0.08, 0.04), (0.55, 0.20, 0.06), (0.10, 0.28, 0.13), (0.07, 0.19, 0.46), (0.25, 0.11, 0.36),
                 (0.05, 0.28, 0.29)]
    mats["roofs"] = [konoha.tile_roof_material(f"TileRoof{i}", c) for i, c in enumerate(roof_cols)]
    mats["tower_roofs"] = [konoha.tile_roof_material(f"TowerRoof{i}", c, cylindrical=True)
                           for i, c in enumerate(roof_cols)]
    mats["glass"] = konoha.window_glass_material("WindowGlass", per_cell=0.9)
    for key, kw in (("plaster", dict(edge=0.35, streak=0.25)), ("parapet", dict(edge=0.3, streak=0.3, dirt=0.0)),
                    ("plaster_street", dict(edge=0.4, streak=0.25, dirt=0.5)),
                    ("red", dict(dirt_h=2.5, dirt=0.35, streak=0.35)), ("red_dark", dict(dirt=0.0, streak=0.3)),
                    ("gate", dict(dirt=0.4, edge=0.3, streak=0.3)), ("stone", dict(dirt=0.3, var=0.03)),
                    ("horn", dict(dirt=0.0, streak=0.35, var=0.0)), ("shutter", dict(dirt=0.0, edge=0.4))):
        konoha.weather(mats[key], **kw)
    konoha.deck_wear(mats["roofdeck"], C_ROOF)
    sign_mats = []
    for ch in ("一楽", "忍具", "団子", "花", "書"):
        sp = os.path.join(textures.OUT, f"sign_{abs(hash(ch)) % 10000}.png")
        textures.kanji_disc(ch[0], sp, fg=(236, 232, 220), bg=(90, 30, 18) if len(sign_mats) % 2 else (25, 40, 70))
        sign_mats.append(konoha.image_material(f"Sign{len(sign_mats)}", sp, rough=0.6))
    mats["signs"] = sign_mats
    hi_path = os.path.join(textures.OUT, "hi.png")
    textures.kanji_disc("火", hi_path, fg=(175, 20, 14), bg=(236, 230, 214), ring=(175, 20, 14))
    mats["hi"] = konoha.image_material("HiMat", hi_path, rough=0.5)
    for k in ("a", "n"):
        p = os.path.join(textures.OUT, f"door_{k}.png")
        textures.kanji_disc("あ" if k == "a" else "ん", p, fg=(170, 25, 18), bg=(236, 228, 210), ring=(150, 22, 16))
        mats["door_" + k] = konoha.door_material("Door_" + k, p)

    tanks = konoha.water_tank_collection(mats["tank"], mats["iron"], mats["rooftop"])
    wins = konoha.window_collection(mats)
    street_mats = dict(mats, plaster=mats["plaster_street"])

    def detailed_building(cx, cy, bw, bd, floors, style, front, idx):
        objs, top = konoha.building(rng, street_mats, tanks, cx, cy, bw, bd, floors, 0.0, idx, style=style, front=front)
        if style == "round":
            return objs, top
        h = objs[0]["h"]
        konoha.facade_details(rng, mats, wins, cx, cy, bw, bd, h, floors, 0.0, idx, front=front)
        if style == "barrel":
            rise = min(bw, bd) * 0.28
            tm = mats["tiles"][int(rng.integers(len(mats["tiles"])))]
            if bw > bd:
                konoha.tile_roof_mesh(f"Tiles{idx}", cx, cy, h, bd, bw, rise, math.pi / 2, tm)
            else:
                konoha.tile_roof_mesh(f"Tiles{idx}", cx, cy, h, bw, bd, rise, 0.0, tm)
        return objs, top

    # Boden: Dorf + Hügel
    gmat = konoha.ground_material()
    cobble = konoha.street_material()
    grass = nature.grass_material("KGrass", c1=(0.035, 0.06, 0.015), c2=(0.10, 0.12, 0.04), dry=(0.22, 0.19, 0.10),
                                  scale=2.0)
    road = fpv.grid_mesh("Road", 12, 300, 2, 2, None, konoha.dirt_road_material(), origin=(0, -148))
    road.location.z = 0.04
    fpv.grid_mesh("VillageGround", 420, 420, 2, 2, None, gmat, origin=(0, 190))
    st = fpv.grid_mesh("MainStreet", 2 * STREET_HW + 2, 178, 2, 2, None, konoha.street_ground_material(half_w=STREET_HW),
                       origin=(0, 88))
    st.location.z = 0.02
    pz = fpv.grid_mesh("Plaza", 140, 100, 2, 2, None, cobble, origin=(0, 226))
    pz.location.z = 0.02
    terr = fpv.grid_mesh("Terrain", 2400, 2400, 360, 360, lambda X, Y: terrain_height(X, Y) - 0.05, grass,
                         origin=(0, 400))

    # Stadtmauer (Bogen, offen zum Felsen)
    wall_mat = nature.rock_material("WallStone", c1=(0.16, 0.15, 0.13), c2=(0.32, 0.30, 0.26), c3=(0.26, 0.24, 0.2),
                                    wet=False, bump=0.5, crack_w=0.4)
    for a in np.arange(-90 - 176, -90 + 180, 3.2):
        if abs(a + 90) < 6.0:  # Toröffnung
            continue
        ang = math.radians(a)
        x = WALL_C[0] + math.cos(ang) * WALL_R
        y = WALL_C[1] + math.sin(ang) * WALL_R
        if y > CLIFF_Y - 20:
            continue
        seg = 2 * math.pi * WALL_R * 3.2 / 360 + 0.4
        konoha.box("Wall", x, y, -1, seg, 3.4, 12.5, wall_mat, rot=ang + math.pi / 2, bevel=0.1)
        konoha.box("WallCap", x, y, 11.5, seg, 4.4, 0.8, mats["gate_roof"], rot=ang + math.pi / 2, bevel=0.05)

    konoha.gate(mats, y=0.0, half_open=10.5, height=19.0)

    # Gebäude
    idx = 0
    foot = []
    tops = []
    # a) entlang der Hauptstraße
    street_roofs = []
    for side in (-1, 1):
        y = 22.0
        while y < 172:
            w = rng.uniform(10, 14)
            d = rng.uniform(11, 15)
            floors = int(rng.integers(2, 5))
            cx = side * (STREET_HW + 1.5 + d / 2)
            style = rng.choice(["flat", "barrel", "barrel", "round", "flat"])
            foot.append((cx, y + w / 2, max(w, d) / 2 + 2.5))
            objs, top = detailed_building(cx, y + w / 2, d, w, floors, style, (-side, 0.0), idx)
            street_roofs.append((side, y + w / 2, objs[0]["h"], style))
            # Banner an der Straßenfassade
            if rng.random() < 0.55:
                ch = rng.choice(["火", "木", "忍", "茶", "楽", "薬", "酒"])
                bp = os.path.join(textures.OUT, f"banner_{ord(ch)}.png")
                if not os.path.exists(bp):
                    textures.kanji_disc(ch, bp, fg=(240, 236, 226), bg=(150, 24, 18))
                bm_ = konoha.image_material(f"Banner{idx}", bp, rough=0.8)
                bpy.ops.mesh.primitive_plane_add(size=1)
                ban = bpy.context.object
                ban.scale = (1.4, 1.0, 3.2)
                ban.location = (side * (STREET_HW + 1.1), y + w * 0.3, 3.4 + 2.0)
                ban.rotation_euler = (math.pi / 2, 0, math.pi / 2 * (-side))
                ban.data.materials.append(bm_)
            idx += 1
            y += w + rng.uniform(1.0, 3.5)
    # b) übriges Dorf (Raster mit Zufall)
    for gx in np.arange(-176, 177, 17.0):
        for gy in np.arange(10, 310, 17.0):
            x = gx + rng.uniform(-2.5, 2.5)
            y = gy + rng.uniform(-2.5, 2.5)
            if abs(x) < STREET_HW + 17 and y < 176:
                continue
            if y < 26 and abs(x) < 34:
                continue
            if 176 < y < 272 and abs(x) < 70:  # Platz + Residenz
                continue
            if math.hypot(x - WALL_C[0], y - WALL_C[1]) > WALL_R - 12:
                continue
            if y > CLIFF_Y - 30:
                continue
            if rng.random() < 0.12:
                continue
            w, d = rng.uniform(8, 13), rng.uniform(8, 13)
            floors = int(rng.integers(1, 5))
            foot.append((x, y, max(w, d) / 2 + 2.5))
            if rng.random() < 0.06:   # runde Stufentürme (Referenz)
                konoha.tiered_tower(f"Tower{idx}", x, y, min(w, d) / 2, int(rng.integers(2, 4)), mats,
                                    mats["tower_roofs"][int(rng.choice([3, 5, 2]))], rng)
            else:
                konoha.building(rng, mats, tanks, x, y, w, d, floors, math.radians(rng.uniform(-6, 6)), idx)
            idx += 1
    # c) Platz-Randbebauung (größere Gebäude)
    for (x, y, w, d, fl, st) in ((52, 205, 22, 16, 5, "barrel"), (50, 246, 18, 18, 4, "round"),
                                 (-62, 198, 20, 14, 4, "hip"), (-64, 258, 16, 16, 5, "flat"),
                                 (30, 272, 14, 12, 3, "flat")):
        foot.append((x, y, max(w, d) / 2 + 3))
        detailed_building(x, y, w, d, fl, st, None, idx)
        idx += 1
    print("buildings", idx)

    _, res_top = konoha.residence(mats, RES_POS[0], RES_POS[1], r=21.0)
    break_fx(mats, DECK_Z)
    for (x, y, rr, tiers, rm) in ((-44, 150, 5.5, 3, 3), (40, 128, 5.0, 2, 5)):   # Stufentürme an der Straße
        konoha.tiered_tower(f"StreetTower{x}", x, y, rr, tiers, mats, mats["tower_roofs"][rm], rng)
        foot.append((x, y, rr + 3))

    # Strommasten + Leitungen entlang der Straße
    poles = []
    for side in (-1, 1):
        prev = None
        for y in np.arange(14, 176, 19.0):
            x = side * (STREET_HW - 0.8)
            konoha.cylinder("Pole", x, y, 0, 0.16, 9.5, mats["wood"], seg=8)
            konoha.box("PoleBar", x, y, 8.6, 1.8, 0.14, 0.14, mats["wood"], rot=0.0, bevel=0)
            if prev is not None:
                for dx in (-0.7, 0.0, 0.7):
                    a = Vector((prev[0] + dx, prev[1], 8.75))
                    b = Vector((x + dx, y, 8.75))
                    pts = [tuple(a.lerp(b, t) - Vector((0, 0, 0.9 * 4 * t * (1 - t)))) for t in np.linspace(0, 1, 12)]
                    konoha.curve_tube("Wire", pts, 0.018, mats["wire"], res=0)
            prev = (x, y)

    # Laternen an sichtbaren Seilen quer über die Straße (pendeln), Wäscheleinen, Marktstände mit Waren,
    # Passanten, aufgeschreckte Vögel, treibende Blätter (Schritt 3.3)
    konoha_life.lantern_lines([33.0, 71.0, 109.0, 147.0], x_end=STREET_HW + 1.4, z_end=7.2, sag=1.1)

    def facade_h(side, y):
        hs = [h for (sd, yc, h, _) in street_roofs if sd == side and abs(yc - y) < 6.0]
        return min(hs) if hs else 0.0
    wl = []
    for y in (52.0, 90.0, 128.0, 160.0):
        z = min(facade_h(-1, y), facade_h(1, y)) - 1.2
        if z > 7.3 and abs(y - TREE_POS[1]) > 14:
            wl.append((y, min(z, 10.5)))
    for y, z in wl:
        konoha_life.laundry_lines([y], x_end=STREET_HW + 1.4, z_end=z, sag=0.7, seed=int(y))
    print("laundry lines", wl)
    for k in range(14):
        side = -1 if k % 2 else 1
        y = 30 + k * 10.5 + rng.uniform(-2, 2)
        x = side * (STREET_HW - 2.6)
        cm = mats["cloth"][rng.integers(3)]
        for dx in (-1.1, 1.1):
            for dy in (-1.4, 1.4):
                konoha.cylinder("StallPost", x + dx, y + dy, 0, 0.06, 2.4, mats["rail"], seg=6)
        konoha.box("StallRoof", x, y, 2.4, 2.8, 3.4, 0.12, cm, rot=math.radians(rng.uniform(-4, 4)), bevel=0)
        konoha.box("StallTable", x, y, 0.8, 2.2, 2.8, 0.12, mats["rail"], bevel=0)
        konoha_life.stall_goods(f"Goods{k}", x, y, rng)
        for q in range(2):
            konoha.box("Crate", x + side * 1.6, y + rng.uniform(-1, 1), 0.0, 0.5, 0.4, 0.35,
                       mats["rail"], rot=rng.uniform(0, 3), bevel=0.02)


    vtemps = konoha_life.villager_templates()
    spots = []
    y = 24.0
    while y < 172.0:
        for side in (-1, 1):
            if rng.random() < 0.55:
                x = side * (rng.uniform(5.0, 7.6) if rng.random() < 0.6 else rng.uniform(11.0, 12.2))
                if abs(y - TREE_POS[1]) < 7 and -2 < x < 10:
                    continue
                yaw = (90.0 if side > 0 else -90.0) + rng.uniform(-45, 45)
                spots.append((x, y + rng.uniform(-1.5, 1.5), yaw))
        y += rng.uniform(3.5, 6.0)
    konoha_life.place_villagers(vtemps, spots, seed=11)
    print("villagers", len(spots))
    konoha_life.birds(14, ((-20, -14), (64, 80), (9, 13)), (30, 14, 9), 3.3, 5.6, seed=21)
    konoha_life.birds(12, ((14, 20), (70, 84), (9, 13)), (-28, 18, 10), 3.45, 5.7, seed=22)
    konoha_life.leaves_gn("StreetLeaves", ((-12, 12), (18, 176), (0.3, 12)), count=700, wind=(1.0, 2.2, -0.3), seed=4)
    konoha_life.leaves_gn("TreeLeaves", ((-6, 14), (104, 134), (0.5, 13)), count=260, wind=(0.8, 1.6, -0.5), seed=6)

    # Hokage-Felsen
    # ockerfarbener Sandstein mit dunklen Laufspuren (Referenzen)
    rock = nature.rock_material("CliffRock", c1=(0.16, 0.10, 0.05), c2=(0.60, 0.42, 0.22), c3=(0.44, 0.28, 0.13),
                                wet=False, moss=(0.05, 0.10, 0.02), moss_amount=0.3, scale=2.0, bump=1.0,
                                crack_w=0.3, strata_scale=1.2, lichen=0.12)
    face_rock = nature.rock_material("FaceRock", c1=(0.22, 0.15, 0.08), c2=(0.66, 0.48, 0.27), c3=(0.50, 0.35, 0.18),
                                     wet=False, scale=1.5, bump=0.5, crack_w=0.12, lichen=0.1,
                                     moss=(0.06, 0.10, 0.03), moss_amount=0.15, cavity=0.75)
    hx_ = np.array([h[0] for h in HEADS], float)
    chin_ = np.array([HEAD_Z + h[1] - 1.2 * HEAD_S for h in HEADS])
    konoha.cliff_mesh("Cliff", -420, 420, CLIFF_Y, -2, 176, rock, seed=8, res=1.0,
                      panel=(-172, 172, 40, 168), chin=lambda X: np.interp(X, hx_, chin_))
    cliff_ob = bpy.data.objects["Cliff"]
    head_path = os.path.join(fpv.ASSETS, "LeePerrySmith.glb")
    tpl = konoha.import_head(head_path)
    tpl.hide_render = True
    tpl.hide_viewport = True
    for (hx, dz, hair) in HEADS:
        parts = konoha.hokage_head(tpl, f"Head_{hair}", (hx, CLIFF_Y + 2.0, HEAD_Z + dz), HEAD_S, face_rock, hair,
                                   face_rock, rng)
        konoha.fuse_parts(f"Hokage_{hair}", parts, voxel=0.22, mat=face_rock)
    cliff_details(cliff_ob, mats, rng)

    # Bäume
    leaf = nature.leaf_material("KLeaves", c1=(0.035, 0.10, 0.018), c2=(0.12, 0.22, 0.04))
    leaf2 = nature.leaf_material("KLeaves2", c1=(0.05, 0.12, 0.02), c2=(0.16, 0.24, 0.05))
    bark = nature.bark_material("KBark")
    variants = [nature.tree_variant(f"KTree{i}", leaf if i % 2 == 0 else leaf2, bark, height=h, crown_r=cr,
                                    n_clusters=nc, leaves_per=90, seed=60 + i, leaf_size=0.4)
                for i, (h, cr, nc) in enumerate(((14, 5.0, 26), (11, 4.2, 22), (17, 6.0, 30), (8, 3.2, 16)))]
    conleaf = nature.leaf_material("SugiLeaves", c1=(0.015, 0.035, 0.012), c2=(0.04, 0.07, 0.025), trans=0.2)
    variants += [nature.tree_variant(f"KSugi{i}", conleaf, bark, height=h, crown_r=cr, n_clusters=nc, leaves_per=110,
                                     seed=80 + i, shape="conical", leaf_size=0.3)
                 for i, (h, cr, nc) in enumerate(((22, 4.0, 34), (16, 3.2, 28)))]
    tree_coll, subs = nature.make_tree_collection("KTrees", variants)
    # großer alter Baum auf der Straße (Manöverpunkt)
    big = nature.tree_variant("BigTree", leaf, bark, height=21, crown_r=7.0, n_clusters=44, leaves_per=140, seed=99,
                              leaf_size=0.45)
    for o in big:
        fpv.link(o)
        o.location = (TREE_POS[0], TREE_POS[1], 0)
    konoha.cylinder("TreeRing", TREE_POS[0], TREE_POS[1], 0, 3.2, 0.6, mats["stone"], seg=32)

    pts, scl = [], []

    def add_forest(n, xr, yr):
        for _ in range(n):
            x = rng.uniform(*xr)
            y = rng.uniform(*yr)
            r = math.hypot(x - WALL_C[0], y - WALL_C[1])
            if r < WALL_R + 12:
                continue
            if abs(x) < 11 and y < 5:  # Straße zum Tor
                continue
            if CLIFF_Y - 25 < y < CLIFF_Y + 34 and abs(x) < 430:  # Felswand
                continue
            z = float(terrain_height(np.array([x]), np.array([y]))[0])
            pts.append((x, y, z - 0.3))
            scl.append(rng.uniform(0.75, 1.5))

    add_forest(1400, (-160, 160), (-260, 20))       # entlang der Zufahrt
    add_forest(6000, (-520, 520), (-340, CLIFF_Y))  # Ring um die Mauer
    add_forest(5000, (-700, 700), (CLIFF_Y, 950))   # Plateau über dem Felsen
    # Büsche entlang der Zufahrt
    for _ in range(500):
        x = rng.choice([-1, 1]) * rng.uniform(7.5, 40)
        y = rng.uniform(-240, -8)
        pts.append((x, y, float(terrain_height(np.array([x]), np.array([y]))[0]) - 0.2))
        scl.append(rng.uniform(0.18, 0.4))
    # Bewuchs auf Felsbändern (nach oben zeigende Flächen der Wand, nicht auf den Gesichtern)
    me = cliff_ob.data
    n = len(me.polygons)
    nrm = np.zeros(n * 3)
    ctr = np.zeros(n * 3)
    area = np.zeros(n)
    me.polygons.foreach_get("normal", nrm)
    me.polygons.foreach_get("center", ctr)
    me.polygons.foreach_get("area", area)
    nrm = nrm.reshape(-1, 3)
    ctr = ctr.reshape(-1, 3)
    ok = (nrm[:, 2] > 0.55) & (ctr[:, 2] > 8) & ~((np.abs(ctr[:, 0]) < 175) & (ctr[:, 2] > 42) & (ctr[:, 2] < 168))
    cand = np.where(ok)[0]
    if len(cand):
        w = area[cand] * nrm[cand, 2] ** 2
        w /= w.sum()
        for k in rng.choice(cand, size=min(900, len(cand)), p=w):
            pts.append(tuple(ctr[k] - np.array([0, 0, 0.3])))
            scl.append(rng.uniform(0.25, 0.7))
    # Bäume im Dorf
    fa = np.array(foot)
    for _ in range(700):
        x = rng.uniform(-170, 170)
        y = rng.uniform(20, 300)
        if math.hypot(x - WALL_C[0], y - WALL_C[1]) > WALL_R - 8 or abs(x) < STREET_HW + 2:
            continue
        if np.any(np.hypot(fa[:, 0] - x, fa[:, 1] - y) < fa[:, 2]):
            continue
        if math.hypot(x - RES_POS[0], y - RES_POS[1]) < 28:
            continue
        pts.append((x, y, 0))
        scl.append(rng.uniform(0.6, 1.1))
    nature.scatter_gn("KForest", tree_coll, len(subs), pts, scales=scl, seed=5)   # Geometry Nodes, Drehung/Größe zufällig
    print("trees", len(pts))

    # Kamera + Kämpfer (Schritt 3.1)
    cam_t, cam_p, _ = camera_positions(frames)
    N_keys, S_keys, _ = fight_plan(cam_t, cam_p)
    nar, sas = stage_fight(N_keys, S_keys, n=frames + 2, cam_pos=cam_p)
    fpv.rim_light([nar.base, sas.base], RIM["elev"], RIM["azim"], strength=RIM["strength"], kelvin=RIM["kelvin"],
                  angle=RIM["angle"])
    fight_fx(nar, sas)
    pos, quats, info = camera_path(frames, N_keys, S_keys)
    info["pos"] = pos
    cam = fpv.make_camera(pos, quats, fov_deg=60.0)
    cam.data.sensor_fit = "VERTICAL"
    cam.data.sensor_height = 36.0          # 9:16 hochkant: lange Seite = 36 mm (Vollformat-Äquivalent)
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
    print(f"Tempo {info['v'].min():.1f}–{info['v'].max():.1f} m/s")
    return sc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/nido_naruto")
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
