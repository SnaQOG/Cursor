"""Welt 2 – Naruto: Konohagakure mit Hokage-Felsen.

Flug (20 s, eine durchgehende Aufnahme, Speed-Ramp 1–33 m/s, fest 35 mm):
  0–2 s    Hook: Naruto und Sasuke prallen auf dem unteren Tor-Querbalken zusammen, Durchflug durchs Tor
  2–6 s    Hauptstraße tief (3,5–5 m) unter den Laternenkabeln, die beiden jagen sich ~15 m voraus
  6–8,6 s  langsam am alten Baum: Abstoß vom Stamm, Zusammenprall vor der Krone, links am Baum vorbei
  8,6–12 s schnell über den Platz, Steigflug über die Brüstung aufs Dach der Hokage-Residenz
  12–17 s  Dachkampf: Sprungtritt/Block, Konter gegen ein Horn, Rasengan gegen Chidori (Klimax 15,0 s)
  17–20 s  ruhiges Schlussbild: Naruto in Untersicht, dahinter die fünf Hokage-Gesichter
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

import fpv  # noqa: E402
import konoha  # noqa: E402
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
    (0.00, (0.0, -30.0, 14.0)),      # Waldweg, Blick hinauf zum unteren Tor-Querbalken (Hook)
    (1.40, (0.0, -10.5, 12.0)),
    (2.20, (0.0, 0.0, 8.5)),         # Durchflug unter dem unteren Querbalken (16 m)
    (3.20, (0.5, 22.0, 5.0)),
    (3.70, (1.5, 33.0, 3.6)),        # unter Laternenkabel 1
    (4.55, (-1.5, 52.0, 4.2)),
    (5.35, (-2.0, 71.0, 3.6)),       # Kabel 2
    (6.30, (-2.5, 92.0, 3.6)),       # Abbremsen vor dem alten Baum
    (7.10, (-4.5, 101.0, 3.4)),      # Kabel 3 (y 109), Blick hinauf: Sprung über die Krone
    (7.90, (-7.5, 112.0, 4.5)),
    (8.60, (-7.0, 123.0, 5.5)),      # links am Baum vorbei
    (9.30, (-5.0, 140.0, 7.5)),
    (9.90, (-5.0, 157.0, 10.5)),     # über Kabel 4 (y 147) hinweg
    (10.55, (-9.0, 177.0, 16.0)),
    (11.15, (-15.0, 194.0, 24.0)),
    (11.60, (-19.5, 205.5, 31.0)),
    (11.90, tuple(roof(0.0, -18.8, 3.8))),   # über die Brüstung zwischen zwei Hörnern
    (12.45, tuple(roof(2.0, -11.5, 2.8))),   # scharf abgebremst: Schlag und Konter ~20 m voraus
    (13.30, tuple(roof(3.4, -11.0, 2.6))),
    (14.45, tuple(roof(4.75, -10.4, 2.4))),  # Aufladen: beide (6,4 m Abstand) ganz im Bild
    (15.00, tuple(roof(4.75, -2.0, 2.0))),   # Vorstoß mit dem Ansturm, Zusammenprall 8,5 m voraus
    (15.90, tuple(roof(6.4, -3.0, 2.8))),    # von der Druckwelle zurück- und hochgedrückt
    (17.10, tuple(roof(6.3, -4.6, 0.35))),   # ganz tief auf dem Dach
    (20.05, tuple(roof(6.7, -0.6, 0.75))),   # Schlussbild: Naruto in Untersicht (8° nach oben), Gesichter darüber
]


def _pchip(tk, yk, t):
    """Monotone kubische Interpolation (Fritsch–Carlson): keine Überschwinger, stetige Geschwindigkeit."""
    tk, yk = np.asarray(tk, float), np.asarray(yk, float)
    h = np.diff(tk)
    d = np.diff(yk) / h
    m = np.zeros_like(yk)
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(yk) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    t = np.asarray(t, float)
    k = np.clip(np.searchsorted(tk, t) - 1, 0, len(tk) - 2)
    u = (t - tk[k]) / h[k]
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return h00 * yk[k] + h10 * h[k] * m[k] + h01 * yk[k + 1] + h11 * h[k] * m[k + 1]


def camera_positions(frames):
    """Kameraorte pro Frame 0..frames+1 entlang einer Catmull-Rom-Kurve durch CAM_KEYS, Bogenlänge über die Zeit
    monoton-kubisch (Speed-Ramp). Rückgabe: t, pos, Tempo (m/s)."""
    n = frames + 2
    t = (np.arange(n) - 1.0) / FPS
    pts = [p for _, p in CAM_KEYS]
    seg = 400
    curve = fpv._catmull_rom(pts, samples_per_seg=seg)
    sa = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    s_knot = [sa[min(k * seg, len(sa) - 1)] for k in range(len(pts))]
    s_knot[-1] = sa[-1]
    s = _pchip([k for k, _ in CAM_KEYS], s_knot, np.clip(t, 0, CAM_KEYS[-1][0]))
    s[t < 0] = s_knot[0] + t[t < 0] * (s[2] - s[1]) * FPS      # Frame 0 (vor dem Start) für Motion-Blur
    pos = np.stack([np.interp(s, sa, curve[:, k]) for k in range(3)], axis=1)
    if (t < 0).any():
        d0 = (curve[1] - curve[0]) / np.linalg.norm(curve[1] - curve[0])
        pos[t < 0] = curve[0] + np.outer(s[t < 0] - s_knot[0], d0)
    v = np.gradient(s) * FPS
    return t, pos, v


def fight_plan(cam_t, cam_pos):
    """Blocking der Kämpfer: Liste (t, Pose, Ort, Blickziel, in der Luft) je Figur, in Welt-Metern.
    Straße: Vorsprung vor der Kamera entlang ihrer Bahn, damit beide im Bild bleiben."""
    def ahead(t, lead, dx=0.0, z=0.0):
        i = int(np.clip(np.searchsorted(cam_t, t), 1, len(cam_t) - 1))
        p = cam_pos[i]
        # Ort `lead` Meter weiter auf der Kamerabahn (über die Bahnpunkte späterer Frames gesucht)
        d = np.linalg.norm(np.diff(cam_pos[i:], axis=0), axis=1).cumsum()
        k = int(np.searchsorted(d, lead))
        q = cam_pos[min(i + k + 1, len(cam_pos) - 1)]
        return Vector((q[0] + dx, q[1], z))

    N, S = [], []
    V = Vector
    # --- Hook: auf dem unteren Tor-Querbalken, Kunai-Zusammenprall 1,15 s, Absprung ins Dorf
    Z = BEAM2_TOP
    for L, sd in ((N, -1), (S, 1)):
        o = -sd
        L += [(0.0, "guard", V((sd * 3.4, -0.2, Z)), V((o * 3.0, -0.2, Z)), False),
              (0.8, "guard", V((sd * 3.0, -0.2, Z)), V((o * 3.0, -0.2, Z)), False),
              (1.02, "run_a" if sd < 0 else "run_b", V((sd * 1.5, -0.2, Z)), V((0, -0.2, Z)), False),
              (1.15, "punch_R" if sd < 0 else "punch_L", V((sd * 0.62, -0.2, Z + 0.15)), V((o, -0.2, Z)), True),
              (1.35, "recoil", V((sd * 1.7, 0.1, Z)), V((o, -0.2, Z)), False),
              (1.6, "jump", V((sd * 2.0, 1.5, Z + 0.8)), V((sd * 2.0, 30, 5)), True),
              (2.3, "jump", V((sd * 2.2, 16.0, 9.0)), V((sd * 2.0, 40, 0)), True)]
    # --- Straße: laufen ~15 m vor der Kamera, zwei Schlagabtausche in der Luft
    runs = np.arange(2.9, 6.35, 0.2)
    clashes = (4.1, 5.65)
    for L, sd in ((N, -1), (S, 1)):
        L.append((2.85, "land", ahead(2.85, 15.0, sd * 2.4), ahead(3.2, 30.0, sd * 2.4), False))
        for k, tr in enumerate(runs):
            if any(abs(tr - tc) < 0.45 for tc in clashes):
                continue
            L.append((tr, "run_a" if (k + (sd > 0)) % 2 == 0 else "run_b", ahead(tr, 15.0, sd * 2.3),
                      ahead(tr + 0.3, 30.0, sd * 2.3), False))
        for k, tc in enumerate(clashes):
            L += [(tc - 0.35, "jump", ahead(tc - 0.35, 15.0, sd * 2.0, 1.2), ahead(tc, 15.0, 0, 2.5), True),
                  (tc, ("kick_R", "punch_R")[k] if sd < 0 else ("punch_L", "kick_L")[k],
                   ahead(tc, 15.0, sd * 0.62, 2.6), ahead(tc, 15.0, -sd * 0.62, 2.6), True),
                  (tc + 0.3, "land", ahead(tc + 0.3, 15.0, sd * 2.4), ahead(tc + 0.6, 30.0, sd * 2.4), False)]
    # --- am alten Baum: Abstoß vom Stamm, Zusammenprall vor der Krone (6,85 s), rechts am Baum vorbei,
    #     Sprint über den Platz, Sprünge auf die Residenz
    tx, ty = TREE_POS
    for L, sd in ((N, -1), (S, 1)):
        o = -sd
        L += [(6.4, "jump", V((tx + sd * 3.5, 104.0, 1.5)), V((tx, 108, 7)), True),
              (6.62, "land", V((tx + sd * 1.2, ty - 1.3, 4.5)), V((tx, 108, 7)), True),        # Fuß am Stamm
              (6.85, ("kick_R" if sd < 0 else "punch_L"), V((tx + sd * 0.62, 108.5, 7.5)),
               V((tx + o * 0.62, 108.5, 7.5)), True),
              (7.05, "recoil", V((tx + sd * 1.8, 109.0, 7.0)), V((tx + o, 108.5, 7.5)), True),
              (7.35, "jump", V((tx + 9.0 + sd * 1.5, 116.0, 4.0)), V((10, 135, 0)), True),
              (7.6, "land", V((9.0 + sd * 1.5, 128.0, 0.0)), V((0, 170, 0)), False)]
        for k, tr in enumerate(np.arange(7.75, 9.75, 0.2)):
            y = 131.0 + (tr - 7.6) * 28.0
            L.append((tr, "run_a" if (k + (sd > 0)) % 2 == 0 else "run_b",
                      V((9.0 + sd * 1.5 - 17.0 * (tr - 7.6) / 2.15, y, 0.0)),
                      V((-20, 205, 0)), False))
        gx = RES_POS[0] + sd * 4.5
        L += [(9.95, "jump", V((gx, 196.0, 4.0)), V((gx, 207, 11)), True),
              (10.25, "land", V((gx, 206.5, 11.2)), V((gx, 214, 19)), False),           # Dach des Torbaus
              (10.5, "jump", V((gx, 208.5, 16.0)), V((gx, 212, 19)), True),
              (10.7, "land", V((gx + sd * 1.0, 210.0, 19.1)), V((gx, 230, 32)), False),  # Ziegelkragen
              (10.9, "jump", V((gx + sd * 2.0, 214.0, 28.0)), V((gx, 235, 32)), True)]
    # --- Dach (relativ zur Dachmitte): Naruto Ost, Sasuke West
    R = roof
    P = V(P_CLASH)
    Y = 1.8   # Linie für Schlag und Konter (1,8 m neben der Mittelspitze)
    N += [(11.3, "land", R(10, Y), R(-3.5, Y), False), (11.6, "guard", R(10, Y), R(-3.5, Y), False),
          (12.1, "run_a", R(7.5, Y), R(-3, Y), False), (12.35, "jump", R(6.0, Y, 0.9), R(3, Y), True),
          (12.6, "kick_R", R(4.7, Y, 0.8), R(3.2, Y, 1.0), True),            # Schlag: Sprungtritt
          (12.75, "recoil", R(5.6, Y, 0.8), R(3, Y), True),                   # geblockt
          (12.92, "recoil", R(7.4, Y - 1.0, 1.3), R(3, Y), True),             # Konter trifft
          (13.12, "recoil", R(15.5, -6.3, 1.1), R(3, Y), True),               # gegen das Horn im Südosten (13,15 s)
          (13.35, "land", R(15.1, -6.0), R(0, 6.5), False),
          (13.6, "jump", R(11.2, 1.5, 1.0), R(0, 6.5), True),
          (13.8, "crouch_charge_R", R(7.95, 6.5), R(1.55, 6.5), False),       # Rasengan
          (14.5, "crouch_charge_R", R(7.9, 6.5), R(1.55, 6.5), False),
          (14.65, "run_b", R(6.9, 6.5), R(1.55, 6.5), False),
          (14.82, "dash_thrust_R", R(6.0, 6.5, 0.6), R(4, 6.5, 1), True),
          (15.0, "dash_thrust_R", R(5.37, 6.5, 1.0), R(4, 6.5, 1), True),     # Zusammenprall
          (15.35, "recoil", R(8.0, 5.4, 2.0), R(0, 6.5, 1), True),
          (15.8, "land", R(8.9, 4.3), R(0, 6.5), False),
          (16.4, "stand", R(8.7, 3.9), R(6.3, -4.6, 1.0), False),             # steht auf, dreht sich zur Kamera
          (17.2, "stand", R(8.4, 3.6), R(6.6, -0.9, 1.2), False),
          (20.1, "stand", R(8.4, 3.6), R(6.6, -0.9, 1.2), False)]
    S += [(11.3, "land", R(-3.5, Y), R(10, Y), False), (11.6, "guard", R(-3.5, Y), R(10, Y), False),
          (12.4, "guard", R(3.0, Y), R(8, Y), False), (12.6, "guard", R(3.3, Y), R(6, Y), False),   # blockt
          (12.9, "kick_L", R(3.6, Y), R(7, Y), False),                         # Drehtritt als Konter
          (13.2, "guard", R(3.0, Y + 0.4), R(15, -6), False),
          (13.7, "crouch_charge_L", R(1.55, 6.5), R(7.95, 6.5), False),       # Chidori
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


def stage_fight(N_keys, S_keys):
    """Figuren anlegen und das Blocking keyframen (Schritt 3.4 ersetzt das durch gebackene Animation)."""
    nar = ninja.naruto((0, 0, 0), 0.0)
    sas = ninja.sasuke((0, 0, 0), 0.0)
    for fig, keys in ((nar, N_keys), (sas, S_keys)):
        for (t, pose, pos, look, air) in sorted(keys, key=lambda k: k[0]):
            d = look - pos
            yaw = math.degrees(math.atan2(-d.x, d.y))
            fig.pose(int(round(t * FPS)) + 1, POSES[pose], loc=(pos.x, pos.y, pos.z), yaw=yaw,
                     ground=None if air else pos.z)
    return nar, sas


def track(keys, t):
    """Ort einer Figur laut Blocking zu Zeiten t (linear zwischen den Schlüsseln), (n, 3)."""
    ks = sorted(keys, key=lambda k: k[0])
    tk = np.array([k[0] for k in ks])
    P = np.array([tuple(k[2]) for k in ks])
    return np.stack([np.interp(t, tk, P[:, j]) for j in range(3)], axis=1)


def camera_path(frames, N_keys, S_keys):
    """Positionen (Speed-Ramp) + Blickführung: vorwiegend auf das Kämpferpaar (Mitte, Brusthöhe), zwischendurch
    in Flugrichtung (Tor-Durchflug), am Ende auf Naruto mit den Gesichtern dahinter."""
    t, pos, v = camera_positions(frames)
    n = len(t)
    a, b = track(N_keys, t), track(S_keys, t)
    pair = (a + b) / 2 + np.array([0, 0, 1.0])
    nar = a + np.array([0, 0, 1.2])
    nar_end = a + np.array([0, 0, 1.3])      # Schlussbild: Blick 8° nach oben, Gesichter über Naruto
    tgt = pair.copy()
    # Konter: Naruto fliegt gegen das Horn -> Blick folgt ihm
    wN = _smooth(12.8, 13.0, t) * (1 - _smooth(13.3, 13.65, t))
    # nach dem Zusammenprall: zwischen Naruto und Klimaxpunkt, ab 16,6 s nur Naruto
    wE = _smooth(15.12, 15.5, t)
    tgt = tgt * (1 - wN[:, None]) + nar * wN[:, None]
    tgt = tgt * (1 - wE[:, None]) + nar * wE[:, None]
    wZ = _smooth(16.8, 17.8, t)
    tgt = tgt * (1 - wZ[:, None]) + nar_end * wZ[:, None]
    # Gewicht Blickziel gegen Flugrichtung: Hook 1, Tor-Durchflug 0, Straße 0,55, Baum 0,85, Anflug 0,6, Dach 1
    keys_w = [(0.0, 1.0), (1.55, 1.0), (1.95, 0.4), (2.55, 0.4), (3.0, 0.55), (6.1, 0.55), (6.4, 0.85),
              (7.4, 0.85), (7.8, 0.5), (11.2, 0.6), (11.9, 1.0), (21.0, 1.0)]
    w = _pchip([k for k, _ in keys_w], [x for _, x in keys_w], np.clip(t, 0, 21.0))
    d = tgt - pos
    look_yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    look_pit = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
    pos_f, quats, info = fpv.fpv_orient(pos, FPS, look_pitch=-2.0, pitch_follow=0.45, bank_gain=1.0, max_bank=30,
                                        micro=1.0, seed=9, look=(w, look_yaw, look_pit))
    info.update(v=v, t=t, look_w=w, tgt=tgt, N=a, S=b)
    return pos_f, quats, info


def _smooth(x0, x1, t):
    u = np.clip((np.asarray(t, float) - x0) / (x1 - x0), 0, 1)
    return u * u * (3 - 2 * u)


def sound_markers(info):
    return [("AMBIENCE", 0.0), ("IMPACT", 1.15), ("WHOOSH", 2.2), ("IMPACT", 4.1), ("WHOOSH", 5.35),
            ("IMPACT", 5.65), ("IMPACT", 6.85), ("WHOOSH", 11.9), ("IMPACT", 12.6), ("IMPACT", 13.15),
            ("BEAT_DROP", T_CLASH), ("AMBIENCE", 17.0)]


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
    st = fpv.grid_mesh("MainStreet", 2 * STREET_HW + 2, 178, 2, 2, None, konoha.sand_street_material(), origin=(0, 88))
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
                                    mats["roofs"][int(rng.choice([3, 5, 2]))], rng)
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
    for (x, y, rr, tiers, rm) in ((-44, 150, 5.5, 3, 3), (40, 128, 5.0, 2, 5)):   # Stufentürme an der Straße
        konoha.tiered_tower(f"StreetTower{x}", x, y, rr, tiers, mats, mats["roofs"][rm], rng)
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

    # Laternenketten quer über die Straße + Marktstände
    for y in np.arange(33, 170, 38.0):
        a = Vector((-(STREET_HW - 0.8), y, 6.9))
        b = Vector((STREET_HW - 0.8, y, 6.9))
        pts_w = [tuple(a.lerp(b, t) - Vector((0, 0, 1.3 * 4 * t * (1 - t)))) for t in np.linspace(0, 1, 16)]
        konoha.curve_tube("LWire", pts_w, 0.02, mats["wire"], res=0)
        for t in np.linspace(0.1, 0.9, 9):
            p = a.lerp(b, t) - Vector((0, 0, 1.3 * 4 * t * (1 - t) + 0.5))
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.28, location=p)
            lo = bpy.context.object
            lo.scale = (1, 1, 1.35)
            for dz in (0.36, -0.36):
                konoha.cylinder("LCap", p.x, p.y, p.z + dz - 0.05, 0.16, 0.1, mats["wire"], seg=10)
            lo.data.materials.append(mats["lantern"])
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
        for q in range(3):
            konoha.box("Crate", x + rng.uniform(-0.8, 0.8), y + rng.uniform(-1, 1), 0.92, 0.5, 0.4, 0.35,
                       mats["rail"], rot=rng.uniform(0, 3), bevel=0.02)

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
    _, subs = nature.make_tree_collection("KTrees", variants)
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
    nature.scatter_instances("KForest", subs, pts, scales=scl, seed=5)
    print("trees", len(pts))

    # Kamera + Kämpfer (Schritt 3.1)
    cam_t, cam_p, _ = camera_positions(frames)
    N_keys, S_keys, _ = fight_plan(cam_t, cam_p)
    nar, sas = stage_fight(N_keys, S_keys)
    fpv.rim_light([nar.base, sas.base], RIM["elev"], RIM["azim"], strength=RIM["strength"], kelvin=RIM["kelvin"],
                  angle=RIM["angle"])
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
