"""Welt 1 – One Piece: Grand Line, Felsnadeln im Meer, Thousand Sunny.

Flug (20 s, konstant 15 m/s, eine durchgehende Aufnahme):
  0–5 s    tief (3 m) über den Wellen, Durchflug zwischen zwei Felsnadeln
  5–12 s   leichter Schwenk rechts, dann Linkskurve mit Schräglage um die große Felsnadel
  ~11–12 s Reveal: die Thousand Sunny kommt hinter der Felsnadel hervor, Breitseite (Steuerbord)
  12–15,5 s Anflug quer auf die Steuerbordseite, Steigflug auf ~9,6 m: die ganze Strohhutbande an Deck
           (Ruffy auf dem Löwenkopf, Jinbei am Steuer, Brook mit Geige, Lysop und Chopper winken,
           Nami an den Mandarinen, Zorro schläft am Mast, Sanji, Robin liest, Franky auf dem Achterkastell)
  ~16 s    Überflug über das Rasendeck zwischen Fockmast und Achterkastell, ~4 m über der Crew
  16–20 s  über die Backbordseite hinaus weiter aufs offene Meer Richtung Inseln
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import bmesh  # noqa: E402,I100
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import crew  # noqa: E402
import fpv  # noqa: E402
import nature  # noqa: E402
import ocean  # noqa: E402
import sunny  # noqa: E402

FPS = 24
SECONDS = 20
SUN_ELEV, SUN_AZIM = 14.0, 195.0   # tiefe Abendsonne aus SSW: Deck frei von Kastell-Schatten, am Ende hinter dem Schiff
SUN_KELVIN = 4300.0
SUN_DISC_DEG = 0.4       # sichtbare Sonnenscheibe (Radius) für die Gegenlicht-Silhouette am Ende
RIM = dict(elev=24.0, azim=25.0, strength=2.2, kelvin=7800.0, angle=3.0)   # kühles Randlicht, nur Schiff + Crew
RIM_FADE = (15.5, 17.0)                                                      # Randlicht aus, bevor die Kamera zurückblickt
# Homogenes Dunst-Volumen: getestet (0.0012/m: +33 % Renderzeit, aber bei 16 Samples extrem verrauscht) -> aus;
# Tiefendunst kommt aus post.py (haze_sky). Mit GPU und 128+ Samples über NIDO_ATMO einschaltbar.
ATMO_DENSITY = float(os.environ.get("NIDO_ATMO", "0.0"))

# ---- Schiff: fährt nach Osten (+X); Steuerbord zeigt nach Süden, zur anfliegenden Kamera
SHIP_HEADING = 0.0
SHIP_SPEED = 3.0
SHIP_START = Vector((-19.0, 201.0, 0.0))   # schon im ersten Bild am Horizont, in der Lücke des Felsentors

# ---- Kamera: feste 28-mm-Brennweite (Vollformat-Äquivalent, lange Bildseite 36 mm), Speed-Ramp,
#      erster Teil weltfest durch die Felsen, zweiter Teil schiffsfest (langsamer Deck-Überflug, Rückblick)
LENS_MM = 28.0
SPEED_KEYS = [(0.0, 18.0), (2.0, 18.0), (2.8, 21.0), (8.8, 21.0), (10.8, 6.5), (11.4, 4.5), (14.8, 4.5),
              (15.8, 7.0), (17.0, 11.0), (18.6, 11.0), (20.2, 8.0)]
WORLD_ROUTE = [(0, -8, 3.0), (0, 35, 3.1), (-1, 78, 3.3), (18, 124, 3.9), (16, 150, 4.6), (14, 166, 5.6)]
SHIP_ROUTE = [(1.0, -22.0, 7.8), (-1.5, -11.0, 9.2), (-1.5, -7.5, 9.2), (-1.45, 0.0, 9.0), (-1.5, 7.5, 9.8),
              (-0.6, 12.5, 11.0), (1.5, 19.0, 13.2), (4.0, 26.0, 16.0), (6.5, 33.0, 18.8), (9.0, 40.0, 21.3),
              (11.5, 47.0, 23.4), (14.0, 54.0, 25.4), (16.5, 61.0, 27.2), (19.0, 68.0, 29.0)]
BLEND_M = 8.0          # Übergang weltfest -> schiffsfest über ±8 m Bahnlänge
LOOK = dict(mid=(-1.0, -0.5, 5.0), castle=(-6.2, -1.8, 10.8),
            ship=(0.0, 0.0, 13.0))


def ship_xform(t):
    """Weltlage des Schiffsrumpfs (ohne Stampfen/Rollen) zur Zeit t: (Ort, Rotationsmatrix 3x3)."""
    h = math.radians(SHIP_HEADING)
    d = Vector((math.cos(h), math.sin(h), 0))
    R = np.array([[math.cos(h), -math.sin(h), 0], [math.sin(h), math.cos(h), 0], [0, 0, 1]])
    return np.array(SHIP_START + d * SHIP_SPEED * t), R


def ship_course(pos=None):
    return SHIP_HEADING, SHIP_START


def _arc(points):
    c = fpv._catmull_rom(points)
    sa = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    return c, sa


def _eval(c, sa, s):
    """Punkt bei Bogenlänge s; außerhalb linear entlang der End-Tangenten verlängert."""
    s = float(s)
    if s < 0:
        d = (c[1] - c[0]) / np.linalg.norm(c[1] - c[0])
        return c[0] + d * s
    if s > sa[-1]:
        d = (c[-1] - c[-2]) / np.linalg.norm(c[-1] - c[-2])
        return c[-1] + d * (s - sa[-1])
    return np.array([np.interp(s, sa, c[:, k]) for k in range(3)])


def _dir(v):
    return math.atan2(v[1], v[0]), math.atan2(v[2], math.hypot(v[0], v[1]))


def _smooth(x0, x1, t):
    u = np.clip((t - x0) / (x1 - x0), 0, 1)
    return u * u * (3 - 2 * u)


def camera_path(frames):
    """Kamerafahrt mit Speed-Ramp. Bis ~9 s weltfeste Route durch die Felsen, danach eine Bahn im Schiffssystem
    (+X Bug, +Y Backbord): Anflug auf Steuerbord, Überflug quer übers Rasendeck mit 4,5 m/s relativ zum
    Schiff, Rückblick-Schwenk und Rückzug, die Sunny steht am Ende vor der tiefen Sonne."""
    n = frames + 2
    t, v, s = fpv.speed_profile(SPEED_KEYS, FPS, n)
    cw, sw = _arc(WORLD_ROUTE)
    Lw = sw[-1]
    t_join = float(np.interp(Lw, s, t))
    loc0, R0 = ship_xform(t_join)
    L0 = R0.T @ (np.array(WORLD_ROUTE[-1], dtype=float) - loc0)
    cl, sl = _arc([tuple(L0)] + SHIP_ROUTE)
    pos = np.zeros((n, 3))
    local = np.zeros((n, 3))
    for i in range(n):
        loc, R = ship_xform(t[i])
        pw = _eval(cw, sw, s[i])
        pl = loc + R @ _eval(cl, sl, s[i] - Lw)
        w = _smooth(Lw - BLEND_M, Lw + BLEND_M, s[i])
        pos[i] = pw * (1 - w) + pl * w
        local[i] = R.T @ (pos[i] - loc)
    # ---- Blickziel-Kette im Schiffssystem (weich überblendet): Schiffsmitte -> Crew auf dem Rasen (voraus) ->
    #      Achterkastell mit Franky -> ganzes Schiff. Stetiger Schwenk gegen den Uhrzeigersinn (das Fock-Segel
    #      würde den Blick zum Bug versperren).
    keys = [(9.0, "mid"), (11.0, "mid"), (11.6, "lawn"), (11.9, "lawn"), (13.5, "castle"), (14.7, "castle"),
            (16.9, "ship"), (99.0, "ship")]
    kt = [k[0] for k in keys]
    look_yaw, look_pit = np.zeros(n), np.zeros(n)

    def target(name, i):
        if name == "lawn":        # mitwandernd ~7 m voraus auf dem Rasen (nie senkrecht nach unten)
            return np.array([local[i, 0] + 1.0, local[i, 1] + 7.0, 4.6])
        return np.array(LOOK[name])
    for i in range(n):
        j = int(np.clip(np.searchsorted(kt, t[i]) - 1, 0, len(keys) - 2))
        u = float(_smooth(kt[j], kt[j + 1], t[i]))
        tl = target(keys[j][1], i) * (1 - u) + target(keys[j + 1][1], i) * u
        loc, R = ship_xform(t[i])
        look_yaw[i], look_pit[i] = _dir(loc + R @ tl - pos[i])
    look_yaw = np.unwrap(look_yaw)
    w = 0.55 * _smooth(9.0, 10.4, t) + 0.45 * _smooth(10.4, 11.0, t)
    pos_f, quats, info = fpv.fpv_orient(pos, FPS, look_pitch=-3.0, pitch_follow=0.5, bank_gain=1.0, max_bank=32,
                                        micro=1.0, seed=5, look=(w, look_yaw, look_pit))
    info.update(t_join=t_join, local=local, s=s, v=v, used=s[-1] - s[0], total=sl[-1] + Lw)
    return pos_f, quats, info


def sound_markers(info):
    """Zeitmarken für das Sounddesign aus der tatsächlichen Bahn."""
    t, local, pos = info["t"], info["local"], info["pos"]

    def when(cond):
        idx = np.where(cond)[0]
        return float(t[idx[0]]) if len(idx) else None
    gate = when(pos[:, 1] > 76.0)
    rock = when(pos[:, 1] > 128.0)
    rail_in = when((local[:, 1] > -7.5) & (t > 5))
    rail_out = when((local[:, 1] > 7.5) & (t > 5))
    m = [("AMBIENCE", 0.0), ("WHOOSH", gate), ("WHOOSH", rock), ("WHOOSH", rail_in), ("BEAT_DROP", rail_in),
         ("WHOOSH", rail_out), ("AMBIENCE", 17.0)]
    return [(name, tt) for name, tt in m if tt is not None]


ROCKS = [
    # (x, y, radius, height, seed, taper)
    (-19, 70, 9.0, 30, 11, 0.35),    # Tor links
    (15, 82, 7.0, 20, 12, 0.45),     # Tor rechts
    (-48, 158, 20.0, 62, 13, 0.25),  # große Felsnadel (seitlich, gibt den Blick aufs Schiff frei)
    (-38, 38, 4.0, 7, 14, 0.5),
    (27, 48, 3.0, 4, 15, 0.6),
    (30, 128, 5.5, 12, 16, 0.4),
    (-52, 118, 8.0, 26, 17, 0.35),
    (64, 178, 6.0, 15, 18, 0.4),
    (-5, 262, 3.5, 5, 19, 0.6),
    (-150, 205, 12.0, 38, 20, 0.3),
    (70, 110, 10.0, 34, 21, 0.3),
    (-107.7, 321.8, 7.0, 18, 22, 0.35),  # neben dem Kurs der Sunny
    (-150, 268, 5.0, 12, 23, 0.4),
    (-205, 345, 11.0, 30, 24, 0.3),
    (-96, 330, 4.0, 7, 25, 0.5),
]


def add_rim_light(ship_root):
    """Kühles Randlicht von schräg hinten (Norden), per Light Linking nur auf Schiff und Crew."""
    coll = bpy.data.collections.new("RimReceivers")
    bpy.context.scene.collection.children.link(coll)
    stack = list(ship_root.children)
    while stack:
        o = stack.pop()
        stack.extend(o.children)
        if o.type in ("MESH", "CURVE"):
            coll.objects.link(o)
    coll.hide_render = False
    rim = fpv.add_sun(RIM["elev"], RIM["azim"], strength=RIM["strength"], color=(1, 1, 1), angle_deg=RIM["angle"],
                      name="RimLight")
    rim.data.use_temperature, rim.data.temperature = True, RIM["kelvin"]
    rim.light_linking.receiver_collection = coll
    for f, e in ((1, RIM["strength"]), (int(RIM_FADE[0] * FPS) + 1, RIM["strength"]), (int(RIM_FADE[1] * FPS) + 1, 0.0)):
        rim.data.energy = e
        rim.data.keyframe_insert("energy", frame=f)
    return rim


def add_atmosphere(density, size=(1600.0, 1600.0, 300.0), center=(0.0, 250.0)):
    """Homogenes Streuvolumen als Box um die Szene (endlich, damit der Himmel nicht zugenebelt wird):
    Tiefendunst und Lichtschleier um die tiefe Sonne (Vorwärtsstreuung)."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bmesh.ops.translate(bm, vec=(center[0], center[1], size[2] / 2 + 0.3), verts=bm.verts)
    mat, nb, out = fpv.new_material("Atmosphere")
    vol = nb.node("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.92, 0.95, 1.0, 1)
    vol.inputs["Density"].default_value = density
    vol.inputs["Anisotropy"].default_value = 0.65
    nb.link(vol.outputs[0], out.inputs["Volume"])
    ob = fpv.mesh_from_bmesh(bm, "Atmosphere", mat)
    ob.visible_shadow = True
    return ob


def foam_builder(shore_img, shore_map, max_d=12.0):
    """Schaum aus GN-Attributen 'wake' + 'hull_foam' und Brandung aus dem Fels-Abstandsfeld."""

    def fn(nb, wp_t):
        wake = nb.out(nb.node("ShaderNodeAttribute", attribute_name="wake"), "Fac")
        hf = nb.out(nb.node("ShaderNodeAttribute", attribute_name="hull_foam"), "Fac")
        shore = shore_factor(nb, shore_img, shore_map, 0.0, 4.5, max_d)
        m = nb.math("MAXIMUM", nb.math("MAXIMUM", nb.math("MULTIPLY", wake, 0.95), hf), nb.math("MULTIPLY", shore, 0.9))
        n = nb.noise(wp_t, scale=0.55, detail=4, rough=0.65)
        n2 = nb.noise(wp_t, scale=2.6, detail=2, rough=0.6)
        nn = nb.math("ADD", nb.out(n, "Fac"), nb.math("MULTIPLY", nb.out(n2, "Fac"), 0.35))
        th = nb.math("SUBTRACT", 1.02, m)
        mr = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(nb.math("SUBTRACT", nn, th), mr.inputs["Value"])
        mr.inputs["From Min"].default_value = -0.15
        mr.inputs["From Max"].default_value = 0.12
        return nb.math("MULTIPLY", mr.outputs[0], nb.math("MINIMUM", nb.math("MULTIPLY", m, 1.6), 0.9))

    return fn


def shore_factor(nb, img, smap, d0, d1, max_d):
    """1 an der Fels-Wasserlinie, 0 ab d1 Meter Abstand."""
    x0, y0, w, h = smap
    pos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    x, y, _ = nb.sep(pos)
    uv = nb.comb(nb.math("DIVIDE", nb.math("SUBTRACT", x, x0), w), nb.math("DIVIDE", nb.math("SUBTRACT", y, y0), h), 0.0)
    tex = nb.n.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "EXTEND"
    tex.interpolation = "Linear"
    nb.link(uv, tex.inputs["Vector"])
    d = nb.math("MULTIPLY", nb.sep(nb.out(tex, "Color"))[0], max_d)
    mr = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(d, mr.inputs["Value"])
    mr.inputs["From Min"].default_value = d1
    mr.inputs["From Max"].default_value = d0
    return mr.outputs[0]


def shallow_tint(shore_img, shore_map, max_d=12.0, color=(0.02, 0.20, 0.19)):
    def fn(nb, col):
        f = shore_factor(nb, shore_img, shore_map, 0.0, 11.0, max_d)
        f = nb.math("MULTIPLY", nb.math("POWER", f, 1.5), 0.75)
        return nb.mix(f, col, color)
    return fn


def island(name, cx, cy, radius, height, seed, mat):
    def hfn(X, Y):
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / radius
        warp = fpv.fbm2(X / radius * 2 + 3, Y / radius * 2, octaves=3, seed=seed + 9) * 0.25
        rr = r + warp
        plateau = np.clip((1.0 - rr) / 0.35, 0, 1)
        plateau = plateau * plateau * (3 - 2 * plateau)
        n = fpv.fbm2(X / radius * 4, Y / radius * 4, octaves=7, seed=seed)
        ridge = 1 - np.abs(fpv.fbm2(X / radius * 6 + 7, Y / radius * 6, octaves=5, seed=seed + 3))
        peak = np.clip(1 - r * 1.6, 0, 1) ** 1.5
        h = height * (0.45 * plateau * (0.8 + 0.4 * n) + 0.55 * peak * (0.6 + 0.6 * ridge))
        return h - 4
    return fpv.grid_mesh(name, radius * 2.6, radius * 2.6, 260, 260, hfn, mat, origin=(cx, cy))


def build(args):
    sc = fpv.reset()
    frames = FPS * SECONDS
    fpv.setup_render(args.out, res=args.res, fps=FPS, seconds=SECONDS, samples=args.samples,
                     motion_blur=not args.no_mblur, mist_depth=6000.0)
    fpv.build_world(sun_elev=SUN_ELEV, sun_azim=SUN_AZIM, sky_strength=0.08, clouds=True, cloud_cover=0.42,
                    cloud_ref=9.0, aerosol=0.5, ozone=2.0,
                    extra_suns=[(SUN_ELEV, SUN_AZIM, SUN_DISC_DEG, 6000.0, (1.0, 0.86, 0.62))])
    fpv.add_sun(SUN_ELEV, SUN_AZIM, strength=5.2, color=(1.0, 1.0, 1.0))
    key = bpy.data.objects["Sun"].data
    key.use_temperature, key.temperature = True, SUN_KELVIN

    pos, quats, info = camera_path(frames)
    info["pos"] = pos
    heading, start = ship_course()
    print(f"Sunny: Kurs {heading:.1f} Grad, Start {tuple(round(v, 2) for v in start)}, "
          f"Übergang schiffsfeste Bahn bei {info['t_join']:.2f} s")

    # Schiff mit der Strohhutbande an Bord
    root, body, _ = sunny.build()
    sunny.animate(root, body, heading, tuple(start), SHIP_SPEED, FPS, frames)
    crew.place_crew(body, frames, sunny)
    add_rim_light(root)
    if ATMO_DENSITY > 0:
        add_atmosphere(ATMO_DENSITY)
    hull = bpy.data.objects["Hull"]
    rng = np.random.default_rng(77)

    # Felsen (Kalk-Karst): Kavität, Regenfahnen, Ocker-Eisenflecken, Nässe-/Algenband
    rmat = nature.rock_material("SeaRock", c1=(0.095, 0.088, 0.078), c2=(0.34, 0.32, 0.28), c3=(0.27, 0.22, 0.16),
                                wet_line=1.8, moss=(0.06, 0.10, 0.03), moss_amount=0.28, lichen=0.22, cavity=0.55,
                                crack_w=0.1)
    rocks = []
    for (x, y, r, h, seed, taper) in ROCKS:
        ob = fpv.rock_mesh(f"Rock{seed}", radius=r, height=h + 4, seed=seed, detail=5, taper=taper, mat=rmat,
                           base_z=-4.0, lumpy=0.22, strata=1.0)
        ob.location = (x, y, 0)
        ob.rotation_euler = (0, 0, seed * 1.7)
        rocks.append((ob, r, h))
        if h > 25:
            cap = fpv.rock_mesh(f"RockCap{seed}", radius=r * 0.72, height=3, seed=seed + 100, detail=4, taper=0.6,
                                mat=None, base_z=h - 3.2, lumpy=0.3)
            cap.location = (x, y, 0)
            cap.data.materials.append(nature.grass_material("CapGrass", c1=(0.04, 0.08, 0.02), c2=(0.12, 0.16, 0.05)))
    bpy.context.view_layer.update()

    # Tropischer Bewuchs: Palmen, Laubbäume, Büsche (Instanzen) + Lianen
    leaf = nature.leaf_material("OPLeaves", c1=(0.025, 0.06, 0.012), c2=(0.08, 0.13, 0.025))
    palm_leaf = nature.leaf_material("PalmLeaf", c1=(0.05, 0.09, 0.015), c2=(0.13, 0.17, 0.035), trans=0.4)
    bark = nature.bark_material("OPBark")
    ptrunk = nature.palm_trunk_material()
    trees = [nature.tree_variant(f"OPTree{i}", leaf, bark, height=h, crown_r=cr, n_clusters=nc, leaves_per=90,
                                 seed=40 + i, leaf_size=0.36)
             for i, (h, cr, nc) in enumerate(((9, 3.4, 22), (6, 2.6, 16), (3.0, 1.8, 10)))]
    palms = [nature.palm_variant(f"OPPalm{i}", palm_leaf, ptrunk, height=hh, n_fronds=nf, frond_len=fl, seed=60 + i,
                                 lean=ln) for i, (hh, nf, fl, ln) in enumerate(((10, 15, 4.3, 0.2), (13, 16, 4.8, 0.3),
                                                                                  (7.5, 13, 3.8, 0.12)))]
    _, subs = nature.make_tree_collection("OPTrees", trees + palms)
    veg_pts, veg_scale, veg_pick = [], [], []
    vine_starts = []
    for ob, r, h in rocks:
        if h < 12:
            continue
        me = ob.data
        mw = ob.matrix_world
        n = len(me.polygons)
        nrm = np.zeros(n * 3)
        ctr = np.zeros(n * 3)
        area = np.zeros(n)
        me.polygons.foreach_get("normal", nrm)
        me.polygons.foreach_get("center", ctr)
        me.polygons.foreach_get("area", area)
        R = np.array(mw.to_3x3())
        nrm = nrm.reshape(-1, 3) @ R.T
        ctr = ctr.reshape(-1, 3) @ R.T + np.array(mw.translation)
        up = np.where((nrm[:, 2] > 0.45) & (ctr[:, 2] > max(5.0, 0.3 * h)))[0]
        if len(up):
            w = area[up] * nrm[up, 2] ** 2
            w /= w.sum()
            nveg = int(min(170, 4 + r * r * 0.25 + h * 0.8))
            for k in rng.choice(up, size=nveg, p=w):
                veg_pts.append(tuple(ctr[k] - np.array([0, 0, 0.25])))
                veg_scale.append(rng.uniform(0.55, 1.2))
        wall = np.where((np.abs(nrm[:, 2]) < 0.35) & (ctr[:, 2] > 0.5 * h) & (ctr[:, 2] < 0.95 * h))[0]
        if len(wall):
            for k in rng.choice(wall, size=min(len(wall), int(6 + r * 1.2)), replace=False):
                vine_starts.append((ctr[k], nrm[k]))
    nature.scatter_instances("OPVeg", subs, veg_pts, scales=veg_scale, seed=3)
    vleaf = nature.leaf_material("VineLeaf", c1=(0.03, 0.07, 0.012), c2=(0.07, 0.12, 0.02))
    nature.vines_mesh("Vines", vine_starts, rng, vleaf, bark, len_range=(3, 10))
    print("vegetation", len(veg_pts), "vines", len(vine_starts))

    # Brandung/Flachwasser-Abstandsfeld
    tile, res = 100.0, args.ocean_res
    x0, y0, nx, ny = -190.0, -10.0, 4, 5  # Kacheln -240..160 / -60..440
    shore_img, shore_map = ocean.shore_distance_image([o for o, _, _ in rocks], -250, -70, 170, 450, cell=0.5)

    # Ozean + Geometry-Nodes-Kielspur
    wmat = ocean.water_material("Sea", deep=(0.005, 0.06, 0.19), shallow=(0.015, 0.21, 0.33),
                                wake_fn=foam_builder(shore_img, shore_map), color_fn=shallow_tint(shore_img, shore_map, color=(0.01, 0.24, 0.30)))
    ocean.animate_time_value(wmat, FPS, frames)
    oc = ocean.make_ocean(x0, y0, nx, ny, tile=tile, res=res, wind=9.5, wave_scale=0.9, chop=1.3, fps=FPS,
                          frames=frames, mat=wmat, direction_deg=-30, alignment=0.4, foam_coverage=0.25)
    ocean.ocean_fx_gn(oc, root, hull, bow_x=sunny.X_B, stern_x=sunny.X_S, hull_halfbeam=sunny.B2)
    far = ocean.water_material("FarSea", deep=(0.005, 0.06, 0.19), shallow=(0.015, 0.21, 0.33), far=True)
    ocean.far_plane(x0 - tile / 2 + 1, x0 - tile / 2 + tile * nx - 1, y0 - tile / 2 + 1, y0 - tile / 2 + tile * ny - 1,
                    mat=far, z=-0.05)

    # Inseln am Horizont
    imat = nature.rock_material("IslandRock", c1=(0.08, 0.075, 0.065), c2=(0.22, 0.2, 0.17), c3=(0.15, 0.13, 0.11),
                                wet=False, moss=(0.05, 0.09, 0.025), moss_amount=0.55, scale=8.0, bump=0.4)
    island("IslandNW", -900, 1150, 420, 260, 31, imat)
    island("IslandNE", 700, 1600, 520, 330, 32, imat)
    island("IslandW", -1700, 420, 380, 180, 33, imat)
    island("IslandFar", -300, 2600, 700, 420, 34, imat)

    # Kamera
    cam = fpv.make_camera(pos, quats, fov_deg=92.0)
    cam.data.sensor_fit = "VERTICAL"
    cam.data.sensor_height = 36.0          # 9:16 hochkant: lange Seite = 36 mm (Vollformat-Äquivalent)
    cam.data.lens_unit = "MILLIMETERS"
    cam.data.lens = LENS_MM
    for name, tt in sound_markers(info):
        sc.timeline_markers.new(name, frame=int(round(tt * FPS)) + 1)
    print(f"Bahnlänge {info['used']:.1f} m, Tempo {info['v'].min():.1f}–{info['v'].max():.1f} m/s, "
          f"Marker {[(n, round(t, 2)) for n, t in sound_markers(info)]}")
    return sc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/nido_op")
    ap.add_argument("--res", type=lambda s: tuple(int(v) for v in s.split("x")), default=(1080, 1920))
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--ocean-res", type=int, default=16)
    ap.add_argument("--frames", default=None, help="z. B. 1,90,180 oder 1-600")
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
