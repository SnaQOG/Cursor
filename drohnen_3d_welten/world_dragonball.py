"""Welt 3 – Dragon Ball: Planet Namek.

Türkiser Himmel mit drei Sonnen, grünliches Meer, Felsnadeln, Tafelberg mit Ajisa-Bäumen,
Namekianer-Kuppelhäusern, den sieben Dragon Balls und Friezas Raumschiff.

Flug (20 s, konstant 18 m/s, eine durchgehende Aufnahme):
  0–6 s    tief über dem Meer zwischen Felsnadeln hindurch
  6–10,5 s Steigflug an der Steilwand des Tafelbergs hinauf
  ~10,8 s  Reveal über der Kante: Dorf, leuchtende Dragon Balls, Raumschiff (vorher verdeckt)
  11–20 s  tief über das Plateau: an den Dragon Balls vorbei, Raumschiff rechts, Richtung Nordkante
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import bmesh  # noqa: E402,I100
import numpy as np  # noqa: E402

import fpv  # noqa: E402
import namek  # noqa: E402
import nature  # noqa: E402
from world_onepiece import foam_builder, shallow_tint  # noqa: E402
import ocean  # noqa: E402

FPS = 24
SECONDS = 20
SPEED = 18.0
SUN_ELEV, SUN_AZIM = 24.0, 300.0          # Hauptsonne (links vorn)
SUNS2 = [(38.0, 40.0, 0.35), (9.0, 22.0, 0.25)]  # zwei weitere Sonnen (elev, azim, Stärke rel.)

MESA_C, MESA_R, MESA_H = (0.0, 250.0), 78.0, 42.0
FAR_TREES = []
DB_C = (1.0, 232.0)
SHIP_C = (40.0, 268.0)

ROUTE = [
    (0, -15, 3.4), (-2, 40, 3.4), (2, 95, 4.2), (1, 128, 12.0), (0, 152, 30.0), (0, 170, 47.0),
    (2, 190, 49.0), (4, 220, 47.5), (7, 252, 47.5), (7, 288, 48.0), (4, 312, 44.0), (1, 336, 43.6),
    (-2, 372, 43.0),
]

SPIRES = [
    # (x, y, r, h, seed)
    (-15, 52, 7.0, 46, 1), (16, 72, 8.0, 58, 2), (-48, 20, 6.0, 30, 3), (42, 18, 5.0, 24, 4),
    (-42, 110, 10.0, 52, 5), (55, 128, 9.0, 40, 6), (-95, 170, 14.0, 70, 7), (110, 200, 12.0, 60, 8),
    (-150, 60, 16.0, 55, 9), (140, 40, 10.0, 48, 10), (25, 120, 4.0, 14, 11), (-22, 150, 5.0, 20, 12),
]


def namek_sky(nb, d):
    x, y, z = nb.sep(d)
    zz = nb.math("MAXIMUM", z, 0.0)
    t = nb.math("POWER", zz, 0.55)
    col = nb.ramp(t, [(0.0, (0.76, 0.90, 0.56)), (0.12, (0.56, 0.82, 0.46)), (0.45, (0.26, 0.60, 0.34)),
                      (1.0, (0.10, 0.40, 0.26))])
    below = nb.math("LESS_THAN", z, 0.0)
    col = nb.mix(below, col, (0.55, 0.74, 0.48))
    # Leuchten um die Hauptsonne
    sd = fpv.sun_dir(SUN_ELEV, SUN_AZIM)
    dp = nb.math("MAXIMUM", nb.vmath("DOT_PRODUCT", d, tuple(sd)), 0.0)
    glow = nb.math("ADD", nb.math("MULTIPLY", nb.math("POWER", dp, 8.0), 0.5),
                   nb.math("MULTIPLY", nb.math("POWER", dp, 64.0), 1.5))
    col = nb.vmath("ADD", col, nb.vmath("SCALE", (1.0, 0.97, 0.85), scale=glow))
    return col


def namek_rock():
    return nature.rock_material("NamekRock", c1=(0.13, 0.095, 0.065), c2=(0.50, 0.38, 0.24), c3=(0.34, 0.24, 0.14),
                                wet_line=1.6, algae=(0.03, 0.06, 0.04), moss=(0.04, 0.17, 0.16), moss_amount=0.3,
                                strata_scale=2.2, bump=0.9, crack_w=0.15, scale=1.5, lichen=0.1,
                                moss_tex=os.path.join(fpv.ASSETS, "grasslight-big.jpg"), moss_tex_scale=4.0)


def mesa_height(X, Y):
    r = np.sqrt((X - MESA_C[0]) ** 2 + (Y - MESA_C[1]) ** 2)
    ang = np.arctan2(Y - MESA_C[1], X - MESA_C[0])
    ca, sa = np.cos(ang), np.sin(ang)
    edge = MESA_R * (1 + 0.08 * np.sin(ang * 3 + 1) + 0.05 * np.sin(ang * 7 + 2)
                     + 0.06 * fpv.fbm2(ca * 3, sa * 3, 3, seed=21)
                     + 0.025 * fpv.fbm2(ca * 25, sa * 25, 3, seed=24))  # Rinnen
    s = edge - r
    W = 16.0
    u = np.clip(s / W, 0, 1)
    # Felsstufen (Schichtbänke)
    k = 5.0
    fu = u * k
    step = np.floor(fu)
    fr = fu - step
    fr = np.clip((fr - 0.55) / 0.45, 0, 1)
    fr = fr * fr * (3 - 2 * fr)
    ut = (step + fr) / k
    ut = np.where(u >= 1.0, 1.0, ut)
    top = MESA_H + 2.0 * fpv.fbm2(X / 40, Y / 40, 4, seed=22) + 0.4 * fpv.fbm2(X / 6, Y / 6, 3, seed=23)
    foot = np.clip((edge + 18 - r) / 18.0, 0, 1) ** 2 * 7
    rough = 0.8 * fpv.fbm2(X / 3, Y / 3, 3, seed=25) * (u > 0) * (u < 1)
    h = foot + ut * (top - foot) + rough - 3
    return h


def build(args):
    sc = fpv.reset()
    frames = FPS * SECONDS
    fpv.setup_render(args.out, res=args.res, fps=FPS, seconds=SECONDS, samples=args.samples,
                     motion_blur=not args.no_mblur, mist_depth=6000.0)
    extra = [(SUN_ELEV, SUN_AZIM, 0.9, 900.0, (1.0, 0.98, 0.9))]
    for (e, a, s) in SUNS2:
        extra.append((e, a, 0.6, 500.0 * s, (1.0, 0.97, 0.9)))
    fpv.build_world(sun_elev=SUN_ELEV, sun_azim=SUN_AZIM, sky_strength=0.85, clouds=True, cloud_cover=0.25,
                    cloud_ref=1.1, custom_sky=namek_sky, extra_suns=extra, cloud_color=(0.95, 1.0, 0.95),
                    cloud_scale=1.3)
    fpv.add_sun(SUN_ELEV, SUN_AZIM, strength=5.6, color=(1.0, 0.92, 0.76), name="Sun1", angle_deg=1.5)
    for i, (e, a, s) in enumerate(SUNS2):
        fpv.add_sun(e, a, strength=4.6 * s, color=((0.92, 0.97, 1.0), (0.93, 1.0, 0.9))[i], name=f"Sun{i + 2}",
                    angle_deg=(2.5, 3.0)[i])
    rng = np.random.default_rng(31)
    top_z = lambda x, y: float(mesa_height(np.array([x]), np.array([y]))[0])
    # Route über dem Plateau: 3,6 m über Grund (tiefer Vorbeiflug an den Dragon Balls);
    # über die Nordkante hinaus bleibt die Höhe erhalten (Blick aufs Meer)
    route = []
    for (x, y, z) in ROUTE:
        if 185 <= y <= 320 and top_z(x, y) > MESA_H - 8:
            z = top_z(x, y) + 3.6
        route.append((x, y, z))

    rock = namek_rock()
    # Tafelberg (Heightfield)
    mesa = fpv.grid_mesh("Mesa", 2 * MESA_R + 50, 2 * MESA_R + 50, 360, 360, mesa_height, rock, origin=MESA_C)
    # Felsnadeln (zylindrisch-vertikal, pilzförmiger Kopf, Schichtbänke)
    spires = []
    for (x, y, r, h, seed) in SPIRES:
        ob = fpv.rock_mesh(f"Spire{seed}", radius=r, height=h + 4, seed=seed + 50, detail=5, taper=-0.35, mat=rock,
                           base_z=-4.0, lumpy=0.18, strata=2.2)
        ob.location = (x, y, 0)
        ob.rotation_euler = (0, 0, seed * 2.1)
        spires.append(ob)
    # ferne Inseln: große Pilzfelsen + Tafelberge
    for k, (x, y, r, h) in enumerate(((-160, 470, 26, 95), (120, 520, 20, 70), (-60, 640, 34, 120),
                                      (260, 700, 30, 85), (-330, 820, 45, 140), (40, 980, 55, 150),
                                      (420, 1050, 50, 120), (-620, 1150, 70, 170))):
        ob = fpv.rock_mesh(f"FarSpire{k}", radius=r, height=h + 4, seed=200 + k, detail=4, taper=-0.3, mat=rock,
                           base_z=-4.0, lumpy=0.2, strata=2.0)
        ob.location = (x, y, 0)
        for j in range(int(r * 0.5)):
            a = rng.uniform(0, 2 * math.pi)
            rr = r * 1.1 * math.sqrt(rng.random())
            FAR_TREES.append((x + math.cos(a) * rr, y + math.sin(a) * rr, h - 0.6))
    for k, (x, y, r, h) in enumerate(((-900, 1400, 260, 150), (800, 1700, 300, 170))):
        def hf(X, Y, x=x, y=y, r=r, h=h, k=k):
            rr = np.sqrt((X - x) ** 2 + (Y - y) ** 2) / r
            e = 1 + 0.12 * fpv.fbm2(X / r * 2, Y / r * 2, 3, seed=40 + k)
            t = np.clip((e - rr) / 0.12, 0, 1)
            return t * t * (3 - 2 * t) * h * (1 + 0.05 * fpv.fbm2(X / 30, Y / 30, 3, seed=50 + k)) - 3
        fpv.grid_mesh(f"Island{k}", r * 2.6, r * 2.6, 180, 180, hf, rock, origin=(x, y))
    bpy.context.view_layer.update()

    # Meer: smaragdgrün, Flachwasser + Brandung an Felsen und Tafelberg (Abstandsfeld)
    shore_img, shore_map = ocean.shore_distance_image(spires + [mesa], -200, -100, 200, 420, cell=0.5,
                                                      name="NamekShore")
    wmat = ocean.water_material("NamekSea", deep=(0.008, 0.06, 0.04), shallow=(0.04, 0.17, 0.11),
                                wake_fn=foam_builder(shore_img, shore_map), foam_amount=0.4,
                                color_fn=shallow_tint(shore_img, shore_map))
    ocean.animate_time_value(wmat, FPS, frames)
    tile = 100.0
    x0, y0, nx, ny = -140.0, -40.0, 3, 5
    ocean.make_ocean(x0, y0, nx, ny, tile=tile, res=13, wind=6.5, wave_scale=0.7, chop=1.1, fps=FPS, frames=frames,
                     mat=wmat, direction_deg=60, alignment=0.3, foam_coverage=0.1)
    far = ocean.water_material("NamekFarSea", deep=(0.008, 0.06, 0.04), shallow=(0.04, 0.17, 0.11), far=True)
    ocean.far_plane(x0 - tile / 2 + 1, x0 - tile / 2 + tile * nx - 1, y0 - tile / 2 + 1, y0 - tile / 2 + tile * ny - 1,
                    mat=far, z=-0.05)

    # Materialien Plateau
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

    # Namekianer-Häuser (glatte Lehm-Kuppeln, Rippen, gehörnte Oberkuppel, Rundfenster)
    houses = [(-20, 204, 6.5), (-36, 224, 8.0), (-22, 252, 5.5), (-46, 252, 6.0), (-28, 276, 7.5), (26, 204, 6.0),
              (30, 228, 5.0), (-16, 298, 6.0), (-46, 292, 5.0), (26, 308, 7.0)]
    for i, (x, y, r) in enumerate(houses):
        namek.house(f"House{i}", x, y, top_z(x, y), r, mats, rng)

    # Dragon Balls in realer Größe (Ø ~0,4 m) auf einem niedrigen, runden Steinsockel links der Flugbahn
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

    # Ajisa-Bäume (verdrehte dünne Stämme, perfekte Kugelkronen)
    leaf = nature.leaf_material("AjisaLeaf", c1=(0.03, 0.13, 0.10), c2=(0.08, 0.26, 0.19), trans=0.3)
    bark = nature.bark_material("AjisaBark", c=(0.16, 0.15, 0.10))
    variants = [namek.ajisa_variant(f"Ajisa{i}", leaf, bark, height=h, crown_r=cr, leaves=int(900 * cr * cr / 4),
                                    seed=70 + i, leaf_size=0.36, turns=tw)
                for i, (h, cr, tw) in enumerate(((7.0, 3.0, 1.4), (9.5, 3.8, 1.8), (5.5, 2.4, 1.2), (11.0, 4.2, 2.1)))]
    _, subs = nature.make_tree_collection("AjisaTrees", variants)
    pos, _, _ = fpv.fpv_path(route, SPEED, FPS, frames, micro=0.0)
    path_xy = pos[:, :2]
    pts, scl = [], []
    avoid = [(x, y, r + 3) for (x, y, r) in houses] + [(DB_C[0], DB_C[1], 5), (SHIP_C[0], SHIP_C[1], 22)]
    groves = [(-56, 205, 9, 8), (50, 210, 8, 6), (-6, 270, 6, 4), (-52, 312, 10, 8), (38, 322, 9, 7),
              (62, 240, 7, 5), (-66, 264, 8, 6), (20, 186, 6, 4), (-28, 184, 6, 5), (-4, 334, 6, 4),
              (24, 250, 5, 3), (-16, 226, 4, 3), (26, 286, 6, 4)]
    cand = []
    for (gx, gy, gr, gn) in groves:
        for _ in range(gn * 3):
            a = rng.uniform(0, 2 * math.pi)
            rr = gr * math.sqrt(rng.random())
            cand.append((gx + math.cos(a) * rr, gy + math.sin(a) * rr))
    for (x, y) in cand:
        if any(math.hypot(x - ax, y - ay) < ar for (ax, ay, ar) in avoid):
            continue
        if np.min(np.hypot(path_xy[:, 0] - x, path_xy[:, 1] - y)) < 11:
            continue
        z = top_z(x, y)
        if z < MESA_H - 6:
            continue
        pts.append((x, y, z - 0.2))
        scl.append(rng.uniform(0.7, 1.25))
    for p in FAR_TREES:
        pts.append(p)
        scl.append(rng.uniform(1.2, 2.2))
    # Findlinge
    boulders = []
    for k in range(45):
        a = rng.uniform(0, 2 * math.pi)
        rr = MESA_R * 0.9 * math.sqrt(rng.random())
        x, y = MESA_C[0] + math.cos(a) * rr, MESA_C[1] + math.sin(a) * rr
        if np.min(np.hypot(path_xy[:, 0] - x, path_xy[:, 1] - y)) < 5 or any(
                math.hypot(x - ax, y - ay) < ar for (ax, ay, ar) in avoid):
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
    for (x, y, r, h, seed) in SPIRES:
        for k in range(int(3 + r * 0.8)):
            a = rng.uniform(0, 2 * math.pi)
            rr = r * 1.1 * math.sqrt(rng.random())
            pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr, h - 0.6))
            scl.append(rng.uniform(0.45, 0.9))
    nature.scatter_instances("Ajisa", subs, pts, scales=scl, seed=4)
    print("ajisa", len(pts))

    # Blaugrünes Gras als echte Halme entlang der Flugbahn (Dichte fällt mit dem Abstand)
    excl = [(x, y, r * 0.95) for (x, y, r) in houses] + [(DB_C[0], DB_C[1], 1.7), (SHIP_C[0], SHIP_C[1], 17.5)]
    excl += boulders
    namek.grass_field("NamekGrass", mesa_height, path_xy, namek.grass_blade_material(), rng, ymin=172, ymax=330,
                      exclude=excl, zmin=MESA_H - 5)

    pos, quats, info = fpv.fpv_path(route, SPEED, FPS, frames, look_pitch=-3.0, pitch_follow=0.5,
                                    bank_gain=1.0, max_bank=30, micro=1.0, seed=13)
    fpv.make_camera(pos, quats, fov_deg=92.0)
    print(f"route length {info['total']:.1f} m, used {info['used']:.1f} m")
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
