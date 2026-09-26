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
from mathutils import Vector  # noqa: E402

import fpv  # noqa: E402
import nature  # noqa: E402
import ocean  # noqa: E402
import textures  # noqa: E402

FPS = 24
SECONDS = 20
SPEED = 18.0
SUN_ELEV, SUN_AZIM = 24.0, 300.0          # Hauptsonne (links vorn)
SUNS2 = [(38.0, 40.0, 0.35), (9.0, 22.0, 0.25)]  # zwei weitere Sonnen (elev, azim, Stärke rel.)

MESA_C, MESA_R, MESA_H = (0.0, 250.0), 78.0, 42.0
FAR_TREES = []
DB_C = (-9.0, 234.0)
SHIP_C = (40.0, 268.0)

ROUTE = [
    (0, -15, 3.4), (-2, 40, 3.4), (2, 95, 4.2), (1, 128, 12.0), (0, 152, 30.0), (0, 170, 47.0),
    (2, 190, 49.0), (4, 220, 47.5), (7, 252, 47.5), (7, 288, 48.0), (4, 324, 47.5), (0, 360, 42.0),
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
    col = nb.ramp(t, [(0.0, (0.62, 0.86, 0.72)), (0.12, (0.42, 0.76, 0.64)), (0.45, (0.18, 0.55, 0.50)),
                      (1.0, (0.07, 0.34, 0.34))])
    below = nb.math("LESS_THAN", z, 0.0)
    col = nb.mix(below, col, (0.45, 0.70, 0.60))
    # Leuchten um die Hauptsonne
    sd = fpv.sun_dir(SUN_ELEV, SUN_AZIM)
    dp = nb.math("MAXIMUM", nb.vmath("DOT_PRODUCT", d, tuple(sd)), 0.0)
    glow = nb.math("ADD", nb.math("MULTIPLY", nb.math("POWER", dp, 8.0), 0.5),
                   nb.math("MULTIPLY", nb.math("POWER", dp, 64.0), 1.5))
    col = nb.vmath("ADD", col, nb.vmath("SCALE", (1.0, 0.97, 0.85), scale=glow))
    return col


def namek_rock():
    return nature.rock_material("NamekRock", c1=(0.12, 0.095, 0.07), c2=(0.44, 0.36, 0.26), c3=(0.30, 0.22, 0.15),
                                wet_line=1.6, algae=(0.03, 0.06, 0.04), moss=(0.05, 0.17, 0.13), moss_amount=0.3,
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


def house(name, x, y, z, r, mats, rng):
    objs = []
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=20, radius=1.0)
    ob = fpv.mesh_from_bmesh(bm, name, mats["white"])
    ob.location = (x, y, z - r * 0.1)
    ob.scale = (r, r, r * rng.uniform(0.85, 1.05))
    objs.append(ob)
    # runde Fenster (zwei Reihen, zufällig)
    for k in range(int(rng.integers(5, 9))):
        a = rng.uniform(0, 2 * math.pi)
        el = rng.choice([0.25, 0.55])
        nrm = Vector((math.cos(a) * math.cos(el), math.sin(a) * math.cos(el), math.sin(el)))
        p = Vector((x, y, z - r * 0.1)) + Vector((nrm.x * r, nrm.y * r, nrm.z * r * 0.95))
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=20, radius1=r * 0.13, radius2=r * 0.13, depth=r * 0.12)
        w = fpv.mesh_from_bmesh(bm, name + "_w", mats["window"])
        w.rotation_mode = "QUATERNION"
        w.rotation_quaternion = nrm.to_track_quat("Z", "Y")
        w.location = p
        objs.append(w)
    # Eingang
    a = rng.uniform(0, 2 * math.pi)
    nrm = Vector((math.cos(a), math.sin(a), 0.1)).normalized()
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=r * 0.26, radius2=r * 0.26, depth=r * 0.2)
    dr = fpv.mesh_from_bmesh(bm, name + "_door", mats["window"])
    dr.rotation_mode = "QUATERNION"
    dr.rotation_quaternion = nrm.to_track_quat("Z", "Y")
    dr.location = Vector((x, y, z + r * 0.12)) + nrm * r * 0.93
    dr.scale = (1, 1.5, 1)
    objs.append(dr)
    # Röhren/Aufsätze oben
    if rng.random() < 0.7:
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=r * 0.12, radius2=r * 0.09, depth=r * 0.6)
        t = fpv.mesh_from_bmesh(bm, name + "_tube", mats["white"])
        t.location = (x + r * 0.25, y, z + r * 0.95)
        objs.append(t)
    return objs


def dragon_ball_materials():
    mats = []
    for n in range(1, 8):
        path = os.path.join(textures.OUT, f"db_star{n}.png")
        textures.star_map(n, path)
        mat, nb, out = fpv.new_material(f"DragonBall{n}")
        glass = nb.node("ShaderNodeBsdfGlass")
        glass.inputs["Color"].default_value = (1.0, 0.55, 0.12, 1)
        glass.inputs["Roughness"].default_value = 0.02
        glass.inputs["IOR"].default_value = 1.45
        em = nb.node("ShaderNodeEmission")
        em.inputs["Color"].default_value = (1.0, 0.45, 0.08, 1)
        em.inputs["Strength"].default_value = 0.55
        add = nb.node("ShaderNodeAddShader")
        nb.link(glass.outputs[0], add.inputs[0])
        nb.link(em.outputs[0], add.inputs[1])
        nb.link(add.outputs[0], out.inputs[0])
        smat, snb, sout = fpv.new_material(f"DBStars{n}")
        uv = snb.coords("UV")
        img = snb.image(path, uv)
        p = fpv.principled(snb, Base_Color=(0.8, 0.05, 0.03), Roughness=0.4)
        p.inputs["Emission Color"].default_value = (1.0, 0.08, 0.03, 1)
        p.inputs["Emission Strength"].default_value = 1.5
        snb.link(snb.out(img, "Alpha"), p.inputs["Alpha"])
        snb.link(p.outputs[0], sout.inputs[0])
        mats.append((mat, smat))
    return mats


def spaceship(mats, cx, cy, z0):
    objs = []
    R = 16.0
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=96, v_segments=48, radius=1.0)
    body = fpv.mesh_from_bmesh(bm, "ShipBody", mats["hull"])
    body.location = (cx, cy, z0 + 7.5)
    body.scale = (R, R, 7.0)
    objs.append(body)
    # Fensterband: Ring runder Fenster auf dem Äquator
    for k in range(28):
        a = k * 2 * math.pi / 28
        nrm = Vector((math.cos(a), math.sin(a), 0.0))
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=1.0, radius2=1.0, depth=0.5)
        w = fpv.mesh_from_bmesh(bm, "ShipWin", mats["shipwin"])
        w.rotation_mode = "QUATERNION"
        w.rotation_quaternion = nrm.to_track_quat("Z", "Y")
        w.location = Vector((cx, cy, z0 + 8.4)) + nrm * (R - 0.05)
        objs.append(w)
    # Ringwulst unten + Kuppel oben
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=96, radius1=R * 0.93, radius2=R * 0.75, depth=1.6)
    rim = fpv.mesh_from_bmesh(bm, "ShipRim", mats["hull_dark"])
    rim.location = (cx, cy, z0 + 3.6)
    objs.append(rim)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
    dome = fpv.mesh_from_bmesh(bm, "ShipDome", mats["hull_dark"])
    dome.location = (cx, cy, z0 + 13.6)
    dome.scale = (4.5, 4.5, 2.2)
    objs.append(dome)
    # Landebeine
    for k in range(3):
        a = k * 2 * math.pi / 3 + 0.4
        p0 = Vector((cx + math.cos(a) * R * 0.55, cy + math.sin(a) * R * 0.55, z0 + 4.0))
        p1 = Vector((cx + math.cos(a) * R * 0.72, cy + math.sin(a) * R * 0.72, z0 - 0.5))
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.6, radius2=0.45, depth=(p1 - p0).length)
        leg = fpv.mesh_from_bmesh(bm, "ShipLeg", mats["hull_dark"])
        leg.rotation_mode = "QUATERNION"
        leg.rotation_quaternion = (p1 - p0).normalized().to_track_quat("Z", "Y")
        leg.location = (p0 + p1) / 2
        objs.append(leg)
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=1.4, radius2=1.1, depth=0.5)
        foot = fpv.mesh_from_bmesh(bm, "ShipFoot", mats["hull_dark"])
        foot.location = p1
        objs.append(foot)
    return objs


def metal_paint(name, color, rough=0.3):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    n = nb.noise(co, scale=3.0, detail=3)
    panels = nb.voronoi(co, scale=1.5, feature="DISTANCE_TO_EDGE")
    seam = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(panels, "Distance"), seam.inputs["Value"])
    seam.inputs["From Min"].default_value = 0.0
    seam.inputs["From Max"].default_value = 0.02
    c = nb.mix(nb.math("MULTIPLY", nb.out(n, "Fac"), 0.25), color, [v * 0.8 for v in color])
    c = nb.mix(nb.math("MULTIPLY", nb.math("SUBTRACT", 1, seam.outputs[0]), 0.5), c, [v * 0.5 for v in color])
    p = fpv.principled(nb, Base_Color=c, Roughness=rough, Metallic=0.3)
    p.inputs["Coat Weight"].default_value = 0.3
    nb.link(nb.bump(seam.outputs[0], strength=0.3, distance=0.02), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def build(args):
    sc = fpv.reset()
    frames = FPS * SECONDS
    fpv.setup_render(args.out, res=args.res, fps=FPS, seconds=SECONDS, samples=args.samples,
                     motion_blur=not args.no_mblur, mist_depth=6000.0)
    extra = [(SUN_ELEV, SUN_AZIM, 0.9, 900.0, (1.0, 0.98, 0.9))]
    for (e, a, s) in SUNS2:
        extra.append((e, a, 0.6, 500.0 * s, (1.0, 0.97, 0.9)))
    fpv.build_world(sun_elev=SUN_ELEV, sun_azim=SUN_AZIM, sky_strength=1.0, clouds=True, cloud_cover=0.25,
                    cloud_ref=1.1, custom_sky=namek_sky, extra_suns=extra, cloud_color=(0.95, 1.0, 0.95),
                    cloud_scale=1.3)
    fpv.add_sun(SUN_ELEV, SUN_AZIM, strength=4.6, color=(1.0, 0.96, 0.86), name="Sun1")
    for i, (e, a, s) in enumerate(SUNS2):
        fpv.add_sun(e, a, strength=4.6 * s, color=(0.95, 1.0, 0.92), name=f"Sun{i + 2}", angle_deg=0.4)
    rng = np.random.default_rng(31)

    # Meer
    wmat = ocean.water_material("NamekSea", deep=(0.006, 0.05, 0.035), shallow=(0.03, 0.15, 0.10))
    ocean.animate_time_value(wmat, FPS, frames)
    tile = 100.0
    x0, y0, nx, ny = -140.0, -40.0, 3, 5
    ocean.make_ocean(x0, y0, nx, ny, tile=tile, res=13, wind=6.5, wave_scale=0.7, chop=1.1, fps=FPS, frames=frames,
                     mat=wmat, direction_deg=60, alignment=0.3, foam_coverage=0.1)
    far = ocean.water_material("NamekFarSea", deep=(0.006, 0.05, 0.035), shallow=(0.03, 0.15, 0.10), far=True)
    ocean.far_plane(x0 - tile / 2 + 1, x0 - tile / 2 + tile * nx - 1, y0 - tile / 2 + 1, y0 - tile / 2 + tile * ny - 1,
                    mat=far, z=-0.05)

    rock = namek_rock()
    # Tafelberg (Heightfield)
    mesa = fpv.grid_mesh("Mesa", 2 * MESA_R + 50, 2 * MESA_R + 50, 360, 360, mesa_height, rock, origin=MESA_C)
    # Felsnadeln mit pilzförmig verbreiterten Köpfen
    for (x, y, r, h, seed) in SPIRES:
        ob = fpv.rock_mesh(f"Spire{seed}", radius=r, height=h + 4, seed=seed + 50, detail=5, taper=-0.35, mat=rock,
                           base_z=-4.0, lumpy=0.18, strata=2.0)
        ob.location = (x, y, 0)
        ob.rotation_euler = (0, 0, seed * 2.1)
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

    # Materialien Plateau
    mats = {
        "white": nature.rock_material("DomeWhite", c1=(0.62, 0.64, 0.60), c2=(0.78, 0.80, 0.76),
                                      c3=(0.70, 0.72, 0.68), wet=False, bump=0.15, crack_w=0.0, lichen=0.05, scale=0.5),
        "window": fpv.simple_mat("DomeWin", (0.02, 0.03, 0.03), rough=0.15),
        "hull": metal_paint("ShipHull", (0.62, 0.62, 0.60)),
        "hull_dark": metal_paint("ShipHullDark", (0.30, 0.30, 0.32)),
        "shipwin": fpv.simple_mat("ShipWin", (0.4, 0.05, 0.05), rough=0.1, emission=(1.0, 0.25, 0.2), estrength=1.2),
    }
    top_z = lambda x, y: float(mesa_height(np.array([x]), np.array([y]))[0])

    # Namekianer-Häuser
    houses = [(-22, 204, 6.5), (-38, 224, 8.0), (-24, 250, 5.5), (-48, 252, 6.0), (-30, 274, 7.5), (26, 204, 6.0),
              (32, 228, 5.0), (-16, 296, 6.0), (-48, 290, 5.0), (26, 308, 7.0)]
    for i, (x, y, r) in enumerate(houses):
        house(f"House{i}", x, y, top_z(x, y), r, mats, rng)

    # Dragon Balls (Kreis) + Glühen
    dbm = dragon_ball_materials()
    zc = top_z(*DB_C)
    for n in range(7):
        a = n * 2 * math.pi / 7 + 0.3
        x, y = DB_C[0] + math.cos(a) * 5.0, DB_C[1] + math.sin(a) * 5.0
        rad = 1.0
        z = top_z(x, y) + rad * 0.9
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=rad)
        ball = fpv.mesh_from_bmesh(bm, f"DB{n + 1}", dbm[n][0])
        ball.location = (x, y, z)
        bm = bmesh.new()
        uvl = bm.loops.layers.uv.new("UVMap")
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=rad * 0.55)
        for f in bm.faces:
            for lp in f.loops:
                co = lp.vert.co.normalized()
                lp[uvl].uv = (0.5 + math.atan2(co.y, co.x) / (2 * math.pi), 0.5 + math.asin(max(-1, min(1, co.z))) / math.pi)
        st = fpv.mesh_from_bmesh(bm, f"DBStars{n + 1}", dbm[n][1])
        st.location = (x, y, z)
        # Sterne zur Flugbahn drehen (ungefähr nach Osten/Kamera)
        st.rotation_euler = (0, 0, rng.uniform(-0.3, 0.3))
        ld = bpy.data.lights.new(f"DBGlow{n}", "POINT")
        ld.energy = 90.0
        ld.color = (1.0, 0.55, 0.15)
        ld.shadow_soft_size = rad * 0.8
        lo = bpy.data.objects.new(f"DBGlow{n}", ld)
        lo.location = (x, y, z + 0.1)
        fpv.link(lo)
        ball.visible_shadow = False
        st.visible_shadow = False

    spaceship(mats, SHIP_C[0], SHIP_C[1], top_z(*SHIP_C) + 1.5)

    # Ajisa-Bäume
    leaf = nature.leaf_material("AjisaLeaf", c1=(0.03, 0.13, 0.10), c2=(0.08, 0.26, 0.19), trans=0.3)
    bark = nature.bark_material("AjisaBark", c=(0.14, 0.13, 0.09))
    variants = [nature.ajisa_variant(f"Ajisa{i}", leaf, bark, height=h, crown_r=cr, leaves=int(900 * cr * cr / 4),
                                     seed=70 + i, leaf_size=0.36)
                for i, (h, cr) in enumerate(((7.0, 3.0), (9.5, 3.8), (5.5, 2.4), (11.0, 4.2)))]
    _, subs = nature.make_tree_collection("AjisaTrees", variants)
    pos, _, _ = fpv.fpv_path(ROUTE, SPEED, FPS, frames, micro=0.0)
    path_xy = pos[:, :2]
    pts, scl = [], []
    avoid = [(x, y, r + 3) for (x, y, r) in houses] + [(DB_C[0], DB_C[1], 9), (SHIP_C[0], SHIP_C[1], 22)]
    # Baumgruppen (Haine) statt Wald
    groves = [(-56, 205, 9, 8), (50, 210, 8, 6), (-6, 268, 6, 4), (-52, 310, 10, 8), (38, 322, 9, 7),
              (62, 240, 7, 5), (-66, 262, 8, 6), (18, 186, 6, 4), (-28, 184, 6, 5), (-2, 330, 6, 4),
              (22, 248, 5, 3), (-20, 226, 4, 3), (24, 285, 6, 4)]
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
    # Findlinge auf dem Plateau
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
    # auf den Felsnadeln
    for (x, y, r, h, seed) in SPIRES:
        for k in range(int(3 + r * 0.8)):
            a = rng.uniform(0, 2 * math.pi)
            rr = r * 1.1 * math.sqrt(rng.random())
            pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr, h - 0.6))
            scl.append(rng.uniform(0.45, 0.9))
    nature.scatter_instances("Ajisa", subs, pts, scales=scl, seed=4)
    print("ajisa", len(pts))

    pos, quats, info = fpv.fpv_path(ROUTE, SPEED, FPS, frames, look_pitch=-3.0, pitch_follow=0.5,
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
