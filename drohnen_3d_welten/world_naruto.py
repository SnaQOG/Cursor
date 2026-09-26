"""Welt 2 – Naruto: Konohagakure mit Hokage-Felsen.

Flug (20 s, konstant 16 m/s, eine durchgehende Aufnahme):
  0–5 s   tief auf der Waldstraße auf das große A-un-Tor zu, Durchflug durch das offene Tor
  5–9 s   Hauptstraße auf Dachhöhe: Wassertanks, Strommasten, Banner ziehen vorbei
  9–12 s  S-Kurve links um den alten Baum auf dem Platz
  12–16 s Steigflug rechts an der roten Hokage-Residenz (火) vorbei
  16–20 s die Hokage-Gesichter im Fels füllen das Bild, Kamera fliegt weiter darauf zu
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import fpv  # noqa: E402
import konoha  # noqa: E402
import nature  # noqa: E402
import sunny  # noqa: E402
import textures  # noqa: E402

FPS = 24
SECONDS = 20
SPEED = 16.0
SUN_ELEV, SUN_AZIM = 17.0, 222.0  # goldene Stunde von links hinten: Gesichter modelliert, lange Schatten

WALL_C, WALL_R = (0.0, 190.0), 190.0
STREET_HW = 12.0
TREE_POS = (4.0, 118.0)
RES_POS = (-20.0, 232.0)
CLIFF_Y = 328.0
HEAD_S = 8.5
HEAD_Z = 85.0
HEADS = [(-104, "hashirama"), (-52, "tobirama"), (0, "hiruzen"), (52, "minato"), (104, "tsunade")]

ROUTE = [
    (0, -80, 4.4), (0, -40, 4.8), (0, 0, 5.5), (0, 35, 7.6), (-2, 75, 10.0), (-7, 112, 11.0),
    (-3, 145, 12.5), (2, 170, 15.0), (11, 194, 25.0), (17, 216, 38.0), (17, 238, 51.0),
    (15, 256, 61.0), (13, 272, 67.0), (12, 290, 72.0),
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
    plateau = north * (152 + 22 * fpv.fbm2(X / 160, Y / 160, 5, seed=4))
    return np.maximum(hills, plateau)


def build(args):
    sc = fpv.reset()
    frames = FPS * SECONDS
    fpv.setup_render(args.out, res=args.res, fps=FPS, seconds=SECONDS, samples=args.samples,
                     motion_blur=not args.no_mblur, mist_depth=6000.0)
    fpv.build_world(sun_elev=SUN_ELEV, sun_azim=SUN_AZIM, sky_strength=0.1, clouds=True, cloud_cover=0.45,
                    cloud_ref=9.0, aerosol=2.4, cloud_color=(1.0, 0.82, 0.62))
    fpv.add_sun(SUN_ELEV, SUN_AZIM, strength=6.2, color=(1.0, 0.74, 0.48), angle_deg=0.3)
    rng = np.random.default_rng(12)

    mats = {
        "plaster": konoha.plaster_material("Plaster"),
        "parapet": konoha.plaster_material("Parapet", window=False),
        "rooftop": fpv.simple_mat("RoofTop", (0.22, 0.2, 0.18), rough=0.9),
        "roofs": [konoha.roof_material("RoofRed", (0.42, 0.08, 0.04)),
                  konoha.roof_material("RoofOrange", (0.55, 0.20, 0.06)),
                  konoha.roof_material("RoofGreen", (0.08, 0.18, 0.10)),
                  konoha.roof_material("RoofBlue", (0.10, 0.14, 0.20)),
                  konoha.roof_material("RoofBrown", (0.25, 0.12, 0.06))],
        "awnings": [fpv.simple_mat("AwnRed", (0.45, 0.06, 0.04), rough=0.7),
                    fpv.simple_mat("AwnBlue", (0.08, 0.14, 0.30), rough=0.7),
                    fpv.simple_mat("AwnCream", (0.60, 0.52, 0.38), rough=0.7)],
        "gate": konoha.roof_material("GateWood", (0.34, 0.10, 0.05), var=0.1),
        "gate_roof": konoha.roof_material("GateRoof", (0.12, 0.10, 0.09), var=0.1),
        "stone": nature.rock_material("GateStone", c1=(0.2, 0.19, 0.17), c2=(0.36, 0.34, 0.3), c3=(0.3, 0.28, 0.25),
                                      wet=False, bump=0.4, crack_w=0.3),
        "red": konoha.roof_material("ResRed", (0.50, 0.09, 0.04), var=0.1),
        "red_dark": konoha.roof_material("ResRedDark", (0.30, 0.05, 0.03), var=0.1),
        "red_roof": konoha.roof_material("ResRoof", (0.46, 0.07, 0.035), var=0.15),
        "glass": fpv.simple_mat("ResGlass", (0.02, 0.025, 0.03), rough=0.1),
        "tank": konoha.rust_metal_material("TankMetal"),
        "plaster_street": konoha.plaster_pbr_material("PlasterStreet"),
        "frame": fpv.simple_mat("WinFrame", (0.09, 0.055, 0.03), rough=0.55),
        "sill": fpv.simple_mat("Sill", (0.42, 0.40, 0.37), rough=0.8),
        "shutter": sunny.paint_material("Shutter", (0.10, 0.22, 0.15), rough=0.5),
        "beam": sunny.wood_material("BeamWood", dark=0.75, board=5.0),
        "pipe": konoha.rust_metal_material("PipeMetal", tint=(0.6, 0.6, 0.6)),
        "ac": sunny.paint_material("ACUnit", (0.62, 0.62, 0.60), rough=0.45),
        "tiles": [konoha.terracotta_material("TileRed", (0.40, 0.07, 0.035)),
                  konoha.terracotta_material("TileOrange", (0.50, 0.17, 0.05)),
                  konoha.terracotta_material("TileSlate", (0.12, 0.14, 0.17))],
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
    st = fpv.grid_mesh("MainStreet", 2 * STREET_HW + 2, 178, 2, 2, None, cobble, origin=(0, 88))
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

    konoha.residence(mats, RES_POS[0], RES_POS[1], r=20.0, h=24.0)

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
    rock = nature.rock_material("CliffRock", c1=(0.085, 0.07, 0.06), c2=(0.36, 0.30, 0.235), c3=(0.24, 0.19, 0.14),
                                wet=False, moss=(0.04, 0.06, 0.02), moss_amount=0.3, scale=2.0, bump=1.0,
                                crack_w=0.3, strata_scale=1.2, lichen=0.25)
    face_rock = nature.rock_material("FaceRock", c1=(0.15, 0.125, 0.10), c2=(0.44, 0.37, 0.29), c3=(0.33, 0.26, 0.19),
                                     wet=False, scale=1.5, bump=0.5, crack_w=0.12, lichen=0.2,
                                     moss=(0.06, 0.09, 0.03), moss_amount=0.18, cavity=0.75)
    konoha.cliff_mesh("Cliff", -420, 420, CLIFF_Y, -2, 160, rock, seed=8, res=1.0,
                      panel=(-150, 150, 45, 140), chin=HEAD_Z - 1.2 * HEAD_S)
    cliff_ob = bpy.data.objects["Cliff"]
    head_path = os.path.join(fpv.ASSETS, "LeePerrySmith.glb")
    tpl = konoha.import_head(head_path)
    tpl.hide_render = True
    tpl.hide_viewport = True
    for (hx, hair) in HEADS:
        parts = konoha.hokage_head(tpl, f"Head_{hair}", (hx, CLIFF_Y + 2.0, HEAD_Z), HEAD_S, face_rock, hair,
                                   face_rock, rng)
        konoha.fuse_parts(f"Hokage_{hair}", parts, voxel=0.2, mat=face_rock)

    # Bäume
    leaf = nature.leaf_material("KLeaves", c1=(0.03, 0.06, 0.012), c2=(0.10, 0.14, 0.03))
    leaf2 = nature.leaf_material("KLeaves2", c1=(0.05, 0.075, 0.015), c2=(0.14, 0.15, 0.04))
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
    ok = (nrm[:, 2] > 0.55) & (ctr[:, 2] > 8) & ~((np.abs(ctr[:, 0]) < 150) & (ctr[:, 2] > 50) & (ctr[:, 2] < 140))
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

    # Kamera
    pos, quats, info = fpv.fpv_path(ROUTE, SPEED, FPS, frames, look_pitch=-2.0, pitch_follow=0.45,
                                    bank_gain=1.0, max_bank=30, micro=1.0, seed=9,
                                    pitch_overrides=[(13.5, 21.0, 16.0)])
    fpv.make_camera(pos, quats, fov_deg=92.0)
    print(f"route length {info['total']:.1f} m, used {info['used']:.1f} m")
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
