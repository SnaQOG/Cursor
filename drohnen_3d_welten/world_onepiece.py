"""Welt 1 – One Piece: Grand Line, Felsnadeln im Meer, Thousand Sunny.

Flug (20 s, konstant 15 m/s, eine durchgehende Aufnahme):
  0–5 s   tief (3 m) über den Wellen, Durchflug zwischen zwei Felsnadeln
  5–9 s   leichter Schwenk rechts, dann Linkskurve mit Schräglage um die große Felsnadel
  ~9 s    Reveal: die Thousand Sunny taucht hinter der Felsnadel auf (bisher verdeckt)
  9–15 s  Anflug auf den Löwenkopf, Steigflug auf ~9 m
  15–17 s Vorbeiflug an Bug, Rumpf und Jolly-Roger-Segel (Schiff rechts)
  17–20 s weiter über das offene Meer Richtung Inseln am Horizont
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
import nature  # noqa: E402
import ocean  # noqa: E402
import sunny  # noqa: E402

FPS = 24
SECONDS = 20
SPEED = 15.0
SUN_ELEV, SUN_AZIM = 22.0, 200.0

ROUTE = [
    (0, -8, 3.0), (0, 40, 3.1), (-1, 80, 3.4), (9, 118, 4.3), (13, 150, 5.4), (4, 184, 7.0),
    (-18, 207, 8.2), (-45, 231, 9.0), (-75, 254, 9.2), (-108, 276, 8.8), (-140, 296, 8.5),
]

SHIP_HEADING = -40.5  # Grad (mathematisch, von +X), Bug zeigt nach Südosten
SHIP_SPEED = 3.0
SHIP_T15 = Vector((-27.9, 233.4, 0.0))  # Position bei t = 15 s


def ship_start():
    h = math.radians(SHIP_HEADING)
    d = Vector((math.cos(h), math.sin(h), 0))
    return SHIP_T15 - d * SHIP_SPEED * 15.0


ROCKS = [
    # (x, y, radius, height, seed, taper)
    (-17, 70, 9.0, 30, 11, 0.35),    # Tor links
    (15, 82, 7.0, 20, 12, 0.45),     # Tor rechts
    (-20, 150, 20.0, 62, 13, 0.25),  # große Felsnadel (verdeckt das Schiff)
    (-38, 38, 4.0, 7, 14, 0.5),
    (27, 48, 3.0, 4, 15, 0.6),
    (30, 128, 5.5, 12, 16, 0.4),
    (-52, 118, 8.0, 26, 17, 0.35),
    (40, 200, 6.0, 15, 18, 0.4),
    (-5, 262, 3.5, 5, 19, 0.6),
    (-150, 205, 12.0, 38, 20, 0.3),
    (70, 110, 10.0, 34, 21, 0.3),
    (-122, 305, 7.0, 18, 22, 0.35),
    (-150, 268, 5.0, 12, 23, 0.4),
    (-205, 345, 11.0, 30, 24, 0.3),
    (-96, 330, 4.0, 7, 25, 0.5),
]


def wake_builder(ship_root):
    """Schaum am Rumpf, Bugwelle und Kielwasser als Maske in Schiffskoordinaten."""

    def fn(nb, wp_t):
        co = nb.coords("Object", ship_root)
        x, y, _ = nb.sep(co)
        ay = nb.math("ABSOLUTE", y)
        # Rumpfkontakt (Superellipse)
        d = nb.math("SQRT", nb.math("ADD", nb.math("POWER", nb.math("DIVIDE", x, 16.2), 2.0),
                                     nb.math("POWER", nb.math("DIVIDE", ay, 5.2), 2.0)))
        ring_in = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(d, ring_in.inputs["Value"])
        ring_in.inputs["From Min"].default_value = 0.92
        ring_in.inputs["From Max"].default_value = 1.0
        ring_out = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(d, ring_out.inputs["Value"])
        ring_out.inputs["From Min"].default_value = 1.35
        ring_out.inputs["From Max"].default_value = 1.05
        ring = nb.math("MULTIPLY", ring_in.outputs[0], ring_out.outputs[0])
        # Bugwellen-Arme: |y| = 0.36 * (15 - x)
        behind_bow = nb.math("SUBTRACT", 16.5, x)
        arm_c = nb.math("MULTIPLY", behind_bow, 0.36)
        arm_d = nb.math("ABSOLUTE", nb.math("SUBTRACT", ay, arm_c))
        arm_w = nb.math("MULTIPLY_ADD", behind_bow, 0.03, 0.7)
        arm = nb.math("SUBTRACT", 1.0, nb.math("DIVIDE", arm_d, arm_w))
        arm = nb.math("MAXIMUM", arm, 0.0)
        arm = nb.math("MULTIPLY", arm, nb.math("MULTIPLY", nb.math("EXPONENT", nb.math("DIVIDE", behind_bow, -32.0)), 0.75))
        arm = nb.math("MULTIPLY", arm, nb.math("GREATER_THAN", behind_bow, 0.0))
        # Kielwasser hinter dem Heck
        u = nb.math("SUBTRACT", -15.0, x)
        upos = nb.math("GREATER_THAN", u, -1.0)
        cw = nb.math("MULTIPLY_ADD", u, 0.07, 3.8)
        center = nb.math("SUBTRACT", 1.0, nb.math("DIVIDE", ay, cw))
        center = nb.math("MAXIMUM", center, 0.0)
        center = nb.math("MULTIPLY", center, nb.math("EXPONENT", nb.math("DIVIDE", nb.math("MAXIMUM", u, 0.0), -38.0)))
        center = nb.math("MULTIPLY", center, upos)
        m = nb.math("MAXIMUM", nb.math("MAXIMUM", ring, arm), center)
        # Aufbrechen mit (driftendem) Rauschen
        n = nb.noise(wp_t, scale=0.55, detail=4, rough=0.65)
        n2 = nb.noise(wp_t, scale=2.5, detail=2, rough=0.6)
        nn = nb.math("ADD", nb.out(n, "Fac"), nb.math("MULTIPLY", nb.out(n2, "Fac"), 0.35))
        th = nb.math("SUBTRACT", 1.02, m)
        mr = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(nb.math("SUBTRACT", nn, th), mr.inputs["Value"])
        mr.inputs["From Min"].default_value = -0.15
        mr.inputs["From Max"].default_value = 0.12
        return nb.math("MULTIPLY", mr.outputs[0], nb.math("MINIMUM", nb.math("MULTIPLY", m, 1.6), 0.85))

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
                    cloud_ref=9.0, aerosol=1.4)
    fpv.add_sun(SUN_ELEV, SUN_AZIM, strength=5.2, color=(1.0, 0.9, 0.78))

    # Schiff
    root, body, _ = sunny.build()
    sunny.animate(root, body, SHIP_HEADING, tuple(ship_start()), SHIP_SPEED, FPS, frames)

    # Ozean
    wmat = ocean.water_material("Sea", deep=(0.003, 0.03, 0.045), shallow=(0.02, 0.10, 0.11),
                                wake_fn=wake_builder(root))
    ocean.animate_time_value(wmat, FPS, frames)
    tile, res = 100.0, args.ocean_res
    x0, y0, nx, ny = -190.0, -10.0, 4, 5  # Kacheln -240..160 / -60..440
    ocean.make_ocean(x0, y0, nx, ny, tile=tile, res=res, wind=9.5, wave_scale=0.9, chop=1.3, fps=FPS,
                     frames=frames, mat=wmat, direction_deg=-30, alignment=0.4, foam_coverage=0.25)
    far = ocean.water_material("FarSea", deep=(0.003, 0.03, 0.045), shallow=(0.02, 0.10, 0.11), far=True)
    ocean.far_plane(x0 - tile / 2 + 1, x0 - tile / 2 + tile * nx - 1, y0 - tile / 2 + 1, y0 - tile / 2 + tile * ny - 1,
                    mat=far, z=-0.05)

    # Vegetation (Baum-Varianten, als Collection-Instanzen)
    leaf = nature.leaf_material("OPLeaves", c1=(0.03, 0.065, 0.012), c2=(0.09, 0.13, 0.025))
    bark = nature.bark_material("OPBark")
    variants = [nature.tree_variant(f"OPTree{i}", leaf, bark, height=h, crown_r=cr, n_clusters=nc, leaves_per=90,
                                    seed=40 + i, leaf_size=0.34)
                for i, (h, cr, nc) in enumerate(((9, 3.4, 22), (6, 2.6, 16), (3.2, 1.8, 10), (12, 4.0, 26)))]
    _, subs = nature.make_tree_collection("OPTrees", variants)
    veg_pts, veg_scale = [], []
    rng = np.random.default_rng(77)

    # Felsen
    rmat = nature.rock_material("SeaRock", c1=(0.085, 0.078, 0.07), c2=(0.27, 0.25, 0.215), c3=(0.21, 0.17, 0.12),
                                wet_line=1.8, moss=(0.07, 0.10, 0.03), moss_amount=0.25, lichen=0.2)
    for (x, y, r, h, seed, taper) in ROCKS:
        ob = fpv.rock_mesh(f"Rock{seed}", radius=r, height=h + 4, seed=seed, detail=5, taper=taper, mat=rmat,
                           base_z=-4.0, lumpy=0.22, strata=1.0)
        ob.location = (x, y, 0)
        ob.rotation_euler = (0, 0, seed * 1.7)
        if h >= 12:
            bpy.context.view_layer.update()
            me = ob.data
            mw = ob.matrix_world
            polys = [(mw @ pl.center, (mw.to_3x3() @ pl.normal).normalized(), pl.area) for pl in me.polygons]
            cand = [(c, n, a) for (c, n, a) in polys if n.z > 0.45 and c.z > max(5.0, 0.3 * h)]
            if cand:
                w = np.array([a * (n.z ** 2) for (_, n, a) in cand])
                w /= w.sum()
                nveg = int(min(160, 4 + r * r * 0.25 + h * 0.8))
                for k in rng.choice(len(cand), size=nveg, p=w):
                    c, n, a = cand[k]
                    veg_pts.append((c.x, c.y, c.z - 0.25))
                    veg_scale.append(rng.uniform(0.55, 1.2))
        # Vegetation oben auf den großen Felsen
        if h > 25:
            cap = fpv.rock_mesh(f"RockCap{seed}", radius=r * 0.72, height=3, seed=seed + 100, detail=4, taper=0.6,
                                mat=None, base_z=h - 3.2, lumpy=0.3)
            cap.location = (x, y, 0)
            cap.data.materials.append(nature.grass_material("CapGrass", c1=(0.04, 0.08, 0.02), c2=(0.12, 0.16, 0.05)))

    nature.scatter_instances("OPVeg", subs, veg_pts, scales=veg_scale, seed=3)
    print("vegetation", len(veg_pts))

    # Inseln am Horizont
    imat = nature.rock_material("IslandRock", c1=(0.08, 0.075, 0.065), c2=(0.22, 0.2, 0.17), c3=(0.15, 0.13, 0.11),
                                wet=False, moss=(0.05, 0.09, 0.025), moss_amount=0.55, scale=8.0, bump=0.4)
    island("IslandNW", -900, 1150, 420, 260, 31, imat)
    island("IslandNE", 700, 1600, 520, 330, 32, imat)
    island("IslandW", -1700, 420, 380, 180, 33, imat)
    island("IslandFar", -300, 2600, 700, 420, 34, imat)

    # Kamera
    pos, quats, info = fpv.fpv_path(ROUTE, SPEED, FPS, frames, look_pitch=-3.0, pitch_follow=0.5,
                                    bank_gain=1.0, max_bank=32, micro=1.0, seed=5)
    fpv.make_camera(pos, quats, fov_deg=92.0)
    print(f"route length {info['total']:.1f} m, used {info['used']:.1f} m")
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
