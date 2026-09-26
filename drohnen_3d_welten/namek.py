"""Planet Namek (Dragon Ball) – Production-Assets v2.

Referenz: Namekianer-Häuser = weiße, steinartige, längliche Kuppeln mit insektenartigen Wandsegmenten,
gehörnter Kuppel obendrauf, runden Fenstern, zwei Ebenen. Ajisa = ein runder Blattballen auf dünnem Stamm.
Namekianische Dragon Balls ≈ Basketball-Größe. Friezas Schiff: runde Untertasse, kurze Landebeine,
Rand schwarz-gelb gemustert.
"""
import math
import os

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

import fpv
import textures

TPS = lambda f: os.path.join(fpv.ASSETS, "tps", f)


# --------------------------------------------------------------------------
# Materialien
# --------------------------------------------------------------------------

def clay_material(name="NamekClay", color=(0.78, 0.79, 0.74)):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    n1 = nb.noise(co, scale=0.25, detail=3)
    n2 = nb.noise(co, scale=3.0, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(n1, "Fac"), 0.3), color, [c * 0.88 for c in color])
    # grünlicher Schmutz unten, Regenfahnen
    z = nb.sep(wpos)[2]
    x, y, _ = nb.sep(wpos)
    st = nb.noise(nb.comb(nb.math("ADD", x, y), nb.math("MULTIPLY", z, 0.1), 0.0), scale=2.0, detail=2)
    sm = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(st, "Fac"), sm.inputs["Value"])
    sm.inputs["From Min"].default_value = 0.55
    sm.inputs["From Max"].default_value = 0.75
    col = nb.mix(nb.math("MULTIPLY", sm.outputs[0], 0.25), col, (0.45, 0.50, 0.44))
    pt = nb.out(nb.node("ShaderNodeNewGeometry"), "Pointiness")
    cav = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(pt, cav.inputs["Value"])
    cav.inputs["From Min"].default_value = 0.49
    cav.inputs["From Max"].default_value = 0.42
    col = nb.mix(nb.math("MULTIPLY", cav.outputs[0], 0.5), col, (0.25, 0.28, 0.24))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.62)
    p.inputs["Subsurface Weight"].default_value = 0.04
    nb.link(nb.bump(nb.out(n2, "Fac"), strength=0.08, distance=0.02), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def ship_hull_material(name="FriezaHull", color=(0.56, 0.53, 0.48)):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    m = nb.mapping(co, scale=(0.12, 0.12, 0.12))
    nrm = nb.image(TPS("tile_rivet_panels_normal.png"), m, colorspace="Non-Color", proj="BOX", blend=0.2)
    orm = nb.image(TPS("tile_rivet_panels_orm.png"), m, colorspace="Non-Color", proj="BOX", blend=0.2)
    ao, rg, _ = nb.sep(nb.out(orm, "Color"))
    nm = nb.node("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = 1.0
    nb.link(nb.out(nrm, "Color"), nm.inputs["Color"])
    n = nb.noise(co, scale=0.08, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(n, "Fac"), 0.3), color, [c * 0.8 for c in color])
    col = nb.vmath("SCALE", col, scale=nb.math("MULTIPLY_ADD", ao, 0.5, 0.5))
    # Staub am unteren Drittel
    z = nb.sep(co)[2]
    dust = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(z, dust.inputs["Value"])
    dust.inputs["From Min"].default_value = 0.0
    dust.inputs["From Max"].default_value = -0.6
    col = nb.mix(nb.math("MULTIPLY", dust.outputs[0], 0.5), col, (0.30, 0.34, 0.28))
    p = fpv.principled(nb, Base_Color=col, Roughness=nb.math("MULTIPLY_ADD", rg, 0.4, 0.25), Metallic=0.45)
    p.inputs["Coat Weight"].default_value = 0.25
    nb.link(nm.outputs[0], p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def hazard_rim_material(name="FriezaRim", stripes=48):
    """Schwarz-gelb gemusterter Rand (Streifen radial über den Umfangswinkel)."""
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    x, y, _ = nb.sep(co)
    ang = nb.math("ARCTAN2", y, x)
    f = nb.math("FRACT", nb.math("MULTIPLY", ang, stripes / (2 * math.pi)))
    yel = nb.math("LESS_THAN", f, 0.5)
    col = nb.mix(yel, (0.02, 0.02, 0.02), (0.75, 0.55, 0.05))
    n = nb.noise(co, scale=2.0, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(n, "Fac"), 0.25), col, (0.12, 0.10, 0.06))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.45, Metallic=0.3)
    p.inputs["Coat Weight"].default_value = 0.2
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def grass_blade_material(name="NamekBlade"):
    mat, nb, out = fpv.new_material(name)
    uv = nb.coords("UV")
    v = nb.sep(uv)[1]
    wn = nb.node("ShaderNodeTexWhiteNoise")
    wn.noise_dimensions = "3D"
    nb.link(nb.coords("Object"), wn.inputs["Vector"])
    root = nb.mix(nb.out(wn, "Value"), (0.018, 0.07, 0.07), (0.03, 0.10, 0.10))
    tip = nb.mix(nb.out(wn, "Value"), (0.08, 0.28, 0.27), (0.14, 0.40, 0.37))
    col = nb.mix(nb.math("POWER", v, 0.8), root, tip)
    p = fpv.principled(nb, Base_Color=col, Roughness=0.55)
    tr = nb.node("ShaderNodeBsdfTranslucent")
    nb.set_in(tr, "Color", col)
    mix = nb.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.3
    nb.link(p.outputs[0], mix.inputs[1])
    nb.link(tr.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    return mat


# --------------------------------------------------------------------------
# Namekianer-Haus
# --------------------------------------------------------------------------

def _curve(name, pts, radius, mat, res=3, profile=None):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1)
    cu.bevel_depth = radius
    cu.bevel_resolution = res
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    fpv.link(ob)
    return ob


def house(name, x, y, z, r, mats, rng, hz=1.25):
    """Längliche Kuppel (Höhe hz·r) mit Meridian-Rippen und Gurtband, gehörnter Oberkuppel mit großem Rundfenster,
    Rundfenster + Bogentür per Boolean ausgeschnitten, glatte Normalen per Data Transfer."""
    loc = Vector((x, y, z - 0.15 * r))
    # Grundkörper
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.0)
    bmesh.ops.scale(bm, vec=(r, r, r * hz), verts=bm.verts)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -0.12 * r * hz], context="VERTS")
    me = bpy.data.meshes.new(name + "_src")
    bm.to_mesh(me)
    bm.free()
    src = bpy.data.objects.new(name + "_normals", me)  # glatte Normalenquelle
    fpv.link(src)
    src.location = loc
    body = bpy.data.objects.new(name, me.copy())
    fpv.link(body)
    body.location = loc
    body.data.materials.append(mats["clay"])
    for p in body.data.polygons:
        p.use_smooth = True
    sol = body.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.25
    sol.offset = -1
    # Ausschnitte: Fenster (2 Reihen) + Tür
    cutters = bpy.data.collections.new(name + "_cut")
    bpy.context.scene.collection.children.link(cutters)
    frames = []
    n_win = int(rng.integers(5, 8))
    door_a = rng.uniform(0, 2 * math.pi)
    for k in range(n_win):
        a = door_a + (k + 1) * 2 * math.pi / (n_win + 1) + rng.uniform(-0.15, 0.15)
        el = rng.choice([0.28, 0.62])
        nrm = Vector((math.cos(a) * math.cos(el), math.sin(a) * math.cos(el), math.sin(el) / hz)).normalized()
        p = Vector((math.cos(a) * math.cos(el) * r, math.sin(a) * math.cos(el) * r, math.sin(el) * r * hz))
        wr = r * rng.uniform(0.10, 0.14)
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=wr, radius2=wr, depth=r * 0.8)
        c = fpv.mesh_from_bmesh(bm, name + "_wc", None)
        bpy.context.scene.collection.objects.unlink(c)
        cutters.objects.link(c)
        c.rotation_mode = "QUATERNION"
        c.rotation_quaternion = nrm.to_track_quat("Z", "Y")
        c.location = loc + p
        frames.append((loc + p, nrm, wr))
    # Tür (Bogen): Zylinder + Kasten
    dn = Vector((math.cos(door_a), math.sin(door_a), 0))
    dp = loc + dn * r * 0.9 + Vector((0, 0, r * 0.25))
    for (kind, sz) in (("cyl", r * 0.26), ("box", r * 0.26)):
        bm = bmesh.new()
        if kind == "cyl":
            bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=sz, radius2=sz, depth=r * 0.9)
        else:
            bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.scale(bm, vec=(2 * sz, 2 * sz, r * 0.9), verts=bm.verts)
            bmesh.ops.translate(bm, vec=(0, -sz, 0), verts=bm.verts)
        c = fpv.mesh_from_bmesh(bm, name + "_dc", None)
        bpy.context.scene.collection.objects.unlink(c)
        cutters.objects.link(c)
        c.rotation_mode = "QUATERNION"
        c.rotation_quaternion = dn.to_track_quat("Z", "Y")
        c.location = dp
    bo = body.modifiers.new("cut", "BOOLEAN")
    bo.operation = "DIFFERENCE"
    bo.operand_type = "COLLECTION"
    bo.collection = cutters
    try:
        bo.solver = "MANIFOLD"
    except TypeError:
        bo.solver = "EXACT"
    bv = body.modifiers.new("bevel", "BEVEL")
    bv.limit_method = "ANGLE"
    bv.angle_limit = math.radians(40)
    bv.width = 0.035
    bv.segments = 2
    fpv.apply_modifiers(body)
    # glatte Normalen von der ungeschnittenen Kuppel übernehmen (nahtlos glatt um die Löcher) und einbacken
    dt = body.modifiers.new("normals", "DATA_TRANSFER")
    dt.object = src
    dt.use_loop_data = True
    dt.data_types_loops = {"CUSTOM_NORMAL"}
    dt.loop_mapping = "POLYINTERP_NEAREST"
    dt.mix_factor = 0.85
    fpv.apply_modifiers(body)
    for c in list(cutters.objects):
        bpy.data.objects.remove(c)
    bpy.data.collections.remove(cutters)
    bpy.data.objects.remove(src)
    objs = [body]
    # Glas + Rahmen in den Öffnungen
    for (p, nrm, wr) in frames:
        bm = bmesh.new()
        bmesh.ops.create_circle(bm, cap_ends=True, segments=32, radius=wr)
        g = fpv.mesh_from_bmesh(bm, name + "_glass", mats["glass"])
        g.rotation_mode = "QUATERNION"
        g.rotation_quaternion = nrm.to_track_quat("Z", "Y")
        g.location = p - nrm * 0.12
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=False, segments=32, radius1=wr * 1.18, radius2=wr * 1.05, depth=0.18)
        fr = fpv.mesh_from_bmesh(bm, name + "_frame", mats["clay_dark"])
        sol = fr.modifiers.new("s", "SOLIDIFY")
        sol.thickness = 0.06
        fr.rotation_mode = "QUATERNION"
        fr.rotation_quaternion = nrm.to_track_quat("Z", "Y")
        fr.location = p + nrm * 0.05
        objs += [g, fr]
    door = fpv.mesh_from_bmesh(_disc_bm(r * 0.25), name + "_door", mats["door"])
    door.rotation_mode = "QUATERNION"
    door.rotation_quaternion = dn.to_track_quat("Z", "Y")
    door.location = dp - dn * 0.25
    door.scale = (1, 1.6, 1)
    objs.append(door)
    # Meridian-Rippen + Gurtband (insektenartige Segmente)
    n_rib = 9
    for k in range(n_rib):
        a = k * 2 * math.pi / n_rib + 0.2
        if abs(((a - door_a + math.pi) % (2 * math.pi)) - math.pi) < 0.35:
            continue
        pts = []
        for t in np.linspace(0.06, 0.93, 24):
            el = t * math.pi / 2
            pts.append(tuple(loc + Vector((math.cos(a) * math.cos(el) * r * 1.02, math.sin(a) * math.cos(el) * r * 1.02,
                                           math.sin(el) * r * hz * 1.01))))
        objs.append(_curve(name + "_rib", pts, r * 0.045, mats["clay"], res=2))
    ring_el = 0.46
    pts = [tuple(loc + Vector((math.cos(a) * math.cos(ring_el) * r * 1.02, math.sin(a) * math.cos(ring_el) * r * 1.02,
                               math.sin(ring_el) * r * hz))) for a in np.linspace(0, 2 * math.pi, 73)]
    objs.append(_curve(name + "_belt", pts, r * 0.035, mats["clay"], res=2))
    # Oberkuppel (2. Ebene) mit großem Rundfenster + Hörner
    top = loc + Vector((0, 0, r * hz * 0.97))
    cup = fpv.mesh_from_bmesh(_sphere_bm(r * 0.32), name + "_cupola", mats["clay"])
    cup.location = top
    cup.scale = (1, 1, 0.85)
    objs.append(cup)
    wa = door_a + math.pi * 0.5
    wn_ = Vector((math.cos(wa), math.sin(wa), 0.25)).normalized()
    big = fpv.mesh_from_bmesh(_disc_bm(r * 0.15), name + "_bigwin", mats["glass"])
    big.rotation_mode = "QUATERNION"
    big.rotation_quaternion = wn_.to_track_quat("Z", "Y")
    big.location = top + wn_ * r * 0.3
    objs.append(big)
    for k in range(4):
        a = k * math.pi / 2 + 0.4
        base = top + Vector((math.cos(a) * r * 0.2, math.sin(a) * r * 0.2, r * 0.18))
        pts = [tuple(base + Vector((math.cos(a) * r * 0.25 * t * t, math.sin(a) * r * 0.25 * t * t, r * 0.55 * t)))
               for t in np.linspace(0, 1, 10)]
        horn = _curve(name + "_horn", pts, r * 0.07, mats["clay_dark"], res=3)
        horn.data.bevel_mode = "ROUND"
        # Verjüngung zur Spitze
        for i, sp in enumerate(horn.data.splines[0].points):
            sp.radius = 1.0 - 0.9 * i / 9
        objs.append(horn)
    return objs


def _sphere_bm(r):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=20, radius=r)
    return bm


def _disc_bm(r):
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, segments=40, radius=r)
    return bm


# --------------------------------------------------------------------------
# Ajisa-Baum (verdrehter dünner Stamm, perfekte Kugelkrone)
# --------------------------------------------------------------------------

def ajisa_variant(name, leaf_mat, bark_mat, height=8.0, crown_r=3.2, leaves=2600, seed=0, leaf_size=0.34,
                  turns=1.6):
    rng = np.random.default_rng(seed)
    n = 40
    ts = np.linspace(0, 1, n)
    lobes = 5
    ns = 30
    verts, faces = [], []
    lean = rng.normal(0, 0.03, 2) * height
    for i, t in enumerate(ts):
        c = np.array([lean[0] * t ** 2 + 0.08 * np.sin(t * 9 + seed), lean[1] * t ** 2 + 0.08 * np.cos(t * 7 + seed),
                      height * t])
        tw = t * turns * 2 * np.pi
        rbase = height * (0.03 - 0.013 * t) * (1 + 0.6 * np.exp(-t * 14))
        for k in range(ns):
            a = 2 * np.pi * k / ns
            rr = rbase * (1 + 0.28 * np.cos(lobes * (a - tw)))  # verdrehte Rippen
            verts.append(tuple(c + np.array([np.cos(a) * rr, np.sin(a) * rr, 0])))
    for i in range(n - 1):
        for k in range(ns):
            k2 = (k + 1) % ns
            faces.append((i * ns + k, i * ns + k2, (i + 1) * ns + k2, (i + 1) * ns + k))
    tme = bpy.data.meshes.new(name + "_trunk")
    tme.from_pydata(verts, [], faces)
    for p in tme.polygons:
        p.use_smooth = True
    tme.materials.append(bark_mat)
    top = np.array([lean[0], lean[1], height])
    c0 = top + np.array([0, 0, crown_r * 0.85])
    lv, lf = [], []
    for j in range(leaves):
        d = rng.normal(0, 1, 3)
        d /= np.linalg.norm(d)
        rr = crown_r * (0.94 + 0.06 * rng.random())  # nahezu perfekte Kugel
        p = c0 + d * rr
        nrm = d + rng.normal(0, 0.45, 3)
        nrm /= np.linalg.norm(nrm)
        t1 = np.cross(nrm, [0, 0, 1.0])
        if np.linalg.norm(t1) < 1e-3:
            t1 = np.array([1.0, 0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(nrm, t1)
        s = leaf_size * (0.7 + 0.6 * rng.random())
        i0 = len(lv)
        lv += [tuple(p + t1 * s), tuple(p + t2 * s * 0.6), tuple(p - t1 * s), tuple(p - t2 * s * 0.6)]
        lf.append((i0, i0 + 1, i0 + 2, i0 + 3))
    lme = bpy.data.meshes.new(name + "_leaves")
    lme.from_pydata(lv, [], lf)
    lme.materials.append(leaf_mat)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=crown_r * 0.93)
    bmesh.ops.translate(bm, vec=tuple(c0), verts=bm.verts)
    cme = bpy.data.meshes.new(name + "_core")
    bm.to_mesh(cme)
    bm.free()
    for p in cme.polygons:
        p.use_smooth = True
    cme.materials.append(leaf_mat)
    return (bpy.data.objects.new(name, tme), bpy.data.objects.new(name + "_l", lme),
            bpy.data.objects.new(name + "_c", cme))


# --------------------------------------------------------------------------
# Gras (echte Halme, Büschel, Dichte fällt mit Abstand zur Flugbahn)
# --------------------------------------------------------------------------

def grass_field(name, height_fn, path_xy, mat, rng, band=30.0, dens=(360.0, 170.0, 8.0), ymin=None, ymax=None,
                exclude=(), zmin=None, max_blades=2_000_000):
    """Halme entlang der Route; Dichte fällt stetig mit dem Abstand (dens = nah-Zusatz, Grund, Abfall-Länge),
    weicher Rand bei `band` – keine sichtbaren Dichte-Stufen (dichtes Gras wirkt dunkler)."""
    pts = []
    seg = path_xy
    d_near, d_base, d_len = dens
    dmax = d_near + d_base
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
    step = 2.0
    for s0 in np.arange(0, L[-1], step):
        i = np.searchsorted(L, s0) - 1
        i = max(0, min(i, len(seg) - 2))
        p = seg[i] + (seg[i + 1] - seg[i]) * ((s0 - L[i]) / max(L[i + 1] - L[i], 1e-6))
        if ymin is not None and (p[1] < ymin or p[1] > ymax):
            continue
        tng = seg[i + 1] - seg[i]
        tng /= np.linalg.norm(tng) + 1e-9
        nrm = np.array([-tng[1], tng[0]])
        k = int(step * band * 2 * dmax)
        off = rng.uniform(0, band, k)
        edge = np.clip((band - off) / 5.0, 0, 1)
        dens_o = (d_base + d_near * np.exp(-off / d_len)) * edge * edge * (3 - 2 * edge)
        keep = rng.random(k) < dens_o / dmax
        off = off[keep] * rng.choice([-1, 1], keep.sum())
        along = rng.uniform(0, step, len(off))
        pts.append(p[None, :] + nrm[None, :] * off[:, None] + tng[None, :] * along[:, None])
    P = np.concatenate(pts, 0)
    for (ex, ey, er) in exclude:
        P = P[np.hypot(P[:, 0] - ex, P[:, 1] - ey) > er]
    Z = height_fn(P[:, 0], P[:, 1])
    if zmin is not None:
        keep = Z > zmin
        P, Z = P[keep], Z[keep]
    if len(P) > max_blades:
        sel = rng.choice(len(P), max_blades, replace=False)
        P, Z = P[sel], Z[sel]
    n = len(P)
    h = rng.uniform(0.12, 0.32, n) * (0.7 + 0.6 * rng.random(n))
    a = rng.uniform(0, 2 * np.pi, n)
    lean = rng.normal(0, 0.3, (n, 2)) * h[:, None]
    w = 0.008 + 0.006 * rng.random(n)
    base = np.stack([P[:, 0], P[:, 1], Z - 0.02], 1)
    side = np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1) * w[:, None]
    tip = base + np.stack([lean[:, 0], lean[:, 1], h], 1)
    verts = np.concatenate([base - side, base + side, tip], 0)
    faces = np.stack([np.arange(n), np.arange(n) + n, np.arange(n) + 2 * n], 1)
    me = bpy.data.meshes.new(name)
    me.vertices.add(3 * n)
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(3 * n)
    me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(n)
    me.polygons.foreach_set("loop_start", (np.arange(n) * 3).astype(np.int32))
    uvl = me.uv_layers.new(name="UVMap")
    luv = np.zeros((n, 3, 2), np.float32)
    luv[:, 2, 1] = 1.0
    luv[:, 1, 0] = 1.0
    uvl.data.foreach_set("uv", luv.ravel())
    me.update()
    ob = bpy.data.objects.new(name, me)
    me.materials.append(mat)
    fpv.link(ob)
    print("grass blades", n)
    return ob


# --------------------------------------------------------------------------
# Dragon Balls (Ø ~0,36 m)
# --------------------------------------------------------------------------

def dragon_balls(center, height_fn, rng, radius=0.18, ring=0.75, light_w=6.0, lift=0.0):
    objs = []
    for n in range(1, 8):
        path = os.path.join(textures.OUT, f"db_star{n}.png")
        textures.star_map(n, path)
        mat, nb, out = fpv.new_material(f"DragonBall{n}")
        glass = nb.node("ShaderNodeBsdfGlass")
        glass.inputs["Color"].default_value = (1.0, 0.52, 0.10, 1)
        glass.inputs["Roughness"].default_value = 0.02
        glass.inputs["IOR"].default_value = 1.5
        em = nb.node("ShaderNodeEmission")
        em.inputs["Color"].default_value = (1.0, 0.42, 0.06, 1)
        em.inputs["Strength"].default_value = 0.6
        add = nb.node("ShaderNodeAddShader")
        nb.link(glass.outputs[0], add.inputs[0])
        nb.link(em.outputs[0], add.inputs[1])
        nb.link(add.outputs[0], out.inputs[0])
        smat, snb, sout = fpv.new_material(f"DBStars{n}")
        img = snb.image(path, snb.coords("UV"))
        p = fpv.principled(snb, Base_Color=(0.85, 0.05, 0.03), Roughness=0.4)
        p.inputs["Emission Color"].default_value = (1.0, 0.08, 0.03, 1)
        p.inputs["Emission Strength"].default_value = 2.0
        snb.link(snb.out(img, "Alpha"), p.inputs["Alpha"])
        snb.link(p.outputs[0], sout.inputs[0])
        a = n * 2 * math.pi / 7 + 0.3
        x, y = center[0] + math.cos(a) * ring * (0.6 + 0.4 * (n % 2)), center[1] + math.sin(a) * ring
        z = (lift if lift else float(height_fn(np.array([x]), np.array([y]))[0])) + radius * 0.92
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=radius)
        ball = fpv.mesh_from_bmesh(bm, f"DB{n}", mat)
        ball.location = (x, y, z)
        ball.visible_shadow = False
        bm = bmesh.new()
        uvl = bm.loops.layers.uv.new("UVMap")
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=radius * 0.55)
        for f in bm.faces:
            for lp in f.loops:
                co = lp.vert.co.normalized()
                lp[uvl].uv = (0.5 + math.atan2(co.y, co.x) / (2 * math.pi),
                              0.5 + math.asin(max(-1, min(1, co.z))) / math.pi)
        st = fpv.mesh_from_bmesh(bm, f"DBStars{n}", smat)
        st.location = (x, y, z)
        st.rotation_euler = (0, 0, rng.uniform(-0.4, 0.4))
        st.visible_shadow = False
        ld = bpy.data.lights.new(f"DBGlow{n}", "POINT")
        ld.energy = light_w
        ld.color = (1.0, 0.55, 0.15)
        ld.shadow_soft_size = radius
        lo = bpy.data.objects.new(f"DBGlow{n}", ld)
        lo.location = (x, y, z)
        fpv.link(lo)
        objs += [ball, st, lo]
    return objs


# --------------------------------------------------------------------------
# Friezas Raumschiff
# --------------------------------------------------------------------------

def spaceship(mats, cx, cy, z0, R=16.0):
    objs = []
    root = bpy.data.objects.new("FriezaShip", None)
    fpv.link(root)
    root.location = (cx, cy, z0)

    def add(ob):
        ob.parent = root
        objs.append(ob)
        return ob

    body = add(fpv.mesh_from_bmesh(_sphere_bm(1.0), "ShipBody", mats["hull"]))
    body.location = (0, 0, 7.5)
    body.scale = (R, R, 6.8)
    for p in body.data.polygons:
        p.use_smooth = True
    # Ringwulst (Torus) mit Warnstreifen
    bm = bmesh.new()
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=128, radius1=R * 1.04, radius2=R * 1.04, depth=1.2)
    rim = add(fpv.mesh_from_bmesh(bm, "ShipRim", mats["rim"]))
    rim.location = (0, 0, 7.5)
    bv = rim.modifiers.new("b", "BEVEL")
    bv.width = 0.4
    bv.segments = 4
    # Fensterband
    for k in range(32):
        a = k * 2 * math.pi / 32
        nrm = Vector((math.cos(a), math.sin(a), 0.25)).normalized()
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.8, radius2=0.8, depth=0.4)
        w = add(fpv.mesh_from_bmesh(bm, "ShipWin", mats["shipwin"]))
        w.rotation_mode = "QUATERNION"
        w.rotation_quaternion = nrm.to_track_quat("Z", "Y")
        w.location = Vector((math.cos(a) * R * 0.955, math.sin(a) * R * 0.955, 9.3))
    # Oberkuppel + Unterseite dunkler
    dome = add(fpv.mesh_from_bmesh(_sphere_bm(1.0), "ShipDome", mats["hull_dark"]))
    dome.location = (0, 0, 13.6)
    dome.scale = (4.8, 4.8, 2.3)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=96, radius1=R * 0.78, radius2=R * 0.55, depth=1.6)
    under = add(fpv.mesh_from_bmesh(bm, "ShipUnder", mats["hull_dark"]))
    under.location = (0, 0, 2.2)
    # Landebeine (6, kurz, mit Füßen)
    for k in range(6):
        a = k * 2 * math.pi / 6 + 0.25
        p0 = Vector((math.cos(a) * R * 0.5, math.sin(a) * R * 0.5, 3.4))
        p1 = Vector((math.cos(a) * R * 0.66, math.sin(a) * R * 0.66, -0.3))
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=0.55, radius2=0.42, depth=(p1 - p0).length)
        leg = add(fpv.mesh_from_bmesh(bm, "ShipLeg", mats["hull_dark"]))
        leg.rotation_mode = "QUATERNION"
        leg.rotation_quaternion = (p1 - p0).normalized().to_track_quat("Z", "Y")
        leg.location = (p0 + p1) / 2
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=1.5, radius2=1.1, depth=0.45)
        foot = add(fpv.mesh_from_bmesh(bm, "ShipFoot", mats["rim"]))
        foot.location = p1
    return objs
