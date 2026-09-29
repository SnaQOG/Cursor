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
    """Blaues Namek-Gras: dunkle Basis, hellere Spitzen; Variation je Halm (Attribut 'blade_rnd': Helligkeit,
    Tönung Richtung Türkis/Violett) und je Büschel ('clump': satter/blasser, trockene hellere Spitzen)."""
    mat, nb, out = fpv.new_material(name)
    uv = nb.coords("UV")
    v = nb.sep(uv)[1]

    def attr(nm):
        n = nb.node("ShaderNodeAttribute")
        n.attribute_name = nm
        return n.outputs["Fac"]
    rnd, cl = attr("blade_rnd"), attr("clump")
    root = nb.mix(rnd, (0.008, 0.035, 0.13), (0.014, 0.055, 0.19))
    tip_a = nb.mix(rnd, (0.035, 0.16, 0.44), (0.07, 0.25, 0.56))
    tip_b = nb.mix(rnd, (0.05, 0.22, 0.40), (0.10, 0.20, 0.52))       # Büschel: türkisstichig / violettstichig
    tip = nb.mix(cl, tip_a, tip_b)
    dry = nb.math("MULTIPLY", nb.math("POWER", v, 3.0), nb.math("MULTIPLY", nb.math("GREATER_THAN", rnd, 0.82), 0.45))
    tip = nb.mix(dry, tip, (0.30, 0.36, 0.48))                        # vereinzelt trockene, blasse Spitzen
    col = nb.mix(nb.math("POWER", v, 0.8), root, tip)
    col = nb.vmath("SCALE", col, scale=nb.math("ADD", 0.8, nb.math("MULTIPLY", cl, 0.4)))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.5)
    p.inputs["Specular IOR Level"].default_value = 0.35
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
        rbase = (0.16 + 0.008 * height) * (1 - 0.35 * t) * (1 + 0.5 * np.exp(-t * 14))  # schlank (Referenz)
        for k in range(ns):
            a = 2 * np.pi * k / ns
            rr = rbase * (1 + 0.15 * np.cos(lobes * (a - tw)))  # leicht verdrehte Rippen
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
    lump_dirs = rng.normal(0, 1, (9, 3))
    lump_dirs /= np.linalg.norm(lump_dirs, axis=1)[:, None]

    def lump(d):  # wolkig-bauschige Kugel (Anime-Krone): ~9 Beulen
        return float(np.max(lump_dirs @ d))

    for j in range(leaves):
        d = rng.normal(0, 1, 3)
        d /= np.linalg.norm(d)
        rr = crown_r * (0.86 + 0.12 * lump(d) ** 3 + 0.04 * rng.random())
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
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=crown_r * 0.86)
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
                exclude=(), zmin=None, max_blades=2_000_000, mask_fn=None, clump=0.0, height=(0.12, 0.32)):
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
    if mask_fn is not None:
        P = P[mask_fn(P[:, 0], P[:, 1])]
    # Büschel: großräumiges Rauschen dünnt aus (kahle Stellen) und verdichtet (Horste)
    cl = 0.5 + 0.5 * fpv.fbm2(P[:, 0] / 2.2, P[:, 1] / 2.2, 3, seed=77)
    if clump:
        keep = rng.random(len(P)) < np.clip(1.0 - clump + clump * 1.6 * cl, 0, 1) ** 1.5
        P, cl = P[keep], cl[keep]
    Z = height_fn(P[:, 0], P[:, 1])
    if zmin is not None:
        keep = Z > zmin
        P, Z, cl = P[keep], Z[keep], cl[keep]
    if len(P) > max_blades:
        sel = rng.choice(len(P), max_blades, replace=False)
        P, Z, cl = P[sel], Z[sel], cl[sel]
    n = len(P)
    h = rng.uniform(*height, n) * (0.7 + 0.6 * rng.random(n)) * (0.75 + 0.5 * cl)
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
    # Attribute: Spitze (für Wind/Druckwelle), Halmhöhe, Zufall je Halm und Büschelwert (Farbvariation im Shader)
    for nm, vals in (("tip", np.concatenate([np.zeros(2 * n), np.ones(n)])), ("bh", np.tile(h, 3)),
                     ("blade_rnd", np.tile(rng.random(n), 3)), ("clump", np.tile(cl, 3))):
        at = me.attributes.new(nm, "FLOAT", "POINT")
        at.data.foreach_set("value", vals.astype(np.float32))
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


# --------------------------------------------------------------------------
# Bodendetails nach den Referenzen: Sandflecken im blauen Gras, rote Pilze
# --------------------------------------------------------------------------

def sand_material(name="NamekSand"):
    """Beige Sand-/Erdflecken mit weich ausfransendem Rand (Alpha aus radialem UV + Rauschen)."""
    mat, nb, out = fpv.new_material(name)
    uv = nb.coords("UV")
    co = nb.coords("Object")
    u, v, _ = nb.sep(uv)
    du, dv = nb.math("SUBTRACT", u, 0.5), nb.math("SUBTRACT", v, 0.5)
    r = nb.math("MULTIPLY", nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", du, du), nb.math("MULTIPLY", dv, dv))), 2.0)
    n1 = nb.noise(co, scale=0.35, detail=4, rough=0.6)
    rr = nb.math("ADD", r, nb.math("MULTIPLY", nb.math("SUBTRACT", nb.out(n1, "Fac"), 0.5), 0.45))
    al = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(rr, al.inputs["Value"])
    al.inputs["From Min"].default_value = 0.62
    al.inputs["From Max"].default_value = 0.95
    al.inputs["To Min"].default_value = 1.0
    al.inputs["To Max"].default_value = 0.0
    n2 = nb.noise(co, scale=2.5, detail=4, rough=0.65)
    col = nb.mix(nb.out(n2, "Fac"), (0.40, 0.28, 0.18), (0.62, 0.46, 0.31))
    n3 = nb.noise(co, scale=14.0, detail=2)
    col = nb.mix(nb.math("MULTIPLY", nb.out(n3, "Fac"), 0.35), col, (0.30, 0.22, 0.15))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.92)
    nb.link(al.outputs[0], p.inputs["Alpha"])
    nb.link(nb.bump(nb.math("ADD", nb.out(n2, "Fac"), nb.math("MULTIPLY", nb.out(n3, "Fac"), 0.4)), strength=0.5,
                    distance=0.03), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def sand_patches(name, patches, height_fn, mat, n_r=10, n_a=48):
    """patches: Liste (cx, cy, rx, ry, rot). Flache Scheiben, die dem Gelände folgen (+3 cm)."""
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    for (cx, cy, rx, ry, rot) in patches:
        ca, sa = math.cos(rot), math.sin(rot)
        grid = []
        for i in range(n_r + 1):
            f = i / n_r
            ring = []
            for k in range(n_a if i else 1):
                a = 2 * math.pi * k / n_a
                lx, ly = f * rx * math.cos(a), f * ry * math.sin(a)
                x, y = cx + lx * ca - ly * sa, cy + lx * sa + ly * ca
                z = float(height_fn(np.array([x]), np.array([y]))[0]) + 0.03
                ring.append((bm.verts.new((x, y, z)), (0.5 + 0.5 * f * math.cos(a), 0.5 + 0.5 * f * math.sin(a))))
            grid.append(ring)
        for k in range(n_a):
            k2 = (k + 1) % n_a
            q = [grid[0][0], grid[1][k], grid[1][k2]]
            f = bm.faces.new([e[0] for e in q])
            for lp, e in zip(f.loops, q):
                lp[uvl].uv = e[1]
        for i in range(1, n_r):
            for k in range(n_a):
                k2 = (k + 1) % n_a
                q = [grid[i][k], grid[i + 1][k], grid[i + 1][k2], grid[i][k2]]
                f = bm.faces.new([e[0] for e in q])
                for lp, e in zip(f.loops, q):
                    lp[uvl].uv = e[1]
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    for p in ob.data.polygons:
        if p.normal.z < 0:
            p.flip()
    return ob


def mushrooms(name, clusters, height_fn, rng):
    """Kleine rote Pilze (Referenz: Manga-Farbtafel/Anime) in Gruppen zu 3–7."""
    cap = fpv.new_material(name + "Cap")
    mat_c, nb, out = cap
    co = nb.coords("Object")
    sp = nb.voronoi(co, scale=22.0, feature="F1")
    dots = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(sp, "Distance"), dots.inputs["Value"])
    dots.inputs["From Min"].default_value = 0.16
    dots.inputs["From Max"].default_value = 0.10
    col = nb.mix(nb.math("MULTIPLY", dots.outputs[0], 0.8), (0.50, 0.025, 0.02), (0.85, 0.80, 0.72))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.35)
    p.inputs["Coat Weight"].default_value = 0.3
    nb.link(p.outputs[0], out.inputs[0])
    mat_s = fpv.simple_mat(name + "Stem", (0.78, 0.74, 0.64), rough=0.6)
    bm_c, bm_s = bmesh.new(), bmesh.new()
    for (cx, cy) in clusters:
        for _ in range(int(rng.integers(3, 8))):
            x, y = cx + rng.normal(0, 0.28), cy + rng.normal(0, 0.28)
            z = float(height_fn(np.array([x]), np.array([y]))[0]) - 0.02
            h = rng.uniform(0.05, 0.16)
            rs = rng.uniform(0.012, 0.026)
            rc = rng.uniform(0.04, 0.11)
            res = bmesh.ops.create_cone(bm_s, cap_ends=True, segments=8, radius1=rs * 1.2, radius2=rs, depth=h)
            bmesh.ops.translate(bm_s, vec=(x, y, z + h / 2), verts=res["verts"])
            res = bmesh.ops.create_uvsphere(bm_c, u_segments=14, v_segments=7, radius=rc)
            lean = rng.normal(0, 0.12, 2)
            M = Matrix.Translation((x + lean[0] * h, y + lean[1] * h, z + h)) @ Matrix.Diagonal((1, 1, 0.55, 1))
            bmesh.ops.transform(bm_c, verts=res["verts"], matrix=M)
    return fpv.mesh_from_bmesh(bm_c, name + "Caps", mat_c), fpv.mesh_from_bmesh(bm_s, name + "Stems", mat_s)


# --------------------------------------------------------------------------
# Tafelberg-Wand (echtes Mesh statt Heightfield: Schichtbänke, Überhänge, Rinnen)
# --------------------------------------------------------------------------

def ring_wall(name, center, edge_fn, top_fn, mat, n_ang=1500, n_z=150, base_z=-3.0, seed=3, lip=(2.0, 4.0, 6.8)):
    """Umlaufende Steilwand um einen Tafelberg. edge_fn(ang) -> Radius der Kante, top_fn(X, Y) -> Plateauhöhe.
    Radius = Kante + Versatz aus großen Beulen, senkrechten Karstrinnen, sägezahnförmigen Schichtbänken (Bank springt
    oben vor), einer vorspringenden Deckbank (Überhang, stellenweise bis ~2,5 m) und einer Brandungskehle.
    Oben rollt die Wand als Felskante nach innen ein und taucht ~6,5 m hinter der Kante unter das Plateau
    (keine Naht, das Heightfield fällt dahinter ab)."""
    cx, cy = center
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * math.pi, n_ang, endpoint=False)
    E = edge_fn(ang)
    ca, sa = np.cos(ang), np.sin(ang)
    Hr = top_fn(cx + (E - lip[1]) * ca, cy + (E - lip[1]) * sa) + 0.05      # Höhe der Felskante je Winkel
    U = np.linspace(0, 1, n_z)[:, None]
    Z = base_z + U * (Hr[None, :] - base_z)
    arc = (ang * np.mean(E))[None, :] + 0 * Z                                  # Meter entlang des Umfangs
    ph = rng.uniform(0, 1000, 6)
    big = fpv.fbm2(arc / 16 + ph[0], Z / 14, 4, seed=seed)                     # große Beulen
    flutes = fpv.fbm2(arc / 2.4 + ph[1] + Z * 0.03, Z / 8, 4, seed=seed + 5)   # unregelmäßige Rinnen
    mid = fpv.fbm2(arc / 4.5 + ph[2], Z / 3.5, 4, seed=seed + 7)
    fine = fpv.fbm2(arc / 0.9 + ph[3], Z / 0.9, 3, seed=seed + 8)
    d = 1.7 * big + 0.22 * flutes + 0.5 * mid + 0.1 * fine
    # Schichtbänke: Dicke 1,4–3,2 m schwankend, leicht geneigt, Stufentiefe je Bank 45–135 %
    bed_h = 2.2
    u = (Z + 1.4 * bed_h * fpv.fbm2(Z * 0.05 + ph[4], arc * 0.004, 2, seed + 13) + 0.015 * arc * 0.3
         + 0.9 * fpv.fbm2(arc / 7 + ph[5], Z * 0.02, 3, seed + 11)) / bed_h
    k = np.floor(u)
    fu = u - k
    hk = np.sin(k * 12.9898 + seed * 78.233) * 43758.5453
    amp = 0.45 + 0.9 * (hk - np.floor(hk))
    d += 0.55 * amp * (fu ** 3 - 0.25)
    # Deckbank: oberste 2–6 m springen vor (Überhang), Stärke wechselt entlang der Kante
    cap_h = 3.0 + 2.5 * (0.5 + 0.5 * fpv.fbm2(arc / 30 + 7.0, 0 * arc + 1.3, 3, seed + 21))
    zc = (Z - (Hr[None, :] - cap_h)) / np.maximum(cap_h, 0.5)
    cap = np.clip(zc * 3.0, 0, 1)
    cap = cap * cap * (3 - 2 * cap) * np.clip((1.0 - zc) * 6.0, 0, 1)
    cap_amp = 1.2 + 1.3 * (0.5 + 0.5 * fpv.fbm2(arc / 22 + 3.3, 0 * arc + 7.7, 3, seed + 22))
    d += cap * cap_amp
    # Brandungskehle an der Wasserlinie
    d -= 1.3 * np.exp(-((Z - 0.9) / 1.3) ** 2)
    d = np.maximum(d, -1.8)
    R = E[None, :] + d
    # Felskante: abgerundet, dann flach nach innen unter das Plateau
    top_d = d[-1]
    rows_r = [E + top_d - 0.5, E - lip[0], E - lip[1], E - lip[2]]
    rows_z = [Hr + 0.12, Hr + 0.1, Hr, top_fn(cx + (E - lip[2]) * ca, cy + (E - lip[2]) * sa) - 0.35]
    R = np.vstack([R] + [r[None, :] for r in rows_r])
    Z = np.vstack([Z] + [z[None, :] for z in rows_z])
    nz = R.shape[0]
    X = cx + R * ca[None, :]
    Y = cy + R * sa[None, :]
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    idx = np.arange(nz * n_ang).reshape(nz, n_ang)
    a = idx[:-1, :]
    b = np.roll(idx[:-1, :], -1, axis=1)
    c = np.roll(idx[1:, :], -1, axis=1)
    dd = idx[1:, :]
    faces = np.stack([a.ravel(), b.ravel(), c.ravel(), dd.ravel()], axis=1)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(len(faces) * 4)
    me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", (np.arange(len(faces)) * 4).astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    me.update()
    ob = bpy.data.objects.new(name, me)
    me.materials.append(mat)
    fpv.link(ob)
    return ob


# --------------------------------------------------------------------------
# Krater (Form entsteht beim Einschlag, Rand glüht und kühlt ab)
# --------------------------------------------------------------------------

def heat_variant(base_mat, name, f0, fps=24, cool_s=3.5, glow=6.0, scorch_col=(0.035, 0.03, 0.028)):
    """Kopie eines Bodenmaterials mit Brandspuren und Glut: Werte 'Scorch'/'Heat' (Value-Knoten, gekeyt) sind vor
    dem Einschlag 0 -> identisch mit dem Boden ringsum. Maske = Attribut 'heat_mask' (Randzone), Glutfarbe
    Schwarzkörper, beim Abkühlen dunkler und röter."""
    mat = base_mat.copy()
    mat.name = name
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    nb = fpv.NB(nt)
    sc_v = nt.nodes.new("ShaderNodeValue")
    sc_v.name = "Scorch"
    ht_v = nt.nodes.new("ShaderNodeValue")
    ht_v.name = "Heat"
    tp_v = nt.nodes.new("ShaderNodeValue")
    tp_v.name = "Temp"
    mask = nb.node("ShaderNodeAttribute")
    mask.attribute_name = "heat_mask"
    cr = nb.noise(nb.coords("Object"), scale=2.2, detail=6)
    crack = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(cr, "Fac"), crack.inputs["Value"])
    crack.inputs["From Min"].default_value = 0.42
    crack.inputs["From Max"].default_value = 0.62
    # Grundfarbe mit Ruß überblenden
    src = bsdf.inputs["Base Color"]
    col_in = src.links[0].from_socket if src.links else src.default_value[:3]
    sm = nb.math("MULTIPLY", sc_v.outputs[0], nb.math("MINIMUM", nb.math("MULTIPLY", mask.outputs["Fac"], 1.6), 1.0))
    col = nb.mix(sm, col_in, scorch_col)
    nb.link(col, src)
    rough = bsdf.inputs["Roughness"]
    # Glut: Maske * Risse * Hitze, Farbe per Schwarzkörper (Temp)
    bb = nb.node("ShaderNodeBlackbody")
    nb.link(tp_v.outputs[0], bb.inputs["Temperature"])
    em = nb.math("MULTIPLY", nb.math("MULTIPLY", mask.outputs["Fac"], crack.outputs[0]), ht_v.outputs[0])
    nb.link(bb.outputs[0], bsdf.inputs["Emission Color"])
    nb.link(nb.math("MULTIPLY", em, glow), bsdf.inputs["Emission Strength"])
    mat.cycles.emission_sampling = "NONE"
    # Zeitverlauf
    fc = lambda t: int(round(t * fps)) + f0
    for nd, keys in ((sc_v, [(f0 - 1, 0.0), (f0, 1.0)]),
                     (ht_v, [(f0 - 1, 0.0), (f0, 1.0), (fc(0.4), 0.85), (fc(cool_s * 0.5), 0.35), (fc(cool_s), 0.12),
                             (fc(cool_s * 2.5), 0.03)]),
                     (tp_v, [(f0, 2400.0), (fc(0.5), 1700.0), (fc(cool_s * 0.5), 1150.0), (fc(cool_s), 850.0)])):
        for f, v in keys:
            nd.outputs[0].default_value = v
            nd.outputs[0].keyframe_insert("default_value", frame=f)
    act = nt.animation_data.action if nt.animation_data else None
    if act:
        for fcu in _action_fcurves(act):
            for kp in fcu.keyframe_points:
                kp.interpolation = "LINEAR" if kp.co[0] > f0 else "CONSTANT"
    _ = rough
    return mat


def _action_fcurves(act):
    try:
        return list(act.fcurves)
    except AttributeError:
        out = []
        for layer in act.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    out += list(cb.fcurves)
        return out


def crater(name, x, y, r, ground_fn, mat, f0, fps=24, seed=1, depth=0.38, rim=0.2, n_r=34, n_a=96, pad=1.6):
    """Krater als Scheibe (Radius pad*r) auf dem Boden: Grundform = Boden (flach), Formschlüssel 'Blast' = Mulde
    mit aufgeworfenem, zerklüftetem Rand; springt beim Einschlag in 3 Frames auf 1. Attribut 'heat_mask' für
    Glut (Muldenrand + Risse). Der Boden darunter braucht ein Loch (punch_hole)."""
    rng = np.random.default_rng(seed)
    rho = np.linspace(0, pad, n_r)[:, None]
    a = np.linspace(0, 2 * math.pi, n_a, endpoint=False)[None, :]
    X = x + rho * r * np.cos(a)
    Y = y + rho * r * np.sin(a)
    Z0 = ground_fn(X, Y)
    edge = rho >= pad - 1e-6
    Z0 = np.where(edge, Z0 - 0.04, Z0 + 0.0)
    wob = 1 + 0.12 * fpv.fbm2(np.cos(a) * 2 + seed, np.sin(a) * 2, 3, seed=seed) + 0 * rho
    q = rho / wob
    bowl = -depth * r * np.clip(1 - q ** 2, 0, 1) ** 1.3
    lip = rim * r * np.exp(-((q - 1.02) / 0.2) ** 2) * (1 + 0.45 * fpv.fbm2(X * 1.3, Y * 1.3, 3, seed=seed + 3))
    rough = 0.05 * r * fpv.fbm2(X * 2.5, Y * 2.5, 3, seed=seed + 5) * np.exp(-((q - 0.9) / 0.5) ** 2)
    fade = np.clip((pad - rho) / (pad - 1.25), 0, 1)
    Z1 = Z0 + (bowl + lip + rough) * fade
    verts0 = np.stack([X.ravel(), Y.ravel(), Z0.ravel()], axis=1)
    verts1 = np.stack([X.ravel(), Y.ravel(), Z1.ravel()], axis=1)
    idx = np.arange(n_r * n_a).reshape(n_r, n_a)
    quads = np.stack([idx[:-1, :].ravel(), np.roll(idx[:-1, :], -1, 1).ravel(), np.roll(idx[1:, :], -1, 1).ravel(),
                      idx[1:, :].ravel()], axis=1)
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts0.tolist(), [], quads.tolist())
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    me.materials.append(mat)
    fpv.link(ob)
    hm = np.exp(-((q - 1.0) / 0.3) ** 2) * 0.9 + np.clip(1 - q, 0, 1) * 0.5
    at = me.attributes.new("heat_mask", "FLOAT", "POINT")
    at.data.foreach_set("value", np.clip(hm, 0, 1).ravel().astype(np.float32))
    ob.shape_key_add(name="Basis")
    sk = ob.shape_key_add(name="Blast")
    sk.data.foreach_set("co", verts1.astype(np.float32).ravel())
    for f, v in ((f0 - 1, 0.0), (f0, 0.55), (f0 + 2, 1.0)):
        sk.value = v
        sk.keyframe_insert("value", frame=f)
    ob.cycles.use_deform_motion = False
    return ob


def punch_hole(ob, holes):
    """Flächen eines Boden-Meshes entfernen, deren Mitte in einem Kreis (x, y, r) liegt."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    kill = [f for f in bm.faces
            if any(math.hypot(f.calc_center_median().x - x, f.calc_center_median().y - y) < r for (x, y, r) in holes)]
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bm.to_mesh(ob.data)
    bm.free()


# --------------------------------------------------------------------------
# Gras in Bewegung: Wind, Druckwellen, verbrannte Kraterflächen (Geometry Nodes, Szenenzeit)
# --------------------------------------------------------------------------

def grass_motion(ob, shocks=(), burns=(), wind=0.35, gust=(0.9, 0.35), sway=0.28):
    """Halmspitzen (Attribute 'tip', 'bh') per GN verschieben. shocks = [(x, y, t0, tempo m/s, stärke, breite)]:
    Front läuft radial nach außen, drückt die Halme weg und lässt sie abklingend nachschwingen;
    burns = [(x, y, r, t0)]: Halme im Krater verschwinden (Spitze auf die Basis). Wind: 4D-Rauschen, Böen
    wandern mit `gust` m/s über das Feld."""
    ng = bpy.data.node_groups.new(ob.name + "Motion", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    g = fpv.NB(ng)
    gi, go = g.node("NodeGroupInput"), g.node("NodeGroupOutput")
    ts = g.node("GeometryNodeInputSceneTime").outputs["Seconds"]
    pos = g.node("GeometryNodeInputPosition").outputs[0]
    px, py, _ = g.sep(pos)

    def attr(nm):
        n = g.node("GeometryNodeInputNamedAttribute")
        n.data_type = "FLOAT"
        n.inputs["Name"].default_value = nm
        return n.outputs["Attribute"]
    tip, bh = attr("tip"), attr("bh")
    # Wind
    nz = g.node("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value = 1.0
    nz.inputs["Detail"].default_value = 2.0
    wp = g.vmath("ADD", g.vmath("SCALE", pos, scale=0.07), g.comb(g.math("MULTIPLY", ts, -gust[0] * 0.07),
                                                                   g.math("MULTIPLY", ts, -gust[1] * 0.07), 0.0))
    g.link(wp, nz.inputs["Vector"])
    g.link(g.math("MULTIPLY", ts, 0.25), nz.inputs["W"])
    cr, cg, _ = g.sep(nz.outputs["Color"])
    fl = g.node("ShaderNodeTexNoise")
    fl.noise_dimensions = "4D"
    fl.inputs["Scale"].default_value = 1.0
    g.link(g.vmath("SCALE", pos, scale=1.3), fl.inputs["Vector"])
    g.link(g.math("MULTIPLY", ts, 1.6), fl.inputs["W"])
    fr, fg, _ = g.sep(fl.outputs["Color"])
    gustf = g.math("MAXIMUM", g.math("MULTIPLY", g.math("SUBTRACT", cr, 0.42), 3.0), 0.0)
    ox = g.math("ADD", g.math("MULTIPLY", g.math("ADD", 0.35, gustf), wind * gust[0] / 0.97),
                g.math("MULTIPLY", g.math("SUBTRACT", fr, 0.5), sway))
    oy = g.math("ADD", g.math("MULTIPLY", g.math("ADD", 0.35, gustf), wind * gust[1] / 0.97),
                g.math("MULTIPLY", g.math("SUBTRACT", fg, 0.5), sway))
    # Druckwellen
    for (sx, sy, t0, v, amp, w) in shocks:
        dx, dy = g.math("SUBTRACT", px, sx), g.math("SUBTRACT", py, sy)
        dist = g.math("MAXIMUM", g.math("SQRT", g.math("ADD", g.math("MULTIPLY", dx, dx), g.math("MULTIPLY", dy, dy))), 0.3)
        age = g.math("SUBTRACT", ts, t0)
        on = g.math("GREATER_THAN", age, 0.0)
        front = g.math("MULTIPLY", g.math("MAXIMUM", age, 0.0), v)
        rel = g.math("DIVIDE", g.math("SUBTRACT", dist, front), w)
        bump = g.math("EXPONENT", g.math("MULTIPLY", g.math("MULTIPLY", rel, rel), -1.0))
        passed = g.math("MINIMUM", g.math("MAXIMUM", g.math("DIVIDE", g.math("SUBTRACT", front, dist), 2 * w), 0.0), 1.0)
        ring = g.math("MULTIPLY", g.math("SINE", g.math("MULTIPLY", age, 9.0)),
                      g.math("EXPONENT", g.math("MULTIPLY", age, -1.4)))
        after = g.math("MULTIPLY", passed, g.math("ADD", g.math("MULTIPLY", ring, 0.45),
                                                  g.math("MULTIPLY", g.math("EXPONENT", g.math("MULTIPLY", age, -0.8)), 0.3)))
        fall = g.math("DIVIDE", 1.0, g.math("ADD", 1.0, g.math("MULTIPLY", dist, 1.0 / (4 * w + 6.0))))
        s = g.math("MULTIPLY", g.math("MULTIPLY", g.math("ADD", bump, after), on), g.math("MULTIPLY", fall, amp))
        ox = g.math("ADD", ox, g.math("MULTIPLY", g.math("DIVIDE", dx, dist), s))
        oy = g.math("ADD", oy, g.math("MULTIPLY", g.math("DIVIDE", dy, dist), s))
    # Länge begrenzen (max. 0,95 der Halmhöhe), Spitze senkt sich beim Umbiegen
    hl = g.math("SQRT", g.math("ADD", g.math("MULTIPLY", ox, ox), g.math("MULTIPLY", oy, oy)))
    k = g.math("DIVIDE", g.math("MINIMUM", hl, 0.95), g.math("MAXIMUM", hl, 1e-4))
    ox, oy = g.math("MULTIPLY", ox, k), g.math("MULTIPLY", oy, k)
    hl = g.math("MINIMUM", hl, 0.95)
    oz = g.math("SUBTRACT", g.math("SQRT", g.math("SUBTRACT", 1.0, g.math("MULTIPLY", hl, hl))), 1.0)
    # verbrannt: Spitze auf die Basis
    burn = None
    for (bx, by, br, bt) in burns:
        dx, dy = g.math("SUBTRACT", px, bx), g.math("SUBTRACT", py, by)
        inside = g.math("LESS_THAN", g.math("ADD", g.math("MULTIPLY", dx, dx), g.math("MULTIPLY", dy, dy)), br * br)
        b = g.math("MULTIPLY", inside, g.math("GREATER_THAN", ts, bt - 0.02))
        burn = b if burn is None else g.math("MAXIMUM", burn, b)
    scale = bh
    if burn is not None:
        keep = g.math("SUBTRACT", 1.0, burn)
        ox, oy = g.math("MULTIPLY", ox, keep), g.math("MULTIPLY", oy, keep)
        oz = g.math("SUBTRACT", g.math("MULTIPLY", oz, keep), burn)
    off = g.vmath("SCALE", g.comb(ox, oy, oz), scale=scale)
    sp = g.node("GeometryNodeSetPosition")
    g.link(gi.outputs[0], sp.inputs["Geometry"])
    g.link(g.math("GREATER_THAN", tip, 0.5), sp.inputs["Selection"])
    g.link(off, sp.inputs["Offset"])
    g.link(sp.outputs[0], go.inputs[0])
    mod = ob.modifiers.new("GrassMotion", "NODES")
    mod.node_group = ng
    ob.cycles.use_deform_motion = False
    return ob
