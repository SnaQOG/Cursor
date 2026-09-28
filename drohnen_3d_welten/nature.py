"""Natur-Materialien und -Objekte: Fels, Gras, Bäume, Inseln."""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector

import fpv


def rock_material(name="Rock", c1=(0.085, 0.078, 0.07), c2=(0.30, 0.27, 0.23), c3=(0.18, 0.16, 0.13),
                  wet_line=1.6, algae=(0.035, 0.05, 0.022), strata_scale=1.3, bump=0.9, moss=None,
                  moss_amount=0.0, scale=1.0, wet=True, lichen=0.15, crack_w=0.12, cavity=0.0, moss_tex=None,
                  moss_tex_scale=5.0, variation=0.0, bedding=0.0, bed_h=1.4, streak=0.55):
    """variation: Streuung pro Objekt (Object Info Random) für Helligkeit, Warm/Kalt-Tönung, Rauschversatz und
    Höhe der Nässelinie; bedding: Stärke dunkler, waagerechter Schichtfugen (Welt-Z, Abstand ~bed_h m);
    streak: Deckkraft der senkrechten Regen-/Sinterstreifen."""
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    z = nb.sep(wpos)[2]
    rnd = rnd2 = None
    if variation:
        rnd = nb.out(nb.node("ShaderNodeObjectInfo"), "Random")
        rnd2 = nb.math("FRACT", nb.math("MULTIPLY", rnd, 7.31))
        co = nb.vmath("ADD", co, nb.comb(nb.math("MULTIPLY", rnd, 37.0), nb.math("MULTIPLY", rnd, 53.0),
                                         nb.math("MULTIPLY", rnd2, 11.0)))
        z = nb.math("ADD", z, nb.math("MULTIPLY", nb.math("SUBTRACT", rnd2, 0.5), 0.6))
    n_big = nb.noise(co, scale=0.035 / scale, detail=3, rough=0.6)
    n_mid = nb.noise(co, scale=0.35 / scale, detail=4, rough=0.65)
    n_fine = nb.noise(co, scale=4.0 / scale, detail=3, rough=0.7)
    # Schichtung (Strata) leicht verzerrt
    wv = nb.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Z", wave_profile="SIN")
    nb.link(nb.vmath("ADD", co, nb.vmath("SCALE", nb.out(n_big, "Color"), scale=6.0)), wv.inputs["Vector"])
    wv.inputs["Scale"].default_value = strata_scale / scale * 0.1
    wv.inputs["Distortion"].default_value = 3.0
    wv.inputs["Detail"].default_value = 1.5
    col = nb.mix(nb.out(n_big, "Fac"), c1, c2)
    col = nb.mix(nb.math("MULTIPLY", nb.out(wv, "Fac"), 0.55), col, c3)
    col = nb.mix(nb.math("MULTIPLY", nb.math("SUBTRACT", nb.out(n_mid, "Fac"), 0.3), 0.8), col, [c * 0.55 for c in c1])
    # Risse (Voronoi-Kanten) dunkel
    vor = nb.voronoi(co, scale=0.6 / scale, feature="DISTANCE_TO_EDGE")
    crack = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(vor, "Distance"), crack.inputs["Value"])
    crack.inputs["From Min"].default_value = 0.0
    crack.inputs["From Max"].default_value = 0.025
    col = nb.mix(nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, crack.outputs[0]), crack_w), col, [c * 0.6 for c in c1])
    # senkrechte Regen-/Sinterstreifen
    stv = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 0.06)), scale=0.9 / scale, detail=2, rough=0.6)
    stm = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(stv, "Fac"), stm.inputs["Value"])
    stm.inputs["From Min"].default_value = 0.5
    stm.inputs["From Max"].default_value = 0.72
    col = nb.mix(nb.math("MULTIPLY", stm.outputs[0], streak), col, [c * 0.45 for c in c1])
    if bedding:
        # Schichtfugen: dünne dunkle Linien in Welt-Z, Dicke/Lage mit großem Rauschen verzogen
        bz = nb.math("DIVIDE", nb.math("ADD", z, nb.math("MULTIPLY", nb.out(n_big, "Fac"), bed_h * 1.4)), bed_h)
        bf = nb.math("FRACT", bz)
        seam = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(nb.math("MINIMUM", bf, nb.math("SUBTRACT", 1.0, bf)), seam.inputs["Value"])
        seam.inputs["From Min"].default_value = 0.07
        seam.inputs["From Max"].default_value = 0.0
        gap = nb.math("MULTIPLY", seam.outputs[0], nb.math("MULTIPLY_ADD", nb.out(n_mid, "Fac"), 0.8, 0.4))
        col = nb.mix(nb.math("MULTIPLY", gap, bedding), col, [c * 0.35 for c in c1])
        # jede zweite Bank minimal heller/wärmer
        alt = nb.math("MULTIPLY", nb.math("FLOOR", nb.math("MULTIPLY", nb.math("FRACT", nb.math("MULTIPLY", bz, 0.5)), 2.0)), 0.18)
        col = nb.mix(nb.math("MULTIPLY", alt, bedding), col, [min(1.0, c * 1.25) for c in c3])
    if lichen:
        lic = nb.noise(co, scale=1.6 / scale, detail=2, rough=0.6)
        lm = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(nb.out(lic, "Fac"), lm.inputs["Value"])
        lm.inputs["From Min"].default_value = 0.62
        lm.inputs["From Max"].default_value = 0.7
        col = nb.mix(nb.math("MULTIPLY", lm.outputs[0], lichen * 3), col, (0.42, 0.40, 0.33))
    if variation:
        v = variation
        val = nb.math("MULTIPLY_ADD", nb.math("SUBTRACT", rnd, 0.5), v, 1.0)
        warm = nb.math("MULTIPLY", nb.math("SUBTRACT", rnd2, 0.5), v)
        tint = nb.comb(nb.math("MULTIPLY", val, nb.math("MULTIPLY_ADD", warm, 0.5, 1.0)), val,
                       nb.math("MULTIPLY", val, nb.math("MULTIPLY_ADD", warm, -0.7, 1.0)))
        col = nb.vmath("MULTIPLY", col, tint)
    rough = nb.math("MULTIPLY_ADD", nb.out(n_fine, "Fac"), 0.2, 0.78)
    moss_mask = None
    if cavity:
        # Vertiefungen (Augenhöhlen, Falten) dunkler, Kanten heller – über Pointiness
        pt = nb.out(nb.node("ShaderNodeNewGeometry"), "Pointiness")
        cv = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(pt, cv.inputs["Value"])
        cv.inputs["From Min"].default_value = 0.47
        cv.inputs["From Max"].default_value = 0.53
        cv.inputs["To Min"].default_value = 1.0 - cavity
        cv.inputs["To Max"].default_value = 1.12
        col = nb.vmath("SCALE", col, scale=cv.outputs[0])
    if moss is not None and moss_amount > 0:
        # Moos/Gras auf nach oben zeigenden Flächen
        nrm = nb.out(nb.node("ShaderNodeNewGeometry"), "Normal")
        up = nb.sep(nrm)[2]
        mm = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(nb.math("ADD", up, nb.math("MULTIPLY", nb.math("SUBTRACT", nb.out(n_mid, "Fac"), 0.5), 0.6)), mm.inputs["Value"])
        mm.inputs["From Min"].default_value = 1.0 - moss_amount
        mm.inputs["From Max"].default_value = 1.0 - moss_amount + 0.12
        mcol = nb.mix(nb.out(n_mid, "Fac"), moss, [c * 0.55 for c in moss])
        mcol = nb.mix(nb.math("MULTIPLY", nb.out(n_big, "Fac"), 0.6), mcol, [min(1.0, c * 1.35) for c in moss])
        mcol = nb.mix(nb.math("MULTIPLY", nb.out(n_fine, "Fac"), 0.3), mcol, [c * 0.7 for c in moss])
        if moss_tex:
            # Foto-Grastextur (three.js grasslight-big) in Weltkoordinaten, auf Moos-Farbe getönt
            gm = nb.mapping(wpos, scale=(1.0 / moss_tex_scale,) * 3)
            gi = nb.image(moss_tex, gm)
            gy = nb.math("MULTIPLY", nb.sep(nb.out(gi, "Color"))[1], 2.4)
            mcol = nb.vmath("MULTIPLY", mcol, nb.comb(gy, gy, gy))
        col = nb.mix(mm.outputs[0], col, mcol)
        moss_mask = mm.outputs[0]
    if wet:
        wl = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(nb.math("ADD", z, nb.math("MULTIPLY", nb.out(n_mid, "Fac"), 1.2)), wl.inputs["Value"])
        wl.inputs["From Min"].default_value = wet_line - 0.2
        wl.inputs["From Max"].default_value = wet_line + 0.8
        dry = wl.outputs[0]
        al = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(nb.math("ADD", z, nb.math("MULTIPLY", nb.out(n_fine, "Fac"), 0.6)), al.inputs["Value"])
        al.inputs["From Min"].default_value = wet_line * 0.55
        al.inputs["From Max"].default_value = wet_line * 0.55 + 0.35
        wetcol = nb.mix(al.outputs[0], algae, nb.vmath("SCALE", col, scale=0.45))
        col = nb.mix(dry, wetcol, col)
        rough = nb.mix(dry, 0.22, rough, dtype="FLOAT")
    p = fpv.principled(nb, Base_Color=col, Roughness=rough)
    h = nb.math("ADD", nb.math("MULTIPLY", nb.out(n_mid, "Fac"), 1.0), nb.math("MULTIPLY", nb.out(n_fine, "Fac"), 0.25))
    h = nb.math("ADD", h, nb.math("MULTIPLY", crack.outputs[0], 0.3 * crack_w))
    if moss_mask is not None:
        h = nb.math("MULTIPLY", h, nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", moss_mask, 0.75)))
    nb.link(nb.bump(h, strength=bump, distance=0.3 * scale), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def grass_material(name="Grass", c1=(0.05, 0.09, 0.02), c2=(0.14, 0.18, 0.05), dry=(0.25, 0.22, 0.10), scale=1.0):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    n1 = nb.noise(co, scale=0.05 / scale, detail=3)
    n2 = nb.noise(co, scale=0.8 / scale, detail=3)
    n3 = nb.noise(co, scale=30 / scale, detail=2)
    col = nb.mix(nb.out(n1, "Fac"), c1, c2)
    col = nb.mix(nb.math("MULTIPLY", nb.math("SUBTRACT", nb.out(n2, "Fac"), 0.45), 1.5), col, dry)
    col = nb.mix(nb.math("MULTIPLY", nb.out(n3, "Fac"), 0.4), col, nb.vmath("SCALE", col, scale=0.6))
    # Erd-/Trockenflecken
    n4 = nb.noise(co, scale=0.02 / scale, detail=3)
    dm = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(n4, "Fac"), dm.inputs["Value"])
    dm.inputs["From Min"].default_value = 0.58
    dm.inputs["From Max"].default_value = 0.72
    col = nb.mix(nb.math("MULTIPLY", dm.outputs[0], 0.7), col, (0.16, 0.13, 0.08))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.85)
    p.inputs["Sheen Weight"].default_value = 0.15
    p.inputs["Sheen Tint"].default_value = (0.8, 1.0, 0.6, 1)
    nb.link(nb.bump(nb.out(n3, "Fac"), strength=0.6, distance=0.05), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def leaf_material(name="Leaves", c1=(0.035, 0.07, 0.015), c2=(0.10, 0.14, 0.03), trans=0.35):
    mat, nb, out = fpv.new_material(name)
    oi = nb.node("ShaderNodeObjectInfo")
    rnd = nb.out(oi, "Random")
    co = nb.coords("Object")
    n = nb.noise(co, scale=0.5, detail=1)
    col = nb.mix(nb.math("ADD", nb.math("MULTIPLY", rnd, 0.6), nb.math("MULTIPLY", nb.out(n, "Fac"), 0.6)), c1, c2)
    p = fpv.principled(nb, Base_Color=col, Roughness=0.6)
    p.inputs["Specular IOR Level"].default_value = 0.35
    tr = nb.node("ShaderNodeBsdfTranslucent")
    nb.set_in(tr, "Color", nb.vmath("SCALE", col, scale=1.3))
    mix = nb.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = trans
    nb.link(p.outputs[0], mix.inputs[1])
    nb.link(tr.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    return mat


def bark_material(name="Bark", c=(0.09, 0.07, 0.05)):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    n = nb.noise(nb.mapping(co, scale=(6, 6, 0.8)), scale=3, detail=3, rough=0.7)
    col = nb.mix(nb.out(n, "Fac"), [x * 0.5 for x in c], c)
    p = fpv.principled(nb, Base_Color=col, Roughness=0.9)
    nb.link(nb.bump(nb.out(n, "Fac"), strength=0.8, distance=0.05), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def terrain(name, size_x, size_y, nx, ny, hfn, mat, origin=(0, 0)):
    return fpv.grid_mesh(name, size_x, size_y, nx, ny, hfn, mat, origin=origin)


def tree_variant(name, leaf_mat, bark_mat, height=12.0, crown_r=4.0, n_clusters=26, leaves_per=110,
                 seed=0, shape="round", leaf_size=0.32):
    """Laubbaum: Stamm + Äste + Blattcluster aus kleinen Rauten (ohne Alpha)."""
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    # Stamm
    trunk_h = height * 0.45
    res = bmesh.ops.create_cone(bm, cap_ends=False, segments=10, radius1=height * 0.035, radius2=height * 0.02,
                                depth=trunk_h + 1.0)
    bmesh.ops.translate(bm, verts=res["verts"], vec=(0, 0, trunk_h / 2 - 0.5))
    trunk_me = bpy.data.meshes.new(name + "_trunk")
    bm.to_mesh(trunk_me)
    bm.free()
    # Blätter
    verts, faces = [], []
    cz = trunk_h + (height - trunk_h) * 0.5
    centers = []
    for k in range(n_clusters):
        if shape == "round":
            d = rng.normal(0, 1, 3)
            d /= np.linalg.norm(d)
            r = crown_r * (0.35 + 0.6 * rng.random() ** 0.5)
            c = np.array([d[0] * r, d[1] * r, cz + d[2] * r * 0.75])
        else:  # conical
            zz = rng.random()
            rr = crown_r * (1 - zz) * rng.random() ** 0.5
            a = rng.uniform(0, 2 * np.pi)
            c = np.array([rr * np.cos(a), rr * np.sin(a), trunk_h * 0.6 + zz * (height - trunk_h * 0.6)])
        centers.append(c)
        cr = crown_r * 0.42
        for j in range(leaves_per):
            d = rng.normal(0, 1, 3)
            d /= np.linalg.norm(d)
            p = c + d * cr * rng.random() ** 0.35 * np.array([1, 1, 0.8])
            nrm = d + rng.normal(0, 0.5, 3)
            nrm /= np.linalg.norm(nrm)
            t1 = np.cross(nrm, [0, 0, 1.0])
            if np.linalg.norm(t1) < 1e-3:
                t1 = np.array([1.0, 0, 0])
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(nrm, t1)
            s = leaf_size * (0.7 + 0.6 * rng.random())
            i0 = len(verts)
            verts += [tuple(p + t1 * s), tuple(p + t2 * s * 0.55), tuple(p - t1 * s), tuple(p - t2 * s * 0.55)]
            faces.append((i0, i0 + 1, i0 + 2, i0 + 3))
    leaf_me = bpy.data.meshes.new(name + "_leaves")
    leaf_me.from_pydata(verts, [], faces)
    # Äste vom Stamm zu den Clustern
    bm = bmesh.new()
    bm.from_mesh(trunk_me)
    for c in centers[: max(6, n_clusters // 2)]:
        start = np.array([0, 0, trunk_h * rng.uniform(0.75, 1.0)])
        v = c - start
        ln = np.linalg.norm(v)
        res = bmesh.ops.create_cone(bm, cap_ends=False, segments=5, radius1=height * 0.012, radius2=height * 0.004,
                                    depth=ln)
        rot = Vector(tuple(v / ln)).to_track_quat("Z", "Y").to_matrix().to_4x4()
        bmesh.ops.transform(bm, verts=res["verts"], matrix=rot)
        bmesh.ops.translate(bm, verts=res["verts"], vec=tuple(start + v / 2))
    bm.to_mesh(trunk_me)
    bm.free()
    trunk_me.materials.append(bark_mat)
    leaf_me.materials.append(leaf_mat)
    for p in trunk_me.polygons:
        p.use_smooth = True
    # zusammenführen
    ob_t = bpy.data.objects.new(name, trunk_me)
    ob_l = bpy.data.objects.new(name + "_l", leaf_me)
    return ob_t, ob_l


def make_tree_collection(name, variants):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    objs = []
    for i, parts in enumerate(variants):
        sub = bpy.data.collections.new(f"{name}_{i}")
        coll.children.link(sub)
        for o in parts:
            sub.objects.link(o)
        objs.append(sub)
    # Sammlung aus Render ausblenden (nur Instanzen sichtbar)
    lc = bpy.context.view_layer.layer_collection.children[name]
    lc.exclude = False
    coll.hide_render = True
    coll.hide_viewport = True
    return coll, objs


def scatter_instances(name, subcolls, points, rots=None, scales=None, seed=0):
    """Instanzen als Empties mit Collection-Instanzen (effizient in Cycles)."""
    rng = np.random.default_rng(seed)
    parent = bpy.data.objects.new(name, None)
    fpv.link(parent)
    out = []
    for i, p in enumerate(points):
        sub = subcolls[rng.integers(0, len(subcolls))]
        e = bpy.data.objects.new(f"{name}_{i}", None)
        e.instance_type = "COLLECTION"
        e.instance_collection = sub
        e.location = tuple(p)
        e.rotation_euler = (0, 0, rots[i] if rots is not None else rng.uniform(0, 2 * np.pi))
        s = scales[i] if scales is not None else rng.uniform(0.8, 1.25)
        e.scale = (s, s, s * rng.uniform(0.9, 1.1))
        e.parent = parent
        fpv.link(e)
        out.append(e)
    return parent


def ajisa_variant(name, leaf_mat, bark_mat, height=8.0, crown_r=3.2, leaves=2600, seed=0, leaf_size=0.34,
                  lean=0.05):
    """Namek-Ajisa-Baum: dünner, gerader Stamm mit kugeliger, buschiger Krone."""
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    top = np.array([rng.normal(0, lean) * height, rng.normal(0, lean) * height, height])
    res = bmesh.ops.create_cone(bm, cap_ends=False, segments=10, radius1=height * 0.03, radius2=height * 0.018,
                                depth=np.linalg.norm(top) + 0.5)
    rot = Vector(tuple(top / np.linalg.norm(top))).to_track_quat("Z", "Y").to_matrix().to_4x4()
    bmesh.ops.transform(bm, verts=res["verts"], matrix=rot)
    bmesh.ops.translate(bm, verts=res["verts"], vec=tuple(top / 2 - np.array([0, 0, 0.25])))
    trunk_me = bpy.data.meshes.new(name + "_trunk")
    bm.to_mesh(trunk_me)
    bm.free()
    c0 = top + np.array([0, 0, crown_r * 0.8])
    verts, faces = [], []
    for j in range(leaves):
        d = rng.normal(0, 1, 3)
        d /= np.linalg.norm(d)
        # Blätter vorwiegend in der äußeren Schale -> buschige Kugel
        rr = crown_r * (0.62 + 0.38 * rng.random() ** 0.4)
        # leicht klumpige Oberfläche
        rr *= 1 + 0.12 * math.sin(d[0] * 5 + seed) * math.sin(d[1] * 4 + d[2] * 3)
        p = c0 + d * rr * np.array([1, 1, 0.92])
        nrm = d + rng.normal(0, 0.6, 3)
        nrm /= np.linalg.norm(nrm)
        t1 = np.cross(nrm, [0, 0, 1.0])
        if np.linalg.norm(t1) < 1e-3:
            t1 = np.array([1.0, 0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(nrm, t1)
        s = leaf_size * (0.7 + 0.6 * rng.random())
        i0 = len(verts)
        verts += [tuple(p + t1 * s), tuple(p + t2 * s * 0.6), tuple(p - t1 * s), tuple(p - t2 * s * 0.6)]
        faces.append((i0, i0 + 1, i0 + 2, i0 + 3))
    # dichter Kern, damit die Krone nicht durchsichtig wirkt
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=crown_r * 0.78)
    bmesh.ops.scale(bm, vec=(1, 1, 0.92), verts=bm.verts)
    bmesh.ops.translate(bm, vec=tuple(c0), verts=bm.verts)
    core_me = bpy.data.meshes.new(name + "_core")
    bm.to_mesh(core_me)
    bm.free()
    leaf_me = bpy.data.meshes.new(name + "_leaves")
    leaf_me.from_pydata(verts, [], faces)
    trunk_me.materials.append(bark_mat)
    leaf_me.materials.append(leaf_mat)
    core_me.materials.append(leaf_mat)
    for p in trunk_me.polygons:
        p.use_smooth = True
    for p in core_me.polygons:
        p.use_smooth = True
    ob_t = bpy.data.objects.new(name, trunk_me)
    ob_l = bpy.data.objects.new(name + "_l", leaf_me)
    ob_c = bpy.data.objects.new(name + "_c", core_me)
    return ob_t, ob_l, ob_c



def palm_variant(name, leaf_mat, trunk_mat, height=11.0, n_fronds=15, frond_len=4.2, seed=0, lean=0.18):
    """Kokospalme: gebogener, geringelter Stamm; Wedel mit hängenden Fiederblättchen."""
    rng = np.random.default_rng(seed)
    # Stamm entlang einer gebogenen Bahn
    n = 24
    ts = np.linspace(0, 1, n)
    ldir = np.array([np.cos(seed * 1.7), np.sin(seed * 1.7), 0.0])
    path = np.stack([ldir[0] * lean * height * ts ** 1.6, ldir[1] * lean * height * ts ** 1.6, height * ts], 1)
    verts, faces = [], []
    sides = 10
    for i, t in enumerate(ts):
        r = 0.22 * (1 - 0.35 * t) * (1 + 0.25 * np.exp(-t * 12))
        for k in range(sides):
            a = 2 * np.pi * k / sides
            verts.append(tuple(path[i] + np.array([np.cos(a) * r, np.sin(a) * r, 0])))
    for i in range(n - 1):
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((i * sides + k, i * sides + k2, (i + 1) * sides + k2, (i + 1) * sides + k))
    tme = bpy.data.meshes.new(name + "_trunk")
    tme.from_pydata(verts, [], faces)
    for p in tme.polygons:
        p.use_smooth = True
    tme.materials.append(trunk_mat)
    # Wedel
    top = path[-1]
    lv, lf = [], []
    for f in range(n_fronds):
        az = 2 * np.pi * f / n_fronds + rng.uniform(-0.2, 0.2)
        el = rng.uniform(-0.1, 0.55) if f % 3 else rng.uniform(0.4, 0.9)
        L = frond_len * rng.uniform(0.8, 1.15)
        hdir = np.array([np.cos(az), np.sin(az), 0.0])
        side = np.array([-np.sin(az), np.cos(az), 0.0])
        m = 22
        rib = []
        for j in range(m):
            s = j / (m - 1)
            d = s * L
            p = top + hdir * d * np.cos(el) + np.array([0, 0, d * np.sin(el) - 0.55 * d * d / L])
            rib.append(p)
        rib = np.array(rib)
        for j in range(1, m - 1):
            s = j / (m - 1)
            tang = rib[j + 1] - rib[j - 1]
            tang /= np.linalg.norm(tang)
            ll = 0.95 * (1 - s ** 1.4) * (0.4 + s * 0.8) * L / 4.2
            for sg in (-1, 1):
                dirl = side * sg * 0.8 + np.array([0, 0, -0.55]) + tang * 0.35
                dirl /= np.linalg.norm(dirl)
                w = 0.05 + 0.03 * (1 - s)
                a0 = rib[j]
                b0 = a0 + dirl * ll
                i0 = len(lv)
                lv += [tuple(a0 - tang * w), tuple(a0 + tang * w), tuple(b0 + tang * w * 0.2), tuple(b0 - tang * w * 0.2)]
                lf.append((i0, i0 + 1, i0 + 2, i0 + 3))
        # Mittelrippe als flaches Band
        for j in range(m - 1):
            a0, b0 = rib[j], rib[j + 1]
            w = 0.035 * (1 - j / m)
            i0 = len(lv)
            lv += [tuple(a0 - side * w), tuple(a0 + side * w), tuple(b0 + side * w), tuple(b0 - side * w)]
            lf.append((i0, i0 + 1, i0 + 2, i0 + 3))
    lme = bpy.data.meshes.new(name + "_fronds")
    lme.from_pydata(lv, [], lf)
    lme.materials.append(leaf_mat)
    return bpy.data.objects.new(name, tme), bpy.data.objects.new(name + "_f", lme)


def palm_trunk_material(name="PalmTrunk"):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    z = nb.sep(co)[2]
    rings = nb.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Z", wave_profile="SAW")
    nb.link(co, rings.inputs["Vector"])
    rings.inputs["Scale"].default_value = 3.2
    rings.inputs["Distortion"].default_value = 1.5
    n = nb.noise(co, scale=6, detail=3)
    col = nb.ramp(nb.out(rings, "Fac"), [(0.0, (0.10, 0.08, 0.06)), (0.6, (0.25, 0.21, 0.16)), (1.0, (0.16, 0.13, 0.1))])
    col = nb.mix(nb.math("MULTIPLY", nb.out(n, "Fac"), 0.4), col, (0.08, 0.07, 0.05))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.9)
    nb.link(nb.bump(nb.math("ADD", nb.out(rings, "Fac"), nb.out(n, "Fac")), strength=0.8, distance=0.03), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def vines_mesh(name, starts, rng, leaf_mat, stem_mat, len_range=(3, 9)):
    """Herabhängende Lianen: starts = Liste (Punkt, Außennormale)."""
    sv, sf, lv, lf = [], [], [], []
    for (p, nrm) in starts:
        p = np.array(p, float)
        nrm = np.array(nrm, float)
        nrm[2] = 0
        if np.linalg.norm(nrm) < 1e-3:
            nrm = np.array([1.0, 0, 0])
        nrm /= np.linalg.norm(nrm)
        Lh = rng.uniform(*len_range)
        m = int(Lh / 0.35) + 2
        pts = []
        ph = rng.uniform(0, 6)
        for j in range(m):
            t = j / (m - 1)
            q = p + nrm * (0.25 + 0.15 * np.sin(t * 7 + ph)) + np.array([0.12 * np.sin(t * 5 + ph), 0.12 * np.cos(t * 4 + ph), -Lh * t])
            pts.append(q)
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            t1 = np.cross(d, [1.0, 0, 0])
            t1 /= np.linalg.norm(t1) + 1e-9
            t2 = np.cross(d, t1)
            t2 /= np.linalg.norm(t2) + 1e-9
            i0 = len(sv)
            for q in (a, b):
                for k in range(4):
                    ang = k * np.pi / 2
                    sv.append(tuple(q + 0.018 * (np.cos(ang) * t1 + np.sin(ang) * t2)))
            for k in range(4):
                k2 = (k + 1) % 4
                sf.append((i0 + k, i0 + k2, i0 + 4 + k2, i0 + 4 + k))
        for j, q in enumerate(pts):
            for _ in range(3):
                d = rng.normal(0, 1, 3)
                d[2] = abs(d[2]) * 0.3 - 0.4
                d /= np.linalg.norm(d)
                s = rng.uniform(0.08, 0.16)
                c = q + d * 0.06
                t1 = np.cross(d, [0, 0, 1.0])
                t1 /= np.linalg.norm(t1) + 1e-9
                i0 = len(lv)
                lv += [tuple(c), tuple(c + t1 * s * 0.5 + d * s), tuple(c + d * s * 1.8), tuple(c - t1 * s * 0.5 + d * s)]
                lf.append((i0, i0 + 1, i0 + 2, i0 + 3))
    sme = bpy.data.meshes.new(name + "_stems")
    sme.from_pydata(sv, [], sf)
    sme.materials.append(stem_mat)
    lme = bpy.data.meshes.new(name + "_leaves")
    lme.from_pydata(lv, [], lf)
    lme.materials.append(leaf_mat)
    so = bpy.data.objects.new(name + "_stems", sme)
    lo = bpy.data.objects.new(name + "_leaves", lme)
    fpv.link(so)
    fpv.link(lo)
    return so, lo


def scatter_gn(name, coll, n_variants, points, scales=None, seed=0):
    """Streuung per Geometry Nodes: ein Punkt-Mesh (Attribute 'variant', 'scale', 'rot_z'), 'Instance on Points'
    wählt die Variante aus den Unter-Collections von `coll` (Collection Info, Kinder getrennt),
    zufällige Drehung um Z und Größe pro Punkt."""
    rng = np.random.default_rng(seed)
    P = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = len(P)
    me = bpy.data.meshes.new(name + "Pts")
    me.vertices.add(n)
    me.vertices.foreach_set("co", P.ravel())
    var = me.attributes.new("variant", "INT", "POINT")
    var.data.foreach_set("value", rng.integers(0, n_variants, n).astype(np.int32))
    sc = me.attributes.new("scale", "FLOAT", "POINT")
    s = np.asarray(scales, np.float32) if scales is not None else rng.uniform(0.8, 1.25, n).astype(np.float32)
    sc.data.foreach_set("value", s)
    rz = me.attributes.new("rot_z", "FLOAT", "POINT")
    rz.data.foreach_set("value", rng.uniform(0, 2 * np.pi, n).astype(np.float32))
    ob = bpy.data.objects.new(name, me)
    fpv.link(ob)
    ng = bpy.data.node_groups.new(name + "GN", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    g = fpv.NB(ng)
    gi, go = g.node("NodeGroupInput"), g.node("NodeGroupOutput")
    mtp = g.node("GeometryNodeMeshToPoints")
    g.link(gi.outputs[0], mtp.inputs["Mesh"])
    ci = g.node("GeometryNodeCollectionInfo")
    ci.transform_space = "ORIGINAL"
    ci.inputs["Collection"].default_value = coll
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    iop = g.node("GeometryNodeInstanceOnPoints")
    g.link(mtp.outputs[0], iop.inputs["Points"])
    g.link(ci.outputs[0], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True

    def attr(nm, dtype):
        a = g.node("GeometryNodeInputNamedAttribute")
        a.data_type = dtype
        a.inputs["Name"].default_value = nm
        return a.outputs["Attribute"]
    g.link(attr("variant", "INT"), iop.inputs["Instance Index"])
    g.link(g.comb(0.0, 0.0, attr("rot_z", "FLOAT")), iop.inputs["Rotation"])
    s_ = attr("scale", "FLOAT")
    g.link(g.comb(s_, s_, s_), iop.inputs["Scale"])
    g.link(iop.outputs[0], go.inputs[0])
    mod = ob.modifiers.new("Scatter", "NODES")
    mod.node_group = ng
    return ob
