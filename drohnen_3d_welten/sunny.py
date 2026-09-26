"""Thousand Sunny (One Piece) – Production-Modell v2.

Lokales System: +X = Bug, +Y = Backbord, +Z = oben, Wasserlinie z = 0.
Brigantine, ~32 m Länge, ~10 m Breite, Großmast ~27 m über Deck.

Referenz-Merkmale: gerundeter brauner Plankenrumpf mit weißen und roten Details,
gelber Löwenkopf mit orangefarbener Blütenmähne und gekreuzten Knochen dahinter,
Rasendeck, zwei Masten mit weißen Rahsegeln (Großsegel mit Strohhut-Jolly-Roger),
Gaffelsegel rot-schwarz gestreift am zweiten Mast, rundes Ausguck-Haus, schwarze Flagge,
Soldier-Dock-Tore mittschiffs, Kanonenpforten, Coup-de-Burst-Öffnung am Heck.
"""
import math

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

import fpv
import textures

L = 32.0
W = 4.9
PLANK_W = 0.30   # Plankenbreite entlang des Umfangs (m)
PLANK_L = 6.5    # Plankenlänge (m)


def sheer(u):
    return 3.9 + 1.6 * np.clip((u - 0.55) / 0.45, 0, 1) ** 2 + 1.2 * np.clip((0.25 - u) / 0.25, 0, 1) ** 2


def keel(u):
    return -2.3 + 2.5 * np.clip((u - 0.78) / 0.22, 0, 1) ** 2 + 0.4 * np.clip((0.08 - u) / 0.08, 0, 1)


def half_beam(u):
    hw = W * (0.80 + 0.20 * np.sin(np.pi / 2 * np.clip(u / 0.28, 0, 1)))
    bow = np.clip((u - 0.58) / 0.42, 0, 1)
    return hw * np.sqrt(np.clip(1 - bow ** 2.1, 0, 1))


def section(v):
    return 1 - (1 - v) ** 2.3 + 0.04 * np.sin(np.pi * v)


def hull_point(u, v, side=1):
    x = -L / 2 + u * L
    return Vector((x, side * half_beam(u) * section(v), keel(u) + (sheer(u) - keel(u)) * v))


def hull_frame(u, v, side=1):
    """Lokales Rahmen-System auf der Rumpfhaut: (Punkt, Tangente längs, Tangente Umfang, Normale)."""
    p = hull_point(u, v, side)
    tu = (hull_point(min(u + 0.002, 1), v, side) - hull_point(max(u - 0.002, 0), v, side)).normalized()
    tv = (hull_point(u, min(v + 0.01, 1), side) - hull_point(u, max(v - 0.01, 0), side)).normalized()
    n = tu.cross(tv) * side
    if n.y * side < 0:
        n = -n
    return p, tu, tv, n.normalized()


# --------------------------------------------------------------------------
# Materialien
# --------------------------------------------------------------------------

def hull_material():
    """Planken aus UV (x = Länge in m, y = Umfang in m) + Farbzonen aus UV 'HullV'."""
    mat, nb, out = fpv.new_material("HullPlanks")
    uvn = nb.node("ShaderNodeUVMap")
    uvn.uv_map = "UVMap"
    ux, uy, _ = nb.sep(uvn.outputs[0])
    hvn = nb.node("ShaderNodeUVMap")
    hvn.uv_map = "HullV"
    hv = nb.sep(hvn.outputs[0])[0]
    co = nb.coords("Object")
    zobj = nb.sep(co)[2]
    # Plankenreihe + versetzte Stöße
    rowf = nb.math("DIVIDE", uy, PLANK_W)
    row = nb.math("FLOOR", rowf)
    t = nb.math("FRACT", rowf)
    rh = nb.node("ShaderNodeTexWhiteNoise")
    rh.noise_dimensions = "1D"
    nb.link(row, rh.inputs["W"])
    segf = nb.math("DIVIDE", nb.math("ADD", ux, nb.math("MULTIPLY", nb.out(rh, "Value"), PLANK_L)), PLANK_L)
    seg = nb.math("FLOOR", segf)
    s = nb.math("FRACT", segf)
    pid = nb.node("ShaderNodeTexWhiteNoise")
    pid.noise_dimensions = "2D"
    nb.link(nb.comb(row, seg, 0.0), pid.inputs["Vector"])
    rnd = nb.out(pid, "Value")
    # prozedurale Holzmaserung entlang der Planke, pro Planke versetzt
    wood, grain = grain_wood(nb, ux, nb.math("MULTIPLY", t, PLANK_W), rnd)
    rgh_v = nb.math("MULTIPLY_ADD", grain, 0.25, 0.5)
    # Fugen
    seam_t = nb.math("LESS_THAN", t, 0.045)
    seam_s = nb.math("MAXIMUM", nb.math("LESS_THAN", s, 0.004), nb.math("GREATER_THAN", s, 0.996))
    seam = nb.math("MAXIMUM", seam_t, seam_s)
    # Farbzonen (v = 0 Kiel .. 1 Schandeck): rotes Band, weißes Band, Unterwasserschiff
    def band(a, b, soft=0.004):
        lo = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(hv, lo.inputs["Value"])
        lo.inputs["From Min"].default_value = a - soft
        lo.inputs["From Max"].default_value = a + soft
        hi = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(hv, hi.inputs["Value"])
        hi.inputs["From Min"].default_value = b + soft
        hi.inputs["From Max"].default_value = b - soft
        return nb.math("MULTIPLY", lo.outputs[0], hi.outputs[0])
    white = band(0.905, 0.965)
    red = band(0.872, 0.893)
    wl = nb.node("ShaderNodeMapRange", clamp=True)
    grime = nb.noise(co, scale=0.5, detail=3)
    nb.link(nb.math("ADD", zobj, nb.math("MULTIPLY", nb.out(grime, "Fac"), 0.25)), wl.inputs["Value"])
    wl.inputs["From Min"].default_value = 0.22
    wl.inputs["From Max"].default_value = 0.30
    above = wl.outputs[0]
    boot = nb.math("MULTIPLY", above, nb.math("LESS_THAN", zobj, 0.45))
    col = wood
    col = nb.mix(red, col, (0.42, 0.05, 0.035))
    col = nb.mix(white, col, (0.80, 0.78, 0.72))
    col = nb.mix(boot, col, (0.02, 0.02, 0.02))
    col = nb.mix(nb.math("SUBTRACT", 1.0, above), col, (0.10, 0.04, 0.03))
    col = nb.mix(nb.math("MULTIPLY", seam, 0.85), col, (0.02, 0.015, 0.01))
    # Salz-/Schmutzfahnen (senkrecht = entlang Umfang)
    stv = nb.noise(nb.comb(nb.math("MULTIPLY", ux, 1.5), nb.math("MULTIPLY", uy, 0.12), 0.0), scale=1.0, detail=2)
    sm = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(stv, "Fac"), sm.inputs["Value"])
    sm.inputs["From Min"].default_value = 0.56
    sm.inputs["From Max"].default_value = 0.72
    col = nb.mix(nb.math("MULTIPLY", sm.outputs[0], 0.18), col, (0.30, 0.28, 0.25))
    paint = nb.math("MAXIMUM", white, red)
    rough = nb.mix(paint, rgh_v, 0.38, dtype="FLOAT")
    rough = nb.mix(nb.math("SUBTRACT", 1.0, above), rough, 0.22, dtype="FLOAT")
    p = fpv.principled(nb, Base_Color=col, Roughness=rough)
    p.inputs["Coat Weight"].default_value = 0.12
    # Plankenwölbung (sin) + Fugen als Bump
    bulge = nb.math("SINE", nb.math("MULTIPLY", t, math.pi))
    h = nb.math("SUBTRACT", nb.math("MULTIPLY", bulge, 1.0), nb.math("MULTIPLY", seam, 0.6))
    nb.link(nb.bump(h, strength=0.55, distance=0.012), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def grain_wood(nb, along, across, rnd, dark=(0.06, 0.028, 0.012), mid=(0.12, 0.058, 0.024),
               light=(0.20, 0.10, 0.045)):
    """Holzmaserung: stark gestrecktes Rauschen (Fasern) + Jahresring-Wellen, Variation pro Brett."""
    v = nb.comb(nb.math("MULTIPLY", along, 0.35), nb.math("ADD", nb.math("MULTIPLY", across, 30.0),
                                                           nb.math("MULTIPLY", rnd, 40.0)), nb.math("MULTIPLY", rnd, 9.0))
    fib = nb.noise(v, scale=1.2, detail=5, rough=0.6, dist=1.5)
    ring = nb.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Y", wave_profile="SIN")
    nb.link(v, ring.inputs["Vector"])
    ring.inputs["Scale"].default_value = 0.8
    ring.inputs["Distortion"].default_value = 6.0
    ring.inputs["Detail"].default_value = 2.0
    g = nb.math("ADD", nb.math("MULTIPLY", nb.out(fib, "Fac"), 0.7), nb.math("MULTIPLY", nb.out(ring, "Fac"), 0.3))
    col = nb.ramp(g, [(0.3, dark), (0.5, mid), (0.72, light)])
    col = nb.vmath("SCALE", col, scale=nb.math("MULTIPLY_ADD", rnd, 0.4, 0.8))
    return col, g


def wood_material(name, tint=None, scale=0.5, rough=0.55, dark=1.0, board=0.22, axis="Z"):
    """Bretter-Holz (senkrechte Bretter bei axis='Z', Maserung entlang der Achse)."""
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    x, y, z = nb.sep(co)
    if axis == "Z":
        along = z
        acr = nb.math("ADD", x, y)
    else:
        along = x
        acr = nb.math("ADD", z, y)
    bf = nb.math("DIVIDE", acr, board)
    bid = nb.math("FLOOR", bf)
    bt = nb.math("FRACT", bf)
    wn = nb.node("ShaderNodeTexWhiteNoise")
    wn.noise_dimensions = "1D"
    nb.link(bid, wn.inputs["W"])
    rnd = nb.out(wn, "Value")
    col, g = grain_wood(nb, along, nb.math("MULTIPLY", bt, board), rnd)
    col = nb.vmath("SCALE", col, scale=dark)
    seam = nb.math("LESS_THAN", bt, 0.05)
    col = nb.mix(nb.math("MULTIPLY", seam, 0.8), col, (0.02, 0.015, 0.01))
    col = _wear(nb, col, edge_col=(0.55, 0.42, 0.3), cav_col=(0.03, 0.02, 0.015))
    p = fpv.principled(nb, Base_Color=col, Roughness=nb.math("MULTIPLY_ADD", g, 0.2, rough - 0.1))
    h = nb.math("SUBTRACT", nb.math("MULTIPLY", g, 0.3), nb.math("MULTIPLY", seam, 0.5))
    nb.link(nb.bump(h, strength=0.4, distance=0.01), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def _wear(nb, col, edge_col, cav_col, edge=0.35, cav=0.5):
    """Kantenabrieb (hell) und Schmutz in Vertiefungen (dunkel) über Pointiness."""
    pt = nb.out(nb.node("ShaderNodeNewGeometry"), "Pointiness")
    n = nb.noise(nb.coords("Object"), scale=6.0, detail=3)
    e = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.math("ADD", pt, nb.math("MULTIPLY", nb.math("SUBTRACT", nb.out(n, "Fac"), 0.5), 0.06)), e.inputs["Value"])
    e.inputs["From Min"].default_value = 0.53
    e.inputs["From Max"].default_value = 0.6
    c = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(pt, c.inputs["Value"])
    c.inputs["From Min"].default_value = 0.47
    c.inputs["From Max"].default_value = 0.40
    col = nb.mix(nb.math("MULTIPLY", e.outputs[0], edge), col, edge_col)
    col = nb.mix(nb.math("MULTIPLY", c.outputs[0], cav), col, cav_col)
    return col


def paint_material(name, color, rough=0.42, wear=True, coat=0.2, under=(0.35, 0.22, 0.12)):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    big = nb.noise(co, scale=0.5, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(big, "Fac"), 0.25), color, [c * 1.12 for c in color])
    fine = nb.noise(co, scale=4.0, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(fine, "Fac"), 0.2), col, [c * 0.75 for c in color])
    if wear:
        col = _wear(nb, col, edge_col=under, cav_col=[c * 0.35 for c in color], edge=0.55, cav=0.45)
    p = fpv.principled(nb, Base_Color=col, Roughness=rough)
    p.inputs["Coat Weight"].default_value = coat
    nb.link(nb.bump(nb.out(fine, "Fac"), strength=0.06, distance=0.01), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def iron_material(name="Iron"):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    n = nb.noise(co, scale=3.0, detail=3)
    rust = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(n, "Fac"), rust.inputs["Value"])
    rust.inputs["From Min"].default_value = 0.55
    rust.inputs["From Max"].default_value = 0.7
    col = nb.mix(rust.outputs[0], (0.05, 0.05, 0.05), (0.18, 0.07, 0.03))
    p = fpv.principled(nb, Base_Color=col, Roughness=nb.mix(rust.outputs[0], 0.45, 0.85, dtype="FLOAT"),
                       Metallic=nb.mix(rust.outputs[0], 0.85, 0.1, dtype="FLOAT"))
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def sail_material(name, decal=None, stripes=None):
    mat, nb, out = fpv.new_material(name)
    uv = nb.coords("UV")
    co = nb.coords("Object")
    base = (0.78, 0.74, 0.64)
    wave = nb.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X")
    nb.link(uv, wave.inputs["Vector"])
    wave.inputs["Scale"].default_value = 11.0
    seams = nb.ramp(nb.out(wave, "Fac"), [(0.0, (0.8, 0.8, 0.8)), (0.05, (1, 1, 1)), (0.95, (1, 1, 1)), (1.0, (0.8, 0.8, 0.8))])
    dirt = nb.noise(co, scale=0.8, detail=3, rough=0.6)
    dcol = nb.ramp(nb.out(dirt, "Fac"), [(0.3, (0.8, 0.76, 0.68)), (0.75, (1.0, 1.0, 1.0))])
    col = nb.mix(1.0, base, seams, blend="MULTIPLY")
    if stripes:
        st = nb.math("FRACT", nb.math("MULTIPLY", nb.sep(uv)[0], stripes))
        red = nb.math("LESS_THAN", st, 0.5)
        col = nb.mix(red, (0.03, 0.03, 0.03), (0.50, 0.05, 0.035))
        col = nb.mix(1.0, col, seams, blend="MULTIPLY")
    col = nb.mix(1.0, col, dcol, blend="MULTIPLY")
    if decal:
        img = nb.image(decal, uv)
        img.extension = "CLIP"
        col = nb.mix(nb.out(img, "Alpha"), col, nb.out(img, "Color"))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.82)
    p.inputs["Sheen Weight"].default_value = 0.25
    tr = nb.node("ShaderNodeBsdfTranslucent")
    nb.set_in(tr, "Color", col)
    mix = nb.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.3
    nb.link(p.outputs[0], mix.inputs[1])
    nb.link(tr.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    return mat


def flag_material():
    mat, nb, out = fpv.new_material("Flag")
    uv = nb.coords("UV")
    img = nb.image(textures.jolly_roger_flag(), uv)
    p = fpv.principled(nb, Base_Color=nb.out(img, "Color"), Roughness=0.85)
    p.inputs["Sheen Weight"].default_value = 0.4
    tr = nb.node("ShaderNodeBsdfTranslucent")
    nb.set_in(tr, "Color", nb.out(img, "Color"))
    mix = nb.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.2
    nb.link(p.outputs[0], mix.inputs[1])
    nb.link(tr.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    return mat


def lawn_materials():
    base, nb, out = fpv.new_material("LawnBase")
    co = nb.coords("Object")
    img = nb.image(fpv.ASSETS + "/bab/textures_grass.jpg", nb.mapping(co, scale=(0.9, 0.9, 0.9)))
    col = nb.mix(1.0, nb.out(img, "Color"), (0.75, 0.9, 0.7), blend="MULTIPLY")
    p = fpv.principled(nb, Base_Color=col, Roughness=0.85)
    nb.link(p.outputs[0], out.inputs[0])
    blade, nb, out = fpv.new_material("GrassBlade")
    uv = nb.coords("UV")
    v = nb.sep(uv)[1]
    oi = nb.out(nb.node("ShaderNodeObjectInfo"), "Random")
    wn = nb.node("ShaderNodeTexWhiteNoise")
    wn.noise_dimensions = "3D"
    nb.link(nb.coords("Object"), wn.inputs["Vector"])
    root = nb.mix(nb.out(wn, "Value"), (0.02, 0.05, 0.01), (0.04, 0.08, 0.015))
    tip = nb.mix(nb.out(wn, "Value"), (0.14, 0.24, 0.04), (0.26, 0.30, 0.08))
    col = nb.mix(v, root, tip)
    p = fpv.principled(nb, Base_Color=col, Roughness=0.6)
    p.inputs["Subsurface Weight"].default_value = 0.0
    tr = nb.node("ShaderNodeBsdfTranslucent")
    nb.set_in(tr, "Color", col)
    mix = nb.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.25
    nb.link(p.outputs[0], mix.inputs[1])
    nb.link(tr.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    return base, blade


# --------------------------------------------------------------------------
# Geometrie-Helfer
# --------------------------------------------------------------------------

def bevel_stack(ob, small=0.012, big=None, angle=30):
    """Bevel-Hierarchie: optional große Silhouetten-Fase (Gewicht), feine Glanzkante, Weighted Normal."""
    if big:
        b1 = ob.modifiers.new("bevel_silhouette", "BEVEL")
        b1.limit_method = "ANGLE"
        b1.angle_limit = math.radians(60)
        b1.width = big
        b1.segments = 4
        b1.profile = 0.5
    b2 = ob.modifiers.new("bevel_micro", "BEVEL")
    b2.limit_method = "ANGLE"
    b2.angle_limit = math.radians(angle)
    b2.width = small
    b2.segments = 2
    b2.harden_normals = True
    wn = ob.modifiers.new("wnormal", "WEIGHTED_NORMAL")
    wn.keep_sharp = True
    return ob


def cyl(name, p0, p1, r0, r1=None, mat=None, verts=16, cap=True):
    r1 = r0 if r1 is None else r1
    p0, p1 = Vector(p0), Vector(p1)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=cap, segments=verts, radius1=r0, radius2=r1, depth=(p1 - p0).length)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    d = (p1 - p0).normalized()
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = d.to_track_quat("Z", "Y")
    ob.location = (p0 + p1) / 2
    return ob


def sphere(name, loc, r, mat=None, scale=(1, 1, 1), seg=32, ring=16):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=ring, radius=r)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    ob.location = loc
    ob.scale = scale
    return ob


def box(name, a, b, mat, bevel=0.03):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    ax, ay, az = a
    bx, by, bz = b
    bmesh.ops.scale(bm, vec=(bx - ax, by - ay, bz - az), verts=bm.verts)
    bmesh.ops.translate(bm, vec=((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2), verts=bm.verts)
    ob = fpv.mesh_from_bmesh(bm, name, mat, smooth=False)
    if bevel:
        bevel_stack(ob, small=bevel)
    return ob


def tubes_mesh(name, segments, r, mat, sides=6):
    """Viele Seil-/Stabzylinder in EINEM Mesh (Takelage, Webeleinen, Nieten-Stäbe)."""
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    for (a, b) in segments:
        a, b = np.array(a, float), np.array(b, float)
        d = b - a
        ln = np.linalg.norm(d)
        if ln < 1e-6:
            continue
        d /= ln
        t1 = np.cross(d, [0, 0, 1.0])
        if np.linalg.norm(t1) < 1e-3:
            t1 = np.cross(d, [1.0, 0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(d, t1)
        i0 = len(verts)
        for p in (a, b):
            for q in ang:
                verts.append(tuple(p + r * (np.cos(q) * t1 + np.sin(q) * t2)))
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((i0 + k, i0 + k2, i0 + sides + k2, i0 + sides + k))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    me.materials.append(mat)
    fpv.link(ob)
    return ob


def catenary(a, b, sag, n=12):
    a, b = np.array(a, float), np.array(b, float)
    pts = [a + (b - a) * t - np.array([0, 0, sag * 4 * t * (1 - t)]) for t in np.linspace(0, 1, n)]
    return list(zip(pts[:-1], pts[1:]))


def spheres_mesh(name, centers, r, mat, subdiv=1, flatten=0.5, normals=None):
    """Nieten: viele kleine Halbkugeln in einem Mesh."""
    bm = bmesh.new()
    for i, c in enumerate(centers):
        res = bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=r)
        nrm = Vector(normals[i]) if normals is not None else Vector((0, 0, 1))
        rot = nrm.to_track_quat("Z", "Y").to_matrix().to_4x4()
        sc = Matrix.Diagonal((1, 1, flatten, 1))
        bmesh.ops.transform(bm, verts=res["verts"], matrix=Matrix.Translation(Vector(c)) @ rot @ sc)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    return ob


# --------------------------------------------------------------------------
# Rumpf, Deck, Rasen
# --------------------------------------------------------------------------

def hull_mesh(mat_hull, mat_transom, nu=200, nv=48):
    us = np.linspace(0, 1, nu)
    vs = np.linspace(0, 1, nv)
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    hvl = bm.loops.layers.uv.new("HullV")
    rows = []
    for u in us:
        x = -L / 2 + u * L
        hw = half_beam(u)
        k = keel(u)
        s = sheer(u)
        pts = [(x, -hw * section(v), k + (s - k) * v) for v in vs[::-1]] + \
              [(x, hw * section(v), k + (s - k) * v) for v in vs[1:]]
        vv = list(vs[::-1]) + list(vs[1:])
        # Umfangslänge vom Kiel aus (für Planken)
        half = np.array([(hw * section(v), k + (s - k) * v) for v in vs])
        g = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(half, axis=0), axis=1))])
        gg = list(g[::-1]) + list(g[1:])
        rows.append([(bm.verts.new(p), gi, vi) for p, gi, vi in zip(pts, gg, vv)])
    nr = len(rows[0])
    for i in range(nu - 1):
        for j in range(nr - 1):
            q = [rows[i][j], rows[i + 1][j], rows[i + 1][j + 1], rows[i][j + 1]]
            try:
                f = bm.faces.new([e[0] for e in q])
            except ValueError:
                continue
            ui = [i, i + 1, i + 1, i]
            for lp, e, uu in zip(f.loops, q, ui):
                lp[uvl].uv = (us[uu] * L, e[1])
                lp[hvl].uv = (e[2], 0.0)
    tf = bm.faces.new([e[0] for e in rows[0][::-1]])
    tf.material_index = 1
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.002)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = fpv.mesh_from_bmesh(bm, "Hull", mat_hull)
    ob.data.materials.append(mat_transom)
    sol = ob.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.22
    sol.offset = -1
    ob.modifiers.new("wn", "WEIGHTED_NORMAL")
    return ob


def deck_z(u):
    return sheer(u) - 1.05


def deck_mesh(mat, nu=90, nw=16):
    us = np.linspace(0.02, 0.985, nu)
    bm = bmesh.new()
    rows = []
    for u in us:
        x = -L / 2 + u * L
        hw = max(half_beam(u) * section(0.86) - 0.25, 0.02)
        z = deck_z(u)
        rows.append([bm.verts.new((x, -hw + 2 * hw * t, z)) for t in np.linspace(0, 1, nw)])
    for i in range(nu - 1):
        for j in range(nw - 1):
            bm.faces.new([rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]])
    return fpv.mesh_from_bmesh(bm, "Deck", mat)


def lawn_blades(mat, u0=0.30, u1=0.86, count=60000, seed=5):
    """Echte Grashalme (Dreiecke) auf dem Rasendeck."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(u0, u1, count)
    hw = np.maximum(half_beam(u) * section(0.86) - 0.4, 0.05)
    x = -L / 2 + u * L
    y = rng.uniform(-1, 1, count) * hw
    z = deck_z(u) + 0.005
    h = rng.uniform(0.05, 0.12, count)
    a = rng.uniform(0, 2 * np.pi, count)
    lean = rng.normal(0, 0.35, (count, 2)) * h[:, None]
    w = 0.006 + 0.004 * rng.random(count)
    base = np.stack([x, y, z], 1)
    side = np.stack([np.cos(a), np.sin(a), np.zeros(count)], 1) * w[:, None]
    tip = base + np.stack([lean[:, 0], lean[:, 1], h], 1)
    verts = np.concatenate([base - side, base + side, tip], 0)
    n = count
    faces = np.stack([np.arange(n), np.arange(n) + n, np.arange(n) + 2 * n], 1)
    me = bpy.data.meshes.new("LawnBlades")
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
    ob = bpy.data.objects.new("LawnBlades", me)
    me.materials.append(mat)
    fpv.link(ob)
    return ob


# --------------------------------------------------------------------------
# Segel
# --------------------------------------------------------------------------

def sail_mesh(name, width, height, bulge, mat, nx=34, ny=30, foot_arch=0.6, taper=0.06, twist=0.0):
    """Rahsegel mit glaubhafter Windform: Bauch mit Maximum bei ~40 % Tiefe, gerundetes Unterliek,
    Spannungsfalten zu den Schothörnern (Noise, gerichtet), leichte Verwindung."""
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    bm = bmesh.new()
    uv_l = bm.loops.layers.uv.new("UVMap")
    rows = []
    for j in range(ny):
        t = j / (ny - 1)
        row = []
        for i in range(nx):
            s = i / (nx - 1)
            sx = 2 * s - 1
            y = sx * width / 2 * (1 - taper * (1 - t))
            foot = foot_arch * (1 - sx * sx) * t ** 6  # Unterliek nach oben gewölbt
            z = -t * height + foot
            depth = bulge * (1 - sx ** 2) * (np.sin(np.pi * min(t, 1) ** 0.85) ** 1.2)
            depth += twist * sx * t
            # Spannungsfalten diagonal zu den unteren Ecken
            fold = 0.05 * np.sin(18 * (abs(sx) - t * 0.9)) * (t ** 2) * (1 - abs(sx) * 0.3)
            row.append(bm.verts.new((depth + fold, y, z)))
        rows.append(row)
    for j in range(ny - 1):
        for i in range(nx - 1):
            f = bm.faces.new([rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]])
            for lp, (ii, jj) in zip(f.loops, [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]):
                lp[uv_l].uv = (1 - ii / (nx - 1), 1 - jj / (ny - 1))
    edges = [rows[0], rows[-1], [r[0] for r in rows], [r[-1] for r in rows]]
    edge_pts = [[tuple(v.co) for v in e] for e in edges]
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    sub = ob.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1
    fpv.displace_obj(ob, "CLOUDS", size=1.3, strength=0.07, depth=2, name=name + "_wr")
    sol = ob.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.025
    return ob, edge_pts


def gaff_sail_mesh(name, luff, gaff_len, boom_len, gaff_rise, mat, nx=24, ny=26, bulge=0.9):
    """Gaffelsegel (trapezförmig, achtern des Mastes). Lokal: -X nach achtern, Z hoch."""
    bm = bmesh.new()
    uv_l = bm.loops.layers.uv.new("UVMap")
    rows = []
    for j in range(ny):
        t = j / (ny - 1)  # 0 unten .. 1 oben
        row = []
        for i in range(nx):
            s = i / (nx - 1)  # 0 am Mast .. 1 achtern
            ln = boom_len + (gaff_len - boom_len) * t
            x = -s * ln
            z = t * luff + s * gaff_rise * t
            y = bulge * np.sin(np.pi * s) * np.sin(np.pi * min(max(t, 0.02), 0.98)) * 0.9
            row.append(bm.verts.new((x, y, z)))
        rows.append(row)
    for j in range(ny - 1):
        for i in range(nx - 1):
            f = bm.faces.new([rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]])
            for lp, (ii, jj) in zip(f.loops, [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]):
                lp[uv_l].uv = (ii / (nx - 1), jj / (ny - 1))
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    fpv.displace_obj(ob, "CLOUDS", size=1.2, strength=0.05, depth=2, name=name + "_wr")
    sol = ob.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.025
    return ob


# --------------------------------------------------------------------------
# Löwen-Galionsfigur
# --------------------------------------------------------------------------

def petal_mesh(name, length, width, thick, mat, bend=0.35):
    """Blütenblatt der Mähne: gebogenes, verdicktes Blatt mit runder Spitze (Quad-Topologie)."""
    nx, ny = 7, 12
    bm = bmesh.new()
    grid = []
    for j in range(ny):
        t = j / (ny - 1)
        row = []
        wt = width * (np.sin(np.pi * min(t * 0.92 + 0.08, 1.0)) ** 0.6)
        for i in range(nx):
            s = (i / (nx - 1)) * 2 - 1
            x = s * wt / 2
            z = t * length
            y = -bend * (t ** 2) * length * 0.35 + 0.08 * (1 - s * s) * width
            row.append(bm.verts.new((x, y, z)))
        grid.append(row)
    for j in range(ny - 1):
        for i in range(nx - 1):
            bm.faces.new([grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]])
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    sol = ob.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = thick
    sol.offset = 0
    sub = ob.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 2
    return ob


def lion_head(body, mats, x0, z0):
    """Löwenkopf: Gesicht ≈ 55 % des Mähnen-Durchmessers, Mähne (2 Kränze) an einer drehbaren Nabe,
    gekreuzte Knochen dahinter wie ein Jolly Roger, Kiefer separat (Gaon-Cannon-Mündung)."""
    K = 1.5
    face_c = Vector((x0 + 1.3, 0, z0 + 2.2))
    parts = []
    # Nabe für die Mähnen-Mechanik (Drehachse = Schiffslängsachse)
    hub = bpy.data.objects.new("ManeHub", None)
    fpv.link(hub)
    hub.parent = body
    hub.location = face_c + Vector((-0.6, 0, 0))
    # Sockel/Hals
    neck = cyl("LionNeck", (x0 - 2.2, 0, z0 - 1.0), (face_c.x - 0.5, 0, face_c.z - 0.3), 1.8, 2.1, mats["mane_dark"], 40)
    parts.append(neck)
    # Rückplatte der Mähne (schließt Lücken)
    back = cyl("ManeBack", (face_c.x - 1.4, 0, face_c.z), (face_c.x - 0.9, 0, face_c.z), 3.9, 3.7, mats["mane_dark"], 48)
    back.parent = hub
    back.location = back.location - hub.location
    face = sphere("LionFace", face_c, 1.55 * K, mats["face"], scale=(0.78, 1.0, 0.95), seg=64, ring=32)
    parts.append(face)
    # Mähnen-Blätter: 2 Kränze à 20, Nabe als Parent (drehbar)
    petal_tpl = {}
    for layer, (n, rad, ln, wd, mk, off, tilt) in enumerate(((20, 1.9 * K, 1.35 * K, 0.95 * K, "mane", 0.0, -12),
                                                             (20, 2.45 * K, 1.5 * K, 1.1 * K, "mane_dark", 9.0, -22))):
        tpl = petal_mesh(f"PetalTpl{layer}", ln, wd, 0.16 * K, mats[mk])
        petal_tpl[layer] = tpl
        for k in range(n):
            a = math.radians(off + k * 360 / n)
            pet = tpl.copy()
            pet.data = tpl.data
            fpv.link(pet)
            dirv = Vector((0, math.cos(a), math.sin(a)))
            base = face_c + dirv * rad * 0.62 + Vector((-0.55 - 0.45 * layer, 0, 0))
            q = dirv.to_track_quat("Z", "X")  # lokale Z = radial
            q = q @ Matrix.Rotation(math.radians(tilt), 4, "X").to_quaternion()
            pet.rotation_mode = "QUATERNION"
            pet.rotation_quaternion = q
            pet.parent = hub
            pet.location = base - hub.location
        bpy.data.objects.remove(tpl)
    # Gekreuzte Knochen hinter der Mähne
    bx = face_c.x - 1.9
    for sgn in (1, -1):
        a = math.radians(38 * sgn)
        d = Vector((0, math.cos(a), math.sin(a)))
        p0 = Vector((bx, 0, face_c.z)) - d * 5.4
        p1 = Vector((bx, 0, face_c.z)) + d * 5.4
        bone = cyl("Bone", p0, p1, 0.42, 0.42, mats["bone"], 20)
        parts.append(bone)
        for p in (p0, p1):
            side = d.cross(Vector((1, 0, 0))).normalized()
            for off in (-0.42, 0.42):
                parts.append(sphere("BoneKnob", p + side * off, 0.55, mats["bone"]))
    # Gesicht: Augen, Brauen, Nase, Kiefer mit Maul
    for sy in (-1, 1):
        parts.append(sphere("Eye", face_c + Vector((1.08, 0.55 * sy, 0.38)) * K, 0.22 * K, mats["black"],
                            scale=(0.6, 1, 1.25)))
        parts.append(sphere("EyeHi", face_c + Vector((1.13, 0.5 * sy, 0.47)) * K, 0.06 * K, mats["white"]))
        parts.append(cyl("Brow", face_c + Vector((1.02, 0.25 * sy, 0.78)) * K,
                         face_c + Vector((0.95, 0.85 * sy, 0.70)) * K, 0.07 * K, 0.05 * K, mats["mane_dark"], 10))
    nose = sphere("Nose", face_c + Vector((1.25 * K, 0, 0.0)), 0.3 * K, mats["mane_dark"], scale=(0.8, 1.2, 0.8))
    parts.append(nose)
    jaw = bpy.data.objects.new("Jaw", None)
    fpv.link(jaw)
    jaw.parent = body
    jaw.location = face_c + Vector((0.4, 0, -0.1))
    pts = []
    for k in range(25):
        a = math.radians(200 + k * (140 / 24))
        pts.append(face_c + Vector((1.12 - 0.1 * abs(math.sin(a)), 0.7 * math.cos(a), -0.2 + 0.45 * math.sin(a))) * K)
    cu = bpy.data.curves.new("Mouth", "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1)
    cu.bevel_depth = 0.06 * K
    cu.bevel_resolution = 3
    mo = bpy.data.objects.new("Mouth", cu)
    mo.data.materials.append(mats["black"])
    fpv.link(mo)
    parts.append(mo)
    for o in parts:
        o.parent = body
    return hub


# --------------------------------------------------------------------------
# Aufbau
# --------------------------------------------------------------------------

def build(name="ThousandSunny"):
    root = bpy.data.objects.new(name, None)  # Kurs
    fpv.link(root)
    body = bpy.data.objects.new(name + "_body", None)  # Stampfen/Rollen/Tauchen
    fpv.link(body)
    body.parent = root
    lawn_base, blade = lawn_materials()
    mats = {
        "hull": hull_material(),
        "transom": wood_material("Transom", dark=0.8, axis="X"),
        "rail": paint_material("RailWhite", (0.80, 0.78, 0.72), rough=0.38),
        "wood": wood_material("CabinWood", dark=1.1),
        "mast": wood_material("MastWood", dark=1.25, board=5.0),
        "lawn": lawn_base,
        "blade": blade,
        "sail": sail_material("Sail"),
        "sail_jr": sail_material("SailJR", textures.jolly_roger_sail()),
        "sail_stripe": sail_material("SailStripe", stripes=7.0),
        "flag": flag_material(),
        "mane": paint_material("Mane", (0.95, 0.62, 0.08), rough=0.45, under=(0.55, 0.35, 0.18)),
        "mane_dark": paint_material("ManeDark", (0.85, 0.36, 0.04), rough=0.45, under=(0.5, 0.3, 0.15)),
        "face": paint_material("LionFace", (0.95, 0.72, 0.16), rough=0.4, under=(0.6, 0.45, 0.25)),
        "bone": paint_material("Bone", (0.86, 0.83, 0.74), rough=0.5),
        "black": fpv.simple_mat("Black", (0.015, 0.012, 0.01), rough=0.3),
        "white": paint_material("WhitePaint", (0.80, 0.78, 0.72), rough=0.38),
        "red": paint_material("RedPaint", (0.45, 0.05, 0.035), rough=0.4),
        "iron": iron_material(),
        "rope": fpv.simple_mat("Rope", (0.24, 0.18, 0.11), rough=0.9),
        "glass": fpv.simple_mat("DarkGlass", (0.015, 0.02, 0.025), rough=0.05),
        "roofblue": paint_material("CrowRoof", (0.12, 0.28, 0.55), rough=0.35),
    }
    parts = []
    parts.append(hull_mesh(mats["hull"], mats["transom"]))
    parts.append(deck_mesh(mats["lawn"]))
    parts.append(lawn_blades(mats["blade"]))

    # Reling-Handlauf (weiß), Speigatten-Leiste (rot)
    for sy in (-1, 1):
        pts = [(-L / 2 + u * L, sy * (half_beam(u) - 0.05), sheer(u) + 0.10) for u in np.linspace(0.0, 0.975, 80)]
        cu = bpy.data.curves.new("Rail", "CURVE")
        cu.dimensions = "3D"
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for p, c in zip(sp.points, pts):
            p.co = (*c, 1)
        cu.bevel_depth = 0.16
        cu.bevel_resolution = 3
        ro = bpy.data.objects.new("Rail", cu)
        ro.data.materials.append(mats["rail"])
        fpv.link(ro)
        parts.append(ro)
        # Kanonenpforten mit Rahmen + Deckel-Scharnier
        for k in range(7):
            u = 0.20 + k * 0.078
            if 0.43 < u < 0.52:  # Platz für das Soldier-Dock-Tor
                continue
            p, tu, tv, n = hull_frame(u, 0.70, sy)
            port = cyl("Port", p - n * 0.2, p + n * 0.09, 0.36, 0.36, mats["iron"], 24)
            parts.append(port)
            inner = cyl("PortIn", p - n * 0.05, p + n * 0.12, 0.27, 0.27, mats["black"], 24)
            parts.append(inner)
        # Soldier-Dock-Tor (mittschiffs, über der Wasserlinie), Rahmen, Rippen, Nieten
        p, tu, tv, n = hull_frame(0.475, 0.50, sy)
        M = Matrix((tu, n, tv)).transposed().to_4x4()
        gw, gh = 4.6, 2.5
        gate_objs = []
        rivets, rnorm = [], []
        for (a, b, m) in (((-gw / 2, 0.05, -gh / 2), (gw / 2, 0.20, gh / 2), "wood"),
                          ((-gw / 2 - 0.18, 0.0, gh / 2), (gw / 2 + 0.18, 0.28, gh / 2 + 0.18), "iron"),
                          ((-gw / 2 - 0.18, 0.0, -gh / 2 - 0.18), (gw / 2 + 0.18, 0.28, -gh / 2), "iron"),
                          ((-gw / 2 - 0.18, 0.0, -gh / 2), (-gw / 2, 0.28, gh / 2), "iron"),
                          ((gw / 2, 0.0, -gh / 2), (gw / 2 + 0.18, 0.28, gh / 2), "iron"),
                          ((-0.06, 0.18, -gh / 2), (0.06, 0.26, gh / 2), "iron")):
            ob = box("DockGate", a, b, mats[m], bevel=0.02)
            ob.matrix_world = Matrix.Translation(p) @ M
            gate_objs.append(ob)
        for k in range(5):  # Rippen
            zz = -gh / 2 + (k + 0.5) * gh / 5
            ob = box("DockRib", (-gw / 2, 0.18, zz - 0.05), (gw / 2, 0.24, zz + 0.05), mats["iron"], bevel=0.01)
            ob.matrix_world = Matrix.Translation(p) @ M
            gate_objs.append(ob)
        for xx in np.arange(-gw / 2 - 0.09, gw / 2 + 0.1, 0.3):
            for zz in (gh / 2 + 0.09, -gh / 2 - 0.09):
                rivets.append(p + M.to_3x3() @ Vector((xx, 0.30, zz)))
                rnorm.append(n)
        for zz in np.arange(-gh / 2, gh / 2 + 0.01, 0.3):
            for xx in (-gw / 2 - 0.09, gw / 2 + 0.09):
                rivets.append(p + M.to_3x3() @ Vector((xx, 0.30, zz)))
                rnorm.append(n)
        rv = spheres_mesh("DockRivets", rivets, 0.045, mats["iron"], normals=rnorm)
        parts += gate_objs + [rv]

    # Heck: Fenster mit Rahmen, Coup-de-Burst-Öffnung mit Eisenring
    for yy in (-2.5, -0.85, 0.85, 2.5):
        parts.append(cyl("SternWin", (-L / 2 - 0.25, yy, 3.3), (-L / 2 + 0.1, yy, 3.3), 0.55, 0.55, mats["glass"], 28))
        parts.append(cyl("SternFrame", (-L / 2 - 0.15, yy, 3.3), (-L / 2 + 0.05, yy, 3.3), 0.7, 0.7, mats["white"], 28))
    parts.append(cyl("CoupDeBurst", (-L / 2 - 0.35, 0, 1.2), (-L / 2 + 0.2, 0, 1.2), 1.15, 1.15, mats["black"], 40))
    parts.append(cyl("CoupRing", (-L / 2 - 0.25, 0, 1.2), (-L / 2 + 0.15, 0, 1.2), 1.45, 1.45, mats["iron"], 40))

    # Achterdeck-Haus mit echten Fensterrahmen, Dachreling, Treppe
    d_aft = deck_z(0.1)
    cab_len, cab_x0 = 7.8, -L / 2 + 0.6
    hwc = half_beam(0.12) * 0.86
    parts.append(box("AftCabin", (cab_x0, -hwc, d_aft), (cab_x0 + cab_len, hwc, d_aft + 3.3), mats["wood"], bevel=0.05))
    for k in range(4):
        xw = cab_x0 + 1.25 + k * 1.8
        for sy in (-1, 1):
            parts.append(box("CabWin", (xw - 0.45, sy * hwc - 0.07, d_aft + 1.3), (xw + 0.45, sy * hwc + 0.07, d_aft + 2.5),
                             mats["glass"], bevel=0.0))
            parts.append(box("CabWinFrame", (xw - 0.55, sy * hwc - 0.1, d_aft + 1.2), (xw + 0.55, sy * hwc + 0.1, d_aft + 1.3),
                             mats["white"], bevel=0.01))
            parts.append(box("CabWinFrameT", (xw - 0.55, sy * hwc - 0.1, d_aft + 2.5), (xw + 0.55, sy * hwc + 0.1, d_aft + 2.6),
                             mats["white"], bevel=0.01))
    parts.append(box("CabRoof", (cab_x0 - 0.25, -hwc - 0.25, d_aft + 3.3), (cab_x0 + cab_len + 0.35, hwc + 0.25, d_aft + 3.6),
                     mats["white"], bevel=0.04))
    for k in range(7):
        parts.append(box("Step", (cab_x0 + cab_len + k * 0.34, -1.3, d_aft), (cab_x0 + cab_len + k * 0.34 + 0.34, 1.3,
                                                                                d_aft + 3.3 - k * 0.47), mats["wood"], bevel=0.02))
    posts = []
    for xx in np.linspace(cab_x0, cab_x0 + cab_len, 12):
        for sy in (-1, 1):
            posts.append(((xx, sy * (hwc + 0.1), d_aft + 3.6), (xx, sy * (hwc + 0.1), d_aft + 4.5)))
    for sy in (-1, 1):
        posts.append(((cab_x0, sy * (hwc + 0.1), d_aft + 4.5), (cab_x0 + cab_len, sy * (hwc + 0.1), d_aft + 4.5)))
    parts.append(tubes_mesh("CabRail", posts, 0.045, mats["white"], sides=8))

    # Masten mit Eisenbändern, Rahen, Gaffel/Baum
    d_main = deck_z(0.45)
    main_x, fore_x = -3.0, 7.0
    main_top, fore_top = d_main + 27.0, d_main + 17.0
    parts.append(cyl("MainMast", (main_x, 0, d_main - 0.5), (main_x, 0, main_top), 0.52, 0.30, mats["mast"], 24))
    parts.append(cyl("ForeMast", (fore_x, 0, d_main - 0.5), (fore_x, 0, fore_top), 0.44, 0.26, mats["mast"], 24))
    for mx, top, r0 in ((main_x, main_top, 0.52), (fore_x, fore_top, 0.44)):
        for zz in np.arange(d_main + 1.5, top - 1.0, 3.0):
            f = (zz - d_main) / (top - d_main)
            rr = r0 + (0.3 - r0) * f
            ring = cyl("MastBand", (mx, 0, zz - 0.08), (mx, 0, zz + 0.08), rr + 0.035, rr + 0.035, mats["iron"], 24)
            parts.append(ring)
    yards = [(main_x, d_main + 21.2, 8.6), (main_x, d_main + 8.8, 9.2), (fore_x, d_main + 12.6, 6.4), (fore_x, d_main + 5.2, 6.8)]
    for (x, z, hw) in yards:
        parts.append(cyl("YardL", (x + 0.45, 0, z), (x + 0.45, -hw, z), 0.24, 0.13, mats["mast"], 12))
        parts.append(cyl("YardR", (x + 0.45, 0, z), (x + 0.45, hw, z), 0.24, 0.13, mats["mast"], 12))
    s_main, e_main = sail_mesh("MainSail", 16.6, 12.2, 2.1, mats["sail_jr"], foot_arch=0.7)
    s_main.location = (main_x + 0.7, 0, d_main + 21.0)
    parts.append(s_main)
    s_fore, e_fore = sail_mesh("ForeSail", 12.2, 7.2, 1.5, mats["sail"], foot_arch=0.5)
    s_fore.location = (fore_x + 0.7, 0, d_main + 12.4)
    parts.append(s_fore)
    # Liektaue entlang der Segelkanten + Schoten zum Deck
    ropes = []
    for (so, ep, loc) in ((s_main, e_main, Vector((main_x + 0.7, 0, d_main + 21.0))),
                          (s_fore, e_fore, Vector((fore_x + 0.7, 0, d_main + 12.4)))):
        for e in ep:
            for a, b in zip(e[:-1], e[1:]):
                ropes.append((tuple(Vector(a) + loc), tuple(Vector(b) + loc)))
        lo = [Vector(ep[1][0]) + loc, Vector(ep[1][-1]) + loc]
        for c in lo:
            u = (c.x + L / 2) / L
            tgt = Vector((c.x - 3.0, np.sign(c.y) * (half_beam(u) - 0.2), sheer(u) + 0.1))
            ropes += catenary(tuple(c), tuple(tgt), 0.15, 8)
    # Gaffelsegel rot-schwarz achtern am Großmast
    gaff = gaff_sail_mesh("GaffSail", luff=10.0, gaff_len=6.5, boom_len=7.2, gaff_rise=2.2, mat=mats["sail_stripe"])
    gaff.location = (main_x - 0.55, 0, d_main + 4.2)
    parts.append(gaff)
    parts.append(cyl("Boom", (main_x - 0.4, 0, d_main + 4.15), (main_x - 7.8, 0, d_main + 4.15), 0.17, 0.12, mats["mast"], 12))
    parts.append(cyl("Gaff", (main_x - 0.4, 0, d_main + 14.2), (main_x - 7.0, 0, d_main + 16.4), 0.15, 0.1, mats["mast"], 12))

    # Ausguck-Haus: Plattform, Geländer, Fensterband mit Rahmen, Kuppel
    cz = d_main + 22.2
    parts.append(cyl("CrowFloor", (main_x, 0, cz - 0.25), (main_x, 0, cz), 2.9, 2.9, mats["wood"], 48))
    parts.append(cyl("CrowRoom", (main_x, 0, cz), (main_x, 0, cz + 2.6), 2.2, 2.2, mats["white"], 48))
    wins = []
    for k in range(12):
        a = k * 2 * math.pi / 12
        d = Vector((math.cos(a), math.sin(a), 0))
        c = Vector((main_x, 0, cz + 1.45)) + d * 2.2
        parts.append(cyl("CrowWin", c - d * 0.08, c + d * 0.05, 0.42, 0.42, mats["glass"], 20))
        parts.append(cyl("CrowWinFrame", c - d * 0.02, c + d * 0.07, 0.52, 0.52, mats["iron"], 20))
    dome = sphere("CrowDome", (main_x, 0, cz + 2.6), 2.25, mats["roofblue"], scale=(1, 1, 0.5), seg=48, ring=24)
    parts.append(dome)
    rail = []
    for k in range(20):
        a = k * 2 * math.pi / 20
        p = (main_x + math.cos(a) * 2.8, math.sin(a) * 2.8)
        rail.append(((p[0], p[1], cz), (p[0], p[1], cz + 1.0)))
        a2 = (k + 1) * 2 * math.pi / 20
        rail.append(((p[0], p[1], cz + 1.0), (main_x + math.cos(a2) * 2.8, math.sin(a2) * 2.8, cz + 1.0)))
    parts.append(tubes_mesh("CrowRail", rail, 0.05, mats["iron"], sides=8))

    # Flagge an der Mastspitze
    bm = bmesh.new()
    uv_l = bm.loops.layers.uv.new("UVMap")
    nx, ny = 24, 16
    fw, fh = 4.4, 2.9
    grid = [[bm.verts.new((-fw * i / (nx - 1), 0, -fh * j / (ny - 1))) for i in range(nx)] for j in range(ny)]
    for j in range(ny - 1):
        for i in range(nx - 1):
            f = bm.faces.new([grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]])
            for lp, (ii, jj) in zip(f.loops, [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]):
                lp[uv_l].uv = (ii / (nx - 1), 1 - jj / (ny - 1))
    flag = fpv.mesh_from_bmesh(bm, "Flag", mats["flag"])
    flag.location = (main_x - 0.2, 0, main_top + 2.7)
    wv = flag.modifiers.new("wave", "WAVE")
    wv.use_x = True
    wv.use_y = False
    wv.use_normal = True
    wv.height = 0.25
    wv.width = 1.6
    wv.speed = -0.12
    parts.append(flag)
    parts.append(cyl("FlagPole", (main_x, 0, main_top), (main_x, 0, main_top + 2.9), 0.08, 0.06, mats["iron"], 8))

    # Takelage: Wanten, Webeleinen, Stagen
    bow_z = sheer(1.0)
    for sy in (-1, 1):
        for (mx, top, u0, n_sh) in ((main_x, main_top, 0.34, 4), (fore_x, fore_top, 0.66, 3)):
            chain = []
            for k in range(n_sh):
                u = u0 + k * 0.03
                a = (-L / 2 + u * L, sy * (half_beam(u) + 0.05), sheer(u) + 0.2)
                b = (mx, 0.35 * sy, top - 1.2)
                ropes.append((a, b))
                chain.append((np.array(a), np.array(b)))
            # Webeleinen alle 0,45 m zwischen benachbarten Wanten
            for (a1, b1), (a2, b2) in zip(chain[:-1], chain[1:]):
                for t in np.arange(0.04, 0.9, 0.45 / np.linalg.norm(b1 - a1)):
                    ropes.append((tuple(a1 + (b1 - a1) * t), tuple(a2 + (b2 - a2) * t)))
    ropes.append(((fore_x, 0, fore_top - 0.5), (main_x, 0, main_top - 0.6)))
    ropes.append(((L / 2 - 0.3, 0, bow_z + 2.0), (fore_x, 0, fore_top - 0.8)))
    rig = tubes_mesh("Rigging", ropes, 0.028, mats["rope"], sides=6)
    parts.append(rig)

    # Galionsfigur
    hub = lion_head(body, mats, L / 2 - 0.4, bow_z - 0.2)

    for p in parts:
        if p.parent is None:
            p.parent = body
    return root, body, mats


def animate(root, body, heading_deg, start, speed, fps, frames, pitch_amp=1.2, roll_amp=2.0, heave_amp=0.25):
    h = math.radians(heading_deg)
    d = Vector((math.cos(h), math.sin(h), 0))
    root.rotation_euler = (0, 0, h)
    body.rotation_mode = "XYZ"
    for f in range(0, frames + 2):
        t = (f - 1) / fps
        root.location = Vector(start) + d * speed * t
        root.keyframe_insert("location", frame=f)
        body.location = (0, 0, heave_amp * math.sin(2 * math.pi * 0.13 * t + 0.4) - 0.15)
        body.rotation_euler = (math.radians(roll_amp) * math.sin(2 * math.pi * 0.085 * t + 1.1),
                               math.radians(pitch_amp) * math.sin(2 * math.pi * 0.11 * t), 0)
        body.keyframe_insert("location", frame=f)
        body.keyframe_insert("rotation_euler", frame=f)
