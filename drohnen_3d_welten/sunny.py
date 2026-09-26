"""Thousand Sunny (One Piece) – prozedurales Modell, realistische Materialien.

Lokales System: +X = Bug, +Y = Backbord, +Z = oben, Wasserlinie z = 0.
Länge ~31 m, Breite ~9,5 m, Großmast ~25 m über Deck.
"""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

import fpv
import textures

L = 30.0
W = 4.7


def sheer(u):
    s = 4.0 + 1.5 * np.clip((u - 0.55) / 0.45, 0, 1) ** 2 + 1.1 * np.clip((0.28 - u) / 0.28, 0, 1) ** 2
    return s


def keel(u):
    return -2.3 + 2.4 * np.clip((u - 0.80) / 0.20, 0, 1) ** 2 + 0.3 * np.clip((0.1 - u) / 0.1, 0, 1)


def half_beam(u):
    hw = W * (0.80 + 0.20 * np.sin(np.pi / 2 * np.clip(u / 0.28, 0, 1)))
    bow = np.clip((u - 0.58) / 0.42, 0, 1)
    hw = hw * np.sqrt(np.clip(1 - bow ** 2.1, 0, 1))
    return hw


def section(v):
    """Querschnitt: 0 = Kiel, 1 = Schandeck. Liefert relative Halbbreite."""
    return 1 - (1 - v) ** 2.3 + 0.04 * np.sin(np.pi * v)


# --------------------------------------------------------------------------
# Materialien
# --------------------------------------------------------------------------

def wood_material(name, tint=(1.0, 0.62, 0.36), dark=1.0, scale=0.25, rough=0.55, lower_dark=True):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    # Planken horizontal: Textur um 90° drehen, Box-Projektion
    m = nb.mapping(co, rot=(0, 0, 0), scale=(scale, scale, scale * 3.0))
    img = nb.image(os.path.join(fpv.ASSETS, "hardwood2_diffuse.jpg"), m, proj="BOX", blend=0.3)
    rgh = nb.image(os.path.join(fpv.ASSETS, "hardwood2_roughness.jpg"), m, colorspace="Non-Color", proj="BOX", blend=0.3)
    bmp = nb.image(os.path.join(fpv.ASSETS, "hardwood2_bump.jpg"), m, colorspace="Non-Color", proj="BOX", blend=0.3)
    col = nb.mix(1.0, nb.out(img, "Color"), tint, blend="MULTIPLY")
    col = nb.vmath("SCALE", col, scale=dark * 1.4)
    # Verwitterung, Salzspuren
    nz = nb.noise(co, scale=0.6, detail=3, rough=0.65)
    grime = nb.ramp(nb.out(nz, "Fac"), [(0.35, (0.55, 0.5, 0.45)), (0.7, (1.1, 1.05, 1.0))])
    col = nb.mix(1.0, col, grime, blend="MULTIPLY")
    rr = nb.math("MULTIPLY_ADD", nb.out(rgh, "Color"), 0.35, rough)
    if lower_dark:
        z = nb.sep(co)[2]
        wl = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(nb.math("ADD", z, nb.math("MULTIPLY", nb.out(nz, "Fac"), 0.4)), wl.inputs["Value"])
        wl.inputs["From Min"].default_value = 0.2
        wl.inputs["From Max"].default_value = 0.9
        col = nb.mix(wl.outputs[0], (0.035, 0.03, 0.025), col)
        rr = nb.mix(wl.outputs[0], 0.25, rr, dtype="FLOAT")
    p = fpv.principled(nb, Base_Color=col, Roughness=rr)
    nb.link(nb.bump(nb.out(bmp, "Color"), strength=0.35, distance=0.02), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def paint_material(name, color, rough=0.45, wear=0.25):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    nz = nb.noise(co, scale=3.0, detail=3, rough=0.7)
    col = nb.mix(nb.math("MULTIPLY", nb.out(nz, "Fac"), wear), color, [c * 0.6 for c in color])
    big = nb.noise(co, scale=0.4, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(big, "Fac"), 0.25), col, [c * 1.15 for c in color])
    p = fpv.principled(nb, Base_Color=col, Roughness=rough)
    p.inputs["Coat Weight"].default_value = 0.15
    nb.link(nb.bump(nb.out(nz, "Fac"), strength=0.08, distance=0.01), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def lawn_material():
    mat, nb, out = fpv.new_material("Lawn")
    co = nb.coords("Object")
    m = nb.mapping(co, scale=(0.35, 0.35, 0.35))
    img = nb.image(os.path.join(fpv.ASSETS, "grasslight-big.jpg"), m)
    nz = nb.noise(co, scale=40, detail=2)
    col = nb.mix(1.0, nb.out(img, "Color"), (0.85, 1.0, 0.7), blend="MULTIPLY")
    p = fpv.principled(nb, Base_Color=col, Roughness=0.8)
    p.inputs["Sheen Weight"].default_value = 0.3
    nb.link(nb.bump(nb.out(nz, "Fac"), strength=0.4, distance=0.02), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def sail_material(name, decal=None):
    mat, nb, out = fpv.new_material(name)
    uv = nb.coords("UV")
    co = nb.coords("Object")
    base = (0.76, 0.71, 0.60)
    # Stoffbahnen (vertikale Nähte) + Verschmutzung
    wave = nb.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X")
    nb.link(uv, wave.inputs["Vector"])
    wave.inputs["Scale"].default_value = 9.0
    wave.inputs["Distortion"].default_value = 0.0
    seams = nb.ramp(nb.out(wave, "Fac"), [(0.0, (0.82, 0.82, 0.82)), (0.06, (1, 1, 1)), (0.94, (1, 1, 1)), (1.0, (0.82, 0.82, 0.82))])
    dirt = nb.noise(co, scale=0.8, detail=3, rough=0.6)
    dcol = nb.ramp(nb.out(dirt, "Fac"), [(0.3, (0.78, 0.74, 0.66)), (0.75, (1.0, 1.0, 1.0))])
    col = nb.mix(1.0, base, seams, blend="MULTIPLY")
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
    mix.inputs[0].default_value = 0.28
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


# --------------------------------------------------------------------------
# Geometrie
# --------------------------------------------------------------------------

def hull_mesh(mat_hull, nu=160, nv=36):
    us = np.linspace(0, 1, nu)
    vs = np.linspace(0, 1, nv)
    verts = []
    for u in us:
        x = -L / 2 + u * L
        hw = half_beam(u)
        k = keel(u)
        s = sheer(u)
        ring = []
        # Steuerbord (y<0) von Schandeck -> Kiel, dann Backbord Kiel -> Schandeck
        for v in vs[::-1]:
            ring.append((x, -hw * section(v), k + (s - k) * v))
        for v in vs[1:]:
            ring.append((x, hw * section(v), k + (s - k) * v))
        verts.append(ring)
    verts = np.array(verts)  # nu x (2nv-1) x 3
    nr = verts.shape[1]
    bm = bmesh.new()
    vv = [[bm.verts.new(tuple(verts[i, j])) for j in range(nr)] for i in range(nu)]
    for i in range(nu - 1):
        for j in range(nr - 1):
            q = [vv[i][j], vv[i + 1][j], vv[i + 1][j + 1], vv[i][j + 1]]
            try:
                bm.faces.new(q)
            except ValueError:
                pass
    # Heckspiegel
    bm.faces.new(vv[0][::-1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.002)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = fpv.mesh_from_bmesh(bm, "Hull", mat_hull)
    sol = ob.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.22
    sol.offset = -1
    ob.modifiers.new("bevel", "WEIGHTED_NORMAL")
    return ob


def deck_mesh(mat, nu=80, nw=14):
    us = np.linspace(0.02, 0.985, nu)
    bm = bmesh.new()
    rows = []
    for u in us:
        x = -L / 2 + u * L
        hw = max(half_beam(u) * section(0.86) - 0.25, 0.02)
        z = sheer(u) - 1.05
        rows.append([bm.verts.new((x, -hw + 2 * hw * t, z)) for t in np.linspace(0, 1, nw)])
    for i in range(nu - 1):
        for j in range(nw - 1):
            bm.faces.new([rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]])
    return fpv.mesh_from_bmesh(bm, "Deck", mat)


def cyl(name, p0, p1, r0, r1=None, mat=None, verts=16, cap=True):
    r1 = r0 if r1 is None else r1
    p0, p1 = Vector(p0), Vector(p1)
    bm = bmesh.new()
    res = bmesh.ops.create_cone(bm, cap_ends=cap, segments=verts, radius1=r0, radius2=r1, depth=(p1 - p0).length)
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


def sail_mesh(name, width, height, bulge, mat, nx=28, ny=28, top_arch=0.3):
    bm = bmesh.new()
    uv_l = bm.loops.layers.uv.new("UVMap")
    rows = []
    for j in range(ny):
        t = j / (ny - 1)
        row = []
        for i in range(nx):
            s = i / (nx - 1)
            y = (s - 0.5) * width * (1 - 0.06 * (1 - t))
            z = -t * height
            # Bauch: parabolisch in beide Richtungen, tiefster Punkt etwas unter der Mitte
            b = bulge * (1 - (2 * s - 1) ** 2) * (1 - (2 * t - 1.1) ** 2 / 1.21)
            b += top_arch * (1 - (2 * s - 1) ** 2) * (1 - t) * 0.0
            row.append(bm.verts.new((b, y, z)))
        rows.append(row)
    for j in range(ny - 1):
        for i in range(nx - 1):
            f = bm.faces.new([rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]])
            for lp, (ii, jj) in zip(f.loops, [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]):
                lp[uv_l].uv = (1 - ii / (nx - 1), 1 - jj / (ny - 1))
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    sub = ob.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 2
    fpv.displace_obj(ob, "CLOUDS", size=1.6, strength=0.12, depth=3, name=name + "_wr")
    sol = ob.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.03
    return ob


def lion_head(parent, mats, x0, z0):
    """Löwen-Galionsfigur mit Sonnenblumen-Mähne."""
    objs = []
    K = 1.45  # Maßstab: Kopf so breit wie der Rumpf
    face_c = Vector((x0 + 1.3, 0, z0 + 2.1))
    # Hals/Sockel am Vorsteven
    objs.append(cyl("LionNeck", (x0 - 2.0, 0, z0 - 0.8), (face_c.x - 0.4, 0, face_c.z - 0.3), 1.7, 2.0, mats["mane_dark"], 32))
    face = sphere("LionFace", face_c, 1.55 * K, mats["face"], scale=(0.78, 1.0, 0.95), seg=64, ring=32)
    objs.append(face)
    # Mähne: zwei Kränze länglicher Blätter in der YZ-Ebene
    for layer, (n, rad, ln, wd, mk, off) in enumerate(((20, 2.05 * K, 1.05 * K, 0.62 * K, "mane", 0.0),
                                                        (20, 2.6 * K, 1.2 * K, 0.72 * K, "mane_dark", 9.0))):
        for k in range(n):
            a = math.radians(off + k * 360 / n)
            dirv = Vector((0, math.cos(a), math.sin(a)))
            c = face_c + dirv * rad + Vector((-0.5 - 0.5 * layer, 0, 0))
            pet = sphere(f"Petal{layer}_{k}", c, 1.0, mats[mk], seg=24, ring=12)
            pet.scale = (0.3 * K, wd, ln)
            pet.rotation_mode = "QUATERNION"
            # lokale Z-Achse zeigt radial nach außen
            q = dirv.to_track_quat("Z", "X")
            pet.rotation_quaternion = q
            objs.append(pet)
    # Augen, Nase, Maul, Augenbrauen
    for sy in (-1, 1):
        e = sphere("Eye", face_c + Vector((1.08, 0.55 * sy, 0.38)) * 1 + Vector((1.08, 0.55 * sy, 0.38)) * (K - 1), 0.22 * K, mats["black"], scale=(0.6, 1, 1.2))
        objs.append(e)
        br = cyl("Brow", face_c + Vector((1.02, 0.25 * sy, 0.78)) * 1 + Vector((1.02, 0.25 * sy, 0.78)) * (K - 1),
                 face_c + Vector((0.98, 0.85 * sy, 0.72)) + Vector((0.98, 0.85 * sy, 0.72)) * (K - 1), 0.07 * K, 0.05 * K, mats["mane_dark"], 8)
        objs.append(br)
    objs.append(sphere("Nose", face_c + Vector((1.25 * K, 0, 0.0)), 0.3 * K, mats["mane_dark"], scale=(0.8, 1.2, 0.8)))
    # Maul als Bogen (Torus-Segment)
    bm = bmesh.new()
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
    objs.append(mo)
    for o in objs:
        o.parent = parent
    return objs


def _box_mesh(name, a, b):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    ax, ay, az = a
    bx, by, bz = b
    bmesh.ops.scale(bm, vec=(bx - ax, by - ay, bz - az), verts=bm.verts)
    bmesh.ops.translate(bm, vec=((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2), verts=bm.verts)
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.04, segments=2, affect="EDGES")
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def rigging(parent, mat, pairs, r=0.035):
    objs = []
    for a, b in pairs:
        o = cyl("Rope", a, b, r, r, mat, 6, cap=False)
        o.parent = parent
        objs.append(o)
    return objs


def build(name="ThousandSunny"):
    root = bpy.data.objects.new(name, None)  # Kurs (nur Gier)
    fpv.link(root)
    body = bpy.data.objects.new(name + "_body", None)  # Stampfen/Rollen/Tauchen
    fpv.link(body)
    body.parent = root

    mats = {
        "hull": wood_material("HullWood", tint=(0.70, 0.46, 0.30), dark=0.62),
        "rail": wood_material("RailWood", tint=(1.0, 0.72, 0.45), dark=1.1, lower_dark=False),
        "mast": wood_material("MastWood", tint=(1.0, 0.70, 0.45), dark=1.0, scale=0.6, lower_dark=False),
        "lawn": lawn_material(),
        "sail": sail_material("Sail"),
        "sail_jr": sail_material("SailJR", textures.jolly_roger_sail()),
        "flag": flag_material(),
        "mane": paint_material("Mane", (0.95, 0.62, 0.08), rough=0.5),
        "mane_dark": paint_material("ManeDark", (0.85, 0.36, 0.04), rough=0.5),
        "face": paint_material("LionFace", (0.98, 0.80, 0.38), rough=0.45),
        "black": fpv.simple_mat("Black", (0.015, 0.012, 0.01), rough=0.3),
        "iron": fpv.simple_mat("Iron", (0.05, 0.05, 0.05), rough=0.45, metal=0.8),
        "rope": fpv.simple_mat("Rope", (0.22, 0.17, 0.11), rough=0.9),
        "glass": fpv.simple_mat("DarkGlass", (0.02, 0.025, 0.03), rough=0.08),
        "white": paint_material("WhitePaint", (0.78, 0.76, 0.72), rough=0.4),
        "blue": paint_material("BluePaint", (0.10, 0.25, 0.55), rough=0.4),
    }
    parts = []
    parts.append(hull_mesh(mats["hull"]))
    parts.append(deck_mesh(mats["lawn"]))

    # Reling-Handlauf (Kurve entlang Schandeck, leicht erhöht)
    for sy in (-1, 1):
        pts = []
        for u in np.linspace(0.0, 0.97, 60):
            pts.append((-L / 2 + u * L, sy * (half_beam(u) - 0.05), sheer(u) + 0.12))
        cu = bpy.data.curves.new("Rail", "CURVE")
        cu.dimensions = "3D"
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for p, c in zip(sp.points, pts):
            p.co = (*c, 1)
        cu.bevel_depth = 0.14
        cu.bevel_resolution = 2
        ro = bpy.data.objects.new("Rail", cu)
        ro.data.materials.append(mats["rail"])
        fpv.link(ro)
        parts.append(ro)
        # Stanzen der Kanonenpforten (runde, dunkle Luken)
        for k in range(7):
            u = 0.22 + k * 0.075
            x = -L / 2 + u * L
            y = sy * (half_beam(u) * section(0.72) + 0.02)
            z = keel(u) + (sheer(u) - keel(u)) * 0.72
            port = cyl("Port", (x, y - sy * 0.25, z), (x, y + sy * 0.06, z), 0.32, 0.32, mats["iron"], 20)
            parts.append(port)
            inner = cyl("PortIn", (x, y - sy * 0.05, z), (x, y + sy * 0.08, z), 0.24, 0.24, mats["black"], 20)
            parts.append(inner)

    # Heckfenster
    for k, yy in enumerate((-2.4, -0.8, 0.8, 2.4)):
        win = cyl("SternWin", (-L / 2 - 0.2, yy, 3.0), (-L / 2 + 0.1, yy, 3.0), 0.55, 0.55, mats["glass"], 24)
        parts.append(win)
        fr = cyl("SternFrame", (-L / 2 - 0.1, yy, 3.0), (-L / 2 + 0.05, yy, 3.0), 0.68, 0.68, mats["white"], 24)
        parts.append(fr)

    # Achterdeck-Haus (Heckaufbau) mit Fenstern und Reling
    d_aft = sheer(0.1) - 1.05
    cab_len, cab_x0 = 7.5, -L / 2 + 0.6
    hwc = half_beam(0.12) * 0.86
    cab = bpy.data.objects.new("AftCabin", _box_mesh("AftCabinMesh", (cab_x0, -hwc, d_aft), (cab_x0 + cab_len, hwc, d_aft + 3.3)))
    fpv.link(cab)
    cab.data.materials.append(mats["rail"])
    parts.append(cab)
    for k in range(4):
        xw = cab_x0 + 1.2 + k * 1.75
        for sy in (-1, 1):
            w = bpy.data.objects.new("CabWin", _box_mesh("CabWinM", (xw - 0.45, sy * hwc - 0.06, d_aft + 1.3), (xw + 0.45, sy * hwc + 0.06, d_aft + 2.5)))
            fpv.link(w)
            w.data.materials.append(mats["glass"])
            parts.append(w)
    roof = bpy.data.objects.new("CabRoof", _box_mesh("CabRoofM", (cab_x0 - 0.2, -hwc - 0.2, d_aft + 3.3), (cab_x0 + cab_len + 0.3, hwc + 0.2, d_aft + 3.55)))
    fpv.link(roof)
    roof.data.materials.append(mats["white"])
    parts.append(roof)
    # Treppe/Stufen vor dem Heckaufbau
    for k in range(6):
        st = bpy.data.objects.new("Step", _box_mesh("StepM", (cab_x0 + cab_len + k * 0.35, -1.2, d_aft), (cab_x0 + cab_len + k * 0.35 + 0.35, 1.2, d_aft + 3.3 - k * 0.55)))
        fpv.link(st)
        st.data.materials.append(mats["rail"])
        parts.append(st)

    # Masten
    d_main = sheer(0.45) - 1.05
    main_x, fore_x = -3.0, 7.0
    main_top = d_main + 26.5
    fore_top = d_main + 16.5
    parts.append(cyl("MainMast", (main_x, 0, d_main - 0.5), (main_x, 0, main_top), 0.5, 0.32, mats["mast"], 20))
    parts.append(cyl("ForeMast", (fore_x, 0, d_main - 0.5), (fore_x, 0, fore_top), 0.42, 0.28, mats["mast"], 20))
    # Rahen
    yards = [(main_x, d_main + 21.2, 8.6), (main_x, d_main + 8.8, 9.2), (fore_x, d_main + 12.6, 6.4), (fore_x, d_main + 5.2, 6.8)]
    for (x, z, hw) in yards:
        parts.append(cyl("Yard", (x + 0.45, -hw, z), (x + 0.45, hw, z), 0.2, 0.2, mats["mast"], 12))
    # Segel (Wind von achtern -> Bauch nach vorn)
    s_main = sail_mesh("MainSail", 16.6, 12.2, 2.0, mats["sail_jr"])
    s_main.location = (main_x + 0.7, 0, d_main + 21.0)
    parts.append(s_main)
    s_fore = sail_mesh("ForeSail", 12.2, 7.2, 1.4, mats["sail"])
    s_fore.location = (fore_x + 0.7, 0, d_main + 12.4)
    parts.append(s_fore)
    # Ausguck (Beobachtungsraum) am Großmast
    parts.append(cyl("CrowNest", (main_x, 0, d_main + 22.0), (main_x, 0, d_main + 24.2), 1.9, 1.9, mats["white"], 32))
    parts.append(cyl("CrowWin", (main_x, 0, d_main + 22.8), (main_x, 0, d_main + 23.6), 1.93, 1.93, mats["glass"], 32, cap=False))
    dome = sphere("CrowDome", (main_x, 0, d_main + 24.2), 1.9, mats["blue"], scale=(1, 1, 0.55))
    parts.append(dome)
    # Flagge an der Mastspitze
    fl = bpy.data.meshes.new("FlagMesh")
    bm = bmesh.new()
    uv_l = bm.loops.layers.uv.new("UVMap")
    nx, ny = 24, 16
    fw, fh = 4.2, 2.8
    grid = [[bm.verts.new((-fw * i / (nx - 1), 0, -fh * j / (ny - 1))) for i in range(nx)] for j in range(ny)]
    for j in range(ny - 1):
        for i in range(nx - 1):
            f = bm.faces.new([grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]])
            for lp, (ii, jj) in zip(f.loops, [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]):
                lp[uv_l].uv = (ii / (nx - 1), 1 - jj / (ny - 1))
    flag = fpv.mesh_from_bmesh(bm, "Flag", mats["flag"])
    flag.location = (main_x - 0.2, 0, main_top + 2.6)
    wv = flag.modifiers.new("wave", "WAVE")
    wv.use_x = True
    wv.use_y = False
    wv.use_normal = True
    wv.height = 0.25
    wv.width = 1.6
    wv.speed = -0.12
    wv.start_position_x = 0.0
    wv.falloff_radius = 0.0
    parts.append(flag)
    parts.append(cyl("FlagPole", (main_x, 0, main_top), (main_x, 0, main_top + 2.8), 0.08, 0.06, mats["iron"], 8))

    # Galionsfigur
    bow_z = sheer(1.0)
    lion = lion_head(body, mats, L / 2 - 0.4, bow_z - 0.2)

    # Takelage (Wanten zu den Mastspitzen)
    pairs = []
    for sy in (-1, 1):
        for k in range(4):
            u = 0.33 + k * 0.035
            pairs.append(((-L / 2 + u * L, sy * half_beam(u), sheer(u) + 0.2), (main_x, 0.35 * sy, main_top - 1.0)))
        for k in range(3):
            u = 0.66 + k * 0.03
            pairs.append(((-L / 2 + u * L, sy * half_beam(u), sheer(u) + 0.2), (fore_x, 0.3 * sy, fore_top - 1.0)))
    pairs.append(((fore_x, 0, fore_top - 0.5), (main_x, 0, main_top - 0.5)))
    pairs.append(((L / 2 - 0.3, 0, bow_z + 1.8), (fore_x, 0, fore_top - 0.8)))
    rigging(body, mats["rope"], pairs)

    for p in parts:
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
                               math.radians(pitch_amp) * math.sin(2 * math.pi * 0.11 * t),
                               0)
        body.keyframe_insert("location", frame=f)
        body.keyframe_insert("rotation_euler", frame=f)
