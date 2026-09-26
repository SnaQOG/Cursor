"""Konohagakure (Naruto): Gebäude, Tor, Hokage-Residenz, Hokage-Felsen."""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

import fpv
import nature
import textures


# --------------------------------------------------------------------------
# Materialien
# --------------------------------------------------------------------------

def plaster_material(name="Plaster", window=True):
    """Putzwände; Farbe variiert pro Objekt, Fenster als Raster in Weltkoordinaten."""
    mat, nb, out = fpv.new_material(name)
    oi = nb.node("ShaderNodeObjectInfo")
    rnd = nb.out(oi, "Random")
    col = nb.ramp(rnd, [(0.0, (0.55, 0.50, 0.41)), (0.2, (0.62, 0.60, 0.55)), (0.4, (0.50, 0.44, 0.36)),
                        (0.55, (0.60, 0.53, 0.42)), (0.7, (0.42, 0.45, 0.42)), (0.82, (0.55, 0.42, 0.34)),
                        (1.0, (0.64, 0.62, 0.58))], interp="CONSTANT")
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    nrm = nb.out(nb.node("ShaderNodeNewGeometry"), "Normal")
    n1 = nb.noise(wpos, scale=0.25, detail=3, rough=0.6)
    n2 = nb.noise(wpos, scale=3.0, detail=2, rough=0.6)
    dirt = nb.ramp(nb.out(n1, "Fac"), [(0.3, (0.72, 0.68, 0.62)), (0.7, (1.0, 1.0, 1.0))])
    col = nb.mix(1.0, col, dirt, blend="MULTIPLY")
    # Schmutzfahnen nach unten (Regen)
    x, y, z = nb.sep(wpos)
    streak = nb.noise(nb.comb(nb.math("ADD", x, y), nb.math("MULTIPLY", z, 0.08), 0.0), scale=1.5, detail=2)
    sm = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(streak, "Fac"), sm.inputs["Value"])
    sm.inputs["From Min"].default_value = 0.55
    sm.inputs["From Max"].default_value = 0.75
    col = nb.mix(nb.math("MULTIPLY", sm.outputs[0], 0.35), col, nb.vmath("SCALE", col, scale=0.6))
    rough = 0.85
    base_col = col
    if window:
        nx, ny, nz = nb.sep(nrm)
        vert = nb.math("LESS_THAN", nb.math("ABSOLUTE", nz), 0.3)
        # u-Koordinate entlang der Wand
        u = nb.math("ADD", nb.math("MULTIPLY", x, nb.math("ABSOLUTE", ny)), nb.math("MULTIPLY", y, nb.math("ABSOLUTE", nx)))
        u = nb.math("ADD", u, nb.math("MULTIPLY", rnd, 17.0))
        fu = nb.math("FRACT", nb.math("DIVIDE", u, 3.1))
        fz = nb.math("FRACT", nb.math("DIVIDE", nb.math("SUBTRACT", z, 0.4), 3.2))
        wu = nb.math("MULTIPLY", nb.math("GREATER_THAN", fu, 0.28), nb.math("LESS_THAN", fu, 0.72))
        wz = nb.math("MULTIPLY", nb.math("GREATER_THAN", fz, 0.32), nb.math("LESS_THAN", fz, 0.78))
        above = nb.math("GREATER_THAN", z, 2.6)
        # nicht jedes Fenster (Zufallsmuster)
        cell = nb.comb(nb.math("FLOOR", nb.math("DIVIDE", u, 3.1)), nb.math("FLOOR", nb.math("DIVIDE", z, 3.2)), rnd)
        wr = nb.node("ShaderNodeTexWhiteNoise")
        wr.noise_dimensions = "3D"
        nb.link(cell, wr.inputs["Vector"])
        keep = nb.math("GREATER_THAN", nb.out(wr, "Value"), 0.25)
        win = nb.math("MULTIPLY", nb.math("MULTIPLY", wu, wz), nb.math("MULTIPLY", above, nb.math("MULTIPLY", vert, keep)))
        # Rahmen
        fr_u = nb.math("MULTIPLY", nb.math("GREATER_THAN", fu, 0.24), nb.math("LESS_THAN", fu, 0.76))
        fr_z = nb.math("MULTIPLY", nb.math("GREATER_THAN", fz, 0.28), nb.math("LESS_THAN", fz, 0.82))
        frame = nb.math("MULTIPLY", nb.math("MULTIPLY", fr_u, fr_z), nb.math("MULTIPLY", above, nb.math("MULTIPLY", vert, keep)))
        glass = nb.mix(nb.out(wr, "Value"), (0.02, 0.025, 0.03), (0.08, 0.07, 0.05))
        col = nb.mix(frame, col, (0.22, 0.13, 0.08))
        col = nb.mix(win, col, glass)
        # Erdgeschoss: Ladenfronten/Türen (dunkle Öffnungen mit Holzrahmen)
        fu2 = nb.math("FRACT", nb.math("DIVIDE", u, 5.2))
        shop_u = nb.math("MULTIPLY", nb.math("GREATER_THAN", fu2, 0.12), nb.math("LESS_THAN", fu2, 0.82))
        shop_z = nb.math("MULTIPLY", nb.math("GREATER_THAN", z, 0.15), nb.math("LESS_THAN", z, 2.45))
        shop = nb.math("MULTIPLY", nb.math("MULTIPLY", shop_u, shop_z), vert)
        sfr_u = nb.math("MULTIPLY", nb.math("GREATER_THAN", fu2, 0.09), nb.math("LESS_THAN", fu2, 0.85))
        sfr = nb.math("MULTIPLY", nb.math("MULTIPLY", sfr_u, nb.math("LESS_THAN", z, 2.65)), vert)
        col = nb.mix(sfr, col, (0.16, 0.09, 0.05))
        col = nb.mix(shop, col, (0.035, 0.03, 0.025))
        # Spritzschmutz am Fuß
        foot = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(z, foot.inputs["Value"])
        foot.inputs["From Min"].default_value = 1.2
        foot.inputs["From Max"].default_value = 0.0
        col = nb.mix(nb.math("MULTIPLY", foot.outputs[0], 0.5), col, (0.12, 0.1, 0.08))
        rough = nb.mix(win, 0.85, 0.08, dtype="FLOAT")
    p = fpv.principled(nb, Base_Color=col, Roughness=rough)
    nb.link(nb.bump(nb.out(n2, "Fac"), strength=0.25, distance=0.02), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def roof_material(name, color, var=0.25):
    """Dachziegel: Wellenband + Versatz, leichte Verwitterung."""
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    oi = nb.node("ShaderNodeObjectInfo")
    rnd = nb.out(oi, "Random")
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    wv = nb.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X", wave_profile="SAW")
    nb.link(co, wv.inputs["Vector"])
    wv.inputs["Scale"].default_value = 1.2
    wv.inputs["Distortion"].default_value = 0.0
    wv2 = nb.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Y", wave_profile="SIN")
    nb.link(co, wv2.inputs["Vector"])
    wv2.inputs["Scale"].default_value = 2.2
    n = nb.noise(wpos, scale=0.4, detail=3)
    c = nb.mix(nb.math("MULTIPLY", rnd, var), color, [v * 0.7 for v in color])
    c = nb.mix(nb.math("MULTIPLY", nb.out(n, "Fac"), 0.5), c, [v * 0.55 for v in color])
    dirt = nb.noise(wpos, scale=2.0, detail=2)
    c = nb.mix(nb.math("MULTIPLY", nb.out(dirt, "Fac"), 0.3), c, (0.15, 0.13, 0.11))
    p = fpv.principled(nb, Base_Color=c, Roughness=0.55)
    h = nb.math("ADD", nb.out(wv, "Fac"), nb.math("MULTIPLY", nb.out(wv2, "Fac"), 0.5))
    nb.link(nb.bump(h, strength=0.5, distance=0.06), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def ground_material():
    mat, nb, out = fpv.new_material("Street")
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    n1 = nb.noise(wpos, scale=0.08, detail=3)
    n2 = nb.noise(wpos, scale=1.2, detail=3)
    vor = nb.voronoi(nb.vmath("SCALE", wpos, scale=1.0), scale=3.2, feature="DISTANCE_TO_EDGE")
    edge = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(vor, "Distance"), edge.inputs["Value"])
    edge.inputs["From Min"].default_value = 0.0
    edge.inputs["From Max"].default_value = 0.06
    c = nb.mix(nb.out(n1, "Fac"), (0.24, 0.20, 0.15), (0.38, 0.32, 0.24))
    c = nb.mix(nb.math("MULTIPLY", nb.math("SUBTRACT", 1, edge.outputs[0]), 0.22), c, (0.12, 0.10, 0.08))
    c = nb.mix(nb.math("MULTIPLY", nb.out(n2, "Fac"), 0.3), c, (0.14, 0.12, 0.09))
    p = fpv.principled(nb, Base_Color=c, Roughness=0.9)
    nb.link(nb.bump(nb.math("ADD", nb.math("MULTIPLY", edge.outputs[0], 0.4), nb.math("MULTIPLY", nb.out(n2, "Fac"), 0.5)), strength=0.3,
                    distance=0.05), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def image_material(name, path, rough=0.6, alpha=False, emission=0.0):
    mat, nb, out = fpv.new_material(name)
    uv = nb.coords("UV")
    img = nb.image(path, uv)
    img.extension = "CLIP"
    p = fpv.principled(nb, Base_Color=nb.out(img, "Color"), Roughness=rough)
    if alpha:
        nb.link(nb.out(img, "Alpha"), p.inputs["Alpha"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


# --------------------------------------------------------------------------
# Geometrie-Helfer
# --------------------------------------------------------------------------

def box(name, cx, cy, z0, w, d, h, mat, rot=0.0, bevel=0.05):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(w, d, h), verts=bm.verts)
    bmesh.ops.translate(bm, vec=(0, 0, h / 2), verts=bm.verts)
    if bevel:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=1, affect="EDGES")
    ob = fpv.mesh_from_bmesh(bm, name, mat, smooth=False)
    ob.location = (cx, cy, z0)
    ob.rotation_euler = (0, 0, rot)
    return ob


def cylinder(name, cx, cy, z0, r, h, mat, seg=32, r_top=None, cap=True):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=cap, segments=seg, radius1=r, radius2=r if r_top is None else r_top, depth=h)
    bmesh.ops.translate(bm, vec=(0, 0, h / 2), verts=bm.verts)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    ob.location = (cx, cy, z0)
    return ob


def barrel_roof(name, cx, cy, z0, w, d, rise, mat, rot=0.0, seg=16):
    """Tonnendach über w (quer) und d (längs)."""
    bm = bmesh.new()
    prof = []
    for i in range(seg + 1):
        a = math.pi * i / seg
        prof.append((-math.cos(a) * w / 2 * 1.04, math.sin(a) * rise))
    vs0 = [bm.verts.new((x, -d / 2 - 0.3, z)) for x, z in prof]
    vs1 = [bm.verts.new((x, d / 2 + 0.3, z)) for x, z in prof]
    for i in range(seg):
        bm.faces.new([vs0[i], vs0[i + 1], vs1[i + 1], vs1[i]])
    bm.faces.new(vs0[::-1])
    bm.faces.new(vs1)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    sol = ob.modifiers.new("s", "SOLIDIFY")
    sol.thickness = 0.25
    ob.location = (cx, cy, z0)
    ob.rotation_euler = (0, 0, rot)
    return ob


def hip_roof(name, cx, cy, z0, w, d, rise, mat, rot=0.0, overhang=0.8):
    w2, d2 = w / 2 + overhang, d / 2 + overhang
    ridge = max(d2 - w2, 0.2)
    verts = [(-w2, -d2, 0), (w2, -d2, 0), (w2, d2, 0), (-w2, d2, 0), (0, -ridge, rise), (0, ridge, rise)]
    faces = [(0, 1, 4), (1, 2, 5, 4), (2, 3, 5), (3, 0, 4, 5)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new(name, me)
    me.materials.append(mat)
    fpv.link(ob)
    sol = ob.modifiers.new("s", "SOLIDIFY")
    sol.thickness = 0.3
    ob.location = (cx, cy, z0)
    ob.rotation_euler = (0, 0, rot)
    return ob


def curve_tube(name, pts, radius, mat, res=2):
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


# --------------------------------------------------------------------------
# Wassertank (Instanz)
# --------------------------------------------------------------------------

def water_tank_collection(mat_tank, mat_leg, mat_cap):
    coll = bpy.data.collections.new("Tanks")
    bpy.context.scene.collection.children.link(coll)
    objs = []
    for i, (r, h) in enumerate(((1.4, 2.6), (1.8, 3.2), (1.1, 2.2))):
        sub = bpy.data.collections.new(f"Tank{i}")
        coll.children.link(sub)
        body = cylinder(f"TankBody{i}", 0, 0, 1.2, r, h, mat_tank, seg=28)
        cap = cylinder(f"TankCap{i}", 0, 0, 1.2 + h, r * 1.05, r * 0.45, mat_cap, seg=28, r_top=0.15)
        for o in (body, cap):
            bpy.context.scene.collection.objects.unlink(o)
            sub.objects.link(o)
        for k in range(4):
            a = k * math.pi / 2 + 0.6
            leg = cylinder(f"TankLeg{i}_{k}", math.cos(a) * r * 0.75, math.sin(a) * r * 0.75, 0, 0.09, 1.25, mat_leg, seg=6)
            bpy.context.scene.collection.objects.unlink(leg)
            sub.objects.link(leg)
        # Ringe
        for k in range(2):
            ring = cylinder(f"TankRing{i}_{k}", 0, 0, 1.2 + h * (0.3 + 0.4 * k), r * 1.02, 0.12, mat_leg, seg=28)
            bpy.context.scene.collection.objects.unlink(ring)
            sub.objects.link(ring)
        objs.append(sub)
    coll.hide_render = True
    coll.hide_viewport = True
    return objs


def place_instance(sub, loc, rot=0.0, scale=1.0, name="Inst"):
    e = bpy.data.objects.new(name, None)
    e.instance_type = "COLLECTION"
    e.instance_collection = sub
    e.location = loc
    e.rotation_euler = (0, 0, rot)
    e.scale = (scale, scale, scale)
    fpv.link(e)
    return e


# --------------------------------------------------------------------------
# Gebäude
# --------------------------------------------------------------------------

def building(rng, mats, tanks, cx, cy, w, d, floors, rot, idx, style=None, front=None):
    h = floors * 3.2 + rng.uniform(-0.3, 0.6)
    style = style or rng.choice(["flat", "barrel", "hip", "round", "flat", "barrel"], p=None)
    objs = []
    if style == "round":
        r = min(w, d) / 2
        objs.append(cylinder(f"B{idx}", cx, cy, 0, r, h, mats["plaster"], seg=40))
        rm = mats["roofs"][rng.integers(len(mats["roofs"]))]
        objs.append(cylinder(f"B{idx}r", cx, cy, h, r * 1.12, r * 0.55, rm, seg=40, r_top=r * 0.2))
        return objs, h + r * 0.55
    objs.append(box(f"B{idx}", cx, cy, 0, w, d, h, mats["plaster"], rot=rot))
    c, s = math.cos(rot), math.sin(rot)
    top = h
    if style == "flat":
        # Brüstung + Dachfläche
        objs.append(box(f"B{idx}p", cx, cy, h, w + 0.3, d + 0.3, 0.7, mats["parapet"], rot=rot))
        objs.append(box(f"B{idx}f", cx, cy, h, w - 0.2, d - 0.2, 0.75, mats["rooftop"], rot=rot, bevel=0))
        top = h + 0.7
        if rng.random() < 0.8:
            nt = 1 if rng.random() < 0.6 else 2
            for k in range(nt):
                ox, oy = rng.uniform(-w / 4, w / 4), rng.uniform(-d / 4, d / 4)
                place_instance(tanks[rng.integers(len(tanks))], (cx + ox * c - oy * s, cy + ox * s + oy * c, h + 0.7),
                               rot=rng.uniform(0, 6.28), scale=rng.uniform(0.85, 1.2), name=f"T{idx}_{k}")
            top = h + 5.5
        if rng.random() < 0.3:
            # kleiner Dachaufbau
            objs.append(box(f"B{idx}k", cx + rng.uniform(-w / 5, w / 5), cy, h + 0.7, w * 0.35, d * 0.35, 2.4,
                            mats["plaster"], rot=rot))
    elif style == "barrel":
        rm = mats["roofs"][rng.integers(len(mats["roofs"]))]
        rise = min(w, d) * 0.28
        if w > d:
            objs.append(barrel_roof(f"B{idx}r", cx, cy, h, d, w, rise, rm, rot=rot + math.pi / 2))
        else:
            objs.append(barrel_roof(f"B{idx}r", cx, cy, h, w, d, rise, rm, rot=rot))
        top = h + rise
    elif style == "hip":
        rm = mats["roofs"][rng.integers(len(mats["roofs"]))]
        rise = min(w, d) * 0.35
        if w > d:
            objs.append(hip_roof(f"B{idx}r", cx, cy, h, d, w, rise, rm, rot=rot + math.pi / 2))
        else:
            objs.append(hip_roof(f"B{idx}r", cx, cy, h, w, d, rise, rm, rot=rot))
        top = h + rise
    # Fassade zur Straße: Markise, Balkone, Schild
    if front is None:
        fx, fy, fw, fd = -s, -c, w, d  # lokale -Y-Seite
        half = d / 2
        frot = rot
    else:
        fx, fy = front
        fw = d if abs(fx) > abs(fy) else w  # Fassadenbreite
        half = (w if abs(fx) > abs(fy) else d) / 2
        frot = math.atan2(fx, -fy)
    if rng.random() < 0.55:
        aw = mats["awnings"][rng.integers(len(mats["awnings"]))]
        objs.append(box(f"B{idx}a", cx + fx * (half + 0.9), cy + fy * (half + 0.9), 2.9, fw * 0.8, 1.8, 0.15, aw,
                        rot=frot, bevel=0))
    if floors >= 2 and rng.random() < 0.6:
        for k in range(1, floors):
            if rng.random() < 0.35:
                continue
            bw = fw * rng.uniform(0.3, 0.6)
            off = rng.uniform(-fw / 4, fw / 4)
            px_, py_ = cx + fx * (half + 0.6) + fy * off * -1, cy + fy * (half + 0.6) + fx * off
            objs.append(box(f"B{idx}bal{k}", px_, py_, 3.2 * k + 0.05, bw, 1.2, 0.18, mats["parapet"], rot=frot, bevel=0.02))
            objs.append(box(f"B{idx}rail{k}", cx + fx * (half + 1.15) + fy * off * -1, cy + fy * (half + 1.15) + fx * off,
                            3.2 * k + 0.23, bw, 0.08, 0.95, mats["rail"], rot=frot, bevel=0))
    if rng.random() < 0.35 and h > 6:
        sg = mats["signs"][rng.integers(len(mats["signs"]))]
        off = rng.choice([-1, 1]) * fw * 0.38
        objs.append(box(f"B{idx}sign", cx + fx * (half + 0.9) + fy * off * -1, cy + fy * (half + 0.9) + fx * off,
                        3.6, 0.25, 1.6, rng.uniform(2.4, 4.0), sg, rot=frot, bevel=0.02))
    return objs, top


# --------------------------------------------------------------------------
# Tor (A-un-Tor)
# --------------------------------------------------------------------------

def gate(mats, y=0.0, half_open=9.5, height=18.0):
    objs = []
    pw = 3.2
    for sx in (-1, 1):
        x = sx * (half_open + pw / 2)
        objs.append(box(f"GatePillar{sx}", x, y, 0, pw, pw, height + 4, mats["gate"], bevel=0.08))
        objs.append(box(f"GateBase{sx}", x, y, 0, pw + 0.8, pw + 0.8, 1.6, mats["stone"], bevel=0.1))
    # Querbalken
    objs.append(box("GateBeam", 0, y, height, 2 * (half_open + pw) + 4, pw * 0.9, 2.6, mats["gate"], bevel=0.08))
    objs.append(box("GateBeam2", 0, y, height - 3.0, 2 * half_open, pw * 0.6, 1.2, mats["gate"], bevel=0.05))
    objs.append(hip_roof("GateRoof", 0, y, height + 2.6, pw * 1.5, 2 * (half_open + pw) + 6, 3.2, mats["gate_roof"],
                         rot=math.pi / 2, overhang=1.2))
    # Tore (offen, nach innen geschwenkt)
    door_w, door_h = half_open, height - 0.4
    for sx, ch in ((-1, "あ"), (1, "ん")):
        path = os.path.join(textures.OUT, f"door_{'a' if sx < 0 else 'n'}.png")
        if not os.path.exists(path):
            textures.kanji_disc(ch, path, fg=(170, 25, 18), bg=(40, 78, 52), ring=(150, 22, 16))
        me = bpy.data.meshes.new(f"Door{sx}")
        bm = bmesh.new()
        uvl = bm.loops.layers.uv.new("UVMap")
        t = 0.7
        # Platte als Quader mit UV auf der Vorderseite
        res = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(door_w, t, door_h), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(-sx * door_w / 2, 0, door_h / 2), verts=bm.verts)
        xmin = 0.0 if sx < 0 else -door_w
        for f in bm.faces:
            for lp in f.loops:
                co = lp.vert.co
                u = (co.x - xmin) / door_w
                u = (u - 0.5) / 0.62 + 0.5
                v = (co.z / door_h - 0.55) / (0.62 * door_w / door_h) + 0.5
                lp[uvl].uv = (u, v)
        bm.to_mesh(me)
        bm.free()
        door = bpy.data.objects.new(f"Door{sx}", me)
        fpv.link(door)
        me.materials.append(mats["door_" + ("a" if sx < 0 else "n")])
        door.location = (sx * half_open, y + 0.2, 0.3)
        door.rotation_euler = (0, 0, sx * math.radians(-78))
        objs.append(door)
    return objs


def door_material(name, path):
    mat, nb, out = fpv.new_material(name)
    uv = nb.coords("UV")
    img = nb.image(path, uv)
    img.extension = "EXTEND"
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    n = nb.noise(wpos, scale=0.8, detail=3)
    # Holzplanken senkrecht
    wv = nb.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X")
    nb.link(nb.coords("Object"), wv.inputs["Vector"])
    wv.inputs["Scale"].default_value = 0.9
    base = nb.mix(nb.out(n, "Fac"), (0.05, 0.12, 0.07), (0.09, 0.18, 0.10))
    inside = nb.math("MULTIPLY", nb.math("GREATER_THAN", nb.sep(uv)[0], 0.0), nb.math("LESS_THAN", nb.sep(uv)[0], 1.0))
    inside = nb.math("MULTIPLY", inside, nb.math("MULTIPLY", nb.math("GREATER_THAN", nb.sep(uv)[1], 0.0),
                                                 nb.math("LESS_THAN", nb.sep(uv)[1], 1.0)))
    c = nb.mix(inside, base, nb.out(img, "Color"))
    c = nb.mix(nb.math("MULTIPLY", nb.out(wv, "Fac"), 0.15), c, (0.02, 0.03, 0.02))
    p = fpv.principled(nb, Base_Color=c, Roughness=0.6)
    nb.link(nb.bump(nb.out(wv, "Fac"), strength=0.3, distance=0.03), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


# --------------------------------------------------------------------------
# Hokage-Residenz
# --------------------------------------------------------------------------

def residence(mats, cx, cy, r=20.0, h=24.0):
    objs = []
    objs.append(cylinder("ResBase", cx, cy, 0, r + 1.5, 2.0, mats["stone"], seg=64))
    objs.append(cylinder("ResBody", cx, cy, 2.0, r, h, mats["red"], seg=64))
    # Fensterbänder
    for k in range(3):
        objs.append(cylinder(f"ResWin{k}", cx, cy, 6 + k * 6.5, r + 0.08, 1.8, mats["glass"], seg=64, cap=False))
    # Dach: Kegel + Rand
    R1, R2, RH = r + 2.4, r * 0.35, 12.0
    objs.append(cylinder("ResEave", cx, cy, h + 2.0, r + 3.0, 1.0, mats["red_dark"], seg=64))
    objs.append(cylinder("ResRoof", cx, cy, h + 3.0, R1, RH, mats["red_roof"], seg=64, r_top=R2))
    objs.append(cylinder("ResTop", cx, cy, h + 3.0 + RH, R2 * 1.05, 3.0, mats["red_dark"], seg=48))
    # 火-Emblem vorn am Dach (Richtung Süden, -Y), fast senkrecht montiert
    disc_r = 5.0
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    bmesh.ops.create_circle(bm, cap_ends=True, segments=48, radius=disc_r)
    for f in bm.faces:
        for lp in f.loops:
            lp[uvl].uv = (lp.vert.co.x / (2 * disc_r) + 0.5, lp.vert.co.y / (2 * disc_r) + 0.5)
    me = bpy.data.meshes.new("HiDisc")
    bm.to_mesh(me)
    bm.free()
    disc = bpy.data.objects.new("HiDisc", me)
    fpv.link(disc)
    me.materials.append(mats["hi"])
    f = 0.25
    rr = R1 - (R1 - R2) * f
    disc.location = (cx, cy - rr - 0.8, h + 3.0 + RH * f + disc_r * 0.75)
    disc.rotation_euler = (math.radians(78), 0, 0)
    objs.append(disc)
    rim = cylinder("HiRim", 0, 0, -0.25, disc_r * 1.08, 0.24, mats["red_dark"], seg=48)
    rim.parent = disc
    rim.location = (0, 0, -0.26)
    objs.append(rim)
    return objs


# --------------------------------------------------------------------------
# Hokage-Felsen
# --------------------------------------------------------------------------

def cliff_mesh(name, x0, x1, y_face, z0, z1, mat, seed=5, res=0.8, panel=None, chin=None):
    """Senkrechte Felswand (displaced grid) + Plateau dahinter."""
    nx = int((x1 - x0) / res)
    nz = int((z1 - z0) / res)
    xs = np.linspace(x0, x1, nx)
    zs = np.linspace(z0, z1, nz)
    X, Z = np.meshgrid(xs, zs)
    n1 = fpv.fbm2(X / 60, Z / 60, octaves=6, seed=seed)
    n2 = fpv.fbm2(X / 12, Z / 12, octaves=5, seed=seed + 1)
    strata = np.sin(Z / 3.1 + 2 * fpv.fbm2(X / 40, Z / 40, 3, seed + 2)) * 0.8
    # senkrechte Klüfte (Säulen) und waagerechte Bänke -> blockige Felsfacetten
    jit = fpv.fbm2(X / 50, Z / 35, 3, seed + 3)
    col_id = np.floor(X / 9.0 + 1.5 * jit)
    col_off = fpv._value_noise(col_id * 1.37, col_id * 0.0 + 0.5, seed + 4)
    bench_id = np.floor(Z / 7.5 + 0.8 * fpv.fbm2(X / 30, Z / 30, 2, seed + 5))
    bench_off = fpv._value_noise(col_id * 0.71 + 3.3, bench_id * 1.13, seed + 6)
    strata = strata + 7.0 * col_off + 3.5 * bench_off
    if panel is not None:
        px0, px1, pz0, pz1 = panel
        mx = np.clip(np.minimum(X - px0, px1 - X) / 30.0, 0, 1)
        mz = np.clip(np.minimum(Z - pz0, pz1 - Z) / 22.0, 0, 1)
        m = mx * mz
        m = m * m * (3 - 2 * m)
    else:
        m = np.zeros_like(X)
    Y = y_face + (8 * n1 + 3.0 * n2) * (1 - 0.8 * m) + strata * (1 - 0.5 * m) + 4 * m
    if chin is not None and panel is not None:
        # unterhalb der Kinne tritt der Fels vor (verdeckt die Schultern des Scans)
        mxc = np.clip(np.minimum(X - panel[0], panel[1] - X) / 30.0, 0, 1)
        cz = np.clip((chin - Z) / 10.0, 0, 1)
        cz = cz * cz * (3 - 2 * cz)
        Y = Y - 13.0 * cz * mxc
    # unregelmäßige Oberkante
    topn = fpv.fbm2(X / 70, np.zeros_like(X) + seed, octaves=4, seed=seed + 11)
    Z = Z + np.clip((Z - (z1 - 40)) / 40, 0, 1) * 22 * topn
    # Oberkante unregelmäßig abrunden (zurückweichend)
    topw = np.clip((Z - (z1 - 18)) / 18, 0, 1)
    Y = Y + topw ** 2 * 25
    # Fuß: Schuttkegel nach vorn
    footw = np.clip(1 - (Z - z0) / 25, 0, 1)
    Y = Y - footw ** 2 * 18
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    idx = np.arange(nx * nz).reshape(nz, nx)
    faces = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], axis=1)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(faces.size)
    me.loops.foreach_set("vertex_index", faces[:, ::-1].astype(np.int32).ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", (np.arange(len(faces)) * 4).astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    me.update()
    me.validate()
    ob = bpy.data.objects.new(name, me)
    me.materials.append(mat)
    fpv.link(ob)
    return ob


def import_head(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    head = [o for o in new if o.type == "MESH"][0]
    for o in new:
        if o is not head:
            bpy.data.objects.remove(o)
    return head


def hokage_head(template, name, loc, scale, mat, hair, mat_hair, rng):
    """Kopf aus dem Scan (Lee Perry-Smith, three.js-Beispiel) + stilisierte Haare aus Fels."""
    ob = template.copy()
    ob.data = template.data
    ob.name = name
    fpv.link(ob)
    ob.hide_render = False
    ob.hide_viewport = False
    sub_ = ob.modifiers.new("sub", "SUBSURF")
    sub_.levels = sub_.render_levels = 1
    ob.location = loc
    ob.scale = (scale, scale, scale)
    ob.material_slots[0].link = "OBJECT"
    ob.material_slots[0].material = mat
    parts = [ob]
    L = Vector(loc)
    s = scale

    def blob(nm, off, size, rot=(0, 0, 0)):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
        o = fpv.mesh_from_bmesh(bm, nm, mat_hair)
        o.location = L + Vector(off) * s
        o.scale = tuple(v * s for v in size)
        o.rotation_euler = rot
        fpv.displace_obj(o, "CLOUDS", size=0.5, strength=0.12, depth=3, name=nm + "_d")
        parts.append(o)
        return o

    def spike(nm, base, tip, r):
        o = cyl_cone(nm, L + Vector(base) * s, L + Vector(tip) * s, r * s, mat_hair)
        parts.append(o)
        return o

    # Maße des Scans (Einheiten): Kopfbreite ~3.6, Stirn vorn y=-2.2 bei z=2.5, Scheitel z=3.97
    if hair == "hashirama":  # lang, glatt, Mittelscheitel
        blob(name + "_cap", (0, 0.45, 2.95), (1.98, 2.35, 1.3))
        for sx in (-1, 1):
            blob(name + f"_long{sx}", (sx * 1.95, 0.35, 0.7), (0.5, 1.5, 2.5))
            spike(name + f"_fr{sx}", (sx * 0.5, -1.9, 3.3), (sx * 1.7, -1.7, 1.6), 0.45)
    elif hair == "tobirama":  # kurz, stachelig
        blob(name + "_cap", (0, 0.5, 2.9), (1.95, 2.3, 1.3))
        for k in range(9):
            a = math.radians(-80 + k * 20)
            spike(name + f"_sp{k}", (math.sin(a) * 1.2, -0.2, 3.4 + math.cos(a) * 0.2),
                  (math.sin(a) * 2.5, -0.5, 4.3 + math.cos(a) * 0.8), 0.45)
    elif hair == "hiruzen":  # zurückweichender Haaransatz, Spitzbart
        blob(name + "_cap", (0, 1.0, 2.7), (1.9, 1.9, 1.35))
        spike(name + "_beard", (0, -1.2, -0.9), (0, -1.45, -1.9), 0.3)
        for sx in (-1, 1):
            blob(name + f"_side{sx}", (sx * 1.8, 0.6, 1.3), (0.4, 1.0, 0.9))
    elif hair == "minato":  # Stachelhaar + lange Strähnen seitlich
        blob(name + "_cap", (0, 0.4, 3.0), (2.0, 2.35, 1.35))
        for k in range(11):
            a = math.radians(-95 + k * 19)
            spike(name + f"_sp{k}", (math.sin(a) * 1.3, -0.1, 3.3 + math.cos(a) * 0.3),
                  (math.sin(a) * 3.1, -0.6, 3.9 + math.cos(a) * 1.4), 0.55)
        for sx in (-1, 1):
            spike(name + f"_bang{sx}", (sx * 1.15, -1.75, 3.0), (sx * 1.75, -1.9, 0.6), 0.38)
    elif hair == "tsunade":  # Pony + Zöpfe
        blob(name + "_cap", (0, 0.45, 2.95), (1.98, 2.35, 1.3))
        for sx in (-1, 1):
            spike(name + f"_bang{sx}", (sx * 0.7, -1.95, 3.3), (sx * 1.6, -2.0, 1.2), 0.42)
            blob(name + f"_tail{sx}", (sx * 1.9, 1.0, -0.6), (0.5, 0.55, 1.9))
    return parts


def cyl_cone(name, p0, p1, r, mat):
    p0, p1 = Vector(p0), Vector(p1)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=r, radius2=r * 0.12, depth=(p1 - p0).length)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    d = (p1 - p0).normalized()
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = d.to_track_quat("Z", "Y")
    ob.location = (p0 + p1) / 2
    fpv.displace_obj(ob, "CLOUDS", size=0.8, strength=0.15 * r, depth=2, subdiv=2, name=name + "_d")
    return ob
