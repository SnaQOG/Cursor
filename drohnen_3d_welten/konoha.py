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
    # Pastellfassaden wie in den Anime-Referenzen: creme, rosa, mintgrün, hellblau, weiß, pfirsich, blassgrün
    col = nb.ramp(rnd, [(0.0, (0.66, 0.57, 0.40)), (0.16, (0.58, 0.36, 0.31)), (0.30, (0.33, 0.45, 0.29)),
                        (0.44, (0.36, 0.45, 0.54)), (0.58, (0.68, 0.66, 0.60)), (0.72, (0.66, 0.44, 0.27)),
                        (0.86, (0.45, 0.53, 0.36)), (1.0, (0.64, 0.58, 0.46))], interp="CONSTANT")
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    nrm = nb.out(nb.node("ShaderNodeNewGeometry"), "Normal")
    tcol, trgh, tnrm = pbr_box(nb, "plaster", nb.coords("Object"), scale=0.35)
    col = nb.vmath("SCALE", nb.mix(1.0, tcol, col, blend="MULTIPLY"), scale=1.25)
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
        # ~25 % der Fenster warm erleuchtet (Zimmerlicht), ~25 % mit hellem Vorhang
        lit = nb.math("MULTIPLY", win, nb.math("GREATER_THAN", nb.out(wr, "Value"), 0.8))
        cur = nb.math("MULTIPLY", win, nb.math("MULTIPLY", nb.math("GREATER_THAN", nb.out(wr, "Value"), 0.62),
                                                nb.math("LESS_THAN", nb.out(wr, "Value"), 0.8)))
        glass = nb.mix(cur, glass, (0.55, 0.50, 0.40))
        glass = nb.mix(lit, glass, (0.30, 0.21, 0.13))
        emis = nb.math("MULTIPLY", lit, 1.1)
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
    if window:
        p.inputs["Emission Color"].default_value = (1.0, 0.62, 0.32, 1)
        nb.link(emis, p.inputs["Emission Strength"])
        mat.cycles.emission_sampling = "NONE"     # sichtbar leuchtend, aber keine Lichtquelle (sonst +15 % Renderzeit)
    nb.link(tnrm, p.inputs["Normal"])
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
        objs[0]["h"] = h
        rm = mats["roofs"][rng.integers(len(mats["roofs"]))]
        objs.append(cylinder(f"B{idx}r", cx, cy, h, r * 1.12, r * 0.55, rm, seg=40, r_top=r * 0.2))
        return objs, h + r * 0.55
    objs.append(box(f"B{idx}", cx, cy, 0, w, d, h, mats["plaster"], rot=rot))
    objs[0]["h"] = h
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

def residence(mats, cx, cy, r=21.0):
    """Hokage-Residenz nach den Anime-Referenzen: orangeroter Rundbau mit Reihen kleiner Fenster,
    ockerfarbenem Ziegel-Kragen auf halber Höhe, 火-Emblem im Ring, Flachdach mit weißen gebogenen
    Hörnern und Mittelspitze, wellige Kabel um den Oberbau, Torbau mit Ziegeldach nach Süden."""
    objs = []
    Z_SK, Z_TOP = 15.0, 31.0
    r2 = r * 0.93
    objs.append(cylinder("ResBase", cx, cy, 0, r + 2.0, 1.5, mats["stone"], seg=72))
    objs.append(cylinder("ResBody", cx, cy, 1.5, r, Z_SK - 1.5, mats["red"], seg=72))
    objs.append(cylinder("ResSkirt", cx, cy, Z_SK, r + 4.2, 4.2, mats["res_tile"], seg=72, r_top=r2 - 0.1))
    objs.append(cylinder("ResSkirtEave", cx, cy, Z_SK - 0.35, r + 4.35, 0.4, mats["red_dark"], seg=72))
    objs.append(cylinder("ResUpper", cx, cy, Z_SK + 4.0, r2, Z_TOP - Z_SK - 4.0, mats["red"], seg=72))
    objs.append(cylinder("ResCap", cx, cy, Z_TOP, r2 + 0.5, 0.6, mats["red_dark"], seg=72))
    objs.append(cylinder("ResParapet", cx, cy, Z_TOP + 0.6, r2 + 0.3, 1.1, mats["red"], seg=72, cap=False))
    objs.append(cylinder("ResRoofDeck", cx, cy, Z_TOP + 0.1, r2 - 0.2, 0.55, mats.get("roofdeck", mats["rooftop"]), seg=72))
    # Reihen kleiner quadratischer Fenster (ein Mesh), Emblem-Bereich im Süden ausgespart
    bm = bmesh.new()
    for (zc, rr, n) in ((5.5, r, 44), (10.0, r, 44), (22.5, r2, 40), (26.5, r2, 40)):
        for j in range(n):
            a = 2 * math.pi * (j + 0.5) / n
            if zc > 20 and abs(math.atan2(math.sin(a + math.pi / 2), math.cos(a + math.pi / 2))) < 0.36:
                continue
            res = bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.scale(bm, vec=(0.25, 1.0, 1.15), verts=res["verts"])
            bmesh.ops.rotate(bm, verts=res["verts"], cent=(0, 0, 0), matrix=Matrix.Rotation(a, 3, "Z"))
            bmesh.ops.translate(bm, vec=(cx + math.cos(a) * (rr - 0.06), cy + math.sin(a) * (rr - 0.06), zc),
                                verts=res["verts"])
    me = bpy.data.meshes.new("ResWindows")
    bm.to_mesh(me)
    bm.free()
    wo = bpy.data.objects.new("ResWindows", me)
    me.materials.append(mats["glass"])
    fpv.link(wo)
    objs.append(wo)
    # 火-Emblem: helle Scheibe mit rotem 火 in dunklem Metallring, nach Süden (-Y)
    disc_r = 3.6
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
    disc.location = (cx, cy - r2 - 0.45, 25.2)
    disc.rotation_euler = (math.radians(90), 0, 0)
    objs.append(disc)
    tube = [(cx + math.cos(t) * (disc_r + 0.3), cy - r2 - 0.5, 25.2 + math.sin(t) * (disc_r + 0.3))
            for t in np.linspace(0, 2 * math.pi, 65)]
    objs.append(curve_tube("HiRingTube", tube, 0.28, mats["iron"], res=3))
    # wellige Kabel/Rohre um den Oberbau (Referenzbild)
    for k, zb in enumerate((20.6, 29.2)):
        pts = []
        for t in np.linspace(0, 2 * math.pi, 181):
            rr = r2 + 0.35
            pts.append((cx + math.cos(t) * rr, cy + math.sin(t) * rr, zb + 0.6 * math.sin(t * 9 + k)))
        objs.append(curve_tube(f"ResCable{k}", pts, 0.2, mats["cable"], res=2))
    # weiße gebogene Hörner auf dem Dachrand + Mittelspitze
    for j in range(8):
        a = math.radians(j * 45 + 22.5)
        d = Vector((math.cos(a), math.sin(a), 0))
        base = Vector((cx, cy, Z_TOP + 0.6)) + d * (r2 - 0.8)
        pts = [tuple(base + d * (1.6 * t * t) + Vector((0, 0, 6.0 * t))) for t in np.linspace(0, 1, 12)]
        cu = bpy.data.curves.new(f"Horn{j}", "CURVE")
        cu.dimensions = "3D"
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for pp, c, t in zip(sp.points, pts, np.linspace(0, 1, len(pts))):
            pp.co = (*c, 1)
            pp.radius = 1.0 - 0.85 * t
        cu.bevel_depth = 0.55
        cu.bevel_resolution = 3
        cu.use_fill_caps = True
        ho = bpy.data.objects.new(f"Horn{j}", cu)
        ho.data.materials.append(mats["horn"])
        fpv.link(ho)
        objs.append(ho)
    objs.append(cylinder("ResSpire", cx, cy, Z_TOP + 0.6, 0.9, 2.6, mats["horn"], seg=24, r_top=0.1))
    # Torbau nach Süden: Block, Ziegel-Walmdach, große Holztüren
    gy = cy - r - 3.0
    objs.append(box("ResGate", cx, gy, 0, 15.0, 8.0, 8.0, mats["red"], bevel=0.1))
    objs.append(hip_roof("ResGateRoof", cx, gy, 8.0, 9.5, 17.5, 3.4, mats["res_tile"], rot=math.pi / 2, overhang=1.0))
    objs.append(box("ResDoor", cx, gy - 4.05, 0.3, 5.0, 0.3, 5.6, mats["beam"], bevel=0.04))
    objs.append(box("ResDoorFrame", cx, gy - 4.2, 5.9, 6.2, 0.4, 0.5, mats["red_dark"], bevel=0.04))
    for sx in (-1, 1):
        objs.append(box("ResDoorPost", cx + sx * 2.85, gy - 4.2, 0, 0.5, 0.4, 6.4, mats["red_dark"], bevel=0.04))
    return objs, Z_TOP


def sand_street_material(name="SandStreet"):
    """Festgetretene, sandfarbene Dorfstraße (Referenz: helle Lehmstraße mit Spuren)."""
    mat, nb, out = fpv.new_material(name)
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    c = nb.image(os.path.join(fpv.ASSETS, "bab", "textures_dirt.jpg"), nb.mapping(wpos, scale=(0.22, 0.22, 0.22)))
    lum = nb.vmath("DOT_PRODUCT", nb.out(c, "Color"), (0.33, 0.33, 0.33))
    n1 = nb.noise(wpos, scale=0.05, detail=3)
    n2 = nb.noise(wpos, scale=0.9, detail=3)
    col = nb.mix(nb.out(n1, "Fac"), (0.44, 0.35, 0.21), (0.66, 0.55, 0.36))
    col = nb.vmath("SCALE", col, scale=nb.math("MULTIPLY_ADD", lum, 0.9, 0.55))
    # Fahrspuren/Fußwege dunkler in Straßenrichtung
    x = nb.sep(wpos)[0]
    track = nb.math("MULTIPLY", nb.math("SINE", nb.math("MULTIPLY", x, 0.9)), nb.math("SUBTRACT", nb.out(n2, "Fac"), 0.3))
    col = nb.mix(nb.math("MULTIPLY", nb.math("MAXIMUM", track, 0.0), 0.35), col, (0.30, 0.23, 0.14))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.93)
    nb.link(nb.bump(nb.math("ADD", lum, nb.math("MULTIPLY", nb.out(n2, "Fac"), 0.4)), strength=0.35, distance=0.04),
            p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def tiered_tower(name, cx, cy, r, tiers, mats, roof_mat, rng):
    """Runder Stufenturm (Referenz: türkiser/blauer Rundturm mit mehreren ausladenden Traufkränzen)."""
    objs = []
    z = 0.0
    rr = r
    for k in range(tiers):
        h = 3.2 * (2 if k == 0 else 1)
        objs.append(cylinder(f"{name}_w{k}", cx, cy, z, rr, h, mats["plaster"], seg=40))
        z += h
        objs.append(cylinder(f"{name}_e{k}", cx, cy, z, rr + 1.6, 1.1, roof_mat, seg=40, r_top=rr * 0.86))
        z += 1.1
        rr *= 0.8
    objs.append(cylinder(f"{name}_top", cx, cy, z, rr * 1.15, rr * 0.9, roof_mat, seg=40, r_top=0.2))
    objs[0]["h"] = z
    return objs, z + rr * 0.9


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
        chz = chin(X) if callable(chin) else chin
        cz = np.clip((chz - Z) / 10.0, 0, 1)
        cz = cz * cz * (3 - 2 * cz)
        Y = Y - 13.0 * cz * mxc
    # tiefe Setzrisse
    Y = add_cracks(Y, X, Z, np.random.default_rng(seed + 20), n=10, x_range=(x0 * 0.9, x1 * 0.9), z_top=z1,
                   panel=panel)
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
    elif hair == "tsunade":  # Pony + Zöpfe + Rautenzeichen (Byakugō) auf der Stirn
        blob(name + "_cap", (0, 0.45, 2.95), (1.98, 2.35, 1.3))
        for sx in (-1, 1):
            spike(name + f"_bang{sx}", (sx * 0.7, -1.95, 3.3), (sx * 1.6, -2.0, 1.2), 0.42)
            blob(name + f"_tail{sx}", (sx * 1.9, 1.0, -0.6), (0.5, 0.55, 1.9))
        dia = blob(name + "_diamond", (0, -2.13, 2.2), (0.17, 0.1, 0.17), rot=(0, math.radians(45), 0))
        dia.modifiers.clear()
    if hair == "tobirama":  # drei Gesichtslinien (Kinn + unter den Augen) als erhabene Grate
        for (a, b) in (((0, -1.33, -0.55), (0, -1.28, -1.05)), ((-0.55, -1.95, 1.05), (-0.75, -1.75, 0.55)),
                       ((0.55, -1.95, 1.05), (0.75, -1.75, 0.55))):
            o = cyl_cone(name + "_line", L + Vector(a) * s, L + Vector(b) * s, 0.07 * s, mat_hair)
            o.modifiers.clear()
            parts.append(o)
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


# --------------------------------------------------------------------------
# v2: Scan-Materialien, Fassaden-Details, Rundziegel, verschmolzene Köpfe, Risse
# --------------------------------------------------------------------------

CC0 = lambda f: os.path.join(fpv.ASSETS, "cc0", f)


def pbr_box(nb, prefix, vec, scale=0.4, blend=0.25):
    """Box-projizierte ambientCG-Maps (color, rough, normal) -> (color, rough, normal)."""
    m = nb.mapping(vec, scale=(scale, scale, scale))
    c = nb.image(CC0(prefix + "_color.jpg"), m, proj="BOX", blend=blend)
    r = nb.image(CC0(prefix + "_rough.jpg"), m, colorspace="Non-Color", proj="BOX", blend=blend)
    n = nb.image(CC0(prefix + "_normal.jpg"), m, colorspace="Non-Color", proj="BOX", blend=blend)
    nm = nb.node("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = 0.8
    nb.link(nb.out(n, "Color"), nm.inputs["Color"])
    return nb.out(c, "Color"), nb.out(r, "Color"), nm.outputs[0]


def plaster_pbr_material(name="PlasterPBR"):
    """Scan-Putz (abgeplatzt) × Gebäudefarbe (pro Objekt), Schmutzfuß, Regenfahnen, Kavität."""
    mat, nb, out = fpv.new_material(name)
    oi = nb.node("ShaderNodeObjectInfo")
    rnd = nb.out(oi, "Random")
    bcol = nb.ramp(rnd, [(0.0, (0.72, 0.64, 0.50)), (0.2, (0.80, 0.78, 0.72)), (0.4, (0.66, 0.56, 0.44)),
                         (0.55, (0.78, 0.68, 0.52)), (0.7, (0.62, 0.66, 0.60)), (0.82, (0.74, 0.54, 0.44)),
                         (1.0, (0.84, 0.82, 0.78))], interp="CONSTANT")
    co = nb.coords("Object")
    tc, tr, tn = pbr_box(nb, "plaster", co, scale=0.35)
    col = nb.mix(1.0, tc, bcol, blend="MULTIPLY")
    col = nb.vmath("SCALE", col, scale=1.25)
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    x, y, z = nb.sep(wpos)
    streak = nb.noise(nb.comb(nb.math("ADD", x, y), nb.math("MULTIPLY", z, 0.08), 0.0), scale=1.5, detail=2)
    sm = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(streak, "Fac"), sm.inputs["Value"])
    sm.inputs["From Min"].default_value = 0.55
    sm.inputs["From Max"].default_value = 0.75
    col = nb.mix(nb.math("MULTIPLY", sm.outputs[0], 0.3), col, nb.vmath("SCALE", col, scale=0.55))
    foot = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(z, foot.inputs["Value"])
    foot.inputs["From Min"].default_value = 1.4
    foot.inputs["From Max"].default_value = 0.0
    col = nb.mix(nb.math("MULTIPLY", foot.outputs[0], 0.55), col, (0.12, 0.10, 0.08))
    p = fpv.principled(nb, Base_Color=col, Roughness=nb.math("MULTIPLY_ADD", tr, 0.3, 0.6))
    nb.link(tn, p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def street_material(name="Cobble"):
    mat, nb, out = fpv.new_material(name)
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    m = nb.mapping(wpos, scale=(0.3, 0.3, 0.3))
    c = nb.image(CC0("paving_color.jpg"), m)
    r = nb.image(CC0("paving_rough.jpg"), m, colorspace="Non-Color")
    n = nb.image(CC0("paving_normal.jpg"), m, colorspace="Non-Color")
    nm = nb.node("ShaderNodeNormalMap")
    nb.link(nb.out(n, "Color"), nm.inputs["Color"])
    # Lehm-/Staub-Flecken über das Pflaster
    dirt = nb.image(os.path.join(fpv.ASSETS, "bab", "textures_ground.jpg"), nb.mapping(wpos, scale=(0.08, 0.08, 0.08)))
    dn = nb.noise(wpos, scale=0.05, detail=3)
    dm = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(dn, "Fac"), dm.inputs["Value"])
    dm.inputs["From Min"].default_value = 0.45
    dm.inputs["From Max"].default_value = 0.65
    col = nb.mix(dm.outputs[0], nb.out(c, "Color"), nb.vmath("SCALE", nb.out(dirt, "Color"), scale=0.8))
    p = fpv.principled(nb, Base_Color=col, Roughness=nb.math("MULTIPLY_ADD", nb.out(r, "Color"), 0.3, 0.62))
    nb.link(nm.outputs[0], p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def dirt_road_material(name="DirtRoad"):
    mat, nb, out = fpv.new_material(name)
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    c = nb.image(os.path.join(fpv.ASSETS, "bab", "textures_dirt.jpg"), nb.mapping(wpos, scale=(0.25, 0.25, 0.25)))
    g = nb.image(os.path.join(fpv.ASSETS, "bab", "textures_rockyGround_basecolor.png"), nb.mapping(wpos, scale=(0.35, 0.35, 0.35)))
    n = nb.noise(wpos, scale=0.06, detail=3)
    col = nb.mix(nb.out(n, "Fac"), nb.out(c, "Color"), nb.out(g, "Color"))
    col = nb.vmath("SCALE", col, scale=0.8)
    p = fpv.principled(nb, Base_Color=col, Roughness=0.92)
    nb.link(nb.bump(nb.math("ADD", nb.sep(nb.out(c, "Color"))[0], nb.out(n, "Fac")), strength=0.4, distance=0.05), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def rust_metal_material(name="TankMetal", tint=(0.75, 0.73, 0.68)):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    tc, tr, tn = pbr_box(nb, "metal", co, scale=0.5)
    col = nb.mix(0.55, tc, nb.vmath("MULTIPLY", tc, tint), blend="MIX")
    p = fpv.principled(nb, Base_Color=col, Roughness=nb.math("MULTIPLY_ADD", tr, 0.4, 0.4), Metallic=0.35)
    nb.link(tn, p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def window_collection(mats):
    """Fenster-Assets: Rahmen (Holz), vertieftes Glas, Sims, optional Fensterläden. Lokal: Fassade = XZ-Ebene,
    Außen = -Y."""
    coll = bpy.data.collections.new("KWindows")
    bpy.context.scene.collection.children.link(coll)
    subs = []
    for i, (w, h, shutters) in enumerate(((1.1, 1.45, False), (1.1, 1.45, True), (1.8, 1.3, False), (0.8, 1.2, False))):
        sub = bpy.data.collections.new(f"KWin{i}")
        coll.children.link(sub)
        objs = []
        t = 0.09
        objs.append(box(f"WinGlass{i}", 0, 0.02, 0, w, 0.04, h, mats["glass"], bevel=0))
        objs.append(box(f"WinFrL{i}", -w / 2, -0.04, 0, t, 0.14, h, mats["frame"], bevel=0.015))
        objs.append(box(f"WinFrR{i}", w / 2, -0.04, 0, t, 0.14, h, mats["frame"], bevel=0.015))
        objs.append(box(f"WinFrT{i}", 0, -0.04, h - t / 2, w + t, 0.14, t, mats["frame"], bevel=0.015))
        objs.append(box(f"WinMul{i}", 0, -0.02, 0, 0.05, 0.08, h, mats["frame"], bevel=0))
        objs.append(box(f"WinSill{i}", 0, -0.12, -0.08, w + 0.3, 0.3, 0.1, mats["sill"], bevel=0.02))
        if shutters:
            for sx in (-1, 1):
                objs.append(box(f"Shut{i}", sx * (w / 2 + 0.33), -0.05, 0, 0.55, 0.05, h, mats["shutter"], bevel=0.01))
        for o in objs:
            bpy.context.scene.collection.objects.unlink(o)
            sub.objects.link(o)
        subs.append(sub)
    coll.hide_render = True
    coll.hide_viewport = True
    return subs


def facade_details(rng, mats, wins, cx, cy, w, d, h, floors, rot, idx, front=None):
    """Echte Fenster auf allen Fassaden, Holzbalken (Ecken + Geschossbänder), Lüftungsrohre, Klimageräte."""
    c, s = math.cos(rot), math.sin(rot)
    faces = [((0, -1), w, d / 2), ((0, 1), w, d / 2), ((-1, 0), d, w / 2), ((1, 0), d, w / 2)]
    for (lx, ly), fw, off in faces:
        nx, ny = lx * c - ly * s, lx * s + ly * c
        ang = math.atan2(-nx, ny) + math.pi  # lokale -Y zeigt nach außen
        tx, ty = -ny, nx
        ncol = max(1, int((fw - 1.2) / 2.6))
        for fl in range(floors):
            if fl == 0 and front is not None and abs(nx * front[0] + ny * front[1]) > 0.9:
                continue  # Erdgeschoss der Straßenseite: Ladenfront (Shader/Markise)
            zz = 0.9 + fl * 3.2
            for k in range(ncol):
                if rng.random() < 0.12:
                    continue
                u = (k + 0.5) / ncol - 0.5
                px = cx + nx * (off + 0.005) + tx * u * (fw - 1.0)
                py = cy + ny * (off + 0.005) + ty * u * (fw - 1.0)
                sub = wins[int(rng.integers(len(wins)))]
                place_instance(sub, (px, py, zz), rot=ang, name=f"W{idx}")
    # Holzbalken: senkrechte Ecken + Geschossbänder (ein Mesh pro Gebäude)
    bm = bmesh.new()
    bw = 0.22
    for (sx, sy_) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = sx * (w / 2), sy_ * (d / 2)
        res = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(bw, bw, h), verts=res["verts"])
        bmesh.ops.translate(bm, vec=(lx, ly, h / 2), verts=res["verts"])
    for fl in range(1, floors + 1):
        zz = fl * 3.2 - 0.15
        for (lx, ly, sx_, sy2) in ((0, -d / 2, w, bw), (0, d / 2, w, bw), (-w / 2, 0, bw, d), (w / 2, 0, bw, d)):
            res = bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.scale(bm, vec=(sx_ + 0.02, sy2 + 0.02, 0.2), verts=res["verts"])
            bmesh.ops.translate(bm, vec=(lx, ly, zz), verts=res["verts"])
    me = bpy.data.meshes.new(f"Beams{idx}")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(f"Beams{idx}", me)
    me.materials.append(mats["beam"])
    fpv.link(ob)
    ob.location = (cx, cy, 0)
    ob.rotation_euler = (0, 0, rot)
    # Lüftungsrohre an einer Seitenfassade (mit Knick über das Dach) + Klimageräte
    for k in range(int(rng.integers(1, 3))):
        lx = rng.uniform(-w / 2 + 0.6, w / 2 - 0.6)
        side = rng.choice([-1, 1])
        ly = side * (d / 2 + 0.18)
        pts = [(lx, ly, 0.3), (lx, ly, h + 0.4), (lx, ly - side * 0.5, h + 0.9), (lx, ly - side * 1.4, h + 1.0)]
        wp = [(cx + px * c - py * s, cy + px * s + py * c, pz) for px, py, pz in pts]
        curve_tube(f"Pipe{idx}_{k}", wp, rng.uniform(0.07, 0.13), mats["pipe"], res=2)
    for k in range(int(rng.integers(0, 3))):
        fl = int(rng.integers(1, max(2, floors)))
        lx = rng.uniform(-w / 2 + 0.8, w / 2 - 0.8)
        side = rng.choice([-1, 1])
        px, py = lx, side * (d / 2 + 0.3)
        box(f"AC{idx}_{k}", cx + px * c - py * s, cy + px * s + py * c, fl * 3.2 - 1.4, 0.9, 0.55, 0.65, mats["ac"],
            rot=rot, bevel=0.03)


def tile_roof_mesh(name, cx, cy, z0, span, length, rise, rot, mat, tile_w=0.30, tile_l=0.42):
    """Rundziegel (Kawara) als echte Geometrie auf einem Tonnendach (Bogen quer über 'span')."""
    # Halbrohr-Ziegel
    seg = 6
    prof = [(math.cos(math.pi * k / seg) * tile_w / 2, math.sin(math.pi * k / seg) * tile_w * 0.45) for k in range(seg + 1)]
    n_arc = max(4, int(math.pi * (span / 2 + rise) / 2 / tile_w * 1.3))
    n_len = max(2, int(length / (tile_l * 0.85)))
    verts, faces = [], []
    for i in range(n_arc):
        a = math.pi * (i + 0.5) / n_arc
        # Punkt auf dem Bogen (Ellipse span/2 x rise), Normale
        ax, az = -math.cos(a) * span / 2 * 1.04, math.sin(a) * rise
        nxv, nzv = -math.cos(a) / (span / 2), math.sin(a) / max(rise, 0.1)
        nl = math.hypot(nxv, nzv)
        nxv, nzv = nxv / nl, nzv / nl
        tx, tz = nzv, -nxv  # Tangente entlang des Bogens
        for j in range(n_len):
            y0 = -length / 2 - 0.2 + j * tile_l * 0.85
            base = len(verts)
            for yy in (y0, y0 + tile_l):
                for (px, pz) in prof:
                    X = ax + tx * px + nxv * (pz + 0.02)
                    Z = az + tz * px + nzv * (pz + 0.02)
                    verts.append((X, yy, Z))
            m = seg + 1
            for k in range(seg):
                faces.append((base + k, base + k + 1, base + m + k + 1, base + m + k))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    me.materials.append(mat)
    fpv.link(ob)
    ob.location = (cx, cy, z0)
    ob.rotation_euler = (0, 0, rot)
    sol = ob.modifiers.new("s", "SOLIDIFY")
    sol.thickness = 0.03
    return ob


def terracotta_material(name, color):
    mat, nb, out = fpv.new_material(name)
    oi = nb.node("ShaderNodeObjectInfo")
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    wn = nb.node("ShaderNodeTexWhiteNoise")
    wn.noise_dimensions = "3D"
    nb.link(nb.vmath("SCALE", wpos, scale=2.5), wn.inputs["Vector"])
    col = nb.mix(nb.math("MULTIPLY", nb.out(wn, "Value"), 0.45), color, [v * 0.6 for v in color])
    n = nb.noise(wpos, scale=0.5, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(n, "Fac"), 0.4), col, (0.14, 0.12, 0.10))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.5)
    p.inputs["Coat Weight"].default_value = 0.15
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def fuse_parts(name, parts, voxel, mat, smooth_iter=4):
    """Kopf + Haar-Felsmassen zu EINEM gemeißelten Mesh verschmelzen (Voxel-Remesh), Teile löschen."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    for o in parts:
        ev = o.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev)
        me.transform(o.matrix_world)
        # offene Scans (Halsöffnung) schließen, sonst verliert das Voxel-Remesh die Gesichtsfläche
        tb = bmesh.new()
        tb.from_mesh(me)
        bnd = [e for e in tb.edges if e.is_boundary]
        if bnd:
            bmesh.ops.holes_fill(tb, edges=bnd, sides=0)
            bmesh.ops.recalc_face_normals(tb, faces=tb.faces)
        tb.to_mesh(me)
        tb.free()
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    fpv.link(ob)
    r = ob.modifiers.new("remesh", "REMESH")
    r.mode = "VOXEL"
    r.voxel_size = voxel
    r.use_smooth_shade = True
    sm = ob.modifiers.new("smooth", "SMOOTH")
    sm.factor = 0.5
    sm.iterations = smooth_iter
    fpv.apply_modifiers(ob)
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    for o in parts:
        bpy.data.objects.remove(o)
    return ob


def add_cracks(Y, X, Z, rng, n=8, x_range=(-400, 400), z_top=150, panel=None, depth=2.2, width=1.2):
    """Tiefe Setzrisse als Polylinien (diagonal nach unten) in die Felswand graben."""
    for _ in range(n):
        x = rng.uniform(*x_range)
        z = rng.uniform(z_top * 0.55, z_top * 0.98)
        pts = [(x, z)]
        for k in range(int(rng.integers(5, 10))):
            x += rng.normal(0, 6)
            z -= rng.uniform(6, 14)
            pts.append((x, z))
        dmin = np.full(X.shape, 1e9)
        for (x1, z1), (x2, z2) in zip(pts[:-1], pts[1:]):
            vx, vz = x2 - x1, z2 - z1
            L2 = vx * vx + vz * vz
            t = np.clip(((X - x1) * vx + (Z - z1) * vz) / L2, 0, 1)
            dx = X - (x1 + t * vx)
            dz = Z - (z1 + t * vz)
            dmin = np.minimum(dmin, np.sqrt(dx * dx + dz * dz))
        groove = depth * np.exp(-(dmin / width) ** 2)
        if panel is not None:
            px0, px1, pz0, pz1 = panel
            inside = (X > px0) & (X < px1) & (Z > pz0) & (Z < pz1)
            groove = np.where(inside, groove * 0.15, groove)
        Y = Y + groove
    return Y


# --------------------------------------------------------------------------
# Schritt 3.3 – Verwitterung, Straßenboden, Ziegel, Glas, Dachbelag
# --------------------------------------------------------------------------

def weather(mat, dirt_h=2.2, dirt=0.45, ground_z=0.0, edge=0.0, edge_col=(0.78, 0.74, 0.66), bevel_r=0.07,
            streak=0.0, var=0.05, streak_scale=1.2):
    """Verwitterung an ein bestehendes Material hängen (Basisfarbe des Principled BSDF wird umgeleitet):
    - Farbe pro Objekt (Object Info → Random): Farbton ±var/2, Helligkeit ±var
    - Schmutz von unten (Welt-z über ground_z bis dirt_h, mit Rauschen ausgefranst)
    - Kantenabrieb über den Bevel-Knoten (Abweichung der gerundeten von der echten Normale), aufgehellt
    - senkrechte Regen-/Laufspuren (gestrecktes Rauschen)."""
    nt = mat.node_tree
    nb = fpv.NB(nt)
    p = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    bc = p.inputs["Base Color"]
    col = bc.links[0].from_socket if bc.is_linked else tuple(bc.default_value)[:3]
    for lk in list(bc.links):
        nt.links.remove(lk)
    geo = nb.node("ShaderNodeNewGeometry")
    wpos = nb.out(geo, "Position")
    x, y, z = nb.sep(wpos)
    if var:
        rnd = nb.out(nb.node("ShaderNodeObjectInfo"), "Random")
        hsv = nb.node("ShaderNodeHueSaturation")
        nb.link(col, hsv.inputs["Color"])
        nb.link(nb.math("ADD", 0.5, nb.math("MULTIPLY", nb.math("SUBTRACT", nb.math("FRACT", nb.math("MULTIPLY", rnd, 7.3)), 0.5),
                                              0.4 * var)), hsv.inputs["Hue"])
        nb.link(nb.math("ADD", 1.0, nb.math("MULTIPLY", nb.math("SUBTRACT", rnd, 0.5), 2 * var)), hsv.inputs["Value"])
        col = hsv.outputs[0]
    if streak:
        sn = nb.noise(nb.comb(nb.math("ADD", x, y), nb.math("MULTIPLY", z, 0.07), 0.0), scale=streak_scale, detail=2)
        sm = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(nb.out(sn, "Fac"), sm.inputs["Value"])
        sm.inputs["From Min"].default_value = 0.55
        sm.inputs["From Max"].default_value = 0.75
        col = nb.mix(nb.math("MULTIPLY", sm.outputs[0], streak), col, nb.vmath("MULTIPLY", col, (0.55, 0.52, 0.48)))
    if edge:
        bev = nb.node("ShaderNodeBevel")
        bev.samples = 4
        bev.inputs["Radius"].default_value = bevel_r
        d = nb.vmath("DOT_PRODUCT", nb.out(bev, "Normal"), nb.out(geo, "Normal"))
        em = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(nb.math("ADD", d, nb.math("MULTIPLY", nb.out(nb.noise(wpos, scale=3.0, detail=3), "Fac"), 0.06)),
                em.inputs["Value"])
        em.inputs["From Min"].default_value = 1.02
        em.inputs["From Max"].default_value = 0.96
        col = nb.mix(nb.math("MULTIPLY", em.outputs[0], edge), col, edge_col)
    if dirt:
        dn = nb.noise(wpos, scale=0.9, detail=3, rough=0.6)
        h = nb.math("ADD", nb.math("SUBTRACT", z, ground_z), nb.math("MULTIPLY", nb.math("SUBTRACT", nb.out(dn, "Fac"), 0.5), 0.9))
        dm = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(h, dm.inputs["Value"])
        dm.inputs["From Min"].default_value = dirt_h
        dm.inputs["From Max"].default_value = 0.0
        f = nb.math("MULTIPLY", nb.math("POWER", dm.outputs[0], 1.6), dirt)
        col = nb.mix(f, col, nb.vmath("MULTIPLY", col, (0.42, 0.37, 0.31)))
    nb.link(col, bc)
    return mat


def street_ground_material(name="StreetGround", half_w=12.0):
    """Hauptstraße: festgetretener Sand mit Steinplatten-Weg in der Mitte (Brick-Muster, Fugen, Risse,
    Farbstreuung pro Platte), Fahrrinnen, rissigen Lehmflächen, Pfützen (glatt, spiegelnd) und dunkleren,
    bewachsenen Rändern an den Hauswänden."""
    mat, nb, out = fpv.new_material(name)
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    x, y, _ = nb.sep(wpos)
    ax = nb.math("ABSOLUTE", x)
    c = nb.image(os.path.join(fpv.ASSETS, "bab", "textures_dirt.jpg"), nb.mapping(wpos, scale=(0.22, 0.22, 0.22)))
    lum = nb.vmath("DOT_PRODUCT", nb.out(c, "Color"), (0.33, 0.33, 0.33))
    n1 = nb.noise(wpos, scale=0.05, detail=3)
    n2 = nb.noise(wpos, scale=0.9, detail=3)
    sand = nb.mix(nb.out(n1, "Fac"), (0.44, 0.35, 0.21), (0.66, 0.55, 0.36))
    sand = nb.vmath("SCALE", sand, scale=nb.math("MULTIPLY_ADD", lum, 0.9, 0.55))
    # Fahrrinnen bei |x| ≈ 5,5 m
    rut = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.math("ABSOLUTE", nb.math("SUBTRACT", ax, 5.5)), rut.inputs["Value"])
    rut.inputs["From Min"].default_value = 0.9
    rut.inputs["From Max"].default_value = 0.2
    sand = nb.mix(nb.math("MULTIPLY", rut.outputs[0], nb.math("MULTIPLY_ADD", nb.out(n2, "Fac"), 0.5, 0.2)), sand,
                  (0.30, 0.23, 0.14))
    # rissige Lehmflächen
    mud = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(nb.noise(wpos, scale=0.11, detail=2), "Fac"), mud.inputs["Value"])
    mud.inputs["From Min"].default_value = 0.56
    mud.inputs["From Max"].default_value = 0.62
    vc = nb.voronoi(wpos, scale=2.2, feature="DISTANCE_TO_EDGE")
    crk = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(vc, "Distance"), crk.inputs["Value"])
    crk.inputs["From Min"].default_value = 0.045
    crk.inputs["From Max"].default_value = 0.0
    cracks = nb.math("MULTIPLY", crk.outputs[0], mud.outputs[0])
    sand = nb.mix(nb.math("MULTIPLY", cracks, 0.8), sand, (0.12, 0.09, 0.06))
    # Steinplatten in der Mitte (Ränder ausgefranst)
    fl = nb.voronoi(wpos, scale=1.05, feature="F1", rand=0.85)
    fe = nb.voronoi(wpos, scale=1.05, feature="DISTANCE_TO_EDGE", rand=0.85)
    jm = nb.node("ShaderNodeMapRange", clamp=True)                  # Fuge: 1 in der Fuge
    nb.link(nb.out(fe, "Distance"), jm.inputs["Value"])
    jm.inputs["From Min"].default_value = 0.05
    jm.inputs["From Max"].default_value = 0.025
    stone = nb.mix(nb.math("MULTIPLY", nb.sep(nb.out(fl, "Color"))[0], 1.0), (0.34, 0.32, 0.29), (0.50, 0.47, 0.41))
    path = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.math("ADD", ax, nb.math("MULTIPLY", nb.out(nb.noise(wpos, scale=0.35, detail=2), "Fac"), 1.4)),
            path.inputs["Value"])
    path.inputs["From Min"].default_value = 3.6
    path.inputs["From Max"].default_value = 3.3
    slab = nb.mix(jm.outputs[0], stone, (0.13, 0.11, 0.09))
    sc_ = nb.voronoi(wpos, scale=1.6, feature="DISTANCE_TO_EDGE")
    sc_m = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(sc_, "Distance"), sc_m.inputs["Value"])
    sc_m.inputs["From Min"].default_value = 0.02
    sc_m.inputs["From Max"].default_value = 0.0
    slab = nb.mix(nb.math("MULTIPLY", sc_m.outputs[0], nb.math("GREATER_THAN", nb.out(n2, "Fac"), 0.55)), slab,
                  (0.14, 0.12, 0.10))
    slab = nb.vmath("SCALE", slab, scale=nb.math("MULTIPLY_ADD", lum, 0.5, 0.75))
    col = nb.mix(path.outputs[0], sand, slab)
    mortar = nb.math("MULTIPLY", jm.outputs[0], path.outputs[0])
    # dunklere, bewachsene Ränder an den Hauswänden
    edge = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.math("ADD", ax, nb.math("MULTIPLY", nb.out(n2, "Fac"), 1.2)), edge.inputs["Value"])
    edge.inputs["From Min"].default_value = half_w - 3.5
    edge.inputs["From Max"].default_value = half_w + 0.6
    col = nb.mix(nb.math("MULTIPLY", edge.outputs[0], 0.6), col, nb.vmath("MULTIPLY", col, (0.5, 0.47, 0.42)))
    weed = nb.math("MULTIPLY", nb.math("POWER", edge.outputs[0], 3.0),
                   nb.math("GREATER_THAN", nb.out(nb.noise(wpos, scale=1.7, detail=3), "Fac"), 0.55))
    col = nb.mix(weed, col, (0.07, 0.11, 0.03))
    # Pfützen: dunkel, glatt, Rand nass
    pn = nb.noise(wpos, scale=0.075, detail=3, rough=0.55)
    pud = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.math("ADD", nb.out(pn, "Fac"), nb.math("MULTIPLY", nb.out(n2, "Fac"), 0.06)), pud.inputs["Value"])
    pud.inputs["From Min"].default_value = 0.672
    pud.inputs["From Max"].default_value = 0.69
    wet = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(pn, "Fac"), wet.inputs["Value"])
    wet.inputs["From Min"].default_value = 0.64
    wet.inputs["From Max"].default_value = 0.675
    col = nb.mix(nb.math("MULTIPLY", wet.outputs[0], 0.6), col, nb.vmath("MULTIPLY", col, (0.55, 0.52, 0.5)))
    col = nb.mix(pud.outputs[0], col, (0.10, 0.10, 0.10))
    rough = nb.mix(wet.outputs[0], 0.93, 0.45, dtype="FLOAT")
    rough = nb.mix(pud.outputs[0], rough, 0.03, dtype="FLOAT")
    p = fpv.principled(nb, Base_Color=col, Roughness=rough)
    h = nb.math("ADD", lum, nb.math("MULTIPLY", nb.out(n2, "Fac"), 0.4))
    h = nb.math("SUBTRACT", h, nb.math("MULTIPLY", nb.math("ADD", mortar, cracks), 0.8))
    h = nb.math("MULTIPLY", h, nb.math("SUBTRACT", 1.0, pud.outputs[0]))       # Pfützen spiegelglatt
    nb.link(nb.bump(h, strength=0.4, distance=0.04), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def tile_roof_material(name, color, cylindrical=False, tile_w=0.32, tile_h=0.26, var=0.18):
    """Ziegeldach: versetzte Ziegelreihen (Brick-Muster) mit Farbstreuung pro Ziegel, dunkle Fugen/Schattenkanten
    als Relief, Moos/Schmutz in den Fugen, Patina-Flecken und Regenstreifen. cylindrical=True: Reihen rund um
    Kegel/Zylinder (Objektkoordinaten, Winkel × Umfang), für die runden Turm- und Stufendächer."""
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    ox, oy, oz = nb.sep(co)
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    if cylindrical:
        ang = nb.math("ARCTAN2", oy, ox)
        u = nb.math("MULTIPLY", ang, 1.6)       # ~10 Ziegel pro Radiant (feste Zahl rund um den Kegel)
        v = oz
    else:
        u, v = nb.math("ADD", ox, oy), oz      # Reihen parallel zur Traufe (gleiche Höhe) für Walm-/Tonnendächer
    br = nb.node("ShaderNodeTexBrick")
    br.offset = 0.5
    nb.link(nb.comb(u, v, 0.0), br.inputs["Vector"])
    br.inputs["Scale"].default_value = 1.0
    br.inputs["Mortar Size"].default_value = 0.018
    br.inputs["Mortar Smooth"].default_value = 0.6
    br.inputs["Brick Width"].default_value = tile_w if not cylindrical else 0.16
    br.inputs["Row Height"].default_value = tile_h
    br.inputs["Color1"].default_value = (*color, 1)
    br.inputs["Color2"].default_value = (*[c * (1 - var) for c in color], 1)
    br.inputs["Mortar"].default_value = (*[c * 0.35 for c in color], 1)
    rnd = nb.out(nb.node("ShaderNodeObjectInfo"), "Random")
    col = nb.out(br, "Color")
    col = nb.vmath("SCALE", col, scale=nb.math("ADD", 0.93, nb.math("MULTIPLY", rnd, 0.14)))
    # Patina (grau-grünlich) und Schmutz
    pat = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(nb.noise(wpos, scale=0.45, detail=4, rough=0.6), "Fac"), pat.inputs["Value"])
    pat.inputs["From Min"].default_value = 0.5
    pat.inputs["From Max"].default_value = 0.75
    col = nb.mix(nb.math("MULTIPLY", pat.outputs[0], 0.45), col, (0.34, 0.36, 0.30))
    moss = nb.math("MULTIPLY", nb.out(br, "Fac"), nb.math("GREATER_THAN", nb.out(nb.noise(wpos, scale=1.3), "Fac"), 0.5))
    col = nb.mix(nb.math("MULTIPLY", moss, 0.7), col, (0.06, 0.09, 0.03))
    x, y, z = nb.sep(wpos)
    sn = nb.noise(nb.comb(nb.math("ADD", x, y), nb.math("MULTIPLY", z, 0.1), 0.0), scale=1.4, detail=2)
    sm = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(sn, "Fac"), sm.inputs["Value"])
    sm.inputs["From Min"].default_value = 0.56
    sm.inputs["From Max"].default_value = 0.74
    col = nb.mix(nb.math("MULTIPLY", sm.outputs[0], 0.4), col, nb.vmath("MULTIPLY", col, (0.5, 0.48, 0.45)))
    p = fpv.principled(nb, Base_Color=col, Roughness=nb.mix(nb.out(br, "Fac"), 0.5, 0.85, dtype="FLOAT"))
    # Relief: Ziegel wölben sich, Unterkante wirft Schattenkante (Sägezahn über die Reihe)
    row = nb.math("FRACT", nb.math("DIVIDE", v, tile_h))
    h = nb.math("SUBTRACT", nb.math("MULTIPLY", row, 0.6), nb.math("MULTIPLY", nb.out(br, "Fac"), 0.8))
    nb.link(nb.bump(h, strength=0.55, distance=0.03), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def window_glass_material(name="WindowGlass", lit=0.22, curtain=0.3, per_cell=None):
    """Fensterglas: spiegelnd (Fresnel), dahinter angedeuteter Raum: ein Teil der Fenster warm erleuchtet,
    ein Teil mit hellem Vorhang, der Rest dunkles Zimmer. Zufall pro Objekt (Instanzen) oder – für ein einziges
    Mesh mit vielen Fenstern – pro Weltzelle (per_cell = Zellgröße in m)."""
    mat, nb, out = fpv.new_material(name)
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    if per_cell:
        wn = nb.node("ShaderNodeTexWhiteNoise")
        wn.noise_dimensions = "3D"
        nb.link(nb.vmath("SCALE", wpos, scale=1.0 / per_cell), wn.inputs["Vector"])
        fl = nb.node("ShaderNodeVectorMath", operation="FLOOR")
        nb.link(nb.vmath("SCALE", wpos, scale=1.0 / per_cell), fl.inputs[0])
        nb.link(fl.outputs[0], wn.inputs["Vector"])
        rnd = nb.out(wn, "Value")
    else:
        rnd = nb.out(nb.node("ShaderNodeObjectInfo"), "Random")
    is_lit = nb.math("LESS_THAN", rnd, lit)
    is_cur = nb.math("MULTIPLY", nb.math("GREATER_THAN", rnd, lit), nb.math("LESS_THAN", rnd, lit + curtain))
    _, _, z = nb.sep(wpos)
    fold = nb.out(nb.noise(nb.comb(nb.math("MULTIPLY", nb.sep(wpos)[0], 9.0), nb.sep(wpos)[1], 0.0), scale=1.0), "Fac")
    room = nb.mix(nb.math("FRACT", nb.math("MULTIPLY", rnd, 13.0)), (0.025, 0.022, 0.02), (0.05, 0.04, 0.03))
    col = nb.mix(is_cur, room, nb.mix(fold, (0.55, 0.50, 0.40), (0.72, 0.66, 0.52)))
    col = nb.mix(is_lit, col, (0.30, 0.22, 0.14))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.04, IOR=1.52)
    p.inputs["Emission Color"].default_value = (1.0, 0.62, 0.32, 1)
    nb.link(nb.math("MULTIPLY", is_lit, nb.math("MULTIPLY_ADD", fold, 0.8, 0.9)), p.inputs["Emission Strength"])
    mat.cycles.emission_sampling = "NONE"
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def deck_wear(mat, center, r_edge=19.3):
    """Dachbelag der Residenz verwittern: dunkler, feuchter Ring an der Brüstung, Wasserflecken, helle
    Laufspuren zur Mitte, Moos in den Fugen am Rand."""
    nt = mat.node_tree
    nb = fpv.NB(nt)
    p = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    bc = p.inputs["Base Color"]
    col = bc.links[0].from_socket
    nt.links.remove(bc.links[0])
    wpos = nb.out(nb.node("ShaderNodeNewGeometry"), "Position")
    x, y, _ = nb.sep(wpos)
    r = nb.math("SQRT", nb.math("ADD", nb.math("POWER", nb.math("SUBTRACT", x, center[0]), 2.0),
                                nb.math("POWER", nb.math("SUBTRACT", y, center[1]), 2.0)))
    n = nb.noise(wpos, scale=0.35, detail=4, rough=0.6)
    ring = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.math("ADD", r, nb.math("MULTIPLY", nb.out(n, "Fac"), 3.0)), ring.inputs["Value"])
    ring.inputs["From Min"].default_value = r_edge - 3.5
    ring.inputs["From Max"].default_value = r_edge + 1.0
    col = nb.mix(nb.math("MULTIPLY", ring.outputs[0], 0.75), col, nb.vmath("MULTIPLY", col, (0.45, 0.43, 0.38)))
    moss = nb.math("MULTIPLY", nb.math("POWER", ring.outputs[0], 2.0),
                   nb.math("GREATER_THAN", nb.out(nb.noise(wpos, scale=2.2), "Fac"), 0.58))
    col = nb.mix(nb.math("MULTIPLY", moss, 0.8), col, (0.07, 0.10, 0.035))
    st = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(nb.noise(wpos, scale=0.16, detail=3), "Fac"), st.inputs["Value"])
    st.inputs["From Min"].default_value = 0.6
    st.inputs["From Max"].default_value = 0.68
    col = nb.mix(nb.math("MULTIPLY", st.outputs[0], 0.35), col, nb.vmath("MULTIPLY", col, (0.62, 0.6, 0.56)))
    walk = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.math("ADD", r, nb.math("MULTIPLY", nb.out(n, "Fac"), 4.0)), walk.inputs["Value"])
    walk.inputs["From Min"].default_value = 11.0
    walk.inputs["From Max"].default_value = 4.0
    col = nb.mix(nb.math("MULTIPLY", walk.outputs[0], 0.25), col, nb.vmath("MULTIPLY", col, (1.35, 1.3, 1.22)))
    nb.link(col, bc)
    return mat
