"""Straßenleben in Konoha (Schritt 3.3): Laternen an Seilen quer über die Straße (pendelnd), Passanten,
Waren an den Marktständen, Wäscheleinen, aufgeschreckte Vögel und treibende Blätter.

Alles ist deterministisch (keine Simulation, die einen Frame-Durchlauf bräuchte): Pendeln über
F-Kurven-Modifier (Sinus-Generator + Rauschen), Blätter über Geometry Nodes mit der Szenenzeit."""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

import fpv
from figures import POSES, Figure
from ninja import cloth_material, hair_material, skin_material


def _fcurves(ob):
    ad = ob.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            for cb in strip.channelbags:
                out += list(cb.fcurves)
    return out


def sway(ob, axis, amp_deg, freq, phase, noise=0.25, base=0.0):
    """Pendeln um eine Achse über F-Kurven-Modifier (Sinus + Rauschen) – ohne Keyframe-Flut, jede Instanz mit
    eigener Phase/Frequenz."""
    ob.rotation_mode = "XYZ"
    ob.keyframe_insert("rotation_euler", index=axis, frame=1)
    for fc in _fcurves(ob):
        if fc.data_path == "rotation_euler" and fc.array_index == axis:
            fc.keyframe_points[0].co = (1.0, base)
            g = fc.modifiers.new("FNGENERATOR")
            g.function_type = "SIN"
            g.amplitude = math.radians(amp_deg)
            g.phase_multiplier = 2 * math.pi * freq / 24.0
            g.phase_offset = phase
            g.value_offset = base
            g.use_additive = False
            if noise:
                nz = fc.modifiers.new("NOISE")
                nz.scale = 30.0
                nz.strength = math.radians(amp_deg) * noise * 2
                nz.phase = phase * 10
                nz.blend_type = "ADD"


def _rope(name, pts, radius, mat):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1)
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    return fpv.link(ob)


def lantern_mesh(mats):
    """Papierlaterne (Chōchin): gerippter Körper, schwarze Deckel, Aufhängeschnur; Ursprung = Aufhängepunkt."""
    bm = bmesh.new()
    rings, seg = 14, 20
    body_top, body_h, R = -0.32, 0.72, 0.3
    verts = []
    for j in range(rings + 1):
        t = j / rings
        z = body_top - t * body_h
        r = R * math.sin(math.pi * (0.12 + 0.76 * t)) * (1 + 0.035 * math.cos(2 * math.pi * 7 * t))   # Rippen
        verts.append([bm.verts.new((r * math.cos(2 * math.pi * k / seg), r * math.sin(2 * math.pi * k / seg), z))
                      for k in range(seg)])
    for j in range(rings):
        for k in range(seg):
            f = bm.faces.new([verts[j][k], verts[j][(k + 1) % seg], verts[j + 1][(k + 1) % seg], verts[j + 1][k]])
            f.material_index = 0
            f.smooth = True
    for zc, rr in ((body_top + 0.02, 0.13), (body_top - body_h - 0.02, 0.15)):
        res = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=rr, radius2=rr, depth=0.07)
        bmesh.ops.translate(bm, vec=(0, 0, zc), verts=res["verts"])
        for f in {f for v in res["verts"] for f in v.link_faces}:
            f.material_index = 1
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=0.012, radius2=0.012, depth=abs(body_top))
    bmesh.ops.translate(bm, vec=(0, 0, body_top / 2), verts=res["verts"])
    for f in {f for v in res["verts"] for f in v.link_faces}:
        f.material_index = 1
    me = bpy.data.meshes.new("LanternMesh")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mats["paper"])
    me.materials.append(mats["cap"])
    return me


def lantern_materials():
    mat, nb, out = fpv.new_material("LanternPaper")
    co = nb.coords("Object")
    rnd = nb.out(nb.node("ShaderNodeObjectInfo"), "Random")
    col = nb.mix(nb.math("GREATER_THAN", rnd, 0.7), (0.62, 0.07, 0.035), (0.80, 0.70, 0.52))  # rot / naturweiß
    band = nb.math("MULTIPLY", nb.math("GREATER_THAN", nb.math("ABSOLUTE", nb.math("ADD", nb.sep(co)[2], 0.68)), 0.28),
                   1.0)
    col = nb.mix(nb.math("MULTIPLY", band, 0.5), col, (0.15, 0.03, 0.02))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.7)
    p.inputs["Subsurface Weight"].default_value = 0.3
    p.inputs["Subsurface Radius"].default_value = (0.3, 0.12, 0.06)
    p.inputs["Emission Color"].default_value = (1.0, 0.42, 0.14, 1)
    p.inputs["Emission Strength"].default_value = 0.9           # Kerzenschein durchs Papier
    nb.link(p.outputs[0], out.inputs[0])
    mat.cycles.emission_sampling = "NONE"
    cap = fpv.simple_mat("LanternCap", (0.02, 0.018, 0.016), rough=0.45)
    rope = fpv.simple_mat("LanternRope", (0.05, 0.04, 0.03), rough=0.8)
    return {"paper": mat, "cap": cap, "rope": rope}


def lantern_lines(ys, x_end, z_end=7.2, sag=1.1, n=9, seed=3):
    """Seile quer über die Straße (sichtbar, 3,5 cm), Laternen hängen daran und pendeln um die Seilachse."""
    rng = np.random.default_rng(seed)
    mats = lantern_materials()
    me = lantern_mesh(mats)
    obs = []
    for k, y in enumerate(ys):
        a = Vector((-x_end, y, z_end))
        b = Vector((x_end, y, z_end))

        def cable(t):
            return a.lerp(b, t) - Vector((0, 0, sag * 4 * t * (1 - t)))
        _rope(f"LanternRope{k}", [tuple(cable(t)) for t in np.linspace(0, 1, 24)], 0.035, mats["rope"])
        for t in np.linspace(0.1, 0.9, n):
            lo = bpy.data.objects.new(f"Lantern{k}", me)
            fpv.link(lo)
            lo.location = cable(t)
            lo.scale = (1.0,) * 3 if rng.random() > 0.3 else (0.85,) * 3
            sway(lo, 0, rng.uniform(4.0, 7.0), rng.uniform(0.3, 0.5), rng.uniform(0, 6.28))   # um die Seilachse
            sway(lo, 1, rng.uniform(1.5, 3.0), rng.uniform(0.2, 0.35), rng.uniform(0, 6.28))
            obs.append(lo)
    return obs


def laundry_lines(ys, x_end, z_end=10.0, sag=0.7, seed=5):
    """Wäscheleinen quer über die Straße in 9–10 m Höhe mit Hemden/Tüchern, die leicht schwingen."""
    rng = np.random.default_rng(seed)
    rope = fpv.simple_mat("LaundryRope", (0.55, 0.50, 0.42), rough=0.8)
    cols = [(0.70, 0.68, 0.62), (0.16, 0.28, 0.52), (0.62, 0.18, 0.12), (0.78, 0.62, 0.30), (0.20, 0.40, 0.28),
            (0.85, 0.83, 0.78), (0.45, 0.22, 0.40)]
    cm = [cloth_material(f"Laundry{i}", c, sheen=0.3) for i, c in enumerate(cols)]
    for m in cm:
        m.use_backface_culling = False
    for k, y in enumerate(ys):
        a = Vector((-x_end, y, z_end))
        b = Vector((x_end, y, z_end))

        def line(t):
            return a.lerp(b, t) - Vector((0, 0, sag * 4 * t * (1 - t)))
        _rope(f"LaundryLine{k}", [tuple(line(t)) for t in np.linspace(0, 1, 20)], 0.012, rope)
        t = 0.06
        while t < 0.94:
            w = rng.uniform(0.4, 0.9)
            h = rng.uniform(0.5, 1.0)
            shirt = rng.random() < 0.45
            bm = bmesh.new()
            if shirt:     # T-Form
                pts = [(-w / 2, 0), (w / 2, 0), (w / 2 + 0.22, -0.12), (w / 2 + 0.12, -0.3), (w / 2, -0.24),
                       (w / 2, -h), (-w / 2, -h), (-w / 2, -0.24), (-w / 2 - 0.12, -0.3), (-w / 2 - 0.22, -0.12)]
            else:
                pts = [(-w / 2, 0), (w / 2, 0), (w / 2, -h), (-w / 2, -h)]
            vs = [bm.verts.new((px, 0.0, pz)) for px, pz in pts]
            bm.faces.new(vs)
            bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=3, use_grid_fill=True)
            for v in bm.verts:        # leichter Wurf/Falten
                v.co.y += 0.05 * math.sin(v.co.x * 9 + k) * (-v.co.z)
            me = bpy.data.meshes.new("Cloth")
            bm.to_mesh(me)
            bm.free()
            me.materials.append(cm[int(rng.integers(len(cm)))])
            ob = bpy.data.objects.new(f"Laundry{k}", me)
            fpv.link(ob)
            ob.location = line(t + w / (2 * 2 * x_end))
            ob.rotation_euler = (0, 0, rng.uniform(-0.1, 0.1))
            sway(ob, 0, rng.uniform(5, 10), rng.uniform(0.35, 0.6), rng.uniform(0, 6.28), noise=0.4)
            t += (w + rng.uniform(0.25, 0.8)) / (2 * x_end)
    return True


VILLAGER_POSES = {
    "look_up": {"spine": (7, 0, 0), "head": (28, 0, 0), "shoulder.R": (0, -8, 0), "shoulder.L": (0, 8, 0)},
    "point_up": {"spine": (6, 0, 0), "head": (24, 0, 0), "shoulder.R": (150, -10, 0), "elbow.R": (10, 0, 0),
                 "shoulder.L": (0, 8, 0)},
    "carry": {"shoulder.R": (50, -8, 0), "elbow.R": (75, 0, 0), "shoulder.L": (50, 8, 0), "elbow.L": (75, 0, 0),
              "spine": (-4, 0, 0)},
    "talk": {"shoulder.R": (25, -12, 0), "elbow.R": (85, 0, 0), "head": (-4, 0, 14), "shoulder.L": (0, 8, 0)},
}


def villager_materials(tag, seed):
    """Stoffe mit Farbe pro Instanz (Object Info → Random wählt aus einer gedeckten Palette)."""
    def palette_cloth(name, pal):
        mat, nb, out = fpv.new_material(name)
        rnd = nb.out(nb.node("ShaderNodeObjectInfo"), "Random")
        stops = [(i / len(pal), c) for i, c in enumerate(pal)]
        col = nb.ramp(nb.math("FRACT", nb.math("MULTIPLY", rnd, 3.7 + seed)), stops, interp="CONSTANT")
        n = nb.noise(nb.coords("Object"), scale=30.0, detail=2)
        col = nb.mix(nb.math("MULTIPLY", nb.out(n, "Fac"), 0.3), col, nb.vmath("SCALE", col, scale=0.75))
        p = fpv.principled(nb, Base_Color=col, Roughness=0.85)
        p.inputs["Sheen Weight"].default_value = 0.35
        nb.link(p.outputs[0], out.inputs[0])
        return mat
    tops = [(0.55, 0.20, 0.12), (0.18, 0.26, 0.42), (0.62, 0.55, 0.40), (0.25, 0.36, 0.22), (0.50, 0.38, 0.22),
            (0.40, 0.18, 0.30), (0.70, 0.66, 0.58)]
    bottoms = [(0.10, 0.10, 0.14), (0.28, 0.22, 0.16), (0.16, 0.18, 0.24), (0.36, 0.33, 0.28)]
    skin = skin_material(f"{tag}Skin", (0.72, 0.47, 0.32) if seed % 2 else (0.62, 0.40, 0.27))
    return {"torso": palette_cloth(f"{tag}Top", tops), "pelvis": palette_cloth(f"{tag}Bottom", bottoms),
            "upper_arm": palette_cloth(f"{tag}Top", tops), "forearm": skin, "hand": skin,
            "thigh": palette_cloth(f"{tag}Bottom", bottoms), "lower": palette_cloth(f"{tag}Bottom", bottoms),
            "foot": cloth_material(f"{tag}Sandal", (0.22, 0.16, 0.10)), "skin": skin}


def villager_templates():
    """6 Passanten-Vorlagen (verschiedene Größe, Statur, Pose, Haar) als versteckte Collections."""
    import crew
    specs = [("look_up", 1.72, 1.0, 0.82, (0.03, 0.025, 0.02)), ("point_up", 1.64, 0.95, 0.3, (0.08, 0.05, 0.03)),
             ("carry", 1.78, 1.15, 0.3, (0.02, 0.02, 0.02)), ("talk", 1.58, 0.9, 0.82, (0.12, 0.08, 0.05)),
             ("look_up", 1.52, 0.9, 0.82, (0.35, 0.33, 0.30)), ("hands_hips", 1.75, 1.1, 0.3, (0.04, 0.03, 0.02))]
    subs = []
    root = bpy.data.collections.new("Villagers")
    bpy.context.scene.collection.children.link(root)
    for i, (pose, h, bulk, sleeve, hair_c) in enumerate(specs):
        tag = f"Vil{i}"
        m = villager_materials(tag, i)
        fig = Figure(tag, h, (0, 0, 0), 0.0)
        fig.body(m, bulk=bulk, sleeve_to=sleeve, pant_to=0.05 if i % 2 else 0.3)
        crew._hair_cap(fig, tag, hair_material(f"{tag}Hair", hair_c))
        rot = VILLAGER_POSES.get(pose) or POSES[pose]
        fig.pose(1, rot, loc=(0, 0, 0), yaw=0.0, ground=0.0)
        sub = bpy.data.collections.new(tag)
        root.children.link(sub)
        stack = [fig.base]
        while stack:
            o = stack.pop()
            stack.extend(o.children)
            for c in list(o.users_collection):
                c.objects.unlink(o)
            sub.objects.link(o)
        subs.append(sub)
    root.hide_render = True
    root.hide_viewport = True
    return subs


def place_villagers(subs, spots, seed=7):
    """spots: Liste (x, y, yaw_deg) -> Collection-Instanzen (Farbe variiert pro Instanz)."""
    rng = np.random.default_rng(seed)
    parent = bpy.data.objects.new("VillagerSpots", None)
    fpv.link(parent)
    for i, (x, y, yaw) in enumerate(spots):
        e = bpy.data.objects.new(f"Villager_{i}", None)
        e.instance_type = "COLLECTION"
        e.instance_collection = subs[int(rng.integers(len(subs)))]
        e.location = (x, y, 0.0)
        e.rotation_euler = (0, 0, math.radians(yaw))
        e.parent = parent
        fpv.link(e)
    return parent


def stall_goods(name, x, y, rng, table_z=0.92, w=2.0, d=2.6):
    """Waren auf dem Markttisch als ein Mesh: Obstpyramiden, Fische, Töpfe, Stoffballen, Körbe."""
    mats = [fpv.simple_mat("GoodsOrange", (0.85, 0.32, 0.03), rough=0.45),
            fpv.simple_mat("GoodsRed", (0.55, 0.05, 0.04), rough=0.4),
            fpv.simple_mat("GoodsGreen", (0.22, 0.42, 0.08), rough=0.5),
            fpv.simple_mat("GoodsFish", (0.55, 0.58, 0.60), rough=0.25, metal=0.3),
            fpv.simple_mat("GoodsPot", (0.42, 0.24, 0.14), rough=0.6),
            fpv.simple_mat("GoodsCloth", (0.20, 0.30, 0.55), rough=0.8),
            fpv.simple_mat("GoodsBasket", (0.45, 0.34, 0.18), rough=0.9)]
    bm = bmesh.new()

    def add(res, mi):
        for f in {f for v in res["verts"] for f in v.link_faces}:
            f.material_index = mi
            f.smooth = True
    kind = int(rng.integers(4))
    for gx in np.linspace(-w / 2 + 0.3, w / 2 - 0.3, 3):
        for gy in np.linspace(-d / 2 + 0.35, d / 2 - 0.35, 3):
            cx, cy = gx + rng.uniform(-0.08, 0.08), gy + rng.uniform(-0.08, 0.08)
            k = (kind + int(rng.integers(2))) % 4
            if k == 0:       # Obst im Korb
                res = bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.2, radius2=0.26, depth=0.12)
                bmesh.ops.translate(bm, vec=(cx, cy, table_z + 0.06), verts=res["verts"])
                add(res, 6)
                mi = int(rng.integers(3))
                for q in range(7):
                    a = q * 2.4
                    rr = 0.13 if q else 0.0
                    res = bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.055)
                    bmesh.ops.translate(bm, vec=(cx + rr * math.cos(a), cy + rr * math.sin(a),
                                                 table_z + 0.16 + (0.07 if q == 0 else 0.0)), verts=res["verts"])
                    add(res, mi)
            elif k == 1:     # Fische nebeneinander
                for q in range(3):
                    res = bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=6, radius=0.1)
                    bmesh.ops.scale(bm, vec=(2.4, 0.7, 0.45), verts=res["verts"])
                    bmesh.ops.translate(bm, vec=(cx, cy - 0.12 + q * 0.12, table_z + 0.05), verts=res["verts"])
                    add(res, 3)
            elif k == 2:     # Töpfe
                for q in range(2):
                    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=14, radius1=0.12, radius2=0.08, depth=0.24)
                    bmesh.ops.translate(bm, vec=(cx - 0.1 + q * 0.2, cy, table_z + 0.12), verts=res["verts"])
                    add(res, 4)
            else:            # Stoffballen
                res = bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.09, radius2=0.09, depth=0.5)
                bmesh.ops.rotate(bm, verts=res["verts"], cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, "X"))
                bmesh.ops.translate(bm, vec=(cx, cy, table_z + 0.09), verts=res["verts"])
                add(res, 5)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    ob.location = (x, y, 0)
    return fpv.link(ob)


def birds(n, start_box, flight, t0, t1, fps=24, seed=9):
    """Aufgeschreckter Vogelschwarm: Körper + zwei Flügel (schlagen mit 6–8 Hz über F-Kurven-Modifier),
    Flugbahn als Bogen von start_box nach flight (Versatz), Start gestaffelt zwischen t0 und t0+0.4 s."""
    rng = np.random.default_rng(seed)
    mat = fpv.simple_mat("Bird", (0.07, 0.065, 0.06), rough=0.6)
    body_me = bpy.data.meshes.new("BirdBody")
    bm = bmesh.new()
    res = bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.09)
    bmesh.ops.scale(bm, vec=(0.7, 2.2, 0.7), verts=res["verts"])
    bm.to_mesh(body_me)
    bm.free()
    body_me.materials.append(mat)
    wing_me = bpy.data.meshes.new("BirdWing")
    bm = bmesh.new()
    vs = [bm.verts.new(c) for c in ((0, -0.08, 0), (0.34, -0.02, 0), (0.30, 0.1, 0), (0, 0.12, 0))]
    bm.faces.new(vs)
    bm.to_mesh(wing_me)
    bm.free()
    wing_me.materials.append(mat)
    (x0, x1), (y0, y1), (z0, z1) = start_box
    for i in range(n):
        b = bpy.data.objects.new(f"Bird{i}", body_me)
        fpv.link(b)
        p0 = Vector((rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(z0, z1)))
        d = Vector(flight) + Vector((rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-2, 4)))
        ts = t0 + rng.uniform(0, 0.4)
        for k, u in enumerate(np.linspace(0, 1, 6)):
            tt = ts + u * (t1 - t0)
            p = p0 + d * u + Vector((0, 0, 3.0 * math.sin(math.pi * u) + 2.0 * u))
            b.location = p
            b.keyframe_insert("location", frame=int(round(tt * fps)) + 1)
        b.location = p0
        b.keyframe_insert("location", frame=1)
        b.rotation_euler = (0, 0, math.atan2(-d.x, d.y))
        for sd in (-1, 1):
            w = bpy.data.objects.new(f"BirdWing{i}", wing_me)
            fpv.link(w)
            w.parent = b
            w.scale = (sd, 1, 1)
            sway(w, 1, 40.0, rng.uniform(6, 8), rng.uniform(0, 6.28) + (0 if sd > 0 else math.pi) * 0, noise=0.1,
                 base=0.0)
            if sd < 0:
                for fc in _fcurves(w):
                    for mdf in fc.modifiers:
                        if mdf.type == "FNGENERATOR":
                            mdf.amplitude = -mdf.amplitude
    return True


def leaves_gn(name, box, count=600, wind=(1.2, 3.0, -0.35), seed=4):
    """Treibende Blätter (Geometry Nodes, Szenenzeit): Punkte zufällig in box, driften mit dem Wind und
    flattern (Sinus), werden am Rand der Box periodisch zurückgesetzt; Blatt-Instanzen drehen sich."""
    (x0, x1), (y0, y1), (z0, z1) = box
    me = bpy.data.meshes.new(name + "Pts")
    rng = np.random.default_rng(seed)
    P = np.stack([rng.uniform(x0, x1, count), rng.uniform(y0, y1, count), rng.uniform(z0, z1, count)], 1)
    me.from_pydata(P.tolist(), [], [])
    ob = bpy.data.objects.new(name, me)
    fpv.link(ob)
    # Blatt-Mesh
    lm = bpy.data.meshes.new(name + "Leaf")
    bm = bmesh.new()
    vs = [bm.verts.new(c) for c in ((0, -0.09, 0), (0.05, 0, 0.01), (0, 0.11, 0), (-0.05, 0, 0.01))]   # ~20 cm
    bm.faces.new(vs)
    bm.to_mesh(lm)
    bm.free()
    mat, nb, out = fpv.new_material(name + "LeafMat")
    rnd = nb.out(nb.node("ShaderNodeObjectInfo"), "Random")
    col = nb.ramp(rnd, [(0.0, (0.10, 0.20, 0.04)), (0.45, (0.30, 0.34, 0.06)), (0.7, (0.55, 0.36, 0.06)),
                        (1.0, (0.45, 0.14, 0.04))])
    p = fpv.principled(nb, Base_Color=col, Roughness=0.6)
    p.inputs["Transmission Weight"].default_value = 0.0
    p.inputs["Subsurface Weight"].default_value = 0.2
    nb.link(p.outputs[0], out.inputs[0])
    lm.materials.append(mat)
    leaf = bpy.data.objects.new(name + "LeafTpl", lm)
    fpv.link(leaf)
    leaf.hide_render = True
    leaf.hide_viewport = True
    ng = bpy.data.node_groups.new(name + "GN", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    g = fpv.NB(ng)
    gi, go = g.node("NodeGroupInput"), g.node("NodeGroupOutput")
    pos = g.node("GeometryNodeInputPosition").outputs[0]
    idx = g.node("GeometryNodeInputIndex").outputs[0]
    ts = g.node("GeometryNodeInputSceneTime").outputs["Seconds"]
    rv = g.node("FunctionNodeRandomValue")
    rv.data_type = "FLOAT"
    g.link(idx, rv.inputs["ID"])
    ph = g.math("MULTIPLY", rv.outputs["Value"], 6.283)
    px, py, pz = g.sep(pos)
    # Drift + Flattern, periodisch in der Box gehalten (Modulo)
    fx = g.math("ADD", g.math("MULTIPLY", ts, wind[0]), g.math("MULTIPLY", g.math("SINE", g.math("ADD", g.math("MULTIPLY", ts, 1.7), ph)), 0.6))
    fy = g.math("ADD", g.math("MULTIPLY", ts, wind[1]), g.math("MULTIPLY", g.math("COSINE", g.math("ADD", g.math("MULTIPLY", ts, 1.3), ph)), 0.6))
    fz = g.math("ADD", g.math("MULTIPLY", ts, wind[2]), g.math("MULTIPLY", g.math("SINE", g.math("ADD", g.math("MULTIPLY", ts, 2.3), ph)), 0.35))

    def wrap(v, off, lo, hi):
        return g.math("ADD", lo, g.math("FLOORED_MODULO", g.math("SUBTRACT", g.math("ADD", v, off), lo), hi - lo))
    npos = g.comb(wrap(px, fx, x0, x1), wrap(py, fy, y0, y1), wrap(pz, fz, z0, z1))
    sp = g.node("GeometryNodeSetPosition")
    g.link(gi.outputs[0], sp.inputs["Geometry"])
    g.link(npos, sp.inputs["Position"])
    mtp = g.node("GeometryNodeMeshToPoints")
    g.link(sp.outputs[0], mtp.inputs["Mesh"])
    oi = g.node("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = leaf
    iop = g.node("GeometryNodeInstanceOnPoints")
    g.link(mtp.outputs[0], iop.inputs["Points"])
    g.link(oi.outputs["Geometry"], iop.inputs["Instance"])
    rot = g.comb(g.math("ADD", g.math("MULTIPLY", ts, 5.0), ph), g.math("ADD", g.math("MULTIPLY", ts, 3.1), g.math("MULTIPLY", ph, 2.0)),
                 g.math("MULTIPLY", ph, 3.0))
    g.link(rot, iop.inputs["Rotation"])
    g.link(g.math("ADD", 0.8, g.math("MULTIPLY", rv.outputs["Value"], 0.8)), iop.inputs["Scale"])
    g.link(iop.outputs[0], go.inputs[0])
    mod = ob.modifiers.new("Leaves", "NODES")
    mod.node_group = ng
    return ob
