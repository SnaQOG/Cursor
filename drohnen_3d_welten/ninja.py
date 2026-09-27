"""Naruto und Sasuke (Shippuden) nach den Model Sheets (決定稿, 2006): Körper aus Skin-Modifier-Skeletten
mit Subdivision, Kleidung als eigene Teile mit Materialzonen, Haar aus Stacheln, Rückenmotive als Decals.

Lokales System der Figur: Blick nach +Y, +Z oben, Füße bei z = 0. Maßstab in Metern
(Naruto 1,66 m ≈ 6,3 Kopfhöhen, Sasuke 1,68 m).
"""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from PIL import Image, ImageDraw, ImageFilter

import fpv
import textures


# --------------------------------------------------------------------------
# Texturen und Materialien
# --------------------------------------------------------------------------

def spiral_decal(path, W=512, col=(196, 40, 24, 255)):
    """Uzumaki-Spirale (Rückenmotiv der Jacke)."""
    im = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pts = []
    for t in np.linspace(0, 3.25 * 2 * math.pi, 400):
        r = W * 0.035 + W * 0.40 * t / (3.25 * 2 * math.pi)
        pts.append((W / 2 + r * math.cos(t), W / 2 + r * math.sin(t)))
    d.line(pts, fill=col, width=int(W * 0.055), joint="curve")
    im.filter(ImageFilter.GaussianBlur(1.0)).save(path)
    return path


def uchiha_decal(path, W=512):
    """Uchiha-Wappen (Fächer): obere Hälfte rot, untere weiß, Griff unten."""
    im = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c, r = W / 2, W * 0.34
    box = [c - r, c - r - W * 0.08, c + r, c + r - W * 0.08]
    d.ellipse([box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 8], fill=(20, 18, 18, 255))
    d.pieslice(box, 180, 360, fill=(190, 22, 18, 255))
    d.pieslice(box, 0, 180, fill=(236, 232, 224, 255))
    d.rectangle([c - W * 0.045, box[3] - 4, c + W * 0.045, box[3] + W * 0.16], fill=(236, 232, 224, 255),
                outline=(20, 18, 18, 255), width=6)
    im.filter(ImageFilter.GaussianBlur(1.0)).save(path)
    return path


def cloth_material(name, color, rough=0.8, sheen=0.4, decal=None, decal_box=None, zones=None):
    """Stoff: feine Webstruktur, Falten-Rauschen, Sheen. zones: Liste (z0, z1, color) in Objekt-z (m) für
    farbige Bereiche (z. B. schwarze Schulterpasse). decal: Bild auf dem Rücken (-Y), decal_box = (x0, x1, z0, z1)."""
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    x, y, z = nb.sep(co)
    n1 = nb.noise(co, scale=40.0, detail=2)
    n2 = nb.noise(co, scale=6.0, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(n2, "Fac"), 0.35), color, [c * 0.72 for c in color])
    for (z0, z1, zc) in zones or ():
        m = nb.math("MULTIPLY", nb.math("GREATER_THAN", z, z0), nb.math("LESS_THAN", z, z1))
        zcol = nb.mix(nb.math("MULTIPLY", nb.out(n2, "Fac"), 0.35), zc, [c * 0.72 for c in zc])
        col = nb.mix(m, col, zcol)
    if decal:
        x0, x1, z0, z1 = decal_box
        u = nb.math("DIVIDE", nb.math("SUBTRACT", x, x0), x1 - x0)
        u = nb.math("SUBTRACT", 1.0, u)                    # von hinten gesehen gespiegelt
        v = nb.math("DIVIDE", nb.math("SUBTRACT", z, z0), z1 - z0)
        img = nb.image(decal, nb.comb(u, v, 0.0))
        img.extension = "CLIP"
        back = nb.math("LESS_THAN", y, -0.02)
        a = nb.math("MULTIPLY", nb.out(img, "Alpha"), back)
        col = nb.mix(a, col, nb.out(img, "Color"))
    p = fpv.principled(nb, Base_Color=col, Roughness=rough)
    p.inputs["Sheen Weight"].default_value = sheen
    nb.link(nb.bump(nb.math("ADD", nb.out(n1, "Fac"), nb.math("MULTIPLY", nb.out(n2, "Fac"), 2.0)), strength=0.25,
                    distance=0.004), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def skin_material(name="Skin", color=(0.70, 0.40, 0.26)):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    n = nb.noise(co, scale=30.0, detail=2)
    col = nb.mix(nb.math("MULTIPLY", nb.out(n, "Fac"), 0.2), color, [c * 0.85 for c in color])
    p = fpv.principled(nb, Base_Color=col, Roughness=0.5)
    p.inputs["Subsurface Weight"].default_value = 0.25
    p.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
    p.inputs["Subsurface Scale"].default_value = 0.01
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def hair_material(name, color, sheen=(1.0, 1.0, 0.9)):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    st = nb.noise(nb.mapping(co, scale=(60.0, 60.0, 8.0)), scale=1.0, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(st, "Fac"), 0.5), color, [c * 0.6 for c in color])
    p = fpv.principled(nb, Base_Color=col, Roughness=0.38)
    p.inputs["Coat Weight"].default_value = 0.2
    p.inputs["Sheen Weight"].default_value = 0.3
    p.inputs["Sheen Tint"].default_value = (*sheen, 1)
    nb.link(nb.bump(nb.out(st, "Fac"), strength=0.4, distance=0.003), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


# --------------------------------------------------------------------------
# Geometrie
# --------------------------------------------------------------------------

def skin_part(name, verts, edges, radii, mat, root=0, subdiv=2):
    """Organischer Körperteil aus einem Skelett (Skin-Modifier + Subdivision). radii: (rx, ry) je Vertex."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], edges, [])
    ob = bpy.data.objects.new(name, me)
    fpv.link(ob)
    ob.modifiers.new("skin", "SKIN")
    if not me.skin_vertices:
        me.skin_vertices.new()
    sv = me.skin_vertices[0].data
    for i, r in enumerate(radii):
        sv[i].radius = r if isinstance(r, (tuple, list)) else (r, r)
        sv[i].use_root = (i == root)
    sub = ob.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = subdiv
    me.materials.append(mat)
    ob.modifiers.new("wn", "WEIGHTED_NORMAL")
    for p in me.polygons:
        p.use_smooth = True
    return ob


def ellipsoid(name, loc, r, mat, seg=32):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=1.0)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    ob.location = loc
    ob.scale = r
    return ob


def spikes(name, center, dirs, lengths, base_r, mat, bend=None):
    """Haarstacheln als gebogene Kegel (ein Mesh)."""
    bm = bmesh.new()
    C = Vector(center)
    for d, ln, br in zip(dirs, lengths, base_r):
        d = Vector(d).normalized()
        side = d.cross(Vector((0, 0, 1)))
        if side.length < 1e-3:
            side = Vector((1, 0, 0))
        side.normalize()
        up = side.cross(d).normalized()
        n_s, n_a = 6, 8
        rings = []
        for i in range(n_s + 1):
            t = i / n_s
            c = C + d * (ln * t) + (Vector(bend) * ln * t * t if bend else Vector())
            r = br * (1 - t) ** 1.1 + 0.002
            ring = []
            for k in range(n_a):
                a = 2 * math.pi * k / n_a
                ring.append(bm.verts.new(c + side * (math.cos(a) * r) + up * (math.sin(a) * r * 0.55)))
            rings.append(ring)
        for i in range(n_s):
            for k in range(n_a):
                k2 = (k + 1) % n_a
                bm.faces.new([rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]])
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    sub = ob.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1
    return ob


def tube(name, pts, r, mat, taper=None, res=3):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for i, (p, c) in enumerate(zip(sp.points, pts)):
        p.co = (*c, 1)
        if taper:
            p.radius = taper[i]
    cu.bevel_depth = r
    cu.bevel_resolution = res
    cu.use_fill_caps = True
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    fpv.link(ob)
    return ob


def ribbon(name, pts, width, mat, twist=0.0):
    """Stoffband (Stirnband-Enden): Streifen entlang einer Kurve, leicht verdreht."""
    bm = bmesh.new()
    prev = None
    for i, p in enumerate(pts):
        p = Vector(p)
        a = twist * i / max(len(pts) - 1, 1)
        s = Vector((math.cos(a), 0, math.sin(a))) * (width / 2)
        pair = (bm.verts.new(p - s), bm.verts.new(p + s))
        if prev:
            bm.faces.new([prev[0], prev[1], pair[1], pair[0]])
        prev = pair
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    sol = ob.modifiers.new("sol", "SOLIDIFY")
    sol.thickness = 0.004
    return ob


def _legs_and_feet(tag, mats, pants, lower, shoe, pant_to=0.36):
    """Hose als EIN Skelett (Becken + beide Beine, durchgehender Schritt) bis pant_to (m), darunter
    Stiefel/Wicklung bis zum Knöchel, Sandale mit Sohle, Riemen und freien Zehen."""
    # Becken als Ellipsoid, Beinketten beginnen im Rumpf -> keine Naht, kein Knick im Schritt
    objs = [ellipsoid(f"{tag}Pelvis", (0, 0, 0.9), (0.152, 0.112, 0.12), pants)]
    lows = []
    for sx in (-1, 1):
        top = Vector((sx * 0.07, 0, 1.0))
        hip = Vector((sx * 0.085, 0, 0.84))
        knee = Vector((sx * 0.095, 0.025, 0.50))
        ankle = Vector((sx * 0.10, 0, 0.10))
        cut = knee.lerp(ankle, (0.50 - pant_to) / 0.40)
        objs.append(skin_part(f"{tag}Leg{sx}", [top, hip, hip.lerp(knee, 0.5), knee, cut],
                              [(i, i + 1) for i in range(4)],
                              [(0.08, 0.08), (0.09, 0.09), (0.079, 0.079), (0.064, 0.064), (0.058, 0.058)], pants))
        lows.append((sx, cut, ankle))
    for sx, cut, ankle in lows:
        objs.append(skin_part(f"{tag}Shin{sx}", [cut + Vector((0, 0, 0.04)), ankle + Vector((0, 0, 0.03))], [(0, 1)],
                              [(0.054, 0.054), (0.046, 0.046)], lower))
        objs.append(skin_part(f"{tag}Foot{sx}", [ankle + Vector((0, -0.02, -0.035)), Vector((sx * 0.10, 0.12, 0.032))],
                              [(0, 1)], [(0.046, 0.04), (0.042, 0.026)], mats["skin"]))
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(0.095, 0.27, 0.025), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(sx * 0.10, 0.05, 0.0125), verts=bm.verts)
        sole = fpv.mesh_from_bmesh(bm, f"{tag}Sole{sx}", shoe, smooth=False)
        bv = sole.modifiers.new("bv", "BEVEL")
        bv.width = 0.01
        bv.segments = 2
        objs.append(sole)
        objs.append(skin_part(f"{tag}Strap{sx}", [ankle + Vector((0, -0.015, -0.01)), ankle + Vector((0, -0.015, 0.09))],
                              [(0, 1)], [(0.052, 0.05), (0.05, 0.048)], shoe))
    return objs


def _arm(tag, sx, pts, radii, mat, hand_mat, hand_r=(0.045, 0.045, 0.05)):
    """Arm-Kette, die IM Rumpf beginnt (nahtloser Schulteransatz), Hand in Verlängerung des Unterarms."""
    root = Vector((sx * 0.06, 0, pts[0][2] - 0.04))
    chain = [root] + [Vector(p) for p in pts]
    objs = [skin_part(f"{tag}Arm{sx}", chain, [(i, i + 1) for i in range(len(chain) - 1)], [radii[0]] + list(radii),
                      mat)]
    d = (chain[-1] - chain[-2]).normalized()
    objs.append(ellipsoid(f"{tag}Hand{sx}", chain[-1] + d * 0.045, hand_r, hand_mat))
    return objs


def naruto(loc, yaw_deg, mats=None):
    """Naruto Uzumaki (Shippuden): schwarz-orange Jacke mit Uzumaki-Spirale am Rücken, orange Hose,
    Oberschenkel-Bandage mit Holster, schwarze Stiefelsandalen, Stirnband mit Blattsymbol und langen Enden,
    gelbes Stachelhaar. Pose: rechter Arm zeigt hinauf zum Hokage-Felsen."""
    tag = "Naruto"
    ORANGE, BLACK = (0.86, 0.26, 0.035), (0.028, 0.028, 0.032)
    sp = spiral_decal(os.path.join(textures.OUT, "uzumaki_spiral.png"))
    m = {
        "jacket": cloth_material("NarutoJacket", ORANGE, zones=[(1.24, 2.0, BLACK), (0.89, 0.97, BLACK)],
                                 decal=sp, decal_box=(-0.12, 0.12, 1.19, 1.43)),
        "sleeve": cloth_material("NarutoSleeve", BLACK),
        "pants": cloth_material("NarutoPants", ORANGE),
        "boot": cloth_material("NarutoBoot", (0.045, 0.045, 0.05), rough=0.6, sheen=0.1),
        "skin": skin_material("NarutoSkin"),
        "hair": hair_material("NarutoHair", (0.95, 0.70, 0.06)),
        "band": cloth_material("NarutoBand", (0.02, 0.025, 0.055)),
        "plate": fpv.simple_mat("NarutoPlate", (0.62, 0.63, 0.66), rough=0.25, metal=1.0),
        "bandage": cloth_material("Bandage", (0.80, 0.78, 0.72)),
        "holster": cloth_material("Holster", (0.03, 0.04, 0.08), rough=0.55),
        "pouch": cloth_material("Pouch", (0.55, 0.54, 0.50)),
    }
    if mats:
        m.update(mats)
    objs = []
    # Rumpf (Jacke): Becken – Bauch – Brust – Hals
    torso = [(0, 0, 0.90), (0, 0.01, 1.05), (0, 0, 1.22), (0, -0.005, 1.37), (0, 0, 1.43)]
    objs.append(skin_part(f"{tag}Torso", torso, [(i, i + 1) for i in range(4)],
                          [(0.155, 0.115), (0.15, 0.11), (0.165, 0.115), (0.14, 0.1), (0.065, 0.065)], m["jacket"]))
    objs.append(skin_part(f"{tag}Collar", [(0, 0, 1.40), (0, 0, 1.49)], [(0, 1)], [(0.075, 0.072), (0.07, 0.068)],
                          m["sleeve"]))
    # Arme: rechts erhoben zum Felsen, links Hand an der Hüfte
    arms = {1: [(0.19, 0, 1.36), (0.23, 0.2, 1.52), (0.25, 0.38, 1.73), (0.255, 0.43, 1.79)],
            -1: [(-0.19, 0, 1.36), (-0.32, -0.05, 1.13), (-0.19, 0.0, 0.99), (-0.15, 0.01, 0.97)]}
    for sx, pts in arms.items():
        objs += _arm(tag, sx, pts[:3], [(0.06, 0.06), (0.051, 0.051), (0.043, 0.043)], m["sleeve"], m["skin"])
    objs += _legs_and_feet(tag, m, m["pants"], m["boot"], m["boot"], pant_to=0.30)
    # Bandage + Holster am rechten Oberschenkel, Tasche hinten rechts
    objs.append(skin_part(f"{tag}Bandage", [(0.092, 0.012, 0.64), (0.093, 0.016, 0.72)], [(0, 1)],
                          [(0.082, 0.082), (0.084, 0.084)], m["bandage"]))
    objs.append(ellipsoid(f"{tag}Holster", (0.17, 0.0, 0.66), (0.022, 0.05, 0.08), m["holster"]))
    objs.append(ellipsoid(f"{tag}Pouch", (0.1, -0.13, 0.92), (0.06, 0.035, 0.06), m["pouch"]))
    # Kopf, Ohren
    H = Vector((0, 0.01, 1.57))
    objs.append(ellipsoid(f"{tag}Head", H, (0.098, 0.108, 0.118), m["skin"]))
    for sx in (-1, 1):
        objs.append(ellipsoid(f"{tag}Ear{sx}", H + Vector((sx * 0.097, -0.005, 0.0)), (0.014, 0.022, 0.03), m["skin"]))
    # Stirnband: Band, Metallplatte vorn, lange Enden hinten (flattern leicht)
    band = [tuple(H + Vector((math.cos(a) * 0.104, math.sin(a) * 0.114, 0.045))) for a in np.linspace(0, 2 * math.pi, 41)]
    objs.append(tube(f"{tag}Band", band, 0.022, m["band"]))
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(0.13, 0.012, 0.045), verts=bm.verts)
    plate = fpv.mesh_from_bmesh(bm, f"{tag}Plate", m["plate"], smooth=False)
    plate.location = H + Vector((0, 0.118, 0.045))
    bv = plate.modifiers.new("bv", "BEVEL")
    bv.width = 0.006
    bv.segments = 2
    objs.append(plate)
    for k, (dx, dz) in enumerate(((0.02, 0.0), (-0.015, -0.025))):
        pts = [tuple(H + Vector((dx + 0.03 * math.sin(t * 5 + k), -0.11 - 0.33 * t, 0.045 + dz - 0.22 * t * t)))
               for t in np.linspace(0, 1, 12)]
        objs.append(ribbon(f"{tag}Tail{k}", pts, 0.035, m["band"], twist=0.9))
    # Stachelhaar: Kappe + ~26 Stacheln (hinten/oben/seitlich), Pony vorn über dem Stirnband
    objs.append(ellipsoid(f"{tag}HairCap", H + Vector((0, -0.01, 0.035)), (0.106, 0.116, 0.108), m["hair"]))
    rng = np.random.default_rng(3)
    dirs, lens, brs = [], [], []
    while len(dirs) < 44:   # Stacheln starten im Kopfzentrum: Länge 0,2–0,32 m ragt 9–20 cm heraus
        az = rng.uniform(0, 2 * math.pi)
        el = rng.uniform(-0.25, 1.35)
        if math.sin(az) > 0.45 and el < 0.55:   # Gesicht frei lassen
            continue
        dirs.append((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el) - 0.15, math.sin(el) + 0.2))
        lens.append(rng.uniform(0.2, 0.32))
        brs.append(rng.uniform(0.06, 0.08))
    objs.append(spikes(f"{tag}Spikes", H + Vector((0, -0.01, 0.06)), dirs, lens, brs, m["hair"]))
    bang = [(math.cos(a) * 0.4, 1.0, -0.8) for a in np.linspace(-1.2, 1.2, 5)]
    objs.append(spikes(f"{tag}Bangs", H + Vector((0, 0.07, 0.1)), bang, [0.07] * 5, [0.03] * 5, m["hair"]))
    return _assemble(tag, objs, loc, yaw_deg)


def sasuke(loc, yaw_deg, mats=None):
    """Sasuke Uchiha (Shippuden): hellgraues Kurzarmhemd mit hohem Kragen und Uchiha-Wappen am Rücken,
    indigofarbener Hüftwickel bis zum Knie, dicker lila Seilgürtel mit Knoten, dunkle Hose,
    graue Beinwickel, Armstulpen, Kusanagi-Schwert schräg im Gürtel, schwarzes Haar (hinten stachelig)."""
    tag = "Sasuke"
    GREY, INDIGO = (0.40, 0.40, 0.46), (0.045, 0.045, 0.16)
    cr = uchiha_decal(os.path.join(textures.OUT, "uchiha_crest.png"))
    m = {
        "shirt": cloth_material("SasukeShirt", GREY, decal=cr, decal_box=(-0.075, 0.075, 1.30, 1.45)),
        "wrap": cloth_material("SasukeWrap", INDIGO),
        "pants": cloth_material("SasukePants", (0.035, 0.035, 0.11)),
        "wraps": cloth_material("SasukeLegWrap", (0.16, 0.16, 0.18)),
        "shoe": cloth_material("SasukeSandal", (0.03, 0.03, 0.035), rough=0.6, sheen=0.1),
        "rope": cloth_material("SasukeRope", (0.20, 0.10, 0.34), rough=0.7),
        "skin": skin_material("SasukeSkin", (0.74, 0.48, 0.35)),
        "hair": hair_material("SasukeHair", (0.012, 0.012, 0.025), sheen=(0.6, 0.7, 1.0)),
        "guard": cloth_material("SasukeGuard", INDIGO),
        "sheath": fpv.simple_mat("Sheath", (0.02, 0.02, 0.022), rough=0.35),
    }
    if mats:
        m.update(mats)
    objs = []
    torso = [(0, 0, 0.93), (0, 0.01, 1.06), (0, 0, 1.23), (0, -0.005, 1.38), (0, 0, 1.44)]
    objs.append(skin_part(f"{tag}Torso", torso, [(i, i + 1) for i in range(4)],
                          [(0.15, 0.11), (0.14, 0.105), (0.16, 0.11), (0.135, 0.098), (0.062, 0.062)], m["shirt"]))
    objs.append(skin_part(f"{tag}Collar", [(0, -0.01, 1.40), (0, -0.015, 1.53)], [(0, 1)],
                          [(0.085, 0.08), (0.09, 0.085)], m["shirt"]))
    # Hüftwickel: weiter Rock bis zum Knie
    objs.append(skin_part(f"{tag}Wrap", [(0, 0, 0.98), (0, 0.0, 0.78), (0, 0.0, 0.55)], [(0, 1), (1, 2)],
                          [(0.17, 0.13), (0.19, 0.15), (0.2, 0.16)], m["wrap"], subdiv=2))
    # Seilgürtel: zwei dicke Windungen + hängender Knoten vorn links
    for k, z in enumerate((0.965, 1.02)):
        pts = [(math.cos(a) * 0.168, math.sin(a) * 0.125, z + 0.01 * math.sin(a * 3)) for a in np.linspace(0, 2 * math.pi, 49)]
        objs.append(tube(f"{tag}Rope{k}", pts, 0.028, m["rope"]))
    objs.append(tube(f"{tag}Knot", [(-0.1, 0.12, 0.99), (-0.12, 0.15, 0.85), (-0.11, 0.14, 0.70)], 0.03, m["rope"]))
    for sx in (-1, 1):
        pts = [(sx * 0.19, 0, 1.37), (sx * 0.235, -0.02, 1.1), (sx * 0.26, 0.03, 0.87)]
        objs.append(skin_part(f"{tag}Sleeve{sx}", [(sx * 0.06, 0, 1.33), pts[0], (sx * 0.222, -0.012, 1.22)],
                              [(0, 1), (1, 2)], [(0.068, 0.066), (0.066, 0.066), (0.062, 0.06)], m["shirt"]))
        objs += _arm(tag, sx, pts, [(0.05, 0.05), (0.041, 0.041), (0.034, 0.034)], m["skin"], m["skin"],
                     hand_r=(0.036, 0.032, 0.055))
        objs.append(skin_part(f"{tag}Guard{sx}", [(sx * 0.252, 0.015, 0.92), (sx * 0.258, 0.026, 0.98)], [(0, 1)],
                              [(0.042, 0.042), (0.044, 0.044)], m["guard"]))
    objs += _legs_and_feet(tag, m, m["pants"], m["wraps"], m["shoe"], pant_to=0.34)
    # Kusanagi schräg hinten im Gürtel
    objs.append(tube(f"{tag}Sheath", [(-0.2, -0.16, 0.62), (0.2, -0.15, 1.18)], 0.022, m["sheath"]))
    objs.append(tube(f"{tag}Hilt", [(0.2, -0.15, 1.18), (0.27, -0.148, 1.29)], 0.017, m["sheath"]))
    # Kopf + Haar: hinten stachelig abstehend, lange Strähnen seitlich vorn
    H = Vector((0, 0.01, 1.60))
    objs.append(ellipsoid(f"{tag}Head", H, (0.096, 0.106, 0.116), m["skin"]))
    for sx in (-1, 1):
        objs.append(ellipsoid(f"{tag}Ear{sx}", H + Vector((sx * 0.095, -0.005, 0.0)), (0.014, 0.022, 0.03), m["skin"]))
    objs.append(ellipsoid(f"{tag}HairCap", H + Vector((0, -0.02, 0.03)), (0.105, 0.115, 0.108), m["hair"]))
    rng = np.random.default_rng(7)
    dirs, lens, brs = [], [], []
    for k in range(26):   # hinten abstehende Stacheln (Referenz: Rückansicht)
        az = rng.uniform(math.radians(195), math.radians(345))
        el = rng.uniform(-0.1, 1.0)
        dirs.append((math.cos(az) * math.cos(el) * 0.85, math.sin(az) * math.cos(el), math.sin(el) + 0.3))
        lens.append(rng.uniform(0.2, 0.31))
        brs.append(rng.uniform(0.055, 0.075))
    objs.append(spikes(f"{tag}Spikes", H + Vector((0, -0.05, 0.05)), dirs, lens, brs, m["hair"], bend=(0, -0.3, 0.15)))
    side = [(sx * 0.9, 0.35, -1.0) for sx in (-1, -0.6, 0.6, 1)]
    objs.append(spikes(f"{tag}Bangs", H + Vector((0, 0.05, 0.08)), side, [0.17, 0.13, 0.13, 0.17], [0.03] * 4,
                       m["hair"]))
    return _assemble(tag, objs, loc, yaw_deg)


def _assemble(tag, objs, loc, yaw_deg):
    root = bpy.data.objects.new(tag, None)
    fpv.link(root)
    for o in objs:
        o.parent = root
    root.location = loc
    root.rotation_euler = (0, 0, math.radians(yaw_deg))
    return root
