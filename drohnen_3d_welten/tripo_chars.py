"""Tripo-3D-Modelle (statische FBX-Meshes mit Farbtextur) als animierbare Figuren auf dem Gelenk-Rig.

- Modell -> Figurenkoordinaten (Blick +Y, rechte Körperseite +X, Füße auf z = 0, Zielhöhe).
- Gelenkpunkte im Modell (aus der Mittellinie des voxelisierten Modells bestimmt, siehe Tabellen unten).
- Ruhelage der Figur (figures.Figure): Proportionen des Modells, aber die Standardrichtungen des Rigs (Arme
  hängend, Beine gerade) – damit gelten alle Posen der Posenbibliothek unverändert.
- Armature in der *Pose des Modells*: Ruhelage jedes Knochens = Modellpose (Ort des Gelenks im Modell, Drehung
  Standardrichtung -> Modellrichtung). Jeder Knochen übernimmt per Copy Transforms die Weltmatrix seines
  Gelenk-Empties; die Verformung ist damit Empty-Welt × (Modellpose)⁻¹. Das Mesh wird nie in eine Grundpose
  zurückgebogen: Verzerrungen hängen nur davon ab, wie weit eine Kampfpose von der Modellpose abweicht.
- Gewichte: Abstand zu den Knochenstrecken in der Modellpose, geteilt durch die Dicke des Glieds (so gewinnt
  das Glied, auf dessen Oberfläche ein Punkt liegt), nur Knochen der eigenen Körperseite, 3 stärkste,
  über die Mesh-Kanten geglättet.
- Schwanz (Freezer): eigene Knochenkette entlang der Mittellinie, direkt animiert (anime_chars.tail_follow).
- Toon-Material aus der Tripo-Farbtextur (harte Licht-/Schattenstufen gegen die Sonnenrichtung, farbige
  Schatten, Randlicht) und Inverted-Hull-Kontur wie bei anime_chars.
"""
import math
import os

import bpy
import numpy as np
from mathutils import Matrix, Vector

import fpv
from anime_chars import LIGHT, add_outline, outline_material
from figures import ORDER, REST, Figure

MODELS = os.path.join(fpv.ASSETS, "models", "dragonball")

TIP = {"head": "head_top", "wrist.R": "hand.R", "wrist.L": "hand.L", "ankle.R": "toe.R", "ankle.L": "toe.L"}
CHILD = {"root": "spine", "spine": "neck", "neck": "head", **TIP}
for _s in ("R", "L"):
    CHILD.update({f"shoulder.{_s}": f"elbow.{_s}", f"elbow.{_s}": f"wrist.{_s}", f"hip.{_s}": f"knee.{_s}",
                  f"knee.{_s}": f"ankle.{_s}"})

# ---- Freezer (Endform): Gelenke im importierten Modell (Höhe 1, Blick -Y, rechte Seite -X)
FREEZER_JOINTS = {
    "root": (-0.046, -0.06, 0.52), "spine": (-0.05, -0.08, 0.60), "neck": (-0.046, -0.042, 0.785),
    "head": (-0.046, -0.05, 0.83), "head_top": (-0.046, -0.08, 0.96),
    "shoulder.R": (-0.135, -0.03, 0.738), "elbow.R": (-0.205, -0.018, 0.650), "wrist.R": (-0.228, -0.095, 0.620),
    "hand.R": (-0.248, -0.14, 0.628),
    "shoulder.L": (0.045, -0.038, 0.748), "elbow.L": (0.132, -0.030, 0.662), "wrist.L": (0.197, -0.060, 0.607),
    "hand.L": (0.24, -0.066, 0.60),
    "hip.R": (-0.095, -0.05, 0.50), "knee.R": (-0.103, 0.06, 0.31), "ankle.R": (-0.102, 0.17, 0.105),
    "toe.R": (-0.102, 0.10, 0.045),
    "hip.L": (0.0, -0.07, 0.50), "knee.L": (0.004, -0.15, 0.345), "ankle.L": (-0.008, -0.162, 0.105),
    "toe.L": (-0.010, -0.232, 0.045),
}
FREEZER_TAIL = [  # Mittellinie vom Ansatz über die Schlaufe bis zur eingerollten Spitze (x, y, z, Radius)
    (-0.03, 0.07, 0.52, 0.055), (0.05, 0.11, 0.516, 0.048), (0.142, 0.114, 0.496, 0.049), (0.202, 0.118, 0.424, 0.046),
    (0.198, 0.118, 0.324, 0.033), (0.138, 0.118, 0.252, 0.034), (0.038, 0.122, 0.248, 0.036),
    (-0.038, 0.094, 0.296, 0.030), (-0.098, 0.11, 0.264, 0.030), (-0.178, -0.006, 0.296, 0.018),
    (-0.242, -0.014, 0.228, 0.014), (-0.238, 0.042, 0.156, 0.008)]
# ---- Son Goku (SSJ): Sprung-/Angriffspose – linker Arm offen nach vorn, rechte Faust an der Hüfte
# zurückgezogen, rechtes Bein nach hinten angewinkelt, linkes Bein gestreckt mit hängendem Fuß
GOKU_JOINTS = {
    "root": (-0.01, 0.14, 0.50), "spine": (0.0, 0.14, 0.58), "neck": (0.0, 0.10, 0.765),
    "head": (-0.01, 0.09, 0.80), "head_top": (-0.02, 0.05, 0.93),
    "shoulder.R": (-0.085, 0.13, 0.755), "elbow.R": (-0.01, 0.265, 0.672), "wrist.R": (-0.10, 0.22, 0.61),
    "hand.R": (-0.125, 0.205, 0.595),
    "shoulder.L": (0.085, 0.07, 0.765), "elbow.L": (0.11, -0.10, 0.705), "wrist.L": (0.115, -0.245, 0.70),
    "hand.L": (0.14, -0.30, 0.755),
    "hip.R": (-0.055, 0.14, 0.47), "knee.R": (-0.10, 0.215, 0.405), "ankle.R": (-0.075, 0.345, 0.335),
    "toe.R": (-0.07, 0.385, 0.27),
    "hip.L": (0.04, 0.12, 0.47), "knee.L": (0.035, 0.115, 0.29), "ankle.L": (0.02, 0.19, 0.09),
    "toe.L": (0.02, 0.235, 0.02),
}
GOKU_RADII = {"root": 0.075, "spine": 0.08, "neck": 0.04, "head": 0.075, "shoulder": 0.042, "elbow": 0.034,
              "wrist": 0.026, "hip": 0.06, "knee": 0.045, "ankle": 0.034}
FREEZER_RADII = {"root": 0.06, "spine": 0.06, "neck": 0.035, "head": 0.08, "shoulder": 0.032, "elbow": 0.026,
                 "wrist": 0.02, "hip": 0.046, "knee": 0.036, "ankle": 0.03}


# ------------------------------------------------------------------------------------------------ Import
MIXAMO = {"root": ("Hips", "head"), "spine": ("Spine1", "head"), "neck": ("Neck", "head"), "head": ("Head", "head"),
          "head_top": ("HeadTop_End", "head")}
for _s, _m in (("R", "Right"), ("L", "Left")):
    MIXAMO.update({f"shoulder.{_s}": (f"{_m}Arm", "head"), f"elbow.{_s}": (f"{_m}ForeArm", "head"),
                   f"wrist.{_s}": (f"{_m}Hand", "head"), f"hand.{_s}": (f"{_m}HandMiddle4", "tail"),
                   f"hip.{_s}": (f"{_m}UpLeg", "head"), f"knee.{_s}": (f"{_m}Leg", "head"),
                   f"ankle.{_s}": (f"{_m}Foot", "head"), f"toe.{_s}": (f"{_m}Toe_End", "head")})


def import_mesh(path, name):
    """FBX/GLB importieren, alle Transformationen ins Mesh übernehmen, nur das Mesh behalten. Hat das Modell ein
    Mixamo-Skelett (Tripo-Auto-Rig), werden dessen Gelenke als Gelenkpunkte zurückgegeben (sonst None)."""
    before = set(bpy.data.objects)
    if path.lower().endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        bpy.ops.import_scene.fbx(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    ob = max((o for o in new if o.type == "MESH"), key=lambda o: len(o.data.vertices))   # GLB: ohne Hilfskugel
    joints = None
    arms = [o for o in new if o.type == "ARMATURE"]
    if arms:
        arm = arms[0]
        bones = {b.name.split(":")[-1]: b for b in arm.data.bones}
        if all(m in bones for m, _ in MIXAMO.values()):
            M = arm.matrix_world
            joints = {j: tuple(M @ (bones[m].head_local if end == "head" else bones[m].tail_local))
                      for j, (m, end) in MIXAMO.items()}
    M = ob.matrix_world.copy()
    ob.parent = None
    for md in list(ob.modifiers):
        ob.modifiers.remove(md)
    ob.vertex_groups.clear()
    ob.data.transform(M)
    ob.matrix_world = Matrix.Identity(4)
    for o in new:
        if o is not ob:
            bpy.data.objects.remove(o, do_unlink=True)
    ob.name = ob.data.name = name
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob, joints


def _image_of(ob, prefer=("color", "basecolor", "base_color", "diffuse", "albedo")):
    """Farbtextur des Modells (bei mehreren Bildern die mit 'Color' im Namen); fehlt sie (nicht mitgeliefert),
    ein graues Ersatzbild."""
    imgs = [n.image for m in ob.data.materials if m and m.node_tree for n in m.node_tree.nodes
            if n.type == "TEX_IMAGE" and n.image]
    ok = [im for im in imgs if im.size[0] > 0]
    for im in ok:
        nm = (im.name + " " + os.path.basename(im.filepath)).lower()
        if any(k in nm for k in prefer) and "normal" not in nm:
            return im
    if ok:
        return ok[0]
    im = bpy.data.images.new("MissingColor", 4, 4)
    im.pixels = [0.7, 0.7, 0.7, 1.0] * 16
    print("WARNUNG: Farbtextur fehlt – graues Ersatzbild", flush=True)
    return im


def _normal_of(ob):
    for m in ob.data.materials:
        if m and m.node_tree:
            for n in m.node_tree.nodes:
                if n.type == "TEX_IMAGE" and n.image and n.image.size[0] > 0 and "normal" in n.image.name.lower():
                    return n.image
    return None


def _shrink(im, size):
    """Textur auf size × size verkleinern (Speicher: 10 Figuren × 4K wären über 1 GB) und eingebettet halten."""
    if im is None or not size or im.size[0] <= size:
        return im
    im.scale(size, size)
    im.pack()
    return im


def real_tex_material(name, image, normal=None, rough=0.55, sat=1.05, value=1.0, normal_strength=0.6, sheen=0.25):
    """Physikalisches Material aus der Tripo-Textur (Figurenlook im echten Licht der Szene): Farbtextur,
    Normal-Map, matte Oberfläche mit etwas Sheen (Stoff)."""
    mat, nb, out = fpv.new_material(name)
    tex = nb.node("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Cubic"
    hs = nb.node("ShaderNodeHueSaturation")
    hs.inputs["Saturation"].default_value = sat
    hs.inputs["Value"].default_value = value
    nb.link(tex.outputs["Color"], hs.inputs["Color"])
    bs = fpv.principled(nb, Base_Color=hs.outputs["Color"], Roughness=rough)
    bs.inputs["Specular IOR Level"].default_value = 0.35
    bs.inputs["Sheen Weight"].default_value = sheen
    if normal is not None:
        normal.colorspace_settings.name = "Non-Color"
        nt = nb.node("ShaderNodeTexImage")
        nt.image = normal
        nm = nb.node("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = normal_strength
        nb.link(nt.outputs["Color"], nm.inputs["Color"])
        nb.link(nm.outputs["Normal"], bs.inputs["Normal"])
    nb.link(bs.outputs[0], out.inputs[0])
    return mat


# ------------------------------------------------------------------------------------------------ Material
def toon_tex_material(name, image, lit=1.0, mid=0.8, shade=(0.52, 0.5, 0.66), bands=(-0.1, 0.35),
                      tint_lit=(1.0, 0.96, 0.9), rim=(1.0, 0.97, 0.9), rim_w=0.4, sat=1.15, value=1.0,
                      diffuse_mix=0.14, hair_glow=0.0):
    """Cel-Shading mit Farbtextur: Textur × Stufe (Schatten / Halbton / Licht aus N·L der Sonne), farbige
    Schatten, Randlicht; hair_glow > 0 lässt goldene Texturbereiche (SSJ-Haar) leicht leuchten."""
    mat, nb, out = fpv.new_material(name)
    tex = nb.node("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Cubic"
    hs = nb.node("ShaderNodeHueSaturation")
    hs.inputs["Saturation"].default_value = sat
    hs.inputs["Value"].default_value = value
    nb.link(tex.outputs["Color"], hs.inputs["Color"])
    base = hs.outputs["Color"]
    geo = nb.node("ShaderNodeNewGeometry")
    N = geo.outputs["Normal"]
    t = nb.math("MULTIPLY_ADD", nb.vmath("DOT_PRODUCT", N, LIGHT["sun"]), 0.5, 0.5)
    band = nb.ramp(t, [(0.0, shade), ((bands[0] + 1) / 2, [mid * k for k in tint_lit]),
                       ((bands[1] + 1) / 2, [lit * k for k in tint_lit])], interp="CONSTANT")
    col = nb.mix(1.0, base, band, blend="MULTIPLY")
    lw = nb.node("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    edge = nb.math("GREATER_THAN", lw.outputs["Facing"], 0.66)
    side = nb.math("MINIMUM", nb.math("MAXIMUM", nb.math("MULTIPLY_ADD", nb.vmath("DOT_PRODUCT", N, LIGHT["rim"]),
                                                          1.6, 0.25), 0.0), 1.0)
    col = nb.mix(nb.math("MULTIPLY", nb.math("MULTIPLY", edge, side), rim_w), col, [r * 1.3 for r in rim],
                 blend="ADD")
    if hair_glow:
        sep = nb.node("ShaderNodeSeparateColor")
        nb.link(tex.outputs["Color"], sep.inputs[0])
        r, g, b = sep.outputs[0], sep.outputs[1], sep.outputs[2]
        gold = nb.math("MINIMUM", nb.math("MAXIMUM", nb.math("MULTIPLY", nb.math("SUBTRACT", nb.math(
            "MINIMUM", r, g), nb.math("ADD", b, 0.22)), 4.0), 0.0), 1.0)
        col = nb.mix(nb.math("MULTIPLY", gold, hair_glow), col, base, blend="ADD")
    em = nb.node("ShaderNodeEmission")
    nb.link(col, em.inputs["Color"])
    df = fpv.principled(nb, Base_Color=base, Roughness=0.9)
    df.inputs["Specular IOR Level"].default_value = 0.0
    mix = nb.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = diffuse_mix
    nb.link(em.outputs[0], mix.inputs[1])
    nb.link(df.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    mat.cycles.emission_sampling = "NONE"
    return mat


# ------------------------------------------------------------------------------------------------ Rig
def _frame(primary, secondary):
    """Orthonormale Basis (x, y, z) mit y = primary, x ~ secondary."""
    y = Vector(primary).normalized()
    x = Vector(secondary) - y * Vector(secondary).dot(y)
    if x.length < 1e-6:
        x = y.orthogonal()
    x.normalize()
    z = x.cross(y)
    return Matrix((x, y, z)).transposed()


def _auto_radii(V, segs, height):
    """Dicke je Glied aus dem Mesh: Punkte, die (absolut) am nächsten an dieser Knochenstrecke liegen und auf
    ihrem mittleren Teil, davon das 40-%-Quantil des Abstands."""
    D = np.zeros((len(V), len(segs)))
    T = np.zeros_like(D)
    for b, (a, c) in enumerate(segs):
        ab = c - a
        t = ((V - a) @ ab) / max(ab @ ab, 1e-12)
        D[:, b] = np.linalg.norm(V - (a + np.clip(t, 0, 1)[:, None] * ab), axis=1)
        T[:, b] = t
    own = np.argmin(D, axis=1)
    out = []
    for b in range(len(segs)):
        m = (own == b) & (T[:, b] > 0.25) & (T[:, b] < 0.75)
        r = float(np.quantile(D[m, b], 0.4)) if m.sum() > 8 else 0.03 * height
        out.append(max(r, 0.012 * height))
    return out


def _geodesic_weights(V, E, segs, rads, spans, power=4.0):
    """Gewichte über Abstände entlang der Oberfläche: je Knochen Saatpunkte = Mesh-Punkte, die klar auf seinem
    Glied liegen (nächster Knochen nach Abstand/Dicke, Abstand < 1,5 Dicken, Projektion im Innenteil `spans`
    der Strecke); dann kürzester Weg über die Mesh-Kanten zu jeder Saat. So bleiben Finger an der Hand,
    Krallen am Fuß und sich berührende Glieder (Oberschenkel, Schwanz am Bein) getrennt. Lose Teile ohne Weg zu
    einer Saat bekommen die über das ganze Teil gemittelten Abstandsgewichte und bewegen sich damit starr."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    n, nb = len(V), len(segs)
    lens = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) + 1e-9
    G = coo_matrix((lens, (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    D = np.zeros((n, nb))
    Tt = np.zeros((n, nb))
    for b, ((a, c), r) in enumerate(zip(segs, rads)):
        ab = c - a
        t = ((V - a) @ ab) / max(ab @ ab, 1e-12)
        D[:, b] = np.linalg.norm(V - (a + np.clip(t, 0, 1)[:, None] * ab), axis=1) / r
        Tt[:, b] = t
    own = np.argmin(D, axis=1)
    geo = np.full((n, nb), np.inf)
    for b in range(nb):
        lo, hi = spans[b]
        seeds = np.where((own == b) & (D[:, b] < 1.5) & (Tt[:, b] >= lo) & (Tt[:, b] <= hi))[0]
        if len(seeds) == 0:
            seeds = np.array([int(np.argmin(D[:, b]))])
        geo[:, b] = dijkstra(G, directed=False, indices=seeds, min_only=True)
    R = np.array(rads)[None, :]
    W = 1.0 / (geo / R + 0.15) ** power
    lost = ~np.isfinite(geo).any(axis=1)
    W[~np.isfinite(W)] = 0.0
    if lost.any():
        _, lab = connected_components(G, directed=False)
        for c in np.unique(lab[lost]):
            m = lab == c
            wl = 1.0 / (D[m] + 0.08) ** 6
            W[m] = (wl / wl.sum(1, keepdims=True)).mean(0)
    return W


def rig_model(name, path, height, joints=None, radii=None, tail=None, n_tail=10, mat_kw=None, outline=0.007,
              look="toon", tex_size=None, smooth=3):
    """Figur aus einem Tripo-Modell: siehe Moduldoku. Rückgabe: figures.Figure mit .rig, .body_parts,
    .face_objs (leer), .tail_bones (falls Schwanz), .mats. look = "toon" (Cel-Shading + Kontur) oder "real"
    (Textur im echten Szenenlicht, ohne Kontur); tex_size verkleinert die Texturen; smooth = Glättungsdurchgänge der
    Gewichte (mehr für weite Kleidung wie Kimono-Ärmel)."""
    fig = Figure(name, height)
    ob, mixamo = import_mesh(path, name + "Body")
    joints = joints or mixamo
    img = _shrink(_image_of(ob), tex_size)
    nrm = _shrink(_normal_of(ob), tex_size) if look == "real" else None
    c = joints["root"]
    zs = np.zeros(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", zs)
    k = height / float(zs[2::3].max())                 # Modellhöhe (Tripo: ~1) -> Zielhöhe
    T = Matrix.Diagonal((k, k, k, 1.0)) @ Matrix.Rotation(math.pi, 4, "Z") @ Matrix.Translation((-c[0], -c[1], 0.0))
    ob.data.transform(T)
    Jm = {j: T @ Vector(p) for j, p in joints.items()}
    # ---- Ruhelage mit Modellproportionen, Standardrichtungen (symmetrisch gemittelt)
    std = {j: (Vector(REST[CHILD[j]][1]) - Vector(REST[j][1])).normalized() for j in CHILD if CHILD[j] in REST}
    std["wrist.R"] = std["elbow.R"]
    std["wrist.L"] = std["elbow.L"]
    std["ankle.R"] = std["ankle.L"] = Vector((0, 0.16, -0.06)).normalized()
    std["head"] = Vector((0, 0, 1))
    # Längen je Seite wie im Modell (nicht gemittelt): jeder Knochen bildet sein Stück Mesh starr ab, gleiche
    # Länge in Modellpose und Ruhelage -> keine Lücken/Dehnungen an Knie und Handgelenk
    L = {j: (Jm[CHILD[j]] - Jm[j]).length for j in CHILD}
    Jr = {"root": Vector((0, 0, Jm["root"].z))}
    for j in ("spine", "neck", "head"):
        par = REST[j][0]
        Jr[j] = Jr[par] + Vector((0, 0, L[par]))
    for base_j, par in (("shoulder", "spine"), ("hip", "root")):
        for sd, sg in (("R", 1), ("L", -1)):
            d = Jm[f"{base_j}.{sd}"] - Jm[par]
            Jr[f"{base_j}.{sd}"] = Jr[par] + Vector((sg * math.hypot(d.x, d.y), 0, d.z))
    for sd in ("R", "L"):
        for a, b in (("shoulder", "elbow"), ("elbow", "wrist"), ("hip", "knee"), ("knee", "ankle")):
            Jr[f"{b}.{sd}"] = Jr[f"{a}.{sd}"] + std[f"{a}.{sd}"] * L[f"{a}.{sd}"]
    for j, tp in TIP.items():
        Jr[tp] = Jr[j] + std[j] * L[j]
    for j in ORDER:
        fig.rest[j] = Jr[j].copy()
        par = REST[j][0]
        fig.J[j].location = Jr[j] - (Jr[par] if par else Vector())
    # Ferse und Ballen relativ zum Sprunggelenk (für den Bodenkontakt in figures.foot_drop): Sohle in der Höhe,
    # die das Gelenk im Modell über dem Boden liegt, Ballen bei der Zehenspitze
    az = min(Jm["ankle.R"].z, Jm["ankle.L"].z)
    foot = max((Jm["toe.R"] - Jm["ankle.R"]).to_2d().length, 0.05 * height)
    fig.foot_offs = [Vector((0, -0.25 * foot, -az)), Vector((0, 0.8 * foot, -az))]
    # ---- Ruhelage der Knochen = Modellpose (Rumpf mit Querachse, Glieder minimal gedreht)
    B = {}
    side_axis = {"root": ("hip.R", "hip.L"), "spine": ("shoulder.R", "shoulder.L"), "neck": ("shoulder.R", "shoulder.L"),
                 "head": ("shoulder.R", "shoulder.L")}
    for j in ORDER:
        d_r, d_m = Jr[CHILD[j]] - Jr[j], Jm[CHILD[j]] - Jm[j]
        if j in side_axis:
            a, b = side_axis[j]
            Rm = _frame(d_m, Jm[a] - Jm[b]) @ _frame(d_r, Jr[a] - Jr[b]).inverted()
        else:
            Rm = d_r.rotation_difference(d_m).to_matrix()
        B[j] = Matrix.Translation(Jm[j]) @ Rm.to_4x4()
    # Schwanz: Kette entlang der Mittellinie, Knochen achsparallel
    tail_pts, tail_r = [], []
    if tail:
        P = np.array([T @ Vector(p[:3]) for p in tail])
        R = np.array([p[3] for p in tail]) * k
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        u = np.linspace(0, s[-1], n_tail + 1)
        tail_pts = [Vector([np.interp(v, s, P[:, k]) for k in range(3)]) for v in u]
        tail_r = np.interp(u, s, R)
    # ---- Gewichte in der Modellpose
    V = np.zeros(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    names, segs, rads = [], [], []
    for j in ORDER:
        names.append(j)
        segs.append((np.array(Jm[j]), np.array(Jm[CHILD[j]])))
        rads.append(radii[j.split(".")[0]] * k if radii else None)
    if not radii:
        rads = _auto_radii(V, segs, height)
    for i in range(len(tail_pts) - 1):
        names.append(f"tail{i}")
        segs.append((np.array(tail_pts[i]), np.array(tail_pts[i + 1])))
        rads.append(max((tail_r[i] + tail_r[i + 1]) / 2, 0.012 * height))
    E = np.zeros(len(ob.data.edges) * 2, np.int64)
    ob.data.edges.foreach_get("vertices", E)
    E = E.reshape(-1, 2)
    span = {"root": (-1.5, 0.85), "head": (0.15, 3.0), "wrist.R": (0.15, 4.0), "wrist.L": (0.15, 4.0),
            "ankle.R": (0.15, 4.0), "ankle.L": (0.15, 4.0)}
    if names[-1].startswith("tail"):
        span[names[-1]] = (0.15, 3.0)
    # Nähte schließen: GLB-Meshes sind an den UV-Nähten aufgetrennt (über 1000 Inseln). Für Oberflächenabstände
    # und Glättung werden Punkte an gleicher Stelle zusammengelegt, die Gewichte danach auf alle Kopien verteilt –
    # sonst springen die Gewichte an jeder Naht und Kleidung reißt auf.
    _, rep, inv = np.unique(np.round(V / (1e-5 * height)).astype(np.int64), axis=0, return_index=True,
                            return_inverse=True)
    inv = inv.ravel()
    Vw = V[rep]
    Ew = inv[E]
    Ew = np.unique(np.sort(Ew[Ew[:, 0] != Ew[:, 1]], axis=1), axis=0)
    W = _geodesic_weights(Vw, Ew, segs, rads, [span.get(nm, (0.15, 0.85)) for nm in names])

    def top3(W):
        top = np.argsort(-W, axis=1)[:, :3]
        Wt = np.take_along_axis(W, top, 1)
        Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-12)
        out = np.zeros_like(W)
        np.put_along_axis(out, top, Wt, 1)
        return out
    W = top3(W)
    cnt = np.bincount(Ew.ravel(), minlength=len(Vw)).astype(float)[:, None]
    for _ in range(smooth):
        acc = np.zeros_like(W)
        np.add.at(acc, Ew[:, 0], W[Ew[:, 1]])
        np.add.at(acc, Ew[:, 1], W[Ew[:, 0]])
        W = 0.5 * W + 0.5 * acc / np.maximum(cnt, 1)
    W = top3(W)[inv]
    for bi, nm in enumerate(names):
        vg = ob.vertex_groups.new(name=nm)
        sel = np.where(W[:, bi] > 1e-4)[0]
        for vi, w in zip(sel.tolist(), W[sel, bi].tolist()):
            vg.add([vi], float(w), "REPLACE")
    # ---- Armature (Ruhelage = Modellpose), Copy Transforms von den Gelenk-Empties
    ad = bpy.data.armatures.new(name + "Rig")
    arm = bpy.data.objects.new(name + "Rig", ad)
    fpv.link(arm)
    arm.parent = fig.base
    bpy.context.view_layer.objects.active = arm
    blen = 0.05 * fig.s
    with bpy.context.temp_override(active_object=arm, object=arm, selected_objects=[arm]):
        bpy.ops.object.mode_set(mode="EDIT")
        eb = {}
        for j in ORDER:
            b = ad.edit_bones.new(j)
            b.head, b.tail = (0, 0, 0), (0, blen, 0)
            b.matrix = B[j]
            eb[j] = b
        for j in ORDER:
            if REST[j][0]:
                eb[j].parent = eb[REST[j][0]]
        for i in range(len(tail_pts) - 1):
            b = ad.edit_bones.new(f"tail{i}")
            b.head = tail_pts[i]
            b.tail = tail_pts[i] + Vector((0, blen, 0))
            b.roll = 0.0
            b.parent = eb["root"] if i == 0 else eb[f"tail{i - 1}"]
            eb[f"tail{i}"] = b
        bpy.ops.object.mode_set(mode="OBJECT")
    for j in ORDER:
        pb = arm.pose.bones[j]
        cst = pb.constraints.new("COPY_TRANSFORMS")
        cst.target = fig.J[j]
        cst.owner_space = cst.target_space = "WORLD"
    tail_bones = [f"tail{i}" for i in range(len(tail_pts) - 1)]
    for bn in tail_bones:
        arm.pose.bones[bn].rotation_mode = "XYZ"
    ob.parent = fig.base
    mod = ob.modifiers.new("Rig", "ARMATURE")
    mod.object = arm
    # ---- Look
    if look == "real":
        mat = real_tex_material(name + "Mat", img, nrm, **(mat_kw or {}))
    else:
        mat = toon_tex_material(name + "Toon", img, **(mat_kw or {}))
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    line = outline_material(name + "Line", (0.02, 0.012, 0.02))
    if look != "real" and outline:
        add_outline(ob, outline * height, line)
    fig.rig = arm
    fig.body_parts = [ob]
    fig.face_objs = []
    fig.tail_bones = tail_bones
    fig.mats = {"body": mat, "line": line}
    fig.model_joints = Jm
    return fig


def freezer():
    """Freezer (Endform) aus dem Tripo-Modell, 1,50 m, Schwanz als 10-gliedrige Kette."""
    return rig_model("Freezer", os.path.join(MODELS, "freezer", "frieza+character+3d+model.fbx"), 1.50,
                     FREEZER_JOINTS, FREEZER_RADII, tail=FREEZER_TAIL, n_tail=10,
                     mat_kw=dict(sat=1.1, shade=(0.55, 0.52, 0.72), rim=(0.92, 0.88, 1.0)))


def goku():
    """Son Goku (Super-Saiyajin) aus dem Tripo-Modell (zweite, feinere Fassung), 1,75 m; das goldene Haar
    leuchtet leicht."""
    return rig_model("Goku", os.path.join(MODELS, "goku2", "dragon+ball+goku+3d+model.fbx"), 1.75,
                     GOKU_JOINTS, GOKU_RADII,
                     mat_kw=dict(sat=1.2, value=1.08, shade=(0.5, 0.46, 0.62), hair_glow=0.35))


# ---- dritte Fassung (Tripo mit Auto-Rig, neutrale Standpose, fein): Gelenke aus dem Mixamo-Skelett
FREEZER3_TAIL = [
    (0.038, -0.002, 0.5, 0.029), (0.022, 0.03, 0.452, 0.027), (-0.002, 0.062, 0.408, 0.026),
    (-0.03, 0.098, 0.38, 0.026), (-0.07, 0.142, 0.356, 0.025), (-0.118, 0.166, 0.34, 0.024), (-0.17, 0.178, 0.332, 0.02),
    (-0.222, 0.186, 0.312, 0.017), (-0.25, 0.178, 0.264, 0.013), (-0.238, 0.142, 0.22, 0.013),
    (-0.21, 0.102, 0.196, 0.009), (-0.178, 0.058, 0.18, 0.008)]


def goku3():
    """Son Goku, dritte Tripo-Fassung (Standpose, Mixamo-Rig), 1,75 m."""
    return rig_model("Goku", os.path.join(MODELS, "goku3", "tripo_convert_9e9f1b26-4f9f-49cc-9e4c-e02278354760.fbx"),
                     1.75, mat_kw=dict(sat=1.2, value=1.08, shade=(0.5, 0.46, 0.62), hair_glow=0.35))


def freezer3():
    """Freezer, dritte Tripo-Fassung (Standpose, Mixamo-Rig), 1,50 m, Schwanz als 10-gliedrige Kette."""
    return rig_model("Freezer", os.path.join(MODELS, "freezer3", "tripo_convert_c3ef3e3b-3cd4-4b30-87dd-28ba80182580.fbx"),
                     1.50, tail=FREEZER3_TAIL, n_tail=10,
                     mat_kw=dict(sat=1.1, shade=(0.55, 0.52, 0.72), rim=(0.92, 0.88, 1.0)))


def freezer4():
    """Freezer, GLB-Fassung (wie die dritte, aber mit eingebetteter Textur): Standpose, Mixamo-Rig, 1,50 m."""
    return rig_model("Freezer", os.path.join(MODELS, "freezer4", "friezafinalform3dmodel.glb"), 1.50,
                     tail=FREEZER3_TAIL, n_tail=10, mat_kw=dict(sat=1.05, shade=(0.58, 0.55, 0.74), rim=(0.92, 0.88, 1.0)),
                     outline=0.004)


def goku5():
    """Son Goku, Super-Saiyajin (GLB-Fassung mit Textur, Standpose, Mixamo-Rig), 1,75 m. Die GLB-Datei war ein
    unvollständiger Download; die reparierte Fassung hat neu berechnete Bind-Matrizen und keine Metallic-Map."""
    return rig_model("Goku", os.path.join(MODELS, "goku5", "gokuactionfigure3dmodel_repariert.glb"), 1.75,
                     mat_kw=dict(sat=1.12, value=1.04, shade=(0.52, 0.48, 0.64), hair_glow=0.25), outline=0.004)
