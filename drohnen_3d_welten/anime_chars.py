"""Son Goku (Super-Saiyajin) und Freezer (Endform) im Anime-Look für den Namek-Kampf (v2).

Aufbau je Figur:
- figures.Figure liefert die Gelenk-Hierarchie (Empties). Blocking und Animation laufen unverändert über
  choreo.bake_fighter; Effekte hängen weiter an den Empties (z. B. J["wrist.R"]).
- Körper: EIN durchgehendes Mesh (Skin-Modifier über ein Skelett mit Muskelradien) statt Segmenten mit
  Kugelgelenken. Es wird im A-Pose erzeugt (Arme/Beine frei -> saubere Gewichte), die Gewichte werden selbst aus
  dem Abstand zu den Knochenstrecken berechnet und das Mesh dann per Linear-Blend-Skinning in die Ruhelage des
  Rigs (Arme hängend) zurückgestellt. Eine Armature mit denselben Gelenken verformt es; jeder Knochen kopiert die
  lokale Drehung seines Empties (gleiche, achsparallele Ruhelage). Freezers Schwanz: eigene Knochenkette mit
  Nachschwingen (tail_follow).
- Kopf als eigenes Mesh (Kiefer, Kinn, Nasenansatz), Gesichtszüge als auf die Kopfoberfläche projizierte
  Decals mit Kontur: Anime-Augen (Lidstrich, Iris, Pupille, Glanzpunkt), Brauen, Nase, Mund. Ausdrücke sind
  Shape Keys (Mund, Brauen, Lider), gesteuert über set_expression().
- Toon-Shading in Cycles ("Shader to RGB" gibt es nur in EEVEE): Licht-/Schattenstufen aus der bekannten
  Sonnenrichtung (N·L -> Color Ramp, konstant), farbige Schatten, Randlicht, Glanzpunkt; ein kleiner Anteil
  echter Diffusion lässt Energie-Lichter (Aura, Strahlen, Explosion) die Figuren färben.
- Konturen per Inverted Hull (Solidify mit gedrehten Normalen; die zur Kamera weisenden Rückseiten sind
  transparent) – Freestyle wäre mit dem Gras der Szene viel zu langsam.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector
from mathutils.bvhtree import BVHTree

import fpv
from figures import REST, Figure

# ------------------------------------------------------------------------------------------------ Shading
LIGHT = {"sun": (0.0, 0.0, 1.0), "rim": (0.0, 1.0, 0.3)}


def set_light(sun_dir, rim_dir):
    """Richtungen *zur* Sonne bzw. zum Randlicht (Welt), vor dem Bau der Figuren setzen."""
    LIGHT["sun"] = tuple(Vector(sun_dir).normalized())
    LIGHT["rim"] = tuple(Vector(rim_dir).normalized())


def _rgba(c):
    return (*c, 1.0) if len(c) == 3 else tuple(c)


def toon_material(name, base, shade, lit=0.92, mid=0.66, bands=(-0.1, 0.35), tint_lit=(1.0, 0.93, 0.84),
                  tint_shade=(0.82, 0.88, 1.0), rim=(1.0, 0.97, 0.9), rim_w=0.55, spec=0.0, glow=0.0,
                  diffuse_mix=0.14):
    """Cel-Shading: drei harte Stufen (Schatten / Halbton / Licht) aus N·L der Sonne, Schatten in eigener,
    farbiger Tönung (kein Schwarz), dünnes Randlicht an der Silhouette, optional Glanzpunkt (spec) und
    Eigenleuchten (glow). diffuse_mix: Anteil echter Diffusion (Effektlichter)."""
    mat, nb, out = fpv.new_material(name)
    geo = nb.node("ShaderNodeNewGeometry")
    N = geo.outputs["Normal"]
    ndl = nb.vmath("DOT_PRODUCT", N, LIGHT["sun"])
    t = nb.math("MULTIPLY_ADD", ndl, 0.5, 0.5)
    c_lit = [b * lit * k for b, k in zip(base, tint_lit)]
    c_mid = [b * mid * k for b, k in zip(base, tint_lit)]
    c_sh = [s * k for s, k in zip(shade, tint_shade)]
    col = nb.ramp(t, [(0.0, c_sh), ((bands[0] + 1) / 2, c_mid), ((bands[1] + 1) / 2, c_lit)], interp="CONSTANT")
    lw = nb.node("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    edge = nb.math("GREATER_THAN", lw.outputs["Facing"], 0.66)
    side = nb.math("MINIMUM", nb.math("MAXIMUM", nb.math("MULTIPLY_ADD", nb.vmath("DOT_PRODUCT", N, LIGHT["rim"]),
                                                          1.6, 0.25), 0.0), 1.0)
    rimv = nb.math("MULTIPLY", nb.math("MULTIPLY", edge, side), rim_w)
    col = nb.mix(rimv, col, [r * 1.3 for r in rim], blend="ADD")
    if spec:
        H = nb.vmath("NORMALIZE", nb.vmath("ADD", geo.outputs["Incoming"], LIGHT["sun"]))
        hs = nb.math("GREATER_THAN", nb.vmath("DOT_PRODUCT", N, H), 0.975)
        col = nb.mix(nb.math("MULTIPLY", hs, spec), col, (1.6, 1.6, 1.6), blend="ADD")
    em = nb.node("ShaderNodeEmission")
    nb.link(col, em.inputs["Color"])
    em.inputs["Strength"].default_value = 1.0 + glow
    df = fpv.principled(nb, Base_Color=base, Roughness=0.9)
    df.inputs["Specular IOR Level"].default_value = 0.0
    mix = nb.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = diffuse_mix
    nb.link(em.outputs[0], mix.inputs[1])
    nb.link(df.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    mat.cycles.emission_sampling = "NONE"
    return mat


def flat_material(name, color, level=1.0):
    """Ungeschattete Fläche (Augenweiß, Iris, Linien)."""
    mat, nb, out = fpv.new_material(name)
    em = nb.node("ShaderNodeEmission")
    em.inputs["Color"].default_value = _rgba(color)
    em.inputs["Strength"].default_value = level
    nb.link(em.outputs[0], out.inputs[0])
    mat.cycles.emission_sampling = "NONE"
    return mat


def outline_material(name, color=(0.012, 0.008, 0.006)):
    """Inverted-Hull-Kontur: Rückseiten (die vorderen Hüllflächen) transparent, Vorderseiten dunkel."""
    mat, nb, out = fpv.new_material(name)
    geo = nb.node("ShaderNodeNewGeometry")
    em = nb.node("ShaderNodeEmission")
    em.inputs["Color"].default_value = _rgba(color)
    tr = nb.node("ShaderNodeBsdfTransparent")
    mix = nb.node("ShaderNodeMixShader")
    nb.link(geo.outputs["Backfacing"], mix.inputs[0])
    nb.link(em.outputs[0], mix.inputs[1])
    nb.link(tr.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    mat.cycles.emission_sampling = "NONE"
    return mat


def add_outline(ob, thickness, mat):
    """Kontur als Hülle: Solidify nach außen, Normalen gedreht, eigene Materialplätze für die Hülle."""
    n = max(len(ob.data.materials), 1)
    if not ob.data.materials:
        ob.data.materials.append(mat)
    for _ in range(n):
        ob.data.materials.append(mat)
    so = ob.modifiers.new("Outline", "SOLIDIFY")
    so.thickness = thickness
    so.offset = 1.0
    so.use_flip_normals = True
    so.use_rim = False
    so.material_offset = n
    so.use_quality_normals = True
    return so


# ------------------------------------------------------------------------------------------------ Körper
def _skin_eval(name, nodes):
    """nodes: [(pos, (rx, ry), parent_index)] -> ausgewertetes Mesh (Skin + Subdivision 1) als numpy-Daten."""
    me = bpy.data.meshes.new(name + "_skel")
    me.from_pydata([tuple(p) for p, _, _ in nodes], [(i, p) for i, (_, _, p) in enumerate(nodes) if p is not None], [])
    ob = bpy.data.objects.new(name + "_skel", me)
    fpv.link(ob)
    sk = ob.modifiers.new("skin", "SKIN")
    sk.branch_smoothing = 0.4
    sk.use_smooth_shade = True
    sv = me.skin_vertices[0].data
    for i, (_, r, par) in enumerate(nodes):
        sv[i].radius = r
        sv[i].use_root = par is None
    sub = ob.modifiers.new("sub", "SUBSURF")
    sub.levels = 1
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    m2 = bpy.data.meshes.new_from_object(ev)
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.meshes.remove(me)
    V = np.zeros(len(m2.vertices) * 3, np.float32)
    m2.vertices.foreach_get("co", V)
    faces = [tuple(p.vertices) for p in m2.polygons]
    bpy.data.meshes.remove(m2)
    return V.reshape(-1, 3).astype(float), faces


def _seg_dist(P, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _rot_about(p, axis, deg):
    return Matrix.Translation(p) @ Matrix.Rotation(math.radians(deg), 4, axis) @ Matrix.Translation(-Vector(p))


class AnimeBody:
    """Skelett (Skin-Knoten in A-Pose) -> gewichtetes, in die Rig-Ruhelage zurückgestelltes Körper-Mesh."""

    ARM_A, LEG_A = 42.0, 7.0          # A-Pose: Arme / Beine seitlich abgespreizt (Grad)

    def __init__(self, fig, extra_bones=()):
        self.fig = fig
        s = fig.s
        self.R = {j: Vector(p) * s for j, (_, p) in REST.items()}
        self.M = {}                    # A-Pose-Transform je Knochen (Ruhelage -> A-Pose)
        for sd, x in (("R", 1), ("L", -1)):
            Ma = _rot_about(self.R[f"shoulder.{sd}"], "Y", -x * self.ARM_A)
            Ml = _rot_about(self.R[f"hip.{sd}"], "Y", -x * self.LEG_A)
            for j in ("shoulder", "elbow", "wrist"):
                self.M[f"{j}.{sd}"] = Ma
            for j in ("hip", "knee", "ankle"):
                self.M[f"{j}.{sd}"] = Ml
        for j in ("root", "spine", "neck", "head"):
            self.M[j] = Matrix.Identity(4)
        self.extra = list(extra_bones)  # [(name, parent, head_rest, tail_rest)]
        for (nm, _, _, _) in self.extra:
            self.M[nm] = Matrix.Identity(4)

    def A(self, j, p):
        """Punkt p (Ruhelage, Figurenmaßstab) in A-Pose des Knochens j."""
        return self.M[j] @ Vector(p)

    def segments(self):
        R = self.R
        s = self.fig.s
        seg = {"root": (R["root"] - Vector((0, 0, 0.1 * s)), R["spine"]),
               "spine": (R["spine"], R["neck"]), "neck": (R["neck"], R["head"]),
               "head": (R["head"], R["head"] + Vector((0, 0, 0.26 * s)))}
        for sd in ("R", "L"):
            seg[f"shoulder.{sd}"] = (R[f"shoulder.{sd}"], R[f"elbow.{sd}"])
            seg[f"elbow.{sd}"] = (R[f"elbow.{sd}"], R[f"wrist.{sd}"])
            seg[f"wrist.{sd}"] = (R[f"wrist.{sd}"], R[f"wrist.{sd}"] + Vector((0, 0.01, -0.12)) * s)
            seg[f"hip.{sd}"] = (R[f"hip.{sd}"], R[f"knee.{sd}"])
            seg[f"knee.{sd}"] = (R[f"knee.{sd}"], R[f"ankle.{sd}"])
            seg[f"ankle.{sd}"] = (R[f"ankle.{sd}"], R[f"ankle.{sd}"] + Vector((0, 0.16, -0.06)) * s)
        for (nm, _, h, t) in self.extra:
            seg[nm] = (Vector(h), Vector(t))
        return seg

    def build(self, name, nodes_A, mat_fn, mats):
        """nodes_A: Skin-Knoten bereits in A-Pose. mat_fn(center_rest, dominant_bone) -> Materialindex."""
        V, faces = _skin_eval(name, nodes_A)
        seg = self.segments()
        names = list(seg)
        D = np.stack([_seg_dist(V, self.A(nm, a) if nm in self.M else a, self.A(nm, b) if nm in self.M else b)
                      for nm, (a, b) in seg.items()], axis=1)
        side = np.array([1 if n.endswith(".R") else (-1 if n.endswith(".L") else 0) for n in names])
        bad = ((side[None, :] > 0) & (V[:, :1] < -0.02 * self.fig.s)) | ((side[None, :] < 0) & (V[:, :1] > 0.02 * self.fig.s))
        W = 1.0 / (D + 0.004 * self.fig.s) ** 6
        W[bad] = 0.0
        top = np.argsort(-W, axis=1)[:, :3]
        Wt = np.take_along_axis(W, top, 1)
        Wt /= Wt.sum(1, keepdims=True)
        # zurück in die Ruhelage (Linear-Blend-Skinning mit den inversen A-Pose-Transforms)
        Minv = {nm: np.array(self.M[nm].inverted()) for nm in names}
        Vh = np.hstack([V, np.ones((len(V), 1))])
        Vr = np.zeros_like(V)
        for k in range(3):
            for bi, nm in enumerate(names):
                sel = top[:, k] == bi
                if sel.any():
                    Vr[sel] += Wt[sel, k:k + 1] * (Vh[sel] @ Minv[nm].T)[:, :3]
        me = bpy.data.meshes.new(name)
        me.from_pydata(Vr.tolist(), [], faces)
        for p in me.polygons:
            p.use_smooth = True
        for m in mats:
            me.materials.append(m)
        dom = np.array(names)[top[:, 0]]
        idx = []
        for p in me.polygons:
            vs = list(p.vertices)
            c = Vr[vs].mean(0)
            db = max(set(dom[vs].tolist()), key=dom[vs].tolist().count)
            idx.append(mat_fn(c / self.fig.s, db))
        me.polygons.foreach_set("material_index", np.array(idx, np.int32))
        ob = bpy.data.objects.new(name, me)
        fpv.link(ob)
        ob.parent = self.fig.base
        for bi, nm in enumerate(names):
            vg = ob.vertex_groups.new(name=nm)
            for k in range(3):
                sel = np.where((top[:, k] == bi) & (Wt[:, k] > 1e-4))[0]
                for vi, w in zip(sel.tolist(), Wt[sel, k].tolist()):
                    vg.add([vi], float(w), "ADD")
        self.body = ob
        return ob

    def armature(self):
        """Armature mit denselben Gelenken (Knochen achsparallel -> lokale Achsen = Empty-Achsen), Drehungen per
        Copy-Rotation von den Empties; Zusatzknochen (Schwanz) werden direkt animiert."""
        fig = self.fig
        ad = bpy.data.armatures.new(fig.name + "Rig")
        arm = bpy.data.objects.new(fig.name + "Rig", ad)
        fpv.link(arm)
        arm.parent = fig.base
        bpy.context.view_layer.objects.active = arm
        with bpy.context.temp_override(active_object=arm, object=arm, selected_objects=[arm]):
            bpy.ops.object.mode_set(mode="EDIT")
            eb = {}
            for j, (par, _) in REST.items():
                b = ad.edit_bones.new(j)
                b.head = self.R[j]
                b.tail = self.R[j] + Vector((0, 0.05 * fig.s, 0))
                b.roll = 0.0
                eb[j] = b
            for j, (par, _) in REST.items():
                if par:
                    eb[j].parent = eb[par]
            for (nm, par, h, t) in self.extra:
                b = ad.edit_bones.new(nm)
                b.head = Vector(h)
                b.tail = Vector(h) + Vector((0, 0.05 * fig.s, 0))
                b.roll = 0.0
                b.parent = eb[par]
                eb[nm] = b
            bpy.ops.object.mode_set(mode="OBJECT")
        for j in REST:
            pb = arm.pose.bones[j]
            pb.rotation_mode = "XYZ"
            c = pb.constraints.new("COPY_ROTATION")
            c.target = fig.J[j]
            c.owner_space = "LOCAL"
            c.target_space = "LOCAL"
            c.euler_order = "XYZ"
        for (nm, _, _, _) in self.extra:
            arm.pose.bones[nm].rotation_mode = "XYZ"
        mod = self.body.modifiers.new("Rig", "ARMATURE")
        mod.object = arm
        self.body.modifiers.move(len(self.body.modifiers) - 1, 0)
        self.arm = arm
        return arm


# ------------------------------------------------------------------------------------------------ Kopf & Gesicht
def head_mesh(name, size, shape_fn, mat, seg=(48, 32)):
    """Kopf aus UV-Kugel, pro Vertex geformt: shape_fn(x, y, z) auf der Einheitskugel -> neue Einheits-Position."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg[0], v_segments=seg[1], radius=1.0)
    for v in bm.verts:
        v.co = Vector(shape_fn(*v.co))
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    return ob


class Face:
    """Decals auf der Kopfoberfläche: 2D-Umrisse (x = rechts der Figur, z = oben, m) werden entlang -Y auf das
    Kopf-Mesh projiziert und um `lift` entlang der Normale abgehoben. Shape Keys: gleiche Punktzahl je Zustand."""

    def __init__(self, head_ob, center):
        self.head = head_ob
        bm = bmesh.new()
        bm.from_mesh(head_ob.data)
        bmesh.ops.transform(bm, matrix=head_ob.matrix_basis, verts=bm.verts)
        self.bvh = BVHTree.FromBMesh(bm)
        bm.free()
        self.C = Vector(center)
        self.objs = []

    def _proj(self, x, z, lift):
        o = self.C + Vector((x, 0.5, z))
        hit, nrm, _, _ = self.bvh.ray_cast(o, Vector((0, -1, 0)))
        if hit is None:
            return self.C + Vector((x, 0.1, z))
        return hit + nrm * lift

    def decal(self, name, outlines, mat, lift=0.0012, keys=None):
        """outlines: Punkte (x, z) eines geschlossenen Umrisses (auch konkav); keys: {Name: Punkte} gleicher
        Länge für Shape Keys. Trianguliert wird in 2D am flächengrößten Zustand (Ear Clipping), damit auch der
        geöffnete Mund saubere Dreiecke hat."""
        from mathutils.geometry import tessellate_polygon

        def area(P):
            return abs(sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))) / 2
        states = [outlines] + (list(keys.values()) if keys else [])
        ref = max(states, key=area)
        tris = tessellate_polygon([[Vector((x, z, 0.0)) for (x, z) in ref]])
        bm = bmesh.new()
        ring = [bm.verts.new(self._proj(x, z, lift)) for (x, z) in outlines]
        for a, b, c in tris:
            try:
                bm.faces.new([ring[a], ring[b], ring[c]])
            except ValueError:
                pass
        ob = fpv.mesh_from_bmesh(bm, name, mat, smooth=False)
        if keys:
            ob.shape_key_add(name="Basis")
            for kn, kp in keys.items():
                sk = ob.shape_key_add(name=kn)
                for v, (x, z) in zip(sk.data, kp):
                    v.co = self._proj(x, z, lift)
        ob.visible_shadow = False
        self.objs.append(ob)
        return ob

    def line(self, name, pts, width, mat, lift=0.0018, keys=None, taper=True):
        """Strich entlang pts (x, z) mit Breite width (Mitte dicker, Enden spitz) als Band."""
        def band(P):
            out_l, out_r = [], []
            n = len(P)
            for i, (x, z) in enumerate(P):
                a = P[max(i - 1, 0)]
                b = P[min(i + 1, n - 1)]
                tx, tz = b[0] - a[0], b[1] - a[1]
                ln = math.hypot(tx, tz) or 1.0
                nx, nz = -tz / ln, tx / ln
                w = width * (math.sin(math.pi * (i + 0.5) / n) ** 0.6 if taper else 1.0) / 2
                out_l.append((x + nx * w, z + nz * w))
                out_r.append((x - nx * w, z - nz * w))
            return out_l + out_r[::-1]
        return self.decal(name, band(pts), mat, lift=lift,
                          keys={k: band(v) for k, v in keys.items()} if keys else None)

    def attach(self, fig):
        for ob in self.objs:
            fig.attach(ob, "head")


def ellipse(cx, cz, rx, rz, n=20, a0=0.0):
    return [(cx + rx * math.cos(a0 + 2 * math.pi * k / n), cz + rz * math.sin(a0 + 2 * math.pi * k / n))
            for k in range(n)]


def resample(pts, n):
    """Polylinie auf n Punkte umverteilen (für Shape Keys mit gleicher Punktzahl)."""
    P = np.array(pts, float)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0, d[-1], n)
    return list(zip(np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])))


def closed_resample(pts, n):
    return resample(list(pts) + [pts[0]], n + 1)[:-1]


def set_expression(fig, keyframes):
    """keyframes: [(frame, {ShapeKeyName: Wert})] – setzt die Werte auf allen Gesichts-Objekten der Figur."""
    for ob in fig.face_objs:
        kb = ob.data.shape_keys
        if not kb:
            continue
        for f, vals in keyframes:
            for kn, kv in vals.items():
                if kn in kb.key_blocks:
                    kb.key_blocks[kn].value = kv
                    kb.key_blocks[kn].keyframe_insert("value", frame=f)


# ------------------------------------------------------------------------------------------------ Goku
def _spikes(name, center, specs, mat, flat=0.5):
    """Haarsträhnen als scharfe, abgeflachte, gebogene Kegel: specs = [(Richtung, Länge, Basisradius, Biegung)]."""
    bm = bmesh.new()
    C = Vector(center)
    for d, ln, br, bend in specs:
        d = Vector(d).normalized()
        side = d.cross(Vector((0, 0, 1)))
        if side.length < 1e-3:
            side = Vector((1, 0, 0))
        side.normalize()
        up = side.cross(d).normalized()
        n_s, n_a = 7, 8
        rings = []
        for i in range(n_s + 1):
            t = i / n_s
            c = C + d * (ln * t) + Vector(bend) * ln * t * t
            if i == n_s:
                rings.append([bm.verts.new(c)] * n_a)          # scharfe Spitze (ein Punkt)
                continue
            r = br * (1 - t) ** 1.25
            rings.append([bm.verts.new(c + side * (math.cos(2 * math.pi * k / n_a) * r)
                                       + up * (math.sin(2 * math.pi * k / n_a) * r * flat)) for k in range(n_a)])
        for i in range(n_s):
            for k in range(n_a):
                k2 = (k + 1) % n_a
                if i == n_s - 1:
                    bm.faces.new([rings[i][k], rings[i][k2], rings[i + 1][0]])
                else:
                    bm.faces.new([rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]])
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    return ob


def goku(expr_keys=None):
    """Son Goku, Super-Saiyajin (Namek-Saga), 1,75 m, Anime-Proportionen: breite Schultern, V-Form, kräftige
    Arme/Oberschenkel, große Stiefel; orangefarbener Gi, blaues Unterhemd (V am Hals, kurze Ärmel), blaue
    Handgelenks-/Gürtelbänder, dunkelblaue Stiefel; goldenes Stachelhaar mit Eigenleuchten; türkise Augen."""
    fig = Figure("Goku", 1.75, (0, 0, 0), 0.0)
    s = fig.s
    ORANGE, ORANGE_SH = (1.0, 0.36, 0.03), (0.60, 0.10, 0.05)
    BLUE, BLUE_SH = (0.07, 0.20, 0.78), (0.03, 0.06, 0.34)
    SKIN, SKIN_SH = (0.96, 0.66, 0.46), (0.66, 0.34, 0.24)
    BOOT, BOOT_SH = (0.06, 0.10, 0.36), (0.02, 0.03, 0.13)
    GOLD, GOLD_SH = (1.0, 0.80, 0.10), (0.85, 0.45, 0.04)
    m = {"gi": toon_material("GokuGiToon", ORANGE, ORANGE_SH),
         "blue": toon_material("GokuBlueToon", BLUE, BLUE_SH),
         "skin": toon_material("GokuSkinToon", SKIN, SKIN_SH, rim_w=0.4),
         "boot": toon_material("GokuBootToon", BOOT, BOOT_SH, spec=0.4),
         "hair": toon_material("GokuHairToon", GOLD, GOLD_SH, lit=0.95, mid=0.8, bands=(-0.25, 0.3), glow=0.12,
                               rim=(1.0, 1.0, 0.8), rim_w=0.7, diffuse_mix=0.05),
         "line": outline_material("GokuLine", (0.02, 0.01, 0.008))}
    body = AnimeBody(fig)
    X = lambda sd: 1 if sd == "R" else -1
    nodes = []

    def add(p, r, par, j=None):
        p = Vector(p) * s
        if j:
            p = body.A(j, p)
        nodes.append((p, (r[0] * s, r[1] * s), par))
        return len(nodes) - 1
    pel = add((0, 0, 0.92), (0.150, 0.112), None)
    wai = add((0, 0.012, 1.04), (0.125, 0.096), pel)
    chl = add((0, 0.022, 1.18), (0.168, 0.12), wai)
    chu = add((0, 0.018, 1.30), (0.19, 0.128), chl)
    nk0 = add((0, 0.0, 1.415), (0.074, 0.070), chu)
    add((0, 0.008, 1.50), (0.057, 0.055), nk0)
    for sd in ("R", "L"):
        x = X(sd)
        cl = add((x * 0.12, 0.0, 1.375), (0.086, 0.08), chu)
        de = add((x * 0.192, 0.0, 1.35), (0.082, 0.08), cl, f"shoulder.{sd}")
        bi = add((x * 0.203, 0.006, 1.235), (0.068, 0.068), de, f"shoulder.{sd}")
        el = add((x * 0.21, 0.0, 1.10), (0.050, 0.050), bi, f"shoulder.{sd}")
        fo = add((x * 0.214, 0.008, 1.02), (0.060, 0.056), el, f"elbow.{sd}")
        wr = add((x * 0.22, 0.0, 0.875), (0.041, 0.037), fo, f"elbow.{sd}")
        ha = add((x * 0.222, 0.006, 0.815), (0.045, 0.036), wr, f"wrist.{sd}")
        add((x * 0.222, 0.014, 0.772), (0.037, 0.031), ha, f"wrist.{sd}")
        hp = add((x * 0.088, 0.0, 0.875), (0.10, 0.10), pel, f"hip.{sd}")
        th = add((x * 0.091, 0.01, 0.70), (0.089, 0.088), hp, f"hip.{sd}")
        kn = add((x * 0.093, 0.0, 0.50), (0.062, 0.062), th, f"hip.{sd}")
        ca = add((x * 0.096, -0.012, 0.40), (0.066, 0.064), kn, f"knee.{sd}")
        b0 = add((x * 0.098, 0.0, 0.31), (0.074, 0.072), ca, f"knee.{sd}")
        an = add((x * 0.10, 0.0, 0.11), (0.062, 0.06), b0, f"knee.{sd}")
        add((x * 0.10, -0.04, 0.045), (0.052, 0.045), an, f"ankle.{sd}")
        ba = add((x * 0.10, 0.09, 0.045), (0.056, 0.042), an, f"ankle.{sd}")
        add((x * 0.10, 0.155, 0.04), (0.046, 0.034), ba, f"ankle.{sd}")
    arm_bones = {f"{j}.{sd}" for j in ("shoulder", "elbow", "wrist") for sd in ("R", "L")}

    def mat_fn(c, db):
        x, y, z = c
        if db in arm_bones:
            if z > 1.305:
                return 0                                   # Gi über der Schulter
            if z > 1.215:
                return 1                                   # kurzer blauer Ärmel des Unterhemds
            if 0.86 < z < 0.935:
                return 1                                   # Handgelenksband
            return 2                                       # Haut
        if z < 0.325:
            return 3                                       # Stiefel
        if db in ("neck", "head") or z > 1.455:
            return 2
        if 1.29 < z <= 1.455 and y > 0 and abs(x) < 0.075 * (z - 1.27) / 0.18:
            return 1                                       # blaues V des Unterhemds am Hals
        return 0                                           # Gi (Oberteil, Hose)
    body.build("GokuBody", nodes, mat_fn, [m["gi"], m["blue"], m["skin"], m["boot"]])
    body.armature()
    add_outline(body.body, 0.011 * s, m["line"])
    sub = body.body.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1
    body.body.modifiers.move(len(body.body.modifiers) - 1, 1)
    # Gürtel mit Knoten, Stiefelschaft-Band
    belt = [(math.cos(a) * 0.162, math.sin(a) * 0.124 + 0.004, 0.97) for a in np.linspace(0, 2 * math.pi, 49)]
    parts = [(_tube("GokuBelt", [tuple(Vector(p) * s) for p in belt], 0.027 * s, m["blue"]), "root")]
    for sd in ("R", "L"):
        x = X(sd)
        ring = [(x * 0.098 + math.cos(a) * 0.078, math.sin(a) * 0.074, 0.325) for a in np.linspace(0, 2 * math.pi, 33)]
        parts.append((_tube(f"GokuBootTop{sd}", [tuple(Vector(p) * s) for p in ring], 0.012 * s, m["blue"]),
                      f"knee.{sd}"))
    for ob, j in parts:
        _mesh_outline(ob, 0.006 * s, m["line"])
        fig.attach(ob, j)
    # Kopf: kantiger Kiefer, Kinn, Wangenknochen, Nasenansatz
    H = Vector((0, 0.012, 1.575)) * s

    def shape(x, y, z):
        if z < 0:
            k = 1 - 0.32 * min(-z / 0.95, 1) ** 1.4
            x *= k
            y = y * (1 - 0.12 * -z) + 0.10 * max(-z - 0.45, 0) * (y > 0)
        if y > 0.6 and -0.45 < z < 0.05 and abs(x) < 0.18:
            y += 0.09 * (1 - abs(x) / 0.18) * (1 - abs(z + 0.2) / 0.25) if abs(z + 0.2) < 0.25 else 0
        if y > 0.3 and -0.1 < z < 0.25:
            y -= 0.035                                     # Augenhöhlen leicht eingetieft
        return x, y, z
    head = head_mesh("GokuHead", (0.094 * s, 0.103 * s, 0.118 * s), shape, m["skin"])
    head.location = H
    ears = [fpv.mesh_from_bmesh(_ell_bm((0.013 * s, 0.022 * s, 0.032 * s)), f"GokuEar{sd}", m["skin"])
            for sd in ("R", "L")]
    for e, x in zip(ears, (1, -1)):
        e.location = H + Vector((x * 0.093, -0.004, -0.004)) * s
    face = Face(head, H)
    _goku_face(face, s)
    face.attach(fig)
    # Haar: große Stacheln oben/hinten, Seiten, eine Strähne in die Stirn; Haarkappe
    cap = fpv.mesh_from_bmesh(_ell_bm((0.1 * s, 0.112 * s, 0.1 * s)), "GokuHairCap", m["hair"])
    cap.location = H + Vector((0, -0.016, 0.045)) * s
    top = H + Vector((0, -0.01, 0.06)) * s
    sp = [((0.0, -0.22, 1.0), 0.30, 0.105, (0, -0.30, 0.05)), ((0.40, -0.25, 1.0), 0.28, 0.095, (0.12, -0.25, 0)),
          ((-0.40, -0.25, 1.0), 0.28, 0.095, (-0.12, -0.25, 0)), ((0.72, -0.40, 0.70), 0.24, 0.085, (0.12, -0.12, 0.12)),
          ((-0.72, -0.40, 0.70), 0.24, 0.085, (-0.12, -0.12, 0.12)), ((0.20, -0.80, 0.60), 0.25, 0.09, (0, -0.1, 0.22)),
          ((-0.20, -0.80, 0.60), 0.25, 0.09, (0, -0.1, 0.22)), ((0.95, -0.25, 0.30), 0.17, 0.07, (0.05, -0.1, 0.14)),
          ((-0.95, -0.25, 0.30), 0.17, 0.07, (-0.05, -0.1, 0.14)), ((0.62, -0.80, 0.10), 0.17, 0.075, (0, -0.1, 0.14)),
          ((-0.62, -0.80, 0.10), 0.17, 0.075, (0, -0.1, 0.14)), ((0.0, -1.0, 0.15), 0.17, 0.08, (0, 0, 0.14))]
    sp = [(d, ln * s, br * s, b) for d, ln, br, b in sp]
    hair = _spikes("GokuHair", top, sp, m["hair"], flat=0.42)
    bangs = _spikes("GokuBangs", H + Vector((0, 0.07, 0.085)) * s,
                    [((0.05, 0.75, -0.55), 0.11 * s, 0.036 * s, (0.05, 0.1, -0.2)),
                     ((0.5, 0.8, -0.15), 0.075 * s, 0.03 * s, (0.1, 0, -0.1)),
                     ((-0.5, 0.8, -0.15), 0.075 * s, 0.03 * s, (-0.1, 0, -0.1))], m["hair"], flat=0.4)
    for ob in (head, *ears, cap, hair, bangs):
        _mesh_outline(ob, 0.006 * s, m["line"])
        fig.attach(ob, "head")
    fig.face_objs = face.objs
    fig.body_parts = [body.body, head, hair, bangs, cap, *ears] + [o for o, _ in parts]
    fig.rig = body.arm
    fig.mats = m
    return fig


def _goku_face(face, s):
    """Anime-Gesicht SSJ-Goku: schmale, scharfe Augen mit türkiser Iris, kräftige, schräge Brauen (Hauthaarfarbe
    mit Kontur), Nasenstrich, Mund. Shape Keys: Clench (Zähne), Shout (Kamehameha-Schrei), Calm (Ende)."""
    k = s
    white = flat_material("GokuEyeWhite", (1.0, 0.98, 0.95), 1.05)
    iris = flat_material("GokuIris", (0.05, 0.62, 0.55), 1.1)
    black = flat_material("GokuInk", (0.015, 0.01, 0.01), 1.0)
    brow = flat_material("GokuBrow", (0.95, 0.72, 0.16), 1.15)
    mouth_in = flat_material("GokuMouthIn", (0.35, 0.05, 0.05), 1.0)
    teeth = flat_material("GokuTeeth", (1.0, 1.0, 0.97), 1.0)
    for x in (1, -1):
        def P(pts):
            return [(x * px * k, pz * k) for px, pz in pts]
        sclera = [(0.013, -0.004), (0.021, 0.006), (0.036, 0.0095), (0.049, 0.007), (0.057, -0.001),
                  (0.048, -0.008), (0.034, -0.0105), (0.021, -0.009)]
        face.decal(f"GokuEyeW{x}", P(sclera), white, lift=0.0008)
        face.decal(f"GokuIris{x}", [(x * px * k, pz * k) for px, pz in ellipse(0.033, -0.0012, 0.0078, 0.0086)], iris,
                   lift=0.0012)
        face.decal(f"GokuPupil{x}", [(x * px * k, pz * k) for px, pz in ellipse(0.033, -0.0012, 0.0033, 0.0042)],
                   black, lift=0.0015)
        face.decal(f"GokuGlint{x}", [(x * px * k, pz * k) for px, pz in ellipse(0.0355, 0.0022, 0.0016, 0.0016, 10)],
                   white, lift=0.0018)
        lid = [(0.011, -0.0025), (0.02, 0.0075), (0.036, 0.011), (0.05, 0.0085), (0.064, 0.0045)]
        lid_calm = [(0.011, -0.001), (0.02, 0.0085), (0.036, 0.0118), (0.05, 0.0095), (0.062, 0.004)]
        face.line(f"GokuLid{x}", P(resample(lid, 14)), 0.0038 * k, black, lift=0.0021,
                  keys={"Calm": P(resample(lid_calm, 14)), "Clench": P(resample(lid, 14)),
                        "Shout": P(resample(lid, 14))})
        face.line(f"GokuLash{x}", P(resample([(0.03, -0.0108), (0.045, -0.0085), (0.056, -0.003)], 8)), 0.0013 * k,
                  black, lift=0.0021)
        br0 = [(0.009, 0.0165), (0.025, 0.021), (0.043, 0.027), (0.062, 0.030)]
        br_angry = [(0.009, 0.0115), (0.025, 0.019), (0.043, 0.027), (0.062, 0.031)]
        br_calm = [(0.009, 0.021), (0.025, 0.024), (0.043, 0.027), (0.062, 0.027)]
        keys = {"Clench": P(resample(br_angry, 12)), "Shout": P(resample(br_angry, 12)), "Calm": P(resample(br_calm, 12))}
        face.line(f"GokuBrowInk{x}", P(resample(br0, 12)), 0.0125 * k, black, lift=0.0018, keys=keys)
        face.line(f"GokuBrow{x}", P(resample(br0, 12)), 0.0085 * k, brow, lift=0.0024, keys=keys)
    face.line("GokuNose", [(0.004 * k, -0.018 * k), (0.008 * k, -0.03 * k), (0.004 * k, -0.036 * k)], 0.0022 * k,
              black, lift=0.0016)
    # Mund: Umriss (Tinte), Innenraum, Zähne – geschlossen = Strich, Clench = Zähne, Shout = weit offen
    def mouth(w, h_top, h_bot, n=24):
        pts = [(-w, 0.0)] + [(-w + 2 * w * t, h_top * math.sin(math.pi * t)) for t in np.linspace(0.1, 0.9, 9)] + \
              [(w, 0.0)] + [(w - 2 * w * t, -h_bot * math.sin(math.pi * t)) for t in np.linspace(0.1, 0.9, 9)]
        return [(px * k, (pz - 0.058) * k) for px, pz in closed_resample(pts, n)]
    closed_ink = mouth(0.015, 0.0006, 0.0010)
    face.decal("GokuMouthInk", closed_ink, black, lift=0.0012,
               keys={"Clench": mouth(0.017, 0.0055, 0.0055), "Shout": mouth(0.018, 0.009, 0.019),
                     "Calm": mouth(0.013, 0.0004, 0.0009)})
    face.decal("GokuMouthIn", mouth(0.014, 0.0001, 0.0001), mouth_in, lift=0.0016,
               keys={"Clench": mouth(0.0155, 0.004, 0.004), "Shout": mouth(0.0165, 0.0075, 0.017),
                     "Calm": mouth(0.012, 0.0001, 0.0001)})
    face.decal("GokuTeeth", mouth(0.013, 0.0001, 0.0001), teeth, lift=0.0019,
               keys={"Clench": mouth(0.0155, 0.004, 0.004), "Shout": mouth(0.0155, 0.0065, -0.004),
                     "Calm": mouth(0.011, 0.0001, 0.0001)})
    # Biss-Linie zwischen den zusammengebissenen Zähnen (nur bei Clench sichtbar)
    face.line("GokuBite", [(-0.0005 * k, -0.058 * k), (0, -0.058 * k), (0.0005 * k, -0.058 * k)], 0.0011 * k, black,
              lift=0.0023, taper=False, keys={"Clench": [(-0.0145 * k, -0.058 * k), (0, -0.058 * k), (0.0145 * k, -0.058 * k)],
                                               "Shout": [(-0.0005 * k, -0.058 * k), (0, -0.058 * k), (0.0005 * k, -0.058 * k)],
                                               "Calm": [(-0.0005 * k, -0.058 * k), (0, -0.058 * k), (0.0005 * k, -0.058 * k)]})


# ------------------------------------------------------------------------------------------------ Freezer
def freezer():
    """Freezer, Endform: 1,50 m, schlank und glatt, weiße Haut mit violetten Partien (Kopfkuppel, Schultern,
    Unterarme, Schienbeine, Brust), langer, dicker, spitz auslaufender Schwanz (eigene Knochenkette), rote
    Augen mit schwerem Lid (arrogant), violette Lippen. Shape Keys: Smirk (kaltes Lächeln), Angry, Shock."""
    fig = Figure("Freezer", 1.50, (0, 0, 0), 0.0)
    s = fig.s
    WHITE, WHITE_SH = (0.96, 0.94, 0.98), (0.56, 0.52, 0.74)
    PURP, PURP_SH = (0.46, 0.10, 0.78), (0.20, 0.03, 0.40)
    m = {"white": toon_material("FreezerWhiteToon", WHITE, WHITE_SH, spec=0.35),
         "purple": toon_material("FreezerGemToon", PURP, PURP_SH, lit=1.15, spec=1.0, rim=(0.9, 0.8, 1.0)),
         "line": outline_material("FreezerLine", (0.03, 0.01, 0.05))}
    # Schwanz: Knochenkette hinten aus dem Becken, bogenförmig nach hinten/unten, Spitze leicht nach oben
    tail_pts = [(0, -0.10, 0.90), (0.02, -0.28, 0.80), (0.05, -0.45, 0.66), (0.06, -0.62, 0.55), (0.04, -0.80, 0.50),
                (0.0, -0.97, 0.52), (-0.04, -1.12, 0.58), (-0.06, -1.24, 0.66)]
    tail_r = [0.075, 0.068, 0.058, 0.048, 0.038, 0.028, 0.018, 0.006]
    TP = [Vector(p) * s for p in tail_pts]
    extra = []
    for i in range(len(TP) - 1):
        extra.append((f"tail{i}", "root" if i == 0 else f"tail{i - 1}", TP[i], TP[i + 1]))
    body = AnimeBody(fig, extra)
    X = lambda sd: 1 if sd == "R" else -1
    nodes = []

    def add(p, r, par, j=None):
        p = Vector(p) * s
        if j:
            p = body.A(j, p)
        nodes.append((p, (r[0] * s, r[1] * s), par))
        return len(nodes) - 1
    pel = add((0, 0, 0.91), (0.125, 0.10), None)
    wai = add((0, 0.01, 1.03), (0.098, 0.082), pel)
    chl = add((0, 0.02, 1.17), (0.128, 0.098), wai)
    chu = add((0, 0.015, 1.29), (0.14, 0.10), chl)
    nk0 = add((0, 0.0, 1.41), (0.056, 0.054), chu)
    add((0, 0.006, 1.50), (0.046, 0.046), nk0)
    for sd in ("R", "L"):
        x = X(sd)
        cl = add((x * 0.10, 0.0, 1.37), (0.062, 0.058), chu)
        de = add((x * 0.185, 0.0, 1.35), (0.058, 0.056), cl, f"shoulder.{sd}")
        bi = add((x * 0.197, 0.004, 1.23), (0.045, 0.045), de, f"shoulder.{sd}")
        el = add((x * 0.21, 0.0, 1.10), (0.036, 0.036), bi, f"shoulder.{sd}")
        fo = add((x * 0.214, 0.006, 1.00), (0.041, 0.039), el, f"elbow.{sd}")
        wr = add((x * 0.22, 0.0, 0.875), (0.028, 0.026), fo, f"elbow.{sd}")
        ha = add((x * 0.222, 0.005, 0.815), (0.034, 0.026), wr, f"wrist.{sd}")
        add((x * 0.222, 0.012, 0.775), (0.028, 0.022), ha, f"wrist.{sd}")
        hp = add((x * 0.085, 0.0, 0.87), (0.078, 0.078), pel, f"hip.{sd}")
        th = add((x * 0.09, 0.008, 0.70), (0.068, 0.066), hp, f"hip.{sd}")
        kn = add((x * 0.093, 0.0, 0.50), (0.044, 0.044), th, f"hip.{sd}")
        ca = add((x * 0.096, -0.012, 0.38), (0.046, 0.044), kn, f"knee.{sd}")
        an = add((x * 0.10, 0.0, 0.10), (0.03, 0.03), ca, f"knee.{sd}")
        ba = add((x * 0.10, 0.07, 0.035), (0.036, 0.024), an, f"ankle.{sd}")
        add((x * 0.10, 0.13, 0.025), (0.022, 0.016), ba, f"ankle.{sd}")
    prev = pel
    for i, (p, r) in enumerate(zip(tail_pts, tail_r)):
        if i == 0:
            prev = add(p, (r, r), pel)
        else:
            prev = add(p, (r, r * 0.92), prev)
    body.build("FreezerBody", nodes, lambda c, db: 0, [m["white"]])
    body.armature()
    add_outline(body.body, 0.009 * s, m["line"])
    sub = body.body.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1
    body.body.modifiers.move(len(body.body.modifiers) - 1, 1)
    # violette Panzerteile (glatt, glänzend) an den Gelenken
    plates = [("GemChest", (0, 0.098, 1.25), (0.075, 0.03, 0.06), "spine")]
    for sd in ("R", "L"):
        x = X(sd)
        plates += [(f"GemShoulder{sd}", (x * 0.19, 0.0, 1.375), (0.062, 0.062, 0.046), f"shoulder.{sd}"),
                   (f"GemForearm{sd}", (x * 0.232, 0.004, 0.98), (0.026, 0.04, 0.075), f"elbow.{sd}"),
                   (f"GemShin{sd}", (x * 0.097, 0.034, 0.30), (0.038, 0.028, 0.10), f"knee.{sd}")]
    for nm, c, r, j in plates:
        ob = fpv.mesh_from_bmesh(_ell_bm(tuple(v * s for v in r)), f"Freezer{nm}", m["purple"])
        ob.location = Vector(c) * s
        _mesh_outline(ob, 0.005 * s, m["line"])
        fig.attach(ob, j)
    # Kopf: großer, runder Schädel, schmales Kinn, violette Kuppel mit Spitze nach vorn
    H = Vector((0, 0.01, 1.58)) * s

    def shape(x, y, z):
        if z < 0:
            k = 1 - 0.30 * min(-z / 0.95, 1) ** 1.2
            x *= k
            y *= 1 - 0.1 * -z
        else:
            y -= 0.06 * z * z                               # Hinterkopf rund
        if y > 0.3 and -0.12 < z < 0.22:
            y -= 0.03
        return x, y, z
    head = head_mesh("FreezerHead", (0.098 * s, 0.108 * s, 0.13 * s), shape, m["white"])
    head.location = H
    dome = head_mesh("FreezerDome", (0.103 * s, 0.113 * s, 0.135 * s),
                     lambda x, y, z: (x, y, z) if z > 0.28 + 0.25 * max(y, 0) - 0.2 * max(-y, 0) - 0.08 * math.exp(-(x / 0.18) ** 2) * (y > 0)
                     else (x * 0.97, y * 0.97, 0.28 + 0.25 * max(y, 0) - 0.2 * max(-y, 0)), m["purple"])
    dome.location = H + Vector((0, -0.004, 0.006)) * s
    face = Face(head, H)
    _freezer_face(face, s)
    face.attach(fig)
    for ob in (head, dome):
        _mesh_outline(ob, 0.005 * s, m["line"])
        fig.attach(ob, "head")
    fig.face_objs = face.objs
    fig.body_parts = [body.body, head, dome]
    fig.rig = body.arm
    fig.tail_bones = [e[0] for e in extra]
    fig.mats = m
    return fig


def _freezer_face(face, s):
    k = s
    white = flat_material("FreezerEyeWhite", (1.0, 0.97, 0.98), 1.05)
    iris = flat_material("FreezerIris", (0.85, 0.04, 0.10), 1.1)
    black = flat_material("FreezerInk", (0.02, 0.005, 0.03), 1.0)
    lips = flat_material("FreezerLips", (0.30, 0.06, 0.42), 1.0)
    shadow = flat_material("FreezerLidShade", (0.62, 0.50, 0.82), 1.0)
    mouth_in = flat_material("FreezerMouthIn", (0.28, 0.03, 0.08), 1.0)
    teeth = flat_material("FreezerTeeth", (1.0, 1.0, 0.98), 1.0)
    for x in (1, -1):
        def P(pts):
            return [(x * px * k, pz * k) for px, pz in pts]
        sclera = [(0.014, -0.002), (0.022, 0.009), (0.036, 0.012), (0.049, 0.009), (0.056, 0.0),
                  (0.049, -0.009), (0.035, -0.012), (0.022, -0.009)]
        face.decal(f"FreezerEyeW{x}", P(sclera), white, lift=0.0008)
        face.decal(f"FreezerIris{x}", P(ellipse(0.034, -0.001, 0.0095, 0.0105)), iris, lift=0.0012)
        face.decal(f"FreezerPupil{x}", P(ellipse(0.034, -0.001, 0.0038, 0.0046)), black, lift=0.0015)
        face.decal(f"FreezerGlint{x}", P(ellipse(0.037, 0.003, 0.0017, 0.0017, 10)), white, lift=0.0018)
        # schweres Oberlid: violette Lidfläche verdeckt das obere Drittel der Iris (arrogant, halb geschlossen)
        lid_top = [(0.012, 0.012), (0.036, 0.016), (0.058, 0.010)]
        lid_low = [(0.058, 0.002), (0.036, 0.0045), (0.012, -0.001)]
        lid_open = [(0.058, 0.007), (0.036, 0.0105), (0.012, 0.004)]
        lid_shock = [(0.058, 0.012), (0.036, 0.016), (0.012, 0.011)]
        face.decal(f"FreezerLid{x}", P(closed_resample(lid_top + lid_low, 16)), shadow, lift=0.0021,
                   keys={"Smirk": P(closed_resample(lid_top + lid_low, 16)),
                         "Angry": P(closed_resample(lid_top + lid_open, 16)),
                         "Shock": P(closed_resample(lid_top + lid_shock, 16))})
        face.line(f"FreezerLidLine{x}", P(resample(lid_low[::-1] + [(0.066, 0.004)], 12)), 0.0030 * k, black,
                  lift=0.0025, keys={"Smirk": P(resample(lid_low[::-1] + [(0.066, 0.004)], 12)),
                                     "Angry": P(resample(lid_open[::-1] + [(0.064, 0.009)], 12)),
                                     "Shock": P(resample(lid_shock[::-1] + [(0.062, 0.014)], 12))})
        face.line(f"FreezerLash{x}", P(resample([(0.03, -0.0118), (0.045, -0.0095), (0.056, -0.002)], 8)),
                  0.0012 * k, black, lift=0.0021)
        # Brauenwulst als violetter Schatten (Freezer hat keine Brauen)
        face.line(f"FreezerRidge{x}", P(resample([(0.012, 0.022), (0.035, 0.027), (0.058, 0.022)], 10)),
                  0.004 * k, shadow, lift=0.0014, keys={"Smirk": P(resample([(0.012, 0.022), (0.035, 0.027), (0.058, 0.022)], 10)),
                                                         "Angry": P(resample([(0.012, 0.015), (0.035, 0.024), (0.058, 0.025)], 10)),
                                                         "Shock": P(resample([(0.012, 0.027), (0.035, 0.032), (0.058, 0.026)], 10))})

    def mouth(w, h_top, h_bot, lift_r=0.0, n=24):
        pts = [(-w, 0.0)] + [(-w + 2 * w * t, h_top * math.sin(math.pi * t)) for t in np.linspace(0.1, 0.9, 9)] + \
              [(w, lift_r)] + [(w - 2 * w * t, -h_bot * math.sin(math.pi * t)) for t in np.linspace(0.1, 0.9, 9)]
        return [(px * k, (pz - 0.062) * k) for px, pz in closed_resample(pts, n)]
    face.decal("FreezerLips", mouth(0.016, 0.0022, 0.0028), lips, lift=0.0012,
               keys={"Smirk": mouth(0.017, 0.0020, 0.0025, 0.0045), "Angry": mouth(0.018, 0.0055, 0.0065),
                     "Shock": mouth(0.013, 0.0075, 0.0105)})
    face.decal("FreezerMouthIn", mouth(0.015, 0.0001, 0.0001), mouth_in, lift=0.0016,
               keys={"Smirk": mouth(0.016, 0.0001, 0.0001, 0.004), "Angry": mouth(0.0165, 0.0035, 0.0045),
                     "Shock": mouth(0.011, 0.006, 0.009)})
    face.decal("FreezerTeeth", mouth(0.014, 0.0001, 0.0001), teeth, lift=0.0019,
               keys={"Smirk": mouth(0.015, 0.0001, 0.0001, 0.004), "Angry": mouth(0.015, 0.003, 0.003),
                     "Shock": mouth(0.010, 0.0045, -0.003)})
    face.line("FreezerMouthLine", [(-0.016 * k, -0.062 * k), (0, -0.0625 * k), (0.016 * k, -0.062 * k)], 0.0011 * k,
              black, lift=0.0022, keys={"Smirk": [(-0.017 * k, -0.062 * k), (0, -0.0622 * k), (0.017 * k, -0.0575 * k)],
                                        "Angry": [(-0.018 * k, -0.062 * k), (0, -0.0625 * k), (0.018 * k, -0.062 * k)],
                                        "Shock": [(-0.013 * k, -0.062 * k), (0, -0.0625 * k), (0.013 * k, -0.062 * k)]})


# ------------------------------------------------------------------------------------------------ Hilfen
def _ell_bm(r, seg=32):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=1.0)
    bmesh.ops.scale(bm, vec=r, verts=bm.verts)
    return bm


def _tube(name, pts, r, mat):
    """Rohr als Mesh (für Kontur per Solidify)."""
    bm = bmesh.new()
    n_a = 10
    P = [Vector(p) for p in pts]
    rings = []
    for i, p in enumerate(P):
        t = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]).normalized()
        a = t.orthogonal().normalized()
        b = t.cross(a)
        rings.append([bm.verts.new(p + (a * math.cos(2 * math.pi * k / n_a) + b * math.sin(2 * math.pi * k / n_a)) * r)
                      for k in range(n_a)])
    for i in range(len(rings) - 1):
        for k in range(n_a):
            k2 = (k + 1) % n_a
            bm.faces.new([rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]])
    return fpv.mesh_from_bmesh(bm, name, mat)


def _mesh_outline(ob, t, mat):
    add_outline(ob, t, mat)


def _fc_eval(ob, path, n, default):
    """Werte einer (gebackenen) F-Kurven-Gruppe path[0..2] für Frames 0..n-1 (ohne Szenenauswertung)."""
    import crew
    out = np.tile(np.array(default, float), (n, 1))
    for fc in crew._fcurves_of(ob):
        if fc.data_path == path:
            out[:, fc.array_index] = [fc.evaluate(f) for f in range(n)]
    return out


def tail_follow(fig, n, fps=24, sway_deg=9.0, seed=3):
    """Sekundärbewegung des Schwanzes: jedes Glied folgt Bewegung und Drehung der Figur verzögert (gedämpft, vom
    Ansatz zur Spitze zunehmend) plus leichtes Pendeln; als Keyframes auf die Schwanzknochen gebacken."""
    import crew
    arm = fig.rig
    t = (np.arange(n) - 1) / fps
    loc = _fc_eval(fig.base, "location", n, (0, 0, 0))
    rot = np.unwrap(_fc_eval(fig.base, "rotation_euler", n, (0, 0, 0)), axis=0)
    root = _fc_eval(fig.J["root"], "rotation_euler", n, (0, 0, 0))
    vel = np.gradient(loc, axis=0) * fps
    yaw = rot[:, 2] + root[:, 2]
    om = np.gradient(yaw) * fps
    vf = vel[:, 0] * -np.sin(yaw) + vel[:, 1] * np.cos(yaw)      # Tempo vorwärts (Figurenraum)
    vs = vel[:, 0] * np.cos(yaw) + vel[:, 1] * np.sin(yaw)       # seitlich
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.28, 2)
    nb = len(fig.tail_bones)
    for i, bn in enumerate(fig.tail_bones):
        w = (i + 1) / nb
        lag = 0.05 + 0.10 * w
        rx = np.clip(vf * 0.8 - vel[:, 2] * 0.9, -30, 30) * w * 0.6 + sway_deg * 0.4 * np.sin(2 * np.pi * 0.7 * t + ph[0] - i * 0.55)
        rz = np.clip(-om * 5.0 + vs * 1.4, -35, 35) * w * 0.6 + sway_deg * np.sin(2 * np.pi * 0.45 * t + ph[1] - i * 0.5)
        R = np.stack([crew._zero_phase(rx, lag, fps), np.zeros(n), crew._zero_phase(rz, lag, fps)], 1)
        crew._bake(arm, f'pose.bones["{bn}"].rotation_euler', np.radians(R))
