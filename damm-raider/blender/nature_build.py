# Felsen und Pflanzen aus CC0-Modellen von Poly Haven, Ergebnis: assets/nature.glb
#
#   python3 -I blender/nature_fetch.py --out <downloads>
#   blender -b --factory-startup -P blender/nature_build.py -- --dl <downloads> --work <arbeitsordner> [id ...]
#
# Felsen: reduziert und gebacken wie die SimReady-Assets (simready_build.py).
# Bäume und Sträucher haben hunderttausende Dreiecke. Sie werden von drei Seiten gerendert und
# als gekreuzte Bildflächen ausgegeben (6 Dreiecke je Pflanze), das Spiel schneidet sie per Alphatest aus.
# Vergleichsbilder und Einzelansichten unter <arbeitsordner>/preview/.
import argparse
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
import simready_build as sb  # noqa: E402

ROOT = os.path.dirname(HERE)
# id, Quelle, Knoten, Dreiecke
ROCKS = [(f'n_rock{i}', 'rock_moss_set_01', f'rock_moss_set_01_rock0{i + 1}', 800) for i in range(6)] + [
    ('n_rock6', 'namaqualand_boulder_05', 'namaqualand_boulder_05', 900),
    ('n_rock7', 'namaqualand_cliff_01', 'drone_rock_01', 1600),
]
# id, Quelle, Knoten, Zielhöhe in Metern, Zellgröße in Pixeln
PLANTS = [
    ('n_fir0', 'fir_sapling_medium', 'fir_sapling_medium_a_LOD0', 14, 512),
    ('n_fir1', 'fir_sapling_medium', 'fir_sapling_medium_b_LOD0', 12, 512),
    ('n_fir2', 'fir_sapling_medium', 'fir_sapling_medium_c_LOD0', 11, 512),
    ('n_leaf0', 'island_tree_02', 'island_tree_02_LOD0', 10, 640),
    # Poly Haven hat keine mitteleuropäischen Laubbäume: Linde und Birke entstehen hier im Skript (gen_tree)
    ('n_lind0', 'gen', 'linde:7', 12, 640),
    ('n_lind1', 'gen', 'linde:19', 10, 640),
    ('n_birch0', 'gen', 'birke:5', 13, 640),
    ('n_bush0', 'searsia_lucida', 'searsia_lucida_a_LOD0', 2.2, 384),
    ('n_bush1', 'searsia_lucida', 'searsia_lucida_c_LOD0', 1.5, 384),
]
VIEWS = (0, 60, 120)


def load_node(dl, src, node):
    """glTF einlesen und nur den gewünschten Knoten behalten, Transformation angewendet."""
    sb.clear_scene()
    bpy.ops.import_scene.gltf(filepath=os.path.join(dl, src, src + '.gltf'))
    keep = next(o for o in bpy.data.objects if o.type == 'MESH' and (o.name == node or o.data.name == node or o.name.startswith(node)))
    for ob in list(bpy.data.objects):
        if ob is not keep and ob.type == 'MESH':
            bpy.data.objects.remove(ob)
    mw = keep.matrix_world.copy()
    keep.parent = None
    keep.data = keep.data.copy()
    keep.data.transform(mw)
    keep.matrix_world.identity()
    for ob in list(bpy.data.objects):
        if ob is not keep:
            bpy.data.objects.remove(ob)
    co = np.empty(len(keep.data.vertices) * 3)
    keep.data.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    return keep, co.min(0), co.max(0)


def gen_tree(kind, seed):
    """Baum per Space Colonization: Äste wachsen auf Zielpunkte in der Krone zu, das ergibt natürliche Silhouetten."""
    sb.clear_scene()
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    birch = kind == 'birke'
    H = 13.0 if birch else 12.0
    base_z = H * (0.3 if birch else 0.26)
    # Krone: Linde breit und eiförmig, Birke schmal, beide mit unregelmäßigem Rand
    cz, rz = (base_z + H) / 2, (H - base_z) / 2
    rxy = H * (0.2 if birch else 0.36)
    pts = []
    while len(pts) < (1500 if birch else 2600):
        p = rng.uniform(-1, 1, 3)
        if p @ p > 1:
            continue
        # eiförmig: unten breiter als oben
        taper = 1.0 - 0.32 * max(0.0, p[2])
        # Beulen im Umriss, damit die Krone nicht wie eine Kugel aussieht
        k = 0.82 + 0.18 * math.sin(p[0] * 5 + seed) * math.cos(p[1] * 4 - seed) * math.sin(p[2] * 3 + seed)
        pts.append((p[0] * rxy * taper * k, p[1] * rxy * taper * k, cz + p[2] * rz * k))
    A = np.array(pts)
    step = 0.3
    P = [np.array((0.0, 0.0, 0.0))]
    parent = [-1]
    # Stamm bis in die Krone
    lean = rng.uniform(-0.04, 0.04, 2)
    while P[-1][2] < base_z * 0.9:
        P.append(P[-1] + np.array((lean[0] * step, lean[1] * step, step)))
        parent.append(len(P) - 2)
    infl, kill = 3.2, 0.55
    gravity = np.array((0, 0, -0.12 if birch else 0.04))
    for _ in range(90):
        if not len(A):
            break
        Pa = np.array(P)
        d = np.linalg.norm(A[:, None, :] - Pa[None, :, :], axis=2)
        near = d.argmin(1)
        dist = d[np.arange(len(A)), near]
        keep = dist > kill
        A = A[keep]
        near, dist = near[keep], dist[keep]
        grow = {}
        for a, n, dd in zip(A, near, dist):
            if dd < infl:
                v = (a - P[n]) / max(dd, 1e-4)
                grow[n] = grow.get(n, 0) + v
        if not grow:
            break
        for n, v in grow.items():
            v = v / (np.linalg.norm(v) + 1e-6) + gravity + rng.normal(0, 0.1, 3)
            P.append(P[n] + v / np.linalg.norm(v) * step)
            parent.append(n)
    n = len(P)
    # Rohrmodell: Astdicke aus den Kindern
    rad = np.full(n, 0.012)
    child_count = np.zeros(n, int)
    for i in range(n - 1, 0, -1):
        rad[parent[i]] = (rad[parent[i]] ** 2.3 + rad[i] ** 2.3) ** (1 / 2.3)
        child_count[parent[i]] += 1
    rad *= (0.15 if birch else 0.25) / rad[0]
    rad = np.maximum(rad, 0.012)
    verts, faces, cols = [], [], []
    crown = Vector((0, 0, cz))
    for i in range(1, n):
        a, b = Vector(P[parent[i]]), Vector(P[i])
        if (b - a).length < 1e-4:
            continue
        axis = (b - a).normalized()
        u = axis.orthogonal().normalized()
        v = axis.cross(u)
        r0, r1 = float(rad[parent[i]]), float(rad[i])
        if birch:
            dark = rnd.random() < 0.14 or a.z < 1.0 or r0 < 0.03
            col = (0.05, 0.045, 0.04) if dark else (0.6, 0.58, 0.53)
        else:
            col = (0.055, 0.042, 0.032)
        seg = 6 if r0 > 0.05 else 4
        i0 = len(verts)
        for k in range(seg):
            ang = k * math.tau / seg
            d = u * math.cos(ang) + v * math.sin(ang)
            verts.append(tuple(a + d * r0))
            verts.append(tuple(b + d * r1))
            cols.extend([col, col])
        for k in range(seg):
            p, q = i0 + k * 2, i0 + ((k + 1) % seg) * 2
            faces.append((p, q, q + 1, p + 1))
    # Blätter an den dünnen Zweigen, in vielen kleinen Büscheln
    for i in range(1, n):
        if rad[i] > (0.04 if birch else 0.055):
            continue
        c = Vector(P[i])
        base = (rnd.uniform(0.13, 0.2), rnd.uniform(0.21, 0.27), rnd.uniform(0.045, 0.07)) if birch else (rnd.uniform(0.07, 0.12), rnd.uniform(0.15, 0.2), rnd.uniform(0.03, 0.05))
        for _ in range(16 if birch else 22):
            p = c + Vector((rnd.gauss(0, 1), rnd.gauss(0, 1), rnd.gauss(0, 0.8))).normalized() * rnd.random() ** 0.7 * (0.5 if birch else 0.62)
            if birch:
                p.z -= rnd.random() * 0.35
            shade = min(1.45, 0.42 + 0.95 * (Vector((p.x / rxy, p.y / rxy, (p.z - cz) / rz)).length)) * (0.9 + 0.25 * (p.z - base_z) / (H - base_z))
            size = rnd.uniform(0.05, 0.085) if birch else rnd.uniform(0.075, 0.12)
            a = Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-0.4, 0.4))).normalized() * size
            b = a.orthogonal().normalized() * size * 0.7
            i0 = len(verts)
            verts.extend([tuple(p - a), tuple(p - b), tuple(p + a), tuple(p + b)])
            c0 = tuple(x * shade * rnd.uniform(0.82, 1.18) for x in base)
            cols.extend([c0, c0, c0, c0])
            faces.append((i0, i0 + 1, i0 + 2, i0 + 3))
    me = bpy.data.meshes.new(kind)
    me.from_pydata(verts, [], faces)
    attr = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set('color', [v for c in cols for v in (c[0], c[1], c[2], 1)])
    mat = bpy.data.materials.new(kind)
    mat.use_nodes = True
    nt = mat.node_tree
    bs = sb.principled(mat)
    vc = nt.nodes.new('ShaderNodeVertexColor')
    vc.layer_name = 'Col'
    nt.links.new(vc.outputs['Color'], bs.inputs['Base Color'])
    bs.inputs['Roughness'].default_value = 0.7
    me.materials.append(mat)
    ob = bpy.data.objects.new(kind, me)
    bpy.context.scene.collection.objects.link(ob)
    co = np.array(verts)
    print('  Baum', kind, n, 'Knoten', len(faces), 'Flächen', flush=True)
    return ob, co.min(0), co.max(0)


def build_rock(aid, src, node, tris, dl, work):
    high, lo, hi = load_node(dl, src, node)
    sb.setup_cycles()
    size = hi - lo
    # einheitliche Größe, Grundfläche auf z = 0
    k = 2.2 / float(size.max())
    me = high.data
    me.transform(Matrix.Scale(k, 4) @ Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
    me.update()
    size = Vector((size * k).tolist())
    a = {'id': aid, 'tris': tris, 'tex': 512, 'sharp': 65, 'lod1': True, 'ao': 0.4}
    low = sb.make_low(a, high, size)
    sb.unwrap(low)
    sb.bake(a, high, low, size)
    os.makedirs(os.path.join(work, 'preview'), exist_ok=True)
    sb.compare_render(os.path.join(work, 'preview', aid + '.png'), a, high, low, size)
    root = bpy.data.objects.new(aid, None)
    bpy.context.scene.collection.objects.link(root)
    low.parent = root
    bpy.data.libraries.write(os.path.join(work, aid + '.blend'), {root, low}, fake_user=False, compress=True)
    print('  fertig', aid, sb.tri_count(high.data), '->', sb.tri_count(low.data), flush=True)


def build_plant(aid, src, node, height, cell, dl, work):
    if src == 'gen':
        kind, seed = node.split(':')
        ob, lo, hi = gen_tree(kind, int(seed))
    else:
        ob, lo, hi = load_node(dl, src, node)
    s = bpy.context.scene
    sb.setup_cycles()
    s.cycles.samples = 96
    s.cycles.max_bounces = 4
    s.cycles.transparent_max_bounces = 16
    s.render.film_transparent = True
    s.render.resolution_x = s.render.resolution_y = cell
    s.view_settings.view_transform = 'Standard'
    s.view_settings.look = 'None'
    s.render.image_settings.file_format = 'PNG'
    s.render.image_settings.color_mode = 'RGBA'
    s.world.use_nodes = True
    bg = next(n for n in s.world.node_tree.nodes if n.type == 'BACKGROUND')
    bg.inputs['Color'].default_value = (1, 1, 1, 1)
    bg.inputs['Strength'].default_value = 0.85
    sun = bpy.data.lights.new('Sun', 'SUN')
    sun.energy = 1.1
    so = bpy.data.objects.new('Sun', sun)
    so.rotation_euler = (math.radians(35), 0, math.radians(25))
    s.collection.objects.link(so)
    # Ausgangslage: Ursprung Mitte Grundfläche
    cx, cy, cz = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2
    extent = float(max(hi[2] - lo[2], math.hypot(hi[0] - lo[0], hi[1] - lo[1]))) * 1.04
    cam = bpy.data.cameras.new('Cam')
    cam.type = 'ORTHO'
    cam.ortho_scale = extent
    co = bpy.data.objects.new('Cam', cam)
    s.collection.objects.link(co)
    s.camera = co
    n = len(VIEWS)
    atlas = np.zeros((cell, cell * n, 4), np.float32)
    os.makedirs(os.path.join(work, 'preview'), exist_ok=True)
    for i, deg in enumerate(VIEWS):
        a = math.radians(deg)
        co.location = (cx + math.sin(a) * extent * 2, cy - math.cos(a) * extent * 2, cz)
        co.rotation_euler = (Vector((cx, cy, cz)) - Vector(co.location)).to_track_quat('-Z', 'Y').to_euler()
        path = os.path.join(work, 'preview', f'{aid}_{deg}.png')
        s.render.filepath = path
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(path)
        px = np.empty(cell * cell * 4, np.float32)
        img.pixels.foreach_get(px)
        px = px.reshape(cell, cell, 4)
        atlas[:, i * cell:(i + 1) * cell] = px
        bpy.data.images.remove(img)
    # Farbe über den Rand ziehen, sonst entstehen helle Säume beim Ausschneiden
    atlas[..., :3] = sb.fill_misses(atlas[..., :3], atlas[..., 3] > 0.3, steps=16)
    sb.clear_scene()
    tex = bpy.data.images.new(aid + '_c', cell * n, cell, alpha=True)
    tex.pixels.foreach_set(atlas.ravel())
    tex.file_format = 'PNG'
    tex.update()
    tex.pack()
    k = height / float(hi[2] - lo[2])
    half = extent * k / 2
    base = (cz - extent / 2 - lo[2]) * k  # Unterkante der Bildfläche relativ zum Boden
    verts, faces, uvs, normals = [], [], [], []
    for i, deg in enumerate(VIEWS):
        a = math.radians(deg)
        rx, ry = math.cos(a), math.sin(a)
        i0 = len(verts)
        for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
            x, y, z = (u * 2 - 1) * half * rx, (u * 2 - 1) * half * ry, base + v * 2 * half
            verts.append((x, y, z))
            uvs.append(((i + u) / n, v))
            # weiche Normalen wie an einer Kugel, damit die Krone Volumen zeigt
            nrm = Vector((x, y, (z - height * 0.45) + height * 0.5)).normalized()
            normals.append(tuple(nrm))
        faces.append((i0, i0 + 1, i0 + 2, i0 + 3))
    me = bpy.data.meshes.new(aid)
    me.from_pydata(verts, [], faces)
    uvl = me.uv_layers.new(name='UVMap')
    for p in me.polygons:
        for li in p.loop_indices:
            uvl.data[li].uv = uvs[me.loops[li].vertex_index]
    me.shade_smooth()
    me.normals_split_custom_set_from_vertices(normals)
    mat = bpy.data.materials.new(aid)
    mat.use_nodes = True
    nt = mat.node_tree
    bs = sb.principled(mat)
    tn = nt.nodes.new('ShaderNodeTexImage')
    tn.image = tex
    nt.links.new(tn.outputs['Color'], bs.inputs['Base Color'])
    nt.links.new(tn.outputs['Alpha'], bs.inputs['Alpha'])
    bs.inputs['Roughness'].default_value = 0.92
    bs.inputs['Metallic'].default_value = 0
    me.materials.append(mat)
    body = bpy.data.objects.new(aid + '__Body', me)
    root = bpy.data.objects.new(aid, None)
    body.parent = root
    bpy.context.scene.collection.objects.link(root)
    bpy.context.scene.collection.objects.link(body)
    bpy.data.libraries.write(os.path.join(work, aid + '.blend'), {root, body}, fake_user=False, compress=True)
    print('  fertig', aid, 'Fläche', round(half * 2, 1), 'm', flush=True)


def assemble(work, out):
    sb.clear_scene()
    col = bpy.context.scene.collection
    for aid in [r[0] for r in ROCKS] + [p[0] for p in PLANTS]:
        path = os.path.join(work, aid + '.blend')
        if not os.path.exists(path):
            continue
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = src.objects
        for ob in dst.objects:
            col.objects.link(ob)
    bpy.ops.export_scene.gltf(
        filepath=out, export_format='GLB', export_extras=False, export_apply=True, export_yup=True,
        export_texcoords=True, export_normals=True, export_tangents=False, export_materials='EXPORT',
        export_image_format='WEBP', export_image_quality=80, export_cameras=False, export_lights=False,
        use_selection=False, export_vertex_color='NONE',
    )
    print(f'geschrieben: {out} ({os.path.getsize(out) / 1e6:.2f} MB)', flush=True)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--dl', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('ids', nargs='*')
    args = ap.parse_args(argv)
    os.makedirs(args.work, exist_ok=True)
    for aid, src, node, tris in ROCKS:
        if (not args.ids and not os.path.exists(os.path.join(args.work, aid + '.blend'))) or aid in args.ids:
            print(aid, flush=True)
            build_rock(aid, src, node, tris, args.dl, args.work)
    for aid, src, node, height, cell in PLANTS:
        if (not args.ids and not os.path.exists(os.path.join(args.work, aid + '.blend'))) or aid in args.ids:
            print(aid, flush=True)
            build_plant(aid, src, node, height, cell, args.dl, args.work)
    assemble(args.work, os.path.join(ROOT, 'assets', 'nature.glb'))


if __name__ == '__main__':
    main()
