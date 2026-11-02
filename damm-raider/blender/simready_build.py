# Wandelt die SimReady-Pakete in spieltaugliche Modelle um und schreibt assets/simready.glb:
# USD einlesen, Polygone reduzieren, neue UVs, alle Materialien auf einen Atlas backen.
#
#   .venv-hf/bin/python -I blender/simready_fetch.py --out <downloads>
#   blender -b --factory-startup -P blender/simready_build.py -- --dl <downloads> --work <arbeitsordner>
#
# Jedes Asset wird einzeln als .blend im Arbeitsordner zwischengespeichert. Mit Asset-IDs wird nur
# das neu gebacken, ohne IDs nur, was noch fehlt; danach wird immer die GLB neu zusammengesetzt.
# Unter <arbeitsordner>/preview/ liegen Vergleichsbilder: Original links, reduziert rechts.
#
# Konventionen wie bei den übrigen Modellen: Z oben, Wurzel = Asset-ID, Teil = <id>__Body.
# Ursprung: Mitte der Grundfläche auf z = 0. rot_z (Grad) dreht das Modell in die Spielrichtung.
import argparse
import glob
import json
import math
import os
import sys
import time

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def lod1_stage(path, work, aid):
    """Scans: Variante mit der reduzierten Stufe wählen und in eine Datei auflösen."""
    from pxr import Usd
    st = Usd.Stage.Open(path)
    for prim in st.Traverse():
        vs = prim.GetVariantSets()
        if 'lod' in vs.GetNames():
            vs.GetVariantSet('lod').SetVariantSelection('1')
    out = os.path.join(work, f'{aid}_lod1.usdc')
    st.Export(out)
    return out


def entry_usd(a, dl):
    base = os.path.join(dl, a['id'], a['folder'], 'simready_usd')
    found = sorted(glob.glob(os.path.join(base, '*.usd')) + glob.glob(os.path.join(base, '*.usda')) + glob.glob(os.path.join(base, '*.usdc')))
    if not found:
        raise RuntimeError(f'keine USD-Datei in {base}')
    return found[0]


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_high(a, dl, work):
    """USD einlesen und alle sichtbaren Meshes zu einem Objekt ohne Transformation verbinden."""
    path = entry_usd(a, dl)
    if a.get('lod1'):
        path = lod1_stage(path, work, a['id'])
    bpy.ops.wm.usd_import(filepath=path, import_cameras=False, import_lights=False, import_guide=False, import_proxy=False,
                          import_render=True, import_all_materials=False, support_scene_instancing=False)
    vl = bpy.context.view_layer
    for ob in list(bpy.data.objects):
        ob.select_set(ob.instance_type != 'NONE')
    if any(ob.instance_type != 'NONE' for ob in bpy.data.objects):
        bpy.ops.object.duplicates_make_real()
    vl.update()
    meshes = []
    for ob in list(bpy.data.objects):
        if ob.type == 'MESH' and not ob.hide_render and not ob.hide_get() and len(ob.data.polygons):
            meshes.append(ob)
    if not meshes:
        raise RuntimeError(f'{a["id"]}: keine Meshes im USD')
    for ob in meshes:
        mw = ob.matrix_world.copy()
        ob.data = ob.data.copy()
        ob.data.transform(mw)
        ob.parent = None
        ob.matrix_world = Matrix.Identity(4)
    for ob in list(bpy.data.objects):
        if ob not in meshes:
            bpy.data.objects.remove(ob)
    bpy.ops.object.select_all(action='DESELECT')
    for ob in meshes:
        ob.select_set(True)
    vl.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    high = vl.objects.active
    high.name = a['id'] + '_high'
    rz = a.get('rot_z', 0)
    if rz:
        high.data.transform(Matrix.Rotation(math.radians(rz), 4, 'Z'))
    co = np.empty(len(high.data.vertices) * 3)
    high.data.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(0), co.max(0)
    high.data.transform(Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
    high.data.update()
    return high, Vector((hi - lo).tolist())


def tri_count(me):
    me.calc_loop_triangles()
    return len(me.loop_triangles)


def bm_area(bm):
    return sum(f.calc_area() for f in bm.faces)


def bm_tris(bm):
    return sum(len(f.verts) - 2 for f in bm.faces)


def drop_small_islands(bm, diag):
    """Schrauben und Aufkleber fliegen aus dem Mesh; ihr Aussehen kommt beim Backen in die Textur."""
    bm.verts.ensure_lookup_table()
    seen = set()
    drop = []
    for start in bm.verts:
        if start.index in seen:
            continue
        stack = [start]
        seen.add(start.index)
        island = []
        while stack:
            v = stack.pop()
            island.append(v)
            for e in v.link_edges:
                o = e.other_vert(v)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
        co = np.array([v.co for v in island])
        dims = co.max(0) - co.min(0)
        size = float(np.linalg.norm(dims))
        if size < diag * 0.012 or (dims.min() < diag * 0.0015 and size < diag * 0.25):
            drop.append(island)
    if drop:
        bmesh.ops.delete(bm, geom=[v for island in drop for v in island], context='VERTS')
    return len(drop)


def make_low(a, high, size):
    """Reduziert das Mesh, ohne die Form zu verlieren.

    Das reine Kollabieren frisst bei CAD-Modellen glatte Flächen auf (Fass wird zum Kegel). Deshalb
    werden zuerst fast ebene Flächen aufgelöst, erst der Rest wird kollabiert. Der Flächeninhalt zeigt,
    ob die Form erhalten blieb; schrumpft er zu stark, gilt das nächstgrößere Ziel.
    """
    low = high.copy()
    low.data = high.data.copy()
    low.name = a['id'] + '__Body'
    bpy.context.scene.collection.objects.link(low)
    me = low.data
    vl = bpy.context.view_layer
    bpy.ops.object.select_all(action='DESELECT')
    low.select_set(True)
    vl.objects.active = low
    if me.has_custom_normals:
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    me.materials.clear()
    for uv in list(me.uv_layers):
        me.uv_layers.remove(uv)
    me.polygons.foreach_set('material_index', [0] * len(me.polygons))
    scan = bool(a.get('lod1'))
    target = a['tris']
    # Geflecht: erst zu einer geschlossenen Hülle machen, das Muster kommt dann aus der Textur
    if a.get('remesh'):
        mod = low.modifiers.new('hull', 'REMESH')
        mod.mode = 'VOXEL'
        mod.voxel_size = a['remesh']
        bpy.ops.object.modifier_apply(modifier=mod.name)
        me = low.data
    bm = bmesh.new()
    bm.from_mesh(me)
    # Nähte an UV- und Materialgrenzen schließen, sonst reißt das Reduzieren die Flächen auf
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=max(size.length * 2e-4, 1e-5))
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    base_area = bm_area(bm)
    if not scan:
        if a.get('small', True):
            drop_small_islands(bm, size.length)
        base_area = bm_area(bm)
        for deg in (0.3, 1, 2, 4, 7, 11, 16):
            if bm_tris(bm) <= target:
                break
            keep = bm.copy()
            bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(deg), use_dissolve_boundaries=False,
                                     verts=bm.verts[:], edges=bm.edges[:], delimit=set())
            if bm_area(bm) < base_area * 0.97:
                bm.free()
                bm = keep
                break
            keep.free()
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
    bm.to_mesh(me)
    bm.free()
    n = tri_count(me)
    if n > target:
        # area im Manifest: Geflecht und kleinteilige Technik dürfen mehr Fläche verlieren
        min_area = base_area * a.get('area', 0.72 if scan else 0.93)
        dg = bpy.context.evaluated_depsgraph_get()
        best = None
        goal = target
        while goal < n:
            mod = low.modifiers.new('reduce', 'DECIMATE')
            mod.decimate_type = 'COLLAPSE'
            mod.ratio = goal / n
            mod.use_collapse_triangulate = True
            dg.update()
            cand = bpy.data.meshes.new_from_object(low.evaluated_get(dg))
            low.modifiers.remove(mod)
            tb = bmesh.new()
            tb.from_mesh(cand)
            ok = bm_area(tb) >= min_area
            tb.free()
            if ok:
                best = cand
                break
            bpy.data.meshes.remove(cand)
            goal = int(goal * 1.5)
        if best:
            low.data = best
            bpy.data.meshes.remove(me)
            me = best
    me.validate(verbose=False)
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(a.get('sharp', 42)))
    me.update()
    return low


def unwrap(low):
    vl = bpy.context.view_layer
    bpy.ops.object.select_all(action='DESELECT')
    low.select_set(True)
    vl.objects.active = low
    low.data.uv_layers.new(name='UVMap')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(58), island_margin=0.004, area_weight=0.6, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(margin=0.005, rotate=True)
    bpy.ops.object.mode_set(mode='OBJECT')


def setup_cycles():
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.refresh_devices()
        for d in prefs.devices:
            d.use = True
        s.cycles.device = 'GPU'
    except (TypeError, KeyError, ValueError) as err:
        print('  keine GPU, backe mit CPU:', err)
        s.cycles.device = 'CPU'
    s.cycles.use_denoising = False
    s.cycles.max_bounces = 2
    if not s.world:
        s.world = bpy.data.worlds.new('World')


def new_image(name, size, data):
    img = bpy.data.images.new(name, size, size, alpha=False, float_buffer=False)
    img.colorspace_settings.name = 'Non-Color' if data else 'sRGB'
    img.generated_color = (0.5, 0.5, 0.5, 1)
    return img


def principled(mat):
    if not mat or not mat.node_tree:
        return None
    return next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)


def surface_output(mat):
    return next((n for n in mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None) \
        or next((n for n in mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL'), None)


def set_emit_source(high, socket):
    """Alle Materialien geben den Wert von socket als Emission aus (auch unter Metall korrekt)."""
    for slot in high.data.materials:
        b = principled(slot)
        if not b:
            continue
        nt = slot.node_tree
        em = nt.nodes.get('__bake_emit') or nt.nodes.new('ShaderNodeEmission')
        em.name = '__bake_emit'
        inp = b.inputs[socket]
        col = em.inputs['Color']
        for l in list(col.links):
            nt.links.remove(l)
        if inp.is_linked:
            nt.links.new(inp.links[0].from_socket, col)
        elif hasattr(inp.default_value, '__len__'):
            col.default_value = tuple(inp.default_value)
        else:
            v = inp.default_value
            col.default_value = (v, v, v, 1)
        nt.links.new(em.outputs['Emission'], surface_output(slot).inputs['Surface'])


def set_emit_white(high):
    """Für die Treffermaske: alles, was der Backstrahl trifft, leuchtet weiß."""
    for slot in high.data.materials:
        if not principled(slot):
            continue
        nt = slot.node_tree
        em = nt.nodes.get('__bake_emit') or nt.nodes.new('ShaderNodeEmission')
        em.name = '__bake_emit'
        col = em.inputs['Color']
        for l in list(col.links):
            nt.links.remove(l)
        col.default_value = (1, 1, 1, 1)
        nt.links.new(em.outputs['Emission'], surface_output(slot).inputs['Surface'])


def restore_surface(high):
    for slot in high.data.materials:
        b = principled(slot)
        if b:
            slot.node_tree.links.new(b.outputs['BSDF'], surface_output(slot).inputs['Surface'])


def fill_misses(arr, valid, steps=40):
    """Pixel, die der Backstrahl verfehlt hat (schwarz), aus den Nachbarn auffüllen."""
    out = arr.copy()
    valid = valid.copy()
    for _ in range(steps):
        if valid.all():
            break
        total = np.zeros_like(out)
        count = np.zeros(valid.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            total += np.roll(out * valid[..., None], (dy, dx), (0, 1))
            count += np.roll(valid, (dy, dx), (0, 1))
        new = ~valid & (count > 0)
        out[new] = total[new] / count[new][..., None]
        valid |= new
    return out


def ensure_material(high):
    """Flächen ohne Material bekämen beim Backen Schwarz: neutrales Grau zuweisen."""
    if not high.data.materials:
        m = bpy.data.materials.new('fallback')
        m.use_nodes = True
        principled(m).inputs['Base Color'].default_value = (0.4, 0.4, 0.4, 1)
        high.data.materials.append(m)


def bake(a, high, low, size):
    s = bpy.context.scene
    tex = a['tex']
    half = max(tex // 2, 256)
    ensure_material(high)
    mat = bpy.data.materials.new(a['id'])
    mat.use_nodes = True
    low.data.materials.append(mat)
    nt = mat.node_tree
    bs = principled(mat)
    node = nt.nodes.new('ShaderNodeTexImage')
    nt.nodes.active = node

    def run(kind, img, samples, **kw):
        node.image = img
        s.cycles.samples = samples
        s.render.bake.use_selected_to_active = True
        s.render.bake.cage_extrusion = size.length * a.get('cage', 0.02)
        s.render.bake.max_ray_distance = size.length * a.get('ray', 0.07)
        s.render.bake.margin = 6
        s.render.bake.margin_type = 'EXTEND'
        s.render.bake.use_clear = True
        s.render.bake.target = 'IMAGE_TEXTURES'
        bpy.ops.object.select_all(action='DESELECT')
        high.select_set(True)
        low.select_set(True)
        bpy.context.view_layer.objects.active = low
        bpy.ops.object.bake(type=kind, **kw)

    def pixels(img):
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        return px.reshape(h, w, 4)

    def put(img, arr):
        img.pixels.foreach_set(arr.ravel())
        img.update()

    # Normalen zuerst, mit den Originalmaterialien (nutzt auch deren Normal-Maps)
    normal = new_image(a['id'] + '_n', tex, True)
    run('NORMAL', normal, 8, normal_space='TANGENT')
    # Verdeckung wird in die Grundfarbe eingerechnet
    s.world.light_settings.distance = size.length * 0.2
    ao = new_image(a['id'] + '_ao', tex, True)
    run('AO', ao, 48)
    set_emit_white(high)
    hit = new_image(a['id'] + '_hit', tex, True)
    run('EMIT', hit, 1)
    set_emit_source(high, 'Base Color')
    base = new_image(a['id'] + '_c', tex, False)
    run('EMIT', base, 12)
    set_emit_source(high, 'Roughness')
    rough = new_image(a['id'] + '_r', half, True)
    run('EMIT', rough, 6)
    set_emit_source(high, 'Metallic')
    metal = new_image(a['id'] + '_m', half, True)
    run('EMIT', metal, 6)
    restore_surface(high)

    # Fehlstellen füllen, dann Verdeckung im gespeicherten sRGB-Wertebereich einrechnen
    valid = pixels(hit)[..., 0] > 0.5
    bpy.data.images.remove(hit)
    step = tex // half
    valid_half = valid[::step, ::step][:half, :half]
    put(normal, fill_misses(pixels(normal), valid))
    pb, pa = fill_misses(pixels(base), valid), fill_misses(pixels(ao), valid)
    k = 1 - a.get('ao', 0.55) * (1 - pa[..., 0])
    pb[..., :3] = pb[..., :3] * k[..., None]
    pb[..., 3] = 1
    put(base, pb)
    bpy.data.images.remove(ao)
    pr, pm = pixels(rough), pixels(metal)
    if valid_half.shape == pr.shape[:2]:
        pr, pm = fill_misses(pr, valid_half), fill_misses(pm, valid_half)
    mr_tex = float(pr[..., 0].std()) > 0.02 or float(pm[..., 0].std()) > 0.03
    bs.inputs['Roughness'].default_value = float(np.median(pr[..., 0]))
    bs.inputs['Metallic'].default_value = float(np.median(pm[..., 0]))
    for n in (node, ):
        nt.nodes.remove(n)
    tb = nt.nodes.new('ShaderNodeTexImage')
    tb.image = base
    nt.links.new(tb.outputs['Color'], bs.inputs['Base Color'])
    tn = nt.nodes.new('ShaderNodeTexImage')
    tn.image = normal
    nm = nt.nodes.new('ShaderNodeNormalMap')
    nt.links.new(tn.outputs['Color'], nm.inputs['Color'])
    nt.links.new(nm.outputs['Normal'], bs.inputs['Normal'])
    images = [base, normal]
    if mr_tex:
        mr = new_image(a['id'] + '_mr', half, True)
        pack = np.ones_like(pr)
        pack[..., 1] = pr[..., 0]
        pack[..., 2] = pm[..., 0]
        put(mr, pack)
        tm = nt.nodes.new('ShaderNodeTexImage')
        tm.image = mr
        sep = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(tm.outputs['Color'], sep.inputs['Color'])
        nt.links.new(sep.outputs['Green'], bs.inputs['Roughness'])
        nt.links.new(sep.outputs['Blue'], bs.inputs['Metallic'])
        images.append(mr)
    bpy.data.images.remove(rough)
    bpy.data.images.remove(metal)
    for img in images:
        img.file_format = 'PNG'
        img.pack()
    return images


def compare_render(path, a, high, low, size):
    """Vergleichsbild: Original links, reduziert rechts."""
    s = bpy.context.scene
    col = s.collection
    made = []
    for name, rot, energy, color in (('Key', (48, 8, -38), 4.0, (1, 0.95, 0.88)), ('Fill', (65, -10, 140), 1.2, (0.75, 0.85, 1)), ('Rim', (-60, 0, 20), 2.0, (1, 0.9, 0.8))):
        d = bpy.data.lights.new(name, 'SUN')
        d.energy = energy
        d.color = color
        d.angle = math.radians(8)
        o = bpy.data.objects.new(name, d)
        o.rotation_euler = [math.radians(v) for v in rot]
        col.objects.link(o)
        made.append(o)
    cam = bpy.data.cameras.new('Cam')
    cam.lens = 60
    co = bpy.data.objects.new('Cam', cam)
    col.objects.link(co)
    made.append(co)
    s.camera = co
    gap = max(size.x, size.y) * 0.65 + size.length * 0.12
    high.location = (-gap, 0, 0)
    low.location = (gap, 0, 0)
    c = Vector((0, 0, size.z * 0.5))
    d = Vector((0.55, -1.0, 0.55)).normalized()
    co.location = c + d * (size.length + 2 * gap) * 1.75
    co.rotation_euler = (c - co.location).to_track_quat('-Z', 'Y').to_euler()
    s.world.use_nodes = True
    bg = next(n for n in s.world.node_tree.nodes if n.type == 'BACKGROUND')
    bg.inputs['Color'].default_value = (0.55, 0.6, 0.68, 1)
    bg.inputs['Strength'].default_value = 0.9
    s.cycles.samples = 48
    s.cycles.use_denoising = False
    s.cycles.max_bounces = 4
    s.render.resolution_x = 900
    s.render.resolution_y = 480
    s.render.film_transparent = False
    s.view_settings.view_transform = 'AgX'
    s.view_settings.look = 'AgX - Medium High Contrast'
    s.render.image_settings.file_format = 'PNG'
    s.render.image_settings.color_mode = 'RGB'
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)
    high.location = (0, 0, 0)
    low.location = (0, 0, 0)
    for o in made:
        bpy.data.objects.remove(o)


def process(a, dl, work):
    t0 = time.time()
    clear_scene()
    setup_cycles()
    high, size = import_high(a, dl, work)
    n_high = tri_count(high.data)
    low = make_low(a, high, size)
    unwrap(low)
    images = bake(a, high, low, size)
    n_low = tri_count(low.data)
    os.makedirs(os.path.join(work, 'preview'), exist_ok=True)
    compare_render(os.path.join(work, 'preview', a['id'] + '.png'), a, high, low, size)
    root = bpy.data.objects.new(a['id'], None)
    bpy.context.scene.collection.objects.link(root)
    low.parent = root
    path = os.path.join(work, a['id'] + '.blend')
    bpy.data.libraries.write(path, {root, low}, fake_user=False, compress=True)
    info = {'id': a['id'], 'high': n_high, 'low': n_low, 'size': [round(v, 3) for v in size], 'images': [tuple(i.size) for i in images], 'seconds': round(time.time() - t0)}
    print('  fertig:', json.dumps(info), flush=True)
    return info


def red_drum(col):
    """Sprengfass: das Stahlfass in Rot mit gelbem Warnband, damit es als gefährlich erkennbar bleibt."""
    src = bpy.data.objects.get('sr_drum_steel__Body')
    if not src:
        return
    me = src.data.copy()
    mat = src.data.materials[0].copy()
    me.materials[0] = mat
    b = principled(mat)
    node = b.inputs['Base Color'].links[0].from_node
    img = node.image.copy()
    img.name = 'sr_drum_red_c'
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    # Band über die UVs: in welchen Pixeln liegt die Höhe des Mantels zwischen 40 und 52 %?
    zs = np.array([v.co.z for v in me.vertices])
    z0, z1 = zs.min(), zs.max()
    band = np.zeros((h, w), bool)
    uv = me.uv_layers[0].data
    me.calc_loop_triangles()
    for t in me.loop_triangles:
        if abs(t.normal.z) > 0.5:
            continue
        pu = np.array([uv[i].uv[0] * w for i in t.loops])
        pv = np.array([uv[i].uv[1] * h for i in t.loops])
        pz = np.array([me.vertices[i].co.z for i in t.vertices])
        x0, x1 = max(int(pu.min()), 0), min(int(pu.max()) + 2, w)
        y0, y1 = max(int(pv.min()), 0), min(int(pv.max()) + 2, h)
        if x0 >= x1 or y0 >= y1:
            continue
        d = (pv[1] - pv[2]) * (pu[0] - pu[2]) + (pu[2] - pu[1]) * (pv[0] - pv[2])
        if abs(d) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        l0 = ((pv[1] - pv[2]) * (gx - pu[2]) + (pu[2] - pu[1]) * (gy - pv[2])) / d
        l1 = ((pv[2] - pv[0]) * (gx - pu[2]) + (pu[0] - pu[2]) * (gy - pv[2])) / d
        l2 = 1 - l0 - l1
        inside = (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)
        rel = (l0 * pz[0] + l1 * pz[1] + l2 * pz[2] - z0) / (z1 - z0)
        band[y0:y1, x0:x1] |= inside & (rel > 0.40) & (rel < 0.52)
    lum = px[..., :3].mean(-1)
    paint = lum < 0.42
    shade = (0.75 + lum * 1.6)[..., None]
    px[..., :3] = np.where(paint[..., None], np.array([0.62, 0.13, 0.09]) * shade, px[..., :3])
    px[..., :3] = np.where((paint & band)[..., None], np.array([0.86, 0.66, 0.14]) * shade, px[..., :3])
    img.pixels.foreach_set(px.ravel())
    img.update()
    img.pack()
    node.image = img
    # Lack statt blankem Metall, sonst blitzt die Sonne als Punkt im Bloom auf
    for name, value in (('Metallic', 0.1), ('Roughness', 0.48)):
        inp = b.inputs[name]
        for link in list(inp.links):
            mat.node_tree.links.remove(link)
        inp.default_value = value
    ob = bpy.data.objects.new('sr_drum_red__Body', me)
    root = bpy.data.objects.new('sr_drum_red', None)
    ob.parent = root
    col.objects.link(root)
    col.objects.link(ob)


def assemble(assets, work, out):
    clear_scene()
    col = bpy.context.scene.collection
    for a in assets:
        path = os.path.join(work, a['id'] + '.blend')
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = src.objects
        for ob in dst.objects:
            col.objects.link(ob)
    red_drum(col)
    bpy.ops.export_scene.gltf(
        filepath=out, export_format='GLB', export_extras=False, export_apply=True, export_yup=True,
        export_texcoords=True, export_normals=True, export_tangents=False, export_materials='EXPORT',
        export_image_format='WEBP', export_image_quality=78, export_cameras=False, export_lights=False,
        use_selection=False, export_vertex_color='NONE',
    )
    print(f'geschrieben: {out} ({os.path.getsize(out) / 1e6:.2f} MB)', flush=True)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--dl', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--only-assemble', action='store_true')
    ap.add_argument('ids', nargs='*')
    args = ap.parse_args(argv)
    os.makedirs(args.work, exist_ok=True)
    with open(os.path.join(HERE, 'simready_assets.json'), encoding='utf-8') as fh:
        assets = json.load(fh)['assets']
    stats_path = os.path.join(args.work, 'stats.json')
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path, encoding='utf-8') as fh:
            stats = json.load(fh)
    if not args.only_assemble:
        for a in assets:
            cached = os.path.exists(os.path.join(args.work, a['id'] + '.blend'))
            if (args.ids and a['id'] not in args.ids) or (not args.ids and cached):
                continue
            print(a['id'], f"({a['repo']} {a['folder']})", flush=True)
            stats[a['id']] = process(a, args.dl, args.work)
            with open(stats_path, 'w', encoding='utf-8') as fh:
                json.dump(stats, fh, indent=1)
    done = [a for a in assets if os.path.exists(os.path.join(args.work, a['id'] + '.blend'))]
    if args.ids and not args.only_assemble and len(done) < len(assets):
        print(f'noch nicht alle gebacken ({len(done)}/{len(assets)}), GLB wird trotzdem geschrieben')
    assemble(done, args.work, os.path.join(ROOT, 'assets', 'simready.glb'))


if __name__ == '__main__':
    main()
