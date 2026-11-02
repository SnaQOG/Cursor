# Kleine Requisiten aus CC0-Modellen von Poly Haven, Ergebnis: assets/props.glb
#
#   python3 -I blender/props_build.py --fetch --out <downloads>      (ohne Blender)
#   blender -b --factory-startup -P blender/props_build.py -- --dl <downloads> --work <arbeitsordner> [id ...]
#
# Alle Requisiten teilen sich einen Texturatlas und ein Material. Das Spiel kann sie so in große
# Meshes zusammenfassen und zeichnet eine ganze Kachel mit einem Draw-Call (Batcher.addModelUV).
import argparse
import json
import os
import sys
import urllib.request

# id im Spiel: (Poly-Haven-ID, Zieldreiecke)
PROPS = {
    'd_wine_barrel': ('wine_barrel_01', 1000), 'd_barrels': ('wooden_barrels_01', 1400),
    'd_crate1': ('wooden_crate_01', 400), 'd_crate2': ('wooden_crate_02', 400), 'd_mil_crate': ('old_military_crate', 500),
    'd_bucket1': ('wooden_bucket_01', 450), 'd_bucket2': ('wooden_bucket_02', 450), 'd_basket': ('wicker_basket_01', 800),
    'd_ladder': ('wooden_ladder', 400), 'd_axe': ('wooden_axe', 250), 'd_broom': ('wooden_broom', 450),
    'd_watering': ('watering_can_metal_01', 500), 'd_pot': ('planter_pot_clay', 350), 'd_plant': ('potted_plant_04', 1200),
    'd_stool': ('wooden_stool_01', 500), 'd_bench': ('painted_wooden_bench', 300), 'd_table': ('wooden_picnic_table', 700),
    'd_firepit': ('stone_fire_pit', 800), 'd_lantern': ('wooden_lantern_01', 500), 'd_jug': ('jug_01', 350),
    'd_ceramic': ('ceramic_pot', 450), 'd_spade': ('rusted_spade_01', 350), 'd_lifebuoy': ('lifebuoy', 350),
    'd_buoy': ('ocean_buoy', 800), 'd_cannon': ('cannon_01', 1800), 'd_tyre': ('old_tyre', 450),
    'd_jerrycan': ('metal_jerrycan_green', 450), 'd_propane': ('propane_tank', 400), 'd_trashcan': ('metal_trash_can', 500),
    'd_barrier': ('concrete_road_barrier_02', 400), 'd_generator': ('portable_generator', 1400), 'd_toolcart': ('tool_cart', 1400),
    'd_can': ('can_rusted', 200), 'd_trashbag': ('trashbag', 450), 'd_cement': ('cement_bag', 150),
    'd_compost': ('compost_bags', 700), 'd_rack': ('worn_metal_rack', 800), 'd_rim': ('rusted_wheel_rim_01', 500),
    'd_searchlight': ('portable_searchlight', 900), 'd_chair': ('painted_wooden_chair_01', 300),
}
COLS, ROWS, CELL = 7, 6, 292
UA = {'User-Agent': 'damm-raider-props'}


def fetch(out):
    def get(url):
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            return r.read()
    for aid, (src, _) in PROPS.items():
        info = json.loads(get(f'https://api.polyhaven.com/files/{src}'))['gltf']['1k']['gltf']
        base = os.path.join(out, src)
        os.makedirs(base, exist_ok=True)
        files = {f'{src}.gltf': info['url'], **{k: v['url'] for k, v in info['include'].items()}}
        for rel, url in files.items():
            path = os.path.join(base, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, 'wb') as fh:
                    fh.write(get(url))
        print(aid, src, flush=True)


def blender_main():
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import simready_build as sb

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def build(aid, src, tris, dl, work):
        sb.clear_scene()
        bpy.ops.import_scene.gltf(filepath=os.path.join(dl, src, src + '.gltf'))
        meshes = [o for o in bpy.data.objects if o.type == 'MESH']
        for ob in meshes:
            mw = ob.matrix_world.copy()
            ob.parent = None
            ob.data = ob.data.copy()
            ob.data.transform(mw)
            ob.matrix_world.identity()
        for ob in list(bpy.data.objects):
            if ob not in meshes:
                bpy.data.objects.remove(ob)
        bpy.ops.object.select_all(action='DESELECT')
        for ob in meshes:
            ob.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        if len(meshes) > 1:
            bpy.ops.object.join()
        high = bpy.context.view_layer.objects.active
        sb.setup_cycles()
        co = np.empty(len(high.data.vertices) * 3)
        high.data.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3)
        lo, hi = co.min(0), co.max(0)
        high.data.transform(Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
        high.data.update()
        size = Vector((hi - lo).tolist())
        a = {'id': aid, 'tris': tris, 'tex': CELL, 'sharp': 50, 'area': 0.8, 'ao': 0.5, 'small': False}
        low = sb.make_low(a, high, size)
        sb.unwrap(low)
        sb.bake(a, high, low, size)
        root = bpy.data.objects.new(aid, None)
        bpy.context.scene.collection.objects.link(root)
        low.parent = root
        bpy.data.libraries.write(os.path.join(work, aid + '.blend'), {root, low}, fake_user=False, compress=True)
        print('  fertig', aid, sb.tri_count(high.data), '->', sb.tri_count(low.data), [round(v, 2) for v in size], flush=True)

    def assemble(work, out):
        sb.clear_scene()
        col = bpy.context.scene.collection
        W, H = COLS * CELL, ROWS * CELL
        colour = np.zeros((H, W, 4), np.float32)
        normal = np.zeros((H, W, 4), np.float32)
        normal[..., :] = (0.5, 0.5, 1, 1)
        colour[..., 3] = 1
        bodies = []
        for k, aid in enumerate(PROPS):
            path = os.path.join(work, aid + '.blend')
            if not os.path.exists(path):
                continue
            with bpy.data.libraries.load(path, link=False) as (src, dst):
                dst.objects = src.objects
            for ob in dst.objects:
                col.objects.link(ob)
            body = next(o for o in dst.objects if o.type == 'MESH')
            m = body.data.materials[0]
            bs = sb.principled(m)
            base = bs.inputs['Base Color'].links[0].from_node.image
            nrm = bs.inputs['Normal'].links[0].from_node.inputs['Color'].links[0].from_node.image
            cx, cy = k % COLS, ROWS - 1 - k // COLS
            for img, target in ((base, colour), (nrm, normal)):
                px = np.empty(img.size[0] * img.size[1] * 4, np.float32)
                img.pixels.foreach_get(px)
                target[cy * CELL:(cy + 1) * CELL, cx * CELL:(cx + 1) * CELL] = px.reshape(img.size[1], img.size[0], 4)
            uv = body.data.uv_layers[0].data
            for d in uv:
                d.uv = ((cx + 0.01 + d.uv[0] * 0.98) / COLS, (cy + 0.01 + d.uv[1] * 0.98) / ROWS)
            bodies.append(body)
        at_c = bpy.data.images.new('props_c', W, H, alpha=False)
        at_n = bpy.data.images.new('props_n', W, H, alpha=False)
        at_n.colorspace_settings.name = 'Non-Color'
        for img, arr in ((at_c, colour), (at_n, normal)):
            img.pixels.foreach_set(arr.ravel())
            img.file_format = 'PNG'
            img.update()
            img.pack()
        mat = bpy.data.materials.new('props')
        mat.use_nodes = True
        nt = mat.node_tree
        bs = sb.principled(mat)
        tc = nt.nodes.new('ShaderNodeTexImage')
        tc.image = at_c
        nt.links.new(tc.outputs['Color'], bs.inputs['Base Color'])
        tn = nt.nodes.new('ShaderNodeTexImage')
        tn.image = at_n
        nm = nt.nodes.new('ShaderNodeNormalMap')
        nt.links.new(tn.outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], bs.inputs['Normal'])
        bs.inputs['Roughness'].default_value = 0.82
        bs.inputs['Metallic'].default_value = 0
        for body in bodies:
            body.data.materials[0] = mat
        bpy.ops.export_scene.gltf(
            filepath=out, export_format='GLB', export_extras=False, export_apply=True, export_yup=True,
            export_texcoords=True, export_normals=True, export_tangents=False, export_materials='EXPORT',
            export_image_format='WEBP', export_image_quality=78, export_cameras=False, export_lights=False,
            use_selection=False, export_vertex_color='NONE',
        )
        print(f'geschrieben: {out} ({os.path.getsize(out) / 1e6:.2f} MB), {len(bodies)} Modelle', flush=True)

    argv = sys.argv[sys.argv.index('--') + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--dl', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('ids', nargs='*')
    args = ap.parse_args(argv)
    os.makedirs(args.work, exist_ok=True)
    for aid, (src, tris) in PROPS.items():
        if (not args.ids and not os.path.exists(os.path.join(args.work, aid + '.blend'))) or aid in args.ids:
            print(aid, flush=True)
            build(aid, src, tris, args.dl, args.work)
    assemble(args.work, os.path.join(root_dir, 'assets', 'props.glb'))


if __name__ == '__main__':
    if '--fetch' in sys.argv:
        fetch(sys.argv[sys.argv.index('--out') + 1])
    else:
        blender_main()
