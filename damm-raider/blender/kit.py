# Modellier-Baukasten für Blender (bpy). Alle Assets von Damm Raider werden
# hiermit prozedural gebaut: Teile werden per bmesh zusammengesetzt, gefast
# und mit Materialien aus einer gemeinsamen Palette versehen.
#
# Koordinaten: Blender Z = oben, +Y = vorne (wird im glTF zu -Z = vorne im Spiel).

import math
import bmesh
import bpy
from mathutils import Matrix, Vector, Euler


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(srgb_to_linear(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4))


# name: (hex, klasse, rauheit, metall, emission)
# klasse: matte / metal -> werden beim Export in Vertexfarben gebacken,
#         emit / glass  -> behalten ein eigenes Material
PALETTE = {
    # ARC
    'arc_shell': ('#dcd6c6', 'matte', 0.45, 0.05, 0),
    'arc_shell2': ('#bdb6a5', 'matte', 0.5, 0.05, 0),
    'arc_panel': ('#8f8c83', 'matte', 0.55, 0.1, 0),
    'arc_frame': ('#4c5054', 'metal', 0.45, 0.5, 0),
    'arc_joint': ('#2b2e32', 'metal', 0.4, 0.5, 0),
    'arc_cable': ('#1b1c1e', 'matte', 0.7, 0, 0),
    'arc_orange': ('#d9652a', 'matte', 0.5, 0.05, 0),
    'arc_red': ('#b8412f', 'matte', 0.5, 0.05, 0),
    'arc_eye': ('#ff4a1c', 'emit', 0.3, 0, 6.0),
    'arc_weak': ('#ffab2a', 'emit', 0.3, 0, 4.0),
    'arc_thrust': ('#62d4ff', 'emit', 0.2, 0, 5.0),
    # Waffen
    'gun_steel': ('#393c40', 'metal', 0.35, 0.6, 0),
    'gun_black': ('#1f2123', 'metal', 0.45, 0.4, 0),
    'gun_polymer': ('#32342f', 'matte', 0.7, 0, 0),
    'gun_wood': ('#7c5334', 'matte', 0.6, 0, 0),
    'gun_wood_dark': ('#5a3b25', 'matte', 0.6, 0, 0),
    'gun_brass': ('#bb8e40', 'metal', 0.3, 0.8, 0),
    'tape': ('#8c8a7d', 'matte', 0.8, 0, 0),
    'paint_orange': ('#d3692b', 'matte', 0.55, 0.05, 0),
    'paint_teal': ('#3f8079', 'matte', 0.55, 0.05, 0),
    'paint_cream': ('#d9cfb3', 'matte', 0.55, 0.05, 0),
    'paint_red': ('#a93b2f', 'matte', 0.55, 0.05, 0),
    'paint_yellow': ('#d9a92b', 'matte', 0.55, 0.05, 0),
    'paint_olive': ('#5d6b3b', 'matte', 0.6, 0.05, 0),
    'paint_blue': ('#3b6f9e', 'matte', 0.55, 0.05, 0),
    'paint_purple': ('#71519c', 'matte', 0.55, 0.05, 0),
    'paint_gold': ('#caa040', 'metal', 0.35, 0.6, 0),
    'lens': ('#6fd8ff', 'emit', 0.1, 0, 1.2),
    # Welt
    'concrete': ('#a9a497', 'matte', 0.9, 0, 0),
    'concrete_dark': ('#8a867b', 'matte', 0.9, 0, 0),
    'concrete_light': ('#c3beaf', 'matte', 0.9, 0, 0),
    'rust': ('#8e5634', 'matte', 0.85, 0.2, 0),
    'rust_dark': ('#6b4029', 'matte', 0.85, 0.2, 0),
    'metal_grey': ('#7a7d80', 'metal', 0.5, 0.5, 0),
    'metal_dark': ('#4b4e51', 'metal', 0.55, 0.5, 0),
    'galv': ('#a2a7a8', 'metal', 0.4, 0.7, 0),
    'chrome': ('#cdd0d1', 'metal', 0.2, 0.9, 0),
    'wood': ('#705639', 'matte', 0.85, 0, 0),
    'wood_dark': ('#4e3b28', 'matte', 0.85, 0, 0),
    'wood_light': ('#9a7d56', 'matte', 0.85, 0, 0),
    'sand': ('#a8986c', 'matte', 0.95, 0, 0),
    'canvas': ('#7e7a5a', 'matte', 0.9, 0, 0),
    'rubber': ('#1e1e1e', 'matte', 0.85, 0, 0),
    'glass': ('#9fc6d0', 'glass', 0.05, 0.1, 0),
    'lamp': ('#fff0c0', 'emit', 0.3, 0, 4.0),
    'hazard': ('#e0b020', 'matte', 0.6, 0, 0),
    'black': ('#222222', 'matte', 0.7, 0, 0),
    'white': ('#e6e2d6', 'matte', 0.6, 0, 0),
    'teal': ('#3e6e68', 'matte', 0.7, 0.1, 0),
    'teal_dark': ('#2c4f4b', 'matte', 0.7, 0.1, 0),
    'orange': ('#d2732e', 'matte', 0.7, 0.1, 0),
    'red': ('#a8392e', 'matte', 0.7, 0.1, 0),
    'olive': ('#5f6b45', 'matte', 0.75, 0.05, 0),
    'olive_dark': ('#454f33', 'matte', 0.8, 0.05, 0),
    'khaki': ('#6b6f4a', 'matte', 0.8, 0.05, 0),
    'car_red': ('#8a3b30', 'matte', 0.55, 0.3, 0),
    'car_blue': ('#566b7a', 'matte', 0.55, 0.3, 0),
    'car_sand': ('#9a8a60', 'matte', 0.55, 0.3, 0),
    'car_green': ('#5d6e4d', 'matte', 0.55, 0.3, 0),
    'car_cream': ('#b8ae92', 'matte', 0.55, 0.3, 0),
    # Natur
    'bark': ('#5a4330', 'matte', 0.95, 0, 0),
    'needles': ('#2f4b2c', 'matte', 0.9, 0, 0),
    'needles2': ('#3d5c35', 'matte', 0.9, 0, 0),
    'leaves': ('#56712f', 'matte', 0.85, 0, 0),
    'leaves2': ('#7d8c3c', 'matte', 0.85, 0, 0),
    'rock': ('#7d776a', 'matte', 0.95, 0, 0),
    'rock_dark': ('#5e5950', 'matte', 0.95, 0, 0),
    'moss': ('#5a6a3c', 'matte', 0.95, 0, 0),
    'grass': ('#7b8a44', 'matte', 0.9, 0, 0),
    # Gegenstände
    'paper': ('#e7e1cd', 'matte', 0.8, 0, 0),
    'cloth': ('#ddd7c8', 'matte', 0.9, 0, 0),
    'med_red': ('#c33b31', 'matte', 0.5, 0, 0),
    'cell_green': ('#6aff9a', 'emit', 0.2, 0, 3.0),
    'cell_blue': ('#5ab8ff', 'emit', 0.2, 0, 3.0),
    'gold': ('#dcae45', 'metal', 0.25, 1.0, 0),
    'bronze': ('#9d6b3c', 'metal', 0.4, 0.9, 0),
    'copper': ('#b8703c', 'metal', 0.35, 0.9, 0),
    'silver': ('#c0c2c4', 'metal', 0.25, 1.0, 0),
    'vinyl': ('#151515', 'matte', 0.3, 0, 0),
    'ceramic_blue': ('#3f6f9e', 'matte', 0.25, 0, 0),
    'ceramic_white': ('#ebe5d6', 'matte', 0.25, 0, 0),
    'pcb': ('#2f6b3a', 'matte', 0.5, 0, 0),
    'plastic_yellow': ('#e1b52d', 'matte', 0.45, 0, 0),
    'plastic_blue': ('#3d7fc2', 'matte', 0.45, 0, 0),
    'plastic_grey': ('#8b8e8f', 'matte', 0.5, 0, 0),
    'chem_orange': ('#e0782c', 'matte', 0.4, 0, 0),
    'leather': ('#5b3a24', 'matte', 0.7, 0, 0),
    # Umfärbbare Lackflächen (werden im Spiel durch eine Farbe ersetzt)
    'key': ('#ff00ff', 'matte', 0.6, 0.15, 0),
    'key_dark': ('#800080', 'matte', 0.7, 0.15, 0),
    # Raider
    'skin': ('#c49a78', 'matte', 0.7, 0, 0),
    'jacket': ('#d9a33a', 'matte', 0.8, 0, 0),
    'jacket_dark': ('#a87a28', 'matte', 0.8, 0, 0),
    'pants': ('#3c4b4a', 'matte', 0.85, 0, 0),
    'pack': ('#c8502c', 'matte', 0.8, 0, 0),
    'helmet': ('#e6dfcf', 'matte', 0.45, 0.05, 0),
    'visor': ('#1c2a30', 'metal', 0.15, 0.6, 0),
    'boots': ('#2c2522', 'matte', 0.8, 0, 0),
    'strap': ('#3a3430', 'matte', 0.8, 0, 0),
}

MATERIALS = {}


def get_mat(name):
    if name in MATERIALS:
        return MATERIALS[name]
    hx, cls, rough, metal, emit = PALETTE[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes['Principled BSDF']
    col = hex_rgb(hx)
    bsdf.inputs['Base Color'].default_value = (*col, 1)
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    if emit:
        bsdf.inputs['Emission Color'].default_value = (*col, 1)
        bsdf.inputs['Emission Strength'].default_value = emit
    if cls == 'glass':
        bsdf.inputs['Alpha'].default_value = 0.3
        m.surface_render_method = 'BLENDED'
    MATERIALS[name] = m
    return m


class Part:
    """Sammelt Geometrie für EIN Mesh-Objekt (mehrere Materialien möglich)."""

    def __init__(self, name):
        self.name = name
        self.bm = bmesh.new()
        self.mats = []

    def _mi(self, mat):
        if mat not in self.mats:
            self.mats.append(mat)
        return self.mats.index(mat)

    def _merge(self, tb, loc, rot, scale, mat, bevel, smooth, seg=1, bevel_angle=30):
        if bevel > 0:
            edges = []
            for e in tb.edges:
                if len(e.link_faces) == 2:
                    a = e.link_faces[0].normal.angle(e.link_faces[1].normal, 0)
                    if a > math.radians(bevel_angle):
                        edges.append(e)
            if edges:
                bmesh.ops.bevel(tb, geom=edges, offset=bevel, offset_type='OFFSET', segments=seg,
                                profile=0.5, affect='EDGES', clamp_overlap=True)
        mi = self._mi(mat)
        for f in tb.faces:
            f.material_index = mi
            f.smooth = smooth
        m = Matrix.LocRotScale(Vector(loc), Euler(rot), Vector(scale))
        bmesh.ops.transform(tb, matrix=m, verts=tb.verts)
        self._append(tb)
        return self

    def _append(self, tb):
        vmap = {}
        for v in tb.verts:
            vmap[v] = self.bm.verts.new(v.co)
        for f in tb.faces:
            try:
                nf = self.bm.faces.new([vmap[v] for v in f.verts])
            except ValueError:
                continue
            nf.material_index = f.material_index
            nf.smooth = f.smooth
        tb.free()

    def paint_top(self, mat, threshold=0.7):
        """Nach oben zeigende Flächen umfärben (Moos, Schnee ...)."""
        mi = self._mi(mat)
        self.bm.normal_update()
        for f in self.bm.faces:
            if f.normal.z > threshold:
                f.material_index = mi
        return self

    # --- Grundformen -----------------------------------------------------
    def box(self, size, loc=(0, 0, 0), rot=(0, 0, 0), mat='concrete', bevel=0.0, taper=None):
        tb = bmesh.new()
        bmesh.ops.create_cube(tb, size=1.0)
        sx, sy, sz = size
        for v in tb.verts:
            v.co.x *= sx
            v.co.y *= sy
            v.co.z *= sz
            if taper and v.co.z > 0:
                v.co.x *= taper[0]
                v.co.y *= taper[1]
        return self._merge(tb, loc, rot, (1, 1, 1), mat, bevel, False)

    def cyl(self, r, h, loc=(0, 0, 0), rot=(0, 0, 0), mat='metal_grey', seg=16, r2=None, bevel=0.0, smooth=True, caps=True):
        tb = bmesh.new()
        bmesh.ops.create_cone(tb, cap_ends=caps, cap_tris=False, segments=seg, radius1=r,
                              radius2=r if r2 is None else r2, depth=h)
        return self._merge(tb, loc, rot, (1, 1, 1), mat, bevel, smooth)

    def sphere(self, r, loc=(0, 0, 0), rot=(0, 0, 0), mat='metal_grey', seg=16, rings=10, scale=(1, 1, 1), smooth=True, hemi=False):
        tb = bmesh.new()
        bmesh.ops.create_uvsphere(tb, u_segments=seg, v_segments=rings, radius=r)
        if hemi:
            kill = [v for v in tb.verts if v.co.z < -1e-4]
            bmesh.ops.delete(tb, geom=kill, context='VERTS')
        return self._merge(tb, loc, rot, scale, mat, 0, smooth)

    def ico(self, r, loc=(0, 0, 0), rot=(0, 0, 0), mat='rock', sub=1, scale=(1, 1, 1), smooth=False, jitter=0.0, seed=1):
        tb = bmesh.new()
        bmesh.ops.create_icosphere(tb, subdivisions=sub, radius=r)
        if jitter:
            import random
            rnd = random.Random(seed)
            for v in tb.verts:
                v.co *= 1 + (rnd.random() - 0.5) * jitter
        return self._merge(tb, loc, rot, scale, mat, 0, smooth)

    def torus(self, R, r, loc=(0, 0, 0), rot=(0, 0, 0), mat='metal_grey', seg=24, rseg=8, smooth=True):
        tb = bmesh.new()
        rings = []
        for i in range(seg):
            a = 2 * math.pi * i / seg
            ring = []
            for j in range(rseg):
                b = 2 * math.pi * j / rseg
                x = (R + r * math.cos(b)) * math.cos(a)
                y = (R + r * math.cos(b)) * math.sin(a)
                z = r * math.sin(b)
                ring.append(tb.verts.new((x, y, z)))
            rings.append(ring)
        for i in range(seg):
            for j in range(rseg):
                a, b = rings[i][j], rings[(i + 1) % seg][j]
                c, d = rings[(i + 1) % seg][(j + 1) % rseg], rings[i][(j + 1) % rseg]
                tb.faces.new((a, b, c, d))
        return self._merge(tb, loc, rot, (1, 1, 1), mat, 0, smooth)

    # Extrudiertes 2D-Profil. pts in (u, v); plane 'XZ' extrudiert entlang Y, 'YZ' entlang X
    def prism(self, pts, depth, loc=(0, 0, 0), rot=(0, 0, 0), mat='gun_steel', plane='YZ', bevel=0.0, center=True):
        tb = bmesh.new()
        vs = []
        for (u, v) in pts:
            if plane == 'YZ':
                vs.append(tb.verts.new((-depth / 2 if center else 0, u, v)))
            elif plane == 'XZ':
                vs.append(tb.verts.new((u, -depth / 2 if center else 0, v)))
            else:  # XY
                vs.append(tb.verts.new((u, v, -depth / 2 if center else 0)))
        f = tb.faces.new(vs)
        n = {'YZ': Vector((depth, 0, 0)), 'XZ': Vector((0, depth, 0)), 'XY': Vector((0, 0, depth))}[plane]
        res = bmesh.ops.extrude_face_region(tb, geom=[f])
        verts = [e for e in res['geom'] if isinstance(e, bmesh.types.BMVert)]
        bmesh.ops.translate(tb, verts=verts, vec=n)
        bmesh.ops.recalc_face_normals(tb, faces=tb.faces)
        return self._merge(tb, loc, rot, (1, 1, 1), mat, bevel, False)

    # Einzelne Fläche (z.B. Grashalme; im Spiel doppelseitig gerendert)
    def face(self, pts, mat='grass'):
        tb = bmesh.new()
        tb.faces.new([tb.verts.new(p) for p in pts])
        mi = self._mi(mat)
        for f in tb.faces:
            f.material_index = mi
            f.smooth = False
        self._append(tb)
        return self

    # Rohr/Balken zwischen zwei Punkten
    def beam(self, a, b, r, mat='metal_dark', seg=8, square=False):
        a, b = Vector(a), Vector(b)
        d = b - a
        L = d.length
        q = Vector((0, 0, 1)).rotation_difference(d.normalized())
        mid = (a + b) / 2
        if square:
            tb = bmesh.new()
            bmesh.ops.create_cube(tb, size=1.0)
            for v in tb.verts:
                v.co.x *= r * 2
                v.co.y *= r * 2
                v.co.z *= L
        else:
            tb = bmesh.new()
            bmesh.ops.create_cone(tb, cap_ends=True, cap_tris=False, segments=seg, radius1=r, radius2=r, depth=L)
        m = Matrix.Translation(mid) @ q.to_matrix().to_4x4()
        mi = self._mi(mat)
        for f in tb.faces:
            f.material_index = mi
            f.smooth = not square
        bmesh.ops.transform(tb, matrix=m, verts=tb.verts)
        self._append(tb)
        return self

    def build(self, collection, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), props=None):
        me = bpy.data.meshes.new(self.name)
        bmesh.ops.recalc_face_normals(self.bm, faces=self.bm.faces)
        self.bm.to_mesh(me)
        self.bm.free()
        for m in self.mats:
            me.materials.append(get_mat(m))
        try:
            me.set_sharp_from_angle(angle=math.radians(40))
        except Exception:
            pass
        ob = bpy.data.objects.new(self.name, me)
        collection.objects.link(ob)
        if parent is not None:
            ob.parent = parent
        ob.location = loc
        ob.rotation_euler = Euler(rot)
        for k, v in (props or {}).items():
            ob[k] = v
        return ob


def empty(name, collection, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), props=None):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_size = 0.2
    collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    ob.location = loc
    ob.rotation_euler = Euler(rot)
    for k, v in (props or {}).items():
        ob[k] = v
    return ob


class Asset:
    """Ein exportierbares Asset: Wurzel-Empty mit benannten Kindern."""

    def __init__(self, name, collection):
        self.name = name
        self.col = collection
        self.root = empty(name, collection)

    def node(self, sub, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), props=None):
        return empty(f'{self.name}__{sub}', self.col, parent or self.root, loc, rot, props)

    def part(self, sub):
        return Part(f'{self.name}__{sub}')

    def add(self, part, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), props=None):
        return part.build(self.col, parent or self.root, loc, rot, props)


def new_collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    MATERIALS.clear()


# ---------------------------------------------------------------- Export
VC_MATS = {}


def vc_material(kind):
    if kind in VC_MATS:
        return VC_MATS[kind]
    m = bpy.data.materials.new(f'VC_{kind}')
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    attr = nt.nodes.new('ShaderNodeVertexColor')
    attr.layer_name = 'Col'
    nt.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.72 if kind == 'Matte' else 0.42
    bsdf.inputs['Metallic'].default_value = 0.05 if kind == 'Matte' else 0.55
    VC_MATS[kind] = m
    return m


def bake_vertex_colors(ob):
    """Matte/Metall-Materialien in Vertexfarben backen -> wenige Materialien im Spiel."""
    me = ob.data
    if not me or not me.materials:
        return
    names = [m.name.split('.')[0] for m in me.materials]
    classes = [PALETTE[n][1] if n in PALETTE else 'keep' for n in names]
    if not any(c in ('matte', 'metal') for c in classes):
        return
    attr = me.color_attributes.new('Col', 'FLOAT_COLOR', 'CORNER')
    cols = [(*hex_rgb(PALETTE[n][0]), 1.0) if n in PALETTE else (1, 1, 1, 1) for n in names]
    new_slots = []
    remap = {}
    for i, c in enumerate(classes):
        if c == 'matte':
            key = 'VC_Matte'
        elif c == 'metal':
            key = 'VC_Metal'
        else:
            key = me.materials[i].name
        if key not in new_slots:
            new_slots.append(key)
        remap[i] = new_slots.index(key)
    data = attr.data
    for p in me.polygons:
        col = cols[p.material_index]
        for li in p.loop_indices:
            data[li].color = col
    old = [m for m in me.materials]
    idx = [remap[p.material_index] for p in me.polygons]
    me.materials.clear()
    for key in new_slots:
        if key == 'VC_Matte':
            me.materials.append(vc_material('Matte'))
        elif key == 'VC_Metal':
            me.materials.append(vc_material('Metal'))
        else:
            me.materials.append(next(m for m in old if m.name == key))
    me.polygons.foreach_set('material_index', idx)
    me.update()
    me.color_attributes.active_color = attr


def export_glb(path):
    for ob in bpy.data.objects:
        if ob.type == 'MESH':
            bake_vertex_colors(ob)
    bpy.ops.export_scene.gltf(
        filepath=path, export_format='GLB', export_extras=True, export_apply=True, export_yup=True,
        export_texcoords=False, export_normals=True, export_materials='EXPORT', export_cameras=False,
        export_lights=False, use_selection=False, export_vertex_color='MATERIAL',
    )


# ---------------------------------------------------------------- Drehkörper
def lathe(part, profile, loc=(0, 0, 0), rot=(0, 0, 0), mat='ceramic_blue', seg=24, smooth=True, scale=(1, 1, 1)):
    """profile: Liste (radius, z) von unten nach oben. Dreht um die Z-Achse."""
    tb = bmesh.new()
    rings = []
    for (r, z) in profile:
        ring = []
        for i in range(seg):
            a = 2 * math.pi * i / seg
            ring.append(tb.verts.new((r * math.cos(a), r * math.sin(a), z)))
        rings.append(ring)
    for k in range(len(rings) - 1):
        for i in range(seg):
            a, b = rings[k][i], rings[k][(i + 1) % seg]
            c, d = rings[k + 1][(i + 1) % seg], rings[k + 1][i]
            tb.faces.new((a, b, c, d))
    if profile[0][0] > 1e-4:
        tb.faces.new(list(reversed(rings[0])))
    if profile[-1][0] > 1e-4:
        tb.faces.new(rings[-1])
    bmesh.ops.remove_doubles(tb, verts=tb.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(tb, faces=tb.faces)
    return part._merge(tb, loc, rot, scale, mat, 0, smooth)


# ---------------------------------------------------------------- Rendern
def setup_render(rx, ry, samples=32, transparent=True):
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    s.cycles.device = 'CPU'
    s.cycles.samples = samples
    s.cycles.use_denoising = True
    s.cycles.max_bounces = 4
    s.render.resolution_x = rx
    s.render.resolution_y = ry
    s.render.resolution_percentage = 100
    s.render.film_transparent = transparent
    s.view_settings.view_transform = 'AgX'
    s.view_settings.look = 'AgX - Medium High Contrast'
    if not s.world:
        s.world = bpy.data.worlds.new('World')
    s.world.use_nodes = True
    bg = s.world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.55, 0.6, 0.68, 1)
    bg.inputs['Strength'].default_value = 0.9


def studio_lights(collection):
    def sun(name, rot, strength, color):
        d = bpy.data.lights.new(name, 'SUN')
        d.energy = strength
        d.color = color
        d.angle = math.radians(8)
        o = bpy.data.objects.new(name, d)
        o.rotation_euler = Euler([math.radians(a) for a in rot])
        collection.objects.link(o)
        return o
    sun('Key', (48, 8, -38), 4.0, (1.0, 0.95, 0.88))
    sun('Fill', (65, -10, 140), 1.2, (0.75, 0.85, 1.0))
    sun('Rim', (-60, 0, 20), 2.0, (1.0, 0.9, 0.8))


def mesh_bounds(objs):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    bpy.context.view_layer.update()
    for ob in objs:
        if ob.type != 'MESH':
            continue
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            lo = Vector((min(lo.x, w.x), min(lo.y, w.y), min(lo.z, w.z)))
            hi = Vector((max(hi.x, w.x), max(hi.y, w.y), max(hi.z, w.z)))
    return lo, hi


def camera_look(collection, loc, target, ortho=None, lens=50):
    cam = bpy.data.cameras.new('Cam')
    if ortho:
        cam.type = 'ORTHO'
        cam.ortho_scale = ortho
    else:
        cam.lens = lens
    ob = bpy.data.objects.new('Cam', cam)
    collection.objects.link(ob)
    ob.location = Vector(loc)
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = ob
    cam.clip_end = 2000
    return ob


def render_to(path, fmt='PNG'):
    s = bpy.context.scene
    s.render.image_settings.file_format = fmt
    if fmt == 'WEBP':
        s.render.image_settings.quality = 88
    s.render.image_settings.color_mode = 'RGBA'
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)


def preview(path, objs, direction=(1.0, -1.4, 0.7), res=(640, 400), samples=24):
    """Schnelle Vorschau einer Objektgruppe (für die Kontrolle beim Modellieren)."""
    col = new_collection('__preview')
    studio_lights(col)
    lo, hi = mesh_bounds(objs)
    c = (lo + hi) / 2
    r = (hi - lo).length / 2
    d = Vector(direction).normalized()
    camera_look(col, c + d * r * 2.6, c, lens=50)
    setup_render(res[0], res[1], samples, transparent=False)
    render_to(path)
