"""Ozean: Blender-Ocean-Modifier (FFT-Wellen) nahe der Route + flache Fernfläche."""
import math

import bpy
import numpy as np

import fpv


def water_material(name="Water", deep=(0.004, 0.028, 0.040), shallow=(0.02, 0.09, 0.10), rough=0.035,
                   foam_attr="foam", wake_fn=None, far=False, foam_amount=1.0):
    """Wasser-Shader. Performance: nur 3D-Rauschen mit animiertem Versatz, wenige Oktaven."""
    mat, nb, out = fpv.new_material(name)
    p = fpv.principled(nb, Roughness=rough, IOR=1.333)
    p.inputs["Specular IOR Level"].default_value = 0.5
    geo = nb.node("ShaderNodeNewGeometry")
    wpos = nb.out(geo, "Position")
    tnode = nb.node("ShaderNodeValue")
    tnode.outputs[0].default_value = 0.0
    tnode.name = "TIME"
    tt = tnode.outputs[0]
    # animierter Versatz (Drift) statt teurem 4D-Rauschen
    drift = nb.comb(nb.math("MULTIPLY", tt, 0.8), nb.math("MULTIPLY", tt, -0.5), nb.math("MULTIPLY", tt, 0.35))
    wp_t = nb.vmath("ADD", wpos, drift)
    if far:
        col = nb.mix(0.25, deep, shallow)
    else:
        z = nb.sep(wpos)[2]
        crest = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(z, crest.inputs["Value"])
        crest.inputs["From Min"].default_value = 0.0
        crest.inputs["From Max"].default_value = 1.2
        col = nb.mix(crest.outputs[0], deep, shallow)
    foam = None
    if not far:
        at = nb.node("ShaderNodeAttribute", attribute_name=foam_attr)
        fnoise = nb.noise(wpos, scale=0.9, detail=3, rough=0.6)
        fm = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(nb.math("MULTIPLY", nb.out(at, "Fac"), nb.math("ADD", nb.out(fnoise, "Fac"), 0.2)), fm.inputs["Value"])
        fm.inputs["From Min"].default_value = 0.45
        fm.inputs["From Max"].default_value = 0.95
        foam = nb.math("MULTIPLY", fm.outputs[0], foam_amount)
    if wake_fn is not None:
        wk = wake_fn(nb, wp_t)
        foam = wk if foam is None else nb.math("MAXIMUM", foam, wk)
    if foam is not None:
        churn = nb.math("MINIMUM", nb.math("MULTIPLY", foam, 2.5), 1.0)
        col = nb.mix(churn, col, (0.03, 0.12, 0.12))
        col = nb.mix(foam, col, (0.62, 0.66, 0.67))
        r = nb.mix(foam, rough, 0.55, dtype="FLOAT")
        nb.link(r, p.inputs["Roughness"])
    nb.link(col, p.inputs["Base Color"])
    # feine Kräuselung
    ripple = nb.noise(wp_t, scale=1.1, detail=2, rough=0.5)
    if far:
        ripple2 = nb.noise(nb.vmath("SCALE", wp_t, scale=0.3), scale=0.12, detail=2, rough=0.5)
        h = nb.math("ADD", nb.out(ripple, "Fac"), nb.math("MULTIPLY", nb.out(ripple2, "Fac"), 2.5))
        bump = nb.bump(h, strength=0.35, distance=0.3)
    else:
        bump = nb.bump(nb.out(ripple, "Fac"), strength=0.12, distance=0.3)
    nb.link(bump, p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def animate_time_value(mat, fps, frames):
    nd = mat.node_tree.nodes.get("TIME")
    if nd is None:
        return
    s = nd.outputs[0]
    s.default_value = 0.0
    s.keyframe_insert("default_value", frame=0)
    s.default_value = (frames + 1) / fps
    s.keyframe_insert("default_value", frame=frames + 1)
    for fc in _fcurves(mat.node_tree):
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"


def _fcurves(idblock):
    ad = idblock.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action
    try:
        return list(act.fcurves)
    except AttributeError:
        # Blender 5: geschichtete Actions
        out = []
        for layer in act.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    out.extend(cb.fcurves)
        return out


def make_ocean(x0, y0, nx, ny, tile=100.0, res=18, wind=9.0, wave_scale=1.0, chop=1.25, fps=30, frames=600,
               mat=None, direction_deg=0.0, alignment=0.35, seed=4, foam_coverage=0.2, spectrum="PHILLIPS",
               time_scale=1.0):
    """Nahbereich-Ozean. (x0,y0) = Mitte der ersten Kachel."""
    me = bpy.data.meshes.new("Ocean")
    ob = bpy.data.objects.new("Ocean", me)
    fpv.link(ob)
    ob.location = (x0, y0, 0.0)
    m = ob.modifiers.new("ocean", "OCEAN")
    m.geometry_mode = "GENERATE"
    m.spatial_size = int(tile)
    m.resolution = res
    m.viewport_resolution = min(res, 8)
    m.repeat_x = nx
    m.repeat_y = ny
    m.wind_velocity = wind
    m.wave_scale = wave_scale
    m.choppiness = chop
    m.wave_alignment = alignment
    m.wave_direction = math.radians(direction_deg)
    m.random_seed = seed
    m.spectrum = spectrum
    m.use_normals = True
    m.use_foam = True
    m.foam_layer_name = "foam"
    m.foam_coverage = foam_coverage
    m.depth = 200
    m.time = 0.0
    m.keyframe_insert("time", frame=0)
    m.time = (frames + 1) / fps * time_scale
    m.keyframe_insert("time", frame=frames + 1)
    for fc in _fcurves(ob):
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"
    if mat is not None:
        me.materials.append(mat)
    ob.cycles.use_deform_motion = False
    return ob


def far_plane(xmin, xmax, ymin, ymax, R=30000.0, mat=None, z=-0.02):
    """Große Fläche mit Loch [xmin,xmax]x[ymin,ymax] (dort liegt der Nahbereich)."""
    v = [(-R, -R), (R, -R), (R, R), (-R, R), (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    verts = [(x, y, z) for x, y in v]
    faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me = bpy.data.meshes.new("FarSea")
    me.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new("FarSea", me)
    if mat is not None:
        me.materials.append(mat)
    fpv.link(ob)
    return ob
