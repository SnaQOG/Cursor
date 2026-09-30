"""Dragon-Ball-Bewegungseffekte an gebackenen Figuren (nach choreo.bake_fighter):

- Ki-Spur: leuchtender, spitz auslaufender Schweif hinter schnellen Vorstößen/Rückstößen. Eine Kurve entlang der
  gebackenen Bahn; sichtbar ist per Bevel-Faktor nur das Stück der letzten `tail_s` Sekunden (Taper auf den
  sichtbaren Teil gemappt: hinten dünn, am Kämpfer dick). Bleibt der Kämpfer stehen, zieht sich die Spur ein.
- Zanzoken: die Figur verschwindet für ein paar Frames (alle Render-Objekte ausgeblendet, auch Aura/Licht) und
  taucht woanders wieder auf; an der alten Stelle bleibt ein flackerndes Nachbild (eingefrorene Kopie des
  verformten Körpers mit eigenen, ausblendbaren Materialien). Beim Wiederauftauchen ein kleiner Luftring.
"""
import math

import bpy
import numpy as np
from mathutils import Euler, Vector

import fpv
import vfx


def F(t, fps=24):
    return int(round(t * fps)) + 1


# ------------------------------------------------------------------------------------------------ Bahn
def _fc_map(ob):
    import crew
    return {(fc.data_path, fc.array_index): fc for fc in crew._fcurves_of(ob)}


def base_track(fig, frames, offset=(0.0, 0.0, 0.0)):
    """Weltposition eines Punkts im Figurenraum (z. B. Körpermitte) zu (auch gebrochenen) Frames, aus den
    gebackenen F-Kurven der Figur-Basis (ohne Szenenauswertung)."""
    fc = _fc_map(fig.base)
    out = []
    off = Vector(offset)
    for f in frames:
        loc = [fc[("location", k)].evaluate(f) if ("location", k) in fc else fig.base.location[k] for k in range(3)]
        rot = [fc[("rotation_euler", k)].evaluate(f) if ("rotation_euler", k) in fc else fig.base.rotation_euler[k]
               for k in range(3)]
        out.append(Vector(loc) + Euler(rot, "XYZ").to_matrix() @ off)
    return np.array([tuple(p) for p in out])


# ------------------------------------------------------------------------------------------------ Ki-Spur
def _trail_taper(name):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "2D"
    sp = cu.splines.new("POLY")
    prof = [(0.0, 0.0), (0.25, 0.18), (0.6, 0.5), (0.85, 0.85), (1.0, 1.0)]
    sp.points.add(len(prof) - 1)
    for p, (x, y) in zip(sp.points, prof):
        p.co = (x, y, 0.0, 1.0)
    ob = bpy.data.objects.new(name, cu)
    fpv.link(ob)
    ob.hide_render = True
    ob.hide_viewport = True
    return ob


def ki_trail(name, fig, t0, t1, color, radius=0.2, tail_s=0.16, fps=24, center_z=0.95, strength=6.0, core=True):
    """Leuchtschweif hinter der Figur für ihre Bewegung zwischen t0 und t1 (Sekunden, Echtzeit der Szene)."""
    sub = 4
    f0, f1 = (t0 * fps + 1), (t1 * fps + 1)
    fr = np.arange(f0, f1 + 1e-6, 1.0 / sub)
    P = base_track(fig, fr, (0.0, 0.0, center_z * fig.s))
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 0.5:
        return None
    u = s / s[-1]
    objs = []
    taper = _trail_taper(name + "Taper")
    layers = [(name + "Glow", radius, vfx.glow_material(name + "GlowMat", color, strength, falloff=1.3, alpha=0.9))]
    if core:
        layers.append((name + "Core", radius * 0.32,
                       vfx.glow_material(name + "CoreMat", tuple(0.35 + 0.65 * c for c in color), strength * 1.4,
                                         kind="core")))
    for nm, r, mat in layers:
        cu = bpy.data.curves.new(nm, "CURVE")
        cu.dimensions = "3D"
        sp = cu.splines.new("POLY")
        sp.points.add(len(P) - 1)
        for p, c in zip(sp.points, P):
            p.co = (*c, 1.0)
        cu.bevel_depth = r
        cu.bevel_resolution = 3
        cu.use_fill_caps = True
        cu.taper_object = taper
        cu.use_map_taper = True
        cu.bevel_factor_mapping_start = "SPLINE"
        cu.bevel_factor_mapping_end = "SPLINE"
        ob = bpy.data.objects.new(nm, cu)
        ob.data.materials.append(mat)
        ob.visible_shadow = False
        ob.cycles.use_motion_blur = False
        fpv.link(ob)
        objs.append(ob)
    # sichtbares Fenster je Frame: [t - tail_s, t] geschnitten mit [t0, t1]
    for f in range(int(math.floor(f0)) - 1, int(math.ceil(f1 + tail_s * fps)) + 2):
        tt = (f - 1) / fps
        a, b = max(tt - tail_s, t0), min(tt, t1)
        vis = b - a > 0.5 / fps
        ua = float(np.interp(a, (fr - 1) / fps, u)) if vis else 0.0
        ub = float(np.interp(b, (fr - 1) / fps, u)) if vis else 0.0
        for ob in objs:
            ob.data.bevel_factor_start = min(ua, 0.999)
            ob.data.bevel_factor_end = max(ub, ua + 1e-3) if vis else 1.0
            ob.data.keyframe_insert("bevel_factor_start", frame=f)
            ob.data.keyframe_insert("bevel_factor_end", frame=f)
            ob.hide_render = not vis
            ob.keyframe_insert("hide_render", frame=f)
    for ob in objs:
        vfx._interp(ob.data.animation_data, "LINEAR")
        _constant(ob, "hide_render")
    return objs


# ------------------------------------------------------------------------------------------------ Zanzoken
def render_objects(root):
    """Alle renderbaren Nachfahren (Mesh, Kurve, Licht) eines Objekts."""
    out, stack = [], list(root.children)
    while stack:
        o = stack.pop()
        stack.extend(o.children)
        if o.type in ("MESH", "CURVE", "LIGHT"):
            out.append(o)
    return out


def _constant(ob, path):
    import crew
    for fc in crew._fcurves_of(ob):
        if fc.data_path == path:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


def key_hidden(ob, frames_values):
    """hide_render je Frame setzen: [(Frame, versteckt?), ...] (stufig)."""
    for f, h in frames_values:
        ob.hide_render = h
        ob.keyframe_insert("hide_render", frame=f)
    _constant(ob, "hide_render")


def vanish(objs, spans):
    """Figur verschwinden lassen: spans = [(f_weg, f_wieder_da), ...] (f_weg einschließlich, f_wieder_da sichtbar)."""
    for ob in objs:
        fv = [(1, False)]
        for fh, fs in spans:
            fv += [(fh, True), (fs, False)]
        key_hidden(ob, fv)


def _ghost_material(mat, cache):
    """Kopie des Materials mit vorgeschaltetem Mix gegen Transparent (Wert 'Fade': 1 = sichtbar, 0 = weg)."""
    if mat.name in cache:
        return cache[mat.name]
    g = mat.copy()
    g.name = mat.name + "Ghost"
    nt = g.node_tree
    outn = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    lk = outn.inputs["Surface"].links
    src = lk[0].from_socket if lk else None
    fade = nt.nodes.new("ShaderNodeValue")
    fade.name = fade.label = "Fade"
    fade.outputs[0].default_value = 1.0
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fade.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    if src is not None:
        nt.links.new(src, mix.inputs[2])
    nt.links.new(mix.outputs[0], outn.inputs["Surface"])
    cache[mat.name] = g
    return g


def afterimage(name, objs, f_snap, f_show, flicker, cache=None):
    """Nachbild: eingefrorene Kopien der (verformten) Objekte im Zustand von Frame f_snap, sichtbar ab f_show mit
    Flackern flicker = [Deckkraft je Frame ab f_show, ...] (danach weg)."""
    cache = {} if cache is None else cache
    sc = bpy.context.scene
    sc.frame_set(f_snap)
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for ob in objs:
        if ob.type != "MESH" or ob.hide_render:
            continue
        ev = ob.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)   # UVs für die Textur
        if len(me.vertices) == 0:
            bpy.data.meshes.remove(me)
            continue
        me.name = f"{name}_{ob.name}"
        for i, m in enumerate(me.materials):
            if m is not None:
                me.materials[i] = _ghost_material(m, cache)
        g = bpy.data.objects.new(me.name, me)
        g.matrix_world = ev.matrix_world.copy()
        g.visible_shadow = False
        g.cycles.use_motion_blur = False
        fpv.link(g)
        out.append(g)
    f_end = f_show + len(flicker)
    for g in out:
        key_hidden(g, [(0, True), (f_show, False), (f_end, True)])
    mats = {m.name: m for m in cache.values()}
    for m in mats.values():
        nd = m.node_tree.nodes["Fade"]
        for k, a in enumerate(flicker):
            nd.outputs[0].default_value = a
            nd.outputs[0].keyframe_insert("default_value", frame=f_show + k)
        vfx._interp(m.node_tree.animation_data, "CONSTANT")
    return out


def pop_ring(name, loc, f0, axis=(0, 0, 1), r_max=1.8, color=(1.0, 0.95, 0.8)):
    """Kleiner Luftring beim Auftauchen/Verschwinden (kurz, dünn)."""
    ob = vfx.shockwave(name, loc, f0, r_max=r_max, dur=6, color=color, thick=0.05, glow=1.2, ior=1.0)
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = Vector(axis).normalized().to_track_quat("Z", "Y")
    return ob


FLICKER = [0.95, 0.35, 0.85, 0.2, 0.7, 0.12, 0.5, 0.06, 0.3, 0.0]
