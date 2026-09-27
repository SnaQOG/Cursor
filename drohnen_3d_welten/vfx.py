"""Energie-Effekte für die Kampfszenen (Cycles): Energiekugeln (Rasengan, Kamehameha-Ladung),
Blitze (Chidori), Strahlen (Kamehameha, Todesstrahl), Licht-Explosionen mit Druckwellenring,
Ki-Salven, Super-Saiyajin-Aura. Alle Effekte sind emissiv, über Keyframes zeitlich gesteuert
(Skalierung/Fade-Werte/Sichtbarkeit) und werfen über Punktlichter Licht auf die Umgebung.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector

import fpv


def _fade_node(nb, name="Fade", value=1.0):
    v = nb.node("ShaderNodeValue")
    v.name = name
    v.label = name
    v.outputs[0].default_value = value
    return v


def key_fade(mat, frames_values, name="Fade", interp="LINEAR"):
    """Fade-Wert eines Effektmaterials über Frames setzen: [(frame, value), ...]."""
    nd = mat.node_tree.nodes[name]
    for f, v in frames_values:
        nd.outputs[0].default_value = v
        nd.outputs[0].keyframe_insert("default_value", frame=f)
    _interp(mat.node_tree.animation_data, interp)


def _interp(ad, interp):
    if ad and ad.action:
        for fc in _fcurves(ad.action):
            for kp in fc.keyframe_points:
                kp.interpolation = interp


def _fcurves(action):
    if hasattr(action, "fcurves") and action.fcurves:
        return list(action.fcurves)
    out = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for cb in strip.channelbags:
                out += list(cb.fcurves)
    return out


def glow_material(name, color, strength=10.0, falloff=1.6, kind="glow", swirl=0.0, alpha=0.75):
    """kind='core': deckend emissiv; 'glow': transparente Hülle, zur Mitte hin hell (Layer Weight Facing);
    swirl > 0: animierte Wirbel-/Rauschstruktur (Wert 'Time' keyframen)."""
    mat, nb, out = fpv.new_material(name)
    fade = _fade_node(nb)
    em = nb.node("ShaderNodeEmission")
    col = color
    strength_s = nb.math("MULTIPLY", fade.outputs[0], strength)
    if swirl:
        t = _fade_node(nb, "Time", 0.0)
        co = nb.coords("Object")
        nz = nb.node("ShaderNodeTexNoise")
        nz.noise_dimensions = "4D"
        nb.link(co, nz.inputs["Vector"])
        nb.link(t.outputs[0], nz.inputs["W"])
        nz.inputs["Scale"].default_value = swirl
        nz.inputs["Detail"].default_value = 4.0
        nz.inputs["Distortion"].default_value = 2.5
        wv = nb.node("ShaderNodeTexWave", wave_type="RINGS", rings_direction="Z", wave_profile="SIN")
        nb.link(nb.vmath("ADD", co, nb.vmath("SCALE", nb.out(nz, "Color"), scale=0.6)), wv.inputs["Vector"])
        wv.inputs["Scale"].default_value = swirl * 0.8
        wv.inputs["Distortion"].default_value = 6.0
        lines = nb.math("POWER", nb.out(wv, "Fac"), 3.0)
        col = nb.mix(lines, color, (1.0, 1.0, 1.0))
        strength_s = nb.math("MULTIPLY", strength_s, nb.math("ADD", 0.45, nb.math("MULTIPLY", lines, 1.4)))
    nb.set_in(em, "Color", col)
    nb.link(strength_s, em.inputs["Strength"])
    if kind == "core":
        tr = nb.node("ShaderNodeBsdfTransparent")
        mix = nb.node("ShaderNodeMixShader")
        nb.link(nb.math("MINIMUM", nb.math("MULTIPLY", fade.outputs[0], 4.0), 1.0), mix.inputs[0])
        nb.link(tr.outputs[0], mix.inputs[1])
        nb.link(em.outputs[0], mix.inputs[2])
        nb.link(mix.outputs[0], out.inputs[0])
    else:
        lw = nb.node("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.5
        a = nb.math("POWER", nb.math("SUBTRACT", 1.0, lw.outputs["Facing"]), falloff)
        a = nb.math("MINIMUM", nb.math("MULTIPLY", a, nb.math("MULTIPLY", fade.outputs[0], alpha)), 1.0)
        tr = nb.node("ShaderNodeBsdfTransparent")
        mix = nb.node("ShaderNodeMixShader")
        nb.link(a, mix.inputs[0])
        nb.link(tr.outputs[0], mix.inputs[1])
        nb.link(em.outputs[0], mix.inputs[2])
        nb.link(mix.outputs[0], out.inputs[0])
    return mat


def _sphere(name, r, mat, seg=32):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=r)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    ob.visible_shadow = False
    return ob


def point_light(name, color, energy=100.0, radius=0.1):
    ld = bpy.data.lights.new(name, "POINT")
    ld.color = color
    ld.energy = energy
    ld.shadow_soft_size = radius
    ob = bpy.data.objects.new(name, ld)
    fpv.link(ob)
    return ob


def key_scale(ob, frames_values):
    for f, v in frames_values:
        ob.scale = (v, v, v) if isinstance(v, (int, float)) else v
        ob.keyframe_insert("scale", frame=f)


def key_energy(light_ob, frames_values):
    for f, v in frames_values:
        light_ob.data.energy = v
        light_ob.data.keyframe_insert("energy", frame=f)


def key_time(mat, f0, f1, speed=0.6, fps=24):
    nd = mat.node_tree.nodes["Time"]
    for f in (f0, f1):
        nd.outputs[0].default_value = (f - f0) / fps * speed * 4
        nd.outputs[0].keyframe_insert("default_value", frame=f)
    _interp(mat.node_tree.animation_data, "LINEAR")


def energy_ball(name, parent, offset, color, radius, f_on, f_full, f_off, light_w=400.0, swirl=True, spin=True):
    """Energiekugel an einer Hand (Rasengan / Kamehameha-Ladung): weißer Kern, Wirbelschale, Glühhülle, Licht."""
    root = bpy.data.objects.new(name, None)
    fpv.link(root)
    root.parent = parent
    root.location = offset
    core = _sphere(name + "Core", radius * 0.5, glow_material(name + "CoreMat", tuple(0.6 + 0.4 * c for c in color),
                                                               14.0, kind="core"))
    shell_mat = glow_material(name + "SwirlMat", color, 7.0, falloff=0.7, swirl=6.0 / radius if swirl else 0.0)
    shell = _sphere(name + "Swirl", radius, shell_mat)
    glow_mat = glow_material(name + "GlowMat", color, 2.5, falloff=2.2, alpha=0.5)
    glow = _sphere(name + "Glow", radius * 1.9, glow_mat)
    for o in (core, shell, glow):
        o.parent = root
    lt = point_light(name + "Light", color, 0.0, radius * 0.6)
    lt.parent = root
    key_scale(root, [(f_on - 1, 0.001), (f_on, 0.05), (f_full, 1.0), (f_off - 1, 1.0), (f_off, 0.001)])
    key_energy(lt, [(f_on - 1, 0.0), (f_full, light_w), (f_off - 1, light_w), (f_off, 0.0)])
    if swirl:
        key_time(shell_mat, f_on, f_off)
    if spin:
        shell.rotation_mode = "XYZ"
        shell.rotation_euler = (0, 0, 0)
        shell.keyframe_insert("rotation_euler", frame=f_on)
        shell.rotation_euler = (math.radians(40), 0, math.radians(360 * (f_off - f_on) / 12))
        shell.keyframe_insert("rotation_euler", frame=f_off)
        _interp(shell.animation_data, "LINEAR")
    return root


def _bolt(rng, start, direction, length, jag, segs=10):
    d = Vector(direction).normalized()
    side = d.orthogonal().normalized()
    up = d.cross(side).normalized()
    pts = [Vector(start)]
    for i in range(1, segs + 1):
        t = i / segs
        off = side * rng.normal(0, jag) + up * rng.normal(0, jag)
        pts.append(Vector(start) + d * length * t + off * (1 - t * 0.3))
    return pts


def _bolt_curve(name, polylines, radius, mat):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    for pts in polylines:
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for i, (p, c) in enumerate(zip(sp.points, pts)):
            p.co = (*c, 1)
            p.radius = 1.0 - 0.7 * i / (len(pts) - 1)
    cu.bevel_depth = radius
    cu.bevel_resolution = 1
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    ob.visible_shadow = False
    fpv.link(ob)
    return ob


def lightning(name, parent, offset, color, f_on, f_off, radius=0.7, n_bolts=7, variants=6, seed=1, light_w=900.0,
              thickness=0.012):
    """Chidori/Blitze: pro Frame eine andere Variante verästelter Blitze (Sichtbarkeit konstant gekeyed),
    heller Kern und flackerndes Punktlicht."""
    rng = np.random.default_rng(seed)
    root = bpy.data.objects.new(name, None)
    fpv.link(root)
    root.parent = parent
    root.location = offset
    mat = glow_material(name + "Mat", tuple(0.75 + 0.25 * c for c in color), 18.0, kind="core")
    vs = []
    for v in range(variants):
        lines = []
        for b in range(n_bolts):
            d = Vector(rng.normal(0, 1, 3)).normalized()
            pts = _bolt(rng, (0, 0, 0), d, radius * rng.uniform(0.5, 1.3), radius * 0.09)
            lines.append(pts)
            k = int(rng.integers(3, 7))
            lines.append(_bolt(rng, pts[k], d + Vector(rng.normal(0, 0.8, 3)), radius * rng.uniform(0.2, 0.5),
                               radius * 0.06, segs=5))
        ob = _bolt_curve(f"{name}V{v}", lines, thickness, mat)
        ob.parent = root
        vs.append(ob)
    core = _sphere(name + "Core", radius * 0.12, glow_material(name + "CoreMat", (0.85, 0.92, 1.0), 16.0, kind="core"))
    core.parent = root
    glow = _sphere(name + "Glow", radius * 0.6, glow_material(name + "GlowMat", color, 2.0, falloff=2.5, alpha=0.5))
    glow.parent = root
    lt = point_light(name + "Light", color, 0.0, 0.2)
    lt.parent = root
    for f in range(f_on - 1, f_off + 2):
        for i, ob in enumerate(vs):
            vis = f_on <= f <= f_off and (f % variants) == i
            ob.hide_render = not vis
            ob.keyframe_insert("hide_render", frame=f)
        lt.data.energy = light_w * rng.uniform(0.5, 1.2) if f_on <= f <= f_off else 0.0
        lt.data.keyframe_insert("energy", frame=f)
    key_scale(core, [(f_on - 1, 0.001), (f_on, 1.0), (f_off, 1.0), (f_off + 1, 0.001)])
    key_scale(glow, [(f_on - 1, 0.001), (f_on, 1.0), (f_off, 1.0), (f_off + 1, 0.001)])
    for ob in vs:
        _interp(ob.animation_data, "CONSTANT")
    return root


def burst(name, loc, f0, color, r_max=6.0, dur=24, light_w=60000.0, ring=True, bolts=True, seed=5, ring_dz=0.0,
          core_s=25.0, glow_s=9.0, glow_alpha=0.32, ring_s=6.0, core_color=(0.9, 0.95, 1.0)):
    """Aufprall-Explosion: weißer Blitz, expandierende Glühkugel, Druckwellenring, Blitzbögen, Lichtspitze."""
    root = bpy.data.objects.new(name, None)
    fpv.link(root)
    root.location = loc
    core_m = glow_material(name + "CoreMat", core_color, core_s, kind="core")
    core = _sphere(name + "Core", 1.0, core_m)
    core.parent = root
    key_scale(core, [(f0 - 1, 0.001), (f0, r_max * 0.12), (f0 + 3, r_max * 0.35), (f0 + 7, r_max * 0.18),
                     (f0 + 10, 0.001)])
    gm = glow_material(name + "GlowMat", color, glow_s, falloff=1.9, alpha=glow_alpha)
    glow = _sphere(name + "Glow", 1.0, gm)
    glow.parent = root
    key_scale(glow, [(f0 - 1, 0.001), (f0, r_max * 0.25), (f0 + int(dur * 0.3), r_max), (f0 + dur, r_max * 1.15)])
    key_fade(gm, [(f0, 1.0), (f0 + int(dur * 0.25), 0.9), (f0 + int(dur * 0.6), 0.35), (f0 + dur, 0.0)])
    if ring:
        rm = glow_material(name + "RingMat", color, ring_s, falloff=0.6, alpha=0.6)
        bm = bmesh.new()
        n = 96
        outer = [bm.verts.new((math.cos(a), math.sin(a), 0)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]
        inner = [bm.verts.new((0.88 * math.cos(a), 0.88 * math.sin(a), 0))
                 for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]
        for i in range(n):
            bm.faces.new([outer[i], outer[(i + 1) % n], inner[(i + 1) % n], inner[i]])
        ringo = fpv.mesh_from_bmesh(bm, name + "Ring", rm)
        ringo.parent = root
        ringo.location = (0, 0, ring_dz)
        ringo.visible_shadow = False
        key_scale(ringo, [(f0 - 1, (0.001, 0.001, 1)), (f0 + 1, (r_max * 0.3, r_max * 0.3, 1)),
                          (f0 + dur, (r_max * 2.6, r_max * 2.6, 1))])
        key_fade(rm, [(f0, 1.0), (f0 + dur, 0.0)])
    if bolts:
        lightning(name + "Arc", root, (0, 0, 0), color, f0, f0 + int(dur * 0.45), radius=r_max * 0.8, n_bolts=9,
                  variants=5, seed=seed, light_w=0.0, thickness=0.05)
    lt = point_light(name + "Light", color, 0.0, r_max * 0.3)
    lt.parent = root
    key_energy(lt, [(f0 - 1, 0.0), (f0, light_w), (f0 + 4, light_w * 0.6), (f0 + dur, 0.0)])
    return root


def beam(name, origin, direction, length, radius, color, f_start, f_full, f_end, light_w=8000.0, wobble=None,
         core_s=14.0, whiten=0.7, glow_s=4.5, core_r=0.42):
    """Energiestrahl (Kamehameha, Todesstrahl): Kern + Glühmantel wachsen von der Quelle bis zur Länge,
    Kopfkugel an der Spitze, Punktlicht an Kopf und Quelle, Ausblenden am Ende.
    length: Zahl oder Liste [(frame, länge)] (Strahlenduell: Länge ändert sich); wobble = (f0, f1, amp):
    pulsierender Radius."""
    d = Vector(direction).normalized()
    root = bpy.data.objects.new(name, None)
    fpv.link(root)
    root.location = origin
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = d.to_track_quat("Z", "Y")
    if isinstance(length, (int, float)):
        lengths = [(f_start, 0.01), (f_full, float(length)), (f_end, float(length))]
    else:
        lengths = [(int(f), max(float(v), 0.01)) for f, v in length]
    rad = [(f_start - 1, 0.001), (f_start, 0.4), (min(f_start + 2, f_full), 1.0), (f_end - 4, 1.0), (f_end, 0.05)]
    if wobble:
        rng = np.random.default_rng(len(name))
        rad += [(f, 1.0 + wobble[2] * rng.uniform(-1, 1)) for f in range(wobble[0], wobble[1], 2)]
        rad.sort()
    cm = glow_material(name + "CoreMat", tuple(whiten + (1 - whiten) * c for c in color), core_s, kind="core")
    gm = glow_material(name + "GlowMat", color, glow_s, falloff=1.1, alpha=0.65)
    parts = []
    for (nm, r, m) in (("Core", radius * core_r, cm), ("Glow", radius, gm)):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=r, radius2=r, depth=1.0)
        bmesh.ops.translate(bm, vec=(0, 0, 0.5), verts=bm.verts)
        ob = fpv.mesh_from_bmesh(bm, name + nm, m)
        ob.visible_shadow = False
        ob.parent = root
        parts.append(ob)
    for ob in parts:
        for f, v in rad:
            ob.scale = (v, v, 1.0)
            ob.keyframe_insert("scale", index=0, frame=f)
            ob.keyframe_insert("scale", index=1, frame=f)
        for f, L in [(f_start - 1, 0.01)] + lengths:
            ob.scale = (1.0, 1.0, L)
            ob.keyframe_insert("scale", index=2, frame=f)
    head_m = glow_material(name + "HeadMat", color, glow_s * 1.8, falloff=0.9, alpha=0.7)
    head = _sphere(name + "Head", radius * 1.6, head_m)
    head.parent = root
    for f, L in [(f_start - 1, 0.0)] + lengths:
        head.location = (0, 0, L)
        head.keyframe_insert("location", frame=f)
    key_scale(head, [(f_start - 1, 0.001), (f_start, 0.6), (f_full, 1.0), (f_end - 4, 1.0), (f_end, 0.001)])
    for nm, zz in (("L0", 0.3), ("L1", None)):
        lt = point_light(name + nm, color, 0.0, radius)
        lt.parent = root
        if zz is None:
            for f, L in lengths:
                lt.location = (0, 0, L)
                lt.keyframe_insert("location", frame=f)
        else:
            lt.location = (0, 0, zz)
        key_energy(lt, [(f_start - 1, 0.0), (f_start + 2, light_w), (f_end - 4, light_w), (f_end, 0.0)])
    return root


def smoke_material(name, color):
    """Staub/Rauch als diffuse Quellwolke: im Kern fast deckend (rauscharm bei wenigen Samples), zur Silhouette
    weich auslaufend, von Rauschen aufgelockert; 'Fade' blendet aus. (Volumen wären realistischer, machen in
    Cycles aber jede Schattenabfrage der Szene mehrfach teurer.)"""
    mat, nb, out = fpv.new_material(name)
    fade = _fade_node(nb)
    co = nb.coords("Object")
    nz = nb.noise(co, scale=2.6, detail=5, rough=0.62)
    lw = nb.node("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    # harte, ausgefranste Kante statt Teiltransparenz (rauschfrei); Ausblenden = Auflösen über die Schwelle
    soft = nb.math("POWER", nb.math("SUBTRACT", 1.0, lw.outputs["Facing"]), 0.6)
    val = nb.math("MULTIPLY", soft, nb.math("ADD", nb.out(nz, "Fac"), 0.25))
    thr = nb.math("ADD", 0.42, nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, fade.outputs[0]), 0.9))
    a = nb.math("MINIMUM", nb.math("MAXIMUM", nb.math("MULTIPLY", nb.math("SUBTRACT", val, thr), 14.0), 0.0), 1.0)
    col = nb.mix(nb.out(nz, "Fac"), [c * 0.6 for c in color], [min(1.0, c * 1.1) for c in color])
    p = fpv.principled(nb, Base_Color=col, Roughness=1.0)
    tr = nb.node("ShaderNodeBsdfTransparent")
    mix = nb.node("ShaderNodeMixShader")
    nb.link(a, mix.inputs[0])
    nb.link(tr.outputs[0], mix.inputs[1])
    nb.link(p.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    return mat


def dust_cloud(name, loc, f0, r_max, color=(0.55, 0.50, 0.42), dur=48, seed=1, puffs=12, rise=0.35):
    """Aufgewirbelte Staub-/Rauchwolke: viele kleine, weiche Quellbüschel, die explosionsartig aufquellen,
    langsam steigen und in der zweiten Hälfte verblassen."""
    rng = np.random.default_rng(seed)
    root = bpy.data.objects.new(name, None)
    fpv.link(root)
    root.location = loc
    m = smoke_material(name + "Mat", color)
    for k in range(puffs):
        r = r_max * rng.uniform(0.22, 0.4)
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
        ob = fpv.mesh_from_bmesh(bm, f"{name}P{k}", m)
        fpv.displace_obj(ob, "CLOUDS", size=0.5, strength=0.18, depth=2, name=f"{name}P{k}_d")
        ob.parent = root
        az = rng.uniform(0, 2 * math.pi)
        rr = r_max * 0.6 * math.sqrt(rng.random())
        base = Vector((math.cos(az) * rr, math.sin(az) * rr, r * 0.3 + rng.uniform(0, 0.4) * r_max))
        ob.rotation_euler = tuple(rng.uniform(0, 6.3, 3))
        for f, sc, lift in ((f0 - 1, 0.001, 0.0), (f0, 0.2, 0.0), (f0 + 5, 0.75, 0.1), (f0 + int(dur * 0.5), 1.0, 0.6),
                            (f0 + dur, 1.2, 1.0)):
            ob.scale = (r * sc, r * sc, r * sc * 0.85)
            ob.keyframe_insert("scale", frame=f)
            ob.location = base * (0.3 + 0.7 * min(sc, 1.0)) + Vector((0, 0, rise * r_max * lift))
            ob.keyframe_insert("location", frame=f)
    key_fade(m, [(f0, 1.0), (f0 + int(dur * 0.4), 0.9), (f0 + dur, 0.0)])
    return root


def debris(name, loc, f0, mat, n=12, speed=14.0, size=0.25, seed=1, fps=24, dur=1.6, ground=None):
    """Weggeschleuderte Gesteinsbrocken (ballistisch, rotierend), landen auf 'ground' (z) und bleiben liegen."""
    rng = np.random.default_rng(seed)
    g = -9.81
    zg = loc[2] if ground is None else ground
    for k in range(n):
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=size * rng.uniform(0.5, 1.4))
        for v in bm.verts:
            v.co *= rng.uniform(0.7, 1.2)
        ob = fpv.mesh_from_bmesh(bm, f"{name}{k}", mat)
        az = rng.uniform(0, 2 * math.pi)
        el = rng.uniform(0.6, 1.3)
        v0 = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el))) * speed * rng.uniform(0.5,
                                                                                                                1.1)
        spin = Vector(rng.normal(0, 8, 3))
        ob.rotation_mode = "XYZ"
        steps = int(dur * fps)
        landed = None
        for i in range(-1, steps + 1, 2):
            t = max(i, 0) / fps
            p = Vector(loc) + v0 * t + Vector((0, 0, 0.5 * g * t * t))
            if p.z < zg and t > 0.1:
                if landed is None:
                    landed = p.copy()
                    landed.z = zg
                p = landed
            ob.location = p
            ob.keyframe_insert("location", frame=f0 + i)
            if landed is None:
                ob.rotation_euler = tuple(spin * t)
                ob.keyframe_insert("rotation_euler", frame=f0 + i)
        ob.scale = (0.001, 0.001, 0.001)
        ob.keyframe_insert("scale", frame=f0 - 1)
        ob.scale = (1, 1, 1)
        ob.keyframe_insert("scale", frame=f0)
        _interp(ob.animation_data, "LINEAR")


def ki_blast(name, p0, p1, f0, f1, color, radius=0.35, impact=True, r_impact=3.0, seed=1):
    """Ki-Kugel fliegt von p0 nach p1 (Bewegungsunschärfe zieht Schweif), Einschlag-Explosion bei p1."""
    m = glow_material(name + "Mat", color, 10.0, falloff=0.8)
    ob = _sphere(name, radius, m, seg=20)
    core = _sphere(name + "Core", radius * 0.5, glow_material(name + "CoreMat", (1, 1, 1), 14.0, kind="core"), seg=16)
    core.parent = ob
    lt = point_light(name + "Light", color, 0.0, radius)
    lt.parent = ob
    for f, p, s in ((f0 - 1, p0, 0.001), (f0, p0, 1.0), (f1, p1, 1.0), (f1 + 1, p1, 0.001)):
        ob.location = p
        ob.keyframe_insert("location", frame=f)
        ob.scale = (s, s, s)
        ob.keyframe_insert("scale", frame=f)
    key_energy(lt, [(f0 - 1, 0.0), (f0, 1500.0), (f1, 1500.0), (f1 + 1, 0.0)])
    _interp(ob.animation_data, "LINEAR")
    if impact:
        burst(name + "Hit", p1, f1, color, r_max=r_impact, dur=16, light_w=15000.0, ring=False, bolts=False, seed=seed)
    return ob


def aura(name, parent, color, f_on, f_off, height=1.9, width=0.75, light_w=600.0):
    """Super-Saiyajin-Aura: flammenförmige, nach oben strömende Glühhülle mit hellem Rand um den Körper,
    Flammenzungen oben, weicher Halo und flackerndes Licht."""
    rim = tuple(min(1.0, c * 0.5 + 0.55) for c in color)

    def shell_mat(nm, strength, alpha, streak_scale):
        mat, nb, out = fpv.new_material(nm)
        fade = _fade_node(nb)
        t = _fade_node(nb, "Time", 0.0)
        co = nb.coords("Object")
        x, y, z = nb.sep(co)
        flow = nb.comb(nb.math("MULTIPLY", x, 4.0), nb.math("MULTIPLY", y, 4.0),
                       nb.math("SUBTRACT", nb.math("MULTIPLY", z, 0.9), t.outputs[0]))
        nz = nb.noise(flow, scale=streak_scale, detail=3, rough=0.55)
        streak = nb.math("MINIMUM", nb.math("MAXIMUM", nb.math("MULTIPLY", nb.math("SUBTRACT", nb.out(nz, "Fac"), 0.42),
                                                                        4.0), 0.0), 1.0)
        lw = nb.node("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.4
        edge = nb.math("POWER", lw.outputs["Facing"], 1.2)
        a = nb.math("ADD", 0.01, nb.math("MULTIPLY", streak, 0.10))
        a = nb.math("ADD", a, nb.math("MULTIPLY", edge, nb.math("ADD", nb.math("MULTIPLY", streak, 0.35), 0.12)))
        a = nb.math("MINIMUM", nb.math("MULTIPLY", a, nb.math("MULTIPLY", fade.outputs[0], alpha)), 1.0)
        em = nb.node("ShaderNodeEmission")
        nb.set_in(em, "Color", nb.mix(edge, color, rim))
        s = nb.math("ADD", nb.math("MULTIPLY", streak, 6.0), nb.math("MULTIPLY", edge, 4.0))
        nb.link(nb.math("MULTIPLY", nb.math("ADD", s, 2.5), strength), em.inputs["Strength"])
        tr = nb.node("ShaderNodeBsdfTransparent")
        mix = nb.node("ShaderNodeMixShader")
        nb.link(a, mix.inputs[0])
        nb.link(tr.outputs[0], mix.inputs[1])
        nb.link(em.outputs[0], mix.inputs[2])
        nb.link(mix.outputs[0], out.inputs[0])
        return mat, t, fade

    def flame_shell(nm, mat, grow):
        ob = _sphere(nm, 1.0, mat, seg=48)
        ob.parent = parent
        ob.location = (0, 0, height * 0.46)
        ob.scale = (width * 0.55 * grow, width * 0.45 * grow, height * 0.6 * grow)
        rng = np.random.default_rng(len(nm))
        ph = rng.uniform(0, 6.3, 3)
        for v in ob.data.vertices:
            x, y, z = v.co
            if z > 0:
                a = math.atan2(y, x)
                tongue = (0.5 + 0.5 * math.sin(a * 5 + ph[0])) ** 3 + 0.6 * (0.5 + 0.5 * math.sin(a * 9 + ph[1])) ** 4
                v.co.z = z * (1.3 + 0.55 * tongue * z)
                k = 1.0 - 0.45 * z
                v.co.x, v.co.y = x * k, y * k
            else:
                v.co.z = z * 0.85
        return ob

    m1, _, _ = shell_mat(name + "Mat", 1.0, 1.0, 2.6)
    flame_shell(name, m1, 1.0)
    m2, _, _ = shell_mat(name + "HaloMat", 0.4, 0.22, 1.6)
    flame_shell(name + "Halo", m2, 1.22)
    for m in (m1, m2):
        key_fade(m, [(f_on - 1, 0.0), (f_on + 3, 1.0), (f_off - 3, 1.0), (f_off, 0.0)])
        nd = m.node_tree.nodes["Time"]
        for f in (f_on - 1, f_off):
            nd.outputs[0].default_value = (f - f_on) / 24.0 * 3.6
            nd.outputs[0].keyframe_insert("default_value", frame=f)
        _interp(m.node_tree.animation_data, "LINEAR")
    lt = point_light(name + "Light", color, 0.0, 0.6)
    lt.parent = parent
    lt.location = (0, 0, height * 0.55)
    rng = np.random.default_rng(3)
    for f in range(f_on - 1, f_off + 1):
        lt.data.energy = light_w * rng.uniform(0.7, 1.1) if f_on <= f < f_off else 0.0
        lt.data.keyframe_insert("energy", frame=f)
    return m1
