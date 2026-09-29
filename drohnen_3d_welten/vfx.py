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
        for f, hid in ((f_start - 2, True), (f_start - 1, False), (f_end + 1, True)):   # ganz weg nach dem Ende
            ob.hide_render = hid
            ob.keyframe_insert("hide_render", frame=f)
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


def aura(name, parent, color, f_on, f_off, height=1.9, width=0.75, light_w=600.0, opacity=1.0, edge_w=1.0,
         strength=1.0):
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
        a = nb.math("ADD", a, nb.math("MULTIPLY", nb.math("MULTIPLY", edge, edge_w),
                                      nb.math("ADD", nb.math("MULTIPLY", streak, 0.35), 0.12)))
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

    m1, _, _ = shell_mat(name + "Mat", strength, opacity, 2.6)
    flame_shell(name, m1, 1.0)
    m2, _, _ = shell_mat(name + "HaloMat", 0.4 * strength, 0.22 * opacity, 1.6)
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


# --------------------------------------------------------------------------
# Impact-Paket (Naruto Schritt 3.5): Funken, Druckwelle, Bruch + Rigid Body
# --------------------------------------------------------------------------

def sparks_gn(name, origin, t0, n=140, speed=(6.0, 15.0), life=(0.2, 0.55), color=(1.0, 0.72, 0.32),
              strength=40.0, radius=(0.01, 0.025), gravity=-9.81, up=0.35, spread_dir=None, seed=3):
    """Funkenregen als Geometry Nodes mit der Szenenzeit: n Punkte starten bei t0 am Ursprung, fliegen ballistisch
    (Zufallsrichtung, nach oben verschoben), leuchten heiß und verschwinden nach ihrer Lebensdauer. Deterministisch,
    jeder Frame einzeln renderbar; Bewegungsunschärfe zieht sie zu Streifen."""
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    fpv.link(ob)
    ob.location = origin
    ng = bpy.data.node_groups.new(name + "GN", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    g = fpv.NB(ng)
    go = g.node("NodeGroupOutput")
    ts = g.node("GeometryNodeInputSceneTime").outputs["Seconds"]
    age = g.math("SUBTRACT", ts, t0)

    def rnd(dtype, lo, hi, sd):
        r = g.node("FunctionNodeRandomValue")
        r.data_type = dtype
        if dtype == "FLOAT_VECTOR":
            r.inputs["Min"].default_value = lo
            r.inputs["Max"].default_value = hi
        else:
            r.inputs[2].default_value = lo
            r.inputs[3].default_value = hi
        r.inputs["Seed"].default_value = sd
        return r.outputs[0] if dtype == "FLOAT_VECTOR" else r.outputs[1]
    d = g.vmath("NORMALIZE", g.vmath("ADD", rnd("FLOAT_VECTOR", (-1, -1, -1), (1, 1, 1), seed), (0, 0, up)))
    if spread_dir is not None:
        d = g.vmath("NORMALIZE", g.vmath("ADD", d, tuple(spread_dir)))
    v = g.vmath("SCALE", d, scale=rnd("FLOAT", speed[0], speed[1], seed + 1))
    lf = rnd("FLOAT", life[0], life[1], seed + 2)
    a = g.math("MAXIMUM", age, 0.0)
    p = g.vmath("ADD", g.vmath("SCALE", v, scale=a), g.comb(0.0, 0.0, g.math("MULTIPLY", 0.5 * gravity, g.math("MULTIPLY", a, a))))
    pts = g.node("GeometryNodePoints")
    pts.inputs["Count"].default_value = n
    g.link(p, pts.inputs["Position"])
    # Radius schrumpft mit dem Alter (abkühlen)
    rad = g.math("MULTIPLY", rnd("FLOAT", radius[0], radius[1], seed + 3),
                 g.math("SUBTRACT", 1.0, g.math("MINIMUM", g.math("DIVIDE", a, lf), 1.0)))
    g.link(rad, pts.inputs["Radius"])
    dead = g.math("MAXIMUM", g.math("LESS_THAN", age, 0.0), g.math("GREATER_THAN", age, lf))
    dg = g.node("GeometryNodeDeleteGeometry")
    g.link(pts.outputs[0], dg.inputs["Geometry"])
    g.link(dead, dg.inputs["Selection"])
    mat, nb, out = fpv.new_material(name + "Mat")
    em = nb.node("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1)
    em.inputs["Strength"].default_value = strength
    nb.link(em.outputs[0], out.inputs[0])
    mat.cycles.emission_sampling = "NONE"
    sm = g.node("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = mat
    g.link(dg.outputs[0], sm.inputs["Geometry"])
    g.link(sm.outputs[0], go.inputs[0])
    mod = ob.modifiers.new("Sparks", "NODES")
    mod.node_group = ng
    ob.visible_shadow = False
    return ob


def shockwave(name, loc, f0, r_max=12.0, dur=12, color=(0.55, 0.75, 1.0), thick=0.35, glow=6.0, ior=1.04, fps=24):
    """Druckwelle als flacher Ring (Geometry Nodes, Szenenzeit): der Radius wächst schnell und bremst ab
    (1 - e^-kt), der Querschnitt bleibt dünn und wird flacher; Material: leichte Brechung (Hitzeflimmern) plus
    leuchtende Kante, blendet mit dem Alter aus. Vor f0 und nach dur Frames ist keine Geometrie da."""
    t0 = (f0 - 1) / fps
    life = dur / fps
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    fpv.link(ob)
    ob.location = loc
    ng = bpy.data.node_groups.new(name + "GN", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    g = fpv.NB(ng)
    go = g.node("NodeGroupOutput")
    ts = g.node("GeometryNodeInputSceneTime").outputs["Seconds"]
    age = g.math("SUBTRACT", ts, t0)
    an = g.math("MINIMUM", g.math("MAXIMUM", g.math("DIVIDE", age, life), 0.0), 1.0)
    rad = g.math("MAXIMUM", g.math("MULTIPLY", r_max, g.math("SUBTRACT", 1.0, g.math("EXPONENT", g.math("MULTIPLY", an, -3.5)))), 0.05)
    cc = g.node("GeometryNodeCurvePrimitiveCircle")
    cc.inputs["Resolution"].default_value = 96
    g.link(rad, cc.inputs["Radius"])
    pc = g.node("GeometryNodeCurvePrimitiveCircle")
    pc.inputs["Resolution"].default_value = 10
    g.link(g.math("MULTIPLY", thick, g.math("SUBTRACT", 1.0, g.math("MULTIPLY", an, 0.6))), pc.inputs["Radius"])
    c2m = g.node("GeometryNodeCurveToMesh")
    g.link(cc.outputs[0], c2m.inputs["Curve"])
    g.link(pc.outputs[0], c2m.inputs["Profile Curve"])
    tf = g.node("GeometryNodeTransform")
    g.link(c2m.outputs[0], tf.inputs["Geometry"])
    tf.inputs["Scale"].default_value = (1.0, 1.0, 0.45)
    st = g.node("GeometryNodeStoreNamedAttribute")
    st.data_type = "FLOAT"
    st.domain = "POINT"
    st.inputs["Name"].default_value = "wave_age"
    g.link(tf.outputs[0], st.inputs["Geometry"])
    g.link(an, st.inputs["Value"])
    dead = g.math("MAXIMUM", g.math("LESS_THAN", age, 0.0), g.math("GREATER_THAN", age, life))
    dg = g.node("GeometryNodeDeleteGeometry")
    g.link(st.outputs[0], dg.inputs["Geometry"])
    g.link(dead, dg.inputs["Selection"])
    mat, nb, out = fpv.new_material(name + "Mat")
    at = nb.node("ShaderNodeAttribute", attribute_name="wave_age")
    fade = nb.math("SUBTRACT", 1.0, nb.math("POWER", nb.out(at, "Fac"), 0.7))
    glass = nb.node("ShaderNodeBsdfRefraction")
    glass.inputs["IOR"].default_value = ior
    glass.inputs["Roughness"].default_value = 0.08
    lw = nb.node("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.25
    em = nb.node("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1)
    nb.link(nb.math("MULTIPLY", nb.math("MULTIPLY", nb.out(lw, "Facing"), glow), fade), em.inputs["Strength"])
    add = nb.node("ShaderNodeAddShader")
    nb.link(glass.outputs[0], add.inputs[0])
    nb.link(em.outputs[0], add.inputs[1])
    tr = nb.node("ShaderNodeBsdfTransparent")
    mix = nb.node("ShaderNodeMixShader")
    nb.link(nb.math("MULTIPLY", fade, 0.6), mix.inputs[0])
    nb.link(tr.outputs[0], mix.inputs[1])
    nb.link(add.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    mat.cycles.emission_sampling = "NONE"
    sm = g.node("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = mat
    g.link(dg.outputs[0], sm.inputs["Geometry"])
    g.link(sm.outputs[0], go.inputs[0])
    mod = ob.modifiers.new("Wave", "NODES")
    mod.node_group = ng
    ob.visible_shadow = False
    return ob


def voronoi_fracture(name, src_bm, seeds, mat, min_verts=4):
    """Voronoi-Bruch eines geschlossenen Meshes (bmesh in Weltkoordinaten): pro Saatpunkt eine konvexe Zelle durch
    Halbraum-Schnitte (bisect + Deckel füllen) -> scharfkantige Bruchstücke, Ursprung im Schwerpunkt."""
    out = []
    S = [Vector(s) for s in seeds]
    for i, si in enumerate(S):
        bm = src_bm.copy()
        for j, sj in enumerate(S):
            if i == j or not bm.verts:
                continue
            mid = (si + sj) / 2
            nrm = (sj - si).normalized()
            res = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=mid,
                                         plane_no=nrm, clear_outer=True)
            cut = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
            if cut:
                bmesh.ops.holes_fill(bm, edges=cut, sides=0)
        if len(bm.verts) < min_verts:
            bm.free()
            continue
        c = sum((v.co for v in bm.verts), Vector()) / len(bm.verts)
        for v in bm.verts:
            v.co -= c
        me = bpy.data.meshes.new(f"{name}{i}")
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = False
        me.materials.append(mat)
        ob = bpy.data.objects.new(f"{name}{i}", me)
        ob.location = c
        fpv.link(ob)
        out.append(ob)
    return out


def rigid_sim(pieces, colliders, f_release, f_end, impulses, fps=24, mass_density=600.0, friction=0.6,
              restitution=0.25):
    """Echte Rigid-Body-Simulation (Blender Bullet) für Bruchstücke: bis f_release kinematisch (stehen an Ort und
    Stelle), danach dynamisch; Impulse über kurz eingeschaltete Kraftfelder (impulses = [(ort, stärke, f0, f1)]).
    Die Simulation wird Frame für Frame durchlaufen und als Keyframes gebacken, danach wird die Physik entfernt
    (jeder Frame ist einzeln renderbar)."""
    sc = bpy.context.scene
    if sc.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    rbw = sc.rigidbody_world
    rbw.point_cache.frame_start = f_release - 2
    rbw.point_cache.frame_end = f_end
    rbw.substeps_per_frame = 10
    rbw.solver_iterations = 20
    coll = rbw.collection
    if coll is None:
        coll = bpy.data.collections.new("RigidBodyWorld")
        rbw.collection = coll

    def add_rb(ob, kind):
        with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob], selected_editable_objects=[ob]):
            bpy.ops.rigidbody.object_add(type=kind)
    for ob in colliders:
        add_rb(ob, "PASSIVE")
        ob.rigid_body.collision_shape = "MESH"
        ob.rigid_body.friction = 0.8
    for ob in pieces:
        add_rb(ob, "ACTIVE")
        rb = ob.rigid_body
        rb.collision_shape = "CONVEX_HULL"
        dims = ob.dimensions
        rb.mass = max(0.2, mass_density * dims.x * dims.y * dims.z * 0.5)
        rb.friction = friction
        rb.restitution = restitution
        rb.collision_margin = 0.01
        rb.kinematic = True
        rb.keyframe_insert("kinematic", frame=f_release - 1)
        rb.kinematic = False
        rb.keyframe_insert("kinematic", frame=f_release)
    fields = []
    for (loc, strength, f0, f1) in impulses:
        bpy.ops.object.effector_add(type="FORCE", location=tuple(loc))
        fo = bpy.context.active_object
        fo.field.falloff_type = "SPHERE"
        fo.field.use_max_distance = True
        fo.field.distance_max = 6.0
        for f, s in ((f0 - 1, 0.0), (f0, strength), (f1, strength), (f1 + 1, 0.0)):
            fo.field.strength = s
            fo.field.keyframe_insert("strength", frame=f)
        fields.append(fo)
    mats = {ob.name: [] for ob in pieces}
    for f in range(f_release - 2, f_end + 1):
        sc.frame_set(f)
        for ob in pieces:
            mats[ob.name].append(ob.matrix_world.copy())
    # Physik entfernen, Bahn als Keyframes
    for ob in pieces + colliders:
        with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
            bpy.ops.rigidbody.object_remove()
    for fo in fields:
        bpy.data.objects.remove(fo, do_unlink=True)
    bpy.ops.rigidbody.world_remove()
    for ob in pieces:
        ob.animation_data_clear()
        ob.rotation_mode = "QUATERNION"
        prev = None
        for k, M in enumerate(mats[ob.name]):
            f = f_release - 2 + k
            loc, q, _ = M.decompose()
            if prev is not None and prev.dot(q) < 0:
                q = -q
            prev = q
            ob.location = loc
            ob.rotation_quaternion = q
            ob.keyframe_insert("location", frame=f)
            ob.keyframe_insert("rotation_quaternion", frame=f)
    sc.frame_set(1)
    return pieces


def dust_gn(name, loc, t0, n=700, r_max=7.0, rise=1.6, life=1.5, color=(0.46, 0.40, 0.33), size=(0.06, 0.28),
            k_drag=3.0, alpha=1.0, seed=5):
    """Staub als Partikel statt Kugel-Wolken: Punkte schießen radial flach über den Boden (Luftwiderstand), steigen
    leicht, werden größer und blassen aus (Alter als Attribut -> Transparenz im Shader). Deterministisch über die
    Szenenzeit; nach der Lebensdauer ist nichts mehr übrig."""
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    fpv.link(ob)
    ob.location = loc
    ng = bpy.data.node_groups.new(name + "GN", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    g = fpv.NB(ng)
    go = g.node("NodeGroupOutput")
    ts = g.node("GeometryNodeInputSceneTime").outputs["Seconds"]
    age = g.math("SUBTRACT", ts, t0)

    def rnd(lo, hi, sd):
        r = g.node("FunctionNodeRandomValue")
        r.data_type = "FLOAT"
        r.inputs[2].default_value = lo
        r.inputs[3].default_value = hi
        r.inputs["Seed"].default_value = sd
        return r.outputs[1]
    ang = rnd(0.0, 6.2832, seed)
    reach = rnd(0.3, 1.0, seed + 1)
    lf = rnd(life * 0.6, life, seed + 2)
    a = g.math("MAXIMUM", age, 0.0)
    an = g.math("MINIMUM", g.math("DIVIDE", a, lf), 1.0)
    rr = g.math("MULTIPLY", g.math("MULTIPLY", reach, r_max),
                g.math("SUBTRACT", 1.0, g.math("EXPONENT", g.math("MULTIPLY", a, -k_drag))))
    z = g.math("MULTIPLY", g.math("MULTIPLY", rnd(0.2, 1.0, seed + 3), rise), g.math("POWER", an, 0.7))
    p = g.comb(g.math("MULTIPLY", rr, g.math("COSINE", ang)), g.math("MULTIPLY", rr, g.math("SINE", ang)), z)
    pts = g.node("GeometryNodePoints")
    pts.inputs["Count"].default_value = n
    g.link(p, pts.inputs["Position"])
    # Körnchen schrumpfen mit dem Alter (statt transparenter Kugeln: kein Transparenz-Rauschen)
    g.link(g.math("MULTIPLY", rnd(size[0], size[1], seed + 4), g.math("SUBTRACT", 1.05, g.math("POWER", an, 1.5))),
           pts.inputs["Radius"])
    st = g.node("GeometryNodeStoreNamedAttribute")
    st.data_type = "FLOAT"
    st.domain = "POINT"
    st.inputs["Name"].default_value = "dust_age"
    g.link(pts.outputs[0], st.inputs["Geometry"])
    g.link(an, st.inputs["Value"])
    dead = g.math("MAXIMUM", g.math("LESS_THAN", age, 0.0), g.math("GREATER_THAN", age, lf))
    dg = g.node("GeometryNodeDeleteGeometry")
    g.link(st.outputs[0], dg.inputs["Geometry"])
    g.link(dead, dg.inputs["Selection"])
    mat, nb, out = fpv.new_material(name + "Mat")
    at = nb.node("ShaderNodeAttribute", attribute_name="dust_age")
    fa = nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, nb.math("POWER", nb.out(at, "Fac"), 3.0)), alpha)
    diff = fpv.principled(nb, Base_Color=color, Roughness=1.0)
    tr = nb.node("ShaderNodeBsdfTransparent")
    mx = nb.node("ShaderNodeMixShader")
    nb.link(fa, mx.inputs[0])
    nb.link(tr.outputs[0], mx.inputs[1])
    nb.link(diff.outputs[0], mx.inputs[2])
    nb.link(mx.outputs[0], out.inputs[0])
    sm = g.node("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = mat
    g.link(dg.outputs[0], sm.inputs["Geometry"])
    g.link(sm.outputs[0], go.inputs[0])
    mod = ob.modifiers.new("Dust", "NODES")
    mod.node_group = ng
    ob.visible_shadow = False
    return ob


def _volume_body(name, mat, seed):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    ob.visible_shadow = True
    ob.cycles.use_motion_blur = False
    return ob


def _volume_material(name, color, dens, fire, seed, noise_scale=3.4, cool=0.8, t_hot=1500.0, lumps=0.95):
    """Prozedurales Volumen (Objektkoordinaten -1..1): Dichte aus zwei 4D-Rauschlagen mit Verzerrung (quellende
    Ballen), Rand je Richtung zwischen ~0,55 und ~0,95 des Radius (kein Kugelumriss); Glut per Schwarzkörper,
    fleckig verteilt (auch an der Oberfläche) und mit dem Alter abkühlend. Gekeyte Value-Knoten: 'Age' (s),
    'Dens' (Dichte-Faktor), 'Fire' (Glut-Faktor)."""
    mat, nb, out = fpv.new_material(name)
    age = nb.node("ShaderNodeValue")
    age.name = "Age"
    dn = nb.node("ShaderNodeValue")
    dn.name = "Dens"
    dn.outputs[0].default_value = dens
    fr = nb.node("ShaderNodeValue")
    fr.name = "Fire"
    fr.outputs[0].default_value = fire
    co = nb.coords("Object")
    r = nb.vmath("LENGTH", co)

    def noise(vec, scale, w_rate, detail, rough, sd):
        n = nb.node("ShaderNodeTexNoise")
        n.noise_dimensions = "4D"
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        nb.link(vec, n.inputs["Vector"])
        nb.link(nb.math("ADD", nb.math("MULTIPLY", age.outputs[0], w_rate), sd), n.inputs["W"])
        return n.outputs["Fac"]
    f0 = noise(co, noise_scale * 0.45, 0.25, 1.0, 0.5, seed * 3.7)                 # große Ballen (Umriss)
    f1 = noise(co, noise_scale, 0.5, 3.0, 0.62, seed * 1.9)                          # Quellstruktur (+ Feinanteil)
    edge = nb.math("ADD", 0.55, nb.math("MULTIPLY", f0, lumps * 0.75))
    shape = nb.math("SUBTRACT", nb.math("SUBTRACT", edge, r), nb.math("MULTIPLY", nb.math("SUBTRACT", f1, 0.5), 0.22))
    billow = nb.math("POWER", nb.math("MINIMUM", nb.math("MAXIMUM", nb.math("MULTIPLY", nb.math("SUBTRACT", f1, 0.25),
                                                                               2.2), 0.0), 1.0), 1.6)
    d = nb.math("MULTIPLY", nb.math("MINIMUM", nb.math("MAXIMUM", nb.math("MULTIPLY", shape, 8.0), 0.0), 1.0),
                nb.math("ADD", 0.12, billow))
    heat = nb.math("MULTIPLY", nb.math("MINIMUM", nb.math("MAXIMUM", nb.math("MULTIPLY", shape, 4.0), 0.0), 1.0),
                   nb.math("MAXIMUM", nb.math("SUBTRACT", nb.math("MULTIPLY", f1, 1.8),
                                               nb.math("MULTIPLY", age.outputs[0], cool)), 0.0))
    heat = nb.math("MINIMUM", heat, 1.0)
    pv = nb.node("ShaderNodeVolumePrincipled")
    pv.inputs["Color"].default_value = (*color, 1)
    pv.inputs["Anisotropy"].default_value = 0.3
    nb.link(nb.math("MULTIPLY", d, dn.outputs[0]), pv.inputs["Density"])
    nb.link(nb.math("MULTIPLY", nb.math("MULTIPLY", heat, heat), fr.outputs[0]), pv.inputs["Blackbody Intensity"])
    nb.link(nb.math("ADD", 800.0, nb.math("MULTIPLY", heat, t_hot)), pv.inputs["Temperature"])
    nb.link(pv.outputs[0], out.inputs["Volume"])
    return mat


def _key_value(mat, name, keys, interp="LINEAR"):
    nd = mat.node_tree.nodes[name]
    for f, v in keys:
        nd.outputs[0].default_value = v
        nd.outputs[0].keyframe_insert("default_value", frame=f)


def volume_blast(name, center, f0, r_fire=9.0, r_smoke=15.0, rise=16.0, dur=100, fps=24, seed=7, fire=18.0):
    """Explosion als prozedurales Volumen – KEINE Fluid-Simulation (Mantaflow bricht in diesem bpy-Build ab):
    Feuerball quillt in ~0,3 s auf (turbulente Dichte, Schwarzkörper-Glut 3350 K -> 750 K), kühlt über ~1,5 s zu
    Ruß ab; ein größerer Rauchkörper wächst aus ihm heraus, rollt (Rauschen läuft mit dem Alter) und steigt auf.
    Zeit über gekeyte Value-Knoten; jeder Frame einzeln renderbar."""
    c = Vector(center)
    fm = _volume_material(name + "FireMat", (0.13, 0.11, 0.10), 1.6, fire, seed, noise_scale=5.0, cool=0.5)
    fb = _volume_body(name + "Fire", fm, seed)
    fb.location = c
    for f, s, dz in ((f0 - 1, 0.001, 0.0), (f0, 0.3, 0.0), (f0 + 3, 0.72, 0.3), (f0 + 8, 0.93, 1.0),
                     (f0 + 24, 1.08, 3.5), (f0 + dur, 1.25, rise * 0.55)):
        fb.scale = (r_fire * s * 1.3,) * 3
        fb.keyframe_insert("scale", frame=f)
        fb.location = c + Vector((0, 0, dz))
        fb.keyframe_insert("location", frame=f)
    _key_value(fm, "Age", [(f0, 0.0), (f0 + dur, dur / fps)])
    _key_value(fm, "Dens", [(f0 - 1, 0.0), (f0, 1.0), (f0 + 12, 1.6), (f0 + 48, 1.4), (f0 + dur, 0.5)])
    _key_value(fm, "Fire", [(f0 - 1, 0.0), (f0, fire * 1.6), (f0 + 6, fire), (f0 + 30, fire * 0.5), (f0 + 60, fire * 0.2),
                            (f0 + dur, fire * 0.08)])
    sm = _volume_material(name + "SmokeMat", (0.20, 0.19, 0.18), 1.2, fire * 0.25, seed + 5, noise_scale=4.5,
                          cool=1.4)
    sb = _volume_body(name + "Smoke", sm, seed + 5)
    for f, s, dz in ((f0 + 7, 0.001, 0.0), (f0 + 8, 0.5, 1.0), (f0 + 20, 0.8, 3.5), (f0 + 48, 1.0, rise * 0.55),
                     (f0 + dur, 1.15, rise)):
        sb.scale = (r_smoke * s * 1.3, r_smoke * s * 1.3, r_smoke * s * 1.45)
        sb.keyframe_insert("scale", frame=f)
        sb.location = c + Vector((0, 0, dz + r_smoke * 0.15 * s))
        sb.keyframe_insert("location", frame=f)
    _key_value(sm, "Age", [(f0, 0.0), (f0 + dur, dur / fps)])
    _key_value(sm, "Dens", [(f0 + 8, 0.0), (f0 + 16, 1.2), (f0 + 60, 1.0), (f0 + dur, 0.55)])
    _key_value(sm, "Fire", [(f0 + 3, fire * 0.4), (f0 + 30, fire * 0.1), (f0 + 60, 0.0)])
    for ob in (fb, sb):
        for f, hid in ((1, True), (f0 - 2, True), (f0 - 1, False)):
            ob.hide_render = hid
            ob.keyframe_insert("hide_render", frame=f)
    return fb, sb
