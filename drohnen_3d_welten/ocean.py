"""Ozean: Blender-Ocean-Modifier (FFT-Wellen) nahe der Route + flache Fernfläche."""
import math

import bpy
import numpy as np

import fpv
import textures


def foam_lace(nb, wp_t, scale=1.3, width=0.12, stretch=1.0):
    """Schaumnetz: verzerrte Voronoi-Kanten, 1 auf den Kanten, 0 in den Zellen.
    stretch < 1 zieht das Netz entlang X zu Windstreifen."""
    if stretch != 1.0:
        wp_t = nb.mapping(wp_t, scale=(stretch, 1.0, 1.0))
    warp = nb.vmath("ADD", wp_t, nb.vmath("SCALE", nb.out(nb.noise(wp_t, scale=0.35, detail=2, rough=0.5), "Color"),
                                          scale=1.6))
    vor = nb.voronoi(warp, scale=scale, feature="DISTANCE_TO_EDGE")
    mr = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(vor, "Distance"), mr.inputs["Value"])
    mr.inputs["From Min"].default_value = width
    mr.inputs["From Max"].default_value = 0.0
    return mr.outputs[0]


def lacy_foam(nb, density, noise_fac, lace, cap=0.95, soft=(0.05, 0.12)):
    """Dichte 0..1 -> Schaum mit Struktur statt Aufkleber-Fleck: dicht = geschlossen mit Löchern,
    mittel = Netz, dünn = einzelne Fäden. Schwelle wandert mit der Dichte über Rauschen + Netz."""
    f = nb.math("ADD", nb.math("MULTIPLY", noise_fac, 0.6), nb.math("MULTIPLY", lace, 0.4))
    t = nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", density, 0.9))
    mr = nb.node("ShaderNodeMapRange", clamp=True)
    mr.interpolation_type = "SMOOTHSTEP"
    nb.link(f, mr.inputs["Value"])
    nb.link(nb.math("SUBTRACT", t, soft[0]), mr.inputs["From Min"])
    nb.link(nb.math("ADD", t, soft[1]), mr.inputs["From Max"])
    return nb.math("MULTIPLY", mr.outputs[0], nb.math("MINIMUM", nb.math("MULTIPLY", density, 1.6), cap))


def water_material(name="Water", deep=(0.004, 0.028, 0.040), shallow=(0.02, 0.09, 0.10), rough=0.035,
                   foam_attr="foam", wake_fn=None, far=False, foam_amount=1.0, color_fn=None, lace=False,
                   view_dark=0.0, micro=0.0):
    """Wasser-Shader. Performance: nur 3D-Rauschen mit animiertem Versatz, wenige Oktaven.
    lace: Ozean-Schaum als Netz (foam_lace/lacy_foam) statt weicher Flecken; view_dark: Blickwinkel-Verlauf
    (senkrecht in die Tiefe dunkler, flach heller); micro: feine Kräuselung mit Windflecken (Stärke)."""
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
    if color_fn is not None:
        col = color_fn(nb, col)
    if view_dark:
        lw = nb.node("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.35
        face = nb.math("POWER", nb.out(lw, "Facing"), 1.5)
        col = nb.mix(face, nb.vmath("SCALE", col, scale=1.0 - view_dark), nb.vmath("SCALE", col, scale=1.0 + 0.25 * view_dark))
    foam = None
    if not far:
        at = nb.node("ShaderNodeAttribute", attribute_name=foam_attr)
        fnoise = nb.noise(wpos, scale=0.9, detail=3, rough=0.6)
        if lace:
            dens = nb.node("ShaderNodeMapRange", clamp=True)
            nb.link(nb.out(at, "Fac"), dens.inputs["Value"])
            dens.inputs["From Min"].default_value = 0.35
            dens.inputs["From Max"].default_value = 1.0
            # offene See: weiche, windgestreckte Schaumfäden, nie geschlossene Flecken
            foam = lacy_foam(nb, nb.math("MULTIPLY", dens.outputs[0], foam_amount), nb.out(fnoise, "Fac"),
                             foam_lace(nb, wp_t, scale=2.8, width=0.08, stretch=0.35), cap=0.5, soft=(0.12, 0.3))
        else:
            fm = nb.node("ShaderNodeMapRange", clamp=True)
            nb.link(nb.math("MULTIPLY", nb.out(at, "Fac"), nb.math("ADD", nb.out(fnoise, "Fac"), 0.2)),
                    fm.inputs["Value"])
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
        if micro:
            # feine Kräuselung (~20 cm), in Windflecken (Katzenpfoten) stärker
            fine = nb.noise(nb.vmath("ADD", wp_t, nb.vmath("SCALE", drift, scale=1.5)), scale=5.0, detail=2, rough=0.55)
            paw = nb.node("ShaderNodeMapRange", clamp=True)
            nb.link(nb.out(nb.noise(wpos, scale=0.025, detail=2, rough=0.5), "Fac"), paw.inputs["Value"])
            paw.inputs["From Min"].default_value = 0.38
            paw.inputs["From Max"].default_value = 0.62
            paw.inputs["To Min"].default_value = 0.35
            bump = nb.bump(nb.math("MULTIPLY", nb.out(fine, "Fac"), paw.outputs[0]), strength=0.1 * micro,
                           distance=0.06, normal=bump)
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


# --------------------------------------------------------------------------
# Geometry-Nodes: Bugwelle, Kelvin-Kielspur, Rumpf-Kontaktschaum
# --------------------------------------------------------------------------

def ocean_fx_gn(ocean_ob, ship_root, hull_ob, bow_x=16.5, stern_x=-16.0, amp=1.0, hull_halfbeam=5.0):
    """Hängt einen GN-Modifier an den Ozean (nach dem Ocean-Modifier):
    - Displacement in Schiffskoordinaten: Aufstau am Bug, Senke mittschiffs, Kelvin-Arme (19,47°),
      Querwellen hinter dem Heck (animiert über Scene Time)
    - Attribute 'wake' (Kielwasser/Bugwelle) und 'hull_foam' (Kontaktzone per Geometry Proximity)
    """
    ng = bpy.data.node_groups.new("OceanFX", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nb = fpv.NB(ng)
    gi = nb.node("NodeGroupInput")
    go = nb.node("NodeGroupOutput")
    pos = nb.node("GeometryNodeInputPosition").outputs[0]
    tsec = nb.node("GeometryNodeInputSceneTime").outputs["Seconds"]
    # Schiffskoordinaten
    oi = nb.node("GeometryNodeObjectInfo")
    oi.transform_space = "RELATIVE"
    oi.inputs["Object"].default_value = ship_root
    inv = nb.node("FunctionNodeInvertMatrix")
    nb.link(oi.outputs["Transform"], inv.inputs[0])
    tp = nb.node("FunctionNodeTransformPoint")
    nb.link(pos, tp.inputs[0])
    nb.link(inv.outputs[0], tp.inputs[1])
    x, y, _ = nb.sep(tp.outputs[0])
    m = nb.math
    yb = m("ABSOLUTE", y)
    db = m("SUBTRACT", bow_x, x)                       # Abstand hinter dem Bug
    dbp = m("MAXIMUM", db, 0.0)
    # Kelvin-Arme
    dev = m("SUBTRACT", yb, m("MULTIPLY", dbp, 0.354))
    wid = m("MULTIPLY_ADD", dbp, 0.04, 0.6)
    arm = m("EXPONENT", m("DIVIDE", m("MULTIPLY", m("MULTIPLY", dev, dev), -1.0), wid))
    arm = m("MULTIPLY", arm, m("EXPONENT", m("DIVIDE", dbp, -55.0)))
    arm = m("MULTIPLY", arm, m("GREATER_THAN", db, 1.0))
    zarm = m("MULTIPLY", m("MULTIPLY", arm, 0.3 * amp),
             m("SINE", m("SUBTRACT", m("MULTIPLY", dbp, 1.0), m("MULTIPLY", tsec, 3.0))))
    # Aufstau am Bug
    bx = m("SUBTRACT", x, bow_x - 1.0)
    bow = m("EXPONENT", m("MULTIPLY", m("ADD", m("DIVIDE", m("MULTIPLY", bx, bx), 5.0),
                                         m("DIVIDE", m("MULTIPLY", y, y), 7.0)), -1.0))
    # Senke mittschiffs
    dy = m("SUBTRACT", yb, 5.4)
    tr = m("MULTIPLY", m("EXPONENT", m("DIVIDE", m("MULTIPLY", dy, dy), -3.0)),
           m("EXPONENT", m("DIVIDE", m("MULTIPLY", x, x), -140.0)))
    # Heck: Querwellen + turbulente Spur
    xs = m("SUBTRACT", stern_x, x)
    xsp = m("MAXIMUM", xs, 0.0)
    spread = m("MULTIPLY_ADD", xsp, 0.25, 3.0)
    gauss_y = m("EXPONENT", m("DIVIDE", m("MULTIPLY", y, y), m("MULTIPLY", m("MULTIPLY", spread, spread), -2.0)))
    ztr = m("MULTIPLY", m("MULTIPLY", m("SINE", m("SUBTRACT", m("MULTIPLY", xsp, 0.8), m("MULTIPLY", tsec, 2.5))),
                          m("EXPONENT", m("DIVIDE", xsp, -35.0))), m("MULTIPLY", gauss_y, 0.12 * amp))
    ztr = m("MULTIPLY", ztr, m("GREATER_THAN", xs, 0.0))
    core = m("MAXIMUM", m("SUBTRACT", 1.0, m("DIVIDE", yb, m("MULTIPLY_ADD", xsp, 0.09, 3.2))), 0.0)
    core = m("MULTIPLY", core, m("MULTIPLY", m("EXPONENT", m("DIVIDE", xsp, -45.0)), m("GREATER_THAN", xs, -2.0)))
    dz = m("ADD", m("ADD", zarm, m("MULTIPLY", bow, 0.55 * amp)), m("ADD", m("MULTIPLY", tr, -0.25 * amp), ztr))
    sp = nb.node("GeometryNodeSetPosition")
    nb.link(gi.outputs[0], sp.inputs["Geometry"])
    nb.link(nb.comb(0.0, 0.0, dz), sp.inputs["Offset"])
    arm_foam = m("MULTIPLY", arm, m("MULTIPLY", m("EXPONENT", m("DIVIDE", dbp, -14.0)), 0.9))
    wake = m("MAXIMUM", m("MAXIMUM", arm_foam, m("MULTIPLY", core, 0.8)), m("MULTIPLY", bow, 0.9))
    st1 = nb.node("GeometryNodeStoreNamedAttribute")
    st1.data_type = "FLOAT"
    st1.domain = "POINT"
    nb.link(sp.outputs[0], st1.inputs["Geometry"])
    st1.inputs["Name"].default_value = "wake"
    nb.link(wake, st1.inputs["Value"])
    # Kontaktschaum am Rumpf: analytische Wasserlinien-Kontur in Schiffskoordinaten
    # (Superellipse |x/a|^p + |y/b(x)|^2; billiger als Geometry Proximity bei Motion-Blur-Neuauswertung)
    half_len = (bow_x - stern_x) / 2 + 0.3
    xc = m("SUBTRACT", x, (bow_x + stern_x) / 2)
    ex = m("POWER", m("ABSOLUTE", m("DIVIDE", xc, half_len)), 2.6)
    ey = m("POWER", m("DIVIDE", yb, hull_halfbeam), 2.0)
    dd = m("SQRT", m("ADD", ex, ey))
    hf = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(dd, hf.inputs["Value"])
    hf.inputs["From Min"].default_value = 1.32
    hf.inputs["From Max"].default_value = 1.0
    st2 = nb.node("GeometryNodeStoreNamedAttribute")
    st2.data_type = "FLOAT"
    st2.domain = "POINT"
    nb.link(st1.outputs[0], st2.inputs["Geometry"])
    st2.inputs["Name"].default_value = "hull_foam"
    nb.link(hf.outputs[0], st2.inputs["Value"])
    nb.link(st2.outputs[0], go.inputs[0])
    mod = ocean_ob.modifiers.new("OceanFX", "NODES")
    mod.node_group = ng
    return mod


def shore_distance_image(rocks, xmin, ymin, xmax, ymax, cell=0.5, max_d=12.0, name="ShoreDist"):
    """Abstandsfeld (Meter) von jedem Wasserpunkt zur Fels-Wasserlinie, als Float-Bild.
    rocks: Liste Blender-Objekte (Mesh, bereits positioniert)."""
    nx = int((xmax - xmin) / cell)
    ny = int((ymax - ymin) / cell)
    D = np.full((ny, nx), max_d, np.float32)
    xs = xmin + (np.arange(nx) + 0.5) * cell
    ys = ymin + (np.arange(ny) + 0.5) * cell
    bpy.context.view_layer.update()
    for ob in rocks:
        mw = ob.matrix_world
        co = np.array([mw @ v.co for v in ob.data.vertices])
        ring = co[np.abs(co[:, 2] - 0.3) < 1.2][:, :2]
        if len(ring) == 0:
            continue
        rx0, ry0 = ring.min(0) - max_d
        rx1, ry1 = ring.max(0) + max_d
        ix = np.where((xs >= rx0) & (xs <= rx1))[0]
        iy = np.where((ys >= ry0) & (ys <= ry1))[0]
        if len(ix) == 0 or len(iy) == 0:
            continue
        GX, GY = np.meshgrid(xs[ix], ys[iy])
        P = np.stack([GX.ravel(), GY.ravel()], 1)
        best = np.full(len(P), max_d, np.float32)
        for k in range(0, len(ring), 256):
            d = np.sqrt(((P[:, None, :] - ring[None, k:k + 256, :]) ** 2).sum(-1)).min(1)
            best = np.minimum(best, d)
        sub = D[np.ix_(iy, ix)]
        D[np.ix_(iy, ix)] = np.minimum(sub, best.reshape(len(iy), len(ix)))
    # als 16-Bit-PNG speichern und laden (generierte Float-Bilder werden im Hintergrundmodus nicht übernommen)
    import os
    from PIL import Image as PILImage
    path = os.path.join(textures.OUT, f"{name}.png")
    arr = (np.clip(D / max_d, 0, 1) * 65535).astype(np.uint16)[::-1]  # Blender: Zeile 0 = unten
    PILImage.fromarray(arr, mode="I;16").save(path)
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    return img, (xmin, ymin, xmax - xmin, ymax - ymin)
