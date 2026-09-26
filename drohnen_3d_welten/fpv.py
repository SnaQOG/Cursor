"""Gemeinsame Bausteine für die NIDO 3D-Welten (Blender 5.0, Cycles).

- Render-Setup (Cycles CPU/GPU, Multilayer-EXR mit Combined/Env/Mist)
- Himmel mit Sonne + prozeduraler Wolkenschicht
- FPV-Kamerafahrt: konstantes Tempo entlang der Route, koordinierte
  Schräglage aus der Kurvenkrümmung, kleine Piloten- und Mikrokorrekturen
- Material- und Geometrie-Helfer
"""
import math
import os
import random

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.environ.get("NIDO_ASSETS", os.path.join(HERE, "assets"))


# --------------------------------------------------------------------------
# Szene / Rendering
# --------------------------------------------------------------------------

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    return sc


def setup_render(outdir, res=(1080, 1920), fps=30, seconds=20, samples=32,
                 motion_blur=True, shutter=0.5, mist_depth=6000.0, seed=7):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.device = "CPU"
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.04
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.denoising_prefilter = "ACCURATE"
    cy.seed = seed
    cy.use_animated_seed = False
    cy.max_bounces = 4
    cy.diffuse_bounces = 2
    cy.glossy_bounces = 2
    cy.transmission_bounces = 6
    cy.volume_bounces = 0
    cy.transparent_max_bounces = 8
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.sample_clamp_indirect = 6.0
    cy.blur_glossy = 1.0
    cy.use_light_tree = True
    sc.render.use_persistent_data = True

    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.fps = fps
    sc.frame_start = 1
    sc.frame_end = int(round(fps * seconds))
    sc.render.use_motion_blur = motion_blur
    sc.render.motion_blur_shutter = shutter
    sc.render.film_transparent = True
    cy.pixel_filter_type = "BLACKMAN_HARRIS"
    cy.filter_width = 1.5

    vl = sc.view_layers[0]
    vl.use_pass_combined = True
    vl.use_pass_mist = True
    vl.use_pass_environment = True

    world = sc.world or bpy.data.worlds.new("World")
    sc.world = world
    world.mist_settings.start = 0.0
    world.mist_settings.depth = mist_depth
    world.mist_settings.falloff = "LINEAR"

    # Passes über den Compositor als einzelne EXR-Dateien schreiben
    # (Blender 5.0 schreibt beim Multilayer-Save nur Combined).
    os.makedirs(outdir, exist_ok=True)
    ng = bpy.data.node_groups.new("NIDO_Passes", "CompositorNodeTree")
    sc.compositing_node_group = ng
    sc.render.use_compositing = True
    sc.render.use_sequencer = False
    rl = ng.nodes.new("CompositorNodeRLayers")
    fo = ng.nodes.new("CompositorNodeOutputFile")
    fo.directory = outdir.rstrip("/") + "/"
    fo.file_name = "f_####"
    fo.format.media_type = "IMAGE"
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "16"
    fo.format.exr_codec = "DWAA"
    for nm, st, src in (("Image", "RGBA", "Image"), ("Mist", "FLOAT", "Mist"), ("Env", "RGBA", "Environment")):
        fo.file_output_items.new(st, nm)
        ng.links.new(rl.outputs[src], fo.inputs[nm])
    sc.render.filepath = os.path.join(outdir, "_unused_")
    return sc


def try_gpu():
    """Nutzt Metal/OptiX/CUDA falls vorhanden (z. B. auf dem Mac), sonst CPU."""
    prefs = bpy.context.preferences
    if "cycles" not in prefs.addons:
        return False
    cp = prefs.addons["cycles"].preferences
    for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            cp.compute_device_type = backend
            cp.get_devices()
            devs = [d for d in cp.devices if d.type == backend]
            if devs:
                for d in cp.devices:
                    d.use = d.type == backend
                bpy.context.scene.cycles.device = "GPU"
                print("GPU backend:", backend)
                return True
        except Exception:
            continue
    return False


# --------------------------------------------------------------------------
# Node-Helfer
# --------------------------------------------------------------------------

class NB:
    """Kleiner Builder für Shader-Nodebäume."""

    def __init__(self, tree):
        self.t = tree
        self.n = tree.nodes
        self.l = tree.links

    def node(self, kind, **props):
        nd = self.n.new(kind)
        for k, v in props.items():
            if k.startswith("in_"):
                key = k[3:]
                self.set_in(nd, key, v)
            else:
                setattr(nd, k, v)
        return nd

    @staticmethod
    def _sock(nd, key, outputs):
        socks = nd.outputs if outputs else nd.inputs
        if isinstance(key, int):
            return socks[key]
        if key in socks:
            # erster *aktiver* Socket dieses Namens (Mix-Node hat Dubletten)
            for s in socks:
                if s.name == key and s.enabled:
                    return s
            return socks[key]
        return socks[key.replace("_", " ")]

    def set_in(self, nd, key, v):
        s = self._sock(nd, key, False)
        if isinstance(v, bpy.types.NodeSocket):
            self.l.new(v, s)
        elif isinstance(v, bpy.types.Node):
            self.l.new(v.outputs[0], s)
        else:
            if s.type == "RGBA" and isinstance(v, (tuple, list)) and len(v) == 3:
                v = (*v, 1.0)
            s.default_value = v

    def out(self, nd, key=0):
        return self._sock(nd, key, True)

    def link(self, a, b):
        self.l.new(a, b)

    # Mathe
    def math(self, op, a, b=None, c=None, clamp=False):
        nd = self.n.new("ShaderNodeMath")
        nd.operation = op
        nd.use_clamp = clamp
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (bpy.types.NodeSocket, bpy.types.Node)):
                self.l.new(v if isinstance(v, bpy.types.NodeSocket) else v.outputs[0], nd.inputs[i])
            else:
                nd.inputs[i].default_value = v
        return nd.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        nd = self.n.new("ShaderNodeVectorMath")
        nd.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (bpy.types.NodeSocket, bpy.types.Node)):
                self.l.new(v if isinstance(v, bpy.types.NodeSocket) else v.outputs[0], nd.inputs[i])
            else:
                nd.inputs[i].default_value = v
        if scale is not None:
            if isinstance(scale, bpy.types.NodeSocket):
                self.l.new(scale, nd.inputs["Scale"])
            else:
                nd.inputs["Scale"].default_value = scale
        return nd.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else nd.outputs[0]

    def mix(self, fac, a, b, blend="MIX", dtype="RGBA", clamp=False):
        nd = self.n.new("ShaderNodeMix")
        nd.data_type = dtype
        if dtype == "RGBA":
            nd.blend_type = blend
            nd.clamp_result = clamp
        suf = {"RGBA": "Color", "FLOAT": "Float", "VECTOR": "Vector"}[dtype]
        byid = {s.identifier: s for s in nd.inputs}
        targets = [byid["Factor_Float"], byid["A_" + suf], byid["B_" + suf]]
        for k, (s, v) in enumerate(zip(targets, (fac, a, b))):
            if isinstance(v, (bpy.types.NodeSocket, bpy.types.Node)):
                self.l.new(v if isinstance(v, bpy.types.NodeSocket) else v.outputs[0], s)
            else:
                if k > 0 and dtype == "RGBA":
                    if isinstance(v, (int, float)):
                        v = (v, v, v, 1.0)
                    elif len(v) == 3:
                        v = (*v, 1.0)
                s.default_value = v
        return {o.identifier: o for o in nd.outputs}["Result_" + suf]

    def ramp(self, fac, stops, interp="LINEAR"):
        nd = self.n.new("ShaderNodeValToRGB")
        cr = nd.color_ramp
        cr.interpolation = interp
        while len(cr.elements) > 1:
            cr.elements.remove(cr.elements[-1])
        for i, (pos, col) in enumerate(stops):
            e = cr.elements[0] if i == 0 else cr.elements.new(pos)
            e.position = pos
            e.color = col if len(col) == 4 else (*col, 1.0)
        self.link(fac if isinstance(fac, bpy.types.NodeSocket) else fac.outputs[0], nd.inputs[0])
        return nd.outputs[0]

    def noise(self, vec=None, scale=5.0, detail=6.0, rough=0.55, lac=2.0, dist=0.0, dims="3D",
              ntype="FBM", w=None, normalize=True):
        nd = self.n.new("ShaderNodeTexNoise")
        nd.noise_dimensions = dims
        nd.noise_type = ntype
        nd.normalize = normalize
        if vec is not None:
            self.link(vec, nd.inputs["Vector"])
        nd.inputs["Scale"].default_value = scale
        nd.inputs["Detail"].default_value = detail
        nd.inputs["Roughness"].default_value = rough
        nd.inputs["Lacunarity"].default_value = lac
        nd.inputs["Distortion"].default_value = dist
        if w is not None and dims in ("4D", "1D"):
            nd.inputs["W"].default_value = w
        return nd

    def voronoi(self, vec=None, scale=5.0, feature="F1", metric="EUCLIDEAN", rand=1.0, detail=0.0, dims="3D"):
        nd = self.n.new("ShaderNodeTexVoronoi")
        nd.voronoi_dimensions = dims
        nd.feature = feature
        nd.distance = metric
        if vec is not None:
            self.link(vec, nd.inputs["Vector"])
        nd.inputs["Scale"].default_value = scale
        nd.inputs["Randomness"].default_value = rand
        if "Detail" in nd.inputs:
            nd.inputs["Detail"].default_value = detail
        return nd

    def coords(self, kind="Object", obj=None):
        nd = self.n.new("ShaderNodeTexCoord")
        if obj is not None:
            nd.object = obj
        return nd.outputs[kind]

    def mapping(self, vec, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        nd = self.n.new("ShaderNodeMapping")
        self.link(vec, nd.inputs["Vector"])
        nd.inputs["Location"].default_value = loc
        nd.inputs["Rotation"].default_value = rot
        nd.inputs["Scale"].default_value = scale
        return nd.outputs[0]

    def sep(self, vec):
        nd = self.n.new("ShaderNodeSeparateXYZ")
        self.link(vec, nd.inputs[0])
        return nd.outputs

    def comb(self, x, y, z):
        nd = self.n.new("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            if isinstance(v, bpy.types.NodeSocket):
                self.link(v, nd.inputs[i])
            else:
                nd.inputs[i].default_value = v
        return nd.outputs[0]

    def image(self, path, vec=None, colorspace="sRGB", proj="FLAT", blend=0.2, interp="Linear"):
        nd = self.n.new("ShaderNodeTexImage")
        img = bpy.data.images.load(path, check_existing=True)
        img.colorspace_settings.name = colorspace
        nd.image = img
        nd.projection = proj
        nd.projection_blend = blend
        nd.interpolation = interp
        if vec is not None:
            self.link(vec, nd.inputs["Vector"])
        return nd

    def bump(self, height, strength=0.5, distance=0.1, normal=None):
        nd = self.n.new("ShaderNodeBump")
        nd.inputs["Strength"].default_value = strength
        nd.inputs["Distance"].default_value = distance
        self.link(height, nd.inputs["Height"])
        if normal is not None:
            self.link(normal, nd.inputs["Normal"])
        return nd.outputs[0]


def new_material(name):
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nb = NB(nt)
    return mat, nb, out


def principled(nb, **kw):
    p = nb.n.new("ShaderNodeBsdfPrincipled")
    for k, v in kw.items():
        nb.set_in(p, k.replace("_", " "), v)
    return p


def simple_mat(name, color, rough=0.5, metal=0.0, emission=None, estrength=0.0, spec=0.5):
    mat, nb, out = new_material(name)
    p = principled(nb, Base_Color=(*color, 1.0) if len(color) == 3 else color, Roughness=rough,
                   Metallic=metal)
    p.inputs["Specular IOR Level"].default_value = spec
    if emission is not None:
        p.inputs["Emission Color"].default_value = (*emission, 1.0)
        p.inputs["Emission Strength"].default_value = estrength
    nb.link(p.outputs[0], out.inputs[0])
    return mat


# --------------------------------------------------------------------------
# Himmel, Sonne, Wolken
# --------------------------------------------------------------------------

def sun_dir(elev_deg, azim_deg):
    """Richtung *zur* Sonne. Azimut 0° = +Y (Norden), 90° = +X (Osten)."""
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return Vector((math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)))


def add_sun(elev_deg, azim_deg, strength=4.0, color=(1.0, 0.95, 0.88), angle_deg=0.55, name="Sun"):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = strength
    ld.color = color
    ld.angle = math.radians(angle_deg)
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    d = sun_dir(elev_deg, azim_deg)
    # Sonnenlicht scheint entlang -Z des Objekts
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (-d).to_track_quat("-Z", "Y")
    return ob


def build_world(sun_elev=15, sun_azim=200, sky_strength=1.0, clouds=True, cloud_cover=0.45,
                cloud_scale=1.0, cloud_height=2000.0, cloud_color=(1.0, 0.97, 0.93),
                air=1.0, aerosol=1.2, ozone=1.0, altitude=50.0, tint=None, custom_sky=None,
                extra_suns=(), cloud_opacity=0.95, cloud_offset=(0.0, 0.0), debug=None, cloud_ref=9.0):
    """Physikalischer Himmel (Multiple Scattering) + 2D-Wolkenschicht.

    custom_sky: optional Funktion(nb, dir_socket) -> color socket (z. B. Namek-Himmel).
    extra_suns: Liste (elev, azim, radius_deg, strength, color) sichtbarer Sonnenscheiben.
    """
    sc = bpy.context.scene
    w = sc.world
    nt = w.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    nb = NB(nt)
    out = nb.node("ShaderNodeOutputWorld")
    bg = nb.node("ShaderNodeBackground")
    nb.link(bg.outputs[0], out.inputs["Surface"])

    dir_ = nb.coords("Generated")  # Weltrichtung
    dirn = nb.vmath("NORMALIZE", dir_)

    if custom_sky is None:
        sky = nb.node("ShaderNodeTexSky")
        sky.sky_type = "MULTIPLE_SCATTERING"
        sky.sun_disc = False
        sky.sun_elevation = math.radians(sun_elev)
        # Sky-Rotation: 0 = Sonne in +Y? Blender: rotation um Z, 0 => +Y? wir kalibrieren
        sky.sun_rotation = math.radians(sun_azim)
        sky.altitude = altitude
        sky.air_density = air
        sky.aerosol_density = aerosol
        sky.ozone_density = ozone
        sky_col = sky.outputs[0]
        if tint is not None:
            sky_col = nb.mix(1.0, sky_col, tint, blend="MULTIPLY")
    else:
        sky_col = custom_sky(nb, dirn)

    col = sky_col
    if clouds:
        x, y, z = nb.sep(dirn)
        zc = nb.math("MAXIMUM", z, 0.015)
        # Projektion auf Wolkenebene
        px = nb.math("MULTIPLY", nb.math("DIVIDE", x, zc), cloud_height)
        py = nb.math("MULTIPLY", nb.math("DIVIDE", y, zc), cloud_height)
        px = nb.math("ADD", px, cloud_offset[0])
        py = nb.math("ADD", py, cloud_offset[1])
        p = nb.comb(px, py, 0.0)
        p = nb.vmath("SCALE", p, scale=1.0 / (3000.0 * cloud_scale))
        warp = nb.noise(p, scale=1.3, detail=2, rough=0.5)
        p2 = nb.vmath("ADD", p, nb.vmath("SCALE", nb.out(warp, "Color"), scale=0.35))
        n1 = nb.noise(p2, scale=2.2, detail=6, rough=0.62, lac=2.1)
        n2 = nb.noise(p2, scale=7.0, detail=3, rough=0.55)
        base = nb.math("ADD", nb.out(n1, "Fac"), nb.math("MULTIPLY", nb.math("SUBTRACT", nb.out(n2, "Fac"), 0.5), 0.35))
        lo = 0.70 - 0.36 * cloud_cover
        mr = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(base, mr.inputs["Value"])
        mr.inputs["From Min"].default_value = lo
        mr.inputs["From Max"].default_value = lo + 0.16
        dens = mr.outputs[0]
        # Horizont-Ausblendung
        hz = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(z, hz.inputs["Value"])
        hz.inputs["From Min"].default_value = 0.01
        hz.inputs["From Max"].default_value = 0.16
        dens = nb.math("MULTIPLY", dens, hz.outputs[0])
        dens = nb.math("MULTIPLY", dens, cloud_opacity)
        # Selbstschattierung: Dichte in Sonnenrichtung verschoben
        sd = sun_dir(sun_elev, sun_azim)
        shift = nb.vmath("ADD", p2, (sd.x * 0.03, sd.y * 0.03, 0.0))
        n3 = nb.noise(shift, scale=2.2, detail=4, rough=0.62, lac=2.1)
        shade = nb.math("SUBTRACT", nb.out(n3, "Fac"), nb.out(n1, "Fac"))
        shade = nb.math("MULTIPLY_ADD", shade, -4.0, 0.75)  # heller auf Sonnenseite
        shade = nb.math("MAXIMUM", nb.math("MINIMUM", shade, 1.25), 0.35)
        # Silberrand Richtung Sonne
        sdot = nb.vmath("DOT_PRODUCT", dirn, tuple(sd))
        glow = nb.math("POWER", nb.math("MAXIMUM", sdot, 0.0), 12.0)
        lit = nb.math("MULTIPLY_ADD", glow, 1.5, shade)
        # Wolkenfarbe: Sonnenfarbe * lit + Himmel-Ambient
        # Einheiten wie der Himmel (cloud_ref ~ Himmelsleuchtdichte am Horizont)
        amb = nb.mix(0.5, sky_col, (0.62 * cloud_ref * 0.5,) * 3, blend="MIX")
        suncol = nb.node("ShaderNodeRGB")
        suncol.outputs[0].default_value = (*cloud_color, 1.0)
        litc = nb.vmath("SCALE", suncol.outputs[0], scale=lit)
        cc = nb.vmath("ADD", nb.vmath("SCALE", litc, scale=0.85 * cloud_ref), nb.vmath("SCALE", amb, scale=0.35))
        col = nb.mix(dens, sky_col, cc)
        dbg = {"dens": dens, "base": base, "cc": cc, "lit": lit, "sky": sky_col}
        if debug in dbg:
            col = dbg[debug]
    for (e, a, rad, stren, c) in extra_suns:
        sd = sun_dir(e, a)
        d = nb.vmath("DOT_PRODUCT", dirn, tuple(sd))
        cosr = math.cos(math.radians(rad))
        mr = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(d, mr.inputs["Value"])
        mr.inputs["From Min"].default_value = cosr - (1 - cosr) * 0.25
        mr.inputs["From Max"].default_value = cosr + (1 - cosr) * 0.05
        halo_mr = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(d, halo_mr.inputs["Value"])
        halo_mr.inputs["From Min"].default_value = math.cos(math.radians(rad * 10))
        halo_mr.inputs["From Max"].default_value = 1.0
        halo = nb.math("POWER", halo_mr.outputs[0], 3.0)
        disc = nb.math("ADD", nb.math("MULTIPLY", mr.outputs[0], stren), nb.math("MULTIPLY", halo, stren * 0.004))
        col = nb.vmath("ADD", col, nb.vmath("SCALE", c, scale=disc))
    nb.link(col, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = sky_strength
    return nb


# --------------------------------------------------------------------------
# FPV-Kamera
# --------------------------------------------------------------------------

def _catmull_rom(points, samples_per_seg=400, alpha=0.5):
    P = [np.array(p, dtype=float) for p in points]
    P = [P[0] + (P[0] - P[1])] + P + [P[-1] + (P[-1] - P[-2])]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]

        def tj(ti, a, b):
            return ti + max(np.linalg.norm(b - a), 1e-6) ** alpha

        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        ts = np.linspace(t1, t2, samples_per_seg, endpoint=False)
        for t in ts:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
            out.append(c)
    out.append(P[-2])
    return np.array(out)


def _gauss_smooth(x, sigma):
    if sigma <= 0:
        return x
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, (r, r), mode="edge")
    return np.convolve(pad, k, mode="valid")


def _smooth_noise(n, fps, freqs_amps, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    s = np.zeros(n)
    for f, a in freqs_amps:
        ph = rng.uniform(0, 2 * np.pi, 3)
        # Summe leicht verstimmter Sinus = organisches Rauschen
        s += a * (0.6 * np.sin(2 * np.pi * f * t + ph[0]) + 0.3 * np.sin(2 * np.pi * f * 1.37 * t + ph[1])
                  + 0.2 * np.sin(2 * np.pi * f * 0.71 * t + ph[2]))
    return s


def fpv_path(points, speed, fps, frames, start_offset=0.0, look_pitch=-4.0, pitch_follow=0.55,
             bank_gain=1.0, max_bank=38.0, bank_smooth_s=0.35, yaw_smooth_s=0.25,
             micro=1.0, seed=3, pitch_overrides=None, yaw_lead_s=0.0, roll_bias=None):
    """Berechnet eine FPV-Kamerafahrt mit konstantem Bahn-Tempo.

    Rückgabe: (positions[n,3], quats[n,4], info dict)
    pitch_overrides: Liste (t0, t1, delta_deg) weicher Blickneigungen.
    """
    curve = _catmull_rom(points)
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s_acc = np.concatenate([[0], np.cumsum(seg)])
    total = s_acc[-1]
    # Oversampling für Motion-Blur: Schlüssel pro Frame, Blender interpoliert.
    n = frames + 2
    t = (np.arange(n) - 1) / fps
    s = start_offset + speed * t
    if s[-1] > total:
        raise ValueError(f"Route zu kurz: {total:.1f} m, benötigt {s[-1]:.1f} m")
    pos = np.stack([np.interp(s, s_acc, curve[:, k]) for k in range(3)], axis=1)
    # vor dem Routenstart linear extrapolieren (Frame 0 für Motion-Blur)
    if s[0] < 0:
        d0 = (curve[1] - curve[0]) / max(np.linalg.norm(curve[1] - curve[0]), 1e-6)
        neg = s < 0
        pos[neg] = curve[0] + np.outer(s[neg], d0)

    # Tangente fein abgetastet
    ds = 0.5
    sc_ = np.clip(s, ds, total - ds)
    fwd = np.stack([np.interp(sc_ + ds, s_acc, curve[:, k]) - np.interp(sc_ - ds, s_acc, curve[:, k])
                    for k in range(3)], axis=1)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    pitch_path = np.arcsin(np.clip(fwd[:, 2], -1, 1))

    # Krümmung -> koordinierte Schräglage
    dyaw = np.gradient(yaw) * fps  # rad/s
    lat = speed * dyaw  # m/s^2
    bank = np.arctan2(lat, 9.81) * bank_gain
    bank = _gauss_smooth(bank, bank_smooth_s * fps)
    bank = np.clip(bank, -math.radians(max_bank), math.radians(max_bank))
    yaw_s = _gauss_smooth(yaw, yaw_smooth_s * fps)
    if yaw_lead_s:
        lead = int(yaw_lead_s * fps)
        yaw_s = np.concatenate([yaw_s[lead:], np.full(lead, yaw_s[-1]) + (yaw_s[-1] - yaw_s[-2]) * np.arange(1, lead + 1)])
    pitch = pitch_path * pitch_follow + math.radians(look_pitch)
    pitch = _gauss_smooth(pitch, 0.3 * fps)
    if pitch_overrides:
        add = np.zeros(n)
        for (t0, t1, dd) in pitch_overrides:
            # weiche Rampe (smoothstep) rein und raus
            ramp_len = min(1.0, (t1 - t0) / 3)
            up = np.clip((t - t0) / ramp_len, 0, 1)
            down = np.clip((t1 - t) / ramp_len, 0, 1)
            w = np.minimum(up, down)
            w = w * w * (3 - 2 * w)
            add += math.radians(dd) * w
        pitch += add

    # Piloten- und Mikrokorrekturen (Grad)
    m = micro
    roll_n = _smooth_noise(n, fps, [(0.35, 0.55 * m), (1.1, 0.25 * m), (6.5, 0.07 * m), (11.0, 0.035 * m)], seed)
    pitch_n = _smooth_noise(n, fps, [(0.3, 0.6 * m), (1.3, 0.25 * m), (7.0, 0.08 * m), (13.0, 0.04 * m)], seed + 1)
    yaw_n = _smooth_noise(n, fps, [(0.25, 0.7 * m), (1.2, 0.25 * m), (7.5, 0.07 * m)], seed + 2)
    pos_n = np.stack([_smooth_noise(n, fps, [(0.4, 0.10 * m), (1.7, 0.03 * m), (9.0, 0.006 * m)], seed + 10 + k)
                      for k in range(3)], axis=1)
    roll = bank + np.radians(roll_n)
    if roll_bias is not None:
        roll += np.radians(roll_bias)
    pitch = pitch + np.radians(pitch_n)
    yaw_f = yaw_s + np.radians(yaw_n)
    pos = pos + pos_n

    quats = []
    prev = None
    for i in range(n):
        f = Vector((math.cos(pitch[i]) * math.cos(yaw_f[i]), math.cos(pitch[i]) * math.sin(yaw_f[i]), math.sin(pitch[i])))
        q = f.to_track_quat("-Z", "Y")
        # Rollen um die Blickachse (positive Kurve links => Schräglage links)
        qr = Quaternion(f, -roll[i])
        q = qr @ q
        if prev is not None and prev.dot(q) < 0:
            q.negate()
        prev = q
        quats.append(q)
    info = dict(total=total, used=s[-1] - s[0], yaw=yaw_f, bank=bank, pitch=pitch, s=s, t=t)
    return pos, quats, info


def make_camera(pos, quats, fov_deg=92.0, name="FPVCam", clip_end=30000.0):
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = 24.0
    cd.lens_unit = "FOV"
    cd.angle = math.radians(fov_deg)
    cd.clip_start = 0.05
    cd.clip_end = clip_end
    cam = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_mode = "QUATERNION"
    for i in range(len(pos)):
        fr = i  # Index 0 entspricht Frame 0 (vor dem ersten Renderframe)
        cam.location = Vector(pos[i])
        cam.rotation_quaternion = quats[i]
        cam.keyframe_insert("location", frame=fr)
        cam.keyframe_insert("rotation_quaternion", frame=fr)
    return cam


# --------------------------------------------------------------------------
# Geometrie-Helfer
# --------------------------------------------------------------------------

def link(ob, coll=None):
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def mesh_from_bmesh(bm, name, mat=None, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    if mat is not None:
        me.materials.append(mat)
    return link(ob)


def grid_mesh(name, size_x, size_y, nx, ny, heightfn=None, mat=None, origin=(0, 0)):
    xs = np.linspace(-size_x / 2, size_x / 2, nx) + origin[0]
    ys = np.linspace(-size_y / 2, size_y / 2, ny) + origin[1]
    X, Y = np.meshgrid(xs, ys)
    Z = heightfn(X, Y) if heightfn else np.zeros_like(X)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    idx = np.arange(nx * ny).reshape(ny, nx)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[1:, :-1].ravel()
    faces = np.stack([a, b, c, d], axis=1)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(len(faces) * 4)
    me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", (np.arange(len(faces)) * 4).astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    # UVs
    uv = me.uv_layers.new(name="UVMap")
    u = ((X - xs[0]) / (xs[-1] - xs[0])).ravel()
    v = ((Y - ys[0]) / (ys[-1] - ys[0])).ravel()
    luv = np.stack([u[faces.ravel()], v[faces.ravel()]], axis=1)
    uv.data.foreach_set("uv", luv.astype(np.float32).ravel())
    me.update()
    me.validate()
    ob = bpy.data.objects.new(name, me)
    if mat is not None:
        me.materials.append(mat)
    return link(ob)


def fbm2(x, y, octaves=6, seed=0, lac=2.0, gain=0.5, scale=1.0):
    """Wertrauschen (numpy) für Heightmaps."""
    rng = np.random.default_rng(seed)
    total = np.zeros_like(x, dtype=float)
    amp = 1.0
    freq = scale
    norm = 0.0
    for o in range(octaves):
        total += amp * _value_noise(x * freq, y * freq, rng.integers(0, 1 << 30))
        norm += amp
        amp *= gain
        freq *= lac
    return total / norm


def _value_noise(x, y, seed):
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi

    def h(i, j):
        v = (i * 374761393 + j * 668265263 + seed * 1442695041) & 0xFFFFFFFF
        v = (v ^ (v >> 13)) * 1274126177 & 0xFFFFFFFF
        v = v ^ (v >> 16)
        return (v & 0xFFFF) / 65535.0 * 2 - 1

    u = xf * xf * (3 - 2 * xf)
    v = yf * yf * (3 - 2 * yf)
    a = h(xi, yi)
    b = h(xi + 1, yi)
    c = h(xi, yi + 1)
    d = h(xi + 1, yi + 1)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


def displace_obj(ob, tex_type="CLOUDS", size=1.0, strength=1.0, depth=4, midlevel=0.5,
                 noise_basis="BLENDER_ORIGINAL", subdiv=0, direction="NORMAL", coords="LOCAL", name=None):
    if subdiv:
        m = ob.modifiers.new("sub", "SUBSURF")
        m.levels = subdiv
        m.render_levels = subdiv
        m.subdivision_type = "SIMPLE"
    tex = bpy.data.textures.new(name or f"tx_{ob.name}_{len(ob.modifiers)}", tex_type)
    if tex_type == "CLOUDS":
        tex.noise_scale = size
        tex.noise_depth = depth
        tex.noise_basis = noise_basis
        tex.noise_type = "SOFT_NOISE"
    elif tex_type == "VORONOI":
        tex.noise_scale = size
        tex.distance_metric = "DISTANCE"
    elif tex_type == "MUSGRAVE":
        tex.noise_scale = size
    m = ob.modifiers.new("disp", "DISPLACE")
    m.texture = tex
    m.strength = strength
    m.mid_level = midlevel
    m.direction = direction
    m.texture_coords = coords
    return m


def apply_modifiers(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    ob.modifiers.clear()
    old = ob.data
    ob.data = me
    bpy.data.meshes.remove(old)
    return ob


def rock_mesh(name, radius=10.0, height=30.0, seed=1, detail=5, stretch=(1, 1, 1), strata=0.0,
              taper=0.3, mat=None, base_z=-4.0, lumpy=0.25):
    """Felsnadel/Seestack aus verformtem Zylinder mit Rauschen (numpy)."""
    rng = np.random.default_rng(seed)
    nr = 96 * (detail - 2)
    nz = int(max(24, height / (radius * 2 * math.pi / nr)))
    nz = min(nz, 400)
    th = np.linspace(0, 2 * math.pi, nr, endpoint=False)
    zs = np.linspace(0, 1, nz)
    TH, ZZ = np.meshgrid(th, zs)
    # Grundradius mit Verjüngung und unregelmäßigem Profil
    prof = 1.0 - taper * ZZ ** 1.5
    prof *= 1 + 0.12 * np.sin(ZZ * 7 + seed) + 0.06 * np.sin(ZZ * 19 + seed * 2)
    X0 = np.cos(TH)
    Y0 = np.sin(TH)
    ph = rng.uniform(0, 1000, 4)
    zw = ZZ * height  # Meter entlang der Höhe
    n_big = fbm2(X0 * 1.2 + ph[0] + ZZ * 0.8, Y0 * 1.2 + ZZ * height / radius * 0.3, octaves=4, seed=seed)
    # senkrechte Karstrinnen (hohe Frequenz um den Umfang, niedrige in der Höhe)
    a_f = max(6.0, radius * 1.1)
    flutes = fbm2(X0 * a_f + ph[1] + zw * 0.025, Y0 * a_f + zw * 0.01, octaves=4, seed=seed + 5)
    mid = fbm2(X0 * radius / 4.0 + ph[2], Y0 * radius / 4.0 + zw / 4.0, octaves=5, seed=seed + 7)
    fine = fbm2(X0 * radius / 1.2 + ph[3], Y0 * radius / 1.2 + zw / 1.2, octaves=3, seed=seed + 8)
    r = radius * prof * (1 + lumpy * n_big + 0.07 * flutes + 0.07 * mid + 0.02 * fine)
    if strata:
        r *= 1 + strata * 0.012 * np.sin(zw * 0.8 + 3 * fbm2(X0 * 2, Y0 * 2 + zw * 0.05, 2, seed + 9))
    # Brandungskehle an der Wasserlinie (Welt-z ~ 0.3 .. 2)
    zworld = base_z + zw
    r *= 1 - 0.10 * np.exp(-((zworld - 0.9) / 1.3) ** 2)
    X = X0 * r * stretch[0]
    Y = Y0 * r * stretch[1]
    Z = base_z + ZZ * height * stretch[2]
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    # Deckel: Kuppel oben
    top_center = np.array([[0, 0, base_z + height * stretch[2] + radius * prof[-1, 0] * 0.25]])
    verts = np.vstack([verts, top_center])
    idx = np.arange(nr * nz).reshape(nz, nr)
    faces = []
    a = idx[:-1, :]
    b = np.roll(idx[:-1, :], -1, axis=1)
    c = np.roll(idx[1:, :], -1, axis=1)
    d = idx[1:, :]
    quads = np.stack([a.ravel(), b.ravel(), c.ravel(), d.ravel()], axis=1)
    top = len(verts) - 1
    tris = [(idx[-1, i], idx[-1, (i + 1) % nr], top) for i in range(nr)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], quads.tolist() + tris)
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    if mat is not None:
        me.materials.append(mat)
    link(ob)
    return ob


def write_meta(path, **kw):
    import json
    with open(path, "w") as f:
        json.dump(kw, f, indent=1, default=float)


def frame_done(outdir, f):
    p = os.path.join(outdir, f"f_{f:04d}Image.exr")
    return os.path.exists(p) and os.path.getsize(p) > 0


def render_frames(outdir, frames=None, step=1):
    """Rendert die (noch fehlenden) Frames; überspringt vorhandene Dateien."""
    sc = bpy.context.scene
    fs = frames or range(sc.frame_start, sc.frame_end + 1, step)
    for f in fs:
        if frame_done(outdir, f):
            continue
        sc.frame_set(f)
        bpy.ops.render.render(write_still=False)
        print(f"FRAME_DONE {f}", flush=True)


def set_output(outdir=None, name="f_####"):
    fo = [n for n in bpy.context.scene.compositing_node_group.nodes if n.bl_idname == "CompositorNodeOutputFile"][0]
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fo.directory = outdir.rstrip("/") + "/"
    fo.file_name = name


def render_still(outdir, name):
    """Einzelbild (Vorschau) mit eigenem Dateinamen."""
    set_output(outdir, name + "_####")
    bpy.ops.render.render(write_still=False)
    f = bpy.context.scene.frame_current
    return os.path.join(outdir, f"{name}_{f:04d}Image.exr")
