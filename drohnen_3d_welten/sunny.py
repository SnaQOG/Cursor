"""Thousand Sunny (One Piece) – v3 nach den offiziellen Model Sheets (三面図 / 決定稿, Folge 313~).

Lokales System: +X = Bug, +Y = Backbord, +Z = oben, Wasserlinie z = 0.
Maße aus den Zeichnungen (Maßstabsleiste 0–100, 1 Einheit ≈ 9 cm):
  Rumpfkörper ~21 m lang und ~12 m breit (sehr bauchig), Gesamtlänge Löwe–Heckkanone ~28 m,
  Rasendeck 4,2 m über Wasser, rote U-Bordwand (Bug 9,3 m, Mitte 5,2 m) mit cremefarbener Kante,
  Voluten und Bullaugen, schwarzes Band und schwarzer Ring mit „1“ (Soldier Dock) mittschiffs,
  roter Bugschild mit gelben Nieten, Vorschiff mit Steuerrad, zweistöckiges Achterkastell
  (Bogenfenster), Rundturm mit rot-gelber Kuppel, große Heckkanone (Gaon Cannon / Coup de Burst),
  Fockmast mit Ausguck-Kuppel und ~20 m breitem Jolly-Roger-Rahsegel, Großmast auf dem
  Achterkastell mit Rahsegel und rot-schwarz gestreiftem Gaffelsegel.
"""
import math

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

import fpv
import textures

X_S, X_B = -11.2, 10.0      # Heck- und Bugende des Rumpfkörpers
X_C = -0.6                  # breiteste Stelle
B2 = 6.0                    # halbe Breite
Z_DECK = 4.2                # Rasendeck
Z_FORE = 8.3                # Vorschiffsdeck (Steuerrad)
Z_ROOF = 9.05               # Dach des Achterkastells
X_CAST = -5.2               # Front des Achterkastells
X_FORE = 5.8                # Achterkante des Vorschiffs
FORE_X, MAIN_X = 2.9, -7.5  # Masten
SD_X, SD_Z, SD_R = -0.2, 1.9, 2.42   # Soldier-Dock-Ring
PLANK_W = 0.32
PLANK_L = 6.0
# Farben (linear) nach dem farbigen Model Sheet
RED = (0.36, 0.018, 0.014)
CREAM = (0.80, 0.63, 0.36)
BLACK = (0.018, 0.016, 0.014)
YELLOW = (0.86, 0.50, 0.025)

# Rückwärtskompatibel für ocean.ocean_fx_gn / world_onepiece
L = X_B - X_S
W = B2


def half_beam(X):
    X = np.asarray(X, float)
    bow = X >= X_C
    t = np.where(bow, (X - X_C) / (X_B - X_C), (X_C - X) / (X_C - X_S))
    p = np.where(bow, 2.4, 2.8)
    return B2 * np.clip(1 - np.clip(t, 0, 1) ** p, 0, 1) ** (1 / p)


def keel(X):
    X = np.asarray(X, float)
    return (-2.8 + 3.3 * np.clip((X - 0.5) / (X_B - 0.5), 0, 1) ** 2.2
            + 5.0 * np.clip((-1.0 - X) / (-1.0 - X_S), 0, 1) ** 2.0)


def sheer(X):
    """Oberkante der Bordwand: U-Form zwischen Achterkastell und Vorschiff (flacher Boden 5,2 m,
    steile Arme bis 7,4/7,2 m, Bugarm steigt weiter bis 9,3 m), achtern Kastell-Sockel 4,5 m."""
    X = np.asarray(X, float)
    g = np.full_like(X, 5.2)
    s = np.clip((X - 1.9) / 1.4, 0, 1)
    g = np.where(X > 1.9, 5.2 + 2.2 * (1 - np.sqrt(1 - s * s)), g)
    g = np.where(X > 3.3, 7.4 + 1.9 * np.clip((X - 3.3) / (X_FORE - 3.3), 0, 1) ** 1.2, g)
    g = np.where(X > X_FORE, 9.3 + 0.3 * np.clip((X - X_FORE) / (X_B - X_FORE), 0, 1), g)
    s = np.clip((-2.5 - X) / 0.9, 0, 1)
    g = np.where(X < -2.5, 5.2 + 2.0 * (1 - np.sqrt(1 - s * s)), g)
    w = np.clip((X - (X_CAST - 0.15)) / 0.15, 0, 1)
    g = np.where(X < X_CAST, 4.5 + (g - 4.5) * w, g)
    return g


def section_y(X, z):
    """Halbe Breite in Höhe z: runde Kimm (Radius ~3,4 m), senkrechte Seiten, leicht ausgestellte Bordwand."""
    X = np.asarray(X, float)
    z = np.asarray(z, float)
    k = keel(X)
    rb = np.clip((sheer(X) - k) * 0.8, 0.1, 3.4)
    zz = np.clip((z - k) / rb, 0, 1)
    f = np.sqrt(np.clip(1 - (1 - zz) ** 2, 0, 1))
    flare = 1 + 0.035 * np.clip((z - Z_DECK) / 4.0, 0, 1)
    return half_beam(X) * f * flare


def hull_frame(X, z, side=1):
    """(Punkt, Tangente längs, Tangente hoch, Außennormale) auf der Rumpfhaut."""
    def P(xx, zz):
        return Vector((xx, side * float(section_y(xx, zz)), zz))
    e = 0.03
    p = P(X, z)
    tx = (P(X + e, z) - P(X - e, z)).normalized()
    tz = (P(X, z + e) - P(X, z - e)).normalized()
    n = tx.cross(tz)
    if n.y * side < 0:
        n = -n
    return p, tx, tz, n.normalized()


def hull_front_point(y, z):
    """Vorderster Rumpfpunkt mit |Breite| = y in Höhe z (für Bugschild-Nieten)."""
    xs = np.linspace(X_C, X_B, 800)
    w = section_y(xs, np.full_like(xs, z))
    ok = np.where((w >= abs(y)) & (sheer(xs) > z + 0.2) & (keel(xs) < z - 0.2))[0]
    if len(ok) == 0:
        return None
    X = float(xs[ok[-1]])
    p = Vector((X, y, z))
    e = 0.03
    # Normale über Nachbarpunkte der Kontur
    def surf(yy, zz):
        ww = section_y(xs, np.full_like(xs, zz))
        k = np.where((ww >= abs(yy)) & (sheer(xs) > zz))[0]
        return Vector((float(xs[k[-1]]) if len(k) else X, yy, zz))
    ty = (surf(y + e, z) - surf(y - e, z)).normalized()
    tz = (surf(y, z + e) - surf(y, z - e)).normalized()
    n = ty.cross(tz)
    if n.x < 0:
        n = -n
    return p, n.normalized()


# --------------------------------------------------------------------------
# Materialien
# --------------------------------------------------------------------------

def grain_wood(nb, along, across, rnd, dark=(0.12, 0.058, 0.022), mid=(0.24, 0.12, 0.045),
               light=(0.40, 0.21, 0.08)):
    """Holzmaserung: stark gestrecktes Rauschen (Fasern) + Jahresring-Wellen, Variation pro Brett."""
    v = nb.comb(nb.math("MULTIPLY", along, 0.35), nb.math("ADD", nb.math("MULTIPLY", across, 30.0),
                                                           nb.math("MULTIPLY", rnd, 40.0)), nb.math("MULTIPLY", rnd, 9.0))
    fib = nb.noise(v, scale=1.2, detail=5, rough=0.6, dist=1.5)
    ring = nb.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Y", wave_profile="SIN")
    nb.link(v, ring.inputs["Vector"])
    ring.inputs["Scale"].default_value = 0.8
    ring.inputs["Distortion"].default_value = 6.0
    ring.inputs["Detail"].default_value = 2.0
    g = nb.math("ADD", nb.math("MULTIPLY", nb.out(fib, "Fac"), 0.7), nb.math("MULTIPLY", nb.out(ring, "Fac"), 0.3))
    col = nb.ramp(g, [(0.3, dark), (0.5, mid), (0.72, light)])
    col = nb.vmath("SCALE", col, scale=nb.math("MULTIPLY_ADD", rnd, 0.4, 0.8))
    return col, g


def wood_material(name, tint=None, scale=0.5, rough=0.55, dark=1.0, board=0.22, axis="Z"):
    """Bretter-Holz (senkrechte Bretter bei axis='Z', Maserung entlang der Achse)."""
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    x, y, z = nb.sep(co)
    if axis == "Z":
        along = z
        acr = nb.math("ADD", x, y)
    else:
        along = x
        acr = nb.math("ADD", z, y)
    bf = nb.math("DIVIDE", acr, board)
    bid = nb.math("FLOOR", bf)
    bt = nb.math("FRACT", bf)
    wn = nb.node("ShaderNodeTexWhiteNoise")
    wn.noise_dimensions = "1D"
    nb.link(bid, wn.inputs["W"])
    rnd = nb.out(wn, "Value")
    col, g = grain_wood(nb, along, nb.math("MULTIPLY", bt, board), rnd)
    col = nb.vmath("SCALE", col, scale=dark)
    seam = nb.math("LESS_THAN", bt, 0.05)
    col = nb.mix(nb.math("MULTIPLY", seam, 0.8), col, (0.02, 0.015, 0.01))
    col = _wear(nb, col, edge_col=(0.55, 0.42, 0.3), cav_col=(0.03, 0.02, 0.015))
    p = fpv.principled(nb, Base_Color=col, Roughness=nb.math("MULTIPLY_ADD", g, 0.2, rough - 0.1))
    h = nb.math("SUBTRACT", nb.math("MULTIPLY", g, 0.3), nb.math("MULTIPLY", seam, 0.5))
    nb.link(nb.bump(h, strength=0.4, distance=0.01), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def _wear(nb, col, edge_col, cav_col, edge=0.35, cav=0.5):
    """Kantenabrieb (hell) und Schmutz in Vertiefungen (dunkel) über Pointiness."""
    pt = nb.out(nb.node("ShaderNodeNewGeometry"), "Pointiness")
    n = nb.noise(nb.coords("Object"), scale=6.0, detail=3)
    e = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.math("ADD", pt, nb.math("MULTIPLY", nb.math("SUBTRACT", nb.out(n, "Fac"), 0.5), 0.06)), e.inputs["Value"])
    e.inputs["From Min"].default_value = 0.53
    e.inputs["From Max"].default_value = 0.6
    c = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(pt, c.inputs["Value"])
    c.inputs["From Min"].default_value = 0.47
    c.inputs["From Max"].default_value = 0.40
    col = nb.mix(nb.math("MULTIPLY", e.outputs[0], edge), col, edge_col)
    col = nb.mix(nb.math("MULTIPLY", c.outputs[0], cav), col, cav_col)
    return col


def paint_material(name, color, rough=0.42, wear=True, coat=0.2, under=(0.35, 0.22, 0.12)):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    big = nb.noise(co, scale=0.5, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(big, "Fac"), 0.25), color, [c * 1.12 for c in color])
    fine = nb.noise(co, scale=4.0, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(fine, "Fac"), 0.2), col, [c * 0.75 for c in color])
    if wear:
        col = _wear(nb, col, edge_col=under, cav_col=[c * 0.35 for c in color], edge=0.55, cav=0.45)
    p = fpv.principled(nb, Base_Color=col, Roughness=rough)
    p.inputs["Coat Weight"].default_value = coat
    nb.link(nb.bump(nb.out(fine, "Fac"), strength=0.06, distance=0.01), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def iron_material(name="Iron"):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    n = nb.noise(co, scale=3.0, detail=3)
    rust = nb.node("ShaderNodeMapRange", clamp=True)
    nb.link(nb.out(n, "Fac"), rust.inputs["Value"])
    rust.inputs["From Min"].default_value = 0.55
    rust.inputs["From Max"].default_value = 0.7
    col = nb.mix(rust.outputs[0], (0.05, 0.05, 0.05), (0.18, 0.07, 0.03))
    p = fpv.principled(nb, Base_Color=col, Roughness=nb.mix(rust.outputs[0], 0.45, 0.85, dtype="FLOAT"),
                       Metallic=nb.mix(rust.outputs[0], 0.85, 0.1, dtype="FLOAT"))
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def sail_material(name, decal=None, stripes=None):
    mat, nb, out = fpv.new_material(name)
    uv = nb.coords("UV")
    co = nb.coords("Object")
    base = (0.78, 0.74, 0.64)
    wave = nb.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X")
    nb.link(uv, wave.inputs["Vector"])
    wave.inputs["Scale"].default_value = 11.0
    seams = nb.ramp(nb.out(wave, "Fac"), [(0.0, (0.8, 0.8, 0.8)), (0.05, (1, 1, 1)), (0.95, (1, 1, 1)), (1.0, (0.8, 0.8, 0.8))])
    dirt = nb.noise(co, scale=0.8, detail=3, rough=0.6)
    dcol = nb.ramp(nb.out(dirt, "Fac"), [(0.3, (0.8, 0.76, 0.68)), (0.75, (1.0, 1.0, 1.0))])
    col = nb.mix(1.0, base, seams, blend="MULTIPLY")
    if stripes:
        st = nb.math("FRACT", nb.math("MULTIPLY", nb.sep(uv)[0], stripes))
        red = nb.math("LESS_THAN", st, 0.5)
        col = nb.mix(red, (0.03, 0.03, 0.03), (0.50, 0.05, 0.035))
        col = nb.mix(1.0, col, seams, blend="MULTIPLY")
    col = nb.mix(1.0, col, dcol, blend="MULTIPLY")
    if decal:
        img = nb.image(decal, uv)
        img.extension = "CLIP"
        col = nb.mix(nb.out(img, "Alpha"), col, nb.out(img, "Color"))
    p = fpv.principled(nb, Base_Color=col, Roughness=0.82)
    p.inputs["Sheen Weight"].default_value = 0.25
    tr = nb.node("ShaderNodeBsdfTranslucent")
    nb.set_in(tr, "Color", col)
    mix = nb.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.3
    nb.link(p.outputs[0], mix.inputs[1])
    nb.link(tr.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    return mat


def flag_material():
    mat, nb, out = fpv.new_material("Flag")
    uv = nb.coords("UV")
    img = nb.image(textures.jolly_roger_flag(), uv)
    p = fpv.principled(nb, Base_Color=nb.out(img, "Color"), Roughness=0.85)
    p.inputs["Sheen Weight"].default_value = 0.4
    tr = nb.node("ShaderNodeBsdfTranslucent")
    nb.set_in(tr, "Color", nb.out(img, "Color"))
    mix = nb.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.2
    nb.link(p.outputs[0], mix.inputs[1])
    nb.link(tr.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    return mat


def lawn_materials():
    base, nb, out = fpv.new_material("LawnBase")
    co = nb.coords("Object")
    img = nb.image(fpv.ASSETS + "/bab/textures_grass.jpg", nb.mapping(co, scale=(0.9, 0.9, 0.9)))
    col = nb.mix(1.0, nb.out(img, "Color"), (0.75, 0.9, 0.7), blend="MULTIPLY")
    p = fpv.principled(nb, Base_Color=col, Roughness=0.85)
    nb.link(p.outputs[0], out.inputs[0])
    blade, nb, out = fpv.new_material("GrassBlade")
    uv = nb.coords("UV")
    v = nb.sep(uv)[1]
    oi = nb.out(nb.node("ShaderNodeObjectInfo"), "Random")
    wn = nb.node("ShaderNodeTexWhiteNoise")
    wn.noise_dimensions = "3D"
    nb.link(nb.coords("Object"), wn.inputs["Vector"])
    root = nb.mix(nb.out(wn, "Value"), (0.02, 0.05, 0.01), (0.04, 0.08, 0.015))
    tip = nb.mix(nb.out(wn, "Value"), (0.14, 0.24, 0.04), (0.26, 0.30, 0.08))
    col = nb.mix(v, root, tip)
    p = fpv.principled(nb, Base_Color=col, Roughness=0.6)
    p.inputs["Subsurface Weight"].default_value = 0.0
    tr = nb.node("ShaderNodeBsdfTranslucent")
    nb.set_in(tr, "Color", col)
    mix = nb.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.25
    nb.link(p.outputs[0], mix.inputs[1])
    nb.link(tr.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs[0])
    return base, blade



def hull_material():
    """Rumpf nach dem farbigen Model Sheet: goldbraune Planken, rote U-Bordwand mit cremefarbener Kante,
    schwarzes Band, roter Bugschild, Soldier-Dock-Ring mit „1“. Zonen aus UV 'Zone' (x = Abstand zur
    Bordwand-Oberkante in m, y = 1 im U-Bereich) und Objektkoordinaten/-normalen."""
    mat, nb, out = fpv.new_material("HullSunny")
    uvn = nb.node("ShaderNodeUVMap")
    uvn.uv_map = "UVMap"
    ux, uy, _ = nb.sep(uvn.outputs[0])
    zn = nb.node("ShaderNodeUVMap")
    zn.uv_map = "Zone"
    dtop, uflag, _ = nb.sep(zn.outputs[0])
    tc = nb.node("ShaderNodeTexCoord")
    co = tc.outputs["Object"]
    x, y, z = nb.sep(co)
    nx = nb.sep(tc.outputs["Normal"])[0]
    # Planken + versetzte Stöße
    rowf = nb.math("DIVIDE", uy, PLANK_W)
    row = nb.math("FLOOR", rowf)
    t = nb.math("FRACT", rowf)
    rh = nb.node("ShaderNodeTexWhiteNoise")
    rh.noise_dimensions = "1D"
    nb.link(row, rh.inputs["W"])
    segf = nb.math("DIVIDE", nb.math("ADD", ux, nb.math("MULTIPLY", nb.out(rh, "Value"), PLANK_L)), PLANK_L)
    seg = nb.math("FLOOR", segf)
    s = nb.math("FRACT", segf)
    pid = nb.node("ShaderNodeTexWhiteNoise")
    pid.noise_dimensions = "2D"
    nb.link(nb.comb(row, seg, 0.0), pid.inputs["Vector"])
    rnd = nb.out(pid, "Value")
    wood, grain = grain_wood(nb, ux, nb.math("MULTIPLY", t, PLANK_W), rnd)
    seam_t = nb.math("LESS_THAN", t, 0.04)
    seam_s = nb.math("MAXIMUM", nb.math("LESS_THAN", s, 0.004), nb.math("GREATER_THAN", s, 0.996))
    seam = nb.math("MAXIMUM", seam_t, seam_s)

    def rng(val, a, b, soft=0.012):
        lo = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(val, lo.inputs["Value"])
        lo.inputs["From Min"].default_value = a - soft
        lo.inputs["From Max"].default_value = a + soft
        hi = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(val, hi.inputs["Value"])
        hi.inputs["From Min"].default_value = b + soft
        hi.inputs["From Max"].default_value = b - soft
        return nb.math("MULTIPLY", lo.outputs[0], hi.outputs[0])

    def step(val, a, b):
        m = nb.node("ShaderNodeMapRange", clamp=True)
        nb.link(val, m.inputs["Value"])
        m.inputs["From Min"].default_value = a
        m.inputs["From Max"].default_value = b
        return m.outputs[0]

    inU = step(uflag, 0.45, 0.55)
    cream_t = nb.math("MULTIPLY", inU, rng(dtop, -0.5, 0.22))
    red = nb.math("MULTIPLY", inU, rng(dtop, 0.22, 2.45))
    cream_l = nb.math("MULTIPLY", inU, rng(dtop, 2.45, 2.57))
    blk = nb.math("MAXIMUM", nb.math("MULTIPLY", inU, rng(dtop, 2.57, 2.9)),
                  nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, inU), rng(z, 3.55, 3.9)))
    # Bugschild (nach vorn zeigende Flächen), cremefarben gerahmt
    fz = nb.math("MULTIPLY", step(z, 0.9, 1.1), step(x, X_B - 4.6, X_B - 4.4))
    front = nb.math("MULTIPLY", step(nx, 0.52, 0.56), fz)
    front_b = nb.math("MULTIPLY", nb.math("MULTIPLY", step(nx, 0.44, 0.47),
                                          nb.math("SUBTRACT", 1.0, step(nx, 0.50, 0.53))), fz)
    # Soldier-Dock: schwarzer Ring, Holztor mit „1“ (nur an den Bordwänden)
    dx = nb.math("SUBTRACT", x, SD_X)
    dz = nb.math("SUBTRACT", z, SD_Z)
    r = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", dx, dx), nb.math("MULTIPLY", dz, dz)))
    side = step(nb.math("ABSOLUTE", y), 3.0, 3.5)
    ring = nb.math("MULTIPLY", rng(r, SD_R - 0.4, SD_R, soft=0.015), side)
    inside = nb.math("MULTIPLY", step(r, SD_R - 0.38, SD_R - 0.42), side)
    D = 3.2
    du = nb.math("MULTIPLY", nb.math("DIVIDE", dx, D), nb.math("MULTIPLY", nb.math("SIGN", y), -1.0))
    dec = nb.image(textures.number_decal("1", textures.OUT + "/soldier_dock_1.png"),
                   nb.comb(nb.math("ADD", du, 0.5), nb.math("ADD", nb.math("DIVIDE", dz, D), 0.5), 0.0))
    dec.extension = "CLIP"
    one = nb.math("MULTIPLY", nb.out(dec, "Alpha"), inside)
    col = wood
    col = nb.mix(red, col, RED)
    col = nb.mix(cream_t, col, CREAM)
    col = nb.mix(cream_l, col, CREAM)
    col = nb.mix(blk, col, BLACK)
    col = nb.mix(front, col, RED)
    col = nb.mix(front_b, col, CREAM)
    col = nb.mix(inside, col, nb.vmath("SCALE", wood, scale=0.85))
    col = nb.mix(one, col, BLACK)
    col = nb.mix(ring, col, BLACK)
    paint = nb.math("MINIMUM", nb.math("ADD", nb.math("ADD", red, nb.math("ADD", cream_t, cream_l)),
                                       nb.math("ADD", nb.math("ADD", blk, front), nb.math("ADD", front_b, ring))), 1.0)
    paint = nb.math("MULTIPLY", paint, nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", inside, nb.math("SUBTRACT", 1.0, ring))))
    # Fugen (auch unter der Farbe sichtbar), Nässe/Algen an der Wasserlinie, Salzfahnen
    col = nb.mix(nb.math("MULTIPLY", seam, nb.math("SUBTRACT", 0.85, nb.math("MULTIPLY", paint, 0.45))), col,
                 (0.02, 0.015, 0.01))
    grime = nb.noise(co, scale=0.5, detail=3)
    wl = step(nb.math("ADD", z, nb.math("MULTIPLY", nb.out(grime, "Fac"), 0.25)), 0.18, 0.32)
    col = nb.mix(nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, wl), 0.7), col,
                 nb.vmath("MULTIPLY", col, (0.35, 0.42, 0.36)))
    stv = nb.noise(nb.comb(nb.math("MULTIPLY", ux, 1.5), nb.math("MULTIPLY", uy, 0.12), 0.0), scale=1.0, detail=2)
    sm = step(nb.out(stv, "Fac"), 0.56, 0.72)
    col = nb.mix(nb.math("MULTIPLY", sm, 0.14), col, (0.30, 0.28, 0.25))
    col = _wear(nb, col, edge_col=(0.45, 0.30, 0.16), cav_col=(0.03, 0.02, 0.015), edge=0.25, cav=0.35)
    rgh_w = nb.math("MULTIPLY_ADD", grain, 0.25, 0.5)
    rough = nb.mix(paint, rgh_w, 0.36, dtype="FLOAT")
    rough = nb.mix(nb.math("SUBTRACT", 1.0, wl), rough, 0.2, dtype="FLOAT")
    p = fpv.principled(nb, Base_Color=col, Roughness=rough)
    p.inputs["Coat Weight"].default_value = 0.15
    bulge = nb.math("SINE", nb.math("MULTIPLY", t, math.pi))
    h = nb.math("SUBTRACT", nb.math("MULTIPLY", bulge, nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", paint, 0.5))),
                nb.math("MULTIPLY", seam, 0.6))
    h = nb.math("ADD", h, nb.math("MULTIPLY", ring, 0.8))
    nb.link(nb.bump(h, strength=0.5, distance=0.012), p.inputs["Normal"])
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def stripe_dome_material(name="DomeStripes", n=12):
    """Kuppel mit abwechselnd roten und gelben Segmenten (Ausguck und Rundturm)."""
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    x, y, _ = nb.sep(co)
    ang = nb.math("ARCTAN2", y, x)
    k = nb.math("FLOOR", nb.math("MULTIPLY", nb.math("ADD", nb.math("DIVIDE", ang, 2 * math.pi), 0.5), n))
    odd = nb.math("MODULO", k, 2.0)
    col = nb.mix(odd, RED, YELLOW)
    fine = nb.noise(co, scale=6.0, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(fine, "Fac"), 0.18), col, nb.vmath("SCALE", col, scale=0.7))
    col = _wear(nb, col, edge_col=(0.45, 0.30, 0.16), cav_col=(0.05, 0.02, 0.01), edge=0.3, cav=0.3)
    p = fpv.principled(nb, Base_Color=col, Roughness=0.4)
    p.inputs["Coat Weight"].default_value = 0.25
    nb.link(p.outputs[0], out.inputs[0])
    return mat


def metal_material(name, color, rough=0.35, metallic=0.85):
    mat, nb, out = fpv.new_material(name)
    co = nb.coords("Object")
    n = nb.noise(co, scale=4.0, detail=3)
    col = nb.mix(nb.math("MULTIPLY", nb.out(n, "Fac"), 0.3), color, [c * 0.6 for c in color])
    p = fpv.principled(nb, Base_Color=col, Roughness=nb.math("MULTIPLY_ADD", nb.out(n, "Fac"), 0.2, rough - 0.1),
                       Metallic=metallic)
    nb.link(p.outputs[0], out.inputs[0])
    return mat


# --------------------------------------------------------------------------
# Geometrie-Helfer
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Geometrie-Helfer
# --------------------------------------------------------------------------

def bevel_stack(ob, small=0.012, big=None, angle=30):
    """Bevel-Hierarchie: optional große Silhouetten-Fase (Gewicht), feine Glanzkante, Weighted Normal."""
    if big:
        b1 = ob.modifiers.new("bevel_silhouette", "BEVEL")
        b1.limit_method = "ANGLE"
        b1.angle_limit = math.radians(60)
        b1.width = big
        b1.segments = 4
        b1.profile = 0.5
    b2 = ob.modifiers.new("bevel_micro", "BEVEL")
    b2.limit_method = "ANGLE"
    b2.angle_limit = math.radians(angle)
    b2.width = small
    b2.segments = 2
    b2.harden_normals = True
    wn = ob.modifiers.new("wnormal", "WEIGHTED_NORMAL")
    wn.keep_sharp = True
    return ob


def cyl(name, p0, p1, r0, r1=None, mat=None, verts=16, cap=True):
    r1 = r0 if r1 is None else r1
    p0, p1 = Vector(p0), Vector(p1)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=cap, segments=verts, radius1=r0, radius2=r1, depth=(p1 - p0).length)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    d = (p1 - p0).normalized()
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = d.to_track_quat("Z", "Y")
    ob.location = (p0 + p1) / 2
    return ob


def sphere(name, loc, r, mat=None, scale=(1, 1, 1), seg=32, ring=16):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=ring, radius=r)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    ob.location = loc
    ob.scale = scale
    return ob


def box(name, a, b, mat, bevel=0.03):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    ax, ay, az = a
    bx, by, bz = b
    bmesh.ops.scale(bm, vec=(bx - ax, by - ay, bz - az), verts=bm.verts)
    bmesh.ops.translate(bm, vec=((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2), verts=bm.verts)
    ob = fpv.mesh_from_bmesh(bm, name, mat, smooth=False)
    if bevel:
        bevel_stack(ob, small=bevel)
    return ob


def tubes_mesh(name, segments, r, mat, sides=6):
    """Viele Seil-/Stabzylinder in EINEM Mesh (Takelage, Webeleinen, Nieten-Stäbe)."""
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    for (a, b) in segments:
        a, b = np.array(a, float), np.array(b, float)
        d = b - a
        ln = np.linalg.norm(d)
        if ln < 1e-6:
            continue
        d /= ln
        t1 = np.cross(d, [0, 0, 1.0])
        if np.linalg.norm(t1) < 1e-3:
            t1 = np.cross(d, [1.0, 0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(d, t1)
        i0 = len(verts)
        for p in (a, b):
            for q in ang:
                verts.append(tuple(p + r * (np.cos(q) * t1 + np.sin(q) * t2)))
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((i0 + k, i0 + k2, i0 + sides + k2, i0 + sides + k))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    me.materials.append(mat)
    fpv.link(ob)
    return ob


def catenary(a, b, sag, n=12):
    a, b = np.array(a, float), np.array(b, float)
    pts = [a + (b - a) * t - np.array([0, 0, sag * 4 * t * (1 - t)]) for t in np.linspace(0, 1, n)]
    return list(zip(pts[:-1], pts[1:]))


def spheres_mesh(name, centers, r, mat, subdiv=1, flatten=0.5, normals=None):
    """Nieten: viele kleine Halbkugeln in einem Mesh."""
    bm = bmesh.new()
    for i, c in enumerate(centers):
        res = bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=r)
        nrm = Vector(normals[i]) if normals is not None else Vector((0, 0, 1))
        rot = nrm.to_track_quat("Z", "Y").to_matrix().to_4x4()
        sc = Matrix.Diagonal((1, 1, flatten, 1))
        bmesh.ops.transform(bm, verts=res["verts"], matrix=Matrix.Translation(Vector(c)) @ rot @ sc)
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    return ob



# --------------------------------------------------------------------------
# Rumpf, Decks, Rasen
# --------------------------------------------------------------------------

def hull_xs():
    xs = list(np.linspace(X_S, X_B, 240)) + [X_CAST, X_CAST - 0.15, X_CAST - 0.02, X_CAST + 0.02]
    return np.array(sorted(set(np.round(xs, 4))))


def hull_mesh(mat_hull, mat_inner, nv=64):
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    zl = bm.loops.layers.uv.new("Zone")
    rows = []
    vs = np.linspace(0, 1, nv) ** 1.25     # dichter an der Kimm
    for X in hull_xs():
        k = float(keel(X))
        G = float(sheer(X))
        zs = k + (G - k) * vs
        ys = section_y(np.full(nv, X), zs)
        half = np.stack([ys, zs], 1)
        g = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(half, axis=0), axis=1))])
        uf = 1.0 if X >= X_CAST - 0.01 else 0.0
        pts, meta = [], []
        for i in range(nv - 1, -1, -1):
            pts.append((X, -ys[i], zs[i]))
            meta.append((g[i], G - zs[i]))
        for i in range(1, nv):
            pts.append((X, ys[i], zs[i]))
            meta.append((g[i], G - zs[i]))
        rows.append(([bm.verts.new(p) for p in pts], meta, uf, float(X)))
    for (va, ma, fa, xa), (vb, mb, fb, xb) in zip(rows[:-1], rows[1:]):
        for j in range(len(va) - 1):
            try:
                f = bm.faces.new([va[j], vb[j], vb[j + 1], va[j + 1]])
            except ValueError:
                continue
            for lp, (mm, ff, xx) in zip(f.loops, [(ma[j], fa, xa), (mb[j], fb, xb), (mb[j + 1], fb, xb),
                                                  (ma[j + 1], fa, xa)]):
                lp[uvl].uv = (xx, mm[0])
                lp[zl].uv = (mm[1], ff)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.002)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = fpv.mesh_from_bmesh(bm, "Hull", mat_hull)
    ob.data.materials.append(mat_inner)
    sol = ob.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.24
    sol.offset = -1
    sol.material_offset = 1
    sol.material_offset_rim = 1
    ob.modifiers.new("wn", "WEIGHTED_NORMAL")
    return ob


def plan_mesh(name, x0, x1, z, mat, inset=0.3, nu=60, nw=16, thick=0.0):
    """Deck/Platte im Grundriss des Rumpfes (Breite aus section_y in Höhe z)."""
    bm = bmesh.new()
    rows = []
    for X in np.linspace(x0, x1, nu):
        hw = max(float(section_y(X, z)) - inset, 0.03)
        rows.append([bm.verts.new((X, -hw + 2 * hw * t, z)) for t in np.linspace(0, 1, nw)])
    for i in range(nu - 1):
        for j in range(nw - 1):
            bm.faces.new([rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]])
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    if thick:
        sol = ob.modifiers.new("solid", "SOLIDIFY")
        sol.thickness = thick
    return ob


def lawn_blades(mat, count=70000, seed=5):
    """Echte Grashalme (Dreiecke) auf dem Rasendeck zwischen Achterkastell und Vorschiff."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_CAST + 0.25, X_FORE - 0.25, count)
    hw = np.maximum(section_y(x, np.full(count, Z_DECK)) - 0.45, 0.05)
    y = rng.uniform(-1, 1, count) * hw
    z = np.full(count, Z_DECK + 0.005)
    h = rng.uniform(0.05, 0.12, count)
    a = rng.uniform(0, 2 * np.pi, count)
    lean = rng.normal(0, 0.35, (count, 2)) * h[:, None]
    w = 0.006 + 0.004 * rng.random(count)
    base = np.stack([x, y, z], 1)
    side = np.stack([np.cos(a), np.sin(a), np.zeros(count)], 1) * w[:, None]
    tip = base + np.stack([lean[:, 0], lean[:, 1], h], 1)
    verts = np.concatenate([base - side, base + side, tip], 0)
    n = count
    faces = np.stack([np.arange(n), np.arange(n) + n, np.arange(n) + 2 * n], 1)
    me = bpy.data.meshes.new("LawnBlades")
    me.vertices.add(3 * n)
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(3 * n)
    me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(n)
    me.polygons.foreach_set("loop_start", (np.arange(n) * 3).astype(np.int32))
    uvl = me.uv_layers.new(name="UVMap")
    luv = np.zeros((n, 3, 2), np.float32)
    luv[:, 2, 1] = 1.0
    luv[:, 1, 0] = 1.0
    uvl.data.foreach_set("uv", luv.ravel())
    me.update()
    ob = bpy.data.objects.new("LawnBlades", me)
    me.materials.append(mat)
    fpv.link(ob)
    return ob


def curve_tube(name, pts, radius, mat, res=3, cyclic=False):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1)
    sp.use_cyclic_u = cyclic
    cu.bevel_depth = radius
    cu.bevel_resolution = res
    cu.use_fill_caps = True
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    fpv.link(ob)
    return ob


def local_matrix(p, tx, n):
    """Lokales System an einer Wand: x = entlang der Wand, y = Außennormale, z = hoch."""
    tz = tx.cross(n).normalized()
    if tz.z < 0:
        tz = -tz
    tx = n.cross(tz).normalized()
    M = Matrix((tx, n, tz)).transposed().to_4x4()
    M.translation = p
    return M


def framed_window(name, M, w, h, mats, arch=False, frame="brown", depth=0.12, bars=True):
    """Fenster in lokalem Wandsystem M (Ursprung = Fenstermitte auf der Wand):
    dunkles Glas, Rahmen, Sprossenkreuz, optional Rundbogen."""
    objs = []
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(w, 0.06, h), verts=bm.verts)
    bmesh.ops.translate(bm, vec=(0, 0.02, 0), verts=bm.verts)
    g = fpv.mesh_from_bmesh(bm, name + "Glass", mats["glass"], smooth=False)
    g.matrix_world = M
    objs.append(g)
    fw = 0.08
    segs = [((-w / 2, 0.06, -h / 2), (w / 2, 0.06, -h / 2)), ((-w / 2, 0.06, -h / 2), (-w / 2, 0.06, h / 2)),
            ((w / 2, 0.06, -h / 2), (w / 2, 0.06, h / 2))]
    if arch:
        pts = [(-w / 2 * math.cos(a), 0.06, h / 2 + w / 2 * math.sin(a)) for a in np.linspace(0, math.pi, 13)]
        segs += list(zip(pts[:-1], pts[1:]))
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=w / 2, radius2=w / 2, depth=0.06)
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, "X"))
        bmesh.ops.translate(bm, vec=(0, 0.02, h / 2), verts=bm.verts)
        ag = fpv.mesh_from_bmesh(bm, name + "ArchGlass", mats["glass"])
        ag.matrix_world = M
        objs.append(ag)
    else:
        segs.append(((-w / 2, 0.06, h / 2), (w / 2, 0.06, h / 2)))
    if bars:
        segs.append(((0, 0.07, -h / 2), (0, 0.07, h / 2 + (w / 2 if arch else 0))))
        segs.append(((-w / 2, 0.07, h * 0.1), (w / 2, 0.07, h * 0.1)))
    fr = tubes_mesh(name + "Frame", segs, fw / 2 if not bars else fw * 0.45, mats[frame], sides=6)
    fr.matrix_world = M
    objs.append(fr)
    return objs


def porthole(name, p, n, r, mats):
    """Bullauge: grauer Metallring + dunkles Glas, nach außen ausgerichtet."""
    o1 = cyl(name + "Ring", p - n * 0.08, p + n * 0.1, r * 1.25, r * 1.25, mats["portring"], 28)
    o2 = cyl(name + "Glass", p - n * 0.02, p + n * 0.13, r, r, mats["glass"], 28)
    return [o1, o2]


def volute(name, p, tx, n, r0, mat, turns=1.6, rad=0.11):
    """Volute (Schnecke) am Ende der U-Bordwand, in der Ebene der Bordwand."""
    tz = tx.cross(n).normalized()
    if tz.z < 0:
        tz = -tz
    pts = []
    for a in np.linspace(0, turns * 2 * math.pi, 60):
        rr = r0 * (1 - a / (turns * 2 * math.pi) * 0.85)
        pts.append(tuple(p + n * 0.12 + tx * (rr * math.cos(a)) + tz * (rr * math.sin(a))))
    return curve_tube(name, pts, rad, mat)


def castle_outline(x0, x1, z, inset, n=30):
    xs = np.linspace(x0, x1, n)
    hw = np.maximum(section_y(xs, np.full_like(xs, z)) - inset, 0.3)
    return [(float(x), -float(w)) for x, w in zip(xs, hw)] + [(float(x), float(w)) for x, w in zip(xs[::-1], hw[::-1])]


def prism(name, outline, z0, z1, mat, smooth=True):
    bm = bmesh.new()
    vs = [bm.verts.new((x, y, z0)) for (x, y) in outline]
    f = bm.faces.new(vs)
    ext = bmesh.ops.extrude_face_region(bm, geom=[f])
    top = [e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=(0, 0, z1 - z0), verts=top)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = fpv.mesh_from_bmesh(bm, name, mat, smooth=smooth)
    bevel_stack(ob, small=0.02, angle=40)
    return ob


def wall_frame(X, z, sy, inset):
    """Punkt/Tangente/Normale an der Kastell-Seitenwand (Kontur section_y - inset)."""
    e = 0.05
    y0 = float(section_y(X, z)) - inset
    y1 = float(section_y(X + e, z)) - inset
    y2 = float(section_y(X - e, z)) - inset
    p = Vector((X, sy * y0, z))
    tx = Vector((2 * e, sy * (y1 - y2), 0)).normalized()
    n = Vector((-tx.y, tx.x, 0)) if sy > 0 else Vector((tx.y, -tx.x, 0))
    if n.y * sy < 0:
        n = -n
    return p, tx, n


def stairs(name, p0, p1, width, steps, mat, axis="X"):
    """Treppe von p0 (unten) nach p1 (oben); Stufen quer zur Laufrichtung, mit Wangen und Handlauf."""
    p0, p1 = Vector(p0), Vector(p1)
    run = p1 - p0
    objs = []
    horiz = Vector((run.x, run.y, 0))
    d = horiz.normalized()
    side = Vector((-d.y, d.x, 0))
    bm = bmesh.new()
    for k in range(steps):
        c = p0 + horiz * ((k + 0.5) / steps)
        ztop = p0.z + run.z * (k + 1) / steps
        res = bmesh.ops.create_cube(bm, size=1.0)
        L_ = horiz.length / steps + 0.05
        M = Matrix((d, side, Vector((0, 0, 1)))).transposed().to_4x4()
        M.translation = Vector((c.x, c.y, ztop - 0.04))
        bmesh.ops.transform(bm, verts=res["verts"], matrix=M @ Matrix.Diagonal((L_, width, 0.08, 1)))
    st = fpv.mesh_from_bmesh(bm, name, mat, smooth=False)
    objs.append(st)
    segs = []
    for s in (-1, 1):
        a = p0 + side * (s * width / 2)
        b = p1 + side * (s * width / 2)
        segs.append((tuple(a), tuple(b)))
        segs.append((tuple(a + Vector((0, 0, 0.95))), tuple(b + Vector((0, 0, 0.95)))))
        for t in np.linspace(0.05, 0.95, 6):
            q = a + (b - a) * t
            segs.append((tuple(q), tuple(q + Vector((0, 0, 0.95)))))
    objs.append(tubes_mesh(name + "Rail", segs, 0.04, mat, sides=6))
    return objs


def balustrade(name, pts, h, mats, spacing=0.22, closed=False):
    """Reling mit gedrechselten Docken (Model Sheet: Zinnenreihe) und Handlauf."""
    segs = []
    P = [Vector(p) for p in pts] + ([Vector(pts[0])] if closed else [])
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int((b - a).length / spacing))
        for t in np.linspace(0, 1, n, endpoint=False):
            q = a + (b - a) * t
            segs.append((tuple(q), tuple(q + Vector((0, 0, h)))))
    posts = tubes_mesh(name + "Posts", segs, 0.035, mats["cream"], sides=6)
    top = curve_tube(name + "Top", [tuple(p + Vector((0, 0, h))) for p in P], 0.07, mats["cream"])
    return [posts, top]


# --------------------------------------------------------------------------
# Segel
# --------------------------------------------------------------------------

def sail_mesh(name, width, height, bulge, mat, nx=34, ny=30, foot_arch=0.6, taper=0.06, twist=0.0, rake=0.0):
    """Rahsegel mit glaubhafter Windform: Bauch mit Maximum bei ~40 % Tiefe, gerundetes Unterliek,
    Spannungsfalten zu den Schothörnern (Noise, gerichtet), leichte Verwindung."""
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    bm = bmesh.new()
    uv_l = bm.loops.layers.uv.new("UVMap")
    rows = []
    for j in range(ny):
        t = j / (ny - 1)
        row = []
        for i in range(nx):
            s = i / (nx - 1)
            sx = 2 * s - 1
            y = sx * width / 2 * (1 - taper * (1 - t))
            foot = foot_arch * (1 - sx * sx) * t ** 6  # Unterliek nach oben gewölbt
            z = -t * height + foot
            depth = bulge * (1 - sx ** 2) * (np.sin(np.pi * min(t, 1) ** 0.85) ** 1.2)
            depth += twist * sx * t
            # Spannungsfalten diagonal zu den unteren Ecken
            fold = 0.05 * np.sin(18 * (abs(sx) - t * 0.9)) * (t ** 2) * (1 - abs(sx) * 0.3)
            row.append(bm.verts.new((depth + fold - rake * t ** 1.5, y, z)))
        rows.append(row)
    for j in range(ny - 1):
        for i in range(nx - 1):
            f = bm.faces.new([rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]])
            for lp, (ii, jj) in zip(f.loops, [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]):
                lp[uv_l].uv = (1 - ii / (nx - 1), 1 - jj / (ny - 1))
    edges = [rows[0], rows[-1], [r[0] for r in rows], [r[-1] for r in rows]]
    edge_pts = [[tuple(v.co) for v in e] for e in edges]
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    sub = ob.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1
    fpv.displace_obj(ob, "CLOUDS", size=1.3, strength=0.07, depth=2, name=name + "_wr")
    sol = ob.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.025
    return ob, edge_pts


def gaff_sail_mesh(name, luff, gaff_len, boom_len, gaff_rise, mat, nx=24, ny=26, bulge=0.9):
    """Gaffelsegel (trapezförmig, achtern des Mastes). Lokal: -X nach achtern, Z hoch."""
    bm = bmesh.new()
    uv_l = bm.loops.layers.uv.new("UVMap")
    rows = []
    for j in range(ny):
        t = j / (ny - 1)  # 0 unten .. 1 oben
        row = []
        for i in range(nx):
            s = i / (nx - 1)  # 0 am Mast .. 1 achtern
            ln = boom_len + (gaff_len - boom_len) * t
            x = -s * ln
            z = t * luff + s * gaff_rise * t
            y = bulge * np.sin(np.pi * s) * np.sin(np.pi * min(max(t, 0.02), 0.98)) * 0.9
            row.append(bm.verts.new((x, y, z)))
        rows.append(row)
    for j in range(ny - 1):
        for i in range(nx - 1):
            f = bm.faces.new([rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]])
            for lp, (ii, jj) in zip(f.loops, [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]):
                lp[uv_l].uv = (ii / (nx - 1), jj / (ny - 1))
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    fpv.displace_obj(ob, "CLOUDS", size=1.2, strength=0.05, depth=2, name=name + "_wr")
    sol = ob.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.025
    return ob



# --------------------------------------------------------------------------
# Löwen-Galionsfigur
# --------------------------------------------------------------------------

def petal_mesh(name, length, width, thick, mat, bend=0.3):
    """Spitzes Mähnen-Blatt (Sonnenstrahl): breite Basis, zur Spitze verjüngt, leicht gewölbt."""
    nx, ny = 7, 12
    bm = bmesh.new()
    grid = []
    for j in range(ny):
        t = j / (ny - 1)
        row = []
        wt = width * (1 - t ** 1.25) * min(1.0, 0.75 + t * 2.5) + 0.02
        for i in range(nx):
            s = (i / (nx - 1)) * 2 - 1
            row.append(bm.verts.new((s * wt / 2, -bend * (t ** 2) * length * 0.35 + 0.1 * (1 - s * s) * width,
                                     t * length)))
        grid.append(row)
    for j in range(ny - 1):
        for i in range(nx - 1):
            bm.faces.new([grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]])
    ob = fpv.mesh_from_bmesh(bm, name, mat)
    sol = ob.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = thick
    sol.offset = 0
    sub = ob.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 2
    return ob


def lion_head(body, mats):
    """Sunny-Löwe (Model Sheet): gelbes rundes Gesicht, Mähne aus spitzen orangefarbenen Strahlen in zwei
    Kränzen an einer drehbaren Nabe, gekreuzte Knochen dahinter, Hals zum Bugschild."""
    c = Vector((X_B + 3.0, 0, 7.1))
    R = 1.75
    parts = []
    hub = bpy.data.objects.new("ManeHub", None)
    fpv.link(hub)
    hub.parent = body
    hub.location = c + Vector((-0.7, 0, 0))
    parts.append(cyl("LionNeck", (X_B - 0.8, 0, 6.3), (c.x - 0.6, 0, c.z - 0.1), 1.55, 1.75, mats["mane_dark"], 40))
    back = cyl("ManeBack", (c.x - 1.25, 0, c.z), (c.x - 0.85, 0, c.z), 2.9, 2.7, mats["mane_dark"], 48)
    back.parent = hub
    back.location = back.location - hub.location
    parts.append(sphere("LionFace", c, R, mats["face"], scale=(0.72, 1.0, 0.97), seg=64, ring=32))
    for layer, (n, rad, ln, wd, mk, off, tilt, dx) in enumerate(((12, 1.5, 2.5, 1.9, "mane", 0.0, -14, -0.75),
                                                                 (12, 1.4, 1.95, 1.4, "mane_light", 15.0, -6, -0.45))):
        tpl = petal_mesh(f"PetalTpl{layer}", ln, wd, 0.2, mats[mk])
        for k in range(n):
            a = math.radians(off + 90 + k * 360 / n)
            pet = tpl.copy()
            pet.data = tpl.data
            fpv.link(pet)
            dirv = Vector((0, math.cos(a), math.sin(a)))
            base = c + dirv * rad + Vector((dx, 0, 0))
            q = dirv.to_track_quat("Z", "X") @ Matrix.Rotation(math.radians(tilt), 4, "X").to_quaternion()
            pet.rotation_mode = "QUATERNION"
            pet.rotation_quaternion = q
            pet.parent = hub
            pet.location = base - hub.location
        bpy.data.objects.remove(tpl)
    # gekreuzte Knochen hinter der Mähne
    bx = c.x - 1.7
    for sgn in (1, -1):
        a = math.radians(35 * sgn)
        d = Vector((0, math.cos(a), math.sin(a)))
        p0 = Vector((bx, 0, c.z - 0.4)) - d * 5.5
        p1 = Vector((bx, 0, c.z - 0.4)) + d * 5.5
        parts.append(cyl("Bone", p0, p1, 0.42, 0.42, mats["bone"], 20))
        for p in (p0, p1):
            sd = d.cross(Vector((1, 0, 0))).normalized()
            for o in (-0.42, 0.42):
                parts.append(sphere("BoneKnob", p + sd * o, 0.56, mats["bone"]))
    # Gesicht: runde Augen, Nase, lächelnder Mund, Wangen
    f = c + Vector((R * 0.72, 0, 0))
    for sy in (-1, 1):
        parts.append(sphere("Eye", f + Vector((-0.12, 0.62 * sy, 0.42)), 0.26, mats["black"], scale=(0.45, 1, 1.2)))
        parts.append(sphere("EyeHi", f + Vector((-0.02, 0.56 * sy, 0.55)), 0.07, mats["white"]))
        parts.append(cyl("Brow", f + Vector((-0.22, 0.32 * sy, 0.95)), f + Vector((-0.3, 0.9 * sy, 0.88)), 0.07, 0.05,
                         mats["mane_dark"], 10))
    parts.append(sphere("Nose", f + Vector((0.2, 0, 0.02)), 0.34, mats["nose"], scale=(0.75, 1.25, 0.8)))
    pts = [tuple(f + Vector((-0.06 - 0.25 * abs(math.cos(a)) ** 2, 0.72 * math.cos(a), -0.35 + 0.34 * math.sin(a))))
           for a in np.linspace(math.radians(200), math.radians(340), 25)]
    parts.append(curve_tube("Mouth", pts, 0.065, mats["black"]))
    for o in parts:
        o.parent = body
    return hub


# --------------------------------------------------------------------------
# Aufbau
# --------------------------------------------------------------------------

def _flag(name, loc, mats, fw=3.6, fh=2.4):
    bm = bmesh.new()
    uv_l = bm.loops.layers.uv.new("UVMap")
    nx, ny = 24, 16
    grid = [[bm.verts.new((-fw * i / (nx - 1), 0, -fh * j / (ny - 1))) for i in range(nx)] for j in range(ny)]
    for j in range(ny - 1):
        for i in range(nx - 1):
            f = bm.faces.new([grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]])
            for lp, (ii, jj) in zip(f.loops, [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]):
                lp[uv_l].uv = (ii / (nx - 1), 1 - jj / (ny - 1))
    flag = fpv.mesh_from_bmesh(bm, name, mats["flag"])
    flag.location = loc
    wv = flag.modifiers.new("wave", "WAVE")
    wv.use_x = True
    wv.use_y = False
    wv.use_normal = True
    wv.height = 0.22
    wv.width = 1.4
    wv.speed = -0.12
    return flag


def round_room(name, cx, z0, h, r, mats, n_win=10, dome_r=None, dome_h=0.72):
    """Rundes Häuschen: cremefarbene Wand, Fensterkranz, rot-gelbe Kuppel mit Knauf."""
    parts = [cyl(name + "Wall", (cx, 0, z0), (cx, 0, z0 + h), r, r, mats["cream"], 48)]
    for k in range(n_win):
        a = (k + 0.5) * 2 * math.pi / n_win
        d = Vector((math.cos(a), math.sin(a), 0))
        p = Vector((cx, 0, z0 + h * 0.55)) + d * r
        M = local_matrix(p, Vector((-d.y, d.x, 0)), d)
        parts += framed_window(name + "Win", M, 0.55 * r / 1.6, h * 0.42, mats, arch=True, frame="brown")
    dr = dome_r or r * 1.06
    parts.append(cyl(name + "Eave", (cx, 0, z0 + h - 0.05), (cx, 0, z0 + h + 0.12), dr + 0.1, dr + 0.1, mats["brown"], 48))
    dome = sphere(name + "Dome", (cx, 0, z0 + h + 0.1), dr, mats["dome"], scale=(1, 1, dome_h), seg=48, ring=24)
    parts.append(dome)
    parts.append(sphere(name + "Knob", (cx, 0, z0 + h + 0.1 + dr * dome_h + 0.12), 0.18, mats["yellow"]))
    return parts


def build(name="ThousandSunny"):
    root = bpy.data.objects.new(name, None)  # Kurs
    fpv.link(root)
    body = bpy.data.objects.new(name + "_body", None)  # Stampfen/Rollen/Tauchen
    fpv.link(body)
    body.parent = root
    lawn_base, blade = lawn_materials()
    mats = {
        "hull": hull_material(),
        "inner": wood_material("BulwarkInner", dark=0.9, axis="X", board=0.3),
        "deckwood": wood_material("DeckWood", dark=0.95, axis="X", board=0.18),
        "brown": wood_material("TrimWood", dark=0.7),
        "mast": wood_material("MastWood", dark=1.0, board=5.0),
        "cream": paint_material("Cream", CREAM, rough=0.42, under=(0.45, 0.3, 0.16)),
        "red": paint_material("RedPaint", RED, rough=0.4),
        "yellow": paint_material("Yellow", YELLOW, rough=0.4),
        "dome": stripe_dome_material(),
        "lawn": lawn_base,
        "blade": blade,
        "sail": sail_material("Sail"),
        "sail_jr": sail_material("SailJR", textures.jolly_roger_sail()),
        "sail_stripe": sail_material("SailStripe", stripes=5.0),
        "flag": flag_material(),
        "mane": paint_material("Mane", (0.85, 0.20, 0.012), rough=0.45, under=(0.55, 0.3, 0.12)),
        "mane_light": paint_material("ManeLight", (0.92, 0.40, 0.02), rough=0.45, under=(0.55, 0.35, 0.15)),
        "mane_dark": paint_material("ManeDark", (0.66, 0.2, 0.025), rough=0.45, under=(0.45, 0.25, 0.1)),
        "face": paint_material("LionFace", (0.93, 0.56, 0.035), rough=0.4, under=(0.6, 0.45, 0.25)),
        "nose": paint_material("LionNose", (0.40, 0.13, 0.03), rough=0.35),
        "bone": paint_material("Bone", (0.86, 0.83, 0.74), rough=0.5),
        "black": fpv.simple_mat("Black", BLACK, rough=0.3),
        "white": paint_material("WhitePaint", (0.85, 0.83, 0.78), rough=0.38),
        "iron": iron_material(),
        "cannon": metal_material("CannonIron", (0.09, 0.065, 0.045), rough=0.45, metallic=0.7),
        "portring": metal_material("PortRing", (0.42, 0.42, 0.41), rough=0.3),
        "rope": fpv.simple_mat("Rope", (0.24, 0.18, 0.11), rough=0.9),
        "glass": fpv.simple_mat("DarkGlass", (0.012, 0.018, 0.025), rough=0.05),
        "leaf": fpv.simple_mat("TangerineLeaf", (0.03, 0.09, 0.02), rough=0.6),
        "fruit": fpv.simple_mat("Tangerine", (0.85, 0.25, 0.02), rough=0.45),
    }
    P = []
    hull = hull_mesh(mats["hull"], mats["inner"])
    P.append(hull)
    P.append(plan_mesh("LawnDeck", X_CAST, X_FORE, Z_DECK, mats["lawn"], inset=0.25))
    P.append(lawn_blades(mats["blade"]))
    P.append(plan_mesh("ForeDeck", X_FORE, X_B - 0.25, Z_FORE, mats["deckwood"], inset=0.25, thick=0.25))

    # Deckskappe (cremefarben) entlang der U-Bordwand und ums Vorschiff
    xs = np.linspace(X_CAST, X_B, 160)
    G = sheer(xs)
    cap = [(float(x), -float(section_y(x, g)) + 0.02, float(g) + 0.06) for x, g in zip(xs, G)]
    cap += [(float(x), float(section_y(x, g)) - 0.02, float(g) + 0.06) for x, g in zip(xs[::-1], G[::-1])]
    P.append(curve_tube("CapRail", cap, 0.15, mats["cream"]))
    # Voluten an den Armen des U, Bullaugen im roten Band
    for sy in (-1, 1):
        for (X, z, r0) in ((3.35, 7.25, 0.55), (-3.45, 7.0, 0.5)):
            p, tx, tz, n = hull_frame(X, z, sy)
            P.append(volute("Volute", p, tx, n, r0, mats["cream"]))
        for (X, dz) in ((4.9, 1.3), (3.85, 1.25), (-3.9, 1.25), (-4.7, 1.25)):
            z = float(sheer(X)) - dz
            p, tx, tz, n = hull_frame(X, z, sy)
            P += porthole("Port", p, n, 0.36, mats)
        for (X, z) in ((3.0, 3.05), (-4.3, 3.0), (6.9, 5.2), (8.0, 4.6)):
            p, tx, tz, n = hull_frame(X, z, sy)
            P += porthole("PortLow", p, n, 0.3, mats)
        # Soldier-Dock-Ring als Relief (folgt der Rumpfhaut)
        pts = []
        for a in np.linspace(0, 2 * math.pi, 73):
            X, z = SD_X + (SD_R - 0.2) * math.cos(a), SD_Z + (SD_R - 0.2) * math.sin(a)
            p, tx, tz, n = hull_frame(X, z, sy)
            pts.append(tuple(p + n * 0.06))
        P.append(curve_tube("DockRing", pts, 0.2, mats["black"], cyclic=True))
    # Bugschild: gelbe Nieten im Bogen unter dem Löwen
    for a in list(np.linspace(math.radians(200), math.radians(340), 9)) + [math.radians(v) for v in (165, 150, 30, 15)]:
        yy, zz = 4.3 * math.cos(a), 6.2 + 4.3 * math.sin(a)
        fp = hull_front_point(yy, zz)
        if fp is None:
            continue
        p, n = fp
        P.append(sphere("Stud", p + n * 0.05, 0.3, mats["yellow"], scale=(1, 1, 1)))

    # Vorschiff: Rückwand mit Tür und Rundfenstern, Treppen, Steuerrad
    hw_f = float(section_y(X_FORE, Z_FORE)) - 0.25
    P.append(box("ForeWall", (X_FORE - 0.3, -hw_f, Z_DECK), (X_FORE, hw_f, Z_FORE), mats["deckwood"], bevel=0.03))
    M = local_matrix(Vector((X_FORE - 0.3, 0, Z_DECK + 1.15)), Vector((0, 1, 0)), Vector((-1, 0, 0)))
    P += framed_window("ForeDoor", M, 1.3, 2.0, mats, arch=True, frame="brown", bars=False)
    for sy in (-1, 1):
        p = Vector((X_FORE - 0.3, sy * 1.9, Z_DECK + 2.6))
        P += porthole("ForePort", p, Vector((-1, 0, 0)), 0.38, mats)
        P += stairs("ForeStairs", (X_FORE - 3.4, sy * 3.5, Z_DECK), (X_FORE - 0.3, sy * 3.5, Z_FORE), 1.2, 14,
                    mats["deckwood"])
    hx = X_FORE + 1.9
    P.append(box("HelmPost", (hx - 0.25, -0.25, Z_FORE), (hx + 0.25, 0.25, Z_FORE + 1.35), mats["brown"], bevel=0.03))
    wheel = []
    for k in range(10):
        a = k * 2 * math.pi / 10
        d = Vector((0, math.cos(a), math.sin(a)))
        c0 = Vector((hx - 0.32, 0, Z_FORE + 1.45))
        wheel.append((tuple(c0), tuple(c0 + d * 1.0)))
        a2 = (k + 1) * 2 * math.pi / 10
        wheel.append((tuple(c0 + d * 0.78), tuple(c0 + Vector((0, math.cos(a2), math.sin(a2))) * 0.78)))
    P.append(tubes_mesh("HelmWheel", wheel, 0.05, mats["brown"], sides=8))

    # Achterkastell: zwei Stockwerke, cremefarben, braune Gesimse, Bogenfenster; Dach mit Balustrade
    X0c, X1c = X_CAST, X_S + 0.35
    out = castle_outline(X0c, X1c, 4.6, 0.14)
    P.append(prism("AftCastle", out, 4.35, Z_ROOF - 0.05, mats["cream"]))
    P.append(prism("CastleBase", castle_outline(X0c + 0.03, X1c, 4.6, 0.08), 4.3, 4.75, mats["brown"]))
    P.append(prism("CastleBelt", castle_outline(X0c + 0.03, X1c, 6.3, 0.08), 6.15, 6.35, mats["brown"]))
    roof = castle_outline(X0c + 0.05, X_S + 0.05, Z_ROOF, -0.12)
    P.append(prism("CastleRoof", roof, Z_ROOF - 0.08, Z_ROOF + 0.18, mats["deckwood"]))
    for sy in (-1, 1):
        for X in (-6.0, -7.05, -8.1, -9.15):   # Obergeschoss: vier Bogenfenster
            p, tx, n = wall_frame(X, 7.2, sy, 0.14)
            P += framed_window("CastleArch", local_matrix(p, tx, n), 0.72, 0.95, mats, arch=True)
        for (X, w, h, arch) in ((-6.2, 1.2, 0.95, False), (-7.65, 0.95, 1.6, True), (-9.0, 0.8, 0.95, False)):
            p, tx, n = wall_frame(X, 5.35 if not arch else 5.3, sy, 0.14)
            P += framed_window("CastleLow", local_matrix(p, tx, n), w, h, mats, arch=arch,
                               bars=not arch)
    # Front zum Rasendeck: Doppeltür, Fenster oben; Seitentreppen hinauf aufs Kastelldach
    front_n = Vector((1, 0, 0))
    M = local_matrix(Vector((X_CAST + 0.02, 0, 5.35)), Vector((0, -1, 0)), front_n)
    P += framed_window("CastleDoor", M, 1.5, 1.9, mats, arch=True, bars=False)
    for yy in (-3.3, -1.4, 1.4, 3.3):
        M = local_matrix(Vector((X_CAST + 0.02, yy, 7.3)), Vector((0, -1, 0)), front_n)
        P += framed_window("CastleFrontWin", M, 0.8, 0.95, mats, arch=True)
    for sy in (-1, 1):
        P += stairs("AftStairs", (X_CAST + 0.75, sy * 1.25, Z_DECK), (X_CAST + 0.75, sy * 5.0, Z_ROOF), 1.1, 16,
                    mats["deckwood"])
    rp = [(x, y, Z_ROOF + 0.18) for (x, y) in castle_outline(X_CAST - 0.35, X_S + 0.1, Z_ROOF, 0.05, n=18)]
    P += balustrade("RoofRail", rp[:18], 0.9, mats)
    P += balustrade("RoofRail2", rp[18:], 0.9, mats)

    # Heckkanone (Gaon Cannon / Coup de Burst) + Rundturm mit rot-gelber Kuppel darüber
    P.append(cyl("Cannon", (-9.4, 0, 4.0), (-14.6, 0, 4.0), 1.85, 1.7, mats["cannon"], 48))
    for X in (-10.9, -12.4, -13.8):
        P.append(cyl("CannonBand", (X - 0.14, 0, 4.0), (X + 0.14, 0, 4.0), 1.95, 1.95, mats["iron"], 48))
    P.append(cyl("CannonMuzzle", (-14.45, 0, 4.0), (-15.1, 0, 4.0), 2.15, 2.1, mats["brown"], 48))
    P.append(cyl("CannonBore", (-14.2, 0, 4.0), (-15.15, 0, 4.0), 1.3, 1.3, mats["black"], 40))
    TX = -12.0
    P.append(cyl("TowerPedestal", (TX, 0, 5.6), (TX, 0, Z_ROOF - 0.1), 1.0, 2.2, mats["cream"], 40))
    P.append(cyl("TowerFloor", (TX, 0, Z_ROOF - 0.12), (TX, 0, Z_ROOF + 0.18), 2.85, 2.85, mats["deckwood"], 48))
    P += round_room("Tower", TX, Z_ROOF + 0.18, 1.9, 2.4, mats, n_win=10)
    ring = [(TX + 2.75 * math.cos(a), 2.75 * math.sin(a), Z_ROOF + 0.18) for a in np.linspace(0, 2 * math.pi, 33)[:-1]]
    P += balustrade("TowerRail", ring, 0.85, mats, closed=True)

    # Mandarinenbäume (Nami) in Pflanzkästen auf dem Rasendeck
    rng = np.random.default_rng(12)
    for (X, yy) in ((-4.15, -2.6), (-4.15, 2.6), (1.0, -3.9), (1.0, 3.9)):
        P.append(box("Planter", (X - 0.55, yy - 0.55, Z_DECK), (X + 0.55, yy + 0.55, Z_DECK + 0.6), mats["brown"],
                     bevel=0.03))
        P.append(cyl("TTrunk", (X, yy, Z_DECK + 0.5), (X, yy, Z_DECK + 1.5), 0.09, 0.06, mats["brown"], 8))
        bush = sphere("TBush", (X, yy, Z_DECK + 2.0), 0.85, mats["leaf"], scale=(1, 1, 0.85))
        fpv.displace_obj(bush, "CLOUDS", size=0.25, strength=0.22, depth=2, subdiv=1, name="TBush_d")
        P.append(bush)
        fr = []
        for _ in range(26):
            d = Vector(rng.normal(size=3)).normalized()
            fr.append(tuple(Vector((X, yy, Z_DECK + 2.0)) + Vector((d.x * 0.88, d.y * 0.88, d.z * 0.76))))
        P.append(spheres_mesh("Tangerines", fr, 0.075, mats["fruit"], subdiv=1, flatten=1.0))

    # Masten: Fockmast (Rasendeck) mit Ausguck, Großmast durch das Achterkastell
    FT, MT = 19.0, 29.6
    P.append(cyl("ForeMast", (FORE_X, 0, Z_DECK - 0.4), (FORE_X, 0, FT + 0.2), 0.46, 0.34, mats["mast"], 24))
    P.append(cyl("MainMast", (MAIN_X, 0, Z_ROOF - 0.2), (MAIN_X, 0, MT), 0.5, 0.26, mats["mast"], 24))
    for mx, z0, top, r0, r1 in ((FORE_X, Z_DECK, FT, 0.46, 0.34), (MAIN_X, Z_ROOF, MT, 0.5, 0.26)):
        for zz in np.arange(z0 + 1.6, top - 0.8, 2.6):
            f = (zz - z0) / (top - z0)
            rr = r0 + (r1 - r0) * f
            P.append(cyl("MastBand", (mx, 0, zz - 0.08), (mx, 0, zz + 0.08), rr + 0.04, rr + 0.04, mats["iron"], 24))
    # Ausguck auf dem Fockmast: Konus, Plattform mit Reling, Rundhaus, Kuppel
    P.append(cyl("CrowCone", (FORE_X, 0, FT - 0.9), (FORE_X, 0, FT + 0.8), 0.45, 1.7, mats["brown"], 40))
    P.append(cyl("CrowFloor", (FORE_X, 0, FT + 0.75), (FORE_X, 0, FT + 0.95), 2.1, 2.1, mats["deckwood"], 48))
    P += round_room("Crow", FORE_X, FT + 0.95, 1.8, 1.6, mats, n_win=8)
    ring = [(FORE_X + 2.0 * math.cos(a), 2.0 * math.sin(a), FT + 0.95) for a in np.linspace(0, 2 * math.pi, 25)[:-1]]
    P += balustrade("CrowRail", ring, 0.75, mats, closed=True)
    crow_top = FT + 0.95 + 1.8 + 0.1 + 1.7 * 0.72
    P.append(cyl("ForePole", (FORE_X, 0, crow_top), (FORE_X, 0, crow_top + 2.6), 0.07, 0.05, mats["iron"], 8))
    P.append(_flag("ForeFlag", (FORE_X - 0.1, 0, crow_top + 2.55), mats, fw=2.6, fh=1.75))
    P.append(cyl("MainPole", (MAIN_X, 0, MT), (MAIN_X, 0, MT + 2.9), 0.08, 0.06, mats["iron"], 8))
    P.append(_flag("MainFlag", (MAIN_X - 0.1, 0, MT + 2.85), mats))
    P.append(cyl("MainTop", (MAIN_X, 0, 25.7), (MAIN_X, 0, 25.9), 1.2, 1.2, mats["deckwood"], 32))

    # Rahen und Segel (Model Sheet: Fock ~20 m breit mit Jolly Roger, Großsegel ~14 m, Gaffel rot-schwarz)
    yards = [(FORE_X, FT - 0.25, 10.0), (MAIN_X, 25.5, 7.2)]
    for (x, z, hw) in yards:
        for sy in (-1, 1):
            P.append(cyl("Yard", (x + 0.5, 0, z), (x + 0.5, sy * hw, z), 0.26, 0.13, mats["mast"], 12))
    s_fore, e_fore = sail_mesh("ForeSail", 19.4, 10.6, 3.1, mats["sail_jr"], foot_arch=0.9, rake=2.2)
    s_fore.location = (FORE_X + 0.75, 0, FT - 0.45)
    P.append(s_fore)
    s_main, e_main = sail_mesh("MainSail", 13.8, 6.2, 1.9, mats["sail"], foot_arch=0.5, rake=0.6)
    s_main.location = (MAIN_X + 0.75, 0, 25.3)
    P.append(s_main)
    gaff = gaff_sail_mesh("GaffSail", luff=6.1, gaff_len=5.6, boom_len=5.6, gaff_rise=0.2, mat=mats["sail_stripe"],
                          bulge=0.8)
    gaff.location = (MAIN_X - 0.5, 0, 12.25)
    P.append(gaff)
    P.append(cyl("Boom", (MAIN_X - 0.4, 0, 12.2), (MAIN_X - 6.3, 0, 12.2), 0.17, 0.12, mats["mast"], 12))
    P.append(cyl("Gaff", (MAIN_X - 0.4, 0, 18.4), (MAIN_X - 6.3, 0, 18.6), 0.15, 0.1, mats["mast"], 12))

    # Takelage: Liektaue, Schoten, Wanten mit Webeleinen, Stagen
    ropes = []
    for (ep, loc) in ((e_fore, s_fore.location), (e_main, s_main.location)):
        loc = Vector(loc)
        for e in ep:
            for a, b in zip(e[:-1], e[1:]):
                ropes.append((tuple(Vector(a) + loc), tuple(Vector(b) + loc)))
    for c in (Vector(e_fore[1][0]) + Vector(s_fore.location), Vector(e_fore[1][-1]) + Vector(s_fore.location)):
        X = -2.9
        tgt = Vector((X, np.sign(c.y) * float(section_y(X, 7.0)), float(sheer(X)) + 0.1))
        ropes += catenary(tuple(c), tuple(tgt), 0.2, 8)
    for sy in (-1, 1):
        for (mx, top, xs_, zfn) in ((FORE_X, FT - 0.9, (1.3, 2.1, 2.9, 3.7), None),
                                    (MAIN_X, 25.0, (-6.6, -7.4, -8.2, -9.0), Z_ROOF + 0.2)):
            chain = []
            for X in xs_:
                if zfn is None:
                    z = float(sheer(X)) + 0.1
                    yw = float(section_y(X, z - 0.1)) + 0.05
                else:
                    z = zfn
                    yw = float(section_y(X, 4.6)) - 0.1
                a = (X, sy * yw, z)
                b = (mx, 0.35 * sy, top)
                ropes.append((a, b))
                chain.append((np.array(a), np.array(b)))
            for (a1, b1), (a2, b2) in zip(chain[:-1], chain[1:]):
                for t in np.arange(0.05, 0.85, 0.45 / np.linalg.norm(b1 - a1)):
                    ropes.append((tuple(a1 + (b1 - a1) * t), tuple(a2 + (b2 - a2) * t)))
    ropes.append(((FORE_X, 0, crow_top - 1.4), (MAIN_X, 0, MT - 0.5)))
    ropes.append(((X_B - 0.4, 0, 9.7), (FORE_X, 0, FT - 1.0)))
    P.append(tubes_mesh("Rigging", ropes, 0.03, mats["rope"], sides=6))

    # Galionsfigur
    lion_head(body, mats)

    for p in P:
        if p.parent is None:
            p.parent = body
    return root, body, mats


def animate(root, body, heading_deg, start, speed, fps, frames, pitch_amp=1.2, roll_amp=2.0, heave_amp=0.25):
    h = math.radians(heading_deg)
    d = Vector((math.cos(h), math.sin(h), 0))
    root.rotation_euler = (0, 0, h)
    body.rotation_mode = "XYZ"
    for f in range(0, frames + 2):
        t = (f - 1) / fps
        root.location = Vector(start) + d * speed * t
        root.keyframe_insert("location", frame=f)
        body.location = (0, 0, heave_amp * math.sin(2 * math.pi * 0.13 * t + 0.4) - 0.15)
        body.rotation_euler = (math.radians(roll_amp) * math.sin(2 * math.pi * 0.085 * t + 1.1),
                               math.radians(pitch_amp) * math.sin(2 * math.pi * 0.11 * t), 0)
        body.keyframe_insert("location", frame=f)
        body.keyframe_insert("rotation_euler", frame=f)
