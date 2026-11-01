# Weltenbrück: Gründergesichter, Stadtrequisiten, Möbel, Fassadenteile,
# Landmarken und Vegetation.
# Ursprung = Bodenmitte, Z = oben. Abweichend von m_world zeigt die
# Vorderseite hier nach -Y (im Spiel +Z), damit Fassadenteile mit derselben
# Drehung wie die Wand nach außen zeigen. Teile namens "...__Paint" sind
# hell und werden im Spiel pro Instanz eingefärbt.
import math
import random

import bmesh
from mathutils import Vector, noise

import kit
from kit import Asset
from m_items import _lamp

R = math.radians

kit.PALETTE.update({
    'stone_face': ('#aaa392', 'matte', 0.95, 0, 0),
    'stone_face2': ('#9b9483', 'matte', 0.95, 0, 0),
    'stone_dark': ('#34302b', 'matte', 0.95, 0, 0),
    'stone_light': ('#c4baa4', 'matte', 0.9, 0, 0),
    'sandstone': ('#b8a688', 'matte', 0.9, 0, 0),
    'brick': ('#8e4a36', 'matte', 0.9, 0, 0),
    'tile': ('#9a5638', 'matte', 0.8, 0, 0),
    'slate': ('#4c5056', 'matte', 0.6, 0.05, 0),
    'iron': ('#2c2a28', 'metal', 0.55, 0.6, 0),
    'paint': ('#ececec', 'matte', 0.7, 0, 0),
    'patina': ('#5f8a78', 'metal', 0.5, 0.6, 0),
    'ceramic_green': ('#3f6b4a', 'matte', 0.25, 0, 0),
    'linen': ('#d8cfb8', 'matte', 0.9, 0, 0),
    'straw': ('#c9b06a', 'matte', 0.9, 0, 0),
    'birch': ('#dcd8cc', 'matte', 0.85, 0, 0),
    'leaves3': ('#45652a', 'matte', 0.85, 0, 0),
    'leaves4': ('#6a8a36', 'matte', 0.85, 0, 0),
    'bloom_red': ('#c8322a', 'matte', 0.8, 0, 0),
    'bloom_blue': ('#4a6ac8', 'matte', 0.8, 0, 0),
    'bloom_white': ('#ece8dc', 'matte', 0.8, 0, 0),
    'bloom_yellow': ('#e0b52a', 'matte', 0.8, 0, 0),
    'veg_red': ('#a8392e', 'matte', 0.7, 0, 0),
    'veg_orange': ('#d9782c', 'matte', 0.7, 0, 0),
    'veg_green': ('#5a8a3a', 'matte', 0.7, 0, 0),
    'core': ('#9ff8ff', 'emit', 0.2, 0, 3.0),
})


# ---------------------------------------------------------------- Hilfen
def weather(part, amp, freq, seed=0):
    """Verwittert eine ganze Teilgeometrie: Ecken entlang der Normalen verschieben."""
    bm = part.bm
    bm.normal_update()
    off = Vector((seed * 13.1, seed * 7.7, seed * 3.3))
    for v in bm.verts:
        n = noise.noise(v.co * freq + off) + 0.5 * noise.noise(v.co * freq * 2.7 + off)
        v.co += v.normal * n * amp


def sphere_cut(part, r, loc, scale, mat, cut, seg=32, rings=20):
    """Kugel, aus der alle Ecken mit cut(co) == True entfernt werden (z. B. Kapuze)."""
    tb = bmesh.new()
    bmesh.ops.create_uvsphere(tb, u_segments=seg, v_segments=rings, radius=r)
    kill = [v for v in tb.verts if cut(v.co)]
    bmesh.ops.delete(tb, geom=kill, context='VERTS')
    part._merge(tb, loc, (0, 0, 0), scale, mat, 0, True)


# ================================================================ Gründergesichter
# 20 m hoch, Ursprung am Sockel, Blick nach -Y.
def _head(p, S='stone_face', D='stone_dark'):
    p.sphere(1.0, loc=(0, 0.6, 10.6), mat=S, seg=40, rings=28, scale=(6.4, 6.2, 8.6))
    p.sphere(1.0, loc=(0, -1.5, 6.6), mat=S, seg=28, rings=16, scale=(5.4, 4.6, 4.2))   # Kiefer
    p.sphere(1.0, loc=(0, -4.4, 4.9), mat=S, seg=24, rings=14, scale=(2.7, 1.7, 1.9))   # Kinn
    p.sphere(1.0, loc=(0, -4.3, 13.1), mat=S, seg=28, rings=12, scale=(5.6, 1.7, 1.25))  # Brauen
    for sx in (-1, 1):
        p.sphere(1.0, loc=(sx * 2.3, -4.95, 11.6), mat=D, seg=20, rings=12, scale=(1.5, 0.7, 1.05))
        p.sphere(1.0, loc=(sx * 2.3, -5.25, 11.5), mat=S, seg=20, rings=12, scale=(0.95, 0.55, 0.62))
        p.sphere(1.0, loc=(sx * 3.3, -3.7, 8.9), mat=S, seg=20, rings=12, scale=(2.2, 1.9, 2.5))  # Wangen
        p.sphere(1.0, loc=(sx * 6.2, 0.6, 10.6), mat=S, seg=16, rings=10, scale=(0.9, 1.6, 2.3))  # Ohren
    # Nase: Profil (y, z) extrudiert entlang X
    p.prism([(-5.2, 12.2), (-7.5, 8.7), (-6.6, 8.1), (-5.0, 8.1)], 1.7, loc=(0, 0, 0), mat=S, plane='YZ', bevel=0.25)
    for sx in (-1, 1):
        p.sphere(1.0, loc=(sx * 0.95, -6.25, 8.45), mat=S, seg=14, rings=10, scale=(0.65, 0.65, 0.55))
        p.sphere(1.0, loc=(sx * 0.6, -6.7, 8.15), mat=D, seg=10, rings=6, scale=(0.3, 0.2, 0.18))
    # Mund
    p.sphere(1.0, loc=(0, -5.95, 7.15), mat=S, seg=20, rings=10, scale=(1.95, 0.6, 0.45))
    p.sphere(1.0, loc=(0, -5.8, 6.4), mat=S, seg=20, rings=10, scale=(1.75, 0.6, 0.45))
    p.box((2.7, 0.25, 0.14), loc=(0, -6.25, 6.78), mat=D)
    # Hals und Felsensockel
    p.cyl(4.8, 4.0, loc=(0, 1.0, 2.0), mat='stone_face2', seg=20, r2=4.2)


def _base_rocks(p, seed):
    rnd = random.Random(seed)
    for i in range(14):
        a = rnd.uniform(0, math.pi * 2)
        r = rnd.uniform(4, 8)
        p.ico(rnd.uniform(1.5, 3.2), loc=(math.cos(a) * r, 2 + math.sin(a) * r * 0.6, rnd.uniform(0, 3)), mat='stone_face2', sub=1, jitter=0.4, seed=seed + i)


def f_seefahrer(col):
    a = Asset('f_seefahrer', col)
    p = a.part('Stone')
    _head(p)
    # Dreispitz
    p.cyl(7.4, 0.55, loc=(0, 0.6, 17.4), mat='stone_face2', seg=36, r2=7.0)
    p.cyl(4.9, 3.4, loc=(0, 0.6, 19.2), mat='stone_face2', seg=28, r2=4.2)
    for k, ang in enumerate((-90, 30, 150)):
        aa = R(ang)
        p.box((7.0, 0.7, 2.6), loc=(math.cos(aa) * 5.4, 0.6 + math.sin(aa) * 5.4, 18.4), rot=(R(-35), 0, aa + R(90)), mat='stone_face2', bevel=0.25)
    # Vollbart und Schnurrbart
    rnd = random.Random(3)
    for i in range(26):
        t = i / 25
        ang = R(-160 + t * 140)
        rr = 5.0 - abs(t - 0.5) * 2.0
        p.ico(rnd.uniform(1.0, 1.6), loc=(math.cos(ang) * rr * 0.9, -2.0 + math.sin(ang) * rr * 0.75, 4.2 + rnd.uniform(-1.4, 2.2)), mat='stone_face', sub=1, jitter=0.35, seed=40 + i)
    for sx in (-1, 1):
        p.sphere(1.0, loc=(sx * 1.4, -6.35, 7.55), rot=(0, R(sx * 20), 0), mat='stone_face', seg=16, rings=8, scale=(1.6, 0.55, 0.5))
    _base_rocks(p, 100)
    weather(p, 0.16, 0.45, 1)
    p.paint_top('moss', 0.86)
    a.add(p)
    return a


def f_schatten(col):
    a = Asset('f_schatten', col)
    p = a.part('Stone')
    _head(p)
    # Kapuze: vorn offen
    sphere_cut(p, 1.0, (0, 1.4, 11.4), (7.8, 7.6, 9.8), 'stone_face2', lambda co: co.y < -0.5 and co.z < 0.75)
    p.box((2.6, 3.0, 2.0), loc=(0, 6.0, 4.6), rot=(R(20), 0, 0), mat='stone_face2', bevel=0.6)
    # Tuchmaske über Mund und Nase
    p.sphere(1.0, loc=(0, -5.2, 6.9), mat='stone_face2', seg=24, rings=14, scale=(4.7, 1.7, 2.5))
    for k in range(4):
        p.box((4.2 - k * 0.4, 0.25, 0.12), loc=(0, -6.85 + k * 0.05, 6.0 + k * 0.6), mat='stone_dark')
    # Stirnband mit Metallplatte
    p.torus(6.35, 0.38, loc=(0, 0.6, 14.3), mat='stone_face2', seg=36, rseg=8)
    p.box((3.4, 0.5, 1.4), loc=(0, -5.95, 14.3), rot=(R(-12), 0, 0), mat='iron', bevel=0.12)
    p.box((1.6, 0.2, 0.18), loc=(0, -6.25, 14.35), rot=(R(-12), 0, 0), mat='stone_dark')
    _base_rocks(p, 200)
    weather(p, 0.14, 0.45, 2)
    p.paint_top('moss', 0.86)
    a.add(p)
    return a


def f_erfinder(col):
    a = Asset('f_erfinder', col)
    p = a.part('Stone')
    _head(p)
    # Wildes Haar
    rnd = random.Random(7)
    for i in range(30):
        ang = rnd.uniform(0, math.pi * 2)
        el = rnd.uniform(0.2, 1.1)
        r = 6.2
        p.ico(rnd.uniform(1.2, 2.0), loc=(math.cos(ang) * r * math.cos(el) * 0.9, 1.5 + math.sin(ang) * r * math.cos(el) * 0.9, 13.5 + math.sin(el) * 5.5), mat='stone_face2', sub=1, jitter=0.5, seed=70 + i)
    # Schnauzbart, gezwirbelt
    for sx in (-1, 1):
        p.sphere(1.0, loc=(sx * 1.5, -6.4, 7.6), rot=(0, R(sx * -18), 0), mat='stone_face', seg=16, rings=8, scale=(1.7, 0.5, 0.45))
        p.torus(0.45, 0.18, loc=(sx * 3.2, -6.0, 7.9), rot=(R(90), 0, 0), mat='stone_face', seg=12, rseg=6)
    weather(p, 0.15, 0.45, 3)
    p.paint_top('moss', 0.86)
    a.add(p)
    # Schutzbrille mit leuchtenden Gläsern (Metall)
    g = a.part('Goggles')
    for sx in (-1, 1):
        g.torus(1.35, 0.38, loc=(sx * 1.8, -5.7, 14.6), rot=(R(80), 0, 0), mat='bronze', seg=20, rseg=8)
    g.torus(6.4, 0.3, loc=(0, 0.6, 14.6), mat='iron', seg=36, rseg=6)
    g.box((0.9, 0.4, 0.4), loc=(0, -6.0, 14.6), mat='bronze')
    a.add(g)
    l = a.part('Lamp')
    for sx in (-1, 1):
        l.cyl(1.05, 0.2, loc=(sx * 1.8, -5.85, 14.6), rot=(R(80), 0, 0), mat='core', seg=20)
    a.add(l)
    return a


def f_waechter(col):
    a = Asset('f_waechter', col)
    p = a.part('Stone')
    _head(p)
    # Helm mit Nasenschutz, Kamm und Wangenklappen
    p.sphere(1.0, loc=(0, 0.6, 12.6), mat='stone_face2', seg=36, rings=20, scale=(7.0, 6.8, 6.6), hemi=True)
    p.torus(6.9, 0.45, loc=(0, 0.6, 12.6), mat='stone_face2', seg=40, rseg=8)
    p.box((0.9, 0.6, 4.6), loc=(0, -6.6, 11.6), rot=(R(-8), 0, 0), mat='stone_face2', bevel=0.2)
    p.box((0.8, 10.5, 2.6), loc=(0, 0.8, 19.6), mat='stone_face2', bevel=0.3)
    for sx in (-1, 1):
        p.box((1.0, 3.8, 6.0), loc=(sx * 5.7, -2.2, 9.6), rot=(0, 0, R(sx * -12)), mat='stone_face2', bevel=0.35)
    # Zopf
    for i in range(7):
        p.sphere(1.0, loc=(5.6, 3.2 - i * 0.15, 9.5 - i * 1.15), mat='stone_face', seg=12, rings=8, scale=(0.9, 0.9, 0.75))
    _base_rocks(p, 400)
    weather(p, 0.15, 0.45, 4)
    p.paint_top('moss', 0.86)
    a.add(p)
    return a


# ================================================================ Fassadenteile
# Fenster: 1 x 1 m nominell (wird auf die Öffnung skaliert), Ursprung unten mittig in der Außenfläche.
# Nur Flächen (im Spiel doppelseitig), damit tausende Fenster wenig kosten.
def _quad_y(p, x0, x1, z0, z1, y, mat):
    p.face([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], mat=mat)


def kc_window(col):
    a = Asset('kc_window', col)
    p = a.part('Paint')
    d = 0.1  # Rahmen sitzt in der Laibung
    f = 0.07
    _quad_y(p, -0.5, -0.5 + f, 0, 1, d, 'paint')
    _quad_y(p, 0.5 - f, 0.5, 0, 1, d, 'paint')
    _quad_y(p, -0.5 + f, 0.5 - f, 1 - f, 1, d, 'paint')
    _quad_y(p, -0.5 + f, 0.5 - f, 0, f, d, 'paint')
    _quad_y(p, -0.025, 0.025, f, 1 - f, d + 0.01, 'paint')
    _quad_y(p, -0.5 + f, 0.5 - f, 0.615, 0.665, d + 0.01, 'paint')
    # Fensterbank (Oberseite und Stirn)
    p.face([(-0.56, -0.08, 0), (0.56, -0.08, 0), (0.56, 0.12, 0), (-0.56, 0.12, 0)], mat='paint')
    p.face([(-0.56, -0.08, -0.07), (0.56, -0.08, -0.07), (0.56, -0.08, 0), (-0.56, -0.08, 0)], mat='paint')
    a.add(p)
    return a


# Fensterladen: ein Flügel 0,5 x 1 m, Scharnier bei x = 0, Flügel nach +X
def kc_shutter(col):
    a = Asset('kc_shutter', col)
    p = a.part('Paint')
    _quad_y(p, 0.0, 0.48, 0.01, 0.99, -0.04, 'paint')
    for z in (0.18, 0.82):
        _quad_y(p, 0.02, 0.46, z - 0.045, z + 0.045, -0.06, 'paint')
    # Strebe (schräg)
    p.face([(0.04, -0.06, 0.2), (0.1, -0.06, 0.2), (0.46, -0.06, 0.78), (0.4, -0.06, 0.78)], mat='paint')
    for x in (0.16, 0.32):
        _quad_y(p, x - 0.006, x + 0.006, 0.01, 0.99, -0.045, 'paint')
    a.add(p)
    return a


# Tür: Flügel 1 x 1 m nominell, Scharnier bei x = 0, Flügel nach +X
def kc_door(col):
    a = Asset('kc_door', col)
    p = a.part('Paint')
    p.box((0.98, 0.06, 0.99), loc=(0.5, 0, 0.5), mat='paint')
    for x in (0.2, 0.4, 0.6, 0.8):
        p.box((0.012, 0.064, 0.98), loc=(x, 0, 0.5), mat='wood_dark')
    a.add(p)
    m = a.part('Iron')
    for z in (0.15, 0.5, 0.85):
        m.box((0.85, 0.075, 0.04), loc=(0.43, 0, z), mat='iron')
    m.torus(0.06, 0.012, loc=(0.85, -0.05, 0.47), rot=(R(90), 0, 0), mat='iron', seg=8, rseg=3)
    a.add(m)
    return a


def kc_chimney(col):
    a = Asset('kc_chimney', col)
    p = a.part('M')
    p.box((0.62, 0.62, 2.0), loc=(0, 0, 1.0), mat='brick')
    p.box((0.78, 0.78, 0.12), loc=(0, 0, 2.0), mat='stone_light')
    for sx in (-1, 1):
        p.box((0.1, 0.5, 0.35), loc=(sx * 0.2, 0, 2.23), mat='brick')
    p.box((0.62, 0.62, 0.06), loc=(0, 0, 2.43), mat='slate')
    a.add(p)
    return a


# Gaube: 1,6 m breit, Front bei y = 0, Körper ragt nach +Y ins Dach
def kc_dormer(col):
    a = Asset('kc_dormer', col)
    p = a.part('M')
    p.box((1.6, 2.4, 1.4), loc=(0, 1.2, 0.2), mat='paint_cream')
    p.box((0.9, 0.05, 0.8), loc=(0, -0.01, 0.45), mat='stone_dark')
    p.box((1.0, 0.07, 0.07), loc=(0, -0.03, 0.85), mat='white')
    p.box((1.0, 0.07, 0.07), loc=(0, -0.03, 0.05), mat='white')
    p.box((0.05, 0.07, 0.8), loc=(0, -0.03, 0.45), mat='white')
    p.prism([(-1.0, 0.9), (1.0, 0.9), (0, 1.85)], 2.8, loc=(0, 1.2, 0), rot=(0, 0, R(90)), mat='tile', plane='YZ')
    a.add(p)
    return a


# Ausleger mit Ladenschild: Wandbefestigung bei y = 0, ragt nach -Y
def kc_sign(col):
    a = Asset('kc_sign', col)
    m = a.part('Iron')
    m.box((0.05, 1.1, 0.05), loc=(0, -0.55, 0.62), mat='iron')
    m.beam((0, -0.05, 0.1), (0, -0.7, 0.6), 0.025, mat='iron', square=True)
    m.torus(0.12, 0.015, loc=(0, -1.0, 0.72), rot=(0, R(90), 0), mat='iron', seg=12, rseg=4)
    a.add(m)
    p = a.part('Paint')
    p.box((0.05, 0.7, 0.55), loc=(0, -0.65, 0.3), mat='paint', bevel=0.02)
    a.add(p)
    return a


# Markise: 2,8 m breit, Wand bei y = 0, schräg nach -Y
def kc_awning(col):
    a = Asset('kc_awning', col)
    p = a.part('Paint')
    p.box((2.8, 1.3, 0.04), loc=(0, -0.6, 0.0), rot=(R(-28), 0, 0), mat='paint')
    p.box((2.8, 0.03, 0.26), loc=(0, -1.18, -0.4), mat='paint')
    a.add(p)
    m = a.part('Iron')
    for sx in (-1.3, 1.3):
        m.beam((sx, -0.02, -0.2), (sx, -1.15, -0.3), 0.02, mat='iron', seg=6)
    a.add(m)
    return a


# ================================================================ Möbel und Behälter
def kc_table(col):
    a = Asset('kc_table', col)
    p = a.part('M')
    p.box((1.4, 0.8, 0.06), loc=(0, 0, 0.73), mat='wood_light', bevel=0.01)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.box((0.07, 0.07, 0.7), loc=(sx * 0.6, sy * 0.32, 0.35), mat='wood')
        p.box((0.06, 0.6, 0.06), loc=(sx * 0.6, 0, 0.18), mat='wood')
    p.box((1.2, 0.05, 0.05), loc=(0, 0, 0.18), mat='wood')
    # etwas Geschirr
    p.cyl(0.08, 0.12, loc=(0.3, 0.1, 0.82), mat='ceramic_white', seg=10)
    p.cyl(0.13, 0.03, loc=(-0.25, -0.1, 0.775), mat='ceramic_white', seg=12)
    a.add(p)
    return a


def kc_chair(col):
    a = Asset('kc_chair', col)
    p = a.part('M')
    p.box((0.44, 0.42, 0.05), loc=(0, 0, 0.46), mat='wood_light')
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.box((0.04, 0.04, 0.45), loc=(sx * 0.18, sy * 0.17, 0.225), mat='wood')
    p.box((0.44, 0.04, 0.5), loc=(0, 0.19, 0.74), mat='wood')
    for x in (-0.12, 0, 0.12):
        p.box((0.06, 0.045, 0.34), loc=(x, 0.19, 0.74), mat='wood_dark')
    a.add(p)
    return a


def kc_bed(col):
    a = Asset('kc_bed', col)
    p = a.part('M')
    p.box((1.0, 2.0, 0.3), loc=(0, 0, 0.25), mat='wood')
    p.box((1.0, 0.07, 1.0), loc=(0, 0.98, 0.5), mat='wood_dark', bevel=0.01)
    p.box((1.0, 0.07, 0.6), loc=(0, -0.98, 0.3), mat='wood_dark', bevel=0.01)
    p.box((0.92, 1.9, 0.16), loc=(0, 0, 0.47), mat='linen', bevel=0.04)
    p.box((0.7, 0.35, 0.12), loc=(0, 0.72, 0.6), mat='cloth', bevel=0.05)
    a.add(p)
    q = a.part('Paint')
    q.box((0.96, 1.2, 0.06), loc=(0, -0.3, 0.58), rot=(R(2), 0, 0), mat='paint', bevel=0.02)
    a.add(q)
    return a


def kc_shelf(col):
    a = Asset('kc_shelf', col)
    p = a.part('M')
    for sx in (-1, 1):
        p.box((0.04, 0.38, 1.9), loc=(sx * 0.58, 0, 0.95), mat='wood')
    for z in (0.05, 0.5, 0.95, 1.4, 1.88):
        p.box((1.2, 0.38, 0.04), loc=(0, 0, z), mat='wood')
    rnd = random.Random(5)
    cols = ['red', 'teal', 'wood_dark', 'paint_cream', 'olive', 'paint_blue']
    for z in (0.07, 0.52, 0.97, 1.42):
        x = -0.52
        while x < 0.5:
            w = rnd.uniform(0.04, 0.09)
            h = rnd.uniform(0.22, 0.36)
            if rnd.random() < 0.85:
                p.box((w, 0.24, h), loc=(x + w / 2, 0, z + h / 2), rot=(0, rnd.uniform(-0.1, 0.1), 0), mat=rnd.choice(cols))
            x += w + 0.01
    a.add(p)
    return a


def kc_stove(col):
    a = Asset('kc_stove', col)
    p = a.part('M')
    p.box((1.0, 0.8, 1.5), loc=(0, 0, 0.75), mat='ceramic_green', bevel=0.03)
    for z in (0.4, 0.8, 1.2):
        p.box((1.03, 0.83, 0.04), loc=(0, 0, z), mat='ceramic_white')
    p.box((1.1, 0.9, 0.12), loc=(0, 0, 1.56), mat='ceramic_white', bevel=0.02)
    p.box((0.6, 0.5, 0.35), loc=(0, 0, 1.79), mat='ceramic_green', bevel=0.03)
    p.box((0.3, 0.05, 0.25), loc=(0, -0.41, 0.35), mat='iron')
    p.box((1.6, 0.4, 0.06), loc=(0.2, -0.6, 0.45), mat='wood')
    a.add(p)
    return a


def c_truhe(col):
    a = Asset('c_truhe', col)
    p = a.part('Body')
    W, D, H = 1.05, 0.62, 0.6
    p.box((W, D, H * 0.72), loc=(0, 0, H * 0.36), mat='wood', bevel=0.02)
    p.cyl(D / 2, W, loc=(0, 0, H * 0.72), rot=(0, R(90), 0), mat='wood', seg=14)
    for x in (-0.38, 0, 0.38):
        p.box((0.06, D + 0.02, H * 0.72), loc=(x, 0, H * 0.36), mat='iron')
        p.torus(D / 2 + 0.005, 0.02, loc=(x, 0, H * 0.72), rot=(0, R(90), 0), mat='iron', seg=16, rseg=4)
    p.box((0.12, 0.04, 0.16), loc=(0, -D / 2 - 0.01, H * 0.62), mat='iron')
    a.add(p)
    _lamp(a, (0.16, 0.02, 0.05), (0.25, -D / 2 - 0.02, H * 0.3))
    return a


def c_schrank(col):
    a = Asset('c_schrank', col)
    p = a.part('Body')
    W, D, H = 1.15, 0.55, 1.95
    p.box((W, D, H), loc=(0, 0, H / 2), mat='wood', bevel=0.015)
    p.box((W + 0.08, D + 0.06, 0.1), loc=(0, 0, H - 0.05), mat='wood_dark', bevel=0.02)
    p.box((W + 0.04, D + 0.04, 0.12), loc=(0, 0, 0.06), mat='wood_dark')
    for sx in (-1, 1):
        p.box((W / 2 - 0.08, 0.03, H - 0.4), loc=(sx * W / 4, -D / 2 - 0.01, H / 2), mat='wood_light', bevel=0.01)
        p.box((0.03, 0.04, 0.16), loc=(sx * 0.06, -D / 2 - 0.035, H * 0.52), mat='iron')
    a.add(p)
    _lamp(a, (0.16, 0.02, 0.05), (0, -D / 2 - 0.035, H * 0.25))
    return a


def c_kommode(col):
    a = Asset('c_kommode', col)
    p = a.part('Body')
    W, D, H = 1.1, 0.5, 0.9
    p.box((W, D, H), loc=(0, 0, H / 2), mat='wood_dark', bevel=0.015)
    p.box((W + 0.06, D + 0.04, 0.05), loc=(0, 0, H + 0.02), mat='wood', bevel=0.01)
    for i in range(3):
        z = 0.18 + i * 0.26
        p.box((W - 0.12, 0.03, 0.22), loc=(0, -D / 2 - 0.01, z), mat='wood', bevel=0.008)
        for sx in (-1, 1):
            p.sphere(0.025, loc=(sx * 0.25, -D / 2 - 0.035, z), mat='bronze', seg=8, rings=6)
    a.add(p)
    _lamp(a, (0.16, 0.02, 0.05), (0, -D / 2 - 0.03, H - 0.06))
    return a


# ================================================================ Straßenrequisiten
def kc_lantern(col):
    a = Asset('kc_lantern', col)
    p = a.part('Iron')
    p.cyl(0.18, 0.35, loc=(0, 0, 0.175), mat='iron', seg=8, r2=0.12)
    p.cyl(0.07, 3.1, loc=(0, 0, 1.9), mat='iron', seg=8, r2=0.055)
    for z in (0.5, 3.3):
        p.torus(0.08, 0.025, loc=(0, 0, z), mat='iron', seg=10, rseg=4)
    # Laternengehäuse
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.box((0.03, 0.03, 0.45), loc=(sx * 0.14, sy * 0.14, 3.72), mat='iron')
    p.box((0.34, 0.34, 0.05), loc=(0, 0, 3.47), mat='iron')
    p.cyl(0.27, 0.32, loc=(0, 0, 4.1), mat='iron', seg=4, r2=0.02)
    p.torus(0.06, 0.015, loc=(0, 0, 4.3), rot=(R(90), 0, 0), mat='iron', seg=10, rseg=4)
    a.add(p)
    l = a.part('Lamp')
    l.box((0.24, 0.24, 0.38), loc=(0, 0, 3.72), mat='lamp')
    a.add(l)
    return a


def kc_bench(col):
    a = Asset('kc_bench', col)
    p = a.part('M')
    for i in range(3):
        p.box((1.8, 0.12, 0.04), loc=(0, -0.14 + i * 0.14, 0.45), mat='wood', bevel=0.005)
    for i in range(2):
        p.box((1.8, 0.04, 0.11), loc=(0, 0.22, 0.65 + i * 0.16), rot=(R(-12), 0, 0), mat='wood', bevel=0.005)
    m = a.part('Iron')
    for sx in (-0.75, 0.75):
        m.box((0.05, 0.42, 0.04), loc=(sx, 0, 0.42), mat='iron')
        m.box((0.05, 0.04, 0.44), loc=(sx, -0.17, 0.22), mat='iron')
        m.box((0.05, 0.04, 0.9), loc=(sx, 0.2, 0.45), rot=(R(-12), 0, 0), mat='iron')
    a.add(p)
    a.add(m)
    return a


# Marktstand 3,6 x 2,2 m, Markise eingefärbt
def kc_stall(col):
    a = Asset('kc_stall', col)
    p = a.part('M')
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.box((0.1, 0.1, 2.5), loc=(sx * 1.7, sy * 1.0, 1.25), mat='wood_dark')
    p.box((3.3, 1.5, 0.9), loc=(0, 0.1, 0.45), mat='wood', bevel=0.02)
    p.box((3.4, 1.6, 0.05), loc=(0, 0.1, 0.92), mat='wood_light')
    rnd = random.Random(11)
    for i in range(10):
        x = -1.4 + i * 0.3
        mat = rnd.choice(['veg_red', 'veg_orange', 'veg_green', 'straw', 'bloom_yellow'])
        p.box((0.28, 0.4, 0.16), loc=(x, -0.3, 1.02), mat='wood_light')
        for k in range(3):
            p.sphere(0.06, loc=(x + rnd.uniform(-0.08, 0.08), -0.3 + rnd.uniform(-0.12, 0.12), 1.12), mat=mat, seg=8, rings=6)
    p.cyl(0.25, 0.5, loc=(1.3, 0.6, 1.2), mat='wood', seg=12)
    a.add(p)
    q = a.part('Paint')
    q.box((3.8, 2.6, 0.04), loc=(0, 0, 2.45), rot=(R(-10), 0, 0), mat='paint')
    q.box((3.8, 0.03, 0.3), loc=(0, -1.29, 2.08), mat='paint')
    a.add(q)
    return a


def kc_cart(col):
    a = Asset('kc_cart', col)
    p = a.part('M')
    p.box((2.4, 1.3, 0.1), loc=(0, 0, 0.6), mat='wood')
    for sy in (-1, 1):
        p.box((2.4, 0.06, 0.45), loc=(0, sy * 0.62, 0.85), mat='wood_light')
    for sx in (-1, 1):
        p.box((0.06, 1.3, 0.45), loc=(sx * 1.17, 0, 0.85), mat='wood_light')
    for sy in (-0.3, 0.3):
        p.beam((1.2, sy, 0.55), (2.9, sy * 0.4, 0.42), 0.04, mat='wood', seg=6)
    for sx in (-0.7, 0.7):
        for sy in (-1, 1):
            p.torus(0.48, 0.05, loc=(sx, sy * 0.72, 0.5), rot=(R(90), 0, 0), mat='wood_dark', seg=20, rseg=6)
            for k in range(6):
                ang = k * math.pi / 3
                p.beam((sx, sy * 0.72, 0.5), (sx + math.cos(ang) * 0.46, sy * 0.72, 0.5 + math.sin(ang) * 0.46), 0.02, mat='wood_dark', seg=4)
    p.beam((-0.7, -0.75, 0.5), (-0.7, 0.75, 0.5), 0.04, mat='iron', seg=6)
    p.beam((0.7, -0.75, 0.5), (0.7, 0.75, 0.5), 0.04, mat='iron', seg=6)
    # Ladung: Säcke
    rnd = random.Random(2)
    for i in range(4):
        p.sphere(0.3, loc=(rnd.uniform(-0.7, 0.7), rnd.uniform(-0.3, 0.3), 0.85), mat='straw', seg=10, rings=8, scale=(1.3, 0.9, 0.7))
    a.add(p)
    return a


# Wassertank auf Stelzen: Radius 1,4, Gesamthöhe 5 m
def kc_tank(col):
    a = Asset('kc_tank', col)
    p = a.part('M')
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.box((0.14, 0.14, 1.9), loc=(sx * 0.85, sy * 0.85, 0.95), mat='wood_dark')
    p.beam((-0.85, -0.85, 0.2), (0.85, -0.85, 1.7), 0.04, mat='wood_dark', square=True)
    p.beam((0.85, 0.85, 0.2), (-0.85, 0.85, 1.7), 0.04, mat='wood_dark', square=True)
    p.box((2.2, 2.2, 0.12), loc=(0, 0, 1.92), mat='wood')
    p.cyl(1.4, 2.6, loc=(0, 0, 3.25), mat='wood', seg=20, r2=1.32)
    for z in (2.3, 3.2, 4.2):
        p.torus(1.38 - (z - 2.3) * 0.03, 0.04, loc=(0, 0, z), mat='iron', seg=24, rseg=4)
    p.cyl(1.5, 0.9, loc=(0, 0, 5.0), mat='rust', seg=12, r2=0.1)
    p.cyl(0.08, 2.4, loc=(1.2, -0.6, 1.2), mat='rust', seg=6)
    a.add(p)
    return a


def kc_fountain(col):
    a = Asset('kc_fountain', col)
    p = a.part('M')
    for k in range(8):
        ang = k * math.pi / 4
        r = 4.8
        p.box((3.95, 0.55, 0.75), loc=(math.cos(ang) * r, math.sin(ang) * r, 0.375), rot=(0, 0, ang + R(90)), mat='stone_light', bevel=0.04)
        p.box((0.7, 0.7, 0.95), loc=(math.cos(ang + R(22.5)) * 5.15, math.sin(ang + R(22.5)) * 5.15, 0.475), mat='sandstone', bevel=0.05)
    p.cyl(0.6, 3.4, loc=(0, 0, 1.7), mat='sandstone', seg=12)
    p.cyl(1.6, 0.35, loc=(0, 0, 3.5), mat='sandstone', seg=20, r2=1.2)
    p.cyl(0.9, 0.4, loc=(0, 0, 3.85), mat='sandstone', seg=16)
    for k in range(4):
        ang = k * math.pi / 2
        p.beam((math.cos(ang) * 0.5, math.sin(ang) * 0.5, 2.4), (math.cos(ang) * 1.1, math.sin(ang) * 1.1, 2.1), 0.06, mat='patina', seg=6)
    a.add(p)
    # Figur: Gildenmeister mit erhobenem Zahnradschwert (Bronze mit Patina)
    s = a.part('Statue')
    s.cyl(0.18, 1.1, loc=(-0.15, 0, 4.6), mat='patina', seg=8)
    s.cyl(0.18, 1.1, loc=(0.15, 0, 4.6), mat='patina', seg=8)
    s.cyl(0.42, 1.3, loc=(0, 0, 5.75), mat='patina', seg=10, r2=0.34)
    s.cyl(0.55, 1.2, loc=(0, 0.05, 5.0), mat='patina', seg=10, r2=0.4)  # Mantel
    s.sphere(0.27, loc=(0, 0, 6.65), mat='patina', seg=12, rings=8)
    s.beam((0.35, 0, 6.2), (0.55, -0.1, 7.1), 0.09, mat='patina', seg=6)
    s.box((0.08, 0.05, 1.5), loc=(0.6, -0.1, 7.7), mat='patina')
    s.torus(0.28, 0.06, loc=(0.6, -0.1, 8.5), rot=(R(90), 0, 0), mat='patina', seg=12, rseg=6)
    s.beam((-0.35, 0, 6.2), (-0.6, -0.2, 5.6), 0.08, mat='patina', seg=6)
    a.add(s)
    return a


def kc_well(col):
    a = Asset('kc_well', col)
    p = a.part('M')
    p.cyl(1.4, 0.9, loc=(0, 0, 0.45), mat='stone_light', seg=16)
    p.cyl(1.15, 0.05, loc=(0, 0, 0.88), mat='stone_dark', seg=16)
    for sx in (-1, 1):
        p.box((0.16, 0.16, 2.4), loc=(sx * 1.2, 0, 1.6), mat='wood_dark')
    p.cyl(0.12, 2.5, loc=(0, 0, 2.3), rot=(0, R(90), 0), mat='wood', seg=8)
    p.box((0.05, 0.05, 0.5), loc=(1.35, 0, 2.05), mat='iron')
    p.beam((0, 0, 2.3), (0, 0, 1.3), 0.012, mat='straw', seg=4)
    p.cyl(0.18, 0.28, loc=(0, 0, 1.15), mat='wood', seg=10, r2=0.22)
    for sy in (-1, 1):
        p.box((2.9, 1.1, 0.06), loc=(0, sy * 0.48, 3.15), rot=(R(sy * 35), 0, 0), mat='tile')
    a.add(p)
    return a


def kc_maypole(col):
    a = Asset('kc_maypole', col)
    p = a.part('M')
    for i in range(20):
        p.cyl(0.28 - i * 0.006, 1.0, loc=(0, 0, 0.5 + i), mat='white' if i % 2 else 'paint_blue', seg=10)
    p.torus(1.3, 0.18, loc=(0, 0, 18.6), mat='leaves', seg=24, rseg=6)
    for k in range(6):
        ang = k * math.pi / 3
        p.beam((0, 0, 18.6), (math.cos(ang) * 1.3, math.sin(ang) * 1.3, 18.6), 0.04, mat='wood', seg=4)
    rnd = random.Random(4)
    for i in range(5):
        z = 6 + i * 2.4
        ang = i * 1.3
        for sx in (-1, 1):
            x, y = math.cos(ang) * sx, math.sin(ang) * sx
            p.beam((0, 0, z), (x * 1.2, y * 1.2, z), 0.03, mat='iron', seg=4)
            p.box((0.9, 0.05, 0.7), loc=(x * 1.3, y * 1.3, z - 0.35), rot=(0, 0, ang), mat=rnd.choice(['paint_red', 'paint_teal', 'paint_yellow', 'paint_blue', 'paint_cream']))
    a.add(p)
    return a


def kc_pinnacle(col):
    a = Asset('kc_pinnacle', col)
    p = a.part('M')
    p.box((0.9, 0.9, 0.9), loc=(0, 0, 0.45), mat='sandstone', bevel=0.03)
    p.cyl(0.62, 2.2, loc=(0, 0, 2.0), rot=(0, 0, R(45)), mat='sandstone', seg=4, r2=0.02)
    for k in range(4):
        ang = k * math.pi / 2 + R(45)
        p.cyl(0.08, 0.3, loc=(math.cos(ang) * 0.4, math.sin(ang) * 0.4, 1.4), mat='sandstone', seg=4, r2=0.01)
    a.add(p)
    return a


# ================================================================ Landmarken
# Erfinderturm-Spitze des Doms: offene Glockenstube (Kern sitzt im Spiel innen), Spitzhelm, Kupferringe
def l_spire(col):
    a = Asset('l_spire', col)
    p = a.part('Stone')
    p.cyl(4.6, 4.0, loc=(0, 0, 2.0), rot=(0, 0, R(22.5)), mat='stone_light', seg=8)
    p.cyl(4.75, 0.3, loc=(0, 0, 4.0), rot=(0, 0, R(22.5)), mat='sandstone', seg=8)
    for k in range(8):
        ang = k * math.pi / 4
        p.box((1.1, 1.1, 7.0), loc=(math.cos(ang) * 4.15, math.sin(ang) * 4.15, 7.5), rot=(0, 0, ang), mat='stone_light', bevel=0.05)
        # Spitzbogen über der Öffnung
        mid = ang + math.pi / 8
        p.prism([(-1.0, 0), (1.0, 0), (0, 1.3)], 0.6, loc=(math.cos(mid) * 4.2, math.sin(mid) * 4.2, 10.0), rot=(0, 0, mid), mat='stone_light', plane='YZ')
    p.cyl(4.6, 3.0, loc=(0, 0, 12.5), rot=(0, 0, R(22.5)), mat='stone_light', seg=8)
    p.cyl(4.85, 0.4, loc=(0, 0, 14.0), rot=(0, 0, R(22.5)), mat='sandstone', seg=8)
    for k in range(8):
        ang = k * math.pi / 4 + R(22.5)
        p.cyl(0.35, 2.6, loc=(math.cos(ang) * 4.5, math.sin(ang) * 4.5, 15.5), mat='sandstone', seg=4, r2=0.02)
    a.add(p)
    s = a.part('Roof')
    s.cyl(4.9, 26.0, loc=(0, 0, 27.0), rot=(0, 0, R(22.5)), mat='slate', seg=8, r2=0.15, smooth=False)
    for k in range(8):
        ang = k * math.pi / 4 + R(22.5)
        for i in range(6):
            t = 0.12 + i * 0.14
            r = 4.9 * (1 - t) + 0.25
            s.box((0.35, 0.35, 0.5), loc=(math.cos(ang) * r, math.sin(ang) * r, 14.5 + t * 26), mat='copper')
    a.add(s)
    m = a.part('Metal')
    for z in (2.2, 12.6):
        m.torus(5.1, 0.28, loc=(0, 0, z), mat='copper', seg=32, rseg=6)
    m.cyl(0.25, 9.0, loc=(0, 0, 44.0), mat='iron', seg=8)
    for z in (41.0, 43.5):
        m.torus(0.9, 0.08, loc=(0, 0, z), mat='copper', seg=16, rseg=4)
    for k in range(4):
        ang = k * math.pi / 2
        m.beam((math.cos(ang) * 4.3, math.sin(ang) * 4.3, 13.0), (math.cos(ang) * 0.4, math.sin(ang) * 0.4, 38.0), 0.12, mat='copper', seg=6)
    a.add(m)
    return a


def l_clock(col):
    a = Asset('l_clock', col)
    p = a.part('M')
    p.box((3.0, 3.0, 5.0), loc=(0, 0, 2.5), mat='brick', bevel=0.02)
    p.box((3.3, 3.3, 0.3), loc=(0, 0, 5.1), mat='stone_light')
    p.cyl(2.6, 4.0, loc=(0, 0, 7.25), rot=(0, 0, R(45)), mat='slate', seg=4, r2=0.05, smooth=False)
    p.cyl(0.06, 1.4, loc=(0, 0, 9.8), mat='gold', seg=6)
    p.sphere(0.18, loc=(0, 0, 9.6), mat='gold', seg=10, rings=6)
    for k in range(4):
        ang = k * math.pi / 2
        x, y = math.cos(ang) * 1.52, math.sin(ang) * 1.52
        p.cyl(0.95, 0.06, loc=(x, y, 3.2), rot=(R(90), 0, ang + R(90)), mat='ceramic_white', seg=24)
        p.torus(0.95, 0.06, loc=(x * 1.01, y * 1.01, 3.2), rot=(R(90), 0, ang + R(90)), mat='gold', seg=24, rseg=4)
        p.box((0.05, 0.05, 0.7), loc=(x * 1.03, y * 1.03, 3.45), rot=(0, 0, ang), mat='iron')
        p.box((0.05, 0.05, 0.5), loc=(x * 1.03, y * 1.03, 3.2), rot=(0, R(70), ang + R(90)), mat='iron')
    a.add(p)
    return a


# Riesenbaum: 44 m hoch, Wurzeln, knorriger Stamm, große Krone
def v_giant(col):
    a = Asset('v_giant', col)
    p = a.part('Trunk')
    rnd = random.Random(21)
    p.cyl(4.4, 3.0, loc=(0, 0, 1.5), mat='bark', seg=14, r2=2.7)
    z = 3.0
    r = 2.7
    for i in range(6):
        h = 4.8
        p.cyl(r, h, loc=(rnd.uniform(-0.15, 0.15), rnd.uniform(-0.15, 0.15), z + h / 2), mat='bark', seg=14, r2=r * 0.9)
        z += h
        r *= 0.9
    for i in range(8):
        ang = i * math.pi / 4 + rnd.uniform(-0.2, 0.2)
        L = rnd.uniform(5, 8)
        p.beam((math.cos(ang) * 1.8, math.sin(ang) * 1.8, 2.2), (math.cos(ang) * L, math.sin(ang) * L, -0.6), rnd.uniform(0.6, 0.9), mat='bark', seg=7)
    tips = []
    for i in range(9):
        ang = i * 2 * math.pi / 9 + rnd.uniform(-0.2, 0.2)
        z0 = rnd.uniform(17, 28)
        L = rnd.uniform(8, 13)
        tip = (math.cos(ang) * L, math.sin(ang) * L, z0 + rnd.uniform(5, 10))
        p.beam((0, 0, z0), tip, rnd.uniform(0.45, 0.8), mat='bark', seg=6)
        tips.append(tip)
    weather(p, 0.25, 0.6, 7)
    p.paint_top('moss', 0.9)
    a.add(p)
    c = a.part('Crown')
    for i, t in enumerate(tips):
        c.ico(rnd.uniform(4.5, 6.5), loc=t, mat=rnd.choice(['leaves', 'leaves3', 'leaves2']), sub=2, jitter=0.3, seed=300 + i, scale=(1, 1, 0.75), smooth=True)
    for i in range(10):
        ang = rnd.uniform(0, math.pi * 2)
        d = rnd.uniform(0, 7)
        c.ico(rnd.uniform(5, 7.5), loc=(math.cos(ang) * d, math.sin(ang) * d, rnd.uniform(33, 42)), mat=rnd.choice(['leaves', 'leaves3', 'leaves4']), sub=2, jitter=0.3, seed=400 + i, scale=(1, 1, 0.7), smooth=True)
    a.add(c)
    return a


# ================================================================ Vegetation
def v_linde(col):
    a = Asset('v_linde', col)
    p = a.part('M')
    rnd = random.Random(31)
    p.cyl(0.35, 4.0, loc=(0, 0, 2.0), r2=0.22, mat='bark', seg=8)
    for (x, y, z) in [(1.2, 0.4, 4.6), (-1.0, 0.7, 4.3), (0.2, -1.2, 4.8), (-0.5, -0.6, 5.2)]:
        p.beam((0, 0, 3.2), (x, y, z), 0.12, mat='bark', seg=5)
    for i in range(9):
        ang = i * 2 * math.pi / 9
        d = rnd.uniform(0.6, 1.8)
        p.ico(rnd.uniform(1.4, 2.1), loc=(math.cos(ang) * d, math.sin(ang) * d, rnd.uniform(4.6, 7.2)), mat=rnd.choice(['leaves', 'leaves2', 'leaves3']), sub=1, jitter=0.35, seed=500 + i)
    p.ico(2.0, loc=(0, 0, 7.6), mat='leaves', sub=1, jitter=0.3, seed=520)
    a.add(p)
    return a


def v_birke(col):
    a = Asset('v_birke', col)
    p = a.part('M')
    rnd = random.Random(41)
    p.cyl(0.16, 6.0, loc=(0, 0, 3.0), r2=0.09, mat='birch', seg=7)
    for i in range(10):
        p.box((0.17, 0.04, 0.12), loc=(0, -0.15, 0.6 + i * 0.5), rot=(0, 0, rnd.uniform(0, 6)), mat='black')
    for i in range(7):
        ang = rnd.uniform(0, math.pi * 2)
        p.ico(rnd.uniform(0.9, 1.3), loc=(math.cos(ang) * 0.8, math.sin(ang) * 0.8, rnd.uniform(4.2, 6.8)), mat=rnd.choice(['leaves4', 'leaves2']), sub=1, jitter=0.4, seed=600 + i, scale=(1, 1, 1.25))
    a.add(p)
    return a


def v_flowers(col):
    a = Asset('v_flowers', col)
    p = a.part('M')
    rnd = random.Random(51)
    mats = ['bloom_red', 'bloom_blue', 'bloom_white', 'bloom_yellow']
    for i in range(14):
        x, y = rnd.uniform(-0.6, 0.6), rnd.uniform(-0.6, 0.6)
        h = rnd.uniform(0.25, 0.55)
        ang = rnd.uniform(0, math.pi)
        dx, dy = math.cos(ang) * 0.02, math.sin(ang) * 0.02
        p.face([(x - dx, y - dy, 0), (x + dx, y + dy, 0), (x + dx, y + dy, h), (x - dx, y - dy, h)], mat='grass')
        s = rnd.uniform(0.05, 0.09)
        m = mats[i % 4]
        p.face([(x - s, y - s, h), (x + s, y - s, h), (x + s, y + s, h + 0.01), (x - s, y + s, h + 0.01)], mat=m)
        p.face([(x - s * 0.8, y, h - s * 0.6), (x + s * 0.8, y, h - s * 0.6), (x + s * 0.8, y, h + s * 0.6), (x - s * 0.8, y, h + s * 0.6)], mat=m)
    a.add(p)
    return a


def v_weed(col):
    a = Asset('v_weed', col)
    p = a.part('M')
    rnd = random.Random(61)
    for i in range(10):
        ang = rnd.uniform(0, math.pi * 2)
        h = rnd.uniform(0.4, 0.95)
        lean = rnd.uniform(0.1, 0.35)
        bx, by = math.cos(ang), math.sin(ang)
        w = 0.05
        p.face([(-by * w, bx * w, 0), (by * w, -bx * w, 0), (bx * lean, by * lean, h), (bx * lean - by * 0.01, by * lean + bx * 0.01, h)], mat=rnd.choice(['grass', 'leaves2', 'straw']))
    for i in range(3):
        ang = rnd.uniform(0, math.pi * 2)
        h = rnd.uniform(0.7, 1.0)
        x, y = math.cos(ang) * 0.2, math.sin(ang) * 0.2
        p.face([(x - 0.04, y, h - 0.12), (x + 0.04, y, h - 0.12), (x + 0.04, y, h + 0.05), (x - 0.04, y, h + 0.05)], mat='straw')
    a.add(p)
    return a


FACES = [f_seefahrer, f_schatten, f_erfinder, f_waechter]
FASSADE = [kc_window, kc_shutter, kc_door, kc_chimney, kc_dormer, kc_sign, kc_awning]
MOEBEL = [kc_table, kc_chair, kc_bed, kc_shelf, kc_stove, c_truhe, c_schrank, c_kommode]
STRASSE = [kc_lantern, kc_bench, kc_stall, kc_cart, kc_tank, kc_fountain, kc_well, kc_maypole, kc_pinnacle]
LANDMARKEN = [l_spire, l_clock, v_giant]
VEG = [v_linde, v_birke, v_flowers, v_weed]
ALL = FACES + FASSADE + MOEBEL + STRASSE + LANDMARKEN + VEG
