# ARC-Maschinen. Blender: Z oben, +Y vorne.
# Trefferzonen stehen als Custom Properties an Objekten/Empties:
#   hit = 'body' | 'weak', r = Radius, mult = Schadensfaktor, armor = 0..1,
#   hp = eigene Lebenspunkte (zerstörbar), thruster = 1 (Absturz bei Verlust)
# Benannte Knoten (z.B. Leg0, Ring, Hull) werden im Spiel animiert.
import math
from kit import Asset, lathe

R = math.radians


def _hit(kind, r, mult=1.0, armor=0.0, hp=None, thruster=False):
    d = {'hit': kind, 'r': r, 'mult': mult, 'armor': armor}
    if hp is not None:
        d['hp'] = hp
    if thruster:
        d['thruster'] = 1
    return d


def _rotor(a, name, loc, r=0.22, hp=18, mult=2.0, blades=3, glow_mat='arc_thrust'):
    """Rotor-Gondel mit Leuchtscheibe und drehbarem Propeller."""
    p = a.part(name)
    p.cyl(r, 0.14, loc=(0, 0, 0), mat='arc_panel', seg=18, bevel=0.01)
    p.torus(r, 0.03, loc=(0, 0, 0.05), mat='arc_frame', seg=20, rseg=6)
    p.cyl(r * 0.25, 0.18, loc=(0, 0, 0.02), mat='arc_joint', seg=10)
    ob = a.add(p, loc=loc, props=_hit('weak', r + 0.05, mult, 0.0, hp, True))
    g = a.part(name + 'Glow')
    g.cyl(r * 0.78, 0.02, mat=glow_mat, seg=18)
    a.add(g, parent=ob, loc=(0, 0, -0.075))
    fan = a.node(name + 'Fan', parent=ob, loc=(0, 0, 0.04))
    f = a.part(name + 'FanM')
    for k in range(blades):
        ang = k * 2 * math.pi / blades
        f.box((r * 0.9, 0.05, 0.012), loc=(math.cos(ang) * r * 0.45, math.sin(ang) * r * 0.45, 0), rot=(R(12), 0, ang), mat='arc_joint')
    a.add(f, parent=fan)
    return ob


def tick(col):
    a = Asset('arc_tick', col)
    b = a.part('Body')
    b.sphere(0.32, loc=(0, 0, 0.35), mat='arc_frame', seg=14, rings=10, scale=(1, 1.2, 0.6))
    b.sphere(0.3, loc=(0, -0.03, 0.4), mat='arc_shell', seg=14, rings=8, scale=(1.02, 1.15, 0.55), hemi=True)
    b.box((0.3, 0.05, 0.04), loc=(0, 0.02, 0.56), mat='arc_orange', bevel=0.01)
    b.box((0.22, 0.14, 0.08), loc=(0, 0.3, 0.36), mat='arc_joint', bevel=0.02)
    for s in (-1, 1):
        b.cyl(0.02, 0.18, loc=(s * 0.09, 0.42, 0.33), rot=(R(80), 0, 0), mat='arc_joint', seg=6)
    a.add(b, props=_hit('body', 0.4))
    e = a.part('Eye')
    e.sphere(0.075, mat='arc_eye', seg=12, rings=8)
    a.add(e, loc=(0, 0.36, 0.42), props=_hit('weak', 0.12, 2.0))
    for i in range(6):
        side = -1 if i < 3 else 1
        k = (i % 3) - 1
        piv = a.node(f'Leg{i}', loc=(side * 0.22, k * 0.2, 0.35))
        l = a.part(f'Leg{i}M')
        l.beam((0, 0, 0), (side * 0.32, k * 0.06, 0.12), 0.03, mat='arc_frame', seg=6)
        l.sphere(0.04, loc=(side * 0.32, k * 0.06, 0.12), mat='arc_joint', seg=8, rings=6)
        l.beam((side * 0.32, k * 0.06, 0.12), (side * 0.48, k * 0.1, -0.33), 0.022, mat='arc_joint', seg=6)
        a.add(l, parent=piv)
    return a


def pop(col):
    a = Asset('arc_pop', col)
    ball = a.node('Ball', loc=(0, 0, 0.5))
    c = a.part('Core')
    c.sphere(0.48, mat='arc_shell', seg=20, rings=14)
    for k in range(3):
        c.torus(0.49, 0.045, rot=(k * 1.1, k * 0.7, 0), mat='arc_orange', seg=28, rseg=6)
    for k in range(6):
        ang = k * math.pi / 3
        c.box((0.14, 0.14, 0.05), loc=(math.cos(ang) * 0.46, math.sin(ang) * 0.46, 0), rot=(0, R(90), ang), mat='arc_panel', bevel=0.01)
    a.add(c, parent=ball, props=_hit('body', 0.55))
    l = a.part('Light')
    l.sphere(0.11, mat='arc_eye', seg=12, rings=8)
    a.add(l, parent=ball, loc=(0, 0, 0.45))
    return a


def wasp(col):
    a = Asset('arc_wasp', col)
    h = a.part('Hull')
    h.sphere(0.42, mat='arc_frame', seg=16, rings=12, scale=(1, 1.3, 0.7))
    h.sphere(0.44, loc=(0, -0.02, 0.04), mat='arc_shell', seg=16, rings=10, scale=(1, 1.25, 0.55), hemi=True)
    h.box((0.1, 0.8, 0.05), loc=(0, 0, 0.27), mat='arc_orange', bevel=0.01)
    h.box((0.3, 0.2, 0.18), loc=(0, 0.4, -0.05), mat='arc_joint', bevel=0.03)
    h.cyl(0.04, 0.5, loc=(0, 0.45, -0.2), rot=(R(90), 0, 0), mat='arc_joint', seg=8)
    h.cyl(0.055, 0.08, loc=(0, 0.7, -0.2), rot=(R(90), 0, 0), mat='arc_frame', seg=8)
    for sx in (-1, 1):
        for sy in (-1, 1):
            h.beam((sx * 0.25, sy * 0.25, 0.05), (sx * 0.7, sy * 0.45, 0.05), 0.04, mat='arc_frame', seg=6, square=True)
    a.add(h, props=_hit('body', 0.5, 1.0, 0.2))
    e = a.part('Eye')
    e.sphere(0.09, mat='arc_eye', seg=12, rings=8)
    a.add(e, loc=(0, 0.53, 0.02), props=_hit('weak', 0.14, 1.6))
    for i, (x, y) in enumerate([(-0.7, 0.45), (0.7, 0.45), (-0.7, -0.45), (0.7, -0.45)]):
        _rotor(a, f'Rotor{i}', (x, y, 0.05))
    return a


def hornet(col):
    a = Asset('arc_hornet', col)
    h = a.part('Hull')
    h.sphere(0.6, mat='arc_frame', seg=16, rings=12, scale=(1.1, 1.5, 0.8))
    h.box((0.9, 0.6, 0.25), loc=(0, -0.4, 0.35), mat='arc_panel', bevel=0.04)
    h.box((0.2, 1.0, 0.06), loc=(0, -0.2, 0.5), mat='arc_orange', bevel=0.01)
    for sx in (-1, 1):
        h.cyl(0.06, 0.8, loc=(sx * 0.35, 0.55, -0.3), rot=(R(90), 0, 0), mat='arc_joint', seg=8)
        h.box((0.16, 0.3, 0.16), loc=(sx * 0.35, 0.2, -0.3), mat='arc_frame', bevel=0.02)
        h.beam((sx * 0.4, -0.5, 0.1), (sx * 0.85, -0.55, 0.1), 0.05, mat='arc_frame', seg=6, square=True)
    a.add(h, props=_hit('body', 0.65, 1.0, 0.35))
    sh = a.part('Shield')
    sh.sphere(0.78, rot=(R(-90), 0, 0), mat='arc_shell', seg=18, rings=10, hemi=True, scale=(1, 1, 0.55))
    sh.torus(0.72, 0.04, rot=(R(90), 0, 0), loc=(0, 0.02, 0), mat='arc_orange', seg=24, rseg=6)
    a.add(sh, loc=(0, 0.35, 0))
    a.node('ShieldHit', loc=(0, 0.55, 0), props=_hit('body', 0.62, 1.0, 0.8))
    e = a.part('Eye')
    e.sphere(0.1, mat='arc_eye', seg=12, rings=8)
    a.add(e, loc=(0, 0.8, 0.05))
    for i, x in enumerate((-0.85, 0.85)):
        _rotor(a, f'Thr{i}', (x, -0.55, 0.1), r=0.3, hp=45, mult=2.2, blades=4)
    return a


def snitch(col):
    a = Asset('arc_snitch', col)
    h = a.part('Hull')
    lathe(h, [(0.2, -0.28), (0.45, -0.18), (0.5, 0.0), (0.42, 0.18), (0.2, 0.26), (0.0, 0.28)], mat='arc_shell', seg=20)
    h.cyl(0.52, 0.06, loc=(0, 0, -0.02), mat='arc_frame', seg=20)
    h.sphere(0.3, loc=(0, 0, -0.25), mat='arc_joint', seg=16, rings=10)
    for k in range(4):
        ang = k * math.pi / 2
        h.beam((math.cos(ang) * 0.45, math.sin(ang) * 0.45, 0), (math.cos(ang) * 0.75, math.sin(ang) * 0.75, 0.05), 0.03, mat='arc_frame', seg=6)
    h.cyl(0.03, 0.6, loc=(0, 0, 0.55), mat='arc_joint', seg=6)
    a.add(h, props=_hit('body', 0.55))
    e = a.part('Eye')
    e.sphere(0.13, mat='arc_eye', seg=14, rings=10)
    a.add(e, loc=(0, 0.05, -0.4), props=_hit('weak', 0.18, 2.0))
    ring = a.node('Ring', loc=(0, 0, 0.05))
    r = a.part('RingM')
    r.torus(0.78, 0.05, mat='arc_panel', seg=32, rseg=6)
    for k in range(3):
        ang = k * 2 * math.pi / 3
        r.box((0.14, 0.08, 0.08), loc=(math.cos(ang) * 0.78, math.sin(ang) * 0.78, 0), rot=(0, 0, ang), mat='arc_orange', bevel=0.01)
    a.add(r, parent=ring)
    b = a.part('Beacon')
    b.sphere(0.08, mat='arc_eye', seg=10, rings=6)
    a.add(b, loc=(0, 0, 0.9))
    return a


def rocketeer(col):
    a = Asset('arc_rocketeer', col)
    h = a.part('Hull')
    h.box((1.8, 2.4, 1.1), mat='arc_frame', bevel=0.08)
    h.box((1.95, 2.3, 0.4), loc=(0, 0, 0.62), mat='arc_shell', bevel=0.1, taper=(0.9, 0.95))
    h.box((1.2, 0.5, 0.25), loc=(0, -0.7, 0.9), mat='arc_orange', bevel=0.04)
    h.box((1.0, 0.6, 0.5), loc=(0, 1.2, 0.05), mat='arc_joint', bevel=0.06)
    for sx in (-1, 1):
        h.beam((sx * 0.9, -0.8, 0.2), (sx * 1.6, -0.9, 0.2), 0.09, mat='arc_frame', seg=6, square=True)
        h.box((0.2, 1.8, 0.2), loc=(sx * 0.95, 0.1, -0.5), mat='arc_panel', bevel=0.03)
    a.add(h, props=_hit('body', 1.35, 1.0, 0.55))
    e = a.part('Eye')
    for sx in (-0.25, 0.25):
        e.sphere(0.12, loc=(sx, 0, 0), mat='arc_eye', seg=12, rings=8)
    a.add(e, loc=(0, 1.52, 0.1), props=_hit('weak', 0.28, 2.0))
    for i, x in enumerate((-1.25, 1.25)):
        pod = a.node(f'Pod{i}', loc=(x, 0.3, -0.1))
        p = a.part(f'Pod{i}M')
        p.box((0.6, 1.4, 0.6), mat='arc_panel', bevel=0.05)
        p.box((0.62, 0.3, 0.62), loc=(0, -0.4, 0), mat='arc_orange', bevel=0.03)
        for k in range(4):
            p.cyl(0.09, 0.1, loc=((k % 2 - 0.5) * 0.28, 0.7, (k // 2 - 0.5) * 0.28), rot=(R(90), 0, 0), mat='arc_joint', seg=10)
        a.add(p, parent=pod)
    for i, x in enumerate((-1.6, 1.6)):
        _rotor(a, f'Thr{i}', (x, -0.9, 0.2), r=0.5, hp=130, mult=2.0, blades=4)
    return a


def leaper(col):
    a = Asset('arc_leaper', col)
    hull = a.node('Hull', loc=(0, 0, 2.2))
    b = a.part('Body')
    b.sphere(1.3, mat='arc_frame', seg=18, rings=12, scale=(1.2, 1.4, 0.7))
    b.sphere(1.42, loc=(0, -0.1, 0.1), mat='arc_shell', seg=18, rings=10, scale=(1.1, 1.3, 0.62), hemi=True)
    b.box((0.4, 2.8, 0.1), loc=(0, -0.1, 0.95), mat='arc_orange', bevel=0.03)
    for sx in (-1, 1):
        b.box((0.9, 1.2, 0.2), loc=(sx * 0.9, -0.3, 0.55), rot=(0, sx * R(-25), 0), mat='arc_shell2', bevel=0.05)
    b.box((1.0, 0.6, 0.6), loc=(0, 1.55, -0.15), mat='arc_joint', bevel=0.08)
    a.add(b, parent=hull, props=_hit('body', 1.5, 1.0, 0.6))
    e = a.part('Eye')
    e.sphere(0.3, mat='arc_weak', seg=14, rings=10)
    a.add(e, parent=hull, loc=(0, 1.85, 0), props=_hit('weak', 0.4, 2.5))
    for i, (x, y) in enumerate([(-1.3, 1.1), (1.3, 1.1), (-1.3, -1.1), (1.3, -1.1)]):
        s = 1 if x > 0 else -1
        hip = a.node(f'Leg{i}', parent=hull, loc=(x, y, 0))
        l = a.part(f'Leg{i}M')
        knee = (s * 1.4, 0, 1.0)
        foot = (s * 2.1, 0, -2.2)
        l.sphere(0.32, mat='arc_joint', seg=12, rings=8)
        l.beam((0, 0, 0), knee, 0.2, mat='arc_panel', seg=8, square=True)
        l.box((0.5, 0.5, 0.9), loc=(s * 0.7, 0, 0.55), rot=(0, s * R(-54), 0), mat='arc_shell', bevel=0.05)
        l.sphere(0.24, loc=knee, mat='arc_joint', seg=10, rings=8)
        l.beam(knee, foot, 0.13, mat='arc_frame', seg=8)
        l.cyl(0.25, 0.3, loc=foot, r2=0.12, mat='arc_joint', seg=10)
        a.add(l, parent=hip)
    return a


def bastion(col):
    a = Asset('arc_bastion', col)
    hull = a.node('Hull', loc=(0, 0, 4.2))
    b = a.part('Body')
    b.box((3.4, 3.8, 2.4), mat='arc_shell', bevel=0.15)
    b.box((3.6, 4.0, 0.5), loc=(0, 0, 1.35), mat='arc_panel', bevel=0.1)
    b.box((3.2, 0.4, 1.6), loc=(0, 2.0, 0.1), mat='arc_shell2', bevel=0.1)
    b.box((3.5, 3.9, 0.3), loc=(0, 0, -1.25), mat='arc_frame', bevel=0.05)
    for sx in (-1, 1):
        b.box((0.3, 3.9, 1.2), loc=(sx * 1.72, 0, 0.2), mat='arc_orange', bevel=0.05)
    b.box((1.8, 0.3, 1.8), loc=(0, -1.95, 0.3), mat='arc_frame', bevel=0.06)
    b.box((1.6, 1.0, 0.8), loc=(0, 0.4, 1.9), mat='arc_panel', bevel=0.08)
    for k in range(3):
        b.cyl(0.08, 0.9, loc=(-0.5 + k * 0.5, -0.3, 2.1), mat='arc_joint', seg=6)
    a.add(b, parent=hull, props=_hit('body', 2.4, 1.0, 0.85))
    e = a.part('Eye')
    e.box((1.2, 0.2, 0.3), mat='arc_eye', bevel=0.03)
    a.add(e, parent=hull, loc=(0, 2.22, 0.7))
    c = a.part('Core')
    c.sphere(0.6, mat='arc_weak', seg=16, rings=10)
    a.add(c, parent=hull, loc=(0, -2.05, 0.3), props=_hit('weak', 0.75, 3.0))
    for i, x in enumerate((-2.1, 2.1)):
        g = a.node(f'Gun{i}', parent=hull, loc=(x, 0.8, 0))
        m = a.part(f'Gun{i}M')
        m.box((0.8, 1.6, 0.8), mat='arc_panel', bevel=0.06)
        m.cyl(0.35, 0.2, loc=(0, 0.85, 0), rot=(R(90), 0, 0), mat='arc_frame', seg=12)
        for k in range(3):
            ang = k * 2 * math.pi / 3
            m.cyl(0.09, 1.6, loc=(math.cos(ang) * 0.18, 1.6, math.sin(ang) * 0.18), rot=(R(90), 0, 0), mat='arc_joint', seg=8)
        m.cyl(0.3, 0.12, loc=(0, 2.2, 0), rot=(R(90), 0, 0), mat='arc_frame', seg=12)
        a.add(m, parent=g)
    for i, (x, y) in enumerate([(-1.7, 1.4), (1.7, 1.4), (-1.7, -1.4), (1.7, -1.4)]):
        hip = a.node(f'Leg{i}', parent=hull, loc=(x, y, -1.0))
        l = a.part(f'Leg{i}M')
        l.sphere(0.5, mat='arc_joint', seg=12, rings=8)
        l.box((0.8, 0.8, 3.0), loc=(0, 0, -1.5), mat='arc_panel', bevel=0.08)
        l.box((0.9, 0.5, 1.2), loc=(0, 0.3, -1.0), mat='arc_shell', bevel=0.06)
        l.cyl(0.25, 3.0, loc=(0, -0.3, -1.5), mat='arc_frame', seg=8)
        l.box((1.3, 1.5, 0.35), loc=(0, 0.1, -3.05), mat='arc_frame', bevel=0.06)
        a.add(l, parent=hip)
    return a


def turret(col):
    a = Asset('arc_turret', col)
    b = a.part('Base')
    b.cyl(0.9, 0.3, loc=(0, 0, 0.15), r2=0.75, mat='arc_panel', seg=16, bevel=0.02)
    b.cyl(0.45, 0.5, loc=(0, 0, 0.5), mat='arc_frame', seg=12)
    for k in range(4):
        ang = k * math.pi / 2 + math.pi / 4
        b.box((0.2, 0.5, 0.25), loc=(math.cos(ang) * 0.75, math.sin(ang) * 0.75, 0.1), rot=(0, 0, ang), mat='arc_joint', bevel=0.02)
    a.add(b)
    head = a.node('Head', loc=(0, 0, 1.0))
    h = a.part('Housing')
    h.box((0.9, 1.1, 0.7), mat='arc_frame', bevel=0.06)
    h.box((1.0, 1.2, 0.2), loc=(0, 0, 0.42), mat='arc_shell', bevel=0.05)
    h.box((0.15, 0.9, 0.4), loc=(-0.52, 0, 0), mat='arc_orange', bevel=0.02)
    h.cyl(0.08, 1.4, loc=(0, 1.1, -0.05), rot=(R(90), 0, 0), mat='arc_joint', seg=10)
    h.cyl(0.13, 0.25, loc=(0, 1.75, -0.05), rot=(R(90), 0, 0), mat='arc_frame', seg=10)
    a.add(h, parent=head, props=_hit('body', 0.75, 1.0, 0.7))
    l = a.part('Lens')
    l.cyl(0.16, 0.1, rot=(R(90), 0, 0), mat='arc_eye', seg=16)
    a.add(l, parent=head, loc=(0, 0.58, 0.12), props=_hit('weak', 0.24, 2.5))
    return a


BUILDERS = [tick, pop, wasp, hornet, snitch, rocketeer, leaper, bastion, turret]


def build_all(col):
    return [b(col) for b in BUILDERS]
