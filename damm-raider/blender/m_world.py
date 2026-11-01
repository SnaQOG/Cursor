# Welt-Assets: Props, Gebäudeteile, Landmarken, Vegetation, Aufzug, Raider.
# Ursprung = Bodenmitte, +Y = vorne, Z = oben. "key"-Flächen werden im Spiel umgefärbt.
import math
import random
from kit import Asset, lathe

R = math.radians


# ================================================================ Props
def p_container(col):
    a = Asset('p_container', col)
    p = a.part('M')
    L, W, H = 6.1, 2.45, 2.6
    p.box((L - 0.1, W - 0.06, H - 0.12), loc=(0, 0, H / 2), mat='key')
    n = 18
    for i in range(n):
        x = -L / 2 + 0.25 + i * (L - 0.5) / (n - 1)
        for s in (-1, 1):
            p.box((0.1, 0.05, H - 0.3), loc=(x, s * (W / 2 - 0.01), H / 2), mat='key_dark')
    for s in (-1, 1):
        p.box((L, 0.12, 0.14), loc=(0, s * (W / 2 - 0.04), H - 0.07), mat='key_dark')
        p.box((L, 0.12, 0.16), loc=(0, s * (W / 2 - 0.04), 0.08), mat='rust_dark')
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (0, 1):
                p.box((0.18, 0.18, 0.14), loc=(sx * (L / 2 - 0.09), sy * (W / 2 - 0.09), 0.07 + sz * (H - 0.14)), mat='metal_dark', bevel=0.01)
    # Türseite (+X)
    p.box((0.06, W - 0.1, H - 0.2), loc=(L / 2 - 0.02, 0, H / 2), mat='key_dark')
    p.box((0.07, 0.03, H - 0.25), loc=(L / 2, 0, H / 2), mat='rust_dark')
    for y in (-0.8, -0.35, 0.35, 0.8):
        p.cyl(0.03, H - 0.3, loc=(L / 2 + 0.05, y, H / 2), mat='galv', seg=6)
        p.box((0.05, 0.25, 0.05), loc=(L / 2 + 0.08, y + 0.1, H * 0.45), mat='galv')
    p.box((0.02, 0.5, 0.35), loc=(-L / 2 - 0.02, -0.5, H * 0.7), mat='paint_cream')
    a.add(p)
    return a


def _wheel(p, loc, r=0.36, w=0.24):
    p.cyl(r, w, loc=loc, rot=(0, R(90), 0), mat='rubber', seg=14, bevel=0.03)
    p.cyl(r * 0.55, w + 0.02, loc=loc, rot=(0, R(90), 0), mat='rust', seg=10)


def p_car_a(col):
    a = Asset('p_car_a', col)
    p = a.part('M')
    L, W = 4.3, 1.8
    p.box((W, L, 0.62), loc=(0, 0, 0.62), mat='key', bevel=0.12)
    p.box((W - 0.1, 1.2, 0.2), loc=(0, 1.35, 0.98), mat='key', bevel=0.08, taper=(0.98, 0.9))
    p.box((W - 0.2, 2.0, 0.55), loc=(0, -0.25, 1.2), mat='key', bevel=0.1, taper=(0.9, 0.7))
    p.box((W - 0.12, 1.95, 0.38), loc=(0, -0.25, 1.2), mat='visor', bevel=0.04, taper=(0.92, 0.72))
    p.box((W + 0.04, 0.2, 0.22), loc=(0, L / 2, 0.45), mat='chrome', bevel=0.05)
    p.box((W + 0.04, 0.2, 0.22), loc=(0, -L / 2, 0.45), mat='rust', bevel=0.05)
    for sx in (-0.62, 0.62):
        p.box((0.3, 0.05, 0.14), loc=(sx, L / 2 + 0.06, 0.68), mat='white', bevel=0.02)
        p.box((0.3, 0.05, 0.12), loc=(sx, -L / 2 - 0.06, 0.7), mat='arc_red', bevel=0.02)
    p.box((0.9, 0.7, 0.03), loc=(0.35, 0.9, 0.94), rot=(0, 0, R(8)), mat='rust')
    for (x, y) in [(-0.85, 1.35), (0.85, 1.35), (-0.85, -1.3), (0.85, -1.3)]:
        _wheel(p, (x, y, 0.36))
    a.add(p)
    return a


def p_car_b(col):
    a = Asset('p_car_b', col)
    p = a.part('M')
    L, W = 4.9, 2.0
    p.box((W, L - 1.0, 1.9), loc=(0, -0.5, 1.2), mat='key', bevel=0.12)
    p.box((W, 1.2, 1.0), loc=(0, L / 2 - 0.6, 0.75), mat='key', bevel=0.12)
    p.box((W - 0.1, 0.4, 0.7), loc=(0, L / 2 - 1.25, 1.55), rot=(R(-25), 0, 0), mat='visor', bevel=0.03)
    for s in (-1, 1):
        p.box((0.04, 0.6, 0.45), loc=(s * (W / 2 + 0.01), L / 2 - 1.3, 1.5), mat='visor')
        p.box((0.05, 2.5, 0.1), loc=(s * (W / 2 + 0.01), -0.7, 0.9), mat='key_dark')
    p.box((W + 0.04, 0.2, 0.25), loc=(0, L / 2, 0.45), mat='rust', bevel=0.05)
    p.box((W - 0.4, 0.04, 1.4), loc=(0, -L / 2 - 0.02 + 0.5 - 0.5, 1.2), mat='key_dark')
    p.box((1.4, 2.8, 0.1), loc=(0, -0.6, 2.2), mat='galv')
    for (x, y) in [(-0.9, 1.6), (0.9, 1.6), (-0.9, -1.5), (0.9, -1.5)]:
        _wheel(p, (x, y, 0.4), r=0.4, w=0.26)
    a.add(p)
    return a


def p_barrier(col):
    a = Asset('p_barrier', col)
    p = a.part('M')
    p.prism([(-0.35, 0), (0.35, 0), (0.3, 0.1), (0.12, 0.3), (0.1, 1.0), (-0.1, 1.0), (-0.12, 0.3), (-0.3, 0.1)], 3.0, mat='concrete_light', plane='YZ', bevel=0.02)
    p.box((3.02, 0.24, 0.12), loc=(0, 0, 0.85), mat='orange')
    for x in (-0.9, 0.2):
        p.box((0.5, 0.25, 0.13), loc=(x, 0, 0.85), mat='white')
    a.add(p)
    return a


def p_sandbags(col):
    a = Asset('p_sandbags', col)
    p = a.part('M')
    rnd = random.Random(3)
    for layer in range(3):
        n = 5 if layer < 2 else 4
        for i in range(n):
            x = -1.2 + i * 0.6 + (0.3 if layer % 2 else 0)
            if x > 1.3:
                continue
            p.sphere(0.34, loc=(x, rnd.uniform(-0.05, 0.05), 0.18 + layer * 0.3), rot=(0, 0, rnd.uniform(-0.2, 0.2)), mat='sand' if rnd.random() < 0.7 else 'canvas', seg=10, rings=6, scale=(1, 0.75, 0.48))
    a.add(p)
    return a


def p_crate(col):
    a = Asset('p_crate', col)
    p = a.part('M')
    s = 1.2
    p.box((s, s, s), loc=(0, 0, s / 2), mat='wood', bevel=0.02)
    for k in range(4):
        z = 0.15 + k * 0.3
        for side in (-1, 1):
            p.box((s + 0.01, 0.02, 0.03), loc=(0, side * s / 2, z), mat='wood_dark')
            p.box((0.02, s + 0.01, 0.03), loc=(side * s / 2, 0, z), mat='wood_dark')
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.box((0.12, 0.12, s + 0.01), loc=(sx * (s / 2 - 0.05), sy * (s / 2 - 0.05), s / 2), mat='wood_light', bevel=0.01)
    p.box((s * 0.6, 0.02, 0.4), loc=(0, -s / 2 - 0.01, s / 2), rot=(0, R(35), 0), mat='wood_light')
    a.add(p)
    return a


def p_barrel(col):
    a = Asset('p_barrel', col)
    p = a.part('M')
    p.cyl(0.42, 1.2, loc=(0, 0, 0.6), mat='red', seg=18)
    for z in (0.08, 0.42, 0.78, 1.12):
        p.torus(0.425, 0.02, loc=(0, 0, z), mat='red', seg=18, rseg=4)
    p.cyl(0.428, 0.16, loc=(0, 0, 0.62), mat='hazard', seg=18, caps=False)
    p.cyl(0.18, 0.03, loc=(0.15, 0.1, 1.21), mat='metal_dark', seg=10)
    a.add(p)
    return a


def p_lamp(col):
    a = Asset('p_lamp', col)
    p = a.part('M')
    p.cyl(0.18, 0.3, loc=(0, 0, 0.15), mat='concrete_dark', seg=10)
    p.cyl(0.09, 6.0, loc=(0, 0, 3.0), r2=0.06, mat='metal_dark', seg=10)
    p.beam((0, 0, 5.8), (1.1, 0, 6.05), 0.05, mat='metal_dark')
    p.box((0.6, 0.3, 0.15), loc=(1.25, 0, 6.02), mat='metal_dark', bevel=0.03)
    a.add(p)
    g = a.part('Glow')
    g.box((0.45, 0.22, 0.03), mat='lamp')
    a.add(g, loc=(1.25, 0, 5.94))
    return a


def p_pylon(col):
    a = Asset('p_pylon', col)
    p = a.part('M')
    H, base, top = 24.0, 3.2, 1.0
    legs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    def at(sx, sy, t):
        w = base + (top - base) * t
        return (sx * w, sy * w, H * t)
    for sx, sy in legs:
        p.beam(at(sx, sy, 0), at(sx, sy, 1), 0.12, mat='galv', square=True)
    n = 7
    for k in range(n):
        t0, t1 = k / n, (k + 1) / n
        for (ax, ay), (bx, by) in [((1, 1), (1, -1)), ((1, -1), (-1, -1)), ((-1, -1), (-1, 1)), ((-1, 1), (1, 1))]:
            p.beam(at(ax, ay, t1), at(bx, by, t1), 0.06, mat='galv', square=True)
            p.beam(at(ax, ay, t0), at(bx, by, t1), 0.04, mat='galv', square=True)
    for z, w in ((H, 6.0), (H - 4.5, 4.5)):
        p.beam((-w, 0, z), (w, 0, z), 0.18, mat='galv', square=True)
        p.beam((-w, 0, z), (0, 0, z - 1.8), 0.06, mat='galv', square=True)
        p.beam((w, 0, z), (0, 0, z - 1.8), 0.06, mat='galv', square=True)
        for x in (-w + 0.4, w - 0.4):
            for k in range(4):
                p.cyl(0.16, 0.12, loc=(x, 0, z - 0.3 - k * 0.14), mat='ceramic_white', seg=10)
    for sx, sy in legs:
        p.box((0.8, 0.8, 0.5), loc=(sx * base, sy * base, 0.1), mat='concrete_dark', bevel=0.04)
    a.add(p)
    return a


def p_tank(col):
    a = Asset('p_tank', col)
    p = a.part('M')
    p.box((3.4, 7.0, 1.3), loc=(0, 0, 1.05), mat='olive_dark', bevel=0.12, taper=(0.9, 0.95))
    for s in (-1, 1):
        p.box((0.8, 7.2, 1.0), loc=(s * 1.85, 0, 0.55), mat='rubber', bevel=0.2)
        for k in range(6):
            p.cyl(0.38, 0.5, loc=(s * 1.85, -2.5 + k, 0.45), rot=(0, R(90), 0), mat='metal_dark', seg=10)
        p.box((0.9, 7.0, 0.1), loc=(s * 1.85, 0, 1.1), mat='olive')
    p.box((2.6, 3.0, 0.9), loc=(0.3, -0.4, 2.1), rot=(0, 0, R(20)), mat='olive', bevel=0.15, taper=(0.85, 0.85))
    p.beam((0.9, 1.2, 2.2), (2.2, 4.4, 1.4), 0.14, mat='olive_dark')
    p.box((1.6, 1.4, 0.6), loc=(-1.2, -2.5, 1.9), rot=(R(10), R(-8), R(30)), mat='rust', bevel=0.05)
    a.add(p)
    return a


def p_busstop(col):
    a = Asset('p_busstop', col)
    p = a.part('M')
    for x in (-1.9, 1.9):
        p.box((0.1, 0.1, 2.5), loc=(x, 0.7, 1.25), mat='teal_dark')
    p.box((4.2, 1.8, 0.12), loc=(0, 0.2, 2.55), mat='teal', bevel=0.03)
    p.box((3.6, 0.4, 0.08), loc=(0, 0.5, 0.5), mat='wood')
    p.box((0.9, 0.08, 1.4), loc=(1.4, 0.8, 1.4), mat='paint_cream')
    a.add(p)
    g = a.part('Glass')
    g.box((3.8, 0.04, 2.0), mat='glass')
    a.add(g, loc=(0, 0.85, 1.4))
    return a


# ================================================================ Gebäudeteile
def k_window(col):
    """Fensterrahmen für eine Öffnung von 1,4 x 1,3 m (Mitte im Ursprung, Wand in der XZ-Ebene)."""
    a = Asset('k_window', col)
    p = a.part('M')
    w, h, d = 1.4, 1.3, 0.14
    p.box((w + 0.12, d + 0.06, 0.08), loc=(0, 0, -h / 2 - 0.04), mat='concrete_light', bevel=0.01)
    p.box((w + 0.1, d, 0.06), loc=(0, 0, h / 2 + 0.02), mat='wood_dark')
    for s in (-1, 1):
        p.box((0.06, d, h), loc=(s * (w / 2 - 0.02), 0, 0), mat='wood_dark')
    p.box((0.04, 0.06, h), loc=(0, 0, 0), mat='wood_dark')
    p.box((w, 0.06, 0.04), loc=(0, 0, h * 0.18), mat='wood_dark')
    a.add(p)
    return a


def k_shutter(col):
    """Halb offener Fensterladen, passend zu k_window (links an der Öffnung)."""
    a = Asset('k_shutter', col)
    p = a.part('M')
    w, h = 1.4, 1.3
    p.box((0.62, 0.04, h - 0.06), loc=(-w / 2 - 0.3, -0.12, 0), rot=(0, 0, R(-25)), mat='paint_teal', bevel=0.01)
    for k in range(3):
        p.box((0.56, 0.05, 0.03), loc=(-w / 2 - 0.3, -0.13, -h / 2 + 0.25 + k * 0.4), rot=(0, 0, R(-25)), mat='teal_dark')
    a.add(p)
    return a


def k_door(col):
    """Türzarge für 1,8 x 2,4 m, Unterkante im Ursprung."""
    a = Asset('k_door', col)
    p = a.part('M')
    w, h, d = 1.8, 2.4, 0.16
    for s in (-1, 1):
        p.box((0.12, d, h), loc=(s * (w / 2 - 0.02), 0, h / 2), mat='metal_dark')
    p.box((w + 0.12, d, 0.14), loc=(0, 0, h - 0.05), mat='metal_dark')
    p.box((w + 0.4, 0.5, 0.08), loc=(0, -0.3, h + 0.15), mat='concrete_light', bevel=0.02)
    p.box((0.4, 0.02, 0.2), loc=(0, -0.09, h + 0.35), mat='paint_cream')
    # Türblatt, aufgeschwungen
    p.box((0.05, 0.85, h - 0.1), loc=(-w / 2 + 0.05, -0.45, h / 2 - 0.02), rot=(0, 0, R(-10)), mat='rust', bevel=0.01)
    p.box((0.07, 0.12, 0.04), loc=(-w / 2 + 0.05, -0.75, h * 0.45), mat='galv')
    a.add(p)
    return a


def k_rolldoor(col):
    """Rolltor-Rahmen für 8 x 7 m (Kraftwerk)."""
    a = Asset('k_rolldoor', col)
    p = a.part('M')
    w, h = 8.0, 7.0
    for s in (-1, 1):
        p.box((0.35, 0.5, h), loc=(s * (w / 2 + 0.1), 0, h / 2), mat='hazard')
        for k in range(7):
            p.box((0.36, 0.52, 0.25), loc=(s * (w / 2 + 0.1), 0, 0.5 + k * 1.0), mat='black')
    p.box((w + 0.6, 0.9, 0.9), loc=(0, 0, h + 0.2), mat='metal_dark', bevel=0.05)
    for k in range(6):
        p.box((w, 0.08, 0.3), loc=(0, 0.1, h - 0.3 - k * 0.32), mat='galv' if k % 2 else 'metal_grey')
    a.add(p)
    return a


def k_vent(col):
    a = Asset('k_vent', col)
    p = a.part('M')
    p.box((1.4, 1.0, 0.8), loc=(0, 0, 0.4), mat='galv', bevel=0.03)
    for k in range(6):
        p.box((1.2, 0.03, 0.05), loc=(0, -0.51, 0.15 + k * 0.1), mat='metal_dark')
    p.cyl(0.25, 0.6, loc=(0.3, 0.2, 1.0), mat='galv', seg=12)
    p.cyl(0.35, 0.12, loc=(0.3, 0.2, 1.35), r2=0.15, mat='metal_dark', seg=12)
    a.add(p)
    return a


def k_ac(col):
    a = Asset('k_ac', col)
    p = a.part('M')
    p.box((1.1, 0.8, 0.9), loc=(0, 0, 0.45), mat='white', bevel=0.03)
    p.cyl(0.33, 0.04, loc=(0, -0.41, 0.48), rot=(R(90), 0, 0), mat='black', seg=16)
    for k in range(3):
        ang = k * 2 * math.pi / 3
        p.box((0.28, 0.02, 0.08), loc=(math.cos(ang) * 0.14, -0.43, 0.48 + math.sin(ang) * 0.14), rot=(0, ang, 0), mat='metal_grey')
    p.torus(0.33, 0.015, loc=(0, -0.44, 0.48), rot=(R(90), 0, 0), mat='metal_dark', seg=18, rseg=4)
    p.beam((0.4, 0.3, 0.1), (0.4, 0.8, 0.1), 0.03, mat='copper')
    a.add(p)
    return a


def k_railing(col):
    """Geländer, 2 m lang entlang X."""
    a = Asset('k_railing', col)
    p = a.part('M')
    for x in (-1.0, 0.0, 1.0):
        p.box((0.06, 0.06, 1.05), loc=(x, 0, 0.52), mat='hazard')
    p.box((2.06, 0.06, 0.06), loc=(0, 0, 1.05), mat='hazard')
    p.box((2.0, 0.04, 0.04), loc=(0, 0, 0.55), mat='hazard')
    p.box((2.0, 0.02, 0.12), loc=(0, 0, 0.06), mat='metal_dark')
    a.add(p)
    return a


# ================================================================ Landmarken
def l_crane(col):
    """Portalkran über dem Überlauf: Stützen bei x=±18, y=±6, Höhe 12."""
    a = Asset('l_crane', col)
    p = a.part('M')
    for sx in (-18, 18):
        for sy in (-6, 6):
            p.box((1.2, 1.2, 12), loc=(sx, sy, 6), mat='orange', bevel=0.05)
            p.box((1.8, 1.8, 0.4), loc=(sx, sy, 0.2), mat='metal_dark', bevel=0.05)
        # Streben erst oberhalb der Durchfahrt (die Dammstraße bleibt frei)
        for k in range(2):
            z0, z1 = 6 + k * 3, 9 + k * 3
            p.beam((sx, -6, z0), (sx, 6, z1), 0.12, mat='orange', square=True)
            p.beam((sx, 6, z0), (sx, -6, z1), 0.12, mat='orange', square=True)
        p.box((1.0, 12.0, 0.6), loc=(sx, 0, 6.0), mat='orange', bevel=0.04)
        p.box((1.4, 13.4, 1.2), loc=(sx, 0, 12.4), mat='orange', bevel=0.05)
    for sy in (-0.8, 0.8):
        p.box((38.4, 0.5, 1.4), loc=(0, sy, 13.3), mat='orange', bevel=0.04)
        for k in range(19):
            x0 = -18 + k * 2
            p.beam((x0, sy, 12.7), (x0 + 2, sy, 13.9), 0.06, mat='orange', square=True)
    p.box((3.0, 2.4, 1.6), loc=(4, 0, 12.2), mat='metal_dark', bevel=0.08)
    p.box((1.6, 1.2, 1.0), loc=(4, 0, 12.8), mat='paint_cream', bevel=0.05)
    p.beam((4, 0, 11.4), (4, 0, 6.0), 0.03, mat='black')
    p.box((0.6, 0.4, 0.5), loc=(4, 0, 5.8), mat='hazard', bevel=0.04)
    p.torus(0.25, 0.06, loc=(4, 0, 5.3), rot=(R(90), 0, 0), mat='metal_dark', seg=14, rseg=6)
    a.add(p)
    return a


def l_turbine(col):
    """Generator im Kraftwerk, Radius 3,4, Höhe 4,5 (+ Aufbau)."""
    a = Asset('l_turbine', col)
    p = a.part('M')
    p.cyl(3.5, 0.4, loc=(0, 0, 0.2), mat='metal_dark', seg=24)
    p.cyl(3.4, 3.6, loc=(0, 0, 2.2), mat='teal', seg=24)
    for k in range(12):
        ang = k * math.pi / 6
        p.box((0.25, 0.3, 3.6), loc=(math.cos(ang) * 3.42, math.sin(ang) * 3.42, 2.2), rot=(0, 0, ang), mat='teal_dark')
    p.cyl(3.5, 0.3, loc=(0, 0, 4.1), mat='metal_dark', seg=24)
    p.cyl(2.2, 1.2, loc=(0, 0, 4.8), r2=1.6, mat='teal', seg=20)
    p.cyl(1.0, 0.6, loc=(0, 0, 5.7), mat='metal_grey', seg=16)
    p.cyl(3.45, 0.3, loc=(0, 0, 3.0), mat='hazard', seg=24, caps=False)
    for k in range(4):
        ang = k * math.pi / 2 + 0.4
        p.beam((math.cos(ang) * 3.3, math.sin(ang) * 3.3, 4.0), (math.cos(ang) * 4.2, math.sin(ang) * 4.2, 6.0), 0.12, mat='galv')
    p.box((1.2, 0.8, 1.6), loc=(0, 4.2, 0.8), mat='hazard', bevel=0.04)
    p.box((0.9, 0.05, 0.6), loc=(0, 3.78, 1.1), mat='black')
    a.add(p)
    g = a.part('Glow')
    for k in range(3):
        g.box((0.14, 0.04, 0.08), loc=(-0.25 + k * 0.25, 3.75, 1.25), mat='cell_green' if k else 'arc_eye')
    a.add(g)
    return a


def l_radar(col):
    a = Asset('l_radar', col)
    p = a.part('M')
    p.box((1.2, 1.2, 0.4), loc=(0, 0, 0.2), mat='metal_dark', bevel=0.04)
    p.cyl(0.25, 3.0, loc=(0, 0, 1.9), mat='galv', seg=10)
    p.sphere(3.2, loc=(0, 0.4, 4.2), rot=(R(-60), 0, 0), mat='concrete_light', seg=24, rings=12, scale=(1, 1, 0.38), hemi=True)
    p.beam((0, 0.6, 4.3), (0, 2.6, 5.4), 0.06, mat='metal_dark')
    p.cyl(0.25, 0.5, loc=(0, 2.7, 5.45), rot=(R(-60), 0, 0), mat='metal_dark', seg=10)
    for k in range(3):
        ang = k * 2 * math.pi / 3
        p.beam((math.cos(ang) * 2.2, 0.4 + math.sin(ang) * 1.1, 4.4 + math.sin(ang) * 1.9), (0, 2.6, 5.4), 0.03, mat='metal_dark')
    a.add(p)
    return a


def l_dome(col):
    """Geodätischer Rahmen für eine Kuppel (Radius 1, wird im Spiel skaliert)."""
    import bmesh as _b
    a = Asset('l_dome', col)
    p = a.part('M')
    tb = _b.new()
    _b.ops.create_icosphere(tb, subdivisions=2, radius=1.0)
    done = set()
    for e in tb.edges:
        v0, v1 = e.verts[0].co, e.verts[1].co
        if v0.z < -0.02 or v1.z < -0.02:
            continue
        key = tuple(sorted([tuple(round(c, 4) for c in v0), tuple(round(c, 4) for c in v1)]))
        if key in done:
            continue
        done.add(key)
        p.beam(tuple(v0), tuple(v1), 0.012, mat='metal_dark', seg=4)
    tb.free()
    p.torus(1.0, 0.025, mat='metal_dark', seg=32, rseg=4)
    a.add(p)
    return a


def l_queen(col):
    """Riesiges Wrack einer ARC-Königin, halb im Boden. ~55 m lang entlang X."""
    a = Asset('l_queen', col)
    p = a.part('M')
    rnd = random.Random(11)
    for i in range(12):
        x = -24 + i * 4.3
        z = 4 + math.sin(i * 0.45) * 3
        p.sphere(2.3, loc=(x, 0, z), rot=(0, rnd.uniform(-0.3, 0.3), 0), mat='arc_frame', seg=14, rings=10, scale=(1.1, 1.0, 0.9))
        p.box((2.0, 3.2, 1.2), loc=(x, 0, z + 2.0), rot=(0, 0, rnd.uniform(-0.1, 0.1)), mat='arc_shell', bevel=0.2)
        if 1 < i < 10:
            for s in (-1, 1):
                # geschwungene Rippen aus drei Segmenten
                pts = [(x, s * 1.5, z + 1.0), (x + 0.3, s * 5.5, z + 2.5), (x + 0.6, s * 9.0, z - 1.0), (x + 0.8, s * 10.0, -2.0)]
                for k in range(3):
                    p.beam(pts[k], pts[k + 1], 0.45 - k * 0.1, mat='arc_shell2' if k else 'arc_shell', square=True)
                p.box((1.5, 3.0, 0.4), loc=(x + 0.3, s * 5.0, z + 2.4), rot=(s * R(-25), 0, 0), mat='arc_panel', bevel=0.1)
    # Kopf
    p.sphere(5.0, loc=(29, 2, 5.5), mat='arc_frame', seg=20, rings=14, scale=(1.1, 0.9, 0.85))
    p.sphere(5.3, loc=(28.5, 2, 6.5), mat='arc_shell', seg=20, rings=12, scale=(1.05, 0.9, 0.7), hemi=True)
    p.box((1.2, 7.0, 0.5), loc=(29, 2, 10.4), mat='arc_orange', bevel=0.1)
    for s in (-1, 1):
        p.beam((33, 2 + s * 2.5, 4.0), (38, 2 + s * 3.5, 1.0), 0.6, mat='arc_joint', square=True)
        p.beam((38, 2 + s * 3.5, 1.0), (40, 2 + s * 1.5, -1.0), 0.45, mat='arc_joint', square=True)
    p.sphere(1.6, loc=(34.2, 2, 6.2), mat='rust_dark', seg=14, rings=10)
    # Beine
    for (x, y, dx, dy, h) in [(-12, -9, -4, -6, 16), (6, 11, 3, 6, 14), (16, -12, 5, -5, 17), (-20, 10, -5, 4, 12)]:
        knee = (x + dx * 0.5, y + dy * 0.5, h)
        p.sphere(1.4, loc=(x, y, 2.0), mat='arc_joint', seg=12, rings=8)
        p.beam((x, y, 2.0), knee, 0.9, mat='arc_panel', square=True)
        p.sphere(1.1, loc=knee, mat='arc_joint', seg=12, rings=8)
        p.beam(knee, (x + dx, y + dy, h * 0.55), 0.6, mat='arc_frame')
    p.paint_top('moss', 0.8)
    a.add(p)
    return a


def l_elevator(col):
    """Extraktionsaufzug. Tür zeigt nach +Y (im Spiel: zur Kartenmitte)."""
    a = Asset('l_elevator', col)
    p = a.part('M')
    p.box((8, 8, 0.4), loc=(0, 0, -0.05), mat='concrete', bevel=0.05)
    p.box((8.02, 0.6, 0.06), loc=(0, 3.7, 0.16), mat='hazard')
    p.box((3.2, 3.2, 0.05), loc=(0, 0, 0.17), mat='hazard')
    for sx in (-1.8, 1.8):
        for sy in (-1.6, 1.6):
            p.box((0.4, 0.4, 5.2), loc=(sx, sy, 2.7), mat='metal_dark', bevel=0.03)
    p.box((4.2, 3.8, 0.4), loc=(0, 0, 5.35), mat='concrete_light', bevel=0.06)
    p.box((3.4, 0.2, 5.0), loc=(0, -1.7, 2.65), mat='metal_grey')
    for s in (-1, 1):
        p.box((0.2, 3.0, 5.0), loc=(s * 1.9, 0, 2.65), mat='metal_grey')
        for k in range(5):
            p.box((0.22, 3.0, 0.08), loc=(s * 1.9, 0, 0.8 + k * 1.0), mat='metal_dark')
    p.box((3.6, 0.3, 0.6), loc=(0, 1.75, 4.95), mat='hazard')
    p.cyl(0.35, 0.3, loc=(0, 0, 5.7), mat='metal_dark', seg=12)
    # Konsole
    p.box((0.6, 0.5, 1.1), loc=(2.8, 2.6, 0.7), mat='metal_dark', bevel=0.03)
    p.box((0.7, 0.55, 0.12), loc=(2.8, 2.55, 1.25), rot=(R(-20), 0, 0), mat='paint_cream', bevel=0.02)
    a.add(p)
    for i, s in enumerate((-1, 1)):
        d = a.part(f'Door{i}')
        d.box((1.6, 0.15, 4.4), loc=(0, 0, 2.2), mat='paint_yellow', bevel=0.02)
        for k in range(4):
            d.box((1.5, 0.17, 0.08), loc=(0, 0, 0.6 + k * 1.1), mat='hazard')
        a.add(d, loc=(s * 0.8, 1.75, 0.2))
    lamp = a.part('Lamp')
    lamp.box((0.4, 0.3, 0.12), mat='lamp')
    lamp.cyl(0.25, 0.4, loc=(-2.8, -2.6, 4.9), mat='lamp', seg=12)
    a.add(lamp, loc=(2.8, 2.6, 1.4))
    return a


# ================================================================ Vegetation
def v_pine(col):
    a = Asset('v_pine', col)
    p = a.part('M')
    p.cyl(0.22, 3.0, loc=(0, 0, 1.5), r2=0.1, mat='bark', seg=7)
    rnd = random.Random(5)
    for k in range(4):
        z = 1.6 + k * 1.3
        r = 2.1 - k * 0.45
        p.cyl(r, 1.9, loc=(rnd.uniform(-0.08, 0.08), rnd.uniform(-0.08, 0.08), z + 0.9), r2=0.05, mat='needles' if k % 2 else 'needles2', seg=8, smooth=False)
    a.add(p)
    return a


def v_tree(col):
    a = Asset('v_tree', col)
    p = a.part('M')
    rnd = random.Random(17)
    p.cyl(0.25, 3.2, loc=(0, 0, 1.6), r2=0.14, mat='bark', seg=7)
    for (x, y, z) in [(0.9, 0.3, 3.4), (-0.8, 0.5, 3.1), (0.1, -0.9, 3.6), (-0.4, -0.5, 4.0)]:
        p.beam((0, 0, 2.4), (x, y, z), 0.09, mat='bark', seg=5)
    blobs = [((0, 0, 4.4), 1.6), ((1.1, 0.4, 3.9), 1.15), ((-1.0, 0.6, 3.7), 1.2), ((0.2, -1.0, 4.0), 1.15),
             ((-0.6, -0.6, 4.6), 1.0), ((0.7, -0.3, 4.9), 1.0), ((0.1, 0.2, 5.4), 1.05), ((-0.3, 0.9, 4.8), 0.95)]
    for i, (c, r) in enumerate(blobs):
        p.ico(r, loc=c, mat=rnd.choice(['leaves', 'leaves2', 'leaves']), sub=1, jitter=0.35, seed=i + 3)
    a.add(p)
    return a


def v_bush(col):
    a = Asset('v_bush', col)
    p = a.part('M')
    for i, (c, r) in enumerate([((0, 0, 0.35), 0.75), ((0.55, 0.2, 0.25), 0.55), ((-0.5, -0.2, 0.25), 0.5)]):
        p.ico(r, loc=c, mat='leaves' if i != 1 else 'leaves2', sub=1, jitter=0.3, seed=i + 20, scale=(1, 1, 0.75))
    a.add(p)
    return a


def _rock(col, name, seed, scale):
    a = Asset(name, col)
    p = a.part('M')
    p.ico(1.0, loc=(0, 0, 0.25), mat='rock', sub=2, jitter=0.45, seed=seed, scale=scale)
    p.ico(0.5, loc=(0.6, 0.3, 0.1), mat='rock_dark', sub=1, jitter=0.4, seed=seed + 1)
    p.paint_top('moss', 0.8)
    a.add(p)
    return a


def v_rock0(col): return _rock(col, 'v_rock0', 31, (1.3, 1.0, 0.7))
def v_rock1(col): return _rock(col, 'v_rock1', 47, (1.0, 1.1, 0.9))
def v_rock2(col): return _rock(col, 'v_rock2', 59, (1.5, 0.9, 0.55))


def v_grass(col):
    a = Asset('v_grass', col)
    p = a.part('M')
    rnd = random.Random(9)
    for k in range(8):
        ang = rnd.uniform(0, 2 * math.pi)
        h = rnd.uniform(0.35, 0.75)
        x, y = math.cos(ang) * 0.1, math.sin(ang) * 0.1
        tx, ty = -math.sin(ang) * 0.045, math.cos(ang) * 0.045
        lean = rnd.uniform(0.05, 0.2)
        p.face([(x - tx, y - ty, 0), (x + tx, y + ty, 0), (x + math.cos(ang) * lean, y + math.sin(ang) * lean, h)], mat='grass' if k % 3 else 'leaves2')
    a.add(p)
    return a


# ================================================================ Raider (Spielfigur)
def ch_raider(col):
    """Hierarchie wie im Spiel: Body > Hips > LegL/LegR > KneeL/KneeR, Body > Torso > Head/ArmL/ArmR/GunMount."""
    a = Asset('ch_raider', col)
    body = a.node('Body')
    hips = a.node('Hips', parent=body, loc=(0, 0, 0.95))
    for side, s in (('L', -1), ('R', 1)):
        leg = a.node(f'Leg{side}', parent=hips, loc=(s * 0.12, 0, 0))
        u = a.part(f'Leg{side}U')
        u.box((0.18, 0.2, 0.5), loc=(0, 0, -0.25), mat='pants', bevel=0.04)
        u.box((0.19, 0.08, 0.14), loc=(s * 0.02, 0.04, -0.22), mat='strap', bevel=0.02)
        a.add(u, parent=leg)
        knee = a.node(f'Knee{side}', parent=leg, loc=(0, 0, -0.48))
        l = a.part(f'Leg{side}L')
        l.box((0.16, 0.18, 0.44), loc=(0, 0, -0.22), mat='pants', bevel=0.04)
        l.box((0.17, 0.08, 0.14), loc=(0, 0.07, -0.02), mat='helmet', bevel=0.03)
        l.box((0.17, 0.28, 0.12), loc=(0, 0.04, -0.44), mat='boots', bevel=0.03)
        l.box((0.18, 0.2, 0.12), loc=(0, -0.01, -0.34), mat='boots', bevel=0.03)
        a.add(l, parent=knee)
    torso = a.node('Torso', parent=body, loc=(0, 0, 0.95))
    t = a.part('TorsoM')
    t.box((0.48, 0.3, 0.6), loc=(0, 0, 0.33), mat='jacket', bevel=0.06, taper=(1.05, 1.0))
    t.box((0.5, 0.32, 0.1), loc=(0, 0, 0.05), mat='strap', bevel=0.02)
    for x in (-0.14, 0.14):
        t.box((0.12, 0.08, 0.12), loc=(x, 0.17, 0.08), mat='canvas', bevel=0.02)
    t.box((0.5, 0.12, 0.12), loc=(0, 0.05, 0.6), mat='arc_red', bevel=0.04)
    t.box((0.2, 0.05, 0.25), loc=(0.1, 0.16, 0.35), mat='jacket_dark', bevel=0.02)
    for s in (-1, 1):
        t.box((0.16, 0.26, 0.1), loc=(s * 0.25, 0, 0.6), mat='jacket_dark', bevel=0.04)
        t.beam((s * 0.15, 0.15, 0.6), (s * 0.12, 0.16, 0.05), 0.02, mat='strap', square=True)
    # Rucksack
    t.box((0.42, 0.26, 0.52), loc=(0, -0.28, 0.38), mat='pack', bevel=0.06)
    t.box((0.44, 0.2, 0.14), loc=(0, -0.28, 0.68), mat='canvas', bevel=0.04)
    t.cyl(0.08, 0.5, loc=(0, -0.3, 0.05), rot=(0, R(90), 0), mat='olive', seg=10)
    t.box((0.3, 0.06, 0.2), loc=(0, -0.43, 0.35), mat='jacket_dark', bevel=0.02)
    t.cyl(0.02, 0.35, loc=(-0.16, -0.36, 0.88), mat='metal_dark', seg=6)
    a.add(t, parent=torso)
    head = a.node('Head', parent=torso, loc=(0, 0, 0.7))
    h = a.part('HeadM')
    h.sphere(0.13, loc=(0, 0, 0.09), mat='skin', seg=12, rings=10)
    h.sphere(0.16, loc=(0, -0.01, 0.12), mat='helmet', seg=16, rings=10, scale=(1, 1.08, 0.9), hemi=True)
    h.box((0.3, 0.3, 0.025), loc=(0, 0, 0.12), mat='helmet', bevel=0.01)
    h.box((0.23, 0.07, 0.06), loc=(0, 0.13, 0.1), mat='visor', bevel=0.015)
    h.cyl(0.03, 0.04, loc=(0.1, 0.1, 0.19), rot=(R(90), 0, 0), mat='lamp', seg=8)
    h.box((0.18, 0.1, 0.08), loc=(0, 0.08, 0.0), mat='arc_red', bevel=0.03)
    a.add(h, parent=head)
    for side, s in (('L', -1), ('R', 1)):
        arm = a.node(f'Arm{side}', parent=torso, loc=(s * 0.29, 0, 0.56))
        m = a.part(f'Arm{side}M')
        m.box((0.13, 0.14, 0.3), loc=(0, 0, -0.14), mat='jacket', bevel=0.03)
        m.box((0.12, 0.13, 0.26), loc=(0, 0, -0.4), mat='jacket_dark', bevel=0.03)
        m.box((0.11, 0.12, 0.1), loc=(0, 0, -0.55), mat='strap', bevel=0.03)
        a.add(m, parent=arm)
    a.node('GunMount', parent=torso, loc=(0.24, 0.3, 0.42))
    return a


PROPS = [p_container, p_car_a, p_car_b, p_barrier, p_sandbags, p_crate, p_barrel, p_lamp, p_pylon, p_tank, p_busstop]
KIT = [k_window, k_shutter, k_door, k_rolldoor, k_vent, k_ac, k_railing]
LANDMARKS = [l_crane, l_turbine, l_radar, l_dome, l_queen, l_elevator]
VEG = [v_pine, v_tree, v_bush, v_rock0, v_rock1, v_rock2, v_grass]
CHARS = [ch_raider]
