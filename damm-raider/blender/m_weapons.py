# Waffenmodelle. Griff im Ursprung, Lauf zeigt nach +Y, oben = +Z.
# Jede Waffe bekommt ein Empty "<id>__Muzzle" an der Laufmündung.
import math
from kit import Asset, lathe

ACCENT = {0: 'paint_cream', 1: 'paint_teal', 2: 'paint_blue', 3: 'paint_purple', 4: 'paint_gold'}
R = math.radians


def _grip(p, loc=(0, 0, 0), mat='gun_polymer', h=0.11, back=0.035, w=0.032):
    # schräger Pistolengriff
    p.prism([(-0.018, 0), (0.022, 0), (0.022 - back, -h), (-0.022 - back, -h)], w, loc=loc, mat=mat, bevel=0.004)


def _trigger(p, y=0.02, z=-0.005):
    p.prism([(0, 0), (0.05, 0), (0.05, -0.006), (0.004, -0.035), (-0.004, -0.035), (0, -0.006)], 0.006, loc=(0, y - 0.01, z), mat='gun_black')
    p.box((0.004, 0.012, 0.02), loc=(0, y + 0.012, z - 0.015), rot=(R(15), 0, 0), mat='gun_black')


def _mag(p, y, z, h=0.12, curve=0.03, w=0.026, d=0.035, mat='gun_black'):
    p.prism([(-d / 2, 0), (d / 2, 0), (d / 2 + curve, -h), (-d / 2 + curve, -h)], w, loc=(0, y, z), mat=mat, bevel=0.003)


def _sight(p, y, z, mat='gun_black'):
    p.box((0.012, 0.015, 0.018), loc=(0, y, z), mat=mat, bevel=0.002)


def _scope(p, y, z, length=0.16, r=0.02, mat='gun_black'):
    p.cyl(r, length, loc=(0, y, z), rot=(R(90), 0, 0), mat=mat, seg=14)
    p.cyl(r * 1.35, 0.035, loc=(0, y + length / 2, z), rot=(R(90), 0, 0), mat=mat, seg=14, r2=r * 1.1)
    p.cyl(r * 1.25, 0.03, loc=(0, y - length / 2, z), rot=(R(90), 0, 0), mat=mat, seg=14, r2=r * 1.05)
    p.box((0.014, 0.02, 0.025), loc=(0, y - 0.03, z - r - 0.008), mat=mat)
    p.box((0.014, 0.02, 0.025), loc=(0, y + 0.03, z - r - 0.008), mat=mat)


def _muzzle(a, y, z):
    a.node('Muzzle', loc=(0, y, z))


def nadel(col):
    a = Asset('w_nadel', col)
    p = a.part('Body')
    acc = ACCENT[0]
    p.box((0.032, 0.19, 0.034), loc=(0, 0.035, 0.058), mat='gun_steel', bevel=0.004)
    for i in range(5):
        p.box((0.034, 0.004, 0.02), loc=(0, -0.045 + i * 0.008, 0.062), mat='gun_black')
    p.box((0.03, 0.13, 0.026), loc=(0, 0.03, 0.03), mat='gun_polymer', bevel=0.004)
    p.cyl(0.009, 0.03, loc=(0, 0.13, 0.058), rot=(R(90), 0, 0), mat='gun_black', seg=10)
    _grip(p, loc=(0, -0.02, 0.022), mat='gun_polymer', h=0.1)
    p.prism([(-0.012, 0), (0.016, 0), (0.016 - 0.03, -0.085), (-0.012 - 0.03, -0.085)], 0.036, loc=(0, -0.022, 0.012), mat=acc, bevel=0.003)
    _trigger(p, y=0.02, z=0.018)
    p.prism([(-0.01, 0), (0.07, 0), (0.07, -0.008), (-0.01, -0.03)], 0.02, loc=(0, -0.005, 0.017), mat='gun_polymer')
    _sight(p, 0.11, 0.08)
    _sight(p, -0.05, 0.08)
    a.add(p)
    _muzzle(a, 0.15, 0.058)
    return a


def stichler(col):
    a = Asset('w_stichler', col)
    acc = ACCENT[1]
    p = a.part('Body')
    p.box((0.045, 0.26, 0.06), loc=(0, 0.06, 0.05), mat='gun_steel', bevel=0.006)
    p.box((0.047, 0.12, 0.02), loc=(0, 0.1, 0.085), mat=acc, bevel=0.004)
    p.cyl(0.018, 0.14, loc=(0, 0.25, 0.055), rot=(R(90), 0, 0), mat='gun_black', seg=12)
    for i in range(4):
        p.cyl(0.021, 0.012, loc=(0, 0.2 + i * 0.025, 0.055), rot=(R(90), 0, 0), mat='gun_steel', seg=12)
    _mag(p, 0.1, 0.022, h=0.17, curve=0.01, w=0.024, d=0.03)
    _grip(p, loc=(0, -0.02, 0.022), h=0.1)
    _trigger(p, y=0.02, z=0.02)
    # Drahtschaft
    p.beam((0, -0.07, 0.07), (0, -0.28, 0.06), 0.006, mat='gun_black')
    p.beam((0, -0.07, 0.03), (0, -0.28, 0.02), 0.006, mat='gun_black')
    p.box((0.04, 0.02, 0.07), loc=(0, -0.285, 0.04), mat='rubber', bevel=0.004)
    _sight(p, 0.17, 0.09)
    p.box((0.02, 0.03, 0.02), loc=(0, -0.04, 0.09), mat='gun_black')
    p.box((0.05, 0.06, 0.012), loc=(0, 0.05, 0.02), mat='tape')
    a.add(p)
    _muzzle(a, 0.33, 0.055)
    return a


def klapper(col):
    a = Asset('w_klapper', col)
    acc = ACCENT[1]
    p = a.part('Body')
    p.box((0.05, 0.3, 0.065), loc=(0, 0.06, 0.055), mat='gun_steel', bevel=0.006)
    p.box((0.058, 0.22, 0.06), loc=(0, 0.31, 0.05), mat=acc, bevel=0.008)
    for i in range(5):
        p.box((0.06, 0.018, 0.014), loc=(0, 0.23 + i * 0.035, 0.066), mat='gun_black')
    p.cyl(0.013, 0.2, loc=(0, 0.5, 0.055), rot=(R(90), 0, 0), mat='gun_black', seg=12)
    p.cyl(0.02, 0.05, loc=(0, 0.6, 0.055), rot=(R(90), 0, 0), mat='gun_steel', seg=8, bevel=0.003)
    # Tragegriff mit Kimme
    p.box((0.016, 0.16, 0.012), loc=(0, 0.06, 0.12), mat='gun_black', bevel=0.003)
    p.box((0.014, 0.014, 0.03), loc=(0, -0.01, 0.1), mat='gun_black')
    p.box((0.014, 0.014, 0.03), loc=(0, 0.13, 0.1), mat='gun_black')
    p.box((0.012, 0.012, 0.045), loc=(0, 0.42, 0.095), mat='gun_black')
    _mag(p, 0.13, 0.025, h=0.17, curve=0.05, w=0.028, d=0.04)
    _grip(p, loc=(0, 0.0, 0.025), h=0.11)
    _trigger(p, y=0.045, z=0.023)
    # Schaft
    p.prism([(0, 0.02), (0, -0.05), (-0.26, -0.07), (-0.27, 0.035), (-0.08, 0.04)], 0.04, loc=(0, -0.09, 0.05), mat='gun_polymer', bevel=0.006)
    p.box((0.044, 0.02, 0.12), loc=(0, -0.365, 0.035), mat='rubber', bevel=0.005)
    p.box((0.06, 0.05, 0.015), loc=(0, 0.3, 0.018), mat='tape')
    a.add(p)
    _muzzle(a, 0.63, 0.055)
    return a


def kessel(col):
    a = Asset('w_kessel', col)
    acc = ACCENT[2]
    p = a.part('Body')
    p.box((0.046, 0.28, 0.06), loc=(0, 0.05, 0.05), mat='gun_steel', bevel=0.006)
    p.prism([(0, 0.03), (0.4, 0.03), (0.4, -0.01), (0.04, -0.035), (0, -0.03)], 0.05, loc=(0, 0.17, 0.045), mat='gun_wood', bevel=0.006)
    p.cyl(0.012, 0.42, loc=(0, 0.55, 0.06), rot=(R(90), 0, 0), mat='gun_black', seg=12)
    p.cyl(0.02, 0.06, loc=(0, 0.76, 0.06), rot=(R(90), 0, 0), mat='gun_steel', seg=10, bevel=0.003)
    p.box((0.048, 0.07, 0.02), loc=(0, 0.33, 0.035), mat=acc, bevel=0.004)
    _mag(p, 0.1, 0.02, h=0.08, curve=0.005, w=0.03, d=0.06)
    _grip(p, loc=(0, -0.02, 0.022), h=0.1, mat='gun_wood_dark')
    _trigger(p, y=0.02, z=0.02)
    p.prism([(0, 0.035), (0, -0.04), (-0.3, -0.08), (-0.31, 0.03), (-0.12, 0.05)], 0.042, loc=(0, -0.08, 0.05), mat='gun_wood', bevel=0.007)
    p.box((0.046, 0.02, 0.13), loc=(0, -0.39, 0.03), mat='rubber', bevel=0.005)
    _scope(p, 0.06, 0.125, length=0.2, r=0.022)
    p.cyl(0.02, 0.004, loc=(0, 0.162, 0.125), rot=(R(90), 0, 0), mat='lens', seg=14)
    a.add(p)
    _muzzle(a, 0.8, 0.06)
    return a


def stier(col):
    a = Asset('w_stier', col)
    acc = ACCENT[2]
    p = a.part('Body')
    p.box((0.05, 0.22, 0.07), loc=(0, 0.04, 0.05), mat='gun_steel', bevel=0.007)
    p.cyl(0.017, 0.55, loc=(0, 0.42, 0.07), rot=(R(90), 0, 0), mat='gun_black', seg=14)
    p.cyl(0.014, 0.45, loc=(0, 0.37, 0.035), rot=(R(90), 0, 0), mat='gun_steel', seg=12)
    p.cyl(0.026, 0.16, loc=(0, 0.3, 0.035), rot=(R(90), 0, 0), mat='gun_wood', seg=12, bevel=0.004)
    for i in range(3):
        p.cyl(0.028, 0.01, loc=(0, 0.24 + i * 0.06, 0.035), rot=(R(90), 0, 0), mat='gun_wood_dark', seg=12)
    p.cyl(0.02, 0.02, loc=(0, 0.6, 0.07), rot=(R(90), 0, 0), mat=acc, seg=12)
    p.box((0.01, 0.012, 0.012), loc=(0, 0.68, 0.092), mat='gun_brass')
    _grip(p, loc=(0, -0.03, 0.022), h=0.1, mat='gun_wood_dark')
    _trigger(p, y=0.01, z=0.02)
    p.prism([(0, 0.04), (0, -0.04), (-0.32, -0.09), (-0.33, 0.035), (-0.1, 0.05)], 0.044, loc=(0, -0.07, 0.05), mat='gun_wood', bevel=0.008)
    p.box((0.048, 0.02, 0.14), loc=(0, -0.4, 0.025), mat='rubber', bevel=0.005)
    # Patronen am Schaft
    for i in range(4):
        p.cyl(0.009, 0.05, loc=(0.025, -0.2 + i * 0.022, 0.03), rot=(0, 0, 0), mat='paint_red', seg=8)
    a.add(p)
    _muzzle(a, 0.7, 0.07)
    return a


def amboss(col):
    a = Asset('w_amboss', col)
    acc = ACCENT[2]
    p = a.part('Body')
    p.box((0.036, 0.1, 0.05), loc=(0, 0.0, 0.05), mat='gun_steel', bevel=0.005)
    p.cyl(0.032, 0.055, loc=(0, 0.05, 0.055), rot=(R(90), 0, 0), mat='gun_steel', seg=12, bevel=0.004)
    for i in range(6):
        ang = i * math.pi / 3
        p.cyl(0.008, 0.057, loc=(math.cos(ang) * 0.02, 0.05, 0.055 + math.sin(ang) * 0.02), rot=(R(90), 0, 0), mat='gun_black', seg=8)
    p.box((0.028, 0.2, 0.032), loc=(0, 0.18, 0.066), mat='gun_steel', bevel=0.004)
    p.cyl(0.013, 0.2, loc=(0, 0.18, 0.058), rot=(R(90), 0, 0), mat='gun_black', seg=12)
    p.box((0.03, 0.2, 0.01), loc=(0, 0.18, 0.086), mat=acc)
    p.box((0.01, 0.012, 0.02), loc=(0, 0.27, 0.095), mat='gun_brass')
    p.box((0.012, 0.02, 0.018), loc=(0, -0.045, 0.08), mat='gun_black')
    p.prism([(-0.02, 0), (0.02, 0), (-0.02, -0.11), (-0.065, -0.11)], 0.04, loc=(0, -0.02, 0.03), mat='gun_wood', bevel=0.006)
    _trigger(p, y=0.015, z=0.025)
    a.add(p)
    _muzzle(a, 0.29, 0.06)
    return a


def donner(col):
    a = Asset('w_donner', col)
    acc = ACCENT[3]
    p = a.part('Body')
    p.box((0.07, 0.38, 0.09), loc=(0, 0.06, 0.06), mat='gun_steel', bevel=0.008)
    p.box((0.074, 0.2, 0.03), loc=(0, 0.02, 0.12), mat=acc, bevel=0.005)
    p.cyl(0.04, 0.3, loc=(0, 0.42, 0.065), rot=(R(90), 0, 0), mat='gun_black', seg=16)
    for i in range(6):
        for s in (-1, 1):
            p.cyl(0.008, 0.02, loc=(s * 0.038, 0.31 + i * 0.04, 0.065), rot=(0, R(90), 0), mat='gun_steel', seg=8)
    p.cyl(0.018, 0.2, loc=(0, 0.66, 0.065), rot=(R(90), 0, 0), mat='gun_black', seg=12)
    p.cyl(0.03, 0.07, loc=(0, 0.78, 0.065), rot=(R(90), 0, 0), mat='gun_steel', seg=10, bevel=0.004)
    # Trommelmagazin
    p.cyl(0.085, 0.07, loc=(0, 0.08, -0.03), rot=(0, R(90), 0), mat='gun_black', seg=20, bevel=0.006)
    p.cyl(0.05, 0.075, loc=(0, 0.08, -0.03), rot=(0, R(90), 0), mat=acc, seg=16)
    # Zweibein (eingeklappt)
    p.beam((0.02, 0.5, 0.03), (0.02, 0.7, 0.025), 0.006, mat='gun_black')
    p.beam((-0.02, 0.5, 0.03), (-0.02, 0.7, 0.025), 0.006, mat='gun_black')
    p.box((0.02, 0.24, 0.02), loc=(0, 0.12, 0.17), mat='gun_black', bevel=0.004)
    p.box((0.02, 0.02, 0.05), loc=(0, 0.02, 0.14), mat='gun_black')
    p.box((0.02, 0.02, 0.05), loc=(0, 0.22, 0.14), mat='gun_black')
    _grip(p, loc=(0, -0.06, 0.025), h=0.11)
    _trigger(p, y=-0.02, z=0.022)
    p.prism([(0, 0.05), (0, -0.05), (-0.25, -0.06), (-0.26, 0.05)], 0.05, loc=(0, -0.13, 0.06), mat='gun_polymer', bevel=0.007)
    p.box((0.052, 0.02, 0.12), loc=(0, -0.395, 0.055), mat='rubber', bevel=0.005)
    a.add(p)
    _muzzle(a, 0.82, 0.065)
    return a


def eisen(col):
    a = Asset('w_eisen', col)
    acc = ACCENT[3]
    p = a.part('Body')
    p.box((0.05, 0.22, 0.07), loc=(0, 0.04, 0.055), mat='gun_steel', bevel=0.007)
    p.cyl(0.02, 0.7, loc=(0, 0.5, 0.065), rot=(R(90), 0, 0), mat='gun_black', seg=14)
    p.box((0.07, 0.1, 0.06), loc=(0, 0.88, 0.065), mat='gun_steel', bevel=0.008)
    for i in range(3):
        p.box((0.074, 0.012, 0.04), loc=(0, 0.855 + i * 0.025, 0.065), mat='gun_black')
    p.prism([(0, 0.03), (0.34, 0.03), (0.34, -0.005), (0.02, -0.04), (0, -0.035)], 0.052, loc=(0, 0.15, 0.045), mat='gun_wood', bevel=0.006)
    p.box((0.054, 0.05, 0.03), loc=(0, 0.1, 0.04), mat=acc, bevel=0.004)
    p.cyl(0.016, 0.06, loc=(0.035, 0.0, 0.07), rot=(0, R(90), 0), mat='gun_steel', seg=10)
    _grip(p, loc=(0, -0.03, 0.022), h=0.1, mat='gun_wood_dark')
    _trigger(p, y=0.01, z=0.02)
    p.prism([(0, 0.04), (0, -0.045), (-0.34, -0.1), (-0.35, 0.04), (-0.12, 0.055)], 0.046, loc=(0, -0.07, 0.05), mat='gun_wood', bevel=0.008)
    p.box((0.05, 0.022, 0.15), loc=(0, -0.43, 0.02), mat='rubber', bevel=0.005)
    _scope(p, 0.08, 0.14, length=0.3, r=0.026)
    p.cyl(0.024, 0.004, loc=(0, 0.232, 0.14), rot=(R(90), 0, 0), mat='lens', seg=14)
    a.add(p)
    _muzzle(a, 0.94, 0.065)
    return a


def sturmwind(col):
    a = Asset('w_sturmwind', col)
    acc = ACCENT[4]
    p = a.part('Body')
    # Bullpup-Gehäuse
    p.prism([(-0.36, 0.0), (0.3, 0.0), (0.36, 0.05), (0.3, 0.1), (-0.1, 0.1), (-0.2, 0.08), (-0.36, 0.08)], 0.06, loc=(0, 0, 0.0), mat='paint_cream', bevel=0.008)
    p.box((0.064, 0.3, 0.02), loc=(0, 0.02, 0.1), mat='gun_black', bevel=0.004)
    p.box((0.066, 0.18, 0.02), loc=(0, 0.2, 0.05), mat=acc, bevel=0.004)
    p.box((0.066, 0.05, 0.05), loc=(0, -0.25, 0.04), mat=acc, bevel=0.004)
    p.box((0.067, 0.24, 0.008), loc=(0, 0.1, 0.075), mat='arc_weak')
    p.cyl(0.015, 0.2, loc=(0, 0.45, 0.05), rot=(R(90), 0, 0), mat='gun_black', seg=12)
    p.cyl(0.026, 0.06, loc=(0, 0.56, 0.05), rot=(R(90), 0, 0), mat='gun_steel', seg=6, bevel=0.004)
    _mag(p, -0.18, 0.0, h=0.12, curve=0.02, w=0.03, d=0.045)
    _grip(p, loc=(0, 0.06, 0.005), h=0.1)
    _trigger(p, y=0.1, z=0.003)
    _scope(p, 0.02, 0.14, length=0.12, r=0.02, mat='gun_black')
    p.cyl(0.018, 0.004, loc=(0, 0.082, 0.14), rot=(R(90), 0, 0), mat='lens', seg=14)
    p.box((0.062, 0.02, 0.11), loc=(0, -0.37, 0.035), mat='rubber', bevel=0.005)
    a.add(p)
    _muzzle(a, 0.6, 0.05)
    return a


BUILDERS = [nadel, stichler, klapper, kessel, stier, amboss, donner, eisen, sturmwind]


def build_all(col):
    return [b(col) for b in BUILDERS]
