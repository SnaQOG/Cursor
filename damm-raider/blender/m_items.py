# Beutebehälter (im Spiel) und Gegenstände (nur für die Inventar-Icons).
import math
from kit import Asset, lathe

R = math.radians


# ================================================================ Behälter
# Ursprung = Bodenmitte. Jeder Behälter hat ein leuchtendes "Lamp"-Teil,
# das im Spiel erlischt, sobald er durchsucht wurde.

def _lamp(a, size, loc):
    l = a.part('Lamp')
    l.box(size, mat='lamp', bevel=0.004)
    a.add(l, loc=loc)


def c_crate(col):
    a = Asset('c_crate', col)
    p = a.part('Body')
    W, D, H = 1.1, 0.8, 0.72
    p.box((W, D, H), loc=(0, 0, H / 2), mat='olive', bevel=0.02)
    for x in (-0.36, 0, 0.36):
        p.box((0.02, D + 0.01, H - 0.1), loc=(x, 0, H / 2), mat='olive_dark')
    p.box((W + 0.02, D + 0.02, 0.07), loc=(0, 0, H - 0.035), mat='olive_dark', bevel=0.01)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.box((0.09, 0.09, H + 0.02), loc=(sx * (W / 2 - 0.03), sy * (D / 2 - 0.03), H / 2), mat='metal_dark', bevel=0.008)
        p.torus(0.06, 0.012, loc=(sx * (W / 2 + 0.01), 0, H * 0.6), rot=(0, R(90), 0), mat='metal_dark', seg=12, rseg=6)
    p.box((0.3, 0.02, 0.12), loc=(0.2, -D / 2 - 0.005, H * 0.45), mat='paint_cream')
    a.add(p)
    _lamp(a, (0.18, 0.02, 0.05), (-0.25, -D / 2 - 0.012, H * 0.45))
    return a


def c_locker(col):
    a = Asset('c_locker', col)
    p = a.part('Body')
    W, D, H = 0.8, 0.55, 1.9
    p.box((W, D, H), loc=(0, 0, H / 2), mat='teal', bevel=0.015)
    p.box((0.012, 0.02, H - 0.1), loc=(0, -D / 2 - 0.005, H / 2), mat='teal_dark')
    for sx in (-0.2, 0.2):
        for k in range(5):
            p.box((0.22, 0.012, 0.012), loc=(sx, -D / 2 - 0.006, H - 0.2 - k * 0.04), mat='teal_dark')
        p.box((0.03, 0.03, 0.14), loc=(sx + (0.13 if sx < 0 else -0.13), -D / 2 - 0.015, H * 0.55), mat='galv', bevel=0.005)
        p.box((0.12, 0.012, 0.07), loc=(sx, -D / 2 - 0.006, H * 0.78), mat='paper')
    p.box((W + 0.02, D + 0.02, 0.05), loc=(0, 0, 0.025), mat='teal_dark')
    a.add(p)
    _lamp(a, (0.3, 0.02, 0.03), (0, -D / 2 - 0.012, H - 0.06))
    return a


def c_toolbox(col):
    a = Asset('c_toolbox', col)
    p = a.part('Body')
    W, D, H = 0.9, 0.45, 0.42
    p.box((W, D, H), loc=(0, 0, H / 2), mat='red', bevel=0.015)
    p.box((W + 0.01, D + 0.01, 0.03), loc=(0, 0, H * 0.72), mat='black')
    p.box((W - 0.1, D - 0.1, 0.06), loc=(0, 0, H + 0.03), mat='red', bevel=0.01)
    p.beam((-0.2, 0, H + 0.06), (-0.2, 0, H + 0.14), 0.012, mat='metal_dark')
    p.beam((0.2, 0, H + 0.06), (0.2, 0, H + 0.14), 0.012, mat='metal_dark')
    p.beam((-0.2, 0, H + 0.14), (0.2, 0, H + 0.14), 0.016, mat='black')
    for sx in (-0.32, 0.32):
        p.box((0.06, 0.03, 0.07), loc=(sx, -D / 2 - 0.01, H * 0.72), mat='galv', bevel=0.004)
    a.add(p)
    _lamp(a, (0.2, 0.02, 0.04), (0, -D / 2 - 0.01, H * 0.35))
    return a


def c_medical(col):
    a = Asset('c_medical', col)
    p = a.part('Body')
    W, D, H = 0.9, 0.4, 1.1
    p.box((W, D, H), loc=(0, 0, H / 2), mat='white', bevel=0.02)
    p.box((W - 0.08, 0.02, H - 0.1), loc=(0, -D / 2 - 0.005, H / 2), mat='ceramic_white', bevel=0.004)
    p.box((0.26, 0.012, 0.08), loc=(0, -D / 2 - 0.018, H * 0.6), mat='med_red')
    p.box((0.08, 0.012, 0.26), loc=(0, -D / 2 - 0.018, H * 0.6), mat='med_red')
    p.box((0.03, 0.03, 0.12), loc=(0.35, -D / 2 - 0.025, H * 0.5), mat='galv')
    p.box((W + 0.02, D + 0.02, 0.06), loc=(0, 0, 0.03), mat='plastic_grey')
    a.add(p)
    _lamp(a, (0.3, 0.02, 0.03), (0, -D / 2 - 0.018, H - 0.07))
    return a


def c_weapon(col):
    a = Asset('c_weapon', col)
    p = a.part('Body')
    W, D, H = 1.6, 0.7, 0.5
    p.box((W, D, H * 0.55), loc=(0, 0, H * 0.275), mat='olive_dark', bevel=0.03)
    p.box((W, D, H * 0.42), loc=(0, 0, H * 0.55 + H * 0.21), mat='olive_dark', bevel=0.03)
    p.box((W + 0.02, D + 0.02, 0.03), loc=(0, 0, H * 0.56), mat='black')
    for x in (-0.55, 0, 0.55):
        p.box((0.1, 0.05, 0.1), loc=(x, -D / 2 - 0.02, H * 0.56), mat='hazard', bevel=0.008)
    for x in (-0.5, -0.2, 0.2, 0.5):
        p.box((0.05, D + 0.02, 0.02), loc=(x, 0, H + 0.005), mat='black')
    p.box((0.4, 0.012, 0.1), loc=(-0.3, -D / 2 - 0.005, H * 0.28), mat='paint_cream')
    a.add(p)
    _lamp(a, (0.25, 0.02, 0.04), (0.35, -D / 2 - 0.012, H * 0.28))
    return a


def c_drawer(col):
    a = Asset('c_drawer', col)
    p = a.part('Body')
    W, D, H = 1.4, 0.7, 0.76
    p.box((W, D, 0.05), loc=(0, 0, H - 0.025), mat='wood_light', bevel=0.01)
    p.box((0.45, D - 0.05, H - 0.05), loc=(-W / 2 + 0.25, 0, (H - 0.05) / 2), mat='metal_grey', bevel=0.01)
    for k in range(3):
        z = 0.12 + k * 0.22
        p.box((0.4, 0.02, 0.18), loc=(-W / 2 + 0.25, -D / 2 + 0.02, z), mat='galv', bevel=0.005)
        p.box((0.12, 0.02, 0.02), loc=(-W / 2 + 0.25, -D / 2, z + 0.03), mat='metal_dark')
    p.box((0.05, D - 0.05, H - 0.05), loc=(W / 2 - 0.04, 0, (H - 0.05) / 2), mat='metal_grey')
    # Schreibmaschine / Papiere
    p.box((0.34, 0.26, 0.1), loc=(0.2, 0.05, H + 0.05), mat='car_green', bevel=0.02)
    p.box((0.22, 0.02, 0.12), loc=(0.2, 0.19, H + 0.12), mat='paper')
    p.box((0.2, 0.28, 0.01), loc=(-0.3, -0.05, H + 0.005), rot=(0, 0, R(12)), mat='paper')
    a.add(p)
    _lamp(a, (0.18, 0.02, 0.03), (-W / 2 + 0.25, -D / 2, 0.66))
    return a


def c_arcprobe(col):
    a = Asset('c_arcprobe', col)
    p = a.part('Body')
    lathe(p, [(0.0, 0.0), (0.45, 0.02), (0.55, 0.3), (0.55, 0.9), (0.42, 1.2), (0.18, 1.35), (0.0, 1.38)], mat='arc_shell', seg=20)
    p.cyl(0.57, 0.08, loc=(0, 0, 0.6), mat='arc_frame', seg=20)
    p.cyl(0.57, 0.08, loc=(0, 0, 0.35), mat='arc_orange', seg=20)
    for k in range(4):
        ang = k * math.pi / 2 + math.pi / 4
        p.box((0.06, 0.4, 0.7), loc=(math.cos(ang) * 0.6, math.sin(ang) * 0.6, 0.35), rot=(0, 0, ang + R(90)), mat='arc_frame', bevel=0.01)
    p.cyl(0.05, 0.4, loc=(0, 0, 1.55), mat='arc_joint', seg=8)
    a.add(p)
    l = a.part('Lamp')
    for k in range(3):
        ang = k * 2 * math.pi / 3
        l.box((0.16, 0.03, 0.06), loc=(math.cos(ang) * 0.53, math.sin(ang) * 0.53, 0.8), rot=(0, 0, ang + R(90)), mat='arc_eye')
    l.sphere(0.06, loc=(0, 0, 1.76), mat='arc_eye', seg=10, rings=6)
    a.add(l)
    return a


def c_arcwreck(col):
    a = Asset('c_arcwreck', col)
    p = a.part('Body')
    p.sphere(0.5, loc=(0, 0, 0.2), mat='arc_frame', seg=14, rings=8, scale=(1.3, 1, 0.55))
    p.sphere(0.55, loc=(0.25, 0.1, 0.35), rot=(R(20), R(-15), 0), mat='arc_shell', seg=14, rings=8, scale=(1.2, 0.9, 0.4), hemi=True)
    p.box((0.7, 0.08, 0.35), loc=(-0.5, -0.3, 0.2), rot=(R(10), R(25), R(30)), mat='arc_shell2', bevel=0.02)
    p.beam((-0.2, 0.3, 0.1), (0.5, 0.6, 0.5), 0.05, mat='arc_joint')
    p.beam((0.3, -0.35, 0.1), (0.8, -0.5, 0.3), 0.04, mat='arc_joint')
    a.add(p)
    _lamp(a, (0.1, 0.1, 0.1), (0.1, -0.35, 0.3))
    return a


CONTAINERS = [c_crate, c_locker, c_toolbox, c_medical, c_weapon, c_drawer, c_arcprobe, c_arcwreck]


# ================================================================ Gegenstände (Icons)
# Funktionen bekommen (asset) und bauen ein Teil; Maßstab egal (wird normiert).

def i_ammo_light(a):
    p = a.part('M')
    p.box((0.12, 0.08, 0.06), loc=(0, 0, 0.03), mat='car_sand', bevel=0.004)
    p.box((0.121, 0.081, 0.02), loc=(0, 0, 0.04), mat='paint_cream')
    for i in range(4):
        x = -0.04 + i * 0.027
        p.cyl(0.006, 0.03, loc=(x, 0, 0.075), mat='gun_brass', seg=10)
        p.cyl(0.006, 0.012, loc=(x, 0, 0.095), r2=0.002, mat='copper', seg=10)
    return p


def i_ammo_medium(a):
    p = a.part('M')
    p.box((0.16, 0.08, 0.1), loc=(0, 0, 0.05), mat='olive', bevel=0.006)
    p.box((0.165, 0.085, 0.015), loc=(0, 0, 0.1), mat='olive_dark', bevel=0.004)
    p.box((0.06, 0.015, 0.012), loc=(0, 0, 0.115), mat='metal_dark')
    p.box((0.08, 0.001, 0.03), loc=(-0.02, -0.041, 0.055), mat='paint_yellow')
    for i in range(3):
        p.cyl(0.007, 0.05, loc=(0.1 + i * 0.018, -0.02, 0.025), mat='gun_brass', seg=10)
        p.cyl(0.007, 0.015, loc=(0.1 + i * 0.018, -0.02, 0.057), r2=0.002, mat='copper', seg=10)
    return p


def i_ammo_heavy(a):
    p = a.part('M')
    for i in range(4):
        x = -0.045 + i * 0.03
        p.cyl(0.011, 0.08, loc=(x, 0, 0.04), mat='gun_brass', seg=12)
        p.cyl(0.011, 0.03, loc=(x, 0, 0.095), r2=0.003, mat='gun_steel', seg=12)
        p.cyl(0.012, 0.006, loc=(x, 0, 0.003), mat='gun_brass', seg=12)
    p.box((0.13, 0.012, 0.02), loc=(0, 0.012, 0.02), mat='metal_dark')
    return p


def i_ammo_shell(a):
    p = a.part('M')
    for i, (x, y) in enumerate([(-0.02, 0), (0.02, 0), (0, 0.03), (0.04, 0.03), (-0.04, 0.03)]):
        p.cyl(0.014, 0.06, loc=(x, y, 0.036), mat='paint_red', seg=12)
        p.cyl(0.0145, 0.015, loc=(x, y, 0.004), mat='gun_brass', seg=12)
    return p


def i_bandage(a):
    p = a.part('M')
    p.cyl(0.05, 0.07, rot=(R(90), 0, 0), mat='cloth', seg=18)
    p.cyl(0.015, 0.072, rot=(R(90), 0, 0), mat='paper', seg=12)
    p.box((0.1, 0.07, 0.004), loc=(0.07, 0, -0.048), mat='cloth')
    p.box((0.02, 0.071, 0.101), loc=(0.02, 0, 0), mat='med_red')
    return p


def i_medkit(a):
    p = a.part('M')
    p.box((0.2, 0.08, 0.14), loc=(0, 0, 0.07), mat='white', bevel=0.012)
    p.box((0.2, 0.082, 0.02), loc=(0, 0, 0.1), mat='plastic_grey')
    p.box((0.06, 0.001, 0.02), loc=(0, -0.041, 0.06), mat='med_red')
    p.box((0.02, 0.001, 0.06), loc=(0, -0.041, 0.06), mat='med_red')
    p.beam((-0.04, 0, 0.14), (-0.04, 0, 0.16), 0.006, mat='black')
    p.beam((0.04, 0, 0.14), (0.04, 0, 0.16), 0.006, mat='black')
    p.beam((-0.04, 0, 0.16), (0.04, 0, 0.16), 0.008, mat='black')
    return p


def i_shieldcell(a):
    p = a.part('M')
    p.cyl(0.035, 0.03, loc=(0, 0, 0.015), mat='metal_dark', seg=16, bevel=0.003)
    p.cyl(0.028, 0.1, loc=(0, 0, 0.08), mat='cell_blue', seg=16)
    for k in range(3):
        ang = k * 2 * math.pi / 3
        p.box((0.01, 0.01, 0.1), loc=(math.cos(ang) * 0.031, math.sin(ang) * 0.031, 0.08), mat='metal_dark')
    p.cyl(0.035, 0.025, loc=(0, 0, 0.14), mat='metal_dark', seg=16, bevel=0.003)
    p.cyl(0.012, 0.015, loc=(0, 0, 0.16), mat='paint_yellow', seg=10)
    return p


def i_grenade(a):
    p = a.part('M')
    p.sphere(0.045, loc=(0, 0, 0.045), mat='olive', seg=14, rings=10, scale=(1, 1, 1.15))
    for k in range(3):
        p.torus(0.045, 0.004, loc=(0, 0, 0.03 + k * 0.02), mat='olive_dark', seg=20, rseg=5)
    p.cyl(0.016, 0.03, loc=(0, 0, 0.1), mat='metal_dark', seg=10)
    p.box((0.012, 0.012, 0.06), loc=(0.02, 0, 0.08), rot=(0, R(-15), 0), mat='metal_grey')
    p.torus(0.014, 0.003, loc=(-0.02, 0, 0.11), rot=(R(90), 0, 0), mat='metal_grey', seg=12, rseg=4)
    return p


def _shield(a, acc, size):
    p = a.part('M')
    p.cyl(0.08 * size, 0.035, loc=(0, 0, 0.018), mat='metal_dark', seg=24, bevel=0.004)
    p.cyl(0.065 * size, 0.012, loc=(0, 0, 0.04), mat=acc, seg=24)
    p.cyl(0.035 * size, 0.012, loc=(0, 0, 0.05), mat='cell_blue', seg=20)
    for k in range(6):
        ang = k * math.pi / 3
        p.box((0.02, 0.012, 0.03), loc=(math.cos(ang) * 0.08 * size, math.sin(ang) * 0.08 * size, 0.02), rot=(0, 0, ang), mat='metal_grey')
    p.box((0.2 * size, 0.03, 0.008), loc=(0, -0.09 * size, 0.01), mat='strap')
    return p


def i_shield_light(a): return _shield(a, 'paint_teal', 0.9)
def i_shield_medium(a): return _shield(a, 'paint_blue', 1.0)
def i_shield_heavy(a): return _shield(a, 'paint_purple', 1.1)


def i_scrap(a):
    p = a.part('M')
    p.box((0.12, 0.08, 0.008), loc=(0, 0, 0.02), rot=(R(8), R(-12), R(20)), mat='rust')
    p.box((0.1, 0.03, 0.03), loc=(0.02, 0.02, 0.04), rot=(0, R(20), R(-30)), mat='metal_grey')
    p.beam((-0.06, -0.04, 0.01), (0.05, 0.05, 0.06), 0.006, mat='rust_dark')
    p.box((0.05, 0.05, 0.01), loc=(-0.04, 0.03, 0.035), rot=(R(30), 0, R(10)), mat='galv')
    return p


def i_metal(a):
    p = a.part('M')
    for k in range(4):
        p.box((0.14, 0.09, 0.008), loc=(0.004 * k, 0.003 * k, 0.005 + k * 0.01), rot=(0, 0, R(k * 4)), mat='galv' if k % 2 else 'metal_grey', bevel=0.002)
    p.box((0.14, 0.02, 0.02), loc=(0, 0.06, 0.06), rot=(0, 0, R(-10)), mat='metal_dark')
    return p


def i_rubber(a):
    p = a.part('M')
    for k in range(3):
        p.torus(0.05 - k * 0.002, 0.009, loc=(0, 0, 0.01 + k * 0.017), mat='rubber', seg=24, rseg=8)
    p.beam((0.05, 0, 0.04), (0.09, -0.03, 0.02), 0.009, mat='rubber')
    return p


def i_fabric(a):
    p = a.part('M')
    p.cyl(0.035, 0.14, rot=(0, R(90), 0), loc=(0, 0, 0.035), mat='canvas', seg=18)
    p.box((0.14, 0.001, 0.07), loc=(0, -0.035, 0.0), rot=(R(-20), 0, 0), mat='canvas')
    p.cyl(0.037, 0.01, rot=(0, R(90), 0), loc=(-0.04, 0, 0.035), mat='strap', seg=18)
    p.cyl(0.037, 0.01, rot=(0, R(90), 0), loc=(0.04, 0, 0.035), mat='strap', seg=18)
    return p


def i_plastic(a):
    p = a.part('M')
    p.box((0.09, 0.05, 0.12), loc=(0, 0, 0.06), mat='plastic_blue', bevel=0.012)
    p.cyl(0.013, 0.02, loc=(0.025, 0, 0.13), mat='plastic_yellow', seg=12)
    p.beam((-0.035, 0, 0.12), (-0.005, 0, 0.13), 0.007, mat='plastic_blue')
    return p


def i_chem(a):
    p = a.part('M')
    lathe(p, [(0, 0), (0.04, 0), (0.042, 0.01), (0.042, 0.1), (0.02, 0.13), (0.014, 0.14), (0.014, 0.16), (0, 0.16)], mat='plastic_grey', seg=16)
    p.cyl(0.0425, 0.05, loc=(0, 0, 0.06), mat='chem_orange', seg=16, caps=False)
    p.cyl(0.016, 0.02, loc=(0, 0, 0.165), mat='black', seg=12)
    return p


def i_electronics(a):
    p = a.part('M')
    p.box((0.14, 0.1, 0.006), loc=(0, 0, 0.003), mat='pcb')
    for (x, y, w, d) in [(-0.03, 0.01, 0.04, 0.04), (0.035, -0.02, 0.03, 0.02), (0.035, 0.025, 0.02, 0.02), (-0.045, -0.03, 0.03, 0.015)]:
        p.box((w, d, 0.008), loc=(x, y, 0.01), mat='black')
    for k in range(6):
        p.cyl(0.004, 0.012, loc=(0.0 + k * 0.01, 0.042, 0.012), mat='gun_brass', seg=6)
    p.cyl(0.008, 0.02, loc=(-0.06, 0.035, 0.016), mat='plastic_blue', seg=10)
    return p


def i_battery(a):
    p = a.part('M')
    p.box((0.14, 0.08, 0.09), loc=(0, 0, 0.045), mat='black', bevel=0.006)
    p.box((0.141, 0.081, 0.02), loc=(0, 0, 0.08), mat='plastic_grey')
    p.cyl(0.01, 0.015, loc=(-0.045, 0, 0.098), mat='med_red', seg=10)
    p.cyl(0.01, 0.015, loc=(0.045, 0, 0.098), mat='galv', seg=10)
    p.box((0.06, 0.001, 0.03), loc=(0, -0.041, 0.045), mat='paint_yellow')
    return p


def _gear(p, r, t, loc, teeth=10, mat='galv'):
    p.cyl(r, t, loc=loc, mat=mat, seg=24)
    for k in range(teeth):
        ang = k * 2 * math.pi / teeth
        p.box((r * 0.3, r * 0.25, t), loc=(loc[0] + math.cos(ang) * r, loc[1] + math.sin(ang) * r, loc[2]), rot=(0, 0, ang), mat=mat)
    p.cyl(r * 0.3, t * 1.2, loc=loc, mat='metal_dark', seg=12)


def i_mech(a):
    p = a.part('M')
    _gear(p, 0.05, 0.015, (0, 0, 0.008))
    _gear(p, 0.032, 0.015, (0.07, 0.03, 0.02), teeth=8, mat='gun_brass')
    p.cyl(0.02, 0.03, loc=(-0.02, -0.06, 0.015), mat='metal_grey', seg=14)
    return p


def i_adv_elec(a):
    p = a.part('M')
    p.box((0.12, 0.08, 0.02), loc=(0, 0, 0.01), mat='metal_dark', bevel=0.004)
    p.box((0.1, 0.06, 0.006), loc=(0, 0, 0.022), mat='pcb')
    p.box((0.04, 0.04, 0.01), loc=(0, 0, 0.028), mat='silver', bevel=0.002)
    p.box((0.1, 0.006, 0.004), loc=(0, -0.035, 0.026), mat='cell_green')
    for k in range(8):
        p.box((0.004, 0.012, 0.004), loc=(-0.035 + k * 0.01, 0.045, 0.012), mat='gun_brass')
    return p


def i_sensors(a):
    p = a.part('M')
    p.box((0.08, 0.06, 0.06), loc=(0, 0, 0.03), mat='plastic_grey', bevel=0.006)
    p.cyl(0.025, 0.03, loc=(0, -0.04, 0.035), rot=(R(90), 0, 0), mat='black', seg=16)
    p.cyl(0.018, 0.004, loc=(0, -0.056, 0.035), rot=(R(90), 0, 0), mat='lens', seg=16)
    p.cyl(0.004, 0.04, loc=(0.03, 0.02, 0.08), mat='metal_dark', seg=6)
    return p


def i_vinyl(a):
    p = a.part('M')
    p.cyl(0.12, 0.004, loc=(0, 0, 0.002), mat='vinyl', seg=40)
    p.cyl(0.04, 0.005, loc=(0, 0, 0.002), mat='paint_orange', seg=24)
    p.box((0.2, 0.2, 0.006), loc=(0.05, 0.03, -0.003), rot=(0, 0, R(10)), mat='paint_teal')
    return p


def i_vase(a):
    p = a.part('M')
    lathe(p, [(0, 0), (0.035, 0), (0.05, 0.03), (0.058, 0.07), (0.05, 0.11), (0.028, 0.15), (0.026, 0.17), (0.034, 0.185), (0.03, 0.19)], mat='ceramic_white', seg=28)
    p.cyl(0.059, 0.025, loc=(0, 0, 0.07), mat='ceramic_blue', seg=28, caps=False)
    p.cyl(0.042, 0.012, loc=(0, 0, 0.12), mat='ceramic_blue', seg=28, caps=False)
    return p


def i_watch(a):
    p = a.part('M')
    p.cyl(0.05, 0.016, rot=(R(80), 0, 0), loc=(0, 0, 0.05), mat='gold', seg=28, bevel=0.003)
    p.cyl(0.043, 0.004, rot=(R(80), 0, 0), loc=(0, -0.008, 0.05), mat='paper', seg=28)
    p.box((0.003, 0.001, 0.03), loc=(0, -0.012, 0.058), rot=(0, R(30), 0), mat='black')
    p.box((0.003, 0.001, 0.02), loc=(0.006, -0.012, 0.05), rot=(0, R(-70), 0), mat='black')
    p.cyl(0.008, 0.012, loc=(0, 0, 0.105), mat='gold', seg=10)
    p.torus(0.012, 0.003, loc=(0, 0, 0.12), rot=(R(90), 0, 0), mat='gold', seg=14, rseg=5)
    return p


def i_camera(a):
    p = a.part('M')
    p.box((0.14, 0.06, 0.085), loc=(0, 0, 0.045), mat='black', bevel=0.008)
    p.box((0.141, 0.061, 0.03), loc=(0, 0, 0.075), mat='silver', bevel=0.004)
    p.cyl(0.03, 0.05, loc=(0, -0.05, 0.04), rot=(R(90), 0, 0), mat='black', seg=20)
    p.cyl(0.024, 0.004, loc=(0, -0.076, 0.04), rot=(R(90), 0, 0), mat='lens', seg=20)
    p.cyl(0.008, 0.01, loc=(0.045, 0, 0.095), mat='silver', seg=10)
    p.box((0.03, 0.02, 0.02), loc=(-0.04, 0, 0.1), mat='silver')
    return p


def i_goldring(a):
    p = a.part('M')
    p.torus(0.035, 0.007, loc=(0, 0, 0.035), rot=(R(90), 0, 0), mat='gold', seg=28, rseg=8)
    p.ico(0.012, loc=(0, 0, 0.075), mat='arc_red', sub=1)
    p.cyl(0.013, 0.008, loc=(0, 0, 0.066), mat='gold', seg=10)
    return p


def i_statuette(a):
    p = a.part('M')
    p.box((0.08, 0.08, 0.03), loc=(0, 0, 0.015), mat='wood_dark', bevel=0.004)
    lathe(p, [(0.03, 0.03), (0.026, 0.06), (0.018, 0.09), (0.03, 0.11), (0.04, 0.13), (0.03, 0.15), (0, 0.155)], mat='bronze', seg=18)
    p.sphere(0.022, loc=(0, 0, 0.175), mat='bronze', seg=14, rings=10, scale=(1, 1, 1.15))
    p.box((0.014, 0.012, 0.05), loc=(0.036, 0, 0.12), rot=(0, R(-30), 0), mat='bronze')
    return p


def i_datacore(a):
    p = a.part('M')
    p.cyl(0.03, 0.12, loc=(0, 0, 0.07), mat='cell_green', seg=16)
    for z in (0.015, 0.125):
        p.cyl(0.042, 0.025, loc=(0, 0, z), mat='metal_dark', seg=6, bevel=0.003)
    for k in range(3):
        ang = k * 2 * math.pi / 3
        p.box((0.01, 0.012, 0.11), loc=(math.cos(ang) * 0.036, math.sin(ang) * 0.036, 0.07), mat='silver')
    p.box((0.02, 0.012, 0.03), loc=(0, 0.03, 0.15), mat='paint_orange')
    return p


def i_arc_alloy(a):
    p = a.part('M')
    p.sphere(0.1, rot=(R(10), 0, 0), mat='arc_shell', seg=16, rings=10, scale=(1, 0.8, 0.35), hemi=True)
    p.box((0.12, 0.02, 0.006), loc=(0, -0.02, 0.034), mat='arc_orange')
    p.box((0.16, 0.12, 0.01), loc=(0, 0, 0.0), mat='arc_frame')
    return p


def i_arc_circuit(a):
    p = a.part('M')
    p.box((0.13, 0.1, 0.012), loc=(0, 0, 0.006), mat='arc_joint', bevel=0.003)
    for k in range(5):
        p.box((0.1, 0.004, 0.003), loc=(0, -0.04 + k * 0.02, 0.013), rot=(0, 0, R(k * 7)), mat='arc_weak')
    p.box((0.035, 0.035, 0.01), loc=(0.02, 0.01, 0.017), mat='arc_frame')
    return p


def i_tick_pod(a):
    p = a.part('M')
    p.sphere(0.05, loc=(0, 0, 0.045), mat='arc_frame', seg=14, rings=10, scale=(1, 1.3, 0.8))
    p.sphere(0.045, loc=(0, 0, 0.055), mat='arc_shell', seg=14, rings=8, scale=(1, 1.25, 0.6), hemi=True)
    p.sphere(0.012, loc=(0, -0.06, 0.05), mat='arc_eye', seg=10, rings=6)
    return p


def i_wasp_driver(a):
    p = a.part('M')
    p.cyl(0.05, 0.04, loc=(0, 0, 0.02), r2=0.042, mat='arc_panel', seg=18, bevel=0.003)
    p.cyl(0.036, 0.008, loc=(0, 0, -0.001), mat='arc_thrust', seg=18)
    for k in range(3):
        ang = k * 2 * math.pi / 3
        p.box((0.09, 0.012, 0.004), loc=(math.cos(ang) * 0.045, math.sin(ang) * 0.045, 0.045), rot=(0, 0, ang), mat='arc_joint')
    p.box((0.08, 0.02, 0.015), loc=(0.07, 0, 0.02), mat='arc_frame')
    return p


def i_pop_trigger(a):
    p = a.part('M')
    p.cyl(0.035, 0.05, loc=(0, 0, 0.025), mat='arc_orange', seg=16, bevel=0.003)
    p.sphere(0.02, loc=(0, 0, 0.055), mat='arc_eye', seg=12, rings=8)
    p.beam((0.03, 0, 0.02), (0.08, 0.02, 0.01), 0.004, mat='arc_cable')
    p.beam((-0.03, 0, 0.02), (-0.07, -0.03, 0.01), 0.004, mat='arc_red')
    return p


def i_hornet_driver(a):
    p = a.part('M')
    p.cyl(0.06, 0.06, loc=(0, 0, 0.03), r2=0.05, mat='arc_panel', seg=20, bevel=0.004)
    p.cyl(0.045, 0.01, loc=(0, 0, 0.0), mat='arc_thrust', seg=20)
    p.box((0.12, 0.03, 0.03), loc=(0.07, 0, 0.04), mat='arc_frame', bevel=0.004)
    p.box((0.05, 0.035, 0.012), loc=(0, 0, 0.066), mat='arc_orange')
    return p


def i_snitch_scanner(a):
    p = a.part('M')
    p.sphere(0.05, loc=(0, 0, 0.03), mat='arc_shell', seg=16, rings=10, hemi=True)
    p.cyl(0.052, 0.02, loc=(0, 0, 0.02), mat='arc_frame', seg=16)
    p.sphere(0.018, loc=(0, -0.045, 0.045), mat='arc_eye', seg=12, rings=8)
    p.cyl(0.004, 0.06, loc=(0.02, 0, 0.1), mat='arc_joint', seg=6)
    return p


def i_turret_optic(a):
    p = a.part('M')
    p.cyl(0.035, 0.1, loc=(0, 0, 0.035), rot=(R(90), 0, 0), mat='arc_frame', seg=18, bevel=0.003)
    p.cyl(0.042, 0.02, loc=(0, -0.05, 0.035), rot=(R(90), 0, 0), mat='arc_shell', seg=18)
    p.cyl(0.028, 0.005, loc=(0, -0.061, 0.035), rot=(R(90), 0, 0), mat='arc_eye', seg=18)
    return p


def i_rocket_thruster(a):
    p = a.part('M')
    lathe(p, [(0.02, 0.0), (0.05, 0.01), (0.06, 0.05), (0.045, 0.09), (0.03, 0.1), (0.0, 0.1)], mat='arc_panel', seg=20)
    p.cyl(0.042, 0.006, loc=(0, 0, 0.004), mat='arc_thrust', seg=18)
    p.cyl(0.062, 0.015, loc=(0, 0, 0.05), mat='arc_orange', seg=20)
    return p


def i_leaper_pulse(a):
    p = a.part('M')
    p.cyl(0.035, 0.14, rot=(0, R(90), 0), loc=(0, 0, 0.035), mat='arc_frame', seg=16)
    p.cyl(0.03, 0.08, rot=(0, R(90), 0), loc=(0, 0, 0.035), mat='arc_weak', seg=16)
    for x in (-0.05, 0.05):
        p.cyl(0.04, 0.02, rot=(0, R(90), 0), loc=(x, 0, 0.035), mat='arc_shell', seg=16)
    return p


def i_arc_core(a):
    p = a.part('M')
    p.sphere(0.045, loc=(0, 0, 0.06), mat='arc_weak', seg=16, rings=10)
    for k in range(4):
        ang = k * math.pi / 2
        p.beam((math.cos(ang) * 0.055, math.sin(ang) * 0.055, 0.0), (math.cos(ang) * 0.055, math.sin(ang) * 0.055, 0.12), 0.007, mat='arc_frame')
    p.cyl(0.07, 0.015, loc=(0, 0, 0.005), mat='arc_joint', seg=18)
    p.cyl(0.07, 0.015, loc=(0, 0, 0.12), mat='arc_joint', seg=18)
    return p


def i_bastion_cell(a):
    p = a.part('M')
    p.box((0.1, 0.1, 0.16), loc=(0, 0, 0.08), mat='arc_frame', bevel=0.01)
    p.box((0.07, 0.102, 0.12), loc=(0, 0, 0.08), mat='arc_weak')
    p.box((0.102, 0.07, 0.12), loc=(0, 0, 0.08), mat='arc_weak')
    p.box((0.11, 0.11, 0.025), loc=(0, 0, 0.165), mat='arc_shell', bevel=0.004)
    p.box((0.11, 0.11, 0.025), loc=(0, 0, 0.0), mat='arc_shell', bevel=0.004)
    return p


ICON_MODELS = {name[2:]: fn for name, fn in globals().items() if name.startswith('i_')}
