"""Welt 3 – Dragon Ball: Planet Namek.

Türkiser Himmel mit drei Sonnen, grünliches Meer, Felsnadeln, Tafelberg mit Ajisa-Bäumen,
Namekianer-Kuppelhäusern, den sieben Dragon Balls und Friezas Raumschiff.

Flug (20 s, konstant 18 m/s, eine durchgehende Aufnahme):
  0–3 s      tief über dem Meer zwischen Felsnadeln hindurch; über dem Tafelberg blitzen Zusammenstöße
  3–8,5 s    Steigflug an der Steilwand hinauf, Goku und Freezer kämpfen hoch über der Kante
  ~8,7 s     über der Kante: Dorf, Dragon Balls, Raumschiff; der Kampf ist jetzt direkt vor der Kamera
  8,5–11,5 s Schlagabtausch 18–24 m vor der Kamera (Schockwellen bei jedem Treffer)
  11,5–13 s  Freezer wird weggeschleudert, feuert Todesstrahlen ins Plateau (Explosionen, Staub, Brocken),
             Goku weicht aus, Freezer lenkt zwei Ki-Kugeln aufs Meer ab
  13–15,5 s  Goku schwebt über der Flugbahn und feuert das Kamehameha; die Kamera fliegt unter ihm durch
             und am Strahl entlang, Strahlenduell mit Freezers Todesstrahl, bei 15,5 s bricht das
             Kamehameha durch -> große Explosion über der Nordkante
  15,5–20 s  weiter über die Kante hinaus aufs Meer, die Explosion verglüht
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import bmesh  # noqa: E402,I100
import numpy as np  # noqa: E402

import dbz  # noqa: E402
import fpv  # noqa: E402
import namek  # noqa: E402
import nature  # noqa: E402
from world_onepiece import foam_builder, shallow_tint  # noqa: E402
import ocean  # noqa: E402
import vfx  # noqa: E402
from figures import POSES  # noqa: E402
from mathutils import Vector  # noqa: E402

FPS = 24
SECONDS = 20
SPEED = 18.0
SUN_ELEV, SUN_AZIM = 24.0, 300.0          # Hauptsonne (links vorn)
SUNS2 = [(38.0, 40.0, 0.35), (9.0, 22.0, 0.25)]  # zwei weitere Sonnen (elev, azim, Stärke rel.)

MESA_C, MESA_R, MESA_H = (0.0, 250.0), 78.0, 42.0
FAR_TREES = []
DB_C = (1.0, 232.0)
SHIP_C = (40.0, 268.0)

ROUTE = [
    (-2, 36, 3.4), (2, 95, 4.2), (1, 128, 12.0), (0, 152, 30.0), (0, 170, 47.0),
    (2, 190, 49.0), (4, 220, 47.5), (7, 252, 47.5), (7, 288, 48.0), (4, 312, 44.0), (1, 336, 43.6),
    (-2, 372, 43.0), (-5, 410, 43.0), (-8, 450, 43.0),
]
PLATEAU = 39.0                  # mittlere Plateauhöhe (Grund unter der Flugbahn)
GOLD, KAME, DEATH, KI = (1.0, 0.62, 0.10), (0.10, 0.36, 1.0), (0.85, 0.25, 1.0), (1.0, 0.80, 0.28)

SPIRES = [
    # (x, y, r, h, seed)
    (-15, 52, 7.0, 46, 1), (16, 72, 8.0, 58, 2), (-48, 20, 6.0, 30, 3), (42, 18, 5.0, 24, 4),
    (-42, 110, 10.0, 52, 5), (55, 128, 9.0, 40, 6), (-95, 170, 14.0, 70, 7), (110, 200, 12.0, 60, 8),
    (-150, 60, 16.0, 55, 9), (140, 40, 10.0, 48, 10), (25, 120, 4.0, 14, 11), (-22, 150, 5.0, 20, 12),
]


def namek_sky(nb, d):
    x, y, z = nb.sep(d)
    zz = nb.math("MAXIMUM", z, 0.0)
    t = nb.math("POWER", zz, 0.55)
    # Referenz (Anime/Manga): gesättigtes Grasgrün, am Horizont hell gelbgrün
    col = nb.ramp(t, [(0.0, (0.80, 0.88, 0.42)), (0.07, (0.50, 0.74, 0.20)), (0.30, (0.12, 0.47, 0.07)),
                      (1.0, (0.035, 0.30, 0.02))])
    below = nb.math("LESS_THAN", z, 0.0)
    col = nb.mix(below, col, (0.55, 0.72, 0.35))
    # Leuchten um die Hauptsonne
    sd = fpv.sun_dir(SUN_ELEV, SUN_AZIM)
    dp = nb.math("MAXIMUM", nb.vmath("DOT_PRODUCT", d, tuple(sd)), 0.0)
    glow = nb.math("ADD", nb.math("MULTIPLY", nb.math("POWER", dp, 8.0), 0.5),
                   nb.math("MULTIPLY", nb.math("POWER", dp, 64.0), 1.5))
    col = nb.vmath("ADD", col, nb.vmath("SCALE", (1.0, 0.97, 0.85), scale=glow))
    # Kamera- und Spiegelstrahlen sehen den kräftigen Himmel; diffuses Licht kommt entsättigt an,
    # damit blaues Gras und beiger Fels ihre Farbe behalten (sonst färbt der grüne Himmel alles türkis)
    lp = nb.node("ShaderNodeLightPath")
    vivid = nb.math("MAXIMUM", lp.outputs["Is Camera Ray"], lp.outputs["Is Glossy Ray"])
    lum = nb.vmath("DOT_PRODUCT", col, (0.2126, 0.7152, 0.0722))
    neutral = nb.mix(0.6, col, nb.comb(lum, lum, lum))
    return nb.mix(vivid, neutral, col)


def namek_rock():
    return nature.rock_material("NamekRock", c1=(0.16, 0.10, 0.12), c2=(0.62, 0.40, 0.27), c3=(0.46, 0.30, 0.24),
                                wet_line=1.6, algae=(0.03, 0.06, 0.04), moss=(0.02, 0.11, 0.34), moss_amount=0.3,
                                strata_scale=2.2, bump=0.9, crack_w=0.15, scale=1.5, lichen=0.1,
                                moss_tex=os.path.join(fpv.ASSETS, "grasslight-big.jpg"), moss_tex_scale=4.0)


def mesa_height(X, Y):
    r = np.sqrt((X - MESA_C[0]) ** 2 + (Y - MESA_C[1]) ** 2)
    ang = np.arctan2(Y - MESA_C[1], X - MESA_C[0])
    ca, sa = np.cos(ang), np.sin(ang)
    edge = MESA_R * (1 + 0.08 * np.sin(ang * 3 + 1) + 0.05 * np.sin(ang * 7 + 2)
                     + 0.06 * fpv.fbm2(ca * 3, sa * 3, 3, seed=21)
                     + 0.025 * fpv.fbm2(ca * 25, sa * 25, 3, seed=24))  # Rinnen
    s = edge - r
    W = 16.0
    u = np.clip(s / W, 0, 1)
    # Felsstufen (Schichtbänke)
    k = 5.0
    fu = u * k
    step = np.floor(fu)
    fr = fu - step
    fr = np.clip((fr - 0.55) / 0.45, 0, 1)
    fr = fr * fr * (3 - 2 * fr)
    ut = (step + fr) / k
    ut = np.where(u >= 1.0, 1.0, ut)
    top = MESA_H + 2.0 * fpv.fbm2(X / 40, Y / 40, 4, seed=22) + 0.4 * fpv.fbm2(X / 6, Y / 6, 3, seed=23)
    foot = np.clip((edge + 18 - r) / 18.0, 0, 1) ** 2 * 7
    rough = 0.8 * fpv.fbm2(X / 3, Y / 3, 3, seed=25) * (u > 0) * (u < 1)
    h = foot + ut * (top - foot) + rough - 3
    return h


def namek_fight(pos, top_z, rock, houses):
    """Goku (SSJ) gegen Freezer, getaktet auf den Kameraflug (pos[i] = Kameraposition in Frame i).
    Alle Nahkampfpositionen werden relativ zur Kamera bestimmt (D m voraus, X m rechts, Z m über dem Plateau),
    damit der Kampf im Bild bleibt, obwohl die Drohne mit 18 m/s geradeaus fliegt."""
    n = len(pos)

    def F(t):
        return int(round(t * FPS)) + 1

    def cam(t):
        return Vector(pos[min(max(F(t), 0), n - 1)])

    def fwd(t):
        i = F(t)
        a, b = Vector(pos[max(i - 6, 0)]), Vector(pos[min(i + 6, n - 1)])
        return Vector((b.x - a.x, b.y - a.y, 0)).normalized()

    def rel(t, D, X, Z, above_cam=False):
        h = fwd(t)
        p = cam(t) + h * D + Vector((h.y, -h.x, 0)) * X
        p.z = (cam(t).z if above_cam else PLATEAU) + Z
        return p

    G = dbz.goku((0, 0, 0), 0.0)
    Z = dbz.freezer((0, 0, 0), 0.0)
    vfx.aura("GokuAura", G.base, GOLD, 1, n + 2, height=1.95, width=0.95, light_w=900.0)
    chest = {G: 1.15 * G.s, Z: 1.15 * Z.s}
    last_yaw = {}

    def key(fig, t, pose, p, look, lean=0.0, extra=None, roll=0.0):
        d = look - p
        yaw = math.degrees(math.atan2(-d.x, d.y))
        if fig in last_yaw:                      # Gier stetig halten (kein 340°-Dreher beim Interpolieren)
            while yaw - last_yaw[fig] > 180:
                yaw -= 360
            while yaw - last_yaw[fig] < -180:
                yaw += 360
        last_yaw[fig] = yaw
        rot = dict(POSES[pose])
        if extra:
            rot.update(extra)
        fig.pose(F(t), rot, loc=(p.x, p.y, p.z - chest[fig]), yaw=yaw, base_rot=(lean, roll, 0))

    # ---- Schlagabtausch: (Zeit, Treffpunkt, Achse Goku->Freezer, Goku-Pose, Freezer-Pose, Stärke)
    teaser = [(1.9, (-6, 222, 74), (1, 0.2, 0.1)), (2.7, (8, 214, 80), (-1, 0.3, -0.2)),
              (3.5, (-3, 230, 70), (1, -0.4, 0.3)), (4.3, (10, 220, 84), (-1, -0.2, 0.1)),
              (5.0, (-8, 212, 76), (1, 0.5, -0.1)), (5.8, (4, 206, 70), (-1, 0.1, 0.3)),
              (6.6, (-4, 200, 66), (1, -0.3, 0.0)), (7.45, (3, 196, 60), (-1, 0.2, 0.2))]
    # Nahkampf 8,5–11 m vor der Kamera, 1,8–2,8 m über Kamerahöhe (Blick ~13° nach oben, oberes Bilddrittel)
    melee = [(8.3, 11, -2.0, 3.2, (1, 0.3, -0.1)), (8.8, 9.5, 2.3, 2.6, (-1, 0.2, 0.2)),
             (9.3, 9, -2.6, 2.0, (1, -0.2, 0.1)), (9.75, 8.5, 1.8, 2.4, (-1, -0.3, -0.2)),
             (10.2, 8.5, -1.0, 1.8, (1, 0.35, 0.25)), (10.65, 9, 2.6, 2.4, (-1, 0.1, -0.1)),
             (11.1, 8.5, -2.0, 2.0, (1, -0.25, 0.0)), (11.5, 9.5, 0.5, 2.6, (-0.3, 1.0, 0.25))]
    clashes = [(t, Vector(c), Vector(a).normalized(), 3.0, 60000.0) for (t, c, a) in teaser]
    clashes += [(t, rel(t, D, X, Zh, True), Vector(a).normalized(), 0.6, 4000.0) for (t, D, X, Zh, a) in melee]
    g_moves = ["punch_R", "kick_R", "punch_L", "kick_L"]
    z_moves = ["guard", "punch_L", "kick_R", "guard"]
    prev = None
    for k, (t, C, ax, r, lw) in enumerate(clashes):
        # Achse im Kamerabezug drehen (Nahkampf: seitlich im Profil sichtbar)
        if t > 8.0:
            h = fwd(t)
            rt = Vector((h.y, -h.x, 0))
            ax = (rt * ax.x + h * ax.y + Vector((0, 0, ax.z))).normalized()
        sep = 1.6 if t > 8 else 3.0
        gA, zA = C - ax * sep, C + ax * sep
        if prev:
            # Bogen zwischen zwei Treffern: bei Seitenwechsel fliegt Goku über, Freezer unter dem anderen durch
            tp, gP, zP, axp = prev
            tm = 0.5 * (tp + t - 0.12)
            up = Vector((0, 0, 2.2 if axp.dot(ax) < 0 else 0.9))
            key(G, tm, "fly", (gP + gA) / 2 + up, (zP + zA) / 2, lean=-60)
            key(Z, tm, "fly", (zP + zA) / 2 - up, (gP + gA) / 2, lean=-40)
        key(G, t - 0.12, "fly", gA, C + ax, lean=-45)
        key(Z, t - 0.12, "fly", zA, C - ax, lean=-30)
        key(G, t, g_moves[k % 4], C - ax * 0.55, C + ax)
        key(Z, t, z_moves[k % 4], C + ax * 0.5, C - ax)
        gR, zR = C - ax * 2.3, C + ax * 2.6
        key(G, t + 0.14, "recoil", gR, C + ax, lean=10)
        key(Z, t + 0.14, "recoil", zR, C - ax, lean=15)
        prev = (t + 0.14, gR, zR, ax)
        b = vfx.burst(f"Hit{k}", C, F(t), (1.0, 0.78, 0.35), r_max=r, dur=10 if t > 8 else 12, light_w=lw,
                      bolts=False, seed=20 + k, core_s=8.0, glow_s=2.0, glow_alpha=0.25, ring_s=2.5,
                      core_color=(1.0, 0.95, 0.8))
        b.rotation_mode = "QUATERNION"
        b.rotation_quaternion = ax.to_track_quat("Z", "Y")

    # ---- 11,5 s: Gokus Schlag schleudert Freezer davon (Salto rückwärts), er fängt sich hoch voraus
    Zp = rel(12.05, 34, 7, 18)
    key(Z, 11.8, "recoil", rel(11.8, 24, 4, 13), cam(11.8), lean=120)
    key(Z, 12.05, "point_R", Zp, cam(12.05) + fwd(12.05) * 24, lean=0)
    # ---- Todesstrahlen ins Plateau, Goku weicht aus und kontert mit einer Ki-Kugel
    for (t, pose, X) in ((11.75, "fly", -1.0), (12.05, "guard", -4.0), (12.33, "guard", 3.0), (12.6, "point_R", 1.0),
                         (12.8, "guard", -1.0)):
        key(G, t, pose, rel(t, 10, X, 2.6, True), Zp, lean=-20 if pose == "fly" else 0)
    shots = [(12.12, -7.0, 24), (12.36, 8.0, 26), (12.58, -9.0, 25)]
    for k, (t, X, D) in enumerate(shots):
        tgt = rel(t + 0.05, D, X, 0)
        if any(math.hypot(tgt.x - hx, tgt.y - hy) < hr + 4 for (hx, hy, hr) in houses):
            X = -X
            tgt = rel(t + 0.05, D, X, 0)
        tgt.z = top_z(tgt.x, tgt.y) + 0.2
        d = tgt - Zp
        aim = math.degrees(math.atan2(d.z, Vector((d.x, d.y)).length))
        key(Z, t - 0.06, "point_R", Zp, tgt, extra={"shoulder.R": (90 + aim, -5, 0)})
        key(Z, t + 0.1, "point_R", Zp, tgt, extra={"shoulder.R": (90 + aim + 12, -5, 0)})
        bpy.context.scene.frame_set(F(t))
        o = Z.J["wrist.R"].matrix_world.translation + d.normalized() * 0.18
        vfx.energy_ball(f"DeathTip{k}", Z.J["wrist.R"], (0, 0, -0.17), DEATH, 0.09, F(t) - 5, F(t) - 1, F(t) + 5,
                        light_w=120.0, swirl=False, spin=False)
        vfx.beam(f"DeathBeam{k}", o, tgt - o, (tgt - o).length, 0.2, DEATH, F(t), F(t) + 2, F(t) + 7,
                 light_w=4000.0, core_s=5.0, whiten=0.3, glow_s=3.0)
        vfx.burst(f"Impact{k}", tgt + Vector((0, 0, 0.8)), F(t) + 2, (1.0, 0.55, 0.85), r_max=4.5, dur=18,
                  light_w=45000.0, bolts=False, seed=40 + k, ring_dz=-0.7, core_s=12.0, glow_s=4.0, ring_s=3.0,
                  core_color=(1.0, 0.9, 0.95))
        vfx.dust_cloud(f"Dust{k}", tgt, F(t) + 3, 5.0, color=(0.50, 0.45, 0.36), dur=60, seed=60 + k)
        vfx.debris(f"Debris{k}_", tgt + Vector((0, 0, 0.3)), F(t) + 2, rock, n=14, speed=13.0, size=0.28,
                   seed=80 + k, ground=tgt.z - 0.1)
    # Gokus Ki-Kugel: Freezer schlägt sie weg, sie schlägt in eine ferne Felsnadel ein
    far = Vector((110, 498, 52))
    bpy.context.scene.frame_set(F(12.62))
    o = G.J["wrist.R"].matrix_world.translation.copy()
    hitp = Zp + Vector((-0.5, -0.5, 0.3))
    vfx.ki_blast("Ki", o, hitp, F(12.62), F(12.84), KI, radius=0.34, impact=False)
    key(Z, 12.78, "guard", Zp, o)
    key(Z, 12.86, "punch_L", Zp, far)
    vfx.burst("KiSwat", hitp, F(12.84), KI, r_max=1.6, dur=10, light_w=8000.0, ring=False, bolts=False, seed=91)
    vfx.ki_blast("KiDeflect", hitp, far, F(12.84), F(13.66), KI, radius=0.34, r_impact=11.0, seed=90)

    # ---- Kamehameha gegen Todesstrahl
    ZK = Vector((3.0, 358.0, 72.0))
    GK = cam(14.35) + Vector((-3.2, 0, 0))
    GK.z = PLATEAU + 9.5
    key(Z, 13.25, "fly", Zp.lerp(ZK, 0.6) + Vector((0, 0, 4)), ZK, lean=-60)
    key(Z, 13.6, "point_R", ZK, GK)
    key(G, 13.2, "kame_charge", GK + Vector((0.3, -0.8, 0.2)), ZK)
    key(G, 13.3, "kame_charge", GK, ZK)
    key(G, 13.85, "kame_charge", GK + Vector((0, 0.05, -0.05)), ZK)
    d = ZK - GK
    aim = math.degrees(math.atan2(d.z, Vector((d.x, d.y)).length))
    fire = {"shoulder.R": (88 + aim, 0, 12), "shoulder.L": (88 + aim, 0, -12)}
    key(G, 13.95, "kame_fire", GK, ZK, extra=fire)
    key(G, 15.6, "kame_fire", GK + Vector((0, -0.3, 0)), ZK, extra=fire)
    key(G, 16.3, "guard", GK + Vector((0, 0.5, 0.4)), ZK)
    aimz = math.degrees(math.atan2(-d.z, Vector((d.x, d.y)).length))
    key(Z, 13.9, "point_R", ZK, GK, extra={"shoulder.R": (90 + aimz, -5, 0)})
    key(Z, 15.4, "point_R", ZK + Vector((0, 0.8, 0)), GK, extra={"shoulder.R": (90 + aimz, -5, 0)})
    vfx.energy_ball("KameCharge", G.J["wrist.R"], (-0.07, 0.06, -0.13), KAME, 0.34, F(13.2), F(13.85), F(14.0),
                    light_w=900.0)
    sc = bpy.context.scene
    sc.frame_set(F(13.95))
    o = (G.J["wrist.R"].matrix_world.translation + G.J["wrist.L"].matrix_world.translation) / 2
    sc.frame_set(F(13.9))
    oz = Z.J["wrist.R"].matrix_world.translation.copy()
    L = (oz - o).length
    dk = (oz - o).normalized()
    o = o + dk * 0.2
    Lc = 0.52 * L
    wob = [(-0.0, 0.01), (0.25, Lc), (0.55, Lc - 2.5), (0.85, Lc + 1.5), (1.1, Lc - 1.0), (1.5, L)]
    kl = [(F(13.95 + dt), v) for dt, v in wob]
    zl = [(F(13.95), 0.01)] + [(F(13.95 + dt), L - v) for dt, v in wob[1:-1]] + [(F(15.45), 0.3)]
    vfx.beam("Kamehameha", o, dk, kl, 0.95, KAME, F(13.95), F(14.2), F(16.0), light_w=26000.0,
             wobble=(F(14.2), F(15.9), 0.12), core_s=5.0, whiten=0.35, glow_s=3.0, core_r=0.3)
    vfx.beam("DeathBeamDuel", oz, -dk, zl, 0.55, DEATH, F(13.95), F(14.2), F(15.5), light_w=12000.0,
             wobble=(F(14.2), F(15.4), 0.15), core_s=5.0, whiten=0.35, glow_s=2.6)
    # Treffpunkt der Strahlen: knisternde Energiekugel, die hin und her drückt und dann zu Freezer rast
    mid = bpy.data.objects.new("DuelPoint", None)
    fpv.link(mid)
    for f, v in kl[1:]:
        mid.location = o + dk * v
        mid.keyframe_insert("location", frame=f)
    vfx.energy_ball("DuelBall", mid, (0, 0, 0), (0.62, 0.55, 1.0), 2.0, F(14.18), F(14.3), F(15.45), light_w=30000.0)
    vfx.lightning("DuelArcs", mid, (0, 0, 0), (0.75, 0.7, 1.0), F(14.2), F(15.45), radius=4.5, n_bolts=10,
                  variants=6, seed=12, light_w=0.0, thickness=0.06)
    b = vfx.burst("DuelMeet", o + dk * Lc, F(14.2), (0.6, 0.55, 1.0), r_max=6.0, dur=18, light_w=90000.0, bolts=False,
                  seed=13, core_s=12.0, glow_s=3.0, ring_s=3.5)
    b.rotation_mode = "QUATERNION"
    b.rotation_quaternion = dk.to_track_quat("Z", "Y")
    # Durchbruch: riesige Explosion um Freezer, Druckwelle, Rauch
    vfx.burst("FinalBlast", ZK, F(15.45), (1.0, 0.62, 0.22), r_max=17.0, dur=60, light_w=450000.0, seed=14,
              core_s=12.0, glow_s=3.5, glow_alpha=0.45, ring_s=4.0, core_color=(1.0, 0.92, 0.7))
    vfx.dust_cloud("FinalSmoke", ZK + Vector((0, 0, -9)), F(16.0), 13.0, color=(0.30, 0.28, 0.27), dur=90, seed=15,
                   puffs=16, rise=0.5)
    for fig_f, s in ((F(15.45), 1.0), (F(15.5), 0.001)):
        Z.base.scale = (s, s, s)
        Z.base.keyframe_insert("scale", frame=fig_f)
    return G, Z


def build(args):
    sc = fpv.reset()
    frames = FPS * SECONDS
    fpv.setup_render(args.out, res=args.res, fps=FPS, seconds=SECONDS, samples=args.samples,
                     motion_blur=not args.no_mblur, mist_depth=6000.0)
    extra = [(SUN_ELEV, SUN_AZIM, 0.9, 900.0, (1.0, 0.98, 0.9))]
    for (e, a, s) in SUNS2:
        extra.append((e, a, 0.6, 500.0 * s, (1.0, 0.97, 0.9)))
    fpv.build_world(sun_elev=SUN_ELEV, sun_azim=SUN_AZIM, sky_strength=0.85, clouds=True, cloud_cover=0.25,
                    cloud_ref=1.1, custom_sky=namek_sky, extra_suns=extra, cloud_color=(1.0, 1.0, 0.78),
                    cloud_scale=1.3)
    fpv.add_sun(SUN_ELEV, SUN_AZIM, strength=5.6, color=(1.0, 0.92, 0.76), name="Sun1", angle_deg=1.5)
    for i, (e, a, s) in enumerate(SUNS2):
        fpv.add_sun(e, a, strength=4.6 * s, color=((0.92, 0.97, 1.0), (0.93, 1.0, 0.9))[i], name=f"Sun{i + 2}",
                    angle_deg=(2.5, 3.0)[i])
    rng = np.random.default_rng(31)
    top_z = lambda x, y: float(mesa_height(np.array([x]), np.array([y]))[0])
    # Route über dem Plateau: 3,6 m über Grund (tiefer Vorbeiflug an den Dragon Balls);
    # über die Nordkante hinaus bleibt die Höhe erhalten (Blick aufs Meer)
    route = []
    for (x, y, z) in ROUTE:
        if 185 <= y <= 320 and top_z(x, y) > MESA_H - 8:
            z = top_z(x, y) + 3.6
        route.append((x, y, z))

    rock = namek_rock()
    # Tafelberg (Heightfield)
    mesa = fpv.grid_mesh("Mesa", 2 * MESA_R + 50, 2 * MESA_R + 50, 360, 360, mesa_height, rock, origin=MESA_C)
    # Felsnadeln (zylindrisch-vertikal, pilzförmiger Kopf, Schichtbänke)
    spires = []
    for (x, y, r, h, seed) in SPIRES:
        ob = fpv.rock_mesh(f"Spire{seed}", radius=r, height=h + 4, seed=seed + 50, detail=5, taper=-0.35, mat=rock,
                           base_z=-4.0, lumpy=0.18, strata=2.2)
        ob.location = (x, y, 0)
        ob.rotation_euler = (0, 0, seed * 2.1)
        spires.append(ob)
    # ferne Inseln: große Pilzfelsen + Tafelberge
    for k, (x, y, r, h) in enumerate(((-160, 470, 26, 95), (120, 520, 20, 70), (-60, 640, 34, 120),
                                      (260, 700, 30, 85), (-330, 820, 45, 140), (40, 980, 55, 150),
                                      (420, 1050, 50, 120), (-620, 1150, 70, 170))):
        ob = fpv.rock_mesh(f"FarSpire{k}", radius=r, height=h + 4, seed=200 + k, detail=4, taper=-0.3, mat=rock,
                           base_z=-4.0, lumpy=0.2, strata=2.0)
        ob.location = (x, y, 0)
        for j in range(int(r * 0.5)):
            a = rng.uniform(0, 2 * math.pi)
            rr = r * 1.1 * math.sqrt(rng.random())
            FAR_TREES.append((x + math.cos(a) * rr, y + math.sin(a) * rr, h - 0.6))
    for k, (x, y, r, h) in enumerate(((-900, 1400, 260, 150), (800, 1700, 300, 170))):
        def hf(X, Y, x=x, y=y, r=r, h=h, k=k):
            rr = np.sqrt((X - x) ** 2 + (Y - y) ** 2) / r
            e = 1 + 0.12 * fpv.fbm2(X / r * 2, Y / r * 2, 3, seed=40 + k)
            t = np.clip((e - rr) / 0.12, 0, 1)
            return t * t * (3 - 2 * t) * h * (1 + 0.05 * fpv.fbm2(X / 30, Y / 30, 3, seed=50 + k)) - 3
        fpv.grid_mesh(f"Island{k}", r * 2.6, r * 2.6, 180, 180, hf, rock, origin=(x, y))
    bpy.context.view_layer.update()

    # Meer: smaragdgrün, Flachwasser + Brandung an Felsen und Tafelberg (Abstandsfeld)
    shore_img, shore_map = ocean.shore_distance_image(spires + [mesa], -200, -100, 200, 420, cell=0.5,
                                                      name="NamekShore")
    wmat = ocean.water_material("NamekSea", deep=(0.012, 0.10, 0.035), shallow=(0.07, 0.30, 0.10),
                                wake_fn=foam_builder(shore_img, shore_map), foam_amount=0.4,
                                color_fn=shallow_tint(shore_img, shore_map, color=(0.05, 0.25, 0.09)))
    ocean.animate_time_value(wmat, FPS, frames)
    tile = 100.0
    x0, y0, nx, ny = -140.0, -40.0, 3, 5
    ocean.make_ocean(x0, y0, nx, ny, tile=tile, res=13, wind=6.5, wave_scale=0.7, chop=1.1, fps=FPS, frames=frames,
                     mat=wmat, direction_deg=60, alignment=0.3, foam_coverage=0.1)
    far = ocean.water_material("NamekFarSea", deep=(0.012, 0.10, 0.035), shallow=(0.07, 0.30, 0.10), far=True)
    ocean.far_plane(x0 - tile / 2 + 1, x0 - tile / 2 + tile * nx - 1, y0 - tile / 2 + 1, y0 - tile / 2 + tile * ny - 1,
                    mat=far, z=-0.05)

    # Materialien Plateau
    mats = {
        "clay": namek.clay_material("NamekClay"),
        "clay_dark": namek.clay_material("NamekClayDark", color=(0.62, 0.64, 0.58)),
        "glass": fpv.simple_mat("DomeGlass", (0.015, 0.03, 0.03), rough=0.06),
        "door": fpv.simple_mat("DomeDoor", (0.03, 0.05, 0.045), rough=0.4),
        "hull": namek.ship_hull_material("FriezaHull"),
        "hull_dark": namek.ship_hull_material("FriezaHullDark", color=(0.30, 0.29, 0.28)),
        "rim": namek.hazard_rim_material(),
        "shipwin": fpv.simple_mat("ShipWin", (0.4, 0.05, 0.05), rough=0.1, emission=(1.0, 0.3, 0.2), estrength=1.2),
    }

    # Namekianer-Häuser (glatte Lehm-Kuppeln, Rippen, gehörnte Oberkuppel, Rundfenster)
    houses = [(-20, 204, 6.5), (-36, 224, 8.0), (-22, 252, 5.5), (-46, 252, 6.0), (-28, 276, 7.5), (26, 204, 6.0),
              (30, 228, 5.0), (-16, 298, 6.0), (-46, 292, 5.0), (26, 308, 7.0)]
    for i, (x, y, r) in enumerate(houses):
        namek.house(f"House{i}", x, y, top_z(x, y), r, mats, rng)

    # Dragon Balls in realer Größe (Ø ~0,4 m) auf einem niedrigen, runden Steinsockel links der Flugbahn
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring_z = mesa_height(DB_C[0] + 1.7 * np.cos(ang), DB_C[1] + 1.7 * np.sin(ang))
    plinth_top = top_z(*DB_C) + 0.4
    base = float(ring_z.min()) - 0.3
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=1.7, radius2=1.5, depth=plinth_top - base)
    slab = fpv.mesh_from_bmesh(bm, "DBPlinth", rock)
    slab.location = (DB_C[0], DB_C[1], (plinth_top + base) / 2)
    namek.dragon_balls(DB_C, mesa_height, rng, radius=0.2, ring=0.8, lift=plinth_top)

    namek.spaceship(mats, SHIP_C[0], SHIP_C[1], top_z(*SHIP_C) + 1.5)

    # Ajisa-Bäume (verdrehte dünne Stämme, perfekte Kugelkronen)
    leaf = nature.leaf_material("AjisaLeaf", c1=(0.004, 0.045, 0.30), c2=(0.03, 0.12, 0.50), trans=0.3)
    bark = nature.bark_material("AjisaBark", c=(0.42, 0.33, 0.18))
    variants = [namek.ajisa_variant(f"Ajisa{i}", leaf, bark, height=h, crown_r=cr, leaves=int(900 * cr * cr / 4),
                                    seed=70 + i, leaf_size=0.36, turns=tw)
                for i, (h, cr, tw) in enumerate(((12.0, 2.8, 0.8), (15.0, 3.3, 1.0), (9.0, 2.3, 0.6), (17.0, 3.6, 1.1)))]
    _, subs = nature.make_tree_collection("AjisaTrees", variants)
    pos, _, _ = fpv.fpv_path(route, SPEED, FPS, frames, micro=0.0)
    path_xy = pos[:, :2]
    pts, scl = [], []
    avoid = [(x, y, r + 3) for (x, y, r) in houses] + [(DB_C[0], DB_C[1], 5), (SHIP_C[0], SHIP_C[1], 22)]
    groves = [(-56, 205, 9, 8), (50, 210, 8, 6), (-6, 270, 6, 4), (-52, 312, 10, 8), (38, 322, 9, 7),
              (62, 240, 7, 5), (-66, 264, 8, 6), (20, 186, 6, 4), (-28, 184, 6, 5), (-4, 334, 6, 4),
              (24, 250, 5, 3), (-16, 226, 4, 3), (26, 286, 6, 4)]
    cand = []
    for (gx, gy, gr, gn) in groves:
        for _ in range(gn * 3):
            a = rng.uniform(0, 2 * math.pi)
            rr = gr * math.sqrt(rng.random())
            cand.append((gx + math.cos(a) * rr, gy + math.sin(a) * rr))
    for (x, y) in cand:
        if any(math.hypot(x - ax, y - ay) < ar for (ax, ay, ar) in avoid):
            continue
        if np.min(np.hypot(path_xy[:, 0] - x, path_xy[:, 1] - y)) < 11:
            continue
        z = top_z(x, y)
        if z < MESA_H - 6:
            continue
        pts.append((x, y, z - 0.2))
        scl.append(rng.uniform(0.7, 1.25))
    for p in FAR_TREES:
        pts.append(p)
        scl.append(rng.uniform(1.2, 2.2))
    # Findlinge
    boulders = []
    for k in range(45):
        a = rng.uniform(0, 2 * math.pi)
        rr = MESA_R * 0.9 * math.sqrt(rng.random())
        x, y = MESA_C[0] + math.cos(a) * rr, MESA_C[1] + math.sin(a) * rr
        if np.min(np.hypot(path_xy[:, 0] - x, path_xy[:, 1] - y)) < 5 or any(
                math.hypot(x - ax, y - ay) < ar for (ax, ay, ar) in avoid):
            continue
        br = rng.uniform(0.6, 2.4)
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=4, radius=br)
        ob = fpv.mesh_from_bmesh(bm, f"Boulder{k}", rock)
        ob.scale = (rng.uniform(0.9, 1.4), rng.uniform(0.8, 1.2), rng.uniform(0.5, 0.8))
        ob.rotation_euler = (0, 0, rng.uniform(0, 6.3))
        fpv.displace_obj(ob, "CLOUDS", size=br * 0.6, strength=br * 0.35, depth=3, name=f"Boulder{k}_d")
        ob.location = (x, y, top_z(x, y) - br * 0.15)
        boulders.append((x, y, br * 1.2))
    for (x, y, r, h, seed) in SPIRES:
        for k in range(int(3 + r * 0.8)):
            a = rng.uniform(0, 2 * math.pi)
            rr = r * 1.1 * math.sqrt(rng.random())
            pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr, h - 0.6))
            scl.append(rng.uniform(0.45, 0.9))
    nature.scatter_instances("Ajisa", subs, pts, scales=scl, seed=4)
    print("ajisa", len(pts))

    # Sandflecken im blauen Gras und rote Pilzgruppen (Referenzbilder)
    prng = np.random.default_rng(58)
    patches = []
    while len(patches) < 26:
        a = prng.uniform(0, 2 * math.pi)
        rr = MESA_R * 0.72 * math.sqrt(prng.random())
        x, y = MESA_C[0] + math.cos(a) * rr, MESA_C[1] + math.sin(a) * rr
        rx = prng.uniform(1.5, 6.0)
        rim = [top_z(x + rx * math.cos(b), y + rx * math.sin(b)) for b in np.linspace(0, 2 * math.pi, 12)]
        if min(rim) < MESA_H - 4.5 or any(math.hypot(x - ax, y - ay) < ar + 2 for (ax, ay, ar) in avoid):
            continue
        patches.append((x, y, rx, rx * prng.uniform(0.45, 0.9), prng.uniform(0, math.pi)))
    namek.sand_patches("SandPatches", patches, mesa_height, namek.sand_material())
    mclusters = []
    while len(mclusters) < 50:
        k = int(prng.integers(0, len(path_xy)))
        p = path_xy[k]
        if not (190 < p[1] < 318):
            continue
        tng = path_xy[min(k + 1, len(path_xy) - 1)] - path_xy[max(k - 1, 0)]
        tng = tng / (np.linalg.norm(tng) + 1e-9)
        x, y = p + np.array([-tng[1], tng[0]]) * prng.uniform(2.5, 14) * prng.choice([-1, 1])
        if any(math.hypot(x - ax, y - ay) < ar for (ax, ay, ar) in avoid) or top_z(x, y) < MESA_H - 5:
            continue
        mclusters.append((float(x), float(y)))
    namek.mushrooms("Mushroom", mclusters, mesa_height, prng)

    # Blaues Gras als echte Halme entlang der Flugbahn (Dichte fällt mit dem Abstand)
    excl = [(x, y, r * 0.95) for (x, y, r) in houses] + [(DB_C[0], DB_C[1], 1.7), (SHIP_C[0], SHIP_C[1], 17.5)]
    excl += boulders
    excl += [(x, y, min(rx, ry) * 0.7) for (x, y, rx, ry, _) in patches]
    namek.grass_field("NamekGrass", mesa_height, path_xy, namek.grass_blade_material(), rng, ymin=172, ymax=330,
                      exclude=excl, zmin=MESA_H - 5)

    pos, quats, info = fpv.fpv_path(route, SPEED, FPS, frames, look_pitch=-3.0, pitch_follow=0.5,
                                    bank_gain=1.0, max_bank=30, micro=1.0, seed=13,
                                    pitch_overrides=[(8.6, 15.9, 6.0)])   # Blick leicht nach oben: Kampf
    fpv.make_camera(pos, quats, fov_deg=92.0)
    namek_fight(pos, top_z, rock, houses)
    bpy.context.scene.cycles.transparent_max_bounces = 16   # Aura, Strahlen, Glühhüllen übereinander
    print(f"route length {info['total']:.1f} m, used {info['used']:.1f} m")
    return sc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/nido_db")
    ap.add_argument("--res", type=lambda s: tuple(int(v) for v in s.split("x")), default=(1080, 1920))
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--no-mblur", action="store_true")
    ap.add_argument("--save-blend", default=None)
    args = ap.parse_args()
    build(args)
    fpv.try_gpu()
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
    frames = None
    if args.frames:
        if "-" in args.frames:
            a, b = args.frames.split("-")
            frames = range(int(a), int(b) + 1)
        else:
            frames = [int(v) for v in args.frames.split(",")]
    fpv.render_frames(args.out, frames)
