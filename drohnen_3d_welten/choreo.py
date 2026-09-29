"""Gemeinsame Choreografie-Bausteine für Kämpfe mit Speed-Ramp-Kamera (Naruto, Namek).

- Kamera aus zeitgestempelten Wegpunkten: Catmull-Rom-Kurve, Bogenlänge über die Zeit monoton-kubisch (Speed-Ramp)
- Hit-Stop als Zeitverzerrung (n Frames Stillstand, danach n Frames Aufholen – kein bleibender Versatz)
- Blocking -> gebackene Figurenanimation (figures.Figure): Ausholen/Überschwingen, Nachziehen, Atmen,
  weiche Bahn mit Bodenkontakt, optional Neigung (Flug) und Blick zur Kamera
- Kamerastöße als abklingendes, verrauschtes Rütteln
"""
import math

import numpy as np
from mathutils import Euler

import fpv
from figures import POSES


def pchip(tk, yk, t):
    """Monotone kubische Interpolation (Fritsch–Carlson): keine Überschwinger, stetige Geschwindigkeit."""
    tk, yk = np.asarray(tk, float), np.asarray(yk, float)
    h = np.diff(tk)
    d = np.diff(yk) / h
    m = np.zeros_like(yk)
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(yk) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    t = np.asarray(t, float)
    k = np.clip(np.searchsorted(tk, t) - 1, 0, len(tk) - 2)
    u = (t - tk[k]) / h[k]
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return h00 * yk[k] + h10 * h[k] * m[k] + h01 * yk[k + 1] + h11 * h[k] * m[k + 1]


def smooth(x0, x1, t):
    u = np.clip((np.asarray(t, float) - x0) / (x1 - x0), 0, 1)
    return u * u * (3 - 2 * u)


def keyed_path(keys, fps, frames):
    """Orte pro Frame 0..frames+1 entlang einer Catmull-Rom-Kurve durch keys = [(t, (x, y, z)), ...];
    Bogenlänge über die Zeit monoton-kubisch (Speed-Ramp). Rückgabe: t, pos, Tempo (m/s)."""
    n = frames + 2
    t = (np.arange(n) - 1.0) / fps
    pts = [p for _, p in keys]
    seg = 400
    curve = fpv._catmull_rom(pts, samples_per_seg=seg)
    sa = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    s_knot = [sa[min(k * seg, len(sa) - 1)] for k in range(len(pts))]
    s_knot[-1] = sa[-1]
    s = pchip([k for k, _ in keys], s_knot, np.clip(t, 0, keys[-1][0]))
    s[t < 0] = s_knot[0] + t[t < 0] * (s[2] - s[1]) * fps      # Frame 0 (vor dem Start) für Motion-Blur
    pos = np.stack([np.interp(s, sa, curve[:, k]) for k in range(3)], axis=1)
    if (t < 0).any():
        d0 = (curve[1] - curve[0]) / np.linalg.norm(curve[1] - curve[0])
        pos[t < 0] = curve[0] + np.outer(s[t < 0] - s_knot[0], d0)
    v = np.gradient(s) * fps
    return t, pos, v


def time_warp(t, holds, fps):
    """Hit-Stop: an jedem (Zeit, Frames) steht die Zeit n Frames still und holt danach in n Frames wieder auf."""
    tau = np.array(t, dtype=float)
    for h, nf in holds:
        H = nf / fps
        a = (t >= h) & (t < h + H)
        b = (t >= h + H) & (t < h + 2 * H)
        tau[a] = h
        tau[b] = h + 2 * (t[b] - h - H)
    return tau


ATTACKS = ("punch_R", "punch_L", "kick_R", "kick_L", "dash_thrust_R", "dash_thrust_L")


def default_style(pose):
    """(Dauer, Ausholen, Überschwingen) je Posenart: Angriffe mit Ausholen und Überschwingen, Treffer-Reaktion
    schnell mit Nachschwingen, Landung federt, Lauf knapp, Halteposen weich."""
    if pose in ATTACKS:
        return 0.36, 0.2, 0.14
    if pose == "recoil":
        return 0.12, 0.0, 0.18
    if pose in ("land", "squat"):
        return 0.14, 0.0, 0.14
    if pose == "jump":
        return 0.2, 0.1, 0.06
    if isinstance(pose, str) and pose.startswith("run"):
        return 0.14, 0.0, 0.05
    if isinstance(pose, str) and pose.startswith("crouch_charge"):
        return 0.3, 0.1, 0.06
    return 0.32, 0.06, 0.07


def dedup_keys(keys):
    ks = []
    for k in sorted(keys, key=lambda k: k[0]):
        if ks and abs(k[0] - ks[-1][0]) < 1e-6:
            ks[-1] = k
        else:
            ks.append(k)
    return ks


def bake_fighter(fig, keys, n, fps, hits, style=default_style, cam_pos=None, look_win=None, seed=0):
    """Blocking -> gebackene Animation. keys = [(t, Pose, Ort, Blickziel, in der Luft[, opts]), ...];
    opts (optional): {"lean": Grad vor/zurück, "roll": Grad seitlich, "rot": {Gelenk: (x, y, z)} zusätzlich}.
    Posenwechsel mit Ausholen/Überschwingen (crew._prog), Nachziehen von Unterarm/Hand/Kopf, Atmen/Mikrobewegung,
    Hit-Stop an Treffern, weiche Bahn (monoton-kubisch) mit Bodenkontakt über Vorwärtskinematik, Blick zur
    Kamera im Fenster look_win."""
    import crew
    ks = dedup_keys(keys)
    tk = np.array([k[0] for k in ks])
    t = (np.arange(n) - 1.0) / fps
    tw = time_warp(t, hits, fps)

    def pose_of(k):
        p = POSES[k[1]] if isinstance(k[1], str) else k[1]
        if len(k) > 5 and k[5] and k[5].get("rot"):
            p = dict(p)
            p.update(k[5]["rot"])
        return p
    # Posenplan
    sched = [(0.0, pose_of(ks[0]), 1.0, 0.0, 0.0)]
    for k in range(1, len(ks)):
        dur, a, o = style(ks[k][1])
        d = min(dur, max(tk[k] - tk[k - 1], 1.0 / fps))
        sched.append((tk[k] - d, pose_of(ks[k]), d, a, o))
    R = {}
    for j in fig.J:
        lag = crew.LAG.get(j.split(".")[0], 0.0)
        R[j] = crew._eval_schedule(sched, j, tw - lag)
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.28, 4)
    br = np.sin(2 * np.pi * 0.28 * t + ph[0])
    R["spine"][:, 0] += 1.0 * br
    R["shoulder.R"][:, 1] += 0.6 * br
    R["shoulder.L"][:, 1] -= 0.6 * br
    R["head"][:, 0] += 1.0 * np.sin(2 * np.pi * 0.41 * t + ph[1])
    R["head"][:, 2] += 1.4 * np.sin(2 * np.pi * 0.23 * t + ph[2])
    # Bahn: Basis-z je Schlüssel (Boden: Fußkontakt), monoton-kubisch dazwischen
    P = np.array([tuple(k[2]) for k in ks], dtype=float)
    air = np.array([k[4] for k in ks])
    bz = np.array([P[k, 2] if air[k] else P[k, 2] - fig.foot_drop(pose_of(ks[k])) for k in range(len(ks))])
    tc = np.clip(tw, tk[0], tk[-1])
    L = np.stack([pchip(tk, P[:, 0], tc), pchip(tk, P[:, 1], tc), pchip(tk, bz, tc)], axis=1)
    seg = np.clip(np.searchsorted(tk, tw, side="right") - 1, 0, len(ks) - 2)
    ground_seg = (~air[seg]) & (~air[seg + 1])
    for f in np.where(ground_seg)[0]:
        u = np.clip((tw[f] - tk[seg[f]]) / max(tk[seg[f] + 1] - tk[seg[f]], 1e-6), 0, 1)
        gz = P[seg[f], 2] * (1 - u) + P[seg[f] + 1, 2] * u
        L[f, 2] = gz - fig.foot_drop({j: tuple(R[j][f]) for j in ("root", "hip.R", "knee.R", "ankle.R",
                                                                  "hip.L", "knee.L", "ankle.L")})
    # Blickrichtung (Yaw) je Schlüssel, stetig; Neigung/Rollen (Flug) je Schlüssel
    yaw_k = np.unwrap(np.array([math.atan2(-(k[3] - k[2]).x, (k[3] - k[2]).y) for k in ks]))
    yaw = pchip(tk, yaw_k, tc)
    opt = lambda k, name: math.radians((k[5] or {}).get(name, 0.0)) if len(k) > 5 else 0.0
    lean_k = np.array([opt(k, "lean") for k in ks])
    roll_k = np.array([opt(k, "roll") for k in ks])
    lean = np.interp(tc, tk, lean_k) if lean_k.any() else np.zeros(n)
    roll = np.interp(tc, tk, roll_k) if roll_k.any() else np.zeros(n)
    if lean_k.any() or roll_k.any():
        import crew as _c
        lean, roll = _c._zero_phase(lean, 0.08, fps), _c._zero_phase(roll, 0.08, fps)
    # Blick zur Kamera (Hals/Kopf), weich ein- und ausgeblendet
    if cam_pos is not None and look_win:
        w = smooth(look_win[0], look_win[0] + 0.6, t) * (1 - smooth(look_win[1] - 0.4, look_win[1], t))
        head = L + np.array([0, 0, 1.5 * fig.s])
        d = cam_pos[:n] - head
        dyaw = np.degrees(np.arctan2(-d[:, 0], d[:, 1]) - yaw)
        dyaw = (dyaw + 180) % 360 - 180
        dpit = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
        ay = crew._zero_phase(np.clip(dyaw - R["head"][:, 2] - R["spine"][:, 2], -60, 60) * w, 0.15)
        ap = crew._zero_phase(np.clip(dpit - R["head"][:, 0] - R["spine"][:, 0], -25, 30) * w, 0.15)
        R["neck"][:, 2] += 0.6 * ay
        R["head"][:, 2] += 0.4 * ay
        R["neck"][:, 0] += 0.5 * ap
        R["head"][:, 0] += 0.5 * ap
    for j, e in fig.J.items():
        e.rotation_mode = "XYZ"
        crew._bake(e, "rotation_euler", np.radians(R[j]))
    crew._bake(fig.base, "location", L)
    fig.base.rotation_mode = "XYZ"
    crew._bake(fig.base, "rotation_euler", np.stack([lean, roll, yaw], axis=1))
    return L


def track(keys, t):
    """Ort einer Figur laut Blocking zu Zeiten t (linear zwischen den Schlüsseln), (n, 3)."""
    ks = sorted(keys, key=lambda k: k[0])
    tk = np.array([k[0] for k in ks])
    P = np.array([tuple(k[2]) for k in ks])
    return np.stack([np.interp(t, tk, P[:, j]) for j in range(3)], axis=1)


def camera_shakes(pos, quats, t, shakes, fps, seed=77, pos_amp=0.03):
    """Kamerastöße (Zeit, Stärke in Grad, Frames): kurzes, stark verrauschtes, abklingendes Rütteln
    (Rollen/Nicken/Gieren + Versatz). Ändert pos/quats an Ort und Stelle."""
    rng = np.random.default_rng(seed)
    for (h, amp, nf) in shakes:
        idx = np.where((t >= h) & (t < h + nf / fps))[0]
        for k, i in enumerate(idx):
            env = (1 - k / len(idx)) ** 1.5
            r = rng.uniform(-1, 1, 3) * math.radians(amp) * env
            quats[i] = quats[i] @ Euler((r[0], r[1] * 0.5, r[2]), "XYZ").to_quaternion()
            pos[i] = pos[i] + rng.uniform(-1, 1, 3) * pos_amp * amp * env
