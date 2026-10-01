"""Reparatur unvollständiger Tripo-GLB-Downloads (Safari „.glb.download“): Ist der Binärteil abgeschnitten, aber
Mesh und Farbtextur vollständig, wird die Datei auf die angegebene Länge aufgefüllt, die Bind-Matrizen der Haut
werden aus den Gelenk-Knoten neu berechnet (Ruhelage = Bindpose) und abgeschnittene Bilder aus den Materialien
entfernt.  Aufruf: python repair_glb.py eingabe.glb ausgabe.glb"""
import json
import struct
import sys

import numpy as np


def _local(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R @ np.diag(n.get("scale", [1, 1, 1]))
    M[:3, 3] = n.get("translation", [0, 0, 0])
    return M


def repair(src, dst):
    d = open(src, "rb").read()
    jlen = struct.unpack("<I", d[12:16])[0]
    J = json.loads(d[20:20 + jlen])
    blen, btype = struct.unpack("<II", d[20 + jlen:28 + jlen])
    B = bytearray(d[28 + jlen:])
    avail = len(B)
    B += b"\0" * (blen - len(B))
    cut = {i for i, bv in enumerate(J["bufferViews"]) if bv.get("byteOffset", 0) + bv["byteLength"] > avail}
    prim_views = {J["accessors"][a]["bufferView"] for m in J["meshes"] for p in m["primitives"]
                  for a in list(p["attributes"].values()) + [p.get("indices")] if a is not None}
    if cut & prim_views:
        raise SystemExit("Mesh-Daten abgeschnitten – nicht reparierbar")
    nodes = J["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

    def glob(i):
        M = _local(nodes[i])
        while i in parent:
            i = parent[i]
            M = _local(nodes[i]) @ M
        return M
    for sk in J.get("skins", []):
        a = J["accessors"][sk["inverseBindMatrices"]]
        bv = J["bufferViews"][a["bufferView"]]
        if a["bufferView"] in cut:
            off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
            ibm = np.stack([np.linalg.inv(glob(j)) for j in sk["joints"]])
            B[off:off + 64 * len(ibm)] = np.ascontiguousarray(ibm.transpose(0, 2, 1)).astype("<f4").tobytes()
    bad_img = {i for i, im in enumerate(J.get("images", [])) if im.get("bufferView") in cut}
    bad_tex = {i for i, t in enumerate(J.get("textures", [])) if t.get("source") in bad_img}
    if bad_img and any(J["images"][i].get("name") == "Color" for i in bad_img):
        raise SystemExit("Farbtextur abgeschnitten – nicht reparierbar")
    for m in J.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        for key in ("metallicRoughnessTexture", "baseColorTexture"):
            if pbr.get(key, {}).get("index") in bad_tex:
                pbr.pop(key)
        for key in ("normalTexture", "occlusionTexture", "emissiveTexture"):
            if m.get(key, {}).get("index") in bad_tex:
                m.pop(key)
    js = json.dumps(J, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = struct.pack("<4sII", b"glTF", 2, 28 + len(js) + len(B)) + struct.pack("<II", len(js), 0x4E4F534A) + js + \
        struct.pack("<II", len(B), btype) + bytes(B)
    open(dst, "wb").write(out)
    print(f"repariert: {dst} ({len(out)} Bytes, {len(cut)} abgeschnittene Puffer ersetzt/entfernt)")


if __name__ == "__main__":
    repair(sys.argv[1], sys.argv[2])
