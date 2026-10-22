r"""
export_glb.py -- VISUALIZATION/EXPORT ONLY: final phone point cloud -> glTF 2.0 binary (.glb).

Modelled on depth-anything.cpp's src/glb_export.cpp (dependency-free GLB: one POINTS primitive with POSITION float32
VEC3 + COLOR_0 normalized uint8 VEC4, no materials; cameras as a separate LINES primitive), with one difference:
the reconstruction geometry is NOT re-projected or re-centred. Point positions are written exactly as stored in
scene.ply (float64 -> float32, the only representable glTF type); the display transform (gravity alignment to glTF
y-up, floor at y = 0, median-centred) is put in the glTF NODE matrix, so viewers apply it at render time and the
vertex data can be compared 1:1 with the PLY.

Inputs (read only):
  experiments\phone_real_v2\improved\scene.ply              final point cloud (binary PLY: double xyz, uchar rgb)
  experiments\phone_real_v2\improved\trajectory_fused.txt   recovered COLMAP camera centres + orientations (fused frame)
  experiments\phone_real_v2\figures\scene_frame.json        gravity frame (side/fwd/up, floor) from phone_report.py
Outputs (this folder):
  final_phone.glb                          point cloud only
  final_phone_with_camera_trajectory.glb   same point cloud + camera path (magenta line), frustums every 10th
                                           camera (yellow), start (green) / end (red) markers
  export_report.json                       counts, bounds, round-trip check
"""
import json
import os
import struct

import numpy as np

ROOT = r"Q:\OPEN_CV_SEM5"
P2 = os.path.join(ROOT, r"experiments\phone_real_v2")
OUT = os.path.dirname(os.path.abspath(__file__))
PLY = os.path.join(P2, r"improved\scene.ply")
TRAJ = os.path.join(P2, r"improved\trajectory_fused.txt")
FRAME = os.path.join(P2, r"figures\scene_frame.json")


def read_ply(path):
    """Minimal binary-little-endian PLY reader (exact bytes; no colour conversion)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        data = f.read()
    assert "format binary_little_endian 1.0" in header, header
    n = int(next(l.split()[2] for l in header if l.startswith("element vertex")))
    props = [l.split()[1:] for l in header if l.startswith("property")]
    tmap = {"double": "<f8", "float": "<f4", "uchar": "u1", "uint8": "u1"}
    dt = np.dtype([(name, tmap[t]) for t, name in props])
    v = np.frombuffer(data, dtype=dt, count=n)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1) if "red" in dt.names else None
    return xyz, rgb, header


def quat_to_R(qx, qy, qz, qw):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def write_glb(path, primitives, node_matrix):
    """primitives: list of (mode, positions float32 Nx3, colors uint8 Nx4). One mesh, one node with node_matrix."""
    bin_ = bytearray(); views, accs, prims = [], [], []

    def add_view(arr):
        while len(bin_) % 4:
            bin_.append(0)
        off = len(bin_); b = arr.tobytes(); bin_.extend(b)
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(b)})
        return len(views) - 1

    for mode, pos, col in primitives:
        pos = np.ascontiguousarray(pos, dtype="<f4"); col = np.ascontiguousarray(col, dtype="u1")
        vp = add_view(pos); vc = add_view(col)
        accs.append({"bufferView": vp, "componentType": 5126, "count": len(pos), "type": "VEC3",
                     "min": [float(x) for x in pos.min(0)], "max": [float(x) for x in pos.max(0)]})
        accs.append({"bufferView": vc, "componentType": 5121, "normalized": True, "count": len(col), "type": "VEC4"})
        prims.append({"attributes": {"POSITION": len(accs) - 2, "COLOR_0": len(accs) - 1}, "mode": mode})
    while len(bin_) % 4:
        bin_.append(0)
    gltf = {"asset": {"version": "2.0", "generator": "Scan2Scene export_glb.py (modelled on depth-anything.cpp glb_export)"},
            "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "matrix": [float(x) for x in node_matrix.T.reshape(-1)],   # glTF: column-major
                       "name": "final_phone_reconstruction"}],
            "meshes": [{"primitives": prims}], "accessors": accs, "bufferViews": views,
            "buffers": [{"byteLength": len(bin_)}]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(bin_), 0x004E4942)); f.write(bin_)
    return total


def read_glb_first_primitive(path):
    """Round-trip reader for verification."""
    b = open(path, "rb").read()
    jl = struct.unpack("<I", b[12:16])[0]
    g = json.loads(b[20:20 + jl]); bin_ = b[20 + jl + 8:]
    a = g["accessors"]; v = g["bufferViews"]
    pa, ca = a[0], a[1]
    pos = np.frombuffer(bin_, "<f4", pa["count"] * 3, v[pa["bufferView"]]["byteOffset"]).reshape(-1, 3)
    col = np.frombuffer(bin_, "u1", ca["count"] * 4, v[ca["bufferView"]]["byteOffset"]).reshape(-1, 4)
    return g, pos, col


def main():
    xyz, rgb, header = read_ply(PLY)
    n = len(xyz)
    col = np.concatenate([rgb if rgb is not None else np.full((n, 3), 200, "u1"), np.full((n, 1), 255, "u1")], 1)
    pos = xyz.astype("<f4")

    # display transform (node matrix only): rows = side, up, fwd  (right-handed: side x up = fwd), floor -> y = 0
    sf = json.load(open(FRAME))
    Rd = np.stack([sf["side"], sf["up"], sf["fwd"]])
    s = xyz @ Rd.T
    t = -np.array([np.median(s[:, 0]), sf["floor"], np.median(s[:, 2])])
    M = np.eye(4); M[:3, :3] = Rd; M[:3, 3] = t

    size_a = write_glb(os.path.join(OUT, "final_phone.glb"), [(0, pos, col)], M)

    # camera trajectory (same fused frame as the cloud; same node matrix)
    tr = np.loadtxt(TRAJ)
    C = tr[:, 1:4]
    diag = np.linalg.norm(np.percentile(xyz, 95, 0) - np.percentile(xyz, 5, 0))
    fs = 0.02 * diag                                          # frustum depth ~2 % of the scene diagonal
    path_seg = np.repeat(C, 2, axis=0)[1:-1]                  # consecutive centres as line segments
    lines, lcol = [path_seg], [np.tile([255, 0, 255, 255], (len(path_seg), 1))]
    w, h = fs * 0.5 * 960 / 540, fs * 0.5
    for i in range(0, len(tr), 10):
        Rc = quat_to_R(*tr[i, 4:8]); c = C[i]
        corners = [c + Rc @ np.array([x, y, fs]) for x, y in ((-w, -h), (w, -h), (w, h), (-w, h))]
        segs = [c, corners[0], c, corners[1], c, corners[2], c, corners[3]]
        for k in range(4):
            segs += [corners[k], corners[(k + 1) % 4]]
        lines.append(np.array(segs)); lcol.append(np.tile([255, 220, 0, 255], (len(segs), 1)))
    for c, rgba in ((C[0], [0, 255, 0, 255]), (C[-1], [255, 40, 40, 255])):   # start / end crosses
        m = fs * 0.6
        segs = [c - [m, 0, 0], c + [m, 0, 0], c - [0, m, 0], c + [0, m, 0], c - [0, 0, m], c + [0, 0, m]]
        lines.append(np.array(segs)); lcol.append(np.tile(rgba, (6, 1)))
    L = np.concatenate(lines).astype("<f4"); LC = np.concatenate(lcol).astype("u1")
    size_b = write_glb(os.path.join(OUT, "final_phone_with_camera_trajectory.glb"), [(0, pos, col), (1, L, LC)], M)

    # round-trip verification against the PLY
    g, rpos, rcol = read_glb_first_primitive(os.path.join(OUT, "final_phone.glb"))
    rep = dict(source_ply=PLY, ply_header=header, points=n, colors_in_ply=rgb is not None,
               glb=dict(final_phone=dict(bytes=size_a),
                        final_phone_with_camera_trajectory=dict(bytes=size_b, cameras=len(tr), frustums=len(range(0, len(tr), 10)),
                                                                line_vertices=len(L))),
               roundtrip=dict(points_equal_count=len(rpos) == n,
                              max_abs_position_diff=float(np.abs(rpos.astype(np.float64) - xyz).max()),
                              position_extent=float(np.abs(xyz).max()),
                              colors_identical=bool(np.array_equal(rcol[:, :3], col[:, :3]))),
               node_matrix_row_major=M.tolist(), frustum_depth_fused_units=float(fs),
               notes="positions stored as in scene.ply (float64->float32 cast only); display transform in node matrix")
    json.dump(rep, open(os.path.join(OUT, "export_report.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "ply_header"}, indent=1))


if __name__ == "__main__":
    main()
