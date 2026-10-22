"""VISUALIZATION-ONLY Poisson meshes of the final phone point cloud (experiments\\phone_real_v2\\improved\\scene.ply).
The input file is only read. Normals: the cloud has none -> estimated (hybrid KNN/radius) and oriented towards the
nearest recovered camera centre. Poisson depths 9 and 10; for each: vertex/triangle counts, runtime, and 'support' =
distance from mesh vertices to the nearest input point (vertices > 5 fused-cm away are surface Poisson invented).
A density-trimmed copy (drop the lowest-density 10 % of vertices) is saved separately, clearly labelled.
No accuracy is claimed for any mesh."""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, r"Q:\OPEN_CV_SEM5\presentation_assets")
from figlib import BG, TEXT, hexc, render  # noqa: E402

P2 = r"Q:\OPEN_CV_SEM5\experiments\phone_real_v2"
OUT = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(P2, r"improved\scene.ply")
pcd = o3d.io.read_point_cloud(src)
info = dict(input=src, input_points=len(pcd.points), input_had_normals=pcd.has_normals())
t = time.time()
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
P = np.asarray(pcd.points)
cams = np.loadtxt(os.path.join(P2, r"improved\trajectory_fused.txt"))[:, 1:4]
_, ci = cKDTree(cams).query(P)
N = np.asarray(pcd.normals)
flip = np.einsum("ij,ij->i", N, cams[ci] - P) < 0
N[flip] *= -1
pcd.normals = o3d.utility.Vector3dVector(N)
info["normals"] = dict(method="estimate_normals hybrid radius 0.08 fused units, max_nn 30; oriented towards nearest camera centre",
                       seconds=time.time() - t, flipped_frac=float(flip.mean()))
tree = cKDTree(P)
sf = json.load(open(os.path.join(P2, r"figures\scene_frame.json")))
Rsc = np.stack([sf["side"], sf["fwd"], sf["up"]])
light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)


def shaded_samples(mesh, n=6_000_000):
    s = mesh.sample_points_uniformly(n)
    s.estimate_normals() if not s.has_normals() else None
    Q = np.asarray(s.points) @ Rsc.T; Q[:, 2] -= sf["floor"]
    Nn = np.asarray(s.normals) @ Rsc.T
    col = np.asarray(s.colors) * (0.45 + 0.55 * np.abs(Nn @ light))[:, None]
    return Q, np.clip(col, 0, 1)


views = []
Pv = P @ Rsc.T; Pv[:, 2] -= sf["floor"]
c0 = np.median(Pv, 0); zc = np.percentile(Pv[:, 2], 97) - 0.25
cams_v = cams @ Rsc.T; cams_v[:, 2] -= sf["floor"]
EYE = [(c0 + [-3.5, -4.5, 3.2], c0, np.array([0, 0, 1.0]), "overall")]
i = 33; Rc = Rsc @ __import__("eval_synth", fromlist=["quat_to_R"]).quat_to_R(*np.loadtxt(os.path.join(P2, r"improved\trajectory_fused.txt"))[i, 4:8]) if False else None
panels = [("point cloud (input, unchanged)", Pv, np.asarray(pcd.colors))]
res = []
for depth in (9, 10):
    t = time.time()
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=1.1, linear_fit=False)
    dt = time.time() - t
    dens = np.asarray(dens)
    V = np.asarray(mesh.vertices)
    d, _ = tree.query(V)
    r = dict(depth=depth, scale=1.1, linear_fit=False, seconds=dt, vertices=len(V), triangles=len(mesh.triangles),
             vertex_support_median=float(np.median(d)), unsupported_frac_gt_5cm=float(np.mean(d > 0.05)),
             unsupported_frac_gt_20cm=float(np.mean(d > 0.20)))
    o3d.io.write_triangle_mesh(os.path.join(OUT, f"phone_poisson_depth{depth}.ply"), mesh, write_ascii=False)
    keep = dens >= np.quantile(dens, 0.10)
    trimmed = mesh.select_by_index(np.flatnonzero(keep))
    dT, _ = tree.query(np.asarray(trimmed.vertices))
    r.update(trimmed_vertices=len(trimmed.vertices), trimmed_triangles=len(trimmed.triangles),
             trimmed_unsupported_frac_gt_5cm=float(np.mean(dT > 0.05)), trimmed_unsupported_frac_gt_20cm=float(np.mean(dT > 0.20)))
    o3d.io.write_triangle_mesh(os.path.join(OUT, f"phone_poisson_depth{depth}_density_trimmed10.ply"), trimmed, write_ascii=False)
    res.append(r)
    for m, lab in ((mesh, f"Poisson depth {depth} (untrimmed)"), (trimmed, f"Poisson depth {depth}, lowest-density 10 % removed")):
        Q, col = shaded_samples(m)
        panels.append((lab, Q, col))
    print(json.dumps(r))
info["poisson"] = res
json.dump(info, open(os.path.join(OUT, "mesh_report.json"), "w"), indent=2)

# figure: identical camera, ceiling cut, same renderer for cloud and mesh surface samples
eye, tgt, up, _ = EYE[0]
fig, axs = plt.subplots(len(panels), 1, figsize=(16, 10 * len(panels)), dpi=90)
fig.patch.set_facecolor(hexc(BG))
for ax, (lab, Q, col) in zip(axs, panels):
    k = Q[:, 2] < zc
    ax.imshow(render(Q[k], col[k], eye, tgt, up, W=1600, H=1000, fov=55, px=2, ss=2, alpha_bg=False, fog=0.2))
    ax.set_title(lab, color=hexc(TEXT), fontsize=18); ax.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "phone_cloud_vs_poisson.png"), facecolor=hexc(BG)); plt.close(fig)
# top view (shows invented surface outside the room)
fig, axs = plt.subplots(1, len(panels), figsize=(8 * len(panels), 8), dpi=90); fig.patch.set_facecolor(hexc(BG))
for ax, (lab, Q, col) in zip(axs, panels):
    k = Q[:, 2] < zc
    ax.imshow(render(Q[k], col[k], c0 + [0, 0.01, 11.0], c0, np.array([0, 1.0, 0]), W=1000, H=1000, fov=60, px=2, ss=1, alpha_bg=False))
    ax.set_title(lab, color=hexc(TEXT), fontsize=14); ax.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "phone_cloud_vs_poisson_top.png"), facecolor=hexc(BG)); plt.close(fig)
print("done")
