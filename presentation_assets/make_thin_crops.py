"""Thin/small-object crops: points inside the 'objects above the desk' region (defined from GT geometry only),
rendered from the same virtual camera for every method, plus the GT cloud."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figlib import *  # noqa: F401,F403

sys.path.insert(0, ROOT)
from surface_metrics import gt_cloud_tum  # noqa: E402
from thin_structure_eval import desk_objects  # noqa: E402

SD = os.path.join(ROOT, r"experiments\spatial_depth\runs")
WF = os.path.join(ROOT, r"experiments\weighted_fusion\runs")
OUTD = os.path.join(ROOT, r"experiments\weighted_fusion\figures")
Z = np.array([0, 0, 1.0])
CASES = {
    "fr1_desk": (FR1, [("baseline (scale+voxel)", SD + r"\fr1_desk\baseline_scale"), ("spatial+voxel", SD + r"\fr1_desk\spatial_perframe_from_fr2_xyz"),
                       ("spatial+weighted", WF + r"\fr1_desk\weighted"), ("spatial+TSDF", WF + r"\fr1_desk\tsdf_spatial")]),
    "fr2_xyz": (FR2, [("baseline (scale+voxel)", SD + r"\fr2_xyz\baseline_scale"), ("spatial+voxel", SD + r"\fr2_xyz\spatial_perframe_from_fr1_desk"),
                      ("spatial+weighted", WF + r"\fr2_xyz\weighted"), ("spatial+TSDF", WF + r"\fr2_xyz\tsdf_spatial")]),
}
for name, (gtd, runs) in CASES.items():
    est_t = np.loadtxt(os.path.join(runs[0][1], "trajectory_fused.txt"))[:, 0]
    G = np.asarray(gt_cloud_tum(gtd, est_t, 0.01).points)
    obj, info = desk_objects(G, up=2)
    O = G[obj]
    lo, hi = np.percentile(O, 1, 0) - 0.05, np.percentile(O, 99, 0) + 0.05
    lo[2] = info["desk_height"] - 0.03
    ctr = (lo + hi) / 2
    span = np.linalg.norm(hi - lo)
    _, _, _, E, _ = sim3_to_gt(os.path.join(runs[0][1], "trajectory_fused.txt"), os.path.join(gtd, "groundtruth.txt"))
    view_dir = E.mean(0) - ctr; view_dir[2] = 0; view_dir /= np.linalg.norm(view_dir)
    eye = ctr + view_dir * span * 0.9 + np.array([0, 0, span * 0.7])
    panels = [("ground truth (Kinect)", G, np.tile([[0.8, 0.82, 0.85]], (len(G), 1)))]
    for lab, rd in runs:
        s, R, t, _, _ = sim3_to_gt(os.path.join(rd, "trajectory_fused.txt"), os.path.join(gtd, "groundtruth.txt"))
        P, C = load_ply(os.path.join(rd, "scene.ply"))
        panels.append((lab, s * P @ R.T + t, C))
    fig, axs = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.0), dpi=130)
    fig.patch.set_facecolor(hexc(BG))
    for ax, (lab, P, C) in zip(axs, panels):
        k = np.all((P >= lo) & (P <= hi), axis=1)
        ax.imshow(render(P[k], C[k], eye, ctr, Z, W=1100, H=850, fov=45, px=2, ss=2, alpha_bg=False, gain=1.1))
        ax.set_title(f"{name} · {lab}\n{k.sum():,} pts in crop", color=hexc(TEXT), fontsize=10); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, f"thin_objects_{name}.png"), facecolor=hexc(BG)); plt.close(fig)
    print("wrote", name)
