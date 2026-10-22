"""Consistency / confidence maps for representative frames (from reconstruct.py --save_confidence).
The last column (GT depth error) is evaluation-only and is NOT used by the method."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = r"Q:\OPEN_CV_SEM5"
sys.path.insert(0, ROOT)
from colmap_to_tum import name_to_timestamp  # noqa: E402
from eval_tum import associate, read_list  # noqa: E402
from figlib import FR1, FR2, sim3_to_gt  # noqa: E402

RUNS = os.path.join(ROOT, r"experiments\weighted_fusion\runs")
OUTD = os.path.join(ROOT, r"experiments\weighted_fusion\figures")
os.makedirs(OUTD, exist_ok=True)
SETS = {"fr2_xyz": (FR2, os.path.join(ROOT, "frames_subset")),
        "fr1_desk": (FR1, os.path.join(ROOT, r"experiments\tum_freiburg1_desk\frames"))}

for ds, (gtd, frames_dir) in SETS.items():
    rd = os.path.join(RUNS, ds, "weighted")
    s = sim3_to_gt(os.path.join(rd, "trajectory_fused.txt"), os.path.join(gtd, "groundtruth.txt"))[0]
    dep = read_list(os.path.join(gtd, "depth.txt"))
    files = sorted(os.listdir(os.path.join(rd, "confidence")))
    picks = [files[int(len(files) * q)] for q in (0.2, 0.5, 0.8)]
    cols = ["RGB", "corrected depth", "multi-view residual (prev)", "multi-view residual (next)",
            "edge factor", "sparse-support factor", "final confidence (red = rejected)", "GT depth error (eval only)"]
    fig, axs = plt.subplots(len(picks), len(cols), figsize=(3.2 * len(cols), 2.7 * len(picks)), dpi=110)
    for row, f in zip(axs, picks):
        z = np.load(os.path.join(rd, "confidence", f))
        stem = os.path.splitext(f)[0]
        img = np.asarray(Image.open(os.path.join(frames_dir, stem + ".png")).convert("RGB"))[::2, ::2]
        d = z["depth"].astype(float)
        rels = sorted([k for k in z.files if k.startswith("rel_")])
        prev = next((z[k].astype(float) for k in rels if k.startswith("rel_-")), None)
        nxt = next((z[k].astype(float) for k in rels if k.startswith("rel_+")), None)
        t = name_to_timestamp(stem)
        a = associate([t], [r[0] for r in dep], 0.02)
        gd = np.asarray(Image.open(os.path.join(gtd, dep[a[t]][1][0])), float)[::2, ::2] / 5000.0 if t in a else None
        conf = z["conf"].astype(float)
        panels = [img, np.where(d > 0, d, np.nan), prev, nxt, z["c_edge"].astype(float), z["c_sparse"].astype(float), conf,
                  np.where((d > 0) & (gd > 0.1), np.abs(d * s / np.maximum(gd, 1e-6) - 1), np.nan) if gd is not None else None]
        cmaps = [None, "viridis", "RdBu_r", "RdBu_r", "magma", "magma", "magma", "inferno"]
        lims = [None, None, (-0.15, 0.15), (-0.15, 0.15), (0, 1), (0, 1), (0, 1), (0, 0.2)]
        for ax, p, cm, lim, title in zip(row, panels, cmaps, lims, cols):
            ax.axis("off")
            if p is None:
                ax.text(0.5, 0.5, "n/a", ha="center", transform=ax.transAxes); continue
            if cm is None:
                ax.imshow(p)
            else:
                im = ax.imshow(p, cmap=cm, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
            if title.startswith("final"):
                rej = z["reject"]
                ov = np.zeros(rej.shape + (4,)); ov[rej] = [1, 0.2, 0.2, 1]
                ax.imshow(ov)
            ax.set_title(title, fontsize=8)
        row[0].set_title(f"{ds} · {stem}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, f"confidence_maps_{ds}.png")); plt.close(fig)
    print("wrote", ds)
