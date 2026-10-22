"""Per-frame view of the spatial correction: sparse residual before (per-frame scale), fitted correction
field, and HELD-OUT residual after (field fitted on 3/4 of the image tiles, scored on the other 1/4)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = r"Q:\OPEN_CV_SEM5"
sys.path.insert(0, ROOT)
import Reconstruct as R  # noqa: E402
from fit_spatial_correction import load, CLAMP, INLIER_TOL  # noqa: E402

OUTD = os.path.join(ROOT, r"experiments\spatial_depth\figures")
os.makedirs(OUTD, exist_ok=True)
CFG = {"fr2_xyz": "calib_fr1_desk_perframe.json", "fr1_desk": "calib_fr2_xyz_perframe.json"}

for ds, cfgf in CFG.items():
    cfg = json.load(open(os.path.join(ROOT, "experiments", "spatial_depth", cfgf)))
    D = load(ds)
    nF = len(D["names"])
    picks = [int(nF * q) for q in (0.25, 0.6, 0.9)]
    fig, axs = plt.subplots(len(picks), 3, figsize=(16, 4.1 * len(picks)), dpi=120)
    for row, fi in zip(axs, picks):
        m = D["frame"] == fi
        img = np.asarray(Image.open(os.path.join(str(D["frames_dir"]), str(D["names"][fi]))).convert("L"))
        H, W = img.shape
        t = D["t"][m]; x, y, ld = D["x"][m], D["y"][m], D["ld"][m]
        before = t - np.median(t)                       # per-frame scale only
        tr = D["fold"][m] != 0; te = ~tr
        beta, _ = R.fit_spatial_frame(cfg["model"], x[tr], y[tr], ld[tr], t[tr], cfg["lam"], INLIER_TOL)
        ldr = (np.percentile(ld[tr], 1), np.percentile(ld[tr], 99))
        after = t[te] - R.spatial_log_factor(cfg["model"], beta, x[te], y[te], np.clip(ld[te], *ldr), CLAMP)
        # dense correction field (shape term only, at the frame's median depth for the log-depth term)
        uu, vv = np.meshgrid(np.arange(0, W, 4) + 0.5, np.arange(0, H, 4) + 0.5)
        gx = (uu - float(D["cx"])) / float(D["fx"]); gy = (vv - float(D["cy"])) / float(D["fy"])
        dep = np.load(os.path.join(ROOT, "experiments", "spatial_depth", "..", "..",
                                   {"fr2_xyz": r"experiments\tum_fr2_xyz\corrected_scale\depth_pred",
                                    "fr1_desk": r"experiments\tum_freiburg1_desk\depth_pred"}[ds],
                                   os.path.splitext(str(D["names"][fi]))[0] + ".npy"))[::4, ::4]
        gld = np.clip(np.log(np.maximum(dep, 1e-6) / float(D["dmed"][m][0])), *ldr)
        field = R.spatial_log_factor(cfg["model"], np.concatenate([[0.0], beta[1:]]), gx.ravel(), gy.ravel(), gld.ravel(), CLAMP)
        u, v = D["u"][m], D["v"][m]
        row[0].imshow(img, cmap="gray", alpha=0.5)
        row[0].scatter(u, v, c=np.expm1(before) * 100, s=6, cmap="RdBu_r", vmin=-10, vmax=10)
        row[0].set_title(f"{ds} · {D['names'][fi]}\nbefore (per-frame scale): median |res| {np.median(np.abs(before))*100:.2f} %", fontsize=10)
        im = row[1].imshow(np.expm1(field.reshape(gx.shape)) * 100, cmap="RdBu_r", vmin=-10, vmax=10, extent=[0, W, H, 0])
        row[1].set_title(f"fitted correction field ({cfg['model']}, λ {cfg['lam']}) [%]", fontsize=10)
        row[2].imshow(img, cmap="gray", alpha=0.5)
        row[2].scatter(u[te], v[te], c=np.expm1(after) * 100, s=8, cmap="RdBu_r", vmin=-10, vmax=10)
        row[2].set_title(f"after, HELD-OUT tiles only: median |res| {np.median(np.abs(after))*100:.2f} % "
                         f"(before on same pts {np.median(np.abs(t[te] - np.median(t[tr])))*100:.2f} %)", fontsize=10)
        for a in row:
            a.axis("off")
    fig.colorbar(im, ax=axs, fraction=0.012, label="residual / correction  %")
    fig.savefig(os.path.join(OUTD, f"correction_field_{ds}.png"), bbox_inches="tight"); plt.close(fig)
    print("wrote", ds)
