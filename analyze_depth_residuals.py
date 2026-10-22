r"""
analyze_depth_residuals.py -- Step 1 of spatial depth correction: look at the residual field BEFORE modelling it.

For every frame, the COLMAP sparse points observed in it are projected into the image and compared with the
Depth Anything prediction at the same pixel, AFTER the existing per-frame scale alignment:

    r = z_colmap_fused / (a_i * d_pred)          (1.0 = perfect; log r is analysed)

Only COLMAP sparse geometry is used here -- no ground-truth depth.

Outputs (experiments\spatial_depth\):
  <ds>_sparse.npz                  cached correspondences (reused by fit_spatial_correction.py)
  residual_pooled_maps.png         median r binned over image position, per dataset
  residual_depth_dependence.png    median r vs relative depth and vs absolute depth
  residual_frames_<ds>.png         per-frame heatmaps on representative frames
  residual_perframe_coeffs.png     per-frame linear (x, y, log-depth) coefficients over the sequence
  residual_summary.json            numeric summary (in-sample robust spread of log r under models A-D)
"""

import csv
import json
import os
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import Reconstruct as R

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "experiments", "spatial_depth")

DATASETS = {
    "fr2_xyz": dict(frames=r"frames_subset", colmap=r"colmap_ws3\sparse\0",
                    recon=r"experiments\tum_fr2_xyz\corrected_scale"),
    "fr1_desk": dict(frames=r"experiments\tum_freiburg1_desk\frames", colmap=r"experiments\tum_freiburg1_desk\colmap_ws\sparse\0",
                     recon=r"experiments\tum_freiburg1_desk\corrected_scale"),
    "synth_room2": dict(frames=r"synth_room2\run\frames", colmap=r"synth_room2\run\colmap_ws\sparse\0",
                        recon=r"synth_room2\recon_scale"),
}


def collect(ds):
    """Sparse correspondences for every frame of a dataset (cached as npz)."""
    cfg = DATASETS[ds]
    cache_npz = os.path.join(OUT, f"{ds}_sparse.npz")
    if os.path.exists(cache_npz):
        return dict(np.load(cache_npz, allow_pickle=True))
    rd = os.path.join(ROOT, cfg["recon"])
    summ = json.load(open(os.path.join(rd, "summary.json")))
    a = summ["args"]
    args = types.SimpleNamespace(max_reproj_err=a["max_reproj_err"], min_track=a["min_track"])
    S = summ["global_scale_S_colmap_units_per_pred_metre"]
    depth_cache = os.path.join(ROOT, a["depth_cache"]) if a.get("depth_cache") else os.path.join(rd, "depth_pred")
    if not os.path.isabs(depth_cache):
        depth_cache = os.path.join(ROOT, depth_cache)
    diag = {r["name"]: r for r in csv.DictReader(open(os.path.join(rd, "alignment_diagnostics.csv")))}
    cams = R.read_colmap_cameras(os.path.join(ROOT, cfg["colmap"], "cameras.txt"))
    poses = R.read_colmap_images(os.path.join(ROOT, cfg["colmap"], "images.txt"))
    idx, xyz, err, trk = R.read_colmap_points3d(os.path.join(ROOT, cfg["colmap"], "points3D.txt"))
    names = sorted(poses, key=lambda n: (R.name_to_timestamp(n), n))
    cols = {k: [] for k in ("frame", "u", "v", "x", "y", "d", "z", "a", "dmed")}
    frame_names = []
    for fi, n in enumerate(names):
        p = os.path.join(depth_cache, os.path.splitext(n)[0] + ".npy")
        if not os.path.exists(p) or n not in diag or diag[n]["status"] != "ok":
            continue
        depth = np.load(p)
        cam = cams[poses[n]["camera_id"]]
        mask = R.valid_depth_mask(depth, a["depth_min"], a["depth_max"], a["edge_thresh"])
        d, z, u, v = R.sparse_correspondences_uv(poses[n], cam, idx, xyz, err, trk, depth, mask, args)
        if len(d) < 10:
            continue
        frame_names.append(n)
        k = len(frame_names) - 1
        cols["frame"].append(np.full(len(d), k)); cols["u"].append(u); cols["v"].append(v)
        cols["x"].append((u + 0.5 - cam["cx"]) / cam["fx"]); cols["y"].append((v + 0.5 - cam["cy"]) / cam["fy"])
        cols["d"].append(d); cols["z"].append(z / S)
        cols["a"].append(np.full(len(d), float(diag[n]["a"])))
        cols["dmed"].append(np.full(len(d), float(np.median(depth[mask]))))
    data = {k: np.concatenate(v) for k, v in cols.items()}
    data["names"] = np.array(frame_names)
    cam0 = next(iter(cams.values()))
    data["W"], data["H"] = cam0["width"], cam0["height"]
    data["fx"], data["fy"], data["cx"], data["cy"] = cam0["fx"], cam0["fy"], cam0["cx"], cam0["cy"]
    data["frames_dir"] = os.path.join(ROOT, cfg["frames"])
    os.makedirs(OUT, exist_ok=True)
    np.savez(cache_npz, **data)
    return data


def robust_inliers(logr, frame, k=3.0):
    """Per-frame MAD rejection of gross sparse outliers (same spirit as the scale fit)."""
    keep = np.zeros(len(logr), bool)
    for f in np.unique(frame):
        m = frame == f
        r = logr[m]
        med = np.median(r); mad = 1.4826 * np.median(np.abs(r - med)) + 1e-6
        keep[m] = np.abs(r - med) <= max(k * mad, np.log1p(0.10))
    return keep


def in_sample_models(D):
    """Robust spread of log r after removing per-frame intercept + shared shape terms (models A-D), in-sample."""
    logr = np.log(D["z"] / (D["a"] * D["d"]))
    keep = robust_inliers(logr, D["frame"])
    f = D["frame"][keep]; y = logr[keep]
    x, yy = D["x"][keep], D["y"][keep]
    ld = np.log(D["d"][keep] / D["dmed"][keep])
    feats = {
        "A_global_scale": np.zeros((len(y), 0)),
        "B_linear_xy": np.stack([x, yy], 1),
        "C_poly2_xy": np.stack([x, yy, x * x, x * yy, yy * yy], 1),
        "D_poly2_xy_logdepth": np.stack([x, yy, x * x, x * yy, yy * yy, ld], 1),
        "E_logdepth_only": np.stack([ld], 1),
    }
    out = {}
    for name, X in feats.items():
        # demean per frame (per-frame intercept is always free = per-frame scale)
        yc = y.copy(); Xc = X.copy()
        for fr in np.unique(f):
            m = f == fr
            yc[m] -= y[m].mean()
            if X.shape[1]:
                Xc[m] -= X[m].mean(0)
        if X.shape[1]:
            beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
            res = yc - Xc @ beta
        else:
            beta = np.zeros(0); res = yc
        out[name] = dict(robust_std_log=float(1.4826 * np.median(np.abs(res - np.median(res)))),
                         mean_abs_pct=float(np.mean(np.abs(np.expm1(res))) * 100), coef=[float(b) for b in beta])
    return out, keep


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {ds: collect(ds) for ds in DATASETS}
    summary = {}
    plt.rcParams.update({"font.size": 11})

    # ---- pooled image-position maps ----
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8), dpi=140)
    for ax, (ds, D) in zip(axs, data.items()):
        logr = np.log(D["z"] / (D["a"] * D["d"]))
        keep = robust_inliers(logr, D["frame"])
        W, H = int(D["W"]), int(D["H"])
        nb = (16, 12)
        s, xe, ye = np.histogram2d(D["u"][keep], D["v"][keep], bins=nb, range=[[0, W], [0, H]], weights=logr[keep])
        c, _, _ = np.histogram2d(D["u"][keep], D["v"][keep], bins=nb, range=[[0, W], [0, H]])
        # binned median
        med = np.full(nb, np.nan)
        bu = np.clip((D["u"][keep] / W * nb[0]).astype(int), 0, nb[0] - 1)
        bv = np.clip((D["v"][keep] / H * nb[1]).astype(int), 0, nb[1] - 1)
        for i in range(nb[0]):
            for j in range(nb[1]):
                m = (bu == i) & (bv == j)
                if m.sum() >= 30:
                    med[i, j] = np.median(logr[keep][m])
        im = ax.imshow(np.expm1(med.T) * 100, extent=[0, W, H, 0], cmap="RdBu_r", vmin=-8, vmax=8)
        ax.set_title(f"{ds}: median residual (z_colmap / a·d − 1), %\n{int(keep.sum()):,} sparse pts, {len(D['names'])} frames")
        ax.set_xlabel("u [px]"); ax.set_ylabel("v [px]")
        plt.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "residual_pooled_maps.png")); plt.close(fig)

    # ---- depth dependence ----
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), dpi=140)
    for ds, D in data.items():
        logr = np.log(D["z"] / (D["a"] * D["d"]))
        keep = robust_inliers(logr, D["frame"])
        for ax, xv, lab in ((axs[0], np.log(D["d"] / D["dmed"]), "log(d / frame median depth)"),
                            (axs[1], np.log(D["a"] * D["d"]), "log(aligned depth)  [fused units]")):
            xk, yk = xv[keep], logr[keep]
            bins = np.quantile(xk, np.linspace(0, 1, 21))
            cen, med, lo, hi = [], [], [], []
            for b0, b1 in zip(bins[:-1], bins[1:]):
                m = (xk >= b0) & (xk < b1)
                if m.sum() > 30:
                    cen.append(np.median(xk[m])); med.append(np.median(yk[m]) * 100)
                    lo.append(np.percentile(yk[m], 25) * 100); hi.append(np.percentile(yk[m], 75) * 100)
            ax.plot(cen, med, "-o", ms=3, label=ds); ax.fill_between(cen, lo, hi, alpha=0.15)
            ax.set_xlabel(lab); ax.set_ylabel("median log residual  [%]"); ax.axhline(0, color="k", lw=0.8)
    axs[0].legend(); axs[0].set_title("depth-dependent bias (relative depth)"); axs[1].set_title("depth-dependent bias (absolute depth)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "residual_depth_dependence.png")); plt.close(fig)

    # ---- per-frame heatmaps on representative frames ----
    for ds in ("fr2_xyz", "fr1_desk", "synth_room2"):
        D = data[ds]
        nF = len(D["names"])
        pick = [int(nF * q) for q in (0.1, 0.4, 0.7, 0.95)]
        fig, axs = plt.subplots(1, 4, figsize=(20, 4.2), dpi=130)
        for ax, fi in zip(axs, pick):
            m = D["frame"] == fi
            r = D["z"][m] / (D["a"][m] * D["d"][m]) - 1
            img = np.asarray(Image.open(os.path.join(str(D["frames_dir"]), str(D["names"][fi]))).convert("L"))
            ax.imshow(img, cmap="gray", alpha=0.55)
            sc = ax.scatter(D["u"][m], D["v"][m], c=r * 100, s=7, cmap="RdBu_r", vmin=-10, vmax=10)
            ax.set_title(f"{ds} · {D['names'][fi]}\n{m.sum()} pts, median |r| {np.median(np.abs(r))*100:.1f} %", fontsize=9)
            ax.axis("off")
        plt.colorbar(sc, ax=axs, fraction=0.012, label="residual %")
        fig.savefig(os.path.join(OUT, f"residual_frames_{ds}.png"), bbox_inches="tight"); plt.close(fig)

    # ---- per-frame linear coefficients (is there a FIXED pattern?) ----
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.2), dpi=140)
    for ds, D in data.items():
        logr = np.log(D["z"] / (D["a"] * D["d"]))
        keep = robust_inliers(logr, D["frame"])
        coefs = []
        for fi in range(len(D["names"])):
            m = keep & (D["frame"] == fi)
            if m.sum() < 40:
                coefs.append([np.nan] * 3); continue
            X = np.stack([np.ones(m.sum()), D["x"][m], D["y"][m], np.log(D["d"][m] / D["dmed"][m])], 1)
            b = np.linalg.lstsq(X, logr[m], rcond=None)[0]
            coefs.append(b[1:])
        coefs = np.array(coefs) * 100
        t = np.linspace(0, 1, len(coefs))
        for k, ax in enumerate(axs):
            ax.plot(t, coefs[:, k], lw=1.2, label=f"{ds} (median {np.nanmedian(coefs[:, k]):+.1f}, IQR {np.nanpercentile(coefs[:, k], 75) - np.nanpercentile(coefs[:, k], 25):.1f})")
        summary.setdefault(ds, {})["perframe_linear_coef_median_pct"] = [float(v) for v in np.nanmedian(coefs, 0)]
        summary[ds]["perframe_linear_coef_iqr_pct"] = [float(v) for v in (np.nanpercentile(coefs, 75, 0) - np.nanpercentile(coefs, 25, 0))]
    for ax, t_ in zip(axs, ("x-slope  [% per unit x/f]", "y-slope  [% per unit y/f]", "log-depth slope  [% per unit log d]")):
        ax.set_title(t_); ax.axhline(0, color="k", lw=0.8); ax.legend(fontsize=8); ax.set_xlabel("normalised sequence time")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "residual_perframe_coeffs.png")); plt.close(fig)

    for ds, D in data.items():
        models, keep = in_sample_models(D)
        summary[ds]["n_points"] = int(len(D["d"])); summary[ds]["n_inliers"] = int(keep.sum())
        summary[ds]["n_frames"] = int(len(D["names"]))
        summary[ds]["in_sample_models"] = models
    json.dump(summary, open(os.path.join(OUT, "residual_summary.json"), "w"), indent=2)
    for ds, s in summary.items():
        print(f"== {ds}: {s['n_frames']} frames, {s['n_points']} pts ({s['n_inliers']} inliers)")
        print("   per-frame linear coef medians [x, y, logd] %:", [round(v, 1) for v in s["perframe_linear_coef_median_pct"]],
              " IQR:", [round(v, 1) for v in s["perframe_linear_coef_iqr_pct"]])
        for m, v in s["in_sample_models"].items():
            print(f"   {m:22s} robust std(log r) {v['robust_std_log']*100:5.2f} %   mean|r| {v['mean_abs_pct']:5.2f} %   coef {[round(c, 3) for c in v['coef']]}")


if __name__ == "__main__":
    main()
