r"""
fit_spatial_correction.py -- choose / learn the spatial depth correction on a CALIBRATION sequence.

Only COLMAP sparse geometry + predicted depth are used (cached by analyze_depth_residuals.py as
experiments\spatial_depth\<ds>_sparse.npz). No ground-truth depth, no dense-cloud error.

Target per sparse point: t = log(z_colmap_fused / d_pred).

Validation = held-out sparse points in UNSEEN IMAGE REGIONS: each frame's image is cut into 4x4 tiles, tiles
are assigned to 4 folds; the model is fitted on 3 folds' tiles and scored on the 4th (median |residual| in %).
  perframe : all coefficients fitted per frame from the training tiles of that frame
  transfer : shape coefficients learned on OTHER frames (4 contiguous frame blocks), only the per-frame
             intercept is fitted on the training tiles of the evaluated frame
Model choice: lowest held-out error; a simpler model wins if within 2 % (relative) of the best.

Writes experiments\spatial_depth\calib_<ds>_{perframe,transfer}.json and cv_<ds>.json.
Usage:  python fit_spatial_correction.py --calib fr2_xyz
"""

import argparse
import json
import os

import numpy as np

import Reconstruct as R

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "experiments", "spatial_depth")
MODELS = ["const", "lin_xy", "logd", "lin_xy_logd", "poly2_xy", "poly2_xy_logd"]  # increasing capacity
LAMBDAS = [0.01, 0.1, 1.0]
CLAMP = 0.2
INLIER_TOL = 0.10
MIN_PTS = 60


def load(ds):
    D = dict(np.load(os.path.join(OUT, f"{ds}_sparse.npz"), allow_pickle=True))
    D["t"] = np.log(D["z"] / D["d"])
    D["ld"] = np.log(D["d"] / D["dmed"])
    W, H = float(D["W"]), float(D["H"])
    tile = (np.clip((D["u"] / W * 4).astype(int), 0, 3) * 4 + np.clip((D["v"] / H * 4).astype(int), 0, 3))
    rng = np.random.default_rng(0)
    fold_of_tile = {}
    D["fold"] = np.zeros(len(tile), int)
    for f in np.unique(D["frame"]):
        perm = rng.permutation(16) % 4           # 4 tiles per fold, different per frame
        m = D["frame"] == f
        D["fold"][m] = perm[tile[m]]
    return D


def predict(model, beta, x, y, ld, ld_range):
    ld = np.clip(ld, *ld_range)
    return R.spatial_log_factor(model, beta, x, y, ld, CLAMP)


def learn_shared_shape(D, frames, model, lam):
    """Pooled shape with per-frame intercepts (per-frame demeaning), robust ridge."""
    Fs, ts = [], []
    for f in frames:
        m = D["frame"] == f
        if m.sum() < MIN_PTS:
            continue
        t = D["t"][m]; inl = R.spatial_inliers(t, INLIER_TOL)
        F = R.spatial_features(model, D["x"][m][inl], D["y"][m][inl], D["ld"][m][inl])
        Fs.append(F - F.mean(0)); ts.append(t[inl] - t[inl].mean())
    F = np.concatenate(Fs); t = np.concatenate(ts)
    if F.shape[1] == 0:
        return np.zeros(0)
    return R.robust_ridge(F, t, lam)[1:]


def cv_error(D, model, lam, mode):
    errs = []
    frames = np.unique(D["frame"])
    blocks = np.array_split(frames, 4)
    shapes = {}
    if mode == "transfer":
        for bi, blk in enumerate(blocks):
            train_frames = np.concatenate([b for bj, b in enumerate(blocks) if bj != bi])
            shapes[bi] = learn_shared_shape(D, train_frames, model, lam)
    for bi, blk in enumerate(blocks):
        for f in blk:
            m = D["frame"] == f
            for k in range(4):
                tr = m & (D["fold"] != k); te = m & (D["fold"] == k)
                if tr.sum() < MIN_PTS or te.sum() < 5:
                    continue
                ld_tr = D["ld"][tr]
                beta, _ = R.fit_spatial_frame(model, D["x"][tr], D["y"][tr], ld_tr, D["t"][tr], lam, INLIER_TOL,
                                              shapes[bi] if mode == "transfer" else None)
                pr = predict(model, beta, D["x"][te], D["y"][te], D["ld"][te],
                             (np.percentile(ld_tr, 1), np.percentile(ld_tr, 99)))
                errs.append(np.abs(D["t"][te] - pr))
    e = np.concatenate(errs)
    return dict(median_abs_pct=float(np.median(e) * 100), mean_abs_pct=float(np.mean(np.minimum(e, 0.5)) * 100), n=int(len(e)))


def choose(rows):
    best = min(r["median_abs_pct"] for r in rows)
    for r in rows:  # rows are ordered simple -> complex, small lambda last within a model? (ordered by capacity)
        if r["median_abs_pct"] <= best * 1.02:
            return r
    return rows[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, choices=["fr2_xyz", "fr1_desk", "synth_room2"])
    ap.add_argument("--also_score", nargs="*", default=[],
                    help="Other datasets on which to REPORT (not select) held-out sparse error of the chosen configs")
    args = ap.parse_args()
    D = load(args.calib)
    table = []
    for mode in ("perframe", "transfer"):
        for model in MODELS:
            lams = [0.0] if model == "const" else sorted(LAMBDAS, reverse=True)   # strongest regularisation first
            for lam in lams:
                if mode == "transfer" and model == "const":
                    continue
                r = dict(mode=mode, model=model, lam=lam, **cv_error(D, model, lam, mode))
                table.append(r)
                print(f"{args.calib} {mode:9s} {model:15s} lam {lam:<5} held-out median |res| {r['median_abs_pct']:.3f} %  "
                      f"(mean {r['mean_abs_pct']:.3f} %, n {r['n']})")
    base = next(r for r in table if r["model"] == "const")
    out = {}
    for mode in ("perframe", "transfer"):
        rows = [base] + [r for r in table if r["mode"] == mode and r["model"] != "const"]
        ch = choose(rows)
        cfg = dict(mode=mode, model=ch["model"], lam=ch["lam"], clamp=CLAMP, calibration_sequence=args.calib,
                   heldout_median_abs_pct=ch["median_abs_pct"], baseline_const_heldout_median_abs_pct=base["median_abs_pct"],
                   shared_shape=None)
        if mode == "transfer" and ch["model"] != "const":
            cfg["shared_shape"] = [float(v) for v in learn_shared_shape(D, np.unique(D["frame"]), ch["model"], ch["lam"])]
        if ch["model"] == "const":
            cfg["mode"] = "perframe"  # nothing to transfer: equivalent to per-frame scale
        json.dump(cfg, open(os.path.join(OUT, f"calib_{args.calib}_{mode}.json"), "w"), indent=2)
        out[mode] = cfg
        print(f"CHOSEN {mode}: {ch['model']} lam {ch['lam']} -> held-out {ch['median_abs_pct']:.3f} % "
              f"(per-frame scale baseline {base['median_abs_pct']:.3f} %)")
    # report-only: the chosen configs scored on other datasets' sparse points (no selection happens here)
    report = {}
    for other in args.also_score:
        Do = load(other)
        report[other] = {"const": cv_error(Do, "const", 0.0, "perframe")}
        for mode, cfg in out.items():
            if cfg["model"] == "const":
                continue
            if mode == "transfer":
                # fixed shape from the calibration sequence, only the per-frame intercept on the test frames
                errs = []
                for f in np.unique(Do["frame"]):
                    m = Do["frame"] == f
                    for k in range(4):
                        tr = m & (Do["fold"] != k); te = m & (Do["fold"] == k)
                        if tr.sum() < MIN_PTS or te.sum() < 5:
                            continue
                        beta, _ = R.fit_spatial_frame(cfg["model"], Do["x"][tr], Do["y"][tr], Do["ld"][tr], Do["t"][tr],
                                                      cfg["lam"], INLIER_TOL, cfg["shared_shape"])
                        pr = predict(cfg["model"], beta, Do["x"][te], Do["y"][te], Do["ld"][te],
                                     (np.percentile(Do["ld"][tr], 1), np.percentile(Do["ld"][tr], 99)))
                        errs.append(np.abs(Do["t"][te] - pr))
                e = np.concatenate(errs)
                report[other][mode] = dict(median_abs_pct=float(np.median(e) * 100), n=int(len(e)))
            else:
                report[other][mode] = cv_error(Do, cfg["model"], cfg["lam"], "perframe")
        print(f"REPORT on {other} (sparse held-out, not used for selection): " +
              ", ".join(f"{k} {v['median_abs_pct']:.3f} %" for k, v in report[other].items()))
    json.dump(dict(calibration=args.calib, table=table, chosen=out, report_on_other=report),
              open(os.path.join(OUT, f"cv_{args.calib}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
