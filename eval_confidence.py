r"""
eval_confidence.py -- EVALUATION ONLY: does the reconstruction-internal confidence predict real depth error?

Uses the per-frame maps written by reconstruct.py --fusion weighted --save_confidence (grid = every 2nd pixel)
and ground-truth depth (TUM Kinect / synthetic renderer) sampled on the same grid. Aligned depth is multiplied by
the trajectory Sim(3) scale before comparison. Reports, pooled over all frames:
  - median / mean relative depth error per confidence quintile
  - per signal: Spearman correlation with error, AUROC for flagging "bad" pixels (error > 10 %)
  - how much the confidence-weighted mean error drops vs the unweighted mean
Usage: python eval_confidence.py --kind tum --gt_dir <seq> --recon_dir <weighted run>
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.stats import rankdata, spearmanr

from colmap_to_tum import name_to_timestamp
from eval_synth import umeyama
from eval_tum import associate, read_list


def auroc(score_bad, is_bad):
    r = rankdata(score_bad)
    n1 = is_bad.sum(); n0 = len(is_bad) - n1
    return float((r[is_bad].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["tum", "synth"], required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--bad", type=float, default=0.10)
    args = ap.parse_args()
    est = np.loadtxt(os.path.join(args.recon_dir, "trajectory_fused.txt"))
    if args.kind == "tum":
        gt = read_list(os.path.join(args.gt_dir, "groundtruth.txt"))
        a = associate(est[:, 0], [r[0] for r in gt], 0.02)
        keep = [i for i, t in enumerate(est[:, 0]) if t in a]
        G = np.array([[float(x) for x in gt[a[est[i, 0]]][1][:3]] for i in keep])
        dep = read_list(os.path.join(args.gt_dir, "depth.txt")); ad = associate(est[:, 0], [r[0] for r in dep], 0.02)
    else:
        g = np.loadtxt(os.path.join(args.gt_dir, "groundtruth.txt")); gi = {int(round(r[0])): r[1:4] for r in g}
        keep = [i for i, t in enumerate(est[:, 0]) if int(round(t)) in gi]
        G = np.array([gi[int(round(est[i, 0]))] for i in keep])
    s, _, _ = umeyama(est[keep, 1:4], G, True)

    cdir = os.path.join(args.recon_dir, "confidence")
    cols = {k: [] for k in ("err", "conf", "c_mv", "c_edge", "c_sparse", "c_frame", "reject")}
    for f in sorted(os.listdir(cdir)):
        t = name_to_timestamp(f)
        if args.kind == "tum":
            if t not in ad:
                continue
            gd = np.asarray(Image.open(os.path.join(args.gt_dir, dep[ad[t]][1][0])), np.float64)[::2, ::2] / 5000.0
        else:
            gd = np.load(os.path.join(args.gt_dir, "depth_gt", f"frame_{int(t):05d}.npy")).astype(np.float64)[::2, ::2]
        z = np.load(os.path.join(cdir, f))
        d = z["depth"].astype(np.float64) * s
        if d.shape != gd.shape:
            continue
        m = (d > 0) & (gd > 0.1) & (gd < 6.0)
        cols["err"].append(np.abs(d[m] / gd[m] - 1))
        for k in ("conf", "c_mv", "c_edge", "c_sparse", "c_frame"):
            cols[k].append(z[k].astype(np.float64)[m] if k in z.files else np.full(m.sum(), np.nan))
        cols["reject"].append(z["reject"][m])
    C = {k: np.concatenate(v) for k, v in cols.items()}
    rng = np.random.default_rng(0)
    idx = rng.choice(len(C["err"]), min(2_000_000, len(C["err"])), replace=False)
    C = {k: v[idx] for k, v in C.items()}
    bad = C["err"] > args.bad
    out = dict(recon_dir=args.recon_dir, pixels=int(len(C["err"])), bad_rate=float(bad.mean()),
               mean_err=float(C["err"].mean()), median_err=float(np.median(C["err"])))
    kept = C["conf"] > 0
    out["mean_err_conf_weighted"] = float(np.sum(C["conf"] * C["err"]) / np.sum(C["conf"]))
    out["dropped_frac"] = float((~kept).mean())
    out["bad_rate_in_dropped"] = float(bad[~kept].mean()) if (~kept).any() else None
    out["bad_rate_in_rejected"] = float(bad[C["reject"]].mean()) if C["reject"].any() else None
    q = np.quantile(C["conf"][kept], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    out["by_conf_quintile"] = []
    for lo, hi in zip(q[:-1], q[1:]):
        sel = kept & (C["conf"] >= lo) & (C["conf"] <= hi)
        out["by_conf_quintile"].append(dict(conf_range=[float(lo), float(hi)], median_err=float(np.median(C["err"][sel])),
                                            mean_err=float(C["err"][sel].mean()), bad_rate=float(bad[sel].mean())))
    out["signals"] = {}
    for k in ("conf", "c_mv", "c_edge", "c_sparse", "c_frame"):
        v = C[k]
        ok = np.isfinite(v)
        if ok.sum() < 1000 or np.nanstd(v) < 1e-6:
            continue
        out["signals"][k] = dict(spearman_with_err=float(spearmanr(v[ok], C["err"][ok])[0]),
                                 auroc_flag_bad=auroc(-v[ok], bad[ok]))
    json.dump(out, open(os.path.join(args.recon_dir, "confidence_eval.json"), "w"), indent=2)
    print(f"{args.recon_dir}: bad(>{args.bad:.0%}) rate {out['bad_rate']*100:.1f}% | mean err {out['mean_err']*100:.2f}% -> "
          f"confidence-weighted {out['mean_err_conf_weighted']*100:.2f}% | dropped {out['dropped_frac']*100:.1f}% of px "
          f"(bad rate there {out['bad_rate_in_dropped'] if out['bad_rate_in_dropped'] is None else round(out['bad_rate_in_dropped']*100, 1)}%)")
    print("   median err by confidence quintile (low->high): " +
          ", ".join(f"{b['median_err']*100:.2f}%" for b in out["by_conf_quintile"]))
    print("   signals: " + ", ".join(f"{k}: rho {v['spearman_with_err']:+.3f} AUROC {v['auroc_flag_bad']:.3f}" for k, v in out["signals"].items()))


if __name__ == "__main__":
    main()
