r"""
eval_aligned_depth.py -- evaluate the depth maps that were ACTUALLY fused (reconstruct.py --save_aligned_depth)
against ground-truth depth (TUM Kinect or synthetic renderer). Evaluation only -- never used for fitting.

Aligned depth is in fused units; it is multiplied by the Sim(3) scale s* of the camera trajectory (so global
scale is removed) before comparison. Reports AbsRel, delta<1.25, and per-frame stability (std of per-frame
AbsRel and of per-frame log bias).

Usage:
  python eval_aligned_depth.py --kind tum   --gt_dir <tum seq> --recon_dir <run with aligned_depth\>
  python eval_aligned_depth.py --kind synth --gt_dir synth_room2 --recon_dir <run>
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from colmap_to_tum import name_to_timestamp
from eval_synth import umeyama
from eval_tum import associate, read_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["tum", "synth"], required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--recon_dir", required=True)
    args = ap.parse_args()

    est = np.loadtxt(os.path.join(args.recon_dir, "trajectory_fused.txt"))
    if args.kind == "tum":
        gt = read_list(os.path.join(args.gt_dir, "groundtruth.txt"))
        a = associate(est[:, 0], [r[0] for r in gt], 0.02)
        keep = [i for i, t in enumerate(est[:, 0]) if t in a]
        G = np.array([[float(x) for x in gt[a[est[i, 0]]][1][:3]] for i in keep])
        dep = read_list(os.path.join(args.gt_dir, "depth.txt"))
        ad = associate(est[:, 0], [r[0] for r in dep], 0.02)
    else:
        g = np.loadtxt(os.path.join(args.gt_dir, "groundtruth.txt"))
        gi = {int(round(r[0])): r[1:4] for r in g}
        keep = [i for i, t in enumerate(est[:, 0]) if int(round(t)) in gi]
        G = np.array([gi[int(round(est[i, 0]))] for i in keep])
    s, _, _ = umeyama(est[keep, 1:4], G, True)

    ad_dir = os.path.join(args.recon_dir, "aligned_depth")
    absrel, d1, bias, n_px = [], [], [], 0
    for f in sorted(os.listdir(ad_dir)):
        t = name_to_timestamp(f)
        if args.kind == "tum":
            if t not in ad:
                continue
            g_ = np.asarray(Image.open(os.path.join(args.gt_dir, dep[ad[t]][1][0])), np.float64)[::2, ::2] / 5000.0
        else:
            g_ = np.load(os.path.join(args.gt_dir, "depth_gt", f"frame_{int(t):05d}.npy")).astype(np.float64)[::2, ::2]
        p = np.load(os.path.join(ad_dir, f)).astype(np.float64) * s
        m = (p > 0) & (g_ > 0.1) & (g_ < 6.0) & np.isfinite(p)
        if m.sum() < 500:
            continue
        r = p[m] / g_[m]
        absrel.append(np.mean(np.abs(r - 1))); d1.append(np.mean(np.maximum(r, 1 / r) < 1.25))
        bias.append(np.median(np.log(r))); n_px += int(m.sum())
    absrel, d1, bias = map(np.array, (absrel, d1, bias))
    res = dict(recon_dir=args.recon_dir, sim3_scale=float(s), frames=len(absrel), pixels=n_px,
               absrel_mean=float(absrel.mean()), absrel_median_frame=float(np.median(absrel)),
               absrel_frame_std=float(absrel.std()), delta1=float(d1.mean()),
               log_bias_frame_std=float(bias.std()))
    json.dump(res, open(os.path.join(args.recon_dir, "depth_eval.json"), "w"), indent=2)
    print(f"{args.recon_dir}: AbsRel {res['absrel_mean']:.4f} (per-frame std {res['absrel_frame_std']:.4f}), "
          f"d1 {res['delta1']:.3f}, per-frame log-bias std {res['log_bias_frame_std']:.4f}, frames {len(absrel)}")


if __name__ == "__main__":
    main()
