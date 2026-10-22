r"""
calibrate_scale.py -- derive a global metric-scale calibration factor k from CALIBRATION sequences.

For each calibration sequence (an uncalibrated reconstruct.py output + that sequence's ground-truth
trajectory) the Sim(3) Umeyama scale s_i = GT metres per fused unit is computed from the camera centres
(the same quantity evo reports as "scale correction"). k is the geometric mean of the s_i.

Use k on a DIFFERENT (test) sequence:
    python reconstruct.py ... --metric_scale_file calib.json

Never include the test sequence here — that would tune the test result on its own ground truth.

Usage:
  python calibrate_scale.py --seq experiments\tum_fr2_xyz\corrected_scale rgbd_dataset_freiburg2_xyz\...\groundtruth.txt
                            [--seq <recon_dir> <groundtruth.txt> ...] --out calib_fr2.json
"""

import argparse
import json
import os

import numpy as np

from eval_synth import umeyama
from eval_tum import associate, read_list


def sim3_scale(traj_path, gt_path):
    est = np.loadtxt(traj_path)
    gt = read_list(gt_path)
    a = associate(est[:, 0], [r[0] for r in gt], 0.02)
    keep = [i for i, t in enumerate(est[:, 0]) if t in a]
    if len(keep) < 3:
        raise SystemExit(f"!! too few associated poses for {traj_path}")
    E = est[keep, 1:4]
    G = np.array([[float(x) for x in gt[a[est[i, 0]]][1][:3]] for i in keep])
    s, _, _ = umeyama(E, G, True)
    return float(s), len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs=2, action="append", required=True, metavar=("RECON_DIR", "GROUNDTRUTH"),
                    help="uncalibrated reconstruct.py output folder + its ground-truth trajectory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per = []
    for rd, gt in args.seq:
        summ = json.load(open(os.path.join(rd, "summary.json")))
        if abs(summ.get("metric_scale_applied", 1.0) - 1.0) > 1e-9:
            raise SystemExit(f"!! {rd} was already calibrated (k={summ['metric_scale_applied']}); use an uncalibrated run")
        s, n = sim3_scale(os.path.join(rd, "trajectory_fused.txt"), gt)
        per.append(dict(recon_dir=os.path.abspath(rd), groundtruth=os.path.abspath(gt), sim3_scale=s, poses=n))
        print(f"{rd}: s = {s:.4f} GT m / fused unit ({n} poses)")
    k = float(np.exp(np.mean([np.log(p["sim3_scale"]) for p in per])))
    json.dump(dict(metric_scale=k, method="geometric mean of Sim(3) camera-centre scales", sequences=per),
              open(args.out, "w"), indent=2)
    print(f"metric_scale k = {k:.4f}  -> {args.out}")


if __name__ == "__main__":
    main()
