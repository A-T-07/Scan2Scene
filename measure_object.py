r"""
measure_object.py -- Independent metric sanity check (used ONLY after reconstruction; never for tuning).

Each observation = one frame + two pixel endpoints of the measured object (in the extracted-frame pixel grid).
Both endpoints are back-projected with the fused (aligned) depth of that frame (median of valid depths in a
small window, so endpoints should sit slightly INSIDE the object, not on its silhouette) and the fused-frame
camera pose. The length is reported
  raw   : in fused units (COLMAP rescaled to Depth Anything's median scale)
  TUM   : x k_TUM ("TUM-calibrated scale", from experiments\calibration, never re-fitted here)
and compared with the tape-measured value. Several observations (different frames) -> median and spread.

Usage:
  python measure_object.py --recon_dir <run> --colmap_dir <sparse\0> --true 2.03 --name "door height"
         --obs frame_00012.jpg 410 95 412 480 --obs frame_00150.jpg 300 60 305 470
"""
import argparse
import json
import os

import numpy as np

import Reconstruct as R
from eval_synth import quat_to_R


def k_tum(root):
    ks = [json.load(open(os.path.join(root, "experiments", "calibration", f"calib_from_{s}.json")))["metric_scale"] for s in ("fr2", "fr1")]
    return float(np.exp(np.mean(np.log(ks))))


def point(depth, rays, u, v, win=3):
    gu, gv = int(round(u / 2)), int(round(v / 2))   # aligned depth is half resolution
    w = depth[max(0, gv - win):gv + win + 1, max(0, gu - win):gu + win + 1]
    w = w[w > 0]
    if len(w) < 3:
        return None
    z = float(np.median(w))
    x, y = rays[min(gv, rays.shape[0] - 1), min(gu, rays.shape[1] - 1)]
    return np.array([x * z, y * z, z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--colmap_dir", required=True)
    ap.add_argument("--true", type=float, required=True, help="tape-measured length in metres")
    ap.add_argument("--name", default="object")
    ap.add_argument("--obs", nargs=5, action="append", required=True, metavar=("FRAME", "U1", "V1", "U2", "V2"))
    a = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    kt = k_tum(root)
    cam = next(iter(R.read_colmap_cameras(os.path.join(a.colmap_dir, "cameras.txt")).values()))
    rays = R.pixel_rays(cam)[::2, ::2]
    tr = {round(r[0]): (r[1:4], quat_to_R(*r[4:8])) for r in np.loadtxt(os.path.join(a.recon_dir, "trajectory_fused.txt"))}
    lens = []
    for f, u1, v1, u2, v2 in a.obs:
        d = np.load(os.path.join(a.recon_dir, "aligned_depth", os.path.splitext(f)[0] + ".npy")).astype(np.float64)
        p1, p2 = point(d, rays, float(u1), float(v1)), point(d, rays, float(u2), float(v2))
        if p1 is None or p2 is None:
            print(f"  {f}: endpoint has no valid depth, skipped"); continue
        lens.append(float(np.linalg.norm(p1 - p2)))       # same camera -> pose cancels
        print(f"  {f}: {lens[-1]:.3f} fused units")
    L = np.array(lens)
    raw, cal = float(np.median(L)), float(np.median(L) * kt)
    res = dict(label="Independent metric sanity check", object=a.name, true_m=a.true, observations=len(L),
               raw_fused_units=raw, raw_rel_error_pct=abs(raw - a.true) / a.true * 100,
               k_TUM=kt, tum_calibrated_m=cal, tum_rel_error_pct=abs(cal - a.true) / a.true * 100,
               spread_rel_pct=float((L.max() - L.min()) / np.median(L) * 100) if len(L) > 1 else None)
    json.dump(res, open(os.path.join(a.recon_dir, "metric_sanity_check.json"), "w"), indent=2)
    print(f"Independent metric sanity check ({a.name}): true {a.true:.3f} m | raw {raw:.3f} (err {res['raw_rel_error_pct']:.1f}%) | "
          f"TUM-calibrated {cal:.3f} m (err {res['tum_rel_error_pct']:.1f}%) | {len(L)} obs, spread {res['spread_rel_pct']}")


if __name__ == "__main__":
    main()
