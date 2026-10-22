r"""
revisit_consistency.py -- GT-free loop-closure / doubling metric.

Back-projects the fused (aligned) depth maps saved by reconstruct.py --save_aligned_depth (half resolution)
with the fused-frame poses, for two frame segments, and measures how far the SAME surfaces land from each other:
  revisit : first K frames  vs  last K frames            (start vs end of the loop)
  control : two adjacent segments in the middle of the sequence (same time, no drift) -> noise floor
For each pair: overlap fraction (A points with a B point within 0.30 units) and the median / p75 nearest-neighbour
distance inside the overlap (units of the fused frame; x k_TUM for "TUM-calibrated" cm).

Usage: python revisit_consistency.py --recon_dir experiments\phone_real_v2\baseline --colmap_dir <sparse\0> [--k 15]
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

import Reconstruct as R
from eval_synth import quat_to_R


def segment_points(recon_dir, cams, names, traj, idx, voxel=0.01):
    ad = os.path.join(recon_dir, "aligned_depth")
    pts = []
    for i in idx:
        n = names[i]
        p = os.path.join(ad, os.path.splitext(n)[0] + ".npy")
        if not os.path.exists(p):
            continue
        d = np.load(p).astype(np.float64)
        rays = R.pixel_rays(cams)[::2, ::2]
        rays = rays[:d.shape[0], :d.shape[1]]
        m = d > 0
        z = d[m]; xy = rays[m]
        X = np.stack([xy[:, 0] * z, xy[:, 1] * z, z], 1)
        C, Rc = traj[i]
        pts.append(X @ Rc.T + C)
    P = np.concatenate(pts)
    k = np.unique(np.floor(P / voxel).astype(np.int64), axis=0, return_index=True)[1]
    return P[k]


def pair_stats(A, B, radius=0.30):
    d, _ = cKDTree(B).query(A, distance_upper_bound=radius)
    ov = np.isfinite(d)
    return dict(overlap_frac=float(ov.mean()), median=float(np.median(d[ov])) if ov.any() else None,
                p75=float(np.percentile(d[ov], 75)) if ov.any() else None, n=int(ov.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--colmap_dir", required=True)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--k_tum", type=float, default=None)
    a = ap.parse_args()
    cams = next(iter(R.read_colmap_cameras(os.path.join(a.colmap_dir, "cameras.txt")).values()))
    poses = R.read_colmap_images(os.path.join(a.colmap_dir, "images.txt"))
    names = sorted(poses, key=R.name_to_timestamp)
    tr = np.loadtxt(os.path.join(a.recon_dir, "trajectory_fused.txt"))
    traj = [(r[1:4], quat_to_R(*r[4:8])) for r in tr]
    n, k = len(names), a.k
    mid = n // 2
    A = segment_points(a.recon_dir, cams, names, traj, range(0, k))
    B = segment_points(a.recon_dir, cams, names, traj, range(n - k, n))
    C1 = segment_points(a.recon_dir, cams, names, traj, range(mid - k, mid))
    C2 = segment_points(a.recon_dir, cams, names, traj, range(mid, mid + k))
    starts = np.array([t[0] for t in traj])
    res = dict(recon_dir=a.recon_dir, frames=n, k=k,
               start_end_camera_distance=float(np.linalg.norm(starts[-1] - starts[0])),
               path_length=float(np.linalg.norm(np.diff(starts, axis=0), axis=1).sum()),
               revisit_start_vs_end=pair_stats(A, B), control_adjacent_mid=pair_stats(C1, C2))
    res["closure_ratio"] = res["start_end_camera_distance"] / res["path_length"]
    if a.k_tum:
        res["k_TUM"] = a.k_tum
    json.dump(res, open(os.path.join(a.recon_dir, "revisit_consistency.json"), "w"), indent=2)
    rv, ct = res["revisit_start_vs_end"], res["control_adjacent_mid"]
    kk = a.k_tum or 1.0
    print(f"{a.recon_dir}: start-end cam dist {res['start_end_camera_distance']:.3f} (path {res['path_length']:.2f}, ratio {res['closure_ratio']*100:.1f}%) | "
          f"revisit overlap {rv['overlap_frac']*100:.0f}% median {rv['median']*100*kk:.2f} cm | control overlap {ct['overlap_frac']*100:.0f}% "
          f"median {ct['median']*100*kk:.2f} cm  ({'TUM-calibrated' if a.k_tum else 'fused units x100'})")


if __name__ == "__main__":
    main()
