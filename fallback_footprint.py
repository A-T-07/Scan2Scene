r"""
fallback_footprint.py -- EVALUATION ONLY. Footprint of the frames that fell back to the global scale
(status fallback_scale*, alignment_diagnostics.csv) in the final cloud: final points whose nearest raw source
point (2 cm) comes only from those frames, their GT distance, and cloud accuracy with / without them.
Same method as experiments\tum_freiburg1_room\eval_extra.py, generalised to TUM + synthetic.
Usage: python fallback_footprint.py --kind tum|synth --gt_dir <gt> --recon_dir <run> --colmap_dir <sparse\0>
"""
import argparse
import csv
import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import Reconstruct as R
from colmap_to_tum import name_to_timestamp
from eval_synth import quat_to_R
from eval_tum import read_list
from surface_metrics import gt_cloud_synth, gt_cloud_tum, load_alignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["tum", "synth"], required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--colmap_dir", required=True)
    a = ap.parse_args()
    diag = list(csv.DictReader(open(os.path.join(a.recon_dir, "alignment_diagnostics.csv"))))
    bad = {os.path.splitext(r["name"])[0] for r in diag if r["status"].startswith("fallback_scale")}
    res = dict(recon_dir=a.recon_dir, fallback_frames=len(bad))
    if not bad:
        json.dump(res, open(os.path.join(a.recon_dir, "fallback_footprint.json"), "w"), indent=2)
        print(f"{a.recon_dir}: no fallback frames"); return
    cams = R.read_colmap_cameras(os.path.join(a.colmap_dir, "cameras.txt"))
    poses = R.read_colmap_images(os.path.join(a.colmap_dir, "images.txt"))
    tr = {round(r[0], 6): (r[1:4], quat_to_R(*r[4:8])) for r in np.loadtxt(os.path.join(a.recon_dir, "trajectory_fused.txt"))}
    rays = {cid: R.pixel_rays(c)[::2, ::2] for cid, c in cams.items()}

    def pts(names):
        P = []
        for n in names:
            fp = os.path.join(a.recon_dir, "aligned_depth", os.path.splitext(n)[0] + ".npy")
            if not os.path.exists(fp):
                continue
            d = np.load(fp).astype(np.float64); m = d > 0
            ry = rays[poses[n]["camera_id"]][:d.shape[0], :d.shape[1]]
            z = d[m]; xy = ry[m]
            C, Rc = tr[round(name_to_timestamp(n), 6)]
            X = np.stack([xy[:, 0] * z, xy[:, 1] * z, z], 1) @ Rc.T + C
            P.append(X[np.unique(np.floor(X / 0.02).astype(np.int64), axis=0, return_index=True)[1]])
        return np.concatenate(P)
    names = [r["name"] for r in diag if r["name"] in poses]
    Pb = pts([n for n in names if os.path.splitext(n)[0] in bad]); Pg = pts([n for n in names if os.path.splitext(n)[0] not in bad])
    F = np.asarray(o3d.io.read_point_cloud(os.path.join(a.recon_dir, "scene.ply")).points)
    db, _ = cKDTree(Pb).query(F, distance_upper_bound=0.02); dg, _ = cKDTree(Pg).query(F, distance_upper_bound=0.02)
    only = np.isfinite(db) & ~np.isfinite(dg)
    is_tum = a.kind == "tum"
    gt_rows = read_list(os.path.join(a.gt_dir, "groundtruth.txt")) if is_tum else np.loadtxt(os.path.join(a.gt_dir, "groundtruth.txt"))
    est, keep, (s, Rr, t) = load_alignment(os.path.join(a.recon_dir, "trajectory_fused.txt"), gt_rows, is_tum)
    G = gt_cloud_tum(a.gt_dir, est[keep, 0], 0.02) if is_tum else gt_cloud_synth(a.gt_dir, 0.02)
    acc = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s * F @ Rr.T + t)).compute_point_cloud_distance(G))
    res.update(final_points=len(F), points_only_from_fallback=int(only.sum()), pct_of_final=float(only.mean() * 100),
               their_acc_median_cm=float(np.median(acc[only]) * 100) if only.any() else None,
               acc_median_all_cm=float(np.median(acc) * 100), acc_median_excluding_cm=float(np.median(acc[~only]) * 100))
    json.dump(res, open(os.path.join(a.recon_dir, "fallback_footprint.json"), "w"), indent=2)
    print(f"{a.recon_dir}: {len(bad)} fallback frames, {res['pct_of_final']:.2f}% of final pts only from them "
          f"(acc {res['their_acc_median_cm']}), acc all {res['acc_median_all_cm']:.2f} excl {res['acc_median_excluding_cm']:.2f} cm")


if __name__ == "__main__":
    main()
