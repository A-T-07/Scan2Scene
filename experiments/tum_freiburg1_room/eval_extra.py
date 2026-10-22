r"""
eval_extra.py -- EVALUATION ONLY supplement for fr1_room (reads reconstruction outputs, never writes to them).

1. Trajectory: per-frame Sim(3)-aligned position error -> ATE RMSE / median / p90 (same Umeyama alignment as eval_tum).
2. Depth (fused aligned depth, half resolution, x Sim(3) scale s*, Kinect GT 0.1-6 m -- same mask as
   eval_aligned_depth.py): per-frame AbsRel, RMSE [m], median relative error, valid pixels; pooled + per group.
3. The 10 frames that fell back to the global scale (status fallback_scale_out_of_range): their depth / pose error
   vs the other frames, and their footprint in the final cloud: final points whose nearest raw source point (2 cm)
   comes ONLY from the fallback frames; accuracy recomputed with those points excluded.
Writes evaluation\eval_extra.json.
"""
import csv
import json
import os
import sys

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

ROOT = r"Q:\OPEN_CV_SEM5"
sys.path.insert(0, ROOT)
import Reconstruct as R  # noqa: E402
from eval_synth import quat_to_R, umeyama  # noqa: E402
from eval_tum import associate, read_list  # noqa: E402
from surface_metrics import gt_cloud_tum  # noqa: E402

SEQ = r"C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_room\rgbd_dataset_freiburg1_room"
EXP = os.path.join(ROOT, r"experiments\tum_freiburg1_room")
COL = os.path.join(EXP, r"colmap_ws\sparse\0")

gt = read_list(os.path.join(SEQ, "groundtruth.txt"))
dep = read_list(os.path.join(SEQ, "depth.txt"))
cam = next(iter(R.read_colmap_cameras(os.path.join(COL, "cameras.txt")).values()))
rays = R.pixel_rays(cam)[::2, ::2]
out = {}
Gc = None
for run in ("baseline", "improved"):
    rd = os.path.join(EXP, run)
    est = np.loadtxt(os.path.join(rd, "trajectory_fused.txt"))
    a = associate(est[:, 0], [r[0] for r in gt], 0.02)
    keep = [i for i, t in enumerate(est[:, 0]) if t in a]
    E = est[keep, 1:4]
    G = np.array([[float(x) for x in gt[a[est[i, 0]]][1][:3]] for i in keep])
    s, Rr, tt = umeyama(E, G, True)
    perr = np.linalg.norm(s * E @ Rr.T + tt - G, axis=1)
    pose_err = {round(est[i, 0], 6): float(e) for i, e in zip(keep, perr)}
    diag = {os.path.splitext(r["name"])[0]: r for r in csv.DictReader(open(os.path.join(rd, "alignment_diagnostics.csv")))}
    bad = {k for k, r in diag.items() if r["status"].startswith("fallback_scale")}
    ad = associate(est[:, 0], [r[0] for r in dep], 0.02)
    rows = []
    for f in sorted(os.listdir(os.path.join(rd, "aligned_depth"))):
        stem = os.path.splitext(f)[0]; t = float(stem)
        if t not in ad:
            continue
        g = np.asarray(Image.open(os.path.join(SEQ, dep[ad[t]][1][0])), np.float64)[::2, ::2] / 5000.0
        p = np.load(os.path.join(rd, "aligned_depth", f)).astype(np.float64) * s
        m = (p > 0) & (g > 0.1) & (g < 6.0)
        if m.sum() < 500:
            continue
        r_ = p[m] / g[m]
        rows.append(dict(frame=stem, bad=stem in bad, absrel=float(np.mean(np.abs(r_ - 1))),
                         rmse=float(np.sqrt(np.mean((p[m] - g[m]) ** 2))), medrel=float(np.median(np.abs(r_ - 1))),
                         valid_px=int(m.sum()), ate=pose_err.get(round(t, 6), np.nan)))
    ab = np.array([r["absrel"] for r in rows]); bmask = np.array([r["bad"] for r in rows])
    grp = lambda sel: dict(frames=int(sel.sum()), absrel_mean=float(ab[sel].mean()),
                           rmse_mean_m=float(np.mean([r["rmse"] for r, k in zip(rows, sel) if k])),
                           medrel_median=float(np.median([r["medrel"] for r, k in zip(rows, sel) if k])),
                           valid_px=int(sum(r["valid_px"] for r, k in zip(rows, sel) if k)),
                           ate_median_m=float(np.nanmedian([r["ate"] for r, k in zip(rows, sel) if k])))
    res = dict(sim3_scale=float(s), ate_rmse_m=float(np.sqrt(np.mean(perr ** 2))), ate_median_m=float(np.median(perr)),
               ate_p90_m=float(np.percentile(perr, 90)), ate_max_m=float(perr.max()),
               depth_all=grp(np.ones(len(rows), bool)), depth_fallback_frames=grp(bmask), depth_other_frames=grp(~bmask),
               fallback_frames=sorted(bad))
    # footprint of the fallback frames in the final cloud
    tr = {round(r[0], 6): (r[1:4], quat_to_R(*r[4:8])) for r in est}

    def pts(frames):
        P = []
        for stem in frames:
            fp = os.path.join(rd, "aligned_depth", stem + ".npy")
            if not os.path.exists(fp):
                continue
            d = np.load(fp).astype(np.float64); mm = d > 0
            z = d[mm]; xy = rays[:d.shape[0], :d.shape[1]][mm]
            C, Rc = tr[round(float(stem), 6)]
            X = np.stack([xy[:, 0] * z, xy[:, 1] * z, z], 1) @ Rc.T + C
            k = np.unique(np.floor(X / 0.02).astype(np.int64), axis=0, return_index=True)[1]
            P.append(X[k])
        return np.concatenate(P)
    Pbad = pts(sorted(bad)); Pgood = pts(sorted(set(diag) - bad))
    F = np.asarray(o3d.io.read_point_cloud(os.path.join(rd, "scene.ply")).points)
    db, _ = cKDTree(Pbad).query(F, distance_upper_bound=0.02)
    dg, _ = cKDTree(Pgood).query(F, distance_upper_bound=0.02)
    only_bad = np.isfinite(db) & ~np.isfinite(dg)
    if Gc is None:
        Gc = gt_cloud_tum(SEQ, est[keep, 0], 0.02)
    Fa = s * F @ Rr.T + tt
    acc = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Fa)).compute_point_cloud_distance(Gc))
    res["fallback_footprint"] = dict(
        final_points=len(F), points_only_from_fallback_frames=int(only_bad.sum()),
        pct_of_final=float(only_bad.mean() * 100),
        their_acc_median_cm=float(np.median(acc[only_bad]) * 100) if only_bad.any() else None,
        acc_median_all_cm=float(np.median(acc) * 100), acc_median_excluding_cm=float(np.median(acc[~only_bad]) * 100),
        within5_all_pct=float(np.mean(acc < 0.05) * 100), within5_excluding_pct=float(np.mean(acc[~only_bad] < 0.05) * 100))
    out[run] = res
    json.dump(dict(per_frame=rows), open(os.path.join(EXP, "evaluation", run, "per_frame_depth_pose.json"), "w"), indent=1)
json.dump(out, open(os.path.join(EXP, "evaluation", "eval_extra.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
