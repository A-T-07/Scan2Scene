r"""
eval_pose_variant.py -- pose-only evaluation of one COLMAP model (TUM groundtruth used ONLY for scoring).

Inputs : a sparse model folder (TXT) + its database (for matching statistics).
Outputs: <out>\trajectory_tum.txt, <out>\pose_eval.json
Metrics:
  registration / models are passed in by the caller (from the sparse folder list)
  evo ATE/RPE, SE(3) and Sim(3) (eval_tum.run_evo), evo scale correction
  own Sim(3) Umeyama per-frame position error -> RMSE / median / p90 / max, and in the problem window
  start/end loop consistency: after Sim(3), error of the displacement between the mean of the first 10 and
     last 10 camera centres vs GT, and the relative rotation error first->last camera
  matching stats from the database: verified pairs, inlier totals, weakest consecutive pair, start<->end pairs
"""
import argparse
import json
import os
import sqlite3
import sys

import numpy as np

ROOT = r"Q:\OPEN_CV_SEM5"
sys.path.insert(0, ROOT)
from colmap_to_tum import read_colmap_images, rotmat_to_qvec  # noqa: E402
from eval_synth import quat_to_R, umeyama  # noqa: E402
from eval_tum import associate, read_list, run_evo  # noqa: E402

SEQ = r"C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_room\rgbd_dataset_freiburg1_room"
WINDOW = (1305031938.5, 1305031941.0)   # problematic region (fallback frames 1939.10-1940.44)


def rot_err_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampled", type=int, default=273)
    ap.add_argument("--n_models", type=int, required=True)
    ap.add_argument("--runtime_s", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    imgs = read_colmap_images(os.path.join(a.model, "images.txt"))
    rows = []
    for name, p in imgs.items():
        Rc = p["R"].T
        C = -Rc @ p["t"]
        rows.append((float(os.path.splitext(name)[0]), *C, *rotmat_to_qvec(Rc)))
    rows.sort()
    traj = os.path.join(a.out, "trajectory_tum.txt")
    np.savetxt(traj, np.array(rows), fmt="%.6f")
    pts = [l.split() for l in open(os.path.join(a.model, "points3D.txt")) if l.strip() and not l.startswith("#")]
    reproj = float(np.mean([float(p[7]) for p in pts]))
    cam = [l.split() for l in open(os.path.join(a.model, "cameras.txt")) if l.strip() and not l.startswith("#")][0]

    evo = run_evo(os.path.join(SEQ, "groundtruth.txt"), traj)

    est = np.array(rows)
    gt = read_list(os.path.join(SEQ, "groundtruth.txt"))
    asc = associate(est[:, 0], [r[0] for r in gt], 0.02)
    keep = [i for i, t in enumerate(est[:, 0]) if t in asc]
    E = est[keep, 1:4]
    Gv = [[float(x) for x in gt[asc[est[i, 0]]][1]] for i in keep]
    G = np.array([g[:3] for g in Gv])
    s, R, t = umeyama(E, G, True)
    Ea = s * E @ R.T + t
    err = np.linalg.norm(Ea - G, axis=1)
    ts = est[keep, 0]
    win = (ts >= WINDOW[0]) & (ts <= WINDOW[1])
    # start/end loop consistency
    k = 10
    d_est = Ea[-k:].mean(0) - Ea[:k].mean(0)
    d_gt = G[-k:].mean(0) - G[:k].mean(0)
    Re_first = R @ quat_to_R(*est[keep[0], 4:8]); Re_last = R @ quat_to_R(*est[keep[-1], 4:8])
    Rg_first = quat_to_R(*Gv[0][3:7]); Rg_last = quat_to_R(*Gv[-1][3:7])
    rel_rot_err = rot_err_deg(Re_first.T @ Re_last, Rg_first.T @ Rg_last)

    # matching statistics
    con = sqlite3.connect(a.db)
    im = {r[0]: r[1] for r in con.execute("SELECT image_id, name FROM images")}
    order = sorted(im, key=lambda i: float(os.path.splitext(im[i])[0]))
    pos = {iid: n for n, iid in enumerate(order)}
    pairs = {}
    for pid, n_in in con.execute("SELECT pair_id, rows FROM two_view_geometries"):
        i2 = pid % 2147483647; i1 = (pid - i2) // 2147483647
        pairs[(min(pos[i1], pos[i2]), max(pos[i1], pos[i2]))] = n_in
    ver = {k_: v for k_, v in pairs.items() if v >= 15}
    cons = [pairs.get((i, i + 1), 0) for i in range(len(order) - 1)]
    tsort = [float(os.path.splitext(im[i])[0]) for i in order]
    wcons = [c for i, c in enumerate(cons) if WINDOW[0] <= tsort[i] <= WINDOW[1]]
    nimg = len(order); edge = 15
    se = [v for (i, j), v in ver.items() if i < edge and j >= nimg - edge]

    res = dict(model=a.model, sampled=a.sampled, registered=len(imgs), registration_pct=100 * len(imgs) / a.sampled,
               n_models=a.n_models, focal_px=float(cam[4]), k1=float(cam[7]), points3D=len(pts), mean_reproj_px=reproj,
               runtime_s=a.runtime_s,
               evo=evo,
               ate_sim3_rmse_m=float(np.sqrt(np.mean(err ** 2))), ate_sim3_median_m=float(np.median(err)),
               ate_sim3_p90_m=float(np.percentile(err, 90)), ate_sim3_max_m=float(err.max()), sim3_scale=float(s),
               window=dict(frames=int(win.sum()), ate_median_m=float(np.median(err[win])) if win.any() else None,
                           ate_max_m=float(err[win].max()) if win.any() else None,
                           min_consecutive_inliers=int(min(wcons)) if wcons else None),
               start_end=dict(displacement_err_m=float(np.linalg.norm(d_est - d_gt)), gt_displacement_m=float(np.linalg.norm(d_gt)),
                              relative_rotation_err_deg=rel_rot_err, verified_start_end_pairs=len(se),
                              max_start_end_inliers=int(max(se)) if se else 0),
               matching=dict(verified_pairs=len(ver), total_inliers=int(sum(ver.values())),
                             median_inliers_per_pair=float(np.median(list(ver.values()))),
                             min_consecutive_inliers=int(min(cons)),
                             nonsequential_pairs_gap_gt_30=int(sum(1 for (i, j) in ver if j - i > 30))))
    json.dump(res, open(os.path.join(a.out, "pose_eval.json"), "w"), indent=2)
    print(f"{a.out}: reg {len(imgs)}/{a.sampled} models {a.n_models} reproj {reproj:.3f} | ATE Sim3 RMSE {res['ate_sim3_rmse_m']*1000:.1f} "
          f"med {res['ate_sim3_median_m']*1000:.1f} p90 {res['ate_sim3_p90_m']*1000:.1f} max {res['ate_sim3_max_m']*1000:.1f} mm | "
          f"evo ATE Sim3 {evo['ape_sim3_rmse_m']*1000:.1f} SE3 {evo['ape_se3_rmse_m']*1000:.1f} RPE Sim3 {evo['rpe_sim3_rmse_m']*1000:.1f} mm "
          f"scale {s:.3f} | window med {res['window']['ate_median_m']*1000:.1f} max {res['window']['ate_max_m']*1000:.1f} mm, min cons inl {res['window']['min_consecutive_inliers']} | "
          f"start/end disp err {res['start_end']['displacement_err_m']*1000:.1f} mm rot {rel_rot_err:.2f} deg, S-E pairs {len(se)} | "
          f"pairs {len(ver)} inliers {res['matching']['total_inliers']}")


if __name__ == "__main__":
    main()
