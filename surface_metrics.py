r"""
surface_metrics.py -- local surface-thickness metric that separates LAYERING from global shape error.

The plane-based thickness in eval_synth / eval_tum measures spread around the *ideal* plane, so a single
but bent surface (depth-shape error) and several stacked copies of a surface (layering) look the same.
This script measures local thickness instead:

  for Q random query points, take all cloud points within radius r (default 10 cm, after Sim(3) alignment to
  the ground-truth frame), fit a plane by PCA, thickness = sqrt(smallest eigenvalue) = RMS distance to the
  local plane. Report the median / p75 over queries. The ground-truth cloud (voxelised at the same spacing)
  gets the same treatment as a reference floor.

It also recomputes accuracy / completeness with the same alignment for convenience.

Usage:
  python surface_metrics.py --kind synth --gt_dir synth_room2 --recon_dir synth_room2\fusion_cmp\voxel
  python surface_metrics.py --kind tum --gt_dir <tum sequence folder> --recon_dir experiments\tum_fr2_xyz\tsdf
"""

import argparse
import json
import os

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from eval_synth import quat_to_R, umeyama
from eval_tum import INTRINSICS, associate, read_list


def local_thickness(P, radius=0.10, queries=30000, min_nb=20, seed=0):
    rng = np.random.default_rng(seed)
    tree = cKDTree(P)
    q = P[rng.choice(len(P), min(queries, len(P)), replace=False)]
    nbs = tree.query_ball_point(q, radius)
    th = []
    for idx in nbs:
        if len(idx) < min_nb:
            continue
        X = P[idx] - P[idx].mean(0)
        ev = np.linalg.eigvalsh(X.T @ X / len(idx))
        th.append(np.sqrt(max(ev[0], 0.0)))
    th = np.array(th)
    return dict(median_cm=float(np.median(th) * 100), p75_cm=float(np.percentile(th, 75) * 100),
                p90_cm=float(np.percentile(th, 90) * 100), n=int(len(th)))


def load_alignment(traj, gt_rows, is_tum):
    est = np.loadtxt(traj)
    if is_tum:
        a = associate(est[:, 0], [r[0] for r in gt_rows], 0.02)
        keep = [i for i, t in enumerate(est[:, 0]) if t in a]
        G = np.array([[float(x) for x in gt_rows[a[est[i, 0]]][1][:3]] for i in keep])
    else:
        gi = {int(round(r[0])): r[1:4] for r in gt_rows}
        keep = [i for i, t in enumerate(est[:, 0]) if int(round(t)) in gi]
        G = np.array([gi[int(round(est[i, 0]))] for i in keep])
    return est, keep, umeyama(est[keep, 1:4], G, True)


def gt_cloud_synth(d, voxel):
    fx, fy, cx, cy, W, H = np.loadtxt(os.path.join(d, "intrinsics.txt"))
    W, H = int(W), int(H)
    g = np.loadtxt(os.path.join(d, "groundtruth.txt"))
    us, vs = np.meshgrid(np.arange(0, W, 2), np.arange(0, H, 2))
    rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    pts = []
    for r in g:
        i = int(round(r[0]))
        z = np.load(os.path.join(d, "depth_gt", f"frame_{i:05d}.npy"))[::2, ::2]
        m = z > 0
        pts.append((rays[m] * z[m][:, None]) @ quat_to_R(*r[4:8]).T + r[1:4])
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts))).voxel_down_sample(voxel)


def gt_cloud_tum(d, est_t, voxel):
    seq = os.path.basename(os.path.normpath(d))
    fx, fy, cx, cy = next(v for k, v in INTRINSICS.items() if k in seq)
    gt = read_list(os.path.join(d, "groundtruth.txt")); dep = read_list(os.path.join(d, "depth.txt"))
    ag = associate(est_t, [r[0] for r in gt], 0.02); ad = associate(est_t, [r[0] for r in dep], 0.02)
    us, vs = np.meshgrid(np.arange(0, 640, 2), np.arange(0, 480, 2))
    rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    pts = []
    for t in est_t:
        if t not in ag or t not in ad:
            continue
        v = [float(x) for x in gt[ag[t]][1]]
        z = np.asarray(Image.open(os.path.join(d, dep[ad[t]][1][0])), np.float64)[::2, ::2] / 5000.0
        m = (z > 0.1) & (z < 6)
        pts.append((rays[m] * z[m][:, None]) @ quat_to_R(*v[3:7]).T + np.array(v[:3]))
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts))).voxel_down_sample(voxel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["synth", "tum"], required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--voxel", type=float, default=0.02, help="GT cloud spacing (match the fused cloud)")
    ap.add_argument("--radius", type=float, default=0.10)
    args = ap.parse_args()

    is_tum = args.kind == "tum"
    gt_rows = read_list(os.path.join(args.gt_dir, "groundtruth.txt")) if is_tum else np.loadtxt(os.path.join(args.gt_dir, "groundtruth.txt"))
    est, keep, (s, R, t) = load_alignment(args.traj or os.path.join(args.recon_dir, "trajectory_fused.txt"), gt_rows, is_tum)
    P = np.asarray(o3d.io.read_point_cloud(os.path.join(args.recon_dir, "scene.ply")).points)
    P = s * P @ R.T + t
    gtc = gt_cloud_tum(args.gt_dir, est[keep, 0], args.voxel) if is_tum else gt_cloud_synth(args.gt_dir, args.voxel)
    G = np.asarray(gtc.points)

    # restrict both clouds to the region the GT observes (fused clouds can extend beyond Kinect range)
    lo, hi = np.percentile(G, 0.5, 0) - 0.1, np.percentile(G, 99.5, 0) + 0.1
    Pin = P[np.all((P >= lo) & (P <= hi), axis=1)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pin))
    acc = np.asarray(pc.compute_point_cloud_distance(gtc)); comp = np.asarray(gtc.compute_point_cloud_distance(pc))
    res = dict(
        recon_dir=args.recon_dir, sim3_scale=float(s), n_fused=len(P), n_fused_in_gt_region=len(Pin), n_gt=len(G),
        local_thickness_fused=local_thickness(Pin, args.radius), local_thickness_gt=local_thickness(G, args.radius),
        accuracy_median_cm=float(np.median(acc) * 100), accuracy_frac_5cm=float(np.mean(acc < 0.05)),
        completeness_median_cm=float(np.median(comp) * 100), completeness_frac_5cm=float(np.mean(comp < 0.05)),
        extent_fused_m=[float(v) for v in (np.percentile(Pin, 99, 0) - np.percentile(Pin, 1, 0))],
        extent_gt_m=[float(v) for v in (np.percentile(G, 99, 0) - np.percentile(G, 1, 0))],
    )
    json.dump(res, open(os.path.join(args.recon_dir, "surface_metrics.json"), "w"), indent=2)
    lf, lg = res["local_thickness_fused"], res["local_thickness_gt"]
    print(f"{args.recon_dir}: local thickness median {lf['median_cm']:.2f} cm (p75 {lf['p75_cm']:.2f}) vs GT {lg['median_cm']:.2f} cm | "
          f"acc med {res['accuracy_median_cm']:.1f} cm, <5cm {res['accuracy_frac_5cm']*100:.0f}% | "
          f"compl med {res['completeness_median_cm']:.1f} cm, <5cm {res['completeness_frac_5cm']*100:.0f}% | "
          f"pts {len(P)} ({len(Pin)} in GT region) | extent {', '.join(f'{v:.2f}' for v in res['extent_fused_m'])} vs GT {', '.join(f'{v:.2f}' for v in res['extent_gt_m'])}")


if __name__ == "__main__":
    main()
