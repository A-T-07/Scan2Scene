r"""
eval_synth.py -- score a reconstruct.py output against synth_render.py ground truth.

Reports:
  1. Trajectory: camera centres (in the fused cloud's frame) vs GT.
       Sim(3) scale s* = GT metres per fused unit  (s* == 1  <=>  fused frame is metric)
       ATE after Sim(3) (shape) and after SE(3) only (metric test, no scale freedom)
  2. Depth (needs alignment_diagnostics.csv + depth cache): raw network depth and
     aligned depth vs GT z-depth per frame.
  3. Cloud: fused cloud mapped into the GT world by the SE(3) / Sim(3) fit from (1),
     compared with a GT cloud back-projected from GT depth + GT poses:
       accuracy (fused->GT), completeness (GT->fused), extents, and the thickness of
       known planar surfaces (floor, walls, ceiling) as a layering/streaking measure.

Usage:
  python eval_synth.py --synth_dir synth_room2 --recon_dir synth_room2\recon_scale
  python eval_synth.py --synth_dir synth_room2 --recon_dir synth_room2\baseline\out --traj synth_room2\estimated_trajectory.txt
"""

import argparse
import csv
import json
import os

import numpy as np
import open3d as o3d

from colmap_to_tum import name_to_timestamp

# Room shell planes (GT world): (axis, coordinate of the visible face, name)
PLANES = [(1, -0.95, "floor"), (2, -2.95, "back wall"), (0, -2.95, "left wall"),
          (0, 2.95, "right wall"), (2, 2.95, "front wall"), (1, 1.95, "ceiling")]
# Object boxes from synth_render.py as (min corner, size); excluded from plane statistics.
OBJECTS = [([-0.8, -0.35, -0.6], [1.6, 0.05, 0.7]), ([-0.25, -0.3, -0.55], [0.5, 0.35, 0.05]),
           ([-0.7, -0.95, 0.4], [0.5, 0.7, 0.06]), ([1.6, -1.0, -2.9], [0.9, 1.6, 0.35]),
           ([2.3, -1.0, -0.6], [0.3, 0.4, 0.3])]


def quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
    ])


def load_tum(path):
    a = np.loadtxt(path)
    return {int(round(r[0])): (r[1:4], quat_to_R(*r[4:8])) for r in a}


def umeyama(src, dst, with_scale):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    c = np.trace(np.diag(S) @ D) / s0.var(0).sum() if with_scale else 1.0
    return c, R, mu_d - c * R @ mu_s


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def gt_cloud(synth_dir, gt, stride=4, voxel=0.01):
    fx, fy, cx, cy, W, H = np.loadtxt(os.path.join(synth_dir, "intrinsics.txt"))
    W, H = int(W), int(H)
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    dirs = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    pts = []
    for i, (C, R) in gt.items():
        d = np.load(os.path.join(synth_dir, "depth_gt", f"frame_{i:05d}.npy"))[::stride, ::stride]
        m = d > 0
        pts.append((dirs[m] * d[m][:, None]) @ R.T + C)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
    return pc.voxel_down_sample(voxel)


def plane_stats(P):
    out = {}
    keep = np.ones(len(P), bool)
    for lo, size in OBJECTS:
        lo = np.array(lo) - 0.1; hi = lo + np.array(size) + 0.2
        keep &= ~np.all((P >= lo) & (P <= hi), axis=1)
    P = P[keep]
    for ax, c, name in PLANES:
        s = P[:, ax] - c
        m = np.abs(s) < 0.15
        if m.sum() < 100:
            out[name] = None
            continue
        s = s[m]
        med = np.median(s)
        out[name] = dict(n=int(m.sum()), offset_m=float(med),
                         robust_std_m=float(1.4826 * np.median(np.abs(s - med))),
                         p90_abs_m=float(np.percentile(np.abs(s - med), 90)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth_dir", required=True)
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--traj", default=None, help="Camera centres in the cloud's frame (default: recon_dir/trajectory_fused.txt)")
    args = ap.parse_args()

    gt = load_tum(os.path.join(args.synth_dir, "groundtruth.txt"))
    est = load_tum(args.traj or os.path.join(args.recon_dir, "trajectory_fused.txt"))
    keys = sorted(set(gt) & set(est))
    G = np.array([gt[k][0] for k in keys]); E = np.array([est[k][0] for k in keys])
    report = {"recon_dir": args.recon_dir}

    # ---- 1. trajectory ----
    s_sim, R_sim, t_sim = umeyama(E, G, True)
    _, R_se3, t_se3 = umeyama(E, G, False)
    ate_sim = np.linalg.norm((s_sim * E @ R_sim.T + t_sim) - G, axis=1)
    ate_se3 = np.linalg.norm((E @ R_se3.T + t_se3) - G, axis=1)
    path_gt = np.linalg.norm(np.diff(G, axis=0), axis=1).sum()
    path_est = np.linalg.norm(np.diff(E, axis=0), axis=1).sum()
    report["trajectory"] = dict(n=len(keys), sim3_scale_gt_per_unit=float(s_sim),
                                ate_sim3_rmse_m=rmse(ate_sim), ate_se3_rmse_m=rmse(ate_se3),
                                path_length_gt_m=float(path_gt), path_length_est_units=float(path_est))

    # ---- 2. depth ----
    diag = os.path.join(args.recon_dir, "alignment_diagnostics.csv")
    if os.path.exists(diag):
        summ = json.load(open(os.path.join(args.recon_dir, "summary.json")))
        cache = summ["args"].get("depth_cache") or os.path.join(args.recon_dir, "depth_pred")
        rows = list(csv.DictReader(open(diag)))
        raw_ratio, raw_absrel, al_absrel, al_absrel_sim, al_logbias, al_d1 = [], [], [], [], [], []
        for r in rows:
            i = int(name_to_timestamp(r["name"]))
            p = os.path.join(cache, os.path.splitext(r["name"])[0] + ".npy")
            if i not in gt or not os.path.exists(p):
                continue
            d = np.load(p).astype(np.float64)
            g = np.load(os.path.join(args.synth_dir, "depth_gt", f"frame_{i:05d}.npy")).astype(np.float64)
            # reconstruct.py uses COLMAP's half-pixel convention; renderer pixel grids match 1:1.
            m = (g > 0) & np.isfinite(d) & (d > 0)
            a, b = float(r["a"]), float(r["b"])
            al = a * d + b
            m &= al > 0
            raw_ratio.append(np.median(g[m] / d[m]))
            raw_absrel.append(np.mean(np.abs(d[m] - g[m]) / g[m]))
            al_absrel.append(np.mean(np.abs(al[m] - g[m]) / g[m]))
            als = s_sim * al
            al_absrel_sim.append(np.mean(np.abs(als[m] - g[m]) / g[m]))
            al_logbias.append(np.median(np.log(g[m] / als[m])))
            al_d1.append(np.mean(np.maximum(als[m] / g[m], g[m] / als[m]) < 1.25))
        pct = lambda x: [float(v) for v in np.percentile(x, [5, 50, 95])]
        report["depth"] = dict(
            frames=len(raw_ratio),
            raw_gt_over_pred_ratio_p5_p50_p95=pct(raw_ratio),
            raw_absrel_mean=float(np.mean(raw_absrel)),
            aligned_absrel_mean_no_rescale=float(np.mean(al_absrel)),
            aligned_absrel_mean_after_sim3_scale=float(np.mean(al_absrel_sim)),
            aligned_delta1_after_sim3_scale=float(np.mean(al_d1)),
            aligned_per_frame_log_bias_std=float(np.std(al_logbias)),
            aligned_per_frame_log_bias_p5_p50_p95=pct(al_logbias),
        )

    # ---- 3. cloud ----
    fused = o3d.io.read_point_cloud(os.path.join(args.recon_dir, "scene.ply"))
    F = np.asarray(fused.points)
    gtc = gt_cloud(args.synth_dir, gt)
    Gp = np.asarray(gtc.points)
    report["cloud"] = {"n_fused": len(F), "n_gt": len(Gp)}
    ext = lambda P: [float(v) for v in (np.percentile(P, 99, 0) - np.percentile(P, 1, 0))]
    report["cloud"]["gt_extent_p1_p99_m"] = ext(Gp)
    report["cloud"]["gt_plane_stats"] = plane_stats(Gp)
    for label, (c, R, t) in [("se3", (1.0, R_se3, t_se3)), ("sim3", (s_sim, R_sim, t_sim))]:
        P = c * F @ R.T + t
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        acc = np.asarray(pc.compute_point_cloud_distance(gtc))
        comp = np.asarray(gtc.compute_point_cloud_distance(pc))
        report["cloud"][label] = dict(
            accuracy_median_m=float(np.median(acc)), accuracy_mean_m=float(np.mean(acc)),
            accuracy_p90_m=float(np.percentile(acc, 90)),
            frac_within_2cm=float(np.mean(acc < 0.02)), frac_within_5cm=float(np.mean(acc < 0.05)),
            completeness_median_m=float(np.median(comp)), completeness_frac_within_5cm=float(np.mean(comp < 0.05)),
            extent_p1_p99_m=ext(P), plane_stats=plane_stats(P))

    out = os.path.join(args.recon_dir, "eval_synth.json")
    json.dump(report, open(out, "w"), indent=2)

    # ---- print ----
    tr = report["trajectory"]
    print(f"== {args.recon_dir}")
    print(f"Trajectory ({tr['n']} poses): Sim3 scale s* = {tr['sim3_scale_gt_per_unit']:.4f} GT m / unit | "
          f"ATE Sim3 {tr['ate_sim3_rmse_m']*1000:.1f} mm | ATE SE3 (no scale) {tr['ate_se3_rmse_m']*1000:.1f} mm | "
          f"path GT {tr['path_length_gt_m']:.3f} m vs est {tr['path_length_est_units']:.3f} units")
    if "depth" in report:
        dp = report["depth"]
        print(f"Depth: raw GT/pred ratio p5/p50/p95 = {', '.join(f'{v:.3f}' for v in dp['raw_gt_over_pred_ratio_p5_p50_p95'])}; "
              f"raw AbsRel {dp['raw_absrel_mean']:.3f}")
        print(f"       aligned AbsRel {dp['aligned_absrel_mean_no_rescale']:.3f} (as-is), "
              f"{dp['aligned_absrel_mean_after_sim3_scale']:.3f} (after s*), delta<1.25 {dp['aligned_delta1_after_sim3_scale']:.3f}; "
              f"per-frame log-bias std {dp['aligned_per_frame_log_bias_std']:.4f}")
    cl = report["cloud"]
    print(f"Cloud: {cl['n_fused']} fused pts, GT {cl['n_gt']} pts; GT extent (m) "
          f"{', '.join(f'{v:.2f}' for v in cl['gt_extent_p1_p99_m'])}")
    for label in ("se3", "sim3"):
        c = cl[label]
        print(f"  [{label}] accuracy median {c['accuracy_median_m']*100:.1f} cm, mean {c['accuracy_mean_m']*100:.1f} cm, "
              f"p90 {c['accuracy_p90_m']*100:.1f} cm, <2cm {c['frac_within_2cm']*100:.0f}%, <5cm {c['frac_within_5cm']*100:.0f}% | "
              f"completeness median {c['completeness_median_m']*100:.1f} cm, <5cm {c['completeness_frac_within_5cm']*100:.0f}% | "
              f"extent {', '.join(f'{v:.2f}' for v in c['extent_p1_p99_m'])}")
        ps = "; ".join(f"{k}: std {v['robust_std_m']*100:.1f} cm, p90 {v['p90_abs_m']*100:.1f} cm, off {v['offset_m']*100:+.1f} cm"
                       for k, v in c["plane_stats"].items() if v)
        print(f"         planes -> {ps}")
    gps = "; ".join(f"{k}: std {v['robust_std_m']*100:.1f} cm" for k, v in cl["gt_plane_stats"].items() if v)
    print(f"  [GT cloud planes] {gps}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
