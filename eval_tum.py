r"""
eval_tum.py -- score a reconstruct.py output on a TUM RGB-D sequence.

1. Trajectory (evo CLI): ATE / RPE RMSE with SE(3) (-a) and Sim(3) (-as) alignment,
   evo's Sim(3) scale correction, number of associated poses.
     SE(3)  = metric test (no scale freedom)
     Sim(3) = trajectory-shape test (global scale removed)
2. Depth vs TUM's Kinect depth (needs alignment_diagnostics.csv + depth cache):
   raw network depth (metres) and aligned depth vs GT per frame.
3. Cloud vs a GT cloud back-projected from TUM depth + GT poses. The fused cloud is
   mapped into the mocap world with the SE(3) / Sim(3) Umeyama fit of the camera centres:
   accuracy, completeness, extents, and thickness of the dominant GT planes (desk, floor...)
   as a layering measure.

Usage:
  python eval_tum.py --seq_dir <tum sequence folder> --recon_dir <reconstruct.py --out folder>
  python eval_tum.py --seq_dir ... --recon_dir output --traj estimated_trajectory.txt   (legacy run)
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from eval_synth import quat_to_R, umeyama, rmse

# Calibrated RGB intrinsics from the TUM benchmark page (depth is registered to RGB).
INTRINSICS = {"freiburg1": (517.3, 516.5, 318.6, 255.3), "freiburg2": (520.9, 521.0, 325.1, 249.7),
              "freiburg3": (535.4, 539.2, 320.1, 247.6)}


def read_list(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        rows.append((float(p[0]), p[1:]))
    return rows


def associate(ts_query, ts_ref, max_diff):
    ts_ref = np.asarray(ts_ref)
    out = {}
    for t in ts_query:
        j = int(np.argmin(np.abs(ts_ref - t)))
        if abs(ts_ref[j] - t) <= max_diff:
            out[t] = j
    return out


def run_evo(gt, est):
    exe_dir = os.path.dirname(sys.executable)
    res = {}
    for tool in ("evo_ape", "evo_rpe"):
        for mode, flags in (("se3", ["-a"]), ("sim3", ["-as"])):
            exe = os.path.join(exe_dir, tool + (".exe" if os.name == "nt" else ""))
            out = subprocess.run([exe, "tum", gt, est, *flags, "-v"], capture_output=True, text=True).stdout
            m = re.search(r"rmse\s+([0-9.eE+-]+)", out)
            n = re.search(r"Found (\d+) of max\. \d+ possible matching timestamps", out) or \
                re.search(r"(\d+) poses", out)
            s = re.search(r"Scale correction:\s*([0-9.eE+-]+)", out)
            res[f"{tool[4:]}_{mode}_rmse_m"] = float(m.group(1)) if m else None
            if n and "n_poses" not in res:
                res["n_poses"] = int(n.group(1))
            if s and mode == "sim3" and tool == "evo_ape":
                res["evo_scale_correction"] = float(s.group(1))
    return res


def plane_thickness(gt_pts, fused_pts, n_planes=2, band=0.3):
    """Dominant planes of the GT cloud; robust spread of the fused points that belong to them.
    A fused point belongs to plane k if its nearest GT point is an inlier of plane k (so objects
    standing on the desk/floor are not counted as plane thickness) and it lies within +-band."""
    out = []
    rest_idx = np.arange(len(gt_pts))
    label = np.full(len(gt_pts), -1)
    planes = []
    for k in range(n_planes):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts[rest_idx]))
        model, inl = pc.segment_plane(0.02, 3, 3000)
        label[rest_idx[inl]] = k
        planes.append(model)
        rest_idx = np.delete(rest_idx, inl)
    _, nn = cKDTree(gt_pts).query(fused_pts)
    for k, model in enumerate(planes):
        n, d = np.array(model[:3]), model[3]
        G = gt_pts[label == k]
        sel = label[nn] == k
        sd = fused_pts[sel] @ n + d
        s = sd[np.abs(sd) < band]
        g = G @ n + d
        rec = dict(normal=[round(float(x), 3) for x in n], gt_inliers=len(G), fused_pts=int(len(s)),
                   gt_robust_std_cm=float(1.4826 * np.median(np.abs(g - np.median(g))) * 100))
        if len(s) > 100:
            med = np.median(s)
            rec.update(fused_offset_cm=float(med * 100),
                       fused_robust_std_cm=float(1.4826 * np.median(np.abs(s - med)) * 100),
                       fused_p5_p95_cm=float((np.percentile(s, 95) - np.percentile(s, 5)) * 100))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--traj", default=None, help="Camera centres in the cloud's frame (default recon_dir/trajectory_fused.txt)")
    ap.add_argument("--gt_frame_step", type=int, default=5, help="Use every Nth evaluated frame for the GT cloud")
    ap.add_argument("--out_name", default="eval_tum.json")
    args = ap.parse_args()

    seq = os.path.basename(os.path.normpath(args.seq_dir))
    fx, fy, cx, cy = next(v for k, v in INTRINSICS.items() if k in seq)
    gt_path = os.path.join(args.seq_dir, "groundtruth.txt")
    traj_path = args.traj or os.path.join(args.recon_dir, "trajectory_fused.txt")
    report = {"sequence": seq, "recon_dir": args.recon_dir, "trajectory_file": traj_path}

    # ---- 1. evo ----
    report["evo"] = run_evo(gt_path, traj_path)

    # ---- own association (for cloud / depth alignment) ----
    gt_rows = read_list(gt_path)
    gt_t = [r[0] for r in gt_rows]
    est = np.loadtxt(traj_path)
    assoc = associate(est[:, 0], gt_t, 0.02)
    keep = [i for i, t in enumerate(est[:, 0]) if t in assoc]
    E = est[keep, 1:4]
    Gpose = {}
    for i in keep:
        t = est[i, 0]; v = [float(x) for x in gt_rows[assoc[t]][1]]
        Gpose[t] = (np.array(v[:3]), quat_to_R(*v[3:7]))
    G = np.array([Gpose[est[i, 0]][0] for i in keep])
    s_sim, R_sim, t_sim = umeyama(E, G, True)
    _, R_se3, t_se3 = umeyama(E, G, False)
    report["own_alignment"] = dict(n=len(keep), sim3_scale_gt_per_unit=float(s_sim),
                                   ate_se3_rmse_m=rmse(np.linalg.norm(E @ R_se3.T + t_se3 - G, axis=1)),
                                   ate_sim3_rmse_m=rmse(np.linalg.norm(s_sim * E @ R_sim.T + t_sim - G, axis=1)))

    # ---- depth association ----
    depth_rows = read_list(os.path.join(args.seq_dir, "depth.txt"))
    d_assoc = associate([est[i, 0] for i in keep], [r[0] for r in depth_rows], 0.02)

    def gt_depth(t):
        return np.asarray(Image.open(os.path.join(args.seq_dir, depth_rows[d_assoc[t]][1][0])), np.float64) / 5000.0

    # ---- 2. depth ----
    diag = os.path.join(args.recon_dir, "alignment_diagnostics.csv")
    if os.path.exists(diag):
        summ = json.load(open(os.path.join(args.recon_dir, "summary.json")))
        cache = summ["args"].get("depth_cache") or os.path.join(args.recon_dir, "depth_pred")
        rows = {float(os.path.splitext(r["name"])[0]): r for r in csv.DictReader(open(diag))}
        raw_ratio, raw_absrel, al_absrel, al_absrel_s, logbias, d1 = [], [], [], [], [], []
        for t in list(d_assoc)[::2]:
            r = rows.get(round(t, 6)) or next((v for k, v in rows.items() if abs(k - t) < 1e-4), None)
            p = os.path.join(cache, os.path.splitext(r["name"])[0] + ".npy") if r else None
            if not p or not os.path.exists(p):
                continue
            d = np.load(p).astype(np.float64); g = gt_depth(t)
            m = (g > 0.1) & (g < 6) & np.isfinite(d) & (d > 0)
            al = float(r["a"]) * d + float(r["b"]); m &= al > 0
            if m.sum() < 1000:
                continue
            raw_ratio.append(np.median(g[m] / d[m])); raw_absrel.append(np.mean(np.abs(d[m] - g[m]) / g[m]))
            al_absrel.append(np.mean(np.abs(al[m] - g[m]) / g[m]))
            als = s_sim * al
            al_absrel_s.append(np.mean(np.abs(als[m] - g[m]) / g[m]))
            logbias.append(np.median(np.log(g[m] / als[m])))
            d1.append(np.mean(np.maximum(als[m] / g[m], g[m] / als[m]) < 1.25))
        pct = lambda x: [float(v) for v in np.percentile(x, [5, 50, 95])]
        report["depth"] = dict(frames=len(raw_ratio), raw_gt_over_pred_p5_p50_p95=pct(raw_ratio),
                               raw_absrel_mean=float(np.mean(raw_absrel)),
                               aligned_absrel_as_is=float(np.mean(al_absrel)),
                               aligned_absrel_after_sim3_scale=float(np.mean(al_absrel_s)),
                               aligned_delta1_after_sim3_scale=float(np.mean(d1)),
                               aligned_per_frame_log_bias_std=float(np.std(logbias)))

    # ---- 3. cloud ----
    H, W = 480, 640
    us, vs = np.meshgrid(np.arange(0, W, 2), np.arange(0, H, 2))
    rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    pts = []
    for t in list(d_assoc)[::args.gt_frame_step]:
        g = gt_depth(t)[::2, ::2]; m = (g > 0.1) & (g < 6)
        C, R = Gpose[t]
        pts.append((rays[m] * g[m][:, None]) @ R.T + C)
    gtc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts))).voxel_down_sample(0.01)
    Gp = np.asarray(gtc.points)
    F = np.asarray(o3d.io.read_point_cloud(os.path.join(args.recon_dir, "scene.ply")).points)
    ext = lambda P: [float(v) for v in (np.percentile(P, 99, 0) - np.percentile(P, 1, 0))]
    report["cloud"] = dict(n_fused=len(F), n_gt=len(Gp), gt_extent_xyz_m=ext(Gp),
                           fused_extent_own_frame_units=ext(F))
    for label, (c, R, t) in (("se3", (1.0, R_se3, t_se3)), ("sim3", (s_sim, R_sim, t_sim))):
        P = c * F @ R.T + t
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        acc = np.asarray(pc.compute_point_cloud_distance(gtc)); comp = np.asarray(gtc.compute_point_cloud_distance(pc))
        report["cloud"][label] = dict(
            accuracy_median_cm=float(np.median(acc) * 100), accuracy_p90_cm=float(np.percentile(acc, 90) * 100),
            frac_within_5cm=float(np.mean(acc < 0.05)), completeness_median_cm=float(np.median(comp) * 100),
            completeness_frac_within_5cm=float(np.mean(comp < 0.05)), extent_xyz_m=ext(P),
            planes=plane_thickness(Gp, P))

    json.dump(report, open(os.path.join(args.recon_dir, args.out_name), "w"), indent=2)

    ev = report["evo"]
    f = lambda x: "n/a" if x is None else f"{x*1000:.1f} mm"
    print(f"== {seq} :: {args.recon_dir}")
    print(f"evo ({ev.get('n_poses')} poses): ATE SE3 {f(ev['ape_se3_rmse_m'])} | RPE SE3 {f(ev['rpe_se3_rmse_m'])} | "
          f"ATE Sim3 {f(ev['ape_sim3_rmse_m'])} | RPE Sim3 {f(ev['rpe_sim3_rmse_m'])} | scale correction {ev.get('evo_scale_correction')}")
    if "depth" in report:
        dp = report["depth"]
        print(f"depth ({dp['frames']} fr): raw GT/pred p5/50/95 {', '.join(f'{v:.3f}' for v in dp['raw_gt_over_pred_p5_p50_p95'])}, "
              f"raw AbsRel {dp['raw_absrel_mean']:.3f} | aligned AbsRel as-is {dp['aligned_absrel_as_is']:.3f}, "
              f"after s* {dp['aligned_absrel_after_sim3_scale']:.3f}, d1 {dp['aligned_delta1_after_sim3_scale']:.3f}, "
              f"per-frame log-bias std {dp['aligned_per_frame_log_bias_std']:.4f}")
    cl = report["cloud"]
    print(f"cloud: {cl['n_fused']} pts; GT extent {', '.join(f'{v:.2f}' for v in cl['gt_extent_xyz_m'])} m")
    for label in ("se3", "sim3"):
        c = cl[label]
        pl = "; ".join(f"plane n={p['normal']}: fused std {p.get('fused_robust_std_cm', float('nan')):.1f} cm "
                       f"(p5-95 {p.get('fused_p5_p95_cm', float('nan')):.1f}), off {p.get('fused_offset_cm', float('nan')):+.1f} cm, "
                       f"GT std {p['gt_robust_std_cm']:.1f} cm" for p in c["planes"])
        print(f"  [{label}] acc median {c['accuracy_median_cm']:.1f} cm, p90 {c['accuracy_p90_cm']:.1f}, <5cm {c['frac_within_5cm']*100:.0f}% | "
              f"compl median {c['completeness_median_cm']:.1f} cm, <5cm {c['completeness_frac_within_5cm']*100:.0f}% | "
              f"extent {', '.join(f'{v:.2f}' for v in c['extent_xyz_m'])} m")
        print(f"         {pl}")


if __name__ == "__main__":
    main()
