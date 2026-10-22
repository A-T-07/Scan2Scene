r"""
thin_structure_eval.py -- EVALUATION ONLY: are small / thin structures preserved?

Structure sets are defined from ground-truth geometry alone (identical for every method):
  high_variation : GT points whose local surface variation lam_min/sum(lam) in a 4 cm ball is in the top 15 %
                   (edges, thin parts, small objects)
  objects        : TUM -> GT points 1.5-40 cm above the dominant raised horizontal plane (the desk), inside its
                   footprint (keyboard, books, monitor base, mugs...);
                   synthetic -> GT points within 3 cm of the known object boxes (desk, monitor, chair, shelf, pot)
For each run (Sim(3)-aligned to GT): completeness at 2 / 5 cm on each set, and the median distance of fused
points that belong to the set region (nearest GT point is in the set, within 20 cm).

Usage: python thin_structure_eval.py --kind tum --gt_dir <seq> --out <json> --run <label> <dir> [--run ...]
"""

import argparse
import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from surface_metrics import gt_cloud_synth, gt_cloud_tum, load_alignment
from eval_tum import read_list

SYN_BOXES = [([-0.8, -0.35, -0.6], [1.6, 0.05, 0.7]), ([-0.25, -0.3, -0.55], [0.5, 0.35, 0.05]),
             ([-0.7, -0.95, 0.4], [0.5, 0.7, 0.06]), ([1.6, -1.0, -2.9], [0.9, 1.6, 0.35]),
             ([2.3, -1.0, -0.6], [0.3, 0.4, 0.3])]


def surface_variation(P, r=0.04):
    tree = cKDTree(P)
    sv = np.full(len(P), np.nan)
    for i, idx in enumerate(tree.query_ball_point(P, r)):
        if len(idx) >= 8:
            X = P[idx] - P[idx].mean(0)
            ev = np.linalg.eigvalsh(X.T @ X)
            sv[i] = ev[0] / max(ev.sum(), 1e-12)
    return sv


def desk_objects(P, up=2):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    rest = pc; best = None
    zlo = np.percentile(P[:, up], 2)
    for _ in range(8):   # large / tilted floors can take several RANSAC planes before the desk appears
        if len(rest.points) < 1000:
            break
        model, inl = rest.segment_plane(0.015, 3, 2000)
        n = np.array(model[:3]); Q = np.asarray(rest.points)[inl]
        if abs(n[up]) > 0.95 and np.median(Q[:, up]) > zlo + 0.4 and len(inl) > 0.03 * len(P):
            if best is None or len(inl) > len(best[1]):
                best = (np.median(Q[:, up]), Q)
        rest = rest.select_by_index(inl, invert=True)
    if best is None:
        return np.zeros(len(P), bool), None
    h, Q = best
    ax = [i for i in range(3) if i != up]
    lo, hi = np.percentile(Q[:, ax], 2, 0), np.percentile(Q[:, ax], 98, 0)
    inside = np.all((P[:, ax] >= lo) & (P[:, ax] <= hi), axis=1)
    height = P[:, up] - h
    return inside & (height > 0.015) & (height < 0.40), dict(desk_height=float(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["tum", "synth"], required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--run", nargs=2, action="append", required=True, metavar=("LABEL", "DIR"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    is_tum = args.kind == "tum"
    first = args.run[0][1]
    gt_rows = read_list(os.path.join(args.gt_dir, "groundtruth.txt")) if is_tum else np.loadtxt(os.path.join(args.gt_dir, "groundtruth.txt"))
    est, keep, _ = load_alignment(os.path.join(first, "trajectory_fused.txt"), gt_rows, is_tum)
    gtc = gt_cloud_tum(args.gt_dir, est[keep, 0], 0.01) if is_tum else gt_cloud_synth(args.gt_dir, 0.01)
    G = np.asarray(gtc.points)
    sv = surface_variation(G)
    thr = np.nanquantile(sv, 0.85)
    sets = {"high_variation": np.nan_to_num(sv, nan=0) >= thr}
    if is_tum:
        sets["objects"], info = desk_objects(G, up=2)
    else:
        m = np.zeros(len(G), bool)
        for lo, size in SYN_BOXES:
            lo = np.array(lo) - 0.03; hi = lo + np.array(size) + 0.06
            m |= np.all((G >= lo) & (G <= hi), axis=1)
        sets["objects"], info = m, {"boxes": "renderer object boxes +3 cm"}
    tree_g = cKDTree(G)
    res = dict(gt_points=len(G), set_sizes={k: int(v.sum()) for k, v in sets.items()}, info=info, runs={})
    for label, rd in args.run:
        _, _, (s, R, t) = load_alignment(os.path.join(rd, "trajectory_fused.txt"), gt_rows, is_tum)
        P = s * np.asarray(o3d.io.read_point_cloud(os.path.join(rd, "scene.ply")).points) @ R.T + t
        tree_p = cKDTree(P)
        dg, _ = tree_p.query(G, distance_upper_bound=0.5)
        dp, ip = tree_g.query(P, distance_upper_bound=0.2)
        r = {}
        for k, m in sets.items():
            own = np.isfinite(dp) & (ip < len(G))
            own[own] = m[ip[own]]
            r[k] = dict(completeness_2cm=float(np.mean(dg[m] < 0.02)), completeness_5cm=float(np.mean(dg[m] < 0.05)),
                        fused_pts=int(own.sum()), accuracy_median_cm=float(np.median(dp[own]) * 100) if own.any() else None)
        res["runs"][label] = r
        print(f"{label:22s} " + " | ".join(f"{k}: compl@2cm {v['completeness_2cm']*100:4.1f}% @5cm {v['completeness_5cm']*100:4.1f}% "
                                          f"acc {v['accuracy_median_cm'] if v['accuracy_median_cm'] is None else round(v['accuracy_median_cm'], 2)} cm (n {v['fused_pts']})"
                                          for k, v in r.items()))
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"set sizes: {res['set_sizes']}  {info}")


if __name__ == "__main__":
    main()
