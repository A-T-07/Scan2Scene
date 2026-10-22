r"""
diagnose_poses.py -- pose-only diagnostics of a COLMAP video run (no dense fusion).

  * per sparse model: registered frames, focal, mean reprojection error, points, frame runs
  * largest model: inter-frame rotation (deg) / translation (COLMAP units, and relative to median step)
  * database: keypoints per image and verified inlier matches between consecutive images -> where tracking breaks
Writes <workdir>\pose_diagnostics.json and pose_diagnostics.png.
Usage: python diagnose_poses.py --workdir experiments\phone_real\pose_test_unmasked
"""

import argparse
import json
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from colmap_to_tum import read_colmap_images, name_to_timestamp


def model_stats(d):
    imgs = read_colmap_images(os.path.join(d, "images.txt"))
    cams = [l.split() for l in open(os.path.join(d, "cameras.txt")) if l.strip() and not l.startswith("#")]
    pts = [l.split() for l in open(os.path.join(d, "points3D.txt")) if l.strip() and not l.startswith("#")]
    err = np.array([float(p[7]) for p in pts]) if pts else np.array([np.nan])
    names = sorted(imgs, key=name_to_timestamp)
    idx = [int(name_to_timestamp(n)) for n in names]
    C = np.array([-imgs[n]["R"].T @ imgs[n]["t"] for n in names])
    Rs = [imgs[n]["R"] for n in names]
    rot = np.array([np.degrees(np.arccos(np.clip((np.trace(Rs[i] @ Rs[i + 1].T) - 1) / 2, -1, 1))) for i in range(len(Rs) - 1)])
    tr = np.linalg.norm(np.diff(C, axis=0), axis=1)
    return dict(path=d, registered=len(names), frame_idx=idx, focal=float(cams[0][4]), k1=float(cams[0][7]) if len(cams[0]) > 7 else None,
                points=len(pts), mean_reproj_px=float(np.nanmean(err)), rot=rot, tr=tr, C=C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--video_fps", type=float, default=None)
    ap.add_argument("--sample_every", type=int, default=None)
    a = ap.parse_args()
    ws = os.path.join(a.workdir, "colmap_ws")
    models = []
    for sub in sorted(os.listdir(os.path.join(ws, "sparse"))):
        d = os.path.join(ws, "sparse", sub)
        if os.path.exists(os.path.join(d, "images.txt")):
            models.append(model_stats(d))
    models.sort(key=lambda m: -m["registered"])
    # database: keypoints + consecutive verified matches
    con = sqlite3.connect(os.path.join(ws, "database.db"))
    im = {r[0]: r[1] for r in con.execute("SELECT image_id, name FROM images")}
    kp = {r[0]: r[1] for r in con.execute("SELECT image_id, rows FROM keypoints")}
    order = sorted(im, key=lambda i: name_to_timestamp(im[i]))
    pairs = {}
    for pid, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries"):
        i2 = pid % 2147483647; i1 = (pid - i2) // 2147483647
        pairs[(min(i1, i2), max(i1, i2))] = rows
    cons = [pairs.get((min(a_, b_), max(a_, b_)), 0) for a_, b_ in zip(order[:-1], order[1:])]
    kps = [kp.get(i, 0) for i in order]
    m0 = models[0]
    rep = dict(frames_sampled=len(order), n_models=len(models),
               models=[dict(registered=m["registered"], pct=100 * m["registered"] / len(order), focal=m["focal"], k1=m["k1"],
                            points=m["points"], mean_reproj_px=m["mean_reproj_px"],
                            first=min(m["frame_idx"]), last=max(m["frame_idx"])) for m in models],
               largest=dict(rot_deg_median=float(np.median(m0["rot"])), rot_deg_max=float(m0["rot"].max()),
                            tr_median=float(np.median(m0["tr"])), tr_max=float(m0["tr"].max()),
                            tr_max_over_median=float(m0["tr"].max() / np.median(m0["tr"])),
                            path_length=float(m0["tr"].sum()),
                            net_displacement=float(np.linalg.norm(m0["C"][-1] - m0["C"][0])),
                            extent_of_camera_positions=[float(v) for v in (m0["C"].max(0) - m0["C"].min(0))],
                            total_rotation_deg=float(m0["rot"].sum())),
               consecutive_inliers_min=int(min(cons)), weakest_pairs=sorted(
                   [(int(name_to_timestamp(im[order[i]])), int(name_to_timestamp(im[order[i + 1]])), int(c)) for i, c in enumerate(cons)],
                   key=lambda x: x[2])[:6],
               keypoints_median=float(np.median(kps)), keypoints_min=int(min(kps)))
    json.dump(rep, open(os.path.join(a.workdir, "pose_diagnostics.json"), "w"), indent=2)
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True, dpi=110)
    x = [int(name_to_timestamp(im[i])) for i in order]
    axs[0].plot(x, kps); axs[0].set_ylabel("SIFT keypoints")
    axs[1].plot(x[:-1], cons); axs[1].set_ylabel("verified inliers\n(i, i+1)"); axs[1].set_yscale("symlog")
    for m in models:
        axs[2].plot(m["frame_idx"][1:], m["rot"], "-o", ms=2, label=f"model: {m['registered']} frames")
    axs[2].set_ylabel("inter-frame rotation [deg]"); axs[2].set_xlabel("sample index"); axs[2].legend()
    for m in models[1:]:
        for ax in axs:
            ax.axvspan(min(m["frame_idx"]), max(m["frame_idx"]), color="red", alpha=0.06)
    fig.tight_layout(); fig.savefig(os.path.join(a.workdir, "pose_diagnostics.png")); plt.close(fig)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
