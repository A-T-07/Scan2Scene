r"""
inspect_video.py -- pre-flight check of a capture before SfM.

Reports resolution / fps / duration / frame count / rotation metadata, per-frame sharpness (variance of the
Laplacian), a rough rotation-vs-translation indicator from consecutive-frame feature matches
(homography vs fundamental-matrix inlier ratio: rotation-only motion is explained by a homography),
and writes a contact sheet + sharpness plot.

Usage: python inspect_video.py --video clip.mp4 --out experiments\phone_real\inspect
"""

import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=3, help="Analyse every Nth frame for motion")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orient = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else None
    auto = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO") else None
    frames, sharp = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
        frames.append(f)
    sharp = np.array(sharp)
    dec_h, dec_w = frames[0].shape[:2]
    # motion: ORB matches between frames `step` apart
    orb = cv2.ORB_create(2000)
    kd = [orb.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None) for f in frames[::args.step]]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    flow, h_ratio = [], []
    for (k1, d1), (k2, d2) in zip(kd[:-1], kd[1:]):
        if d1 is None or d2 is None:
            flow.append(np.nan); h_ratio.append(np.nan); continue
        m = bf.match(d1, d2)
        if len(m) < 30:
            flow.append(np.nan); h_ratio.append(np.nan); continue
        p1 = np.float32([k1[x.queryIdx].pt for x in m]); p2 = np.float32([k2[x.trainIdx].pt for x in m])
        flow.append(float(np.median(np.linalg.norm(p2 - p1, axis=1))))
        Hm, hin = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        Fm, fin = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0)
        h_ratio.append(float(hin.sum() / max(fin.sum(), 1)) if hin is not None and fin is not None else np.nan)
    flow, h_ratio = np.array(flow), np.array(h_ratio)
    res = dict(video=os.path.abspath(args.video), container_w=W, container_h=H, decoded_w=dec_w, decoded_h=dec_h,
               fps=fps, frames=len(frames), frame_count_meta=n, duration_s=len(frames) / fps,
               orientation_meta=orient, orientation_auto=auto,
               sharpness_p10_p50_p90=[float(v) for v in np.percentile(sharp, [10, 50, 90])],
               blurry_frac_below_half_median=float(np.mean(sharp < 0.5 * np.median(sharp))),
               flow_px_per_step_p50_p90=[float(v) for v in np.nanpercentile(flow, [50, 90])],
               homography_to_F_inlier_ratio_p50=float(np.nanmedian(h_ratio)),
               near_pure_rotation_frac=float(np.nanmean(h_ratio > 0.9)))
    json.dump(res, open(os.path.join(args.out, "inspect.json"), "w"), indent=2)
    # contact sheet (16 frames)
    idx = np.linspace(0, len(frames) - 1, 16).astype(int)
    th = [cv2.resize(frames[i], (dec_w // 4, dec_h // 4)) for i in idx]
    for t, i in zip(th, idx):
        cv2.putText(t, f"{i} ({i / fps:.1f}s) sharp {sharp[i]:.0f}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    rows = [np.hstack(th[r * 4:(r + 1) * 4]) for r in range(4)]
    cv2.imwrite(os.path.join(args.out, "contact_sheet.jpg"), np.vstack(rows))
    fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True, dpi=110)
    t = np.arange(len(frames)) / fps
    axs[0].plot(t, sharp); axs[0].set_ylabel("sharpness (Lap var)"); axs[0].axhline(0.5 * np.median(sharp), color="r", ls="--")
    tt = t[::args.step][1:len(flow) + 1]
    axs[1].plot(tt, flow); axs[1].set_ylabel(f"median flow px / {args.step} fr")
    axs[2].plot(tt, h_ratio); axs[2].set_ylabel("H/F inlier ratio"); axs[2].axhline(0.9, color="r", ls="--"); axs[2].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "motion_sharpness.png")); plt.close(fig)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
