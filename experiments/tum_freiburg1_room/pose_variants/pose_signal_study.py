r"""
pose_signal_study.py -- DESIGN EVIDENCE ONLY (no pipeline change). Which reconstruction-time pose-quality signals,
computed from the CONTROL COLMAP model + database + existing alignment diagnostics, track pose error?

Signals (per registered frame, no ground truth involved):
  n_obs        triangulated observations in the image (images.txt POINTS2D with point3D_id >= 0)
  reproj_img   per-image median reprojection error: project each observed 3D point with the image pose/camera
  track_med    median track length of the image's 3D points
  inl_nb       verified inliers to the previous/next frame (database), min of the two
  step_ratio   translation step / local median step (window +-5)
  rot_step     rotation to the previous frame [deg]
  accel        |C[i-1] - 2 C[i] + C[i+1]| / local median step (trajectory non-smoothness)
  align_res    existing per-frame depth<->COLMAP median residual (alignment_diagnostics.csv, 'frame' signal)
Targets (ground truth, evaluation only):
  ate_abs      Sim(3)-aligned absolute position error (accumulated drift)
  rpe_local    |est step - GT step| after Sim(3) for (i-1 -> i+1)   (local pose error)
Reports Spearman rho and AUROC for flagging the worst 10 % frames of each target.
"""
import csv
import json
import os
import sqlite3
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = r"Q:\OPEN_CV_SEM5"
sys.path.insert(0, ROOT)
import Reconstruct as R  # noqa: E402
from eval_synth import umeyama  # noqa: E402
from eval_tum import associate, read_list  # noqa: E402

SEQ = r"C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_room\rgbd_dataset_freiburg1_room"
EXP = os.path.join(ROOT, r"experiments\tum_freiburg1_room")
MODEL = os.path.join(EXP, r"colmap_ws\sparse\0")
DB = os.path.join(EXP, r"colmap_ws\database.db")

cam = next(iter(R.read_colmap_cameras(os.path.join(MODEL, "cameras.txt")).values()))
idx, xyz, err, trk = R.read_colmap_points3d(os.path.join(MODEL, "points3D.txt"))
# parse images.txt with 2D coordinates
imgs = {}
with open(os.path.join(MODEL, "images.txt")) as f:
    lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
for i in range(0, len(lines) - 1, 2):
    p = lines[i].split()
    if len(p) < 10:
        continue
    q = list(map(float, p[1:5])); t = np.array(list(map(float, p[5:8])))
    obs = lines[i + 1].split()
    uv = np.array(obs, float).reshape(-1, 3) if obs else np.zeros((0, 3))
    imgs[p[9]] = dict(R=R.qvec_to_rotmat(*q), t=t, uv=uv[uv[:, 2] >= 0])
names = sorted(imgs, key=lambda n: float(os.path.splitext(n)[0]))
C = np.array([-imgs[n]["R"].T @ imgs[n]["t"] for n in names])
Rs = [imgs[n]["R"] for n in names]

sig = {k: [] for k in ("n_obs", "reproj_img", "track_med")}
for n in names:
    d = imgs[n]
    rows = np.array([idx[int(pid)] for pid in d["uv"][:, 2] if int(pid) in idx])
    X = xyz[rows] @ d["R"].T + d["t"]
    u, v = R.project(X, cam)
    obs = d["uv"][[int(pid) in idx for pid in d["uv"][:, 2]], :2] - 0.5   # COLMAP uv -> pixel-index convention of project()
    sig["n_obs"].append(len(rows))
    sig["reproj_img"].append(float(np.median(np.hypot(u - obs[:, 0], v - obs[:, 1]))))
    sig["track_med"].append(float(np.median(trk[rows])))
# database: verified inliers with neighbours
con = sqlite3.connect(DB)
im = {r[1]: r[0] for r in con.execute("SELECT image_id, name FROM images")}
pairs = {}
for pid, n_in in con.execute("SELECT pair_id, rows FROM two_view_geometries"):
    i2 = pid % 2147483647; i1 = (pid - i2) // 2147483647
    pairs[(min(i1, i2), max(i1, i2))] = n_in
ids = [im[n] for n in names]
inl = [pairs.get((min(a, b), max(a, b)), 0) for a, b in zip(ids[:-1], ids[1:])]
sig["inl_nb"] = [min(inl[max(i - 1, 0)], inl[min(i, len(inl) - 1)]) for i in range(len(names))]
step = np.r_[np.linalg.norm(np.diff(C, axis=0), axis=1), np.nan]
med = np.array([np.nanmedian(step[max(0, i - 5):i + 6]) for i in range(len(names))])
sig["step_ratio"] = list(np.nan_to_num(step / med, nan=1.0))
rot = [0.0] + [np.degrees(np.arccos(np.clip((np.trace(Rs[i - 1] @ Rs[i].T) - 1) / 2, -1, 1))) for i in range(1, len(Rs))]
sig["rot_step"] = rot
acc = [0.0] + [float(np.linalg.norm(C[i - 1] - 2 * C[i] + C[i + 1]) / med[i]) for i in range(1, len(C) - 1)] + [0.0]
sig["accel"] = acc
diag = {r["name"]: r for r in csv.DictReader(open(os.path.join(EXP, r"improved\alignment_diagnostics.csv")))}
sig["align_res"] = [float(diag[n]["med_rel_res"]) if n in diag and diag[n]["med_rel_res"] not in ("", "nan") else np.nan for n in names]

# ground-truth targets (evaluation only)
ts = np.array([float(os.path.splitext(n)[0]) for n in names])
gt = read_list(os.path.join(SEQ, "groundtruth.txt"))
a = associate(ts, [r[0] for r in gt], 0.02)
keep = np.array([t in a for t in ts])
G = np.array([[float(x) for x in gt[a[t]][1][:3]] if t in a else [np.nan] * 3 for t in ts])
s, Rr, tt = umeyama(C[keep], G[keep], True)
Ea = s * C @ Rr.T + tt
ate = np.linalg.norm(Ea - G, axis=1)
rpe = np.full(len(ts), np.nan)
for i in range(1, len(ts) - 1):
    rpe[i] = np.linalg.norm((Ea[i + 1] - Ea[i - 1]) - (G[i + 1] - G[i - 1]))
win = (ts >= 1305031938.5) & (ts <= 1305031941.0)


def auroc(score, bad):
    r = rankdata(score); n1 = bad.sum(); n0 = len(bad) - n1
    return float((r[bad].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


# orientation: which direction means "bad pose"
bad_high = {"n_obs": False, "reproj_img": True, "track_med": False, "inl_nb": False, "step_ratio": True,
            "rot_step": True, "accel": True, "align_res": True}
out = {}
for target, y in (("ate_abs", ate), ("rpe_local", rpe)):
    ok = np.isfinite(y)
    worst = y >= np.nanpercentile(y, 90)
    out[target] = {}
    for k, v in sig.items():
        v = np.array(v, float); m = ok & np.isfinite(v)
        sc = v if bad_high[k] else -v
        out[target][k] = dict(spearman=float(spearmanr(sc[m], y[m])[0]), auroc_worst10=auroc(sc[m], worst[m]))
out["window_vs_rest"] = {k: dict(window_median=float(np.nanmedian(np.array(v, float)[win])),
                                 rest_median=float(np.nanmedian(np.array(v, float)[~win])))
                         for k, v in sig.items()}
out["targets"] = dict(ate_window_median_m=float(np.nanmedian(ate[win])), ate_rest_median_m=float(np.nanmedian(ate[~win])),
                      rpe_window_median_m=float(np.nanmedian(rpe[win])), rpe_rest_median_m=float(np.nanmedian(rpe[~win])))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_signal_study.json"), "w"), indent=2)
print(f"{'signal':12s} | {'ATE rho':>8s} {'ATE AUC':>8s} | {'RPE rho':>8s} {'RPE AUC':>8s} | window vs rest (median)")
for k in sig:
    a1, a2 = out["ate_abs"][k], out["rpe_local"][k]
    w = out["window_vs_rest"][k]
    print(f"{k:12s} | {a1['spearman']:+8.3f} {a1['auroc_worst10']:8.3f} | {a2['spearman']:+8.3f} {a2['auroc_worst10']:8.3f} | "
          f"{w['window_median']:.3f} vs {w['rest_median']:.3f}")
print("targets:", {k: round(v * 1000, 1) for k, v in out["targets"].items()}, "mm")
