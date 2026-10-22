r"""
reconstruct.py -- COLMAP poses + Depth Anything V2 metric depth -> fused point cloud

Pipeline:
  1. Read COLMAP cameras / images / points3D (plain-text export).
  2. Predict per-frame depth with Depth Anything V2 Metric-Indoor (cached as .npy).
  3. Depth <-> SfM alignment, using COLMAP's sparse points as the common reference:
       a. For each frame, project the sparse points it observes -> (pixel, COLMAP depth z).
          Sample predicted depth d at the same pixels.
       b. GLOBAL scale S (COLMAP units per predicted metre) = median over frames of the
          per-frame median of log(z / d). The whole COLMAP model (poses + sparse points)
          is divided by S once. This defines the output coordinate system: COLMAP's
          world frame, rescaled to the depth network's *median* metric scale.
       c. PER-FRAME alignment of the predicted depth to the rescaled sparse depth
          (z_m = z / S):  --align scale   d' = a_i * d        (robust log-ratio fit)
                          --align affine  d' = a_i * d + b_i  (RANSAC + LSQ refine)
                          --align global  d' = d               (a_i = 1, b_i = 0)
                          --align none    raw COLMAP units, no alignment (for comparison)
          Bad fits (too few points, few inliers, implausible scale) fall back to global.
  4. Vectorised back-projection with the (rescaled) COLMAP pose, depth-edge filtering.
  5. Concatenate, voxel-downsample, statistical outlier removal, save scene.ply.

Outputs in --out:
  scene.ply                  fused cloud (coordinate system described in summary.json)
  alignment_diagnostics.csv  per-frame alignment statistics
  summary.json               sequence-level summary + all arguments
  trajectory_fused.txt       camera centres in the fused frame, TUM format (for evo)
  depth_pred\*.npy           cached network depth (metres, float32, image resolution)

Fusion (--fusion):
  voxel  concatenate back-projected points, voxel-downsample, outlier removal (baseline, default)
  tsdf   integrate the aligned depth maps into an Open3D ScalableTSDFVolume (--tsdf_voxel, --tsdf_trunc,
         --depth_trunc), extract the zero-level surface as a point cloud (+ mesh with --tsdf_mesh)

Optional metric calibration (OFF by default): --metric_scale k or --metric_scale_file calib.json
  multiplies the output frame by k. Derive k with calibrate_scale.py from OTHER sequences only.

Usage:
  python reconstruct.py --frames frames/ --colmap_dir sparse/0 --out output/ --no_viewer
  python reconstruct.py ... --fusion tsdf --tsdf_voxel 0.02 --tsdf_trunc 0.08 --no_viewer
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
from PIL import Image

from colmap_to_tum import name_to_timestamp, rotmat_to_qvec


# ---------- 1. COLMAP parsing ----------

def read_colmap_cameras(cameras_txt):
    cams = {}
    with open(cameras_txt, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model, w, h = parts[1], int(parts[2]), int(parts[3])
            p = list(map(float, parts[4:]))
            k1 = k2 = p1 = p2 = 0.0
            if model == "PINHOLE":
                fx, fy, cx, cy = p
            elif model == "SIMPLE_PINHOLE":
                fx = fy = p[0]; cx, cy = p[1], p[2]
            elif model == "SIMPLE_RADIAL":
                fx = fy = p[0]; cx, cy, k1 = p[1], p[2], p[3]
            elif model == "RADIAL":
                fx = fy = p[0]; cx, cy, k1, k2 = p[1], p[2], p[3], p[4]
            elif model == "OPENCV":
                fx, fy, cx, cy, k1, k2, p1, p2 = p
            else:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")
            cams[cam_id] = dict(model=model, width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy,
                                k1=k1, k2=k2, p1=p1, p2=p2)
    return cams


def qvec_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
    ])


def read_colmap_images(images_txt):
    """Returns {name: {'R': world->cam, 't', 'camera_id', 'point3D_ids'}}"""
    poses = {}
    with open(images_txt, "r") as f:
        lines = [l for l in f if not l.startswith("#")]
    # Two lines per image; the POINTS2D line can be empty, so don't drop blank lines blindly.
    lines = [l.rstrip("\n") for l in lines]
    while lines and not lines[-1].strip():
        lines.pop()
    for i in range(0, len(lines) - 1, 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        obs = lines[i + 1].split()
        ids = np.array(obs[2::3], dtype=np.int64) if obs else np.zeros(0, np.int64)
        poses[parts[9]] = dict(R=qvec_to_rotmat(qw, qx, qy, qz), t=np.array([tx, ty, tz]),
                               camera_id=int(parts[8]), point3D_ids=np.unique(ids[ids >= 0]))
    return poses


def read_colmap_points3d(points_txt):
    """Returns (id -> row index dict, xyz Nx3, reprojection error N, track length N)"""
    ids, xyz, err, trk = [], [], [], []
    with open(points_txt, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            ids.append(int(p[0]))
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
            err.append(float(p[7]))
            trk.append((len(p) - 8) // 2)
    index = {pid: k for k, pid in enumerate(ids)}
    return index, np.array(xyz).reshape(-1, 3), np.array(err), np.array(trk)


# ---------- 2. Camera model helpers ----------
# COLMAP pixel convention: the centre of the top-left pixel is (0.5, 0.5).

def distortion_delta(x, y, cam):
    k1, k2, p1, p2 = cam["k1"], cam["k2"], cam["p1"], cam["p2"]
    r2 = x * x + y * y
    radial = k1 * r2 + k2 * r2 * r2
    dx = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dy = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return dx, dy


def project(Xc, cam):
    """Camera-frame points -> pixel indices (u, v) with distortion."""
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    dx, dy = distortion_delta(x, y, cam)
    u = cam["fx"] * (x + dx) + cam["cx"] - 0.5
    v = cam["fy"] * (y + dy) + cam["cy"] - 0.5
    return u, v


def pixel_rays(cam):
    """Undistorted normalised coords (x/z, y/z) for every pixel, shape (H, W, 2)."""
    W, H = cam["width"], cam["height"]
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    xd = (uu - cam["cx"]) / cam["fx"]
    yd = (vv - cam["cy"]) / cam["fy"]
    x, y = xd.copy(), yd.copy()
    if any(cam[k] != 0.0 for k in ("k1", "k2", "p1", "p2")):
        for _ in range(20):  # fixed-point inversion of the distortion model
            dx, dy = distortion_delta(x, y, cam)
            x, y = xd - dx, yd - dy
    return np.stack([x, y], axis=-1)


def undistort_maps(cam, rays):
    """Returns f(depth, color) -> (depth, color) resampled onto the ideal pinhole grid (same fx, fy, cx, cy),
    so the TSDF (pinhole-only) sees undistorted images. Depth uses nearest-neighbour to avoid blending edges."""
    import cv2
    if all(cam[k] == 0.0 for k in ("k1", "k2", "p1", "p2")):
        return lambda d, c: (d, c)
    H, W = rays.shape[:2]
    Xc = np.concatenate([rays.reshape(-1, 2), np.ones((H * W, 1))], axis=1)
    u, v = project(Xc, cam)
    mx, my = u.reshape(H, W).astype(np.float32), v.reshape(H, W).astype(np.float32)
    return lambda d, c: (cv2.remap(d, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0),
                         cv2.remap(c, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0))


# ---------- 3. Depth estimation ----------

def load_depth_model(model_name, device):
    from transformers import pipeline
    print(f"Loading {model_name} ...")
    return pipeline(task="depth-estimation", model=model_name, device=0 if device == "cuda" else -1)


def predict_depth(depth_pipe, img, H, W):
    # NOTE: result["depth"] is an 8-bit visualisation normalised to 0..255, NOT depth.
    # The metric prediction is result["predicted_depth"].
    result = depth_pipe(img)
    d = result["predicted_depth"]
    if not torch.is_tensor(d):
        d = torch.as_tensor(np.asarray(d))
    d = d.float().squeeze()
    if tuple(d.shape) != (H, W):
        d = torch.nn.functional.interpolate(d[None, None], size=(H, W), mode="bilinear",
                                            align_corners=False)[0, 0]
    return d.cpu().numpy().astype(np.float32)


def valid_depth_mask(d, depth_min, depth_max, edge_thresh):
    m = np.isfinite(d) & (d > depth_min) & (d < depth_max)
    if edge_thresh > 0 and m.any():
        # Relative depth range in a 3x3 neighbourhood; large => depth discontinuity
        # ("flying pixels" between foreground and background).
        dd = np.where(m, d, np.nan)
        pad = np.pad(dd, 1, mode="edge")
        H, W = d.shape
        stack = np.stack([pad[i:i + H, j:j + W] for i in range(3) for j in range(3)])
        with np.errstate(invalid="ignore"), np.errstate(all="ignore"):
            rng = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
            rel = rng / np.maximum(d, 1e-6)
        m &= np.isfinite(rel) & (rel < edge_thresh)
    return m


# ---------- 4. Depth <-> SfM correspondences and fitting ----------

def sparse_correspondences(pose, cam, pts_index, pts_xyz, pts_err, pts_trk, depth, mask, args):
    """Sparse COLMAP points observed in this frame -> (pred depth d, COLMAP depth z)."""
    d, z, _, _ = sparse_correspondences_uv(pose, cam, pts_index, pts_xyz, pts_err, pts_trk, depth, mask, args)
    return d, z


def sparse_correspondences_uv(pose, cam, pts_index, pts_xyz, pts_err, pts_trk, depth, mask, args):
    """As sparse_correspondences, plus the integer pixel coordinates (u, v) of each correspondence."""
    empty = (np.zeros(0), np.zeros(0), np.zeros(0, np.int64), np.zeros(0, np.int64))
    rows = [pts_index[i] for i in pose["point3D_ids"] if i in pts_index]
    if not rows:
        return empty
    rows = np.array(rows)
    rows = rows[(pts_err[rows] <= args.max_reproj_err) & (pts_trk[rows] >= args.min_track)]
    if len(rows) == 0:
        return empty
    Xc = pts_xyz[rows] @ pose["R"].T + pose["t"]
    Xc = Xc[Xc[:, 2] > 1e-9]
    if len(Xc) == 0:
        return empty
    u, v = project(Xc, cam)
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    H, W = depth.shape
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z = ui[inside], vi[inside], Xc[inside, 2]
    ok = mask[vi, ui]
    return depth[vi[ok], ui[ok]].astype(np.float64), z[ok], ui[ok], vi[ok]


def fit_scale(d, z, inlier_tol):
    """Robust z = a*d in log space: median + MAD rejection, then mean of inliers."""
    r = np.log(z) - np.log(d)
    c = np.median(r)
    inl = np.ones_like(r, bool)
    for _ in range(3):
        mad = 1.4826 * np.median(np.abs(r[inl] - c))
        thr = max(3.0 * mad, np.log1p(inlier_tol))
        inl = np.abs(r - c) <= thr
        if inl.sum() == 0:
            break
        c = np.mean(r[inl])
    return np.exp(c), 0.0, inl


def fit_affine(d, z, inlier_tol, iters=300, seed=0):
    """Robust z = a*d + b: 2-point RANSAC (relative residual threshold), LSQ refine."""
    rng = np.random.default_rng(seed)
    n = len(d)
    best = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(d[i] - d[j]) < 1e-3 * max(d[i], d[j]):
            continue
        a = (z[i] - z[j]) / (d[i] - d[j])
        if a <= 0:
            continue
        b = z[i] - a * d[i]
        inl = np.abs(a * d + b - z) / z <= inlier_tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 2:
        return None
    inl = best
    for _ in range(3):
        A = np.stack([d[inl], np.ones(inl.sum())], axis=1)
        a, b = np.linalg.lstsq(A, z[inl], rcond=None)[0]
        new = np.abs(a * d + b - z) / z <= inlier_tol
        if new.sum() < 2 or np.array_equal(new, inl):
            break
        inl = new
    return a, b, inl


# ---------- 4b. Spatially varying depth correction (experimental, --align spatial) ----------
# Model (log domain):  log(z / d) = beta0 + F(x, y, ld) @ beta_shape
#   x, y = (u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy   normalised image coordinates
#   ld   = log(d / median frame depth)                 relative predicted depth
# beta0 plays the role of the per-frame scale; the shape term is deliberately low-order, ridge-regularised,
# robustly fitted, clipped to the observed depth range and clamped in magnitude.

SPATIAL_MODELS = {
    "const": [],
    "lin_xy": ["x", "y"],
    "logd": ["ld"],
    "lin_xy_logd": ["x", "y", "ld"],
    "poly2_xy": ["x", "y", "xx", "xy", "yy"],
    "poly2_xy_logd": ["x", "y", "xx", "xy", "yy", "ld"],
}


def spatial_features(model, x, y, ld):
    cols = {"x": x, "y": y, "ld": ld, "xx": x * x, "xy": x * y, "yy": y * y}
    names = SPATIAL_MODELS[model]
    return np.stack([cols[n] for n in names], axis=1) if names else np.zeros((len(x), 0))


def robust_ridge(F, t, lam, iters=6):
    """Huber-IRLS fit of t = b0 + F @ b, ridge penalty lam * n * |b|^2 on the shape terms only."""
    n, p = F.shape
    X = np.concatenate([np.ones((n, 1)), F], axis=1)
    P = np.diag([0.0] + [lam * n] * p)
    w = np.ones(n)
    beta = np.zeros(p + 1)
    for _ in range(iters):
        Xw = X * w[:, None]
        beta = np.linalg.solve(X.T @ Xw + P + 1e-12 * np.eye(p + 1), Xw.T @ t)
        r = t - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        c = 1.345 * s
        w = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-12))
    return beta


def spatial_inliers(t, inlier_tol):
    med = np.median(t)
    mad = 1.4826 * np.median(np.abs(t - med))
    return np.abs(t - med) <= max(3.0 * mad, np.log1p(inlier_tol))


def fit_spatial_frame(model, x, y, ld, t, lam, inlier_tol, shared_shape=None):
    """Returns (beta_full, inlier mask). With shared_shape (transfer mode) only the intercept is fitted."""
    inl = spatial_inliers(t, inlier_tol)
    F = spatial_features(model, x[inl], y[inl], ld[inl])
    if shared_shape is not None:
        g = F @ np.asarray(shared_shape) if F.shape[1] else np.zeros(inl.sum())
        b0 = np.median(t[inl] - g)
        return np.concatenate([[b0], shared_shape]), inl
    return robust_ridge(F, t[inl], lam), inl


def spatial_log_factor(model, beta, x, y, ld, clamp):
    """log(corrected / d) = beta0 + clip(F @ beta_shape, +-clamp)."""
    F = spatial_features(model, x, y, ld)
    g = F @ np.asarray(beta[1:]) if F.shape[1] else np.zeros(len(x))
    return beta[0] + np.clip(g, -clamp, clamp)


# ---------- 4c. Multi-view consistency + confidence (experimental, --fusion weighted) ----------
# All maps live on the back-projection grid (every `stride`-th pixel). Only reconstruction-internal signals
# are used: the corrected depth of neighbouring frames, COLMAP poses/intrinsics and sparse points.

def mv_forward_check(ds_i, rays_g, Rc2w_i, C_i, Rw2c_j, C_j, cam_j, ds_j, s):
    """Signed relative residual of frame i's points seen from frame j:
         rel = (z of i's point in j - j's depth at that pixel) / j's depth
       ~0 consistent, < 0 point lies in front of j's surface (free-space conflict), > 0 hidden in j.
       NaN where j has no valid depth / the point projects outside j."""
    rel = np.full(ds_i.shape, np.nan, np.float32)
    m = ds_i > 0
    if not m.any():
        return rel
    z = ds_i[m]; xy = rays_g[m]
    P = np.stack([xy[:, 0] * z, xy[:, 1] * z, z], 1) @ Rc2w_i.T + C_i
    Xc = (P - C_j) @ Rw2c_j.T
    front = Xc[:, 2] > 1e-6
    if not front.any():
        return rel
    u, v = project(Xc[front], cam_j)
    gu, gv = np.round(u / s).astype(np.int64), np.round(v / s).astype(np.int64)
    Hg, Wg = ds_j.shape
    ok = (gu >= 0) & (gu < Wg) & (gv >= 0) & (gv < Hg)
    idx = np.flatnonzero(front)[ok]
    dj = ds_j[gv[ok], gu[ok]]
    good = dj > 0
    vals = np.full(m.sum(), np.nan, np.float32)
    vals[idx[good]] = (Xc[front][ok][good, 2] - dj[good]) / dj[good]
    rel[m] = vals
    return rel


def pose_confidence(poses, names, pts_index, pts_trk, ref=0.5, floor=0.2):
    """Opt-in frame-level pose reliability from the COLMAP model (no ground truth, no extra inputs).
       r_obs = triangulated observations / sequence median,  r_trk = median track length / sequence median
       c_pose = max(floor, sqrt(min(1, r_obs/ref) * min(1, r_trk/ref)))
    Returns {name: (c_pose, n_obs, track_med)}."""
    stats = {}
    for n in names:
        ids = [pts_index[i] for i in poses[n]["point3D_ids"] if i in pts_index]
        stats[n] = (len(ids), float(np.median(pts_trk[ids])) if ids else 0.0)
    med_obs = max(float(np.median([v[0] for v in stats.values()])), 1e-9)
    med_trk = max(float(np.median([v[1] for v in stats.values()])), 1e-9)
    out = {}
    for n, (n_obs, trk) in stats.items():
        c = np.sqrt(min(1.0, n_obs / med_obs / ref) * min(1.0, trk / med_trk / ref))
        out[n] = (float(max(floor, c)), n_obs, trk)
    return out


def frame_confidence(ds, rels, sparse_grid, frame_res, s, width, cfg, pose_c=1.0):
    """Interpretable per-pixel confidence in [0, 1] and a hard-reject mask.
       mv     : multi-view agreement (exp(-0.5 (rel/tau)^2) when visible & not hidden; 0.5 when unknown / hidden)
       edge   : local 3x3 relative depth range (exp(-0.5 (g/g0)^2))
       sparse : 0.5 + 0.5 exp(-0.5 (dist/rho)^2), dist to nearest COLMAP sparse point in this frame
       frame  : exp(-0.5 (frame alignment residual / r0)^2)
       reject : more neighbours report a strong free-space conflict (rel < -tau_hard) than confirm (|rel| <= tau)."""
    from scipy.ndimage import distance_transform_edt
    m = ds > 0
    sig = cfg["signals"]
    parts = {}
    reject = np.zeros(ds.shape, bool)
    if rels:
        cs, n_conf, n_ok = [], np.zeros(ds.shape, np.int16), np.zeros(ds.shape, np.int16)
        for r in rels:
            vis = np.isfinite(r)
            c = np.full(ds.shape, 0.5, np.float32)
            sel = vis & (r <= cfg["tau"])
            c[sel] = np.exp(-0.5 * (r[sel] / cfg["tau"]) ** 2)
            cs.append(c)
            n_conf += (vis & (r < -cfg["tau_hard"])).astype(np.int16)
            n_ok += (vis & (np.abs(r) <= cfg["tau"])).astype(np.int16)
        parts["mv"] = np.mean(cs, axis=0)
        reject = (n_conf > n_ok) & m
    else:
        parts["mv"] = np.full(ds.shape, 0.5, np.float32)
    dd = np.where(m, ds, np.nan)
    pad = np.pad(dd, 1, mode="edge")
    Hh, Ww = ds.shape
    st = np.stack([pad[i:i + Hh, j:j + Ww] for i in range(3) for j in range(3)])
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN neighbourhoods (invalid pixels) are expected
        g = (np.nanmax(st, 0) - np.nanmin(st, 0)) / np.maximum(ds, 1e-6)
    parts["edge"] = np.where(np.isfinite(g), np.exp(-0.5 * (g / cfg["edge_g0"]) ** 2), 0).astype(np.float32)
    if sparse_grid is not None and sparse_grid.any():
        dist = distance_transform_edt(~sparse_grid) * s
        rho = cfg["sparse_radius"] * width
        parts["sparse"] = (0.5 + 0.5 * np.exp(-0.5 * (dist / rho) ** 2)).astype(np.float32)
    else:
        parts["sparse"] = np.full(ds.shape, 0.5, np.float32)
    fr = frame_res if np.isfinite(frame_res) else cfg["frame_r0"]
    parts["frame"] = np.full(ds.shape, np.exp(-0.5 * (fr / cfg["frame_r0"]) ** 2), np.float32)
    if "pose" in sig:
        parts["pose"] = np.full(ds.shape, pose_c, np.float32)
    conf = np.ones(ds.shape, np.float32)
    for k_ in sig:
        conf *= parts[k_]
    conf[~m] = 0
    if "mv" in sig:
        conf[reject] = 0
    conf[conf < cfg["conf_min"]] = 0
    return conf, parts, reject


def weighted_voxel_fusion(P, C, w, voxel, w_min):
    """Confidence-weighted mean position / colour per voxel; voxels with total weight < w_min are dropped."""
    key = np.floor(P / voxel).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    _, inv = np.unique(lin, return_inverse=True)
    W = np.bincount(inv, w)
    Pm = np.stack([np.bincount(inv, w * P[:, i]) for i in range(3)], 1)
    Cm = np.stack([np.bincount(inv, w * C[:, i]) for i in range(3)], 1)
    keep = W >= w_min
    return Pm[keep] / W[keep, None], Cm[keep] / W[keep, None], W[keep]


def median_filter_1d(x, valid, win):
    """Median of x over a sliding window, using only entries flagged valid."""
    if win <= 1:
        return x.copy()
    h = win // 2
    out = x.copy()
    for i in range(len(x)):
        lo, hi = max(0, i - h), min(len(x), i + h + 1)
        w = x[lo:hi][valid[lo:hi]]
        if len(w):
            out[i] = np.median(w)
    return out


# ---------- 5. Back-projection ----------

def backproject(depth, color, rays, R_c2w, C, mask, stride):
    """Vectorised: depth (H,W) + per-pixel rays -> world points (N,3) and colours (N,3)."""
    s = (slice(None, None, stride), slice(None, None, stride))
    m = mask[s]
    z = depth[s][m]
    xy = rays[s][m]
    P = np.stack([xy[:, 0] * z, xy[:, 1] * z, z], axis=1)
    Pw = P @ R_c2w.T + C
    return Pw.astype(np.float32), (color[s][m] / 255.0).astype(np.float32)


# ---------- 6. Main pipeline ----------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--frames", required=True, help="Folder of frames (same ones COLMAP used)")
    ap.add_argument("--colmap_dir", required=True, help="Folder with cameras.txt, images.txt, points3D.txt")
    ap.add_argument("--out", default="output")
    ap.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size in output units (metres unless --align none)")
    ap.add_argument("--stride", "--pixel_stride", dest="stride", type=int, default=2,
                    help="PIXEL stride for back-projection (does not skip frames)")
    ap.add_argument("--frame_step", type=int, default=1, help="Use every Nth registered frame")
    ap.add_argument("--align", choices=["none", "global", "scale", "affine", "spatial"], default="scale",
                    help="Per-frame depth alignment model (see module docstring)")
    ap.add_argument("--smooth_window", type=int, default=1,
                    help="Temporal median window over per-frame fit parameters (1 = off)")
    ap.add_argument("--min_corr", type=int, default=20, help="Min sparse correspondences for a per-frame fit")
    ap.add_argument("--min_inlier_frac", type=float, default=0.5)
    ap.add_argument("--inlier_tol", type=float, default=0.10, help="Relative depth residual counted as inlier")
    ap.add_argument("--max_scale_dev", type=float, default=1.5,
                    help="Reject per-frame scale outside [1/x, x] of the global scale")
    ap.add_argument("--max_reproj_err", type=float, default=2.0, help="Max COLMAP reprojection error of sparse points (px)")
    ap.add_argument("--min_track", type=int, default=3, help="Min track length of sparse points")
    ap.add_argument("--depth_min", type=float, default=0.1, help="Valid range of raw network depth (m)")
    ap.add_argument("--depth_max", type=float, default=20.0)
    ap.add_argument("--depth_trunc", type=float, default=8.0, help="Drop aligned depth beyond this (output units)")
    ap.add_argument("--edge_thresh", type=float, default=0.10,
                    help="Drop pixels whose 3x3 relative depth range exceeds this (0 = off)")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    ap.add_argument("--recompute_depth", action="store_true", help="Ignore cached depth predictions")
    ap.add_argument("--depth_cache", default=None, help="Depth cache folder (default: <out>/depth_pred)")
    ap.add_argument("--no_viewer", action="store_true", help="Do not open the Open3D viewer at the end")
    # ---- fusion ----
    ap.add_argument("--fusion", choices=["voxel", "tsdf", "weighted"], default="voxel",
                    help="voxel = concatenate back-projected points + voxel downsample (baseline); "
                         "tsdf = Open3D ScalableTSDFVolume integration of the aligned depth maps")
    ap.add_argument("--tsdf_voxel", type=float, default=0.02, help="TSDF voxel length (output units)")
    ap.add_argument("--tsdf_trunc", type=float, default=0.08, help="TSDF truncation distance (output units)")
    ap.add_argument("--tsdf_mesh", action="store_true", help="Also extract and save a triangle mesh (scene_mesh.ply)")
    # ---- confidence-weighted fusion (experimental, --fusion weighted); thresholds fixed a priori ----
    ap.add_argument("--conf_signals", default="mv,edge,sparse,frame",
                    help="Comma list of confidence factors: mv (multi-view), edge, sparse, frame, pose (opt-in, "
                         "experimental); 'none' = uniform weights")
    ap.add_argument("--conf_pose_ref", type=float, default=0.5,
                    help="pose factor: support ratio (vs sequence median) at which a frame gets full weight")
    ap.add_argument("--conf_pose_floor", type=float, default=0.2, help="pose factor: minimum frame weight")
    ap.add_argument("--conf_tau", type=float, default=0.05, help="Relative multi-view agreement scale")
    ap.add_argument("--conf_tau_hard", type=float, default=0.15, help="Free-space conflict counted as strong beyond this")
    ap.add_argument("--conf_edge", type=float, default=0.05, help="Relative 3x3 depth range scale for the edge factor")
    ap.add_argument("--conf_sparse_radius", type=float, default=0.10, help="Sparse-support radius as a fraction of image width")
    ap.add_argument("--conf_frame_res", type=float, default=0.05, help="Frame alignment residual scale")
    ap.add_argument("--conf_min", type=float, default=0.05, help="Points below this confidence are dropped")
    ap.add_argument("--conf_max_gap", type=int, default=3, help="Search up to this many frames away for a neighbour")
    ap.add_argument("--conf_min_overlap", type=float, default=0.2, help="Min fraction of pixels a neighbour must see")
    ap.add_argument("--weighted_wmin", type=float, default=0.5, help="Drop voxels whose total confidence is below this")
    ap.add_argument("--save_confidence", action="store_true", help="Save per-frame confidence / consistency maps (npz)")
    # ---- experimental spatial depth correction (--align spatial) ----
    ap.add_argument("--spatial_mode", choices=["perframe", "transfer"], default="perframe",
                    help="perframe: fit the low-order field per frame from its own sparse points; "
                         "transfer: shared shape from a calibration sequence (needs --spatial_params)")
    ap.add_argument("--spatial_model", choices=list(SPATIAL_MODELS), default="lin_xy_logd")
    ap.add_argument("--spatial_lambda", type=float, default=0.1, help="Ridge penalty (per point) on shape terms")
    ap.add_argument("--spatial_clamp", type=float, default=0.2, help="Max |log| of the shape factor (0.2 = +-22%%)")
    ap.add_argument("--spatial_min_pts", type=int, default=60, help="Min sparse points for a spatial fit")
    ap.add_argument("--spatial_params", default=None,
                    help="JSON from fit_spatial_correction.py (mode/model/lam/clamp/shared_shape); overrides the above")
    ap.add_argument("--save_aligned_depth", action="store_true",
                    help="Save aligned depth maps (half resolution, float16) for depth evaluation")
    # ---- optional global metric calibration (OFF by default) ----
    ap.add_argument("--metric_scale", type=float, default=None,
                    help="Multiply the output frame (poses + depth) by this factor, e.g. derived with "
                         "calibrate_scale.py from OTHER sequences. Never tune it on the test sequence.")
    ap.add_argument("--metric_scale_file", default=None, help="JSON written by calibrate_scale.py (key: metric_scale)")
    ap.add_argument("--mask_dir", default=None,
                    help="COLMAP-style masks (<image name>.png, black = ignore); masked pixels are not fused")
    args = ap.parse_args()

    t_start = time.time()
    timing = {}
    os.makedirs(args.out, exist_ok=True)
    cache_dir = args.depth_cache or os.path.join(args.out, "depth_pred")
    os.makedirs(cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cams = read_colmap_cameras(os.path.join(args.colmap_dir, "cameras.txt"))
    poses = read_colmap_images(os.path.join(args.colmap_dir, "images.txt"))
    pts_index, pts_xyz, pts_err, pts_trk = read_colmap_points3d(os.path.join(args.colmap_dir, "points3D.txt"))
    names = sorted(poses, key=lambda n: (name_to_timestamp(n) if any(c.isdigit() for c in n) else 0, n))
    names = names[::max(1, args.frame_step)]
    print(f"COLMAP: {len(poses)} registered frames ({len(names)} used), {len(pts_xyz)} sparse points, "
          f"{len(cams)} camera(s)")
    rays_cache = {cid: pixel_rays(c) for cid, c in cams.items()}

    # ---- pass 1: depth prediction + sparse correspondences ----
    t0 = time.time()
    depth_pipe = None
    frames = []
    for name in names:
        pose = poses[name]
        cam = cams[pose["camera_id"]]
        H, W = cam["height"], cam["width"]
        rec = dict(name=name, status="ok")
        frame_path = os.path.join(args.frames, name)
        if not os.path.exists(frame_path):
            rec["status"] = "missing_image"
            frames.append(rec)
            continue
        cache = os.path.join(cache_dir, os.path.splitext(os.path.basename(name))[0] + ".npy")
        if os.path.exists(cache) and not args.recompute_depth:
            depth = np.load(cache)
        else:
            if depth_pipe is None:
                depth_pipe = load_depth_model(args.model, device)
            depth = predict_depth(depth_pipe, Image.open(frame_path).convert("RGB"), H, W)
            np.save(cache, depth)
        if depth.shape != (H, W):
            rec["status"] = f"depth_shape_{depth.shape}"
            frames.append(rec)
            continue
        mask = valid_depth_mask(depth, args.depth_min, args.depth_max, args.edge_thresh)
        if args.mask_dir:
            mp = os.path.join(args.mask_dir, name + ".png")
            if os.path.exists(mp):
                user_mask = np.array(Image.open(mp).convert("L"))
                if user_mask.shape == mask.shape:
                    mask &= user_mask > 0
        d_s, z_s, u_s, v_s = sparse_correspondences_uv(pose, cam, pts_index, pts_xyz, pts_err, pts_trk, depth, mask, args)
        rec.update(depth=depth, mask=mask, d_s=d_s, z_s=z_s, n_valid_px=int(mask.sum()), n_corr=len(d_s),
                   x_s=(u_s + 0.5 - cam["cx"]) / cam["fx"], y_s=(v_s + 0.5 - cam["cy"]) / cam["fy"],
                   dmed=float(np.median(depth[mask])) if mask.any() else np.nan)
        if rec["n_valid_px"] == 0:
            rec["status"] = "empty_depth"
        frames.append(rec)
    timing["depth_and_correspondences_s"] = time.time() - t0

    # ---- global scale: COLMAP units per predicted metre ----
    per_frame_logr = []
    for f in frames:
        if f["status"] == "ok" and f["n_corr"] >= args.min_corr:
            f["raw_scale"] = float(np.exp(np.median(np.log(f["z_s"]) - np.log(f["d_s"]))))
            per_frame_logr.append(np.log(f["raw_scale"]))
    if not per_frame_logr:
        sys.exit("!! No frame has enough sparse correspondences to estimate the depth<->COLMAP scale.")
    per_frame_logr = np.array(per_frame_logr)
    S = float(np.exp(np.median(per_frame_logr)))
    spread = float(1.4826 * np.median(np.abs(per_frame_logr - np.log(S))))
    if args.align == "none":
        unit_scale, out_units = 1.0, "COLMAP (arbitrary) units; depth NOT aligned"
    else:
        unit_scale, out_units = S, "COLMAP frame rescaled by 1/S (depth network's median metric scale)"
    # Optional external calibration k: applied AFTER alignment to poses and depth alike, so the
    # per-frame fits (and their sanity bounds) are unchanged; only the output frame is rescaled.
    k = 1.0
    if args.metric_scale_file:
        k = float(json.load(open(args.metric_scale_file))["metric_scale"])
    if args.metric_scale is not None:
        k = float(args.metric_scale)
    if not np.isfinite(k) or k <= 0:
        sys.exit(f"!! invalid metric scale {k}")
    if k != 1.0:
        out_units += f", then multiplied by external calibration k = {k:.4f}"
    print(f"\nGlobal scale S = {S:.4f} COLMAP units per predicted metre "
          f"(1 COLMAP unit = {1/S:.4f} m); per-frame scale spread (robust std of log) = {spread:.3f}")

    # ---- per-frame alignment in the rescaled frame ----
    for f in frames:
        f.update(a=1.0, b=0.0, n_inliers=0, n_outliers=0, med_rel_res=np.nan, fit_ok=False)
        if f["status"] != "ok":
            continue
        d, z = f["d_s"], f["z_s"] / unit_scale
        if args.align in ("none", "global"):
            if len(d):
                res = np.abs(d - z) / z
                f.update(med_rel_res=float(np.median(res)), n_inliers=int((res <= args.inlier_tol).sum()))
                f["n_outliers"] = len(d) - f["n_inliers"]
            continue
        if len(d) < args.min_corr:
            f["status"] = "fallback_few_corr"
            continue
        fit = fit_scale(d, z, args.inlier_tol) if args.align in ("scale", "spatial") else fit_affine(d, z, args.inlier_tol)
        if fit is None:
            f["status"] = "fallback_fit_failed"
            continue
        a, b, inl = fit
        f.update(n_inliers=int(inl.sum()), n_outliers=int((~inl).sum()), fit_a=float(a), fit_b=float(b),
                 sparse_z_median=float(np.median(z)))
        # Implied scale at the median sparse depth, relative to the global scale.
        dm = np.median(d)
        implied = (a * dm + b) / dm
        if inl.sum() < args.min_inlier_frac * len(d):
            f["status"] = "fallback_low_inliers"
        elif not (1 / args.max_scale_dev <= implied <= args.max_scale_dev):
            f["status"] = "fallback_scale_out_of_range"
        elif args.align == "affine" and (a * np.percentile(d, 5) + b <= 0 or abs(b) > 0.5 * np.median(z)):
            f["status"] = "fallback_bad_shift"
        else:
            f.update(a=float(a), b=float(b), fit_ok=True)

    if args.align in ("scale", "affine") and args.smooth_window > 1:
        ok = np.array([f.get("fit_ok", False) for f in frames])
        A = np.array([np.log(f["a"]) for f in frames])
        B = np.array([f["b"] for f in frames])
        A_s, B_s = median_filter_1d(A, ok, args.smooth_window), median_filter_1d(B, ok, args.smooth_window)
        for f, la, bb in zip(frames, A_s, B_s):
            if f["status"] == "ok" or f["status"].startswith("fallback"):
                f["a"], f["b"] = float(np.exp(la)), float(bb)

    # ---- experimental spatial correction on top of a successful per-frame scale fit ----
    sp = None
    if args.align == "spatial":
        sp = dict(mode=args.spatial_mode, model=args.spatial_model, lam=args.spatial_lambda,
                  clamp=args.spatial_clamp, shared_shape=None, source="command line")
        if args.spatial_params:
            cfg = json.load(open(args.spatial_params))
            sp.update({k: cfg[k] for k in ("mode", "model", "lam", "clamp", "shared_shape") if k in cfg})
            sp["source"] = os.path.abspath(args.spatial_params)
        if sp["mode"] == "transfer" and sp["shared_shape"] is None:
            sys.exit("!! --spatial_mode transfer needs --spatial_params with a learned shared_shape")
        n_sp = 0
        for f in frames:
            f["spatial_ok"] = False
            if f["status"] != "ok" or not f.get("fit_ok") or len(f["d_s"]) < max(args.min_corr, args.spatial_min_pts):
                continue
            t = np.log(f["z_s"] / unit_scale) - np.log(f["d_s"])
            ld = np.log(f["d_s"] / f["dmed"])
            beta, inl = fit_spatial_frame(sp["model"], f["x_s"], f["y_s"], ld, t, sp["lam"], args.inlier_tol,
                                          sp["shared_shape"] if sp["mode"] == "transfer" else None)
            if not np.all(np.isfinite(beta)) or not (1 / args.max_scale_dev <= np.exp(beta[0]) <= args.max_scale_dev):
                continue
            f.update(spatial_ok=True, spatial_beta=[float(b) for b in beta],
                     spatial_ld_range=(float(np.percentile(ld, 1)), float(np.percentile(ld, 99))))
            n_sp += 1
        print(f"Spatial correction ({sp['mode']}, {sp['model']}, lambda {sp['lam']}, clamp {sp['clamp']}): "
              f"{n_sp} frames corrected, others keep the per-frame scale")

    def sparse_prediction(f):
        """Aligned (pre-calibration) depth predicted at the sparse points, for residual diagnostics."""
        if f.get("spatial_ok"):
            ld = np.clip(np.log(f["d_s"] / f["dmed"]), *f["spatial_ld_range"])
            return f["d_s"] * np.exp(spatial_log_factor(sp["model"], f["spatial_beta"], f["x_s"], f["y_s"], ld, sp["clamp"]))
        return f["a"] * f["d_s"] + f["b"]

    # residuals after the final (possibly smoothed) alignment
    for f in frames:
        if f["status"] in ("ok",) or f["status"].startswith("fallback"):
            if len(f["d_s"]):
                z = f["z_s"] / unit_scale
                res = np.abs(sparse_prediction(f) - z) / z
                f["med_rel_res"] = float(np.median(res))
                if args.align in ("scale", "affine") and not f["fit_ok"]:
                    f["n_inliers"] = int((res <= args.inlier_tol).sum())
                    f["n_outliers"] = len(res) - f["n_inliers"]

    # ---- pass 2: back-projection (voxel) or TSDF integration ----
    t0 = time.time()
    all_pts, all_cols, traj = [], [], []
    wf = []  # per-frame grids for --fusion weighted
    grid_cache = {}
    if args.save_aligned_depth:
        os.makedirs(os.path.join(args.out, "aligned_depth"), exist_ok=True)
    tsdf = None
    if args.fusion == "tsdf":
        tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=args.tsdf_voxel, sdf_trunc=args.tsdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        tsdf_intr = {cid: o3d.camera.PinholeCameraIntrinsic(c["width"], c["height"], c["fx"], c["fy"],
                                                           c["cx"] - 0.5, c["cy"] - 0.5) for cid, c in cams.items()}
        undist = {cid: undistort_maps(c, rays_cache[cid]) for cid, c in cams.items()}
    print(f"\n{'frame':<22}{'valid_px':>9}{'corr':>6}{'raw_S':>8}{'a':>8}{'b':>8}{'inl':>6}{'out':>5}"
          f"{'medres':>8}{'points':>8}  status")
    for f in frames:
        f["n_points"] = 0
        pose = poses[f["name"]]
        R_c2w = pose["R"].T
        C = k * (-R_c2w @ pose["t"] / unit_scale)
        traj.append((f["name"], C, R_c2w))
        if f["status"] == "ok" or f["status"].startswith("fallback"):
            if f.get("spatial_ok"):
                cam_ = cams[pose["camera_id"]]
                if pose["camera_id"] not in grid_cache:
                    Hh, Ww = f["depth"].shape
                    uu, vv = np.meshgrid(np.arange(Ww) + 0.5, np.arange(Hh) + 0.5)
                    grid_cache[pose["camera_id"]] = (((uu - cam_["cx"]) / cam_["fx"]).ravel(), ((vv - cam_["cy"]) / cam_["fy"]).ravel())
                gx, gy = grid_cache[pose["camera_id"]]
                dep = f["depth"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ldm = np.clip(np.log(np.maximum(dep, 1e-6) / f["dmed"]), *f["spatial_ld_range"]).ravel()
                lf = spatial_log_factor(sp["model"], f["spatial_beta"], gx, gy, ldm, sp["clamp"]).reshape(dep.shape)
                aligned = k * dep * np.exp(lf)
            else:
                aligned = k * (f["a"] * f["depth"] + f["b"])
            m = f["mask"] & np.isfinite(aligned) & (aligned > 0) & (aligned <= args.depth_trunc)
            if args.save_aligned_depth:
                np.save(os.path.join(args.out, "aligned_depth", os.path.splitext(os.path.basename(f["name"]))[0] + ".npy"),
                        np.where(m, aligned, 0)[::2, ::2].astype(np.float16))
            if m.any():
                img = np.array(Image.open(os.path.join(args.frames, f["name"])).convert("RGB"))
                if img.shape[:2] != aligned.shape:
                    img = np.array(Image.fromarray(img).resize(aligned.shape[::-1]))
                if args.fusion == "weighted":
                    s_ = args.stride
                    cam_w = cams[pose["camera_id"]]
                    wf.append(dict(f=f, ds=np.where(m, aligned, 0)[::s_, ::s_].astype(np.float32),
                                   col=np.ascontiguousarray(img[::s_, ::s_]), Rc2w=R_c2w, C=C, Rw2c=pose["R"],
                                   cam=cam_w, cid=pose["camera_id"],
                                   su=np.round((f["x_s"] * cam_w["fx"] + cam_w["cx"] - 0.5) / s_).astype(np.int64),
                                   sv=np.round((f["y_s"] * cam_w["fy"] + cam_w["cy"] - 0.5) / s_).astype(np.int64)))
                    f["n_points"] = int((wf[-1]["ds"] > 0).sum())
                elif tsdf is None:
                    P, Cc = backproject(aligned, img, rays_cache[pose["camera_id"]], R_c2w, C, m, args.stride)
                    ok = np.isfinite(P).all(axis=1)
                    all_pts.append(P[ok]); all_cols.append(Cc[ok])
                    f["n_points"] = int(ok.sum())
                else:
                    d_pin, c_pin = undist[pose["camera_id"]](np.where(m, aligned, 0).astype(np.float32), img)
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        o3d.geometry.Image(np.ascontiguousarray(c_pin)), o3d.geometry.Image(d_pin),
                        depth_scale=1.0, depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False)
                    extr = np.eye(4)
                    extr[:3, :3] = pose["R"]
                    extr[:3, 3] = -pose["R"] @ C
                    tsdf.integrate(rgbd, tsdf_intr[pose["camera_id"]], extr)
                    f["n_points"] = int((d_pin > 0).sum())
        print(f"{os.path.basename(f['name']):<22}{f.get('n_valid_px', 0):>9}{f.get('n_corr', 0):>6}"
              f"{f.get('raw_scale', np.nan):>8.3f}{f['a']:>8.3f}{f['b']:>8.3f}{f['n_inliers']:>6}"
              f"{f['n_outliers']:>5}{f['med_rel_res']:>8.3f}{f['n_points']:>8}  {f['status']}")
        f.pop("depth", None); f.pop("mask", None)
    timing["backprojection_s"] = time.time() - t0

    # ---- multi-view consistency + confidence (weighted fusion only) ----
    if args.fusion == "weighted":
        t0 = time.time()
        s_ = args.stride
        sig_list = [] if args.conf_signals.strip().lower() in ("", "none") else \
            [x.strip() for x in args.conf_signals.split(",") if x.strip()]
        cfg = dict(signals=sig_list, tau=args.conf_tau,
                   tau_hard=args.conf_tau_hard, edge_g0=args.conf_edge, sparse_radius=args.conf_sparse_radius,
                   frame_r0=args.conf_frame_res, conf_min=args.conf_min)
        rays_g = {cid: r[::s_, ::s_] for cid, r in rays_cache.items()}
        pose_conf = {}
        if "pose" in sig_list:
            pose_conf = pose_confidence(poses, [e["f"]["name"] for e in wf], pts_index, pts_trk,
                                        args.conf_pose_ref, args.conf_pose_floor)
        if args.save_confidence:
            os.makedirs(os.path.join(args.out, "confidence"), exist_ok=True)
        wP, wC, wW = [], [], []
        for i, e in enumerate(wf):
            f = e["f"]
            valid = e["ds"] > 0
            rels, used = [], []
            if "mv" in cfg["signals"]:
                for direction in (-1, 1):
                    for gap in range(1, args.conf_max_gap + 1):
                        j = i + direction * gap
                        if j < 0 or j >= len(wf):
                            break
                        e2 = wf[j]
                        rel = mv_forward_check(e["ds"], rays_g[e["cid"]], e["Rc2w"], e["C"], e2["Rw2c"], e2["C"],
                                               e2["cam"], e2["ds"], s_)
                        ov = float(np.isfinite(rel)[valid].mean()) if valid.any() else 0.0
                        if ov >= args.conf_min_overlap:
                            rels.append(rel); used.append(j - i)
                            break
            sg = np.zeros(e["ds"].shape, bool)
            ok_s = (e["su"] >= 0) & (e["su"] < sg.shape[1]) & (e["sv"] >= 0) & (e["sv"] < sg.shape[0])
            sg[e["sv"][ok_s], e["su"][ok_s]] = True
            pc = pose_conf.get(f["name"])
            if pc:
                f.update(pose_conf=pc[0], n_obs=pc[1], track_med=pc[2])
            conf, parts, reject = frame_confidence(e["ds"], rels, sg, f.get("med_rel_res", np.nan), s_,
                                                   e["cam"]["width"], cfg, pc[0] if pc else 1.0)
            keep = conf > 0
            z = e["ds"][keep]; xy = rays_g[e["cid"]][keep]
            P = np.stack([xy[:, 0] * z, xy[:, 1] * z, z], 1) @ e["Rc2w"].T + e["C"]
            wP.append(P.astype(np.float32)); wC.append((e["col"][keep] / 255.0).astype(np.float32)); wW.append(conf[keep])
            f.update(n_points=int(keep.sum()), conf_mean=float(conf[valid].mean()) if valid.any() else 0.0,
                     rejected_frac=float((valid & ~keep).mean() / max(valid.mean(), 1e-9)),
                     mv_neighbours=",".join(str(u) for u in used),
                     mv_consistent_frac=float(np.nanmean([(np.abs(r[valid]) <= cfg["tau"]).mean() for r in rels])) if rels else np.nan)
            if args.save_confidence:
                np.savez_compressed(os.path.join(args.out, "confidence", os.path.splitext(os.path.basename(f["name"]))[0] + ".npz"),
                                    conf=conf.astype(np.float16), reject=reject, depth=e["ds"].astype(np.float16),
                                    **{f"c_{k_}": v.astype(np.float16) for k_, v in parts.items()},
                                    **{f"rel_{u:+d}": r.astype(np.float16) for u, r in zip(used, rels)})
            e["col"] = None
            old = i - args.conf_max_gap
            if old >= 0:
                wf[old]["ds"] = None  # no longer needed as a neighbour
        timing["confidence_s"] = time.time() - t0

    # ---- fuse / denoise / voxelize ----
    t0 = time.time()
    if args.fusion == "weighted":
        n_raw = int(sum(len(p) for p in wP))
        if n_raw == 0:
            sys.exit("!! No confident points survived. Check depth range / confidence settings.")
        Pm, Cm, Wv = weighted_voxel_fusion(np.concatenate(wP).astype(np.float64), np.concatenate(wC).astype(np.float64),
                                           np.concatenate(wW).astype(np.float64), args.voxel_size, args.weighted_wmin)
        del wP, wC, wW
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(Pm)
        merged.colors = o3d.utility.Vector3dVector(np.clip(Cm, 0, 1))
    elif tsdf is None:
        n_raw = int(sum(len(p) for p in all_pts))
        if n_raw == 0:
            sys.exit("!! No points were back-projected from any frame. Check depth range / alignment diagnostics.")
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(np.concatenate(all_pts).astype(np.float64))
        merged.colors = o3d.utility.Vector3dVector(np.concatenate(all_cols).astype(np.float64))
        del all_pts, all_cols
        merged = merged.voxel_down_sample(args.voxel_size)
    else:
        n_raw = int(sum(f["n_points"] for f in frames))  # depth pixels integrated
        if n_raw == 0:
            sys.exit("!! No depth pixels were integrated into the TSDF. Check depth range / alignment diagnostics.")
        merged = tsdf.extract_point_cloud()
        if args.tsdf_mesh:
            mesh = tsdf.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(os.path.join(args.out, "scene_mesh.ply"), mesh)
        del tsdf
    n_vox = len(merged.points)
    if n_vox > 20:
        merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    timing["fusion_s"] = time.time() - t0

    out_path = os.path.join(args.out, "scene.ply")
    o3d.io.write_point_cloud(out_path, merged)

    # ---- diagnostics files ----
    cols = ["name", "status", "n_valid_px", "n_corr", "raw_scale", "a", "b", "n_inliers", "n_outliers",
            "med_rel_res", "n_points", "fit_a", "fit_b", "sparse_z_median", "spatial_ok", "spatial_beta",
            "conf_mean", "rejected_frac", "mv_neighbours", "mv_consistent_frac", "pose_conf", "n_obs", "track_med"]
    with open(os.path.join(args.out, "alignment_diagnostics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for f in frames:
            w.writerow([f.get(c, "") for c in cols])
    with open(os.path.join(args.out, "trajectory_fused.txt"), "w") as fh:
        for name, C, Rc in traj:
            qx, qy, qz, qw = rotmat_to_qvec(Rc)
            fh.write(f"{name_to_timestamp(name):.6f} {C[0]:.6f} {C[1]:.6f} {C[2]:.6f} "
                     f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    statuses = {}
    for f in frames:
        statuses[f["status"]] = statuses.get(f["status"], 0) + 1
    used = [f for f in frames if f["n_points"] > 0]
    a_vals = np.array([f["a"] for f in used]) if used else np.array([np.nan])
    raw = np.array([f["raw_scale"] for f in frames if "raw_scale" in f])
    timing["total_s"] = time.time() - t_start
    summary = dict(
        frames_registered=len(poses), frames_considered=len(frames), frames_fused=len(used),
        status_counts=statuses, align=args.align, output_units=out_units,
        global_scale_S_colmap_units_per_pred_metre=S, metres_per_colmap_unit=1 / S,
        per_frame_raw_scale_rel_to_S_p5_p50_p95=[float(x) for x in np.percentile(raw / S, [5, 50, 95])],
        per_frame_log_scale_robust_std=spread,
        a_p5_p50_p95=[float(x) for x in np.nanpercentile(a_vals, [5, 50, 95])],
        median_corr_per_frame=float(np.median([f.get("n_corr", 0) for f in frames])),
        median_rel_residual_after_alignment=float(np.nanmedian([f["med_rel_res"] for f in used])) if used else None,
        points_backprojected=n_raw, points_after_voxel=n_vox, points_final=len(merged.points),
        fusion=args.fusion,
        fusion_params=({"tsdf_voxel": args.tsdf_voxel, "tsdf_trunc": args.tsdf_trunc, "depth_trunc": args.depth_trunc}
                       if args.fusion == "tsdf" else {"voxel_size": args.voxel_size, "pixel_stride": args.stride,
                                                      "depth_trunc": args.depth_trunc,
                                                      **({"conf_signals": args.conf_signals, "conf_tau": args.conf_tau,
                                                          "conf_tau_hard": args.conf_tau_hard, "conf_edge": args.conf_edge,
                                                          "conf_sparse_radius": args.conf_sparse_radius,
                                                          "conf_frame_res": args.conf_frame_res, "conf_min": args.conf_min,
                                                          "weighted_wmin": args.weighted_wmin,
                                                          "conf_pose_ref": args.conf_pose_ref,
                                                          "conf_pose_floor": args.conf_pose_floor}
                                                         if args.fusion == "weighted" else {})}),
        metric_scale_applied=k,
        spatial=(dict(sp, shared_shape=sp["shared_shape"], frames_corrected=int(sum(bool(f.get("spatial_ok")) for f in frames)))
                 if sp else None),
        timing_s=timing, args=vars(args),
    )
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n===== Reconstruction summary =====")
    print(f"Frames: {len(poses)} registered, {len(frames)} considered, {len(used)} fused; status {statuses}")
    print(f"Alignment: --align {args.align}; output frame: {out_units}")
    print(f"Global scale S = {S:.4f} COLMAP units / predicted m  ->  1 COLMAP unit = {1/S:.4f} m")
    print(f"Per-frame raw scale / S (p5, p50, p95): "
          f"{', '.join(f'{x:.3f}' for x in summary['per_frame_raw_scale_rel_to_S_p5_p50_p95'])}")
    print(f"Per-frame a (p5, p50, p95): {', '.join(f'{x:.3f}' for x in summary['a_p5_p50_p95'])}")
    print(f"Median sparse correspondences/frame: {summary['median_corr_per_frame']:.0f}; "
          f"median relative residual after alignment: {summary['median_rel_residual_after_alignment']:.4f}")
    if args.fusion == "tsdf":
        print(f"Fusion: TSDF (voxel {args.tsdf_voxel}, trunc {args.tsdf_trunc}, depth_trunc {args.depth_trunc}); "
              f"{n_raw} depth px integrated -> {n_vox} surface points -> {len(merged.points)} after outlier removal")
    elif args.fusion == "weighted":
        cm = [f["conf_mean"] for f in frames if "conf_mean" in f]
        rj = [f["rejected_frac"] for f in frames if "rejected_frac" in f]
        print(f"Fusion: confidence-weighted voxel ({args.conf_signals}); mean confidence {np.mean(cm):.3f}, "
              f"rejected {np.mean(rj) * 100:.1f}% of valid px; {n_raw} confident points -> {n_vox} voxels -> "
              f"{len(merged.points)} after outlier removal ({timing.get('confidence_s', 0):.1f} s for consistency/confidence)")
    else:
        print(f"Points: {n_raw} back-projected -> {n_vox} after {args.voxel_size} voxel -> "
              f"{len(merged.points)} after outlier removal")
    if k != 1.0:
        print(f"External metric calibration applied: k = {k:.4f}")
    print(f"Runtime: {timing['total_s']:.1f} s (depth+corr {timing['depth_and_correspondences_s']:.1f}, "
          f"backproj {timing['backprojection_s']:.1f}, fusion {timing['fusion_s']:.1f})")
    print(f"Saved {out_path}")

    if not args.no_viewer:
        o3d.visualization.draw_geometries([merged])


if __name__ == "__main__":
    main()
