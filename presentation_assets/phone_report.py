"""Phone experiment v2: GT-free diagnostics + identical-view visualisations of baseline vs improved.
Scene frame: gravity-aligned from the camera up-vectors, floor = dominant horizontal RANSAC plane at z=0.
Dimensions are given in the fused frame (network scale) and, separately, x k_TUM ("TUM-calibrated scale")."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image

from figlib import *  # noqa: F401,F403

sys.path.insert(0, ROOT)
from eval_synth import quat_to_R  # noqa: E402
from surface_metrics import local_thickness  # noqa: E402

P2 = os.path.join(ROOT, r"experiments\phone_real_v2")
OUTD = os.path.join(P2, "figures"); os.makedirs(OUTD, exist_ok=True)
FR = os.path.join(P2, r"pose_A_nomask_960\frames")
K_TUM = float(np.exp(np.mean([np.log(json.load(open(os.path.join(ROOT, rf"experiments\calibration\calib_from_{s}.json")))["metric_scale"])
                               for s in ("fr2", "fr1")])))
plt.rcParams.update({"font.family": "Segoe UI", "text.color": hexc(TEXT), "axes.labelcolor": hexc(MUTED),
                     "xtick.color": hexc(MUTED), "ytick.color": hexc(MUTED), "axes.edgecolor": "#2a3542", "font.size": 12})

# ---- scene frame from the (shared) trajectory ----
tr = np.loadtxt(os.path.join(P2, r"baseline\trajectory_fused.txt"))
Cw = tr[:, 1:4]; Rc = [quat_to_R(*r[4:8]) for r in tr]
up = -np.mean([R[:, 1] for R in Rc], axis=0); up /= np.linalg.norm(up)       # camera y points down
fwd = np.mean([R[:, 2] for R in Rc], axis=0); fwd -= up * (fwd @ up); fwd /= np.linalg.norm(fwd)
side = np.cross(up, fwd)
Rsc = np.stack([side, fwd, up])                                               # world -> scene rows


def to_scene(P):
    return P @ Rsc.T


runs = {}
for name in ("baseline", "improved"):
    pc = o3d.io.read_point_cloud(os.path.join(P2, name, "scene.ply"))
    runs[name] = [to_scene(np.asarray(pc.points)), np.asarray(pc.colors)]
cams = to_scene(Cw)
# floor: lowest large horizontal plane of the baseline cloud
Pb = runs["baseline"][0]
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pb[::5]))
floor_z = None
rest = pc
for _ in range(6):
    model, inl = rest.segment_plane(0.02, 3, 2000)
    n = np.array(model[:3]); Q = np.asarray(rest.points)[inl]
    if abs(n[2]) > 0.95 and np.median(Q[:, 2]) < np.median(cams[:, 2]) and (floor_z is None or np.median(Q[:, 2]) < floor_z):
        floor_z = float(np.median(Q[:, 2]))
    rest = rest.select_by_index(inl, invert=True)
for k in runs:
    runs[k][0][:, 2] -= floor_z
cams[:, 2] -= floor_z


def plane_thickness(P, n_planes=6):
    """Largest planes (floor / walls / ceiling / wardrobe): robust spread of points within +-15 cm."""
    out = []
    rest = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[::3]))
    for _ in range(n_planes):
        if len(rest.points) < 5000:
            break
        model, inl = rest.segment_plane(0.03, 3, 3000)
        n, d = np.array(model[:3]), model[3]
        Q = np.asarray(rest.points)
        sd = Q @ n + d
        band = np.abs(sd) < 0.15
        s = sd[band]; med = np.median(s)
        kind = "horizontal" if abs(n[2]) > 0.9 else ("vertical" if abs(n[2]) < 0.2 else "oblique")
        out.append(dict(kind=kind, normal=[round(float(v), 2) for v in n], inliers=len(inl),
                        robust_std_cm=float(1.4826 * np.median(np.abs(s - med)) * 100),
                        p5_p95_cm=float((np.percentile(s, 95) - np.percentile(s, 5)) * 100)))
        rest = rest.select_by_index(inl, invert=True)
    return out


report = dict(k_TUM=K_TUM, floor_offset=floor_z)
for name, (P, C) in runs.items():
    s = json.load(open(os.path.join(P2, name, "summary.json")))
    ext = np.percentile(P, 99, 0) - np.percentile(P, 1, 0)
    lt = local_thickness(P[::4], radius=0.10)
    planes = plane_thickness(P)
    diag = list(csv.DictReader(open(os.path.join(P2, name, "alignment_diagnostics.csv"))))
    rej = [float(r["rejected_frac"]) for r in diag if r.get("rejected_frac")]
    report[name] = dict(points=len(P), extent_fused_units=[float(v) for v in ext],
                        extent_TUM_calibrated_m=[float(v * K_TUM) for v in ext],
                        local_thickness_cm=lt, planes=planes,
                        median_sparse_residual=s["median_rel_residual_after_alignment"],
                        rejected_pct=float(np.mean(rej) * 100) if rej else None,
                        conf_mean=float(np.mean([float(r["conf_mean"]) for r in diag if r.get("conf_mean")])) if rej else None,
                        runtime_s=s["timing_s"]["total_s"])
report["camera_path"] = dict(length_fused=float(np.linalg.norm(np.diff(cams, axis=0), axis=1).sum()),
                             start_end_distance=float(np.linalg.norm(cams[-1] - cams[0])),
                             height_above_floor_median=float(np.median(cams[:, 2])))
json.dump(report, open(os.path.join(P2, "phone_report.json"), "w"), indent=2)

# ---------------- visualisations (identical cameras) ----------------
ctr = np.median(runs["baseline"][0], 0)
views = {
    "main": (ctr + np.array([-3.5, -4.5, 3.2]), ctr, 55),
    "front": (ctr + np.array([0.0, -6.5, 0.6]), ctr, 55),
    "side": (ctr + np.array([-6.5, 0.0, 0.6]), ctr, 55),
}


def pair(fname, eye, tgt, fov, crop=None, px=2, lines=True):
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), dpi=140); fig.patch.set_facecolor(hexc(BG))
    for ax, (name, (P, C)) in zip(axs, runs.items()):
        k = np.ones(len(P), bool) if crop is None else np.all((P >= crop[0]) & (P <= crop[1]), axis=1)
        ln = [(cams, CYAN, 3)] if lines else []
        ax.imshow(render(P[k], C[k], np.array(eye, float), np.array(tgt, float), np.array([0, 0, 1.0]), W=1600, H=1000,
                         fov=fov, px=px, ss=2, fog=0.2, gain=1.1, alpha_bg=False, lines=ln))
        lab = "baseline: scale + voxel" if name == "baseline" else "improved: spatial + weighted (mv,edge)"
        ax.set_title(f"{lab}  ·  {k.sum():,} pts", color=hexc(TEXT)); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, fname), facecolor=hexc(BG)); plt.close(fig)


# cut the ceiling for the perspective views
zcut = np.percentile(runs["baseline"][0][:, 2], 97)
big = np.array([1e9, 1e9, 1e9])
for v, (eye, tgt, fov) in views.items():
    pair(f"cloud_{v}.png", eye, tgt, fov, crop=(-big, np.array([1e9, 1e9, zcut - 0.25])))

# top-down density
allp = np.concatenate([runs[k][0] for k in runs])
rng_ = [np.percentile(allp[:, 0], [0.5, 99.5]), np.percentile(allp[:, 1], [0.5, 99.5])]
hs = {k: np.histogram2d(P[:, 0], P[:, 1], bins=500, range=rng_)[0] for k, (P, _) in runs.items()}
vmax = max(np.log1p(h).max() for h in hs.values())
fig, axs = plt.subplots(1, 2, figsize=(16, 7.5), dpi=140); fig.patch.set_facecolor(hexc(BG))
for ax, (k, h) in zip(axs, hs.items()):
    ax.imshow(np.log1p(h.T), origin="lower", extent=[*rng_[0], *rng_[1]], cmap="magma", vmin=0, vmax=vmax, aspect="equal")
    ax.plot(cams[:, 0], cams[:, 1], color=hexc(CYAN), lw=1.2); ax.plot(*cams[0, :2], "o", color=hexc(CYAN))
    ax.set_title(f"top view · {k}", color=hexc(TEXT)); ax.set_facecolor(hexc(BG))
    ax.set_xlabel("x [fused units]"); ax.set_ylabel("y [fused units]")
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "topdown.png"), facecolor=hexc(BG)); plt.close(fig)

# trajectory plot
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140)
t = np.arange(len(cams)) / 6.0
sc = axs[0].scatter(cams[:, 0], cams[:, 1], c=t, cmap="viridis", s=12); axs[0].plot(cams[:, 0], cams[:, 1], lw=0.6, color="k")
axs[0].set_aspect("equal"); axs[0].set_title("camera trajectory (top view, gravity-aligned)"); plt.colorbar(sc, ax=axs[0], label="time [s]")
axs[0].set_xlabel("x [fused units]"); axs[0].set_ylabel("y [fused units]")
step = np.linalg.norm(np.diff(cams, axis=0), axis=1)
rot = [np.degrees(np.arccos(np.clip((np.trace(Rc[i].T @ Rc[i + 1]) - 1) / 2, -1, 1))) for i in range(len(Rc) - 1)]
axs[1].plot(t[1:], step / np.median(step), label="translation / median"); axs[1].plot(t[1:], rot, label="rotation [deg]")
axs[1].plot(t, cams[:, 2], label="camera height [fused units]")
axs[1].legend(); axs[1].set_xlabel("time [s]"); axs[1].set_title("per-sample motion")
for a in axs:
    a.set_facecolor("white")
plt.rcParams.update({"text.color": "black"})
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "trajectory.png"), facecolor="white"); plt.close(fig)

# depth + confidence examples (improved run saves both)
files = sorted(os.listdir(os.path.join(P2, r"improved\confidence")))
picks = [files[int(len(files) * q)] for q in (0.1, 0.35, 0.6, 0.85)]
fig, axs = plt.subplots(len(picks), 4, figsize=(18, 2.6 * len(picks)), dpi=110)
for row, f in zip(axs, picks):
    z = np.load(os.path.join(P2, r"improved\confidence", f))
    stem = os.path.splitext(f)[0]
    img = np.asarray(Image.open(os.path.join(FR, stem + ".jpg")).convert("RGB"))[::2, ::2]
    d = z["depth"].astype(float)
    rels = [z[k].astype(float) for k in z.files if k.startswith("rel_")]
    ov = np.zeros(z["reject"].shape + (4,)); ov[z["reject"]] = [1, 0.2, 0.2, 1]
    for ax, p, cm, lim, ttl in zip(row, [img, np.where(d > 0, d, np.nan), rels[0] if rels else None, z["conf"].astype(float)],
                                   [None, "viridis", "RdBu_r", "magma"], [None, None, (-0.15, 0.15), (0, 1)],
                                   [stem, "corrected depth", "multi-view residual (neighbour)", "confidence (red = rejected)"]):
        ax.axis("off"); ax.set_title(ttl, fontsize=9)
        if p is None:
            continue
        ax.imshow(p) if cm is None else ax.imshow(p, cmap=cm, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
        if ttl.startswith("confidence"):
            ax.imshow(ov)
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "depth_confidence_examples.png")); plt.close(fig)
json.dump(dict(side=side.tolist(), fwd=fwd.tolist(), up=up.tolist(), floor=floor_z), open(os.path.join(OUTD, "scene_frame.json"), "w"))
print(json.dumps({k: (v if k not in ("baseline", "improved") else {kk: vv for kk, vv in v.items() if kk != "planes"}) for k, v in report.items()}, indent=1))
for name in ("baseline", "improved"):
    print(name, "planes:", [(p["kind"], p["inliers"], round(p["robust_std_cm"], 1), round(p["p5_p95_cm"], 1)) for p in report[name]["planes"]])
