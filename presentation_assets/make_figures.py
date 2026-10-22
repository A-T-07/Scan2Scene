"""Render every figure used in the Review-1 deck from existing experiment outputs.
Run: python make_figures.py   (writes PNGs into this folder)"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial import cKDTree

from figlib import *  # noqa: F401,F403

os.chdir(OUT)
Z = np.array([0, 0, 1.0]); Y = np.array([0, 1.0, 0])

fam = [f.name for f in font_manager.fontManager.ttflist]
SANS = "Segoe UI" if "Segoe UI" in fam else "DejaVu Sans"
MONO = "Consolas" if "Consolas" in fam else "DejaVu Sans Mono"
plt.rcParams.update({"font.family": SANS, "text.color": hexc(TEXT), "axes.labelcolor": hexc(MUTED),
                     "xtick.color": hexc(MUTED), "ytick.color": hexc(MUTED), "axes.edgecolor": "#2a3542",
                     "font.size": 15, "axes.linewidth": 1.0})


def styled_ax(ax):
    ax.set_facecolor((0, 0, 0, 0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#1f2a36", lw=0.8)
    ax.tick_params(labelsize=13)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(MONO)


def get(ply, tr, gt, with_scale=True):
    s, R, t, E, G = sim3_to_gt(tr, gt, with_scale)
    P, C = load_ply(ply)
    return s * P @ R.T + t, C, G, E, (s, R, t)


def traj_rot(tr, gt, sRt):
    """Camera-to-world rotations in the GT frame for frustum drawing."""
    s, R, t = sRt
    est = np.loadtxt(tr)
    return [R @ quat_to_R(*r[4:8]) for r in est]


def save(im, name):
    im.save(name)
    print("wrote", name, im.size)


EXP = os.path.join(ROOT, "experiments")
cases = {
    "fr2": (EXP + r"\tum_fr2_xyz\corrected_scale\scene.ply", EXP + r"\tum_fr2_xyz\corrected_scale\trajectory_fused.txt", FR2 + r"\groundtruth.txt"),
    "fr2_legacy": (ROOT + r"\output\scene.ply", ROOT + r"\estimated_trajectory.txt", FR2 + r"\groundtruth.txt"),
    "fr1": (EXP + r"\tum_freiburg1_desk\corrected_scale\scene.ply", EXP + r"\tum_freiburg1_desk\corrected_scale\trajectory_fused.txt", FR1 + r"\groundtruth.txt"),
    "syn": (ROOT + r"\synth_room2\recon_scale\scene.ply", ROOT + r"\synth_room2\recon_scale\trajectory_fused.txt", ROOT + r"\synth_room2\groundtruth.txt"),
}
D = {k: get(*v) for k, v in cases.items()}

# ---------------- point-cloud renders ----------------
P, C, G, E, sRt = D["fr2"]
Rs = traj_rot(cases["fr2"][1], cases["fr2"][2], sRt)
fr_lines = []
for i in range(0, len(E), 30):
    fr_lines += [(seg, ICE, 1.6) for seg in frustum_lines(E[i], Rs[i], scale=0.14)]
hero = render(P, C, (-2.3, -1.9, 2.9), (1.7, -0.9, 0.5), Z, W=3000, H=1690, fov=52, px=2, ss=2, fog=0.25, gain=1.18,
              lines=[(E, CYAN, 3)] + fr_lines, sat=1.05)
save(hero, "hero_fr2.png")

save(render(P, C, (-1.7, -2.3, 2.5), (1.7, -0.8, 0.5), Z, W=2400, H=1500, fov=50, px=2, ss=2, fog=0.25,
            gain=1.18, lines=[(E, CYAN, 3)]), "fr2_cloud.png")

# legacy vs corrected: identical virtual camera, both Sim(3)-aligned into the TUM mocap frame
cam = ((-2.6, -4.2, 4.6), (1.6, -0.9, 0.4))
for key, name in (("fr2_legacy", "legacy_fr2.png"), ("fr2", "corrected_fr2.png")):
    P2, C2, G2, E2, _ = D[key]
    save(render(P2, C2, cam[0], cam[1], Z, W=2000, H=1400, fov=50, px=2, ss=2, fog=0.35,
                lines=[(G2, CYAN, 3)]), name)

P, C, G, E, sRt = D["fr1"]
Rs = traj_rot(cases["fr1"][1], cases["fr1"][2], sRt)
fr_lines = []
for i in range(0, len(E), 6):
    fr_lines += [(seg, ICE, 1.4) for seg in frustum_lines(E[i], Rs[i], scale=0.12)]
save(render(P, C, (0.9, -1.9, 2.9), (-0.1, 1.3, 0.6), Z, W=2400, H=1500, fov=55, px=3, ss=2, fog=0.45,
            lines=[(E, CYAN, 3)] + fr_lines), "fr1_cloud.png")

P, C, G, E, sRt = D["syn"]
k = (P[:, 1] < 1.7) & (P[:, 2] < 2.7) & (P[:, 0] < 2.7)
Rs = traj_rot(cases["syn"][1], cases["syn"][2], sRt)
fr_lines = []
for i in range(0, len(E), 12):
    fr_lines += [(seg, ICE, 1.6) for seg in frustum_lines(E[i], Rs[i], scale=0.25)]
save(render(P[k], C[k], (4.6, 3.6, 4.4), (0, -0.7, -1.0), Y, W=2400, H=1500, fov=55, px=2, ss=2, fog=0.3,
            lines=[(E, CYAN, 4)] + fr_lines), "synth_cloud.png")

# ---------------- synthetic trajectory: GT vs estimate (Sim3) ----------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
fig.patch.set_alpha(0)
gt = np.loadtxt(ROOT + r"\synth_room2\groundtruth.txt")
ax.plot(gt[:, 1], gt[:, 3], color=hexc(MUTED), lw=6, alpha=0.45, label="renderer ground truth")
ax.plot(E[:, 0], E[:, 2], color=hexc(CYAN), lw=2.2, label="COLMAP estimate (Sim(3)-aligned)")
ax.scatter(E[::10, 0], E[::10, 2], s=22, color=hexc(CYAN), zorder=3)
ax.set_xlabel("x  [m]"); ax.set_ylabel("z  [m]"); ax.set_aspect("equal")
styled_ax(ax)
ax.legend(frameon=False, fontsize=13, loc="lower center", labelcolor=hexc(TEXT))
fig.tight_layout(); fig.savefig("synth_traj.png", transparent=True); plt.close(fig)
print("wrote synth_traj.png")

# ---------------- fr1: GT vs Sim3 vs SE3 alignment ----------------
tr, gtp = cases["fr1"][1], cases["fr1"][2]
_, _, _, E_sim, Gf = sim3_to_gt(tr, gtp, True)
_, _, _, E_se3, _ = sim3_to_gt(tr, gtp, False)
fig, ax = plt.subplots(figsize=(8, 6.2), dpi=220)
fig.patch.set_alpha(0)
ax.plot(Gf[:, 0], Gf[:, 1], color=hexc(TEXT), lw=5, alpha=0.35, label="TUM ground truth")
ax.plot(E_sim[:, 0], E_sim[:, 1], color=hexc(CYAN), lw=2.4, label="Sim(3): ATE 22.5 mm")
ax.plot(E_se3[:, 0], E_se3[:, 1], color=hexc(CORAL), lw=2.4, ls=(0, (5, 2)), label="SE(3): ATE 212.4 mm")
ax.set_xlabel("x  [m]"); ax.set_ylabel("y  [m]"); ax.set_aspect("equal")
styled_ax(ax)
ax.legend(frameon=False, fontsize=13, loc="upper left", labelcolor=hexc(TEXT))
fig.tight_layout(); fig.savefig("fr1_se3_vs_sim3.png", transparent=True); plt.close(fig)
print("wrote fr1_se3_vs_sim3.png")

# ---------------- per-frame depth scale (true metres per predicted metre) ----------------
series = []
for label, rd, key, color in (("TUM fr2_xyz", EXP + r"\tum_fr2_xyz\corrected_scale", "fr2", CYAN),
                               ("TUM fr1_desk", EXP + r"\tum_freiburg1_desk\corrected_scale", "fr1", ICE),
                               ("synth_room2", ROOT + r"\synth_room2\recon_scale", "syn", VIOLET)):
    summ = json.load(open(rd + r"\summary.json"))
    S = summ["global_scale_S_colmap_units_per_pred_metre"]
    s_gt = D[key][4][0]  # GT metres per fused unit
    rows = [r for r in csv.DictReader(open(rd + r"\alignment_diagnostics.csv")) if r["raw_scale"]]
    y = np.array([float(r["raw_scale"]) / S * s_gt for r in rows])
    series.append((label, y, color))
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=220)
fig.patch.set_alpha(0)
ax.axhline(1.0, color=hexc(TEXT), lw=1.2, ls=(0, (4, 3)), alpha=0.7)
ax.text(0.005, 1.015, "metric = 1.0", color=hexc(TEXT), fontsize=12, ha="left", va="bottom", fontfamily=MONO,
        transform=ax.get_yaxis_transform())
for label, y, color in series:
    x = np.linspace(0, 1, len(y))
    ax.plot(x, y, color=hexc(color), lw=2.2, label=f"{label}  (median {np.median(y):.3f})")
ax.set_xlabel("normalised sequence time"); ax.set_ylabel("true depth / predicted depth")
ax.set_ylim(0.55, 1.12)
styled_ax(ax)
ax.legend(frameon=False, fontsize=13, loc="lower left", ncol=3, labelcolor=hexc(TEXT))
fig.tight_layout(); fig.savefig("depth_scale_per_frame.png", transparent=True); plt.close(fig)
print("wrote depth_scale_per_frame.png", [(l, round(float(np.median(y)), 3)) for l, y, _ in series])

# ---------------- surface thickness: vertical slice through fr1 desk ----------------
from eval_tum import INTRINSICS  # noqa: E402
fx, fy, cx, cy = INTRINSICS["freiburg1"]
gt_rows = read_list(FR1 + r"\groundtruth.txt"); dep_rows = read_list(FR1 + r"\depth.txt")
est = np.loadtxt(tr)
a_gt = associate(est[:, 0], [r[0] for r in gt_rows], 0.02); a_d = associate(est[:, 0], [r[0] for r in dep_rows], 0.02)
us, vs = np.meshgrid(np.arange(0, 640, 2), np.arange(0, 480, 2))
rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
gpts = []
for t in est[:, 0]:
    if t not in a_gt or t not in a_d:
        continue
    v = [float(x) for x in gt_rows[a_gt[t]][1]]
    d = np.asarray(Image.open(os.path.join(FR1, dep_rows[a_d[t]][1][0])), np.float64)[::2, ::2] / 5000.0
    m = (d > 0.1) & (d < 6)
    gpts.append((rays[m] * d[m][:, None]) @ quat_to_R(*v[3:7]).T + np.array(v[:3]))
Gc = np.concatenate(gpts)
P, C, _, _, _ = D["fr1"]
x0 = float(np.median(P[:, 0]))
best = None
for x0c in np.linspace(np.percentile(Gc[:, 0], 20), np.percentile(Gc[:, 0], 80), 9):
    n = np.sum(np.abs(Gc[:, 0] - x0c) < 0.015)
    if best is None or n > best[1]:
        best = (x0c, n)
x0 = best[0]
gs = Gc[np.abs(Gc[:, 0] - x0) < 0.015]; fs = P[np.abs(P[:, 0] - x0) < 0.015]
fig, ax = plt.subplots(figsize=(11, 4.8), dpi=220)
fig.patch.set_alpha(0)
ax.scatter(gs[:, 1], gs[:, 2], s=0.8, color="#9aa6b2", alpha=0.55, lw=0, label="Kinect depth (ground truth)", zorder=2)
ax.scatter(fs[:, 1], fs[:, 2], s=5.0, color=hexc(CYAN), alpha=0.85, lw=0, label="our fused cloud (Sim(3)-aligned)", zorder=3)
ax.set_xlabel("y  [m]"); ax.set_ylabel("height z  [m]"); ax.set_aspect("equal")
# zoom on the desk top (densest horizontal GT band)
zh = np.histogram(gs[:, 2], bins=np.arange(0.3, 1.4, 0.02))
zdesk = zh[1][np.argmax(zh[0])]
ax.set_xlim(np.percentile(gs[:, 1], 2) - 0.05, np.percentile(gs[:, 1], 98) + 0.05)
ax.set_ylim(zdesk - 0.3, zdesk + 0.45)
print("desk height ~", round(float(zdesk), 3))
styled_ax(ax)
lg = ax.legend(frameon=False, fontsize=13, loc="upper left", markerscale=8, labelcolor=hexc(TEXT))
fig.tight_layout(); fig.savefig("fr1_slice.png", transparent=True); plt.close(fig)
print("wrote fr1_slice.png at x =", round(x0, 3))

# ---------------- slide 2 strip: RGB frame + predicted depth ----------------
fr1_frames = sorted(os.listdir(EXP + r"\tum_freiburg1_desk\frames"))
fname = fr1_frames[20]
Image.open(EXP + r"\tum_freiburg1_desk\frames\\" + fname).convert("RGB").save("strip_rgb.png")
dep = np.load(EXP + r"\tum_freiburg1_desk\depth_pred\\" + os.path.splitext(fname)[0] + ".npy")
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list("deck", ["#0B0E12", "#12324a", "#1d6f8a", "#38D6E8", "#E8FCFF"])
dn = (dep - np.percentile(dep, 2)) / (np.percentile(dep, 98) - np.percentile(dep, 2))
Image.fromarray((cm(1 - np.clip(dn, 0, 1))[:, :, :3] * 255).astype(np.uint8)).save("strip_depth.png")
print("wrote strip_rgb/depth from", fname, "depth range m", round(float(dep.min()), 2), round(float(dep.max()), 2))

# ---------------- bug figure: old 8-bit input vs predicted depth ----------------
frame = np.asarray(Image.open("bug_frame.png").convert("RGB")).astype(np.float32)
disp = np.load("bug_display_uint8.npy"); pred = np.load("bug_predicted.npy")
kept = (disp > 0) & (disp <= 8)
dark = frame * 0.28
over = dark.copy(); over[kept] = 0.35 * frame[kept] + 0.65 * np.array(CORAL)
Image.fromarray(over.astype(np.uint8)).save("bug_old_kept.png")
pn = (pred - pred.min()) / (pred.max() - pred.min())
Image.fromarray((cm(1 - pn)[:, :, :3] * 255).astype(np.uint8)).save("bug_new_depth.png")
print("wrote bug figures; kept %", round(float(kept.mean()) * 100, 1))

# ---------------- stage thumbnails ----------------
Image.open(ROOT + r"\synth_room2\run\frames\frame_00030.jpg").convert("RGB").save("thumb_synth.png")
Image.open(os.path.join(ROOT, "frames_subset", sorted(os.listdir(ROOT + r"\frames_subset"))[120])).convert("RGB").save("thumb_fr2.png")
Image.open(EXP + r"\tum_freiburg1_desk\frames\\" + fr1_frames[40]).convert("RGB").save("thumb_fr1.png")

# ---------------- compact, large-type variants for small slide slots ----------------
def compact_ax(ax):
    styled_ax(ax)
    ax.tick_params(labelsize=15)


# synthetic trajectory inset
P_, C_, G_, E_, _ = D["syn"]
fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=240)
fig.patch.set_alpha(0)
ax.plot(gt[:, 1], gt[:, 3], color=hexc(MUTED), lw=7, alpha=0.45, label="ground truth")
ax.plot(E_[:, 0], E_[:, 2], color=hexc(CYAN), lw=2.4, label="estimate (Sim3)")
ax.set_xlabel("x [m]", fontsize=16); ax.set_ylabel("z [m]", fontsize=16)
compact_ax(ax)
ax.legend(frameon=False, fontsize=15, loc="lower center", labelcolor=hexc(TEXT))
fig.tight_layout(); fig.savefig("synth_traj_compact.png", transparent=True); plt.close(fig)

# per-frame depth scale
fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=240)
fig.patch.set_alpha(0)
ax.axhline(1.0, color=hexc(TEXT), lw=1.2, ls=(0, (4, 3)), alpha=0.7)
ax.text(0.01, 1.02, "metric = 1.0", color=hexc(TEXT), fontsize=14, ha="left", va="bottom", fontfamily=MONO,
        transform=ax.get_yaxis_transform())
for (label_, y, color), short in zip(series, ("fr2", "fr1", "synth")):
    ax.plot(np.linspace(0, 1, len(y)), y, color=hexc(color), lw=2.2, label=short)
ax.set_ylim(0.55, 1.18); ax.set_xlabel("sequence time", fontsize=16); ax.set_ylabel("true / predicted", fontsize=16)
compact_ax(ax)
ax.legend(frameon=False, fontsize=15, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, labelcolor=hexc(TEXT),
          handlelength=1.2, columnspacing=1.0, borderaxespad=0.1)
fig.tight_layout(); fig.savefig("depth_scale_compact.png", transparent=True); plt.close(fig)

# desk slice
fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=240)
fig.patch.set_alpha(0)
ax.scatter(gs[:, 1], gs[:, 2], s=0.6, color="#9aa6b2", alpha=0.55, lw=0, label="Kinect", zorder=2)
ax.scatter(fs[:, 1], fs[:, 2], s=3.0, color=hexc(CYAN), alpha=0.85, lw=0, label="ours", zorder=3)
ax.set_xlim(np.percentile(gs[:, 1], 2) - 0.05, np.percentile(gs[:, 1], 98) + 0.05)
ax.set_ylim(zdesk - 0.25, zdesk + 0.4)
ax.set_xlabel("y [m]", fontsize=16); ax.set_ylabel("height [m]", fontsize=16)
compact_ax(ax)
ax.legend(frameon=False, fontsize=15, loc="lower left", bbox_to_anchor=(0.0, 1.0), markerscale=6, labelcolor=hexc(TEXT),
          ncol=2, borderaxespad=0.1)
fig.tight_layout(); fig.savefig("fr1_slice_compact.png", transparent=True); plt.close(fig)
print("wrote compact variants")

# ---------------- slide backgrounds (grid + faint point field from the real fr2 cloud) ----------------
def background(name, grid_alpha=1.0, cloud_alpha=0.0, vignette=0.55, seed=0):
    W, H = 2560, 1440
    img = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    minor = ((xx % 40 == 0) | (yy % 40 == 0)).astype(np.float32)
    major = ((xx % 200 == 0) | (yy % 200 == 0)).astype(np.float32)
    img += grid_alpha * (minor[..., None] * np.array([7, 10, 13]) + major[..., None] * np.array([12, 18, 24]))
    rng = np.random.default_rng(seed)
    n = 2600
    px_, py_ = rng.integers(0, W, n), rng.integers(0, H, n)
    img[py_, px_] += np.array([20, 38, 44]) * rng.random(n)[:, None]
    if cloud_alpha > 0:
        P, C, *_ = D["fr2"]
        cl = render(P, C, (-2.3, -1.9, 2.9), (1.7, -0.9, 0.5), Z, W=W, H=H, fov=52, px=1, ss=1, fog=0.3)
        a = np.asarray(cl).astype(np.float32) / 255
        lum = a[..., :3].mean(-1, keepdims=True)
        tint = lum * np.array(CYAN, np.float32)
        img = img * (1 - cloud_alpha * a[..., 3:]) + (tint * 0.9 + img * 0.1) * cloud_alpha * a[..., 3:]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img *= (1 - vignette * np.clip(r - 0.35, 0, 1))[..., None]
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(name, quality=92)
    print("wrote", name)


background("bg_grid.jpg")
background("bg_cloud.jpg", grid_alpha=0.7, cloud_alpha=0.16, seed=1)
background("bg_quiet.jpg", grid_alpha=0.45, vignette=0.4, seed=2)
print("done")
