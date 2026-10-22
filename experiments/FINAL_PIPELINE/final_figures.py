"""FINAL_PIPELINE figures (visualisation only; clouds are read, never modified).
Conventions: TUM/synthetic clouds are displayed in the ground-truth frame via the trajectory Sim(3) fit (display only);
baseline and final share the same transform, virtual camera, point size, fog and colour gain. Phone clouds use the
gravity-aligned scene frame saved by phone_report.py (figures\\scene_frame.json)."""
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"Q:\OPEN_CV_SEM5\presentation_assets")
from figlib import *  # noqa: F401,F403
import make_compare_figures as mcf  # noqa: E402

E = os.path.join(ROOT, "experiments")
FIG = os.path.join(E, r"FINAL_PIPELINE\figures")
os.makedirs(FIG, exist_ok=True)
FR1ROOM = r"C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_room\rgbd_dataset_freiburg1_room"
Z = np.array([0, 0, 1.0])
plt.rcParams.update({"font.family": "Segoe UI", "text.color": hexc(TEXT), "font.size": 13})
RENDER = dict(px=2, ss=2, fog=0.2, gain=1.1, alpha_bg=False)


def aligned(rd, gt):
    s, R, t, E_, G = sim3_to_gt(os.path.join(rd, "trajectory_fused.txt"), gt)
    P, C = load_ply(os.path.join(rd, "scene.ply"))
    return s * P @ R.T + t, C, E_, (s, R, t)


def pose_view(rd, gt, sRt, frame_name, pull=0.4):
    """Virtual camera = recovered camera pose of `frame_name` (mapped into the display frame), pulled back."""
    s, R, t = sRt
    est = np.loadtxt(os.path.join(rd, "trajectory_fused.txt"))
    ts = float(os.path.splitext(frame_name)[0])
    r = est[np.argmin(np.abs(est[:, 0] - ts))]
    Rc = R @ quat_to_R(*r[4:8]); C = s * R @ r[1:4] + t
    fwd, up = Rc[:, 2], -Rc[:, 1]
    return C - pull * fwd, C + 2.0 * fwd, up


def single(P, C, eye, tgt, up, fname, title, lines=(), W=2400, H=1500, fov=55, crop=None):
    k = np.ones(len(P), bool) if crop is None else crop(P)
    im = render(P[k], C[k], np.array(eye, float), np.array(tgt, float), np.array(up, float), W=W, H=H, fov=fov,
                lines=list(lines), **RENDER)
    fig = plt.figure(figsize=(W / 150, H / 150 + 0.4), dpi=150); fig.patch.set_facecolor(hexc(BG))
    ax = fig.add_axes([0, 0, 1, H / (H + 60)]); ax.imshow(im); ax.axis("off")
    fig.suptitle(title, color=hexc(TEXT), fontsize=14, y=0.995)
    fig.savefig(os.path.join(FIG, fname), facecolor=hexc(BG)); plt.close(fig)
    print("wrote", fname)


def pair(clouds, eye, tgt, up, fname, title, lines=(), fov=55, crop=None, W=1600, H=1000):
    fig, axs = plt.subplots(1, 2, figsize=(2 * W / 100, H / 100 + 0.8), dpi=100); fig.patch.set_facecolor(hexc(BG))
    for ax, (lab, P, C) in zip(axs, clouds):
        k = np.ones(len(P), bool) if crop is None else crop(P)
        ax.imshow(render(P[k], C[k], np.array(eye, float), np.array(tgt, float), np.array(up, float), W=W, H=H, fov=fov,
                         lines=list(lines), **RENDER))
        ax.set_title(f"{lab} · {k.sum():,} pts", color=hexc(TEXT), fontsize=15); ax.axis("off")
    fig.suptitle(title, color=hexc(TEXT), fontsize=16)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), facecolor=hexc(BG)); plt.close(fig)
    print("wrote", fname)


# ============ A + B(fr1_room): final reconstruction and baseline vs final ============
gtr = os.path.join(FR1ROOM, "groundtruth.txt")
Pf, Cf, Ef, sRt = aligned(os.path.join(E, r"tum_freiburg1_room\improved"), gtr)
Pb, Cb, _, _ = aligned(os.path.join(E, r"tum_freiburg1_room\baseline"), gtr)
nocei = lambda P: P[:, 2] < 2.3
ctr = np.array([0.25, -0.1, 0.9])
traj = [(Ef, CYAN, 3)]
single(Pf, Cf, ctr + [-4.5, -5.5, 4.6], ctr, Z, "fr1_room_final_main.png",
       "TUM fr1_room · final (spatial + weighted, mv,edge) · ceiling cut · cyan = recovered camera path", traj, crop=nocei)
single(Pf, Cf, ctr + [0, 0.01, 11.5], ctr, np.array([0, 1.0, 0]), "fr1_room_final_top.png",
       "TUM fr1_room · final · top view (ceiling cut)", traj, fov=50, crop=nocei)
single(Pf, Cf, ctr + [0, -8.5, 0.6], ctr, Z, "fr1_room_final_side.png", "TUM fr1_room · final · side view", crop=nocei)
# trajectory: GT vs Sim(3)-aligned estimate
est = np.loadtxt(os.path.join(E, r"tum_freiburg1_room\improved\trajectory_fused.txt"))
from eval_tum import associate, read_list  # noqa: E402
gtl = read_list(gtr); a = associate(est[:, 0], [r[0] for r in gtl], 0.02)
G = np.array([[float(x) for x in gtl[a[t]][1][:3]] for t in est[:, 0] if t in a])
Ea = Ef[[i for i, t in enumerate(est[:, 0]) if t in a]]
fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
plt.rcParams.update({"text.color": "black"})
ax.plot(G[:, 0], G[:, 1], color="0.55", lw=5, alpha=0.6, label="TUM ground truth")
ax.plot(Ea[:, 0], Ea[:, 1], color="#1f9fb3", lw=1.8, label="COLMAP estimate (Sim3-aligned, ATE RMSE 89.5 mm)")
ts = est[[i for i, t in enumerate(est[:, 0]) if t in a], 0]
w = (ts >= 1305031938.5) & (ts <= 1305031941.0)
ax.plot(Ea[w, 0], Ea[w, 1], "o", color="#e0553b", ms=4, label="28-30 s problem region")
ax.plot(*G[0, :2], "k^", ms=9, label="start"); ax.plot(*G[-1, :2], "ks", ms=8, label="end")
ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(fontsize=9)
ax.set_title("TUM fr1_room camera trajectory (top view)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fr1_room_trajectory.png")); plt.close(fig)
plt.rcParams.update({"text.color": hexc(TEXT)})
print("wrote fr1_room_trajectory.png")

frames = sorted(os.listdir(os.path.join(E, r"tum_freiburg1_room\frames")))
clouds = [("baseline: scale + voxel", Pb, Cb), ("final: spatial + weighted (mv,edge)", Pf, Cf)]
pair(clouds, ctr + [-4.5, -5.5, 4.6], ctr, Z, "compare_fr1_room_overall.png",
     "TUM fr1_room · baseline vs final · identical camera and render settings", traj, crop=nocei)
for tag, idx, ttl in (("wall", 199, "wall / plane close-up (desk against white wall)"),
                      ("detail", 54, "object / detail close-up (desk with books, phone, gamepad)")):
    eye, tgt, up = pose_view(os.path.join(E, r"tum_freiburg1_room\improved"), gtr, sRt, frames[idx])
    pair(clouds, eye, tgt, up, f"compare_fr1_room_{tag}.png",
         f"TUM fr1_room · {ttl} · camera = recovered pose of input frame {idx}", fov=70)
    shutil.copy2(os.path.join(E, r"tum_freiburg1_room\frames", frames[idx]), os.path.join(FIG, f"compare_fr1_room_{tag}_input_frame.png"))
# top-down density pair (same extent / bins / colour scale)
allp = np.concatenate([Pb, Pf])
rng_ = [np.percentile(allp[:, 0], [0.5, 99.5]), np.percentile(allp[:, 1], [0.5, 99.5])]
hs = [np.histogram2d(P[:, 0], P[:, 1], bins=600, range=rng_)[0] for P in (Pb, Pf)]
vmax = max(np.log1p(h).max() for h in hs)
fig, axs = plt.subplots(1, 2, figsize=(18, 8.5), dpi=150); fig.patch.set_facecolor(hexc(BG))
for ax, h, (lab, _, _) in zip(axs, hs, clouds):
    ax.imshow(np.log1p(h.T), origin="lower", extent=[*rng_[0], *rng_[1]], cmap="magma", vmin=0, vmax=vmax, aspect="equal")
    ax.plot(Ef[:, 0], Ef[:, 1], color=hexc(CYAN), lw=1); ax.set_title(f"top-down point density · {lab}", color=hexc(TEXT))
    ax.set_xlabel("x [m, GT frame]", color=hexc(MUTED)); ax.set_ylabel("y [m]", color=hexc(MUTED)); ax.tick_params(colors=hexc(MUTED))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "compare_fr1_room_topdown.png"), facecolor=hexc(BG)); plt.close(fig)
print("wrote compare_fr1_room_topdown.png")

# ============ B: held-out + synthetic comparisons (existing renderer, unchanged) ============
for ds, base, fin in (("fr1_desk", r"tum_freiburg1_desk\corrected_scale", r"pose_conf_experiment\fr1_desk\arm1_mv_edge"),
                      ("fr2_xyz", r"tum_fr2_xyz\corrected_scale", r"pose_conf_experiment\fr2_xyz\arm1_mv_edge"),
                      ("synth_room2", r"..\synth_room2\recon_scale", r"pose_conf_experiment\synth_room2\arm1_mv_edge")):
    mcf.main(ds, os.path.normpath(os.path.join(E, base)), "baseline: scale + voxel", os.path.join(E, fin),
             "final: spatial + weighted (mv,edge)", os.path.join(FIG, f"compare_{ds}"))

# ============ C: phone, final only ============
P2 = os.path.join(E, "phone_real_v2")
sf = json.load(open(os.path.join(P2, r"figures\scene_frame.json")))
Rsc = np.stack([sf["side"], sf["fwd"], sf["up"]])
pc_P, pc_C = load_ply(os.path.join(P2, r"improved\scene.ply"))
Pp = pc_P @ Rsc.T; Pp[:, 2] -= sf["floor"]
trp = np.loadtxt(os.path.join(P2, r"improved\trajectory_fused.txt"))
cams = trp[:, 1:4] @ Rsc.T; cams[:, 2] -= sf["floor"]
zc = np.percentile(Pp[:, 2], 97) - 0.25
cutp = lambda P: P[:, 2] < zc
c0 = np.median(Pp, 0)
lab = "phone (IMG_8682) · final: spatial + weighted (mv,edge) · fused units, NOT metric"
single(Pp, pc_C, c0 + [-3.5, -4.5, 3.2], c0, Z, "phone_final_main.png", lab + " · ceiling cut", [(cams, CYAN, 3)], crop=cutp)
single(Pp, pc_C, c0 + [0, -6.5, 0.6], c0, Z, "phone_final_front.png", lab + " · front", crop=cutp)
single(Pp, pc_C, c0 + [-6.5, 0, 0.6], c0, Z, "phone_final_side.png", lab + " · side", crop=cutp)
single(Pp, pc_C, c0 + [0, 0.01, 11.0], c0, np.array([0, 1.0, 0]), "phone_final_top.png", lab + " · top view",
       [(cams, CYAN, 2)], crop=cutp, fov=55)
Rs = [Rsc @ quat_to_R(*r[4:8]) for r in trp]
for tag, tsec in (("shelves_desk", 5.5), ("window_bed", 12.0), ("second_desk", 30.0)):
    i = min(int(round(tsec * 6)), len(cams) - 1)
    Rc = Rs[i]; fwd = Rc[:, 2]
    single(Pp, pc_C, cams[i] - 0.6 * fwd, cams[i] + 2.0 * fwd, -Rc[:, 1], f"phone_final_closeup_{tag}.png",
           f"phone · final · close-up '{tag}' · camera = recovered pose of sampled frame {i} (t = {tsec:.1f} s)", fov=70,
           W=1920, H=1080)
    shutil.copy2(os.path.join(P2, r"pose_A_nomask_960\frames", f"frame_{i:05d}.jpg"), os.path.join(FIG, f"phone_final_closeup_{tag}_input_frame.jpg"))
for f in ("trajectory.png", "depth_confidence_examples.png"):   # reuse existing figures unchanged (copied)
    shutil.copy2(os.path.join(P2, "figures", f), os.path.join(FIG, "phone_" + f))
print("done")
