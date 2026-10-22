"""Close-ups for phone_real_v2: render baseline vs improved from the recovered camera pose at the time each object
is in view (pulled back 0.6 units along the view axis), next to the input frame. Also re-draws the trajectory plot."""
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

P2 = os.path.join(ROOT, r"experiments\phone_real_v2")
OUTD = os.path.join(P2, "figures")
FR = os.path.join(P2, r"pose_A_nomask_960\frames")
sf = json.load(open(os.path.join(OUTD, "scene_frame.json")))
Rsc = np.stack([sf["side"], sf["fwd"], sf["up"]])
tr = np.loadtxt(os.path.join(P2, r"baseline\trajectory_fused.txt"))
runs = {}
for name in ("baseline", "improved"):
    pc = o3d.io.read_point_cloud(os.path.join(P2, name, "scene.ply"))
    P = np.asarray(pc.points) @ Rsc.T; P[:, 2] -= sf["floor"]
    runs[name] = (P, np.asarray(pc.colors))
cams = tr[:, 1:4] @ Rsc.T; cams[:, 2] -= sf["floor"]
Rs = [Rsc @ quat_to_R(*r[4:8]) for r in tr]

# sample index = time * 6 (6 fps sampling); times read from the input contact sheet
SHOTS = [("door", 1.0), ("shelves + desk", 5.5), ("bed", 8.0), ("window", 12.0), ("wardrobe", 26.0), ("second desk", 30.0)]
fig, axs = plt.subplots(len(SHOTS), 3, figsize=(18, 3.4 * len(SHOTS)), dpi=110)
fig.patch.set_facecolor(hexc(BG))
for row, (label, t) in zip(axs, SHOTS):
    i = min(int(round(t * 6)), len(cams) - 1)
    C, R = cams[i], Rs[i]
    fwd, upc = R[:, 2], -R[:, 1]
    eye = C - 0.6 * fwd
    tgt = C + 2.0 * fwd
    img = Image.open(os.path.join(FR, f"frame_{i:05d}.jpg")).convert("RGB")
    row[0].imshow(img); row[0].set_title(f"{label} · input frame {i} (t = {t:.1f} s)", color=hexc(TEXT), fontsize=10)
    for ax, name in zip(row[1:], runs):
        P, Cc = runs[name]
        ax.imshow(render(P, Cc, eye, tgt, upc, W=960, H=540, fov=70, px=2, ss=2, alpha_bg=False, gain=1.05))
        ax.set_title(f"{label} · {name}", color=hexc(TEXT), fontsize=10)
    for ax in row:
        ax.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "closeups.png"), facecolor=hexc(BG)); plt.close(fig)

# trajectory plot (readable titles)
plt.rcParams.update({"text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"})
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140)
tt = np.arange(len(cams)) / 6.0
sc = axs[0].scatter(cams[:, 0], cams[:, 1], c=tt, cmap="viridis", s=12); axs[0].plot(cams[:, 0], cams[:, 1], lw=0.6, color="k")
for label, t in SHOTS:
    i = min(int(round(t * 6)), len(cams) - 1)
    axs[0].annotate(label, cams[i, :2], fontsize=8, xytext=(4, 4), textcoords="offset points")
axs[0].set_aspect("equal"); axs[0].set_title("camera trajectory (top view, gravity-aligned)"); plt.colorbar(sc, ax=axs[0], label="time [s]")
axs[0].set_xlabel("x [fused units]"); axs[0].set_ylabel("y [fused units]")
step = np.linalg.norm(np.diff(cams, axis=0), axis=1)
rot = [np.degrees(np.arccos(np.clip((np.trace(Rs[i].T @ Rs[i + 1]) - 1) / 2, -1, 1))) for i in range(len(Rs) - 1)]
axs[1].plot(tt[1:], step / np.median(step), label="translation / median step")
axs[1].plot(tt[1:], rot, label="rotation per sample [deg]")
axs[1].plot(tt, cams[:, 2], label="camera height above floor [fused units]")
axs[1].legend(); axs[1].set_xlabel("time [s]"); axs[1].set_title("per-sample motion (6 samples / s)")
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "trajectory.png")); plt.close(fig)
print("wrote closeups + trajectory")
