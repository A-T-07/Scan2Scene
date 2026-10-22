"""Baseline (voxel concatenation) vs TSDF: identical-view renders, top-down density maps and
cross-sections against ground truth. Output: experiments\\tsdf_comparison\\*.png"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figlib import *  # noqa: F401,F403

sys.path.insert(0, ROOT)
from surface_metrics import gt_cloud_synth, gt_cloud_tum  # noqa: E402

OUTD = os.path.join(ROOT, r"experiments\tsdf_comparison")
os.makedirs(OUTD, exist_ok=True)
plt.rcParams.update({"font.family": "Segoe UI", "text.color": hexc(TEXT), "axes.labelcolor": hexc(MUTED),
                     "xtick.color": hexc(MUTED), "ytick.color": hexc(MUTED), "axes.edgecolor": "#2a3542", "font.size": 13})
Z = np.array([0, 0, 1.0]); Yv = np.array([0, 1.0, 0])

CASES = {
    "fr2_xyz": dict(base=ROOT + r"\experiments\tum_fr2_xyz\corrected_scale", tsdf=ROOT + r"\experiments\tum_fr2_xyz\tsdf",
                    gt=FR2, kind="tum", up=Z, eye=(-1.7, -2.3, 2.5), tgt=(1.7, -0.8, 0.5), top=(0, 1), slice_axis=0),
    "fr1_desk": dict(base=ROOT + r"\experiments\tum_freiburg1_desk\corrected_scale", tsdf=ROOT + r"\experiments\tum_freiburg1_desk\tsdf",
                     gt=FR1, kind="tum", up=Z, eye=(0.9, -1.9, 2.9), tgt=(-0.1, 1.3, 0.6), top=(0, 1), slice_axis=0),
    "synth_room2": dict(base=ROOT + r"\synth_room2\fusion_cmp\voxel", tsdf=ROOT + r"\synth_room2\fusion_cmp\tsdf_t008",
                        gt=ROOT + r"\synth_room2", kind="synth", up=Yv, eye=(4.6, 3.6, 4.4), tgt=(0, -0.7, -1.0), top=(0, 2), slice_axis=0),
}


def aligned(rd, gtdir, kind):
    gtp = os.path.join(gtdir, "groundtruth.txt")
    s, R, t, E, G = sim3_to_gt(os.path.join(rd, "trajectory_fused.txt"), gtp)
    P, C = load_ply(os.path.join(rd, "scene.ply"))
    return s * P @ R.T + t, C, E


def ax_style(ax):
    ax.set_facecolor(hexc(BG))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#1f2a36", lw=0.6)


for name, c in CASES.items():
    Pb, Cb, E = aligned(c["base"], c["gt"], c["kind"])
    Pt, Ct, _ = aligned(c["tsdf"], c["gt"], c["kind"])
    if c["kind"] == "synth":  # cut-away: drop ceiling / front / right walls for the perspective view
        cut = lambda P: (P[:, 1] < 1.7) & (P[:, 2] < 2.7) & (P[:, 0] < 2.7)
    else:
        cut = lambda P: np.ones(len(P), bool)
    ims = []
    for P, C, lab in ((Pb, Cb, "baseline: voxel concatenation"), (Pt, Ct, "TSDF fusion")):
        k = cut(P)
        ims.append((render(P[k], C[k], c["eye"], c["tgt"], c["up"], W=1600, H=1000, fov=52, px=2, ss=2, fog=0.25,
                           gain=1.15, alpha_bg=False, lines=[(E, CYAN, 3)]), lab))
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.4), dpi=150)
    fig.patch.set_facecolor(hexc(BG))
    for ax, (im, lab) in zip(axs, ims):
        ax.imshow(im); ax.set_title(f"{name} · {lab}", color=hexc(TEXT), fontsize=15); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, f"{name}_view.png"), facecolor=hexc(BG)); plt.close(fig)

    # top-down density, identical extent / bins / colour scale
    a0, a1 = c["top"]
    allp = np.concatenate([Pb, Pt])
    rng_ = [np.percentile(allp[:, a0], [0.5, 99.5]), np.percentile(allp[:, a1], [0.5, 99.5])]
    hs = [np.histogram2d(P[:, a0], P[:, a1], bins=500, range=rng_)[0] for P in (Pb, Pt)]
    vmax = max(np.log1p(h).max() for h in hs)
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.2), dpi=150)
    fig.patch.set_facecolor(hexc(BG))
    for ax, h, lab in zip(axs, hs, ("baseline: voxel concatenation", "TSDF fusion")):
        ax.imshow(np.log1p(h.T), origin="lower", extent=[*rng_[0], *rng_[1]], cmap="magma", vmin=0, vmax=vmax, aspect="equal")
        ax.plot(E[:, a0], E[:, a1], color=hexc(CYAN), lw=1.2)
        ax.set_title(f"{name} · top view · {lab}", color=hexc(TEXT)); ax_style(ax)
        ax.set_xlabel("xyz"[a0] + " [m]"); ax.set_ylabel("xyz"[a1] + " [m]")
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, f"{name}_topdown.png"), facecolor=hexc(BG)); plt.close(fig)

    # cross-section: thin slab through the densest GT region, plotted with GT underneath
    if c["kind"] == "tum":
        est_t = np.loadtxt(os.path.join(c["base"], "trajectory_fused.txt"))[:, 0]
        G = np.asarray(gt_cloud_tum(c["gt"], est_t, 0.01).points)
    else:
        G = np.asarray(gt_cloud_synth(c["gt"], 0.01).points)
    ax_i = c["slice_axis"]
    cands = np.linspace(np.percentile(G[:, ax_i], 25), np.percentile(G[:, ax_i], 75), 11)
    x0 = max(cands, key=lambda v: np.sum(np.abs(G[:, ax_i] - v) < 0.015))
    other = [i for i in range(3) if i != ax_i]
    h_ax = 2 if c["kind"] == "tum" else 1           # vertical axis
    w_ax = [i for i in other if i != h_ax][0]
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(hexc(BG))
    for ax, P, lab in zip(axs, (Pb, Pt), ("baseline: voxel concatenation", "TSDF fusion")):
        g = G[np.abs(G[:, ax_i] - x0) < 0.015]; p = P[np.abs(P[:, ax_i] - x0) < 0.015]
        ax.scatter(g[:, w_ax], g[:, h_ax], s=0.8, color="#9aa6b2", alpha=0.6, lw=0, label="ground truth")
        ax.scatter(p[:, w_ax], p[:, h_ax], s=3.0, color=hexc(CYAN), alpha=0.8, lw=0, label="fused cloud")
        ax.set_aspect("equal"); ax_style(ax)
        ax.set_title(f"{name} · slice {'xyz'[ax_i]} = {x0:.2f} m ± 1.5 cm · {lab}", color=hexc(TEXT), fontsize=13)
        ax.set_xlabel("xyz"[w_ax] + " [m]"); ax.set_ylabel("xyz"[h_ax] + " [m]")
        ax.legend(frameon=False, labelcolor=hexc(TEXT), markerscale=6, loc="upper right")
    lo = np.percentile(G[np.abs(G[:, ax_i] - x0) < 0.015][:, [w_ax, h_ax]], [1, 99], axis=0)
    axs[0].set_xlim(lo[0, 0] - 0.1, lo[1, 0] + 0.1); axs[0].set_ylim(lo[0, 1] - 0.1, lo[1, 1] + 0.15)
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, f"{name}_slice.png"), facecolor=hexc(BG)); plt.close(fig)
    print("wrote", name)
