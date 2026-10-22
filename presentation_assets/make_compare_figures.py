"""Generic A-vs-B cloud comparison (identical virtual camera, top-down density, GT cross-section).
Usage: python make_compare_figures.py <fr2_xyz|fr1_desk|synth_room2> <run_A_dir> <label_A> <run_B_dir> <label_B> <out_dir>"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figlib import *  # noqa: F401,F403

sys.path.insert(0, ROOT)
from surface_metrics import gt_cloud_synth, gt_cloud_tum  # noqa: E402

plt.rcParams.update({"font.family": "Segoe UI", "text.color": hexc(TEXT), "axes.labelcolor": hexc(MUTED),
                     "xtick.color": hexc(MUTED), "ytick.color": hexc(MUTED), "axes.edgecolor": "#2a3542", "font.size": 13})
Z = np.array([0, 0, 1.0]); Yv = np.array([0, 1.0, 0])
VIEWS = {
    "fr2_xyz": dict(gt=FR2, kind="tum", up=Z, eye=(-1.7, -2.3, 2.5), tgt=(1.7, -0.8, 0.5), top=(0, 1)),
    "fr1_desk": dict(gt=FR1, kind="tum", up=Z, eye=(0.9, -1.9, 2.9), tgt=(-0.1, 1.3, 0.6), top=(0, 1)),
    "synth_room2": dict(gt=ROOT + r"\synth_room2", kind="synth", up=Yv, eye=(4.6, 3.6, 4.4), tgt=(0, -0.7, -1.0), top=(0, 2)),
}


def aligned(rd, gtdir):
    s, R, t, E, G = sim3_to_gt(os.path.join(rd, "trajectory_fused.txt"), os.path.join(gtdir, "groundtruth.txt"))
    P, C = load_ply(os.path.join(rd, "scene.ply"))
    return s * P @ R.T + t, C, E


def ax_style(ax):
    ax.set_facecolor(hexc(BG))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#1f2a36", lw=0.6)


def main(name, rA, lA, rB, lB, outd):
    os.makedirs(outd, exist_ok=True)
    c = VIEWS[name]
    Pa, Ca, E = aligned(rA, c["gt"]); Pb, Cb, _ = aligned(rB, c["gt"])
    cut = (lambda P: (P[:, 1] < 1.7) & (P[:, 2] < 2.7) & (P[:, 0] < 2.7)) if c["kind"] == "synth" else (lambda P: np.ones(len(P), bool))
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.4), dpi=150); fig.patch.set_facecolor(hexc(BG))
    for ax, (P, C, lab) in zip(axs, ((Pa, Ca, lA), (Pb, Cb, lB))):
        k = cut(P)
        ax.imshow(render(P[k], C[k], c["eye"], c["tgt"], c["up"], W=1600, H=1000, fov=52, px=2, ss=2, fog=0.25, gain=1.15,
                         alpha_bg=False, lines=[(E, CYAN, 3)]))
        ax.set_title(f"{name} · {lab}", color=hexc(TEXT), fontsize=15); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(outd, f"{name}_view.png"), facecolor=hexc(BG)); plt.close(fig)

    a0, a1 = c["top"]
    allp = np.concatenate([Pa, Pb])
    rng_ = [np.percentile(allp[:, a0], [0.5, 99.5]), np.percentile(allp[:, a1], [0.5, 99.5])]
    hs = [np.histogram2d(P[:, a0], P[:, a1], bins=500, range=rng_)[0] for P in (Pa, Pb)]
    vmax = max(np.log1p(h).max() for h in hs)
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.2), dpi=150); fig.patch.set_facecolor(hexc(BG))
    for ax, h, lab in zip(axs, hs, (lA, lB)):
        ax.imshow(np.log1p(h.T), origin="lower", extent=[*rng_[0], *rng_[1]], cmap="magma", vmin=0, vmax=vmax, aspect="equal")
        ax.plot(E[:, a0], E[:, a1], color=hexc(CYAN), lw=1.2)
        ax.set_title(f"{name} · top view · {lab}", color=hexc(TEXT)); ax_style(ax)
        ax.set_xlabel("xyz"[a0] + " [m]"); ax.set_ylabel("xyz"[a1] + " [m]")
    fig.tight_layout(); fig.savefig(os.path.join(outd, f"{name}_topdown.png"), facecolor=hexc(BG)); plt.close(fig)

    if c["kind"] == "tum":
        G = np.asarray(gt_cloud_tum(c["gt"], np.loadtxt(os.path.join(rA, "trajectory_fused.txt"))[:, 0], 0.01).points)
    else:
        G = np.asarray(gt_cloud_synth(c["gt"], 0.01).points)
    cands = np.linspace(np.percentile(G[:, 0], 25), np.percentile(G[:, 0], 75), 11)
    x0 = max(cands, key=lambda v: np.sum(np.abs(G[:, 0] - v) < 0.015))
    h_ax = 2 if c["kind"] == "tum" else 1
    w_ax = 1 if c["kind"] == "tum" else 2
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), dpi=150, sharex=True, sharey=True); fig.patch.set_facecolor(hexc(BG))
    g = G[np.abs(G[:, 0] - x0) < 0.015]
    for ax, P, lab in zip(axs, (Pa, Pb), (lA, lB)):
        p = P[np.abs(P[:, 0] - x0) < 0.015]
        ax.scatter(g[:, w_ax], g[:, h_ax], s=0.8, color="#9aa6b2", alpha=0.6, lw=0, label="ground truth")
        ax.scatter(p[:, w_ax], p[:, h_ax], s=3.0, color=hexc(CYAN), alpha=0.8, lw=0, label="fused cloud")
        ax.set_aspect("equal"); ax_style(ax)
        ax.set_title(f"{name} · slice x = {x0:.2f} m ± 1.5 cm · {lab}", color=hexc(TEXT), fontsize=13)
        ax.set_xlabel("xyz"[w_ax] + " [m]"); ax.set_ylabel("xyz"[h_ax] + " [m]")
        ax.legend(frameon=False, labelcolor=hexc(TEXT), markerscale=6, loc="upper right")
    lo = np.percentile(g[:, [w_ax, h_ax]], [1, 99], axis=0)
    axs[0].set_xlim(lo[0, 0] - 0.1, lo[1, 0] + 0.1); axs[0].set_ylim(lo[0, 1] - 0.1, lo[1, 1] + 0.15)
    fig.tight_layout(); fig.savefig(os.path.join(outd, f"{name}_slice.png"), facecolor=hexc(BG)); plt.close(fig)
    print("wrote", name, "->", outd)


if __name__ == "__main__":
    main(*sys.argv[1:7])
