"""EVALUATION-ONLY diagnostic: are the extra verified long-range pairs plausible? For every verified pair with an
index gap > 30, the GT camera viewing directions are compared; pairs whose optical axes differ by > 90 deg cannot
see the same surface from a normal indoor distance and are counted as implausible (likely false matches)."""
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, r"Q:\OPEN_CV_SEM5")
from eval_synth import quat_to_R  # noqa: E402
from eval_tum import associate, read_list  # noqa: E402

SEQ = r"C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_room\rgbd_dataset_freiburg1_room"
PV = os.path.dirname(os.path.abspath(__file__))
gt = read_list(os.path.join(SEQ, "groundtruth.txt"))
gts = [r[0] for r in gt]
for v in ("control", "variant_B_sequential", "variant_C_global"):
    con = sqlite3.connect(os.path.join(PV, v, "colmap_ws", "database.db"))
    im = {r[0]: float(os.path.splitext(r[1])[0]) for r in con.execute("SELECT image_id, name FROM images")}
    order = sorted(im, key=im.get); pos = {i: k for k, i in enumerate(order)}
    a = associate(list(im.values()), gts, 0.02)
    zax = {i: quat_to_R(*[float(x) for x in gt[a[t]][1][3:7]])[:, 2] for i, t in im.items() if t in a}
    cen = {i: np.array([float(x) for x in gt[a[t]][1][:3]]) for i, t in im.items() if t in a}
    ang, inl, dist = [], [], []
    for pid, n in con.execute("SELECT pair_id, rows FROM two_view_geometries"):
        if n < 15:
            continue
        i2 = pid % 2147483647; i1 = (pid - i2) // 2147483647
        if abs(pos[i1] - pos[i2]) <= 30 or i1 not in zax or i2 not in zax:
            continue
        ang.append(np.degrees(np.arccos(np.clip(zax[i1] @ zax[i2], -1, 1)))); inl.append(n)
        dist.append(np.linalg.norm(cen[i1] - cen[i2]))
    ang, inl = np.array(ang), np.array(inl)
    bad = ang > 90
    print(f"{v:22s} long-range verified pairs {len(ang):5d} | axis angle median {np.median(ang):5.1f} deg | "
          f">90 deg (implausible): {bad.sum():4d} ({bad.mean() * 100:4.1f}%), their median inliers {np.median(inl[bad]) if bad.any() else 0:.0f} "
          f"(plausible pairs {np.median(inl[~bad]):.0f})")
