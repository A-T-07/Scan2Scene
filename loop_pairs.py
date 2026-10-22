r"""
loop_pairs.py -- which non-sequential (loop) image pairs did COLMAP verify?
Lists verified pairs whose frame-index gap exceeds --min_gap, and specifically start<->end pairs.
Usage: python loop_pairs.py --workdir experiments\phone_real_v2\pose_A_nomask_960 [--min_gap 30] [--edge 15]
"""
import argparse
import os
import sqlite3

import numpy as np

from colmap_to_tum import name_to_timestamp


def loop_pairs(workdir, min_gap=30, edge=15, min_inliers=15):
    con = sqlite3.connect(os.path.join(workdir, "colmap_ws", "database.db"))
    im = {r[0]: int(name_to_timestamp(r[1])) for r in con.execute("SELECT image_id, name FROM images")}
    n = max(im.values()) + 1
    out = []
    for pid, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries"):
        if rows < min_inliers:
            continue
        i2 = pid % 2147483647; i1 = (pid - i2) // 2147483647
        a, b = sorted((im[i1], im[i2]))
        if b - a >= min_gap:
            out.append((a, b, rows))
    start_end = [p for p in out if p[0] < edge and p[1] >= n - edge]
    return dict(frames=n, loop_pairs=len(out), start_end_pairs=len(start_end),
                start_end_best=sorted(start_end, key=lambda p: -p[2])[:5],
                largest_gap_pairs=sorted(out, key=lambda p: -(p[1] - p[0]))[:5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--min_gap", type=int, default=30)
    ap.add_argument("--edge", type=int, default=15)
    a = ap.parse_args()
    print(loop_pairs(a.workdir, a.min_gap, a.edge))
