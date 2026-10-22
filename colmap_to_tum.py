"""
colmap_to_tum.py — export COLMAP's estimated camera poses in TUM trajectory
format, so you can score them against TUM's groundtruth.txt with evo.

TUM format per line:
    timestamp tx ty tz qx qy qz qw

- timestamp: taken directly from the image filename (TUM's own convention)
- tx ty tz: camera CENTER in world coordinates (not the raw COLMAP translation,
  which is world-to-camera, not camera position)
- qx qy qz qw: camera-to-world rotation as a quaternion

Usage:
    python colmap_to_tum.py --colmap_dir colmap_ws3\\sparse\\0 --out estimated_trajectory.txt
"""

import argparse
import os
import re
import numpy as np


def read_colmap_images(images_txt):
    """Returns {filename: {'R': 3x3 world->cam, 't': 3, 'camera_id': int}}"""
    poses = {}
    with open(images_txt, "r") as f:
        # Keep blank lines: an image's POINTS2D line can be empty, and dropping it would
        # shift every following pair.
        lines = [l for l in f if not l.startswith("#")]
    while lines and not lines[-1].strip():
        lines.pop()
    for i in range(0, len(lines) - 1, 2):  # every registered image = 2 lines in this file
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        poses[name] = dict(R=qvec_to_rotmat(qw, qx, qy, qz), t=np.array([tx, ty, tz]), camera_id=cam_id)
    return poses


def qvec_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
    ])


def rotmat_to_qvec(R):
    """3x3 rotation matrix -> (qx, qy, qz, qw), TUM's quaternion order."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return qx, qy, qz, qw


def name_to_timestamp(name):
    """TUM filenames are the timestamp itself, e.g. "1311867170.462290.png";
    extracted video frames ("frame_00012.jpg") use the trailing frame index."""
    stem = os.path.splitext(os.path.basename(name))[0]
    try:
        return float(stem)
    except ValueError:
        return float(re.findall(r"\d+", stem)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap_dir", required=True, help="Folder with images.txt (plain-text COLMAP export)")
    ap.add_argument("--out", default="estimated_trajectory.txt")
    args = ap.parse_args()

    poses = read_colmap_images(os.path.join(args.colmap_dir, "images.txt"))
    print(f"Loaded {len(poses)} registered poses")

    rows = []
    for name, pose in poses.items():
        timestamp = name_to_timestamp(name)

        R_w2c, t_w2c = pose["R"], pose["t"]
        R_c2w = R_w2c.T
        center = -R_c2w @ t_w2c  # camera center in world coordinates

        qx, qy, qz, qw = rotmat_to_qvec(R_c2w)
        rows.append((timestamp, center[0], center[1], center[2], qx, qy, qz, qw))

    rows.sort(key=lambda r: r[0])

    with open(args.out, "w") as f:
        for r in rows:
            f.write(f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f} {r[5]:.6f} {r[6]:.6f} {r[7]:.6f}\n")

    print(f"Wrote {len(rows)} poses to {args.out}")


if __name__ == "__main__":
    main()
