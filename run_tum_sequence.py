r"""
run_tum_sequence.py -- full corrected pipeline + evaluation on one TUM RGB-D sequence.

  rgb/ (every --frame_step-th frame) -> COLMAP (video / sequential, --single_camera 1)
  -> reconstruct.py --align scale and --align affine (shared depth cache)
  -> eval_tum.py for both (evo SE3 + Sim3, depth vs Kinect, cloud vs GT cloud)

Nothing is overwritten: each experiment goes to --out (default experiments\tum_<sequence>).

Usage:
  python run_tum_sequence.py --seq_dir Q:\OPEN_CV_SEM5\rgbd_dataset_freiburg1_desk --frame_step 2
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, log=None):
    print(">", " ".join(cmd))
    if log:
        with open(log, "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"!! failed ({r.returncode})" + (f", see {log}" if log else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True, help="TUM sequence folder (contains rgb/, depth/, groundtruth.txt)")
    ap.add_argument("--colmap_exe", default=os.path.join(HERE, "colmap-x64-windows-cuda", "bin", "colmap.exe"))
    ap.add_argument("--frame_step", type=int, default=2, help="Use every Nth RGB frame")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    seq = os.path.basename(os.path.normpath(args.seq_dir))
    out = os.path.abspath(args.out or os.path.join(HERE, "experiments", "tum_" + seq.replace("rgbd_dataset_", "")))
    frames = os.path.join(out, "frames")
    ws = os.path.join(out, "colmap_ws")
    os.makedirs(frames, exist_ok=True); os.makedirs(ws, exist_ok=True)
    t0 = time.time()

    rgb = sorted(os.listdir(os.path.join(args.seq_dir, "rgb")))
    for name in rgb[::args.frame_step]:
        dst = os.path.join(frames, name)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(args.seq_dir, "rgb", name), dst)
    n = len(rgb[::args.frame_step])
    print(f"{n} frames (every {args.frame_step}th of {len(rgb)}) -> {frames}")

    sparse0 = os.path.join(ws, "sparse", "0")
    if not os.path.exists(os.path.join(sparse0, "images.txt")):
        t = time.time()
        run([args.colmap_exe, "automatic_reconstructor", "--workspace_path", ws, "--image_path", frames,
             "--data_type", "video", "--dense", "0", "--single_camera", "1"], log=os.path.join(out, "colmap_log.txt"))
        print(f"COLMAP {time.time() - t:.0f} s")
        for sub in os.listdir(os.path.join(ws, "sparse")):
            d = os.path.join(ws, "sparse", sub)
            subprocess.run([args.colmap_exe, "model_converter", "--input_path", d, "--output_path", d,
                            "--output_type", "TXT"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for sub in sorted(os.listdir(os.path.join(ws, "sparse"))):
        it = os.path.join(ws, "sparse", sub, "images.txt")
        if os.path.exists(it):
            k = sum(1 for l in open(it) if l.strip() and not l.startswith("#")) // 2
            print(f"model {sub}: ~{k} registered of {n} ({100 * k / n:.1f}%)")

    cache = os.path.join(out, "depth_pred")
    for mode in ("scale", "affine"):
        rd = os.path.join(out, f"corrected_{mode}")
        run([sys.executable, os.path.join(HERE, "reconstruct.py"), "--frames", frames, "--colmap_dir", sparse0,
             "--out", rd, "--align", mode, "--stride", str(args.stride), "--depth_cache", cache, "--no_viewer"],
            log=os.path.join(out, f"corrected_{mode}_log.txt"))
        run([sys.executable, os.path.join(HERE, "eval_tum.py"), "--seq_dir", args.seq_dir, "--recon_dir", rd])
    print(f"Done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
