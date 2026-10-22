r"""
process_video.py -- run the FULL pipeline on any video file in one command:

    video.mp4 -> extract frames -> COLMAP (pose+intrinsics) -> registration report
    -> TUM trajectory export (+ evo if --gt given) -> reconstruct.py (depth + fusion)

The input video is never modified; everything goes into --workdir.

Usage:
    python process_video.py --video my_room.mp4 --colmap_exe "Q:\OPEN_CV_SEM5\colmap-x64-windows-cuda\bin\colmap.exe"

Optional:
    --workdir Q:\OPEN_CV_SEM5\experiments\my_room   (default: <video_name>_run next to the video)
    --fps 5                    frame extraction rate
    --mask_rect x0,y0,x1,y1    fixed image-space overlay to ignore (pixels, repeatable). Frames are
                               kept intact; a COLMAP mask (black = ignore) is written per frame, so no
                               features are extracted there, and reconstruct.py drops those pixels.
    --crop x0,y0,x1,y1         crop every frame to this box instead (changes the image; use sparingly)
    --skip_reconstruct         stop after COLMAP + report + trajectory export (inspect poses first)
    --gt groundtruth.txt       TUM ground truth -> run evo_ape / evo_rpe with Sim(3) alignment
    --stride 4                 passed to reconstruct.py's pixel stride
    --voxel_size 0.02
    --no_viewer
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from colmap_to_tum import name_to_timestamp, read_colmap_images, rotmat_to_qvec


def parse_box(s):
    x0, y0, x1, y1 = (int(v) for v in s.split(","))
    return x0, y0, x1, y1


def extract_frames(video_path, out_dir, fps, crop=None, mask_rects=(), mask_dir=None, max_width=None):
    os.makedirs(out_dir, exist_ok=True)
    if mask_rects:
        os.makedirs(mask_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(video_fps / fps))

    mask = None
    count, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count % frame_interval == 0:
            if crop:
                x0, y0, x1, y1 = crop
                frame = frame[y0:y1, x0:x1]
            if max_width and frame.shape[1] > max_width:
                h = int(round(frame.shape[0] * max_width / frame.shape[1]))
                frame = cv2.resize(frame, (max_width, h), interpolation=cv2.INTER_AREA)
            name = f"frame_{saved:05d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), frame)
            if mask_rects:
                if mask is None:
                    mask = np.full(frame.shape[:2], 255, np.uint8)
                    for x0, y0, x1, y1 in mask_rects:
                        mask[max(0, y0):y1, max(0, x0):x1] = 0
                    print(f"Mask: {len(mask_rects)} rect(s), {100 * (mask == 0).mean():.2f}% of pixels ignored")
                # COLMAP convention: mask for "frame.jpg" is "frame.jpg.png", black = ignore
                cv2.imwrite(os.path.join(mask_dir, name + ".png"), mask)
            saved += 1
        count += 1
    cap.release()
    print(f"Extracted {saved} frames (from {count} total, every {frame_interval}th = ~{video_fps / frame_interval:.1f} fps) -> {out_dir}")
    return saved


def run(cmd, label, log=None):
    print(f"\n=== {label} ===")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    if log:
        with open(log, "w") as fh:
            result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n!! {label} failed (exit code {result.returncode}). Stopping here." + (f" See {log}" if log else ""))
        sys.exit(1)


def colmap_report(colmap_exe, sparse_root, n_frames):
    """Convert every sparse model to TXT and summarise registration / intrinsics."""
    models = []
    for sub in sorted(os.listdir(sparse_root), key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0)):
        d = os.path.join(sparse_root, sub)
        if not os.path.isdir(d):
            continue
        subprocess.run([colmap_exe, "model_converter", "--input_path", d, "--output_path", d,
                        "--output_type", "TXT"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(os.path.join(d, "images.txt")):
            continue
        poses = read_colmap_images(os.path.join(d, "images.txt"))
        cams = [l.split() for l in open(os.path.join(d, "cameras.txt")) if l.strip() and not l.startswith("#")]
        n_pts = sum(1 for l in open(os.path.join(d, "points3D.txt")) if l.strip() and not l.startswith("#"))
        idx = sorted(int(name_to_timestamp(n)) for n in poses)
        # contiguous runs of registered frame indices -> gaps show where tracking was lost
        runs, start = [], idx[0] if idx else None
        for a, b in zip(idx, idx[1:] + [None]):
            if b is None or b != a + 1:
                runs.append((start, a)); start = b
        models.append(dict(path=d, registered=len(poses), points3D=n_pts, cameras=len(cams),
                           camera=" ".join(cams[0][1:]) if cams else None, frame_runs=runs))
    models.sort(key=lambda m: -m["registered"])
    best = models[0] if models else None
    rep = dict(frames_extracted=n_frames, n_models=len(models), models=models,
               registered_best=best["registered"] if best else 0,
               registration_pct=100.0 * (best["registered"] if best else 0) / max(1, n_frames))
    print("\n=== COLMAP registration report ===")
    print(f"Frames extracted: {n_frames}; sparse models: {len(models)}")
    for i, m in enumerate(models):
        print(f"  model {os.path.basename(m['path'])}: {m['registered']} frames ({100 * m['registered'] / max(1, n_frames):.1f}%), "
              f"{m['points3D']} points, camera: {m['camera']}, registered frame runs: {m['frame_runs']}")
    return rep, best


def export_trajectory(images_txt, out_path):
    poses = read_colmap_images(images_txt)
    rows = []
    for name, p in poses.items():
        R_c2w = p["R"].T
        c = -R_c2w @ p["t"]
        rows.append((name_to_timestamp(name), *c, *rotmat_to_qvec(R_c2w)))
    rows.sort()
    with open(out_path, "w") as f:
        for r in rows:
            f.write(" ".join(f"{v:.6f}" for v in r) + "\n")
    C = np.array([r[1:4] for r in rows])
    steps = np.linalg.norm(np.diff(C, axis=0), axis=1)
    med = float(np.median(steps)) if len(steps) else 0.0
    jumps = [(int(rows[i][0]), int(rows[i + 1][0]), float(s / med)) for i, s in enumerate(steps) if med > 0 and s > 5 * med]
    print(f"Trajectory: {len(rows)} poses -> {out_path}; path length {steps.sum():.3f} (COLMAP units), "
          f"median step {med:.4f}, steps > 5x median: {jumps if jumps else 'none'}")
    return dict(path=out_path, n=len(rows), path_length_units=float(steps.sum()), median_step=med, jumps_gt_5x_median=jumps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video file")
    ap.add_argument("--colmap_exe", required=True, help="Full path to colmap.exe")
    ap.add_argument("--workdir", default=None, help="Output folder (default: <video_name>_run next to the video)")
    ap.add_argument("--fps", type=float, default=5.0, help="Frames per second to extract")
    ap.add_argument("--mask_rect", action="append", default=[], type=parse_box,
                    help="x0,y0,x1,y1 pixel box of a fixed overlay to ignore (repeatable)")
    ap.add_argument("--crop", type=parse_box, default=None, help="x0,y0,x1,y1 crop applied to every frame")
    ap.add_argument("--max_width", type=int, default=None,
                    help="Downscale extracted frames to this width (aspect kept), e.g. 960 for 4K phone video")
    ap.add_argument("--person_mask", action="store_true",
                    help="Mask people (DeepLabV3, dilated) out of COLMAP features and depth fusion")
    ap.add_argument("--skip_reconstruct", action="store_true", help="Stop after COLMAP + report + trajectory")
    ap.add_argument("--gt", default=None, help="TUM-format ground truth for evo scoring (only if it really exists)")
    ap.add_argument("--stride", type=int, default=4, help="Passed to reconstruct.py's pixel stride")
    ap.add_argument("--voxel_size", type=float, default=0.02)
    ap.add_argument("--align", default="scale", help="Passed to reconstruct.py --align")
    ap.add_argument("--fusion", default="voxel", choices=["voxel", "tsdf", "weighted"], help="Passed to reconstruct.py --fusion")
    ap.add_argument("--spatial_params", default=None,
                    help="Calibration JSON for --align spatial (e.g. experiments\\spatial_depth\\calib_fr2_xyz_perframe.json)")
    ap.add_argument("--conf_signals", default=None, help="Passed to reconstruct.py --conf_signals (weighted fusion)")
    ap.add_argument("--reconstruct_script", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reconstruct.py"))
    ap.add_argument("--no_viewer", action="store_true", help="Don't open the Open3D viewer at the end")
    args = ap.parse_args()

    t0 = time.time()
    timing = {}
    video_path = os.path.abspath(args.video)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    workdir = os.path.abspath(args.workdir or os.path.join(os.path.dirname(video_path), f"{video_name}_run"))
    frames_dir = os.path.join(workdir, "frames")
    mask_dir = os.path.join(workdir, "masks") if (args.mask_rect or args.person_mask) else None
    colmap_ws = os.path.join(workdir, "colmap_ws")
    sparse_root = os.path.join(colmap_ws, "sparse")
    out_dir = os.path.join(workdir, "output")
    os.makedirs(colmap_ws, exist_ok=True)
    print(f"Working directory: {workdir}")

    # 1. Extract frames
    t = time.time()
    n_frames = extract_frames(video_path, frames_dir, args.fps, args.crop, args.mask_rect, mask_dir, args.max_width)
    timing["extract_s"] = time.time() - t
    if args.person_mask:
        from make_person_masks import person_masks
        t = time.time()
        person_masks(frames_dir, mask_dir, preview=os.path.join(workdir, "person_mask_preview.jpg"))
        timing["person_mask_s"] = time.time() - t
    if n_frames < 10:
        print("Warning: very few frames extracted -- check --fps or the video itself.")

    # 2. COLMAP: sparse reconstruction (sequential matching, since this is a video)
    t = time.time()
    cmd = [
        args.colmap_exe, "automatic_reconstructor",
        "--workspace_path", colmap_ws,
        "--image_path", frames_dir,
        "--data_type", "video",
        "--dense", "0",
        # One phone = one camera. Without this COLMAP solves a separate focal
        # length per frame, which on the synthetic test ranged 335-1925 px.
        "--single_camera", "1",
    ]
    if mask_dir:
        cmd += ["--mask_path", mask_dir]
    run(cmd, "COLMAP automatic_reconstructor", log=os.path.join(workdir, "colmap_log.txt"))
    timing["colmap_s"] = time.time() - t

    if not os.path.isdir(sparse_root) or not os.listdir(sparse_root):
        print(f"\n!! COLMAP did not produce any model under {sparse_root} -- reconstruction failed "
              f"(too little overlap / too few features).")
        sys.exit(1)

    # 3. Registration report + trajectory export (largest model)
    rep, best = colmap_report(args.colmap_exe, sparse_root, n_frames)
    if best is None:
        sys.exit("!! No readable sparse model.")
    rep["trajectory"] = export_trajectory(os.path.join(best["path"], "images.txt"),
                                          os.path.join(workdir, "trajectory_colmap.txt"))
    rep["timing_s"] = timing
    rep["args"] = {k: v for k, v in vars(args).items()}
    with open(os.path.join(workdir, "colmap_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"Registration: {rep['registered_best']}/{n_frames} = {rep['registration_pct']:.1f}% in the largest model; "
          f"COLMAP time {timing['colmap_s']:.0f} s")

    # 4. Optional evo scoring (only with real ground truth)
    if args.gt:
        scripts = os.path.dirname(sys.executable)
        traj = rep["trajectory"]["path"]
        for tool in ("evo_ape", "evo_rpe"):
            exe = os.path.join(scripts, tool + (".exe" if os.name == "nt" else ""))
            subprocess.run([exe, "tum", args.gt, traj, "-as"])

    if args.skip_reconstruct:
        print("\n--skip_reconstruct given: stopping before depth fusion.")
        return

    # 5. Depth + fusion
    t = time.time()
    cmd = [
        sys.executable, args.reconstruct_script,
        "--frames", frames_dir,
        "--colmap_dir", best["path"],
        "--out", out_dir,
        "--stride", str(args.stride),
        "--voxel_size", str(args.voxel_size),
        "--align", args.align,
        "--fusion", args.fusion,
    ]
    if args.spatial_params:
        cmd += ["--spatial_params", args.spatial_params]
    if args.conf_signals:
        cmd += ["--conf_signals", args.conf_signals]
    if mask_dir:
        cmd += ["--mask_dir", mask_dir]
    if args.no_viewer:
        cmd.append("--no_viewer")
    run(cmd, "reconstruct.py (depth + alignment + fusion)")
    timing["reconstruct_s"] = time.time() - t
    print(f"\nAll done in {time.time() - t0:.0f} s. Point cloud saved under: {os.path.join(out_dir, 'scene.ply')}")


if __name__ == "__main__":
    main()
