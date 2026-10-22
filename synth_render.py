r"""
synth_render.py -- render a synthetic static textured room seen by a translating
camera, as a sanity-check input for the pipeline (COLMAP + depth + fusion).

Writes to --out (default Q:\OPEN_CV_SEM5\synth_room):
    frames\frame_00000.jpg ...   rendered RGB frames
    depth_gt\frame_00000.npy ... ground-truth z-depth in metres (float32, 0 = no hit)
    groundtruth.txt              TUM trajectory (timestamp = frame index), camera-to-world
    intrinsics.txt               fx fy cx cy W H
    synth_room.mp4               the frames encoded as a video (--fps)

Usage:
    python synth_render.py
    python synth_render.py --out Q:\OPEN_CV_SEM5\synth_room --n_frames 90 --fps 15
"""

import argparse
import os
import cv2
import numpy as np
import open3d as o3d

from colmap_to_tum import rotmat_to_qvec

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=r"Q:\OPEN_CV_SEM5\synth_room")
ap.add_argument("--n_frames", type=int, default=90)
ap.add_argument("--fps", type=float, default=15.0, help="Frame rate of the output mp4")
args = ap.parse_args()

np.random.seed(3)
frames_dir = os.path.join(args.out, "frames")
depth_dir = os.path.join(args.out, "depth_gt")
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# ---- Build a static room with distinct objects (good SIFT corners, no people, no motion) ----
def box(w, h, d, t):
    m = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)
    m.translate(t)
    return m

floor     = box(6, 0.05, 6, [-3, -1, -3])
wallB     = box(6, 3.0, 0.05, [-3, -1, -3])          # back wall
wallL     = box(0.05, 3.0, 6, [-3, -1, -3])          # left wall
wallR     = box(0.05, 3.0, 6, [2.95, -1, -3])        # right wall
wallF     = box(6, 3.0, 0.05, [-3, -1, 2.95])        # front wall
ceiling   = box(6, 0.05, 6, [-3, 1.95, -3])
desk      = box(1.6, 0.05, 0.7, [-0.8, -0.35, -0.6])
monitor   = box(0.5, 0.35, 0.05, [-0.25, -0.3, -0.55])
chairback = box(0.5, 0.7, 0.06, [-0.7, -0.95, 0.4])
bookshelf = box(0.9, 1.6, 0.35, [1.6, -1.0, -2.9])
plant_pot = box(0.3, 0.4, 0.3, [2.3, -1.0, -0.6])

# First N_TEXTURED parts (room shell) get the mosaic texture; the rest are flat-coloured objects.
parts = [floor, wallB, wallL, wallR, wallF, ceiling, desk, monitor, chairback, bookshelf, plant_pot]
N_TEXTURED = 6
tri_offsets = [0]
mesh = o3d.geometry.TriangleMesh()
for p in parts:
    mesh += p
    tri_offsets.append(tri_offsets[-1] + len(p.triangles))
mesh.compute_triangle_normals()
tri_normals = np.asarray(mesh.triangle_normals)

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

# ---- Irregular mosaic texture (rich local variation -> good for SIFT, not pure periodic) ----
palette = np.array([
    [0.85,0.25,0.25],[0.25,0.55,0.85],[0.95,0.85,0.30],[0.30,0.70,0.40],
    [0.55,0.35,0.75],[0.90,0.55,0.20],[0.20,0.20,0.20],[0.90,0.90,0.90],
    [0.65,0.45,0.30],[0.35,0.75,0.75],[0.80,0.40,0.60],[0.45,0.45,0.45],
])
def mosaic_color(x, y, plane_id, cell=0.22):
    # Proper integer hash (splitmix-style) so the pattern never repeats. The old
    # linear hash reduced to 3*ix + 7*iy + ix*iy (mod 12), which tiles every 12
    # cells (~2.6 m) and made COLMAP match the wrong copy of the pattern.
    ix = np.floor(x/cell).astype(np.int64).astype(np.uint64)
    iy = np.floor(y/cell).astype(np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = ix*np.uint64(0x9E3779B97F4A7C15) ^ iy*np.uint64(0xC2B2AE3D27D4EB4F) ^ np.uint64(plane_id)*np.uint64(0x165667B19E3779F9)
        h ^= h >> np.uint64(31); h *= np.uint64(0xBF58476D1CE4E5B9); h ^= h >> np.uint64(29)
    return palette[(h % np.uint64(palette.shape[0])).astype(np.int64)]

base_colors = np.array([
    [0.55,0.5,0.45],   # floor (mosaic overrides)
    [0.75,0.78,0.85],  # back wall (mosaic overrides)
    [0.75,0.78,0.85],  # left wall (mosaic overrides)
    [0.75,0.78,0.85],  # right wall (mosaic overrides)
    [0.75,0.78,0.85],  # front wall (mosaic overrides)
    [0.85,0.85,0.85],  # ceiling (mosaic overrides)
    [0.35,0.22,0.12],  # desk
    [0.08,0.08,0.08],  # monitor
    [0.5,0.3,0.2],     # chair
    [0.4,0.28,0.18],   # bookshelf
    [0.25,0.45,0.25],  # plant
])

W, H = 640, 480
fx = fy = 520.0
cx, cy = W/2, H/2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
light_dir = np.array([0.4, -0.8, 0.4]); light_dir /= np.linalg.norm(light_dir)

us, vs = np.meshgrid(np.arange(W), np.arange(H))
# camera-frame ray directions with z = 1, so z-depth = t * 1
cam_dirs = np.stack([(us-cx)/fx, (vs-cy)/fy, np.ones_like(us, dtype=np.float64)], axis=-1)

def look_at(eye, target):
    world_up = np.array([0.0, -1.0, 0.0])
    zaxis = target - eye; zaxis /= np.linalg.norm(zaxis)
    xaxis = np.cross(world_up, zaxis); xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)
    return np.stack([xaxis, yaxis, zaxis], axis=0)  # R_w2c

def render_frame(eye, target):
    R_w2c = look_at(eye, target)
    R_c2w = R_w2c.T

    # Build rays ourselves so t_hit is unambiguous: direction has camera-z = 1,
    # hence t_hit == z-depth.
    world_dirs = cam_dirs @ R_c2w.T
    rays = np.concatenate([np.broadcast_to(eye, world_dirs.shape), world_dirs], axis=-1)
    ans = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
    depth = ans["t_hit"].numpy().astype(np.float64)
    valid = np.isfinite(depth)
    prim_ids = ans["primitive_ids"].numpy().astype(np.int64)
    prim_ids[~valid] = 0
    part_idx = np.clip(np.searchsorted(tri_offsets, prim_ids, side="right") - 1, 0, len(parts)-1)

    hit_pts = eye + world_dirs * np.where(valid, depth, 0)[..., None]

    img = np.zeros((H, W, 3), dtype=np.float64)
    img[:] = base_colors[part_idx]
    normals = tri_normals[prim_ids]

    # Texture coords on the room shell: the two world axes orthogonal to the face normal
    tex = (part_idx < N_TEXTURED) & valid
    n_axis = np.argmax(np.abs(normals[tex]), axis=1)
    p = hit_pts[tex]
    u_ax = np.where(n_axis == 0, 2, 0)
    v_ax = np.where(n_axis == 1, 2, 1)
    rows = np.arange(len(p))
    img[tex] = mosaic_color(p[rows, u_ax], p[rows, v_ax], part_idx[tex])

    shade = np.clip(np.abs(normals @ light_dir), 0.35, 1.0)
    img = img * shade[..., None]
    img[~valid] = [1.0, 1.0, 1.0]
    img = np.clip(img*255, 0, 255).astype(np.uint8)

    depth_z = np.where(valid, depth, 0).astype(np.float32)
    return img, depth_z, R_c2w

# ---- Camera path: real translation (walking sweep), not pure rotation ----
n = args.n_frames
target = np.array([0.3, -0.5, -1.3])
video_path = os.path.join(args.out, "synth_room.mp4")
vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

gt_lines = []
for i in range(n):
    t = i / (n-1)
    eye = np.array([-1.6 + 3.2*t, -0.15 + 0.1*np.sin(t*3*np.pi), -2.2 + 1.4*np.sin(t*np.pi)])
    img, depth_z, R_c2w = render_frame(eye, target)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(frames_dir, f"frame_{i:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    np.save(os.path.join(depth_dir, f"frame_{i:05d}.npy"), depth_z)
    vw.write(bgr)

    qx, qy, qz, qw = rotmat_to_qvec(R_c2w)
    gt_lines.append(f"{i:.6f} {eye[0]:.6f} {eye[1]:.6f} {eye[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    if i % 15 == 0:
        print(f"rendered {i+1}/{n}")

vw.release()
with open(os.path.join(args.out, "groundtruth.txt"), "w") as f:
    f.writelines(gt_lines)
with open(os.path.join(args.out, "intrinsics.txt"), "w") as f:
    f.write(f"{fx} {fy} {cx} {cy} {W} {H}\n")

print(f"Rendered {n} frames -> {frames_dir}")
print(f"Video -> {video_path}")
print(f"GT depth -> {depth_dir}, GT trajectory -> {os.path.join(args.out, 'groundtruth.txt')}")
