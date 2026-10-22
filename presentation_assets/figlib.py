"""Shared helpers for presentation figures: point-cloud splat renderer + data loading.
Everything is rendered from existing experiment outputs (no new experiments)."""
import os
import sys
import numpy as np
import open3d as o3d
from PIL import Image, ImageFilter

ROOT = r"Q:\OPEN_CV_SEM5"
sys.path.insert(0, ROOT)
from eval_synth import umeyama, quat_to_R  # noqa: E402
from eval_tum import read_list, associate  # noqa: E402

OUT = os.path.join(ROOT, "presentation_assets")
FR2 = os.path.join(ROOT, r"rgbd_dataset_freiburg2_xyz\rgbd_dataset_freiburg2_xyz")
FR1 = r"C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_desk\rgbd_dataset_freiburg1_desk"

# palette (matches the deck)
BG = (11, 14, 18)
CYAN = (56, 214, 232)
ICE = (168, 230, 240)
VIOLET = (155, 123, 255)
CORAL = (255, 107, 107)
TEXT = (242, 245, 248)
MUTED = (138, 151, 166)


def hexc(c):
    return "#%02x%02x%02x" % c


def load_ply(path):
    pc = o3d.io.read_point_cloud(path)
    return np.asarray(pc.points), np.asarray(pc.colors)


def sim3_to_gt(traj_path, gt_path, with_scale=True):
    """Umeyama fit of camera centres (fused frame -> GT world). Returns (s, R, t, E_aligned, G)."""
    est = np.loadtxt(traj_path)
    if gt_path.endswith("groundtruth.txt") and "synth" not in gt_path:
        gt = read_list(gt_path)
        a = associate(est[:, 0], [r[0] for r in gt], 0.02)
        keep = [i for i, t in enumerate(est[:, 0]) if t in a]
        G = np.array([[float(x) for x in gt[a[est[i, 0]]][1][:3]] for i in keep])
    else:
        g = np.loadtxt(gt_path)
        gi = {int(round(r[0])): r[1:4] for r in g}
        keep = [i for i, t in enumerate(est[:, 0]) if int(round(t)) in gi]
        G = np.array([gi[int(round(est[i, 0]))] for i in keep])
    E = est[keep, 1:4]
    s, R, t = umeyama(E, G, with_scale)
    return s, R, t, s * E @ R.T + t, G


def look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, -u, f])  # rows: x right, y down, z forward (world->cam rotation)


def render(P, C, eye, target, up, W=2400, H=1350, fov=55, px=2, ss=2, fog=0.0, lines=(), alpha_bg=True,
           bg=BG, sat=1.0, gain=1.0):
    """Z-buffered splat render. lines: list of (Nx3 polyline, rgb, width_px)."""
    Wr, Hr = W * ss, H * ss
    R = look_at(np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float))
    Xc = (P - eye) @ R.T
    z = Xc[:, 2]
    m = z > 0.05
    Xc, z, col = Xc[m], z[m], C[m]
    f = 0.5 * Wr / np.tan(np.radians(fov) / 2)
    u = (f * Xc[:, 0] / z + Wr / 2).astype(np.int64)
    v = (f * Xc[:, 1] / z + Hr / 2).astype(np.int64)
    ok = (u >= 0) & (u < Wr) & (v >= 0) & (v < Hr)
    u, v, z, col = u[ok], v[ok], z[ok], col[ok]
    # colour grading
    col = np.clip(col * gain, 0, 1)
    if sat != 1.0:
        g = col.mean(1, keepdims=True); col = np.clip(g + (col - g) * sat, 0, 1)
    if fog > 0:
        zn = (z - np.percentile(z, 2)) / max(1e-6, np.percentile(z, 98) - np.percentile(z, 2))
        k = np.clip(1 - fog * zn, 0.25, 1)[:, None]
        col = col * k + np.array(bg) / 255 * (1 - k)
    order = np.argsort(-z)  # far first; near overwrite
    u, v, col = u[order], v[order], col[order]
    img = np.zeros((Hr, Wr, 4), np.float32)
    r = px * ss
    offs = [(dx, dy) for dx in range(-r // 2, r - r // 2) for dy in range(-r // 2, r - r // 2)
            if dx * dx + dy * dy <= (r / 2 + 0.5) ** 2]
    for dx, dy in offs:
        uu, vv = u + dx, v + dy
        k = (uu >= 0) & (uu < Wr) & (vv >= 0) & (vv < Hr)
        img[vv[k], uu[k], :3] = col[k]
        img[vv[k], uu[k], 3] = 1
    im = Image.fromarray((img * 255).astype(np.uint8), "RGBA")
    if lines:
        from PIL import ImageDraw
        d = ImageDraw.Draw(im)
        for L, rgb, w in lines:
            Lc = (np.asarray(L) - eye) @ R.T
            Lc = Lc[Lc[:, 2] > 0.05]
            pts = [(f * p[0] / p[2] + Wr / 2, f * p[1] / p[2] + Hr / 2) for p in Lc]
            if len(pts) > 1:
                d.line(pts, fill=tuple(rgb) + (255,), width=int(w * ss), joint="curve")
    im = im.resize((W, H), Image.LANCZOS)
    if not alpha_bg:
        base = Image.new("RGBA", im.size, tuple(bg) + (255,))
        base.alpha_composite(im)
        im = base
    return im


def frustum_lines(C, Rc2w, scale=0.08, aspect=4 / 3):
    """Small camera frustum (as polylines) at centre C with camera-to-world rotation."""
    w, h, d = scale * aspect / 2, scale / 2, scale
    corners = [np.array([x, y, d]) for x, y in [(-w, -h), (w, -h), (w, h), (-w, h)]]
    cw = [C + Rc2w @ c for c in corners]
    segs = [np.array([C, c]) for c in cw] + [np.array(cw + [cw[0]])]
    return segs


def crop_alpha(im, pad=20):
    bbox = im.getchannel("A").getbbox()
    if not bbox:
        return im
    l, t, r, b = bbox
    return im.crop((max(0, l - pad), max(0, t - pad), min(im.width, r + pad), min(im.height, b + pad)))
