r"""
make_person_masks.py -- conservative per-frame person masks for COLMAP and depth fusion.

torchvision DeepLabV3-MobileNetV3-Large (VOC classes, 'person' = 15). Every pixel classified as person is
removed, the mask is dilated (default 15 px) to also drop the blurry silhouette border, and written in COLMAP's
convention: <frame name>.png next to nothing else, 255 = keep, 0 = ignore. Existing masks in the folder
(e.g. from --mask_rect) are combined (AND).

Usage: python make_person_masks.py --frames <frames dir> --masks <masks dir> [--dilate 15] [--preview out.jpg]
"""

import argparse
import os

import cv2
import numpy as np
import torch


def person_masks(frames_dir, masks_dir, dilate=15, min_prob=0.5, preview=None):
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = deeplabv3_mobilenet_v3_large(weights=weights).eval().to(dev)
    person = weights.meta["categories"].index("person")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    os.makedirs(masks_dir, exist_ok=True)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
    names = sorted(n for n in os.listdir(frames_dir) if n.lower().endswith((".jpg", ".png")))
    frac, prev = [], []
    for i, n in enumerate(names):
        img = cv2.cvtColor(cv2.imread(os.path.join(frames_dir, n)), cv2.COLOR_BGR2RGB)
        x = (torch.from_numpy(img).permute(2, 0, 1).float().to(dev) / 255 - mean) / std
        with torch.no_grad():
            prob = torch.softmax(model(x[None])["out"][0], 0)[person].cpu().numpy()
        m = (prob > min_prob).astype(np.uint8)
        m = cv2.dilate(m, kern) > 0
        keep = np.where(m, 0, 255).astype(np.uint8)
        mp = os.path.join(masks_dir, n + ".png")
        if os.path.exists(mp):
            old = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if old is not None and old.shape == keep.shape:
                keep = np.minimum(keep, old)
        cv2.imwrite(mp, keep)
        frac.append(float(m.mean()))
        if preview and i % max(1, len(names) // 12) == 0 and len(prev) < 12:
            ov = img.copy(); ov[m] = (0.4 * ov[m] + 0.6 * np.array([255, 60, 60])).astype(np.uint8)
            prev.append(cv2.resize(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), (img.shape[1] // 2, img.shape[0] // 2)))
    if preview and prev:
        while len(prev) % 6:
            prev.append(np.zeros_like(prev[0]))
        cv2.imwrite(preview, np.vstack([np.hstack(prev[r * 6:(r + 1) * 6]) for r in range(len(prev) // 6)]))
    frac = np.array(frac)
    print(f"Person masks: {len(names)} frames, person present in {np.mean(frac > 0.001) * 100:.0f}% of frames, "
          f"masked area mean {frac.mean() * 100:.1f}% / max {frac.max() * 100:.1f}% of the image -> {masks_dir}")
    return frac


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--dilate", type=int, default=15)
    ap.add_argument("--preview", default=None)
    a = ap.parse_args()
    person_masks(a.frames, a.masks, a.dilate, preview=a.preview)
