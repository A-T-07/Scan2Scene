# Scan2Scene — 3D scene reconstruction from monocular RGB video

Computer Vision course project (Amrita School of Artificial Intelligence, Coimbatore) — Ashwin Tyagi, C Rishitha, Karthikeya Y,
 Kaushal S.

Ordinary RGB video → COLMAP camera poses → Depth Anything V2 (Metric-Indoor-Large) dense depth → per-frame spatial
depth alignment to the SfM sparse points → confidence-weighted voxel fusion → coloured 3D point cloud.

**Start here:** [`experiments/FINAL_PIPELINE/FINAL_STATUS.md`](experiments/FINAL_PIPELINE/FINAL_STATUS.md) — frozen
configuration, results, rejected alternatives, limitations.

| Document | Content |
|---|---|
| `experiments/FINAL_PIPELINE/final_results.md` | final TUM RGB-D + synthetic benchmark (Sim3-aligned) |
| `experiments/FINAL_PIPELINE/baseline_vs_final.md` | baseline (scale + voxel) vs final (spatial + weighted) |
| `experiments/FINAL_PIPELINE/PHONE_RESULTS.md` | real phone reconstruction (no metric ground truth) |
| `experiments/FINAL_PIPELINE/figures/` | renders and comparisons |

## Frozen configuration

```
python process_video.py --video <clip> --colmap_exe <path\to\colmap.exe> --fps 6 --max_width 960 --stride 2 \
       --voxel_size 0.02 --align spatial --spatial_params experiments/spatial_depth/calib_fr2_xyz_perframe.json \
       --fusion weighted --conf_signals mv,edge --no_viewer
```

Requirements: `requirements.txt` (PyTorch with CUDA, transformers, Open3D, OpenCV, evo) and COLMAP 4.1.1.
View a result: `python view_cloud.py <scene.ply>`.

Not in this repository (size / licence / privacy): datasets (TUM RGB-D is available from TUM), extracted frames,
depth caches, COLMAP workspaces, point clouds (`.ply`), meshes, input videos, Python environments and COLMAP binaries.

All 3D accuracy figures are after Sim(3) alignment to ground truth; the phone reconstruction is not metrically validated.
