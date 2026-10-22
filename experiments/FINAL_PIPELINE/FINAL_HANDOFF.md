# Scan2Scene — Final Project Handoff

Frozen 2026-09-30. Numbers are copied from existing result files (paths in §11); nothing in the reconstruction was
re-run for this document. Detailed companions: `FINAL_STATUS.md`, `PHONE_RESULTS.md`, `baseline_vs_final.md`,
`final_results.md`.

## 1. Project objective

3D scene reconstruction from ordinary monocular phone video: no depth sensor, no stereo rig, no calibration target.

## 2. Final pipeline

Input video → frame sampling (6 fps, 960 px wide for phone; every N-th RGB frame for TUM) → COLMAP SfM (one shared
camera) → camera poses + sparse 3D points → Depth Anything V2 Metric-Indoor-Large (per-frame dense depth) → spatial depth
alignment (sequence scale + per-frame low-order correction fitted to that frame's COLMAP sparse points) →
confidence-weighted voxel fusion (multi-view agreement × depth-edge confidence, 2 cm voxels) → coloured 3D point cloud
(`scene.ply`) → visualization / export (PNG renders, GLB + interactive viewer).

## 3. Frozen configuration

```
reconstruct.py --align spatial --spatial_params experiments\spatial_depth\calib_fr2_xyz_perframe.json
               --fusion weighted --conf_signals mv,edge --stride 2 --voxel_size 0.02 --depth_trunc 8.0 --no_viewer
```

- COLMAP 4.1.1, `automatic_reconstructor --data_type video --single_camera 1 --dense 0` (SIMPLE_RADIAL, SIFT 8192,
  guided matching, sequential overlap 10 + vocabulary-tree loop detection).
- Depth model `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`.
- Spatial calibration: `poly2_xy_logd`, λ = 0.01, clamp ±0.2 (log), min 60 sparse points per frame.
- Weighted fusion: τ = 0.05, τ_hard = 0.15, edge scale 0.05, conf_min 0.05, neighbours ≤ 3 frames, overlap ≥ 0.2,
  voxel weight ≥ 0.5; statistical outlier removal (20, 2.0).
- Metric calibration OFF (TUM factor 0.811 reported separately only).
- Every parameter: `final_config.json`.

## 4. Final validated datasets

| Dataset | Role | Frames | Ground truth |
|---|---|---|---|
| TUM fr1_room | development | 273 sampled (every 5th), 272 registered | mocap trajectory + Kinect depth |
| TUM fr1_desk | held-out | 62 (every 10th), 62 registered | mocap + Kinect |
| TUM fr2_xyz | held-out (spatial calibration derived from its own sparse points, no GT) | 367, 367 registered | mocap + Kinect |
| synth_room2 | synthetic | 90, 90 registered | renderer trajectory + depth |
| phone IMG_8682.MOV | real demonstration | 213 (6 fps), 213 registered | **none** (internal consistency only) |
| TUM fr2_desk | additional unseen-sequence validation (completed, reported separately) | 297 (every 10th), 297 registered | mocap + Kinect |

## 5. Final quantitative results

All trajectory and 3D numbers are after Sim(3) alignment to ground truth (global scale removed) — not absolute
metric accuracy. Depth AbsRel uses `eval_aligned_depth.py` and thickness `surface_metrics.py` on every row; 3D cloud
metrics use `eval_tum.py` (TUM, Kinect GT) or `eval_synth.py` (synthetic, renderer GT), which are not interchangeable.

| Dataset | Trajectory: ATE Sim3 RMSE | Trajectory: RPE Sim3 | Depth: AbsRel | 3D: median err | 3D: P90 | 3D: within 5 cm | 3D: completeness <5 cm | Surface thickness | Points |
|---|---|---|---|---|---|---|---|---|---|
| fr1_room (TUM, dev) | 89.5 mm | 28.2 mm | 9.84 % | 7.95 cm | 34.0 cm | 37.9 % | 68.8 % | 3.19 cm | 1,875,200 |
| fr1_desk (TUM, held-out) | 22.5 mm | 25.5 mm | 4.07 % | 5.23 cm | 23.0 cm | 48.8 % | 83.2 % | 2.89 cm | 308,540 |
| fr2_xyz (TUM, held-out) | 7.4 mm | 3.0 mm | 4.11 % | 6.25 cm | 32.7 cm | 43.9 % | 92.9 % | 3.96 cm | 1,613,959 |
| synth_room2 (synthetic) | 7.6 mm | 3.0 mm | 1.91 % | 5.96 cm | 16.8 cm | 45.2 % | 70.2 % | 1.51 cm | 715,333 |

Absolute scale is biased (Depth Anything overestimates depth): Sim3 scale correction 0.817 (fr2_xyz), 0.805
(fr1_desk), 0.843 (fr1_room), 0.754 (synthetic). Phone internal-consistency metrics are in §6 (no ground truth).

### 5b. Additional unseen-sequence validation — TUM fr2_desk

Run once with exactly the frozen configuration (no tuning), every 10th RGB frame (a priori, as fr2_xyz); RGB only for
reconstruction, TUM depth / groundtruth only for evaluation. Script `experiments\run_fr2_desk_validation.ps1`, outputs
`experiments\tum_fr2_desk_validation\final\`. Not used to select or change anything; reported separately.

| Registered | ATE Sim3 RMSE | RPE Sim3 | ATE SE3 | Sim3 scale | Depth AbsRel | 3D median err | P90 | Within 5 cm | Completeness <5 cm | Thickness | Points | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 297/297, 1 model, 1.05 px | 11.9 mm | 6.1 mm | 275.7 mm | 0.864 | 4.54 % | 4.72 cm | 50.5 cm | 51.7 % | 95.4 % | 3.38 cm | 1,913,072 | 231.7 s (incl. depth inference) |

- evo associated 219 of the 297 poses with ground truth (mocap timestamps do not cover every sampled frame).
- All 297 frames passed the depth–SfM scale check (no fallback frames); median depth-vs-COLMAP residual 2.18 %.
- Behaviour is consistent with the other TUM sequences (Sim3 scale 0.86, depth AbsRel ≈ 4.5 %); this is one extra
  sequence of the same Kinect, not evidence of general or metric accuracy.

## 6. Phone reconstruction (no independent metric ground truth)

- 213/213 frames registered, 1 COLMAP model, 0.90 px mean reprojection error.
- Final point cloud: 3,629,554 points (`experiments\phone_real_v2\improved\scene.ply`).
- Depth-vs-COLMAP sparse residual 2.96 % (baseline 4.76 %) — agreement with SfM, not with true geometry.
- Local surface thickness 3.65 fused-cm (baseline 3.78).
- Revisit consistency (same surfaces, first vs last 15 frames): 10.20 cm final vs 6.88 cm baseline at TUM-calibrated
  scale, against a 1.0–2.6 cm adjacent-segment floor — the start–end inconsistency (doubled walls) is **worse** with the
  final configuration on this sequence.
- No object was measured: absolute metric accuracy of the phone reconstruction is unknown; units are fused units.

## 7. What was tested (documented experiments, not re-run)

- **Baseline scale + voxel:** per-frame scale alignment + voxel concatenation; best completeness on 3 of 4 datasets.
- **Spatial alignment:** per-frame low-order correction; depth AbsRel −12 % (fr2) / −15 % (fr1_desk) / −51 % (synthetic)
  vs per-frame scale. A *transferred* (fixed) correction field made depth worse and was rejected.
- **Weighted fusion (mv,edge):** confidence-weighted voxels; lower cloud error than plain voxel fusion at a 1–4 point
  completeness cost; far better thin-object completeness than TSDF.
- **TSDF:** thinner surfaces, but 5 cm coverage fell sharply (e.g. fr2 91 → 41 %) and accuracy was mixed → optional mode only.
- **Stronger COLMAP matching:** sequential overlap 30 / exhaustive matching → ATE 212.5 / 249.1 mm vs 83.5 mm control → rejected.
- **Pose confidence:** held-out gain +0.06 / +0.04 / 0.00 cm → failed the pre-registered adoption rule.
- **Frame confidence:** held-out gain +0.02 / +0.31 / 0.00 cm → failed the rule (1 of 3 datasets).
- Also documented: affine alignment (fr2 19.7 vs 8.7 cm), global-scale-only alignment (21.8 vs 8.7 cm), person masking
  on the phone (caused a pose jump).

## 8. Why the final configuration was frozen

Against the baseline, spatial alignment + weighted fusion improved median 3D error, P90 error, points within 5 cm,
depth AbsRel and surface thickness on all four GT datasets (fr1_room 9.64 → 7.95 cm, fr1_desk 5.74 → 5.23, fr2_xyz
8.71 → 6.25, synthetic 9.19 → 5.96 cm median error). Completeness remained better for the baseline on fr1_room and
synth_room2. Every further candidate (TSDF, stronger matching, pose / frame confidence) either failed a pre-defined
rule or traded accuracy / coverage unfavourably. No significance testing was done.

## 9. Known limitations

- **Monocular scale ambiguity:** SfM scale is arbitrary; the depth network's scale is biased (Sim3 scale 0.75–0.84).
- **COLMAP trajectory drift:** fr1_room ATE ≈ 80–90 mm, weakest at 28–30 s (10 frames fall back to global scale).
- **Loop closure:** not solved — start/end same-surface discrepancy ≈ 6.3–6.5 cm on fr1_room; doubled walls on the phone.
- **Depth Anything bias:** overestimates depth; residual per-frame shape error ≈ 4–5 % AbsRel on TUM.
- **Spatial correction transfer:** must be fitted per frame; a fixed transferred field does not generalise.
- **Thin / textureless surfaces:** errors concentrate at object boundaries and small objects; stripes / gaps on plain walls.
- **Reconstruction artifacts:** doubled walls, speckle, points leaking through windows; a person walks through fr1_room (unmasked).
- **No independent metric ground truth for the phone.**

## 10. Visualization

- **PLY:** `experiments\phone_real_v2\improved\scene.ply` (authoritative result; `view_cloud.py` opens it in Open3D).
- **GLB:** `experiments\FINAL_PIPELINE\visualization\final_phone.glb` and `final_phone_with_camera_trajectory.glb`
  (glTF POINTS + camera LINES, modelled on depth-anything.cpp's exporter; positions as in the PLY, colours identical).
- **Interactive viewer (use this for the demo):** `experiments\FINAL_PIPELINE\visualization\v2\open_viewer.bat`.
  Presentation mode by default (light background, round RGB points, no trajectory, overview into the room);
  analysis mode adds the COLMAP camera path, start/end markers and optional frustums; top-view button.
- v1 viewer kept at `visualization\open_viewer.bat`. Static figures: `experiments\FINAL_PIPELINE\figures\`.
- Poisson mesh (`mesh\`): optional, visualization only, of limited use (invents surface in unobserved regions).

## 11. Reproducibility

- **Scripts:** `Reconstruct.py`, `process_video.py`, `colmap_to_tum.py`; evaluation `eval_tum.py`, `eval_synth.py`,
  `eval_aligned_depth.py`, `surface_metrics.py`, `revisit_consistency.py`; reports `FINAL_PIPELINE\collect_final.py`,
  `baseline_vs_final.py`, `synth_rpe.py`, `final_figures.py`; export `visualization\export_glb.py`.
- **Configuration:** `FINAL_PIPELINE\final_config.json`, `experiments\spatial_depth\calib_fr2_xyz_perframe.json`,
  COLMAP control options (`experiments\tum_freiburg1_room\colmap_ws\sparse\project.ini`).
- **Datasets:** TUM RGB-D fr1_room, fr1_desk, fr2_xyz (and fr2_desk) from the TUM website; synthetic room from
  `synth_render.py`; phone clip IMG_8682.MOV (not published).
- **Outputs:** fr1_room `experiments\tum_freiburg1_room\{improved,evaluation}`; fr1_desk / fr2_xyz / synthetic
  `experiments\pose_conf_experiment\<dataset>\arm1_mv_edge`; phone `experiments\phone_real_v2\improved`.
- **Phone command:**
  `python process_video.py --video <clip> --colmap_exe <colmap.exe> --fps 6 --max_width 960 --stride 2 --voxel_size 0.02
  --align spatial --spatial_params experiments\spatial_depth\calib_fr2_xyz_perframe.json --fusion weighted
  --conf_signals mv,edge --no_viewer`
- **Environment:** Python 3.11 venv (`3d-reconstruction\3d-reconstruction\venv`), PyTorch CUDA, Open3D 0.19, evo.
- **Code repository:** https://github.com/A-T-07/Scan2Scene (private; datasets, clouds, videos excluded).

## 12. Final status

**COMPLETE:** reconstruction pipeline · quantitative validation · phone reconstruction · final visualization ·
documentation · figures

**OPTIONAL:** fr2_desk additional validation (completed, §5b) · Poisson mesh

**NOT REQUIRED:** further algorithm tuning
