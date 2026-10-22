# Final Pipeline Status

Monocular RGB video → 3D point cloud. Frozen 2026-09-30. All numbers below are copied from existing evaluation files
(paths in §9); nothing was re-run to produce this document.

## 1. Frozen configuration

```
reconstruct.py --align spatial --spatial_params experiments\spatial_depth\calib_fr2_xyz_perframe.json
               --fusion weighted --conf_signals mv,edge
               --stride 2 --voxel_size 0.02 --depth_trunc 8.0 --no_viewer
```

- COLMAP 4.1.1 (CUDA), `automatic_reconstructor --data_type video --single_camera 1 --dense 0` with the control options
  (SIMPLE_RADIAL, SIFT 8192 features, guided matching, sequential overlap 10 + vocabulary-tree loop detection).
- Depth: `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` (`predicted_depth` output).
- Spatial calibration file contents: model `poly2_xy_logd`, λ = 0.01, clamp 0.2 (±22 %).
- Weighted-fusion thresholds (defaults, fixed a priori): τ = 0.05, τ_hard = 0.15, edge scale 0.05, conf_min 0.05,
  neighbour search ≤ 3 frames, min overlap 0.2, voxel weight ≥ 0.5; statistical outlier removal (20 nb, 2.0 σ).
- Metric calibration: OFF. The TUM-derived factor k = 0.811 is only ever reported separately as "TUM-calibrated".
- Full parameter record: `final_config.json`.

## 2. Pipeline

1. **Input:** phone video (sampled at 6 fps, downscaled to 960 px wide) or TUM RGB frames (RGB only; no sensor depth).
2. **COLMAP SfM:** camera poses, one shared camera, sparse 3D points.
3. **Depth Anything V2 Metric-Indoor-Large:** per-frame dense depth.
4. **Spatial depth alignment:** sequence scale S from COLMAP sparse points; then per frame, a robust low-order
   (poly2 in image position + relative log-depth) correction field fitted to that frame's own sparse points. Frames that
   fail the scale check fall back to the global scale.
5. **Confidence-weighted voxel fusion:** per-pixel confidence = multi-view agreement with neighbouring frames × depth-edge
   factor; confidence-weighted mean per 2 cm voxel; outlier removal.
6. **Output:** coloured point cloud `scene.ply` + diagnostics (`summary.json`, `alignment_diagnostics.csv`, trajectory).

## 3. Quantitative results (final configuration)

| Dataset | Role | Registered | ATE Sim3 RMSE (mm) | RPE Sim3 (mm) | Depth AbsRel | Median 3D err (cm) | P90 (cm) | Within 5 cm | Completeness <5 cm | Surface thickness (cm) | Points | Runtime (s, cached depth) | Cloud-metric script |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fr1_room | development | 272/273 | 89.5 | 28.2 | 9.84% | 7.95 | 34.0 | 37.9% | 68.8% | 3.19 | 1,875,200 | 33.7 | eval_tum.py |
| fr1_desk | held-out | 62/62 | 22.5 | 25.5 | 4.07% | 5.23 | 23.0 | 48.8% | 83.2% | 2.89 | 308,540 | 10.4 | eval_tum.py |
| fr2_xyz | held-out (spatial calibration derived from its own sparse points, no GT) | 367/367 | 7.4 | 3.0 | 4.11% | 6.25 | 32.7 | 43.9% | 92.9% | 3.96 | 1,613,959 | 89.4 | eval_tum.py |
| synth_room2 | synthetic | 90/90 | 7.6 | 3.0 | 1.91% | 5.96 | 16.8 | 45.2% | 70.2% | 1.51 | 715,333 | 13.4 | eval_synth.py (renderer GT; not directly comparable to eval_tum.py) |

- **Sim3 vs absolute:** every 3D and trajectory number above is after a Sim(3) (similarity) alignment to ground truth,
  i.e. global scale is removed. Absolute (SE3) accuracy is much worse because the depth network's scale is biased: evo's
  Sim3 scale correction was 0.817 (fr2_xyz), 0.805 (fr1_desk), 0.843 (fr1_room, fused frame), 0.754 (synthetic render).
- Depth AbsRel (`eval_aligned_depth.py`) and surface thickness (`surface_metrics.py`) use the same script on every row;
  3D accuracy/completeness use `eval_tum.py` (TUM) and `eval_synth.py` (synthetic), which are not interchangeable.
- Cross-dataset comparison of 3D numbers is not meaningful (different scenes, sampling and ground-truth sources).

## 4. Baseline vs final

Baseline `--align scale --fusion voxel` vs final, same frames/poses/depth (details: `baseline_vs_final.md`).

| Dataset | Median 3D err (cm) | P90 (cm) | Within 5 cm | Completeness <5 cm | Depth AbsRel | Thickness (cm) |
|---|---|---|---|---|---|---|
| fr1_room | 9.64 → **7.95** | 42.4 → **34.0** | 33.1 → **37.9 %** | **72.4** → 68.8 % | 9.99 → **9.84 %** | 3.34 → **3.19** |
| fr1_desk | 5.74 → **5.23** | 25.6 → **23.0** | 46.2 → **48.8 %** | 83.2 → 83.2 % (tie) | 4.79 → **4.07 %** | 3.06 → **2.89** |
| fr2_xyz | 8.71 → **6.25** | 40.3 → **32.7** | 35.6 → **43.9 %** | 90.4 → **92.9 %** | 4.89 → **4.11 %** | 4.00 → 3.96 (≈ tie) |
| synth_room2 | 9.19 → **5.96** | 31.1 → **16.8** | 35.1 → **45.2 %** | **78.6** → 70.2 % | 3.93 → **1.91 %** | 2.54 → **1.51** |

Spatial alignment + weighted fusion improved median / P90 3D error, points within 5 cm and depth AbsRel on all four
datasets, and surface thickness (marginally on fr2_xyz). Completeness remained better for the baseline on fr1_room and
synth_room2 (and the completeness median on fr1_desk). The final cloud has 9–46 % fewer points. No significance testing
was done; differences of a few tenths are treated as ties. Trajectory metrics are identical (same poses).

## 5. Phone reconstruction (no metric ground truth)

Separate report: `PHONE_RESULTS.md`. IMG_8682.MOV (4K, 59.9 fps, 35.5 s) → 213 frames at 6 fps, 960×540.

- **COLMAP [measured]:** 213/213 registered, 1 model, 0.90 px reprojection error, no step > 2.94× median.
- **Reconstruction [measured]:** baseline 5,277,004 pts vs final 3,629,554 pts; depth-vs-COLMAP residual 4.76 % → 2.96 %
  (agreement with SfM, not with truth); local thickness 3.78 → 3.65 fused-cm; 0.56 % of pixels rejected by confidence.
- **Consistency [measured / TUM-calibrated]:** start vs end surfaces differ by 6.88 cm (baseline) / 10.20 cm (final)
  TUM-calibrated, against a 1–2.6 cm adjacent-segment floor. The final configuration is **worse** than the baseline here.
- **Qualitative:** full room recognisable; doubled/thick walls from start–end inconsistency; gaps and stripes on plain surfaces.
- **Metric accuracy:** not validated — no measured object in this recording. Any dimension × 0.811 is a TUM-calibrated estimate only.

## 6. What was tested and rejected

| Approach | Evidence | Why not adopted |
|---|---|---|
| Stronger sequential matching (overlap 30, loop period 5, 100 loop images) | fr1_room pose-only: ATE Sim3 212.5 mm vs control 83.5 mm (mean of 3 mapper runs; control spread 79.6–89.5) | 2.5× worse trajectory |
| Exhaustive matching (all 37,128 pairs) | ATE Sim3 249.1 mm vs 83.5 mm; start/end error 201.6 vs 60.8 mm | 3× worse trajectory; cause not established (grossly implausible pairs < 1 %) |
| Pose confidence signal (track length × observations) | held-out median-error change +0.06 / +0.04 / 0.00 cm (fr1_desk / fr2_xyz / synth) | failed the pre-registered rule (needs ≥ 0.3 cm on 2 of 3) |
| Frame confidence signal (alignment residual) | held-out +0.02 / +0.31 / 0.00 cm (large gain only on fr1_room, the development set, via its 10 fallback frames) | failed the pre-registered rule (1 of 3) |
| TSDF fusion as default | thinner surfaces (local thickness fr2 4.00 → 2.29, fr1_desk 3.06 → 1.97 cm) but 5 cm coverage fr2 91 → 41 %, fr1_desk 63 → 47 %; accuracy mixed. With spatial alignment (surface_metrics.py): fr2 median error 7.31 cm / coverage 32 % vs 4.25 cm / 94 % for weighted mv,edge | kept only as an optional surface / mesh mode (`--fusion tsdf`) |
| Per-frame affine depth alignment | fr2_xyz cloud error 19.7 cm vs 8.7 cm (scale); 141 / 367 fits rejected | ill-conditioned over narrow depth ranges |
| Transferred (fixed) spatial correction field | AbsRel fr1_desk 5.28 % vs 4.79 % baseline; synth 8.14 % vs 3.93 % | residual field is scene-specific; per-frame fitting used instead |
| Person masking on phone v2 | 8.8× pose jump at samples 83–86 + spurious 5-frame model | hurt tracking; person occupies < 1 % of image |
| Global-scale-only alignment | fr2 cloud error 21.8 cm vs 8.7 cm with per-frame scale | superseded by per-frame / spatial alignment |

## 7. Limitations

- **COLMAP trajectory / drift:** fr1_room ATE Sim3 ≈ 80–90 mm (median 45–48 mm), weakest at 28–30 s, where 10 frames fail
  depth–SfM scale checks and are fused at the global scale (≈ 5 % of the cloud; measured effect on accuracy ≤ 0.12 cm).
  Start/end same-surface discrepancy ≈ 6.3–6.5 cm vs a ≈ 1.1 cm noise floor.
- **Loop closure is not solved.** COLMAP loop detection is active, but start/end inconsistency remains on fr1_room and the
  phone sequence; the phone start–end discrepancy is larger with the final configuration than with the baseline.
- **Depth scale bias:** Depth Anything overestimates depth (Sim3 scale 0.75–0.84). The pipeline output is not metric.
- **No independent metric ground truth for the phone.** Metric accuracy of the phone reconstruction is unknown.
- **TUM calibration limits:** k = 0.811 comes from two sequences of one Kinect camera; leave-one-out residual scale was
  0.986 / 1.015 on TUM, but transfer to phone cameras is unverified (phone camera-height check suggests it may underestimate).
- **Spatial correction is not universally transferable:** it must be fitted per frame; a transferred field made depth worse.
- **Thin structures / textureless surfaces:** remaining error concentrates at object boundaries, thin and small objects;
  plain walls / wardrobe doors produce stripes and gaps; completeness is lower than the baseline on 2–3 of 4 datasets.
- **Dynamic content:** fr1_room contains a person walking through part of the sequence (around input frame 163); no masking is applied.
- **Confidence quality varies:** AUROC of the combined confidence for flagging bad (> 10 % error) pixels 0.65 on fr1_room
  vs 0.72 (fr2_xyz) / 0.86 (fr1_desk); these were measured with the four-signal confidence used in the weighted-fusion study.

## 8. Reproducibility

- Final flags: see §1 (identical for every dataset; `final_config.json` holds every parameter).
- Code: `Q:\OPEN_CV_SEM5\Reconstruct.py`, `process_video.py` (video → frames → COLMAP; `--max_width 960 --fps 6` for phone),
  `colmap_to_tum.py`. Calibration input: `experiments\spatial_depth\calib_fr2_xyz_perframe.json`.
- Phone end-to-end:
  `python process_video.py --video <clip> --colmap_exe <colmap.exe> --fps 6 --max_width 960 --stride 2 --voxel_size 0.02 --align spatial
  --spatial_params experiments\spatial_depth\calib_fr2_xyz_perframe.json --fusion weighted --conf_signals mv,edge --no_viewer`
- TUM: copy every Nth RGB frame (fr1_room every 5th), run COLMAP as in §1, then `reconstruct.py` with the §1 flags
  (`experiments\run_tum_fr1_room_dense.ps1`).
- Evaluation: `eval_tum.py`, `eval_synth.py`, `eval_aligned_depth.py`, `surface_metrics.py`, `revisit_consistency.py`;
  report generation: `FINAL_PIPELINE\collect_final.py`, `baseline_vs_final.py`, `synth_rpe.py`, `final_figures.py`.
- Environment: Python 3.11 venv (`3d-reconstruction\3d-reconstruction\venv`), PyTorch CUDA, Open3D 0.19, evo; COLMAP 4.1.1.

## 9. Output inventory

| What | Where |
|---|---|
| Frozen config / final tables | `experiments\FINAL_PIPELINE\final_config.json`, `final_results.{json,md}`, `baseline_vs_final.{json,md}`, `synth_rpe.json` |
| Phone report | `experiments\FINAL_PIPELINE\PHONE_RESULTS.md` |
| Figures | `experiments\FINAL_PIPELINE\figures\` (fr1_room final + trajectory, baseline-vs-final for all 4 datasets, phone final views / close-ups) |
| fr1_room final / baseline clouds + evaluation | `experiments\tum_freiburg1_room\{improved,baseline}\scene.ply`, `...\evaluation\` |
| fr1_desk / fr2_xyz / synth final clouds + evaluation | `experiments\pose_conf_experiment\<dataset>\arm1_mv_edge\` |
| Phone final / baseline clouds | `experiments\phone_real_v2\{improved,baseline}\scene.ply` |
| Pose-variant study | `experiments\tum_freiburg1_room\pose_variants\` |
| Pose / frame confidence study | `experiments\pose_conf_experiment\` |
| TSDF / spatial / weighted studies | `experiments\tsdf_comparison\`, `experiments\spatial_depth\`, `experiments\weighted_fusion\` |
| Optional visualization mesh | `experiments\FINAL_PIPELINE\mesh\` (see `mesh\MESH_NOTES.md`) |

## 10. Project status

**COMPLETE**
- End-to-end RGB-only pipeline (video / frames → COLMAP → Depth Anything V2 → spatial alignment → weighted fusion).
- Frozen configuration and parameter record.
- TUM evaluation on 3 real sequences (1 development, 2 held-out) + 1 synthetic, Sim3-aligned, RGB-only.
- Controlled ablations (matching variants, pose / frame confidence, TSDF, affine / transferred alignment).
- Real phone reconstruction (qualitative demonstration with internal consistency metrics).

**OPTIONAL**
- `--fusion tsdf` surface / mesh mode; `--metric_scale` TUM calibration; Poisson visualization mesh.

**NOT VALIDATED**
- Metric accuracy on phone footage (no measured object; loop-closing recording prepared but not captured).
- Absolute (SE3) metric accuracy on TUM without Sim3 alignment.
- Generalisation beyond the tested indoor scenes and cameras.
