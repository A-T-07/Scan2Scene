# Real phone reconstruction — results (phone_real_v2)

**Status: qualitative demonstration. This sequence has NO independent metric ground truth. Nothing in this report is a
validated metric accuracy.**

Labels used below:

- **[M] measured** — computed directly from reconstruction outputs (internal quantities; not compared to any ground truth)
- **[I] internal / TUM-calibrated estimate** — derived with an assumption (e.g. the TUM-derived scale factor k = 0.811); NOT independently validated
- **[Q] qualitative** — visual observation from rendered figures

Units: *COLMAP units* = arbitrary SfM scale. *Fused units* = the reconstruction output frame (COLMAP rescaled by the depth
network's median scale; 1 fused unit = 2.414 COLMAP units for this run). "cm" values in fused units are fused units × 100
unless marked TUM-calibrated.

## 1. Source video

| Quantity | Value | Label |
|---|---|---|
| File | `C:\Users\tyagi\Downloads\IMG_8682.MOV` | — |
| Resolution / frame rate | 3840×2160, 59.88 fps (the user's pre-inspection note of 910×512 / 30 fps did not match the decoded file) | [M] |
| Duration / frames | 35.5 s, 2,125 frames | [M] |
| Sharpness (variance of Laplacian, p10/p50/p90, 4K) | 10.8 / 24.4 / 38.3; 12.2 % of frames below half the median (clustered at 0–3 s and 25.5–29.5 s) | [M] |
| Content | single room sweep with walking translation: door → shelves/desk → bed/window → wardrobe → second desk → window → door | [Q] |

Source: `experiments\phone_real_v2\inspect\inspect.json`, `inspect\contact_sheet.jpg`, `inspect\motion_sharpness.png`.

## 2. Frame sampling and COLMAP poses (shared by baseline and final)

| Quantity | Value | Label |
|---|---|---|
| Sampling | 6 fps (every 10th frame), downscaled to 960×540 (`process_video.py --max_width 960`) | — |
| Sampled / registered | 213 / 213 (100 %) | [M] |
| COLMAP models | 1 | [M] |
| Camera | SIMPLE_RADIAL, f = 419.9 px, k1 ≈ −0.0002 (single shared camera) | [M] |
| Mean reprojection error | 0.90 px; 21,392 sparse points | [M] |
| Inter-sample rotation median / max | 1.34° / 4.72°; total 348.8° | [M] |
| Inter-sample translation median / max | 0.090 / 0.263 COLMAP units (max = 2.94× median; no step > 5× median) | [M] |
| Weakest consecutive verified pair | 259 inliers | [M] |
| Camera path length | 20.74 COLMAP units = 8.59 fused units | [M] |
| COLMAP runtime | 28 s extraction + 118 s SfM | [M] |

Person-mask ablation (pose_B_mask_960): also 213/213 registered, but a 5-frame spurious second model and an 8.8× pose
jump at samples 83–86; the unmasked poses (pose_A) were used. Sources: `pose_A_nomask_960\colmap_report.json`,
`pose_A_nomask_960\pose_diagnostics.json`, `pose_B_mask_960\colmap_report.json`.

## 3. Dense reconstruction — baseline vs final

Baseline: `--align scale --fusion voxel`. Final (frozen): `--align spatial --fusion weighted --conf_signals mv,edge`
(spatial calibration `experiments\spatial_depth\calib_fr2_xyz_perframe.json`). Identical frames, poses, depth cache,
voxel 0.02, depth truncation 8.0, pixel stride 2.

| Quantity | Baseline (scale + voxel) | Final (spatial + weighted, mv,edge) | Label |
|---|---|---|---|
| Frames fused | 213 (all `ok`) | 213 (all `ok`, spatial fit on all 213) | [M] |
| Final points | 5,277,004 | 3,629,554 | [M] |
| Raw back-projected points | 27,260,639 | 27,107,170 confident | [M] |
| Median depth-vs-COLMAP sparse residual | 4.76 % | 2.96 % | [M] (agreement with COLMAP, not with true geometry) |
| Local surface thickness, median (10 cm neighbourhoods) | 3.78 (p75 4.03) fused-cm | 3.65 (p75 4.11) fused-cm | [M] |
| Confidence | — | mean 0.872; 0.56 % of valid pixels rejected | [M] |
| Runtime | 129.7 s (incl. 104.7 s depth inference) | 63.7 s (depth cached; 35.6 s consistency/confidence) | [M] |
| Global scale S | 2.414 COLMAP units per predicted metre | same | [M] |
| Extent (p1–p99, gravity-aligned) | 8.82 × 9.32 × 4.28 fused units | 8.62 × 9.48 × 4.42 fused units | [M] |
| Extent × k_TUM | 7.15 × 7.56 × 3.47 m | 6.99 × 7.69 × 3.58 m | [I] — axis-aligned extents of a room rotated ~45°, including out-of-room points; overstates room size; NOT a room measurement |

Sources: `baseline\summary.json`, `improved\summary.json`, `phone_report.json`.

Plane-fit "thickness" in `phone_report.json` (robust std ≈ 8–11 cm, p5–p95 ≈ 26 cm for every plane in both runs) is
dominated by the ±15 cm evaluation band and neighbouring structures; it is not a usable thickness measure and is not reported.

## 4. Trajectory and revisit consistency (internal)

| Quantity | Baseline | Final | Label |
|---|---|---|---|
| Start/end camera distance | 2.347 fused units (27.3 % of path) | same (same poses) | [M] — the recording did not return to its start; not a drift measurement by itself |
| Camera height above reconstructed floor (median) | 1.007 fused units | same | [M]; ×k_TUM = 0.82 m [I] — low for a hand-held phone, suggests the TUM-calibrated scale may underestimate here |
| Same surfaces, first 15 vs last 15 frames (median distance) | 8.48 fused-cm (overlap 84 %) → 6.88 cm TUM-calibrated | 12.57 fused-cm (overlap 86 %) → 10.20 cm TUM-calibrated | [M] / [I] |
| Noise floor: adjacent mid-sequence segments | 3.23 fused-cm → 2.62 cm TUM-cal. | 1.22 fused-cm → 0.99 cm TUM-cal. | [M] / [I] |

Interpretation: start-vs-end surfaces disagree by ~2.6× (baseline) and ~10× (final) the adjacent-segment noise floor —
a real global inconsistency (doubled walls). The final run's noise floor is biased low because its own confidence already
checks adjacent frames; the start-vs-end value is the fair comparison, and on it the **final run is worse than the baseline**.
COLMAP's loop detection found start↔end matches in this run (e.g. frames 10↔212, 107 inliers), so the residual doubling is
not simply a missing loop constraint. Sources: `baseline\revisit_consistency.json`, `improved\revisit_consistency.json`,
`loop_pairs.py` output.

## 5. Observed artifacts [Q]

From `experiments\phone_real_v2\figures\` (cloud_main/front/side, topdown, closeups, depth_confidence_examples):

- the full room outline, door, shelves, desk, bed, window, wardrobe and second desk are recognisable in both runs;
- walls appear as doubled / thickened bands in the top view, strongest along the start/end wall;
- striping and gaps on untextured surfaces (wardrobe doors, white walls, window cloth);
- the start-of-sequence close-up at the door is blurred (motion blur in the first 3 s);
- a small number of points leak outside the room through the window / grille;
- the final run shows slightly crisper wall bands in the top view and fewer points; the visual difference is modest.

## 6. Limitations

- **No independent metric ground truth.** No object was tape-measured for this recording; metric accuracy cannot be
  verified. Dimensions × k_TUM are TUM-calibrated estimates only (k = 0.811 from TUM fr2_xyz / fr1_desk, a Kinect camera).
- Depth-vs-COLMAP residual measures agreement with the sparse SfM geometry, not with the true scene.
- Global inconsistency between start and end of the sweep (doubled walls) is not resolved by the final fusion
  configuration; on the revisit metric it is worse than the baseline.
- Local surface thickness changes only marginally (3.78 → 3.65 fused-cm).
- A loop-closing recording with a measured object (`experiments\run_phone_loop.ps1`, `measure_object.py`) was prepared
  but has not been recorded.
