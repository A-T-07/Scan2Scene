# Baseline vs final configuration

Baseline: `--align scale --fusion voxel`. Final (frozen): `--align spatial --fusion weighted --conf_signals mv,edge`.
Same frames, COLMAP poses, cached depth, voxel 0.02, depth truncation 8.0 and pixel stride 2 within each dataset. All values come from existing evaluation files (see baseline_vs_final.json for exact paths); nothing was rerun.

**Comparability:** compare baseline vs final *within* a row of one dataset. Across datasets, 3D metrics are not directly comparable (different scenes, ground-truth sources and sampling); synth_room2 3D metrics come from `eval_synth.py` (renderer ground truth), TUM rows from `eval_tum.py` (Kinect depth). All 3D metrics are after Sim(3) alignment, not absolute metric evaluation. Trajectory metrics are identical for baseline and final (same COLMAP poses).

## fr1_room (development)

| Metric | Baseline | Final | Better |
|---|---|---|---|
| Median 3D error (cm, Sim3) | 9.64 | 7.95 | final |
| P90 3D error (cm, Sim3) | 42.4 | 34.0 | final |
| Points within 5 cm (%) | 33.1 | 37.9 | final |
| Completeness median (cm) | 1.49 | 1.85 | baseline |
| Completeness within 5 cm (%) | 72.4 | 68.8 | baseline |
| Depth AbsRel (%) | 9.99 | 9.84 | final |
| Local surface thickness (cm) | 3.34 | 3.19 | final |
| Start-vs-end revisit (fused units ×100) | 7.66 | 7.52 | final |
| Final points | 2,457,590 | 1,875,200 | — |
| Runtime (s) | 168.9 | 33.7 | — |
| ATE Sim3 RMSE (mm) | 89.5 | 89.5 | same poses |
| RPE Sim3 (mm) | 28.2 | 28.2 | same poses |

- Runtimes are not comparable: one run includes depth-network inference, the other used cached depth.

## fr1_desk (held-out)

| Metric | Baseline | Final | Better |
|---|---|---|---|
| Median 3D error (cm, Sim3) | 5.74 | 5.23 | final |
| P90 3D error (cm, Sim3) | 25.6 | 23.0 | final |
| Points within 5 cm (%) | 46.2 | 48.8 | final |
| Completeness median (cm) | 1.14 | 1.39 | baseline |
| Completeness within 5 cm (%) | 83.2 | 83.2 | tie |
| Depth AbsRel (%) | 4.79 | 4.07 | final |
| Local surface thickness (cm) | 3.06 | 2.89 | final |
| Start-vs-end revisit (fused units ×100) | not available | 1.64 | — |
| Final points | 397,020 | 308,540 | — |
| Runtime (s) | 2.2 | 10.4 | — |
| ATE Sim3 RMSE (mm) | 22.5 | 22.5 | same poses |
| RPE Sim3 (mm) | 25.5 | 25.5 | same poses |

- Baseline cloud metrics are from the earlier identical-configuration run (397,020 points, identical to the 397,020-point run used for depth/thickness/runtime).
- Baseline revisit consistency was never computed for this dataset: not available.

## fr2_xyz (held-out)

| Metric | Baseline | Final | Better |
|---|---|---|---|
| Median 3D error (cm, Sim3) | 8.71 | 6.25 | final |
| P90 3D error (cm, Sim3) | 40.3 | 32.7 | final |
| Points within 5 cm (%) | 35.6 | 43.9 | final |
| Completeness median (cm) | 0.95 | 0.93 | final |
| Completeness within 5 cm (%) | 90.4 | 92.9 | final |
| Depth AbsRel (%) | 4.89 | 4.11 | final |
| Local surface thickness (cm) | 4.00 | 3.96 | final |
| Start-vs-end revisit (fused units ×100) | not available | 12.62 | — |
| Final points | 1,764,277 | 1,613,959 | — |
| Runtime (s) | 64.0 | 89.4 | — |
| ATE Sim3 RMSE (mm) | 7.4 | 7.4 | same poses |
| RPE Sim3 (mm) | 3.0 | 3.0 | same poses |

- Baseline cloud metrics are from the earlier identical-configuration run (1,764,277 points, identical to the 1,764,277-point run used for depth/thickness/runtime).
- Final spatial calibration (`calib_fr2_xyz_perframe.json`) was derived from fr2_xyz's own COLMAP sparse points (no GT).
- Baseline revisit consistency was never computed for this dataset: not available.

## synth_room2 (synthetic)

| Metric | Baseline | Final | Better |
|---|---|---|---|
| Median 3D error (cm, Sim3) | 9.19 | 5.96 | final |
| P90 3D error (cm, Sim3) | 31.1 | 16.8 | final |
| Points within 5 cm (%) | 35.1 | 45.2 | final |
| Completeness median (cm) | 1.35 | 2.03 | baseline |
| Completeness within 5 cm (%) | 78.6 | 70.2 | baseline |
| Depth AbsRel (%) | 3.93 | 1.91 | final |
| Local surface thickness (cm) | 2.54 | 1.51 | final |
| Start-vs-end revisit (fused units ×100) | not available | 2.36 | — |
| Final points | 1,324,283 | 715,333 | — |
| Runtime (s) | 6.4 | 13.4 | — |
| ATE Sim3 RMSE (mm) | 7.6 | 7.6 | same poses |
| RPE Sim3 (mm) | 3.0 | 3.0 | same poses |

- Baseline cloud metrics are from the earlier identical-configuration run (1,324,283 points, identical to the 1,324,283-point run used for depth/thickness/runtime).
- Baseline revisit consistency was never computed for this dataset: not available.

synth_room2 RPE (both columns): trajectory-only evo run on the final run's trajectory_fused.txt (FINAL_PIPELINE\synth_rpe.json); valid for both because the COLMAP poses are identical.

## Summary of measured changes

- **median 3D error:** final better on fr1_room, fr1_desk, fr2_xyz, synth_room2; baseline better on none.
- **P90 3D error:** final better on fr1_room, fr1_desk, fr2_xyz, synth_room2; baseline better on none.
- **points within 5 cm:** final better on fr1_room, fr1_desk, fr2_xyz, synth_room2; baseline better on none.
- **depth AbsRel:** final better on fr1_room, fr1_desk, fr2_xyz, synth_room2; baseline better on none.
- **local surface thickness:** final better on fr1_room, fr1_desk, fr2_xyz, synth_room2; baseline better on none.
- **completeness median:** final better on fr2_xyz; baseline better on fr1_room, fr1_desk, synth_room2.
- **completeness within 5 cm:** final better on fr2_xyz; baseline better on fr1_room, fr1_desk, synth_room2.

The final configuration improved several 3D accuracy / surface metrics on every dataset, while completeness remained better for the baseline on fr1_room and synth_room2 (and the completeness median on fr1_desk). No significance testing was done; differences of a few tenths of a cm or percentage point (e.g. fr2_xyz thickness 4.00 → 3.96 cm, fr1_room revisit 7.66 → 7.52) should be treated as ties.
