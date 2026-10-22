| Dataset | Role | Registered | ATE Sim3 RMSE (mm) | RPE Sim3 (mm) | Depth AbsRel | Median 3D err (cm) | P90 (cm) | Within 5 cm | Completeness <5 cm | Surface thickness (cm) | Points | Runtime (s, cached depth) | Cloud-metric script |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fr1_room | development | 272 | 89.5 | 28.2 | 9.84% | 7.95 | 34.0 | 37.9% | 68.8% | 3.19 | 1,875,200 | 33.7 | eval_tum.py |
| fr1_desk | held-out | 62 | 22.5 | 25.5 | 4.07% | 5.23 | 23.0 | 48.8% | 83.2% | 2.89 | 308,540 | 10.4 | eval_tum.py |
| fr2_xyz | held-out (spatial calibration derived from its own sparse points, no GT) | 367 | 7.4 | 3.0 | 4.11% | 6.25 | 32.7 | 43.9% | 92.9% | 3.96 | 1,613,959 | 89.4 | eval_tum.py |
| synth_room2 | synthetic | 90 | 7.6 | 3.0 | 1.91% | 5.96 | 16.8 | 45.2% | 70.2% | 1.51 | 715,333 | 13.4 | eval_synth.py (renderer GT; not directly comparable to eval_tum.py) |

All 3D metrics are after Sim(3) alignment to ground truth (not absolute metric evaluation). Depth AbsRel: eval_aligned_depth.py; thickness: surface_metrics.py (same script for all rows).
ATE/RPE: evo (-as, delta = 1 sampled frame) for all rows; synth_room2 RPE from synth_rpe.py (trajectory-only evo run on the existing trajectory_fused.txt vs the renderer trajectory, see synth_rpe.json); synth_room2 ATE from eval_synth.py (7.6 mm, identical to evo's 7.63 mm).
