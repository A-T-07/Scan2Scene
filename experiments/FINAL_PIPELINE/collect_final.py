"""Freeze record + final results table for the approved pipeline (reads existing outputs only)."""
import json
import os

E = r"Q:\OPEN_CV_SEM5\experiments"
OUT = os.path.join(E, "FINAL_PIPELINE")
j = lambda *p: json.load(open(os.path.join(E, *p)))

s = j("tum_freiburg1_room", "improved", "summary.json")
a = s["args"]
calib = json.load(open(os.path.join(r"Q:\OPEN_CV_SEM5", a["spatial_params"])))
config = dict(
    colmap="COLMAP 4.1.1 (commit a0d785f, CUDA); automatic_reconstructor --data_type video --single_camera 1 --dense 0 "
           "(SIMPLE_RADIAL, SIFT 8192 feats, guided matching, sequential overlap 10 + vocab-tree loop detection)",
    depth_model=a["model"], alignment="--align spatial (per-frame low-order field fitted to COLMAP sparse points)",
    spatial_calibration_file=a["spatial_params"], spatial_calibration=calib,
    fusion="--fusion weighted (confidence-weighted 2 cm voxel average + statistical outlier removal nb=20, std=2.0)",
    conf_signals=a["conf_signals"],
    fixed_parameters={k: a[k] for k in ("voxel_size", "stride", "depth_trunc", "depth_min", "depth_max", "edge_thresh",
                                        "min_corr", "inlier_tol", "max_scale_dev", "max_reproj_err", "min_track",
                                        "conf_tau", "conf_tau_hard", "conf_edge", "conf_min", "conf_max_gap",
                                        "conf_min_overlap", "weighted_wmin")},
    frame_sampling={"fr1_room": "every 5th RGB frame (6 fps), 640x480", "fr1_desk": "every 10th (3 fps), 640x480",
                    "fr2_xyz": "every 10th, 640x480", "synth_room2": "all 90 rendered frames, 640x480",
                    "phone_real_v2": "6 fps, downscaled to 960x540 (--max_width 960)"},
    metric_calibration="OFF (no --metric_scale); TUM-calibrated factor 0.811 is reported separately only",
)
json.dump(config, open(os.path.join(OUT, "final_config.json"), "w"), indent=2)

rows = []
for name, role, run, ev, kind in (
        ("fr1_room", "development", r"tum_freiburg1_room\improved", r"tum_freiburg1_room\evaluation\improved", "tum"),
        ("fr1_desk", "held-out", r"pose_conf_experiment\fr1_desk\arm1_mv_edge", r"pose_conf_experiment\fr1_desk\arm1_mv_edge", "tum"),
        ("fr2_xyz", "held-out (spatial calibration derived from its own sparse points, no GT)", r"pose_conf_experiment\fr2_xyz\arm1_mv_edge", r"pose_conf_experiment\fr2_xyz\arm1_mv_edge", "tum"),
        ("synth_room2", "synthetic", r"pose_conf_experiment\synth_room2\arm1_mv_edge", r"pose_conf_experiment\synth_room2\arm1_mv_edge", "synth")):
    sm, de, su = j(ev, "surface_metrics.json"), j(ev, "depth_eval.json"), j(run, "summary.json")
    if kind == "tum":
        t = j(ev, "eval_tum.json"); c = t["cloud"]["sim3"]; evo = t["evo"]
        r = dict(ate=evo["ape_sim3_rmse_m"] * 1000, rpe=evo["rpe_sim3_rmse_m"] * 1000, acc=c["accuracy_median_cm"],
                 p90=c["accuracy_p90_cm"], w5=c["frac_within_5cm"] * 100, comp=c["completeness_frac_within_5cm"] * 100,
                 script="eval_tum.py")
    else:
        t = j(ev, "eval_synth.json"); c = t["cloud"]["sim3"]
        rp = os.path.join(OUT, "synth_rpe.json")   # trajectory-only evo run (synth_rpe.py); eval_synth.py has no RPE
        rpe = json.load(open(rp))["rpe_sim3_rmse_m"] * 1000 if os.path.exists(rp) else None
        r = dict(ate=t["trajectory"]["ate_sim3_rmse_m"] * 1000, rpe=rpe, acc=c["accuracy_median_m"] * 100,
                 p90=c["accuracy_p90_m"] * 100, w5=c["frac_within_5cm"] * 100, comp=c["completeness_frac_within_5cm"] * 100,
                 script="eval_synth.py (renderer GT; not directly comparable to eval_tum.py)")
    r.update(dataset=name, role=role, registered=f"{su['frames_registered']}", absrel=de["absrel_mean"] * 100,
             thick=sm["local_thickness_fused"]["median_cm"], points=su["points_final"], runtime=su["timing_s"]["total_s"])
    rows.append(r)
json.dump(rows, open(os.path.join(OUT, "final_results.json"), "w"), indent=2)
md = ["| Dataset | Role | Registered | ATE Sim3 RMSE (mm) | RPE Sim3 (mm) | Depth AbsRel | Median 3D err (cm) | P90 (cm) | Within 5 cm | Completeness <5 cm | Surface thickness (cm) | Points | Runtime (s, cached depth) | Cloud-metric script |",
      "|" + "---|" * 14]
for r in rows:
    rpe = f"{r['rpe']:.1f}" if r["rpe"] is not None else "not computed"
    md.append(f"| {r['dataset']} | {r['role']} | {r['registered']} | {r['ate']:.1f} | {rpe} | {r['absrel']:.2f}% | {r['acc']:.2f} | "
              f"{r['p90']:.1f} | {r['w5']:.1f}% | {r['comp']:.1f}% | {r['thick']:.2f} | {r['points']:,} | {r['runtime']:.1f} | {r['script']} |")
md.append("\nAll 3D metrics are after Sim(3) alignment to ground truth (not absolute metric evaluation). "
          "Depth AbsRel: eval_aligned_depth.py; thickness: surface_metrics.py (same script for all rows).\n"
          "ATE/RPE: evo (-as, delta = 1 sampled frame) for all rows; synth_room2 RPE from synth_rpe.py "
          "(trajectory-only evo run on the existing trajectory_fused.txt vs the renderer trajectory, see synth_rpe.json); "
          "synth_room2 ATE from eval_synth.py (7.6 mm, identical to evo's 7.63 mm).")
open(os.path.join(OUT, "final_results.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
