"""Baseline (--align scale --fusion voxel) vs final (--align spatial --fusion weighted --conf_signals mv,edge),
from already-computed evaluation outputs only. Writes baseline_vs_final.json and baseline_vs_final.md."""
import json
import os

R = r"Q:\OPEN_CV_SEM5"
OUT = os.path.join(R, r"experiments\FINAL_PIPELINE")


def j(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


SRC = {
    "fr1_room": dict(
        base=dict(cloud=r"experiments\tum_freiburg1_room\evaluation\baseline\eval_tum.json",
                  depth=r"experiments\tum_freiburg1_room\evaluation\baseline\depth_eval.json",
                  surf=r"experiments\tum_freiburg1_room\evaluation\baseline\surface_metrics.json",
                  summ=r"experiments\tum_freiburg1_room\baseline\summary.json",
                  revisit=r"experiments\tum_freiburg1_room\evaluation\baseline\revisit_consistency.json"),
        final=dict(cloud=r"experiments\tum_freiburg1_room\evaluation\improved\eval_tum.json",
                   depth=r"experiments\tum_freiburg1_room\evaluation\improved\depth_eval.json",
                   surf=r"experiments\tum_freiburg1_room\evaluation\improved\surface_metrics.json",
                   summ=r"experiments\tum_freiburg1_room\improved\summary.json",
                   revisit=r"experiments\tum_freiburg1_room\evaluation\improved\revisit_consistency.json"), kind="tum"),
    "fr1_desk": dict(
        base=dict(cloud=r"experiments\tum_freiburg1_desk\corrected_scale\eval_tum_rerun.json",
                  depth=r"experiments\spatial_depth\runs\fr1_desk\baseline_scale\depth_eval.json",
                  surf=r"experiments\spatial_depth\runs\fr1_desk\baseline_scale\surface_metrics.json",
                  summ=r"experiments\spatial_depth\runs\fr1_desk\baseline_scale\summary.json",
                  summ_cloud=r"experiments\tum_freiburg1_desk\corrected_scale\summary.json", revisit=None),
        final=dict(cloud=r"experiments\pose_conf_experiment\fr1_desk\arm1_mv_edge\eval_tum.json",
                   depth=r"experiments\pose_conf_experiment\fr1_desk\arm1_mv_edge\depth_eval.json",
                   surf=r"experiments\pose_conf_experiment\fr1_desk\arm1_mv_edge\surface_metrics.json",
                   summ=r"experiments\pose_conf_experiment\fr1_desk\arm1_mv_edge\summary.json",
                   revisit=r"experiments\pose_conf_experiment\fr1_desk\arm1_mv_edge\revisit_consistency.json"), kind="tum"),
    "fr2_xyz": dict(
        base=dict(cloud=r"experiments\tum_fr2_xyz\corrected_scale\eval_tum_rerun.json",
                  depth=r"experiments\spatial_depth\runs\fr2_xyz\baseline_scale\depth_eval.json",
                  surf=r"experiments\spatial_depth\runs\fr2_xyz\baseline_scale\surface_metrics.json",
                  summ=r"experiments\spatial_depth\runs\fr2_xyz\baseline_scale\summary.json",
                  summ_cloud=r"experiments\tum_fr2_xyz\corrected_scale\summary.json", revisit=None),
        final=dict(cloud=r"experiments\pose_conf_experiment\fr2_xyz\arm1_mv_edge\eval_tum.json",
                   depth=r"experiments\pose_conf_experiment\fr2_xyz\arm1_mv_edge\depth_eval.json",
                   surf=r"experiments\pose_conf_experiment\fr2_xyz\arm1_mv_edge\surface_metrics.json",
                   summ=r"experiments\pose_conf_experiment\fr2_xyz\arm1_mv_edge\summary.json",
                   revisit=r"experiments\pose_conf_experiment\fr2_xyz\arm1_mv_edge\revisit_consistency.json"), kind="tum"),
    "synth_room2": dict(
        base=dict(cloud=r"synth_room2\recon_scale\eval_synth.json",
                  depth=r"experiments\spatial_depth\runs\synth_room2\baseline_scale\depth_eval.json",
                  surf=r"experiments\spatial_depth\runs\synth_room2\baseline_scale\surface_metrics.json",
                  summ=r"experiments\spatial_depth\runs\synth_room2\baseline_scale\summary.json",
                  summ_cloud=r"synth_room2\recon_scale\summary.json", revisit=None),
        final=dict(cloud=r"experiments\pose_conf_experiment\synth_room2\arm1_mv_edge\eval_synth.json",
                   depth=r"experiments\pose_conf_experiment\synth_room2\arm1_mv_edge\depth_eval.json",
                   surf=r"experiments\pose_conf_experiment\synth_room2\arm1_mv_edge\surface_metrics.json",
                   summ=r"experiments\pose_conf_experiment\synth_room2\arm1_mv_edge\summary.json",
                   revisit=r"experiments\pose_conf_experiment\synth_room2\arm1_mv_edge\revisit_consistency.json"), kind="synth"),
}


def metrics(src, kind):
    c, d, s, sm = j(src["cloud"]), j(src["depth"]), j(src["surf"]), j(src["summ"])
    out = {}
    if kind == "tum":
        cc = c["cloud"]["sim3"]; ev = c["evo"]
        out.update(acc=cc["accuracy_median_cm"], p90=cc["accuracy_p90_cm"], w5=cc["frac_within_5cm"] * 100,
                   comp_med=cc["completeness_median_cm"], comp5=cc["completeness_frac_within_5cm"] * 100,
                   ate=ev["ape_sim3_rmse_m"] * 1000, rpe=ev["rpe_sim3_rmse_m"] * 1000, ate_se3=ev["ape_se3_rmse_m"] * 1000)
    else:
        cc = c["cloud"]["sim3"]
        out.update(acc=cc["accuracy_median_m"] * 100, p90=cc["accuracy_p90_m"] * 100, w5=cc["frac_within_5cm"] * 100,
                   comp_med=cc["completeness_median_m"] * 100, comp5=cc["completeness_frac_within_5cm"] * 100,
                   ate=c["trajectory"]["ate_sim3_rmse_m"] * 1000,
                   # same COLMAP poses for baseline and final -> RPE from the trajectory-only evo run (synth_rpe.json)
                   rpe=j(r"experiments\FINAL_PIPELINE\synth_rpe.json")["rpe_sim3_rmse_m"] * 1000,
                   ate_se3=c["trajectory"]["ate_se3_rmse_m"] * 1000)
    out.update(absrel=d["absrel_mean"] * 100 if d else None, thick=s["local_thickness_fused"]["median_cm"] if s else None,
               points=sm["points_final"], runtime=sm["timing_s"]["total_s"],
               depth_inference_in_runtime=sm["timing_s"]["depth_and_correspondences_s"] > 30)
    rv = j(src["revisit"]) if src.get("revisit") else None
    out["revisit"] = rv["revisit_start_vs_end"]["median"] * 100 if rv else None
    if src.get("summ_cloud"):  # cloud metrics come from an equivalent earlier run: verify it is the same reconstruction
        out["cloud_run_points"] = j(src["summ_cloud"])["points_final"]
        out["same_reconstruction"] = out["cloud_run_points"] == out["points"]
    return out


res = {ds: dict(baseline=metrics(v["base"], v["kind"]), final=metrics(v["final"], v["kind"]), kind=v["kind"], sources=v)
       for ds, v in SRC.items()}
json.dump(res, open(os.path.join(OUT, "baseline_vs_final.json"), "w"), indent=2)

ROWS = [("acc", "Median 3D error (cm, Sim3)", "lower", 2), ("p90", "P90 3D error (cm, Sim3)", "lower", 1),
        ("w5", "Points within 5 cm (%)", "higher", 1), ("comp_med", "Completeness median (cm)", "lower", 2),
        ("comp5", "Completeness within 5 cm (%)", "higher", 1), ("absrel", "Depth AbsRel (%)", "lower", 2),
        ("thick", "Local surface thickness (cm)", "lower", 2), ("revisit", "Start-vs-end revisit (fused units ×100)", "lower", 2),
        ("points", "Final points", None, 0), ("runtime", "Runtime (s)", None, 1),
        ("ate", "ATE Sim3 RMSE (mm)", "lower", 1), ("rpe", "RPE Sim3 (mm)", "lower", 1)]
md = ["# Baseline vs final configuration", "",
      "Baseline: `--align scale --fusion voxel`. Final (frozen): `--align spatial --fusion weighted --conf_signals mv,edge`.",
      "Same frames, COLMAP poses, cached depth, voxel 0.02, depth truncation 8.0 and pixel stride 2 within each dataset. "
      "All values come from existing evaluation files (see baseline_vs_final.json for exact paths); nothing was rerun.", "",
      "**Comparability:** compare baseline vs final *within* a row of one dataset. Across datasets, 3D metrics are not directly "
      "comparable (different scenes, ground-truth sources and sampling); synth_room2 3D metrics come from `eval_synth.py` "
      "(renderer ground truth), TUM rows from `eval_tum.py` (Kinect depth). All 3D metrics are after Sim(3) alignment, "
      "not absolute metric evaluation. Trajectory metrics are identical for baseline and final (same COLMAP poses).", ""]
for ds, r in res.items():
    b, f = r["baseline"], r["final"]
    md += [f"## {ds} ({'development' if ds == 'fr1_room' else 'synthetic' if ds == 'synth_room2' else 'held-out'})", "",
           "| Metric | Baseline | Final | Better |", "|---|---|---|---|"]
    for k, lab, better, p in ROWS:
        bv, fv = b.get(k), f.get(k)
        fmt = (lambda v: "not available" if v is None else (f"{v:,.0f}" if p == 0 else f"{v:.{p}f}"))
        if bv is None or fv is None or better is None:
            w = "—"
        elif abs(bv - fv) < 10 ** (-p):
            w = "tie"
        else:
            w = "final" if (fv < bv) == (better == "lower") else "baseline"
        if k in ("ate", "rpe"):
            w = "same poses"
        md.append(f"| {lab} | {fmt(bv)} | {fmt(fv)} | {w} |")
    notes = []
    if b.get("same_reconstruction") is not None:
        notes.append(f"Baseline cloud metrics are from the earlier identical-configuration run ({b['cloud_run_points']:,} points, "
                     f"{'identical' if b['same_reconstruction'] else 'DIFFERENT'} to the {b['points']:,}-point run used for depth/thickness/runtime).")
    if b["depth_inference_in_runtime"] != f["depth_inference_in_runtime"]:
        notes.append("Runtimes are not comparable: one run includes depth-network inference, the other used cached depth.")
    if ds == "fr2_xyz":
        notes.append("Final spatial calibration (`calib_fr2_xyz_perframe.json`) was derived from fr2_xyz's own COLMAP sparse points (no GT).")
    if b.get("revisit") is None:
        notes.append("Baseline revisit consistency was never computed for this dataset: not available.")
    md += [""] + [f"- {n}" for n in notes] + [""]
if res["synth_room2"]["final"]["rpe"] is not None:
    md.insert(md.index("## Summary of measured changes") if "## Summary of measured changes" in md else len(md),
              "synth_room2 RPE (both columns): trajectory-only evo run on the final run's trajectory_fused.txt "
              "(FINAL_PIPELINE\\synth_rpe.json); valid for both because the COLMAP poses are identical.\n")


def verdicts(key):
    return {ds: ("final" if (r["final"][key] < r["baseline"][key]) == (key not in ("w5", "comp5")) else "baseline")
            for ds, r in res.items() if r["final"][key] is not None and r["baseline"][key] is not None
            and abs(r["final"][key] - r["baseline"][key]) >= 0.01}


md += ["## Summary of measured changes", ""]
for key, lab in (("acc", "median 3D error"), ("p90", "P90 3D error"), ("w5", "points within 5 cm"), ("absrel", "depth AbsRel"),
                 ("thick", "local surface thickness"), ("comp_med", "completeness median"), ("comp5", "completeness within 5 cm")):
    v = verdicts(key)
    fin = [d for d, w in v.items() if w == "final"]; bas = [d for d, w in v.items() if w == "baseline"]
    ties = [d for d in res if d not in v]
    md.append(f"- **{lab}:** final better on {', '.join(fin) or 'none'}; baseline better on {', '.join(bas) or 'none'}"
              + (f"; tie on {', '.join(ties)}" if ties else "") + ".")
md += ["", "The final configuration improved several 3D accuracy / surface metrics on every dataset, while completeness "
       "remained better for the baseline on fr1_room and synth_room2 (and the completeness median on fr1_desk). "
       "No significance testing was done; differences of a few tenths of a cm or percentage point (e.g. fr2_xyz thickness "
       "4.00 → 3.96 cm, fr1_room revisit 7.66 → 7.52) should be treated as ties."]
open(os.path.join(OUT, "baseline_vs_final.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
