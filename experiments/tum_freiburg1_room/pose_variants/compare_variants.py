"""Collect pose_eval.json of every variant / mapper repeat into one comparison (comparison.json + printed table)."""
import json
import os

import numpy as np

PV = os.path.dirname(os.path.abspath(__file__))
CONTROL_ORIGINAL_RUNTIME_S = 174.0  # from the original automatic_reconstructor log (0.201+0.561+0.002+2.136 min)
rows = {}
for v in ("control", "variant_B_sequential", "variant_C_global"):
    runs = []
    for e in sorted(d for d in os.listdir(os.path.join(PV, v)) if d.startswith("eval")):
        runs.append(json.load(open(os.path.join(PV, v, e, "pose_eval.json"))))
    summ = json.load(open(os.path.join(PV, v, "run_summary.json")))
    rows[v] = dict(runs=runs, summary=summ)
out = {}
for v, d in rows.items():
    R = d["runs"]
    f = lambda key: [r[key] for r in R]
    g = lambda fn: [fn(r) for r in R]
    rt = CONTROL_ORIGINAL_RUNTIME_S if v == "control" else d["summary"]["match_s"] + d["summary"]["runs"][0]["map_s"]
    out[v] = dict(
        registered=g(lambda r: f"{r['registered']}/{r['sampled']}"), models=f("n_models"), reproj_px=f("mean_reproj_px"),
        ate_rmse_mm=[x * 1000 for x in f("ate_sim3_rmse_m")], ate_median_mm=[x * 1000 for x in f("ate_sim3_median_m")],
        ate_p90_mm=[x * 1000 for x in f("ate_sim3_p90_m")], ate_max_mm=[x * 1000 for x in f("ate_sim3_max_m")],
        rpe_sim3_mm=g(lambda r: r["evo"]["rpe_sim3_rmse_m"] * 1000), sim3_scale=f("sim3_scale"),
        window_median_mm=g(lambda r: r["window"]["ate_median_m"] * 1000), window_max_mm=g(lambda r: r["window"]["ate_max_m"] * 1000),
        window_min_cons_inliers=g(lambda r: r["window"]["min_consecutive_inliers"]),
        start_end_disp_err_mm=g(lambda r: r["start_end"]["displacement_err_m"] * 1000),
        start_end_rot_err_deg=g(lambda r: r["start_end"]["relative_rotation_err_deg"]),
        start_end_pairs=g(lambda r: r["start_end"]["verified_start_end_pairs"]),
        verified_pairs=g(lambda r: r["matching"]["verified_pairs"]), total_inliers=g(lambda r: r["matching"]["total_inliers"]),
        nonseq_pairs=g(lambda r: r["matching"]["nonsequential_pairs_gap_gt_30"]),
        min_cons_inliers=g(lambda r: r["matching"]["min_consecutive_inliers"]),
        runtime_first_run_s=rt, mapper_s=[x.get("map_s") for x in d["summary"]["runs"]],
        match_s=d["summary"].get("match_s"))
json.dump(out, open(os.path.join(PV, "comparison.json"), "w"), indent=2)
fmt = lambda xs, p=1: f"{np.mean(xs):.{p}f} [{min(xs):.{p}f}-{max(xs):.{p}f}]"
for v, o in out.items():
    print(f"\n== {v}  (runs: {len(o['models'])})")
    for k, val in o.items():
        if isinstance(val, list) and val and isinstance(val[0], (int, float)) and val[0] is not None:
            print(f"   {k:26s} {fmt(val, 3 if k in ('reproj_px', 'sim3_scale', 'start_end_rot_err_deg') else 1)}   runs {[round(x, 3) for x in val]}")
        else:
            print(f"   {k:26s} {val}")
