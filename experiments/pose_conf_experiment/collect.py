"""Consolidated table for the pose-confidence experiment + mechanical application of the pre-registered rule.
Primary accuracy/completeness: eval_tum.py (TUM, Sim3) / eval_synth.py (synthetic, Sim3)."""
import json
import os

E = os.path.dirname(os.path.abspath(__file__))
ARMS = ["arm1_mv_edge", "arm2_mv_edge_pose", "arm3_mv_edge_frame", "arm4_mv_edge_pose_frame"]
DS = ["fr1_room", "fr1_desk", "fr2_xyz", "synth_room2"]
HELD_OUT = ["fr1_desk", "fr2_xyz", "synth_room2"]


def load(ds, arm):
    d = os.path.join(E, ds, arm)
    if not os.path.exists(os.path.join(d, "summary.json")):
        return None
    j = lambda f: json.load(open(os.path.join(d, f))) if os.path.exists(os.path.join(d, f)) else None
    s, sm, de, rv, fp = j("summary.json"), j("surface_metrics.json"), j("depth_eval.json"), j("revisit_consistency.json"), j("fallback_footprint.json")
    if ds == "synth_room2":
        c = j("eval_synth.json")["cloud"]["sim3"]
        acc = (c["accuracy_median_m"] * 100, c["accuracy_p90_m"] * 100, c["frac_within_5cm"] * 100)
        comp = (c["completeness_median_m"] * 100, c["completeness_frac_within_5cm"] * 100)
    else:
        c = j("eval_tum.json")["cloud"]["sim3"]
        acc = (c["accuracy_median_cm"], c["accuracy_p90_cm"], c["frac_within_5cm"] * 100)
        comp = (c["completeness_median_cm"], c["completeness_frac_within_5cm"] * 100)
    return dict(acc_med=acc[0], acc_p90=acc[1], acc_5=acc[2], comp_med=comp[0], comp_5=comp[1],
                thick=sm["local_thickness_fused"]["median_cm"], points=s["points_final"], runtime=s["timing_s"]["total_s"],
                absrel=de["absrel_mean"] * 100, revisit_cm=rv["revisit_start_vs_end"]["median"] * 100 if rv and rv["revisit_start_vs_end"]["median"] else None,
                fallback=fp.get("fallback_frames") if fp else None, fb_pct=fp.get("pct_of_final") if fp else None,
                fb_acc_excl=fp.get("acc_median_excluding_cm") if fp else None)


R = {ds: {a: load(ds, a) for a in ARMS} for ds in DS}
json.dump(R, open(os.path.join(E, "results.json"), "w"), indent=2)
hdr = f"{'dataset':11s} {'arm':24s} {'acc med':>7s} {'p90':>6s} {'<5cm':>5s} {'comp med':>8s} {'comp<5':>6s} {'thick':>5s} {'points':>9s} {'time s':>6s} {'AbsRel%':>7s} {'revisit':>7s} {'fallback':>13s}"
print(hdr)
for ds in DS:
    for a in ARMS:
        r = R[ds][a]
        if not r:
            print(f"{ds:11s} {a:24s} (missing)"); continue
        fb = f"{r['fallback']}fr/{r['fb_pct']:.1f}%" if r['fallback'] else "0"
        rv = f"{r['revisit_cm']:.2f}" if r["revisit_cm"] is not None else "-"
        print(f"{ds:11s} {a:24s} {r['acc_med']:7.2f} {r['acc_p90']:6.1f} {r['acc_5']:5.1f} {r['comp_med']:8.2f} {r['comp_5']:6.1f} "
              f"{r['thick']:5.2f} {r['points']:9,d} {r['runtime']:6.1f} {r['absrel']:7.3f} {rv:>7s} {fb:>13s}")
print()
# decision rule (pre-registered): vs arm1 on held-out datasets
for arm in ARMS[1:]:
    better, worse, comp_ok = 0, 0, True
    detail = []
    for ds in HELD_OUT:
        b, x = R[ds]["arm1_mv_edge"], R[ds][arm]
        if not b or not x:
            detail.append(f"{ds}: missing"); comp_ok = False; continue
        d_acc = b["acc_med"] - x["acc_med"]          # positive = improvement
        d_comp = x["comp_5"] - b["comp_5"]
        better += d_acc >= 0.3; worse += d_acc < -0.3; comp_ok &= d_comp >= -2.0
        detail.append(f"{ds}: acc {d_acc:+.2f} cm, comp<5 {d_comp:+.1f} pp")
    ok = better >= 2 and worse == 0 and comp_ok
    print(f"{arm}: improves>=0.3cm on {better}/3, worse>0.3cm on {worse}, completeness within -2pp: {comp_ok} -> {'ADOPT' if ok else 'REJECT'}  | " + "; ".join(detail))
