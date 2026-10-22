"""Trajectory-only RPE/ATE for synth_room2 (final configuration run), via evo -- the same eval_tum.run_evo call used
for all TUM rows (evo_ape / evo_rpe, -a = SE3, -as = Sim3, delta = 1 frame). Reads existing files only.
  estimate : experiments\pose_conf_experiment\synth_room2\arm1_mv_edge\trajectory_fused.txt
  reference: synth_room2\groundtruth.txt (renderer trajectory, TUM format, timestamp = frame index)"""
import json
import os
import sys

import numpy as np

ROOT = r"Q:\OPEN_CV_SEM5"
sys.path.insert(0, ROOT)
from eval_tum import run_evo  # noqa: E402

est = os.path.join(ROOT, r"experiments\pose_conf_experiment\synth_room2\arm1_mv_edge\trajectory_fused.txt")
gt = os.path.join(ROOT, r"synth_room2\groundtruth.txt")
e, g = np.loadtxt(est), np.loadtxt(gt)
common = len(set(np.round(e[:, 0], 3)) & set(np.round(g[:, 0], 3)))
res = dict(estimate=est, reference=gt, estimate_poses=len(e), reference_poses=len(g), common_timestamps=common,
           method="eval_tum.run_evo: evo_ape/evo_rpe tum <gt> <est> with -a (SE3) and -as (Sim3), default delta=1 frame",
           **run_evo(gt, est))
json.dump(res, open(os.path.join(ROOT, r"experiments\FINAL_PIPELINE\synth_rpe.json"), "w"), indent=2)
print(json.dumps(res, indent=1))
