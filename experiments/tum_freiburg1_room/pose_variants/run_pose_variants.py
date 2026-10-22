r"""
run_pose_variants.py -- controlled POSE-ONLY COLMAP variants for TUM fr1_room (RGB frames only, no GT, no TUM depth).

All variants: the same 273 frames, the same SIFT features (a copy of the control database), and the control's
project.ini (COLMAP 4.1.1, as written by automatic_reconstructor --data_type video --single_camera 1). Only the
MATCHING options differ. Each variant has an isolated workspace/database. The mapper is not seeded
(Mapper.random_seed=-1), so it is run 3x per variant to measure run-to-run spread.

  control              : original model (untouched, evaluated from a copy) + 2 mapper repeats on a copy of its DB
  variant_B_sequential : sequential_matcher, overlap 10 -> 30, loop_detection_period 10 -> 5,
                         loop_detection_num_images 50 -> 100 (guided matching already on in the control;
                         loop_detection_min_index_distance does not exist in COLMAP 4.1.1)
  variant_C_global     : exhaustive_matcher (all 37,128 pairs), same verification / guided matching
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = r"Q:\OPEN_CV_SEM5"
EXP = os.path.join(ROOT, r"experiments\tum_freiburg1_room")
PV = os.path.join(EXP, "pose_variants")
COLMAP = os.path.join(ROOT, r"colmap-x64-windows-cuda\bin\colmap.exe")
PY = sys.executable
FRAMES = os.path.join(EXP, "frames")
CTRL_WS = os.path.join(EXP, "colmap_ws")
BASE_INI = open(os.path.join(CTRL_WS, "sparse", "project.ini")).read()
REPEATS = 3


def ini(path, db, overrides, top=None):
    """Copy of the control project.ini with new paths / overrides. With --project_path, COLMAP 4.1.1 reads ALL
    options from the file (command-line options are ignored), so per-command keys like output_path go in `top`."""
    txt = BASE_INI
    if top:
        txt = "".join(f"{k}={v}\n" for k, v in top.items()) + txt
    txt = re.sub(r"(?m)^database_path=.*$", "database_path=" + db.replace("\\", "\\\\"), txt)
    txt = re.sub(r"(?m)^image_path=.*$", "image_path=" + FRAMES.replace("\\", "\\\\"), txt)
    section = None
    lines = []
    for line in txt.splitlines():
        m = re.match(r"^\[(.+)\]$", line)
        if m:
            section = m.group(1)
        elif section and "=" in line:
            key = line.split("=", 1)[0]
            if f"{section}.{key}" in overrides:
                line = f"{key}={overrides[section + '.' + key]}"
        lines.append(line)
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def run(cmd, log):
    t = time.time()
    with open(log, "a") as fh:
        fh.write("> " + " ".join(cmd) + "\n"); fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(cmd)} -- see {log}")
    return time.time() - t


def fresh_db(dst):
    shutil.copy2(os.path.join(CTRL_WS, "database.db"), dst)
    con = sqlite3.connect(dst)
    con.execute("DELETE FROM matches"); con.execute("DELETE FROM two_view_geometries"); con.commit()
    con.execute("VACUUM"); con.close()


def models(sparse_dir, log):
    out = []
    for sub in sorted(os.listdir(sparse_dir)):
        d = os.path.join(sparse_dir, sub)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "images.bin")):
            run([COLMAP, "model_converter", "--input_path", d, "--output_path", d, "--output_type", "TXT"], log)
            n = sum(1 for l in open(os.path.join(d, "images.txt")) if l.strip() and not l.startswith("#")) // 2
            out.append((n, d))
    return sorted(out, reverse=True)


def evaluate(model_dir, db, out, n_models, runtime, log):
    run([PY, os.path.join(PV, "eval_pose_variant.py"), "--model", model_dir, "--db", db, "--out", out,
         "--n_models", str(n_models), "--runtime_s", f"{runtime:.1f}"], log)


def variant(name, matcher, overrides):
    vd = os.path.join(PV, name); ws = os.path.join(vd, "colmap_ws"); os.makedirs(ws, exist_ok=True)
    log = os.path.join(vd, "colmap_log.txt")
    db = os.path.join(ws, "database.db")
    fresh_db(db)
    p = ini(os.path.join(vd, "project.ini"), db, overrides)
    t_match = run([COLMAP, matcher, "--project_path", p], log)
    summary = dict(variant=name, matcher=matcher, overrides=overrides, match_s=t_match, runs=[])
    for r in range(REPEATS):
        sp = os.path.join(ws, "sparse") if r == 0 else os.path.join(vd, "repeats", f"rep{r}")
        os.makedirs(sp, exist_ok=True)
        pm = ini(os.path.join(vd, f"project_mapper_rep{r}.ini"), db, overrides, top={"output_path": sp})
        t_map = run([COLMAP, "mapper", "--project_path", pm], log)
        ms = models(sp, log)
        evaluate(ms[0][1], db, os.path.join(vd, "eval" if r == 0 else f"eval_rep{r}"), len(ms), t_match + t_map, log)
        summary["runs"].append(dict(rep=r, map_s=t_map, models=[m[0] for m in ms]))
    if True:
        subprocess.run([PY, os.path.join(ROOT, "diagnose_poses.py"), "--workdir", vd], stdout=open(os.path.join(vd, "pose_diagnostics.txt"), "w"),
                       stderr=subprocess.STDOUT)
    json.dump(summary, open(os.path.join(vd, "run_summary.json"), "w"), indent=2)
    print(f"{name}: matching {t_match:.0f} s, mapper {[round(r['map_s']) for r in summary['runs']]} s, models {[r['models'] for r in summary['runs']]}")


def control():
    vd = os.path.join(PV, "control"); ws = os.path.join(vd, "colmap_ws"); os.makedirs(ws, exist_ok=True)
    log = os.path.join(vd, "colmap_log.txt")
    shutil.copy2(os.path.join(CTRL_WS, "database.db"), os.path.join(ws, "database.db"))
    if not os.path.exists(os.path.join(ws, "sparse")):
        shutil.copytree(os.path.join(CTRL_WS, "sparse"), os.path.join(ws, "sparse"))
    db = os.path.join(ws, "database.db")
    # original runtime from the original automatic_reconstructor log (extraction + matching + mapping)
    txt = open(os.path.join(EXP, "colmap_log.txt"), errors="ignore").read()
    orig_min = sum(float(x) for x in re.findall(r"Elapsed time: ([0-9.]+) \[minutes\]", txt))
    ms = [(n, d) for n, d in models(os.path.join(ws, "sparse"), log)]
    evaluate(ms[0][1], db, os.path.join(vd, "eval"), len(ms), orig_min * 60, log)
    p = ini(os.path.join(vd, "project.ini"), db, {})
    summary = dict(variant="control", note="original model evaluated from a copy; repeats = mapper re-runs on the same DB",
                   original_runtime_s=orig_min * 60, runs=[dict(rep=0, models=[m[0] for m in ms])])
    for r in range(1, REPEATS):
        sp = os.path.join(vd, "repeats", f"rep{r}"); os.makedirs(sp, exist_ok=True)
        pm = ini(os.path.join(vd, f"project_mapper_rep{r}.ini"), db, {}, top={"output_path": sp})
        t_map = run([COLMAP, "mapper", "--project_path", pm], log)
        mr = models(sp, log)
        evaluate(mr[0][1], db, os.path.join(vd, f"eval_rep{r}"), len(mr), t_map, log)
        summary["runs"].append(dict(rep=r, map_s=t_map, models=[m[0] for m in mr]))
    subprocess.run([PY, os.path.join(ROOT, "diagnose_poses.py"), "--workdir", vd], stdout=open(os.path.join(vd, "pose_diagnostics.txt"), "w"),
                   stderr=subprocess.STDOUT)
    json.dump(summary, open(os.path.join(vd, "run_summary.json"), "w"), indent=2)
    print(f"control: original {orig_min * 60:.0f} s, repeats {[round(r.get('map_s', 0)) for r in summary['runs']]} s, models {[r['models'] for r in summary['runs']]}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["control", "B", "C"]
    if "control" in which:
        control()
    if "B" in which:
        variant("variant_B_sequential", "sequential_matcher",
                {"SequentialMatching.overlap": "30", "SequentialMatching.loop_detection_period": "5",
                 "SequentialMatching.loop_detection_num_images": "100"})
    if "C" in which:
        variant("variant_C_global", "exhaustive_matcher", {})
