# Independent evaluation of the fr1_room RGB-only reconstructions (TUM depth + groundtruth used ONLY here).
# Tools that can only write into the run folder have their JSON moved to evaluation\ right after, so
# baseline\ and improved\ end up unchanged.
$ErrorActionPreference = "Continue"
Set-Location Q:\OPEN_CV_SEM5
$py = ".\3d-reconstruction\3d-reconstruction\venv\Scripts\python.exe"
$SEQ = "C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_room\rgbd_dataset_freiburg1_room"
$O = "experiments\tum_freiburg1_room"
$E = "Q:\OPEN_CV_SEM5\$O\evaluation"
New-Item -ItemType Directory -Force $E | Out-Null
foreach ($r in "baseline", "improved") {
  $rd = "$O\$r"; $ed = "$E\$r"
  New-Item -ItemType Directory -Force $ed | Out-Null
  "==== $r : eval_tum.py"
  & $py eval_tum.py --seq_dir $SEQ --recon_dir $rd --out_name "$ed\eval_tum.json" *> "$ed\eval_tum.log"; "exit $LASTEXITCODE"
  "==== $r : surface_metrics.py"
  & $py surface_metrics.py --kind tum --gt_dir $SEQ --recon_dir $rd *> "$ed\surface_metrics.log"; "exit $LASTEXITCODE"
  if (Test-Path "$rd\surface_metrics.json") { Move-Item "$rd\surface_metrics.json" "$ed\surface_metrics.json" -Force }
  "==== $r : eval_aligned_depth.py"
  & $py eval_aligned_depth.py --kind tum --gt_dir $SEQ --recon_dir $rd *> "$ed\eval_aligned_depth.log"; "exit $LASTEXITCODE"
  if (Test-Path "$rd\depth_eval.json") { Move-Item "$rd\depth_eval.json" "$ed\depth_eval.json" -Force }
  "==== $r : revisit_consistency.py"
  & $py revisit_consistency.py --recon_dir $rd --colmap_dir "$O\colmap_ws\sparse\0" --k 15 *> "$ed\revisit_consistency.log"; "exit $LASTEXITCODE"
  if (Test-Path "$rd\revisit_consistency.json") { Move-Item "$rd\revisit_consistency.json" "$ed\revisit_consistency.json" -Force }
}
"==== improved : eval_confidence.py"
& $py eval_confidence.py --kind tum --gt_dir $SEQ --recon_dir "$O\improved" *> "$E\improved\eval_confidence.log"; "exit $LASTEXITCODE"
if (Test-Path "$O\improved\confidence_eval.json") { Move-Item "$O\improved\confidence_eval.json" "$E\improved\confidence_eval.json" -Force }
"ALL DONE"
