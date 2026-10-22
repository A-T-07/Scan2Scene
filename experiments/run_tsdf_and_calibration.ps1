# TSDF fusion + leave-one-sequence-out metric calibration on TUM (no new COLMAP, cached depth only).
# TSDF parameters were fixed on synthetic data BEFORE this script: --tsdf_voxel 0.02 --tsdf_trunc 0.08 --depth_trunc 8.0
$ErrorActionPreference = "Stop"
Set-Location Q:\OPEN_CV_SEM5
$py = ".\3d-reconstruction\3d-reconstruction\venv\Scripts\python.exe"
$FR2 = "Q:\OPEN_CV_SEM5\rgbd_dataset_freiburg2_xyz\rgbd_dataset_freiburg2_xyz"
$FR1 = "C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_desk\rgbd_dataset_freiburg1_desk"
$seq = @{
  fr2 = @{ frames = "frames_subset"; colmap = "colmap_ws3\sparse\0"; cache = "experiments\tum_fr2_xyz\corrected_scale\depth_pred"; exp = "experiments\tum_fr2_xyz"; gt = $FR2 }
  fr1 = @{ frames = "experiments\tum_freiburg1_desk\frames"; colmap = "experiments\tum_freiburg1_desk\colmap_ws\sparse\0"; cache = "experiments\tum_freiburg1_desk\depth_pred"; exp = "experiments\tum_freiburg1_desk"; gt = $FR1 }
}
function Recon($s, $out, $extra) {
  $a = @("reconstruct.py", "--frames", $s.frames, "--colmap_dir", $s.colmap, "--out", $out, "--align", "scale", "--stride", "2",
         "--depth_cache", $s.cache, "--no_viewer") + $extra
  & $py @a *> "$out.log"
  if ($LASTEXITCODE -ne 0) { throw "reconstruct failed: $out" }
  Get-Content "$out.log" | Select-String "^Fusion|^Points|^Runtime|External metric"
}

# 1) TSDF (same frames / poses / alignment as the corrected_scale baseline)
foreach ($k in "fr2", "fr1") {
  $s = $seq[$k]; $out = "$($s.exp)\tsdf"
  "== $k TSDF"; Recon $s $out @("--fusion", "tsdf", "--tsdf_voxel", "0.02", "--tsdf_trunc", "0.08", "--depth_trunc", "8.0", "--tsdf_mesh")
}

# 2) leave-one-sequence-out calibration (calibration uses ONLY the other sequence's ground truth)
New-Item -ItemType Directory -Force experiments\calibration | Out-Null
& $py calibrate_scale.py --seq experiments\tum_fr2_xyz\corrected_scale "$FR2\groundtruth.txt" --out experiments\calibration\calib_from_fr2.json
& $py calibrate_scale.py --seq experiments\tum_freiburg1_desk\corrected_scale "$FR1\groundtruth.txt" --out experiments\calibration\calib_from_fr1.json
"== fr1 calibrated with k from fr2"; Recon $seq.fr1 "experiments\tum_freiburg1_desk\calibrated_from_fr2" @("--metric_scale_file", "experiments\calibration\calib_from_fr2.json")
"== fr2 calibrated with k from fr1"; Recon $seq.fr2 "experiments\tum_fr2_xyz\calibrated_from_fr1" @("--metric_scale_file", "experiments\calibration\calib_from_fr1.json")

# 3) evaluation
foreach ($k in "fr2", "fr1") {
  $s = $seq[$k]
  foreach ($d in "corrected_scale", "tsdf") {
    & $py surface_metrics.py --kind tum --gt_dir $s.gt --recon_dir "$($s.exp)\$d"
    & $py eval_tum.py --seq_dir $s.gt --recon_dir "$($s.exp)\$d" --out_name ("eval_tum" + $(if ($d -eq "tsdf") { "" } else { "_rerun" }) + ".json") | Select-String "^evo|acc median"
  }
}
& $py eval_tum.py --seq_dir $FR1 --recon_dir experiments\tum_freiburg1_desk\calibrated_from_fr2 | Select-String "^evo|\[se3\] acc"
& $py eval_tum.py --seq_dir $FR2 --recon_dir experiments\tum_fr2_xyz\calibrated_from_fr1 | Select-String "^evo|\[se3\] acc"
"ALL DONE"
