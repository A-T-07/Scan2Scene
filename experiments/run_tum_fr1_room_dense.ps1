# TUM freiburg1_room dense reconstruction -- RGB frames + COLMAP model only (no TUM depth, no groundtruth).
# Settings identical to experiments\run_phone_real_v2.ps1 (phone_real_v2 baseline / improved).
$ErrorActionPreference = "Continue"
Set-Location Q:\OPEN_CV_SEM5
$py = ".\3d-reconstruction\3d-reconstruction\venv\Scripts\python.exe"
$O = "experiments\tum_freiburg1_room"
$common = @("--frames", "$O\frames", "--colmap_dir", "$O\colmap_ws\sparse\0", "--depth_cache", "$O\depth_pred",
            "--stride", "2", "--voxel_size", "0.02", "--depth_trunc", "8.0", "--no_viewer", "--save_aligned_depth")
$runs = [ordered]@{
  baseline = @("--align", "scale", "--fusion", "voxel")
  improved = @("--align", "spatial", "--spatial_params", "experiments\spatial_depth\calib_fr2_xyz_perframe.json",
               "--fusion", "weighted", "--conf_signals", "mv,edge", "--save_confidence")
}
foreach ($name in $runs.Keys) {
  $a = @("reconstruct.py", "--out", "$O\$name") + $common + $runs[$name]
  "> $py $($a -join ' ')"
  $t = Get-Date
  & $py @a *> "$O\$name.log"
  "== $name exit $LASTEXITCODE, wall $([int]((Get-Date) - $t).TotalSeconds) s"
}
