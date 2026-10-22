# Real phone experiment v2 (IMG_8682.MOV). Poses: pose_A_nomask_960 (213/213, single model).
# Settings are the pre-defined project settings; nothing tuned on this clip.
$ErrorActionPreference = "Continue"
Set-Location Q:\OPEN_CV_SEM5
$py = ".\3d-reconstruction\3d-reconstruction\venv\Scripts\python.exe"
$P = "experiments\phone_real_v2"
$src = "$P\pose_A_nomask_960"
$common = @("--frames", "$src\frames", "--colmap_dir", "$src\colmap_ws\sparse\0", "--depth_cache", "$P\depth_pred",
            "--stride", "2", "--voxel_size", "0.02", "--depth_trunc", "8.0", "--no_viewer", "--save_aligned_depth")
$runs = [ordered]@{
  baseline = @("--align", "scale", "--fusion", "voxel")
  improved = @("--align", "spatial", "--spatial_params", "experiments\spatial_depth\calib_fr2_xyz_perframe.json",
               "--fusion", "weighted", "--conf_signals", "mv,edge", "--save_confidence")
}
foreach ($name in $runs.Keys) {
  $out = "$P\$name"
  $a = @("reconstruct.py", "--out", $out) + $common + $runs[$name]
  & $py @a *> "$out.log"
  "== $name exit $LASTEXITCODE"
  Get-Content "$out.log" | Select-String "COLMAP:|Global scale S|Spatial corr|^Fusion|^Points|^Runtime|Median sparse"
}
