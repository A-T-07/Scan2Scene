# Spatial depth correction: baseline vs spatial (per-frame, hyper-params from the OTHER sequence) vs transfer.
# Everything uses cached depth, the existing COLMAP models, voxel fusion, identical masks / voxel / truncation.
# Outputs: experiments\spatial_depth\runs\<dataset>\<method>\
$ErrorActionPreference = "Stop"
Set-Location Q:\OPEN_CV_SEM5
$py = ".\3d-reconstruction\3d-reconstruction\venv\Scripts\python.exe"
$SD = "experiments\spatial_depth"
$FR2 = "Q:\OPEN_CV_SEM5\rgbd_dataset_freiburg2_xyz\rgbd_dataset_freiburg2_xyz"
$FR1 = "C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_desk\rgbd_dataset_freiburg1_desk"
$D = @{
  fr2_xyz = @{ frames = "frames_subset"; colmap = "colmap_ws3\sparse\0"; cache = "experiments\tum_fr2_xyz\corrected_scale\depth_pred"; gt = $FR2; kind = "tum"; other = "fr1_desk" }
  fr1_desk = @{ frames = "experiments\tum_freiburg1_desk\frames"; colmap = "experiments\tum_freiburg1_desk\colmap_ws\sparse\0"; cache = "experiments\tum_freiburg1_desk\depth_pred"; gt = $FR1; kind = "tum"; other = "fr2_xyz" }
  synth_room2 = @{ frames = "synth_room2\run\frames"; colmap = "synth_room2\run\colmap_ws\sparse\0"; cache = "synth_room2\recon_scale\depth_pred"; gt = "synth_room2"; kind = "synth"; other = "fr2_xyz" }
}
foreach ($ds in "synth_room2", "fr1_desk", "fr2_xyz") {
  $c = $D[$ds]; $o = $c.other
  $runs = [ordered]@{
    "baseline_scale"                 = @("--align", "scale")
    "spatial_perframe_from_$o"       = @("--align", "spatial", "--spatial_params", "$SD\calib_${o}_perframe.json")
    "spatial_transfer_from_$o"       = @("--align", "spatial", "--spatial_params", "$SD\calib_${o}_transfer.json")
  }
  if ($ds -eq "synth_room2") { $runs["spatial_perframe_from_fr1_desk"] = @("--align", "spatial", "--spatial_params", "$SD\calib_fr1_desk_perframe.json") }
  foreach ($name in $runs.Keys) {
    $out = "$SD\runs\$ds\$name"
    New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
    $a = @("reconstruct.py", "--frames", $c.frames, "--colmap_dir", $c.colmap, "--out", $out, "--stride", "2",
           "--depth_cache", $c.cache, "--fusion", "voxel", "--voxel_size", "0.02", "--depth_trunc", "8.0",
           "--save_aligned_depth", "--no_viewer") + $runs[$name]
    & $py @a *> "$out.log"
    if ($LASTEXITCODE -ne 0) { throw "reconstruct failed: $out" }
    "== $ds / $name"
    Get-Content "$out.log" | Select-String "^Spatial|^Points|^Runtime|Median sparse"
    & $py surface_metrics.py --kind $c.kind --gt_dir $c.gt --recon_dir $out
    & $py eval_aligned_depth.py --kind $c.kind --gt_dir $c.gt --recon_dir $out
  }
}
"ALL DONE"
