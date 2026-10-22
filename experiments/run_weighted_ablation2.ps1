# Remaining signal ablations (same a-priori thresholds as all other weighted runs).
$ErrorActionPreference = "Continue"
Set-Location Q:\OPEN_CV_SEM5
$py = ".\3d-reconstruction\3d-reconstruction\venv\Scripts\python.exe"
$SD = "experiments\spatial_depth"
$RUNROOT = "experiments\weighted_fusion\runs"
$FR2 = "Q:\OPEN_CV_SEM5\rgbd_dataset_freiburg2_xyz\rgbd_dataset_freiburg2_xyz"
$FR1 = "C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_desk\rgbd_dataset_freiburg1_desk"
$D = [ordered]@{
  synth_room2 = @{ frames = "synth_room2\run\frames"; colmap = "synth_room2\run\colmap_ws\sparse\0"; cache = "synth_room2\recon_scale\depth_pred"; gt = "synth_room2"; kind = "synth"; calib = "fr2_xyz" }
  fr1_desk = @{ frames = "experiments\tum_freiburg1_desk\frames"; colmap = "experiments\tum_freiburg1_desk\colmap_ws\sparse\0"; cache = "experiments\tum_freiburg1_desk\depth_pred"; gt = $FR1; kind = "tum"; calib = "fr2_xyz" }
  fr2_xyz = @{ frames = "frames_subset"; colmap = "colmap_ws3\sparse\0"; cache = "experiments\tum_fr2_xyz\corrected_scale\depth_pred"; gt = $FR2; kind = "tum"; calib = "fr1_desk" }
}
foreach ($ds in $D.Keys) {
  $c = $D[$ds]
  $spatial = @("--align", "spatial", "--spatial_params", "$SD\calib_$($c.calib)_perframe.json")
  foreach ($sig in "sparse", "frame", "mv,edge,sparse", "mv,edge,frame") {
    $name = "abl_" + ($sig -replace ",", "_")
    $rundir = "$RUNROOT\$ds\$name"
    if (Test-Path "$rundir\surface_metrics.json") { continue }
    $a = @("reconstruct.py", "--frames", $c.frames, "--colmap_dir", $c.colmap, "--out", $rundir, "--stride", "2",
           "--depth_cache", $c.cache, "--voxel_size", "0.02", "--depth_trunc", "8.0", "--no_viewer",
           "--fusion", "weighted", "--conf_signals", $sig) + $spatial
    & $py @a *> "$rundir.log"
    if ($LASTEXITCODE -ne 0) { throw "reconstruct failed: $rundir" }
    & $py surface_metrics.py --kind $c.kind --gt_dir $c.gt --recon_dir $rundir | Select-String "local"
  }
}
"ALL DONE"
