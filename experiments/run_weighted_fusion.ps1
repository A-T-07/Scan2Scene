# Confidence-weighted fusion experiments. Thresholds are the a-priori defaults in reconstruct.py; nothing is tuned here.
# Baseline / spatial runs are reused from experiments\spatial_depth\runs (identical frames, poses, voxel, truncation).
$ErrorActionPreference = "Continue"   # python warnings on stderr must not abort; exit codes are checked explicitly
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
  $runs = [ordered]@{
    "weighted"     = @("--fusion", "weighted", "--save_confidence", "--save_aligned_depth")
    "abl_none"     = @("--fusion", "weighted", "--conf_signals", "none")
    "abl_mv"       = @("--fusion", "weighted", "--conf_signals", "mv")
    "abl_edge"     = @("--fusion", "weighted", "--conf_signals", "edge")
    "abl_mv_edge"  = @("--fusion", "weighted", "--conf_signals", "mv,edge")
    "tsdf_spatial" = @("--fusion", "tsdf", "--tsdf_voxel", "0.02", "--tsdf_trunc", "0.08")
  }
  foreach ($name in $runs.Keys) {
    $rundir = "$RUNROOT\$ds\$name"
    if (Test-Path "$rundir\surface_metrics.json") { "== $ds / $name (already done)"; continue }
    New-Item -ItemType Directory -Force (Split-Path $rundir) | Out-Null
    $a = @("reconstruct.py", "--frames", $c.frames, "--colmap_dir", $c.colmap, "--out", $rundir, "--stride", "2",
           "--depth_cache", $c.cache, "--voxel_size", "0.02", "--depth_trunc", "8.0", "--no_viewer") + $spatial + $runs[$name]
    & $py @a *> "$rundir.log"
    if ($LASTEXITCODE -ne 0) { throw "reconstruct failed: $rundir" }
    "== $ds / $name"
    Get-Content "$rundir.log" | Select-String "^Fusion|^Points|^Runtime"
    & $py surface_metrics.py --kind $c.kind --gt_dir $c.gt --recon_dir $rundir
  }
  & $py eval_aligned_depth.py --kind $c.kind --gt_dir $c.gt --recon_dir "$RUNROOT\$ds\weighted"
}
"ALL DONE"
