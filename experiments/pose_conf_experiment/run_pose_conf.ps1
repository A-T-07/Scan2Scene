# Controlled pose-confidence experiment. Only --conf_signals differs between arms; every other parameter is fixed
# and identical across datasets. Outputs: experiments\pose_conf_experiment\<dataset>\<arm>\
param([string[]]$Datasets = @("fr1_room", "fr1_desk", "fr2_xyz", "synth_room2"))
$ErrorActionPreference = "Continue"
Set-Location Q:\OPEN_CV_SEM5
$py = ".\3d-reconstruction\3d-reconstruction\venv\Scripts\python.exe"
$OUT = "experiments\pose_conf_experiment"
$D = @{
  fr1_room = @{ frames = "experiments\tum_freiburg1_room\frames"; colmap = "experiments\tum_freiburg1_room\colmap_ws\sparse\0"
                cache = "experiments\tum_freiburg1_room\depth_pred"; gt = "C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_room\rgbd_dataset_freiburg1_room"; kind = "tum" }
  fr1_desk = @{ frames = "experiments\tum_freiburg1_desk\frames"; colmap = "experiments\tum_freiburg1_desk\colmap_ws\sparse\0"
                cache = "experiments\tum_freiburg1_desk\depth_pred"; gt = "C:\Users\tyagi\Downloads\rgbd_dataset_freiburg1_desk\rgbd_dataset_freiburg1_desk"; kind = "tum" }
  fr2_xyz = @{ frames = "frames_subset"; colmap = "colmap_ws3\sparse\0"
               cache = "experiments\tum_fr2_xyz\corrected_scale\depth_pred"; gt = "Q:\OPEN_CV_SEM5\rgbd_dataset_freiburg2_xyz\rgbd_dataset_freiburg2_xyz"; kind = "tum" }
  synth_room2 = @{ frames = "synth_room2\run\frames"; colmap = "synth_room2\run\colmap_ws\sparse\0"
                   cache = "synth_room2\recon_scale\depth_pred"; gt = "synth_room2"; kind = "synth" }
}
$arms = [ordered]@{ "arm1_mv_edge" = "mv,edge"; "arm2_mv_edge_pose" = "mv,edge,pose"; "arm3_mv_edge_frame" = "mv,edge,frame"; "arm4_mv_edge_pose_frame" = "mv,edge,pose,frame" }
foreach ($ds in $Datasets) {
  $c = $D[$ds]
  foreach ($arm in $arms.Keys) {
    $rd = "$OUT\$ds\$arm"
    New-Item -ItemType Directory -Force $rd | Out-Null
    $a = @("reconstruct.py", "--frames", $c.frames, "--colmap_dir", $c.colmap, "--out", $rd, "--depth_cache", $c.cache,
           "--stride", "2", "--voxel_size", "0.02", "--depth_trunc", "8.0", "--no_viewer", "--save_aligned_depth",
           "--align", "spatial", "--spatial_params", "experiments\spatial_depth\calib_fr2_xyz_perframe.json",
           "--fusion", "weighted", "--conf_signals", $arms[$arm], "--conf_pose_ref", "0.5", "--conf_pose_floor", "0.2")
    & $py @a *> "$rd\reconstruct.log"
    if ($LASTEXITCODE -ne 0) { "!! $ds/$arm reconstruct FAILED"; continue }
    if ($c.kind -eq "tum") { & $py eval_tum.py --seq_dir $c.gt --recon_dir $rd *> "$rd\eval_tum.log" }
    else { & $py eval_synth.py --synth_dir $c.gt --recon_dir $rd *> "$rd\eval_synth.log" }
    & $py surface_metrics.py --kind $c.kind --gt_dir $c.gt --recon_dir $rd *> "$rd\surface_metrics.log"
    & $py eval_aligned_depth.py --kind $c.kind --gt_dir $c.gt --recon_dir $rd *> "$rd\eval_aligned_depth.log"
    & $py revisit_consistency.py --recon_dir $rd --colmap_dir $c.colmap --k 15 *> "$rd\revisit_consistency.log"
    & $py fallback_footprint.py --kind $c.kind --gt_dir $c.gt --recon_dir $rd --colmap_dir $c.colmap *> "$rd\fallback_footprint.log"
    "== $ds / $arm done"
  }
}
"ALL DONE"
