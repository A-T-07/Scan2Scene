# Phone loop-closure experiment (v3). Usage:
#   .\experiments\run_phone_loop.ps1 -Video "C:\path\clip.MOV" [-Out experiments\phone_real_v3]
# Identical settings to phone v2 (6 fps, 960 px, single camera, no person mask unless -PersonMask).
# Stops after the pose test if registration < 95 %, >1 model, or a step > 5x median.
param([Parameter(Mandatory = $true)][string]$Video, [string]$Out = "experiments\phone_real_v3", [switch]$PersonMask)
$ErrorActionPreference = "Continue"
Set-Location Q:\OPEN_CV_SEM5
$py = ".\3d-reconstruction\3d-reconstruction\venv\Scripts\python.exe"
$pose = "$Out\pose"
& $py inspect_video.py --video $Video --out "$Out\inspect" --step 15 | Select-String "decoded|fps|duration|blurry|rotation_frac"
$pv = @("process_video.py", "--video", $Video, "--colmap_exe", "Q:\OPEN_CV_SEM5\colmap-x64-windows-cuda\bin\colmap.exe",
        "--workdir", (Resolve-Path -LiteralPath (New-Item -ItemType Directory -Force $pose)).Path, "--fps", "6", "--max_width", "960", "--skip_reconstruct")
if ($PersonMask) { $pv += "--person_mask" }
& $py @pv | Select-String "Extracted|Person masks|model \d|Registration|Trajectory"
& $py diagnose_poses.py --workdir $pose > "$pose\pose_diagnostics.txt"
& $py loop_pairs.py --workdir $pose
$rep = Get-Content "$pose\colmap_report.json" | ConvertFrom-Json
$gate = ($rep.registration_pct -ge 95) -and (@($rep.models | Where-Object { $_.points3D -gt 0 }).Count -eq 1) -and ($rep.trajectory.jumps_gt_5x_median.Count -eq 0)
"POSE GATE: registration $([math]::Round($rep.registration_pct,1))%, models with points $(@($rep.models | Where-Object { $_.points3D -gt 0 }).Count), jumps $($rep.trajectory.jumps_gt_5x_median.Count) -> $(if ($gate) {'PASS'} else {'FAIL'})"
if (-not $gate) { "Stopping before dense fusion."; exit 1 }
$common = @("--frames", "$pose\frames", "--colmap_dir", "$pose\colmap_ws\sparse\0", "--depth_cache", "$Out\depth_pred",
            "--stride", "2", "--voxel_size", "0.02", "--depth_trunc", "8.0", "--no_viewer", "--save_aligned_depth")
$runs = [ordered]@{
  baseline = @("--align", "scale", "--fusion", "voxel")
  improved = @("--align", "spatial", "--spatial_params", "experiments\spatial_depth\calib_fr2_xyz_perframe.json",
               "--fusion", "weighted", "--conf_signals", "mv,edge", "--save_confidence")
}
foreach ($name in $runs.Keys) {
  $a = @("reconstruct.py", "--out", "$Out\$name") + $common + $runs[$name]
  & $py @a *> "$Out\$name.log"
  "== $name exit $LASTEXITCODE"
  Get-Content "$Out\$name.log" | Select-String "^Fusion|^Points|^Runtime|Median sparse"
  & $py revisit_consistency.py --recon_dir "$Out\$name" --colmap_dir "$pose\colmap_ws\sparse\0" --k_tum 0.8110
}
"DONE -- next: figures (phone_report / closeups) and measure_object.py with the tape measurement"
