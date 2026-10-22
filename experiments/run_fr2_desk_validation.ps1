# Additional unseen-sequence validation: TUM fr2_desk with the FROZEN final configuration (nothing tuned).
# RGB frames only for reconstruction; TUM depth + groundtruth used only by the evaluation scripts afterwards.
$ErrorActionPreference = "Continue"
Set-Location Q:\OPEN_CV_SEM5
$py = ".\3d-reconstruction\3d-reconstruction\venv\Scripts\python.exe"
$SEQ = "C:\Users\tyagi\Downloads\rgbd_dataset_freiburg2_desk"
$O = "experiments\tum_fr2_desk_validation"
New-Item -ItemType Directory -Force "$O\frames", "$O\colmap_ws" | Out-Null
$f = Get-ChildItem "$SEQ\rgb" -Filter *.png | Sort-Object Name
for ($i = 0; $i -lt $f.Count; $i += 10) { Copy-Item $f[$i].FullName "$O\frames\" }   # every 10th frame, as fr2_xyz
"frames: $((Get-ChildItem "$O\frames").Count)"
$t = Get-Date
.\colmap-x64-windows-cuda\bin\colmap.exe automatic_reconstructor --workspace_path "$O\colmap_ws" --image_path "$O\frames" --data_type video --dense 0 --single_camera 1 *> "$O\colmap_log.txt"
foreach ($s in Get-ChildItem "$O\colmap_ws\sparse" -Directory) { .\colmap-x64-windows-cuda\bin\colmap.exe model_converter --input_path $s.FullName --output_path $s.FullName --output_type TXT *> $null }
"colmap s: $([int]((Get-Date) - $t).TotalSeconds)"
& $py diagnose_poses.py --workdir $O > "$O\pose_diagnostics.txt" 2>&1
$a = @("reconstruct.py", "--frames", "$O\frames", "--colmap_dir", "$O\colmap_ws\sparse\0", "--out", "$O\final", "--depth_cache", "$O\depth_pred",
       "--stride", "2", "--voxel_size", "0.02", "--depth_trunc", "8.0", "--no_viewer", "--save_aligned_depth",
       "--align", "spatial", "--spatial_params", "experiments\spatial_depth\calib_fr2_xyz_perframe.json",
       "--fusion", "weighted", "--conf_signals", "mv,edge")
& $py @a *> "$O\final_reconstruct.log"
"reconstruct exit $LASTEXITCODE"
& $py eval_tum.py --seq_dir $SEQ --recon_dir "$O\final" *> "$O\final\eval_tum.log"
& $py eval_aligned_depth.py --kind tum --gt_dir $SEQ --recon_dir "$O\final" *> "$O\final\eval_aligned_depth.log"
& $py surface_metrics.py --kind tum --gt_dir $SEQ --recon_dir "$O\final" *> "$O\final\surface_metrics.log"
"ALL DONE"
