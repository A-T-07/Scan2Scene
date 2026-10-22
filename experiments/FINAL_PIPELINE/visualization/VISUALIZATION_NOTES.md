# Interactive visualization of the final phone reconstruction (export only)

Nothing in the reconstruction pipeline, COLMAP settings, evaluation scripts, final results or `scene.ply` was changed.
`scene.ply` MD5 is identical before and after export (049BD0DE85ABF10714EAECCAF16992A2).

## Reference: depth-anything.cpp

Inspected `https://github.com/mudler/depth-anything.cpp` (`docs/EXPORT.md`, `src/glb_export.cpp`):

- dependency-free glTF 2.0 binary (GLB) writer, parity-checked against a NumPy reference (`scripts/parity_glb.py`);
- the reconstruction is **one `POINTS` primitive** (mode 0): `POSITION` float32 VEC3 + `COLOR_0` normalized uint8 VEC4,
  no materials, no normals, no mesh;
- cameras are a separate **`LINES` primitive** (mode 1): an 8-segment frustum per camera, coloured per camera
  (HSV hue), sized from the 5th–95th percentile bounding-box diagonal;
- points are transformed OpenCV→glTF (Y and Z negated) and centred on their per-axis median; confidence-filtered, ≤ 1 M points;
- it also exports PLY and COLMAP text/binary models; the repository ships **no viewer** — the GLB is opened in any
  glTF viewer, which provides orbit / zoom.

Conclusion: a coloured point-cloud GLB with camera line primitives is the same representation; normals / a mesh are
not needed.

## What was produced

| File | Content | Size |
|---|---|---|
| `final_phone.glb` | final point cloud only | 58.1 MB |
| `final_phone_with_camera_trajectory.glb` | same point cloud + camera path and markers | 58.1 MB |
| `viewer.html` | three.js viewer (orbit / zoom / pan, point size, trajectory toggle, top view) | — |
| `open_viewer.bat` | starts a local web server on port 8765 and opens the viewer | — |
| `export_glb.py` | exporter (re-run to regenerate) | — |
| `export_report.json` | counts, round-trip check, node matrix | — |

- **Source PLY:** `experiments\phone_real_v2\improved\scene.ply` (final configuration: spatial + weighted, mv,edge).
- **Points:** 3,629,554 (all points; no filtering, sub-sampling or confidence threshold).
- **Format:** glTF 2.0 binary; mesh with `POINTS` primitive (`POSITION` float32, `COLOR_0` uint8 normalized RGBA,
  alpha = 255), no materials, no normals.
- **Geometry modified? No.** Vertex positions are the PLY positions, cast float64 → float32 (glTF has no float64
  position type). Round-trip max |difference| = 2.4 × 10⁻⁷ fused units (coordinates up to 5.6 units). No re-centring,
  re-projection or filtering is baked into the data. The display transform — rotation to the gravity-aligned frame of
  `phone_report.py` (glTF y-up), floor at y = 0, median-centred in x/z — is stored only in the glTF node matrix and
  applied by the viewer.
- **Colours preserved? Yes.** RGB bytes copied directly from the PLY; round-trip identical. The viewer displays them
  without sRGB re-encoding.
- **Mesh generated? No.** The Poisson mesh (`..\mesh\`) is not used here; the point cloud remains the result.
- **Camera trajectory** (`..._with_camera_trajectory.glb`): 213 recovered COLMAP camera centres from
  `improved\trajectory_fused.txt` (same fused frame as the cloud, same node matrix) as a magenta polyline; yellow
  frustums every 10th camera (depth 0.24 fused units ≈ 2 % of the scene diagonal, 16:9); green cross = start, red
  cross = end. These are separate `LINES` primitives, clearly distinct from the reconstruction.

## How to open

1. Double-click `open_viewer.bat` (uses the project venv's Python to serve this folder), or run in this folder:
   `python -m http.server 8765` and open `http://127.0.0.1:8765/viewer.html`.
2. Without a server: open `viewer.html` directly and use **Open .glb…** (or drag the .glb onto the page).
3. Other glTF viewers (e.g. online glTF viewers, Blender) can open the .glb files; support for `POINTS`
   primitives and vertex colours varies between viewers.

Tested in the Claude desktop browser pane: both files load (3,629,554 points), orbit / zoom / top view work,
the room renders upright and the trajectory lies inside the room; no console errors.

## Limitations

- Visualization only; it inherits every limitation of the reconstruction (doubled walls from the start–end
  inconsistency, gaps on plain surfaces, non-metric fused units — see `..\PHONE_RESULTS.md`).
- Point sprites have a fixed world size; dense areas look solid, sparse areas look speckled; there is no lighting
  (colours only), as in depth-anything.cpp's export.
- `viewer.html` loads three.js 0.160 from the jsdelivr CDN, so it needs an internet connection.
- 58 MB per file: loads in a few seconds locally; heavy for sharing online. The files are not in the Git repo
  (`.gitignore` allow-list excludes `.glb`).
- The start/end markers and path show the recovered COLMAP trajectory, not ground truth.
