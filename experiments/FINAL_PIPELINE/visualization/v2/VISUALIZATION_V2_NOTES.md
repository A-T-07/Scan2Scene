# Viewer v2 — presentation-ready interactive view (visualization only)

**Open:** double-click `open_viewer.bat` (serves `..\` on http://127.0.0.1:8766 with the project venv's Python and
opens `v2/viewer.html`). Requires an internet connection for three.js 0.160 (jsdelivr CDN).

**Data:** loads the existing, unchanged export `..\final_phone_with_camera_trajectory.glb` (3,629,554 points from
`experiments\phone_real_v2\improved\scene.ply`, positions as in the PLY, RGB identical; see `..\VISUALIZATION_NOTES.md`).
No new export was made; `scene.ply` and the v1 viewer are untouched.

**Presentation mode (default):** light neutral background; RGB points rendered as round sprites (disc texture +
alpha test) with perspective-aware size (slider); trajectory, markers and frustums hidden; start view = elevated
oblique overview into the room. The top ~12 % of the room's height (ceiling band) is hidden by a display-only
clipping plane so the interior is visible from above; the data are not altered.

**Buttons:** Inside view (from one end of the room, looking along it), Overview, Top view, Light / Dark background,
point size, collapsible panel.

**Analysis mode (optional):** recovered COLMAP camera path (magenta), start (green) / end (red) markers, camera
frustums (yellow, every 10th camera; off until ticked). The single LINES primitive of the export is split in the
viewer by vertex colour so each can be toggled.

**Not implemented (by design):** eye-dome lighting / custom splatting shaders — skipped for robustness.

**Tested:** Claude desktop browser pane — loads 3,629,554 points, overview shows the room (window, door, bed, desk),
analysis mode shows path + markers, no console errors.

**Limitations:** inherits the reconstruction's artifacts (doubled walls, gaps on plain surfaces); units are fused,
not metric; 58 MB file loads in a few seconds locally.
