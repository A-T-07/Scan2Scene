# Optional Poisson mesh — VISUALIZATION ONLY

Input: `experiments\phone_real_v2\improved\scene.ply` (3,629,554 points), read only — MD5 identical before and after
(049BD0DE85ABF10714EAECCAF16992A2). Meshes are not used by any evaluation and no accuracy is claimed for them.
Script: `make_poisson_mesh.py`; numbers: `mesh_report.json`.

**Normals:** the input cloud has no normals. They were estimated (hybrid KNN/radius: radius 0.08 fused units, max 30
neighbours) and oriented towards the nearest recovered camera centre.

**Poisson:** Open3D `create_from_point_cloud_poisson`, scale 1.1, linear_fit off. Open3D logged one warning
("bad average roots: 4", depth 9).

| Mesh | Vertices | Triangles | Time | Vertices > 5 cm from any input point | > 20 cm | File |
|---|---|---|---|---|---|---|
| depth 9 | 1,638,598 | 3,316,860 | 9 s | 3.2 % | 0.33 % | `phone_poisson_depth9.ply` (121 MB) |
| depth 9, lowest-density 10 % removed | 1,474,512 | 2,959,097 | — | 0.42 % | 0 % | `phone_poisson_depth9_density_trimmed10.ply` (108 MB) |
| depth 10 | 11,244,482 | 22,826,008 | 44 s | 0.69 % | 0.06 % | `phone_poisson_depth10.ply` (830 MB) |
| depth 10, lowest-density 10 % removed | 10,118,461 | 20,375,246 | — | 0 % | 0 % | `phone_poisson_depth10_density_trimmed10.ply` (745 MB) |

"cm" = fused units × 100 (not metric). The density trim is a display choice, not a validated filter.

**Visual assessment** (`phone_cloud_vs_poisson.png`, `phone_cloud_vs_poisson_top.png`, identical camera and renderer for
all panels):

- Untrimmed Poisson fills unobserved regions: the top view shows smooth, textureless patches bridging the unobserved
  central floor area, and extra blobs outside the room. This surface is invented, not reconstructed.
- The doubled / thick walls of the point cloud become thick or layered sheets; the mesh does not remove the
  start–end inconsistency, it hides it.
- Vertex colours are speckled and generally less legible than the point cloud; fine objects (clothes, desk items)
  are less recognisable.
- Density trimming removes most of the invented floor fill at the cost of holes.

**Verdict:** of limited use. The trimmed depth-9 mesh can serve as a rough overview surface, but it is not
more informative than the point cloud and must not be presented as more accurate. Depth 10 adds size (≈ 0.8 GB), not
clarity, at this point density. The point cloud remains the result.
