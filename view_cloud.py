r"""
view_cloud.py -- interactive Open3D viewer for a reconstructed point cloud (read-only; the file is never modified).

Mouse: left-drag rotate | right-drag / ctrl+left-drag pan | wheel zoom
Keys:  +/- point size | R reset view | B toggle background | Q / Esc quit

Usage:
  python view_cloud.py experiments\phone_real_v2\improved\scene.ply
  python view_cloud.py experiments\phone_real_v2\baseline\scene.ply
"""
import argparse
import os

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", help="Path to scene.ply")
    ap.add_argument("--point_size", type=float, default=1.5)
    a = ap.parse_args()

    pcd = o3d.io.read_point_cloud(a.ply)
    if len(pcd.points) == 0:
        raise SystemExit(f"No points in {a.ply}")
    print(f"{a.ply}: {len(pcd.points):,} points, colours: {pcd.has_colors()}")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"{os.path.basename(os.path.dirname(os.path.abspath(a.ply)))} - {os.path.basename(a.ply)}",
                      width=1600, height=900)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = a.point_size
    opt.background_color = np.array([0.04, 0.05, 0.07])
    opt.light_on = False                     # show the stored RGB colours as-is

    dark = [True]

    def toggle_bg(v):
        dark[0] = not dark[0]
        v.get_render_option().background_color = np.array([0.04, 0.05, 0.07]) if dark[0] else np.ones(3)
        return False

    def size(delta):
        def f(v):
            o = v.get_render_option()
            o.point_size = float(np.clip(o.point_size + delta, 1.0, 10.0))
            return False
        return f

    vis.register_key_callback(ord("B"), toggle_bg)
    vis.register_key_callback(ord("="), size(+0.5))   # '+' key (unshifted)
    vis.register_key_callback(ord("-"), size(-0.5))
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
