"""Step 1 of the mesh pipeline: dump every mesh object of the TurboSquid .blend to .npz.

Run with Blender's Python module (``pip install bpy``):

    python tools/export_blend.py Rehabilitation_Exoskeleton_Indego_Running_Pose.blend out_dir/

Each object is written in *world* coordinates, converted from cm to m. The base
mesh is used (the subdivision-surface modifiers are ignored on purpose: they
only add render smoothness and 4-16x the triangles).
"""
import os
import sys

import bpy
import numpy as np


def main(path: str, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=path)
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        me = ob.data
        me.calc_loop_triangles()
        V = np.array([(ob.matrix_world @ v.co)[:] for v in me.vertices], dtype=np.float64) / 100.0
        F = np.array([t.vertices[:] for t in me.loop_triangles], dtype=np.int64)
        mats = [m.name if m else "" for m in me.materials]
        fm = np.array([t.material_index for t in me.loop_triangles], dtype=np.int64)
        np.savez(os.path.join(out, ob.name + ".npz"), V=V, F=F, fm=fm, mats=np.array(mats))
        print(ob.name, V.shape, F.shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
