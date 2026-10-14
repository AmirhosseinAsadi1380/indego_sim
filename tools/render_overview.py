"""Render results/model_overview.png: the SCI user standing in the exo with crutches, two views, with the
IMU locations (hip unit, thighs, add-on trunk IMU) shown as small spheres.

    MUJOCO_GL=egl python tools/render_overview.py
"""
import os
import sys

import imageio
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indego_sim import viz                                                # noqa: E402
from indego_sim.config import SupportParams                               # noqa: E402
from indego_sim.controllers.indego_fsm import IndegoFSMController         # noqa: E402
from indego_sim.human.sci import get_preset                               # noqa: E402
from indego_sim.sim import HumanExoSim                                    # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "model_overview.png")
sim = HumanExoSim(subject=get_preset("sci_t10_aisA"), support=SupportParams(kind="crutches"), schedule=[])
sim.reset(settle=1.0)
sim.run(IndegoFSMController(), 1.0)
m = sim.m
m.vis.global_.offwidth, m.vis.global_.offheight = 1200, 1000
for name in ("exo_imu_hip", "exo_imu_thigh_l", "exo_imu_thigh_r", "trunk_imu"):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    if i >= 0:
        m.site_rgba[i] = [1.0, 0.55, 0.0, 1.0]
        m.site_size[i][0] = 0.018
opt = mujoco.MjvOption()
opt.sitegroup[4] = True
imgs = []
for az in (135, 90):
    r = mujoco.Renderer(m, 1000, 1200)
    cam = viz.make_camera(sim, distance=2.0, azimuth=az, elevation=-8)
    cam.lookat[2] = 0.75
    r.update_scene(sim.d, cam, scene_option=opt)
    viz.add_crutches(sim, r.scene)
    imgs.append(r.render().copy())
imageio.imwrite(OUT, np.concatenate(imgs, axis=1))
print("wrote", os.path.abspath(OUT))
