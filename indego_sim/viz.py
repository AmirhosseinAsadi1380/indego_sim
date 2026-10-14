"""Rendering helpers (offscreen MuJoCo renderer -> mp4/gif/png)."""
from __future__ import annotations

import numpy as np


def add_crutches(sim, scene):
    """Draw the crutches (tip -> shoulder) and their load as user geoms."""
    import mujoco
    sup = getattr(sim, "support", None)
    if sup is None or not sup.enabled or sup.sp.kind == "harness":
        return
    for k, side in enumerate("lr"):
        if scene.ngeom >= scene.maxgeom:
            return
        P = sup.hand(sim.d, side)
        T = sup.tips[side].copy()
        if sup.lifted[side]:
            T = T + np.array([0.0, 0.0, 0.12])
        g = scene.geoms[scene.ngeom]
        load = min(1.0, sup.f[k] / (sup.sp.max_load_frac * sup.BW + 1e-9))
        rgba = np.array([0.55 + 0.45 * load, 0.6 - 0.4 * load, 0.65 - 0.5 * load, 1.0], dtype=np.float32)
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, T, P)
        scene.ngeom += 1


def make_camera(sim, distance=2.8, azimuth=135.0, elevation=-12.0, track=True):
    import mujoco
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    cam.lookat[:] = sim.d.subtree_com[sim.root]
    cam.lookat[2] = 0.75
    return cam


def record(sim, controller, duration, path, fps=30, width=960, height=720, azimuth=135.0, distance=2.8,
           elevation=-12.0, show_forces=False, slowmo=1.0):
    """Run `controller` for `duration` seconds while rendering to `path` (.mp4/.gif). Returns the log."""
    import imageio
    import mujoco
    m, d = sim.m, sim.d
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    r = mujoco.Renderer(m, height, width)
    opt = mujoco.MjvOption()
    if show_forces:
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    cam = make_camera(sim, distance, azimuth, elevation)
    frames = []
    obs = sim.observe()
    n = int(round(duration / sim.ctrl_dt))
    every = max(1, int(round((1.0 / fps) / sim.ctrl_dt * slowmo)))   # slowmo < 1 -> slow motion
    for k in range(n):
        tau = controller(obs) if sim.with_exo else None
        obs = sim.step(tau)
        if k % every == 0:
            com = d.subtree_com[sim.root]
            cam.lookat[0] += 0.1 * (com[0] - cam.lookat[0])
            cam.lookat[1] += 0.1 * (com[1] - cam.lookat[1])
            r.update_scene(d, cam, opt)
            add_crutches(sim, r.scene)
            frames.append(r.render().copy())
        if sim.fell and k > 0 and (sim.d.time - sim.t0) > 0.5 + [e[0] for e in sim.events if e[1] == "fall"][0]:
            break
    imageio.mimsave(path, frames, fps=fps) if path.endswith(".gif") else imageio.mimsave(path, frames, fps=fps, quality=7)
    try:
        r.close()
    except Exception:
        pass
    return sim.get_log()


def snapshot(sim, path, azimuth=135.0, distance=2.8, elevation=-12.0, width=960, height=720):
    import imageio
    import mujoco
    m, d = sim.m, sim.d
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    r = mujoco.Renderer(m, height, width)
    cam = make_camera(sim, distance, azimuth, elevation)
    r.update_scene(d, cam)
    imageio.imwrite(path, r.render())
