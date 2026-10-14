"""Template: plug your own controller into the Gymnasium environment.

Your controller gets exactly what the Indego firmware gets (obs dict at 200 Hz) and returns 4 joint
torques. The env randomises the participant/exo/sensors each episode (domain randomisation) and returns
the ground truth in `info` for evaluation (cuff forces, falls, COM, crutch loads ...).

    python examples/03_your_controller_env.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indego_sim.controllers.indego_fsm import IndegoFSMController   # noqa: E402
from indego_sim.env import IndegoEnv                                 # noqa: E402


class MyController:
    """Replace with yours. obs keys: q, qd, tau_est, current, acc_/gyr_thigh_l|r|hip, hip_quat, stale, t
    (+ acc_trunk, gyr_trunk, trunk_quat when the add-on trunk IMU is fitted: SensorParams.trunk_imu)."""

    def __init__(self):
        self.inner = IndegoFSMController()

    def reset(self, env):
        # estimator constants for this episode's (possibly randomised) participant model
        self.inner = IndegoFSMController.from_model(env.sim.m)

    def __call__(self, obs):
        return self.inner(obs)


env = IndegoEnv(subjects=["sci_t10_aisA", "sci_t12_aisD"], support="crutches", schedule=[(1.0, "walk")],
                episode_s=12.0, randomize=0.10)
ctrl = MyController()
print("observation vector:", env.observation_space.shape, "(34 stock Indego values + 10 trunk IMU)")
for ep in range(2):
    obs, info = env.reset(seed=ep)
    ctrl.reset(env)
    ret, n, peak_cuff = 0.0, 0, 0.0
    while True:
        action = ctrl(info["raw_obs"])            # or a policy on the flat `obs` vector
        obs, r, terminated, truncated, info = env.step(action)
        ret += r
        n += 1
        peak_cuff = max(peak_cuff, float(np.max(np.abs(info["interaction_force"]))))
        if terminated or truncated:
            break
    steps = sum(1 for _, e in info["events"] if "heel" in str(e))
    print(f"episode {ep}: subject={info['subject']:14s} t={info['t']:5.1f}s fell={info['fell']} "
          f"heel-strikes={steps} return={ret:7.2f} peak cuff force={peak_cuff:5.0f} N")
env.close()
