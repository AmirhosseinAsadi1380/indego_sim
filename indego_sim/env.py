"""Controller-facing environment (Gymnasium API) for designing / training Indego controllers.

* observation: what the Indego firmware can see at 200 Hz, flattened (see ExoSensors.flatten):
      q[4], qd[4], tau_est[4], acc/gyr thigh_l, thigh_r, hip [3 each], hip_quat[4]   -> 34 values
      + acc_trunk[3], gyr_trunk[3], trunk_quat[4] when the add-on trunk IMU is fitted    -> 44 values
        (SensorParams.trunk_imu = "sternum" | "back" (default "sternum"), or "none" for the stock Indego)
  optionally stacked over `history` control periods (your controller's memory)
* action: joint torque command [N*m] for (hip_l, knee_l, hip_r, knee_r), clipped by the drive model
  (current limits, thermal fold-back, delays happen *after* this, in the simulated drive)
* info: the ground truth you never get on the real device - human joint angles, COM, GRFs, cuff
  interaction wrenches, crutch loads, intent, muscle activity, falls - for rewards and analysis

Every reset can draw a new participant (`subjects`), new session script, and randomised exo / sensor /
human parameters (domain randomisation, `randomize` = relative spread) so controllers do not overfit
to one set of guesses.

    env = IndegoEnv(subjects=["sci_t10_aisA", "sci_t12_aisD"], support="crutches", randomize=0.15)
    obs, info = env.reset(seed=1)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Callable, List, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:  # minimal fallback
    gym, spaces = None, None
    _Base = object

from .config import ExoParams, SensorParams, SimConfig, SupportParams, VolitionParams, randomize as _rand
from .human.sci import get_preset
from .model.builder import build_model
from .sim import HumanExoSim


def default_reward(env: "IndegoEnv", obs, info) -> float:
    """Progress + upright - interaction effort - electrical power; large penalty on falls. Replace freely."""
    r = 2.0 * info["com_vel_x"] * env.dt
    r -= 1e-5 * float(np.sum(np.square(info["interaction_force"]))) * env.dt
    r -= 1e-4 * float(np.sum(np.abs(info["exo_power"]))) * env.dt
    r -= 0.5 * abs(info["torso_pitch"]) * env.dt
    if info["fell"]:
        r -= 10.0
    return r


class IndegoEnv(_Base):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, subjects: List[str] = ("sci_t10_aisA",), support: str = "crutches",
                 schedule=((1.0, "walk"),), episode_s: float = 20.0, history: int = 1, randomize: float = 0.0,
                 reward_fn: Callable = default_reward, exo: ExoParams = None, sensors: SensorParams = None,
                 volition: VolitionParams = None, support_params: SupportParams = None, sim: SimConfig = None,
                 settle: float = 1.0, seed: int = 0, render_mode: Optional[str] = None):
        self.subjects = list(subjects)
        self.support_kind = support
        self.schedule = list(schedule)
        self.episode_s = episode_s
        self.history = history
        self.rand = randomize
        self.reward_fn = reward_fn
        self.exo0 = exo or ExoParams()
        self.sens0 = sensors or SensorParams()
        self.vol0 = volition or VolitionParams()
        self.sup0 = support_params or SupportParams(kind=support)
        self.sup0.kind = support
        self.sim_cfg = sim or SimConfig()
        self.settle = settle
        self.render_mode = render_mode
        self.np_random_ = np.random.default_rng(seed)
        self._models = {}
        self.sim = None
        self.dt = 1.0 / self.exo0.control_rate_hz
        n_obs = (34 + (10 if self.sens0.trunk_imu != "none" else 0)) * history
        lim = np.array([p.joint_peak_torque() for p in (self.exo0.hip_motor, self.exo0.knee_motor) * 2])
        lim = np.array([lim[0], lim[1], lim[0], lim[1]])
        if spaces is not None:
            self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float64)
            self.action_space = spaces.Box(-lim, lim, (4,), np.float64)
        self._renderer = None

    # ------------------------------------------------------------------------------------------
    def _model(self, subj_name, exo):
        # the compiled model depends on the subject's anthropometry and on the exo build parameters;
        # parameters that are only *read* at run time (gains, delays, noise) do not need a rebuild
        key = (subj_name, repr(exo))
        if key not in self._models:
            if len(self._models) > 4:
                self._models.pop(next(iter(self._models)))
            self._models[key] = build_model(exo, get_preset(subj_name), self.sim_cfg, self.sens0)
        return self._models[key]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random_ = np.random.default_rng(seed)
        rng = self.np_random_
        subj_name = self.subjects[int(rng.integers(len(self.subjects)))]
        subject = get_preset(subj_name)
        exo, sens, vol, sup = (copy.deepcopy(x) for x in (self.exo0, self.sens0, self.vol0, self.sup0))
        if self.rand > 0:
            # drive physics, cuffs, sensors, volition, arm-support behaviour and spasticity are all randomised;
            # the model is recompiled (~1 s) because masses/cuffs/motor constants live in the compiled model
            exo = _rand(exo, rng, self.rand)                    # recursive: motors, cuffs, masses, AFO, ...
            sens = _rand(sens, rng, self.rand)
            # strap misalignment of the add-on trunk IMU (additive, up to +-5 deg at rand = 0.15)
            sens.trunk_imu_mount_pitch_deg += float(rng.uniform(-1, 1)) * 5.0 * min(1.0, self.rand / 0.15)
            vol = _rand(vol, rng, self.rand)
            sup = _rand(sup, rng, self.rand)
            sup.kind = self.support_kind
            subject.spasticity = _rand(subject.spasticity, rng, self.rand)
            model = build_model(exo, subject, self.sim_cfg, sens)
        else:
            model = self._model(subj_name, self.exo0)
        self.sim = HumanExoSim(exo=exo, subject=subject, volition=vol, support=sup, sensors=sens,
                               sim=self.sim_cfg, schedule=list(self.schedule), seed=int(rng.integers(1 << 31)),
                               model=model, log_muscles=False, log_rate_hz=50)
        obs = self.sim.reset(settle=self.settle)
        self.subject_name = subj_name
        self._hist = deque([self.sim.sensors.flatten(obs)] * self.history, maxlen=self.history)
        self._x_prev = float(self.sim.d.subtree_com[self.sim.root][0])
        return self._obs(), self._info(obs)

    def _obs(self):
        return np.concatenate(list(self._hist))

    def _info(self, obs):
        s = self.sim
        d = s.d
        com = d.subtree_com[s.root]
        w = s.interaction_wrenches()
        F = np.concatenate([w[k][:3] for k in sorted(w)])
        from .model.utils import body_tilt
        pitch, roll = body_tilt(d.xmat[s.torso].reshape(3, 3) @ s.human.R_torso0.T)
        vx = (float(com[0]) - self._x_prev) / self.dt
        self._x_prev = float(com[0])
        return dict(t=d.time - s.t0, subject=self.subject_name, com=com.copy(), com_vel_x=vx,
                    torso_pitch=pitch, torso_roll=roll, grf_l=s._grf["l"].copy(), grf_r=s._grf["r"].copy(),
                    crutch_force=s.support.F.copy(), interaction_force=F, interaction=w,
                    exo_power=s.drive.power(d), exo_temp=s.drive.winding_temp(d), human_q=d.qpos[s._hq].copy(),
                    intent=s.human.intent, events=list(s.events), fell=s.fell, raw_obs=obs)

    def step(self, action):
        obs = self.sim.step(np.asarray(action, float))
        self._hist.append(self.sim.sensors.flatten(obs))
        info = self._info(obs)
        r = float(self.reward_fn(self, obs, info)) if self.reward_fn else 0.0
        terminated = bool(self.sim.fell)
        truncated = info["t"] >= self.episode_s
        return self._obs(), r, terminated, truncated, info

    def render(self):
        import mujoco
        from .viz import add_crutches, make_camera
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.sim.m, 480, 640)
            self._cam = make_camera(self.sim)
        com = self.sim.d.subtree_com[self.sim.root]
        self._cam.lookat[:2] = com[:2]
        self._renderer.update_scene(self.sim.d, self._cam)
        add_crutches(self.sim, self._renderer.scene)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
