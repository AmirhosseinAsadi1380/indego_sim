"""Dynamic sagittal walking controller (foot placement), an alternative to the Indego-like FSM.

The Indego-like FSM (indego_fsm.py) steps quasi-statically: it waits in double support until the user's posture asks
for the next step, which needs the user (and crutches) to bring the body over the front foot. This controller walks
the way planar biped robots do (Raibert stepping / SIMBICON, Yin et al. 2007): it keeps stepping at a fixed rhythm
and balances the body by where it puts each swing foot.

* swing leg: the ANKLE travels (min-jerk, in x relative to the hip) from where it was to a landing point set every
  period by Raibert's rule, v*T/2 ahead of the COM plus `k_speed` x (v - v_des) (COM and its velocity from the
  exo's own estimate), while the knee flexes for toe clearance and extends before touch-down; the swing hip tracks the
  WORLD-frame thigh angle (joint angle minus hip-unit pitch) that puts the ankle there for the knee's measured bend
  (so an early knee extension does not throw the foot past the landing point);
* stance knee held almost straight;
* stance hip: keeps the hip unit (pelvis) at its pitch set-point and cancels the swing hip's reaction on the pelvis
  (tau_stance = tau_pelvis - tau_swing); the pelvis has priority within the drives' torque limit;
* gravity compensation of the swing leg.

It uses only what the device senses (encoders, hip-unit AHRS, the add-on trunk IMU through the shared COM
estimator). There are no foot sensors on the stock Indego: steps change on time - or, with the optional foot switches
(SensorParams.foot_switches), at swing-foot contact as in SIMBICON. It starts stepping `start_after` seconds
after calibration; the quiet-standing forward lean gives the first step its forward motion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .indego_fsm import (HIP_L, HIP_R, KNEE_L, KNEE_R, IndegoFSMController, IndegoFSMParams, _knee_bump,
                         _minjerk, estimator_constants)


@dataclass
class DynamicWalkParams(IndegoFSMParams):
    step_time: float = 0.6                              # s per step (single support, time-based) [design]
    start_after: float = 1.0                            # s standing after calibration before the first step
    v_des: float = 0.35                                 # m/s walking speed [design; Indego users 0.2-0.5]
    k_speed: float = 0.15                               # s: landing point shift per m/s of speed error (Raibert)
    place_ahead: float = 0.0                            # m extra landing offset
    max_reach: float = 0.40                             # m landing point limits relative to the hip
    min_reach: float = -0.25
    swing_reach_at: float = 0.6                         # fraction of the step at which the swing thigh reaches its goal
    swing_knee_peak: float = np.radians(55.0)
    swing_knee_peak_at: float = 0.35
    stance_knee_q: float = np.radians(3.0)
    kp_swing_hip: float = 250.0
    kd_swing_hip: float = 6.0
    vel_tau: float = 0.03                               # s low-pass of the COM velocity estimate
    goal_tau: float = 0.08                              # s low-pass of the landing point (the velocity estimate is noisy)
    goal_freeze_at: float = 0.7                         # fraction of the step after which the landing point is fixed
    # with foot switches (SensorParams.foot_switches): the step ends at swing-foot contact after this fraction of
    # step_time, at the latest after max_step_frac x step_time
    min_step_frac: float = 0.6
    max_step_frac: float = 1.5
    contact_tol: float = 0.05                           # m a contact only ends the step once the swing ankle is this close
                                                        # to its landing point (a toe scuff earlier in the swing does not)


class DynamicWalkController(IndegoFSMController):
    """See module docstring. Shares the Indego FSM's calibration and COM estimator; replaces its state machine."""

    def __init__(self, p: DynamicWalkParams = None):
        super().__init__(p or DynamicWalkParams())

    @classmethod
    def from_model(cls, m, **kw):
        return cls(DynamicWalkParams(**{**estimator_constants(m), **kw}))

    def reset(self):
        super().reset()
        self.walking = False
        self.t_step = 0.0
        self.swing = "r"
        self.c_prev = None
        self.v_com = 0.0
        self.x_start = 0.0
        self.x_goal = 0.0

    def _stance_ankle(self, est):
        return est["ankle_l"] if self.swing == "r" else est["ankle_r"]

    def __call__(self, obs):
        p = self.p
        self.t += p.dt
        q, qd = np.asarray(obs["q"], float), np.asarray(obs["qd"], float)
        est = self.estimate(obs)
        self.est = est
        if self.t < p.calib_time:                       # same quiet-standing calibration as the FSM
            self.com0 = est["com_rel_mid"] if self.com0 is None else \
                self.com0 + (est["com_rel_mid"] - self.com0) * min(1.0, p.dt / 0.2)
            self.phi_cal = est["phi"] if self.phi_cal is None else \
                self.phi_cal + (est["phi"] - self.phi_cal) * min(1.0, p.dt / 0.2)
            rr = np.array([est["roll"], est.get("roll_trunk", 0.0)])
            self.roll_ref = rr if self.roll_ref is None else self.roll_ref + (rr - self.roll_ref) * min(1.0, p.dt / 0.2)
            if "theta" in est:
                ref = np.array([est["phi"], est["theta"]])
                self.trunk_ref = ref if self.trunk_ref is None else \
                    self.trunk_ref + (ref - self.trunk_ref) * min(1.0, p.dt / 0.2)
        phi, phidot = est["phi"], float(obs["gyr_hip"][1])
        # COM relative to the stance ankle and its velocity (the stance foot does not move)
        c = est["com"] - self._stance_ankle(est)
        if self.c_prev is not None:
            a = p.dt / (p.vel_tau + p.dt)
            self.v_com += a * ((c - self.c_prev) / p.dt - self.v_com)
        self.c_prev = c

        if not self.walking:
            if self.t > p.calib_time + p.start_after:
                self.walking = True
                self._new_step(q, phi)
            tau = np.array(p.kp, float) * (np.array(p.stand_q) - q) - np.array(p.kd, float) * qd
            self.last_tau = np.clip(tau, -p.tau_max, p.tau_max)
            self.state = self.STAND
            return self.last_tau

        self.t_step += p.dt
        if "foot_contact" in obs:
            touched = obs["foot_contact"][0 if self.swing == "l" else 1] > 0.5
            near = est[f"ankle_{self.swing}"] >= self.x_goal - p.contact_tol
            switch = (self.t_step >= p.min_step_frac * p.step_time and touched and near) or \
                self.t_step >= p.max_step_frac * p.step_time
        else:
            switch = self.t_step >= p.step_time
        if switch:
            self.swing = "l" if self.swing == "r" else "r"
            self.steps_total += 1
            self.n_steps += 1
            self._new_step(q, phi)
            c = est["com"] - self._stance_ankle(est)
            self.c_prev = c
        self.state = self.STEP
        s = min(self.t_step / p.step_time, 1.0)
        sw, kn = (HIP_R, KNEE_R) if self.swing == "r" else (HIP_L, KNEE_L)
        st, kst = (HIP_L, KNEE_L) if self.swing == "r" else (HIP_R, KNEE_R)

        # foot placement (Raibert): land v*T/2 ahead of the COM (steady walking) plus a correction for the speed error;
        # positions relative to the hip
        x_land = est["com"] + self.v_com * p.step_time / 2 + p.k_speed * (self.v_com - p.v_des) + p.place_ahead
        if s < p.goal_freeze_at:                          # committed footstep: filtered, then frozen
            self.x_goal += min(1.0, p.dt / p.goal_tau) * (float(np.clip(x_land, p.min_reach, p.max_reach)) - self.x_goal)
        reach = max(p.swing_reach_at, 0.05)

        def plan(s_):
            x_ = self.x_start + (self.x_goal - self.x_start) * _minjerk(s_ / reach)
            k_ = p.stance_knee_q + (p.swing_knee_peak - p.stance_knee_q) * _knee_bump(min(s_, 1.0), p.swing_knee_peak_at)
            return x_, k_
        x_des, k_des = plan(s)
        th_des = self._thigh_for(x_des, q[kn])           # for the knee's ACTUAL bend: the ankle lands where planned
        x_n, k_n = plan(s + 0.01)                         # velocity feed-forward of the planned thigh motion: the
        thd_des = (self._thigh_for(x_n, k_n) - self._thigh_for(x_des, k_des)) / (0.01 * p.step_time)   # damping
        th_sw = q[sw] - phi                               # (kd is limited by the loop latency) then only acts on errors
        thd_sw = qd[sw] - phidot
        tau = np.zeros(4)
        tau[sw] = p.kp_swing_hip * (th_des - th_sw) + p.kd_swing_hip * (thd_des - thd_sw)
        # swing-leg gravity compensation (as in the FSM)
        sh_world = th_sw - q[kn]
        tau[sw] += p.leg_mass * 9.81 * p.leg_com * np.sin(th_sw)
        # knees
        kp = np.array(p.kp, float)
        kd = np.array(p.kd, float)
        tau[kn] = kp[kn] * (k_des - q[kn]) - kd[kn] * qd[kn] - p.shank_mass * 9.81 * p.shank_com * np.sin(sh_world)
        tau[kst] = p.kp_stance_knee * (p.stance_knee_q - q[kst]) - kd[kst] * qd[kst]
        # stance hip: pelvis pitch set-point, cancelling the swing hip's reaction on the pelvis
        phi_ref = (self.phi_cal or 0.0) + p.pelvis_pitch_des
        tau_pel = float(np.clip(-p.kp_pelvis * (phi - phi_ref) - p.kd_pelvis * phidot, -p.tau_max, p.tau_max))
        # the pelvis has priority: the swing hip only gets what leaves the stance hip within its torque limit
        tau[sw] = float(np.clip(tau[sw], tau_pel - p.tau_max, tau_pel + p.tau_max))
        tau[st] = tau_pel - tau[sw]
        tau = np.clip(tau, -p.tau_max, p.tau_max)
        self.last_tau = tau
        self._traj_q = q.copy()
        self.log.append((self.t, self.state, self.swing, est["com_rel_mid"], est["com_rel_front"], *q))
        return tau

    def _thigh_for(self, x, knee):
        """World thigh angle that puts the ankle `x` ahead of the hip with exo knee angle `knee`."""
        p = self.p
        bend = knee - p.shank_offset                      # angle between thigh and knee->ankle line
        l = np.sqrt(p.L_thigh ** 2 + p.L_shank ** 2 + 2 * p.L_thigh * p.L_shank * np.cos(bend))
        beta = np.arctan2(p.L_shank * np.sin(bend), p.L_thigh + p.L_shank * np.cos(bend))
        return float(np.arcsin(np.clip(x / l, -0.95, 0.95)) + beta)

    def _new_step(self, q, phi):
        self.t_step = 0.0
        sw, kn = (HIP_R, KNEE_R) if self.swing == "r" else (HIP_L, KNEE_L)
        th = q[sw] - phi
        self.x_start = float(self.p.L_thigh * np.sin(th) + self.p.L_shank * np.sin(th - q[kn] + self.p.shank_offset))
        self.x_goal = float(self.est["com"]) if self.est else 0.0
