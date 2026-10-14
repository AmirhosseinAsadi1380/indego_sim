"""Indego drive electronics as the controller sees them.

The physics of each joint drive (current loop, back-EMF, voltage limit, rotor inertia,
friction, cogging, copper heating) runs inside MuJoCo's `dcmotor` actuator. This class
adds the firmware-level behaviour around it:

* joint-torque command -> motor-torque setpoint (divide by gear), clipped to the
  present current limit
* I^2t / winding-temperature fold-back from peak to continuous current
* normally-locked knee brakes (engage on fault / command / power loss)
* telemetry: current, winding temperature, electrical power
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..config import ExoParams


class ExoDrive:
    def __init__(self, m: mujoco.MjModel, info, exo: ExoParams):
        self.m, self.info, self.exo = m, info, exo
        self.act = np.array([m.actuator(a).id for a in info.exo_actuators])
        self.dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in info.exo_joints])
        self.gear = np.array(info.gears, float)
        self.mp = [exo.hip_motor if "hip" in j else exo.knee_motor for j in info.exo_joints]
        self.K = np.array([p.K for p in self.mp])
        self.i_peak = np.array([p.i_peak for p in self.mp])
        self.i_cont = np.array([p.i_cont for p in self.mp])
        self.t_max = np.array([p.t_winding_max for p in self.mp])
        self.ambient = np.array([p.ambient_C for p in self.mp])
        self.base_friction = m.dof_frictionloss[self.dof].copy()
        # dcmotor activation layout (MuJoCo order): [slew][integral][temperature][bristle][current]
        self.T_idx, self.i_idx = [], []
        for a, p in zip(self.act, self.mp):
            k = m.actuator_actadr[a]
            ti = k if (p.thermal_R > 0 or p.thermal_C > 0) else None
            if ti is not None:
                k += 1
            if p.friction_model == "lugre" and p.lugre_stiffness > 0:
                k += 1
            ii = k if p.te > 0 else None
            self.T_idx.append(ti)
            self.i_idx.append(ii)
        self.brake = np.zeros(4, bool)
        self.fault = False
        self.last_cmd = np.zeros(4)

    def current_limit(self, d):
        Tw = self.winding_temp(d)
        lim = self.i_peak.copy()
        fold = np.clip((Tw - (self.t_max - 15.0)) / 15.0, 0.0, 1.0)      # linear fold-back over the last 15 K
        lim = lim - fold * (self.i_peak - self.i_cont)
        lim = np.where(Tw > self.t_max, 0.5 * self.i_cont, lim)
        return lim

    def command(self, d, tau_joint):
        tau = np.asarray(tau_joint, float).copy()
        if self.fault:
            tau[:] = 0.0
        lim = self.current_limit(d) * self.K * self.gear
        tau = np.clip(tau, -lim, lim)
        self.last_cmd = tau
        d.ctrl[self.act] = tau / self.gear
        return tau

    def set_brake(self, knee_l: bool, knee_r: bool):
        """Engage/release the normally-locked knee brakes (modelled as large dry friction)."""
        for k, on in ((1, knee_l), (3, knee_r)):
            self.brake[k] = on
            self.m.dof_frictionloss[self.dof[k]] = self.base_friction[k] + (self.exo.knee_brake_torque if on else 0.0)

    # --- telemetry ------------------------------------------------------------------------------
    def winding_temp(self, d):
        return np.array([self.ambient[k] + (d.act[i] if i is not None else 0.0) for k, i in enumerate(self.T_idx)])

    def current(self, d):
        out = np.empty(4)
        for k, i in enumerate(self.i_idx):
            out[k] = d.act[i] if i is not None else d.actuator_force[self.act[k]] / self.K[k]
        return out

    def joint_torque(self, d):
        """Torque the drive actually applies to the joint (after limits; excludes joint friction)."""
        return d.qfrc_actuator[self.dof].copy()

    def power(self, d):
        i = self.current(d)
        R = np.array([p.R for p in self.mp])
        omega_m = d.qvel[self.dof] * self.gear
        return i * i * R + self.K * i * omega_m   # electrical input power per motor (W)
