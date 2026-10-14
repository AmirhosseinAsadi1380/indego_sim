"""What the exo controller can measure: the Indego's onboard sensors with realistic
sampling, latency (native MuJoCo sensor interval + delay), quantisation, noise and bias.

Observation dict (all arrays ordered hip_l, knee_l, hip_r, knee_r):
    q        joint angle [rad] (output side, 14-bit absolute encoder by default)
    qd       joint velocity [rad/s] (filtered finite difference of the encoder, like firmware)
    tau_est  motor-current-based torque estimate [N*m] (K*i*gear, no friction!)
    current  motor current [A]
    acc_thigh_l/r, gyr_thigh_l/r, acc_hip, gyr_hip   IMU specific force [m/s^2], rate [rad/s]
    hip_quat  hip-unit orientation estimate (w,x,y,z)
    acc_trunk, gyr_trunk, trunk_quat   optional add-on upper-trunk IMU (SensorParams.trunk_imu != "none");
              axes = hip-unit IMU axes in the reference posture (x forward, y left, z up)
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..config import SensorParams


class ExoSensors:
    def __init__(self, m: mujoco.MjModel, info, sp: SensorParams, rng: np.random.Generator, ctrl_dt: float,
                 vel_source: str = "diff"):
        self.m, self.info, self.sp, self.rng = m, info, sp, rng
        self.dt = ctrl_dt
        self.vel_source = vel_source
        self.res = 2 * np.pi / (2 ** sp.enc_bits)
        self.gear = np.array(info.gears, float)
        self.K = None

        def adr(name):
            s = m.sensor(name)
            return slice(s.adr[0], s.adr[0] + s.dim[0])
        self.s_enc = [adr(f"enc_{j}") for j in info.exo_joints]
        self.s_vel = [adr(f"vel_{j}") for j in info.exo_joints]
        self.s_trq = [adr(f"trq_{j}") for j in info.exo_joints]
        self.s_encbl = [adr(f"encbl_{j}") if jb else None for j, jb in zip(info.exo_joints, info.exo_backlash_joints)]
        self.s_velbl = [adr(f"velbl_{j}") if jb else None for j, jb in zip(info.exo_joints, info.exo_backlash_joints)]
        self.imus = {"thigh_l": "exo_imu_thigh_l", "thigh_r": "exo_imu_thigh_r", "hip": "exo_imu_hip"}
        self.has_trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "quat_trunk_imu") >= 0
        if self.has_trunk:
            self.imus["trunk"] = "trunk_imu"
        self.s_acc = {k: adr(f"acc_{v}") for k, v in self.imus.items()}
        self.s_gyr = {k: adr(f"gyr_{v}") for k, v in self.imus.items()}
        self.s_quat = {"hip": adr("quat_exo_imu_hip")}
        if self.has_trunk:
            self.s_quat["trunk"] = adr("quat_trunk_imu")
        self.reset()

    def reset(self):
        sp = self.sp
        self.acc_bias = {k: self.rng.normal(0, sp.imu_acc_bias, 3) for k in self.imus}
        self.gyr_bias = {k: self.rng.normal(0, sp.imu_gyro_bias, 3) for k in self.imus}
        self.q_prev = None
        self.qd_f = np.zeros(4)
        self.last = None

    def read(self, d, K):
        sp, rng = self.sp, self.rng
        if self.last is not None and sp.dropout_prob > 0 and rng.random() < sp.dropout_prob:
            out = dict(self.last)
            out["stale"] = True
            return out
        sd = d.sensordata
        q = np.array([sd[s][0] for s in self.s_enc])
        qd_true = np.array([sd[s][0] for s in self.s_vel])
        for k, (sb, sv) in enumerate(zip(self.s_encbl, self.s_velbl)):
            if sb is not None:
                q[k] += sd[sb][0]
                qd_true[k] += sd[sv][0]
        qm = np.round(q / self.res) * self.res + rng.normal(0, np.radians(sp.enc_noise_deg), 4)
        if self.vel_source == "diff":
            raw = np.zeros(4) if self.q_prev is None else (qm - self.q_prev) / self.dt
            a = 1 - np.exp(-2 * np.pi * sp.vel_filter_hz * self.dt)
            self.qd_f += a * (raw - self.qd_f)
            qd = self.qd_f.copy()
        else:
            qd = qd_true + rng.normal(0, 0.01, 4)
        self.q_prev = qm
        i_meas = np.array([sd[s][0] for s in self.s_trq]) / K + rng.normal(0, sp.current_noise, 4)
        out = dict(t=d.time, q=qm, qd=qd, current=i_meas, tau_est=i_meas * K * self.gear, stale=False)
        for k in self.imus:
            out[f"acc_{k}"] = sd[self.s_acc[k]] + self.acc_bias[k] + rng.normal(0, sp.imu_acc_noise, 3)
            out[f"gyr_{k}"] = sd[self.s_gyr[k]] + self.gyr_bias[k] + rng.normal(0, sp.imu_gyro_noise, 3)
        for k, sl in self.s_quat.items():
            qh = sd[sl].copy() + rng.normal(0, 0.002, 4)
            out[f"{k}_quat"] = qh / np.linalg.norm(qh)
        self.last = out
        return out

    KEYS = ["q", "qd", "tau_est", "acc_thigh_l", "gyr_thigh_l", "acc_thigh_r", "gyr_thigh_r", "acc_hip",
            "gyr_hip", "hip_quat"]
    TRUNK_KEYS = ["acc_trunk", "gyr_trunk", "trunk_quat"]

    @staticmethod
    def flatten(obs) -> np.ndarray:
        """34 values (stock Indego sensors) + 10 when the trunk IMU is fitted."""
        keys = ExoSensors.KEYS + (ExoSensors.TRUNK_KEYS if "trunk_quat" in obs else [])
        return np.concatenate([np.atleast_1d(obs[k]) for k in keys])

    @property
    def n_flat(self) -> int:
        return 34 + (10 if self.has_trunk else 0)
