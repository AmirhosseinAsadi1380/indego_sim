"""Two more reference controllers that only use the onboard sensors.

TransparentController
    "Zero-torque" / transparent mode for people with leg volition: compensates the drive's own
    friction (Coulomb + viscous, smoothed through zero velocity) and the exo segments' gravity.
    How transparent it really is shows up in the logged cuff interaction forces.

AssistAsNeeded
    Phase-based assistance: the gait phase is estimated from the thigh angle/velocity phase portrait
    (no foot sensors needed); the assistance is an impedance around a template trajectory whose gain
    adapts down while the person tracks well and up when they fall behind (forgetting-factor rule).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..human.gait_library import DEFAULT_GAIT

HIP_L, KNEE_L, HIP_R, KNEE_R = 0, 1, 2, 3


@dataclass
class TransparentParams:
    coulomb: tuple = (1.5, 1.5, 1.5, 1.5)        # N*m joint side, from your friction identification
    viscous: tuple = (0.25, 0.25, 0.25, 0.25)    # N*m*s/rad
    v_smooth: float = 0.05                       # rad/s: tanh smoothing of the Coulomb term (avoid chatter)
    friction_gain: float = 0.8                   # <1: under-compensate (over-compensation is destabilising)
    gravity_comp: bool = True
    thigh_mass: float = 3.3 + 0.55               # exo thigh + knee actuator [kg]
    thigh_com: float = 0.20                      # m below the hip axis
    shank_mass: float = 1.2                      # exo shank + AFO [kg]
    shank_com: float = 0.22                      # m below the knee axis
    L_thigh: float = 0.405
    tau_max: float = 15.0


class TransparentController:
    def __init__(self, p: TransparentParams = None):
        self.p = p or TransparentParams()

    def reset(self):
        pass

    def __call__(self, obs):
        p = self.p
        q, qd = obs["q"], obs["qd"]
        tau = p.friction_gain * (np.array(p.coulomb) * np.tanh(qd / p.v_smooth) + np.array(p.viscous) * qd)
        if p.gravity_comp:
            w, x, y, z = obs["hip_quat"]
            phi = np.arctan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y))     # pelvis pitch (+ forward)
            g = 9.81
            for hip, knee in ((HIP_L, KNEE_L), (HIP_R, KNEE_R)):
                th = q[hip] - phi                    # world thigh angle (+ = knee ahead)
                sh = th - q[knee]                    # world shank angle
                # hip: + = flexion; gravity on a forward-swung segment pulls it back -> add + torque
                tau[hip] += g * (p.thigh_mass * p.thigh_com * np.sin(th)
                                 + p.shank_mass * (p.L_thigh * np.sin(th) + p.shank_com * np.sin(sh)))
                tau[knee] += -g * p.shank_mass * p.shank_com * np.sin(sh)
        return np.clip(tau, -p.tau_max, p.tau_max)


@dataclass
class AANParams:
    k_max: tuple = (60.0, 40.0, 60.0, 40.0)     # N*m/rad maximum assistance stiffness
    d_ratio: float = 0.08
    k0: float = 0.5                             # initial gain fraction
    adapt_up: float = 2.0                       # 1/s per rad of tracking error
    forget: float = 0.15                        # 1/s gain decay (Emken / Banala style)
    phase_filter_hz: float = 3.0
    cadence_spm: float = 60.0                   # nominal cadence used only to scale the template velocity
    tau_max: float = 40.0
    transparent: TransparentParams = field(default_factory=TransparentParams)


class AssistAsNeeded:
    def __init__(self, p: AANParams = None, dt: float = 0.005):
        self.p = p or AANParams()
        self.dt = dt
        self.base = TransparentController(self.p.transparent)
        self.reset()

    def reset(self):
        self.k = np.full(4, self.p.k0)
        self.th_f = np.zeros(2)
        self.thd_f = np.zeros(2)
        self.phase = np.zeros(2)
        self.amp = np.array([0.3, 0.3])
        self.center = np.zeros(2)

    def _phase(self, obs):
        """Thigh phase portrait: phase = atan2(-thigh velocity / omega, thigh angle - centre)."""
        p = self.p
        w, x, y, z = obs["hip_quat"]
        phi = np.arctan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y))
        th = np.array([obs["q"][HIP_L], obs["q"][HIP_R]]) - phi
        thd = np.array([obs["qd"][HIP_L], obs["qd"][HIP_R]])
        a = 1 - np.exp(-2 * np.pi * p.phase_filter_hz * self.dt)
        self.th_f += a * (th - self.th_f)
        self.thd_f += a * (thd - self.thd_f)
        b = self.dt / 3.0                                # slow tracking of amplitude/centre of the thigh motion
        self.center += b * (self.th_f - self.center)
        self.amp += b * (np.abs(self.th_f - self.center) * 1.4 - self.amp)
        omega = 2 * np.pi * p.cadence_spm / 120.0
        ang = np.arctan2(-self.thd_f / omega, (self.th_f - self.center) / np.maximum(self.amp, 0.05))
        # 0 = maximum hip flexion (~ heel strike) -> map to template phase where hip flexion peaks (~0.87)
        self.phase = (ang / (2 * np.pi) + 0.87) % 1.0
        return self.phase

    def __call__(self, obs):
        p = self.p
        ph = self._phase(obs)
        q, qd = obs["q"], obs["qd"]
        rate = p.cadence_spm / 120.0
        q_ref = np.zeros(4)
        qd_ref = np.zeros(4)
        for k, (hip, knee) in enumerate(((HIP_L, KNEE_L), (HIP_R, KNEE_R))):
            q_ref[hip] = DEFAULT_GAIT.angle("hip_flexion", ph[k])
            qd_ref[hip] = DEFAULT_GAIT.velocity("hip_flexion", ph[k], rate)
            q_ref[knee] = DEFAULT_GAIT.angle("knee_angle", ph[k])
            qd_ref[knee] = DEFAULT_GAIT.velocity("knee_angle", ph[k], rate)
        e = q_ref - q
        # assist-as-needed adaptation: gain grows with |error|, decays otherwise
        self.k += self.dt * (p.adapt_up * np.abs(e) - p.forget * self.k)
        self.k = np.clip(self.k, 0.0, 1.0)
        K = self.k * np.array(p.k_max)
        tau = K * e + p.d_ratio * K * (qd_ref - qd) + self.base(obs)
        return np.clip(tau, -p.tau_max, p.tau_max)
