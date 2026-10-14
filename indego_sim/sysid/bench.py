"""Virtual test bench: one Indego joint drive (same MuJoCo `dcmotor` model, friction, delays and
sensors as in the full simulator) driving a rigid link, optionally clamped.

Use it to
* rehearse an identification protocol before taking the real exo to the bench,
* check that the fitting code in `actuator_id.py` recovers known parameters (tests/),
* see how much each non-ideality (delay, Coulomb friction, quantisation, current noise)
  biases a given identification method.

The link stands in for an exo segment (or exo segment + passive leg) hanging under gravity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..config import MotorParams, SensorParams
from ..model.builder import _add_dcmotor, _sensor


@dataclass
class LinkParams:
    mass: float = 3.3               # kg (Indego thigh segment incl. knee actuator) [est]
    com: float = 0.20               # m from the joint axis
    inertia_com: float = 0.045      # kg m^2 about the COM (sagittal axis)
    gravity: bool = True


@dataclass
class BenchResult:
    t: np.ndarray
    cmd: np.ndarray                 # commanded joint torque [N*m]
    q: np.ndarray                   # measured joint angle [rad] (quantised, noisy, delayed)
    qd: np.ndarray                  # firmware velocity estimate [rad/s]
    current: np.ndarray             # measured motor current [A]
    q_true: np.ndarray
    qd_true: np.ndarray
    tau_motor_joint: np.ndarray     # true drive torque at the joint (after limits, before joint friction) [N*m]
    temp: np.ndarray                # winding temperature [degC]
    constraint: np.ndarray = field(default_factory=lambda: np.zeros(0))   # clamp reaction (locked bench) [N*m]


def make_bench(mp: MotorParams = None, link: LinkParams = None, locked: bool = False, timestep: float = 0.0005,
               control_rate_hz: float = 200.0, sensors: SensorParams = None, q0: float = 0.0):
    """Return (model, data) of the single-joint bench. Joint angle 0 = link hanging straight down."""
    mp = mp or MotorParams()
    link = link or LinkParams()
    sp = sensors or SensorParams()
    spec = mujoco.MjSpec()
    spec.modelname = "indego_joint_bench"
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER if mp.friction_model == "lugre" \
        else mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if not link.gravity:
        spec.option.gravity = [0.0, 0.0, 0.0]
    b = spec.worldbody.add_body()
    b.name = "link"
    b.pos = [0.0, 0.0, 1.0]
    b.explicitinertial = True
    b.mass = link.mass
    b.ipos = [0.0, 0.0, -link.com]
    b.inertia = [link.inertia_com, link.inertia_com, 0.1 * link.inertia_com]
    j = b.add_joint()
    j.name = "joint"
    j.type = mujoco.mjtJoint.mjJNT_HINGE
    j.axis = [0.0, -1.0, 0.0]                         # + = link swings forward (+x), like hip/knee flexion
    if mp.friction_model == "coulomb":
        j.frictionloss = mp.coulomb * mp.gear
    g = b.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.fromto = [0, 0, 0, 0, 0, -2 * link.com]
    g.size = [0.03, 0, 0]
    g.contype = g.conaffinity = 0
    g.mass = 0.0
    g.density = 0.0
    if locked:
        e = spec.add_equality()
        e.type = mujoco.mjtEq.mjEQ_JOINT
        e.name1 = "joint"
        e.data[:5] = [q0, 0, 0, 0, 0]
        e.solref = [2 * timestep, 1.0]
    _add_dcmotor(spec, "act", "joint", mp, mp.gear, timestep)
    per, dl = 1.0 / control_rate_hz, sp.sensor_delay
    T, O = mujoco.mjtSensor, mujoco.mjtObj
    _sensor(spec, "enc", T.mjSENS_JOINTPOS, O.mjOBJ_JOINT, "joint", per, dl, timestep)
    _sensor(spec, "vel", T.mjSENS_JOINTVEL, O.mjOBJ_JOINT, "joint", per, dl, timestep)
    _sensor(spec, "trq", T.mjSENS_ACTUATORFRC, O.mjOBJ_ACTUATOR, "act", per, dl, timestep)
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[0] = q0
    mujoco.mj_forward(m, d)
    return m, d


def simulate(m, d, u: np.ndarray, mp: MotorParams = None, fs: float = 200.0, mode: str = "torque",
             kp: float = 150.0, kd: float = 4.0, sensors: SensorParams = None, seed: int = 0) -> BenchResult:
    """Drive the bench with `u` sampled at `fs`.

    mode="torque": u = joint torque command [N*m] (what you send in torque/current mode)
    mode="position": u = joint angle reference [rad] tracked by an on-board-style PD at `fs`
                     (constant-velocity sweeps for the friction map)
    """
    mp = mp or MotorParams()
    sp = sensors or SensorParams()
    rng = np.random.default_rng(seed)
    nphys = int(round((1.0 / fs) / m.opt.timestep))
    a_id = m.actuator("act").id
    s_enc, s_vel, s_trq = (m.sensor(n).adr[0] for n in ("enc", "vel", "trq"))
    res = 2 * np.pi / 2 ** sp.enc_bits
    act_T = m.actuator_actadr[a_id] if (mp.thermal_R > 0 or mp.thermal_C > 0) else None
    lim = mp.K * mp.i_peak * mp.gear
    n = len(u)
    out = {k: np.zeros(n) for k in ("t", "cmd", "q", "qd", "current", "q_true", "qd_true", "tau", "temp", "cons")}
    q_prev, qd_f = None, 0.0
    a_f = 1 - np.exp(-2 * np.pi * sp.vel_filter_hz / fs)
    for k in range(n):
        sd = d.sensordata
        qm = np.round(sd[s_enc] / res) * res + rng.normal(0, np.radians(sp.enc_noise_deg))
        raw = 0.0 if q_prev is None else (qm - q_prev) / (1.0 / fs)
        qd_f += a_f * (raw - qd_f)
        q_prev = qm
        i_meas = sd[s_trq] / mp.K + rng.normal(0, sp.current_noise)
        if mode == "torque":
            cmd = float(np.clip(u[k], -lim, lim))
        else:
            ref_v = (u[min(k + 1, n - 1)] - u[k]) * fs
            cmd = float(np.clip(kp * (u[k] - qm) + kd * (ref_v - qd_f), -lim, lim))
        d.ctrl[a_id] = cmd / mp.gear
        out["t"][k], out["cmd"][k], out["q"][k], out["qd"][k], out["current"][k] = d.time, cmd, qm, qd_f, i_meas
        out["q_true"][k], out["qd_true"][k] = d.qpos[0], d.qvel[0]
        out["tau"][k] = d.qfrc_actuator[0]
        out["temp"][k] = mp.ambient_C + (d.act[act_T] if act_T is not None else 0.0)
        out["cons"][k] = d.qfrc_constraint[0]
        mujoco.mj_step(m, d, nstep=nphys)
    return BenchResult(t=out["t"], cmd=out["cmd"], q=out["q"], qd=out["qd"], current=out["current"],
                       q_true=out["q_true"], qd_true=out["qd_true"], tau_motor_joint=out["tau"], temp=out["temp"],
                       constraint=out["cons"])


def true_parameters(mp: MotorParams, link: LinkParams) -> dict:
    """Joint-side parameters an identification of this bench should recover."""
    N = mp.gear
    return dict(
        inertia=link.inertia_com + link.mass * link.com ** 2 + mp.rotor_inertia * N ** 2,
        viscous=mp.viscous * N ** 2,
        coulomb=mp.coulomb * N,
        mgl=link.mass * 9.81 * link.com if link.gravity else 0.0,
        torque_constant=mp.K,
        delay=mp.cmd_delay,
        rotor_inertia_reflected=mp.rotor_inertia * N ** 2,
    )
