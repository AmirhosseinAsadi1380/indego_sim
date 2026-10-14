"""Calibrate the spinal/spasticity model of a participant with the Wartenberg pendulum test.

Clinical test: seated, thigh supported horizontally, the examiner lifts the shank to full knee
extension and releases it. In a relaxed healthy person the shank swings 5-6 times; with quadriceps
spasticity the first swing is cut short (catch) and the motion dies quickly. Common metrics:
    first-swing excursion  A1 = initial angle - first minimum (deg of knee flexion reached)
    relaxation index       R2 = A1 / A0, A0 = initial angle - final resting angle   (healthy ~1.6+)
    number of oscillations, time to rest

Measure it with an electrogoniometer/IMU on the shank (or the Indego thigh/shank encoders with the
exo doffed of its motors), then call `fit_pendulum` to fit the TSRT threshold / velocity sensitivity /
gain of the knee extensors (and optionally the passive stiffness scale) of that participant.

Everything here runs the same muscle + reflex code as the walking simulator (no exo attached).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import optimize

from ..config import SubjectParams
from ..human.muscles import MuscleModel
from ..human.reflexes import SpinalCord
from ..human.sci import apply_to_model, muscle_capacity
from ..model.builder import build_model


@dataclass
class PendulumResult:
    t: np.ndarray
    knee_deg: np.ndarray              # knee flexion (+), 0 = fully extended
    quad_excitation: np.ndarray       # mean reflex excitation of the knee extensors
    metrics: dict


def _seated_model(subject: SubjectParams, side: str):
    m, info, spec = build_model(subject=subject, with_exo=False, base="fixed", lift=0.6, return_spec=True)
    other = "l" if side == "r" else "r"
    fixes = {f"hip_flexion_{side}": np.radians(90), f"hip_flexion_{other}": np.radians(90),
             f"hip_adduction_{side}": 0.0, f"hip_adduction_{other}": 0.0,
             f"hip_rotation_{side}": 0.0, f"hip_rotation_{other}": 0.0,
             f"knee_angle_{other}": np.radians(90), "lumbar_flexion": 0.0, "lumbar_bending": 0.0,
             "lumbar_rotation": 0.0}
    for j, v in fixes.items():
        if spec.joint(j) is None:
            continue
        e = spec.add_equality()
        e.type = mujoco.mjtEq.mjEQ_JOINT
        e.name1 = j
        e.data[:5] = [v, 0, 0, 0, 0]
    m = spec.compile()
    return m, info


def pendulum_test(subject: SubjectParams, side: str = "r", T: float = 8.0, start_knee_deg: float = 0.0,
                  overrides: dict = None, seed: int = 0, model=None, dt_tick: float = 0.001) -> PendulumResult:
    """Simulate the pendulum test. `overrides`: {"lam": x, "mu": x, "gain": x, "tonic": x, "passive": x}
    applied to the knee extensors of `side` (TSRT threshold, velocity term, gain, tonic drive, passive
    stiffness scale) - the parameters `fit_pendulum` estimates."""
    subject = copy.deepcopy(subject)
    m, info = model if model is not None else _seated_model(subject, side)
    m = copy.copy(m)
    d = mujoco.MjData(m)
    mm = MuscleModel(m, info.muscles)
    cap = muscle_capacity(mm, subject)
    apply_to_model(m, mm, subject, cap)
    rng = np.random.default_rng(seed)
    sc = SpinalCord(mm, subject, dt_tick, rng)
    sc.spasm_rate = 0.0
    sc.spasm_trigger = 0.0
    quad = [i for i in mm.group_indices("knee_extensors") if mm.side[i] == side]
    rf = [i for i in range(mm.n) if mm.base[i] == "recfem" and mm.side[i] == side]
    quad = sorted(set(quad + rf))
    ov = overrides or {}
    if "lam" in ov:
        sc.lam[quad] = ov["lam"]
    if "mu" in ov:
        sc.mu[quad] = ov["mu"]
    if "gain" in ov:
        sc.k_sp[quad] = ov["gain"]
    if "tonic" in ov:
        sc.tonic[quad] = ov["tonic"]
    if "passive" in ov:
        for i in quad:
            a = mm.ids[i]
            m.actuator_biasprm[a, 7] *= ov["passive"]
            m.actuator_gainprm[a, 7] *= ov["passive"]
    # initial posture: seated, knee extended to start angle
    from ..model.utils import apply_couplings
    q = m.qpos0.copy()
    for j, v in {f"hip_flexion_{side}": 90, f"hip_flexion_{'l' if side == 'r' else 'r'}": 90,
                 f"knee_angle_{'l' if side == 'r' else 'r'}": 90, f"knee_angle_{side}": start_knee_deg}.items():
        q[m.jnt_qposadr[m.joint(j).id]] = np.radians(v)
    d.qpos[:] = apply_couplings(m, q)
    mujoco.mj_forward(m, d)
    sc.reset(sc.relative_length(d), np.zeros(mm.n), mm.forces(d) / np.abs(mm.F0))
    kadr = m.jnt_qposadr[m.joint(f"knee_angle_{side}").id]
    nphys = max(1, int(round(dt_tick / m.opt.timestep)))
    n = int(T / dt_tick)
    t = np.zeros(n)
    kn = np.zeros(n)
    qe = np.zeros(n)
    for k in range(n):
        e = sc.step(d.time, d)
        d.ctrl[mm.ids] = e                      # complete SCI: reflex drive only, no voluntary activation
        mujoco.mj_step(m, d, nstep=nphys)
        t[k], kn[k], qe[k] = d.time, np.degrees(d.qpos[kadr]), e[quad].mean()
    return PendulumResult(t=t, knee_deg=kn, quad_excitation=qe, metrics=pendulum_metrics(t, kn))


def pendulum_metrics(t, knee_deg, start=None) -> dict:
    """Wartenberg metrics from a knee-flexion trace (deg, + flexion)."""
    k = np.asarray(knee_deg)
    start = k[0] if start is None else start
    rest = float(np.mean(k[int(0.85 * len(k)):]))
    # first swing: first local maximum of flexion
    dk = np.diff(k)
    peaks = np.where((dk[:-1] > 0) & (dk[1:] <= 0))[0] + 1
    valleys = np.where((dk[:-1] < 0) & (dk[1:] >= 0))[0] + 1
    first = int(peaks[0]) if len(peaks) else int(np.argmax(k))
    A1 = float(k[first] - start)
    A0 = float(rest - start)
    big = [p for p in peaks if k[p] - rest > 3.0]
    return dict(first_swing_deg=A1, rest_deg=rest, R2=(A1 / A0 if abs(A0) > 1e-6 else np.nan),
                n_oscillations=len(big), first_peak_time=float(t[first] - t[0]),
                first_return_deg=float(k[valleys[valleys > first][0]]) if np.any(valleys > first) else np.nan)


def fit_pendulum(t_meas, knee_meas_deg, subject: SubjectParams, side: str = "r",
                 params=("lam", "mu", "gain"), x0=None, bounds=None, max_iter: int = 60, grid: int = 5,
                 verbose: bool = False):
    """Fit knee-extensor TSRT parameters so the simulated pendulum trace matches a measured one
    (Nelder-Mead on the RMS angle error over the measured window). Returns (best overrides, result)."""
    t_meas = np.asarray(t_meas) - t_meas[0]
    knee_meas_deg = np.asarray(knee_meas_deg)
    T = float(t_meas[-1])
    model = _seated_model(copy.deepcopy(subject), side)
    default = {"lam": 0.95, "mu": 0.15, "gain": 2.0, "tonic": 0.0, "passive": 1.0}
    bnds = {"lam": (0.3, 1.5), "mu": (0.0, 0.6), "gain": (0.0, 8.0), "tonic": (0.0, 0.1), "passive": (0.3, 3.0)}
    bnds.update(bounds or {})
    x0 = np.array([default[p] if x0 is None else x0[i] for i, p in enumerate(params)])
    lo = np.array([bnds[p][0] for p in params])
    hi = np.array([bnds[p][1] for p in params])

    def cost(x):
        x = np.clip(x, lo, hi)
        r = pendulum_test(subject, side, T=T, start_knee_deg=float(knee_meas_deg[0]),
                          overrides=dict(zip(params, x)), model=model)
        sim = np.interp(t_meas, r.t - r.t[0], r.knee_deg)
        c = float(np.sqrt(np.mean((sim - knee_meas_deg) ** 2)))
        if verbose:
            print(np.round(x, 3), round(c, 2))
        return c
    # coarse grid first (the cost is piecewise flat in places because the reflex has a threshold), then
    # Nelder-Mead from the best grid point with a simplex spanning ~15 % of each bound
    if grid > 1:
        axes = [np.linspace(lo[i] + 0.1 * (hi[i] - lo[i]), hi[i] - 0.1 * (hi[i] - lo[i]), grid) for i in range(len(params))]
        best_c, best_x = np.inf, x0
        for xg in np.array(np.meshgrid(*axes)).reshape(len(params), -1).T:
            c = cost(xg)
            if c < best_c:
                best_c, best_x = c, xg
        x0 = best_x
    simplex = [x0] + [x0 + np.eye(len(params))[i] * 0.15 * (hi - lo)[i] for i in range(len(params))]
    sol = optimize.minimize(cost, x0, method="Nelder-Mead",
                            options=dict(maxiter=max_iter, xatol=1e-3, fatol=0.05, initial_simplex=np.array(simplex)))
    best = dict(zip(params, np.clip(sol.x, lo, hi)))
    res = pendulum_test(subject, side, T=T, start_knee_deg=float(knee_meas_deg[0]), overrides=best, model=model)
    return best, res
