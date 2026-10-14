"""Muscle-level utilities for the MyoLeg muscles (MuJoCo Hill-type muscles, rigid tendon).

* normalised fibre length / velocity (used by spinal reflexes)
* force-generating gain and passive bias at the current state
* effective moment arms on the *independent* coordinates (MyoLeg couples the knee
  translations/patella to knee_angle with polynomials; we fold those in)
* static optimisation: desired joint torques -> muscle activations with per-muscle
  capacity bounds (paralysis / weakness) via bounded least squares
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from ..config import MUSCLE_GROUPS
from ..model.utils import coupling_derivatives

# independent human DOFs driven by leg muscles (per side)
LEG_DOFS = ["hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle", "subtalar_angle", "mtp_angle"]


class MuscleModel:
    def __init__(self, m: mujoco.MjModel, muscle_names: Sequence[str]):
        self.m = m
        self.names = list(muscle_names)
        self.ids = np.array([m.actuator(n).id for n in self.names])
        self.n = len(self.ids)
        gp = m.actuator_gainprm[self.ids]
        self.range0, self.range1 = gp[:, 0], gp[:, 1]
        self.F0 = gp[:, 2].copy()
        self.vmax = gp[:, 6]
        self.lr = m.actuator_lengthrange[self.ids].copy()
        self.L0 = (self.lr[:, 1] - self.lr[:, 0]) / (self.range1 - self.range0)
        self.side = np.array([n[-1] for n in self.names])
        self.base = [n[:-2] for n in self.names]
        self.group_of = {}
        for g, members in MUSCLE_GROUPS.items():
            for i, b in enumerate(self.base):
                if b in members:
                    self.group_of.setdefault(i, []).append(g)
        self.dof_names = [f"{j}_{s}" for s in ("l", "r") for j in LEG_DOFS]
        self.dofs = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.dof_names])
        self._dense = np.zeros((m.nu, m.nv))
        self.rom_lo, self.rom_hi = self._rom_length_range(m)
        self.solver = "fista"
        self.iters = 150

    def _rom_length_range(self, m, n: int = 400, seed: int = 0):
        """Muscle-tendon length range over the anatomical joint ROM (what a clinician moves through when
        grading spasticity), by sampling leg-joint configurations biased toward the limits."""
        from ..model.utils import apply_couplings
        d = mujoco.MjData(m)
        rng = np.random.default_rng(seed)
        jids = [m.joint(nm).id for nm in self.dof_names]
        qa = np.array([m.jnt_qposadr[j] for j in jids])
        lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -1.0 for j in jids])
        hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else 1.0 for j in jids])
        Lmin = np.full(self.n, np.inf)
        Lmax = np.full(self.n, -np.inf)
        base_q = m.qpos0.copy()
        for k in range(n):
            u = rng.random(len(jids))
            pick = rng.random(len(jids))
            u = np.where(pick < 0.3, 0.0, np.where(pick < 0.6, 1.0, u))
            q = base_q.copy()
            q[qa] = lo + u * (hi - lo)
            d.qpos[:] = apply_couplings(m, q)
            mujoco.mj_fwdPosition(m, d)
            L = d.actuator_length[self.ids]
            Lmin = np.minimum(Lmin, L)
            Lmax = np.maximum(Lmax, L)
        return Lmin, np.maximum(Lmax, Lmin + 1e-6)

    def rom_fraction(self, d):
        """Position of each muscle within its length range over the joint ROM (0 = shortest, 1 = longest)."""
        return (d.actuator_length[self.ids] - self.rom_lo) / (self.rom_hi - self.rom_lo)

    # -- state ---------------------------------------------------------------------------
    def norm_length(self, d):
        L = d.actuator_length[self.ids]
        return self.range0 + (L - self.lr[:, 0]) / (self.lr[:, 1] - self.lr[:, 0]) * (self.range1 - self.range0)

    def norm_velocity(self, d):
        """Lengthening positive, in optimal-fibre-lengths per second."""
        return d.actuator_velocity[self.ids] / self.L0

    def forces(self, d):
        """Tension (positive N)."""
        return -d.actuator_force[self.ids]

    def gain_bias(self, d):
        """Vectorised MuJoCo muscle gain/bias (same formulas as mju_muscleGain/Bias)."""
        gp = self.m.actuator_gainprm[self.ids]
        lmin, lmax, fpmax, fvmax = gp[:, 4], gp[:, 5], gp[:, 7], gp[:, 8]
        L = self.norm_length(d)
        V = d.actuator_velocity[self.ids] / (self.L0 * self.vmax)
        a = 0.5 * (lmin + 1)
        b = 0.5 * (1 + lmax)
        FL = np.zeros(self.n)
        e = 1e-15
        m1 = (L >= lmin) & (L <= a)
        x = (L - lmin) / np.maximum(e, a - lmin); FL[m1] = 0.5 * x[m1] ** 2
        m2 = (L > a) & (L <= 1)
        x = (1 - L) / np.maximum(e, 1 - a); FL[m2] = 1 - 0.5 * x[m2] ** 2
        m3 = (L > 1) & (L <= b)
        x = (L - 1) / np.maximum(e, b - 1); FL[m3] = 1 - 0.5 * x[m3] ** 2
        m4 = (L > b) & (L <= lmax)
        x = (lmax - L) / np.maximum(e, lmax - b); FL[m4] = 0.5 * x[m4] ** 2
        y = fvmax - 1
        FV = np.where(V <= -1, 0.0, np.where(V <= 0, (V + 1) ** 2,
                      np.where(V <= y, fvmax - (y - V) ** 2 / np.maximum(e, y), fvmax)))
        F0 = self.F0
        x1 = (L - 1) / np.maximum(e, b - 1)
        FP = np.where(L <= 1, 0.0, np.where(L <= b, 0.5 * fpmax * x1 ** 2, fpmax * (0.5 + (L - b) / np.maximum(e, b - 1))))
        return -F0 * FL * FV, -F0 * FP  # force = g*act + b  (negative = tension)

    def gain_bias_reference(self, d):
        m = self.m
        g = np.empty(self.n)
        b = np.empty(self.n)
        for k, i in enumerate(self.ids):
            g[k] = mujoco.mju_muscleGain(d.actuator_length[i], d.actuator_velocity[i], m.actuator_lengthrange[i],
                                         m.actuator_acc0[i], m.actuator_gainprm[i][:9])
            b[k] = mujoco.mju_muscleBias(d.actuator_length[i], m.actuator_lengthrange[i], m.actuator_acc0[i],
                                         m.actuator_biasprm[i][:9])
        return g, b

    def moment_arms(self, d):
        """Effective generalized-force map R (n_dof x n_muscle) on LEG_DOFS: qfrc = R @ force."""
        m = self.m
        mujoco.mju_sparse2dense(self._dense, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
        Mfull = self._dense[self.ids].T.copy()          # nv x n_muscle
        for cdof, (ddof, deriv) in coupling_derivatives(m, d.qpos).items():
            Mfull[ddof] += deriv * Mfull[cdof]
        return Mfull[self.dofs]

    # -- static optimisation ---------------------------------------------------------------
    def static_optimisation(self, d, tau_des, cap, weights=None, w_act=1e-4, a_prev=None, w_smooth=0.0):
        """Activations a in [0, cap] minimising |R(g a + b) - tau|^2_W + w_act |a/cap|^2 (+ smoothness)."""
        R = self.moment_arms(d)
        g, b = self.gain_bias(d)
        A = R * g[None, :]
        rhs = tau_des - R @ b
        W = np.ones(len(tau_des)) if weights is None else np.asarray(weights)
        rows = [A * W[:, None]]
        rv = [rhs * W]
        reg = np.sqrt(w_act) * np.diag(1.0 / np.maximum(cap, 0.05)) * np.sqrt(np.abs(self.F0).mean())
        rows.append(reg)
        rv.append(np.zeros(self.n))
        if w_smooth > 0 and a_prev is not None:
            rows.append(np.sqrt(w_smooth) * np.eye(self.n) * np.sqrt(np.abs(self.F0).mean()))
            rv.append(np.sqrt(w_smooth) * a_prev * np.sqrt(np.abs(self.F0).mean()))
        ub = np.maximum(cap, 1e-6)
        Aall, ball = np.vstack(rows), np.concatenate(rv)
        if self.solver == "bvls":
            res = lsq_linear(Aall, ball, bounds=(np.zeros(self.n), ub), method="bvls", max_iter=200)
            return np.clip(res.x, 0, ub)
        # FISTA projected gradient on the normal equations, warm-started (fast, ~0.3 ms)
        H = Aall.T @ Aall
        f = Aall.T @ ball
        Lc = np.linalg.norm(H, 2)
        x = np.clip(a_prev if a_prev is not None else np.zeros(self.n), 0, ub)
        y, t = x.copy(), 1.0
        for _ in range(self.iters):
            xn = np.clip(y - (H @ y - f) / Lc, 0, ub)
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            y = xn + ((t - 1) / tn) * (xn - x)
            x, t = xn, tn
        return x

    def group_indices(self, group: str, side: str = None) -> List[int]:
        return [i for i, gs in self.group_of.items() if group in gs and (side is None or self.side[i] == side)]
