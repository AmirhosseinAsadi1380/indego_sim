"""Normative level-walking kinematics and moments vs gait cycle (0 = ipsilateral heel strike).

Values are key points read from the classic normative curves (Winter 2009,
"Biomechanics and Motor Control of Human Movement", ~1.3 m/s, natural cadence;
Perry & Burnfield 2010) and interpolated with periodic cubic splines. They are
approximations meant as a *template* for human intent and exo trajectories -
replace them with your own lab's motion-capture averages (``GaitLibrary.from_arrays``).

Sign conventions (match the model): hip flexion +, hip adduction +, knee flexion +,
ankle dorsiflexion +. Moments are *internal* joint moments in N*m/kg with the same
sign convention (e.g. negative knee moment = knee-extensor moment).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.interpolate import CubicSpline

_KEYS_DEG = {
    "hip_flexion": [(0, 30), (10, 27), (20, 18), (30, 8), (40, 0), (50, -8), (55, -10), (60, -6), (65, 3),
                    (70, 14), (75, 23), (80, 30), (85, 33), (90, 33), (95, 31.5), (100, 30)],
    "knee_angle": [(0, 4), (5, 10), (10, 16), (15, 19), (20, 16), (30, 8), (40, 4), (45, 5), (50, 10), (55, 20),
                   (60, 35), (65, 50), (70, 59), (73, 61), (78, 57), (85, 40), (90, 22), (95, 7), (100, 4)],
    "ankle_angle": [(0, 0), (5, -4), (10, -2), (20, 4), (30, 7.5), (40, 10), (45, 9.5), (50, 5), (55, -4),
                    (60, -14), (63, -17), (68, -12), (75, -4), (80, 0), (90, 1), (100, 0)],
    "hip_adduction": [(0, 1), (10, 4.5), (20, 6), (30, 5), (40, 3), (50, 1), (60, -3.5), (70, -5.5), (80, -3),
                      (90, 0), (100, 1)],
}
_KEYS_MOMENT = {  # N*m/kg internal
    "hip_flexion": [(0, -0.45), (8, -0.65), (15, -0.4), (25, -0.1), (35, 0.25), (45, 0.6), (52, 0.75), (57, 0.55),
                    (62, 0.25), (70, 0.08), (80, 0.0), (88, -0.2), (95, -0.4), (100, -0.45)],
    "knee_angle": [(0, 0.15), (4, -0.2), (10, -0.75), (15, -0.7), (22, -0.35), (30, 0.0), (40, 0.15), (45, 0.15),
                   (50, 0.0), (56, -0.18), (62, -0.08), (70, 0.0), (80, 0.1), (90, 0.25), (95, 0.2), (100, 0.15)],
    "ankle_angle": [(0, 0.03), (5, 0.1), (10, -0.05), (20, -0.45), (30, -0.85), (40, -1.25), (47, -1.5),
                    (52, -1.4), (57, -0.85), (62, -0.2), (66, 0.0), (90, 0.0), (100, 0.03)],
    "hip_adduction": [(0, 0.0), (8, -0.55), (15, -0.8), (25, -0.65), (35, -0.6), (45, -0.75), (52, -0.6),
                      (60, -0.1), (70, 0.0), (90, 0.0), (100, 0.0)],  # negative = abductor moment
}


def _spline(keys, scale=1.0):
    x = np.array([k[0] for k in keys], float) / 100.0
    y = np.array([k[1] for k in keys], float) * scale
    y[-1] = y[0]
    return CubicSpline(x, y, bc_type="periodic")


class GaitLibrary:
    def __init__(self, angles_deg=None, moments=None):
        a = angles_deg or _KEYS_DEG
        mo = moments or _KEYS_MOMENT
        self.q = {k: _spline(v, np.pi / 180) for k, v in a.items()}
        self.tau = {k: _spline(v) for k, v in mo.items()}

    @classmethod
    def from_arrays(cls, phase: np.ndarray, angles_rad: Dict[str, np.ndarray], moments: Dict[str, np.ndarray] = None):
        """Build from your own averaged gait cycle (phase in [0,1], arrays same length, periodic)."""
        g = cls.__new__(cls)
        g.q = {k: CubicSpline(phase, np.r_[v[:-1], v[0]], bc_type="periodic") for k, v in angles_rad.items()}
        g.tau = {k: CubicSpline(phase, np.r_[v[:-1], v[0]], bc_type="periodic") for k, v in (moments or {}).items()}
        return g

    def angle(self, joint, phase, amp=1.0):
        return amp * self.q[joint](phase % 1.0)

    def velocity(self, joint, phase, phase_rate, amp=1.0):
        return amp * self.q[joint](phase % 1.0, 1) * phase_rate

    def moment(self, joint, phase):
        return self.tau[joint](phase % 1.0) if joint in self.tau else 0.0

    @staticmethod
    def stance(phase, toe_off=0.62):
        return (phase % 1.0) < toe_off


DEFAULT_GAIT = GaitLibrary()
