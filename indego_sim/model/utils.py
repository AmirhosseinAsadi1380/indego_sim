"""Small MuJoCo helpers shared by the runtime modules."""
from __future__ import annotations

import mujoco
import numpy as np


def apply_couplings(m: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Set every joint-coupled coordinate (MyoLeg knee/patella polynomials) consistent with its driver."""
    q = qpos
    for i in range(m.neq):
        if m.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT or m.eq_obj2id[i] < 0:
            continue
        j1, j2 = m.eq_obj1id[i], m.eq_obj2id[i]
        x = q[m.jnt_qposadr[j2]] - m.qpos0[m.jnt_qposadr[j2]]
        c = m.eq_data[i][:5]
        q[m.jnt_qposadr[j1]] = m.qpos0[m.jnt_qposadr[j1]] + c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4
    return q


def coupling_derivatives(m: mujoco.MjModel, qpos: np.ndarray) -> dict:
    """d q_coupled / d q_driver for every joint-coupling equality: {coupled_dof: (driver_dof, deriv)}."""
    out = {}
    for i in range(m.neq):
        if m.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT or m.eq_obj2id[i] < 0:
            continue
        j1, j2 = m.eq_obj1id[i], m.eq_obj2id[i]
        x = qpos[m.jnt_qposadr[j2]] - m.qpos0[m.jnt_qposadr[j2]]
        c = m.eq_data[i][:5]
        out[m.jnt_dofadr[j1]] = (m.jnt_dofadr[j2], c[1] + 2 * c[2] * x + 3 * c[3] * x ** 2 + 4 * c[4] * x ** 3)
    return out


def joint_qadr(m, name):
    return m.jnt_qposadr[m.joint(name).id]


def joint_dadr(m, name):
    return m.jnt_dofadr[m.joint(name).id]


def quat_to_euler_zyx(q):
    """(w,x,y,z) -> roll (x), pitch (y), yaw (z)."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def body_tilt(xmat):
    """Pitch (+ forward) and roll (+ to the left... i.e. top moves +y) of a body's z-axis vs vertical."""
    z = np.asarray(xmat).reshape(3, 3)[:, 2]
    return np.arctan2(z[0], z[2]), np.arctan2(z[1], z[2])
