"""Upper-limb support: forearm crutches / walker / harness.

People with SCI walk in the Indego with crutches or a walker; the arms carry balance and
part of the body weight. The arms themselves are not simulated. Each crutch is a
massless strut between a ground tip P and a hand point H rigidly attached to the trunk,
so it can only *push* along its own axis:

    F_i = f_i * (H_i - P_i) / |H_i - P_i|,   0 <= f_i <= f_max

The person chooses where to plant the tips (a stance tripod ~35 cm ahead of the feet;
during 4-point gait each crutch is lifted and re-planted ahead in early swing of the
same-side leg) and how hard to push (f_i), using a delayed postural controller on the
whole-body COM and trunk tilt. The force direction therefore comes from geometry, not
from a free "stabilising wrench": crutches ahead can stop a forward fall but cannot
pull a person who sits back, which matches clinical experience.

Loads are logged per arm and can be compared directly with instrumented crutches.
Tune the gains and tip geometry with your own crutch-load data (docs/CALIBRATION.md).
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from ..config import SupportParams


def _bounded_ls2(A, b, f_pref, lam, ub):
    """min |A f - b|^2 + lam |f - f_pref|^2  s.t. 0 <= f <= ub, for 2 variables (exact active-set search)."""
    H = A.T @ A + lam * np.eye(2)
    g = A.T @ b + lam * f_pref

    def cost(f):
        return float(np.sum((A @ f - b) ** 2) + lam * np.sum((f - f_pref) ** 2))
    try:
        f = np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        f = f_pref.copy()
    if np.all(f >= 0) and np.all(f <= ub):
        return f
    best, bc = None, np.inf
    for i in range(2):
        j = 1 - i
        for fi in (0.0, ub[i]):
            fj = (g[j] - H[j, i] * fi) / H[j, j]
            fj = min(max(fj, 0.0), ub[j])
            cand = np.zeros(2)
            cand[i], cand[j] = fi, fj
            cst = cost(cand)
            if cst < bc:
                best, bc = cand, cst
    return best


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _skew(r):
    """Matrix S with S @ F = r x F."""
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


class UpperLimbSupport:
    def __init__(self, m: mujoco.MjModel, sp: SupportParams, body_weight: float, dt: float, scale: float = 1.0):
        self.m, self.sp, self.dt = m, sp, dt
        self.BW = body_weight
        self.torso = m.body("torso").id
        self.arm = m.body("Arm_attachment").id
        self.root = m.body("Full Body").id
        self.feet = {s: (m.body(f"calcn_{s}").id, m.body(f"toes_{s}").id) for s in "lr"}
        self.s = scale
        # load-transfer points in the (reference-aligned) trunk frame relative to Arm_attachment. With the
        # elbow nearly extended the crutch force line runs through the shoulder, so we apply it there.
        self.hand_local = {"l": np.array([0.0, 0.19, 0.0]) * scale, "r": np.array([0.0, -0.19, 0.0]) * scale}
        self.tip_ahead = sp.tip_ahead * scale
        self.tip_lateral = sp.tip_lateral * scale
        self.nd = max(1, int(round(sp.reaction_delay / dt)))
        self.F = np.zeros((2, 3))
        self.f = np.zeros(2)
        self.side_capacity = {"l": 1.0, "r": 1.0}
        self.target_pitch = 0.0
        self.target_speed = 0.0
        self.lateral_bias = 0.0
        self.com_shift = 0.0
        self.x_ref = None
        self._brake = True
        self.single_support = 0.0          # 0..1 set by the simulator from the person's foot-load sensing
        self.foot_load = {"l": 0.0, "r": 0.0}   # vertical GRF per foot (N), set by the simulator
        self.walk_intent = 0.0                  # 0..1 the person intends to walk (pushes harder on the crutches)
        self.expected_stance = None             # "l"/"r" while the person anticipates / feels a step
        self.anticipation = 0.0                 # 0..1 weighting of the expected stance foot
        self.mass = body_weight / 9.81
        self.enabled = sp.kind != "none"
        self.tips = {"l": np.zeros(3), "r": np.zeros(3)}
        self.lifted = {"l": False, "r": False}
        self.cop = {"l": None, "r": None}          # per-foot centre of pressure, set by the simulator
        self.moving_crutches = False
        self.lean_frac = 0.0                       # 0..1 how far into the step-request lean the person is
        self.pending_crutch = None                 # crutch about to be / being moved (set by the simulator)
        self.extra_lean = 0.0                      # m extra forward lean while waiting for the exo to step

    # --------------------------------------------------------------------------------------
    def reset(self, d):
        self.R_torso0 = d.xmat[self.torso].reshape(3, 3).copy()
        self.h_nom = d.subtree_com[self.root][2]
        self.buf = deque(maxlen=self.nd + 2)
        s = self._state(d)
        for _ in range(self.nd + 2):
            self.buf.append(s)
        self.F[:] = 0.0
        self.f[:] = 0.0
        self.extra_lean = 0.0
        self.pending_crutch = None
        self.trolley = None
        self.L0 = [None, None]
        self.L_prev = [0.0, 0.0]
        self.perp = np.zeros((2, 3))
        for side in "lr":
            self.plant(d, side)
        self.f_base = self.sp.weight_bearing_frac * self.BW / 2

    def _trunk_R(self, d):
        return d.xmat[self.torso].reshape(3, 3) @ self.R_torso0.T

    def hand(self, d, side):
        return d.xpos[self.arm] + self._trunk_R(d) @ self.hand_local[side]

    def plant(self, d, side, ahead=None):
        """Put the crutch tip down ahead of the body (called at reset and at the end of a crutch swing)."""
        com = d.subtree_com[self.root]
        fx = 0.5 * (d.xpos[self.feet[side][0]][0] + d.xpos[self.feet[side][1]][0])
        # within reach: ahead of the body (a crutch planted far ahead of the shoulder can only push the person
        # back), and at least level with the front of that side's foot
        x = max(com[0] + (self.tip_ahead if ahead is None else ahead), fx + 0.09 * self.s)
        y = com[1] + (self.tip_lateral if side == "l" else -self.tip_lateral)
        self.tips[side] = np.array([x, y, 0.0])
        if hasattr(self, "L0"):
            self.L0["lr".index(side)] = None          # new contact: the arm length is re-set at touch-down

    def _state(self, d):
        R = self._trunk_R(d)
        z, x = R[:, 2], R[:, 0]
        com = d.subtree_com[self.root].copy()
        foot = {s: 0.5 * (d.xpos[self.feet[s][0]] + d.xpos[self.feet[s][1]]) for s in "lr"}
        bos = 0.5 * (foot["l"] + foot["r"])
        # where each foot actually bears load (centre of pressure); the foot centre when unloaded
        cop = {s: (self.cop[s].copy() if self.cop.get(s) is not None else foot[s].copy()) for s in "lr"}
        return dict(pitch=np.arctan2(z[0], z[2]), roll=np.arctan2(z[1], z[2]), yaw=np.arctan2(x[1], x[0]),
                    com=com, bos=bos, foot=foot, cop=cop, load=dict(self.foot_load))

    # --------------------------------------------------------------------------------------
    def step(self, d):
        s_now = self._state(d)
        self.buf.append(s_now)
        if not self.enabled:
            return self.F
        sp = self.sp
        s, s1 = self.buf[0], self.buf[1]                     # delayed perception (and one tick later)
        vcom = (s1["com"] - s["com"]) / self.dt
        rate_p = (s1["pitch"] - s["pitch"]) / self.dt
        rate_r = (s1["roll"] - s["roll"]) / self.dt
        if sp.kind == "harness":
            # overhead body-weight-support harness on a robotic trolley (ZeroG-style): two straps from the vest
            # (at the shoulders) to a spreader bar `harness_height` above them; the trolley follows the person
            # horizontally with a lag, so the straps tilt (pendulum restoring force) only when the trunk moves
            # faster than the trolley or tilts. Vertical unloading = weight_bearing_frac*BW + safety spring.
            c_now = s_now["com"]
            if self.trolley is None:
                self.trolley = c_now[:2].copy()
            self.trolley += self.dt / sp.trolley_lag * (c_now[:2] - self.trolley)
            fz = sp.weight_bearing_frac * self.BW + sp.k_height * max(0.0, self.h_nom - 0.02 - c_now[2]) \
                - sp.d_height * min(0.0, s_now["com"][2] - s["com"][2]) / max(self.nd * self.dt, self.dt)
            fz = float(np.clip(fz, 0.0, 2 * sp.max_load_frac * self.BW))
            for k, side in enumerate("lr"):
                A = self.hand(d, side)
                top = np.array([self.trolley[0], self.trolley[1] + (0.2 if side == "l" else -0.2) * self.s,
                                A[2] + sp.harness_height])
                u = top - A
                u /= max(np.linalg.norm(u), 1e-6)
                self.F[k] = (0.5 * fz / max(u[2], 0.3)) * u          # strap tension giving fz/2 vertical
            self.f = np.linalg.norm(self.F, axis=1)
            return self.F

        # ---- whole-body balance + trunk orientation through the crutches -----------------------
        # pivot = load-weighted centre of the feet the person feels loaded (delayed perception)
        fl, fr = max(s["load"]["l"], 0.0), max(s["load"]["r"], 0.0)
        tot = fl + fr
        wl, wr = (0.5, 0.5) if tot < 0.2 * self.BW else (fl / tot, fr / tot)
        # anticipation: a trained exo user loads the stance side (and the swing-side crutch) before and
        # during the device's step, i.e. balances about the expected stance foot
        if self.expected_stance is not None and self.anticipation > 0:
            a = self.anticipation
            ws = (1.0, 0.0) if self.expected_stance == "l" else (0.0, 1.0)
            wl, wr = (1 - a) * wl + a * ws[0], (1 - a) * wr + a * ws[1]
        # balance pivot = (load-weighted) centre of pressure under the feet: the moment balance about it is
        # exact, so passive ankle torque (AFO, stiff/spastic plantarflexors) is accounted for
        o = wl * s["cop"]["l"] + wr * s["cop"]["r"]
        o[2] = 0.0
        mid = 0.5 * (s["foot"]["l"] + s["foot"]["r"])
        # actually standing on one foot (perceived loads only; anticipation shifts the pivot, not this)
        single = tot >= 0.2 * self.BW and min(fl, fr) / tot < 0.1
        walking = self.walk_intent > 0.5 and self.target_speed > 0      # stepping has started
        if single:
            st = "l" if fl > fr else "r"
            y_des = s["foot"][st][1] - (0.015 if st == "l" else -0.015)
        else:
            y_des = mid[1] + self.lateral_bias
            if self.pending_crutch in ("l", "r"):
                # before lifting a crutch the COM must be (laterally) inside the opposite foot: the remaining
                # crutch can only push the body toward the lifted side. Shift only as far as that requires.
                mg = (sp.lift_lat_margin + sp.lift_shift) * self.s
                if self.pending_crutch == "l":
                    y_des = min(y_des, s["foot"]["l"][1] - mg)
                else:
                    y_des = max(y_des, s["foot"]["r"][1] + mg)
        # the person predicts the present state from the delayed one (internal model, ~Smith predictor)
        lag = self.nd * self.dt * sp.prediction
        c = s["com"] + vcom * lag
        # exo gait is a sequence of quasi-static postures:
        #  * standing / double support: COM a little ahead of the middle of the feet (com_ahead)
        #  * requesting a step: lean until the COM is `lean_ahead` beyond the FRONT foot (only once the
        #    crutches are planted again)
        #  * while the exo steps (single support, or the person feels it coming): COM over the stance foot
        front = max(s["foot"]["l"][0], s["foot"]["r"][0])
        stepping = single or (walking and self.anticipation > 0.75 and self.expected_stance is not None)
        x_rest = 0.5 * (s["foot"]["l"][0] + s["foot"]["r"][0]) + sp.com_ahead
        v_ref = 0.0
        if stepping:
            # while the exo steps, the body travels forward inside the tripod (stance foot + crutch tips): the
            # person only stops it falling back behind the stance foot or getting too close to the crutch line,
            # and brakes above the expected walking speed
            st_ = self.expected_stance if (self.expected_stance is not None and not single) else \
                ("l" if fl > fr else "r")
            lo = s["foot"][st_][0] + sp.com_ahead
            hi = max(lo, min(self.tips["l"][0], self.tips["r"][0]) - sp.tip_margin * self.s) \
                if self.sp.kind in ("crutches", "walker") else lo + sp.max_ahead_walking
            x_des = float(np.clip(c[0], lo, hi))
            v_ref = max(self.target_speed, 0.0) if lo < c[0] < hi else 0.0   # "fall" into the step at walking speed
        elif self.moving_crutches:
            # advancing the crutches after a step: hold the body where it is (no retreat to mid-feet)
            x_des = float(np.clip(c[0], x_rest, front + sp.com_ahead))
        else:
            lean = float(np.clip(self.lean_frac, 0.0, 1.0))
            x_des = x_rest + lean * max(0.0, front + sp.lean_ahead + self.extra_lean * self.s - x_rest)
        x_des = min(x_des, front + sp.max_ahead_walking)
        ax = sp.kp_com * (x_des - c[0]) + sp.kd_com * (v_ref - vcom[0])
        ay = sp.kp_lat_com * (y_des - c[1]) + sp.kd_lat_com * (0.0 - vcom[1])
        r = c - o
        M_req = self.mass * np.array([r[1] * 9.81 - r[2] * ay, r[2] * ax - r[0] * 9.81])
        # trunk orientation (world frame): delayed + predicted feedback and intrinsic arm impedance
        p_pred = s["pitch"] + rate_p * lag
        r_pred = s["roll"] + rate_r * lag
        w_now = d.cvel[self.torso][:3]                     # world angular velocity (rad/s)
        tp = self.target_pitch
        m_pitch = -sp.k_tilt * (p_pred - tp) - sp.d_tilt * rate_p \
            - sp.k_tilt_int * (s_now["pitch"] - tp) - sp.d_tilt_int * w_now[1]
        m_roll = -sp.k_tilt * r_pred - sp.d_tilt * rate_r - sp.k_tilt_int * s_now["roll"] + sp.d_tilt_int * w_now[0]
        M_trunk = np.array([-m_roll, m_pitch])          # roll (+ toward left) is a rotation about -x
        # heading: keep facing the walking direction (yaw 0 = heading at reset)
        rate_y = _wrap(s1["yaw"] - s["yaw"]) / self.dt
        y_pred = s["yaw"] + rate_y * lag
        M_yaw = -sp.k_yaw * _wrap(y_pred) - sp.d_yaw * rate_y - sp.k_yaw_int * _wrap(s_now["yaw"]) \
            - sp.d_yaw_int * w_now[2]
        c_t = d.xipos[self.torso]
        # preferred loading: along each crutch axis
        e_h = sp.k_height * max(0.0, self.h_nom - 0.02 - c[2])
        f_pref_tot = 2 * self.f_base + self.single_support * sp.single_support_frac * self.BW + e_h \
            + self.walk_intent * sp.walking_extra_frac * self.BW
        avail = np.array([self.side_capacity["l"], self.side_capacity["r"]]) > 0
        n_av = max(int(avail.sum()), 1)
        rows, rhs, U, L = [], [], np.zeros((2, 3)), np.zeros(2)
        Jb = np.zeros((2, 6))
        Jt = np.zeros((2, 6))
        Jz = np.zeros((1, 6))
        Jy = np.zeros((1, 6))
        R_pref = np.zeros(6)
        for k, side in enumerate("lr"):
            P = self.tips[side]
            H = self.hand(d, side)
            u = H - P
            L[k] = max(np.linalg.norm(u), 1e-6)
            U[k] = u / L[k]
            if not avail[k]:
                continue
            Jb[:, 3 * k:3 * k + 3] = _skew(P - o)[:2]
            Jt[:, 3 * k:3 * k + 3] = _skew(P - c_t)[:2]
            Jy[0, 3 * k:3 * k + 3] = _skew(P - c_t)[2]
            Jz[0, 3 * k + 2] = 1.0
            R_pref[3 * k:3 * k + 3] = U[k] * f_pref_tot / n_av
        lam = 1e-3
        wb = np.array([sp.w_lateral, 1.0])[:, None]          # falling sideways matters more than AP drift
        A = np.vstack([wb * Jb, sp.w_trunk * Jt, sp.w_yaw * Jy, sp.w_load * Jz, np.sqrt(lam) * np.eye(6)])
        b = np.concatenate([wb[:, 0] * M_req, sp.w_trunk * M_trunk, [sp.w_yaw * M_yaw], [sp.w_load * f_pref_tot],
                            np.sqrt(lam) * R_pref])
        caps = np.array([self.side_capacity["l"], self.side_capacity["r"]]) * sp.max_load_frac * self.BW
        R = np.zeros((2, 3))
        free = [k for k in range(2) if avail[k]]
        # bounded allocation: solve, project onto what each crutch can transmit, fix the crutches that hit a
        # limit and re-solve for the others (exact for the 2-crutch case)
        for _ in range(2):
            if not free:
                break
            cols = np.concatenate([np.arange(3 * k, 3 * k + 3) for k in free])
            b_eff = b - A @ R.reshape(6) + A[:, cols] @ R.reshape(6)[cols]
            x = np.linalg.lstsq(A[:, cols], b_eff, rcond=None)[0]
            changed = []
            for j, k in enumerate(free):
                want = x[3 * j:3 * j + 3]
                R[k] = self._project(want, U[k], L[k], caps[k])
                if np.linalg.norm(R[k] - want) > 1e-6:
                    changed.append(k)
            if not changed or len(changed) == len(free):
                break
            free = [k for k in free if k not in changed]
        self.debug = dict(M_req=M_req, M_trunk=M_trunk, M_yaw=M_yaw, R=R.copy(), f_pref=f_pref_tot, x_des=x_des, y_des=y_des,
                          com=c.copy(), o=o, single=single)
        a = self.dt / (0.05 + self.dt)               # ~50 ms force rise time
        if sp.k_arm <= 0:
            self.F += a * (R - self.F)
        else:
            # intrinsic arm: the extended, co-contracted arm + crutch is a stiff strut of commanded length L0
            # (no delay); the delayed postural controller only moves L0 to obtain the axial load it wants.
            # The off-axis part (made by the shoulder) keeps the 50 ms muscle force rise.
            for k, side in enumerate("lr"):
                if not avail[k]:
                    self.F[k] = 0.0
                    self.L0[k] = None
                    continue
                Lk, u = L[k], U[k]
                if self.L0[k] is None:
                    self.L0[k], self.L_prev[k] = Lk, Lk
                Ld = (Lk - self.L_prev[k]) / self.dt
                self.L_prev[k] = Lk
                f_des = max(0.0, float(R[k] @ u))
                perp_des = R[k] - f_des * u if f_des > 0 else R[k] - float(R[k] @ u) * u
                f_spring = sp.k_arm * (self.L0[k] - Lk) - sp.c_arm * Ld
                f_ax = float(np.clip(f_spring, 0.0, caps[k]))
                # move the commanded length so the steady-state load approaches what the person wants
                # while the exo steps the person "locks" the arms (stiff tripod) and re-sets them in double support
                t_set = sp.arm_set_time_stepping if stepping else sp.arm_set_time
                self.L0[k] += self.dt / t_set * (f_des - max(f_spring, 0.0)) / sp.k_arm
                self.L0[k] = min(self.L0[k], Lk + caps[k] / sp.k_arm)      # cannot command beyond full effort
                self.perp[k] += a * (perp_des - self.perp[k])
                self.F[k] = self._project(f_ax * u + self.perp[k], u, Lk, caps[k]) if f_ax > 0 else 0.0
        for k in range(2):
            if not avail[k]:
                self.F[k] = 0.0
        self.f = np.linalg.norm(self.F, axis=1)
        return self.F

    def _project(self, R, u, L, cap):
        """Project a desired crutch reaction onto what the crutch + arm can transmit."""
        sp = self.sp
        if R[2] <= 0.0:
            return np.zeros(3)
        a = float(R @ u)                               # axial part (push only)
        perp = R - a * u
        if a <= 0.0:
            a, perp = 0.0, perp
        pm = np.linalg.norm(perp)
        pmax = sp.shoulder_moment_max / L
        if pm > pmax:
            perp *= pmax / pm
        R = a * u + perp
        if R[2] <= 0.0:
            return np.zeros(3)
        h = np.hypot(R[0], R[1])
        if h > sp.friction * R[2]:
            R[:2] *= sp.friction * R[2] / h
        n = np.linalg.norm(R)
        if n > cap:
            R *= cap / n
        return R

    def apply(self, d):
        if not self.enabled:
            return
        c = d.xipos[self.torso]
        f = self.F.sum(0)
        tq = np.zeros(3)
        for k, side in enumerate("lr"):
            # the reaction's line of action passes through the tip (the arm carries the off-axis moment)
            P = self.tips[side] if self.sp.kind != "harness" else self.hand(d, side)
            tq += np.cross(P - c, self.F[k])
        d.xfrc_applied[self.torso, :3] = f
        d.xfrc_applied[self.torso, 3:] = tq


class Therapist:
    """Therapist guarding through the gait belt: a delayed, force-limited impedance on the pelvis that helps
    the person reach the COM position and upright posture they are trying to hold (it does not walk for them).
    With level "contact_guard" it only acts when the COM error exceeds a deadband (catching near-falls)."""

    def __init__(self, m: mujoco.MjModel, tp, dt: float, support: "UpperLimbSupport"):
        self.m, self.tp, self.dt, self.sup = m, tp, dt, support
        self.enabled = tp.level != "none"
        self.pelvis = m.body("pelvis").id
        self.root = m.body("Full Body").id
        self.nd = max(1, int(round(tp.reaction_delay / dt)))
        self.F = np.zeros(3)
        self.M = np.zeros(3)
        self.work = 0.0                       # N*s impulse delivered (|F| integrated)
        self.peak = 0.0

    def reset(self, d):
        self.R0 = d.xmat[self.pelvis].reshape(3, 3).copy()
        self.buf = deque(maxlen=self.nd + 2)
        for _ in range(self.nd + 2):
            self.buf.append(self._state(d))
        self.F[:] = 0.0
        self.M[:] = 0.0
        self.work = 0.0
        self.peak = 0.0

    def _state(self, d):
        R = d.xmat[self.pelvis].reshape(3, 3) @ self.R0.T
        z = R[:, 2]
        return dict(com=d.subtree_com[self.root].copy(), pitch=np.arctan2(z[0], z[2]), roll=np.arctan2(z[1], z[2]))

    def step(self, d):
        self.buf.append(self._state(d))
        if not self.enabled:
            return
        tp, lvl = self.tp, self.tp.level
        s, s1 = self.buf[0], self.buf[1]
        v = (s1["com"] - s["com"]) / self.dt
        dbg = getattr(self.sup, "debug", None)
        if dbg is None:
            return
        err = np.array([dbg["x_des"] - s["com"][0], dbg["y_des"] - s["com"][1]])
        db = tp.deadband[lvl]
        n = np.linalg.norm(err)
        e = err * max(0.0, n - db) / n if n > 1e-9 else err * 0.0
        f = tp.kp * e - tp.kd * v[:2] * (1.0 if n > db else 0.3)
        cap = tp.force_cap[lvl]
        fn = np.linalg.norm(f)
        if fn > cap:
            f *= cap / fn
        rp, rr = (s1["pitch"] - s["pitch"]) / self.dt, (s1["roll"] - s["roll"]) / self.dt
        mp = -tp.k_tilt * (s["pitch"] - self.sup.target_pitch) - tp.d_tilt * rp
        mr = -tp.k_tilt * s["roll"] - tp.d_tilt * rr
        mc = tp.moment_cap[lvl]
        M = np.array([-np.clip(mr, -mc, mc), np.clip(mp, -mc, mc), 0.0])
        a = self.dt / (0.08 + self.dt)
        self.F[:2] += a * (f - self.F[:2])
        self.M += a * (M - self.M)
        fn = float(np.linalg.norm(self.F))
        self.work += fn * self.dt
        self.peak = max(self.peak, fn)

    def apply(self, d):
        if not self.enabled:
            return
        d.xfrc_applied[self.pelvis, :3] = self.F
        d.xfrc_applied[self.pelvis, 3:] = self.M


class Planarizer:
    """Planarizing boom on the pelvis (test rig, see PlanarizerParams): holds the pelvis's sideways position and its
    lateral axis (no roll, no heading change) with a stiff, undelayed spring-damper; forward/vertical translation and
    pitch are left free. Logs the sideways force and the moment it has to deliver."""

    def __init__(self, m: mujoco.MjModel, pp, dt: float):
        self.m, self.pp, self.dt = m, pp, dt
        self.enabled = bool(pp.enabled)
        self.pelvis = m.body("pelvis").id
        self.F = np.zeros(3)
        self.M = np.zeros(3)
        self._v = np.zeros(6)
        self.impulse = 0.0                     # N*s sideways force integrated
        self.peak = 0.0                        # N
        self.peak_moment = 0.0                 # N*m

    def reset(self, d):
        self.y0 = float(d.xipos[self.pelvis][1])
        # the pelvis-fixed axis that points along world y (left) at reset
        self.ey_body = d.xmat[self.pelvis].reshape(3, 3).T @ np.array([0.0, 1.0, 0.0])
        self.F[:] = 0.0
        self.M[:] = 0.0
        self.impulse = 0.0
        self.peak = 0.0
        self.peak_moment = 0.0

    def step(self, d):
        if not self.enabled:
            return
        pp = self.pp
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, self._v, 0)   # [ang, lin], world
        w, v = self._v[:3], self._v[3:]
        self.F[1] = -pp.k_lat * (d.xipos[self.pelvis][1] - self.y0) - pp.d_lat * v[1]
        # rotate the pelvis's lateral axis back onto world y: the error a x e_y has no y part, so pitch is free
        a = d.xmat[self.pelvis].reshape(3, 3) @ self.ey_body
        self.M[:] = pp.k_rot * np.array([-a[2], 0.0, a[0]]) - pp.d_rot * np.array([w[0], 0.0, w[2]])
        self.impulse += abs(self.F[1]) * self.dt
        self.peak = max(self.peak, abs(self.F[1]))
        self.peak_moment = max(self.peak_moment, float(np.linalg.norm(self.M)))

    def apply(self, d):
        if not self.enabled:
            return
        d.xfrc_applied[self.pelvis, :3] += self.F           # on top of the therapist (sim zeroes the pelvis each tick)
        d.xfrc_applied[self.pelvis, 3:] += self.M
