"""A baseline controller that behaves like the Indego's walking mode, after
Quintero, Farris & Goldfarb (2012) "A Method for the Autonomous Control of Lower Limb
Exoskeletons for Persons With Paraplegia": a finite-state machine whose transitions are
triggered by the user's posture (center-of-pressure / COM estimate from the exo's own
kinematics: lean forward -> step, stay back/upright -> stop), with joint-space trajectory
tracking during each step.

States:  STAND --(COM ahead)--> STEP --> DOUBLE_SUPPORT (weight transfer)
         DOUBLE_SUPPORT --(COM over front foot)--> STEP (other leg)
         DOUBLE_SUPPORT --(no lean for ds_max_wait)--> CLOSE (feet together) --> STAND

It only uses what `ExoSensors` provides (encoders + hip-unit AHRS, at the control rate), plus the
optional add-on upper-trunk IMU when it is fitted (`SensorParams.trunk_imu`): the stock device cannot
see the trunk flexing at the waist (L5/S1), which moves ~1/3 of the body mass, so with stock sensors
the estimator assumes the trunk keeps its quiet-standing angle to the pelvis.
Use it (1) as a reference behaviour to validate the simulator against your own session
logs and (2) as a template for your own controller.

COM estimator: a sagittal segment model (pelvis group, trunk above L5, thigh, shank, foot per leg)
whose masses and COM vectors are expressed in the IMU frames. The defaults are the nominal model's
(1.75 m, 75 kg + 12.3 kg exo); `IndegoFSMParams.from_model(m)` measures them on any (scaled) model.
On the real device compute the same numbers from anthropometric tables (e.g. de Leva 1996) plus the
exo's mass properties.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIP_L, KNEE_L, HIP_R, KNEE_R = 0, 1, 2, 3


@dataclass
class IndegoFSMParams:
    dt: float = 0.005
    # posture / geometry known to the device (set from the fitting)
    stand_q: tuple = (0.0, -0.05, 0.0, -0.05)          # hip_l, knee_l, hip_r, knee_r [rad]
    L_thigh: float = 0.404                              # exo hip axis -> knee axis [m]
    L_shank: float = 0.400                              # exo knee axis -> ankle [m]
    # sagittal COM estimator: mass fractions (human + exo) and COM vectors (x fwd, z up) in the IMU frames
    use_trunk_imu: bool = True                          # use the add-on trunk IMU when the observation has one
    m_pelvis: float = 0.244                             # pelvis + sacrum + hip unit
    r_pelvis: tuple = (0.0036, 0.0702)                  # its COM from the hip-axis midpoint, hip-IMU frame [m]
    m_trunk: float = 0.340                              # everything above L5/S1 (trunk, head, arms)
    r_l5: tuple = (-0.0418, 0.1495)                     # L5/S1 from the hip-axis midpoint, hip-IMU frame [m]
    r_trunk: tuple = (0.0081, 0.2507)                   # trunk COM from L5/S1 in quiet standing, hip-IMU frame [m]
    m_thigh: float = 0.135                              # thigh + exo thigh segment (each leg)
    c_thigh: float = 0.498                              # COM along hip->knee
    m_shank: float = 0.056                              # shank + exo shank (each leg)
    c_shank: float = 0.441                              # COM along knee->ankle
    m_foot: float = 0.017                               # foot + AFO (each leg)
    foot_dx: float = 0.042                              # foot COM ahead of the ankle [m]
    shank_offset: float = np.radians(-1.44)             # knee->ankle line vs exo shank axis
    # step shape
    step_time: float = 1.2                              # s per step [Indego slow walking ~0.8-1.5 s]
    hip_flex: float = np.radians(36.0)                  # swing-hip target at heel strike
    hip_ext: float = np.radians(-8.0)                   # stance-hip at the end of the step
    knee_peak: float = np.radians(60.0)                 # swing-knee peak flexion
    knee_peak_at: float = 0.5                           # fraction of the step (late knee extension = toe clearance)
    hip_delay: float = 0.0                              # optional swing-hip delay (fraction of the step)
    swing_hip_one_sided: bool = True                    # swing hip assists flexion but never brakes it below the goal
    hip_reach: float = 0.55                             # swing hip reaches its goal at this fraction of the step
                                                        # (leads the knee, as in normal swing)
    knee_stance: float = np.radians(-3.0)               # drive the stance knee into full extension
    first_step_scale: float = 1.0
    ds_shift: float = np.radians(5.0)                   # double-support weight transfer (both hips extend)
    ds_shift_time: float = 0.6
    ds_hold_actual: bool = True                         # DS holds the measured end-of-step posture
    ds_kp_hip: float = -1.0                             # N*m/rad hip stiffness in double support after a step; < 0 = kp.
                                                        # A user with leg function shifts their own weight in double
                                                        # support (hips and knees locked + both feet down = no motion
                                                        # possible), so the hips must give way: set e.g. 50-100
    # transitions
    trigger: str = "lean"                               # "lean": user posture (Indego Personal); "auto": steps on a
                                                        # timer; "external": call request_step() (therapist app)
    auto_ds_time: float = 1.2                           # s of double support between steps in "auto" mode
    stand_trigger_dx: float = 0.02                      # COM ahead of its quiet-standing position [m] -> first step
    ds_trigger_dx: float = 0.0                          # COM relative to the front ankle [m] -> next step ...
    ds_trigger_len_gain: float = 0.35                   # ... minus this x step length: like the Vanderbilt/Indego
                                                        # controller, the threshold drops linearly with step length
    ds_lean_dx: float = 0.02                            # and the user must have shifted the COM this far forward of
                                                        # its rearmost position in this double support (deliberate lean)
    lat_dy: float = 0.012                               # lateral COM shift toward the stance foot needed to step [m]
                                                        # (Vanderbilt/Indego: frontal-plane lean selects the leg); 0 = off
    h_lat: float = 0.87                                 # whole-body COM height for the lateral estimate [m]
    h_lat_trunk: float = 0.085                          # trunk mass fraction x its COM height above L5 [m]
    trigger_hold: float = 0.10
    ds_min_time: float = 1.2                            # minimum double-support time (crutch moves)
    ds_max_wait: float = 6.0                            # no trigger for this long in DS -> close feet, stop
    calib_time: float = 1.0                             # quiet-standing calibration at start-up
    # gains (N*m/rad, N*m*s/rad). With ~11 ms loop latency (sensor + ZOH + driver + velocity
    # filter) and compliant cuffs, kd above ~8 causes a 10-15 Hz limit cycle in this model.
    kp: tuple = (250.0, 200.0, 250.0, 200.0)
    kd: tuple = (6.0, 6.0, 6.0, 6.0)
    kp_stance_knee: float = 400.0
    tau_max: float = 60.0
    # world-frame swing + pelvis stabilisation through the stance hip (SIMBICON-like). False = pure
    # relative joint-trajectory tracking.
    world_frame: bool = False
    pelvis_pitch_des: float = np.radians(2.0)           # relative to the pitch measured in quiet standing
    swing_compensation: bool = True                     # stance hip also cancels the swing-hip reaction on the
                                                        # pelvis (SIMBICON: tau_st = tau_pelvis - tau_sw)
    kp_pelvis: float = 350.0
    kd_pelvis: float = 25.0
    gravity_comp: bool = True
    leg_mass: float = 17.0                              # human + exo leg used for swing gravity compensation [est]
    leg_com: float = 0.32
    shank_mass: float = 6.0
    shank_com: float = 0.22


def estimator_constants(m, key: int = 0) -> dict:
    """Measure the COM-estimator constants on a model built by `build_model` (standing keyframe `key`):
    segment mass fractions, COM vectors in the hip-IMU frame, leg lengths. Returns IndegoFSMParams fields."""
    import mujoco
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    seg_root = {"lumbar5": "trunk", "femur_l": "thigh_l", "femur_r": "thigh_r", "exo_thigh_l": "thigh_l",
                "exo_thigh_r": "thigh_r", "tibia_l": "shank_l", "tibia_r": "shank_r", "exo_shank_l": "shank_l",
                "exo_shank_r": "shank_r", "talus_l": "foot_l", "talus_r": "foot_r", "exo_afo_l": "foot_l",
                "exo_afo_r": "foot_r"}

    def seg_of(b):
        while b > 0:
            nm = m.body(b).name
            if nm in seg_root:
                return seg_root[nm]
            b = m.body_parentid[b]
        return "pelvis"
    M, S = {}, {}
    for b in range(1, m.nbody):
        g = seg_of(b)
        M[g] = M.get(g, 0.0) + m.body_mass[b]
        S[g] = S.get(g, np.zeros(3)) + m.body_mass[b] * d.xipos[b]
    Mt = sum(M.values())
    com = {g: S[g] / M[g] for g in M}
    ja = lambda j: d.xanchor[m.joint(j).id].copy()  # noqa: E731
    H = {s: ja(f"exo_hip_{s}") for s in "lr"}
    K = {s: ja(f"exo_knee_{s}") for s in "lr"}
    A = {s: d.xpos[m.body(f"talus_{s}").id].copy() for s in "lr"}
    Hm = 0.5 * (H["l"] + H["r"])
    R = d.site_xmat[m.site("exo_imu_hip").id].reshape(3, 3)
    hu = lambda v: tuple(float(x) for x in (R.T @ v)[[0, 2]])  # noqa: E731
    L5 = ja("lumbar_flexion")
    u, v = (K["l"] - H["l"])[[0, 2]], (A["l"] - K["l"])[[0, 2]]
    th, sh = np.arctan2(u[0], -u[1]), np.arctan2(v[0], -v[1])
    return dict(m_pelvis=M["pelvis"] / Mt, r_pelvis=hu(com["pelvis"] - Hm), m_trunk=M["trunk"] / Mt,
                r_l5=hu(L5 - Hm), r_trunk=hu(com["trunk"] - L5),
                m_thigh=0.5 * (M["thigh_l"] + M["thigh_r"]) / Mt,
                c_thigh=float(np.dot((com["thigh_l"] - H["l"])[[0, 2]], u) / np.dot(u, u)),
                m_shank=0.5 * (M["shank_l"] + M["shank_r"]) / Mt,
                c_shank=float(np.dot((com["shank_l"] - K["l"])[[0, 2]], v) / np.dot(v, v)),
                m_foot=0.5 * (M["foot_l"] + M["foot_r"]) / Mt, foot_dx=float(com["foot_l"][0] - A["l"][0]),
                L_thigh=float(np.linalg.norm(u)), L_shank=float(np.linalg.norm(v)), shank_offset=float(sh - th))


def _pitch(qt):
    """Forward pitch (rotation about the left axis) of an IMU frame from its quaternion (w,x,y,z)."""
    w, x, y, z = qt
    return np.arctan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y))


def _roll(qt):
    """Sideways lean (+ = toward the left, y axis) of an IMU frame's up axis, from its quaternion (w,x,y,z)."""
    w, x, y, z = qt
    return np.arctan2(2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def _minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def _minjerk_d(s):
    s = np.clip(s, 0.0, 1.0)
    return 30 * s * s * (1 - s) ** 2


def _knee_bump(s, peak_at):
    ph = np.clip(s / (2 * peak_at), 0, 1) if s < peak_at else 0.5 + 0.5 * np.clip((s - peak_at) / (1 - peak_at), 0, 1)
    return np.sin(np.pi * ph)


class IndegoFSMController:
    STAND, STEP, DS, CLOSE = "stand", "step", "double_support", "close"

    def __init__(self, p: IndegoFSMParams = None):
        self.p = p or IndegoFSMParams()
        self.reset()

    def reset(self):
        p = self.p
        self.state = self.STAND
        self.t = 0.0
        self.t_state = 0.0
        self.trigger_t = 0.0
        self.swing = "r"
        self.n_steps = 0               # steps since standing (reset by CLOSE: the next one is a first step again)
        self.steps_total = 0           # all completed steps (for reporting)
        self.closes = 0                # double-support time-outs that closed the feet
        self.q_start = np.array(p.stand_q, float)
        self.q_goal = np.array(p.stand_q, float)
        self.com0 = None
        self.phi0 = 0.0
        self.phi_cal = None
        self.trunk_ref = None          # (pelvis pitch, trunk pitch) in quiet standing
        self.ds_com_min = np.inf
        self.roll_ref = None           # (hip-unit roll, trunk roll) in quiet standing
        self.est = {}
        self._requested = False
        self.auto_walk = True          # "auto" mode: set False to stop after the current step
        self.log = []

    def request_step(self):
        """External trigger (therapist/app or a test harness) for the next step."""
        self._requested = True

    # --- estimates ---------------------------------------------------------------------------
    @staticmethod
    def pelvis_pitch(obs):
        """Forward pitch of the hip unit from its AHRS quaternion (w,x,y,z)."""
        return _pitch(obs["hip_quat"])

    @classmethod
    def from_model(cls, m, **kw):
        """Controller with the COM-estimator constants measured on model `m` (see estimator_constants)."""
        return cls(IndegoFSMParams(**{**estimator_constants(m), **kw}))

    def estimate(self, obs):
        """Sagittal COM relative to the ankles (static COP proxy) from the exo's own kinematics, the hip-unit
        AHRS and, when fitted, the trunk IMU (lumbar flexion relative to quiet standing)."""
        p = self.p
        q = obs["q"]
        phi = _pitch(obs["hip_quat"])
        out = {"phi": phi, "lumbar": 0.0}
        delta = 0.0
        if p.use_trunk_imu and "trunk_quat" in obs:
            theta = _pitch(obs["trunk_quat"])
            out["theta"] = theta
            if self.trunk_ref is not None:
                delta = (theta - self.trunk_ref[1]) - (phi - self.trunk_ref[0])
                out["lumbar"] = delta
                out["trunk_lean"] = theta - self.trunk_ref[1]       # trunk pitch change since calibration

        def rx(r, a):
            return r[0] * np.cos(a) + r[1] * np.sin(a)
        mx = p.m_pelvis * rx(p.r_pelvis, phi) + p.m_trunk * (rx(p.r_l5, phi) + rx(p.r_trunk, phi + delta))
        ms = p.m_pelvis + p.m_trunk
        ankles = []
        for hip, knee in ((HIP_L, KNEE_L), (HIP_R, KNEE_R)):
            th = q[hip] - phi                           # world thigh angle (+ = knee ahead of hip)
            sh = th - q[knee] + p.shank_offset          # world shank angle (+ = ankle ahead of knee)
            kx = p.L_thigh * np.sin(th)
            ax = kx + p.L_shank * np.sin(sh)
            mx += p.m_thigh * p.c_thigh * kx + p.m_shank * (kx + p.c_shank * (ax - kx)) + p.m_foot * (ax + p.foot_dx)
            ms += p.m_thigh + p.m_shank + p.m_foot
            ankles.append(ax)
        # lateral COM shift from quiet standing (+ = toward the left foot): whole body rolling about the feet
        # (hip-unit AHRS) plus, with the trunk IMU, the trunk bending sideways relative to the pelvis
        rp = _roll(obs["hip_quat"])
        out["roll"] = rp
        if self.roll_ref is not None:
            lat = p.h_lat * np.sin(rp - self.roll_ref[0])
            if "trunk_quat" in obs and p.use_trunk_imu:
                rt = _roll(obs["trunk_quat"])
                out["roll_trunk"] = rt
                lat += p.h_lat_trunk * (np.sin(rt - self.roll_ref[1]) - np.sin(rp - self.roll_ref[0]))
            out["lat"] = lat
        else:
            out["roll_trunk"] = _roll(obs["trunk_quat"]) if "trunk_quat" in obs else 0.0
            out["lat"] = 0.0
        com = mx / ms
        out["com"] = com
        out["ankle_l"], out["ankle_r"] = ankles
        out["com_rel_mid"] = com - 0.5 * (ankles[0] + ankles[1])
        out["com_rel_front"] = com - max(ankles)
        return out

    # --- trajectories --------------------------------------------------------------------------
    def _step_goal(self, closing=False):
        p = self.p
        q = np.array(p.stand_q, float)
        if closing:
            return q
        sc = p.first_step_scale if self.n_steps == 0 else 1.0
        sw, st = (HIP_R, HIP_L) if self.swing == "r" else (HIP_L, HIP_R)
        q[sw] = p.hip_flex * sc
        q[st] = p.hip_ext * sc
        q[KNEE_L] = q[KNEE_R] = p.knee_stance
        return q

    def _desired(self):
        p = self.p
        if self.state in (self.STEP, self.CLOSE):
            T = p.step_time if self.state == self.STEP else 0.8 * p.step_time
            s = self.t_state / T
            b, bd = _minjerk(s), _minjerk_d(s) / T
            q = self.q_start + (self.q_goal - self.q_start) * b
            qd = (self.q_goal - self.q_start) * bd
            kn = KNEE_R if self.swing == "r" else KNEE_L
            sw = HIP_R if self.swing == "r" else HIP_L
            span = max(p.hip_reach - p.hip_delay, 0.05)
            sh = np.clip((s - p.hip_delay) / span, 0, 1)                     # swing-hip profile
            q[sw] = self.q_start[sw] + (self.q_goal[sw] - self.q_start[sw]) * _minjerk(sh)
            qd[sw] = (self.q_goal[sw] - self.q_start[sw]) * _minjerk_d(sh) / (T * span)
            peak = p.knee_peak if self.state == self.STEP else 0.7 * p.knee_peak
            bump = _knee_bump(s, p.knee_peak_at)
            q[kn] = p.knee_stance + (peak - p.knee_stance) * bump
            qd[kn] = (peak - p.knee_stance) * (_knee_bump(s + 1e-3, p.knee_peak_at) - bump) / (1e-3 * T)
            return q, qd
        if self.state == self.DS and self.n_steps > 0:
            s = self.t_state / p.ds_shift_time
            q = self.q_start.copy()
            qd = np.zeros(4)
            q[[HIP_L, HIP_R]] -= p.ds_shift * _minjerk(s)
            qd[[HIP_L, HIP_R]] -= p.ds_shift * _minjerk_d(s) / p.ds_shift_time
            return q, qd
        return self.q_goal.copy(), np.zeros(4)

    def _weight_on_stance(self, est) -> bool:
        """The user has shifted the COM sideways toward the stance foot of the next step (swing = self.swing)."""
        if self.p.lat_dy <= 0:
            return True
        return est["lat"] > self.p.lat_dy if self.swing == "r" else est["lat"] < -self.p.lat_dy

    def _start(self, state, q, closing=False):
        self.phi0 = self.est.get("phi", 0.0)
        self.state = state
        self.t_state = 0.0
        self.trigger_t = 0.0
        self.q_start = q.copy()
        self.q_goal = self._step_goal(closing)

    # --- main ------------------------------------------------------------------------------------
    def __call__(self, obs):
        p = self.p
        self.t += p.dt
        self.t_state += p.dt
        q, qd = obs["q"], obs["qd"]
        est = self.estimate(obs)
        self.est = est
        if self.t < p.calib_time:
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

        if self.state == self.STAND:
            if p.trigger == "lean":
                want = self.t > p.calib_time and est["com_rel_mid"] - self.com0 > p.stand_trigger_dx \
                    and self.t_state > 0.3 and self._weight_on_stance(est)
            else:
                want = self._requested or (p.trigger == "auto" and self.t > p.calib_time and self.t_state > 0.3
                                           and self.auto_walk)
            if want and self._requested:
                self._requested = False
                self.trigger_t = p.trigger_hold
            self.trigger_t = self.trigger_t + p.dt if want else 0.0
            if self.trigger_t >= p.trigger_hold:
                self._start(self.STEP, q)
        elif self.state == self.DS:
            if p.trigger == "lean":
                step_len = abs(est["ankle_l"] - est["ankle_r"])
                if self.t_state > 0.3:
                    self.ds_com_min = min(self.ds_com_min, est["com_rel_front"])
                want = self.t_state > p.ds_min_time and \
                    est["com_rel_front"] > p.ds_trigger_dx - p.ds_trigger_len_gain * step_len and \
                    est["com_rel_front"] - self.ds_com_min > p.ds_lean_dx and self._weight_on_stance(est)
            elif p.trigger == "auto":
                want = self.t_state > p.auto_ds_time and self.auto_walk
            else:
                want = self._requested and self.t_state > p.ds_min_time
                if want:
                    self._requested = False
                    self.trigger_t = p.trigger_hold
            self.trigger_t = self.trigger_t + p.dt if want else 0.0
            if self.trigger_t >= p.trigger_hold:
                self._start(self.STEP, q)
            elif self.t_state > p.ds_max_wait:
                self.closes += 1
                self._start(self.CLOSE, q, closing=True)
        elif self.state == self.STEP and self.t_state >= p.step_time:
            self.n_steps += 1
            self.steps_total += 1
            self.swing = "l" if self.swing == "r" else "r"
            self.state, self.t_state = self.DS, 0.0
            self.ds_com_min = np.inf
            # hold the posture actually reached (both feet are on the ground: driving to the planned goal
            # would only create internal forces and pitch the pelvis)
            self.q_start = np.array(q, float) if p.ds_hold_actual else self.q_goal.copy()
        elif self.state == self.CLOSE and self.t_state >= 0.8 * p.step_time:
            self.state, self.t_state, self.n_steps = self.STAND, 0.0, 0
            self.q_goal = np.array(p.stand_q, float)

        q_des, qd_des = self._desired()
        self._traj_q = q_des.copy()
        kp = np.array(p.kp, float)
        kd = np.array(p.kd, float)
        stance_pelvis_ctrl = None
        if p.world_frame and self.state in (self.STEP, self.CLOSE):
            # swing hip: trajectory interpreted as a WORLD thigh angle (add the pelvis pitch back)
            sw = HIP_R if self.swing == "r" else HIP_L
            st = HIP_L if self.swing == "r" else HIP_R
            q_des[sw] = q_des[sw] + est["phi"] - self.phi0
            stance_pelvis_ctrl = st
        if self.state in (self.STEP, self.CLOSE):
            kst = KNEE_L if self.swing == "r" else KNEE_R
            kp[kst] = p.kp_stance_knee
        else:
            kp[[KNEE_L, KNEE_R]] = p.kp_stance_knee
        if self.state == self.DS and self.n_steps > 0 and p.ds_kp_hip >= 0:
            kp[[HIP_L, HIP_R]] = p.ds_kp_hip
        e, ed = q_des - q, qd_des - qd
        if p.swing_hip_one_sided and self.state in (self.STEP, self.CLOSE):
            # never pull the swing thigh back while it is on its way to the goal: knee flexion with a still
            # loaded foot (pre-swing) throws the thigh forward, and braking it destabilises the pelvis
            sw = HIP_R if self.swing == "r" else HIP_L
            ahead = q[sw] > q_des[sw] and q[sw] < self.q_goal[sw] + (q_des[sw] - self._traj_q[sw])
            if ahead:
                e[sw] = 0.0
                ed[sw] = max(ed[sw], 0.0)
        tau = kp * e + kd * ed
        if p.gravity_comp and self.state in (self.STEP, self.CLOSE):
            sw, kn = (HIP_R, KNEE_R) if self.swing == "r" else (HIP_L, KNEE_L)
            th_world = q[sw] - est["phi"]
            sh_world = th_world - q[kn]
            tau[sw] += p.leg_mass * 9.81 * p.leg_com * np.sin(th_world)
            tau[kn] += -p.shank_mass * 9.81 * p.shank_com * np.sin(sh_world)
        if stance_pelvis_ctrl is not None:
            # stance hip holds the pelvis (and trunk) upright in the world frame; +hip torque pitches the pelvis back
            phidot = obs["gyr_hip"][1]
            # (+hip torque with the foot planted tilts the pelvis forward -> negative feedback)
            phi_ref = (self.phi_cal or 0.0) + p.pelvis_pitch_des
            tau_pel = -p.kp_pelvis * (est["phi"] - phi_ref) - p.kd_pelvis * phidot
            st, sw = stance_pelvis_ctrl, (HIP_R if stance_pelvis_ctrl == HIP_L else HIP_L)
            if p.swing_compensation:
                # both hip torques act on the pelvis (+ = forward pitch); cancel the swing-hip reaction
                tau[st] = tau_pel - tau[sw]
            else:
                tau[st] = tau_pel + 0.3 * (p.kp[st] * (q_des[st] - q[st]))
        tau = np.clip(tau, -p.tau_max, p.tau_max)
        self.last_tau = tau
        self.log.append((self.t, self.state, self.swing, est["com_rel_mid"], est["com_rel_front"], *q_des))
        return tau
