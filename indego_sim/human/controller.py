"""Human neuromuscular controller = volition + (delayed) voluntary motor control + spinal reflexes.

    intent (VolitionGenerator)
        │  planned joint trajectories from the gait template at the *human's* phase
        ▼
    voluntary layer, every `so_period` (cortical/long-loop, delay `long_loop_delay`)
        tau_des = stiffness*(Kp e + Kd edot) + feedforward moments (+ balance strategy)
        activations = static optimisation(tau_des) bounded by capacity*effort  (SCI weakness)
        ▼
    spinal layer, every tick (stretch reflex, spasticity, clonus, spasms)
        ▼
    muscle excitation = clip(voluntary + reflex, 0, 1)  ->  MuJoCo muscle activation dynamics

Trunk: 3 lumbar torque generators track the lean intent (strength from trunk_strength).
"""
from __future__ import annotations

import dataclasses
from collections import deque

import mujoco
import numpy as np

from ..config import SubjectParams, VolitionParams
from ..model.utils import body_tilt
from ..controllers.indego_fsm import _minjerk, _minjerk_d
from .gait_library import DEFAULT_GAIT, GaitLibrary
from .muscles import LEG_DOFS, MuscleModel
from .reflexes import SpinalCord
from .sci import apply_to_model, muscle_capacity
from .volition import Intent, VolitionGenerator

# per-kg PD gains of the voluntary layer (N*m/rad/kg, N*m*s/rad/kg) [model; tuned for stability with 60 ms delay]
_KP = {"hip_flexion": 3.0, "hip_adduction": 3.0, "hip_rotation": 1.0, "knee_angle": 3.0, "ankle_angle": 3.0,
       "subtalar_angle": 0.8, "mtp_angle": 0.3}
_KD_RATIO = 0.12
_W = {"hip_flexion": 1.0, "hip_adduction": 1.0, "hip_rotation": 0.4, "knee_angle": 1.0, "ankle_angle": 1.0,
      "subtalar_angle": 0.4, "mtp_angle": 0.2}


class HumanController:
    def __init__(self, m: mujoco.MjModel, info, subject: SubjectParams, vp: VolitionParams,
                 rng: np.random.Generator, tick_dt: float, so_period: float = 0.01, schedule=None,
                 gait: GaitLibrary = DEFAULT_GAIT, balance: str = "auto", ff_gain: float = 0.6):
        self.m, self.subject, self.vp, self.dt = m, subject, vp, tick_dt
        self.mm = MuscleModel(m, info.muscles)
        self.cap = muscle_capacity(self.mm, subject)
        apply_to_model(m, self.mm, subject, self.cap)
        self.spinal = SpinalCord(self.mm, subject, tick_dt, rng, healthy_gain=1.0)
        self.has_leg_volition = bool(self.cap.max() > 0.05)
        self.vol = VolitionGenerator(vp, rng, schedule, has_leg_volition=self.has_leg_volition)
        self.gait = gait
        self.so_period = so_period
        self.ff_gain = ff_gain
        self.mass = subject.mass
        self.balance = balance
        self.dof_names = self.mm.dof_names
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.dof_names])
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.dof_names])
        base = [n[:-2] for n in self.dof_names]
        self.kp = np.array([_KP[b] for b in base]) * self.mass
        self.kd = self.kp * _KD_RATIO
        self.w = np.array([_W[b] for b in base])
        self.trunk_j = ["lumbar_flexion", "lumbar_bending", "lumbar_rotation"]
        self.trunk_q = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.trunk_j])
        self.trunk_d = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.trunk_j])
        self.trunk_act = np.array([m.actuator(f"trunk_{n.split('_')[1]}").id for n in self.trunk_j])
        self.nd = max(1, int(round(subject.long_loop_delay / tick_dt)))
        self.ndt = max(1, int(round(subject.trunk_delay / tick_dt)))
        self.torso = m.body("torso").id
        self.pelvis = m.body("pelvis").id
        self.root = m.body("Full Body").id
        self.intent = Intent()
        self.a_vol = np.zeros(self.mm.n)
        self.tau_des = np.zeros(len(self.dof_names))
        self.e_reflex = np.zeros(self.mm.n)
        self.enable = {"voluntary": True, "reflex": True, "trunk": True}
        self.postural_com = None   # set by the simulator: desired minus actual COM x (delayed), m
        self.postural_gain = 1.0   # rad of trunk lean per m of COM-behind-target [model]
        self.exo_ctx = None        # set by the simulator: {"swing": leg felt moved or None, "next_swing": side,
                                   #                        "load": {"l": N, "r": N} vertical foot loads}
        self.s_len = subject.height / 1.75
        self._jacp = np.zeros((3, m.nv))

    # ------------------------------------------------------------------------------------------
    def reset(self, d):
        self.vol.reset()
        s = self._snap(d)
        self.buf = deque([s] * (max(self.nd, self.ndt) + 2), maxlen=max(self.nd, self.ndt) + 2)
        self.next_so = d.time
        self.a_vol[:] = 0
        sp = self.spinal
        sp.reset(sp.relative_length(d), np.zeros(self.mm.n), self.mm.forces(d) / np.abs(self.mm.F0))
        self.R_torso0 = d.xmat[self.torso].reshape(3, 3).copy()
        self.R_pelvis0 = d.xmat[self.pelvis].reshape(3, 3).copy()
        self.stand_ref = d.qpos[self.qadr].copy()
        ank_x = [d.xpos[self.m.body(f"talus_{s}").id][0] for s in "lr"]
        self.stand_com_dx = float(d.subtree_com[self.root][0] - np.mean(ank_x))   # quiet-standing COM over the feet
        self.tqd_f = np.zeros(3)
        self._com_prev = None
        self.cop_int = 0.0
        nb = self.nd + 12
        self.bbuf = deque([self._body_snap(d)] * nb, maxlen=nb)
        self.ws = {"swing": "r", "t": 0.0, "on": False}
        self.exo_tau_buf = deque([np.zeros(4)] * max(1, int(round(self.subject.long_loop_delay / self.so_period))),
                                 maxlen=max(1, int(round(self.subject.long_loop_delay / self.so_period))))
        self._swing_prev, self._t_swing0, self._x_swing0 = None, 0.0, 0.0

    def _body_snap(self, d):
        """What the person senses about the whole body (vestibular/visual/proprioceptive), for the delayed buffer."""
        pitch, _ = body_tilt(d.xmat[self.pelvis].reshape(3, 3) @ self.R_pelvis0.T)
        load = (self.exo_ctx or {}).get("load", {"l": 0.0, "r": 0.0})
        return dict(pitch=float(pitch), rate=float(d.cvel[self.pelvis][1]), com=float(d.subtree_com[self.root][0]),
                    ank={s: float(d.xpos[self.m.body(f"talus_{s}").id][0]) for s in "lr"},
                    load={s: float(load[s]) for s in "lr"})

    def _snap(self, d):
        return (d.qpos[self.qadr].copy(), d.qvel[self.dadr].copy(), d.qpos[self.trunk_q].copy(),
                d.qvel[self.trunk_d].copy())

    # ------------------------------------------------------------------------------------------
    def step(self, d, heel_strikes=None, track_err=None) -> Intent:
        t = d.time
        self.intent = it = self.vol.update(t, self.dt, heel_strikes, track_err)
        self.buf.append(self._snap(d))
        if self.vp.gait_mode == "walk":
            self.bbuf.append(self._body_snap(d))
        q_del, qd_del, _, _ = self.buf[-1 - self.nd]
        _, _, tq_del, tqd_del = self.buf[-1 - self.ndt]

        if self.has_leg_volition and self.enable["voluntary"] and t >= self.next_so - 1e-12:
            self.next_so = t + self.so_period
            self.tau_des = self._voluntary_torque(d, it, q_del, qd_del)
            cap = self.cap * np.clip(it.effort * (self.vp.walk_effort if it.walking else 1.0), 0.0, 1.0)
            self.a_vol = self.mm.static_optimisation(d, self.tau_des, cap, weights=self.w, a_prev=self.a_vol)

        self.e_reflex = self.spinal.step(t, d) if self.enable["reflex"] else np.zeros(self.mm.n)
        u = np.clip(self.a_vol + self.e_reflex, 0.0, 1.0)
        d.ctrl[self.mm.ids] = u

        # trunk: intrinsic (no delay, stands in for muscle visco-elasticity and co-contraction)
        #        + long-loop feedback (delayed, low-passed velocity)
        sgt = max(self.subject.trunk_strength, 0.05)
        kpi, kdi = 3.0 * self.mass * sgt, 0.25 * self.mass * sgt
        kpd, kdd = 2.0 * self.mass * sgt, 0.10 * self.mass * sgt
        a = self.dt / (1.0 / (2 * np.pi * 5.0) + self.dt)
        self.tqd_f += a * (tqd_del - self.tqd_f)
        target = np.array([self.trunk_pitch_target(it), 0.0, 0.0])
        q_now, qd_now = d.qpos[self.trunk_q], d.qvel[self.trunk_d]
        tau_t = kpi * (target - q_now) - kdi * qd_now + kpd * (target - tq_del) - kdd * self.tqd_f
        lo, hi = self.m.actuator_ctrlrange[self.trunk_act].T
        d.ctrl[self.trunk_act] = np.clip(tau_t, lo, hi) if self.enable["trunk"] else 0.0
        return it

    def trunk_pitch_target(self, it: Intent) -> float:
        """Intended trunk pitch (rad, + forward): half the whole-body lean intent plus a postural
        strategy that leans the trunk forward when the (delayed) COM drifts back. Shared with the arms."""
        tgt = self.vp.trunk_lean_fraction * it.lean_pitch
        if self.has_leg_volition and self.vp.gait_mode in ("exo", "follow", "walk"):
            return tgt          # balances with the legs / the exo; the trunk-lean compensation is a crutch user's strategy
        if self.postural_com is not None:
            tgt += float(np.clip(self.postural_gain * self.postural_com, -0.1, 0.2))
        return tgt

    # ------------------------------------------------------------------------------------------
    def _reference(self, it: Intent):
        n = len(self.dof_names)
        qr = self.stand_ref.copy()
        qdr = np.zeros(n)
        tff = np.zeros(n)
        if not it.walking:
            return qr, qdr, tff
        amp = 1.0
        for k, name in enumerate(self.dof_names):
            b, s = name[:-2], name[-1]
            ph = it.phase_l if s == "l" else it.phase_r
            if b in self.gait.q:
                qr[k] = self.gait.angle(b, ph, amp)
                qdr[k] = self.gait.velocity(b, ph, it.phase_rate, amp)
            if b in self.gait.tau:
                tff[k] = self.ff_gain * self.mass * self.gait.moment(b, ph)
        # model sign: MyoLeg knee_angle + = flexion, ankle_angle + = dorsiflexion (same as template)
        return qr, qdr, tff

    def _voluntary_torque(self, d, it: Intent, q_del, qd_del):
        if it.walking and self.vp.gait_mode == "exo":
            return self._exo_user_torque(d, it, q_del, qd_del)
        if it.walking and self.vp.gait_mode == "follow":
            return self._follow_torque(d, q_del, qd_del)
        if it.walking and self.vp.gait_mode == "walk":
            return self._walk_torque(d, it, q_del, qd_del)
        qr, qdr, tff = self._reference(it)
        stiff = it.stiffness * (1.5 if self.vp.cooperation < 0 else 1.0)
        tau = stiff * (self.kp * (qr - q_del) + self.kd * (qdr - qd_del)) + tff
        if self._use_balance():
            tau += self._balance_torque(d, it)
        return tau

    def _ankle_strategy(self, d, it: Intent):
        """Quiet-standing ankle strategy: plantar/dorsiflexion torque from the COM position/velocity
        relative to the ankles (gain above the gravitational toppling stiffness m*g) and hip ab/adduction
        for the lateral COM (load/unload strategy)."""
        n = len(self.dof_names)
        tau = np.zeros(n)
        com = d.subtree_com[self.root]
        if self._com_prev is None:
            self._com_prev = com.copy()
        v = (com - self._com_prev) / self.so_period
        self._com_prev = com.copy()
        ank = {s: d.xpos[self.m.body(f"talus_{s}").id] for s in "lr"}
        mid = 0.5 * (ank["l"] + ank["r"])
        g = 9.81
        kx, dx = 1.5 * self.mass * g, 0.35 * self.mass * g
        x_ref = mid[0] + 0.03 + 0.9 * np.sin(it.lean_pitch)
        t_ap = -(kx * (com[0] - x_ref) + dx * v[0])               # + = dorsiflexion torque at the ankle
        ky, dy = 1.2 * self.mass * g, 0.3 * self.mass * g
        f_lat = -(ky * (com[1] - mid[1]) + dy * v[1])
        for s in "lr":
            tau[self.dof_names.index(f"ankle_angle_{s}")] += 0.5 * t_ap
            sg = 1.0 if s == "l" else -1.0
            # pushing the COM toward +y: abduct the left stance hip / adduct the right (pelvis shifts)
            tau[self.dof_names.index(f"hip_adduction_{s}")] += 0.5 * sg * f_lat * 0.09
        return tau

    def _walk_torque(self, d, it, q_del, qd_del):
        """VolitionParams.gait_mode == "walk": the person's own walking (see HumanWalkParams), from the state sensed one
        long-loop delay ago, extrapolated by `predict`; the swing-leg gravity compensation uses the current posture."""
        wp, m, ix = self.vp.walk, self.mass, self.dof_names.index
        tau = np.zeros(len(self.dof_names))
        held = [k for k, nm in enumerate(self.dof_names)
                if nm[:-2] in ("hip_adduction", "hip_rotation", "subtalar_angle", "mtp_angle")]
        tau[held] = self.kp[held] * (self.stand_ref[held] - q_del[held]) - self.kd[held] * qd_del[held]
        b, b0 = self.bbuf[-1 - self.nd], self.bbuf[-11 - self.nd]
        v = (b["com"] - b0["com"]) / (10 * self.dt)
        if wp.predict > 0:          # internal forward model: the person extrapolates the sensed state over `predict` s
            q_del = q_del + wp.predict * qd_del
            b = dict(b, com=b["com"] + wp.predict * v, pitch=b["pitch"] + wp.predict * b["rate"])
        ws = self.ws
        T = wp.swing_time

        def thigh(side):                                   # world thigh angle (+ = knee ahead of the hip)
            return q_del[ix(f"hip_flexion_{side}")] - b["pitch"]

        def pd(dof, target, k, vel=0.0):
            j = ix(dof)
            return k * m * (target - q_del[j]) + wp.kd_ratio * k * m * (vel - qd_del[j])
        if not ws["on"]:
            # gait initiation: lean forward on the ankles (standing control, COM shifted ahead) and take the first
            # step once the COM is init_dx ahead of the ankles (at the latest after init_max)
            ws.setdefault("t_init", 0.0)
            ws["t_init"] += self.so_period
            if b["com"] - 0.5 * (b["ank"]["l"] + b["ank"]["r"]) < self.stand_com_dx + wp.init_dx \
                    and ws["t_init"] < wp.init_max:
                it0 = dataclasses.replace(it, walking=False, lean_pitch=wp.init_lean)
                qr, qdr, tff = self._reference(it0)
                return it.stiffness * (self.kp * (qr - q_del) + self.kd * (qdr - qd_del)) + tff \
                    + self._balance_torque(d, it0)
            ws.update(on=True, swing="r", t=0.0, th0=thigh("r"), ph="swing")
        ws["t"] += self.so_period
        sw = ws["swing"]
        st = "l" if sw == "r" else "r"
        if ws["ph"] == "swing":
            # the swing ends at the felt touch-down once the swing foot is past the stance foot (a scuff behind it
            # does not count), at the latest after max_swing_frac
            landed = b["load"][sw] > 0.15 * m * 9.81 and b["ank"][sw] - b["ank"][st] > wp.min_step
            if (ws["t"] > wp.min_swing_frac * T and landed) or ws["t"] > wp.max_swing_frac * T:
                ws.update(ph="ds", t=0.0)
        if ws["ph"] == "ds":
            # double support: weight onto the front (just landed) foot - the trailing ankle pushes off - and the
            # trailing leg is lifted only once it has (nearly) unloaded, at the latest after ds_max
            lead, trail = sw, st
            if b["load"][trail] < wp.ds_unload * m * 9.81 or ws["t"] > wp.ds_max:
                sw, st = trail, lead
                ws.update(ph="swing", swing=sw, t=0.0, th0=thigh(sw))
            else:
                tot = max(b["load"]["l"], 0.0) + max(b["load"]["r"], 0.0) or 1.0
                tq = -(wp.kp_pelvis * m * (b["pitch"] - wp.pelvis_pitch) + wp.kd_ratio * wp.kp_pelvis * m * b["rate"])
                for side in "lr":
                    share = max(b["load"][side], 0.0) / tot
                    tau[ix(f"hip_flexion_{side}")] += share * tq
                    tau[ix(f"knee_angle_{side}")] += pd(f"knee_angle_{side}", wp.stance_knee, wp.kp_knee_st)
                    tau[ix(f"ankle_angle_{side}")] += pd(f"ankle_angle_{side}", wp.stance_ankle, wp.kp_ankle_st)
                tau[ix(f"ankle_angle_{trail}")] -= wp.ds_push * m
                return tau
        s = ws["t"] / T
        # swing hip: min-jerk from the thigh angle at lift-off to the foot-placement angle
        th_d = wp.theta0 + wp.c_d * (b["com"] - b["ank"][st]) + wp.c_v * v
        r = max(wp.swing_reach_at, 0.05)
        th_ref = ws["th0"] + (th_d - ws["th0"]) * _minjerk(s / r)
        thd_ref = (th_d - ws["th0"]) * _minjerk_d(s / r) / (r * T)
        j = ix(f"hip_flexion_{sw}")
        th, thd = thigh(sw), qd_del[j] - b["rate"]
        tau_sw = wp.kp_hip * m * (th_ref - th) + wp.kd_ratio * wp.kp_hip * m * (thd_ref - thd)
        # swing knee: (nearly) relaxed while the hip pulls the thigh forward - it folds by itself, as in human swing
        # (large active knee flexion needs the biarticular hamstrings, which extend the hip) - then extends for the
        # touch-down after knee_extend_at, once the thigh is forward of knee_ext_thigh (an earlier extension puts
        # the foot on the ground behind the body)
        if s < 1.0 and (s < wp.knee_extend_at or th < wp.knee_ext_thigh):
            tau_kn = pd(f"knee_angle_{sw}", wp.swing_knee_flex, wp.kp_knee_early)
        else:
            tau_kn = pd(f"knee_angle_{sw}", wp.swing_knee_ext, wp.kp_knee)
        # swing-leg weight (femur and tibia subtrees: thigh+shank+foot about the hip, shank+foot about the knee)
        if wp.g_comp:
            for body, dof in ((f"femur_{sw}", f"hip_flexion_{sw}"), (f"tibia_{sw}", f"knee_angle_{sw}")):
                bid = self.m.body(body).id
                mujoco.mj_jacSubtreeCom(self.m, d, self._jacp, bid)
                g_t = self.m.body_subtreemass[bid] * 9.81 * self._jacp[2, self.dadr[ix(dof)]]
                if dof.startswith("hip"):
                    tau_sw += wp.g_comp * g_t
                else:
                    tau_kn += wp.g_comp * g_t
        tau[j] += tau_sw
        tau[ix(f"knee_angle_{sw}")] += tau_kn
        tau[ix(f"ankle_angle_{sw}")] += pd(f"ankle_angle_{sw}", wp.swing_ankle, wp.kp_ankle)
        tau[ix(f"knee_angle_{st}")] += pd(f"knee_angle_{st}", wp.stance_knee, wp.kp_knee_st)
        tau[ix(f"ankle_angle_{st}")] += pd(f"ankle_angle_{st}", wp.stance_ankle, wp.kp_ankle_st) \
            - wp.push_off * m * float(s > 0.7)
        # speed: lean forward to walk faster (SIMBICON), toward v_des
        lean = wp.pelvis_pitch + float(np.clip(wp.k_lean * (wp.v_des - v), -wp.max_lean, wp.max_lean))
        tq = -(wp.kp_pelvis * m * (b["pitch"] - lean) + wp.kd_ratio * wp.kp_pelvis * m * b["rate"])
        tau[ix(f"hip_flexion_{st}")] += tq - tau_sw
        return tau

    def _follow_torque(self, d, q_del, qd_del):
        """VolitionParams.gait_mode == "follow": the person lets the exo walk them (hip/knee motion and foot placement
        are the device's) but bears weight - a loaded knee is held extended in proportion to the load on that leg -,
        regulates the walking speed with the ankles (centre of pressure ahead of the COM to slow down, behind it to
        speed up, toward `exo_walk_speed`; each foot's COP within its heel..toe, weighted by its load) and holds the
        joints the exo does not drive (hip rotation, subtalar, toes) at their standing posture."""
        ix = self.dof_names.index
        tau = np.zeros(len(self.dof_names))
        held = [k for k, nm in enumerate(self.dof_names) if nm[:-2] in ("hip_rotation", "subtalar_angle", "mtp_angle")]
        tau[held] = self.kp[held] * (self.stand_ref[held] - q_del[held]) - self.kd[held] * qd_del[held]
        load = (self.exo_ctx or {}).get("load", {"l": 1.0, "r": 1.0})
        tot = max(load["l"], 0.0) + max(load["r"], 0.0) or 1.0
        for s_ in "lr":
            k_ = ix(f"knee_angle_{s_}")
            tau[k_] += max(load[s_], 0.0) / tot * (self.kp[k_] * (self.stand_ref[k_] - q_del[k_]) - self.kd[k_] * qd_del[k_])
        # going with the device: the person feels the exo's push on the swinging leg through the cuffs and adds their own
        # effort in the same direction (`follow_assist` x the exo torque felt one reaction delay ago)
        self.exo_tau_buf.append(np.asarray((self.exo_ctx or {}).get("exo_tau", np.zeros(4)), float))
        tau_felt = self.exo_tau_buf[0]
        for s_, (jh, jk) in (("l", (0, 1)), ("r", (2, 3))):
            share = 1.0 - max(load[s_], 0.0) / tot                # 1 = unloaded (swinging) leg
            tau[ix(f"hip_flexion_{s_}")] += self.vp.follow_assist * share * tau_felt[jh]
            tau[ix(f"knee_angle_{s_}")] += self.vp.follow_assist * share * tau_felt[jk]
        com = d.subtree_com[self.root]
        if self._com_prev is None:
            self._com_prev = com.copy()
        v = (com - self._com_prev) / self.so_period
        self._com_prev = com.copy()
        p_cop = com[0] + self.vp.follow_speed_gain * (v[0] - self.vp.exo_walk_speed)
        heel, toe = 0.04 * self.s_len, 0.13 * self.s_len
        for s_ in "lr":
            a_x = d.xpos[self.m.body(f"talus_{s_}").id][0]
            tau[ix(f"ankle_angle_{s_}")] += -max(load[s_], 0.0) * (float(np.clip(p_cop, a_x - heel, a_x + toe)) - a_x)
        return tau

    # --- exo-led walking: a person with leg volition cooperating with the exo's walking mode ----------
    def _exo_user_torque(self, d, it: Intent, q_del, qd_del):
        """VolitionParams.gait_mode == "exo": exo-led, quasi-static walking. The exo drives hips and knees (no voluntary
        sagittal hip/knee torque: the person goes with it); the person does what the device cannot, with the strength
        their motor scores allow. Which leg the exo is moving is felt (`exo_ctx`, set by the simulator).
          double support: weight onto the next stance (front) foot - COM forward to its mid-foot plus the small lean
                          that requests the step, sideways over it - with both ankles, the trailing ankle pushing off
                          (heel rise lets the body move over the front foot) and the hip ab/adductors; pelvis level;
          exo step:       stance ankle keeps the COM over the stance mid-foot, stance hip abductors keep the pelvis
                          level, swing ankle lifted for clearance, swing foot placed sideways by the COM state.
        "Mid-foot" = the COM-ahead-of-ankle offset of the person's own quiet standing (measured at reset); the step is
        requested by shifting `request_lean` further."""
        n = len(self.dof_names)
        ix = self.dof_names.index
        tau = np.zeros(n)
        vp, g, m = self.vp, 9.81, self.mass
        held = [k for k, nm in enumerate(self.dof_names) if nm[:-2] in ("hip_rotation", "subtalar_angle", "mtp_angle")]
        tau[held] = self.kp[held] * (self.stand_ref[held] - q_del[held]) - self.kd[held] * qd_del[held]
        com = d.subtree_com[self.root]
        if self._com_prev is None:
            self._com_prev = com.copy()
        v = (com - self._com_prev) / self.so_period
        self._com_prev = com.copy()
        ank = {s: d.xpos[self.m.body(f"talus_{s}").id] for s in "lr"}
        pitch, roll = body_tilt(d.xmat[self.pelvis].reshape(3, 3) @ self.R_pelvis0.T)
        w = d.cvel[self.pelvis][:3]
        tq_roll = -(8.0 * m * roll + 0.8 * m * (-w[0]))           # pelvis roll -> stance hip ab/adduction
        # a person with hip muscles keeps the pelvis (and the trunk on it) upright with the stance hip(s), as in
        # walking (+ hip flexion torque with the foot planted pitches the pelvis forward)
        tq_pitch = -(8.0 * m * (pitch - 0.5 * it.lean_pitch) + 0.8 * m * w[1])
        sg = {"l": 1.0, "r": -1.0}                               # + = toward the left (+y)
        ctx = self.exo_ctx or {}
        swing = ctx.get("swing")
        if swing is None:
            stance = "l" if ctx.get("next_swing", "r") == "r" else "r"     # DS: the next stance foot
        else:
            stance = "l" if swing == "r" else "r"
        # sagittal: centre-of-pressure control about the stance (DS: next stance) foot, gravity included - the COP is
        # put ahead of the COM to slow it and behind it to let it go; each foot's COP stays between its heel and toes
        # (flat-foot balance: the person does not lift a heel to balance)
        # targets (x, relative to the stance foot; a planted foot can only carry the COP from ~4 cm behind to ~13 cm
        # ahead of its ankle, so the COM has to cross the stance foot during the exo step, not before it):
        #   parallel feet:   quiet-standing lean + `request_lean` (asks the device for the first step)
        #   split stance:    weight onto the front foot until the COM is `transfer_target` from its ankle (default 5 cm
        #                    behind it: past the device's double-support threshold), the exo then steps
        #   exo step:        the COM rolls forward from where it was at the start of the step at `exo_walk_speed`,
        #                    up to 10 cm ahead of the stance ankle
        if swing is not None and self._swing_prev != swing:
            self._t_swing0, self._x_swing0 = d.time, float(com[0])
        self._swing_prev = swing
        split = swing is None and ank[stance][0] - ank["l" if stance == "r" else "r"][0] > 0.10 * self.s_len
        if swing is not None:
            x_ref = min(self._x_swing0 + vp.exo_walk_speed * (d.time - self._t_swing0), ank[stance][0] + 0.10 * self.s_len)
        elif split:
            x_ref = ank[stance][0] + vp.transfer_target * self.s_len
        else:
            x_ref = ank[stance][0] + self.stand_com_dx + vp.request_lean
        roll_on = swing is not None and x_ref < ank[stance][0] + 0.10 * self.s_len
        # slow postural adaptation (~1 s) removes the offset that passive ankle stiffness / the AFO would leave
        self.cop_int = float(np.clip(self.cop_int + self.so_period * (com[0] - x_ref), -0.05, 0.05))
        v_ref = vp.exo_walk_speed if roll_on else 0.0
        p_cop = com[0] + 1.5 * (com[0] - x_ref) + 1.0 * self.cop_int + 0.35 * (v[0] - v_ref)
        load = {k_: max(f, 0.0) for k_, f in ctx.get("load", {"l": 0.5 * m * g, "r": 0.5 * m * g}).items()}
        heel, toe = 0.04 * self.s_len, 0.13 * self.s_len

        def ankle_cop(s_):                                       # + = dorsiflexion
            return -load[s_] * (float(np.clip(p_cop, ank[s_][0] - heel, ank[s_][0] + toe)) - ank[s_][0])
        # weight bearing: the person holds a loaded knee extended with their own quadriceps (the exo only moves the
        # swinging leg), in proportion to the load on that leg
        tot_load = sum(load.values()) or 1.0
        for s_ in "lr":
            k_ = ix(f"knee_angle_{s_}")
            tau[k_] += (load[s_] / tot_load) * (self.kp[k_] * (self.stand_ref[k_] - q_del[k_]) - self.kd[k_] * qd_del[k_])
        # sideways: COM over the stance foot, a little inside it (hip ab/adductors move the pelvis; gravity's
        # toppling stiffness is m*g per m of sideways shift, so the gain is above it)
        y_ref = ank[stance][1] - sg[stance] * 0.015
        m_lat = -(2.0 * m * g * (com[1] - y_ref) + 0.5 * m * g * v[1])        # N*m, both hips, + = toward +y
        if swing is None:
            trail = "l" if stance == "r" else "r"
            for s_ in "lr":
                tau[ix(f"ankle_angle_{s_}")] += ankle_cop(s_)
            # the weight goes forward at a controlled speed (at most exo_walk_speed, slowing down near the target)
            v_tgt = float(np.clip(vp.transfer_gain * (x_ref - com[0]), 0.0, vp.exo_walk_speed))
            drive = float(np.clip((v_tgt - v[0]) / 0.1, 0.0, 1.0))
            if ank[stance][0] - ank[trail][0] > 0.10 * self.s_len:            # split stance: trailing ankle pushes off
                tau[ix(f"ankle_angle_{trail}")] -= vp.push_off * m * drive
            for s_ in "lr":
                tau[ix(f"hip_adduction_{s_}")] += 0.5 * sg[s_] * m_lat - 0.5 * sg[s_] * tq_roll
            # weight transfer with the own hip and knee muscles: a forward push of the pelvis toward the next stance
            # foot, made by both planted legs (virtual force at the hips -> joint torques through each leg's geometry)
            f_x = float(np.clip(1.5 * m * g * (v_tgt - v[0]), -0.15 * m * g, 0.15 * m * g))
            tot = sum(load.values()) or 1.0
            push = {}
            for s_ in "lr":
                mujoco.mj_jac(self.m, d, self._jacp, None, ank[s_], self.m.body(f"talus_{s_}").id)
                push[s_] = {jn: -self._jacp[0, self.dadr[ix(jn)]] * f_x * load[s_] / tot
                            for jn in (f"hip_flexion_{s_}", f"knee_angle_{s_}")}
            # pelvis-neutral: only loaded legs can push on the pelvis, and the hip moments of the loaded legs cancel
            # (their sum is the upright control), so the pelvis translates instead of tipping; an unloaded leg's hip
            # torque would only swing that leg, so it gets none
            hip_sum = push["l"]["hip_flexion_l"] + push["r"]["hip_flexion_r"]
            for s_ in "lr":
                w_s = load[s_] / tot
                tau[ix(f"hip_flexion_{s_}")] += push[s_][f"hip_flexion_{s_}"] + w_s * (tq_pitch - hip_sum)
                tau[ix(f"knee_angle_{s_}")] += push[s_][f"knee_angle_{s_}"]
            return tau
        tau[ix(f"ankle_angle_{stance}")] += ankle_cop(stance)
        tau[ix(f"hip_adduction_{stance}")] += -sg[stance] * tq_roll
        tau[ix(f"hip_flexion_{stance}")] += tq_pitch
        k = ix(f"ankle_angle_{swing}")
        tau[k] += self.kp[k] * (np.radians(3.0) - q_del[k]) - self.kd[k] * qd_del[k]
        k = ix(f"hip_adduction_{swing}")
        dy = com[1] - ank[stance][1]
        tau[k] += self.kp[k] * (self.stand_ref[k] - q_del[k]) - self.kd[k] * qd_del[k] \
            - sg[swing] * self.kp[k] * (0.5 * dy + 0.2 * v[1])
        return tau

    # --- SIMBICON-style balance for unsupported walkers ----------------------------------------
    def _use_balance(self):
        # anyone with leg volition balances with their own legs too (crutches add to it, they do not replace it)
        return self.balance == "simbicon" or (self.balance == "auto" and self.has_leg_volition)

    def _balance_torque(self, d, it: Intent):
        """Stance hip stabilises the trunk in world frame; swing hip places the foot by COM state."""
        n = len(self.dof_names)
        tau = np.zeros(n)
        R = d.xmat[self.pelvis].reshape(3, 3) @ self.R_pelvis0.T
        pitch, roll = body_tilt(R)
        w = d.cvel[self.pelvis][:3]
        kt, dt_ = 8.0 * self.mass, 0.8 * self.mass
        tq_pitch = -(kt * (pitch - 0.5 * it.lean_pitch) + dt_ * w[1])
        tq_roll = -(kt * roll + dt_ * (-w[0]))
        if not it.walking:
            # (+hip flexion torque with the feet planted pitches the pelvis forward; +adduction of the left
            #  stance hip rolls the pelvis toward -y, of the right one toward +y)
            for s in "lr":
                tau[self.dof_names.index(f"hip_flexion_{s}")] += 0.5 * tq_pitch
                sg = 1.0 if s == "l" else -1.0
                tau[self.dof_names.index(f"hip_adduction_{s}")] += -0.5 * sg * tq_roll
            tau += self._ankle_strategy(d, it)
            return tau
        stance = "r" if self.gait.stance(it.phase_r) and not self.gait.stance(it.phase_l) else (
            "l" if self.gait.stance(it.phase_l) and not self.gait.stance(it.phase_r) else None)
        com = d.subtree_com[self.root]
        vcom = d.subtree_linvel[self.root] if d.subtree_linvel.any() else np.zeros(3)
        if stance is None:
            for s in "lr":
                tau[self.dof_names.index(f"hip_flexion_{s}")] += 0.5 * tq_pitch
            return tau
        swing = "l" if stance == "r" else "r"
        ankle = d.xpos[self.m.body(f"talus_{stance}").id]
        dx, dy = com[0] - ankle[0], com[1] - ankle[1]
        cd, cv = 0.5, 0.2
        k_sw = self.kp[self.dof_names.index(f"hip_flexion_{swing}")]
        tau[self.dof_names.index(f"hip_flexion_{swing}")] += k_sw * (cd * dx + cv * (vcom[0] - it.speed))
        sg = 1.0 if swing == "l" else -1.0
        k_sa = self.kp[self.dof_names.index(f"hip_adduction_{swing}")]
        tau[self.dof_names.index(f"hip_adduction_{swing}")] += -sg * k_sa * (cd * dy + cv * vcom[1])
        sgs = 1.0 if stance == "l" else -1.0
        tau[self.dof_names.index(f"hip_flexion_{stance}")] += tq_pitch
        tau[self.dof_names.index(f"hip_adduction_{stance}")] += -sgs * tq_roll
        return tau
