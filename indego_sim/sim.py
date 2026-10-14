"""HumanExoSim: one closed-loop session = MyoLeg human + Indego + support + your controller.

Timing (defaults):
    physics            0.5 ms   (MuJoCo, dcmotor current/thermal dynamics, delays, contacts)
    spinal / support   1 ms     (reflexes, crutch forces)
    voluntary (SO)     10 ms    (static optimisation of the voluntary drive)
    exo controller     5 ms     (Indego Explorer runs at 200 Hz) -> HumanExoSim.step()

Your controller only ever sees `obs` (onboard sensors). Everything else is logged as
ground truth for analysis (interaction forces at every cuff, human joint angles,
muscle activations, GRFs, crutch loads, intent, events).
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

import mujoco
import numpy as np

from .config import ExoParams, SensorParams, SimConfig, SubjectParams, SupportParams, VolitionParams
from .exo.drive import ExoDrive
from .exo.sensors import ExoSensors
from .human.controller import HumanController
from .human.support import Planarizer, Therapist, UpperLimbSupport
from .model.builder import build_model
from .model.utils import apply_couplings, body_tilt


class HumanExoSim:
    def __init__(self, exo: ExoParams = None, subject: SubjectParams = None, volition: VolitionParams = None,
                 support: SupportParams = None, sensors: SensorParams = None, sim: SimConfig = None,
                 schedule=None, seed: int = 0, with_exo: bool = True, model=None, log_rate_hz: float = 200.0,
                 log_muscles: bool = True, balance: str = "auto", base: str = "free", lift: float = 0.0):
        self.exo_p = exo or ExoParams()
        self.subject = subject or SubjectParams()
        self.vol_p = volition or VolitionParams()
        self.sup_p = support or SupportParams()
        if sim is not None and sim.planar:
            # sagittal-plane model: the pelvis is held in the plane too (copy: the caller's object stays as it was)
            self.sup_p = copy.deepcopy(self.sup_p)
            self.sup_p.planarizer.enabled = True
        self.sen_p = sensors or SensorParams()
        self.sim_p = sim or SimConfig()
        self.with_exo = with_exo
        self.rng = np.random.default_rng(seed)
        if model is None:
            self.m, self.info = build_model(self.exo_p, self.subject, self.sim_p, self.sen_p, with_exo=with_exo,
                                            base=base, lift=lift)
        else:
            self.m, self.info = model
            self.m = copy.copy(self.m)
        m = self.m
        self.d = mujoco.MjData(m)
        self.tick = 1.0 / self.sim_p.human_rate_hz
        self.nphys = max(1, int(round(self.tick / m.opt.timestep)))
        self.ctrl_dt = 1.0 / self.exo_p.control_rate_hz
        self.nticks = max(1, int(round(self.ctrl_dt / self.tick)))
        if abs(self.nticks * self.tick - self.ctrl_dt) > 1e-9 or abs(self.nphys * m.opt.timestep - self.tick) > 1e-9:
            raise ValueError("control period must be a multiple of the human tick, and the tick of the physics step")
        self.human = HumanController(m, self.info, self.subject, self.vol_p, self.rng, self.tick, schedule=schedule,
                                     balance=balance)
        self.human.support_kind = self.sup_p.kind
        self.BW = (self.info.human_mass) * 9.81
        self.support = UpperLimbSupport(m, self.sup_p, self.BW, self.tick, scale=self.subject.height / 1.75)
        self.therapist = Therapist(m, self.sup_p.therapist, self.tick, self.support)
        self.planarizer = Planarizer(m, self.sup_p.planarizer, self.tick)
        if with_exo:
            self.drive = ExoDrive(m, self.info, self.exo_p)
            self.sensors = ExoSensors(m, self.info, self.sen_p, self.rng, self.ctrl_dt)
            self.K = np.array([p.K for p in self.drive.mp])
        self.root = m.body("Full Body").id
        self.torso = m.body("torso").id
        self._pelvis = m.body("pelvis").id
        self.foot_bodies = {s: {m.body(f"calcn_{s}").id, m.body(f"toes_{s}").id, m.body(f"talus_{s}").id} for s in "lr"}
        self.floor_bodies = {0}
        self.log_every = max(1, int(round((1.0 / log_rate_hz) / self.tick)))
        self.log_muscles = log_muscles
        self._bushing_dofs = {k: np.array([m.jnt_dofadr[m.joint(j).id] for j in js])
                              for k, js in self.info.bushings.items()}
        self._bushing_q = {k: np.array([m.jnt_qposadr[m.joint(j).id] for j in js])
                           for k, js in self.info.bushings.items()}
        self.human_joints = ["hip_flexion_l", "hip_adduction_l", "knee_angle_l", "ankle_angle_l",
                             "hip_flexion_r", "hip_adduction_r", "knee_angle_r", "ankle_angle_r",
                             "lumbar_flexion", "lumbar_bending"]
        self._hq = np.array([m.jnt_qposadr[m.joint(j).id] for j in self.human_joints])
        if with_exo:
            self._exo_q = np.array([m.jnt_qposadr[m.joint(j).id] for j in self.info.exo_joints])
            self._exo_d = np.array([m.jnt_dofadr[m.joint(j).id] for j in self.info.exo_joints])
            self._exo_bq = [m.jnt_qposadr[m.joint(j).id] if j else None for j in self.info.exo_backlash_joints]
        self.callbacks: List[Callable] = []

    # ----------------------------------------------------------------------------------------
    def reset(self, settle: float = 0.6, hold_controller=None, qpos: Optional[np.ndarray] = None):
        """Reset to the standing keyframe and let contacts, cuffs and reflexes settle while the
        exo holds the joints (default PD hold) and the person stands still."""
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        if qpos is not None:
            d.qpos[:] = qpos
            apply_couplings(m, d.qpos)
        mujoco.mj_forward(m, d)
        if self.with_exo:
            self.sensors.reset()
            self.drive.fault = False
        self.human.reset(d)
        self.support.reset(d)
        self.therapist.reset(d)
        self.planarizer.reset(d)
        self.hs_state = {s: True for s in "lr"}
        self._swing_start = {"l": None, "r": None}
        self._next_swing = "r"
        self._current_swing = None
        self._felt_next = "r"
        self._crutch_lift_until = {"l": -1.0, "r": -1.0}
        self._unloaded_time = {"l": 0.0, "r": 0.0}
        self._load_f = {"l": self.BW / 2, "r": self.BW / 2}
        self._crutch_lift_from = {"l": -1.0, "r": -1.0}
        self._crutch_queue = []
        self._lean_wait = 0.0
        self._crutch_wait_from = 0.0
        self._crutch_next_ok = -1.0
        self._new_heel_strike = {"l": False, "r": False}
        self._settle_t = 0.0
        self._last_gait_event = -10.0
        self._crutch_moved_for = {"l": False, "r": False}
        self.hs_last = {s: -1.0 for s in "lr"}
        self.log = {}
        self._tick_count = 0
        self.fell = False
        self.events = []
        self._grf = {"l": np.zeros(3), "r": np.zeros(3)}
        self._cop = {"l": None, "r": None}
        if settle > 0:
            hold = hold_controller or _HoldPD()
            sched_backup = self.human.vol.schedule
            self.human.vol.schedule = []
            n = int(round(settle / self.ctrl_dt))
            obs = self.observe()
            for _ in range(n):
                tau = hold(obs) if self.with_exo else None
                obs = self.step(tau, record=False)
            self.human.vol.schedule = sched_backup
            self.human.vol.reset()
            self._new_heel_strike = {"l": False, "r": False}
            self.events = []
            self.therapist.work, self.therapist.peak = 0.0, 0.0
            self.planarizer.impulse, self.planarizer.peak, self.planarizer.peak_moment = 0.0, 0.0, 0.0
            self.t0 = d.time
            # shift scripted session so t=0 is the end of settling
            self.human.vol.schedule = [(t + d.time, c) for t, c in sched_backup]
        else:
            self.t0 = d.time
        self.log = {}
        return self.observe()

    def observe(self):
        if not self.with_exo:
            return {"t": self.d.time}
        obs = self.sensors.read(self.d, self.K)
        if self.sen_p.foot_switches:
            thr = self.sen_p.foot_switch_threshold * self.BW
            obs["foot_contact"] = np.array([float(self._grf[s][2] > thr) for s in "lr"])
        return obs

    # ----------------------------------------------------------------------------------------
    def step(self, tau_cmd=None, record: bool = True):
        """Advance one exo control period with joint torque command tau_cmd (hip_l, knee_l, hip_r, knee_r)."""
        m, d = self.m, self.d
        if self.with_exo:
            self.drive.command(d, np.zeros(4) if tau_cmd is None else tau_cmd)
        for _ in range(self.nticks):
            hs = self._contacts()
            err = None
            self.human.exo_ctx = {"swing": self._felt_swing(), "next_swing": self._felt_next,
                                  "load": {s: float(self._grf[s][2]) for s in "lr"},
                                  "exo_tau": self.drive.last_cmd if self.with_exo else np.zeros(4)}
            it = self.human.step(d, hs, err)
            self._update_support_targets(it)
            self.support.step(d)
            self.support.apply(d)
            d.xfrc_applied[self._pelvis] = 0.0          # pelvis rig forces (therapist, planarizer) are re-set each tick
            self.therapist.step(d)
            self.therapist.apply(d)
            self.planarizer.step(d)
            self.planarizer.apply(d)
            for cb in self.callbacks:
                cb(self, d)
            mujoco.mj_step(m, d, nstep=self.nphys)
            self._tick_count += 1
            if record and self._tick_count % self.log_every == 0:
                self._record(it)
            if not self.fell and self._fall_check(d):
                self.fell = True
                self.events.append((d.time - self.t0 if hasattr(self, "t0") else d.time, "fall"))
        return self.observe()

    # ----------------------------------------------------------------------------------------
    def _update_support_targets(self, it):
        s = self.support
        s.com_shift = it.com_shift
        s.lean_frac = it.lean_pitch / max(np.radians(self.vol_p.lean_trigger_deg), 1e-6)
        s.walk_intent += (self.tick / (0.3 + self.tick)) * ((1.0 if it.walking else 0.0) - s.walk_intent)
        if it.walking and not self.human.has_leg_volition:
            # the person knows which leg the exo moves next and sees it lift (visual/vestibular, delayed)
            swing = self._current_swing or self._next_swing
            stance = "l" if swing == "r" else "r"
            sw_knee = self.d.qpos[self.m.jnt_qposadr[self.m.joint(f"knee_angle_{swing}").id]]
            target = 1.0 if (self._current_swing is not None or sw_knee > np.radians(8)) else self.vol_p.anticipation
            if self._crutch_phase() and target < 1.0:
                target = 0.0           # land, re-set the arms and move the crutches before transferring weight
            s.expected_stance = stance
            s.anticipation += (self.tick / (0.15 + self.tick)) * (target - s.anticipation)
        else:
            s.expected_stance = None
            s.anticipation = 0.0
        # "lean until it steps": fully leaned, nothing happening -> lean a little further; relax once the exo
        # starts the step (the person feels / sees the swing leg go)
        exo_moving = False
        if it.walking and not self.human.has_leg_volition:
            sw = self._current_swing or self._next_swing
            knee = self.d.qpos[self.m.jnt_qposadr[self.m.joint(f"knee_angle_{sw}").id]]
            exo_moving = self._current_swing is not None or knee > np.radians(8)
        waiting = it.walking and not self.human.has_leg_volition and not exo_moving and not s.moving_crutches \
            and s.lean_frac >= 0.95
        self._lean_wait = self._lean_wait + self.tick if waiting else 0.0
        if waiting and self._lean_wait > self.vol_p.lean_patience:
            s.extra_lean = min(s.extra_lean + self.vol_p.lean_ramp * self.tick, self.vol_p.lean_extra_max)
        elif not waiting:
            s.extra_lean = max(0.0, s.extra_lean - 0.3 * self.tick)
        stepping = (self.d.time - self._last_gait_event) < 2.5 if it.walking else False
        # perceived foot loads (proprioception/pressure, low-passed ~4 Hz)
        a_f = self.tick / (self.tick + 1.0 / (2 * np.pi * 4.0))
        for side in "lr":
            self._load_f[side] += a_f * (max(self._grf[side][2], 0.0) - self._load_f[side])
        fl, fr = self._load_f["l"], self._load_f["r"]
        s.foot_load = {"l": fl, "r": fr}
        share = min(fl, fr) / max(0.5 * (fl + fr), 1e-6)            # 1 = even double support, 0 = single
        ss_target = float(np.clip(1.0 - share / 0.4, 0.0, 1.0))
        s.single_support += (self.tick / (0.08 + self.tick)) * (ss_target - s.single_support)
        for side, f in (("l", fl), ("r", fr)):
            unloaded = f < 0.10 * max(fl + fr, 1e-6)
            self._unloaded_time[side] = self._unloaded_time[side] + self.tick if unloaded else 0.0
            if not unloaded:
                self._crutch_moved_for[side] = False
        s.target_speed = it.speed if (stepping or exo_moving) else 0.0
        st = s.buf[0] if hasattr(s, "buf") else None
        dbg = getattr(s, "debug", None)
        if dbg is not None and st is not None:
            # same COM target the arms work toward (incl. single-support / crutch-move logic), delayed
            self.human.postural_com = dbg["x_des"] - st["com"][0]
        elif st is not None:
            self.human.postural_com = (st["bos"][0] + self.sup_p.com_ahead + it.com_shift) - st["com"][0]
        s.target_pitch = self.human.trunk_pitch_target(it)
        # lateral weight shift. People with leg volition shift with their own gait phase; exo users
        # learn the alternation and shift onto the next stance leg before/through each exo step.
        if not it.walking:
            target_bias = 0.0
        elif self.human.has_leg_volition:
            target_bias = 0.04 * np.sin(2 * np.pi * it.phase_r)
        elif self._crutch_phase():
            target_bias = 0.0          # centred while the crutches are advanced after a step
        else:
            target_bias = self.vol_p.weight_shift * (1.0 if self._next_swing == "r" else -1.0)
        a = self.tick / (self.vol_p.weight_shift_time + self.tick)
        s.lateral_bias += a * (target_bias - s.lateral_bias)
        if self.sup_p.kind in ("crutches", "walker") and self.sup_p.four_point_lift:
            t = self.d.time
            pat = self.sup_p.crutch_pattern if self.sup_p.kind == "crutches" else "walker"
            if pat in ("2point", "4point"):
                # a crutch moves when the person feels a leg come off the ground; never both at once
                for side in "lr":
                    if self._unloaded_time[side] > 0.12 and not self._crutch_moved_for[side]:
                        c = side if pat == "4point" else ("r" if side == "l" else "l")
                        other = "l" if c == "r" else "r"
                        if not s.lifted[other]:
                            self._crutch_lift_until[c] = t + self.sup_p.crutch_swing_time
                            self._crutch_lift_from[c] = t
                            self._crutch_moved_for[side] = True
            else:
                # "ds" (crutches advanced one after the other) or "walker" (lifted as a unit) once both feet are
                # settled after a step. The person only lifts when it feels safe: nothing else in the air and the
                # COM not ahead of where they want it (the remaining support must be able to hold it).
                side = self._settled_after_step() if it.walking else None
                if side is not None:
                    first = "r" if side == "l" else "l"
                    # the crutch on the side of the leg that just stepped goes first: the weight is still on
                    # the other side, so the remaining (other) crutch can hold the body
                    self._crutch_queue = ["lr"] if pat == "walker" else [side, first]
                    self._crutch_next_ok = max(self._crutch_next_ok, t + self.sup_p.reaction_delay)
                    self._crutch_wait_from = t
                busy = any(self._crutch_lift_from[c] <= t < self._crutch_lift_until[c] for c in "lr")
                if self._crutch_queue and not busy and t >= self._crutch_next_ok:
                    dbg = getattr(s, "debug", None)
                    safe = dbg is None or (dbg["x_des"] - dbg["com"][0] > -self.sup_p.lift_margin)
                    if not safe and pat != "walker":
                        # COM ahead of where the person wants it: still move a crutch that has ended up under the
                        # shoulder (it cannot stop a forward fall) or after waiting too long
                        c0 = self._crutch_queue[0]
                        safe = (s.tips[c0][0] - s.hand(self.d, c0)[0] < self.sup_p.tip_useless * s.s) or \
                            (t - self._crutch_wait_from > self.sup_p.crutch_wait_max)
                    if safe and pat != "walker":
                        safe = self._lateral_safe(self._crutch_queue[0])
                    if safe:
                        cs = self._crutch_queue.pop(0)
                        dur = 0.3 if pat == "walker" else self.sup_p.crutch_swing_time
                        for c in cs:
                            self._crutch_lift_from[c], self._crutch_lift_until[c] = t, t + dur
                        self._crutch_next_ok = t + dur + 0.1
            s.moving_crutches = bool(self._crutch_queue) or any(
                self._crutch_lift_from[c] <= t < self._crutch_lift_until[c] for c in "lr")
            up = [c for c in "lr" if self._crutch_lift_from[c] <= t < self._crutch_lift_until[c]]
            s.pending_crutch = up[0] if up else (self._crutch_queue[0] if self._crutch_queue and pat != "walker"
                                                 else None)
            for c in "lr":
                lifted = self._crutch_lift_from.get(c, -1.0) <= t < self._crutch_lift_until.get(c, -1.0)
                if s.lifted[c] and not lifted:
                    s.plant(self.d, c)
                s.lifted[c] = lifted
                s.side_capacity[c] = 0.0 if lifted else 1.0
        else:
            s.side_capacity = {"l": 1.0, "r": 1.0}
        # re-plant a crutch that ended up far behind the hand (e.g. walker / 2-point gait)
        for side in "lr":
            if s.enabled and s.sp.kind != "harness" and not s.lifted[side]:
                if s.hand(self.d, side)[0] - s.tips[side][0] > 0.15:
                    s.plant(self.d, side)

    def _felt_swing(self):
        """The leg a person with sensation feels the exo moving: knee flexed (> 12 deg) by the exo while that foot carries
        little of the weight (< 30 %); a stance knee that yields under load is not a step. None in double support -
        including before the moved foot is loaded, so the weight transfer can start at touch-down. Also tracks the
        alternation (`_felt_next`), as the person does."""
        kn = {s: self.d.qpos[self.m.jnt_qposadr[self.m.joint(f"knee_angle_{s}").id]] for s in "lr"}
        load = {s: max(float(self._grf[s][2]), 0.0) for s in "lr"}
        tot = load["l"] + load["r"] + 1e-6
        moved = [s for s in "lr" if kn[s] > np.radians(12.0) and load[s] < 0.3 * tot]
        sw = moved[0] if len(moved) == 1 else None
        if sw is not None:
            self._felt_next = "l" if sw == "r" else "r"
        return sw

    def _crutch_phase(self):
        """After a step (crutch pattern "ds"): from heel strike until both crutches have been advanced."""
        if self.sup_p.kind != "crutches" or self.sup_p.crutch_pattern != "ds" or not self.sup_p.four_point_lift:
            return False
        return any(self._new_heel_strike.values()) or bool(self._crutch_queue) or any(self.support.lifted.values())

    def _lateral_safe(self, c):
        """The person lifts crutch `c` only when the (perceived) COM is inside the opposite foot laterally."""
        s = self.support
        st = s.buf[0]
        y = st["com"][1]
        mg = self.sup_p.lift_lat_margin * s.s
        return y <= st["foot"]["l"][1] - mg if c == "l" else y >= st["foot"]["r"][1] + mg

    def _settled_after_step(self):
        """Return the side of the last step once both feet have felt loaded for 0.25 s afterwards."""
        for side in "lr":
            if self._new_heel_strike.get(side):
                both = min(self._load_f["l"], self._load_f["r"]) > 0.2 * self.BW
                self._settle_t = self._settle_t + self.tick if both else 0.0
                if self._settle_t > 0.25:
                    self._new_heel_strike[side] = False
                    self._settle_t = 0.0
                    return side
        return None

    def _contacts(self):
        m, d = self.m, self.d
        grf = {"l": np.zeros(3), "r": np.zeros(3)}
        mom = {"l": np.zeros(3), "r": np.zeros(3)}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for side in "lr":
                fb = self.foot_bodies[side]
                if (b1 in fb and b2 == 0) or (b2 in fb and b1 == 0):
                    mujoco.mj_contactForce(m, d, i, f6)
                    R = c.frame.reshape(3, 3)
                    fw = R.T @ f6[:3]
                    if b1 in fb:
                        fw = -fw
                    grf[side] += fw
                    mom[side] += c.pos * max(fw[2], 0.0)
        self._grf = grf
        for side in "lr":
            if grf[side][2] > 0.02 * self.BW:
                cop = mom[side] / max(grf[side][2], 1e-6)
                self._cop[side] = cop if self._cop[side] is None else self._cop[side] + 0.2 * (cop - self._cop[side])
            else:
                self._cop[side] = None
        self.support.cop = dict(self._cop)
        hs = {}
        thr_on, thr_off = 0.15 * self.BW, 0.03 * self.BW
        for side in "lr":
            fz = grf[side][2]
            hs[side] = False
            if not self.hs_state[side] and fz > thr_on and d.time - self.hs_last[side] > 0.3:
                self.hs_state[side] = True
                self.hs_last[side] = d.time
                hs[side] = True
            elif self.hs_state[side] and fz < thr_off:
                self.hs_state[side] = False
                self._swing_start[side] = d.time
                self._current_swing = side
                self._last_gait_event = d.time
                self.events.append((d.time - getattr(self, "t0", 0.0), f"toe_off_{side}"))
            if hs[side]:
                self.events.append((d.time - getattr(self, "t0", 0.0), f"heel_strike_{side}"))
                # a real step (not a load flicker) = the foot was off the ground for a while
                ss = self._swing_start[side]
                if ss is not None and d.time - ss > 0.25:      # a load flicker must not cancel a pending step
                    self._new_heel_strike[side] = True
                self._last_gait_event = d.time
                if self._current_swing == side:
                    self._current_swing = None
                self._next_swing = "l" if side == "r" else "r"
        return hs

    def _fall_check(self, d):
        h = d.subtree_com[self.root][2]
        pitch, roll = body_tilt(d.xmat[self.torso].reshape(3, 3) @ self.human.R_torso0.T)
        return h < 0.6 * self.support.h_nom or abs(pitch) > np.radians(55) or abs(roll) > np.radians(45)

    # ----------------------------------------------------------------------------------------
    def interaction_wrenches(self):
        """{bushing: [Fx, Fy, Fz, Mx, My, Mz]} applied BY the exo ON the human, in exo-segment axes."""
        d = self.d
        out = {}
        for k, dofs in self._bushing_dofs.items():
            out[k] = d.qfrc_spring[dofs] + d.qfrc_damper[dofs]
        return out

    def exo_angles(self):
        q = self.d.qpos[self._exo_q].copy()
        for k, a in enumerate(self._exo_bq):
            if a is not None:
                q[k] += self.d.qpos[a]
        return q

    def _record(self, it):
        d, L = self.d, self.log
        t = d.time - getattr(self, "t0", 0.0)

        def put(k, v):
            L.setdefault(k, []).append(np.array(v, dtype=float) if np.ndim(v) else float(v))
        put("t", t)
        put("human_q", d.qpos[self._hq])
        put("com", d.subtree_com[self.root])
        pitch, roll = body_tilt(d.xmat[self.torso].reshape(3, 3) @ self.human.R_torso0.T)
        put("torso_pitch", pitch)
        put("torso_roll", roll)
        put("grf_l", self._grf["l"])
        put("grf_r", self._grf["r"])
        put("support_force", self.support.F.copy())
        put("therapist_force", self.therapist.F.copy())
        if self.planarizer.enabled:
            put("planarizer_force", self.planarizer.F.copy())
            put("planarizer_moment", self.planarizer.M.copy())
        put("intent_walk", float(it.walking))
        put("intent_phase_r", it.phase_r)
        put("intent_lean", it.lean_pitch)
        put("effort", it.effort)
        put("reflex_mean", self.human.e_reflex.mean())
        if self.log_muscles:
            put("muscle_act", d.act[self.m.actuator_actadr[self.human.mm.ids]])
            put("muscle_vol", self.human.a_vol)
            put("muscle_reflex", self.human.e_reflex)
        if self.with_exo:
            put("exo_q", self.exo_angles())
            put("exo_qd", d.qvel[self._exo_d])
            put("exo_cmd", self.drive.last_cmd)
            put("exo_tau", self.drive.joint_torque(d))
            put("exo_current", self.drive.current(d))
            put("exo_temp", self.drive.winding_temp(d))
            put("exo_power", self.drive.power(d))
            for k, w in self.interaction_wrenches().items():
                put(f"F_{k}", w)

    def get_log(self) -> Dict[str, np.ndarray]:
        out = {k: np.asarray(v) for k, v in self.log.items()}
        out["events"] = np.array(self.events + [(t - getattr(self, "t0", 0.0), f"spasm_{k}_{s}")
                                                for t, k, s, a, du in self.human.spinal.spasm_events], dtype=object)
        out["muscle_names"] = np.array(self.human.mm.names)
        out["human_joint_names"] = np.array(self.human_joints)
        return out

    def run(self, controller, duration: float, progress: bool = False):
        """Convenience loop: controller(obs) -> tau."""
        obs = self.observe()
        n = int(round(duration / self.ctrl_dt))
        for k in range(n):
            tau = controller(obs) if self.with_exo else None
            obs = self.step(tau)
            if self.fell:
                break
            if progress and k % int(1 / self.ctrl_dt) == 0:
                print(f"t={self.d.time - self.t0:6.2f}s")
        return self.get_log()


class _HoldPD:
    """Default posture hold used during settling: hips/knees at 0 with gravity-level gains."""

    def __init__(self, kp=(200, 200, 200, 200), kd=(8, 8, 8, 8), q0=None):
        self.kp, self.kd = np.array(kp, float), np.array(kd, float)
        self.q0 = np.zeros(4) if q0 is None else np.asarray(q0)

    def __call__(self, obs):
        return self.kp * (self.q0 - obs["q"]) - self.kd * obs["qd"]
