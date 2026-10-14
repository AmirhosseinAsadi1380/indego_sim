"""Human volition: what the person *intends*, generated independently of the exoskeleton.

The intent is the part of a real session your controller can never see directly but
always feels: when the person wants to go, how fast, how stiffly they hold their own
plan, whether they lead, follow or resist the device, and the spurious leans and
hesitations that cause false triggers.

Model
-----
* A scripted *session* (list of (time, command)) plus stochastic events.
* An internal gait oscillator (phase 0 = right heel strike) with stride-to-stride
  variability and timing jitter, entrained to the actual heel strikes with a strength
  set by `cooperation` (cooperative people adapt to the device, resisting people don't).
* For people without leg volition (complete SCI) intent shows up as trunk lean and
  crutch use: the lean target rises with `lean_rise_time` when they want to walk
  (the Indego steps on forward lean) and returns upright to stop.
* Effort "slacking": effort decays while the device does the work well.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ..config import VolitionParams


@dataclass
class Intent:
    walking: bool = False
    phase_l: float = 0.0
    phase_r: float = 0.0
    phase_rate: float = 0.0           # strides / s
    lean_pitch: float = 0.0           # rad, desired trunk pitch (+ forward)
    lean_roll: float = 0.0            # rad, (+ toward left)
    com_shift: float = 0.0            # m, desired forward COM shift over the feet
    speed: float = 0.0                # m/s intended
    effort: float = 1.0
    stiffness: float = 1.0
    events: List[str] = field(default_factory=list)


def _wrap(x):
    return (x + 0.5) % 1.0 - 0.5


class VolitionGenerator:
    def __init__(self, vp: VolitionParams, rng: np.random.Generator, schedule: List[Tuple[float, str]] = None,
                 has_leg_volition: bool = True, step_length: float = 0.55):
        self.vp, self.rng = vp, rng
        self.schedule = sorted(schedule if schedule is not None else [(1.0, "walk")])
        self.has_leg_volition = has_leg_volition
        self.step_length = step_length
        self.reset()

    def reset(self):
        self.mode = "stand"
        self.stopping = False
        self.phase = 0.0
        self.stride_T = 120.0 / self.vp.cadence_spm
        self.cur_T = self.stride_T
        self.lean = 0.0
        self.lean_spurious_until = -1.0
        self.effort = self.vp.effort
        self.sched_i = 0
        self.log = []
        self.cadence_spm = self.vp.cadence_spm

    def _command(self, cmd, t):
        if cmd == "walk" and self.mode != "walk":
            self.mode = "walk"
            self.stopping = False
            self.phase = 0.52            # start with right leg entering swing
            self._new_stride()
        elif cmd == "stop" and self.mode == "walk":
            self.stopping = True
        elif cmd.startswith("cadence="):
            self.cadence_spm = float(cmd.split("=")[1])
            self.stride_T = 120.0 / self.cadence_spm
        elif cmd == "lean":
            self.lean_spurious_until = t + self.rng.uniform(0.5, 1.0)
        self.log.append((t, cmd))

    def _new_stride(self):
        vp = self.vp
        self.cur_T = self.stride_T * max(0.5, 1.0 + vp.cadence_cv * self.rng.standard_normal())

    def update(self, t: float, dt: float, heel_strikes: dict = None, assist_tracking_err: float = None) -> Intent:
        vp = self.vp
        events = []
        while self.sched_i < len(self.schedule) and self.schedule[self.sched_i][0] <= t:
            self._command(self.schedule[self.sched_i][1], t)
            events.append(self.schedule[self.sched_i][1])
            self.sched_i += 1
        if self.mode == "walk" and vp.stop_prob_per_step > 0 and self.rng.random() < vp.stop_prob_per_step * dt / self.cur_T * 2:
            self.stopping = True
            events.append("random_stop")
        if self.mode != "walk" and vp.random_lean_rate_per_min > 0 and self.rng.random() < vp.random_lean_rate_per_min / 60 * dt:
            self._command("lean", t)
            events.append("spurious_lean")

        # oscillator
        if self.mode == "walk":
            prev = self.phase
            self.phase += dt / self.cur_T
            if heel_strikes:
                k = 0.3 + 0.6 * max(0.0, vp.cooperation)
                off = vp.phase_offset_s / self.cur_T
                for side, hs in heel_strikes.items():
                    if hs:
                        ph_s = self.phase if side == "r" else self.phase + 0.5
                        err = _wrap(off + vp.phase_jitter_s * self.rng.standard_normal() / self.cur_T - ph_s)
                        self.phase += k * err
            if np.floor(self.phase) > np.floor(prev):
                self._new_stride()
            # stop at the next double support after a stop request
            ph = self.phase % 1.0
            if self.stopping and (abs(_wrap(ph - 0.0)) < 0.03 or abs(_wrap(ph - 0.5)) < 0.03):
                self.mode = "stand"
                self.stopping = False
                events.append("stopped")

        # lean intent (Indego: lean forward to step, upright to stop)
        want = vp.lean_trigger_deg if (self.mode == "walk" and not self.stopping) else 0.0
        if t < self.lean_spurious_until:
            want = max(want, vp.lean_trigger_deg * 1.2)
        if self.has_leg_volition and self.mode == "walk":
            want = min(want, 4.0)   # able walkers lean only slightly
        a = dt / max(vp.lean_rise_time, 1e-3)
        self.lean += a * (np.radians(want) - self.lean)

        # slacking
        if vp.slack_rate > 0 and assist_tracking_err is not None:
            target = 0.2 if assist_tracking_err < 0.05 else vp.effort
            self.effort += vp.slack_rate * dt * (target - self.effort)

        ph_r = self.phase % 1.0
        ph_l = (self.phase + 0.5) % 1.0
        walking = self.mode == "walk"
        rate = 1.0 / self.cur_T if walking else 0.0
        roll = np.radians(3.0) * np.sin(2 * np.pi * ph_r) if walking else 0.0  # weight shift toward stance side
        speed = 2 * self.step_length * rate if self.has_leg_volition else (vp.expected_speed if walking else 0.0)
        return Intent(walking=walking, phase_l=ph_l, phase_r=ph_r, phase_rate=rate, lean_pitch=self.lean,
                      lean_roll=roll, com_shift=vp.com_shift_gain * np.sin(self.lean), speed=speed,
                      effort=self.effort, stiffness=vp.stiffness_scale, events=events)
