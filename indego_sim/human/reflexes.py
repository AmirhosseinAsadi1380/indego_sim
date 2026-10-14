"""Spinal layer: segmental stretch reflexes, spasticity, clonus and spasms.

Each muscle i gets an excitation from delayed proprioceptive signals:

* normal short-latency stretch reflex (small gains; healthy and incomplete SCI)
* spastic hyper-reflexia, tonic-stretch-reflex-threshold (TSRT) model
  (Levin & Feldman 1994; Jobin & Levin 2000): the reflex fires when the muscle's
  relative length exceeds a velocity-dependent threshold
        p_i(t - d_i) > lambda_i - mu_i * pdot_i(t - d_i)
  where p is the position in the muscle's own length range (0 = shortest over the
  joint ROM, 1 = longest).  Lower lambda / higher mu / gain = more spastic.
* optional force/force-rate feedback (Falisse et al. 2018, which found it explains
  responses to passive stretch better than velocity feedback alone)
* clonus emerges from delay + high gain (no special casing); plantarflexors get
  an extra gain factor
* spasms: Poisson-timed (plus stretch-triggered) extensor or flexor bursts

All parameters derived from Modified Ashworth scores are ESTIMATES meant to be
calibrated per participant (see sysid/spasticity_calibration.py).
"""
from __future__ import annotations

import numpy as np

from ..config import MUSCLE_GROUPS, SubjectParams
from .muscles import MuscleModel

# delays (s): afferent + efferent conduction + synaptic, by muscle location
_DELAY = {
    "hip": 0.020, "thigh": 0.025, "shank": 0.035,
}
_LOCATION = {
    **{b: "hip" for b in ["iliacus", "psoas", "glmax1", "glmax2", "glmax3", "glmed1", "glmed2", "glmed3", "glmin1",
                          "glmin2", "glmin3", "piri", "addbrev", "addlong", "addmagDist", "addmagIsch", "addmagMid",
                          "addmagProx", "tfl"]},
    **{b: "thigh" for b in ["vasint", "vaslat", "vasmed", "recfem", "bflh", "bfsh", "semimem", "semiten", "grac", "sart"]},
    **{b: "shank" for b in ["soleus", "gaslat", "gasmed", "tibant", "tibpost", "perbrev", "perlong", "fdl", "fhl",
                            "edl", "ehl"]},
}
# MAS -> (lambda_p, mu_p [s], gain, tonic)   [est]
_MAS_TABLE = np.array([
    # MAS, lambda (static threshold, fraction of the muscle's length range over the joint ROM), mu [ROM-fraction
    # per (L0/s)], gain, tonic. The static threshold stays near end-range for mild spasticity; the velocity term
    # makes a fast clinical stretch (~2 optimal-fibre-lengths/s) catch at lambda - 2*mu.  [est; calibrate with
    # sysid/spasticity.py - e.g. MAS 1.5 quadriceps gives a pendulum-test R2 of ~1.15 vs ~1.45 relaxed]
    [0.0, 2.00, 0.00, 0.0, 0.00],
    [1.0, 0.98, 0.10, 1.5, 0.00],
    [1.5, 0.96, 0.14, 2.0, 0.005],
    [2.0, 0.93, 0.18, 2.5, 0.01],
    [3.0, 0.85, 0.22, 3.0, 0.02],
    [4.0, 0.72, 0.28, 4.0, 0.04],
])
EXTENSOR_SPASM = ["hip_extensors", "hip_adductors", "knee_extensors", "ankle_plantarflexors"]
FLEXOR_SPASM = ["hip_flexors", "knee_flexors", "ankle_dorsiflexors"]


def mas_params(mas: float):
    t = _MAS_TABLE
    return tuple(np.interp(mas, t[:, 0], t[:, k]) for k in range(1, 5))


class SpinalCord:
    def __init__(self, mm: MuscleModel, subject: SubjectParams, dt: float, rng: np.random.Generator,
                 healthy_gain: float = 1.0):
        self.mm, self.dt, self.rng = mm, dt, rng
        n = mm.n
        sp = subject.spasticity
        self.delay_s = np.array([_DELAY[_LOCATION.get(b, "thigh")] for b in mm.base]) * sp.delay_scale
        self.delay = np.maximum(1, np.round(self.delay_s / dt).astype(int))
        self.nbuf = int(self.delay.max()) + 2
        # normal stretch reflex (small): velocity and length components
        # relaxed-muscle values: with 0.03/(L0/s) a healthy pendulum test was over-damped (R2 ~1.0); 0.005
        # gives R2 ~1.45 and several oscillations (the MyoLeg knee range limit of 120 deg caps R2 at ~1.4)
        self.g_vel_n = np.full(n, 0.005 * healthy_gain)   # excitation per (L0/s) of stretch velocity [est/calib]
        self.g_len_n = np.full(n, 0.03 * healthy_gain)    # per unit relative length beyond 0.9 [est]
        # spastic TSRT parameters per muscle from MAS of its group(s)
        self.lam = np.full(n, 2.0)
        self.mu = np.zeros(n)
        self.k_sp = np.zeros(n)
        self.tonic = np.zeros(n)
        mas_by_group = dict(sp.mas)
        for i in range(n):
            groups = mm.group_of.get(i, [])
            mas = max([mas_by_group.get(g, mas_by_group.get(f"{g}_{mm.side[i]}", 0.0)) for g in groups] + [0.0])
            lam, mu, k, tone = mas_params(mas)
            self.lam[i], self.mu[i], self.k_sp[i], self.tonic[i] = lam, mu, k, tone
            if "ankle_plantarflexors" in groups:
                self.k_sp[i] *= sp.clonus_gain_boost
        self.use_tsrt = sp.model in ("tsrt", "both")
        self.use_force = sp.model in ("force_rate", "both")
        self.g_force = np.where(self.k_sp > 0, 0.3, 0.0)        # [est] per unit F/F0 above 0.1
        self.g_frate = np.where(self.k_sp > 0, 0.03, 0.0)       # [est] per (F/F0)/s
        # spasm process
        self.spasm_rate = sp.spasm_rate_per_min / 60.0
        self.spasm_trigger = sp.spasm_stretch_trigger
        self.spasm_amp_rng, self.spasm_dur_rng = sp.spasm_amplitude, sp.spasm_duration
        self.spasm = None
        self.spasm_events = []
        self.max_reflex = 0.6        # spastic reflex EMG rarely approaches maximal activation [est]
        self.reset()

    def reset(self, p=None, pdot=None, f=None):
        n = self.mm.n
        self.buf_p = np.tile(p if p is not None else np.full(n, 0.5), (self.nbuf, 1))
        self.buf_v = np.tile(pdot if pdot is not None else np.zeros(n), (self.nbuf, 1))
        self.buf_f = np.tile(f if f is not None else np.zeros(n), (self.nbuf, 1))
        self.head = 0
        self.spasm = None
        self.last_out = np.zeros(n)
        self.v_f = np.zeros(n)

    def relative_length(self, d):
        """Position in the muscle's length range over the anatomical joint ROM (0 short, 1 long), i.e. the
        quantity a clinician's passive stretch explores - the scale of the TSRT thresholds in _MAS_TABLE."""
        return self.mm.rom_fraction(d)

    def relative_velocity(self, d):
        """Spindle velocity signal: fibre velocity in optimal lengths/s (lengthening +)."""
        return self.mm.norm_velocity(d)

    def step(self, t, d, gate=None):
        mm = self.mm
        p = self.relative_length(d)
        v_raw = self.relative_velocity(d)
        a = self.dt / (self.dt + 1.0 / (2 * np.pi * 25.0))      # spindle/afferent low-pass ~25 Hz
        self.v_f += a * (v_raw - self.v_f)
        v = self.v_f
        f = mm.forces(d) / np.abs(mm.F0)
        self.head = (self.head + 1) % self.nbuf
        self.buf_p[self.head], self.buf_v[self.head], self.buf_f[self.head] = p, v, f
        idx = (self.head - self.delay) % self.nbuf
        cols = np.arange(mm.n)
        pd_, vd_, fd_ = self.buf_p[idx, cols], self.buf_v[idx, cols], self.buf_f[idx, cols]
        fprev = self.buf_f[(idx - 1) % self.nbuf, cols]
        fdot = (fd_ - fprev) / self.dt

        e = self.g_vel_n * np.maximum(0.0, vd_) + self.g_len_n * np.maximum(0.0, pd_ - 0.9)
        if self.use_tsrt:
            e += self.k_sp * np.maximum(0.0, pd_ - (self.lam - self.mu * vd_))
        if self.use_force:
            e += self.g_force * np.maximum(0.0, fd_ - 0.1) + self.g_frate * np.maximum(0.0, fdot)
        e += self.tonic
        if gate is not None:
            e = e * gate
        e += self._spasm(t, vd_)
        self.last_out = np.clip(e, 0.0, self.max_reflex)
        return self.last_out

    # --- spasms --------------------------------------------------------------------------------
    def _spasm(self, t, vd):
        mm = self.mm
        if self.spasm is None:
            trig = self.rng.random() < self.spasm_rate * self.dt
            if not trig and self.spasm_trigger > 0:
                # rapid stretch of hip flexors (hip extension) is a classic extensor-spasm trigger
                hf = mm.group_indices("hip_flexors")
                if len(hf) and vd[hf].max() > 1.0 and self.rng.random() < self.spasm_trigger * self.dt * 20:
                    trig = True
            if trig:
                kind = "extensor" if self.rng.random() < 0.6 else "flexor"
                side = self.rng.choice(["l", "r", "both"], p=[0.35, 0.35, 0.3])
                groups = EXTENSOR_SPASM if kind == "extensor" else FLEXOR_SPASM
                idx = [i for g in groups for i in mm.group_indices(g) if side == "both" or mm.side[i] == side]
                amp = self.rng.uniform(*self.spasm_amp_rng)
                dur = self.rng.uniform(*self.spasm_dur_rng)
                self.spasm = dict(t0=t, dur=dur, amp=amp, idx=np.array(sorted(set(idx))), kind=kind, side=side)
                self.spasm_events.append((t, kind, side, amp, dur))
            return 0.0
        s = self.spasm
        tau = t - s["t0"]
        if tau > s["dur"]:
            self.spasm = None
            return 0.0
        ramp = min(1.0, tau / 0.15, (s["dur"] - tau) / 0.2)
        trem = 1.0 + 0.3 * np.sin(2 * np.pi * 6.0 * tau)       # clonic modulation ~6 Hz
        out = np.zeros(mm.n)
        out[s["idx"]] = s["amp"] * max(ramp, 0.0) * trem
        return out
