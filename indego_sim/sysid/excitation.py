"""Excitation signals for identifying the Indego joint drives (and any other plant).

All functions return arrays sampled at `fs` (the rate at which you can command the drive,
200 Hz on the Indego research interface). Amplitudes are in whatever unit you command
(joint torque [N*m] or motor current [A]).

Practical guidance (see docs/CALIBRATION.md):
* chirp / multisine around a small bias torque that keeps the transmission loaded on one
  flank (avoids backlash/stiction dominating the FRF); repeat with zero bias to *see* them;
* PRBS for time-domain grey-box fitting (rich in frequency, constant amplitude);
* slow triangular velocity sweeps (under velocity control) for the friction-velocity map;
* torque steps / ramps for breakaway (static) friction and command latency.
"""
from __future__ import annotations

import numpy as np


def chirp(fs: float, T: float, f0: float = 0.2, f1: float = 15.0, amp: float = 3.0, bias: float = 0.0,
          log: bool = True, taper: float = 0.5) -> np.ndarray:
    """Swept sine from f0 to f1 Hz over T seconds (logarithmic by default) with cosine tapers."""
    t = np.arange(int(round(T * fs))) / fs
    if log:
        k = (f1 / f0) ** (1.0 / T)
        phase = 2 * np.pi * f0 * (k ** t - 1.0) / np.log(k)
    else:
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t ** 2)
    u = amp * np.sin(phase)
    return bias + u * _taper(len(t), int(taper * fs))


def multisine(fs: float, T: float, f_min: float = 0.2, f_max: float = 15.0, n_freq: int = 40, amp_rms: float = 2.0,
              bias: float = 0.0, seed: int = 0, periods: int = 3) -> tuple:
    """Schroeder-phased multisine (low crest factor) on a log grid of `n_freq` frequencies that are exact
    harmonics of the period T/periods. Returns (u, freqs). Use the last `periods-1` periods for the FRF."""
    Tp = T / periods
    df = 1.0 / Tp
    ks = np.unique(np.round(np.geomspace(max(f_min, df), f_max, n_freq) / df).astype(int))
    freqs = ks * df
    n = len(freqs)
    phases = -np.pi * np.arange(1, n + 1) * np.arange(n) / n           # Schroeder phases
    t = np.arange(int(round(T * fs))) / fs
    u = np.sum(np.sin(2 * np.pi * freqs[:, None] * t[None, :] + phases[:, None]), axis=0)
    u *= amp_rms / np.sqrt(np.mean(u ** 2))
    return bias + u, freqs


def prbs(fs: float, T: float, amp: float = 3.0, bias: float = 0.0, hold: float = 0.02, order: int = 9,
         seed: int = 1) -> np.ndarray:
    """Pseudo-random binary sequence (maximum-length LFSR) with clock period `hold` seconds."""
    taps = {5: (5, 3), 6: (6, 5), 7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5), 10: (10, 7), 11: (11, 9)}[order]
    state = np.ones(order, int)
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 2, order)
    if not state.any():
        state[0] = 1
    n = int(round(T * fs))
    per = max(1, int(round(hold * fs)))
    out = np.empty(n)
    bit = 1
    for k in range(n):
        if k % per == 0:
            fb = 0
            for tp in taps:
                fb ^= state[tp - 1]
            state = np.roll(state, 1)
            state[0] = fb
            bit = state[-1]
        out[k] = amp if bit else -amp
    return bias + out


def steps(fs: float, levels, hold: float = 1.0, ramp: float = 0.0) -> np.ndarray:
    """Piecewise-constant levels (optionally ramped) - breakaway friction and latency tests."""
    segs = []
    prev = 0.0
    for L in levels:
        nr = int(round(ramp * fs))
        if nr > 0:
            segs.append(np.linspace(prev, L, nr, endpoint=False))
        segs.append(np.full(int(round(hold * fs)), float(L)))
        prev = L
    return np.concatenate(segs)


def triangle_velocity(fs: float, speeds, amplitude: float = 0.6, center: float = 0.6) -> np.ndarray:
    """Position reference (rad) of constant-speed back-and-forth sweeps for a velocity/position loop:
    each speed (rad/s) is used for one up and one down stroke of +-amplitude around `center`."""
    out = [np.full(int(fs), center - amplitude)]
    for v in speeds:
        n = max(2, int(round(2 * amplitude / v * fs)))
        up = np.linspace(center - amplitude, center + amplitude, n)
        out += [up, np.full(int(0.5 * fs), center + amplitude), up[::-1], np.full(int(0.5 * fs), center - amplitude)]
    return np.concatenate(out)


def _taper(n, m):
    w = np.ones(n)
    if m > 0 and 2 * m < n:
        r = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        w[:m] = r
        w[-m:] = r[::-1]
    return w
