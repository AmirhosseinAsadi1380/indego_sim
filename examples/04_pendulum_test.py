"""Calibrate a participant's quadriceps spasticity from a Wartenberg pendulum test.

1. simulates the test for a healthy person and for the T10 AIS A preset,
2. makes a synthetic "measured" trace (a participant whose true parameters differ from the preset),
3. fits the TSRT threshold / velocity term / gain to it with sysid.spasticity.fit_pendulum.

With a real participant: record the knee angle (goniometer/IMU, ~100+ Hz) during the release, then
call fit_pendulum(t, knee_deg, get_preset("..."), side="r").

    python examples/04_pendulum_test.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indego_sim.human.sci import get_preset                                     # noqa: E402
from indego_sim.sysid.spasticity import fit_pendulum, pendulum_test             # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

healthy = pendulum_test(get_preset("healthy"), T=6.0)
sci = pendulum_test(get_preset("sci_t10_aisA"), T=6.0)
true_ov = {"mu": 0.30, "gain": 4.0}      # stronger velocity-dependent catch than the MAS 1.5 preset
meas = pendulum_test(get_preset("sci_t10_aisA"), T=4.0, overrides=true_ov, seed=5)
rng = np.random.default_rng(0)
t_m = meas.t[::10] - meas.t[0]
k_m = meas.knee_deg[::10] + rng.normal(0, 0.5, len(t_m))          # 100 Hz goniometer with 0.5 deg noise
# one release at one speed cannot separate threshold, velocity term and gain well: fix the static threshold
# (from MAS / end-range catch) and fit velocity term + gain; add Tardieu stretches at 2-3 speeds to fit all three
best, fit = fit_pendulum(t_m, k_m, get_preset("sci_t10_aisA"), params=("mu", "gain"), max_iter=40, grid=5)

for name, r in (("healthy", healthy), ("SCI preset", sci), ("'measured'", meas), ("fitted", fit)):
    mt = r.metrics
    print(f"{name:12s} first swing {mt['first_swing_deg']:6.1f} deg   R2 {mt['R2']:.2f}   oscillations "
          f"{mt['n_oscillations']}   rest {mt['rest_deg']:.0f} deg")
print("true parameters  :", true_ov)
print("fitted parameters:", {k: round(float(v), 3) for k, v in best.items()})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(healthy.t - healthy.t[0], healthy.knee_deg, label="healthy (relaxed)")
    ax[0].plot(sci.t - sci.t[0], sci.knee_deg, label="T10 AIS A preset (MAS 1.5 quads)")
    ax[0].invert_yaxis()
    ax[0].set(xlabel="t [s]", ylabel="knee flexion [deg]", title="Simulated pendulum test")
    ax[0].legend()
    ax[1].plot(t_m, k_m, ".", ms=3, label="'measured'")
    ax[1].plot(fit.t - fit.t[0], fit.knee_deg, label="fitted model")
    ax[1].invert_yaxis()
    ax[1].set(xlabel="t [s]", title="Fitting the TSRT parameters")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pendulum_test.png"), dpi=110)
    print("figure ->", os.path.abspath(os.path.join(OUT, "pendulum_test.png")))
except ImportError:
    pass
