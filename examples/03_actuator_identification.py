"""Identify an Indego joint drive and write the result into the simulator parameters.

This script runs the whole protocol of docs/CALIBRATION.md section 2 on the *virtual bench*
(sysid/bench.py), so you can see what each test gives and how accurate it is, before doing it on the
real exo. To use your own data, replace the `bench.simulate(...)` calls by your logged arrays
(time, commanded torque, measured current, joint angle, firmware velocity) at 200 Hz.

    python examples/03_actuator_identification.py            # -> results/actuator_id.png + printed table
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indego_sim.config import MotorParams                                # noqa: E402
from indego_sim.sysid import actuator_id as aid, bench, excitation as ex  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
fs = 200.0

# "truth" = the drive you are trying to identify (here: the default model with a few changes so the
# identification has something to find)
true_mp = MotorParams(coulomb=0.075, viscous=5.0e-4, rotor_inertia=5.5e-6, cmd_delay=0.004)
link = bench.LinkParams()                       # thigh segment hanging from the hip joint
truth = bench.true_parameters(true_mp, link)

# --- test A: constant-velocity sweeps with the segment horizontal / gravity-balanced -> friction map
m, d = bench.make_bench(true_mp, bench.LinkParams(gravity=False))
ref = ex.triangle_velocity(fs, [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.8], amplitude=0.6, center=0.0)
rA = bench.simulate(m, d, ref, true_mp, fs, mode="position", kp=150, kd=4)   # low kd: ~10 ms loop latency
v_ref = np.gradient(ref) * fs
pts = aid.steady_state_points(rA.t, v_ref, rA.current * true_mp.K * true_mp.gear)
fr = aid.friction_map(pts[:, 0], pts[:, 1], v_max=2.0)

# --- test B: torque steps -> command latency (command vs measured current)
m, d = bench.make_bench(true_mp, link)
uB = ex.steps(fs, [0, 4, -4, 6, -6, 0] * 3, hold=0.5)
rB = bench.simulate(m, d, uB, true_mp, fs)
delay = aid.estimate_delay(rB.cmd, rB.current * true_mp.K * true_mp.gear, fs)

# --- test C: multisine with the segment hanging -> inertia + gravity (friction from test A fixed)
m, d = bench.make_bench(true_mp, link)
uC, freqs = ex.multisine(fs, 30.0, 0.3, 10.0, 40, amp_rms=2.5, periods=3)
rC = bench.simulate(m, d, uC, true_mp, fs)
rb = aid.fit_rigid_body(rC.t, rC.q, rC.current * true_mp.K * true_mp.gear, fs, friction=fr)

# --- test D (cross-check): FRF torque -> velocity
n0 = len(uC) // 3
f, H, coh = aid.frf(rC.current[n0:] * true_mp.K * true_mp.gear, rC.qd[n0:], fs, nperseg=len(uC) // 3)
ffit = aid.fit_frf_mass_damper(f, H, coh, f_band=(0.3, 10.0))

# --- test E: segment detached (only the output flange / a light known adapter) -> reflected rotor inertia
adapter = bench.LinkParams(mass=0.3, com=0.03, inertia_com=2e-4)
m, d = bench.make_bench(true_mp, adapter)
uE, _ = ex.multisine(fs, 20.0, 1.0, 20.0, 40, amp_rms=3.0, periods=2)
rE = bench.simulate(m, d, uE, true_mp, fs)
rbE = aid.fit_rigid_body(rE.t, rE.q, rE.current * true_mp.K * true_mp.gear, fs, friction=fr, cutoff=30.0, gravity=False)
J_adapter = adapter.inertia_com + adapter.mass * adapter.com ** 2
rotor_from_E = (rbE.inertia - J_adapter) / true_mp.gear ** 2

ident = aid.to_motor_params(MotorParams(), true_mp.gear, rb, fr, link_inertia=link.inertia_com + link.mass * link.com ** 2,
                            delay=delay)
rows = [
    ("joint inertia J [kg m^2]", truth["inertia"], rb.inertia),
    ("Coulomb friction [N m]", truth["coulomb"], fr.coulomb),
    ("viscous friction [N m s/rad]", truth["viscous"], fr.viscous),
    ("gravity m*g*l [N m]", truth["mgl"], rb.mgl),
    # measured between command and *measured* current = driver latency + sensing pipeline
    ("command->measured-current delay [s]", true_mp.cmd_delay + 0.003 + 0.5 / fs, delay),
    # the reflected rotor inertia is ~2 % of J with the segment attached: identify it with the segment
    # detached (or with a well-known CAD inertia) - see docs/CALIBRATION.md
    ("rotor inertia, segment attached [kg m^2]", true_mp.rotor_inertia, ident.rotor_inertia),
    ("rotor inertia, segment detached (E) [kg m^2]", true_mp.rotor_inertia, rotor_from_E),
]
print(f"{'parameter':40s} {'true':>10s} {'identified':>12s} {'error %':>8s}")
for name, tv, iv in rows:
    print(f"{name:40s} {tv:10.4g} {iv:12.4g} {100 * (iv - tv) / tv:8.1f}")
print("\nFRF cross-check:", {k: round(v, 4) for k, v in ffit.items()},
      "\n(FRF damping includes the equivalent damping of Coulomb friction at this amplitude; the stiffness is gravity",
      "linearised about the motion, so expect it below m*g*l for large swings)")
print("\nThe delay row compares with cmd_delay + sensor_delay + half a sample (what the measurement contains).",
      "\nThe rotor-inertia row shows why it must be identified with the segment detached: with the segment",
      "attached it is ~2 % of the joint inertia, so a 3 % error in J becomes a 100+ % error in the rotor inertia.",
      "\nEven detached, 200 Hz logging cannot resolve it well (test E); use the motor/gearbox datasheet value or",
      "log the drive at >= 1 kHz for that one test.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    vv = np.linspace(-2, 2, 400)
    ax[0].plot(pts[:, 0], pts[:, 1], "o", label="steady-state points")
    ax[0].plot(vv, np.sign(vv) * (fr.coulomb + (fr.static - fr.coulomb) * np.exp(-(vv / max(fr.stribeck_vel, 1e-3)) ** 2))
               + fr.viscous * vv + fr.offset, label="fit")
    ax[0].set(xlabel="joint velocity [rad/s]", ylabel="drive torque [N m]", title="A: friction map")
    ax[0].legend()
    sl = slice(0, int(3 * fs))
    ax[1].plot(rB.t[sl], rB.cmd[sl], label="command")
    ax[1].plot(rB.t[sl], rB.current[sl] * true_mp.K * true_mp.gear, label="N K i (measured)")
    ax[1].set(xlabel="t [s]", ylabel="N m", title=f"B: latency = {1000 * delay:.1f} ms")
    ax[1].legend()
    m_ = (f > 0.2) & (f < 12)
    ax[2].loglog(f[m_], np.abs(H[m_]), label="H1 estimate")
    s = 2j * np.pi * f[m_]
    Hm = s * np.exp(-s * ffit["delay"]) / (ffit["inertia"] * s ** 2 + ffit["viscous"] * s + ffit["stiffness"])
    ax[2].loglog(f[m_], np.abs(Hm), "--", label="fit")
    ax[2].set(xlabel="Hz", ylabel="|velocity / torque|", title="D: FRF (multisine)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "actuator_id.png"), dpi=110)
    print("figure ->", os.path.abspath(os.path.join(OUT, "actuator_id.png")))
except ImportError:
    pass
