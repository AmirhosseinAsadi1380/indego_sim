"""Closed-loop walking session: complete-SCI user (T10 AIS A) + Indego-like FSM controller + crutches.

The person leans forward to request each step (volition), shifts weight onto the stance leg, moves the
crutches in double support and balances with their arms; the controller only sees the onboard sensors
(+ the add-on upper-trunk IMU unless --no-trunk-imu). Prints when the first step fired and how well the
controller's COM estimate matched the simulator's ground truth.

STATUS (see docs/STATUS.md): with the default (uncalibrated) arm/crutch strategy this user typically
manages 1-2 steps with the baseline FSM before a fall in double support. Use it to study step initiation,
foot clearance, crutch loading and failure modes, and calibrate the support model on your own session data.

    python examples/02_sci_walking_fsm.py [--video] [--duration 20] [--support crutches|walker|harness]
                                          [--no-trunk-imu] [--seed 0]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indego_sim.config import SensorParams, SupportParams                # noqa: E402
from indego_sim.controllers.indego_fsm import IndegoFSMController        # noqa: E402
from indego_sim.human.sci import get_preset                              # noqa: E402
from indego_sim.sim import HumanExoSim                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", action="store_true")
ap.add_argument("--duration", type=float, default=20.0)
ap.add_argument("--support", default="crutches")
ap.add_argument("--subject", default="sci_t10_aisA")
ap.add_argument("--trigger", default="lean", help="lean | auto (timer, e.g. with the harness)")
ap.add_argument("--no-trunk-imu", action="store_true", help="stock Indego sensors only")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

sup = SupportParams(kind=args.support)
if args.support == "harness":
    sup.weight_bearing_frac = 0.4
sens = SensorParams(trunk_imu="none" if args.no_trunk_imu else "sternum")
sim = HumanExoSim(subject=get_preset(args.subject), support=sup, sensors=sens, schedule=[(2.0, "walk")],
                  seed=args.seed)
# COM-estimator constants measured on this participant's model (on the real device: anthropometric tables)
fsm = IndegoFSMController.from_model(sim.m, trigger=args.trigger)
truth = []


def _true_com(sim_, d):        # ground truth for scoring the controller's estimate (not visible to it)
    ax = [d.xpos[sim_.m.body(f"talus_{s}").id][0] for s in "lr"]
    truth.append((d.time - getattr(sim_, "t0", 0.0), d.subtree_com[sim_.root][0] - 0.5 * (ax[0] + ax[1])))


sim.reset(settle=1.5)
sim.callbacks.append(_true_com)
if args.video:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from indego_sim import viz
    log = viz.record(sim, fsm, args.duration, os.path.join(OUT, f"walking_{args.subject}_{args.support}.mp4"),
                     fps=30, azimuth=110)
else:
    log = sim.run(fsm, args.duration)

ev = [(round(float(t), 2), e) for t, e in log["events"]]
print("events:", ev)
print(f"steps completed by the FSM: {fsm.steps_total} (feet closed after a double-support time-out: "
      f"{fsm.closes}); fell: {sim.fell}"
      + (f" at t={log['t'][-1]:.1f} s" if sim.fell else ""))
L = np.array([(r[0], r[3]) for r in fsm.log])
starts = [r[0] for r0, r in zip(fsm.log[:-1], fsm.log[1:]) if r[1] == "step" and r0[1] != "step"]
print(f"walk intent at 2.0 s; first step triggered at {starts[0]:.2f} s" if starts else "no step triggered")
T = np.array(truth)
t_fall = min([float(t) for t, e in log["events"] if e == "fall"] + [np.inf])
sel = (L[:, 0] > fsm.p.calib_time) & (L[:, 0] < t_fall - 0.5)      # score while the user is still upright
err = L[sel, 1] - np.interp(L[sel, 0], T[:, 0], T[:, 1])
print(f"controller COM estimate vs truth ({'stock sensors' if args.no_trunk_imu else 'with trunk IMU'}): "
      f"bias {1000 * err.mean():+.0f} mm, SD {1000 * err.std():.0f} mm")
F_cr = np.linalg.norm(log["support_force"], axis=2)
print(f"crutch load mean/peak: {F_cr.sum(1).mean():.0f} / {F_cr.sum(1).max():.0f} N "
      f"({100 * F_cr.sum(1).max() / sim.BW:.0f} %BW peak)")
print(f"peak exo torque |hip|/|knee|: {np.abs(log['exo_tau'][:, [0, 2]]).max():.0f} / "
      f"{np.abs(log['exo_tau'][:, [1, 3]]).max():.0f} N m; winding temp max {log['exo_temp'].max():.1f} C")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = log["t"]
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t, np.degrees(log["exo_q"][:, 0]), label="hip L")
    ax[0].plot(t, np.degrees(log["exo_q"][:, 2]), label="hip R")
    ax[0].plot(t, np.degrees(log["exo_q"][:, 1]), "--", label="knee L")
    ax[0].plot(t, np.degrees(log["exo_q"][:, 3]), "--", label="knee R")
    ax[0].set(ylabel="exo joint angle [deg]")
    ax[0].legend(ncol=4, fontsize=8)
    ax[1].plot(t, log["grf_l"][:, 2], label="foot L")
    ax[1].plot(t, log["grf_r"][:, 2], label="foot R")
    ax[1].plot(t, F_cr[:, 0], ":", label="crutch L")
    ax[1].plot(t, F_cr[:, 1], ":", label="crutch R")
    ax[1].set(ylabel="vertical load [N]")
    ax[1].legend(ncol=4, fontsize=8)
    ax[2].plot(t, np.degrees(log["intent_lean"]), label="intended lean [deg]")
    ax[2].plot(t, np.degrees(log["torso_pitch"]), label="torso pitch [deg]")
    ax[2].plot(t, 100 * (log["com"][:, 0] - log["com"][0, 0]), label="COM progression [cm]")
    ax[2].plot(T[:, 0], 100 * T[:, 1], "k", lw=0.8, label="true COM - mid-ankle [cm]")
    ax[2].plot(L[:, 0], 100 * L[:, 1], "k:", lw=0.8, label="exo estimate [cm]")
    ax[2].set(xlabel="t [s]")
    ax[2].legend(fontsize=8)
    for a in ax:
        for tt, e in ev:
            if "heel" in str(e):
                a.axvline(tt, color="0.85", lw=0.8)
    fig.tight_layout()
    fn = os.path.join(OUT, f"walking_{args.subject}_{args.support}.png")
    fig.savefig(fn, dpi=110)
    print("figure ->", os.path.abspath(fn))
except ImportError:
    pass
