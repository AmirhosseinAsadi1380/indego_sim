"""Quiet standing of a person with complete T10 SCI in the Indego, with forearm crutches.

Shows what your controller would see (onboard sensors) next to the ground truth the simulator logs:
cuff interaction forces, crutch loads, COM sway, reflex/spasm activity of the paralysed muscles.

    python examples/01_quiet_standing.py [--video] [--spasms]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indego_sim.config import SupportParams                              # noqa: E402
from indego_sim.controllers.indego_fsm import IndegoFSMController        # noqa: E402
from indego_sim.human.sci import get_preset                              # noqa: E402
from indego_sim.sim import HumanExoSim                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", action="store_true")
ap.add_argument("--spasms", action="store_true", help="raise the spasm rate so one shows up in 10 s")
ap.add_argument("--duration", type=float, default=10.0)
args = ap.parse_args()
NAME = "standing_extensor_spasm" if args.spasms else "quiet_standing_sci"
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

subject = get_preset("sci_t10_aisA")
if args.spasms:
    subject.spasticity.spasm_rate_per_min = 12.0
sim = HumanExoSim(subject=subject, support=SupportParams(kind="crutches"), schedule=[], seed=2)
ctrl = IndegoFSMController()          # stays in STAND (nobody leans): PD hold of the standing posture
sim.reset(settle=1.0)
if args.video:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from indego_sim import viz
    log = viz.record(sim, ctrl, args.duration, os.path.join(OUT, NAME + ".mp4"), fps=30)
else:
    log = sim.run(ctrl, args.duration)

t = log["t"]
com = log["com"]
F_cr = np.linalg.norm(log["support_force"], axis=2)
print(f"fell: {sim.fell}")
print(f"COM sway (std) x/y: {1000 * com[:, 0].std():.1f} / {1000 * com[:, 1].std():.1f} mm")
print(f"crutch load mean (L/R): {F_cr[:, 0].mean():.0f} / {F_cr[:, 1].mean():.0f} N "
      f"= {100 * F_cr.sum(1).mean() / (sim.BW):.0f} %BW")
for k in ("F_exo_thigh_cuff_l", "F_exo_shank_cuffP_l", "F_exo_belt"):
    if k in log:
        print(f"{k:24s} mean |F| = {np.linalg.norm(log[k][:, :3], axis=1).mean():6.1f} N")
print("spasms:", [e for e in log["events"] if str(e[1]).startswith("spasm")])

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    ax[0, 0].plot(t, 1000 * (com[:, 0] - com[0, 0]), label="AP")
    ax[0, 0].plot(t, 1000 * (com[:, 1] - com[0, 1]), label="ML")
    ax[0, 0].set(ylabel="COM displacement [mm]", title="Centre of mass")
    ax[0, 0].legend()
    ax[0, 1].plot(t, F_cr[:, 0], label="left crutch")
    ax[0, 1].plot(t, F_cr[:, 1], label="right crutch")
    ax[0, 1].plot(t, log["grf_l"][:, 2], "--", label="left foot")
    ax[0, 1].plot(t, log["grf_r"][:, 2], "--", label="right foot")
    ax[0, 1].set(ylabel="N", title="Crutch and foot loads")
    ax[0, 1].legend(fontsize=8)
    for k, lab in (("F_exo_thigh_cuff_l", "thigh cuff L"), ("F_exo_shank_cuffP_l", "shank cuff L (prox)"),
                   ("F_exo_belt", "hip belt")):
        if k in log:
            ax[1, 0].plot(t, np.linalg.norm(log[k][:, :3], axis=1), label=lab)
    ax[1, 0].set(xlabel="t [s]", ylabel="|F| [N]", title="Cuff interaction forces (ground truth)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(t, log["exo_tau"][:, 0], label="hip L")
    ax[1, 1].plot(t, log["exo_tau"][:, 1], label="knee L")
    ax[1, 1].plot(t, log["reflex_mean"] * 100, "k:", label="mean reflex excitation x100")
    ax[1, 1].set(xlabel="t [s]", ylabel="N m", title="Exo joint torque / spinal activity")
    ax[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, NAME + ".png"), dpi=110)
    print("figure ->", os.path.abspath(os.path.join(OUT, NAME + ".png")))
except ImportError:
    pass
