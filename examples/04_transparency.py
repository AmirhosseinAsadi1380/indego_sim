"""How transparent is the exo for a person with leg volition? Zero torque vs friction+gravity compensation.

A healthy participant stands quietly (own ankle/hip balance strategy, no crutches) while the exo runs
(a) zero torque command, (b) TransparentController. The cuff interaction forces are the
ground truth you would like to minimise; compare them to what the onboard current-based torque
estimate suggests.

    python examples/04_transparency.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indego_sim.config import SupportParams                                 # noqa: E402
from indego_sim.controllers.transparent import TransparentController        # noqa: E402
from indego_sim.human.sci import get_preset                                 # noqa: E402
from indego_sim.sim import HumanExoSim                                      # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
res = {}
for name, ctrl in (("zero torque", lambda obs: np.zeros(4)), ("transparent", TransparentController())):
    sim = HumanExoSim(subject=get_preset("healthy"), support=SupportParams(kind="none"),
                      schedule=[], seed=1)
    sim.reset(settle=1.0)
    log = sim.run(ctrl, 8.0)
    F = np.concatenate([log[k][:, :3] for k in log if k.startswith("F_exo_thigh_cuff")], axis=1)
    res[name] = (log, F)
    print(f"{name:12s} fell={sim.fell}  thigh-cuff |F| rms = {np.sqrt(np.mean(F ** 2)):.1f} N   "
          f"hip torque rms = {np.sqrt(np.mean(log['exo_tau'][:, [0, 2]] ** 2)):.2f} N m")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for name, (log, F) in res.items():
        ax[0].plot(log["t"], np.linalg.norm(F[:, :3], axis=1), label=name)
        ax[1].plot(log["t"], log["exo_tau"][:, 0], label=name)
    ax[0].set(xlabel="t [s]", ylabel="|F| left thigh cuff [N]", title="Interaction force (ground truth)")
    ax[1].set(xlabel="t [s]", ylabel="N m", title="Left hip drive torque")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transparency.png"), dpi=110)
    print("figure ->", os.path.abspath(os.path.join(OUT, "transparency.png")))
except ImportError:
    pass
