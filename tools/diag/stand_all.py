"""Quiet-standing regression for all presets (10 s): fall, COM sway, crutch load.

    python tools/diag/stand_all.py
"""
from multiprocessing import Pool

import numpy as np
from _common import SupportParams, HumanExoSim, get_preset  # noqa: F401
from indego_sim.controllers.indego_fsm import IndegoFSMController
from indego_sim.controllers.transparent import TransparentController


def run(args):
    subj, kind = args
    sim = HumanExoSim(subject=get_preset(subj), support=SupportParams(kind=kind), schedule=[], seed=1,
                      log_muscles=False, log_rate_hz=50)
    sim.reset(settle=1.0)
    log = sim.run(TransparentController() if subj == "healthy" else IndegoFSMController(), 10.0)
    F = np.linalg.norm(log["support_force"], axis=2).sum(1)
    return (f"{subj:14s} {kind:9s} fell={sim.fell} sway x/y={1000 * log['com'][:, 0].std():.1f}/"
            f"{1000 * log['com'][:, 1].std():.1f} mm crutch={100 * F.mean() / sim.BW:.0f}%BW")


if __name__ == "__main__":
    jobs = [("sci_t10_aisA", "crutches"), ("sci_t5_aisA", "crutches"), ("sci_l2_aisC", "crutches"),
            ("sci_t12_aisD", "crutches"), ("healthy", "none")]
    with Pool(2) as p:
        for r in p.map(run, jobs):
            print(r)
