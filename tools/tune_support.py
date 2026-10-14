"""Fit the person's arm/crutch strategy (SupportParams + a few VolitionParams) to statistics measured in
your own sessions, with CMA-ES (pip install cma). Edit TARGETS and SPACE, then:

    python tools/tune_support.py --iters 15 --out results/support_fit.jsonl

Each evaluation runs one simulated session (default 20 s, ~20-40 s wall time); two processes in parallel.
The objective is a weighted squared error between simulated and measured statistics (+ a fall penalty),
not "walk as far as possible" - the goal is a model of *your* participant, falls included if they fall.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- what you measured (example numbers - replace with your participant's) --------------------------
TARGETS = {
    # name: (measured value, weight, unit-scale)
    "stand_crutch_load_frac": (0.15, 1.0, 0.05),   # mean total crutch load in quiet standing / body weight
    "walk_crutch_load_frac": (0.35, 1.0, 0.05),    # mean total crutch load while walking / body weight
    "walk_crutch_peak_frac": (0.60, 0.5, 0.10),    # peak total crutch load / body weight
    "stand_com_sway_mm": (8.0, 0.5, 4.0),          # COM AP std in quiet standing
    "first_step_delay_s": (2.5, 0.5, 1.0),         # walk intent -> first toe-off
    "steps_in_20s": (10.0, 1.0, 2.0),              # completed exo steps
}
SUBJECT = "sci_t10_aisA"
SCHEDULE = [(2.0, "walk")]
T = 20.0

SPACE = [  # (group, name, lo, hi)
    ("sup", "com_ahead", 0.0, 0.08),
    ("sup", "lean_ahead", 0.02, 0.12),
    ("sup", "tip_ahead", 0.2, 0.45),
    ("sup", "max_load_frac", 0.3, 0.55),
    ("sup", "k_arm", 0.0, 20000.0),
    ("sup", "kp_com", 2.0, 12.0),
    ("sup", "kd_com", 1.0, 8.0),
    ("sup", "kp_lat_com", 3.0, 12.0),
    ("vol", "lean_trigger_deg", 3.0, 12.0),
    ("vol", "weight_shift", 0.0, 0.06),
]
_model = None


def decode(x):
    cfg = {"sup": {}, "vol": {}}
    for xi, (g, n, lo, hi) in zip(x, SPACE):
        cfg[g][n] = float(lo + (hi - lo) * np.clip(xi, 0, 1))
    return cfg


def simulate(cfg):
    global _model
    from indego_sim.config import SupportParams, VolitionParams
    from indego_sim.controllers.indego_fsm import IndegoFSMController
    from indego_sim.human.sci import get_preset
    from indego_sim.model.builder import build_model
    from indego_sim.sim import HumanExoSim
    if _model is None:
        _model = build_model(subject=get_preset(SUBJECT))
    sim = HumanExoSim(subject=get_preset(SUBJECT), support=SupportParams(**cfg["sup"]),
                      volition=VolitionParams(**cfg["vol"]), schedule=SCHEDULE, model=_model, log_muscles=False,
                      log_rate_hz=50)
    c = IndegoFSMController()
    sim.reset(settle=1.0)
    log = sim.run(c, T)
    t = log["t"]
    Fz = log["support_force"][:, :, 2].sum(1) / sim.BW
    walk_t = SCHEDULE[0][0]
    stand = t < walk_t
    toe = [e[0] for e in sim.events if str(e[1]).startswith("toe_off")]
    stats = {
        "stand_crutch_load_frac": float(Fz[stand].mean()) if stand.any() else np.nan,
        "walk_crutch_load_frac": float(Fz[~stand].mean()) if (~stand).any() else np.nan,
        "walk_crutch_peak_frac": float(Fz.max()),
        "stand_com_sway_mm": float(1000 * log["com"][stand, 0].std()) if stand.sum() > 5 else np.nan,
        "first_step_delay_s": (toe[0] - walk_t) if toe else T,
        "steps_in_20s": float(c.n_steps),
    }
    return stats, sim.fell


def objective(x):
    try:
        stats, fell = simulate(decode(x))
    except Exception as e:  # noqa: BLE001
        return 1e3, {"err": str(e)}
    cost = 0.0
    for k, (target, w, scale) in TARGETS.items():
        v = stats.get(k, np.nan)
        cost += w * (((v - target) / scale) ** 2 if np.isfinite(v) else 25.0)
    return cost, dict(stats, fell=fell)


if __name__ == "__main__":
    import cma
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--out", default="results/support_fit.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    es = cma.CMAEvolutionStrategy([0.5] * len(SPACE), 0.25, {"bounds": [0, 1], "popsize": a.pop, "seed": 1,
                                                             "verbose": -9})
    best = (np.inf, None, None)
    with Pool(2) as pool, open(a.out, "w") as f:
        for it in range(a.iters):
            X = es.ask()
            t0 = time.time()
            res = pool.map(objective, X)
            es.tell(X, [r[0] for r in res])
            for x, (c, info) in zip(X, res):
                f.write(json.dumps(dict(it=it, cost=c, stats=info, cfg=decode(x)), default=float) + "\n")
                if c < best[0]:
                    best = (c, decode(x), info)
            f.flush()
            print(f"iter {it}: best cost {best[0]:.2f} {best[2]} ({time.time() - t0:.0f} s)", flush=True)
    print(json.dumps(best[1], indent=1))
