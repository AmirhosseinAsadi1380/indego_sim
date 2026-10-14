"""Tune the person's own walking (HumanWalkParams, gait_mode "walk") for a subject / delay / device with CMA-ES.

Starts from the set tuned for the subject preset (or the defaults) and searches all HumanWalkParams within the
bounds below. Cost per run: -(time walked, counted only while the person keeps moving forward at >= vmin, + pw x
forward distance capped at vcap x time - 4 x backward distance), averaged over seeds and variants. Survival alone
is not enough: it converges to marching in place. Validate the result with eval_own_walk.py (>= 4 seeds, 30 s).

    python tools/diag/tune_own_walk.py '<cfg json>' out.jsonl [iters=40] [popsize=16] [T=20] [seeds=0,1]
        [--variants '[{"ctrl":"zero"},{"ctrl":"transparent"}]'] [--sigma 0.1] [--vmin 0.15] [--vcap 0.6] [--pw 1]

The config is merged into the scenario of eval_own_walk.py. Prints the best parameter set (a "vol.walk" dict).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_own_walk import own_walk_cfg, run  # noqa: E402

BOUNDS = {
    "swing_time": (0.3, 0.7), "min_swing_frac": (0.3, 0.9), "max_swing_frac": (1.0, 2.0), "theta0": (0.1, 0.6),
    "c_d": (0.0, 1.5), "c_v": (0.0, 0.8), "swing_knee_flex": (0.5, 1.4), "kp_knee_early": (0.0, 3.0),
    "knee_extend_at": (0.2, 0.8), "knee_ext_thigh": (-0.3, 0.5), "swing_knee_ext": (0.0, 0.3),
    "swing_reach_at": (0.4, 1.0), "g_comp": (0.0, 1.5), "swing_ankle": (-0.1, 0.5), "stance_knee": (0.0, 0.3),
    "stance_ankle": (-0.2, 0.35), "push_off": (0.0, 1.5), "pelvis_pitch": (-0.1, 0.2), "v_des": (0.2, 0.8),
    "k_lean": (0.0, 0.5), "min_step": (-0.05, 0.2), "init_lean": (0.0, 0.2), "init_dx": (0.0, 0.12),
    "ds_unload": (0.05, 0.5), "ds_max": (0.05, 0.6), "ds_push": (0.0, 1.2), "kp_hip": (0.5, 6.0),
    "kp_knee": (0.5, 6.0), "kp_ankle": (0.5, 10.0), "kp_knee_st": (1.0, 10.0), "kp_ankle_st": (1.0, 10.0),
    "kp_pelvis": (1.0, 25.0), "kd_ratio": (0.03, 0.2), "predict": (0.0, 0.08),
}
NAMES = list(BOUNDS)


def cost(runs, vmin, vcap, pw):
    c = 0.0
    for r in runs:
        t, prog = r["t"], r["prog"]
        t_eff = min(t, 2.0 + max(prog, 0.0) / vmin)
        c -= t_eff + pw * max(min(prog, vcap * t), 0.0) - 4.0 * max(-prog, 0.0)
    return c / len(runs)


def job(args):
    cfg, seed, T = args
    r = run((cfg, seed, T))
    return dict(seed=seed, t=r["fell"] if r["fell"] is not None else T, prog=r["distance_m"])


if __name__ == "__main__":
    import cma
    from indego_sim.config import HumanWalkParams
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("out")
    ap.add_argument("iters", nargs="?", type=int, default=40)
    ap.add_argument("popsize", nargs="?", type=int, default=16)
    ap.add_argument("T", nargs="?", type=float, default=20.0)
    ap.add_argument("seeds", nargs="?", default="0,1")
    ap.add_argument("--variants", default="[{}]")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--vmin", type=float, default=0.15)
    ap.add_argument("--vcap", type=float, default=0.6)
    ap.add_argument("--pw", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    base = own_walk_cfg(json.loads(a.cfg))
    start = {**HumanWalkParams().__dict__, **base["vol"]["walk"]}
    x0 = [float(np.clip((start[k] - lo) / (hi - lo), 0.02, 0.98)) for k, (lo, hi) in BOUNDS.items()]
    variants = json.loads(a.variants)
    seeds = [int(s) for s in a.seeds.split(",")]

    def decode(x):
        cfg = json.loads(json.dumps(base))
        cfg["vol"]["walk"] = {**cfg["vol"]["walk"],
                              **{k: round(float(lo + (hi - lo) * xi), 4) for (k, (lo, hi)), xi in
                                 zip(BOUNDS.items(), np.clip(x, 0, 1))}}
        return cfg

    def variant(cfg, v):
        cfg = json.loads(json.dumps(cfg))
        for k, val in v.items():
            if isinstance(val, dict):
                cfg.setdefault(k, {}).update(val)
            else:
                cfg[k] = val
        return cfg

    es = cma.CMAEvolutionStrategy(x0, a.sigma, {"bounds": [0, 1], "popsize": a.popsize, "seed": 3, "verbose": -9})
    best = (np.inf, None)
    with Pool(a.workers) as pool, open(a.out, "w") as f:
        for it in range(a.iters):
            t0 = time.time()
            X = es.ask()
            cfgs = [decode(x) for x in X]
            jobs = [(variant(c, v), s, a.T) for c in cfgs for v in variants for s in seeds]
            res = pool.map(job, jobs)
            n = len(variants) * len(seeds)
            costs = []
            for i, c in enumerate(cfgs):
                runs = res[i * n:(i + 1) * n]
                costs.append(cost(runs, a.vmin, a.vcap, a.pw))
                f.write(json.dumps(dict(it=it, cost=costs[-1], runs=runs, walk=c["vol"]["walk"])) + "\n")
                if costs[-1] < best[0]:
                    best = (costs[-1], c["vol"]["walk"], runs)
            f.flush()
            es.tell(X, costs)
            print(f"iter {it}: best {best[0]:.2f} {[(r['t'], r['prog']) for r in best[2]]} ({time.time() - t0:.0f} s)",
                  flush=True)
    print(json.dumps(best[1]))
