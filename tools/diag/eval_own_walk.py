"""Score the person's own walking (gait_mode "walk") over several seeds: time on their feet, distance, steps.

The config is merged into the own-walking scenario: sagittal model, no crutches or help, zero-torque exo, 30 ms
long-loop delay, T12 AIS D, walking parameters tuned for the subject preset (HumanWalkParams.for_subject) unless
"vol": {"walk": {...}} is given.

    python tools/diag/eval_own_walk.py '<cfg json>' [seeds=0,1,2,3] [T=30] [workers=4]
    python tools/diag/eval_own_walk.py '{}'                                      # T12 AIS D, zero-torque exo
    python tools/diag/eval_own_walk.py '{"subj":"healthy","ctrl":"transparent"}'
    python tools/diag/eval_own_walk.py '{"subject":{"long_loop_delay":0.06}}'   # default reaction delay
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import make_sim  # noqa: E402
from indego_sim.config import TUNED_WALK  # noqa: E402

BASE = {"ctrl": "zero", "subj": "sci_t12_aisD", "subject": {"long_loop_delay": 0.03}, "sup": {"kind": "none"},
        "sim": {"planar": True}, "vol": {"gait_mode": "walk"}}


def own_walk_cfg(cfg: dict) -> dict:
    """`cfg` merged (one level deep) into the own-walking scenario."""
    out = json.loads(json.dumps(BASE))
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    if "walk" not in out["vol"]:
        out["vol"]["walk"] = dict(TUNED_WALK.get(out["subj"], {}))
    return out


def run(args):
    cfg, seed, T = args
    t0 = time.time()
    sim, ctl, obs = make_sim(cfg, seed, log_rate_hz=10.0)
    m, d, hc = sim.m, sim.d, sim.human
    tal = {s: m.body(f"talus_{s}").id for s in "lr"}
    x0 = d.subtree_com[sim.root][0]
    prev, steps, t_last, x_last, unstable = None, [], 0.0, x0, False
    for _ in range(int(round(T / sim.ctrl_dt))):
        obs = sim.step(ctl(obs))
        if d.time - sim.t0 < t_last:                   # MuJoCo reset the data after a numerical blow-up: a fall
            unstable = True
            break
        t_last, x_last = d.time - sim.t0, d.subtree_com[sim.root][0]
        sw = hc.ws.get("swing") if hc.ws.get("on") else None
        if sw is not None and sw != prev:
            if prev is not None:                       # a new swing starts: the other foot has landed
                steps.append(d.xpos[tal[prev]][0] - d.xpos[tal[sw]][0])
            prev = sw
        if sim.fell:
            break
    steps = np.array(steps)
    return dict(seed=seed, fell=round(t_last, 2) if sim.fell or unstable else None,
                distance_m=round(float(x_last - x0), 2), steps=len(steps),
                backward_steps=int((steps < 0).sum()), step_length_m=round(float(steps.mean()), 3) if len(steps) else 0.0,
                wall_s=round(time.time() - t0))


if __name__ == "__main__":
    cfg = own_walk_cfg(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
    seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2,3").split(",")]
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    with Pool(min(W, len(seeds))) as pool:
        res = pool.map(run, [(cfg, s, T) for s in seeds])
    for r in res:
        print(json.dumps(r))
    falls = [r for r in res if r["fell"] is not None]
    dist = [r["distance_m"] for r in res]
    print(f"SUMMARY {cfg['subj']} ctrl={cfg['ctrl']} delay={cfg['subject'].get('long_loop_delay')} s: "
          f"{len(res) - len(falls)}/{len(res)} walked {T:.0f} s, distance {min(dist):.2f}-{max(dist):.2f} m, "
          f"steps {np.mean([r['steps'] for r in res]):.0f}" + (f", falls at {[r['fell'] for r in falls]} s" if falls else ""))
