"""Compare swing shapes: per step, toe clearance (toe body above its standing height, 20-80 % of the step),
step length (ankle travel) and the step phase at which the swing foot first loads the ground.

    python tools/diag/step_sweep.py '[{"name":"a","fsm":{"hip_flex":0.63}}, {"name":"b","fsm":{...}}]' [seed] [T] [workers]
"""
import json
import sys
from multiprocessing import Pool

from _common import make_sim


def run(args):
    cfg, seed, T = args
    sim, fsm, obs = make_sim(cfg, seed, log_rate_hz=20)
    m, d = sim.m, sim.d
    toes = {s: m.body(f"toes_{s}").id for s in "lr"}
    tal = {s: m.body(f"talus_{s}").id for s in "lr"}
    z0, steps, cur = None, [], None
    for k in range(int(T / sim.ctrl_dt)):
        obs = sim.step(fsm(obs))
        if z0 is None and fsm.t > 0.5:
            z0 = {s: d.xpos[toes[s]][2] for s in "lr"}
        if fsm.state == fsm.STEP:
            s, ph = fsm.swing, fsm.t_state / fsm.p.step_time
            if cur is None or cur["done"] or cur["side"] != s:
                cur = dict(side=s, t=round(d.time - sim.t0, 2), x0=d.xpos[tal[s]][0], clearance=9.0,
                           contact_at=None, done=False)
                steps.append(cur)
            if 0.2 < ph < 0.8:
                cur["clearance"] = min(cur["clearance"], d.xpos[toes[s]][2] - z0[s])
                if cur["contact_at"] is None and ph > 0.25 and sim._grf[s][2] > 0.05 * sim.BW:
                    cur["contact_at"] = round(ph, 2)
        elif cur is not None and not cur["done"]:
            cur.update(length=round(float(d.xpos[tal[cur["side"]]][0] - cur["x0"]), 3),
                       clearance=round(cur["clearance"], 3), done=True)
        if sum(st["done"] for st in steps) >= cfg.get("nsteps", 6) or sim.fell:
            break
    return dict(cfg=cfg.get("name", ""), fell=round(d.time - sim.t0, 2) if sim.fell else None,
                steps=[{k: v for k, v in st.items() if k not in ("x0", "done")} for st in steps])


if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    with Pool(W) as p:
        for r in p.map(run, [(c, seed, T) for c in cfgs]):
            print(json.dumps(r))
