"""Score walking over several seeds: steps, fall time, progression, COM-estimate error, help needed (therapist,
planarizer), and how much of the time the person hangs on the crutches (> 70 % body weight on the arms).

    python tools/diag/eval_walk.py '<cfg json>' [seeds=0,1,2,3] [T=25] [workers=2]
    python tools/diag/eval_walk.py '{}'                                         # default crutch walking
    python tools/diag/eval_walk.py '{"sens":{"trunk_imu":"none"}}'               # stock sensors only
    python tools/diag/eval_walk.py '{"sup":{"therapist":{"level":"mod"}}}'       # ideal-ish helper (control vs model)
    python tools/diag/eval_walk.py '{"sup":{"planarizer":{"enabled":true}},"fsm":{"lat_dy":0}}'   # sagittal plane only
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from _common import make_sim


def run(args):
    cfg, seed, T = args
    t0 = time.time()
    sim, fsm, obs = make_sim(cfg, seed)
    m, d = sim.m, sim.d
    tal = [m.body("talus_l").id, m.body("talus_r").id]
    err, starts, last = [], [], fsm.state
    x0 = d.subtree_com[sim.root][0]
    ank0 = np.array([d.xpos[b][0] for b in tal])
    hang = []
    for k in range(int(T / sim.ctrl_dt)):
        tau = fsm(obs)
        if fsm.state != last and fsm.state == fsm.STEP:
            starts.append(round(d.time - sim.t0, 2))
        last = fsm.state
        obs = sim.step(tau)
        if k % 4 == 0 and fsm.t > fsm.p.calib_time:
            ax = [d.xpos[b][0] for b in tal]
            err.append(fsm.est["com_rel_mid"] - (d.subtree_com[sim.root][0] - 0.5 * (ax[0] + ax[1])))
            hang.append(float(np.sum(sim.support.f)) > 0.7 * sim.BW)
        if sim.fell:
            break
    err = np.array(err[:-100] if sim.fell and len(err) > 150 else err)     # drop the fall itself
    return dict(seed=seed, steps=fsm.steps_total, closes=fsm.closes, step_starts=starts[:10],
                fell=round(d.time - sim.t0, 2) if sim.fell else None,
                progression_m=round(float(d.subtree_com[sim.root][0] - x0), 2),
                foot_progression_m=round(float(np.mean([d.xpos[b][0] for b in tal] - ank0)), 2),
                crutch_hang_frac=round(float(np.mean(hang)), 2) if hang else 0.0,
                est_bias_mm=round(1000 * float(err.mean()), 1) if len(err) else None,
                est_sd_mm=round(1000 * float(err.std()), 1) if len(err) else None,
                therapist_impulse_Ns=round(sim.therapist.work, 1), therapist_peak_N=round(sim.therapist.peak, 1),
                planarizer_impulse_Ns=round(sim.planarizer.impulse, 1), planarizer_peak_N=round(sim.planarizer.peak, 1),
                wall_s=round(time.time() - t0))


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2,3").split(",")]
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    with Pool(W) as pool:
        res = pool.map(run, [(cfg, s, T) for s in seeds])
    for r in res:
        print(json.dumps(r))
    st = [r["steps"] for r in res]
    print(f"SUMMARY steps mean {np.mean(st):.1f} (min {min(st)}, max {max(st)}), "
          f"falls {sum(r['fell'] is not None for r in res)}/{len(res)}, "
          f"progression mean {np.mean([r['progression_m'] for r in res]):.2f} m "
          f"(feet {np.mean([r['foot_progression_m'] for r in res]):.2f} m)")
