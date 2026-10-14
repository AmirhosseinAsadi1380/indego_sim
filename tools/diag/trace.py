"""Print the state of one walking trial every `every` seconds: FSM state, true COM vs feet, the controller's
estimate, crutch forces / tips / lifts, the person's COM targets, foot loads, trunk lean, exo angles.

    python tools/diag/trace.py '<cfg json>' [seed] [T0] [T1] [every]
"""
import json
import sys

import numpy as np
from _common import make_sim

cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
T0, T1 = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.0, 15.0)
every = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
sim, fsm, obs = make_sim(cfg, seed)
m, d = sim.m, sim.d
B = lambda n: d.xpos[m.body(n).id]  # noqa: E731
nxt = T0
for k in range(int(T1 / sim.ctrl_dt)):
    obs = sim.step(fsm(obs))
    t = d.time - sim.t0
    if t >= nxt:
        nxt += every
        sp, db = sim.support, getattr(sim.support, "debug", {})
        c = d.subtree_com[sim.root]
        fl, fr = 0.5 * (B("calcn_l") + B("toes_l")), 0.5 * (B("calcn_r") + B("toes_r"))
        print(f"t={t:5.2f} {fsm.state[:5]:5s}{fsm.swing} com=({c[0]:+.2f},{c[1]:+.3f}) "
              f"feet L=({fl[0]:+.2f},{fl[1]:+.2f}) R=({fr[0]:+.2f},{fr[1]:+.2f}) "
              f"est_front={fsm.est.get('com_rel_front', 0):+.3f} lat={fsm.est.get('lat', 0):+.3f} "
              f"trunk={np.degrees(fsm.est.get('trunk_lean', 0)):+.0f} "
              f"grf={sim._grf['l'][2]:3.0f}/{sim._grf['r'][2]:3.0f} "
              f"crutchF L={sp.F[0].round(0)} R={sp.F[1].round(0)} tips x={sp.tips['l'][0]:+.2f}/{sp.tips['r'][0]:+.2f} "
              f"lift={int(sp.lifted['l'])}{int(sp.lifted['r'])} pend={sp.pending_crutch} "
              f"xdes={db.get('x_des', 0):+.2f} ydes={db.get('y_des', 0):+.3f} "
              f"therapist={np.linalg.norm(sim.therapist.F):3.0f}N q={np.degrees(obs['q']).round(0)}")
    if sim.fell:
        print("FELL", round(t, 2))
        break
