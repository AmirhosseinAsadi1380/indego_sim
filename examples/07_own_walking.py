"""A person walking on their own in the Indego (sagittal plane), no crutches, no therapist.

The person's own walking control (VolitionParams.gait_mode = "walk", HumanWalkParams) drives their muscles; the exo
runs zero torque or its transparent mode (friction + exo-weight compensation). Assumptions of this scenario (see
docs/STATUS.md): sagittal-plane model (SimConfig.planar), a fast long-loop reaction delay (30 ms instead of the
60 ms default) that the person compensates with an internal forward model. The walking parameters are tuned per
subject preset (HumanWalkParams.for_subject): "sci_t12_aisD" (incomplete SCI, the default) and "healthy".

    python examples/07_own_walking.py                         # T12 AIS D, 30 s, zero-torque exo
    python examples/07_own_walking.py --subject healthy --exo transparent --video
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indego_sim.config import HumanWalkParams, SimConfig, SupportParams, VolitionParams  # noqa: E402
from indego_sim.controllers.transparent import TransparentController        # noqa: E402
from indego_sim.human.sci import get_preset                                 # noqa: E402
from indego_sim.sim import HumanExoSim                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subject", choices=("sci_t12_aisD", "healthy"), default="sci_t12_aisD")
ap.add_argument("--exo", choices=("zero", "transparent"), default="zero")
ap.add_argument("--duration", type=float, default=30.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--delay", type=float, default=0.03, help="long-loop reaction delay [s]")
ap.add_argument("--video", action="store_true")
args = ap.parse_args()

subject = get_preset(args.subject)
subject.long_loop_delay = args.delay
vol = VolitionParams(gait_mode="walk", walk=HumanWalkParams.for_subject(args.subject))
sim = HumanExoSim(subject=subject, support=SupportParams(kind="none"), volition=vol,
                  sim=SimConfig(planar=True), schedule=[(2.0, "walk")], seed=args.seed, log_muscles=False)
obs = sim.reset(settle=1.5)
ctrl = TransparentController() if args.exo == "transparent" else (lambda o: np.zeros(4))
m, d, hc = sim.m, sim.d, sim.human
x0 = d.subtree_com[sim.root][0]

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
if args.video:
    from indego_sim import viz
    for g in range(m.ngeom):                          # hide MyoSuite's room (visual only) - the walk leaves it
        if m.geom_bodyid[g] == 0 and m.geom_type[g] == 7:
            m.geom_group[g] = 5                       # not drawn (default visible groups 0-2)
    path = os.path.join(OUT, f"own_walking_{args.subject}_{args.exo}.mp4")
    viz.record(sim, ctrl, args.duration, path, azimuth=90.0, distance=3.0, elevation=-5.0)
    print("video ->", os.path.abspath(path))
    steps = None
else:
    steps, prev = 0, None
    for _ in range(int(round(args.duration / sim.ctrl_dt))):
        obs = sim.step(ctrl(obs))
        sw = hc.ws.get("swing") if hc.ws.get("on") else None
        if sw is not None and sw != prev:
            steps += prev is not None
            prev = sw
        if sim.fell:
            break
t = d.time - sim.t0
dist = d.subtree_com[sim.root][0] - x0
print(f"{args.subject} exo={args.exo} delay={args.delay * 1e3:.0f} ms  fell={sim.fell}  t={t:.1f} s  distance={dist:.2f} m  "
      f"speed={dist / max(t - 2.0, 1e-3):.2f} m/s" + (f"  steps={steps}" if steps is not None else ""))
