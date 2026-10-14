# Results

Produced by the examples and diagnostics of this repository (current model, flat floor); the console output of each
run is in `logs/`. Regenerate any of them with the command in the table.

## The person walking on their own in the exo

Sagittal-plane model, no crutches or help, 30 ms long-loop reaction delay, `HumanWalkParams` tuned for the person.

| | file | command |
|---|---|---|
| T12 AIS D (incomplete SCI), zero-torque exo, 30 s: 17.6 m, no fall | `own_walking_sci_t12_aisD_zero.mp4` | `python examples/07_own_walking.py --video` |
| healthy strength, zero-torque exo, 30 s: 20.3 m, no fall | `own_walking_healthy_zero.mp4` | `python examples/07_own_walking.py --subject healthy --video` |

Scores over seeds (`tools/diag/eval_own_walk.py`):

| person | exo | delay | walked the full time | distance | log |
|---|---|---|---|---|---|
| T12 AIS D | zero torque | 30 ms | 8/8 seeds x 30 s | 17.6-18.0 m (56 steps) | `logs/eval_own_walk_t12_zero.txt` |
| T12 AIS D | zero torque | 30 ms | 2/2 seeds x 60 s | 37.3-37.4 m | `logs/eval_own_walk_t12_60s.txt` |
| T12 AIS D | transparent | 30 ms | 6/8 seeds x 30 s (falls at 16.5, 26.0 s) | 8.2-17.0 m | `logs/eval_own_walk_t12_transparent.txt` |
| T12 AIS D | zero torque | 60 ms (default, not re-tuned) | 0/4 (falls at 8.8-10.2 s) | 2.4-3.3 m | `logs/eval_own_walk_t12_delay60ms.txt` |
| healthy | zero torque | 30 ms | 8/8 seeds x 30 s | 20.3-20.5 m (53 steps) | `logs/eval_own_walk_healthy_zero.txt` |
| healthy | transparent | 30 ms | 8/8 seeds x 30 s | 19.4-20.0 m | `logs/eval_own_walk_healthy_transparent.txt` |

## Standing, exo-led walking, identification

| result | file | command |
|---|---|---|
| T10 AIS A standing with crutches, 10 s: COM sway 10 mm, crutches 22 % BW | `quiet_standing_sci.*` | `python examples/01_quiet_standing.py --video` |
| the same with forced spasms (extensor spasms of the left leg) | `standing_extensor_spasm.*` | `python examples/01_quiet_standing.py --spasms` |
| all presets stand (crutches; healthy without support) | `logs/stand_all.txt` | `python tools/diag/stand_all.py` |
| T10 AIS A walking with crutches and the baseline FSM: one clean step, fall at 13.1 s | `walking_sci_t10_aisA_crutches.*` | `python examples/02_sci_walking_fsm.py --video` |
| the same over 4 seeds: 1 step, falls at 10.5-15.1 s | `logs/eval_walk_fsm_crutches.txt` | `python tools/diag/eval_walk.py '{}' 0,1,2,3 30` |
| overhead harness + timer stepping: 8 steps, no fall, no net progress, hip motors ~115 C | `walking_sci_t10_aisA_harness.png` | `python examples/02_sci_walking_fsm.py --support harness --trigger auto` |
| drive identification on a virtual bench (Coulomb -0.1 %, viscous -3 %, inertia +3 %) | `actuator_id.png` | `python examples/03_actuator_identification.py` |
| spasticity from a pendulum test (R2 healthy 1.44, T10 A 1.17; fitted velocity term within 2 %) | `pendulum_test.png` | `python examples/04_pendulum_test.py` |
| Gymnasium env, two randomised episodes | `logs/05_your_controller_env.txt` | `python examples/05_your_controller_env.py` |
| transparency, healthy standing: thigh-cuff force 11.9 N rms (zero torque) vs 10.3 N (transparent) | `transparency.png` | `python examples/06_transparency.py` |
| model overview render | `model_overview.png` | `python tools/render_overview.py` |
