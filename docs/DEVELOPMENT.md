# Development notes

Setup, commands, conventions and the rules the model is developed by. What works and what does not is in
[STATUS.md](STATUS.md), changes in [CHANGELOG.md](CHANGELOG.md), every parameter with its provenance in
[PARAMETERS.md](PARAMETERS.md).

## Setup

```bash
python -m pip install "mujoco==3.15.0" numpy scipy pyyaml gymnasium matplotlib "imageio[ffmpeg]" cma pytest
git clone https://github.com/MyoHub/myo_sim && python -m pip install --no-deps -e myo_sim   # MyoLeg model
python -m pytest -q tests          # 16 tests, must stay green (a few minutes)
```

Developed with Python 3.12-3.13 and mujoco 3.15.0 on Linux and Windows. Headless video on Linux: `MUJOCO_GL=egl`;
on macOS / Windows leave `MUJOCO_GL` unset (GLFW).

Exo meshes: the Indego segment meshes in `indego_sim/assets/indego/` are made from a commercial 3D model and are
not distributed. Without them every exo segment is drawn and weighted as a box with the mesh's mass properties
(same mass, centre of mass within 0.1 mm, inertia within 0.3 %), so results do not depend on having them. To build
them from your own copy of the model: `pip install bpy`, then `tools/export_blend.py` -> `tools/process_indego_mesh.py`.

## Commands

| what | command | time |
|---|---|---|
| tests | `python -m pytest -q tests` | a few min |
| a person walking on their own in the exo | `python examples/01_own_walking.py [--subject healthy] [--exo transparent] [--video]` | ~1 min |
| quiet standing (SCI, crutches) | `python examples/02_quiet_standing.py [--video] [--spasms]` | ~1 min |
| Gymnasium env + your controller | `python examples/03_your_controller_env.py` | ~1 min |
| transparency (healthy standing) | `python examples/04_transparency.py` | ~2 min |
| actuator identification on a virtual bench | `python examples/05_actuator_identification.py` | ~2 min |
| spasticity from a pendulum test | `python examples/06_pendulum_test.py` | ~10 min |
| FSM walking score over seeds | `python tools/diag/eval_walk.py '<cfg json>' 0,1,2,3 25` | ~1 min / 2 seeds / core |
| own-walking score over seeds | `python tools/diag/eval_own_walk.py '<cfg json>' 0,1,2,3 30` | ~2 min |
| tune the own walking for a person | `python tools/diag/tune_own_walk.py '<cfg json>' out.jsonl` | ~1 h on 16 cores |
| step-shape comparison | `python tools/diag/step_sweep.py '[{...},{...}]'` | |
| trace one trial | `python tools/diag/trace.py '<cfg json>' seed T0 T1 every` | |
| standing regression, all presets | `python tools/diag/stand_all.py` | ~3 min |
| fit the crutch/arm strategy to session statistics | `python tools/tune_support.py` | |
| regenerate docs/PARAMETERS.md | `python tools/gen_parameters_md.py` (after any config.py change) | |
| regenerate results/model_overview.png | `python tools/render_overview.py` | |

Diagnostic configs (JSON): `{"subj":..., "walk_at":2.0, "ctrl":..., "fsm":{IndegoFSMParams}, "sup":{SupportParams,
"therapist":{TherapistParams}, "planarizer":{PlanarizerParams}}, "vol":{VolitionParams}, "sens":{SensorParams},
"exo":{ExoParams}, "sim":{SimConfig}, "subject":{SubjectParams overrides}}`; `"ctrl"`: `"fsm"` (default),
`"dynamic"`, `"transparent"`, `"aan"`, `"zero"`. Sagittal-plane model: `"sim":{"planar":true}` (locks the
frontal/transverse joints and holds the pelvis in the plane; the FSM's lateral check is off). The person walking on
their own: `{"ctrl":"zero","subj":"sci_t12_aisD","subject":{"long_loop_delay":0.03},"sup":{"kind":"none"},
"sim":{"planar":true},"vol":{"gait_mode":"walk"}}` (the default scenario of `eval_own_walk.py`).

## Layout

```
indego_sim/config.py          every parameter (dataclasses) with provenance tags [doc]/[lit]/[est]/[model]
indego_sim/model/builder.py   MjSpec assembly: MyoLeg + 3 lumbar DOF, subject scaling, shoes, flat floor, exo
                              segments, 6-DOF cuff bushings, dcmotor drives, sensors (incl. trunk IMU), keyframe
indego_sim/exo/               drive firmware layer (current limits, fold-back), onboard sensors (ExoSensors)
indego_sim/human/             muscles + static optimisation, reflexes/spasticity, SCI presets, volition,
                              HumanController (own walking, exo cooperation), support.py (crutches, walker,
                              harness, therapist, planarizer)
indego_sim/sim.py             HumanExoSim: closed loop, timing, crutch-move scheduling, contacts, logging
indego_sim/controllers/       indego_fsm.py (baseline Indego-like FSM), dynamic_walk.py (foot-placement walking),
                              transparent.py (+ assist-as-needed)
indego_sim/env.py             Gymnasium IndegoEnv (44-D obs: 34 stock + 10 trunk IMU), domain randomisation
indego_sim/sysid/             excitation, virtual bench, actuator_id, spasticity (pendulum test)
tools/diag/                   walking diagnostics and tuning
```

Timing: physics 0.5 ms (implicitfast), human/support tick 1 kHz, voluntary static optimisation every 10 ms
with a 60 ms delay (`SubjectParams.long_loop_delay`), controller 200 Hz with 3 ms sensor + 2 ms command delay
(~10-11 ms loop latency).

## Conventions and pitfalls

* Frames: x forward, y left, z up. Exo joints: + = flexion for hip and knee. Pitch + = forward, roll + = toward
  the left. IMU frames are aligned with the world in the reference posture (qpos0).
* Joint-space PD on the exo: kd above ~8 N·m·s/rad gives a 10-15 Hz limit cycle (loop latency + compliant
  cuffs). Bench sweeps use kp 150 / kd 4.
* dcmotor LuGre friction only with the Euler integrator (implicit integrators add spurious damping); default
  friction is Coulomb via joint frictionloss.
* MyoLeg knee coupling: call `model.utils.apply_couplings` whenever you set qpos by hand.
* MjSpec: look up all `spec.geom(name)` references **before** adding new geoms - lookups after additions
  returned the wrong elements (see `_add_shoes`).
* Joint stiffness/damping in MjSpec are 3-arrays (`[k, 0, k3]`).
* The trunk IMU is strap-mounted: use relative angles (trunk vs pelvis, change since quiet standing).
* MyoSuite's scene stands the model on a 1.05 m-radius pedestal; `builder._flat_ground` replaces it by an unbounded
  floor. If walking results ever end near 1 m of progress, check the ground first.
* Searching walking parameters: score forward progress, not only survival (survival alone converges to marching in
  place), and validate over >= 4 seeds and 30 s. Tuned sets are specific to the person, delay and device.
* config.py comments are parsed into docs/PARAMETERS.md: inline `# note [tag]`, continuation lines indented
  under the comment column; field-level `#` lines are section comments.
* Every parameter keeps a provenance tag. Do not present [est]/[model] values as measured.

## Rules for changes

* Do not tune the simulated person just so a controller passes. Every human-model change needs a physical
  justification (what real users do) and must keep quiet standing (`tools/diag/stand_all.py`) and the tests
  green. Evaluate walking over >= 4 seeds, not a single trace.
* Keep docs/STATUS.md honest and add an entry to docs/CHANGELOG.md for every behaviour change.
* Do not commit the exo meshes (`indego_sim/assets/indego/`, see .gitignore).

## Next steps

1. Own walking for the users of interest: robustness (pushes, slopes, the transparent exo), the default 60 ms delay,
   slower speeds (Indego users walk at 0.2-0.5 m/s); then use the walking person to test exo controllers
   (transparent, assist-as-needed).
2. Re-run the exo-led walking evaluations (`gait_mode` "exo"/"follow", crutch walking with help) on the flat floor.
3. Calibrate the person with session data ([CALIBRATION.md](CALIBRATION.md) §5): crutch loads or tip placement,
   trunk-IMU traces, exo step timing / joint logs; fit `SupportParams` / `VolitionParams` with
   `tools/tune_support.py`.
4. If the real Indego step trajectories and trigger thresholds are available, put them in `IndegoFSMParams` (or
   replay them) - then any sim/real mismatch is a modelling error.
5. Per-episode randomisation of the crutch strategy in `IndegoEnv`; sit-to-stand; 3D (sideways) balance.
