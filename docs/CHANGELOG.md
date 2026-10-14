# Changelog

## Unreleased

Floor (behaviour change for every scenario)
* MyoSuite's scene stood the model on a pedestal: a collision cylinder of 1.05 m radius (top at z = 0.015 m) with
  the floor plane 0.4 m lower, so anyone who walked more than ~1 m stepped off its edge and fell. The builder now
  raises the floor plane to the pedestal top (same contact parameters) and switches the cylinder's collision off
  (`model.builder._flat_ground`; its visual mesh stays). Within the old pedestal the surface is the same, but the
  foot contacts are now generated against a plane instead of a cylinder, so results move slightly: default crutch
  walking (`eval_walk.py '{}'`, seeds 0-3, 30 s) still makes 1 step and falls 4/4, now at 10.5/14.9/10.2/15.1 s
  (was 12.5/11.0/13.2/12.3 s); quiet standing (`stand_all.py`) still stands for all presets (sway changes by a few
  mm); example 05 episode 0 2.5 s, 7 heel strikes, return -14.37, 417 N (was -13.91, 419 N), episode 1 9.8 s,
  -12.05, 156 N (was 9.2 s, -11.90, 158 N). Earlier walking results that progressed more than ~1 m (several in
  docs/STATUS.md) ended at the pedestal edge.

The person walks on their own (`VolitionParams.gait_mode = "walk"`, sagittal model; defaults of other modes
unchanged)
* `HumanWalkParams` / `HumanController._walk_torque`: SIMBICON-style walking executed through the muscles (static
  optimisation, capacity-limited) on the state sensed one long-loop delay ago, extrapolated by an internal forward
  model (`predict`). Gait initiation (lean forward on the ankles, first step once the COM is `init_dx` ahead); swing:
  the thigh's world angle goes to the foot-placement angle theta0 + c_d*d + c_v*v on a min-jerk path with velocity
  feed-forward, the knee is held loosely while it folds and extends for touch-down once the thigh is forward, the
  swing leg's weight is compensated, stance knee/ankle are held, push-off late in the swing, the stance hip holds the
  pelvis (forward lean when slower than `v_des`) and cancels the swing hip's reaction; the swing ends at the felt
  touch-down ahead of the stance foot; double support: weight onto the front foot with trailing-ankle push-off until
  the trailing leg unloads (a loaded leg cannot swing). Tuned with CMA-ES (30 ms long-loop delay -
  `SubjectParams.long_loop_delay`, default 0.06 s -, `SimConfig.planar`, no crutches or help): the defaults for the
  incomplete-SCI preset `sci_t12_aisD` - zero-torque Indego 8/8 seeds walk 30 s, 17.6-18.0 m (0.64 m/s), 56 steps,
  60 s 2/2 (37.3 m); transparent controller 6/8 (falls at 16.5 and 26 s) -, and `HumanWalkParams.for_subject
  ("healthy")` (`TUNED_WALK`) - zero-torque 8/8 (20.3-20.5 m, 0.72 m/s), transparent 8/8 (19.4-20.0 m). Each set is
  specific to its person (the T12 set makes the healthy preset fall in 8-14 s and vice versa).
* `examples/07_own_walking.py` (`--subject sci_t12_aisD|healthy`, `--exo zero|transparent`, `--delay`, `--video`;
  the video hides MyoSuite's room, which the walk leaves). The floor plane is drawn 100 m wide.
* 16 tests (unbounded flat floor; T12 AIS D walks on their own in the exo, 6 s).
* `VolitionParams.walk`/`walk_effort` and the other walking-strategy settings are excluded from `randomize`
  (domain randomisation of `IndegoEnv` reproduces the earlier episodes for the same seeds).

Repository
* The exo segment meshes are no longer needed: the builder uses them when they are installed and otherwise gives
  every segment a box with the mesh's mass properties (`model.builder.MESH_BOXES`: same mass, centre of mass within
  0.1 mm, inertia within 0.3 %); the segment geometry is in `model.builder.INDEGO_GEOMETRY`
  (`assets/indego/indego_geometry.json`, when present, takes precedence). The meshes are git-ignored.
* `tools/diag/eval_own_walk.py` (own-walking score over seeds) and `tools/diag/tune_own_walk.py` (CMA-ES tuning of
  `HumanWalkParams` for a person / delay / device).
* `docs/DEVELOPMENT.md`: setup, commands, conventions, rules for changes, next steps.

Test rig and diagnostics (no change to the default behaviour: with the planarizer off, walking and standing
results are bit-identical to 0.2)
* `PlanarizerParams` / `human.support.Planarizer`: optional planarizing boom on the pelvis (`SupportParams.planarizer`,
  off by default) that holds sideways position, roll and heading with a stiff, undelayed spring-damper and leaves
  forward/vertical translation and pitch free; sideways force and moment are logged (`planarizer_force`,
  `planarizer_moment`). Excluded from `randomize` (a rig, not the person). Use it with `fsm.lat_dy = 0`: the pelvis
  cannot shift sideways, so the FSM's lateral weight check would never pass.
* `IndegoFSMController.steps_total` and `.closes`: `n_steps` is reset when a double-support time-out closes the feet,
  so it under-counted walking trials; `tools/diag/eval_walk.py` and example 02 now report `steps_total`.
* `eval_walk.py` also reports feet-closures, foot progression (COM progression includes the fall), the fraction of
  time the person hangs on the crutches (> 70 % BW), and planarizer help; `tools/diag/_common.py` accepts
  `"planarizer": {...}` in `"sup"`.
* `tools/gen_parameters_md.py` writes LF line endings on every OS (Windows wrote CRLF).
* 14 tests (planarizer holds the pelvis in the sagittal plane under a 150 N sideways push).

Sagittal-plane model and incomplete-SCI users walking without support (experimental; defaults unchanged)
* `SimConfig.planar`: sagittal-plane model - lumbar side-bend/rotation, hip ab/adduction/rotation, subtalar and the
  exo's hip ab/adduction/twist compliance are held at their standing values (joint equality constraints, qpos layout
  unchanged) and the planarizer holds the pelvis. Diagnostic configs take `"sim": {...}`; with `"planar": true` the
  FSM's lateral weight check defaults to off.
* `VolitionParams.gait_mode = "exo"` (people with leg volition): the exo's walking mode drives the steps and the
  person cooperates with their own muscles - lets the exo move hips/knees, keeps the pelvis upright with the stance
  hip(s), holds a loaded knee extended, centre-of-pressure ankle control within the flat-foot limits (gravity
  included, slow postural adaptation), requests the first step with a learned `request_lean`, shifts the weight to
  5 cm behind the front ankle in double support (speed-limited, trailing-ankle `push_off` + a pelvis-neutral hip/knee
  push, `transfer_gain`), lets the COM roll over the stance foot at `exo_walk_speed` during an exo step, lifts the
  swing toes; sideways weight shift / foot placement with the hip ab/adductors (3D). Felt from knee motion and foot
  load (`HumanExoSim._felt_swing`). Without it ("own", the default) a walker follows their own gait template and
  fights an exo in walking mode.
* `IndegoFSMParams.ds_kp_hip`: hip stiffness in double support (default: unchanged). A user who shifts their own
  weight needs the hips to give way (hips and knees locked + both feet down = no motion possible).
* `tools/diag/_common.py`: `"subject": {...}` overrides SubjectParams fields (e.g. motor scores).

Exo-led dynamic walking (experimental; defaults unchanged)
* `controllers/dynamic_walk.py` - `DynamicWalkController`: sagittal dynamic walking by foot placement (Raibert
  rule on the device's own COM estimate and velocity, swing ankle planned in ankle space with leg IK for the measured
  knee bend + velocity feed-forward, stance hip keeps the hip-unit pitch with priority over the swing hip inside the
  torque limit, landing point filtered then frozen); steps end on time or, with foot switches, at touch-down near the
  landing point. Diagnostics: `"ctrl": "dynamic"` (also `"transparent"`, `"aan"`, `"zero"` for people walking
  themselves).
* `VolitionParams.gait_mode = "follow"`: the exo walks the person, who bears weight on the loaded knee, regulates
  speed with the ankles (`follow_speed_gain`, `exo_walk_speed`) and adds `follow_assist` x the exo torque felt on the
  swinging leg (one reaction delay late); `walk_effort` scales leg effort while walking.
* `SensorParams.foot_switches` (add-on, off by default): `obs["foot_contact"]` (not part of the 44-D flat observation).
* `VolitionParams.transfer_target` ("exo" mode): where the weight transfer puts the COM relative to the front ankle.

## 0.2 — trunk IMU, footwear, crutch strategy

Sensors and controller
* Add-on upper-trunk IMU (`SensorParams.trunk_imu = "sternum" | "back" | "none"`, own rate, latency and
  strap misalignment); `ExoSensors` returns `acc_trunk`, `gyr_trunk`, `trunk_quat`; flat observation and
  `IndegoEnv` grow from 34 to 44 values (34 with `trunk_imu="none"`).
* FSM COM estimator rewritten as a segment model in the IMU frames (`IndegoFSMController.from_model(m)`
  measures the constants on any model); uses the trunk IMU's lumbar angle when present; lateral COM
  estimate from hip-unit (+ trunk) roll.
* FSM triggers after the Vanderbilt controller: double-support threshold drops with step length
  (`ds_trigger_len_gain`), needs a deliberate forward shift (`ds_lean_dx`) and the weight shifted toward
  the next stance foot (`lat_dy`). New swing defaults (hip 36°, knee 60° peaking at 50 % of the step)
  for toe clearance in shoes; `ds_max_wait` 3 → 6 s.

Human model
* Shoe soles (9.5 cm wide, 12° toe spring) replace MyoLeg's narrow bare-foot contact capsules.
* Crutch/arm model: heading (yaw) control; crutch lift order and lateral-safety rule; no deadlock when a
  crutch ends up under the shoulder; tips re-planted within reach; weight transfer only after the crutches
  are advanced; the body travels forward inside the crutch tripod during an exo step; the person leans
  further when the device does not respond (`lean_patience`, `lean_ramp`, `lean_extra_max`).

Tools, tests, docs
* `tools/render_overview.py`; `tools/gen_parameters_md.py` no longer merges section comments into the
  previous parameter. 13 tests (trunk IMU placement/observation, shoe soles, trunk-flexion COM estimate).
* Example 02: `--no-trunk-imu`, `--seed`, reports first-step time and COM-estimate accuracy vs ground truth.
* `tools/diag/`: walking diagnostics (multi-seed score incl. therapist help, swing-shape sweep, trial trace,
  standing regression); development notes with setup, conventions and next steps (now `docs/DEVELOPMENT.md`).
