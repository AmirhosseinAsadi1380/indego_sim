# Modelling notes, conventions and pitfalls

## Frames and signs

* World: x forward, y left, z up (the MyoLeg composite faces −y and is rotated 90° about z).
* Exo joints: hip + = flexion, knee + = flexion, AFO hinge + = plantarflexion. Torque commands follow
  the same signs. Order everywhere: hip_l, knee_l, hip_r, knee_r.
* MyoLeg: knee_angle + = flexion, ankle_angle + = dorsiflexion; hip_adduction + = adduction (both sides).
* Pitch + = forward lean, roll + = top toward +y (left).
* With the feet planted, + hip-flexion torque pitches the pelvis forward; + adduction of the left
  stance hip rolls the pelvis toward −y, of the right toward +y. (Getting these wrong produced a
  healthy-standing collapse earlier; the balance code is commented accordingly.)

## Timing

physics 0.5 ms (implicitfast) · spinal/arm layer 1 kHz · voluntary static optimisation every 10 ms on
60 ms-delayed state · exo controller 200 Hz with sensor sample-and-hold + 3 ms acquisition delay and
2 ms command delay. Total loop latency seen by the controller ≈ 10–11 ms.

**Consequence**: joint PD damping above ~8 N·m·s/rad gives a 10–15 Hz limit cycle with the compliant
cuffs (the FSM uses kd = 6). The same applies to position-controlled bench sweeps (use kd ≈ 4).

## Drives (MuJoCo `dcmotor`)

* Torque-mode driver (`mjINPUT_FF`) with back-EMF compensation, voltage limit, electrical time
  constant, thermal state (winding temperature changes resistance), cogging; rotor inertia via
  actuator `armature` and viscous friction via actuator `damping` (both reflected by gear²).
* **LuGre + implicit integrators**: the bristle damping σ₁·N² enters the implicit damping matrix and
  adds ~290 N·m·s/rad of spurious damping at the joint. The default friction model is therefore
  Coulomb via joint `frictionloss` (works with any integrator); LuGre is allowed only with
  `integrator="euler"` (the builder raises otherwise).
* `frictionloss` is a soft constraint: at very low speed the friction torque is below Fc and stick
  phases are not ideal Coulomb. Identification therefore excludes |q̇| < `v_min` (0.3 rad/s).
* Firmware layer (`ExoDrive`): torque → current setpoint, peak/continuous current with linear thermal
  fold-back over the last 15 K, knee brakes as extra dry friction.

## Interface

* Each cuff is a body with 3 slide + 3 hinge joints (stiffness [k, 0, k₃] polynomial: strain
  stiffening) whose small mass/inertia is the soft-tissue wobbling mass, welded to the bone with
  solref = 2·dt. A static test showed the weld deflects µm vs mm for the bushing, i.e. the weld is
  effectively rigid and the bushing is the compliance.
* Logged interaction wrench = spring + damper force of the bushing DOFs (`F_exo_*`, exo-segment axes).

## Muscles and voluntary control

* Hill-type MuJoCo muscles (MyoLeg F0, L0, tendon slack); vectorised gain/bias match
  `mju_muscleGain/Bias` exactly (tested). Moment arms from the sparse actuator moment matrix with the
  knee/patella coupling derivatives folded in (checked against finite differences).
* Voluntary drive: delayed PD + template feedforward → static optimisation (FISTA warm start,
  activation bounds = capacity × effort, smoothness term) every 10 ms. SCI weakness enters only as
  capacity bounds and atrophy (F0 scaling).
* Trunk: 3 lumbar torque generators with intrinsic (undelayed) + delayed long-loop terms. A purely
  delayed (80 ms) trunk PD was unstable.
* Balance for people with leg volition and no support: ankle strategy (COM vs ankles, gain 1.5·m·g),
  hip strategy for the pelvis, SIMBICON-like swing placement while walking.

## Spinal layer

* TSRT spasticity: excitation k·max(0, p(t−d) − (λ − μ·v(t−d))), with p the position of the muscle in
  its **length range over the anatomical joint ROM** (computed by sampling joint configurations, so λ
  is a fraction of the ROM like a clinician's catch angle) and v the fibre stretch velocity in L₀/s
  (25 Hz afferent filter). Delays 20/25/35 ms (hip/thigh/shank muscles).
* MAS → (λ, μ, gain, tonic) table in `reflexes.py`, calibrated so the simulated pendulum test grades
  healthy (R2 ≈ 1.45) > MAS 1.5 (≈1.17) > MAS 2 (≈1.05).
* Normal (relaxed) stretch reflex gain 0.005 per L₀/s: 0.03 made a healthy pendulum test over-damped.
* Reflex output capped at 0.6; clonus emerges from delay × gain (`clonus_gain_boost` on
  plantarflexors); spasms are Poisson or stretch-triggered extensor/flexor bursts with 6 Hz modulation.

## Upper-limb support

* **Crutch physics**: a crutch pinned at its rubber tip transmits a ground reaction R whose line of
  action passes through the tip. R must be push-only, inside the friction cone and below the per-arm
  cap; the part not along the crutch axis needs a shoulder moment (`shoulder_moment_max`). Applied to
  the trunk as force R with moment (tip − trunk COM) × R. (A pure axial strut is the special case
  `shoulder_moment_max = 0`.)
* Horizontal crutch forces have no moment about a ground-level pivot: crutches ahead of the feet can
  stop a forward fall but cannot pull a person who sits back — backward falls must be prevented by
  posture (trunk lean, COM kept ahead), as in reality.
* **Controller**: delayed (120 ms, Smith-predictor-like extrapolation) whole-body balance about the
  feet's **centre of pressure** (so passive ankle torque is accounted for) + intrinsic trunk
  orientation impedance; allocation by weighted least squares with lateral balance prioritised
  (`w_lateral`), then projection on each crutch's feasible set and re-solve (bounded LS).
* **Intrinsic arm stiffness**: each extended arm + crutch is a stiff strut of commanded length L₀
  (`k_arm`); the person moves L₀ to get the load they want (`arm_set_time`) and "locks" the arms while
  the exo steps (`arm_set_time_stepping`).
* **Heading (yaw)**: the arms also hold the heading (`k_yaw_int`/`d_yaw_int` intrinsic, `k_yaw`/`d_yaw`
  delayed, weight `w_yaw`): with the tips ~0.66 m apart, opposite fore-aft tip forces twist the body.
  Without this row in the allocation any asymmetric tip force left a free yaw moment and the simulated
  user spun round (−140°) during single support.
* COM targets (sagittal):
  * standing / double support: mid-feet + `com_ahead`;
  * requesting a step: front foot + `lean_ahead` (+ up to `lean_extra_max` if the device does not
    respond within `lean_patience`, "lean until it steps");
  * while the exo steps: the body may travel forward inside the tripod (stance foot … crutch tips −
    `tip_margin`) at the expected walking speed — the person "falls" into the step instead of pinning the
    COM over the stance foot (which made every step land with the body static and far behind);
  * while advancing crutches after a step: hold where the body is (no retreat to mid-feet).
* Lateral: a weight shift (`weight_shift`, `anticipation`) toward the next stance foot, but only **after
  the crutches have been advanced** (land → re-set the arms → move the crutches → transfer weight → lean).
* Crutch moves ("ds" pattern): after a step is settled both crutches are advanced one at a time — the
  stepping side first (the weight is still on the other side). A crutch is lifted only when (a) the COM
  is not ahead of its target (`lift_margin`) — or the tip is no longer ahead of the shoulder
  (`tip_useless`) or the person has waited `crutch_wait_max` — and (b) the COM is laterally at least
  `lift_lat_margin` inside the opposite foot: the remaining crutch can only push the body *toward* the
  lifted side, so a fall toward it could not be caught. Tips are re-planted within reach (`tip_ahead`
  ahead of the body, at least level with that side's toes); a crutch planted far ahead of the shoulder can
  only push the person back.
* Harness: two straps from the shoulders to a spreader bar on a robotic trolley that follows the
  person with a lag — vertical BWS plus pendulum restoring forces. Therapist: force/moment-limited
  impedance on the pelvis toward the person's own target, logged as an assistance measure.

## Baseline controller (Indego-like FSM)

States stand → step → double support → (step | close → stand), after the Vanderbilt controller the
Indego descends from (Quintero, Farris & Goldfarb 2012).

* **COM estimate**: sagittal segment model — pelvis group, trunk above L5/S1, thigh, shank and foot per
  leg — with mass fractions and COM vectors expressed in the IMU frames (`IndegoFSMParams.m_*`, `r_*`,
  `c_*`; `IndegoFSMController.from_model(m)` measures them on any scaled model, on the device use
  anthropometric tables + the exo's mass properties). With stock sensors the trunk is assumed to keep
  its quiet-standing angle to the pelvis, so **trunk flexion is invisible**: 10° of flexion moves the
  true COM 1.4 cm, and the stock estimate misses all of it (the first-step threshold is 2 cm). With the
  add-on trunk IMU the lumbar angle change since calibration is added (static error 0 in the test).
* **Lateral COM estimate**: whole-body roll from the hip-unit AHRS (+ trunk roll relative to the pelvis
  with the trunk IMU).
* **Triggers**: first step when the COM estimate is `stand_trigger_dx` ahead of quiet standing; later
  steps when it is ahead of the front ankle by `ds_trigger_dx − ds_trigger_len_gain × step length` (the
  Vanderbilt threshold drops linearly with step length) **and** the user has shifted it `ds_lean_dx`
  forward of its rearmost position in this double support (a deliberate lean, not leftover momentum);
  both also require the weight to be shifted sideways toward the next stance foot (`lat_dy`; frontal
  lean selects the leg). Otherwise the exo lifted a still-loaded leg.
* Min-jerk hip trajectory that leads the knee (`hip_reach`), knee "bump" peaking late (`knee_peak_at`
  0.5) so the foot does not drop while the hip is already at its goal, stance knee driven into
  extension, swing hip never braked below its goal (`swing_hip_one_sided`), double support holds the
  reached posture with a small weight-transfer extension. Optional world-frame swing with stance-hip
  pelvis control (SIMBICON-style, off by default). Triggers: lean / auto (timer) / external
  (`request_step()`).

## Footwear

MyoLeg's bare-foot contact is two lengthwise capsules ~4 cm apart (and ~1 cm apart under the toes): the
foot rolls sideways almost freely, so in double support the whole body could tip sideways about the feet
(both feet rolled 13° onto their edges). Exo users stand in shoes on an AFO footplate, so
`SubjectParams.shoes` (default on) replaces those capsules by a heel-to-ball sole box and a forefoot box,
`shoe_width` wide (9.5 cm at 1.75 m), with `shoe_toe_spring_deg` (12°) of upward bend under the toes —
without it the flat forefoot caught the ground in late swing and every step stopped at ~20 cm.

## Add-on trunk IMU

`SensorParams.trunk_imu = "sternum" | "back" | "none"`: a site on the person's torso, placed from the
rib-cage / vertebra meshes (mid-sternum, or T3–T5 spinous processes) + 1.5–2.5 cm for strap and housing,
axes aligned with the hip-unit IMU in the reference posture, optional strap misalignment
(`trunk_imu_mount_pitch_deg`, randomised ±5° by the env). It has its own rate and latency
(`trunk_imu_rate_hz`, `trunk_imu_delay`) because add-on IMUs are often wireless (100 Hz, 10–30 ms).
Observation: `acc_trunk`, `gyr_trunk`, `trunk_quat` (44-D flat vector instead of 34). Use relative
angles (trunk vs pelvis, change since quiet standing) so mounting errors cancel.

## Performance

~1× real time for a full SCI session on one core (80 muscles, 1 kHz reflex layer, 0.5 ms physics);
model build ~1 s. Use `interface_model="weld"` and `log_muscles=False` for speed.
