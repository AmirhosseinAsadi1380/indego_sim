# Calibrating the model to your exo and your participants

The defaults are a mix of Indego documentation, literature and engineering estimates
(`PARAMETERS.md` tags each one). The order below is the order in which errors hurt a controller most.

## 1. Fit and anthropometry (per participant, 10 min)

* `SubjectParams.height`, `mass`; the builder scales MyoLeg segments and the exo thigh/shank shells to
  the femur and tibia lengths (the Indego is adjusted the same way).
* Cuff positions are taken from the mesh (thigh cuff centre at 0.29 m below the hip axis, shank cuffs at
  0.12 / 0.32 m below the knee). Measure them on the fitted device.
* Joint-axis misalignment: `ExoParams.hip_axis_offset` / `knee_axis_offset` (sagittal, metres). Even
  1–2 cm changes the cuff shear forces noticeably; photograph the fitted participant from the side.
* AFO: choose `afo_type="free"` or `"quasi_static"`; for the latter measure the AFO's stiffness with a
  torque wrench at ±5–10° (`afo_stiffness`, N·m/rad) and its neutral angle (`afo_neutral_deg`).
* Shoes: sole width at the ball of the foot (`shoe_width`) and the toe spring of the shoe the participant
  walks in (`shoe_toe_spring_deg`, measured with the shoe on a flat surface). Both matter: sole width
  sets sideways stability in double support, toe spring sets toe clearance at the end of swing.

## 2. Joint drives — how to identify the Indego actuator dynamics

The drive model (MuJoCo `dcmotor` + firmware layer) has these joint-side quantities:
inertia J (link + N²·rotor), viscous b, Coulomb Fc (optionally static/Stribeck), gravity term, command
latency, current-loop time constant, torque constant (current → torque), thermal R/C, optional
backlash. `examples/03_actuator_identification.py` runs the whole protocol on the virtual bench
(`sysid/bench.py`) so you can rehearse it and see the expected accuracy.

**Data to log** (Indego research interface, 200 Hz): time, commanded torque/current, measured motor
current, joint angle, firmware velocity, and if available winding temperature. Record at the highest
rate the interface allows for test E.

**Setup**: exo off the person, hip unit clamped to a rigid stand so the leg segment hangs freely
(hip test) or the shank hangs from the knee with the thigh clamped (knee test). Remove the AFO/shoe or
weigh it. Use torque/current mode with software limits well below peak; an e-stop within reach.

| test | excitation (`sysid/excitation.py`) | gives | fit (`sysid/actuator_id.py`) |
|---|---|---|---|
| A friction map | `triangle_velocity` constant-speed strokes, 0.05–1.8 rad/s, segment horizontal or gravity-balanced, **PD gains low (kp≈150, kd≈4)** – higher kd limit-cycles with the ~10 ms loop latency | Fc, b (Stribeck if well-tracked points < 0.1 rad/s) | `steady_state_points`, `friction_map` |
| B latency | `steps` ±4–6 N·m | command → measured-current delay (= driver latency + sensing pipeline) | `estimate_delay` |
| C inertia & gravity | `multisine` 0.3–10 Hz, 2–3 N·m rms, segment hanging | J and m·g·l with friction from A fixed (two-stage) | `fit_rigid_body(..., friction=fr)` |
| D cross-check | same data as C | FRF velocity/torque (+coherence) | `frf`, `fit_frf_mass_damper` |
| E rotor inertia | segment detached / light adapter, multisine to 20 Hz, ≥1 kHz logging | N²·rotor inertia (≈2 % of J with the segment on — not identifiable from C) | `fit_rigid_body(gravity=False)` or use the datasheet |
| F current loop | output clamped, current steps | time constant `te` and delay | `fit_current_loop` |
| G thermal | output clamped, constant current 60–120 s, winding temperature or resistance | `thermal_R`, `thermal_C` | `fit_thermal` |
| H backlash (optional) | clamped output, slow ±torque reversals, motor-side encoder if exposed | dead band, contact stiffness | `TransmissionParams(mode="backlash")` |

Expected accuracy on the bench model with realistic noise/quantisation: Coulomb ±1 %, viscous ±5 %,
J ±3–5 % (stick phases excluded via `v_min`), m·g·l ±2 %, delay ±1 sample. Then
`aid.to_motor_params(...)` writes the values into a `MotorParams` you pass as
`ExoParams(hip_motor=..., knee_motor=...)`.

**Validate**: replay test C's excitation on the bench model with the identified parameters and compare
angle traces (VAF > 90 %). Repeat hip and knee, left and right; differences between units are real and
worth keeping per side.

**Torque constant**: if you have a joint torque sensor or a known load (dead weight on a lever),
regress joint torque on current to get N·K·η directly; otherwise `K` absorbs gear efficiency.

## 3. Cuffs and soft tissue

`BushingParams` per interface: translational (N/m) and rotational (N·m/rad) stiffness/damping,
strain-stiffening, soft-tissue wobbling mass. Two practical methods:

1. **Pull test**: participant seated, exo segment fixed, push/pull the cuff with a force gauge in
   AP and vertical directions while filming a marker on the cuff and one on the skin over bone;
   fit F = k·x + k₃·x³ (k₃ = `cubic`·k/range²).
2. **From walking data**: with mocap on the exo and bony landmarks, the cuff–bone relative motion vs
   the exo joint torque gives the effective stiffness; set the bushing so that simulated relative
   motion matches.

Typical thigh soft tissue under a strap: 5–30 kN/m normal, 2–6 kN/m shear. The simulator logs the
bushing wrenches (`F_exo_*` in the log) — the quantity you will want to minimise in transparent
modes.

## 4. Spasticity, clonus and spasms (per participant)

* Start from the clinical exam: Modified Ashworth per muscle group (`SpasticityParams.mas`),
  ISNCSCI motor scores and NLI/AIS (`SubjectParams`).
* **Pendulum test** (quadriceps): record the knee angle after release; `sysid.spasticity.fit_pendulum`
  fits the TSRT velocity term and gain (fix the static threshold from the MAS / catch angle).
  The simulated relaxed healthy R2 is ≈1.45 (the MyoLeg knee range of 120° caps it), MAS 1.5 ≈1.17,
  MAS 2 ≈1.05. One release cannot separate threshold, velocity term and gain — add Tardieu stretches
  at 2–3 speeds.
* **Clonus**: count ankle beats and their frequency (5–8 Hz) after a quick dorsiflexion;
  `clonus_gain_boost` and `delay_scale` set it.
* **Spasms**: rate (`spasm_rate_per_min`), triggers (rapid hip extension → `spasm_stretch_trigger`),
  amplitude/duration from EMG if you have it. Note: with the quasi-static AFO, a bilateral extensor
  spasm of default amplitude tips the simulated user backward (`results/standing_extensor_spasm.*`) —
  check this against your participants and your AFO.

## 5. How the person uses their arms and leans (the least validated part)

Measure, in a normal session: crutch axial loads (instrumented crutches or force insoles under the
crutch tips), crutch tip placement (video), trunk IMU pitch/roll, step timing from the exo, ideally
COM from mocap or a pelvis IMU. Then tune `SupportParams` (and `VolitionParams.lean_trigger_deg`,
`weight_shift`, `anticipation`) so the simulated statistics match — e.g. mean/peak crutch load per phase,
lean before step trigger, time from walk intent to first step, COM sway in quiet standing. Key knobs:

| parameter | effect |
|---|---|
| `com_ahead`, `lean_ahead` | where the person holds the COM when standing / when requesting a step |
| `tip_ahead`, `tip_lateral` | crutch placement |
| `max_load_frac`, `shoulder_moment_max` | arm strength |
| `k_arm`, `arm_set_time(_stepping)` | how stiffly the arms hold the trunk (locked elbows) vs adapt |
| `kp_com`, `kd_com`, `kp_lat_com`, `kd_lat_com`, `reaction_delay`, `prediction` | postural controller |
| `w_lateral`, `w_trunk`, `w_load` | priorities when the crutches cannot do everything |
| `crutch_pattern`, `lift_margin`, `lift_lat_margin`, `crutch_wait_max` | when crutches move |
| `k_yaw_int`, `k_yaw`, `w_yaw` | how firmly the arms hold the heading |
| `tip_margin`, `VolitionParams.expected_speed` | how far / how fast the body travels during an exo step |
| `VolitionParams.lean_patience`, `lean_ramp`, `lean_extra_max` | leaning further when the device does not respond |

A CMA-ES driver for this kind of fit is in `tools/tune_support.py` (edit the target statistics).

## 6. Sensors

Record 30 s with the exo standing still: encoder noise/quantisation, IMU noise and bias
(`SensorParams`). Measure the controller-visible latency by commanding a step and timing the first
encoder change on the 200 Hz stream (it contains `cmd_delay`, mechanics and `sensor_delay`).

Add-on trunk IMU: set `trunk_imu` to where you strap it ("sternum" or "back"), `trunk_imu_rate_hz` and
`trunk_imu_delay` to what reaches your controller (tap the sternum while logging the exo stream and the
IMU stream together: the delay between the hip-unit and trunk-IMU acceleration spikes is the extra
latency). Strap misalignment is removed by calibrating in quiet standing; check it by recording 10 s of
quiet standing and comparing trunk-minus-pelvis pitch across donnings (`trunk_imu_mount_pitch_deg` covers
the residual).

## 7. Validate the whole loop

Run the *same controller* (e.g. the Indego FSM with your step settings) on the device and in the
simulator with the participant's calibrated parameters and compare: joint trajectories, motor currents,
trigger timing, crutch loads, cuff forces (if instrumented). Disagreements usually point to cuff
compliance, AFO stiffness or the person's strategy (§5), in that order.
