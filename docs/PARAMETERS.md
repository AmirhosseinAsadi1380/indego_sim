# Parameters and provenance

Generated from `indego_sim/config.py` by `tools/gen_parameters_md.py`. Tags: **[doc]** from Indego
documentation / papers on the Indego, **[lit]** from the biomechanics / SCI literature, **[est]** an
engineering estimate that you should measure, **[model]** a modelling choice (tuned for stability or
behaviour, not a physical constant), **[calib]** set by a calibration in this package.

Anything tagged [est] or [model] is a candidate for the calibration procedures in `CALIBRATION.md`.

## `MotorParams`

One Indego joint drive (BLDC motor + reduction), mapped onto MuJoCo `dcmotor`.  Motor-side quantities (torque, friction, inertia) are reflected to the joint by `gear` (torque x N, inertia x N^2) inside MuJoCo.

| parameter | default | notes |
|---|---|---|
| `gear` | `24.0` | [doc:Quintero2011] 24:1 reduction (prototype); [est] for commercial unit |
| `K` | `0.051` | N*m/A effective torque const incl. ~85% gear efficiency [est] |
| `R` | `0.35` | ohm terminal resistance [est] (200 W-class BLDC) |
| `te` | `1.0e-4` | s electrical time constant L/R [est] |
| `di_dt_max` | `0.0` | A/s current slew limit (0 = off) [est] |
| `v_max` | `28.0` | V drive voltage limit (29.6 V Li-ion pack) [doc:Quintero2011 battery] |
| `i_peak` | `49.0` | A peak current -> 60 Nm joint peak [doc:Farris2012 up to 65 Nm; Quintero2011 prototype ~40 Nm] |
| `i_cont` | `9.8` | A continuous current -> 12 Nm joint continuous [doc:Quintero2011] |
| `rotor_inertia` | `4.5e-6` | kg m^2 rotor + first gear stage [est] -> 2.6e-3 kg m^2 reflected |
| `viscous` | `4.3e-4` | N*m*s/rad motor side (0.25 joint side) [est] |
| `coulomb` | `0.0625` | N*m motor side (1.5 Nm joint side) [est] (Indego knee stiction noted in [doc:Sanchez2023]) |
| `static` | `0.08` | N*m motor side breakaway (1.9 Nm joint) [est] |
| `stribeck_vel` | `2.4` | rad/s motor side (0.1 rad/s joint) [est] |
| `friction_model` | `"coulomb"` | "coulomb" (joint frictionloss, any integrator) / "lugre" (needs Euler integrator) |
| `lugre_stiffness` | `2.0e3` | N*m/rad motor side bristle stiffness [model] |
| `lugre_damping` | `0.5` | N*m*s/rad motor side bristle damping [model] |
| `cogging_amp` | `0.005` | N*m motor side (~1% rated) [est] |
| `cogging_poles` | `24` | [est] |
| `thermal_R` | `2.5` | K/W winding->ambient [est] |
| `thermal_C` | `40.0` | J/K winding heat capacity [est] (tau ~100 s) |
| `temp_coef` | `0.0039` | 1/K copper [lit] |
| `ambient_C` | `25.0` | degC |
| `t_winding_max` | `120.0` | degC driver fold-back threshold [est] |
| `cmd_delay` | `0.002` | s command latency controller->driver [est] |

## `TransmissionParams`

| parameter | default | notes |
|---|---|---|
| `mode` | `"rigid"` | "rigid" / "backlash" (dual-joint decomposition) |
| `backlash_deg` | `0.5` | total free play, joint side [est] (chain/gear drive) |
| `stiffness` | `3000.0` | N*m/rad joint-side transmission stiffness [est] |
| `damping` | `2.0` | N*m*s/rad [est] |

## `BushingParams`

6-DOF visco-elastic bushing (cuff strap + soft tissue), in segment frame (x AP, y ML, z axial).

| parameter | default | notes |
|---|---|---|
| `k_lin` | `(15000.0, 20000.0, 4000.0)` | N/m |
| `c_lin` | `(150.0, 150.0, 80.0)` | N*s/m |
| `k_rot` | `(60.0, 60.0, 15.0)` | N*m/rad (about x, y, z) |
| `c_rot` | `(0.6, 0.6, 0.2)` | N*m*s/rad |
| `wobble_mass` | `0.4` | kg soft-tissue "wobbling mass" coupled through the bushing [lit 0.2-1] |
| `wobble_inertia` | `2.0e-3` | kg m^2 [lit] |
| `range_lin` | `0.04` | m soft limit (strap bottoming out) [est] |
| `range_rot` | `0.35` | rad soft limit [est] |
| `cubic` | `1.0` | strain stiffening: k3 = cubic*k/range^2 -> force doubles at the range [lit soft tissue] |

## `ExoParams`

| parameter | default | notes |
|---|---|---|
| `variant` | `"personal"` |  |
| `mass_hip_unit` | `2.8` | kg incl. 0.68 kg battery [doc:Quintero2011 battery], CPU, belt [est split] |
| `mass_thigh` | `3.3` | kg incl. hip+knee motors/gearing [est split] |
| `actuator_mass` | `0.55` | kg per motor+gear, lumped at the joint axis [est] |
| `mass_shank` | `1.0` | kg [est split] |
| `mass_afo` | `0.2` | kg carbon AFO plate [est] |
| `battery_pos` | `(-0.12, 0.0, 0.12)` | in hip-unit frame [est] (posterior) |
| `hip_range_deg` | `(-30.0, 110.0)` |  |
| `knee_range_deg` | `(-10.0, 105.0)` |  |
| `hip_abd_stiffness` | `600.0` | N*m/rad frame + cuff compliance in ab/adduction [est, ~10 Nm/deg] |
| `hip_abd_damping` | `3.0` | [est] |
| `hip_abd_range_deg` | `6.0` |  |
| `hip_rot_stiffness` | `150.0` | N*m/rad thigh-shell twist compliance [est] |
| `hip_rot_damping` | `1.0` |  |
| `afo_type` | `"quasi_static"` | "quasi_static": bidirectional spring about afo_neutral_deg (hinged/sprung AFO); "free": no AFO torque, the person's own passive tissue/spasticity/voluntary drive |
| `afo_stiffness` | `150.0` | N*m/rad about neutral (~2.6 Nm/deg) [est; lit hinged/carbon AFOs 1-10 Nm/deg] |
| `afo_stiffness_cubic` | `0.0` | N*m/rad^3 optional progressive stiffening toward end of range |
| `afo_damping` | `2.0` | N*m*s/rad [est] |
| `afo_neutral_deg` | `5.0` | AFO neutral in slight dorsiflexion (common for exo users) [est] |
| `afo_range_deg` | `(-25.0, 45.0)` | hinge travel (DF negative, PF positive) = anatomical ROM, soft |
| `hip_motor` | `MotorParams` |  |
| `knee_motor` | `MotorParams` |  |
| `transmission` | `TransmissionParams` |  |
| `knee_brake_torque` | `60.0` | N*m holding torque of normally-locked knee brake [est] [doc:Quintero2011 brakes] |
| `belt` | `_belt` |  |
| `thigh_cuff` | `_thigh_cuff` |  |
| `shank_cuff_prox` | `_shank_cuff` |  |
| `shank_cuff_dist` | `_shank_cuff` |  |
| `shoe` | `_shoe` |  |
| `interface_model` | `"bushing"` | "bushing" (6-DOF springs) / "weld" (isotropic soft weld, faster) |
| `hip_axis_offset` | `lambda: {"l": (0.0, 0.0), "r": (0.0, 0.0)}` |  |
| `knee_axis_offset` | `lambda: {"l": (0.0, 0.0), "r": (0.0, 0.0)}` |  |
| `control_rate_hz` | `200.0` | [doc:Sanchez2023 Indego Explorer signals at 200 Hz] |

## `SensorParams`

| parameter | default | notes |
|---|---|---|
| `enc_bits` | `14` | absolute joint encoder resolution [est] |
| `enc_noise_deg` | `0.03` | [est] |
| `vel_filter_hz` | `40.0` | onboard velocity estimate low-pass [est] |
| `imu_acc_noise` | `0.04` | m/s^2 std at 200 Hz [est, MEMS] |
| `imu_gyro_noise` | `0.003` | rad/s std [est] |
| `imu_acc_bias` | `0.05` | m/s^2 constant bias std (per run) [est] |
| `imu_gyro_bias` | `0.002` | rad/s [est] |
| `current_noise` | `0.15` | A std on measured motor current [est] |
| `sensor_delay` | `0.003` | s acquisition->controller latency [est] |
| `dropout_prob` | `0.0` | probability a control cycle reuses stale data [est] |
| `trunk_imu` | `"sternum"` | "sternum" (mid-sternum, chest strap) / "back" (between the shoulder blades, ~T4) / "none"  [lab add-on] |
| `trunk_imu_rate_hz` | `200.0` | sample rate seen by the controller [est; many wireless IMUs: 100 Hz] |
| `trunk_imu_delay` | `0.003` | s latency [est; wired/CAN ~3 ms, BLE/Wi-Fi IMUs 10-30 ms] |
| `trunk_imu_mount_pitch_deg` | `0.0` | strap misalignment about the left axis (+ = sensor tilted forward) [est; the controller should calibrate in quiet standing] |
| `foot_switches` | `False` | obs["foot_contact"] = [left, right] in {0, 1} [lab add-on] |
| `foot_switch_threshold` | `0.05` | fraction of body weight on a foot that counts as contact [est] |

## `SpasticityParams`

Hyper-reflexive stretch reflex (tonic stretch reflex threshold model, Levin & Feldman 1994; plus force-rate feedback option after Falisse et al. 2018).

| parameter | default | notes |
|---|---|---|
| `mas` | `dict` | Modified Ashworth per group 0..4 (1+ -> 1.5) |
| `model` | `"tsrt"` | "tsrt" / "force_rate" / "both" |
| `delay_scale` | `1.0` |  |
| `clonus_gain_boost` | `1.0` | extra gain on plantarflexors (clonus-prone) [est] |
| `spasm_rate_per_min` | `0.0` | spontaneous spasm rate [est; 0-3 /min typical in chronic SCI] |
| `spasm_stretch_trigger` | `0.0` | probability per strong hip-extension stretch [est] |
| `spasm_amplitude` | `(0.1, 0.35)` | excitation added to the spasm group [est; calibrate from EMG] |
| `spasm_duration` | `(0.5, 2.5)` |  |

## `SubjectParams`

| parameter | default | notes |
|---|---|---|
| `name` | `"healthy_default"` |  |
| `height` | `1.75` | m (MyoLeg nominal ~1.75 m) |
| `mass` | `75.0` | kg (MyoLeg nominal ~75 kg) |
| `sci` | `False` |  |
| `nli` | `"none"` | neurological level of injury e.g. "T10", "L3" |
| `ais` | `"E"` | A..E |
| `motor_scores` | `dict` | {"l": {"hip_flexors": 3, ...}} |
| `atrophy` | `1.0` | Fmax multiplier for paralysed muscles [lit 0.5-0.8 chronic complete SCI] |
| `passive_stiffness_scale` | `dict` | per group passive force x |
| `trunk_strength` | `1.0` | 0..1 trunk control (T4-T10 partial) |
| `shoes` | `True` |  |
| `shoe_width` | `0.095` | m sole width at 1.75 m (scaled with height) [est, adult shoe] |
| `shoe_toe_spring_deg` | `12.0` | upward bend of the forefoot sole (walking shoes ~10-20 deg) [est] |
| `spasticity` | `SpasticityParams` |  |
| `long_loop_delay` | `0.06` | s voluntary/long-latency feedback delay |
| `trunk_delay` | `0.08` |  |

## `HumanWalkParams`

The person's own walking (VolitionParams.gait_mode = "walk"): SIMBICON-style balance (Yin et al. 2007) executed through the muscles (static optimisation, capacity-limited) on the state sensed one long-loop delay ago, extrapolated by an internal forward model (`predict`). States: gait initiation (lean forward on the ankles, first step once the COM is ahead), swing (thigh to the foot-placement angle on a min-jerk path; knee relaxed while it folds, extended for touch-down; swing-leg weight compensated; stance knee/ankle held, push-off late in the swing, pelvis held by the stance hip) ending at the felt touch-down ahead of the stance foot, then double support (weight onto the front foot, trailing-ankle push-off) until the trailing leg unloads. Gains in N*m/rad per kg body mass. Defaults: tuned (tools/diag/tune_own_walk.py) for the incomplete-SCI preset "sci_t12_aisD" with a 30 ms long-loop delay in the zero-torque Indego, sagittal model (SimConfig.planar); other presets: HumanWalkParams.for_subject(name). Other people, delays or devices need their own tuning. [model]

| parameter | default | notes |
|---|---|---|
| `swing_time` | `0.482` | s nominal swing; touch-down ends it after min_swing_frac, max_swing_frac at most |
| `min_swing_frac` | `0.707` |  |
| `max_swing_frac` | `1.358` |  |
| `theta0` | `0.381` | rad swing thigh WORLD angle aimed at (+ = knee ahead of hip) |
| `c_d` | `1.001` | rad per m of COM ahead of the stance ankle (foot placement) |
| `c_v` | `0.078` | rad per m/s of COM velocity |
| `swing_knee_flex` | `1.377` | rad swing knee target in the first part of the swing (clearance) ... |
| `kp_knee_early` | `0.21` | ... held loosely: the knee folds by itself as the hip pulls the thigh forward |
| `knee_extend_at` | `0.302` | fraction of the swing after which the swing knee extends (gain kp_knee) ... |
| `knee_ext_thigh` | `0.385` | rad ... once the swing thigh's world angle is past this |
| `swing_knee_ext` | `0.18` | rad swing knee target for the touch-down |
| `swing_reach_at` | `0.705` | fraction of the swing at which the thigh reaches its foot-placement angle |
| `g_comp` | `0.473` | x the swing leg's own weight compensated (feed-forward) |
| `swing_ankle` | `0.454` | rad swing ankle (+ = dorsiflexion, toe clearance) |
| `stance_knee` | `0.009` | rad |
| `stance_ankle` | `-0.013` | rad stance ankle target |
| `push_off` | `0.837` | N*m/kg extra plantarflexion of the trailing (about to swing) stance ankle |
| `pelvis_pitch` | `0.051` | rad pelvis pitch set-point (+ = forward) |
| `v_des` | `0.703` | m/s walking speed the person aims for |
| `k_lean` | `0.37` | rad of extra forward pelvis lean per m/s below v_des |
| `max_lean` | `0.15` | rad limit of that extra lean |
| `min_step` | `0.069` | m the swing foot must be ahead of the stance foot for its touch-down to end the swing |
| `init_lean` | `0.105` | rad gait initiation: forward lean (ankle strategy) before the first step ... |
| `init_dx` | `0.038` | m ... taken once the COM is this much further ahead of the ankles than in quiet standing ... |
| `init_max` | `1.5` | s ... or after this long |
| `ds_unload` | `0.347` | x body weight: the trailing leg is lifted once its load is below this ... |
| `ds_max` | `0.128` | s ... or after this long in double support |
| `ds_push` | `0.952` | N*m/kg trailing-ankle plantarflexion (push-off) in double support |
| `kp_hip` | `3.121` | swing hip |
| `kp_knee` | `2.547` | swing knee |
| `kp_ankle` | `4.496` | swing ankle |
| `kp_knee_st` | `1.159` | stance knee (carries body weight) |
| `kp_ankle_st` | `6.865` | stance ankle |
| `kp_pelvis` | `8.943` | pelvis pitch through the stance hip(s) |
| `kd_ratio` | `0.048` | s: kd = kd_ratio x kp |
| `predict` | `0.024` | s internal forward model: the delayed state is extrapolated this far ahead |

## `VolitionParams`

What the person *wants* to do, independent of what the exo does.

| parameter | default | notes |
|---|---|---|
| `cooperation` | `0.5` | -1 resist, 0 passive, 1 lead/drive [model] |
| `effort` | `1.0` | fraction of capacity used for tracking own plan |
| `walk` | `HumanWalkParams` | gait_mode "walk" |
| `walk_effort` | `1.0` | x effort while walking (0 = the person relaxes the legs and lets the exo walk them; the trunk stays active) [model] |
| `stiffness_scale` | `1.0` | how stiffly they hold their own plan (co-contraction) |
| `cadence_spm` | `90.0` | intended steps/min |
| `cadence_cv` | `0.03` | step-to-step variability (healthy ~2-3 %, SCI higher) [lit] |
| `phase_offset_s` | `0.0` | systematic lead (+) / lag (-) vs exo step [model] |
| `phase_jitter_s` | `0.03` | random timing jitter per step [lit] |
| `slack_rate` | `0.0` | 1/s rate at which effort decays when exo does the work [model] |
| `lean_trigger_deg` | `7.0` | whole-body forward lean the user has learned triggers a step [est] |
| `com_shift_gain` | `0.9` | m of forward COM shift per rad of intended lean (logged intent; the crutch user's lean target is SupportParams.lean_ahead) |
| `trunk_lean_fraction` | `0.2` | part of the lean made by flexing the trunk (invisible to the exo's posture estimate) vs. whole-body lean about the ankles [est] |
| `expected_speed` | `0.35` | m/s progression a user without leg volition expects in the exo [lit 0.2-0.5] |
| `weight_shift` | `0.035` | m lateral COM shift toward the next stance foot (exo users) [est] |
| `weight_shift_time` | `0.25` | s time constant of the shift |
| `anticipation` | `0.5` | 0..1 how much a trained exo user pre-loads the stance side before a step |
| `lean_rise_time` | `0.35` | s |
| `lean_patience` | `1.0` | s an exo user waits, fully leaned, before leaning further [est] |
| `lean_ramp` | `0.03` | m/s extra forward COM lean while the device does not respond [est] |
| `lean_extra_max` | `0.06` | m maximum extra lean (relaxed once the exo starts the step) [est] |
| `stop_prob_per_step` | `0.0` |  |
| `random_lean_rate_per_min` | `0.0` | spurious leans (false triggers) [model] |
| `gait_mode` | `"own"` | "own": the person follows their own gait template (+ balance strategy); "walk": the person walks on their own, SIMBICON-style balance through the muscles (HumanWalkParams; exo zero torque / transparent; sagittal model); "exo": the exo's walking mode drives hips and knees and the person cooperates (lets the legs be moved; ankle lean + trailing-ankle push-off, swing-foot clearance, sideways balance with the hip ab/adductors, trunk); "follow": the exo walks the person (e.g. DynamicWalkController), who bears weight on the loaded knee and keeps the trunk upright [model] |
| `request_lean` | `0.03` | m COM shift past quiet standing (over the next stance foot) a trained user makes to request an exo step: just past the device's threshold ("exo" mode) [est] |
| `push_off` | `0.8` | N*m/kg trailing-ankle plantarflexion while requesting the next exo step ("exo" mode; capped by the person's plantarflexor capacity) [est; healthy push-off peaks at ~1.5 N*m/kg] |
| `exo_walk_speed` | `0.2` | m/s the COM rolls forward over the stance foot during an exo step ("exo" mode; keeps the momentum that carries it onto the front foot) [est; Indego 0.2-0.5] |
| `follow_speed_gain` | `0.3` | s: ankle COP shift ahead of the COM per m/s above exo_walk_speed ("follow") [est] |
| `follow_assist` | `0.0` | "follow": the person adds this fraction of the exo torque they feel on the swinging leg (one reaction delay late), in the same direction [est] |
| `transfer_target` | `-0.05` | m where the weight transfer puts the COM relative to the front ankle before the next exo step (- = behind; "exo" mode) [est] |
| `transfer_gain` | `1.5` | 1/s: double-support weight transfer - forward COM speed the person aims for per m it is still behind the next stance foot (capped at exo_walk_speed), made with trailing-ankle push-off and their own hip/knee muscles ("exo" mode) [est] |

## `TherapistParams`

A physical therapist guarding / assisting through the gait belt (hip unit), as in clinical exo sessions. The assistance actually delivered is logged, so 'how much help did the person need with this controller' becomes an outcome measure (independent / contact guard / min / mod assist).

| parameter | default | notes |
|---|---|---|
| `level` | `"none"` | "none" / "contact_guard" / "min" / "mod" |
| `force_cap` | `lambda: {"contact_guard": 60.0, "min": 120.0, "mod": 250.0}` | N [est] |
| `moment_cap` | `lambda: {"contact_guard": 10.0, "min": 25.0, "mod": 50.0}` | N*m [est] |
| `deadband` | `lambda: {"contact_guard": 0.06, "min": 0.02, "mod": 0.0}` | m COM error |
| `kp` | `1500.0` | N/m toward the person's own intended COM position [est] |
| `kd` | `300.0` | N*s/m |
| `k_tilt` | `150.0` | N*m/rad pelvis/trunk pitch & roll |
| `d_tilt` | `20.0` |  |
| `reaction_delay` | `0.15` | s |

## `PlanarizerParams`

Test rig, not part of the person: a planarizing boom on the pelvis that confines the body to the sagittal plane, as used to develop gait controllers for devices that only act in that plane (the Indego has no frontal-plane actuation). Stiff, undelayed spring-damper against sideways translation, roll and heading change of the pelvis; forward and vertical translation and pitch stay free. The force it delivers is logged: it is the frontal-plane balance the person (or a therapist) would otherwise have to provide.

| parameter | default | notes |
|---|---|---|
| `enabled` | `False` |  |
| `k_lat` | `1.0e5` | N/m sideways stiffness at the pelvis COM [model] |
| `d_lat` | `5.0e3` | N*s/m [model] |
| `k_rot` | `5.0e3` | N*m/rad roll and heading stiffness [model] |
| `d_rot` | `400.0` | N*m*s/rad [model] |

## `SupportParams`

Upper-limb support (crutches / walker). Each crutch transmits a ground reaction R_i whose line of action passes through its tip; R_i respects push-only, the tip friction cone, a per-arm load cap and a shoulder-moment cap (the part of R_i not along the crutch axis must be made by the shoulder/elbow/wrist). The person chooses R_i with a delayed postural controller (whole-body COM about the feet + trunk orientation) plus intrinsic, undelayed arm impedance.

| parameter | default | notes |
|---|---|---|
| `kind` | `"crutches"` | "none" / "crutches" / "walker" / "harness" (overhead BWS on a trolley) |
| `harness_height` | `0.5` | m spreader bar above the shoulders (sets the pendulum restoring stiffness) |
| `trolley_lag` | `0.3` | s the robotic trolley lags the person horizontally |
| `max_load_frac` | `0.45` | per arm, fraction of body weight [lit: 20-50 %BW crutch loads in SCI exo gait] |
| `friction` | `0.8` | crutch tip friction |
| `shoulder_moment_max` | `40.0` | N*m per arm for off-axis crutch force [est; 0 = pure axial strut] |
| `k_tilt` | `0.0` | N*m/rad extra delayed trunk-orientation feedback via the arms [model; >~50 oscillates] |
| `d_tilt` | `0.0` |  |
| `k_tilt_int` | `150.0` | N*m/rad intrinsic (undelayed) arm/shoulder stiffness on trunk orientation |
| `d_tilt_int` | `25.0` |  |
| `k_yaw` | `60.0` | N*m/rad delayed heading feedback toward the walking direction [model] |
| `d_yaw` | `10.0` |  |
| `k_yaw_int` | `150.0` | N*m/rad intrinsic (undelayed) arm stiffness against trunk twist [model] |
| `d_yaw_int` | `25.0` |  |
| `w_yaw` | `0.7` | allocation weight of the heading moment |
| `k_arm` | `15000.0` | N/m intrinsic axial stiffness of extended arm + crutch (0 = pure force control) [est; co-contracted arm, short-range stiffness + 30-50 ms stretch reflexes] |
| `c_arm` | `300.0` | N*s/m |
| `arm_set_time` | `0.2` | s time constant with which the person re-sets the arm length to get the crutch load they want |
| `arm_set_time_stepping` | `5.0` | s the same while the exo is stepping (large = arms locked, stiff tripod) |
| `w_trunk` | `0.7` | allocation weight of the trunk-orientation moment vs whole-body balance |
| `w_lateral` | `3.0` | allocation priority of the lateral (roll) balance moment over the AP one |
| `w_load` | `0.05` | allocation weight of the preferred total crutch load |
| `k_lat` | `800.0` | N/m lateral COM stabilisation (differential crutch push) |
| `d_lat` | `200.0` |  |
| `k_ap` | `600.0` | N/m antero-posterior COM over the base of support |
| `d_ap` | `300.0` |  |
| `k_height` | `4000.0` | N/m vertical unloading when sagging |
| `d_height` | `400.0` |  |
| `weight_bearing_frac` | `0.15` | baseline arm weight-bearing (fraction BW, both arms) |
| `single_support_frac` | `0.25` | extra arm weight-bearing while one foot is off the ground [lit 30-50 %BW total] |
| `com_ahead` | `0.03` | m desired COM ahead of mid-foot when standing (tripod lean) |
| `lean_ahead` | `0.08` | m beyond the FRONT foot the person leans to request a step [est] |
| `kp_com` | `4.0` | 1/s^2 desired COM acceleration per m of error (crutch balance) |
| `kd_com` | `3.0` | 1/s |
| `kp_lat_com` | `6.0` | lateral (weight shift onto the stance leg) |
| `kd_lat_com` | `5.0` |  |
| `walking_extra_frac` | `0.12` | extra crutch weight-bearing while walking (fraction BW, total) |
| `max_ahead_walking` | `0.15` | m the COM may get ahead of the front foot before the person brakes |
| `tip_margin` | `0.12` | m during an exo step the COM may travel up to this far behind the crutch tips (inside the stance-foot + crutch tripod) [model] |
| `lean_speed` | `0.15` | m/s forward drift allowed while leaning to request the first step |
| `tip_ahead` | `0.25` | m crutch/walker tips placed ahead of the feet |
| `tip_lateral` | `0.33` | m crutch tips lateral of the COM (scaled with height) [est] |
| `pref_weight` | `0.003` | how strongly the person sticks to their preferred crutch load vs balance |
| `prediction` | `1.0` | 0..1 how much the person compensates their own reaction delay (internal model) |
| `reaction_delay` | `0.12` | s upper-limb reaction delay [lit 100-150 ms] |
| `four_point_lift` | `True` | crutches are lifted and re-planted during gait |
| `crutch_pattern` | `"ds"` | "ds": advance both crutches in double support after each step (typical early exo training); "2point": opposite crutch moves with the swing leg; "4point": same-side crutch moves with the swing leg |
| `crutch_swing_time` | `0.35` | s a crutch is off the ground |
| `lift_margin` | `0.03` | m the COM may be ahead of its target when the person lifts a crutch |
| `lift_lat_margin` | `0.03` | m the COM must be inside the opposite foot (laterally) before a crutch is lifted: the remaining crutch can only push the body toward its own opposite side, so a fall toward the lifted crutch cannot be caught [model] |
| `lift_shift` | `0.01` | m extra lateral margin aimed for before lifting a crutch [model] |
| `tip_useless` | `0.05` | m a crutch whose tip is less than this ahead of the shoulder cannot stop a forward fall, so the person moves it even with the COM ahead [model] |
| `crutch_wait_max` | `1.5` | s after which a pending crutch is moved anyway (AP check only) [est] |
| `therapist` | `TherapistParams` |  |
| `planarizer` | `PlanarizerParams` |  |

## `SimConfig`

| parameter | default | notes |
|---|---|---|
| `timestep` | `0.0005` | s physics step [model] |
| `human_rate_hz` | `1000.0` | spinal reflex / crutch update rate [model] |
| `stand_lean_deg` | `6.0` | initial whole-body forward lean about the ankles (COM ahead of ankles) |
| `integrator` | `"implicitfast"` |  |
| `floor_friction` | `1.0` |  |
| `seed` | `0` |  |
| `planar` | `False` | sagittal-plane model: frontal/transverse joints locked (lumbar side-bend and rotation, hip ab/adduction and rotation, subtalar, exo hip ab/adduction and twist compliance) and the pelvis held by the planarizer (SupportParams) [model] |

