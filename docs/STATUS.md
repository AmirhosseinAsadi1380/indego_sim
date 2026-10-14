# Status: what is verified, what is not

## Verified (tests/ and examples/)

| item | result |
|---|---|
| Mesh → exo model | 4 segments + AFO plates from a commercial 3D model of the Indego, un-posed and mirrored; total exo mass 12.3 kg (Personal). Without the (not distributed) meshes each segment is a mass-equivalent box: same mass, centre of mass within 0.1 mm, inertia within 0.3 %; tests and own walking give the same results |
| Drive model | 10 N·m command → 8.2 A after the 2 ms delay; peak torque clipped at 60 N·m; joint inertia matches link + N²·rotor |
| Muscle model | vectorised gain/bias identical to MuJoCo; moment arms match finite differences |
| Sensors | quantisation, sample-and-hold, delay and noise as configured |
| Add-on trunk IMU | placed on the mid-sternum (1.30 m, on the chest surface) or between the shoulder blades (T3–T4); 44-D observation with it, 34-D without (`trunk_imu="none"`) |
| COM estimate (FSM) | static trunk flexion 10/20/30°: true COM shift 1.4/2.8/4.1 cm; error with trunk IMU 0.0 cm, stock sensors −1.4/−2.8/−4.0 cm (misses it entirely). In closed-loop walking until the fall: bias +4…+6 mm, SD 4–7 mm with the trunk IMU vs bias up to 26 mm, SD up to 71 mm with stock sensors |
| Step initiation | walk intent at 2.0 s → first step at ~6.3-6.5 s with the trunk IMU, ~6.75 s with stock sensors (4 seeds) |
| Actuator identification (bench) | Coulomb −0.1 %, viscous −3 %, joint inertia +3 %, m·g·l +2 %, command→current delay within half a sample; rotor inertia needs a detached-segment test at ≥1 kHz |
| Spasticity | pendulum-test R2: relaxed healthy 1.44, T10 A (MAS 1.5 quads) 1.17, T5 A 1.03–1.05; fitting recovers the velocity term within 2 % and the gain within 12 % |
| Quiet standing with crutches (10 s) | T10 A, T5 A, L2 C, T12 D all stand; T10 A: COM sway 10 mm, crutch load 22 % BW, cuff forces 24–51 N |
| Quiet standing, healthy, no support | stable in zero-torque and transparent modes; thigh-cuff force 11.9 → 10.3 N rms with friction + gravity compensation |
| First exo step (T10 A, crutches) | clean ~0.30 m step, no toe contact during swing, heading held within ±1.5° |
| Unsupported standing, incomplete SCI | T12 AIS D stands without crutches or help, 30 s, 4/4 seeds, in 3D and in the sagittal-plane model (`SimConfig.planar`); L2 AIS C (ankles 1/5) falls at 0.3 s |
| Planarizer (test rig) | pelvis held within 2 mm sideways and < 1 deg roll under a 150 N push on the trunk; forward motion and pitch free; off = bit-identical results |
| Floor | unbounded plane at the height of MyoSuite's old pedestal (whose 1.05 m-radius collision cylinder made anyone walking more than ~1 m step off and fall) |
| The person walks on their own in the exo | `gait_mode="walk"`, 30 ms long-loop delay, sagittal model, no crutches or help. T12 AIS D (incomplete SCI, `HumanWalkParams` defaults): zero-torque exo 8/8 seeds walk 30 s, 17.6-18.0 m (0.64 m/s), 56 steps, none backward; 60 s 2/2 (37.3 m); transparent controller 6/8 (falls at 16.5 and 26 s). Healthy (`HumanWalkParams.for_subject("healthy")`): zero-torque 8/8 (20.3-20.5 m, 0.72 m/s), transparent 8/8 (19.4-20.0 m), 60 s 2/2. `examples/01_own_walking.py` |
| Gymnasium env | 44-D observation (34 stock + 10 trunk IMU), randomised episodes incl. trunk-IMU strap misalignment, ground-truth info |

## Not working yet / limitations

The walking results below were measured before the floor fix (CHANGELOG, Unreleased): runs that progressed more
than ~1 m may have ended at the edge of MyoSuite's pedestal rather than by a loss of balance - re-run them before
relying on their fall times.

* **The person's own walking (`gait_mode="walk"`) is tuned per person**: one parameter set each for T12 AIS D (the
  defaults) and healthy strength, both with a 30 ms long-loop reaction delay - a stronger assumption than the 60 ms
  default (long-latency reflexes are ~50-80 ms; 30 ms is the short-latency range), partly offset by the person's
  internal forward model (`predict`). With the default 60 ms delay T12 AIS D falls at 8.8-10.2 s after 10-12 steps
  (not re-tuned); each set makes the other preset fall (8-14 s); L2 AIS C (ankles 1/5) cannot even stand
  unsupported. In the transparent exo T12 AIS D falls in 2/8 seeds (16.5 and 26 s). Other people, delays or devices
  need their own tuning (`tools/diag/tune_own_walk.py`, ~1 h on 16 cores). The gaits are faster than typical Indego
  users (0.64-0.72 vs 0.2-0.5 m/s), sagittal only, and untested against pushes, slopes or uneven ground.
* **Closed-loop walking of the complete-SCI user with crutches**: the first step is clean, but the user
  falls in the double support after it (1 step over 4 seeds, falls at 10–15 s).
  Mechanism: after the step the COM sits near mid-stance,
  ~10–15 cm behind the front ankle; the person advances both crutches (~2.5 s) and then cannot bring the
  COM forward — crutches planted ahead can only push back and a complete-SCI user has no push-off —
  and while waiting with the weight shifted sideways toward the next stance foot a slow lateral drift
  builds up until they fall. On the real device this is where the person's own strategy (trunk lean,
  pushing/pulling on the crutches, timing) decides the outcome, and it is the least validated part of
  the model. Calibrate it on session data (CALIBRATION.md §5, `tools/tune_support.py`) before drawing
  conclusions about controllers from walking trials. The trunk IMU helps the controller (earlier,
  better-informed triggers) but cannot fix the simulated person's strategy.
* **Control vs modelling (October 2026 runs, `eval_walk.py`, seeds 0-3, 30 s)**:

  | condition | completed steps | falls | foot progression | help needed |
  |---|---|---|---|---|
  | crutches, no help | 1, 1, 1, 1 | 4/4 at 11.0-13.2 s | 0.10-0.25 m | - |
  | + therapist "min" (<= 120 N) | 5, 4, 2, 2 | 4/4 at 19.1-28.5 s | 0.50-1.26 m | 56-72 N on average, at the cap 26-51 % of the time |
  | + therapist "mod" (<= 250 N) | 2, 2, 2, 2 | 4/4 at 8.0-10.9 s | 0.57-0.77 m | 158-167 N on average |
  | planarizer (sagittal plane only) | 2, 2, 2, 2 | 4/4 at 9.7-12.3 s | 0.23-0.25 m | planarizer 14-17 N sideways on average |
  | planarizer + therapist "min" | 5, 4, 5, 5 | 4/4 at 18.8-24.5 s | 1.45-1.59 m | therapist 42-64 N, planarizer 40-62 N on average |

  The criterion "with a helper the same FSM walks, so the controller is adequate" is not met, so these runs do not
  clear the controller; what they show:
  * Sideways: the FSM's lateral estimate tracks the true COM shift (same sign, mostly within a few mm), and its
    lateral check correctly refuses steps while the COM is over the swing foot. Without help the forward and the
    sideways step conditions are met at different times (forward right after a step, sideways after the crutch
    moves, by when the COM has dropped back). Disabling the check (`lat_dy = 0`) makes the person step with the
    weight on the swing leg and fall sooner. "mod" help destabilises the sideways balance (a delayed 250 N spring
    toward a target that jumps between the feet), so "mod" is not a clean probe of the controller.
  * Forward (planarizer runs, all sideways falls removed): the falls are sagittal. After a 0.30 m step the COM is
    ~11 cm behind the front ankle and nothing can carry it forward: no ankle push-off, crutches planted ahead can
    only push back, and stance-hip extension tips the trunk back instead of vaulting the body over the stance foot.
    With shorter steps (28-32 deg hip flexion) the COM lands ahead of the front foot and runs away; the person then
    hangs on the crutches (both at their load cap, feet sliding) - a state the fall check does not flag, now
    reported by `eval_walk.py` as `crutch_hang_frac`.
  * Tried and not kept (no improvement over 4 seeds): an exo-driven double-support weight transfer (leg IK moving
    the hips toward the front ankle; with a 0.30 m step and the rear heel down the rear leg is ~5 cm too short for
    it), adaptive foot placement from the COM estimate, 2-point/4-point crutch patterns with a short double
    support, a minimum double support matched to the crutch moves (2.6 s), body-weight-supported harness walking
    with the planarizer (never falls, but ~0 net progression).
* **Incomplete SCI walking with the exo, no support (sagittal-plane model, `gait_mode="exo"`)**: T12 AIS D takes
  2-4 steps and falls at 9.5-18 s (best setting so far: `exo_walk_speed` 0.25, FSM `ds_min_time` 1.2 s,
  `hip_flex` 32 deg with `first_step_scale` 1.125; 0.4-1.45 m). Healthy is similar. What is understood: a planted foot
  can carry the COP only from ~4 cm behind to ~13 cm ahead of the ankle, so the COM has to cross the stance foot
  during each exo step; with the fixed exo step (1.2 s, fixed hip goals) the step length and the COM's progress per
  cycle drift apart, the front foot lands ahead of the COM without taking load and the person falls forward over the
  rear foot. Fixed timing and step length are the Indego's; matching them to the user (or adapting them to the COM
  state) is the open problem. In 3D the same users fall sideways after the first step (hip abductors 3/5); out of
  scope for now.
* **Exo-led dynamic walking, incomplete SCI, no support (sagittal model, `DynamicWalkController` + `gait_mode
  "follow"`, foot switches)**: best configuration found (CMA-ES over controller and cooperation parameters) takes
  8-38 steps and falls at 6.8-18 s over 4 seeds (`eval_walk.py`), with little net progress (-0.5..+1.0 m). Limits
  identified: (1) the knee drives overheat - mean |knee torque| ~30 N*m against 12 N*m continuous, windings at
  ~110 C after ~15 s and thermal fold-back to 40-46 N*m before the fall; a compliant exo stance knee keeps them cool
  (<60 C) but the gait then falls in 4-9 s; (2) the swing leg is under-damped - the exo's derivative gain is capped at
  ~8 N*m*s/rad by loop latency and cuff compliance (more damping is worse even at 1 kHz control), so swing feet
  overshoot or scuff; (3) no ankle actuation on the device. A quasi-static exo-driven variant (hips moved over the front
  foot in double support) was tried and removed (front foot slips while unloaded, transfer oscillates). The person's
  own walking controller (`gait_mode "own"`) falls in 1.5-2 s even in the sagittal model, with zero-torque,
  transparent or assist-as-needed exo.
* Overhead harness + timer-triggered stepping (40 % BWS) does not fall (8 steps in 20 s), but the body
  hangs back in the harness with both hips flexed 20–35° and makes no net progress; the hip motors heat
  to ~115 °C holding that posture. Not a usable walking demo.
* **Gait-template walking** (`gait_mode="own"`, the default: gait-template tracking + static optimisation +
  SIMBICON-like balance) falls after ~2.3 s in every tested combination (no support or crutches ×
  zero-torque, transparent or assist-as-needed exo; the T12 AIS D episode in `examples/03_your_controller_env.py` falls at 2.5 s).
  Standing works. Use `gait_mode="walk"` for a person who walks on their own.
* A bilateral extensor spasm of default amplitude makes the standing SCI user fall backward with the
  quasi-static AFO when spasms are forced. Plausible, but check
  against your participants and AFO stiffness.
* The MyoLeg knee range is 0–120°, which caps pendulum-test excursions.
* Many parameters are estimates ([est] in PARAMETERS.md), notably the cuff stiffnesses, the motor
  thermal capacity, the Indego mass split, the shoe geometry and all human-strategy gains.
* No sit-to-stand, stairs or turning; sagittal-plane FSM only (plus the lateral weight check); ground is
  flat, rigid and unbounded.
