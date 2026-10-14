# indego_sim — human + Indego exoskeleton simulation for controller design

A MuJoCo / MyoSuite simulation of a person (healthy or with spinal cord injury) wearing an
Indego-like lower-limb exoskeleton, built so that **your controller sees what it would see on the
real device**: the Indego's onboard sensors at 200 Hz with latency, quantisation and noise, driving
DC-motor models with current limits, friction, delay and heating, through compliant cuffs on a
musculoskeletal human. The human has their own intent (volition), delayed voluntary control,
spinal reflexes, spasticity, clonus and spasms, and balances with crutches, a walker, a harness or a
therapist.

```
 volition (wants to stand / lean / step, cadence, effort)        your controller (obs -> 4 torques, 200 Hz)
        │                                                                ▲                 │
        ▼                                                                │ encoders, IMUs, │
 voluntary layer (SO, 60 ms delay) ─┐                                    │ motor current   ▼
 spinal layer (reflex, spasticity,  ├─► 80 Hill muscles (MyoLeg) ◄── cuffs (6-DOF bushings) ◄── Indego frame
   clonus, spasms; 1 kHz)          ─┘   + trunk                          hip belt, thigh, shank, shoe/AFO
 arms: crutches / walker / harness / therapist (delayed + intrinsic)     dcmotor drives (current loop, back-EMF,
                                                                          friction, cogging, thermal, delay)
```

## Quick start

```bash
pip install -r requirements.txt                   # mujoco (dcmotor), numpy, scipy, gymnasium, matplotlib, imageio, cma
git clone https://github.com/MyoHub/myo_sim && pip install --no-deps -e myo_sim   # MyoLeg model (MyoSuite)
python examples/07_own_walking.py --video         # incomplete-SCI person (T12 AIS D) walking on their own in the exo
python examples/01_quiet_standing.py --video      # SCI T10 AIS A standing with crutches
python examples/02_sci_walking_fsm.py --video     # Indego-like FSM, lean-to-step, crutches (+ --no-trunk-imu)
python examples/03_actuator_identification.py     # bench protocol for the drives (answers "how do I identify...")
python examples/04_pendulum_test.py               # calibrate spasticity from a Wartenberg pendulum test
python examples/05_your_controller_env.py         # Gymnasium env + domain randomisation, plug in your controller
python examples/06_transparency.py                # zero-torque vs friction/gravity-compensated transparent mode
pytest -q tests                                   # 16 tests
```

Outputs go to `results/` (index with the current results: [`results/README.md`](results/README.md)). Setup details,
all commands and the development rules: [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## Minimal use

```python
from indego_sim.sim import HumanExoSim
from indego_sim.human.sci import get_preset
from indego_sim.config import SupportParams, ExoParams

sim = HumanExoSim(subject=get_preset("sci_t10_aisA"), support=SupportParams(kind="crutches"),
                  exo=ExoParams(afo_type="quasi_static"), schedule=[(2.0, "walk")])
obs = sim.reset(settle=1.5)                 # dict: q, qd, tau_est, current, acc_/gyr_thigh_l|r|hip, hip_quat
                                            #  + acc_trunk, gyr_trunk, trunk_quat (add-on trunk IMU)
for _ in range(2000):                       # 10 s at 200 Hz
    tau = my_controller(obs)                # 4 joint torques [N m]: hip_l, knee_l, hip_r, knee_r
    obs = sim.step(tau)
log = sim.get_log()                         # ground truth: cuff wrenches, GRFs, crutch loads, muscles, intent...
```

## What is in the model

| part | what is modelled | where |
|---|---|---|
| Exo geometry | hip unit / thigh / shank / AFO plate measured on a 3D model of the Indego (un-posed, axes found from the shells), scaled to the subject's femur/tibia. The segment meshes are not distributed; without them each segment is a box with the mesh's mass properties | `model/builder.py`, `tools/` |
| Exo inertia | masses from the Indego spec (12.3 kg Personal; `ExoParams.therapy()` 17.7 kg), distributed on the segment shells, battery and motors as point masses | `config.ExoParams` |
| Joints | powered hip & knee (Indego ranges), passive frontal-plane hip compliance and thigh twist, AFO either **free** or **quasi-static spring** (never locked), optional gear backlash, knee brakes | `model/builder.py` |
| Human–exo interface | 6-DOF visco-elastic bushings with strain stiffening + soft-tissue wobbling mass at hip belt, thigh cuff, 2 shank cuffs, shoe; axis misalignment | `BushingParams` |
| Drives | MuJoCo `dcmotor`: torque-mode driver with back-EMF compensation, voltage limit, current loop lag, rotor inertia, viscous + Coulomb (or LuGre) friction, cogging, copper heating; firmware current limits and thermal fold-back | `exo/drive.py` |
| Sensors | 14-bit encoders, filtered-difference velocity, motor current, three IMUs, hip AHRS; MuJoCo sensor `interval`/`delay` + noise, bias, dropouts | `exo/sensors.py` |
| Add-on trunk IMU | strapped on the sternum or between the shoulder blades (placed from the rib-cage mesh), own rate / latency / strap misalignment; `SensorParams.trunk_imu="none"` for the stock device | `model/builder.py` |
| Human | MyoLeg 80 muscles / 29 DOF + 3 lumbar DOF, scaled to height/mass; shoe soles (9.5 cm wide, toe spring) instead of MyoLeg's narrow bare-foot contacts | `model/builder.py` |
| Volition | stand / lean / walk / stop / cadence session scripts, gait oscillator entrained by heel strikes, cooperation/resistance, slacking, spurious leans | `human/volition.py` |
| Voluntary control | 60 ms delayed PD + gait-template feedforward → static optimisation (muscle capacity limits = SCI weakness); ankle/hip balance strategy. Own walking (`gait_mode="walk"`): SIMBICON-style foot placement and trunk control through the muscles, gait initiation, double-support weight transfer, internal forward model; parameter sets tuned per person | `human/controller.py`, `config.HumanWalkParams` |
| Spinal layer | stretch reflex, spasticity (TSRT threshold model from Modified Ashworth), clonus (delay × gain), Poisson & stretch-triggered extensor/flexor spasms; location-dependent delays | `human/reflexes.py` |
| SCI | ISNCSCI motor scores / NLI / AIS → per-muscle capacity, disuse atrophy, passive stiffness; presets T10 A, T5 A, L2 C, T12 D | `human/sci.py` |
| Upper limbs | crutches/walker: ground reaction through each tip within the friction cone, per-arm load cap, shoulder-moment cap, stiff "locked-arm" struts while the exo steps, delayed postural controller (COM about the feet' centre of pressure + trunk orientation + heading), crutch moves in double support with lateral-safety and reach rules, leaning further when the exo does not respond; overhead BWS harness on a trolley; therapist guarding (logged assistance) | `human/support.py` |
| Controllers | Indego-like FSM (lean-to-step from the exo's own COM estimate — segment model, uses the trunk IMU when fitted — step-length-dependent double-support threshold and lateral weight check as in the Vanderbilt controller, min-jerk steps, lean/auto/external triggers), foot-placement dynamic walking (experimental), transparent mode, assist-as-needed | `controllers/` |
| RL / design env | Gymnasium env, 44-D obs at 200 Hz (34 stock + 10 trunk IMU), ground truth in `info`, per-episode randomisation of drives, cuffs, sensors (incl. trunk-IMU strap misalignment), volition, support, spasticity | `env.py` |
| System ID | excitation signals, virtual single-joint bench, friction map, rigid-body + delay fit, FRF, current loop, thermal; pendulum-test spasticity fit | `sysid/` |

## Status — read this before trusting a result

See [`docs/STATUS.md`](docs/STATUS.md). In short: model construction, drives, sensors (incl. the
trunk IMU), muscles, reflexes, quiet standing (all SCI presets with crutches; healthy without support)
and the identification tools are verified. **A person who walks on their own** (incomplete SCI T12 AIS D, or
healthy) walks in the zero-torque Indego for 30 s on 8/8 seeds in the sagittal-plane model, with a 30 ms reaction
delay and parameters tuned for that person (`examples/07_own_walking.py`). **Exo-led walking of a complete-SCI user
with crutches is not robust yet**: the user takes a clean first step (~0.3 m) with the baseline FSM but falls in
the double support after it. The arm/crutch strategy of the person is the least validated part of the model and
the first thing to calibrate on your own session data (`docs/CALIBRATION.md` §5).

What changed in this version: [`docs/CHANGELOG.md`](docs/CHANGELOG.md). Every parameter has a provenance tag ([doc] / [lit] / [est] / [model]) in
[`docs/PARAMETERS.md`](docs/PARAMETERS.md); modelling decisions and pitfalls are in
[`docs/MODELING_NOTES.md`](docs/MODELING_NOTES.md).

## Layout

```
indego_sim/
  config.py            all parameters (dataclasses) + YAML loading + randomisation
  sim.py               HumanExoSim: one closed-loop session, logging, events
  env.py               Gymnasium environment
  model/builder.py     MjSpec assembly of MyoLeg + Indego, cuffs, drives, sensors, keyframe
  exo/                 drive firmware layer, onboard sensors
  human/               muscles + static optimisation, reflexes, SCI, volition, controller, support
  controllers/         Indego-like FSM, foot-placement dynamic walking, transparent, assist-as-needed
  sysid/               excitation, bench, actuator_id, spasticity (pendulum test)
  assets/indego/       exo segment meshes (optional, not distributed; see docs/DEVELOPMENT.md)
tools/                 mesh processing, docs generator, overview renderer, crutch-strategy fitting
tools/diag/            walking diagnostics, own-walking evaluation and tuning
examples/ tests/ docs/ results/
```

The exo segment meshes are made from a commercial 3D model whose licence does not allow redistributing them;
they are git-ignored, and the model builds without them.
