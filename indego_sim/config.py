"""All simulation parameters in one place, with provenance.

Every number carries a source tag in the comment:
  [doc:<ref>]  taken from a publication / datasheet (see docs/PARAMETERS.md)
  [est]        engineering estimate - REPLACE with your measurement
  [lit]        typical value from biomechanics literature (range given)
  [model]      numerical/modeling choice, not a physical property

Parameters are plain dataclasses so they can be overridden from YAML
(``load_yaml``) or programmatically, and randomized (``randomize``).
"""
from __future__ import annotations

import copy
import dataclasses as dc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Exoskeleton
# ----------------------------------------------------------------------------


@dataclass
class MotorParams:
    """One Indego joint drive (BLDC motor + reduction), mapped onto MuJoCo `dcmotor`.

    Motor-side quantities (torque, friction, inertia) are reflected to the joint by
    `gear` (torque x N, inertia x N^2) inside MuJoCo.
    """
    gear: float = 24.0                 # [doc:Quintero2011] 24:1 reduction (prototype); [est] for commercial unit
    K: float = 0.051                   # N*m/A effective torque const incl. ~85% gear efficiency [est]
    R: float = 0.35                    # ohm terminal resistance [est] (200 W-class BLDC)
    te: float = 1.0e-4                 # s electrical time constant L/R [est]
    di_dt_max: float = 0.0             # A/s current slew limit (0 = off) [est]
    v_max: float = 28.0                # V drive voltage limit (29.6 V Li-ion pack) [doc:Quintero2011 battery]
    i_peak: float = 49.0               # A peak current -> 60 Nm joint peak [doc:Farris2012 up to 65 Nm; Quintero2011 prototype ~40 Nm]
    i_cont: float = 9.8                # A continuous current -> 12 Nm joint continuous [doc:Quintero2011]
    rotor_inertia: float = 4.5e-6      # kg m^2 rotor + first gear stage [est] -> 2.6e-3 kg m^2 reflected
    viscous: float = 4.3e-4            # N*m*s/rad motor side (0.25 joint side) [est]
    coulomb: float = 0.0625            # N*m motor side (1.5 Nm joint side) [est] (Indego knee stiction noted in [doc:Sanchez2023])
    static: float = 0.08               # N*m motor side breakaway (1.9 Nm joint) [est]
    stribeck_vel: float = 2.4          # rad/s motor side (0.1 rad/s joint) [est]
    friction_model: str = "coulomb"    # "coulomb" (joint frictionloss, any integrator) | "lugre" (needs Euler integrator)
    lugre_stiffness: float = 2.0e3     # N*m/rad motor side bristle stiffness [model]
    lugre_damping: float = 0.5         # N*m*s/rad motor side bristle damping [model]
    cogging_amp: float = 0.005         # N*m motor side (~1% rated) [est]
    cogging_poles: int = 24            # [est]
    thermal_R: float = 2.5             # K/W winding->ambient [est]
    thermal_C: float = 40.0            # J/K winding heat capacity [est] (tau ~100 s)
    temp_coef: float = 0.0039          # 1/K copper [lit]
    ambient_C: float = 25.0            # degC
    t_winding_max: float = 120.0       # degC driver fold-back threshold [est]
    cmd_delay: float = 0.002           # s command latency controller->driver [est]

    def joint_peak_torque(self) -> float:
        return self.K * self.i_peak * self.gear

    def joint_cont_torque(self) -> float:
        return self.K * self.i_cont * self.gear


@dataclass
class TransmissionParams:
    mode: str = "rigid"                # "rigid" | "backlash" (dual-joint decomposition)
    backlash_deg: float = 0.5          # total free play, joint side [est] (chain/gear drive)
    stiffness: float = 3000.0          # N*m/rad joint-side transmission stiffness [est]
    damping: float = 2.0               # N*m*s/rad [est]


@dataclass
class BushingParams:
    """6-DOF visco-elastic bushing (cuff strap + soft tissue), in segment frame (x AP, y ML, z axial)."""
    k_lin: Tuple[float, float, float] = (15000.0, 20000.0, 4000.0)   # N/m
    c_lin: Tuple[float, float, float] = (150.0, 150.0, 80.0)         # N*s/m
    k_rot: Tuple[float, float, float] = (60.0, 60.0, 15.0)           # N*m/rad (about x, y, z)
    c_rot: Tuple[float, float, float] = (0.6, 0.6, 0.2)              # N*m*s/rad
    wobble_mass: float = 0.4           # kg soft-tissue "wobbling mass" coupled through the bushing [lit 0.2-1]
    wobble_inertia: float = 2.0e-3     # kg m^2 [lit]
    range_lin: float = 0.04            # m soft limit (strap bottoming out) [est]
    range_rot: float = 0.35            # rad soft limit [est]
    cubic: float = 1.0                 # strain stiffening: k3 = cubic*k/range^2 -> force doubles at the range [lit soft tissue]


def _thigh_cuff():
    return BushingParams()  # [lit] thigh soft tissue under strap 5-30 kN/m; shear 2-6 kN/m


def _shank_cuff():
    return BushingParams(k_lin=(12000.0, 15000.0, 3000.0), c_lin=(120.0, 120.0, 60.0),
                         k_rot=(40.0, 40.0, 10.0), c_rot=(0.4, 0.4, 0.15), wobble_mass=0.25,
                         wobble_inertia=1.0e-3)


def _belt():
    return BushingParams(k_lin=(30000.0, 30000.0, 20000.0), c_lin=(300.0, 300.0, 300.0),
                         k_rot=(1500.0, 1500.0, 800.0), c_rot=(12.0, 12.0, 6.0), wobble_mass=1.0,
                         wobble_inertia=0.02, range_lin=0.05, range_rot=0.3)


def _shoe():
    # foot strapped into AFO + shoe: heel can slip a little, but rotation is tightly limited
    return BushingParams(k_lin=(25000.0, 25000.0, 40000.0), c_lin=(200.0, 200.0, 300.0),
                         k_rot=(600.0, 1500.0, 200.0), c_rot=(2.0, 4.0, 1.0), wobble_mass=0.15,
                         wobble_inertia=5.0e-4, range_lin=0.02, range_rot=0.15)


@dataclass
class ExoParams:
    variant: str = "personal"
    # --- mass properties (Personal ~12 kg [doc:Indego spec 12 kg]; Therapy 17.7 kg [doc:Ekso spec])
    mass_hip_unit: float = 2.8         # kg incl. 0.68 kg battery [doc:Quintero2011 battery], CPU, belt [est split]
    mass_thigh: float = 3.3            # kg incl. hip+knee motors/gearing [est split]
    actuator_mass: float = 0.55        # kg per motor+gear, lumped at the joint axis [est]
    mass_shank: float = 1.0            # kg [est split]
    mass_afo: float = 0.2              # kg carbon AFO plate [est]
    battery_pos: Tuple[float, float, float] = (-0.12, 0.0, 0.12)  # in hip-unit frame [est] (posterior)
    # --- joint limits [doc:FDA K173530 hip 110 flex / 30 ext; Quintero2011 knee 105 flex / 10 hyperext]
    hip_range_deg: Tuple[float, float] = (-30.0, 110.0)
    knee_range_deg: Tuple[float, float] = (-10.0, 105.0)
    # --- passive frontal-plane compliance at the hip (Indego has no abd joint;
    #     "ab/adduction is limited by embedded compliance" [doc:Quintero2011])
    hip_abd_stiffness: float = 600.0   # N*m/rad frame + cuff compliance in ab/adduction [est, ~10 Nm/deg]
    hip_abd_damping: float = 3.0       # [est]
    hip_abd_range_deg: float = 6.0
    hip_rot_stiffness: float = 150.0   # N*m/rad thigh-shell twist compliance [est]
    hip_rot_damping: float = 1.0
    # --- AFO / foot plate (ankle hinge at the anatomical ankle). Never locked: in practice the ankle is either
    # left to the person ("free") or held quasi-statically by a sprung hinge ("quasi_static").
    afo_type: str = "quasi_static"     # "quasi_static": bidirectional spring about afo_neutral_deg (hinged/sprung AFO);
                                       # "free": no AFO torque, the person's own passive tissue/spasticity/voluntary drive
    afo_stiffness: float = 150.0       # N*m/rad about neutral (~2.6 Nm/deg) [est; lit hinged/carbon AFOs 1-10 Nm/deg]
    afo_stiffness_cubic: float = 0.0   # N*m/rad^3 optional progressive stiffening toward end of range
    afo_damping: float = 2.0           # N*m*s/rad [est]
    afo_neutral_deg: float = 5.0       # AFO neutral in slight dorsiflexion (common for exo users) [est]
    afo_range_deg: Tuple[float, float] = (-25.0, 45.0)  # hinge travel (DF negative, PF positive) = anatomical ROM, soft
    # --- drives
    hip_motor: MotorParams = field(default_factory=MotorParams)
    knee_motor: MotorParams = field(default_factory=MotorParams)
    transmission: TransmissionParams = field(default_factory=TransmissionParams)
    knee_brake_torque: float = 60.0    # N*m holding torque of normally-locked knee brake [est] [doc:Quintero2011 brakes]
    # --- human interface
    belt: BushingParams = field(default_factory=_belt)
    thigh_cuff: BushingParams = field(default_factory=_thigh_cuff)
    shank_cuff_prox: BushingParams = field(default_factory=_shank_cuff)
    shank_cuff_dist: BushingParams = field(default_factory=_shank_cuff)
    shoe: BushingParams = field(default_factory=_shoe)
    interface_model: str = "bushing"   # "bushing" (6-DOF springs) | "weld" (isotropic soft weld, faster)
    # --- fit / misalignment (exo axis minus anatomical axis, sagittal, metres) [est: typical 0-3 cm]
    hip_axis_offset: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {"l": (0.0, 0.0), "r": (0.0, 0.0)})
    knee_axis_offset: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {"l": (0.0, 0.0), "r": (0.0, 0.0)})
    # --- control interface
    control_rate_hz: float = 200.0     # [doc:Sanchez2023 Indego Explorer signals at 200 Hz]

    @staticmethod
    def therapy() -> "ExoParams":
        p = ExoParams(variant="therapy", mass_hip_unit=5.6, mass_thigh=4.3, mass_shank=1.4, mass_afo=0.25)
        return p  # 17.7 kg total [doc:Ekso Indego Therapy spec sheet]; split [est]

    def total_mass(self) -> float:
        return self.mass_hip_unit + 2 * (self.mass_thigh + self.mass_shank + self.mass_afo)


@dataclass
class SensorParams:
    enc_bits: int = 14                 # absolute joint encoder resolution [est]
    enc_noise_deg: float = 0.03        # [est]
    vel_filter_hz: float = 40.0        # onboard velocity estimate low-pass [est]
    imu_acc_noise: float = 0.04        # m/s^2 std at 200 Hz [est, MEMS]
    imu_gyro_noise: float = 0.003      # rad/s std [est]
    imu_acc_bias: float = 0.05         # m/s^2 constant bias std (per run) [est]
    imu_gyro_bias: float = 0.002       # rad/s [est]
    current_noise: float = 0.15        # A std on measured motor current [est]
    sensor_delay: float = 0.003        # s acquisition->controller latency [est]
    dropout_prob: float = 0.0          # probability a control cycle reuses stale data [est]
    # optional add-on IMU strapped to the person's upper trunk (not part of the stock Indego)
    trunk_imu: str = "sternum"         # "sternum" (mid-sternum, chest strap) | "back" (between the shoulder
                                       # blades, ~T4) | "none"  [lab add-on]
    trunk_imu_rate_hz: float = 200.0   # sample rate seen by the controller [est; many wireless IMUs: 100 Hz]
    trunk_imu_delay: float = 0.003     # s latency [est; wired/CAN ~3 ms, BLE/Wi-Fi IMUs 10-30 ms]
    trunk_imu_mount_pitch_deg: float = 0.0   # strap misalignment about the left axis (+ = sensor tilted forward)
                                       # [est; the controller should calibrate in quiet standing]
    # optional foot switches / insole sensors (not on the stock Indego; common research add-on)
    foot_switches: bool = False        # obs["foot_contact"] = [left, right] in {0, 1} [lab add-on]
    foot_switch_threshold: float = 0.05  # fraction of body weight on a foot that counts as contact [est]


# ----------------------------------------------------------------------------
# Human
# ----------------------------------------------------------------------------

# Spinal segment innervation used to derive SCI muscle capacity
# (Kendall; ISNCSCI key muscles: L2 hip flex, L3 knee ext, L4 ankle DF, L5 EHL, S1 ankle PF)
MUSCLE_GROUPS: Dict[str, List[str]] = {
    "hip_flexors": ["iliacus", "psoas", "sart", "tfl", "recfem"],
    "hip_extensors": ["glmax1", "glmax2", "glmax3", "bflh", "semimem", "semiten", "addmagIsch", "addmagDist"],
    "hip_abductors": ["glmed1", "glmed2", "glmed3", "glmin1", "glmin2", "glmin3", "piri"],
    "hip_adductors": ["addbrev", "addlong", "addmagMid", "addmagProx", "grac"],
    "knee_extensors": ["vasint", "vaslat", "vasmed"],
    "knee_flexors": ["bfsh"],
    "ankle_plantarflexors": ["soleus", "gaslat", "gasmed", "tibpost", "perbrev", "perlong", "fdl", "fhl"],
    "ankle_dorsiflexors": ["tibant", "edl", "ehl"],
}
GROUP_SEGMENT = {  # dominant root level per group (for SCI level -> capacity)
    "hip_flexors": 2, "hip_adductors": 3, "knee_extensors": 3, "ankle_dorsiflexors": 4,
    "hip_abductors": 5, "hip_extensors": 5, "knee_flexors": 5, "ankle_plantarflexors": 6,
}  # lumbar L1..L5 -> 1..5, S1 -> 6
MOTOR_SCORE_TO_CAPACITY = {0: 0.0, 1: 0.03, 2: 0.12, 3: 0.3, 4: 0.6, 5: 1.0}  # [est] max voluntary drive


@dataclass
class SpasticityParams:
    """Hyper-reflexive stretch reflex (tonic stretch reflex threshold model, Levin & Feldman 1994;
    plus force-rate feedback option after Falisse et al. 2018)."""
    mas: Dict[str, float] = field(default_factory=dict)  # Modified Ashworth per group 0..4 (1+ -> 1.5)
    model: str = "tsrt"                # "tsrt" | "force_rate" | "both"
    delay_scale: float = 1.0
    clonus_gain_boost: float = 1.0     # extra gain on plantarflexors (clonus-prone) [est]
    spasm_rate_per_min: float = 0.0    # spontaneous spasm rate [est; 0-3 /min typical in chronic SCI]
    spasm_stretch_trigger: float = 0.0 # probability per strong hip-extension stretch [est]
    spasm_amplitude: Tuple[float, float] = (0.1, 0.35)    # excitation added to the spasm group [est; calibrate from EMG]
    spasm_duration: Tuple[float, float] = (0.5, 2.5)


@dataclass
class SubjectParams:
    name: str = "healthy_default"
    height: float = 1.75               # m (MyoLeg nominal ~1.75 m)
    mass: float = 75.0                 # kg (MyoLeg nominal ~75 kg)
    # neurological status
    sci: bool = False
    nli: str = "none"                  # neurological level of injury e.g. "T10", "L3"
    ais: str = "E"                     # A..E
    motor_scores: Dict[str, Dict[str, int]] = field(default_factory=dict)  # {"l": {"hip_flexors": 3, ...}}
    atrophy: float = 1.0               # Fmax multiplier for paralysed muscles [lit 0.5-0.8 chronic complete SCI]
    passive_stiffness_scale: Dict[str, float] = field(default_factory=dict)  # per group passive force x
    trunk_strength: float = 1.0        # 0..1 trunk control (T4-T10 partial)
    # footwear: exo users stand in shoes (+ AFO footplate). MyoLeg's bare-foot contact is two lengthwise capsules
    # ~4 cm apart, which rolls sideways almost freely; a shoe sole gives the real ~9-10 cm support width.
    shoes: bool = True
    shoe_width: float = 0.095          # m sole width at 1.75 m (scaled with height) [est, adult shoe]
    shoe_toe_spring_deg: float = 12.0  # upward bend of the forefoot sole (walking shoes ~10-20 deg) [est]
    spasticity: SpasticityParams = field(default_factory=SpasticityParams)
    # neural timing [lit]
    long_loop_delay: float = 0.06      # s voluntary/long-latency feedback delay
    trunk_delay: float = 0.08


@dataclass
class HumanWalkParams:
    """The person's own walking (VolitionParams.gait_mode = "walk"): SIMBICON-style balance (Yin et al. 2007) executed
    through the muscles (static optimisation, capacity-limited) on the state sensed one long-loop delay ago,
    extrapolated by an internal forward model (`predict`). States: gait initiation (lean forward on the ankles, first
    step once the COM is ahead), swing (thigh to the foot-placement angle on a min-jerk path; knee relaxed while it
    folds, extended for touch-down; swing-leg weight compensated; stance knee/ankle held, push-off late in the swing,
    pelvis held by the stance hip) ending at the felt touch-down ahead of the stance foot, then double support (weight
    onto the front foot, trailing-ankle push-off) until the trailing leg unloads. Gains in N*m/rad per kg body mass.
    Defaults: tuned (tools/diag/tune_own_walk.py) for the incomplete-SCI preset "sci_t12_aisD" with a 30 ms
    long-loop delay in the zero-torque Indego, sagittal model (SimConfig.planar); other presets:
    HumanWalkParams.for_subject(name). Other people, delays or devices need their own tuning. [model]"""
    swing_time: float = 0.482          # s nominal swing; touch-down ends it after min_swing_frac, max_swing_frac at most
    min_swing_frac: float = 0.707
    max_swing_frac: float = 1.358
    theta0: float = 0.381              # rad swing thigh WORLD angle aimed at (+ = knee ahead of hip)
    c_d: float = 1.001                 # rad per m of COM ahead of the stance ankle (foot placement)
    c_v: float = 0.078                 # rad per m/s of COM velocity
    swing_knee_flex: float = 1.377     # rad swing knee target in the first part of the swing (clearance) ...
    kp_knee_early: float = 0.21        # ... held loosely: the knee folds by itself as the hip pulls the thigh forward
    knee_extend_at: float = 0.302      # fraction of the swing after which the swing knee extends (gain kp_knee) ...
    knee_ext_thigh: float = 0.385      # rad ... once the swing thigh's world angle is past this
    swing_knee_ext: float = 0.18       # rad swing knee target for the touch-down
    swing_reach_at: float = 0.705      # fraction of the swing at which the thigh reaches its foot-placement angle
    g_comp: float = 0.473              # x the swing leg's own weight compensated (feed-forward)
    swing_ankle: float = 0.454         # rad swing ankle (+ = dorsiflexion, toe clearance)
    stance_knee: float = 0.009         # rad
    stance_ankle: float = -0.013       # rad stance ankle target
    push_off: float = 0.837            # N*m/kg extra plantarflexion of the trailing (about to swing) stance ankle
    pelvis_pitch: float = 0.051        # rad pelvis pitch set-point (+ = forward)
    v_des: float = 0.703               # m/s walking speed the person aims for
    k_lean: float = 0.37               # rad of extra forward pelvis lean per m/s below v_des
    max_lean: float = 0.15             # rad limit of that extra lean
    min_step: float = 0.069            # m the swing foot must be ahead of the stance foot for its touch-down to end
                                       # the swing
    init_lean: float = 0.105           # rad gait initiation: forward lean (ankle strategy) before the first step ...
    init_dx: float = 0.038             # m ... taken once the COM is this much further ahead of the ankles than in
                                       # quiet standing ...
    init_max: float = 1.5              # s ... or after this long
    ds_unload: float = 0.347           # x body weight: the trailing leg is lifted once its load is below this ...
    ds_max: float = 0.128              # s ... or after this long in double support
    ds_push: float = 0.952             # N*m/kg trailing-ankle plantarflexion (push-off) in double support
    kp_hip: float = 3.121              # swing hip
    kp_knee: float = 2.547             # swing knee
    kp_ankle: float = 4.496            # swing ankle
    kp_knee_st: float = 1.159          # stance knee (carries body weight)
    kp_ankle_st: float = 6.865         # stance ankle
    kp_pelvis: float = 8.943           # pelvis pitch through the stance hip(s)
    kd_ratio: float = 0.048            # s: kd = kd_ratio x kp
    predict: float = 0.024             # s internal forward model: the delayed state is extrapolated this far ahead

    @classmethod
    def for_subject(cls, name: str) -> "HumanWalkParams":
        """The walking parameters tuned for a subject preset (TUNED_WALK); the defaults (tuned for sci_t12_aisD)
        for any other name."""
        return cls(**TUNED_WALK.get(name, {}))


# HumanWalkParams tuned per subject preset (30 ms long-loop delay, zero-torque Indego, sagittal model); the
# HumanWalkParams defaults are the "sci_t12_aisD" set. [model]
TUNED_WALK: Dict[str, Dict[str, float]] = {
    "healthy": {"swing_time": 0.428, "min_swing_frac": 0.72, "theta0": 0.439, "c_d": 0.987, "c_v": 0.035,
                "swing_knee_flex": 1.4, "swing_knee_ext": 0.113, "swing_ankle": 0.406, "stance_knee": 0.012,
                "stance_ankle": 0.14, "push_off": 1.202, "pelvis_pitch": 0.004, "kp_hip": 3.081, "kp_knee": 1.871,
                "kp_knee_st": 2.003, "kp_ankle": 7.748, "kp_ankle_st": 6.337, "kp_pelvis": 10.697, "kd_ratio": 0.049,
                "v_des": 0.717, "k_lean": 0.433, "min_step": 0.039, "max_swing_frac": 1.437, "swing_reach_at": 0.731,
                "g_comp": 0.538, "knee_extend_at": 0.246, "kp_knee_early": 0.359, "predict": 0.026,
                "knee_ext_thigh": 0.493, "ds_unload": 0.37, "ds_max": 0.139, "ds_push": 1.15, "init_lean": 0.039,
                "init_dx": 0.067},
}


@dataclass
class VolitionParams:
    """What the person *wants* to do, independent of what the exo does."""
    cooperation: float = 0.5           # -1 resist, 0 passive, 1 lead/drive [model]
    effort: float = 1.0                # fraction of capacity used for tracking own plan
    walk: HumanWalkParams = field(default_factory=HumanWalkParams)   # gait_mode "walk"
    walk_effort: float = 1.0           # x effort while walking (0 = the person relaxes the legs and lets the exo walk
                                       # them; the trunk stays active) [model]
    stiffness_scale: float = 1.0       # how stiffly they hold their own plan (co-contraction)
    cadence_spm: float = 90.0          # intended steps/min
    cadence_cv: float = 0.03           # step-to-step variability (healthy ~2-3 %, SCI higher) [lit]
    phase_offset_s: float = 0.0        # systematic lead (+) / lag (-) vs exo step [model]
    phase_jitter_s: float = 0.03       # random timing jitter per step [lit]
    slack_rate: float = 0.0            # 1/s rate at which effort decays when exo does the work [model]
    lean_trigger_deg: float = 7.0      # whole-body forward lean the user has learned triggers a step [est]
    com_shift_gain: float = 0.9        # m of forward COM shift per rad of intended lean (logged intent; the crutch
                                       # user's lean target is SupportParams.lean_ahead)
    trunk_lean_fraction: float = 0.2   # part of the lean made by flexing the trunk (invisible to the exo's
                                       # posture estimate) vs. whole-body lean about the ankles [est]
    expected_speed: float = 0.35       # m/s progression a user without leg volition expects in the exo [lit 0.2-0.5]
    weight_shift: float = 0.035        # m lateral COM shift toward the next stance foot (exo users) [est]
    weight_shift_time: float = 0.25    # s time constant of the shift
    anticipation: float = 0.5          # 0..1 how much a trained exo user pre-loads the stance side before a step
    lean_rise_time: float = 0.35       # s
    lean_patience: float = 1.0         # s an exo user waits, fully leaned, before leaning further [est]
    lean_ramp: float = 0.03            # m/s extra forward COM lean while the device does not respond [est]
    lean_extra_max: float = 0.06       # m maximum extra lean (relaxed once the exo starts the step) [est]
    stop_prob_per_step: float = 0.0
    random_lean_rate_per_min: float = 0.0  # spurious leans (false triggers) [model]
    # people WITH leg volition (incomplete SCI, healthy): who makes the steps
    gait_mode: str = "own"             # "own": the person follows their own gait template (+ balance strategy);
                                       # "walk": the person walks on their own, SIMBICON-style
                                       # balance through the muscles (HumanWalkParams; exo zero torque / transparent;
                                       # sagittal model); "exo": the exo's walking mode drives hips and knees and the
                                       # person cooperates (lets the legs be moved; ankle lean + trailing-ankle push-off,
                                       # swing-foot clearance, sideways balance with the hip ab/adductors, trunk);
                                       # "follow": the exo walks the person (e.g. DynamicWalkController), who bears
                                       # weight on the loaded knee and keeps the trunk upright [model]
    request_lean: float = 0.03         # m COM shift past quiet standing (over the next stance foot) a trained user makes
                                       # to request an exo step: just past the device's threshold ("exo" mode) [est]
    push_off: float = 0.8              # N*m/kg trailing-ankle plantarflexion while requesting the next exo step
                                       # ("exo" mode; capped by the person's plantarflexor capacity) [est; healthy
                                       # push-off peaks at ~1.5 N*m/kg]
    exo_walk_speed: float = 0.2        # m/s the COM rolls forward over the stance foot during an exo step ("exo" mode;
                                       # keeps the momentum that carries it onto the front foot) [est; Indego 0.2-0.5]
    follow_speed_gain: float = 0.3     # s: ankle COP shift ahead of the COM per m/s above exo_walk_speed ("follow") [est]
    follow_assist: float = 0.0         # "follow": the person adds this fraction of the exo torque they feel on the
                                       # swinging leg (one reaction delay late), in the same direction [est]
    transfer_target: float = -0.05     # m where the weight transfer puts the COM relative to the front ankle before the
                                       # next exo step (- = behind; "exo" mode) [est]
    transfer_gain: float = 1.5         # 1/s: double-support weight transfer - forward COM speed the person aims for per m
                                       # it is still behind the next stance foot (capped at exo_walk_speed), made with
                                       # trailing-ankle push-off and their own hip/knee muscles ("exo" mode) [est]


@dataclass
class TherapistParams:
    """A physical therapist guarding / assisting through the gait belt (hip unit), as in clinical exo
    sessions. The assistance actually delivered is logged, so 'how much help did the person need with
    this controller' becomes an outcome measure (independent / contact guard / min / mod assist)."""
    level: str = "none"                # "none" | "contact_guard" | "min" | "mod"
    force_cap: dict = field(default_factory=lambda: {"contact_guard": 60.0, "min": 120.0, "mod": 250.0})  # N [est]
    moment_cap: dict = field(default_factory=lambda: {"contact_guard": 10.0, "min": 25.0, "mod": 50.0})  # N*m [est]
    deadband: dict = field(default_factory=lambda: {"contact_guard": 0.06, "min": 0.02, "mod": 0.0})     # m COM error
    kp: float = 1500.0                 # N/m toward the person's own intended COM position [est]
    kd: float = 300.0                  # N*s/m
    k_tilt: float = 150.0              # N*m/rad pelvis/trunk pitch & roll
    d_tilt: float = 20.0
    reaction_delay: float = 0.15       # s


@dataclass
class PlanarizerParams:
    """Test rig, not part of the person: a planarizing boom on the pelvis that confines the body to the sagittal
    plane, as used to develop gait controllers for devices that only act in that plane (the Indego has no
    frontal-plane actuation). Stiff, undelayed spring-damper against sideways translation, roll and heading change
    of the pelvis; forward and vertical translation and pitch stay free. The force it delivers is logged: it is the
    frontal-plane balance the person (or a therapist) would otherwise have to provide."""
    enabled: bool = False
    k_lat: float = 1.0e5               # N/m sideways stiffness at the pelvis COM [model]
    d_lat: float = 5.0e3               # N*s/m [model]
    k_rot: float = 5.0e3               # N*m/rad roll and heading stiffness [model]
    d_rot: float = 400.0               # N*m*s/rad [model]


@dataclass
class SupportParams:
    """Upper-limb support (crutches / walker). Each crutch transmits a ground reaction R_i whose
    line of action passes through its tip; R_i respects push-only, the tip friction cone, a per-arm
    load cap and a shoulder-moment cap (the part of R_i not along the crutch axis must be made by the
    shoulder/elbow/wrist). The person chooses R_i with a delayed postural controller (whole-body COM
    about the feet + trunk orientation) plus intrinsic, undelayed arm impedance."""
    kind: str = "crutches"             # "none" | "crutches" | "walker" | "harness" (overhead BWS on a trolley)
    harness_height: float = 0.5        # m spreader bar above the shoulders (sets the pendulum restoring stiffness)
    trolley_lag: float = 0.3           # s the robotic trolley lags the person horizontally
    max_load_frac: float = 0.45        # per arm, fraction of body weight [lit: 20-50 %BW crutch loads in SCI exo gait]
    friction: float = 0.8              # crutch tip friction
    shoulder_moment_max: float = 40.0  # N*m per arm for off-axis crutch force [est; 0 = pure axial strut]
    k_tilt: float = 0.0                # N*m/rad extra delayed trunk-orientation feedback via the arms [model; >~50 oscillates]
    d_tilt: float = 0.0
    k_tilt_int: float = 150.0          # N*m/rad intrinsic (undelayed) arm/shoulder stiffness on trunk orientation
    d_tilt_int: float = 25.0
    # heading (yaw) through the arms: with the tips ~0.66 m apart, opposite fore-aft tip forces twist the body.
    # Without this term any asymmetric tip force spins the user about the vertical axis in single support.
    k_yaw: float = 60.0                # N*m/rad delayed heading feedback toward the walking direction [model]
    d_yaw: float = 10.0
    k_yaw_int: float = 150.0           # N*m/rad intrinsic (undelayed) arm stiffness against trunk twist [model]
    d_yaw_int: float = 25.0
    w_yaw: float = 0.7                 # allocation weight of the heading moment
    k_arm: float = 15000.0             # N/m intrinsic axial stiffness of extended arm + crutch (0 = pure force control)
                                       # [est; co-contracted arm, short-range stiffness + 30-50 ms stretch reflexes]
    c_arm: float = 300.0               # N*s/m
    arm_set_time: float = 0.2          # s time constant with which the person re-sets the arm length to get the
                                       # crutch load they want
    arm_set_time_stepping: float = 5.0 # s the same while the exo is stepping (large = arms locked, stiff tripod)
    w_trunk: float = 0.7               # allocation weight of the trunk-orientation moment vs whole-body balance
    w_lateral: float = 3.0             # allocation priority of the lateral (roll) balance moment over the AP one
    w_load: float = 0.05               # allocation weight of the preferred total crutch load
    k_lat: float = 800.0               # N/m lateral COM stabilisation (differential crutch push)
    d_lat: float = 200.0
    k_ap: float = 600.0                # N/m antero-posterior COM over the base of support
    d_ap: float = 300.0
    k_height: float = 4000.0           # N/m vertical unloading when sagging
    d_height: float = 400.0
    weight_bearing_frac: float = 0.15  # baseline arm weight-bearing (fraction BW, both arms)
    single_support_frac: float = 0.25  # extra arm weight-bearing while one foot is off the ground [lit 30-50 %BW total]
    com_ahead: float = 0.03            # m desired COM ahead of mid-foot when standing (tripod lean)
    lean_ahead: float = 0.08           # m beyond the FRONT foot the person leans to request a step [est]
    kp_com: float = 4.0                # 1/s^2 desired COM acceleration per m of error (crutch balance)
    kd_com: float = 3.0                # 1/s
    kp_lat_com: float = 6.0            # lateral (weight shift onto the stance leg)
    kd_lat_com: float = 5.0
    walking_extra_frac: float = 0.12   # extra crutch weight-bearing while walking (fraction BW, total)
    max_ahead_walking: float = 0.15    # m the COM may get ahead of the front foot before the person brakes
    tip_margin: float = 0.12           # m during an exo step the COM may travel up to this far behind the crutch
                                       # tips (inside the stance-foot + crutch tripod) [model]
    lean_speed: float = 0.15           # m/s forward drift allowed while leaning to request the first step
    tip_ahead: float = 0.25            # m crutch/walker tips placed ahead of the feet
    tip_lateral: float = 0.33          # m crutch tips lateral of the COM (scaled with height) [est]
    pref_weight: float = 0.003         # how strongly the person sticks to their preferred crutch load vs balance
    prediction: float = 1.0            # 0..1 how much the person compensates their own reaction delay (internal model)
    reaction_delay: float = 0.12       # s upper-limb reaction delay [lit 100-150 ms]
    four_point_lift: bool = True       # crutches are lifted and re-planted during gait
    crutch_pattern: str = "ds"         # "ds": advance both crutches in double support after each step (typical
                                       # early exo training); "2point": opposite crutch moves with the swing leg;
                                       # "4point": same-side crutch moves with the swing leg
    crutch_swing_time: float = 0.35    # s a crutch is off the ground
    lift_margin: float = 0.03          # m the COM may be ahead of its target when the person lifts a crutch
    lift_lat_margin: float = 0.03      # m the COM must be inside the opposite foot (laterally) before a crutch
                                       # is lifted: the remaining crutch can only push the body toward its own
                                       # opposite side, so a fall toward the lifted crutch cannot be caught [model]
    lift_shift: float = 0.01           # m extra lateral margin aimed for before lifting a crutch [model]
    tip_useless: float = 0.05          # m a crutch whose tip is less than this ahead of the shoulder cannot stop a
                                       # forward fall, so the person moves it even with the COM ahead [model]
    crutch_wait_max: float = 1.5       # s after which a pending crutch is moved anyway (AP check only) [est]
    therapist: TherapistParams = field(default_factory=TherapistParams)
    planarizer: PlanarizerParams = field(default_factory=PlanarizerParams)


@dataclass
class SimConfig:
    timestep: float = 0.0005           # s physics step [model]
    human_rate_hz: float = 1000.0      # spinal reflex / crutch update rate [model]
    stand_lean_deg: float = 6.0        # initial whole-body forward lean about the ankles (COM ahead of ankles)
    integrator: str = "implicitfast"
    floor_friction: float = 1.0
    seed: int = 0
    planar: bool = False               # sagittal-plane model: frontal/transverse joints locked (lumbar side-bend and
                                       # rotation, hip ab/adduction and rotation, subtalar, exo hip ab/adduction and
                                       # twist compliance) and the pelvis held by the planarizer (SupportParams) [model]


def load_yaml(obj, path_or_dict):
    """Recursively override dataclass fields from a YAML file or dict."""
    import yaml
    d = path_or_dict
    if isinstance(path_or_dict, str):
        with open(path_or_dict) as f:
            d = yaml.safe_load(f) or {}
    return _update(obj, d)


def _update(obj, d):
    for k, v in d.items():
        if not hasattr(obj, k):
            raise KeyError(f"{type(obj).__name__} has no field '{k}'")
        cur = getattr(obj, k)
        if dc.is_dataclass(cur) and isinstance(v, dict):
            _update(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, (list, tuple)):
            setattr(obj, k, tuple(v))
        else:
            setattr(obj, k, v)
    return obj


def randomize(p, rng: np.random.Generator, rel: float = 0.25,
              skip=("gear", "cogging_poles", "enc_bits", "control_rate_hz", "timestep", "human_rate_hz", "ambient_C",
                    "t_winding_max", "mass_hip_unit_battery", "trunk_imu_rate_hz", "planarizer",
                    # walking-strategy settings of people with leg volition (not part of the domain randomisation)
                    "request_lean", "push_off", "exo_walk_speed", "transfer_gain", "transfer_target",
                    "follow_speed_gain", "follow_assist", "walk_effort", "walk", "foot_switch_threshold")):
    """Return a copy with every positive float field multiplied by U(1-rel, 1+rel) (log-uniform)."""
    q = copy.deepcopy(p)

    def rec(o):
        for f in dc.fields(o):
            v = getattr(o, f.name)
            if f.name in skip:
                continue
            if dc.is_dataclass(v):
                rec(v)
            elif isinstance(v, float) and v > 0:
                setattr(o, f.name, float(v * np.exp(rng.uniform(np.log(1 - rel), np.log(1 + rel)))))
            elif isinstance(v, tuple) and all(isinstance(x, float) for x in v) and all(x > 0 for x in v):
                setattr(o, f.name, tuple(float(x * np.exp(rng.uniform(np.log(1 - rel), np.log(1 + rel)))) for x in v))
    rec(q)
    return q
