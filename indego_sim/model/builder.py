"""Programmatic construction of the coupled MyoLeg + Indego model (MuJoCo MjSpec).

Topology (all frames: x forward, y left, z up at the reference pose)::

    world
     └─ Full Body (free joint)            MyoLeg 80-muscle legs + passive torso scaffold
         ├─ sacrum → lumbar5 (+3 trunk DOF added here, torque-driven) → … torso, head
         └─ pelvis
             ├─ femur_l/r → tibia → talus → calcn → toes        (human, muscle driven)
             └─ exo_hipunit  [6-DOF belt bushing]                (Indego hip unit)
                 └─ exo_thigh_s  [abd + twist compliance, exo_hip_s hinge, dcmotor]
                     ├─ exo_thigh_cuff_s [6-DOF bushing]  ══weld══ femur_s
                     └─ exo_shank_s  [exo_knee_s hinge, dcmotor, brake]
                         ├─ exo_shank_cuffP_s / cuffD_s [bushings] ══weld══ tibia_s
                         └─ exo_afo_s [passive AFO ankle hinge]
                             └─ exo_shoe_s [bushing] ══weld══ calcn_s

Interaction compliance lives in the bushing *joints* (exact physical stiffness and
damping in N/m, N*m/rad); the welds only close the kinematic loops and are made
much stiffer than the bushings.
With transmission.mode == "backlash" each drive gets an extra motor-side hinge and a
dead-band joint (MuJoCo's dual-joint backlash decomposition).

The exo segment meshes (indego_sim/assets/indego/*.stl, made with tools/process_indego_mesh.py from a commercial 3D
model) are not distributed. Without them each segment is drawn and weighted as a box with the mesh's mass
properties (same centre of mass, principal axes and inertia), so the dynamics are unchanged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mujoco
import numpy as np

from ..config import ExoParams, SensorParams, SimConfig, SubjectParams
from .utils import apply_couplings

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "indego")
SIDES = ("l", "r")
HUMAN_NOMINAL_HEIGHT = 1.75
HUMAN_NOMINAL_MASS = 74.975

# Indego segment geometry (m) measured on the un-posed 3D model, in the segment frames of
# tools/process_indego_mesh.py (which also writes it to assets/indego/indego_geometry.json; that file wins)
INDEGO_GEOMETRY = {
    "thigh_length_m": 0.42415637471036327,
    "shank_length_knee_to_afo_m": 0.30173804898282536,
    "afo_attach_in_shank_frame": [0.0015882520588113285, 0.0, -0.301733868929823],
    "thigh_bar_lateral_y_m": 0.16616278648376465,
    "thigh_cuff_centre_in_thigh": [0.001646305572566564, 0.08989309400320053, -0.28855080146850137],
    "shank_cuff_prox_centre_in_shank": [0.023262752951894283, 0.09247095247737974, -0.1170132953013188],
    "shank_cuff_dist_centre_in_shank": [0.018619090555277815, 0.08096261966150228, -0.31646539536660706],
    "afo_plate_z_min_in_afo": -0.18389203850857233,
}

# Mass properties of each segment mesh at unit scale and unit mass (shell inertia): centre of mass and principal
# axes in the segment frame, half-sizes of the solid box with the same principal inertia. Used when the meshes are
# not installed.
MESH_BOXES = {
    "afo_plate_l": ([0.06179, 0.0892, -0.16171], [0.5147, 0.46467, 0.66156, 0.28549], [0.03216, 0.07559, 0.11927]),
    "afo_plate_r": ([0.06179, -0.0892, -0.16171], [0.28549, 0.66156, 0.46467, 0.5147], [0.03216, 0.07559, 0.11927]),
    "hipunit_frame": ([-0.05739, 1e-05, 0.13611], [0.47644, 0.4771, -0.52167, 0.52272], [0.11667, 0.17031, 0.21389]),
    "hipunit_led": ([-0.03115, 0.0, 0.18576], [0.59537, 0.59537, -0.38149, 0.38149], [0.00062, 0.00576, 0.31037]),
    "hipunit_pads": ([-0.03975, -3e-05, 0.14198], [0.43184, 0.43296, -0.55889, 0.56009], [0.12048, 0.16109, 0.19121]),
    "shank_cuff_l": ([0.01782, 0.08154, -0.2029], [0.56925, -0.01688, 0.01291, 0.82189], [0.05929, 0.06606, 0.17686]),
    "shank_cuff_r": ([0.01784, -0.08159, -0.20273], [0.82209, 0.01279, -0.017, 0.56896], [0.05931, 0.06607, 0.1768]),
    "shank_frame_l": ([0.00113, 0.14487, -0.19095], [0.64178, -0.05407, 0.05422, 0.76306], [0.03082, 0.05346, 0.15551]),
    "shank_frame_r": ([0.00096, -0.14474, -0.19109], [0.76541, 0.05462, -0.05389, 0.63896], [0.03088, 0.05369, 0.15564]),
    "thigh_cuff_l": ([0.02865, 0.07427, -0.28608], [-0.2086, 0.76482, 0.11828, 0.59795], [0.04851, 0.08862, 0.10322]),
    "thigh_cuff_r": ([0.02869, -0.0743, -0.2861], [0.20859, 0.76496, -0.11839, 0.59776], [0.04852, 0.08861, 0.10322]),
    "thigh_frame_l": ([-0.00085, 0.17738, -0.22482], [0.65708, -0.01214, 0.00557, 0.7537], [0.04152, 0.08029, 0.24014]),
    "thigh_frame_r": ([-0.00089, -0.17737, -0.22493], [0.75383, 0.00558, -0.01226, 0.65693], [0.04154, 0.08028, 0.23974]),
}


@dataclass
class ModelInfo:
    """Names/indices the runtime needs; filled by build_model()."""
    exo_joints: List[str] = field(default_factory=list)          # actuated hinges, order hip_l, knee_l, hip_r, knee_r
    exo_dofs: List[str] = field(default_factory=lambda: ["hip_l", "knee_l", "hip_r", "knee_r"])
    exo_actuators: List[str] = field(default_factory=list)
    exo_backlash_joints: List[Optional[str]] = field(default_factory=list)  # dead-band joints (backlash mode)
    gears: List[float] = field(default_factory=list)
    trunk_actuators: List[str] = field(default_factory=list)
    muscles: List[str] = field(default_factory=list)
    bushings: Dict[str, List[str]] = field(default_factory=dict)  # bushing name -> its 6 joint names
    bushing_body: Dict[str, str] = field(default_factory=dict)
    welds: Dict[str, str] = field(default_factory=dict)          # bushing name -> equality name
    exo_sensors: List[str] = field(default_factory=list)
    stand_qpos: Optional[np.ndarray] = None
    exo_mass: float = 0.0
    human_mass: float = 0.0
    geometry: Dict = field(default_factory=dict)
    with_exo: bool = True


def _quat_from_mat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q


def _compile(spec):
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _load_myolegs():
    try:
        from myo_sim.build.compose import build_spec
    except ImportError as e:  # pragma: no cover
        raise ImportError("myo_sim is required: pip install myo-sim  (or clone MyoHub/myo_sim)") from e
    spec = build_spec("myolegs")
    _flat_ground(spec)
    return spec


def _flat_ground(spec):
    """MyoSuite's scene stands the model on a pedestal: a collision cylinder of 1.05 m radius whose top is at
    z = 0.015, with the floor plane 0.4 m lower - anyone who walks more than ~1 m steps off its edge and falls.
    Replace it by an unbounded floor at the pedestal height with the same contact parameters (standing and stepping
    on it are unchanged). The scene's room and pedestal meshes (visual only) are hidden and a plain sky is drawn
    instead, so renders show the ground the model actually stands on."""
    floor = cyl = None
    for g in spec.worldbody.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE and g.name == "floor":
            floor = g
        elif g.type == mujoco.mjtGeom.mjGEOM_CYLINDER and g.contype:
            cyl = g
        elif g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.group = 5                                   # not drawn by default (visible groups 0-2)
    if floor is None or cyl is None:
        return
    floor.pos = [floor.pos[0], floor.pos[1], cyl.pos[2] + cyl.size[1]]
    floor.size = [100.0, 100.0, floor.size[2]]          # drawn size only (a plane collides everywhere)
    cyl.contype = cyl.conaffinity = 0
    sky = spec.add_texture()
    sky.name = "indego_sim_sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.93, 0.94, 0.96]
    sky.rgb2 = [0.62, 0.66, 0.72]
    sky.width = sky.height = 512


# ----------------------------------------------------------------------------
# subject scaling (uniform geometric scale + mass scale)
# ----------------------------------------------------------------------------

def _scale_spec(spec, s: float, ms: float):
    """Uniformly scale lengths by s and masses by ms (inertia by ms*s^2).

    Muscle max force is scaled by ms**(2/3) (cross-sectional area ~ mass^(2/3)) and the
    muscle length ranges by s.
    """
    if abs(s - 1) < 1e-9 and abs(ms - 1) < 1e-9:
        return
    for b in spec.bodies:
        if b.name == "world":
            continue
        b.pos = np.asarray(b.pos) * s
        b.ipos = np.asarray(b.ipos) * s
        if b.mass > 0:
            b.mass = b.mass * ms
            b.inertia = np.asarray(b.inertia) * ms * s * s
            if np.any(np.asarray(b.fullinertia)):
                b.fullinertia = np.asarray(b.fullinertia) * ms * s * s
    for j in spec.joints:
        j.pos = np.asarray(j.pos) * s
        if j.type == mujoco.mjtJoint.mjJNT_SLIDE:
            j.range = np.asarray(j.range) * s
        j.damping = j.damping * ms * s * s
        j.armature = j.armature * ms * s * s
    for g in spec.geoms:
        if g.parent.name == "world" and g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        g.pos = np.asarray(g.pos) * s
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            g.size = np.asarray(g.size) * s
        if np.any(np.isfinite(g.fromto)) and not np.all(np.asarray(g.fromto) == 0):
            g.fromto = np.asarray(g.fromto) * s
    for me in spec.meshes:
        me.scale = np.asarray(me.scale) * s
    for site in spec.sites:
        site.pos = np.asarray(site.pos) * s
        site.size = np.asarray(site.size) * s
    for e in spec.equalities:
        if e.type == mujoco.mjtEq.mjEQ_JOINT:
            # polynomial coupling q1 = c0 + c1 q2 + ...: translation joints scale with s
            j1 = spec.joint(e.name1)
            if j1 is not None and j1.type == mujoco.mjtJoint.mjJNT_SLIDE:
                e.data = np.asarray(e.data) * np.r_[np.full(5, s), np.ones(len(e.data) - 5)]
    for a in spec.actuators:
        if a.gaintype == mujoco.mjtGain.mjGAIN_MUSCLE:
            gp = np.asarray(a.gainprm).copy()
            if gp[2] > 0:
                gp[2] *= ms ** (2 / 3)
            a.gainprm = gp
            bp = np.asarray(a.biasprm).copy()
            if bp[2] > 0:
                bp[2] *= ms ** (2 / 3)
            a.biasprm = bp
            a.lengthrange = np.asarray(a.lengthrange) * s


# ----------------------------------------------------------------------------
# helpers to add things
# ----------------------------------------------------------------------------

def _add_mesh(spec, name, scale=(1, 1, 1)):
    """Register an Indego segment mesh; returns (mesh name, or None when the mesh is not installed, base name,
    scale)."""
    base = name.split("@")[0]
    path = os.path.join(ASSET_DIR, f"indego_{base}.stl")
    if not os.path.exists(path):
        return None, base, np.asarray(scale, float)
    me = spec.add_mesh()
    me.name = name
    me.file = path
    me.scale = list(scale)
    me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    return name, base, np.asarray(scale, float)


MATERIAL_RGBA = {"indego_frame": [0.26, 0.27, 0.30, 1], "indego_soft": [0.10, 0.10, 0.11, 1],
                 "indego_carbon": [0.16, 0.16, 0.18, 1], "indego_led": [0.25, 0.95, 0.95, 1]}


def _visual(body, mesh, material, mass=0.0, group=1, offset=(0.0, 0.0, 0.0)):
    """Segment shell (no contacts) carrying `mass`: the mesh from _add_mesh, or its mass-equivalent box."""
    name, base, scale = mesh
    g = body.add_geom()
    if name is not None:
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = name
        g.pos = list(offset)
    else:
        pos, quat, half = (np.asarray(v, float) for v in MESH_BOXES[base])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = list(np.asarray(offset) + scale * pos)
        g.quat = list(quat)
        g.size = [max(h * np.linalg.norm(scale * R[:, i]), 1e-4) for i, h in enumerate(half)]
    g.material = material
    g.rgba = MATERIAL_RGBA[material]   # MyoLeg's default class sets a geom rgba that would override the material
    g.contype = 0
    g.conaffinity = 0
    g.group = group
    g.mass = mass
    return g


def _point_mass(body, pos, mass, radius=0.03):
    g = body.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [radius, 0, 0]
    g.pos = list(pos)
    g.mass = mass
    g.contype = 0
    g.conaffinity = 0
    g.group = 4
    g.rgba = [0.9, 0.3, 0.1, 0.3]
    return g


def _collision_capsule(body, p0, p1, r, name):
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.fromto = list(p0) + list(p1)
    g.size = [r, 0, 0]
    g.mass = 0.0
    g.contype = 1          # collides with the floor only (human geoms have conaffinity 0)
    g.conaffinity = 0
    g.group = 3
    g.rgba = [0.2, 0.6, 0.9, 0.3]
    return g


def _add_bushing(spec, parent, name, pos, bp, info: ModelInfo, weld_to: str, weld_solref, mass=0.06,
                 interface_model="bushing"):
    """Body with 3 slide + 3 hinge visco-elastic joints, welded to a human segment."""
    b = parent.add_body()
    b.name = name
    b.pos = list(pos)
    joints = []
    if interface_model == "bushing":
        axes = np.eye(3)
        for i, ax in enumerate("xyz"):
            j = b.add_joint()
            j.name = f"{name}_t{ax}"
            j.type = mujoco.mjtJoint.mjJNT_SLIDE
            j.axis = axes[i]
            j.stiffness = [bp.k_lin[i], 0.0, bp.k_lin[i] * bp.cubic / bp.range_lin ** 2]
            j.damping = [bp.c_lin[i], 0.0, 0.0]
            j.armature = bp.wobble_mass
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            j.range = [-bp.range_lin, bp.range_lin]
            joints.append(j.name)
        for i, ax in enumerate("xyz"):
            j = b.add_joint()
            j.name = f"{name}_r{ax}"
            j.type = mujoco.mjtJoint.mjJNT_HINGE
            j.axis = axes[i]
            j.stiffness = [bp.k_rot[i], 0.0, bp.k_rot[i] * bp.cubic / bp.range_rot ** 2]
            j.damping = [bp.c_rot[i], 0.0, 0.0]
            j.armature = bp.wobble_inertia
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            j.range = [-bp.range_rot, bp.range_rot]
            joints.append(j.name)
    g = b.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [0.012, 0, 0]
    g.mass = mass
    g.contype = 0
    g.conaffinity = 0
    g.group = 4
    g.rgba = [0.95, 0.75, 0.1, 0.6]
    s = b.add_site()
    s.name = f"{name}_site"
    s.size = [0.01, 0, 0]
    s.group = 4
    e = spec.add_equality()
    e.name = f"{name}_weld"
    e.type = mujoco.mjtEq.mjEQ_WELD
    e.objtype = mujoco.mjtObj.mjOBJ_BODY
    e.name1 = name
    e.name2 = weld_to
    data = np.zeros(11)
    data[10] = 1.0            # torquescale; relpose quat = 0 -> use reference-pose relative pose
    e.data = data
    if interface_model == "bushing":
        e.solref = list(weld_solref)
        e.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    else:  # isotropic soft weld: timeconst from axial stiffness & wobble mass (approximate)
        k = float(np.mean(bp.k_lin))
        tc = max(2 * weld_solref[0], np.sqrt(bp.wobble_mass / k))
        e.solref = [tc, 1.0]
        e.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
    info.bushings[name] = joints
    info.bushing_body[name] = name
    info.welds[name] = e.name
    return b


def _add_dcmotor(spec, name, joint, mp, gear, timestep):
    a = spec.add_actuator()
    a.name = name
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.target = joint
    tau_peak_motor = mp.K * mp.i_peak
    a.set_to_dcmotor(
        motorconst=[mp.K, 0.0],
        resistance=mp.R,
        saturation=[tau_peak_motor, 0.0, mp.di_dt_max],
        inductance=[0.0, mp.te],
        cogging=[mp.cogging_amp, float(mp.cogging_poles), 0.0],
        controller=[0.0, 0.0, 0.0, 0.0, 0.0, mp.v_max],
        thermal=[mp.thermal_R, mp.thermal_C, 0.0, mp.temp_coef, mp.ambient_C, mp.ambient_C],
        lugre=([mp.lugre_stiffness, mp.lugre_damping, mp.coulomb, mp.static, mp.stribeck_vel]
               if mp.friction_model == "lugre" else [0.0, 0.0, 0.0, 0.0, 0.0]),
        ctrlspec=int(mujoco.mjtCtrlInput.mjINPUT_FF),
    )
    a.gear = [gear, 0, 0, 0, 0, 0]
    a.armature = mp.rotor_inertia
    a.damping = [mp.viscous, 0.0, 0.0]
    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    a.ctrlrange = [-tau_peak_motor, tau_peak_motor]
    if mp.cmd_delay > 0:
        n = int(np.ceil(mp.cmd_delay / timestep)) + 1
        a.nsample = n
        a.delay = mp.cmd_delay
    return a


def _sensor(spec, name, stype, objtype, objname, period, delay, timestep, refname=None, reftype=None):
    s = spec.add_sensor()
    s.name = name
    s.type = stype
    s.objtype = objtype
    s.objname = objname
    if refname is not None:
        s.reftype = reftype
        s.refname = refname
    if period > 0 or delay > 0:
        s.nsample = int(np.ceil(max(delay, period) / max(period, timestep))) + 3
        if period > 0:
            s.interval = [period, 0.0]
        s.delay = delay
    return s


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------

def build_model(exo: Optional[ExoParams] = None, subject: Optional[SubjectParams] = None,
                sim: Optional[SimConfig] = None, sensors: Optional[SensorParams] = None,
                with_exo: bool = True, return_spec: bool = False, base: str = "free", lift: float = 0.0):
    """Build the coupled model.

    base: "free" (normal), or "fixed" = pelvis welded to the world at standing height + `lift`
          (bench tests: exo hanging in the air with the passive human leg, system identification).
    """
    exo = exo or ExoParams()
    subject = subject or SubjectParams()
    sim = sim or SimConfig()
    sensors = sensors or SensorParams()
    info = ModelInfo(with_exo=with_exo)
    gfile = os.path.join(ASSET_DIR, "indego_geometry.json")
    geom = {**INDEGO_GEOMETRY, **(json.load(open(gfile)) if os.path.exists(gfile) else {})}
    info.geometry = dict(geom)

    spec = _load_myolegs()
    spec.modelname = "myoleg_indego" if with_exo else "myoleg"
    spec.option.timestep = sim.timestep
    spec.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{sim.integrator.upper()}")
    if with_exo and "lugre" in (exo.hip_motor.friction_model, exo.knee_motor.friction_model) \
            and sim.integrator.lower() != "euler":
        raise ValueError("LuGre friction in MuJoCo's dcmotor needs integrator='euler' "
                         "(implicit integrators add spurious bristle damping, see docs/MODELING_NOTES.md)")
    spec.option.iterations = 50
    spec.option.ls_iterations = 20
    spec.compiler.inertiagrouprange = [0, 5]

    # ---- re-orient: x forward, y left, z up (MyoLeg composite faces -y)
    fb = spec.body("Full Body")
    fb.quat = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]

    # ---- subject scaling
    s_len = subject.height / HUMAN_NOMINAL_HEIGHT
    s_mass = subject.mass / HUMAN_NOMINAL_MASS
    _scale_spec(spec, s_len, s_mass)

    # ---- trunk: 3 torque-driven lumbar DOF at L5/S1 (scaffold is rigid in MyoLeg)
    m0, d0 = _compile(spec)
    R_l5 = d0.body("lumbar5").xmat.reshape(3, 3)
    trunk_axes = {"lumbar_flexion": [0, 1, 0], "lumbar_bending": [-1, 0, 0], "lumbar_rotation": [0, 0, 1]}
    trunk_ranges = {"lumbar_flexion": (-0.5, 1.0), "lumbar_bending": (-0.5, 0.5), "lumbar_rotation": (-0.5, 0.5)}
    l5 = spec.body("lumbar5")
    for jn, ax in trunk_axes.items():
        j = l5.add_joint()
        j.name = jn
        j.type = mujoco.mjtJoint.mjJNT_HINGE
        j.axis = R_l5.T @ np.asarray(ax, float)
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.range = list(trunk_ranges[jn])
        j.stiffness = [40.0 * s_mass * s_len, 0.0, 0.0]   # passive spine stiffness [lit 20-100 Nm/rad]
        j.damping = [5.0 * s_mass * s_len, 0.0, 0.0]
        j.armature = 0.02
        a = spec.add_actuator()
        a.name = f"trunk_{jn.split('_')[1]}"
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.target = jn
        a.set_to_motor()
        lim = 250.0 * s_mass * s_len * max(subject.trunk_strength, 1e-3)
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = [-lim, lim]
        info.trunk_actuators.append(a.name)

    if subject.shoes:
        _add_shoes(spec, subject, s_len)

    # materials
    for nm, spec_, shin in [("indego_frame", 0.5, 0.6), ("indego_soft", 0.1, 0.1),
                            ("indego_carbon", 0.7, 0.8), ("indego_led", 0.3, 0.3)]:
        rgba = MATERIAL_RGBA[nm]
        mat = spec.add_material()
        mat.name = nm
        mat.rgba = rgba
        mat.specular = spec_
        mat.shininess = shin
        if nm == "indego_led":
            mat.emission = 1.0

    if with_exo:
        _build_exo(spec, exo, sim, sensors, info, geom, s_len)

    # ---- compile and find a standing pose with feet on the floor
    m, d = _compile(spec)
    if with_exo and exo.transmission.mode == "backlash":
        # MuJoCo's negative solref is acceleration-level: scale physical N*m/rad by the
        # DOF's inverse inertia at the reference pose so the tooth contact has ~ the requested stiffness
        for jb in info.exo_backlash_joints:
            dof = m.jnt_dofadr[m.joint(jb).id]
            iw = float(m.dof_invweight0[dof])
            js = spec.joint(jb)
            js.solref_limit = [-exo.transmission.stiffness * iw, -exo.transmission.damping * iw]
        m, d = _compile(spec)
    info.muscles = [m.actuator(i).name for i in range(m.nu) if m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE]
    info.human_mass = float(sum(m.body_mass[i] for i in range(m.nbody) if not m.body(i).name.startswith("exo_")))
    info.exo_mass = float(sum(m.body_mass[i] for i in range(m.nbody) if m.body(i).name.startswith("exo_")))
    zmin = np.inf
    for gi in range(m.ngeom):
        if m.geom_contype[gi] and m.geom_bodyid[gi] > 0 and m.geom(gi).name.startswith(("foot", "bofoot")):
            zmin = min(zmin, _geom_lowest_z(m, d, gi))
    q = apply_couplings(m, m.qpos0.copy())
    q[2] -= zmin - 0.0005
    if sim.stand_lean_deg and base == "free":
        q = _lean_pose(m, info, q, np.radians(sim.stand_lean_deg), with_exo)
    if base == "fixed":
        fbody = spec.body("Full Body")
        fbody.pos = [q[0], q[1], q[2] + lift]
        for j in list(spec.joints):
            if j.name == "root":
                spec.delete(j)
        m = spec.compile()
        q = apply_couplings(m, m.qpos0.copy())
    info.stand_qpos = q
    k = spec.add_key()
    k.name = "stand"
    k.qpos = q
    if sim.planar:
        _lock_frontal_transverse(spec, m, q)
    m = spec.compile()
    if return_spec:
        return m, info, spec
    return m, info


PLANAR_LOCKED = ("lumbar_bending", "lumbar_rotation", "hip_adduction_l", "hip_adduction_r", "hip_rotation_l",
                 "hip_rotation_r", "subtalar_angle_l", "subtalar_angle_r", "exo_hip_l_abd", "exo_hip_r_abd",
                 "exo_hip_l_twist", "exo_hip_r_twist")


def _lock_frontal_transverse(spec, m, q):
    """SimConfig.planar: hold every frontal/transverse joint at its standing-keyframe value (joint equality, stiff;
    the joints stay in qpos, so indexing and the muscle model are unchanged)."""
    for name in PLANAR_LOCKED:
        try:
            j = m.joint(name)
        except KeyError:
            continue
        e = spec.add_equality()
        e.name = f"planar_{name}"
        e.type = mujoco.mjtEq.mjEQ_JOINT
        e.objtype = mujoco.mjtObj.mjOBJ_JOINT
        e.name1 = name
        data = np.zeros(11)
        data[0] = q[m.jnt_qposadr[j.id]] - m.qpos0[m.jnt_qposadr[j.id]]      # q1 - q1_ref = data[0]
        e.data = data
        e.solref = [0.004, 1.0]


def _lean_pose(m, info, q, lean, with_exo):
    """Whole-body forward lean about the ankles with flat feet (ankles dorsiflexed by `lean`)."""
    from .utils import apply_couplings as _ac
    d = mujoco.MjData(m)
    q = q.copy()
    # rotate the free-joint orientation by +lean about world y (top forward)
    qy = np.array([np.cos(lean / 2), 0.0, np.sin(lean / 2), 0.0])
    qn = np.zeros(4)
    mujoco.mju_mulQuat(qn, qy, q[3:7])
    q[3:7] = qn
    for s in ("l", "r"):
        q[m.jnt_qposadr[m.joint(f"ankle_angle_{s}").id]] = lean          # MyoLeg: + = dorsiflexion
        if with_exo:
            q[m.jnt_qposadr[m.joint(f"exo_afo_ankle_{s}").id]] = -lean   # AFO hinge: + = plantarflexion
    _ac(m, q)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    zmin, xs = np.inf, []
    for gi in range(m.ngeom):
        if m.geom_contype[gi] and m.geom_bodyid[gi] > 0 and m.geom(gi).name.startswith(("foot", "bofoot")):
            zmin = min(zmin, _geom_lowest_z(m, d, gi))
    q[2] -= zmin - 0.0005
    q[0] -= 0.5 * (d.body("talus_l").xpos[0] + d.body("talus_r").xpos[0])   # ankles at x = 0
    return q


def _add_shoes(spec, subject: SubjectParams, s_len: float):
    """Replace the bare-foot contact capsules by shoe soles: a heel-to-ball box on the calcaneus and a forefoot
    box on the toes, at the same sole height, `shoe_width` wide (the MTP joint keeps the forefoot flexible)."""
    m0, d0 = _compile(spec)
    W = subject.shoe_width * s_len
    h = 0.012 * s_len
    plan, disable = [], []
    for side in SIDES:
        geoms = {"calcn": [], "toes": []}
        for gi in range(m0.ngeom):
            nm = m0.geom(gi).name
            b = m0.body(m0.geom_bodyid[gi]).name
            if m0.geom_contype[gi] and nm.startswith(("foot", "bofoot")) and b in (f"calcn_{side}", f"toes_{side}"):
                geoms[b.split("_")[0]].append(gi)
        if not geoms["calcn"] or not geoms["toes"]:
            continue
        zb = min(_geom_lowest_z(m0, d0, gi) for gi in geoms["calcn"] + geoms["toes"])

        def x_ext(gi):
            R, p, sz, t = d0.geom_xmat[gi].reshape(3, 3), d0.geom_xpos[gi], m0.geom_size[gi], m0.geom_type[gi]
            if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                a = abs(R[0, 2]) * sz[1] + sz[0]
            elif t in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                a = float(np.sum(np.abs(R[0, :]) * sz[:3]))
            else:
                a = m0.geom_rbound[gi]
            return p[0] - a, p[0] + a
        heel = min(x_ext(gi)[0] for gi in geoms["calcn"])
        tip = max(x_ext(gi)[1] for gi in geoms["toes"])
        mtp = d0.body(f"toes_{side}").xpos[0]
        yc = float(np.mean([d0.geom_xpos[gi][1] for gi in geoms["calcn"]]))
        ref = spec.geom(m0.geom(geoms["calcn"][0]).name)
        props = (list(ref.friction), list(ref.solref), list(ref.solimp), float(ref.margin))
        disable += [spec.geom(m0.geom(gi).name) for gi in geoms["calcn"] + geoms["toes"]]
        a = np.radians(subject.shoe_toe_spring_deg)
        for seg, x0, x1, w in (("calcn", heel, mtp, W), ("toes", mtp, tip, 0.95 * W)):
            body = d0.body(f"{seg}_{side}")
            R, xb = body.xmat.reshape(3, 3).copy(), body.xpos.copy()
            Rg = np.eye(3)
            c = np.array([0.5 * (x0 + x1), yc, zb + 0.5 * h])
            if seg == "toes" and a > 0:
                # toe spring: forefoot sole pitched up about its rear (MTP) bottom edge
                Rg = np.array([[np.cos(a), 0.0, -np.sin(a)], [0.0, 1.0, 0.0], [np.sin(a), 0.0, np.cos(a)]])
                c = np.array([x0, yc, zb]) + Rg @ np.array([0.5 * (x1 - x0), 0.0, 0.5 * h])
            plan.append((f"{seg}_{side}", f"{'foot' if seg == 'calcn' else 'bofoot'}_shoe_{side}",
                         [0.5 * (x1 - x0), 0.5 * w, 0.5 * h], list(R.T @ (c - xb)), list(_quat_from_mat(R.T @ Rg)),
                         props))
    for gs in disable:
        gs.contype, gs.conaffinity = 0, 0
    for body, name, size, pos, quat, (fr, sr, si, mg) in plan:
        g = spec.body(body).add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size, g.pos, g.quat = size, pos, quat
        g.contype, g.conaffinity, g.condim = 1, 0, 3
        g.friction, g.solref, g.solimp, g.margin = fr, sr, si, mg
        g.rgba = [0.12, 0.12, 0.13, 1.0]
        g.group = 1


def _geom_lowest_z(m, d, gi):
    R = d.geom_xmat[gi].reshape(3, 3)
    p = d.geom_xpos[gi]
    t = m.geom_type[gi]
    size = m.geom_size[gi]
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        a = R[:, 2] * size[1]
        return min(p[2] + a[2], p[2] - a[2]) - size[0]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return p[2] - size[0]
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return p[2] - np.sqrt(np.sum((R[2, :] * size[:3]) ** 2))
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return p[2] - np.sum(np.abs(R[2, :]) * size[:3])
    return p[2] - m.geom_rbound[gi]


def _build_exo(spec, exo: ExoParams, sim: SimConfig, sp: SensorParams, info: ModelInfo, geom: dict, s_len: float):
    m0, d0 = _compile(spec)
    dt = sim.timestep
    weld_solref = [2.0 * dt, 1.0]

    P = {}
    for s in SIDES:
        P[f"hjc_{s}"] = d0.body(f"femur_{s}").xpos.copy()
        P[f"kjc_{s}"] = d0.body(f"tibia_{s}").xpos.copy()
        P[f"ajc_{s}"] = d0.body(f"talus_{s}").xpos.copy()
        P[f"calcn_{s}"] = d0.body(f"calcn_{s}").xpos.copy()
        P[f"toes_{s}"] = d0.body(f"toes_{s}").xpos.copy()
    xp = d0.body("pelvis").xpos.copy()
    Rp = d0.body("pelvis").xmat.reshape(3, 3).copy()

    hjc_mid = 0.5 * (P["hjc_l"] + P["hjc_r"])
    O_hu = hjc_mid.copy()                          # hip-unit origin on the (nominal) hip axis
    hu = spec.body("pelvis").add_body()
    hu.name = "exo_hipunit"
    hu.pos = Rp.T @ (O_hu - xp)
    hu.quat = _quat_from_mat(Rp.T)
    # belt bushing joints: centred at the belt (~12 cm above, 3 cm behind the hip axis)
    belt_c = np.array([-0.03, 0.0, 0.12]) * s_len
    bp = exo.belt
    belt_joints = []
    for i, ax in enumerate("xyz"):
        j = hu.add_joint()
        j.name = f"exo_belt_t{ax}"
        j.type = mujoco.mjtJoint.mjJNT_SLIDE
        j.axis = np.eye(3)[i]
        j.stiffness = [bp.k_lin[i], 0.0, bp.k_lin[i] * bp.cubic / bp.range_lin ** 2]
        j.damping = [bp.c_lin[i], 0.0, 0.0]
        j.armature = bp.wobble_mass
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.range = [-bp.range_lin, bp.range_lin]
        belt_joints.append(j.name)
    for i, ax in enumerate("xyz"):
        j = hu.add_joint()
        j.name = f"exo_belt_r{ax}"
        j.type = mujoco.mjtJoint.mjJNT_HINGE
        j.axis = np.eye(3)[i]
        j.pos = belt_c
        j.stiffness = [bp.k_rot[i], 0.0, bp.k_rot[i] * bp.cubic / bp.range_rot ** 2]
        j.damping = [bp.c_rot[i], 0.0, 0.0]
        j.armature = bp.wobble_inertia
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.range = [-bp.range_rot, bp.range_rot]
        belt_joints.append(j.name)
    info.bushings["exo_belt"] = belt_joints
    info.bushing_body["exo_belt"] = "exo_hipunit"

    # hip unit visuals and mass: shell mass + battery point mass
    m_batt = 0.68
    for nm, mat, frac in [("hipunit_frame", "indego_frame", 0.75), ("hipunit_pads", "indego_soft", 0.25)]:
        _visual(hu, _add_mesh(spec, nm), mat, mass=(exo.mass_hip_unit - m_batt) * frac)
    _visual(hu, _add_mesh(spec, "hipunit_led"), "indego_led", mass=0.0)
    _point_mass(hu, np.asarray(exo.battery_pos) * s_len, m_batt, 0.04)
    imu = hu.add_site()
    imu.name = "exo_imu_hip"
    imu.pos = [-0.15 * s_len, 0, 0.12 * s_len]
    imu.size = [0.01, 0, 0]
    imu.group = 4

    # ---- legs
    L_th_mesh = geom["thigh_length_m"]
    L_sh_mesh = geom["shank_length_knee_to_afo_m"]
    plate_depth = -geom["afo_plate_z_min_in_afo"]
    for s in SIDES:
        sign = 1.0 if s == "l" else -1.0
        hjc, kjc, ajc = P[f"hjc_{s}"], P[f"kjc_{s}"], P[f"ajc_{s}"]
        hip_off = np.array([exo.hip_axis_offset[s][0], 0.0, exo.hip_axis_offset[s][1]])
        knee_off = np.array([exo.knee_axis_offset[s][0], 0.0, exo.knee_axis_offset[s][1]])
        O_th = np.array([hjc[0], O_hu[1], hjc[2]]) + hip_off     # exo hip axis point (world)
        O_sh = np.array([kjc[0], O_hu[1], kjc[2]]) + knee_off    # exo knee axis point (world)
        L_th = O_th[2] - O_sh[2]
        L_ank = O_sh[2] - ajc[2]                                  # knee axis -> anatomical ankle height
        foot_sole_z = ajc[2] - 0.062 * s_len                      # MyoLeg ankle->sole ~6.2 cm
        knee_to_sole = O_sh[2] - foot_sole_z
        s_th = L_th / L_th_mesh
        s_sh = max(0.5, (knee_to_sole - plate_depth) / L_sh_mesh)
        y_bar = sign * geom["thigh_bar_lateral_y_m"] * s_len

        # --- thigh (with hip drive)
        th_parent = hu
        th_pos_in_parent = O_th - O_hu
        thigh, motor_joint, link_joint = _add_drive(
            spec, th_parent, f"exo_thigh_{s}", f"exo_hip_{s}", th_pos_in_parent, exo, "hip", y_bar, s_len)
        for nm, mat, mass in [(f"thigh_frame_{s}", "indego_frame", exo.mass_thigh - 2 * exo.actuator_mass - 0.25),
                              (f"thigh_cuff_{s}", "indego_soft", 0.25)]:
            mesh = _add_mesh(spec, nm + f"@{s_th:.4f}", scale=(1, 1, s_th))
            _visual(thigh, mesh, mat, mass=mass)
        _point_mass(thigh, [0, y_bar, 0], exo.actuator_mass)                 # hip motor + gearing
        _point_mass(thigh, [0, y_bar, -L_th], exo.actuator_mass)             # knee motor + gearing
        _collision_capsule(thigh, [0, y_bar, -0.02], [0, y_bar, -L_th + 0.02], 0.035, f"exo_thigh_col_{s}")
        st = thigh.add_site()
        st.name = f"exo_imu_thigh_{s}"
        st.pos = [0.0, y_bar, -0.5 * L_th]
        st.size = [0.01, 0, 0]
        st.group = 4
        info.exo_joints.append(motor_joint)
        info.exo_backlash_joints.append(None if link_joint == motor_joint else link_joint)

        # thigh cuff bushing on the femur line
        zc = geom["thigh_cuff_centre_in_thigh"][2] * s_th
        t = (-zc) / max(hjc[2] - kjc[2], 1e-6)
        pc_world = hjc + t * (kjc - hjc)
        _add_bushing(spec, thigh, f"exo_thigh_cuff_{s}", pc_world - O_th, exo.thigh_cuff, info,
                     f"femur_{s}", weld_solref, interface_model=exo.interface_model)

        # --- shank (with knee drive)
        shank, kmotor_joint, klink_joint = _add_drive(
            spec, thigh, f"exo_shank_{s}", f"exo_knee_{s}", O_sh - O_th, exo, "knee", y_bar, s_len)
        for nm, mat, mass in [(f"shank_frame_{s}", "indego_frame", exo.mass_shank - 0.2),
                              (f"shank_cuff_{s}", "indego_soft", 0.2)]:
            mesh = _add_mesh(spec, nm + f"@{s_sh:.4f}", scale=(1, 1, s_sh))
            _visual(shank, mesh, mat, mass=mass)
        _collision_capsule(shank, [0, y_bar * 0.85, -0.03], [0, y_bar * 0.7, -L_ank + 0.05], 0.03, f"exo_shank_col_{s}")
        info.exo_joints.append(kmotor_joint)
        info.exo_backlash_joints.append(None if klink_joint == kmotor_joint else klink_joint)
        for tag, key, bpar in [("P", "shank_cuff_prox_centre_in_shank", exo.shank_cuff_prox),
                               ("D", "shank_cuff_dist_centre_in_shank", exo.shank_cuff_dist)]:
            zc = geom[key][2] * s_sh
            t = (-zc) / max(kjc[2] - ajc[2], 1e-6)
            pc_world = kjc + t * (ajc - kjc)
            _add_bushing(spec, shank, f"exo_shank_cuff{tag}_{s}", pc_world - O_sh, bpar, info,
                         f"tibia_{s}", weld_solref, interface_model=exo.interface_model)

        # --- AFO: passive hinge at the anatomical ankle height
        afo = shank.add_body()
        afo.name = f"exo_afo_{s}"
        O_afo = np.array([ajc[0], O_sh[1], ajc[2]])
        afo.pos = O_afo - O_sh
        ja = afo.add_joint()
        ja.name = f"exo_afo_ankle_{s}"
        ja.type = mujoco.mjtJoint.mjJNT_HINGE
        ja.axis = [0, 1, 0]
        if exo.afo_type not in ("quasi_static", "free"):
            raise ValueError(f"afo_type must be 'quasi_static' or 'free', got {exo.afo_type!r}")
        k_afo = exo.afo_stiffness if exo.afo_type == "quasi_static" else 0.0
        k3_afo = exo.afo_stiffness_cubic if exo.afo_type == "quasi_static" else 0.0
        ja.stiffness = [k_afo, 0.0, k3_afo]
        ja.damping = [exo.afo_damping if exo.afo_type == "quasi_static" else 0.2, 0.0, 0.0]
        ja.armature = 1e-4
        ja.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        ja.springref = -np.radians(exo.afo_neutral_deg)       # + = plantarflexion; neutral in dorsiflexion
        ja.range = list(np.radians(exo.afo_range_deg))         # anatomical travel only (soft end-stops)
        ja.solref_limit = [0.02, 1.0]
        # plate mesh: its strut attaches to the shank at A (in shank frame z = -L_sh_mesh*s_sh)
        A_shank = np.array([geom["afo_attach_in_shank_frame"][0], 0.0, -L_sh_mesh * s_sh])
        _visual(afo, _add_mesh(spec, f"afo_plate_{s}"), "indego_carbon", mass=exo.mass_afo,
                offset=A_shank - (O_afo - O_sh))
        # shoe/foot bushing at the foot centre (between heel and forefoot), on the sole
        foot_c = 0.5 * (P[f"calcn_{s}"] + P[f"toes_{s}"])
        foot_c[2] = foot_sole_z + 0.02 * s_len
        _add_bushing(spec, afo, f"exo_shoe_{s}", foot_c - O_afo, exo.shoe, info, f"calcn_{s}",
                     weld_solref, mass=0.05, interface_model=exo.interface_model)
        info.geometry[f"thigh_scale_{s}"] = s_th
        info.geometry[f"shank_scale_{s}"] = s_sh

    # ---- order: hip_l, knee_l, hip_r, knee_r
    # ---- actuators (MuJoCo dcmotor, torque-mode driver with back-EMF compensation)
    for jn in info.exo_joints:
        mp = exo.hip_motor if "hip" in jn else exo.knee_motor
        a = _add_dcmotor(spec, f"act_{jn}", jn, mp, mp.gear, dt)
        info.exo_actuators.append(a.name)
        info.gears.append(mp.gear)

    # ---- onboard sensors (sampled at the control rate, delayed)
    per = 1.0 / exo.control_rate_hz
    dl = sp.sensor_delay
    T = mujoco.mjtSensor
    O = mujoco.mjtObj
    for jm, jb, an in zip(info.exo_joints, info.exo_backlash_joints, info.exo_actuators):
        info.exo_sensors.append(_sensor(spec, f"enc_{jm}", T.mjSENS_JOINTPOS, O.mjOBJ_JOINT, jm, per, dl, dt).name)
        info.exo_sensors.append(_sensor(spec, f"vel_{jm}", T.mjSENS_JOINTVEL, O.mjOBJ_JOINT, jm, per, dl, dt).name)
        if jb is not None:  # output-side encoder sees motor angle + dead-band deflection
            info.exo_sensors.append(_sensor(spec, f"encbl_{jm}", T.mjSENS_JOINTPOS, O.mjOBJ_JOINT, jb, per, dl, dt).name)
            info.exo_sensors.append(_sensor(spec, f"velbl_{jm}", T.mjSENS_JOINTVEL, O.mjOBJ_JOINT, jb, per, dl, dt).name)
        info.exo_sensors.append(_sensor(spec, f"trq_{jm}", T.mjSENS_ACTUATORFRC, O.mjOBJ_ACTUATOR, an, per, dl, dt).name)
    for site in ["exo_imu_thigh_l", "exo_imu_thigh_r", "exo_imu_hip"]:
        info.exo_sensors.append(_sensor(spec, f"acc_{site}", T.mjSENS_ACCELEROMETER, O.mjOBJ_SITE, site, per, dl, dt).name)
        info.exo_sensors.append(_sensor(spec, f"gyr_{site}", T.mjSENS_GYRO, O.mjOBJ_SITE, site, per, dl, dt).name)
    info.exo_sensors.append(_sensor(spec, "quat_exo_imu_hip", T.mjSENS_FRAMEQUAT, O.mjOBJ_SITE, "exo_imu_hip", per, dl, dt).name)

    # ---- optional upper-trunk IMU (strapped to the person, read by the exo controller)
    if sp.trunk_imu != "none":
        _add_trunk_imu_site(spec, m0, d0, sp)
        per_t, dl_t = 1.0 / sp.trunk_imu_rate_hz, sp.trunk_imu_delay
        for nm, st in (("acc", T.mjSENS_ACCELEROMETER), ("gyr", T.mjSENS_GYRO), ("quat", T.mjSENS_FRAMEQUAT)):
            info.exo_sensors.append(_sensor(spec, f"{nm}_trunk_imu", st, O.mjOBJ_SITE, "trunk_imu", per_t, dl_t, dt).name)


def _add_trunk_imu_site(spec, m0, d0, sp: SensorParams):
    """Place the add-on trunk IMU on the scaled torso: on the skin over mid-sternum (T5-T6 level) or between
    the shoulder blades (T3-T5 spinous processes), found from the torso meshes, +1-2 cm for strap/housing.
    Its axes match the hip-unit IMU in the reference posture (x forward, y left, z up), optionally pitched
    by a mounting error."""
    tid = m0.body("torso").id
    R = d0.xmat[tid].reshape(3, 3)
    x0 = d0.xpos[tid]
    up, fwd = R.T @ np.array([0.0, 0.0, 1.0]), R.T @ np.array([1.0, 0.0, 0.0])   # world axes in torso frame
    verts, rib = [], None
    for g in range(m0.ngeom):
        if m0.geom_bodyid[g] != tid or m0.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m0.geom_dataid[g]
        V = m0.mesh_vert[m0.mesh_vertadr[mid]:m0.mesh_vertadr[mid] + m0.mesh_vertnum[mid]]
        Vb = V @ d0.geom_xmat[g].reshape(3, 3).T + d0.geom_xpos[g]               # world
        Vb = (Vb - x0) @ R                                                         # torso frame
        verts.append(Vb)
        if rib is None or len(Vb) > len(rib):
            rib = Vb                                                               # rib cage = largest mesh
    h, f = rib @ up, rib @ fwd
    lat = rib - np.outer(h, up) - np.outer(f, fwd)
    mid_line = np.linalg.norm(lat, axis=1) < 0.02 * max(np.ptp(h) / 0.32, 0.5)   # 2 cm band at nominal size
    hmin, hmax = h.min(), h.max()
    if sp.trunk_imu == "sternum":
        band = mid_line & (h > hmin + 0.50 * (hmax - hmin)) & (h < hmin + 0.70 * (hmax - hmin))
        k = np.argmax(np.where(band, f, -np.inf))
        pos = rib[k] + 0.015 * fwd
    elif sp.trunk_imu == "back":
        allv = np.vstack(verts)
        ha, fa = allv @ up, allv @ fwd
        la = np.linalg.norm(allv - np.outer(ha, up) - np.outer(fa, fwd), axis=1)
        band = (la < 0.03) & (ha > hmin + 0.65 * (hmax - hmin)) & (ha < hmin + 0.85 * (hmax - hmin))
        k = np.argmin(np.where(band, fa, np.inf))
        pos = allv[k] - 0.025 * fwd                                                # skin + housing
    else:
        raise ValueError(f"unknown trunk_imu location {sp.trunk_imu!r}")
    site = spec.body("torso").add_site()
    site.name = "trunk_imu"
    site.pos = list(pos)
    a = np.radians(sp.trunk_imu_mount_pitch_deg)
    Rmount = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    site.quat = _quat_from_mat(R.T @ Rmount)
    site.size = [0.012, 0, 0]
    site.group = 4


def _add_drive(spec, parent, body_name, joint_name, pos, exo: ExoParams, kind, y_bar, s_len):
    """Add an actuated exo segment. Returns (link body, motor joint name, link joint name).

    rigid:    parent -> link [compliance joints] + hinge(joint_name)
    backlash: parent -> motor body [hinge joint_name_motor, actuated] -> link [deadband hinge joint_name]
    """
    rng = np.radians(exo.hip_range_deg if kind == "hip" else exo.knee_range_deg)
    sign = 1.0 if body_name.endswith("_l") else -1.0
    # + = flexion for both: hip flexion swings the thigh forward = rotation about -y;
    # knee flexion swings the shank backward = rotation about +y
    axis = np.array([0.0, -1.0, 0.0]) if kind == "hip" else np.array([0.0, 1.0, 0.0])
    tr = exo.transmission

    def add_compliance(b):
        if kind != "hip":
            return
        ja = b.add_joint()
        ja.name = f"{joint_name}_abd"
        ja.type = mujoco.mjtJoint.mjJNT_HINGE
        ja.axis = [sign * 1.0, 0.0, 0.0]  # + = abduction for both sides
        ja.pos = [0.0, y_bar, 0.10 * s_len]
        ja.stiffness = [exo.hip_abd_stiffness, 0.0, 0.0]
        ja.damping = [exo.hip_abd_damping, 0.0, 0.0]
        ja.armature = 0.002
        ja.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        ja.range = [-np.radians(exo.hip_abd_range_deg), np.radians(exo.hip_abd_range_deg)]
        jr = b.add_joint()
        jr.name = f"{joint_name}_twist"
        jr.type = mujoco.mjtJoint.mjJNT_HINGE
        jr.axis = [0.0, 0.0, -sign * 1.0]  # + = internal rotation
        jr.pos = [0.0, y_bar, 0.0]
        jr.stiffness = [exo.hip_rot_stiffness, 0.0, 0.0]
        jr.damping = [exo.hip_rot_damping, 0.0, 0.0]
        jr.armature = 0.002
        jr.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        jr.range = [-0.25, 0.25]

    mp = exo.hip_motor if kind == "hip" else exo.knee_motor
    joint_friction = mp.coulomb * mp.gear if mp.friction_model == "coulomb" else 0.0

    def hinge(b, name, rng_, frictionloss=0.0):
        j = b.add_joint()
        j.name = name
        j.type = mujoco.mjtJoint.mjJNT_HINGE
        j.axis = axis
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.range = list(rng_)
        j.solref_limit = [0.004, 1.0]     # mechanical hard stops
        j.frictionloss = frictionloss
        return j

    if tr.mode == "rigid":
        link = parent.add_body()
        link.name = body_name
        link.pos = list(pos)
        add_compliance(link)
        hinge(link, joint_name, rng, joint_friction)
        return link, joint_name, joint_name
    elif tr.mode == "backlash":
        mb = parent.add_body()
        mb.name = body_name + "_drive"
        mb.pos = list(pos)
        add_compliance(mb)
        jm = hinge(mb, joint_name, rng, joint_friction)
        g = mb.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [0.01, 0, 0]
        g.mass = 0.02
        g.contype = 0
        g.conaffinity = 0
        g.group = 4
        link = mb.add_body()
        link.name = body_name
        jl = link.add_joint()
        jl.name = joint_name + "_backlash"
        jl.type = mujoco.mjtJoint.mjJNT_HINGE
        jl.axis = axis
        half = 0.5 * np.radians(tr.backlash_deg)
        jl.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        jl.range = [-half, half]
        # tooth contact: direct stiffness/damping via negative solref (converted to the
        # acceleration-level values MuJoCo expects in build_model's second pass)
        jl.solref_limit = [-tr.stiffness, -tr.damping]
        jl.damping = [0.01, 0.0, 0.0]
        return link, jm.name, jl.name
    raise ValueError(tr.mode)
