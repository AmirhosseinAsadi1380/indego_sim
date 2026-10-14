"""Fast regression tests (pytest -q). The slow closed-loop checks are marked `slow`."""
import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indego_sim.config import ExoParams, MotorParams, SensorParams, SupportParams      # noqa: E402
from indego_sim.human.muscles import MuscleModel                                         # noqa: E402
from indego_sim.human.sci import get_preset                                              # noqa: E402
from indego_sim.model.builder import build_model                                         # noqa: E402


@pytest.fixture(scope="module")
def model():
    return build_model(subject=get_preset("sci_t10_aisA"))


def test_model_builds_with_expected_masses(model):
    m, info = model
    assert m.nu > 80 and len(info.exo_actuators) == 4
    assert 11.0 < info.exo_mass < 14.0                       # Indego ~12 kg + AFOs
    assert 60.0 < info.human_mass < 90.0


def test_standing_keyframe_feet_on_floor(model):
    m, info = model
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)
    for s in "lr":
        assert abs(d.xpos[m.body(f"calcn_{s}").id][2]) < 0.08


def test_drive_torque_and_current(model):
    from indego_sim.exo.drive import ExoDrive
    m, info = model
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    drv = ExoDrive(m, info, ExoParams())
    drv.command(d, np.array([10.0, 0, 0, 0]))
    for _ in range(40):                                       # 20 ms > command delay + current rise
        mujoco.mj_step(m, d)
    mp = MotorParams()
    i = drv.current(d)[0]
    assert abs(i - 10.0 / (mp.K * mp.gear)) < 0.5
    # peak torque limit
    tau = drv.command(d, np.array([500.0, 0, 0, 0]))
    assert abs(tau[0] - mp.joint_peak_torque()) < 1e-6


def test_muscle_gain_bias_matches_mujoco(model):
    m, info = model
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)
    mm = MuscleModel(m, info.muscles)
    g, b = mm.gain_bias(d)
    g2, b2 = mm.gain_bias_reference(d)
    assert np.allclose(g, g2, rtol=1e-6, atol=1e-6) and np.allclose(b, b2, rtol=1e-6, atol=1e-6)


def test_rom_fraction_in_unit_range(model):
    m, info = model
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)
    p = MuscleModel(m, info.muscles).rom_fraction(d)
    assert np.all(p > -0.05) and np.all(p < 1.05)


def test_sensor_latency_and_quantisation(model):
    from indego_sim.exo.sensors import ExoSensors
    m, info = model
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    sp = SensorParams(enc_noise_deg=0.0)
    sens = ExoSensors(m, info, sp, np.random.default_rng(0), 0.005)
    mujoco.mj_forward(m, d)
    obs = sens.read(d, np.full(4, MotorParams().K))
    res = 2 * np.pi / 2 ** sp.enc_bits
    assert np.allclose(np.round(obs["q"] / res) * res, obs["q"])


def test_actuator_identification_recovers_bench_parameters():
    from indego_sim.sysid import actuator_id as aid, bench, excitation as ex
    fs = 200.0
    mp = MotorParams(coulomb=0.075, viscous=5e-4)
    m, d = bench.make_bench(mp, bench.LinkParams(gravity=False))
    ref = ex.triangle_velocity(fs, [0.1, 0.2, 0.4, 0.8, 1.2], amplitude=0.6, center=0.0)
    r = bench.simulate(m, d, ref, mp, fs, mode="position")
    pts = aid.steady_state_points(r.t, np.gradient(ref) * fs, r.current * mp.K * mp.gear)
    fr = aid.friction_map(pts[:, 0], pts[:, 1])
    assert abs(fr.coulomb - 1.8) < 0.15 and abs(fr.viscous - 0.288) < 0.08
    link = bench.LinkParams()
    m, d = bench.make_bench(mp, link)
    u, _ = ex.multisine(fs, 30.0, 0.3, 10.0, 40, amp_rms=2.5, periods=3)
    r = bench.simulate(m, d, u, mp, fs)
    rb = aid.fit_rigid_body(r.t, r.q, r.current * mp.K * mp.gear, fs, friction=fr)
    tr = bench.true_parameters(mp, link)
    assert abs(rb.inertia / tr["inertia"] - 1) < 0.08 and abs(rb.mgl / tr["mgl"] - 1) < 0.08


def test_pendulum_spasticity_ordering():
    from indego_sim.sysid.spasticity import pendulum_test
    r_h = pendulum_test(get_preset("healthy"), T=4.0).metrics["R2"]
    r_s = pendulum_test(get_preset("sci_t5_aisA"), T=4.0).metrics["R2"]
    assert r_h > r_s + 0.2


def test_trunk_imu_on_sternum_and_in_observation(model):
    from indego_sim.exo.sensors import ExoSensors
    m, info = model
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)
    site = m.site("trunk_imu")
    assert m.site_bodyid[site.id] == m.body("torso").id
    # on the front of the chest: well ahead of the thoracic spine, at mid-sternum height
    assert d.site_xpos[site.id][0] - d.xpos[m.body("torso").id][0] > 0.10
    assert 1.15 < d.site_xpos[site.id][2] < 1.45
    sens = ExoSensors(m, info, SensorParams(), np.random.default_rng(0), 0.005)
    obs = sens.read(d, np.full(4, MotorParams().K))
    assert {"acc_trunk", "gyr_trunk", "trunk_quat"} <= set(obs)
    assert ExoSensors.flatten(obs).shape == (44,) and sens.n_flat == 44
    m2, info2 = build_model(subject=get_preset("sci_t10_aisA"), sensors=SensorParams(trunk_imu="none"))
    assert mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_SENSOR, "quat_trunk_imu") < 0
    sens2 = ExoSensors(m2, info2, SensorParams(trunk_imu="none"), np.random.default_rng(0), 0.005)
    assert ExoSensors.flatten(sens2.read(mujoco.MjData(m2), np.full(4, MotorParams().K))).shape == (34,)


def test_shoe_soles_replace_narrow_foot_contacts(model):
    m, info = model
    for s in "lr":
        g = m.geom(f"foot_shoe_{s}").id
        assert m.geom_contype[g] == 1 and 2 * m.geom_size[g][1] > 0.085      # ~9.5 cm wide sole
        assert m.geom_contype[m.geom(f"foot_col1_{s}").id] == 0              # bare-foot capsules disabled


def _obs_from_state(m, d, info):
    """Noise-free, undelayed onboard signals straight from the state (for estimator tests)."""
    q = np.array([d.qpos[m.jnt_qposadr[m.joint(j).id]] for j in info.exo_joints])
    quat = {}
    for k, site in (("hip_quat", "exo_imu_hip"), ("trunk_quat", "trunk_imu")):
        qt = np.zeros(4)
        mujoco.mju_mat2Quat(qt, d.site_xmat[m.site(site).id])
        quat[k] = qt
    return dict(q=q, qd=np.zeros(4), **quat)


def test_trunk_imu_com_estimate_sees_trunk_flexion(model):
    from indego_sim.controllers.indego_fsm import IndegoFSMController
    m, info = model
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)

    def truth():
        ax = [d.xpos[m.body(f"talus_{s}").id][0] for s in "lr"]
        return d.subtree_com[0][0] - 0.5 * (ax[0] + ax[1])
    with_imu, stock = IndegoFSMController.from_model(m), IndegoFSMController.from_model(m, use_trunk_imu=False)
    obs = _obs_from_state(m, d, info)
    for c in (with_imu, stock):
        e = c.estimate(obs)
        c.trunk_ref = np.array([e["phi"], e.get("theta", 0.0)])
        assert abs(e["com_rel_mid"] - truth()) < 0.01                      # quiet standing
    d.qpos[m.jnt_qposadr[m.joint("lumbar_flexion").id]] += 0.3            # bend forward at the waist (~17 deg)
    mujoco.mj_forward(m, d)
    obs = _obs_from_state(m, d, info)
    err_imu = abs(with_imu.estimate(obs)["com_rel_mid"] - truth())
    err_stock = abs(stock.estimate(obs)["com_rel_mid"] - truth())
    assert err_imu < 0.01 and err_stock > 2 * err_imu


@pytest.mark.slow
def test_sci_quiet_standing_with_crutches(model):
    from indego_sim.controllers.indego_fsm import IndegoFSMController
    from indego_sim.sim import HumanExoSim
    sim = HumanExoSim(subject=get_preset("sci_t10_aisA"), support=SupportParams(kind="crutches"), schedule=[],
                      model=model)
    sim.reset(settle=1.0)
    log = sim.run(IndegoFSMController(), 5.0)
    assert not sim.fell
    assert log["com"][:, 0].std() < 0.02


@pytest.mark.slow
def test_planarizer_holds_pelvis_in_sagittal_plane(model):
    from indego_sim.config import PlanarizerParams
    from indego_sim.controllers.indego_fsm import IndegoFSMController
    from indego_sim.sim import HumanExoSim
    sup = SupportParams(kind="crutches", planarizer=PlanarizerParams(enabled=True))
    sim = HumanExoSim(subject=get_preset("sci_t10_aisA"), support=sup, schedule=[], model=model)
    sim.reset(settle=0.5)
    pel = sim.m.body("pelvis").id
    y0 = sim.d.xipos[pel][1]

    def push(s, d):                                   # 150 N sideways on the trunk for 0.3 s
        if 0.2 < d.time - s.t0 < 0.5:
            d.xfrc_applied[s.torso, 1] += 150.0
    sim.callbacks.append(push)
    ctrl, obs, worst_y, worst_roll = IndegoFSMController(), sim.observe(), 0.0, 0.0
    for _ in range(200):                              # 1 s
        obs = sim.step(ctrl(obs))
        worst_y = max(worst_y, abs(sim.d.xipos[pel][1] - y0))
        lateral_axis = sim.d.xmat[pel].reshape(3, 3) @ sim.planarizer.ey_body       # world y at reset
        worst_roll = max(worst_roll, abs(np.arcsin(lateral_axis[2])))   # pelvis roll (the trunk may bend sideways)
    assert not sim.fell
    assert worst_y < 0.005 and worst_roll < np.radians(1.0)
    assert sim.planarizer.peak > 100.0                # it carried the push
    assert sim.planarizer.F[0] == 0.0 and sim.planarizer.F[2] == 0.0     # forward / vertical stay free


def test_floor_is_flat_and_unbounded(model):
    """MyoSuite's 1.05 m-radius pedestal is replaced by an unbounded floor at its height (walking must not step off)."""
    m, _ = model
    world = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and m.geom_contype[g]]
    assert [m.geom_type[g] for g in world] == [mujoco.mjtGeom.mjGEOM_PLANE]
    assert abs(m.geom_pos[world[0]][2] - 0.015) < 1e-6


@pytest.mark.slow
def test_person_walks_on_their_own_in_the_exo():
    """gait_mode "walk" with its tuned defaults: the incomplete-SCI preset T12 AIS D with a 30 ms reaction delay walks
    in the zero-torque exo (sagittal plane) - several steps forward, no fall."""
    from indego_sim.config import HumanWalkParams, SimConfig, VolitionParams
    from indego_sim.sim import HumanExoSim
    assert HumanWalkParams.for_subject("healthy").theta0 != HumanWalkParams().theta0      # per-preset tuned sets
    subject = get_preset("sci_t12_aisD")
    subject.long_loop_delay = 0.03
    sim = HumanExoSim(subject=subject, support=SupportParams(kind="none"), volition=VolitionParams(gait_mode="walk"),
                      sim=SimConfig(planar=True), schedule=[(0.5, "walk")], seed=0, log_muscles=False)
    obs = sim.reset(settle=1.5)
    x0 = sim.d.subtree_com[sim.root][0]
    steps, prev = 0, None
    for _ in range(int(round(6.0 / sim.ctrl_dt))):
        obs = sim.step(np.zeros(4))
        sw = sim.human.ws.get("swing") if sim.human.ws.get("on") else None
        if sw is not None and sw != prev:
            steps += prev is not None
            prev = sw
    assert not sim.fell
    assert steps >= 8 and sim.d.subtree_com[sim.root][0] - x0 > 2.5


@pytest.mark.slow
def test_env_api():
    from indego_sim.env import IndegoEnv
    env = IndegoEnv(episode_s=0.5)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (44,)
    obs, r, term, trunc, info = env.step(np.zeros(4))
    assert np.isfinite(r) and "interaction_force" in info
