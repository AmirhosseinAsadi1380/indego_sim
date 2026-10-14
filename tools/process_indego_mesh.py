"""Step 2 of the mesh pipeline: turn the posed TurboSquid Indego into canonical,
simulation-ready per-body meshes.

What it does
------------
1. Groups the 17 Blender objects into the 4 rigid exo bodies
   (hip unit, thigh segment, shank segment, AFO foot plate). The plate is split
   off the shank mesh by connected components.
2. Finds the joint axes from geometry: the thigh segment's lateral shell is a
   "stadium" whose two semicircular ends are centred on the hip and knee motor
   axes. The fitted thigh length is 0.424 m (Indego femur range 0.35-0.47 m).
3. Un-poses the "running" pose: the left leg (stance leg, foot flat) is rotated
   to neutral (thigh and shank vertical, plate flat), the hip unit is pitched
   back by HIP_UNIT_PITCH_DEG, then the left leg is mirrored to make the right
   leg (the source mesh is symmetric: same vertex counts per side).
4. Expresses every body in its own joint frame (x forward, y left, z up):
     hip unit : origin on the hip-axis line, mid-sagittal plane
     thigh    : origin on the hip axis (same point as hip unit origin)
     shank    : origin on the knee axis
     afo      : origin where the AFO strut meets the shank (bottom of shank)
5. Decimates for real-time viewing and writes binary STL + a geometry JSON
   (segment lengths, cuff centres, plate extents) used by the model builder.

Usage:  python tools/process_indego_mesh.py <npz_dir> <out_assets_dir>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import trimesh

try:
    import fast_simplification
except ImportError:  # decimation is optional
    fast_simplification = None

HIP_UNIT_PITCH_DEG = -20.0  # rotate hip unit top-backwards so the lumbar pad is ~vertical

GROUPS = {
    "hipunit_frame": ["body_pivot"],
    "hipunit_pads": ["body_mount_front_pivot", "body_mount_rear_pivot", "pillow_pivot"],
    "hipunit_led": ["led_pivot"],
    "thigh_frame": ["hip_L_pivot"],
    "thigh_cuff": ["hip_attachment_L001_pivot", "hip_attachment_L002_pivot"],
    "shank_frame": ["knee_L_pivot"],  # plate removed below
    "shank_cuff": ["knee_mount_L001_pivot", "knee_mount_L002_pivot"],
}
TARGET_FACES = {
    "hipunit_frame": 24000, "hipunit_pads": 8000, "hipunit_led": 600,
    "thigh_frame": 14000, "thigh_cuff": 3000,
    "shank_frame": 10000, "shank_cuff": 4000, "afo_plate": 2500,
}


def load(npz_dir, name):
    z = np.load(os.path.join(npz_dir, name + ".npz"))
    return trimesh.Trimesh(z["V"], z["F"], process=False)


def concat(meshes):
    return trimesh.util.concatenate(meshes)


def ry(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def stadium_ends(V):
    """Centres of the two semicircular ends of a stadium-shaped bar (sagittal x-z)."""
    P = V[:, [0, 2]]
    c = P.mean(0)
    _, _, W = np.linalg.svd(P - c, full_matrices=False)
    u = W[0] if W[0][1] > 0 else -W[0]
    n = np.array([-u[1], u[0]])
    s, t = (P - c) @ u, (P - c) @ n
    r = (t.max() - t.min()) / 2
    tc = (t.max() + t.min()) / 2
    top = c + (s.max() - r) * u + tc * n
    bot = c + (s.min() + r) * u + tc * n
    return top, bot, r


def pca_dir_down(V):
    P = V[:, [0, 2]]
    c = P.mean(0)
    _, _, W = np.linalg.svd(P - c, full_matrices=False)
    u = W[0] if W[0][1] < 0 else -W[0]
    return u  # unit vector pointing down the segment (x, z)


def angle_to_vertical(d):
    """R_y angle that maps sagittal direction d=(dx,dz) (pointing down) onto (0,-1)."""
    return np.arctan2(-d[0], -d[1]) * -1.0


def transform(mesh, R, p_from, p_to):
    m = mesh.copy()
    m.vertices = (np.asarray(m.vertices) - p_from) @ R.T + p_to
    return m


def mirror_y(mesh):
    m = mesh.copy()
    v = np.asarray(m.vertices).copy()
    v[:, 1] *= -1
    m.vertices = v
    m.faces = np.asarray(m.faces)[:, ::-1]
    return m


def decimate(mesh, target):
    if fast_simplification is None or len(mesh.faces) <= target:
        return mesh
    ratio = 1.0 - target / len(mesh.faces)
    v, f = fast_simplification.simplify(np.asarray(mesh.vertices, np.float32),
                                        np.asarray(mesh.faces, np.int32), target_reduction=ratio)
    return trimesh.Trimesh(v, f, process=True)


def main(npz_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    parts = {g: concat([load(npz_dir, n) for n in names]) for g, names in GROUPS.items()}

    # --- split the foot plate (largest component touching the floor) off the shank
    comps = parts["shank_frame"].split(only_watertight=False)
    plate = [c for c in comps if c.bounds[0][2] < 0.01]
    assert len(plate) == 1, "expected exactly one floor-touching component (AFO plate)"
    plate = plate[0]
    parts["shank_frame"] = concat([c for c in comps if c.bounds[0][2] >= 0.01])
    parts["afo_plate"] = plate

    # --- joint axes from the thigh bars' lateral shells (both sides agree on the hip axis)
    def shell(name):
        cs = sorted(load(npz_dir, name).split(only_watertight=False), key=lambda c: -len(c.vertices))
        return [c for c in cs if c.bounds[1][1] - c.bounds[0][1] < 0.03][0]  # thin lateral plate
    topL, botL, _ = stadium_ends(np.asarray(shell("hip_L_pivot").vertices))
    topR, _, _ = stadium_ends(np.asarray(shell("hip_R_pivot").vertices))
    H = np.array([(topL[0] + topR[0]) / 2, 0.0, (topL[1] + topR[1]) / 2])  # hip axis point (y=0)
    K = np.array([botL[0], 0.0, botL[1]])                                   # left knee axis point
    thigh_len = float(np.linalg.norm(K[[0, 2]] - H[[0, 2]]))

    # --- un-pose the left thigh: rotate about hip axis until H->K is vertical
    d_th = (K[[0, 2]] - H[[0, 2]]) / thigh_len
    # we want R_y(beta) d = (0,-1)  =>  beta = atan2(dx, -dz)
    beta = np.arctan2(d_th[0], -d_th[1])
    R_th = ry(beta)
    assert np.allclose(R_th @ np.array([d_th[0], 0, d_th[1]]), [0, 0, -1], atol=1e-9)

    # --- shank direction from the shank frame body (largest components above the cuffs)
    sh_main = sorted(parts["shank_frame"].split(only_watertight=False), key=lambda c: -len(c.vertices))
    sh_bar = [c for c in sh_main if (c.bounds[1][2] - c.bounds[0][2]) > 0.2][0]
    d_sh = pca_dir_down(np.asarray(sh_bar.vertices))
    gamma = np.arctan2(d_sh[0], -d_sh[1])
    R_sh = ry(gamma)

    # AFO attachment point: top of the plate's strut (highest plate vertices)
    pv = np.asarray(plate.vertices)
    top_strut = pv[pv[:, 2] > pv[:, 2].max() - 0.01]
    A = np.array([top_strut[:, 0].mean(), 0.0, top_strut[:, 2].mean()])
    shank_len = float(np.linalg.norm((R_sh @ (A - K))[[0, 2]]))
    A_in_shank = R_sh @ (A - K)          # A expressed in the (un-posed) shank frame

    # --- canonical per-body meshes (each in its own joint frame)
    out = {}
    Rhu = ry(np.radians(HIP_UNIT_PITCH_DEG))    # R_y(+a) pitches top forward; negative = top backwards
    for g in ["hipunit_frame", "hipunit_pads", "hipunit_led"]:
        out[g] = transform(parts[g], Rhu, H, np.zeros(3))
    for g in ["thigh_frame", "thigh_cuff"]:
        out[g + "_l"] = transform(parts[g], R_th, H, np.zeros(3))
    for g in ["shank_frame", "shank_cuff"]:
        out[g + "_l"] = transform(parts[g], R_sh, K, np.zeros(3))
    # plate keeps its world orientation (flat); origin at the strut top A
    out["afo_plate_l"] = transform(parts["afo_plate"], np.eye(3), A, np.zeros(3))

    # cuff centres in their segment frames (used for bushing locations)
    def centre(m):
        return np.asarray(m.bounding_box.centroid)
    thigh_cuff = out["thigh_cuff_l"]
    shank_cuffs = sorted(out["shank_cuff_l"].split(only_watertight=False), key=lambda c: -c.bounds[1][2])
    # the two shank cuffs are the two vertical clusters of the shank_cuff group
    sc = np.asarray(out["shank_cuff_l"].vertices)
    zmid = (sc[:, 2].max() + sc[:, 2].min()) / 2
    prox = sc[sc[:, 2] > zmid]
    dist = sc[sc[:, 2] <= zmid]

    plate_v = np.asarray(out["afo_plate_l"].vertices)
    geom = {
        "source": "TurboSquid 'Rehabilitation Exoskeleton Indego Running Pose' (.blend), un-posed",
        "hip_axis_world_in_source": H.tolist(),
        "thigh_length_m": thigh_len,
        "shank_length_knee_to_afo_m": shank_len,
        "thigh_flexion_in_source_deg": float(np.degrees(beta)),
        "shank_angle_in_source_deg": float(np.degrees(gamma)),
        "afo_attach_in_shank_frame": A_in_shank.tolist(),
        "thigh_bar_lateral_y_m": float(np.median(np.asarray(out["thigh_frame_l"].vertices)[:, 1])),
        "thigh_cuff_centre_in_thigh": centre(thigh_cuff).tolist(),
        "shank_cuff_prox_centre_in_shank": [float(prox[:, 0].mean()), float(prox[:, 1].mean()), float(prox[:, 2].mean())],
        "shank_cuff_dist_centre_in_shank": [float(dist[:, 0].mean()), float(dist[:, 1].mean()), float(dist[:, 2].mean())],
        "afo_plate_x_range_in_afo": [float(plate_v[:, 0].min()), float(plate_v[:, 0].max())],
        "afo_plate_z_min_in_afo": float(plate_v[:, 2].min()),
        "afo_plate_y_mean_in_afo": float(plate_v[plate_v[:, 2] < plate_v[:, 2].min() + 0.01][:, 1].mean()),
        "hipunit_bounds": np.asarray(out["hipunit_frame"].bounds).tolist(),
        "hip_unit_pitch_correction_deg": HIP_UNIT_PITCH_DEG,
    }

    # mirror left -> right
    for k in list(out):
        if k.endswith("_l"):
            out[k[:-2] + "_r"] = mirror_y(out[k])

    for k, m in out.items():
        base = k[:-2] if k.endswith(("_l", "_r")) else k
        md = decimate(m, TARGET_FACES.get(base, 8000))
        md.export(os.path.join(out_dir, f"indego_{k}.stl"))
        print(f"{k:18s} faces {len(m.faces):6d} -> {len(md.faces):6d}  bounds {np.round(md.bounds, 3).tolist()}")
    with open(os.path.join(out_dir, "indego_geometry.json"), "w") as f:
        json.dump(geom, f, indent=2)
    print(json.dumps(geom, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
