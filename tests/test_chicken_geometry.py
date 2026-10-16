"""Chicken skin assets and neck geometry (pure numpy, no Isaac)."""

import xml.etree.ElementTree as ET

import numpy as np

from hybrid_control.sim_interfaces.isaac.ridgeback.chicken_geometry import (
    ASSET_DIR, CHICKEN_URDF, catmull_rom, load_chicken_visuals, read_obj, rpy_to_quat_xyzw, tube, tube_topology,
)

FRANKA_URDF = ASSET_DIR.parent / "franka" / "lula_franka_gen.urdf"


def test_urdf_reuses_franka_joints():
    chicken = {j.get("name"): j for j in ET.parse(CHICKEN_URDF).getroot().findall("joint")}
    franka = {j.get("name"): j for j in ET.parse(FRANKA_URDF).getroot().findall("joint")}
    for name in [f"panda_joint{i}" for i in range(1, 9)] + ["panda_hand_joint", "panda_finger_joint1"]:
        assert ET.tostring(chicken[name]) == ET.tostring(franka[name]), name
    mount = chicken["arm_mount_joint"]
    assert (mount.find("parent").get("link"), mount.find("child").get("link")) == ("base_link", "panda_link0")


def test_urdf_platform_is_0p4_square():
    base = next(l for l in ET.parse(CHICKEN_URDF).getroot().findall("link") if l.get("name") == "base_link")
    assert base.find("collision/geometry/box").get("size") == "0.4 0.4 0.28"


def test_visuals_load_and_meshes_are_closed():
    visuals = load_chicken_visuals()
    assert {v.link for v in visuals} == {"base_link", "panda_hand", "panda_leftfinger", "panda_rightfinger"}
    assert {"body", "head", "beak_upper", "beak_lower"} <= {v.name for v in visuals}
    for path in {v.mesh for v in visuals}:
        v, n, f = read_obj(path)
        assert len(v) == len(n) and f.max() < len(v)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)


def test_rpy_quaternion_matches_urdf_convention():
    x, y, z, w = rpy_to_quat_xyzw(0.0, np.pi / 2, 0.0)  # pitch: +x turns to -z
    np.testing.assert_allclose([x, y, z, w], [0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)], atol=1e-12)
    x, y, z, w = rpy_to_quat_xyzw(0.3, -0.2, 1.1)
    assert abs(x * x + y * y + z * z + w * w - 1.0) < 1e-12


def test_catmull_rom_passes_through_end_points():
    p = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3], [-0.2, 0.0, 0.6], [0.3, 0.0, 0.7], [0.4, 0.0, 0.75]])
    c = catmull_rom(p, 50)
    np.testing.assert_allclose(c[0], p[0], atol=1e-9)
    np.testing.assert_allclose(c[-1], p[-1], atol=1e-9)
    assert min(np.linalg.norm(c - p[2], axis=1)) < 0.02


def test_tube_rings_have_the_requested_radii_and_outward_normals():
    centre = catmull_rom(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.4, 0.0, 0.8]]), 20)
    radii = np.linspace(0.08, 0.05, 20)
    points, normals = tube(centre, radii, 12)
    dist = np.linalg.norm(points.reshape(20, 12, 3) - centre[:, None], axis=-1)
    np.testing.assert_allclose(dist, radii[:, None].repeat(12, 1), atol=1e-9)
    np.testing.assert_allclose((points - np.repeat(centre, 12, 0)) / np.repeat(radii, 12)[:, None], normals, atol=1e-9)
    counts, indices = tube_topology(20, 12)
    assert len(counts) == 19 * 12 and indices.max() == 20 * 12 - 1
    # quad winding: the face normal points away from the centre line
    a, b, c = (points[indices[k::4][0]] for k in range(3))
    face = np.cross(b - a, c - b)
    assert np.dot(face, a - centre[0]) > 0
