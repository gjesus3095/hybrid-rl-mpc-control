import numpy as np
import pytest

pytest.importorskip("torch")  # the path builder imports the scenario module, which needs torch
pytest.importorskip("scipy")

from hybrid_control.rl.evaluation.scenarios import REF_DT, SPAWN_POSES
from hybrid_control.rl.evaluation.text_path import GLYPHS, PHASES, Q_WALL, TextCfg, build_text_path, glyph_points, stroke_yz


@pytest.fixture(scope="module")
def path():
    return build_text_path()


def test_every_glyph_is_one_connected_stroke_inside_its_box():
    cfg = TextCfg()
    for char in GLYPHS:
        uv, _ = glyph_points(char)
        assert uv[:, 0].min() >= -1e-9 and uv[:, 0].max() <= 0.6 + 1e-9 and uv[:, 1].min() >= -1e-9 and uv[:, 1].max() <= 1 + 1e-9, char
        yz = stroke_yz(char, 0.0, cfg)  # the dense stroke has no gaps
        assert np.linalg.norm(np.diff(yz, axis=0), axis=1).max() < 1.5e-3, char
        assert np.ptp(yz[:, 0]) <= 0.6 * cfg.height + 1e-6 and np.ptp(yz[:, 1]) <= cfg.height + 1e-6, char


def test_path_starts_at_the_spawn_pose_and_ends_parked_still(path):
    pos, quat = SPAWN_POSES["impedance"]
    assert np.allclose(path.poses[0, :3], pos, atol=1e-5) and abs(path.poses[0, 3:] @ np.array(quat)) > 1 - 1e-6
    assert np.allclose(path.poses[-1, 3:], Q_WALL, atol=1e-5)
    assert np.allclose(path.poses[-1, :3], path.poses[-30, :3], atol=1e-6)  # held at the end
    assert 60 < path.duration < 100


def test_every_letter_is_written_once_on_the_wall_at_the_commanded_speed(path):
    cfg = path.cfg
    assert [s["char"] for s in path.strokes] == list(cfg.text)
    pos = path.poses[:, :3]
    assert np.allclose(pos[path.pen_down, 0], cfg.x_wall)  # pen down = on the wall plane
    for s in path.strokes:
        seg = pos[s["start"]:s["stop"]]
        speed = np.linalg.norm(np.diff(seg, axis=0), axis=1) / REF_DT
        assert speed.max() <= cfg.write_speed * 1.01
        assert np.median(speed) == pytest.approx(cfg.write_speed, rel=0.02)  # constant cruise speed
        assert np.all(path.letter[s["start"]:s["stop"]] == path.strokes.index(s))
        # the written points lie on the planned stroke
        d = np.linalg.norm(seg[:, None, 1:] - s["yz"][None, :, :], axis=-1).min(axis=1)
        assert d.max() < 1e-3
    assert path.pen_down.sum() == sum(s["stop"] - s["start"] for s in path.strokes)


def test_text_reads_left_to_right_from_the_robot_side_and_stays_in_the_workspace(path):
    centres = [s["yz"][:, 0].mean() for s in path.strokes]
    assert np.all(np.diff(centres) < 0)  # the robot faces +x, so reading left to right means decreasing y
    pos = path.poses[:, :3]
    assert pos[:, 2].min() > 0.45 and pos[:, 2].max() < 0.9  # letters around the shoulder height
    assert pos[:, 0].min() >= 0.45 - 1e-6 and pos[:, 0].max() <= path.cfg.x_wall + 1e-6  # in front of the robot, up to the wall
    lifted = path.phase == PHASES.index("hop")
    assert np.allclose(pos[lifted][len(pos[lifted]) // 2, 0], path.cfg.x_wall - path.cfg.lift, atol=1e-6)


def test_motion_is_smooth_with_bounded_accelerations(path):
    pos = path.poses[:, :3]
    accel = np.linalg.norm(np.diff(pos, 2, axis=0), axis=1) / REF_DT**2
    assert accel.max() < 1.5  # corners are rounded, hops and lifts are rest-to-rest quintics
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / REF_DT
    assert speed.max() < 0.6
    quat = path.poses[:, 3:]
    assert np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-4)


def test_unknown_characters_are_rejected():
    with pytest.raises(ValueError, match="no glyph"):
        build_text_path(TextCfg(text="MPCX"))
