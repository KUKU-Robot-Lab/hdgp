"""layout — scene constants, configuration sampling, gate config (no Isaac)."""

import json

import numpy as np

from openarm.agnostic.modules import robot_profiles
from openarm.agnostic.modules.iker import rotations
from openarm.agnostic.tasks.iker_shoe import layout


def test_first_configuration_is_pinned_to_the_seed():
    first, second = layout.sample_configs(2)
    values = (first.index, round(first.move_x, 6), round(first.move_y, 6), round(first.move_yaw_deg, 6), round(first.other_x, 6))
    assert values == (0, 0.241444, 0.134132, -4.040029, 0.27183)
    assert second.index == 1


def test_sampled_configurations_stay_inside_the_spec_ranges():
    for config in layout.sample_configs(200):
        assert layout.MOVE_X_RANGE[0] <= config.move_x <= layout.MOVE_X_RANGE[1]
        assert layout.MOVE_Y_RANGE[0] <= config.move_y <= layout.MOVE_Y_RANGE[1]
        assert layout.MOVE_YAW_DEG_RANGE[0] <= config.move_yaw_deg <= layout.MOVE_YAW_DEG_RANGE[1]
        assert layout.OTHER_X_RANGE[0] <= config.other_x <= layout.OTHER_X_RANGE[1]


def test_rack_is_checked_before_the_table():
    rack, table = layout.supports()
    assert rack.name == "rack" and rack.contains_xy(0.27, layout.OTHER_SHOE_Y)
    assert not rack.contains_xy(0.27, 0.14) and table.contains_xy(0.27, 0.14)


def test_rack_keypoints_form_the_inset_grid():
    grid = layout.rack_keypoints()
    assert grid.shape == (12, 3)
    assert np.allclose(grid[0], [0.16, -0.28, 0.325]) and np.allclose(grid[-1], [0.38, -0.07, 0.325])


def test_moving_shoe_leaves_a_hand_width_beside_the_rack():
    # The grasping fingers reach across the shoe toward the rack; a 5.2 cm slot below the taller rack wall caught the
    # hand (stage-1 training, 2026-09-14).
    meta = json.loads(layout.SHOE_META_PATH.read_text())
    hull = np.asarray(meta["objects"][layout.MOVING_SHOE]["hull_local"], dtype=float)

    def gap(config):
        position, quat = layout.shoe_start_poses(config, meta)[layout.MOVING_SHOE]
        world = hull @ rotations.quat_wxyz_to_matrix(quat).T + position
        return float(world[:, 1].min()) - layout.RACK_Y_RANGE[1]

    configs = layout.sample_configs(200)
    assert gap(configs[0]) >= 0.095
    assert min(gap(config) for config in configs) >= 0.06


def test_shoe_start_poses_rest_on_their_supports_with_the_config_yaw():
    rest = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    meta = {"objects": {name: {"rest_rotation": rest, "rest_height": 0.05} for name in (layout.MOVING_SHOE, layout.OTHER_SHOE)}}
    config = layout.SceneConfig(index=0, move_x=0.25, move_y=0.14, move_yaw_deg=90.0, other_x=0.27)
    poses = layout.shoe_start_poses(config, meta)
    move_pos, move_quat = poses[layout.MOVING_SHOE]
    other_pos, other_quat = poses[layout.OTHER_SHOE]
    assert np.allclose(move_pos, [0.25, 0.14, layout.TABLE_TOP_Z + 0.05 + layout.SPAWN_CLEARANCE_M])
    assert np.allclose(other_pos, [0.27, layout.OTHER_SHOE_Y, layout.RACK_TOP_Z + 0.05 + layout.SPAWN_CLEARANCE_M])
    assert np.allclose(other_quat, [0.5, -0.5, 0.5, -0.5])
    long_axis = rotations.quat_wxyz_to_matrix(move_quat)[:, 2]  # local z is the shoe length
    assert np.allclose(long_axis, [0.0, 1.0, 0.0], atol=1e-9)  # rest points it along +x; +90 deg yaw turns it to +y


def test_gate_config_takes_the_palm_box_from_the_robot_profile():
    profile = robot_profiles.PROFILES[layout.PROFILE_NAME]
    cfg = layout.gate_config()
    assert cfg.palm_box_min == tuple(profile.palm_box_min) and cfg.palm_box_max == tuple(profile.palm_box_max)


def test_paths_resolve_inside_the_hdgp_checkout():
    assert (layout.HDGP_ROOT / "source" / "openarm").is_dir()
    assert layout.TABLE_USD_PATH.is_file()
