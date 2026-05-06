from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


SIM2REAL_SCRIPTS = Path("/home/user/rl_ws/sim2real/scripts")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SIM2REAL_SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_demo_recorder_contract_is_importable_without_ros2() -> None:
    recorder = _load_script("ros2_demo_recorder")

    assert recorder.RIGHT_ARM_TOPIC == "/isaacsim/right_arm_cmd"
    assert recorder.RIGHT_HAND_TOPIC == "/isaacsim/right_hand_cmd"
    assert recorder.LEFT_ARM_TOPIC == "/isaacsim/left_arm_cmd"
    assert recorder.LEFT_GRIPPER_TOPIC == "/isaacsim/left_gripper_cmd"
    assert recorder.CURL_JOINT_IDX == [1, 5, 9, 13, 18]


def test_demo_recorder_builds_18d_action_from_joint_targets() -> None:
    recorder = _load_script("ros2_demo_recorder")

    current_pose = np.array([0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0])
    target_pose = np.array([0.11, 0.18, 0.33, 0.0, 0.0, 0.0, 1.0])

    def fk(joints: np.ndarray) -> np.ndarray:
        return current_pose if np.allclose(joints, np.zeros(7)) else target_pose

    right_hand = np.zeros(20)
    right_hand[recorder.CURL_JOINT_IDX] = recorder.CURL_MAX

    action = recorder.build_pour_mimic_action(
        current_right_arm=np.zeros(7),
        target_right_arm=np.ones(7) * 0.1,
        target_right_hand=right_hand,
        current_left_arm=np.zeros(7),
        target_left_arm=np.ones(7) * 0.2,
        right_palm_fk=fk,
    )

    assert action.shape == (18,)
    np.testing.assert_allclose(action[:3], np.array([0.01, -0.02, 0.03]) / 0.30, atol=1e-7)
    np.testing.assert_allclose(action[3:6], [0.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(action[6:11], np.ones(5), atol=1e-7)
    np.testing.assert_allclose(action[11:18], np.ones(7) * 2.0, atol=1e-7)


def test_demo_recorder_rejects_bad_command_dimensions() -> None:
    recorder = _load_script("ros2_demo_recorder")

    try:
        recorder.TeleopCommandState(
            right_arm=np.zeros(6),
            right_hand=np.zeros(20),
            left_arm=np.zeros(7),
            left_gripper=0.0,
        )
    except ValueError as exc:
        assert "right_arm expects 7 values" in str(exc)
    else:
        raise AssertionError("TeleopCommandState accepted a malformed right arm command")


def test_demo_recorder_writes_mimic_hdf5_minimum_structure(tmp_path) -> None:
    recorder = _load_script("ros2_demo_recorder")
    output = tmp_path / "pour_one_demo.hdf5"

    episode = recorder.EpisodeBuffer()
    episode.set_initial_state(
        {
            "articulation": {
                "robot": {
                    "joint_pos": np.zeros((1, 7), dtype=np.float32),
                },
            },
        }
    )
    episode.append(
        obs=np.zeros(91, dtype=np.float32),
        action=np.ones(18, dtype=np.float32),
        reward=0.0,
        done=False,
    )

    writer = recorder.HDF5DemoWriter(output, env_name="Pour-Mimic-V1-Mimic-v0")
    writer.set_env_args({"dt": 1.0 / 300.0, "decimation": 5, "num_envs": 1})
    writer.write_episode(episode, success=True)
    writer.close()

    with h5py.File(output, "r") as handle:
        assert "data" in handle
        assert "Pour-Mimic-V1-Mimic-v0" in handle["data"].attrs["env_args"]
        demo_name = next(iter(handle["data"].keys()))
        demo = handle["data"][demo_name]
        assert "initial_state" in demo
        assert demo["actions"].shape == (1, 18)


def test_ros2_teleop_device_returns_isaaclab_se3_tuple() -> None:
    device_mod = _load_script("ros2_teleop_device")

    device = device_mod.Se3ROS2Device(use_ros=False)
    device.update_command(
        delta_pos=[0.01, -0.02, 0.03],
        delta_rot=[0.1, 0.0, -0.1],
        gripper=0.5,
        reset=True,
    )

    delta_pos, delta_rot, gripper, reset = device.advance()
    np.testing.assert_allclose(delta_pos, [0.01, -0.02, 0.03])
    np.testing.assert_allclose(delta_rot, [0.1, 0.0, -0.1])
    assert gripper == 0.5
    assert reset is True
