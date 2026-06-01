from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from prepare_pour_robomimic_dataset import build_dataset, discover_demo_paths  # noqa: E402


def _write_demo(path: Path, frames: int = 5) -> None:
    with h5py.File(path, "w") as f:
        demo = f.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.arange(frames * 18, dtype=np.float32).reshape(frames, 18))
        obs = demo.create_group("obs")
        obs.create_dataset("actor_obs", data=np.arange(frames * 91, dtype=np.float32).reshape(frames, 91))
        obs.create_dataset("right_joint_pos", data=np.ones((frames, 27), dtype=np.float32))
        obs.create_dataset("right_joint_vel", data=np.ones((frames, 27), dtype=np.float32) * 2.0)
        obs.create_dataset("left_joint_pos", data=np.ones((frames, 7), dtype=np.float32) * 3.0)
        obs.create_dataset("left_joint_vel", data=np.ones((frames, 7), dtype=np.float32) * 4.0)
        obs.create_dataset("tip_force_norm", data=np.ones((frames, 5), dtype=np.float32) * 0.5)
        obs.create_dataset("prev_actions", data=np.zeros((frames, 18), dtype=np.float32))
        obs.create_dataset("right_arm_joint_pos", data=np.ones((frames, 7), dtype=np.float32))
        obs.create_dataset("right_hand_joint_pos", data=np.ones((frames, 20), dtype=np.float32))
        obs.create_dataset("left_arm_joint_pos", data=np.ones((frames, 7), dtype=np.float32))
        obs.create_dataset("left_gripper_joint_pos", data=np.ones((frames, 2), dtype=np.float32) * 0.044)


def test_discover_demo_paths_skips_missing_a4(tmp_path: Path) -> None:
    _write_demo(tmp_path / "pour_v1_a1.hdf5")
    _write_demo(tmp_path / "pour_v1_a2.hdf5")
    _write_demo(tmp_path / "pour_v1_a5.hdf5")

    paths, missing = discover_demo_paths(tmp_path, demo_ids=(1, 2, 3, 4, 5))

    assert [p.name for p in paths] == ["pour_v1_a1.hdf5", "pour_v1_a2.hdf5", "pour_v1_a5.hdf5"]
    assert missing == [3, 4]


def test_build_dataset_writes_robot_and_fixed_object_policy_dataset(tmp_path: Path) -> None:
    _write_demo(tmp_path / "pour_v1_a1.hdf5", frames=5)
    _write_demo(tmp_path / "pour_v1_a2.hdf5", frames=4)
    out = tmp_path / "pour_bc_robomimic.hdf5"

    report = build_dataset(tmp_path, out, demo_ids=(1, 2, 4))

    assert report.kept == 2
    assert report.skipped == 0
    assert report.missing_demo_ids == [4]

    with h5py.File(out, "r") as f:
        assert f["data"].attrs["total"] == 7
        assert json.loads(f["data"].attrs["env_args"])["env_name"] == "Pour-Mimic"
        assert f["mask/train"].shape == (2,)

        demo0 = f["data/demo_0"]
        assert demo0.attrs["source_file"] == "pour_v1_a1.hdf5"
        assert demo0.attrs["num_samples"] == 4
        assert demo0["actions"].shape == (4, 18)
        assert demo0["obs/policy"].shape == (4, 105)
        assert demo0["next_obs/policy"].shape == (4, 105)
        assert demo0["obs/source_cup_pose"].shape == (4, 7)
        assert demo0["obs/target_cup_pose"].shape == (4, 7)
        assert demo0["obs/tip_force_norm"].shape == (4, 5)
        assert demo0["dones"][-1]
        assert np.allclose(demo0["obs/policy"][0, :91], np.arange(91, dtype=np.float32))
        assert np.allclose(demo0["next_obs/policy"][0, :91], np.arange(91, 182, dtype=np.float32))
        assert np.allclose(demo0["obs/policy"][0, 91:98], np.array([0.27, -0.10, 0.277, 1.0, 0.0, 0.0, 0.0]))
        assert np.allclose(demo0["obs/policy"][0, 98:105], np.array([0.27, 0.10, 0.277, 1.0, 0.0, 0.0, 0.0]))


def test_build_dataset_can_mask_left_and_right_control_scopes(tmp_path: Path) -> None:
    _write_demo(tmp_path / "pour_v1_a1.hdf5", frames=5)

    left_out = tmp_path / "left.hdf5"
    right_out = tmp_path / "right.hdf5"
    build_dataset(tmp_path, left_out, demo_ids=(1,), control_scope="left")
    build_dataset(tmp_path, right_out, demo_ids=(1,), control_scope="right")

    with h5py.File(left_out, "r") as f:
        actions = f["data/demo_0/actions"][:]
        assert np.allclose(actions[:, :11], 0.0)
        assert np.allclose(actions[:, 11:18], np.arange(5 * 18, dtype=np.float32).reshape(5, 18)[:4, 11:18])
        assert f["data/demo_0"].attrs["control_scope"] == "left"

    with h5py.File(right_out, "r") as f:
        actions = f["data/demo_0/actions"][:]
        assert np.allclose(actions[:, :11], np.arange(5 * 18, dtype=np.float32).reshape(5, 18)[:4, :11])
        assert np.allclose(actions[:, 11:18], 0.0)
        assert f["data/demo_0"].attrs["control_scope"] == "right"


def test_build_dataset_accepts_consolidated_input_file(tmp_path: Path) -> None:
    raw = tmp_path / "raw.hdf5"
    with h5py.File(raw, "w") as f:
        data = f.create_group("data")
        for idx in range(2):
            demo = data.create_group(f"demo_{idx}")
            demo.create_dataset("actions", data=np.ones((5, 18), dtype=np.float32) * idx)
            obs = demo.create_group("obs")
            obs.create_dataset("actor_obs", data=np.ones((5, 91), dtype=np.float32) * idx)
            obs.create_dataset("right_joint_pos", data=np.ones((5, 27), dtype=np.float32))
            obs.create_dataset("right_joint_vel", data=np.ones((5, 27), dtype=np.float32) * 2.0)
            obs.create_dataset("left_joint_pos", data=np.ones((5, 7), dtype=np.float32) * 3.0)
            obs.create_dataset("left_joint_vel", data=np.ones((5, 7), dtype=np.float32) * 4.0)
            obs.create_dataset("tip_force_norm", data=np.ones((5, 5), dtype=np.float32) * 0.5)
            obs.create_dataset("prev_actions", data=np.zeros((5, 18), dtype=np.float32))

    out = tmp_path / "bc.hdf5"
    report = build_dataset(tmp_path, out, input_file=raw)

    assert report.kept == 2
    assert report.missing_demo_ids == []
    with h5py.File(out, "r") as f:
        assert f["data"].attrs["num_demos"] == 2
        assert f["data"].attrs["total"] == 8
        assert f["data/demo_1"].attrs["source_file"] == "raw.hdf5"


def test_build_dataset_sorts_consolidated_demo_keys_numerically(tmp_path: Path) -> None:
    raw = tmp_path / "raw.hdf5"
    with h5py.File(raw, "w") as f:
        data = f.create_group("data")
        for name, marker in (("demo_10", 10.0), ("demo_2", 2.0), ("demo_1", 1.0)):
            demo = data.create_group(name)
            demo.create_dataset("actions", data=np.ones((5, 18), dtype=np.float32) * marker)
            obs = demo.create_group("obs")
            obs.create_dataset("actor_obs", data=np.ones((5, 91), dtype=np.float32) * marker)
            obs.create_dataset("right_joint_pos", data=np.ones((5, 27), dtype=np.float32))
            obs.create_dataset("right_joint_vel", data=np.ones((5, 27), dtype=np.float32))
            obs.create_dataset("left_joint_pos", data=np.ones((5, 7), dtype=np.float32))
            obs.create_dataset("left_joint_vel", data=np.ones((5, 7), dtype=np.float32))
            obs.create_dataset("tip_force_norm", data=np.ones((5, 5), dtype=np.float32))
            obs.create_dataset("prev_actions", data=np.zeros((5, 18), dtype=np.float32))

    out = tmp_path / "bc.hdf5"
    build_dataset(tmp_path, out, input_file=raw)

    with h5py.File(out, "r") as f:
        assert f["data/demo_0"].attrs["source_demo"] == "demo_1"
        assert f["data/demo_1"].attrs["source_demo"] == "demo_2"
        assert f["data/demo_2"].attrs["source_demo"] == "demo_10"


def test_build_dataset_rejects_dummy_object_only_demo(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "pour_v1_a1.hdf5", "w") as f:
        demo = f.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.zeros((4, 18), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("actor_obs", data=np.zeros((4, 91), dtype=np.float32))
        obs.create_dataset("datagen_info/object_pose/source_cup", data=np.zeros((4, 4, 4), dtype=np.float32))

    report = build_dataset(tmp_path, tmp_path / "out.hdf5", demo_ids=(1,))

    assert report.kept == 0
    assert report.skipped == 1
    assert report.entries[0]["reason"] == "missing_robot_sensor_obs"


def test_train_wrapper_registers_task_before_delegating() -> None:
    wrapper = _SCRIPT_DIR / "train_pour_mimic_robomimic.py"
    text = wrapper.read_text(encoding="utf-8")

    assert "pour_v1_mimic" in text
    assert "scripts/imitation_learning/robomimic/train.py" in text
    assert "Pour-Mimic" in text
    assert "hdgp/log" in text or "HDGP_LOG_DIR" in text
    assert "pour_v1_from_bags_slave_right_robot_object_bc_robomimic.hdf5" in text


def test_sim_replay_builder_resets_from_demo_first_frame_and_records_env_policy() -> None:
    script = _SCRIPT_DIR / "replay_pour_demos_to_robomimic_dataset.py"
    text = script.read_text(encoding="utf-8")

    assert "right_arm_joint_pos" in text
    assert "right_hand_joint_pos" in text
    assert "left_joint_pos" in text
    assert "write_joint_state_to_sim" in text
    assert "env.scene.reset()" in text
    assert "write_root_pose_to_sim" in text
    assert "write_root_velocity_to_sim" in text
    assert "_passive_cups_for_scope" in text
    assert "obs[\"policy\"]" in text or "obs['policy']" in text
