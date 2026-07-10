import json

import h5py
import numpy as np
import pytest

from r2s_autotune.config import RealTrackConfig
from r2s_autotune.joint_contract import JointContractError, load_manifest
from r2s_autotune.load_real_track import load_real_track
from r2s_autotune.paths import asset_manifest

TESOLLO_MANIFEST = asset_manifest("openarm_tesollo_sensor_rl")

# teleop HDF5는 legacy 이름을 JSON 문자열 attr로 저장한다.
LEGACY_NAMES = ["rj_dg_1_2", "rj_dg_2_2", "rj_dg_3_2"]
CANONICAL_NAMES = ("r_hj_thumb_2", "r_hj_index_2", "r_hj_middle_2")


def _write(path, names, steps=50, with_velocity=True, measured_names=None):
    rng = np.random.default_rng(0)
    q_cmd = rng.normal(size=(steps, len(names))).astype(np.float32)
    q_real = q_cmd * 0.9
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("timestamps_ns", data=np.arange(steps) * 10_000_000)
        obs = demo.create_group("obs")
        for key, array, attr in (
            ("q_cmd", q_cmd, names),
            ("q_real", q_real, measured_names or names),
        ):
            dataset = obs.create_dataset(key, data=array)
            dataset.attrs["joint_names"] = json.dumps(list(attr))
        if with_velocity:
            dataset = obs.create_dataset("dq_real", data=np.zeros_like(q_real))
            dataset.attrs["joint_names"] = json.dumps(list(measured_names or names))


def _config(path, with_velocity=True):
    return RealTrackConfig(
        hdf5=path,
        demo_key="demo_0",
        command_dataset="obs/q_cmd",
        measured_dataset="obs/q_real",
        velocity_dataset="obs/dq_real" if with_velocity else None,
        dt=0.01,
    )


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(TESOLLO_MANIFEST)


def test_legacy_joint_names_are_normalized_to_canonical(tmp_path, manifest):
    path = tmp_path / "track.hdf5"
    _write(path, LEGACY_NAMES)

    track = load_real_track(_config(path), manifest)

    assert track.joint_names == CANONICAL_NAMES


def test_time_is_rebased_to_zero(tmp_path, manifest):
    path = tmp_path / "track.hdf5"
    _write(path, LEGACY_NAMES)

    track = load_real_track(_config(path), manifest)

    assert track.time[0] == 0.0
    assert track.num_steps == 50


def test_velocity_is_finite_differenced_when_absent(tmp_path, manifest):
    path = tmp_path / "track.hdf5"
    _write(path, LEGACY_NAMES, with_velocity=False)

    track = load_real_track(_config(path, with_velocity=False), manifest)

    assert track.dq_real is not None
    assert track.dq_real.shape == track.q_real.shape
    assert np.any(track.dq_real != 0.0)


def test_command_and_measured_joint_order_mismatch_is_rejected(tmp_path, manifest):
    path = tmp_path / "track.hdf5"
    _write(path, LEGACY_NAMES, measured_names=list(reversed(LEGACY_NAMES)))

    with pytest.raises(JointContractError, match="different joint order"):
        load_real_track(_config(path), manifest)


def test_missing_joint_names_attr_is_rejected(tmp_path, manifest):
    path = tmp_path / "track.hdf5"
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_group("obs").create_dataset("q_cmd", data=np.zeros((5, 3)))

    with pytest.raises(JointContractError, match="missing 'joint_names' attr"):
        load_real_track(_config(path, with_velocity=False), manifest)


def test_select_keeps_requested_joints_and_leaves_original_untouched(tmp_path, manifest):
    path = tmp_path / "track.hdf5"
    _write(path, LEGACY_NAMES)
    track = load_real_track(_config(path), manifest)

    subset = track.select(["r_hj_index_2"])

    assert subset.joint_names == ("r_hj_index_2",)
    assert subset.q_cmd.shape[1] == 1
    assert track.joint_names == CANONICAL_NAMES  # 원본 불변


def test_select_rejects_unknown_joint(tmp_path, manifest):
    path = tmp_path / "track.hdf5"
    _write(path, LEGACY_NAMES)
    track = load_real_track(_config(path), manifest)

    with pytest.raises(JointContractError, match="no such joints"):
        track.select(["r_hj_pinky_2"])
