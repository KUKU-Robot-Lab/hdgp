from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 모듈 동적 로드 (Isaac 의존 없음)
# ---------------------------------------------------------------------------
_TASK_DIR = Path(__file__).resolve().parents[1]
_LOADER_PATH = _TASK_DIR / "il_demo_loader.py"
_DEMO_A11 = Path("/home/user/rl_ws/datasets/pour_v1_a11.hdf5")
_DEMO_A12 = Path("/home/user/rl_ws/datasets/pour_v1_a12.hdf5")

spec = importlib.util.spec_from_file_location("il_demo_loader", _LOADER_PATH)
il_demo_loader = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = il_demo_loader
spec.loader.exec_module(il_demo_loader)

load_il_trajectory = il_demo_loader.load_il_trajectory
ILTrajectory = il_demo_loader.ILTrajectory


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def traj_a11():
    return load_il_trajectory(_DEMO_A11)


def test_arm_joint_shape(traj_a11: ILTrajectory) -> None:
    """arm_joints shape = (T, 7)."""
    assert traj_a11.arm_joints.ndim == 2
    assert traj_a11.arm_joints.shape[1] == 7
    T = traj_a11.arm_joints.shape[0]
    assert T > 100


def test_finger_curl_shape(traj_a11: ILTrajectory) -> None:
    """finger_curl shape = (T, 5) — tesollo 5 fingers."""
    assert traj_a11.finger_curl.ndim == 2
    assert traj_a11.finger_curl.shape[1] == 5
    assert traj_a11.finger_curl.shape[0] == traj_a11.arm_joints.shape[0]


def test_finger_curl_is_not_all_ones(traj_a11: ILTrajectory) -> None:
    """실제 right_hand_curl을 로드 — buffer.py의 hardcoded 1.0 버그 재현 안 됨."""
    assert not np.allclose(traj_a11.finger_curl, 1.0), (
        "finger_curl이 모두 1.0 → buffer.py의 np.ones 버그가 복사된 것"
    )


def test_finger_shows_initial_open_state(traj_a11: ILTrajectory) -> None:
    """t=0에서 손가락 3-4번(index 2, 3)이 open (0에 가까운) 상태."""
    # right_hand_curl t=0: [-0.517, 0.0, 0.0, 0.0, 0.0]
    # finger 1-4 (index 1-4)는 0에 가까움
    t0_fingers = traj_a11.finger_curl[0, 1:]  # finger 1-4
    assert np.all(np.abs(t0_fingers) < 0.5), (
        f"t=0에서 손가락이 open 상태여야 함, got: {t0_fingers}"
    )


def test_finger_shows_grasped_state_at_end(traj_a11: ILTrajectory) -> None:
    """에피소드 후반 손가락 1-4번이 0.5 이상 grasped 상태."""
    T = traj_a11.finger_curl.shape[0]
    last_quarter = traj_a11.finger_curl[3 * T // 4 :, 1:]  # finger 1-4
    assert np.mean(last_quarter) > 0.5, (
        f"에피소드 후반 finger curl 평균이 0.5 미만: {np.mean(last_quarter):.3f}"
    )


def test_length_consistent(traj_a11: ILTrajectory) -> None:
    """arm_joints, finger_curl, pour_mask 길이가 동일."""
    T = traj_a11.arm_joints.shape[0]
    assert traj_a11.finger_curl.shape[0] == T
    assert traj_a11.pour_mask.shape[0] == T


def test_pour_mask_is_bool(traj_a11: ILTrajectory) -> None:
    """pour_mask dtype = bool."""
    assert traj_a11.pour_mask.dtype == bool


def test_hand_joints_shape(traj_a11: ILTrajectory) -> None:
    """hand_joints shape = (T, 20) — tesollo 20 DOF."""
    assert traj_a11.hand_joints.ndim == 2
    assert traj_a11.hand_joints.shape[1] == 20
    assert traj_a11.hand_joints.shape[0] == traj_a11.arm_joints.shape[0]


def test_load_multiple_demos() -> None:
    """여러 demo 파일을 각각 독립적으로 로드 가능."""
    if not _DEMO_A12.exists():
        pytest.skip("pour_v1_a12.hdf5 없음")
    traj_a12 = load_il_trajectory(_DEMO_A12)
    assert traj_a12.arm_joints.shape[1] == 7
    assert traj_a12.finger_curl.shape[1] == 5
    assert traj_a12.hand_joints.shape[1] == 20
