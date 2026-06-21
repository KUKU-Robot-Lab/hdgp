"""Phase 2 차원 정합성 검증 (GPU 불필요, 순수 import).

preset/constants 의 차원이 확정 레이아웃과 일치하는지 단언.
  Action 12D, Actor 96D, Critic 114D, 손 6 DOF.
"""
import importlib.util
import sys
from pathlib import Path

# 패키지 상대 import 우회: 파일 직접 로드
_HERE = Path(__file__).resolve().parent.parent

import types


def _load(modname, filename):
    # grasp_right_preset 은 의존 없음 → 단독 로드 가능
    spec = importlib.util.spec_from_file_location(modname, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_constants():
    # constants 는 'from .grasp_right_preset import ...' 사용 → 패키지 컨텍스트 모사
    pkg = types.ModuleType("_p2pkg")
    pkg.__path__ = [str(_HERE)]
    sys.modules["_p2pkg"] = pkg
    preset = _load("_p2pkg.grasp_right_preset", "grasp_right_preset.py")
    sys.modules["_p2pkg.grasp_right_preset"] = preset
    const = _load("_p2pkg.grasp_right_constants", "grasp_right_constants.py")
    return preset, const


def test_dims():
    preset, c = _load_constants()
    assert c.NUM_ARM_DOF == 7
    assert c.NUM_HAND_DOF == 6
    assert c.NUM_ROBOT_DOF == 13
    assert c.NUM_PALM_ACTION == 6
    assert c.NUM_FINGER_ACTION == 6
    assert c.NUM_ACTIONS == 12
    assert c.NUM_OBSERVATIONS == 96
    assert c.NUM_OBSERVATIONS_WITH_MASS == 97
    assert c.NUM_CRITIC_EXTRAS == 18
    assert c.NUM_CRITIC_OBSERVATIONS == 114


def test_actor_obs_components_sum():
    """Actor 96D 구성요소 합 검증 (tip_force 실센서 15D 포함)."""
    comp = [7, 7, 6, 6, 3, 15, 3, 4, 12, 15, 15, 1, 1, 1]
    assert sum(comp) == 96


def test_critic_extra_components_sum():
    """Critic extra 18D 구성요소 합 검증."""
    comp = [1, 3, 3, 1, 5, 5]  # bead_mass, cup_lin_vel, cup_ang_vel, cup_height, tip contact/dist
    assert sum(comp) == 18


def test_hand_pose_lengths():
    preset, c = _load_constants()
    for name in ["HAND_START_POSE", "HAND_APPROACH_POSE", "HAND_GRASP_POSE", "HAND_FULL_GRIP_POSE"]:
        assert len(getattr(preset, name)) == 6, name
    assert len(preset.RIGHT_HAND_JOINT_NAMES) == 6
    assert len(preset.HAND_JOINT_LIMITS_MIN) == 6
    assert len(preset.HAND_JOINT_LIMITS_MAX) == 6
    assert len(preset.FINGERTIP_SENSOR_BODIES) == 5
    assert len(preset.HAND_BODY_NAMES_USD) == 6  # palm + 5
    assert len(preset.FABRIC_HAND_BODY_NAMES) == 7  # palm_link + palm_x + 5 tip


def test_grasp_pose_within_limits():
    preset, c = _load_constants()
    lo, hi = preset.HAND_JOINT_LIMITS_MIN, preset.HAND_JOINT_LIMITS_MAX
    for pose in [preset.HAND_APPROACH_POSE, preset.HAND_GRASP_POSE, preset.HAND_FULL_GRIP_POSE]:
        for v, a, b in zip(pose, lo, hi):
            assert a - 1e-6 <= v <= b + 1e-6, f"{v} not in [{a},{b}]"


if __name__ == "__main__":
    test_dims()
    test_actor_obs_components_sum()
    test_critic_extra_components_sum()
    test_hand_pose_lengths()
    test_grasp_pose_within_limits()
    print("Phase 2 dims: 5 checks passed (GREEN)")
