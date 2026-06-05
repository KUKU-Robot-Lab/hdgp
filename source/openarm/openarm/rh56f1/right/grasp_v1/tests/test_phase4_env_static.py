"""Phase 4 env.py 정적 정합성 (GPU/isaaclab 불필요).

env.py 의 Tesollo→RH56F1 마이그레이션이 구문/토큰 수준에서 일관적인지 단언.
GPU 통합검증 전에 회귀를 잡는 경량 게이트.
"""
import ast
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent / "grasp_right_env.py"


def _src():
    return ENV.read_text()


def test_syntax_valid():
    ast.parse(_src())


def test_no_tesollo_fabric_tokens():
    s = _src()
    for tok in ["OpenArmTeosllo", "open_tesollo_fabric", "open_tesollo_integrator",
                "tesollo_hand_abduction", "tesollo_hand_curl", "tesollo_hand_pip", "tesollo_hand_dip"]:
        assert tok not in s, f"잔재 토큰: {tok}"


def test_uses_rh56f1_fabric():
    s = _src()
    assert "OpenArmRh56f1PoseFabric" in s
    assert "openarm_rh56f1_pose_fabric" in s
    assert s.count("OpenArmRh56f1PoseFabric(") == 2  # main + reset fabric


def test_no_tesollo_hand_dims_4b():
    """4b: 손 제어 6D 직접. 20D 슬라이스/Tesollo 마스크 없음."""
    s = _src()
    assert "[:, 6:26]" not in s and "actions[:, 6:26]" not in s
    assert "finger * 4 + 1" not in s
    assert "6:NUM_ACTIONS" in s


def test_sensors_consolidated_4c():
    """4c: distal=tip, middle=zeros. 제거된 센서/cfg/Tesollo body 참조 없음."""
    s = _src()
    for tok in ["self._distal_sensor", "self._middle_sensor",
                "distal_sensor_cfg", "middle_sensor_cfg",
                "rl_dg_", "NUM_DISTAL_SENSORS", "NUM_MIDDLE_SENSORS"]:
        assert tok not in s, f"잔재: {tok}"


def test_obs_layout_4d():
    """4d: obs 빌더 102/117. Tesollo 144/174 주석/critic extra 제거."""
    s = _src()
    assert "144D" not in s and "174D" not in s, "stale 144/174 주석 잔재"
    assert "Actor 102D" in s and "Critic 117D" in s
    assert "tip_contact_binary" in s          # critic privileged
    assert "tip_force_xyz_norm" in s          # actor 실 fingertip 센서
    # critic 에서 distal_force_norm/middle_force_norm 제거됨 (obs 에서 미사용)
    assert "distal_force_norm" not in s
    assert "middle_force_norm =" not in s


if __name__ == "__main__":
    test_syntax_valid()
    test_no_tesollo_fabric_tokens()
    test_uses_rh56f1_fabric()
    test_no_tesollo_hand_dims_4b()
    test_sensors_consolidated_4c()
    test_obs_layout_4d()
    print("Phase 4a/4b/4c/4d env static: 6 checks passed (GREEN)")
