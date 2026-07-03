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
    """4c: distal=tip, middle=근위 sim-only ContactSensor. 제거된 센서/cfg/Tesollo body 참조 없음."""
    s = _src()
    for tok in ["self._distal_sensor",
                "distal_sensor_cfg", "middle_sensor_cfg",
                "rl_dg_", "NUM_DISTAL_SENSORS", "NUM_MIDDLE_SENSORS"]:
        assert tok not in s, f"잔재: {tok}"
    # middle(proximal) 접촉 센서는 이제 실재 — envelope 계측/critic 용.
    assert "self._middle_sensors" in s
    assert "right_middle_contact_links" in s


def test_obs_layout_4d():
    """4d: obs 빌더 96/119. Tesollo 144/174 주석/critic extra 제거."""
    s = _src()
    assert "144D" not in s and "174D" not in s, "stale 144/174 주석 잔재"
    assert "Observations: Actor 96D" in s and "Critic 119D" in s
    assert "tip_contact_binary" in s          # critic privileged
    assert "tip_force_xyz_norm" in s          # actor 실 fingertip 센서
    # critic 에서 distal_force_norm/middle_force_norm 제거됨 (obs 에서 미사용)
    assert "distal_force_norm" not in s
    assert "middle_force_norm =" not in s


def test_pregrasp_ik_targets_palm_sensor_reference():
    """Fabric IK 가 r_hl_palm_sensor 를 직접 제어 → palm_link offset 변환 제거(항등)."""
    s = _src()
    # Tesollo palm_link offset(3.4cm 오차)은 제거되어야 한다.
    assert "_PALM_SENSOR_OFFSET_IN_FABRIC_PALM" not in s
    # 함수는 항등으로 유지(호출부 시그니처 보존).
    assert "def _fabric_palm_pose_from_sensor_target" in s
    assert "palm_sensor[:, 0] = flat_x + self.cfg.pregrasp_offset_x" in s
    assert "pregrasp_sensor_pos = obj_pos_local + self.pregrasp_offset.unsqueeze(0) + noise" in s
    assert s.count("_fabric_palm_pose_from_sensor_target(") >= 3


def test_reward_has_precontact_approach_term_and_total_log():
    """RH56F1 needs dense pre-contact reward because separate phalanx contacts do not exist."""
    s = _src()
    assert "compute_grasp_phase_finger_targets(" in s
    assert "compute_late_grasp_full_grip_mask(" in s
    assert "cup_inward_xy = self.object_pos[:, :2] - self.grasp_anchor_palm_pose_buf[:, :2]" in s
    assert "float(self.cfg.grasp_palm_inward_offset)" in s
    assert "cup_center = self.object_pos" in s
    assert "compute_grasp_reward_terms(" in s
    assert 'reward_terms["approach"]' in s
    assert 'reward_terms["grasp"]' in s
    assert 'reward_terms["lift"]' in s
    assert 'reward_terms["post_lift_contact_loss"]' in s
    assert 'reward_terms["stabilize"]' in s
    assert 'reward_terms["success_bonus"]' in s
    assert "r1c_full_grasp" not in s
    assert "r1b_force_balance" not in s
    assert "r2_tip_bonus" not in s
    assert "r5_quality_lift" not in s
    assert "log_grasp_v2_common_scalars(" in s
    assert '"reward/approach"' in s
    assert '"reward/grasp"' in s
    assert '"reward/lift"' in s
    assert '"reward/post_lift_contact_loss"' in s
    assert '"reward/stabilize"' in s
    assert '"reward/success_bonus"' in s
    assert '"reward/action_smooth"' in s
    assert '"reward/stability"' in s
    assert 'self.extras["reward/full_grasp_bonus"]' not in s
    assert 'self.extras["reward/force_balance"]' not in s
    assert 'self.extras["reward/tip_approach_bonus"]' not in s
    assert 'self.extras["reward/grasp_quality_lift"]' not in s
    assert '"task/stable_rate"' in s
    assert '"task/cup_lin_vel"' in s
    assert '"task/cup_ang_vel"' in s
    assert '"task/action_delta_norm"' in s
    assert '"task/contact_delta"' in s
    assert 'self.extras["debug/rh56f1/task/prelift_force_ratio"]' in s
    assert "transport" not in s
    assert '"reward/total"' in s


if __name__ == "__main__":
    test_syntax_valid()
    test_no_tesollo_fabric_tokens()
    test_uses_rh56f1_fabric()
    test_no_tesollo_hand_dims_4b()
    test_sensors_consolidated_4c()
    test_obs_layout_4d()
    test_pregrasp_ik_targets_palm_sensor_reference()
    test_reward_has_precontact_approach_term_and_total_log()
    print("Phase 4a/4b/4c/4d env static: 8 checks passed (GREEN)")
