from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _text(filename: str) -> str:
    return (ROOT / filename).read_text(encoding="utf-8")


def _load_finger_action_utils():
    path = ROOT / "finger_action_utils.py"
    spec = importlib.util.spec_from_file_location("grasp_v2_finger_action_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_five_finger_actions_drive_absolute_twenty_joint_synergy() -> None:
    module = _load_finger_action_utils()
    open_pose = torch.arange(20, dtype=torch.float32)
    closed_pose = open_pose + 4.0
    lower = torch.full((20,), -100.0)
    upper = torch.full((20,), 100.0)
    actions = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])

    target = module.compute_absolute_finger_targets(
        finger_action=actions,
        open_pose=open_pose,
        closed_pose=closed_pose,
        lower_limits=lower,
        upper_limits=upper,
    )

    expected_blend = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]]).repeat_interleave(4, dim=1)
    assert torch.allclose(target, open_pose.unsqueeze(0) + 4.0 * expected_blend)


def test_finger_close_is_contact_gated_adaptive() -> None:
    # ① 접촉-게이트 적응 폐쇄: 손가락이 중간마디(_3) 접촉까지 점진 폐쇄 후 동결.
    # 고정 포즈 lerp(compute_lift_finger_targets) 제어를 대체.
    env = _text("grasp_right_env.py")
    preset = _text("grasp_right_preset.py")

    assert "HAND_FULL_GRIP_POSE" in preset
    assert "finger_close_buf" in env
    assert "middle_binary_contact_buf" in env
    assert "self.hand_full_grip_pose" in env
    assert "finger_close_speed" in env
    # 고정 lerp 제어 제거 확인
    assert "compute_lift_finger_targets(" not in env
    assert "self.lift_finger_pos_buf" not in env


def test_finger_close_is_per_joint_contact_gated() -> None:
    # 관절별 적응 폐쇄: PIP@middle, DIP@distal|tip 독립 동결, MCP 무게이트 full close.
    # 손가락당 1-DOF(repeat_interleave 후 lerp) 제어를 대체.
    env = _text("grasp_right_env.py")

    # 관절별 (N,20) 버퍼
    assert "self.finger_close_buf = torch.zeros(self.num_envs, NUM_HAND_DOF" in env
    # 3개 접촉 밴드 모두 게이트 입력으로 사용
    assert "self.binary_contact_buf.float()" in env
    assert "self.distal_binary_contact_buf.float()" in env
    assert "self.middle_binary_contact_buf.float()" in env
    # 관절별 게이트 스택 → (N,20)
    assert "gate20 = torch.stack" in env
    assert "cmd.repeat_interleave(4" in env
    # 1-DOF lerp(close_buf를 repeat_interleave 후 lerp) 제거 확인
    assert "self.finger_close_buf.repeat_interleave(4" not in env


def test_lift_latch_gate_disabled_envelope_via_reward() -> None:
    # 인벨롭 latch hard 게이트는 비활성(=0): success를 죽이면서 envelope은 못 만듦.
    # envelope은 grasp/lift 보상 credit(soft gradient)으로 유도한다.
    cfg = _text("grasp_right_env_cfg.py")
    utils = _text("grasp_right_utils.py")

    assert "lift_start_min_envelope_fingers: int = 0" in cfg
    # 게이트 배선은 보존(재활성 가능), compute_lift_readiness가 지원만
    assert "min_envelope_fingers" in utils


def test_reward_is_dextrah_four_terms() -> None:
    # DEXTRAH compute_rewards 이식 계약: 4항(거리 기반) + in_success_region.
    # 접촉 synergy 항(compute_grasp_reward_terms)은 사용하지 않는다.
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")
    reward_body = env.split("def _get_rewards", 1)[1].split("return total", 1)[0]

    for name in (
        "lift_goal_offset_z",
        "object_goal_tol",
        "hand_to_object_weight",
        "hand_to_object_sharpness",
        "object_to_goal_weight",
        "object_to_goal_sharpness",
        "lift_weight",
        "lift_sharpness",
        "finger_curl_reg_weight",
    ):
        assert name in cfg

    for term in (
        "hand_to_object_reward",
        "object_to_goal_reward",
        "finger_curl_reg",
        "lift_reward",
        "in_success_region",
    ):
        assert term in reward_body

    # hand_to_object: palm+5손끝 MAX 거리 (OpenArm 포팅 규약)
    assert ".max(dim=-1).values" in reward_body
    # 구 접촉 synergy 코어 미사용
    assert "compute_grasp_reward_terms(" not in reward_body


def test_goal_is_settle_anchored() -> None:
    # baseline 버그 수정 계약: settle 종료 시 안착점 스냅샷 → object_init_pos/goal 갱신.
    # goal = 안착점 + lift_goal_offset_z (spawn 상대 height_delta 폐기).
    env = _text("grasp_right_env.py")

    assert "self.episode_length_buf == int(self.cfg.settle_steps)" in env
    assert "self.object_init_pos[snap] = self.object_pos[snap]" in env
    assert "self.object_goal[snap, 2] += float(self.cfg.lift_goal_offset_z)" in env


def test_single_phase_no_scripted_lift() -> None:
    # DEXTRAH 단일 phase 계약: scripted joint7 lift/latch/freeze 없이
    # 정책이 전 구간 Fabrics arm을 연속 제어한다.
    env = _text("grasp_right_env.py")
    apply_body = env.split("def _apply_action", 1)[1].split("def ", 1)[0]

    assert "lift_progress" not in apply_body
    assert "arm_target = self.fabric_q[:, :NUM_ARM_DOF]" in apply_body
    assert "compute_lift_readiness" not in env
    assert "is_lift_phase" not in env
