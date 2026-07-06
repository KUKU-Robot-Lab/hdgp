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


def test_envelope_credited_in_grasp_and_lift_reward() -> None:
    # envelope(중간/원위 wrap)을 grasp/lift 보상에 credit → tip-farming 차단.
    core = (
        Path(__file__).resolve().parents[4] / "common" / "grasp_reward_core.py"
    ).read_text(encoding="utf-8")
    env = _text("grasp_right_env.py")

    # 공유 코어: optional envelope_frac (RH56F1 None=기존 동작)
    assert "envelope_frac: torch.Tensor | None = None" in core
    # grasp 보상이 envelope_frac을 포함
    assert "0.40 * envelope_frac" in core
    # lift/stabilize graded_contact가 envelope-aware
    assert "0.5 * graded_contact + 0.5 * env_quality" in core
    # env: 중간·원위 마디 접촉으로 envelope_frac 계산해 전달
    assert "middle_binary_contact_buf.float().mean" in env
    assert "distal_binary_contact_buf.float().mean" in env
    assert "envelope_frac=envelope_frac" in env


def test_post_lift_and_success_are_grip_consistent() -> None:
    # envelope wrap이 tip을 mid/dist로 옮겨도 처벌하지 않도록 post_lift 페널티·success를
    # grip(임의 마디 접촉) 기준으로. tip-only면 wrap↔tip 진동 유발.
    core = (
        Path(__file__).resolve().parents[4] / "common" / "grasp_reward_core.py"
    ).read_text(encoding="utf-8")
    env = _text("grasp_right_env.py")

    # 공유 코어: optional grip_frac (RH56F1 None=tip 유지)
    assert "grip_frac: torch.Tensor | None = None" in core
    # post_lift_contact_loss가 grip_frac 사용
    assert "tip_contact_frac if grip_frac is None else grip_frac" in core
    # env: 임의 마디(tip|middle|distal) 접촉 손가락 수
    assert "num_grip_fingers" in env
    assert "grip_frac=grip_frac" in env
    # success_now가 grip 기반(full_grip) + 엄지-컵 접촉 명시 요구(거짓 4지 그립 배제).
    # distal/middle Cup-only 필터 후 num_grip_fingers>=success_min_grip_fingers & thumb_cup_grip.
    assert "num_grip_fingers >= int(self.cfg.success_min_grip_fingers)" in env
    assert "thumb_cup_grip = any_finger_contact[:, 0]" in env
    assert "& full_grip_bool" in env


def test_reward_uses_rh56f1_shared_core_terms() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")
    reward_body = env.split("def _get_rewards", 1)[1].split("return total", 1)[0]

    for name in (
        "approach_weight",
        "grasp_weight",
        "lift_reward_weight",
        "stabilize_weight",
        "success_bonus_weight",
        "post_lift_contact_loss_weight",
        "action_smooth_weight",
        "stability_reward_weight",
    ):
        assert name in cfg

    for term in (
        "compute_grasp_reward_terms(",
        'reward_terms["approach"]',
        'reward_terms["grasp"]',
        'reward_terms["lift"]',
        'reward_terms["stabilize"]',
        'reward_terms["success_bonus"]',
        'reward_terms["post_lift_contact_loss"]',
        'reward_terms["action_smooth"]',
        'reward_terms["stability"]',
    ):
        assert term in reward_body

    for removed in (
        "r1b_force_balance",
        "r1c_full_grasp",
        "r2_tip_bonus",
        "r5_quality_lift",
        "prelift_rim_lift_penalty",
    ):
        assert removed not in reward_body
