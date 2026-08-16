from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _text(filename: str) -> str:
    return (ROOT / filename).read_text(encoding="utf-8")


def _load_finger_action_utils():
    path = ROOT / "finger_action_utils.py"
    spec = importlib.util.spec_from_file_location("grasp_v1_finger_action_utils", path)
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
    env = _text("grasp_left_env.py")
    preset = _text("grasp_left_preset.py")

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
    env = _text("grasp_left_env.py")

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
    cfg = _text("grasp_left_env_cfg.py")
    utils = _text("grasp_left_utils.py")

    assert "lift_start_min_envelope_fingers: int = 0" in cfg
    # 게이트 배선은 보존(재활성 가능), compute_lift_readiness가 지원만
    assert "min_envelope_fingers" in utils


def test_envelope_credited_in_grasp_and_lift_reward() -> None:
    # envelope(중간/원위 wrap)을 grasp/lift 보상에 credit → tip-farming 차단.
    core = (
        Path(__file__).resolve().parents[1] / "grasp_reward.py"
    ).read_text(encoding="utf-8")
    env = _text("grasp_left_env.py")

    # 전용 모듈: envelope_frac 는 graded_contact 용으로 남아 있다
    assert "envelope_frac: torch.Tensor | None = None" in core
    # grasp 보상의 envelope 비중은 cfg knob(grasp_envelope_credit, 기본 0.40)로 노출된다.
    # ※구 계약은 리터럴 "0.40 * envelope_frac" 였으나 knob 화되며 사라졌다.
    assert '_cfg_float(cfg, "grasp_envelope_credit", 0.40)' in core
    # tip 항은 (1-credit)/0.60 으로 비례 축소 → 합=1 유지 → credit 을 올려도 grasp 최대치 불변
    # (감쌈만 하고 안 드는 국소최적을 구조적으로 차단; reward-audit Check1 근거)
    assert "_tip_scale = (1.0 - _ecred) / 0.60" in core
    # lift/stabilize graded_contact 가 envelope-aware (mix 는 lift_envelope_mix knob)
    assert '_cfg_float(cfg, "lift_envelope_mix", 0.5)' in core
    assert "(1.0 - _emix) * graded_contact + _emix * env_quality" in core
    # env: 중간·원위 마디 접촉으로 envelope_frac 계산해 전달
    assert "middle_binary_contact_buf.float().mean" in env
    assert "distal_binary_contact_buf.float().mean" in env
    assert "envelope_frac=envelope_frac" in env


def test_wrap_depth_and_retention_contract() -> None:
    """08.16 감쌈 깊이(per-finger mid AND distal)와 래치 대비 유지 페널티 계약.

    배경: ADR 만렙 후 난이도가 상수인 구간에서도 감쌈만 단조 침식했다. 원인은
    ①grasp(감쌈 credit)가 pre_lift_gate 로 리프트 순간 꺼지고 ②post_lift 페널티가
    grip_frac(마디 무관 OR)이라 중간마디를 잃어도 비용이 0 이었던 것.
    """
    core = (
        Path(__file__).resolve().parents[1] / "grasp_reward.py"
    ).read_text(encoding="utf-8")
    env = _text("grasp_left_env.py")

    # grasp_v1 전용 모듈이므로 optional 이 아니라 **필수 인자**다(공유 core 호환 불필요).
    assert "wrap_frac: torch.Tensor," in core
    assert "wrap_at_latch: torch.Tensor," in core
    # grasp credit 이 느슨한 envelope_frac 이 아니라 깊이(wrap_frac)를 직접 참조
    assert "+ _ecred * wrap_frac.clamp(0.0, 1.0)" in core
    assert "_ecred * envelope_frac.clamp(0.0, 1.0)" not in core
    # 유지 페널티는 **절대 깊이가 아니라 래치 대비 감소분** — 유지하면 비용 0이라
    # 보상 기준선이 이동하지 않는다(절대 깊이 처벌은 리프트를 억제해 REVISE 됨)
    assert "torch.relu(wrap_at_latch.clamp(0.0, 1.0) - wrap_frac.clamp(0.0, 1.0))" in core
    assert 'wrap_retention_loss_weight' in core

    # env: per-finger AND 로 깊이 산출 + 래치 순간 스냅샷 + 리셋 클리어
    assert "middle_binary_contact_buf & self.distal_binary_contact_buf" in env
    assert "self.wrap_at_latch_buf = torch.where(" in env
    assert "self.wrap_at_latch_buf[env_ids] = 0.0" in env
    # 회피 경로("얕게 래치하면 잃을 게 없다") 감시용 로깅이 있어야 한다
    assert 'self.extras["contact/wrap_at_latch"]' in env


def test_retighten_and_tipping_signal_contract() -> None:
    """래치 후 재조임 권한 + 회전 외란 구간의 실패 신호 복원."""
    env = _text("grasp_left_env.py")

    # 파지력 = stiffness×(target−actual) 오버슈트뿐인데 동결이 첫 접촉에서 걸려
    # 오버슈트≈0 으로 고정된다 → 래치 후 동결 해제로 "더 조일" 권한을 준다
    assert 'getattr(self.cfg, "retighten_after_latch", False)' in env
    assert "gate20 = gate20 * (~self.lift_ready_latched_buf).float().unsqueeze(1)" in env
    # 회전 외란은 래치 후에만 걸리는데 tipped 종료가 그 구간 전체에서 꺼져 있었다 →
    # 스크립트 램프 구간만 억제하고 hold 구간에서는 복원
    assert 'getattr(self.cfg, "tipping_active_after_lift_ramp", False)' in env
    assert "is_scripted_phase = self.is_lift_phase & _ramp_left" in env


def test_post_lift_and_success_are_grip_consistent() -> None:
    # envelope wrap이 tip을 mid/dist로 옮겨도 처벌하지 않도록 post_lift 페널티·success를
    # grip(임의 마디 접촉) 기준으로. tip-only면 wrap↔tip 진동 유발.
    core = (
        Path(__file__).resolve().parents[1] / "grasp_reward.py"
    ).read_text(encoding="utf-8")
    env = _text("grasp_left_env.py")

    # grip_frac(마디 무관 OR)은 breadth 유지용으로 그대로 사용
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
    cfg = _text("grasp_left_env_cfg.py")
    env = _text("grasp_left_env.py")
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
