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
    # ★08.16 PIP/DIP 분리: 손가락 스칼라 복사(cmd.repeat_interleave)를 채널 전개로 대체.
    #   [_1,_2,_3,_4] 에 [ch0,ch1,ch2,ch2] 를 대응시킨다.
    assert "cmd.repeat_interleave(4" not in env
    assert "cmd_ch[:, :, 0], cmd_ch[:, :, 1], cmd_ch[:, :, 2], cmd_ch[:, :, 2]" in env
    # 1-DOF lerp(close_buf를 repeat_interleave 후 lerp) 제거 확인
    assert "self.finger_close_buf.repeat_interleave(4" not in env


def test_finger_action_is_absolute_level_not_ratchet() -> None:
    # ★08.16 래칫 제거. 구 advance = speed × cmd ≥ 0 은 단조 증가만 가능해
    # 탐색 노이즈만으로 close_buf 가 1.0 에 포화했다(close_frac_max 첫 구간부터 1.0).
    # 그 상태로 채널만 분리하면 세 채널이 전부 1.0 이 되어 비율이 안 생긴다 —
    # 즉 PIP/DIP 분리와 래칫 제거는 한 묶음이어야 의미가 있다.
    env = _text("grasp_right_env.py")
    consts = _text("grasp_right_constants.py")

    # 절대 목표를 향해 변화율 상한으로 이동(감소 가능)
    assert "delta = (cmd20 - self.finger_close_buf).clamp(-_rate, _rate)" in env
    assert "advance = delta * (1.0 - gate20)" in env
    # 구 단방향 누적 형태가 남아 있지 않을 것
    assert "float(self.cfg.finger_close_speed) * cmd20" not in env
    # 접촉 동결은 유지되어야 한다(감쌈 생성 메커니즘 — 제거 시 3지 국소최적 회귀)
    assert "(1.0 - gate20)" in env

    # 액션 차원 계약: 6 palm + 15 finger(5×3), squeeze 철회
    assert "NUM_FINGER_CHANNELS = 3" in consts
    assert "NUM_FINGER_ACTION = NUM_FINGERTIPS * NUM_FINGER_CHANNELS" in consts
    assert "NUM_SQUEEZE_ACTION = 0" in consts
    # squeeze 잔재 없음
    assert "squeeze_action" not in env
    assert "squeeze_cmd_buf" not in env


def test_four_finger_coupling_is_per_channel() -> None:
    # couple_four_fingers 는 3지 국소최적 차단용이므로 유지하되, **채널별로** 평균내야
    # 4지가 같은 자세를 공유하면서도 그 자세의 외전/MCP/PIP 비율은 자유로울 수 있다.
    env = _text("grasp_right_env.py")
    assert "finger_action[:, 1:5, :].mean(dim=1, keepdim=True)" in env
    assert "_common4.expand(-1, 4, -1)" in env


def test_lift_latch_gate_disabled_envelope_via_reward() -> None:
    # 인벨롭 latch hard 게이트는 비활성(=0): success를 죽이면서 envelope은 못 만듦.
    # envelope은 grasp/lift 보상 credit(soft gradient)으로 유도한다.
    cfg = _text("grasp_right_env_cfg.py")
    utils = _text("grasp_right_utils.py")

    assert "lift_start_min_envelope_fingers: int = 0" in cfg
    # 게이트 배선은 보존(재활성 가능), compute_lift_readiness가 지원만
    assert "min_envelope_fingers" in utils


def test_envelope_credited_in_grasp_and_lift_reward() -> None:
    """2026-08-20 재설계: 감쌈 유도가 contact_envelope_credit 한 곳으로 통합됐다.

    구 계약(grasp_envelope_credit + lift_envelope_mix + wrap_retention 3원화)은 폐기.
    손끝 항이 grasp quality 의 95% 를 차지해 손끝 파지가 최적해였던 것이 실패 원인이라,
    envelope 비중이 지배적(기본 0.75)이어야 한다.
    """
    core = (
        Path(__file__).resolve().parents[1] / "grasp_reward.py"
    ).read_text(encoding="utf-8")
    env = _text("grasp_right_env.py")

    assert '_cfg_float(cfg, "contact_envelope_credit", 0.75)' in core
    assert "(1.0 - _env_credit) * tip_contact_frac" in core
    assert "_env_credit * envelope_frac" in core
    # 구 3원화 knob 이 되살아나면 실패
    for dead in ("grasp_envelope_credit", "lift_envelope_mix", "wrap_retention_loss_weight"):
        assert dead not in core, f"구 knob 부활: {dead}"
    # env: 중간·원위 마디 접촉으로 envelope_frac 계산해 전달
    assert "middle_binary_contact_buf.float().mean" in env
    assert "distal_binary_contact_buf.float().mean" in env
    assert "envelope_frac=envelope_frac" in env


def test_wrap_depth_and_retention_contract() -> None:
    """2026-08-20: 감쌈 유지 페널티 제거를 고정한다.

    감쌈을 잃으면 contact_quality 가 곧바로 떨어져 grasp 보상이 자동 감소하므로
    별도 페널티는 중복이었다(구 설계 페널티 5개 → 2개). 래치 스냅샷 기준선
    (wrap_at_latch)도 보상에서 빠졌다 — 다만 **진단 로깅으로는 유지**한다.
    """
    core = (
        Path(__file__).resolve().parents[1] / "grasp_reward.py"
    ).read_text(encoding="utf-8")
    env = _text("grasp_right_env.py")

    for dead in ("wrap_at_latch", "wrap_frac", "post_lift_contact_loss"):
        assert dead not in core, f"보상에서 제거됐어야 할 항: {dead}"
    # 로깅은 유지 — 감쌈 침식 진단은 계속 필요하다
    assert 'self.extras["contact/wrap_at_latch"]' in env
    assert "middle_binary_contact_buf & self.distal_binary_contact_buf" in env


def test_retighten_and_tipping_signal_contract() -> None:
    """래치 후 재조임 권한 + 회전 외란 구간의 실패 신호 복원."""
    env = _text("grasp_right_env.py")

    # 파지력 = stiffness×(target−actual) 오버슈트뿐인데 동결이 첫 접촉에서 걸려
    # 오버슈트≈0 으로 고정된다 → 래치 후 동결 해제로 "더 조일" 권한을 준다
    assert 'getattr(self.cfg, "retighten_after_latch", False)' in env
    assert "gate20 = gate20 * (~self.lift_ready_latched_buf).float().unsqueeze(1)" in env
    # 회전 외란은 래치 후에만 걸리는데 tipped 종료가 그 구간 전체에서 꺼져 있었다 →
    # 스크립트 램프 구간만 억제하고 hold 구간에서는 복원
    assert 'getattr(self.cfg, "tipping_active_after_lift_ramp", False)' in env
    assert "is_scripted_phase = self.is_lift_phase & _ramp_left" in env


def test_post_lift_and_success_are_grip_consistent() -> None:
    """success_flag 는 보상에서 분리돼 ADR·warm export·로깅 전용으로 남는다.

    2026-08-20 재설계에서 success_bonus 보상 항은 hold 로 대체됐지만, success_flag
    자체는 파이프라인 하류(ADR 트리거, warm 뱅크 입구)가 쓰므로 정의가 유지돼야 한다.
    """
    env = _text("grasp_right_env.py")

    assert "self.success_flag" in env
    assert "adr/trigger_metric" in env
    # 실리프트 증거(P5) 유지 — 들지 않은 latch 가 ADR 을 오염시키던 것을 막는다
    assert "_lifted_evidence" in env
    assert "_upright_evidence" in env


def test_reward_uses_rh56f1_shared_core_terms() -> None:
    """2026-08-20 재설계 계약: 4항 + 페널티 2개, latch 게이트 0개."""
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")
    reward_body = env.split("def _get_rewards", 1)[1].split("return total", 1)[0]

    for name in (
        "approach_weight",
        "grasp_weight",
        "contact_envelope_credit",
        "lift_weight",
        "hold_weight",
        "upright_soft_scale_deg",
        "push_penalty_weight",
        "action_smooth_weight",
    ):
        assert name in cfg, f"신 가중치 누락: {name}"

    # 구 8항 가중치가 되살아나면 실패
    for dead in (
        "lift_reward_weight",
        "stabilize_weight",
        "success_bonus_weight",
        "post_lift_contact_loss_weight",
        "stability_reward_weight",
        "approach_xy_penalty_weight",
    ):
        assert dead not in cfg, f"구 가중치 부활: {dead}"

    assert "compute_grasp_reward_terms(" in reward_body
    # 항·factor 를 전부 로깅한다(구 설계는 곱셈 factor 를 하나도 안 내보내 진단 불가였다)
    assert 'self.extras[f"reward/{_k}"]' in reward_body
    assert 'self.extras[f"reward_factor/{_k}"]' in reward_body
