from __future__ import annotations

from pathlib import Path

import torch


ENV_PATH = Path(__file__).resolve().parents[1] / "grasp_right_env.py"
CFG_PATH = Path(__file__).resolve().parents[1] / "grasp_right_env_cfg.py"

CONTACT_FORCE_MAX = 10.0


def _middle_contact_reward(
    *,
    middle_force_raw: torch.Tensor,
    middle_binary: torch.Tensor,
    lift_latched: torch.Tensor,
    middle_contact_weight: float,
    envelope_bonus_weight: float,
    min_middle_contacts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """env 배선과 동일한 공식 (순수 torch 재현)."""
    middle_norm = (middle_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)
    middle_contact_count = middle_binary.float().sum(dim=-1)
    pre_lift_gate = (~lift_latched).float()
    reward = middle_contact_weight * pre_lift_gate * middle_norm.sum(dim=-1)
    bonus = (
        envelope_bonus_weight
        * pre_lift_gate
        * (middle_contact_count >= int(min_middle_contacts)).float()
    )
    return reward, bonus


def test_middle_contact_rewards_wrap_not_tip_poke() -> None:
    # env0: 끝만 poke (중간마디 접촉 0) → 보상 0
    # env1: 4지 중간마디 감쌈(wrap) → 보상 + envelope bonus
    middle_force = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 3.0, 0.5],
        ]
    )
    middle_binary = middle_force > 0.1
    lift_latched = torch.zeros(2, dtype=torch.bool)

    reward, bonus = _middle_contact_reward(
        middle_force_raw=middle_force,
        middle_binary=middle_binary,
        lift_latched=lift_latched,
        middle_contact_weight=3.0,
        envelope_bonus_weight=3.0,
        min_middle_contacts=4,
    )

    assert reward[0].item() == 0.0       # tip-only poke는 보상 없음
    assert reward[1].item() > 0.0        # wrap은 보상
    assert bonus[0].item() == 0.0
    assert bonus[1].item() == 3.0        # ≥4 중간마디 → envelope bonus


def test_middle_contact_only_in_prelift_phase() -> None:
    # latch 후(lift phase)에는 0 — grasp 단계만 shaping
    middle_force = torch.full((1, 5), 5.0)
    middle_binary = middle_force > 0.1
    latched = torch.ones(1, dtype=torch.bool)

    reward, bonus = _middle_contact_reward(
        middle_force_raw=middle_force,
        middle_binary=middle_binary,
        lift_latched=latched,
        middle_contact_weight=3.0,
        envelope_bonus_weight=3.0,
        min_middle_contacts=4,
    )

    assert reward.item() == 0.0
    assert bonus.item() == 0.0


def test_middle_contact_per_finger_norm_is_clamped() -> None:
    # 과압박(>force_max) 해킹 방지: per-finger norm 1.0 clamp
    over = torch.full((1, 5), 100.0)  # 10N 한참 초과
    middle_binary = over > 0.1
    reward, _ = _middle_contact_reward(
        middle_force_raw=over,
        middle_binary=middle_binary,
        lift_latched=torch.zeros(1, dtype=torch.bool),
        middle_contact_weight=3.0,
        envelope_bonus_weight=3.0,
        min_middle_contacts=4,
    )
    # 5 fingers × clamp(1.0) × weight 3.0 = 15.0 (무한 압박해도 상한)
    assert reward.item() == 15.0


def test_env_wires_middle_contact_reward_into_total() -> None:
    env_src = ENV_PATH.read_text(encoding="utf-8")
    cfg_src = CFG_PATH.read_text(encoding="utf-8")

    # env에 실제 배선 (orphan config가 아니라 total에 더해짐)
    assert "middle_contact_reward" in env_src
    assert "middle_envelope_bonus" in env_src
    assert "self.cfg.middle_contact_weight" in env_src
    assert "total + middle_contact_reward + middle_envelope_bonus" in env_src
    assert "(~self.lift_ready_latched_buf).float()" in env_src
    assert '"reward/middle_contact"' in env_src

    # cfg weight 활성화 (Phase L)
    assert "middle_contact_weight: float = 3.0" in cfg_src
    assert "middle_contact_envelope_bonus_weight: float = 3.0" in cfg_src


def test_success_lifted_gate_uses_unified_v1_height() -> None:
    # 방향B(v1 정렬): test4~17의 이중 hold(success_lift_height 0.015 ≠ lift_success_height 0.025)를
    # 폐기하고 v1 단일 계약(0.04)으로 통일 — lifted 게이트와 보상 saturation이 같은 문턱을 쓴다.
    env_src = ENV_PATH.read_text(encoding="utf-8")
    cfg_src = CFG_PATH.read_text(encoding="utf-8")

    assert "success_lift_height: float = 0.04" in cfg_src
    assert "lift_success_height: float = 0.04" in cfg_src
    # lifted 게이트가 success_lift_height를 쓰는지 (lift_success_height가 아니라)
    lifted_line = [
        ln for ln in env_src.splitlines()
        if "lifted" in ln and "self.object_pos[:, 2]" in ln
    ]
    assert lifted_line, "lifted 게이트 라인을 찾지 못함"
    assert "self.cfg.success_lift_height" in lifted_line[0]
