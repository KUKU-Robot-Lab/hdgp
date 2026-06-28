from __future__ import annotations

from pathlib import Path

import torch


ENV_PATH = Path(__file__).resolve().parents[1] / "grasp_right_env.py"
CFG_PATH = Path(__file__).resolve().parents[1] / "grasp_right_env_cfg.py"


def _num_envelope(tip: torch.Tensor, middle: torch.Tensor, distal: torch.Tensor) -> torch.Tensor:
    """env 배선과 동일: 손가락 engaged = tip ∨ middle ∨ distal."""
    return (tip | middle | distal).sum(dim=-1).long()


def test_envelope_counts_finger_wrapped_by_any_phalanx() -> None:
    # 현재 학습 상태 모사: 검지(2)는 tip 빠지고 middle로 감쌈, ring(4)은 tip만, 나머지 tip+
    # 결과: 5손가락 모두 어떤 마디로든 접촉 → envelope 5
    tip    = torch.tensor([[True,  False, True,  True,  True ]])
    middle = torch.tensor([[True,  True,  False, False, True ]])
    distal = torch.tensor([[False, False, False, False, True ]])
    env = _num_envelope(tip, middle, distal)
    assert env.tolist() == [5]


def test_envelope_requires_each_finger_to_touch_somewhere() -> None:
    # 검지가 어느 마디로도 안 닿으면 envelope 4 (전부 감싸기 미달)
    tip    = torch.tensor([[True,  False, True,  True,  True ]])
    middle = torch.tensor([[True,  False, False, False, True ]])
    distal = torch.tensor([[False, False, False, False, True ]])
    env = _num_envelope(tip, middle, distal)
    assert env.tolist() == [4]


def test_tip_only_grasp_still_counts_when_tips_touch() -> None:
    # 모든 손가락 tip만 닿아도 envelope 5 (tip도 한 마디이므로 포함)
    tip    = torch.ones(1, 5, dtype=torch.bool)
    middle = torch.zeros(1, 5, dtype=torch.bool)
    distal = torch.zeros(1, 5, dtype=torch.bool)
    assert _num_envelope(tip, middle, distal).tolist() == [5]


def test_env_wires_envelope_count_into_grasp_gates() -> None:
    env_src = ENV_PATH.read_text(encoding="utf-8")
    cfg_src = CFG_PATH.read_text(encoding="utf-8")

    # envelope buffer 계산 (tip ∨ middle ∨ distal)
    assert "num_envelope_contacts_buf" in env_src
    assert "self.binary_contact_buf" in env_src
    assert "self.middle_binary_contact_buf" in env_src
    assert "self.distal_binary_contact_buf" in env_src
    # 게이트가 envelope count를 사용
    assert "num_tip_contacts = self.num_envelope_contacts_buf" in env_src
    assert "full_tip_contact = self.num_envelope_contacts_buf >= self._adr_min_contacts" in env_src
    # 실험 A: success 요구는 ≥4 envelope(≥5 sustained는 떨림으로 미달), grasp 보상은 여전히 5 유도
    assert "(3.0, 4.0)" in cfg_src
