"""엄지+대향2지 precision grasp mask 계약 (Phase 1에서 구현).

패키지 경로 import는 상위 __init__.py가 isaaclab을 끌어와 GPU 없이 실패하므로,
grasp_v1/tests 관행대로 grasp_right_utils.py를 파일 경로에서 직접 로드한다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "grasp_right_utils.py"
SPEC = importlib.util.spec_from_file_location("grasp_adapt_utils", MODULE_PATH)
_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = _mod
SPEC.loader.exec_module(_mod)


def _mask(tip_contact_bool: torch.Tensor) -> torch.Tensor:
    return _mod.compute_precision_grasp_mask(tip_contact_bool)


def test_thumb_plus_two_opposing_is_grasp():
    # 엄지(idx0) + 검지 + 중지 접촉 → True
    tip = torch.tensor([[True, True, True, False, False]])
    assert bool(_mask(tip)[0]) is True


def test_no_thumb_is_not_grasp():
    # 엄지 없이 4지 접촉 → False (엄지 대향이 파지의 핵심)
    tip = torch.tensor([[False, True, True, True, True]])
    assert bool(_mask(tip)[0]) is False


def test_thumb_plus_one_is_not_grasp():
    # 엄지 + 1지만 → False (대향 2지 미만)
    tip = torch.tensor([[True, True, False, False, False]])
    assert bool(_mask(tip)[0]) is False
