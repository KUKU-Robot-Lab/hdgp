"""radial 압축 순수 함수 계약: 감싸기→radial↑, 손끝(축방향)→radial↓, inward만 양수."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "grasp_right_utils.py"
SPEC = importlib.util.spec_from_file_location("grasp_adapt_utils_radial", MODULE_PATH)
_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = _mod
SPEC.loader.exec_module(_mod)

compute_radial_compression = _mod.compute_radial_compression

# 컵: 중심 원점, up축 +z, 반경 0.045
CENTER = torch.tensor([[0.0, 0.0, 0.0]])
AXIS = torch.tensor([[0.0, 0.0, 1.0]])
R = 0.045


def test_envelope_high_radial():
    # 4접촉이 컵 옆면 사방(+x,-x,+y,-y)에서 안으로 미는 감싸기 → radial 큼
    pos = torch.tensor([[[R, 0, 0], [-R, 0, 0], [0, R, 0], [0, -R, 0]]])
    force = torch.tensor([[[-5.0, 0, 0], [5.0, 0, 0], [0, -5.0, 0], [0, 5.0, 0]]])  # 모두 inward
    mask = torch.ones(1, 4)
    out = compute_radial_compression(pos, force, CENTER, AXIS, mask)
    assert float(out[0]) > 15.0  # 4×5N inward ≈ 20


def test_fingertip_axial_low_radial():
    # 손끝이 컵 위 테두리를 축방향(-z)으로 누름 → radial 성분 거의 0
    pos = torch.tensor([[[R, 0, 0.08], [-R, 0, 0.08]]])
    force = torch.tensor([[[0, 0, -5.0], [0, 0, -5.0]]])  # 축방향, radial 0
    mask = torch.ones(1, 2)
    out = compute_radial_compression(pos, force, CENTER, AXIS, mask)
    assert float(out[0]) < 1.0


def test_outward_force_not_counted():
    # 바깥으로 미는(당기는) 힘은 inward relu로 0
    pos = torch.tensor([[[R, 0, 0]]])
    force = torch.tensor([[[5.0, 0, 0]]])  # outward
    mask = torch.ones(1, 1)
    out = compute_radial_compression(pos, force, CENTER, AXIS, mask)
    assert float(out[0]) < 1e-5


def test_mask_excludes_contact():
    pos = torch.tensor([[[R, 0, 0]]])
    force = torch.tensor([[[-5.0, 0, 0]]])  # inward
    mask = torch.zeros(1, 1)  # 마스크로 배제
    out = compute_radial_compression(pos, force, CENTER, AXIS, mask)
    assert float(out[0]) < 1e-5
