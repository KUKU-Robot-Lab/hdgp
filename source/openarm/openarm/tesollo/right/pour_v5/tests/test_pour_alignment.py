"""pour_point 3D 정렬 score 단위 테스트 (Stage B r_pour_xy+r_descend 통합).

설계: r_align = exp(-scale · ‖pour_point − (target_opening + [0,0,z_margin])‖₃ᴅ)
  - xy(빗나감)와 z(높이)를 단일 3D 거리로 동시 처리.
  - 목표점은 target 입구가 아니라 그 z_margin 위 (두 컵 충돌 방지).
"""
from __future__ import annotations

from pathlib import Path

import torch

from openarm.tesollo.right.pour_v5.pour_right_utils import pour_alignment_score


TASK_DIR = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (TASK_DIR / path).read_text()


def test_score_is_one_at_aim_point() -> None:
    # Arrange: pour_point가 target_opening + z_margin 위에 정확히 위치
    z_margin = 0.05
    target_opening = torch.tensor([[0.4, 0.1, 0.39]])
    pour_point = torch.tensor([[0.4, 0.1, 0.39 + z_margin]])

    # Act
    score = pour_alignment_score(pour_point, target_opening, z_margin, scale=8.0)

    # Assert: 정확히 조준 → score ≈ 1
    assert torch.allclose(score, torch.ones(1), atol=1e-5)


def test_xy_miss_reduces_score() -> None:
    z_margin = 0.05
    target_opening = torch.tensor([[0.4, 0.1, 0.39]])
    aimed = torch.tensor([[0.4, 0.1, 0.44]])
    xy_off = torch.tensor([[0.5, 0.1, 0.44]])  # x로 10cm 빗나감

    s_aim = pour_alignment_score(aimed, target_opening, z_margin, scale=8.0)
    s_off = pour_alignment_score(xy_off, target_opening, z_margin, scale=8.0)

    assert (s_off < s_aim).all()


def test_z_miss_reduces_score() -> None:
    z_margin = 0.05
    target_opening = torch.tensor([[0.4, 0.1, 0.39]])
    aimed = torch.tensor([[0.4, 0.1, 0.44]])         # z 정확
    z_high = torch.tensor([[0.4, 0.1, 0.60]])        # 16cm 고공

    s_aim = pour_alignment_score(aimed, target_opening, z_margin, scale=8.0)
    s_high = pour_alignment_score(z_high, target_opening, z_margin, scale=8.0)

    assert (s_high < s_aim).all()


def test_monotonic_decreasing_with_distance() -> None:
    z_margin = 0.0
    target_opening = torch.tensor([[0.0, 0.0, 0.0]])
    near = torch.tensor([[0.02, 0.0, 0.0]])
    far = torch.tensor([[0.20, 0.0, 0.0]])

    s_near = pour_alignment_score(near, target_opening, z_margin, scale=8.0)
    s_far = pour_alignment_score(far, target_opening, z_margin, scale=8.0)

    assert (s_near > s_far).all()
    assert (s_far > 0.0).all()  # 항상 양수 gradient (죽지 않음)


def test_batch_shape() -> None:
    z_margin = 0.05
    target_opening = torch.zeros(4, 3)
    pour_point = torch.randn(4, 3)
    score = pour_alignment_score(pour_point, target_opening, z_margin, scale=8.0)
    assert score.shape == (4,)


def test_stage_b_reward_uses_single_3d_align_term() -> None:
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")

    stage_b = env.split("# Stage B — pour-point / tilt / bead", maxsplit=1)[1].split(
        "# ============================================================\n        # Outcome",
        maxsplit=1,
    )[0]

    assert "pour_alignment_score" in env
    assert "r_align = (" in stage_b
    assert "r_pour_xy" not in stage_b
    assert "r_descend" not in stage_b
    assert "weight_align" in cfg
    assert "pour_align_scale" in cfg
    assert "pour_align_z_margin" in cfg


def test_g_ready_gate_uses_pour_point_not_origin() -> None:
    """[H1] Stage B 게이트 g_ready는 origin(cup_center)이 아닌 pour_point(mouth_xy) 기준.

    origin 기준이면 "origin만 target 위에 두면 Stage B 만점" 회피해 → pour_point가
    입구 밖이어도 보상(spill). 비드는 pour_point에서 나오므로 게이트도 pour_point 기준이어야.
    """
    env = _read("pour_right_env.py")

    gate = env.split("g_ready = torch.sigmoid(", maxsplit=1)[1].split(")", maxsplit=1)[0]

    # 게이트 입력은 mouth_xy_distance(pour_point), cup_center가 아님
    assert "_mouth_xy_distance" in gate
    assert "_cup_center_xy_dist" not in gate
    # transport(r_approach)는 여전히 cup_center 기준 유지 (단일 변경 범위 확인)
    assert "_cup_center_xy_dist - self.cfg.cup_transport_saturate_xy" in env


def test_align_gate_is_3d_not_xy_only() -> None:
    """[H2] bead align_gate는 xy-only가 아닌 3D(z 포함) 정렬 종속.

    bead_in은 상태기반(fraction) 보상이라 xy-only gate면 z 고공 나쁜 자세서도 통과 →
    그 자세에 최적화(z=16cm 고착). pour_point xyz 정렬돼야 bead_in 발현되도록 3D로 막는다.
    """
    env = _read("pour_right_env.py")

    gate_block = env.split("align_gate = ", maxsplit=1)[1].split("\n        r_bead_in", maxsplit=1)[0]

    # gate가 3D 정렬 score(pour_alignment_score) 기반, xy-only exp가 아님
    assert "pour_alignment_score" in gate_block
    assert "torch.exp(-self.cfg.align_gate_scale * self._mouth_xy_distance)" not in env
    # bead_in / drain이 이 3D gate에 종속
    assert "r_bead_in = align_gate" in env
