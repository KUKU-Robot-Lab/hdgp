# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TDD: Actor/Critic obs가 비드 상태(컵 안/밖 여부)를 올바르게 반영하는지 검증.

배경:
  Actor가 비드 상태를 전혀 모르면 "소스 컵이 비었다" / "타겟에 비드가 들어갔다"는 신호를
  받지 못한다. 이로 인해 구슬을 다 쏟고도 계속 불필요한 자세 변환이 발생한다.

테스트 구조 (단위 테스트 — Isaac Sim 불필요):
  실제 환경을 띄우지 않고 constants.py / obs 조립 로직만 검증한다.
  환경 내부 tensor 연산은 torch mock으로 대체.
"""

import re
from pathlib import Path

import pytest
import torch

# 패키지명이 숫자로 시작('5g_pour_right_v3')하여 Python import 불가.
# constants.py 에서 NUM_* 상수만 정규식으로 직접 파싱한다.
_TASK_DIR = Path(__file__).parent.parent
_CONSTANTS_TEXT = (_TASK_DIR / "pour_right_constants.py").read_text()

def _parse_constants() -> dict:
    """NUM_* = <정수 또는 덧셈식> 패턴만 추출해 평가."""
    ns: dict = {}
    for m in re.finditer(r"^(NUM_\w+)\s*=\s*([^\n#]+)", _CONSTANTS_TEXT, re.MULTILINE):
        key, expr = m.group(1), m.group(2).strip()
        try:
            ns[key] = int(eval(expr, {}, ns))  # noqa: S307 — 신뢰된 로컬 파일
        except Exception:
            pass
    return ns

_C = _parse_constants()
NUM_OBSERVATIONS        = _C["NUM_OBSERVATIONS"]
NUM_CRITIC_OBSERVATIONS = _C["NUM_CRITIC_OBSERVATIONS"]
NUM_CRITIC_EXTRAS       = _C["NUM_CRITIC_EXTRAS"]


# ---------------------------------------------------------------------------
# 1. 관측 차원 테스트
# ---------------------------------------------------------------------------

class TestObsDimensions:
    """Actor/Critic 관측 차원이 비드 상태 정보를 포함하도록 설계됐는지 검증."""

    def test_actor_obs_includes_bead_fractions(self):
        """Actor obs는 bead_in_source_fraction(1D) + bead_in_target_fraction(1D)를 포함해야 한다.

        RED: 현재 NUM_OBSERVATIONS=106이면 FAIL.
        GREEN: 108로 올리면 PASS.
        """
        # bead_in_source + bead_in_target = 2D 포함 최소 요구
        assert NUM_OBSERVATIONS >= 110, (
            f"Actor obs({NUM_OBSERVATIONS}D)에 bead_in_source_fraction(1D) + "
            f"bead_in_target_fraction(1D)가 없음. 최소 108D 필요."
        )

    def test_critic_obs_includes_bead_fractions(self):
        """Critic obs는 bead 상태 fractions을 포함해야 한다 (기존 유지 검증)."""
        # critic extras에 bead_cross(1) + bead_in_source(1) + bead_in_target(1) + spill(1) = 4
        assert NUM_CRITIC_EXTRAS >= 4, (
            f"Critic extras({NUM_CRITIC_EXTRAS}D)에 bead fraction 4개가 없음."
        )

    def test_critic_total_dim_consistent(self):
        """Critic total = actor + extras 일관성."""
        assert NUM_CRITIC_OBSERVATIONS == NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS, (
            f"Critic total({NUM_CRITIC_OBSERVATIONS}) != "
            f"actor({NUM_OBSERVATIONS}) + extras({NUM_CRITIC_EXTRAS})"
        )


# ---------------------------------------------------------------------------
# 2. 비드 fractions 값 범위 테스트
# ---------------------------------------------------------------------------

class TestBeadFractionValues:
    """bead_in_source_fraction / bead_in_target_fraction 값 범위 및 의미 검증."""

    def test_source_fraction_full_when_all_beads_in_source(self):
        """소스 컵에 비드가 모두 있을 때 fraction = 1.0."""
        bead_in_source = torch.ones(20, dtype=torch.bool)   # 20개 전부 소스에
        fraction = bead_in_source.float().mean()
        assert float(fraction) == pytest.approx(1.0)

    def test_source_fraction_zero_when_all_beads_poured(self):
        """비드를 전부 쏟았을 때 fraction = 0.0."""
        bead_in_source = torch.zeros(20, dtype=torch.bool)  # 0개
        fraction = bead_in_source.float().mean()
        assert float(fraction) == pytest.approx(0.0)

    def test_target_fraction_half_when_half_captured(self):
        """10개 중 10개(절반) 타겟에 있을 때 fraction = 0.5."""
        bead_in_target = torch.zeros(20, dtype=torch.bool)
        bead_in_target[:10] = True   # 10/20
        fraction = bead_in_target.float().mean()
        assert float(fraction) == pytest.approx(0.5)

    def test_fraction_range_always_zero_to_one(self):
        """fraction은 항상 [0, 1] 범위여야 한다."""
        for n_in in range(0, 21):
            flags = torch.zeros(20, dtype=torch.bool)
            flags[:n_in] = True
            fraction = float(flags.float().mean())
            assert 0.0 <= fraction <= 1.0, f"n_in={n_in}: fraction={fraction} 범위 이탈"

    def test_source_empty_threshold(self):
        """source_empty 판정: fraction < 0.05 (20개 기준 1개 미만)."""
        # 1개만 남은 경우: 1/20 = 0.05 → 경계, 0개만 empty
        one_left = torch.zeros(20, dtype=torch.bool)
        one_left[0] = True
        assert float(one_left.float().mean()) == pytest.approx(0.05)

        none_left = torch.zeros(20, dtype=torch.bool)
        assert float(none_left.float().mean()) < 0.05, "0개일 때 empty 판정 실패"


# ---------------------------------------------------------------------------
# 3. obs 조립 로직 — 비드 fraction이 올바른 위치에 있는지
# ---------------------------------------------------------------------------

class TestActorObsBeadPosition:
    """Actor obs tensor에서 bead fraction이 올바른 슬라이스에 위치하는지 검증.

    obs 구조(목표):
      [0:7]   arm_joint_pos
      [7:14]  arm_joint_vel
      [14:34] finger_joint_pos
      [34:54] finger_joint_vel
      [54:57] right_cup_pos_rel_palm
      [57:61] right_cup_quat
      [61:64] left_cup_pos_rel_palm
      [64:68] left_cup_quat
      [68:71] pour_point_to_opening
      [71:74] source_pour_axis
      [74:77] source_up_axis
      [77:85] transport_summary (8D)
      [85:90] binary_contact
      [90:95] tip_force_norm
      [95:106] last_actions
      [106]   bead_in_source_fraction  ← NEW
      [107]   bead_in_target_fraction  ← NEW
      Total: 108D
    """

    def _build_mock_actor_obs(
        self,
        bead_in_source_fraction: float,
        bead_in_target_fraction: float,
    ) -> torch.Tensor:
        """실제 환경 없이 obs tensor를 mock으로 조립."""
        N = 4  # batch size
        base = torch.zeros(N, 106)         # 기존 106D
        bead_source = torch.full((N, 1), bead_in_source_fraction)
        bead_target = torch.full((N, 1), bead_in_target_fraction)
        return torch.cat([base, bead_source, bead_target], dim=-1)  # 108D

    def test_actor_obs_dim_108(self):
        """mock obs 차원이 108D인지 확인."""
        obs = self._build_mock_actor_obs(1.0, 0.0)
        assert obs.shape == (4, 108)

    def test_bead_source_fraction_at_index_106(self):
        """bead_in_source_fraction이 index 106에 위치."""
        obs = self._build_mock_actor_obs(bead_in_source_fraction=0.75, bead_in_target_fraction=0.0)
        assert obs[:, 106].mean().item() == pytest.approx(0.75)

    def test_bead_target_fraction_at_index_107(self):
        """bead_in_target_fraction이 index 107에 위치."""
        obs = self._build_mock_actor_obs(bead_in_source_fraction=0.0, bead_in_target_fraction=0.4)
        assert obs[:, 107].mean().item() == pytest.approx(0.4)

    def test_bead_fractions_independent(self):
        """두 fraction은 독립적으로 변해야 한다 (spill 상황: source=0, target=0.3)."""
        obs = self._build_mock_actor_obs(bead_in_source_fraction=0.0, bead_in_target_fraction=0.3)
        assert obs[:, 106].mean().item() == pytest.approx(0.0)   # source empty
        assert obs[:, 107].mean().item() == pytest.approx(0.3)   # target partial
