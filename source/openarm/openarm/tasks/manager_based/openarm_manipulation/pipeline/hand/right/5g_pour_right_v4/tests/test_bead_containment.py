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

"""TDD: critic obs 차원 정확성과 bead mouth crossing 로직 검증.

배경:
  test1 학습에서 bead_in_target_fraction = 0%, bead_cross_fraction = 0%
  → 실제 구슬이 타겟에 안 들어가는지, 아니면 계산 버그인지 확인 필요.

테스트 범위 (Isaac Sim 불필요, 순수 torch 계산):
  1. Critic obs extras 33D 구성 검증 (constants.py 기준)
  2. mouth crossing 감지 로직 (prev_z/curr_z/radius 조건)
  3. bead_in_target 감지 로직 (z-bounds + radius 조건)
  4. 초기화 값(prev_z=10.0)의 first-frame 동작 확인
"""

import re
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# 상수 파싱 (pour_right_constants.py)
# ---------------------------------------------------------------------------
_TASK_DIR = Path(__file__).parent.parent
_CONSTANTS_TEXT = (_TASK_DIR / "pour_right_constants.py").read_text()


def _parse_constants() -> dict:
    ns: dict = {}
    for m in re.finditer(r"^(NUM_\w+)\s*=\s*([^\n#]+)", _CONSTANTS_TEXT, re.MULTILINE):
        key, expr = m.group(1), m.group(2).strip()
        try:
            ns[key] = int(eval(expr, {}, ns))  # noqa: S307 — 신뢰된 로컬 파일
        except Exception:
            pass
    return ns


_C = _parse_constants()
NUM_OBSERVATIONS = _C["NUM_OBSERVATIONS"]
NUM_CRITIC_EXTRAS = _C["NUM_CRITIC_EXTRAS"]
NUM_CRITIC_OBSERVATIONS = _C["NUM_CRITIC_OBSERVATIONS"]

# cup_big.usd 기준 컵 파라미터 (pour_right_env_cfg.py 와 동일)
TARGET_INNER_RADIUS = 0.041
TARGET_INSIDE_Z_MIN = -0.070
TARGET_INSIDE_Z_MAX = 0.100
TARGET_MOUTH_Z = 0.100      # 림 높이 (crossing 기준) = z_max 와 같음
PREV_Z_INIT = 10.0          # _prev_bead_target_local_z 초기값


# ===========================================================================
# 1. Critic obs extras 33D 구성 검증
# ===========================================================================

class TestCriticObsDim:
    """pour_right_constants.py의 NUM_CRITIC_EXTRAS=33이 실제 구성과 일치하는지 확인."""

    def test_critic_extras_count_by_component(self):
        """extras 항목을 직접 합산해 33D 인지 검증.

        constants.py 도큐스트링에 열거된 항목 기준:
          left_arm_joint_pos(9) + left_arm_joint_vel(9)
          + distal_contact_binary(5) + distal_contact_norm(5)
          + cup_height_delta(1)
          + g_align_xy(1) + g_clear(1) + g_tilt(1) + g_pour(1)
          = 33

        NOTE: critic_obs cat 구성과 동일해야 함.
        """
        left_arm = 9 + 9            # pos + vel
        distal = 5 + 5              # binary + norm
        cup_height_delta = 1
        gates = 1 + 1 + 1 + 1  # g_align_xy, g_clear, g_tilt, g_pour

        total = left_arm + distal + cup_height_delta + gates
        assert total == 33, f"직접 합산 결과 {total}D != 33D"
        assert NUM_CRITIC_EXTRAS == 33

    def test_critic_total_is_actor_plus_extras(self):
        """NUM_CRITIC_OBSERVATIONS == NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS."""
        assert NUM_CRITIC_OBSERVATIONS == NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS, (
            f"{NUM_CRITIC_OBSERVATIONS} != {NUM_OBSERVATIONS} + {NUM_CRITIC_EXTRAS}"
        )

    def test_critic_total_is_143(self):
        """절대값으로 143D 확인 (110 actor + 33 extras)."""
        assert NUM_CRITIC_OBSERVATIONS == 143


# ===========================================================================
# 2. Mouth crossing 로직 검증
#    실제 _compute_bead_flags() 내부 mouth_crossed_now 조건을 torch로 재현
# ===========================================================================

def _compute_mouth_crossed(
    bead_xy_to_target: torch.Tensor,   # (N, K)
    prev_z: torch.Tensor,               # (N, K)
    curr_z: torch.Tensor,               # (N, K)
    inner_radius: float = TARGET_INNER_RADIUS,
    mouth_z: float = TARGET_MOUTH_Z,
) -> torch.Tensor:
    """pour_right_env.py 838-842줄과 동일한 crossing 판정."""
    return (
        (bead_xy_to_target <= inner_radius)
        & (prev_z > mouth_z)
        & (curr_z <= mouth_z)
    )


class TestMouthCrossing:
    """bead가 target cup 입구(rim)를 통과했는지 판정 로직 검증."""

    def test_crossing_detected_when_bead_falls_from_above(self):
        """정상 케이스: 비드가 림 위에서 아래로 낙하할 때 crossing 감지."""
        xy = torch.tensor([[0.020]])      # 반경 안 (0.020 < 0.041)
        prev_z = torch.tensor([[0.120]])  # 림 위 (> 0.100)
        curr_z = torch.tensor([[0.050]])  # 림 아래 (<= 0.100)

        result = _compute_mouth_crossed(xy, prev_z, curr_z)
        assert result[0, 0].item() is True

    def test_crossing_not_detected_when_bead_stays_above_rim(self):
        """비드가 림 위에 머물면 crossing 없음."""
        xy = torch.tensor([[0.020]])
        prev_z = torch.tensor([[0.150]])
        curr_z = torch.tensor([[0.110]])  # 여전히 림 위

        result = _compute_mouth_crossed(xy, prev_z, curr_z)
        assert result[0, 0].item() is False

    def test_crossing_not_detected_when_bead_outside_radius(self):
        """비드가 반경 밖이면 z 조건이 맞아도 crossing 없음."""
        xy = torch.tensor([[0.060]])      # 반경 밖 (0.060 > 0.041)
        prev_z = torch.tensor([[0.120]])
        curr_z = torch.tensor([[0.050]])

        result = _compute_mouth_crossed(xy, prev_z, curr_z)
        assert result[0, 0].item() is False

    def test_crossing_not_detected_when_bead_enters_from_side(self):
        """비드가 측면으로 진입 (항상 z <= 0.100): prev_z도 <= 0.100이면 crossing 없음.

        이것은 의도된 동작 — pour 태스크에서 비드는 위에서 낙하해야 하므로
        lateral entry는 카운트하지 않아도 무방.
        """
        xy = torch.tensor([[0.020]])
        prev_z = torch.tensor([[0.080]])  # 이미 림 아래
        curr_z = torch.tensor([[0.040]])

        result = _compute_mouth_crossed(xy, prev_z, curr_z)
        assert result[0, 0].item() is False, (
            "lateral entry (prev_z < mouth_z)는 crossing으로 카운트하지 않아야 함"
        )

    def test_crossing_at_exact_rim_boundary(self):
        """경계값: prev_z 정확히 mouth_z (> 조건 불만족 → crossing 없음)."""
        xy = torch.tensor([[0.020]])
        prev_z = torch.tensor([[0.100]])  # 경계: > mouth_z 불만족
        curr_z = torch.tensor([[0.050]])

        result = _compute_mouth_crossed(xy, prev_z, curr_z)
        assert result[0, 0].item() is False, "prev_z == mouth_z는 crossing 아님 (strict >)"

    def test_crossing_detected_when_curr_z_exactly_at_mouth(self):
        """경계값: curr_z 정확히 mouth_z (== 포함) → crossing 감지."""
        xy = torch.tensor([[0.020]])
        prev_z = torch.tensor([[0.110]])
        curr_z = torch.tensor([[0.100]])  # 경계: <= 만족

        result = _compute_mouth_crossed(xy, prev_z, curr_z)
        assert result[0, 0].item() is True

    def test_crossing_cumulative_with_or_operation(self):
        """crossing 플래그는 누적 OR → 한번 True면 계속 True."""
        xy = torch.tensor([[0.020]])
        prev_z = torch.tensor([[0.110]])
        curr_z = torch.tensor([[0.050]])

        crossed = torch.zeros(1, 1, dtype=torch.bool)
        crossed |= _compute_mouth_crossed(xy, prev_z, curr_z)
        assert crossed[0, 0].item() is True

        # 다음 프레임: 비드가 이미 컵 안에 있어 crossing 조건 불만족
        crossed |= _compute_mouth_crossed(
            torch.tensor([[0.020]]),
            torch.tensor([[0.050]]),  # prev_z <= mouth_z → no new crossing
            torch.tensor([[0.020]]),
        )
        assert crossed[0, 0].item() is True, "이전 crossing은 유지돼야 함"

    def test_initial_prev_z_10_triggers_crossing_on_first_frame(self):
        """첫 프레임 초기화 동작: prev_z=10.0이면 rim 아래 비드는 항상 crossing.

        이것은 _prev_bead_target_local_z 초기값(10.0)의 의도된 동작.
        warmstart 시 beads는 source cup(target에서 멀리) → xy > inner_radius → 실제론 무해.
        """
        xy_near = torch.tensor([[0.020]])   # 반경 안
        prev_z_init = torch.tensor([[PREV_Z_INIT]])  # 10.0
        curr_z = torch.tensor([[0.050]])    # 림 아래

        result = _compute_mouth_crossed(xy_near, prev_z_init, curr_z)
        assert result[0, 0].item() is True, (
            "초기 prev_z=10.0, 반경 안, z<=mouth_z → crossing True (초기화 동작)"
        )

    def test_initial_prev_z_safe_when_bead_outside_radius(self):
        """warmstart 정상 케이스: source cup의 beads는 target에서 멀리 → crossing 없음."""
        xy_far = torch.tensor([[0.500]])    # 반경 밖 (source cup은 target에서 0.5m 이상)
        prev_z_init = torch.tensor([[PREV_Z_INIT]])
        curr_z = torch.tensor([[0.050]])

        result = _compute_mouth_crossed(xy_far, prev_z_init, curr_z)
        assert result[0, 0].item() is False, "source cup beads는 target radius 밖 → safe"

    def test_multiple_beads_batch(self):
        """배치 케이스: 20개 비드 중 일부만 crossing."""
        n_beads = 20
        xy = torch.full((1, n_beads), 0.020)   # 모두 반경 안
        prev_z = torch.full((1, n_beads), 0.110)
        curr_z = torch.full((1, n_beads), 0.050)

        # 절반은 반경 밖으로 설정
        xy[0, 10:] = 0.060

        result = _compute_mouth_crossed(xy, prev_z, curr_z)
        assert result[0, :10].all().item() is True
        assert result[0, 10:].any().item() is False
        assert result.float().mean().item() == pytest.approx(0.5)


# ===========================================================================
# 3. bead_in_target 감지 로직 검증
# ===========================================================================

def _compute_bead_in_target(
    bead_xy_to_target: torch.Tensor,  # (N, K)
    pos_z: torch.Tensor,               # (N, K)  컵 로컬 z
    inner_radius: float = TARGET_INNER_RADIUS,
    z_min: float = TARGET_INSIDE_Z_MIN,
    z_max: float = TARGET_INSIDE_Z_MAX,
) -> torch.Tensor:
    """pour_right_env.py 817-821줄과 동일한 in_target 판정."""
    return (
        (bead_xy_to_target <= inner_radius)
        & (pos_z >= z_min)
        & (pos_z <= z_max)
    )


class TestBeadInTarget:
    """bead가 target cup 내부에 있는지 판정 로직 검증."""

    def test_bead_inside_cup_detected(self):
        """컵 중심 + z 범위 안 → in_target=True."""
        xy = torch.tensor([[0.020]])
        z = torch.tensor([[0.000]])  # 컵 중간

        result = _compute_bead_in_target(xy, z)
        assert result[0, 0].item() is True

    def test_bead_at_rim_level_is_in_target(self):
        """z = target_inside_z_max (0.100, 림 높이) → in_target=True (경계 포함)."""
        xy = torch.tensor([[0.020]])
        z = torch.tensor([[0.100]])

        result = _compute_bead_in_target(xy, z)
        assert result[0, 0].item() is True

    def test_bead_above_rim_not_in_target(self):
        """z > 0.100 (림 위) → in_target=False."""
        xy = torch.tensor([[0.020]])
        z = torch.tensor([[0.101]])  # 림보다 1mm 위

        result = _compute_bead_in_target(xy, z)
        assert result[0, 0].item() is False

    def test_bead_at_bottom_is_in_target(self):
        """z = target_inside_z_min (-0.070) → in_target=True (경계 포함)."""
        xy = torch.tensor([[0.020]])
        z = torch.tensor([[-0.070]])

        result = _compute_bead_in_target(xy, z)
        assert result[0, 0].item() is True

    def test_bead_below_bottom_not_in_target(self):
        """z < -0.070 (컵 바닥 아래) → in_target=False."""
        xy = torch.tensor([[0.020]])
        z = torch.tensor([[-0.071]])

        result = _compute_bead_in_target(xy, z)
        assert result[0, 0].item() is False

    def test_bead_outside_radius_not_in_target(self):
        """xy > inner_radius → in_target=False (z가 맞아도)."""
        xy = torch.tensor([[0.042]])  # 반경보다 1mm 밖
        z = torch.tensor([[0.000]])

        result = _compute_bead_in_target(xy, z)
        assert result[0, 0].item() is False

    def test_bead_at_radius_boundary_is_in_target(self):
        """xy == inner_radius (0.041) → in_target=True (경계 포함, <=)."""
        xy = torch.tensor([[0.041]])
        z = torch.tensor([[0.000]])

        result = _compute_bead_in_target(xy, z)
        assert result[0, 0].item() is True

    def test_bead_fraction_from_mixed_batch(self):
        """20개 비드 중 5개만 컵 안 → fraction=0.25."""
        n_beads = 20
        xy = torch.full((1, n_beads), 0.060)  # 기본: 반경 밖
        z = torch.full((1, n_beads), 0.000)
        xy[0, :5] = 0.020                      # 5개만 반경 안

        result = _compute_bead_in_target(xy, z)
        fraction = result.float().mean().item()
        assert fraction == pytest.approx(0.25)


# ===========================================================================
# 4. crossing과 in_target의 관계 검증
# ===========================================================================

class TestCrossingVsInTarget:
    """mouth_crossed와 bead_in_target의 관계 — 독립적임을 확인."""

    def test_in_target_does_not_require_crossing(self):
        """bead_in_target은 crossing 없이도 True 가능 (초기 배치 등).

        실제로는 warmstart에서 source cup에만 있으므로 문제없지만,
        두 지표가 독립적으로 동작함을 명확히 한다.
        """
        # bead가 컵 안 (in_target=True)
        xy = torch.tensor([[0.020]])
        z = torch.tensor([[0.000]])
        in_target = _compute_bead_in_target(xy, z)
        assert in_target[0, 0].item() is True

        # 그런데 이전 프레임도 컵 안이었다면 crossing은 없음
        prev_z = torch.tensor([[0.050]])  # 이미 컵 안 (< mouth_z)
        crossed = _compute_mouth_crossed(xy, prev_z, z)
        assert crossed[0, 0].item() is False, "이미 컵 안에 있던 bead는 crossing 아님"

    def test_crossing_before_in_target(self):
        """crossing이 감지된 후 다음 프레임에 in_target이 True가 되는 정상 흐름."""
        xy = torch.tensor([[0.020]])

        # 프레임 1: 림 통과
        crossed = _compute_mouth_crossed(xy, torch.tensor([[0.110]]), torch.tensor([[0.090]]))
        assert crossed[0, 0].item() is True

        # 프레임 2: 컵 내부에 안착
        in_target = _compute_bead_in_target(xy, torch.tensor([[0.000]]))
        assert in_target[0, 0].item() is True
