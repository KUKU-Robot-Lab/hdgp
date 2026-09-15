"""보상 게이트용 에피소드 단계 래치 — 순수 torch (Isaac 불요).

★09.15 사용자: "기본 핸드 자세에서 컵으로 접근 → 접근한 상태에서 인벨롭 파지 → 리프트하면서 조임(이번엔 제외)".
  여섯 라운드 동안 생성 보상이 단계를 건너뛴 자세(엄지만 대기·기울이며 테이블 누르기·손가락 등으로 기대기)에 점수를 줬다.
  env 가 단계 완료를 **에피소드 래치**로 만들어 ctx(`approach_done`·`envelope_done`)로 넘기고, 생성기는 그 순서대로 보상한다.
★09.15 사용자 "컵에 다가가는 palm_ee_x · 손가락 방향(palm_ee_z)" — 리셋 자세가 이미 법선 +y · 손가락 +x 라 접근은 이 방향을 유지한다.
★09.15 23:2x 사용자 결정 "C자 사전파지로 재정의" — 기본 손 자세(thumb_2 대향 잠금 −1.57, 나머지 0)의 엄지는 손바닥 법선 방향으로
  손바닥면보다 **118 mm** 앞으로 곧게 뻗는다(URDF collider FK). 그래서 "손바닥 중심 2 cm + 기본 자세 + 무접촉"은 성립하지 않았고
  fj_stage_i01 의 접근 래치는 엄지로 컵을 누르며 섰다(e700–720 엄지 접촉 0.33–0.45). 접근 완료 = 컵이 엄지와 네 손가락 사이에 든
  C자: 컵 축이 palm_ee 보다 손가락 방향으로 약 R 앞 · 손바닥면이 컵 옆면 가까이 · 시작 방향 · 기본 손 자세 · 손가락·엄지 무접촉.
★로그 퍼널(`stage_funnel.py`)과 정의가 다르다 — 퍼널은 관찰용(접근 = 간극 ≤ 5 cm), 게이트는 보상 순서용이라 더 엄격하다.
★수치는 이 파일 한 곳. `t2r/context.py` 주석이 같은 값을 영어로 적는다(`tests/test_grasp_gates.py` 가 대조).
"""

from __future__ import annotations

import torch

#: 접근 완료 — palm_ee(손바닥면, collider 면과 0.5 mm)에서 컵 옆면까지 손바닥 법선 방향 거리 상한 [m]
APPROACH_PLANE_GAP_M = 0.02
#: 접근 완료 — 같은 거리의 하한 [m]. ★09.15 서버 부팅 스모크: 상한만 두면 컵이 손등 뒤(음수 간극)여도 조건이 섰다
#:   (먼 출발 무작위 행동에서 gap 통과 0.28). 손바닥면은 컵을 뚫지 못하므로 −1 cm 아래는 "컵이 손바닥 앞이 아니다".
APPROACH_PLANE_GAP_MIN_M = -0.01
#: 접근 완료 — 컵 축이 palm_ee 보다 손가락 방향으로 `R + 이 값` 만큼 앞 [m], 하한·상한.
#:   하한: 기본 자세 엄지 collider 는 palm_ee 보다 손목 쪽(손가락 방향 −38~−10 mm)에서 뻗는다 → 컵 단면이 엄지를 5 mm 이상 비킨다.
#:   상한: 네 손가락(손가락 방향 +16~+176 mm)이 감쌀 수 있는 자리.
APPROACH_ALONG_MIN_OFFSET_M = -0.005
APPROACH_ALONG_MAX_OFFSET_M = 0.02
#: 접근 완료 — 손 방향(env-local 축 = world 축, 환경 원점은 평행이동뿐). 리셋 자세 FK(short-tl URDF)의 palm_ee x(법선) = +y ·
#:   z(손가락) = +x 와 같다 — `tests/test_grasp_gates.py` 가 시작 자세 FK 와 대조한다.
APPROACH_PALM_NORMAL_DIR = (0.0, 1.0, 0.0)
APPROACH_FINGER_DIR = (1.0, 0.0, 0.0)
#: 위 두 방향 각각의 cos 하한(≈ 45°)
APPROACH_ORIENT_MIN = 0.7
#: 접근 완료 — 움직이는 손 관절의 정규화 각(0 = 하한, 1 = 상한)이 기본 자세에서 벗어난 최대치.
#:   ★09.16 사용자 "0.3 으로 완화" — 기본 자세는 움직이는 손가락 관절 대부분이 하한(액션 a = −1)이고 탐색 σ = 1 이라 0.15 는
#:   13관절 동시 통과가 i02 e240 에서 0.8 %(평균 −1 에서도 관절당 ≈ 62 % → 13관절 ≈ 0.2 %). "30 % 이상 오므리지 않았다" 로 본다.
APPROACH_POSE_TOL = 0.3
#: 접근 조건 이름 — `approach_conditions` 열 순서 = 로그 `stage/approach_ok_<이름>_now`
APPROACH_CONDITIONS = ("gap", "along", "height", "orient", "pose", "no_touch")
#: 인벨롭 완료 — 컵에 닿은 손가락 수(엄지 포함, 손가락마다 마디 하나라도) 하한
ENVELOPE_MIN_DIGITS = 4
#: 인벨롭 완료 — 조건이 연속으로 유지돼야 하는 스텝 수
ENVELOPE_HOLD_STEPS = 5
#: "닿았다"로 셀 컵 접촉력 [N]
TOUCH_N = 0.1


def c_pregrasp_geometry(palm_pos: torch.Tensor, palm_normal: torch.Tensor, finger_dir: torch.Tensor,
                        cup_pos: torch.Tensor, cup_axis: torch.Tensor, cup_radius: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """C자 사전파지 기하 (각 (N,) [m]).

    · plane_gap    = (컵 축 − palm_ee)·법선 − R — 손바닥면에서 컵 옆면까지(축에 수직 성분으로 잰다)
    · along_offset = (컵 축 − palm_ee)·손가락 방향 − R — 컵 단면의 손목 쪽 끝이 palm_ee 보다 손가락 방향으로 얼마나 앞인가
    · height       = palm_ee 의 띠 중심(cup_pos) 기준 축 방향 높이
    """
    v = cup_pos - palm_pos
    axial = (v * cup_axis).sum(-1)
    v_perp = v - axial.unsqueeze(-1) * cup_axis
    plane_gap = (v_perp * palm_normal).sum(-1) - cup_radius
    along_offset = (v_perp * finger_dir).sum(-1) - cup_radius
    return plane_gap, along_offset, -axial


def hand_orientation(palm_normal: torch.Tensor, finger_dir: torch.Tensor) -> torch.Tensor:
    """min(법선·+y, 손가락·+x) (N,). 1 = 시작 자세 방향 그대로."""
    def _dir(d: tuple[float, float, float]) -> torch.Tensor:
        return torch.tensor(d, dtype=palm_normal.dtype, device=palm_normal.device)

    normal = (palm_normal * _dir(APPROACH_PALM_NORMAL_DIR)).sum(-1)
    finger = (finger_dir * _dir(APPROACH_FINGER_DIR)).sum(-1)
    return torch.minimum(normal, finger)


def pose_deviation(hand_q_norm: torch.Tensor, default_q_norm: torch.Tensor,
                   movable: torch.Tensor) -> torch.Tensor:
    """움직이는 관절만 본 |정규화 각 − 기본 자세| 최대치 (N,). 잠긴 관절(폭 ≤ 0.05 rad)은 정규화가 노이즈를 키워 뺀다."""
    dev = (hand_q_norm - default_q_norm.unsqueeze(0)).abs()
    return torch.where(movable.unsqueeze(0), dev, torch.zeros_like(dev)).amax(dim=-1)


def approach_conditions(*, plane_gap: torch.Tensor, along_offset: torch.Tensor, height: torch.Tensor,
                        half_height: torch.Tensor, orient: torch.Tensor, pose_dev: torch.Tensor,
                        digit_touch: torch.Tensor) -> torch.Tensor:
    """접근 조건별 통과 (N,6) bool — 열 순서 = `APPROACH_CONDITIONS`. digit_touch (N,) = 손가락·엄지 마디 중 하나라도 컵에 닿음."""
    gap_ok = (plane_gap >= APPROACH_PLANE_GAP_MIN_M) & (plane_gap <= APPROACH_PLANE_GAP_M)
    along_ok = (along_offset >= APPROACH_ALONG_MIN_OFFSET_M) & (along_offset <= APPROACH_ALONG_MAX_OFFSET_M)
    return torch.stack([gap_ok, along_ok, height.abs() <= half_height,
                        orient >= APPROACH_ORIENT_MIN, pose_dev <= APPROACH_POSE_TOL, ~digit_touch], dim=1)


def update_gates(approach: torch.Tensor, envelope_count: torch.Tensor, envelope: torch.Tensor, *,
                 approach_ok: torch.Tensor, palm_touch: torch.Tensor, finger_touch: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """이 스텝 상태로 래치를 갱신한 **새** 텐서 (approach (N,) bool, envelope_count (N,) long, envelope (N,) bool).

    approach_ok (N,) = `approach_conditions(...).all(dim=1)`. finger_touch (N,5) bool — 손가락 0 = 엄지.
    인벨롭은 이번 스텝까지 켜진 접근 래치 뒤에만 센다.
    """
    approach_new = approach | approach_ok
    holding = (approach_new & palm_touch & finger_touch[:, 0]
               & (finger_touch.sum(dim=1) >= ENVELOPE_MIN_DIGITS))
    count_new = torch.where(holding, envelope_count + 1, torch.zeros_like(envelope_count))
    envelope_new = envelope | (count_new >= ENVELOPE_HOLD_STEPS)
    return approach_new, count_new, envelope_new
