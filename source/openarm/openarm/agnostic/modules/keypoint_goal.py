"""keypoint_goal — 축대칭 키포인트 · 목표열 샘플러 · 진행 추적기 · 허용오차 커리큘럼.

`agnostic/tasks/grasp_kp/DESIGN.md` §2·§8 의 구현. **torch 만** import 한다(isaaclab 금지 —
시스템 python3 pytest 로 돈다). 쿼터니언은 전부 **wxyz**, 위치는 env-local.

왜 축대칭 키포인트인가: 컵은 yaw 가 과제 무관이라 키포인트를 물체 z 축 위에만 두면
d(o,g) = max_i ‖kp_i(o) − kp_i(g)‖ 가 yaw 에 불변이고 tilt·이동에는 단조로 커진다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

KEYPOINT_AXIAL_UNIT = ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0 / 3.0), (0.0, 0.0, -1.0 / 3.0))
NUM_KEYPOINTS = len(KEYPOINT_AXIAL_UNIT)

_SENTINEL = -1.0  # closest_* 의 "아직 관측 없음" 표식(음수 거리는 존재하지 않으므로 안전)


# =============================================================================
# 쿼터니언 헬퍼 (isaaclab.utils.math 규약: wxyz, quat_mul(q1,q2)=q1⊗q2)
# =============================================================================
def _check_shape(name: str, t: torch.Tensor, last: int) -> None:
    if t.ndim < 1 or t.shape[-1] != last:
        raise ValueError(f"{name}: expected trailing dim {last}, got shape {tuple(t.shape)}")


def quat_apply(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q 로 v 를 회전. q (...,4) wxyz, v (...,3) → (...,3)."""
    _check_shape("quat_apply.q", q_wxyz, 4)
    _check_shape("quat_apply.v", v, 3)
    xyz = q_wxyz[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q_wxyz[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """해밀턴 곱 a⊗b (a 를 나중에 적용). 둘 다 (...,4) wxyz."""
    _check_shape("quat_mul.a", a, 4)
    _check_shape("quat_mul.b", b, 4)
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """angle (...,) rad · axis (...,3) → (...,4) wxyz. 축은 정규화한다(영벡터면 항등)."""
    _check_shape("quat_from_angle_axis.axis", axis, 3)
    norm = axis.norm(dim=-1, keepdim=True)
    unit = torch.where(norm > 0, axis / norm.clamp_min(1e-12), torch.zeros_like(axis))
    half = 0.5 * angle.unsqueeze(-1)
    return torch.cat((torch.cos(half), unit * torch.sin(half)), dim=-1)


def random_small_rotation(n: int, max_deg: float, device) -> torch.Tensor:
    """무작위 축 · U(−max_deg, max_deg) 각도의 회전 (n,4). max_deg==0 → 항등."""
    if max_deg < 0:
        raise ValueError(f"random_small_rotation: max_deg must be ≥ 0, got {max_deg}")
    ident = torch.zeros(n, 4, device=device)
    ident[:, 0] = 1.0
    if max_deg == 0.0:
        return ident
    axis = torch.randn(n, 3, device=device)
    angle = (torch.rand(n, device=device) * 2.0 - 1.0) * math.radians(max_deg)
    return quat_from_angle_axis(angle, axis)


# =============================================================================
# 키포인트
# =============================================================================
def keypoint_offsets(half_height: float, device) -> torch.Tensor:
    """물체 프레임 키포인트 오프셋 (4,3) = half_height · 단위 축 배열."""
    if half_height <= 0:
        raise ValueError(f"keypoint_offsets: half_height must be > 0, got {half_height}")
    return torch.tensor(KEYPOINT_AXIAL_UNIT, dtype=torch.float32, device=device) * half_height


def keypoints_world(pos: torch.Tensor, quat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """(N,3)·(N,4)·(4,3) → (N,4,3). 위치 프레임은 호출자가 통일한다(env-local)."""
    _check_shape("keypoints_world.pos", pos, 3)
    _check_shape("keypoints_world.quat", quat, 4)
    if offsets.shape != (NUM_KEYPOINTS, 3):
        raise ValueError(f"keypoints_world: offsets must be ({NUM_KEYPOINTS},3), got {tuple(offsets.shape)}")
    if pos.shape[0] != quat.shape[0]:
        raise ValueError(f"keypoints_world: N mismatch pos {pos.shape[0]} vs quat {quat.shape[0]}")
    n = pos.shape[0]
    q = quat[:, None, :].expand(n, NUM_KEYPOINTS, 4)
    off = offsets[None, :, :].expand(n, NUM_KEYPOINTS, 3)
    return pos[:, None, :] + quat_apply(q, off)


def keypoint_max_dist(kp_a: torch.Tensor, kp_b: torch.Tensor) -> torch.Tensor:
    """d(o,g) = max_i ‖kp_i(a) − kp_i(b)‖ → (N,)."""
    if kp_a.shape != kp_b.shape or kp_a.ndim != 3 or kp_a.shape[1:] != (NUM_KEYPOINTS, 3):
        raise ValueError(f"keypoint_max_dist: expected (N,{NUM_KEYPOINTS},3) pair, got {tuple(kp_a.shape)} vs {tuple(kp_b.shape)}")
    return (kp_a - kp_b).norm(dim=-1).max(dim=1).values


# =============================================================================
# 목표열
# =============================================================================
@dataclass(frozen=True)
class GoalSeqCfg:
    first_xy_range: float = 0.05
    first_z_range: tuple = (0.12, 0.20)
    first_tilt_deg: float = 0.0
    delta_distance: float = 0.08
    delta_rotation_deg: float = 0.0
    box_min: tuple = (-1e9,) * 3  # env-local 절대 박스(클램프)
    box_max: tuple = (1e9,) * 3
    success_steps: int = 10
    force_consecutive: bool = False
    max_goals: int = 50

    def __post_init__(self):
        if len(self.first_z_range) != 2 or self.first_z_range[0] > self.first_z_range[1]:
            raise ValueError(f"GoalSeqCfg.first_z_range must be (lo ≤ hi), got {self.first_z_range}")
        if len(self.box_min) != 3 or len(self.box_max) != 3:
            raise ValueError("GoalSeqCfg.box_min/box_max must be 3-tuples")
        if any(lo > hi for lo, hi in zip(self.box_min, self.box_max)):
            raise ValueError(f"GoalSeqCfg box_min {self.box_min} > box_max {self.box_max}")
        if self.success_steps < 1 or self.max_goals < 1:
            raise ValueError("GoalSeqCfg.success_steps/max_goals must be ≥ 1")


def _clamp_box(pos: torch.Tensor, cfg: GoalSeqCfg) -> torch.Tensor:
    lo = torch.tensor(cfg.box_min, dtype=pos.dtype, device=pos.device)
    hi = torch.tensor(cfg.box_max, dtype=pos.dtype, device=pos.device)
    return torch.maximum(torch.minimum(pos, hi), lo)


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo


def sample_first_goal(settled_pos: torch.Tensor, settled_quat: torch.Tensor, cfg: GoalSeqCfg):
    """리셋 시 첫 목표: settled + [U(±xy), U(±xy), U(z_lo, z_hi)], 자세 = settled (tilt 0 이면 그대로)."""
    _check_shape("sample_first_goal.settled_pos", settled_pos, 3)
    _check_shape("sample_first_goal.settled_quat", settled_quat, 4)
    n, dev = settled_pos.shape[0], settled_pos.device
    r = cfg.first_xy_range
    delta = torch.stack(
        (_uniform(n, -r, r, dev), _uniform(n, -r, r, dev), _uniform(n, cfg.first_z_range[0], cfg.first_z_range[1], dev)),
        dim=-1,
    )
    pos = _clamp_box(settled_pos + delta, cfg)
    # 왜 pre-multiply: world 프레임 기울임(물체 프레임 축이 아니라 world 축 기준)
    quat = quat_mul(random_small_rotation(n, cfg.first_tilt_deg, dev), settled_quat)
    return pos, quat


def sample_delta_goal(prev_pos: torch.Tensor, prev_quat: torch.Tensor, cfg: GoalSeqCfg):
    """다음 목표: 이전 **목표** 에서 ±delta_distance 균일 이동 → 박스 클램프, 회전은 world pre-multiply."""
    _check_shape("sample_delta_goal.prev_pos", prev_pos, 3)
    _check_shape("sample_delta_goal.prev_quat", prev_quat, 4)
    n, dev = prev_pos.shape[0], prev_pos.device
    d = cfg.delta_distance
    pos = _clamp_box(prev_pos + (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * d, cfg)
    quat = quat_mul(random_small_rotation(n, cfg.delta_rotation_deg, dev), prev_quat)
    return pos, quat


# =============================================================================
# 진행 추적기
# =============================================================================
class GoalTrackers:
    """단순 텐서 컨테이너. closest_* 는 −1 센티널로 시작(첫 관측이 기준선이 된다)."""

    def __init__(self, num_envs: int, num_tips: int, device):
        if num_envs < 1 or num_tips < 1:
            raise ValueError(f"GoalTrackers: num_envs/num_tips must be ≥ 1, got {num_envs}/{num_tips}")
        self.closest_kp = torch.full((num_envs,), _SENTINEL, device=device)
        self.closest_ft = torch.full((num_envs, num_tips), _SENTINEL, device=device)
        self.near_goal_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.successes = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prev_episode_successes = torch.zeros(num_envs, dtype=torch.long, device=device)

    def clear_goal(self, ids) -> None:
        """목표 전진: 키포인트 진행 기준선·손끝 기준선·근접 카운터 초기화."""
        self.closest_kp[ids] = _SENTINEL
        self.closest_ft[ids] = _SENTINEL
        self.near_goal_steps[ids] = 0

    def full_reset(self, ids) -> None:
        """에피소드 리셋: 직전 에피소드 성공수를 커리큘럼 지표로 남기고 전부 초기화."""
        self.prev_episode_successes[ids] = self.successes[ids]
        self.successes[ids] = 0
        self.clear_goal(ids)


def progress_delta(curr: torch.Tensor, closest: torch.Tensor):
    """(delta, new_closest). closest<0 센티널 → delta 0·new=curr; 아니면 delta=clamp(closest−curr, min 0)·new=min."""
    if curr.shape != closest.shape:
        raise ValueError(f"progress_delta: shape mismatch curr {tuple(curr.shape)} vs closest {tuple(closest.shape)}")
    fresh = closest < 0
    delta = torch.where(fresh, torch.zeros_like(curr), (closest - curr).clamp_min(0.0))
    new_closest = torch.where(fresh, curr, torch.minimum(closest, curr))
    return delta, new_closest


def update_near_goal(kp_dist: torch.Tensor, tol: float, trackers: GoalTrackers, cfg: GoalSeqCfg):
    """near_goal = d ≤ tol; near_goal_steps 갱신(force_consecutive 면 놓치면 0). is_success = steps ≥ success_steps."""
    if kp_dist.shape != trackers.near_goal_steps.shape:
        raise ValueError(f"update_near_goal: kp_dist {tuple(kp_dist.shape)} vs trackers {tuple(trackers.near_goal_steps.shape)}")
    near_goal = kp_dist <= tol
    steps = trackers.near_goal_steps + near_goal.long()
    if cfg.force_consecutive:
        steps = torch.where(near_goal, steps, torch.zeros_like(steps))
    trackers.near_goal_steps = steps
    is_success = steps >= cfg.success_steps
    return near_goal, is_success


# =============================================================================
# 허용오차 커리큘럼
# =============================================================================
class ToleranceCurriculum:
    """interval 프레임마다 mean(prev_episode_successes) ≥ threshold 면 tol ← max(tol·factor, floor)."""

    def __init__(self, start: float, floor: float, factor: float = 0.9, interval: int = 3000, success_threshold: float = 2.0):
        if not (0.0 < floor <= start):
            raise ValueError(f"ToleranceCurriculum: need 0 < floor ≤ start, got floor={floor} start={start}")
        if not (0.0 < factor < 1.0) or interval < 1:
            raise ValueError(f"ToleranceCurriculum: factor∈(0,1), interval≥1; got {factor}/{interval}")
        self._tol = float(start)
        self._floor = float(floor)
        self._factor = float(factor)
        self._interval = int(interval)
        self._threshold = float(success_threshold)
        self._frames = 0

    @property
    def tol(self) -> float:
        return self._tol

    @property
    def frames(self) -> int:
        return self._frames

    def update(self, prev_episode_successes: torch.Tensor) -> bool:
        """매 프레임 호출. tol 이 실제로 줄었을 때만 True."""
        self._frames += 1
        if self._frames % self._interval != 0:
            return False
        if float(prev_episode_successes.float().mean()) < self._threshold or self._tol <= self._floor:
            return False
        self._tol = max(self._tol * self._factor, self._floor)
        return True
