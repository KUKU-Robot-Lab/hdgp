"""progress-only 보상 7항 + hand_floor 기하 벌점 + cmd_rate 지령변화 벌점 (순수 torch, isaaclab 금지).

SimToolReal 식 보상: 접촉 센서 항이 **0개**다. 신호는 전부 기하(거리 감소·높이·목표 근접)와
속도 벌점이라 실기에서 그대로 계산할 수 있다. 계약은 `tasks/grasp_kp/DESIGN.md` §3·§8.

    total, terms, out = compute_progress_reward(obj_z=..., ..., cfg=ProgressRewardCfg())

`terms` 는 `PROGRESS_REWARD_TERMS` 순서 그대로의 dict(로깅용), `out` 은 env 가 다음 스텝에
되먹여야 하는 상태(lifted 래치·최소거리)다. 모듈은 상태를 갖지 않는다 — 되먹임은 env 몫.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PROGRESS_REWARD_TERMS = (
    "fingertip_progress",
    "lift",
    "lift_bonus",
    "keypoint_progress",
    "goal_bonus",
    "arm_vel",
    "hand_vel",
    "hand_floor",
    "cmd_rate",
)

# 왜: 손끝 진행량 clamp 상한(10 m) — 사실상 무한이지만 NaN/inf 값이 폭주하는 것만 막는다.
FT_PROGRESS_CLIP = 10.0
# 왜: 키포인트 진행량 clamp 상한(100 m) — 위와 동일 목적(SimToolReal 상수 그대로).
KP_PROGRESS_CLIP = 100.0


@dataclass(frozen=True)
class ProgressRewardCfg:
    """DESIGN §3 의 계수. env cfg 는 이 필드를 접두사 `rw_` 로 그대로 노출한다."""

    ft_scale: float = 50.0
    lift_scale: float = 20.0
    lift_base: float = 0.05
    lift_clip: float = 0.5
    lift_bonus: float = 300.0
    lift_latch_height: float = 0.10
    kp_scale: float = 200.0
    goal_bonus: float = 1000.0
    success_steps: int = 10
    arm_vel_scale: float = 0.03
    hand_vel_scale: float = 0.003
    hand_floor_penalty: float = 10.0
    hand_floor_z: float = 0.215
    hand_floor_max: float = 5.0
    # ★09.07 리프트 후 지령 변화 벌점 = −scale · cmd_rate · lifted. 0 = 끔. 측도(cmd_rate)는 env 가 준다 —
    #   A: 리미터 **전** 원지령 변화 / 리미터 상한(위치·회전 평균). 1.0 = 리미터에 딱 맞춤, 비유계(작동점 ≈10).
    #   B: 팔 액션 1차 차분 RMS / 2 ∈ [0, 1]. 크기는 트랙 cfg(`rw_cmd_rate_scale`)가 작동점에 맞춰 정한다.
    #   리프트 전엔 0 — 억제 항을 처음부터 켜면 탐색이 죽는다(fab_test14: σ −41%, 리프트 350 epoch 지연).
    cmd_rate_scale: float = 0.1

    def __post_init__(self):
        if self.success_steps < 1:
            raise ValueError(f"success_steps must be ≥ 1, got {self.success_steps}")
        if self.hand_floor_max < 0.0 or self.lift_clip < 0.0:
            raise ValueError("hand_floor_max / lift_clip must be non-negative")
        if self.cmd_rate_scale < 0.0:
            raise ValueError(f"cmd_rate_scale must be non-negative, got {self.cmd_rate_scale}")


def _progress_delta(curr: torch.Tensor, closest: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`keypoint_goal.progress_delta` 의 사본(동일 의미) — 동시 작성 중인 파일과의 import 경합 회피.

    closest < 0 (센티널) → delta 0, new = curr. 아니면 delta = clamp(closest − curr, min 0),
    new = min(closest, curr). (N,) 과 (N,K) 모두 지원. 후퇴(curr > closest)는 0 — 진행만 보상.
    """
    if curr.shape != closest.shape:
        raise ValueError(f"progress_delta shape mismatch: curr {tuple(curr.shape)} vs closest {tuple(closest.shape)}")
    sentinel = closest < 0.0
    delta = torch.where(sentinel, torch.zeros_like(curr), torch.clamp(closest - curr, min=0.0))
    new_closest = torch.where(sentinel, curr, torch.minimum(closest, curr))
    return delta, new_closest


def _check_shapes(
    n: int,
    *,
    obj_z, settled_z, lifted_prev, ft_dist, closest_ft, kp_dist, closest_kp,
    near_goal, arm_qd, hand_qd, hand_z_min, cmd_rate,
) -> None:
    """부팅 시 차원 불일치를 시끄럽게 잡는다(조용한 브로드캐스트 금지)."""
    vec = dict(obj_z=obj_z, settled_z=settled_z, lifted_prev=lifted_prev, kp_dist=kp_dist,
               closest_kp=closest_kp, near_goal=near_goal, hand_z_min=hand_z_min, cmd_rate=cmd_rate)
    for name, t in vec.items():
        if t.shape != (n,):
            raise ValueError(f"{name} must be ({n},), got {tuple(t.shape)}")
    for name, t in dict(ft_dist=ft_dist, closest_ft=closest_ft, arm_qd=arm_qd, hand_qd=hand_qd).items():
        if t.dim() != 2 or t.shape[0] != n:
            raise ValueError(f"{name} must be ({n}, K), got {tuple(t.shape)}")
    if ft_dist.shape != closest_ft.shape:
        raise ValueError(f"ft_dist {tuple(ft_dist.shape)} vs closest_ft {tuple(closest_ft.shape)}")
    if lifted_prev.dtype != torch.bool or near_goal.dtype != torch.bool:
        raise TypeError("lifted_prev / near_goal must be bool tensors")


def compute_progress_reward(
    *,
    obj_z: torch.Tensor,
    settled_z: torch.Tensor,
    lifted_prev: torch.Tensor,
    ft_dist: torch.Tensor,
    closest_ft: torch.Tensor,
    kp_dist: torch.Tensor,
    closest_kp: torch.Tensor,
    near_goal: torch.Tensor,
    arm_qd: torch.Tensor,
    hand_qd: torch.Tensor,
    hand_z_min: torch.Tensor,
    cmd_rate: torch.Tensor,
    cfg: ProgressRewardCfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """DESIGN §3 보상. 반환 (total (N,), terms dict, out dict).

    - lifted 는 sticky 래치(에피소드 리셋에서만 해제 — env 가 lifted_prev 를 False 로 준다).
    - 리프트 전 항(fingertip_progress·lift)은 lifted 에서 0, keypoint_progress 는 lifted 전 0.
    - closest_* 되먹임: fingertip 은 리프트 후에도 계속 갱신(값은 무해, 게이트가 0 으로 만든다).
    - cmd_rate(N,) ≥ 0 는 env 가 준 정규화 지령 변화율 — lifted 에서만 −cmd_rate_scale 배로 벌한다.
    """
    n = obj_z.shape[0]
    _check_shapes(n, obj_z=obj_z, settled_z=settled_z, lifted_prev=lifted_prev, ft_dist=ft_dist,
                  closest_ft=closest_ft, kp_dist=kp_dist, closest_kp=closest_kp, near_goal=near_goal,
                  arm_qd=arm_qd, hand_qd=hand_qd, hand_z_min=hand_z_min, cmd_rate=cmd_rate)

    dz = obj_z - settled_z
    lifted = (dz > cfg.lift_latch_height) | lifted_prev
    just_lifted = lifted & ~lifted_prev
    not_lifted = (~lifted).float()
    lifted_f = lifted.float()

    ft_delta, new_closest_ft = _progress_delta(ft_dist, closest_ft)
    kp_delta, new_closest_kp = _progress_delta(kp_dist, closest_kp)

    terms = {
        "fingertip_progress": cfg.ft_scale * ft_delta.clamp(0.0, FT_PROGRESS_CLIP).sum(dim=-1) * not_lifted,
        "lift": cfg.lift_scale * (cfg.lift_base + dz).clamp(0.0, cfg.lift_clip) * not_lifted,
        "lift_bonus": cfg.lift_bonus * just_lifted.float(),
        "keypoint_progress": cfg.kp_scale * kp_delta.clamp(0.0, KP_PROGRESS_CLIP) * lifted_f,
        "goal_bonus": (cfg.goal_bonus / cfg.success_steps) * near_goal.float(),
        "arm_vel": -cfg.arm_vel_scale * arm_qd.abs().sum(dim=-1),
        "hand_vel": -cfg.hand_vel_scale * hand_qd.abs().sum(dim=-1),
        # 왜: 센서 없이 상판 관통을 벌하는 기하 항 — 상판(hand_floor_z) 아래 깊이에 비례, 상한 hand_floor_max.
        "hand_floor": -(cfg.hand_floor_penalty * torch.relu(cfg.hand_floor_z - hand_z_min)).clamp(max=cfg.hand_floor_max),
        # 왜 상한이 없나: 작동점(≈10×)에서 clamp 되면 항이 상수가 되어 μ 에 기울기가 없다(reward-clamp-kills-gradient).
        #   유계인 것은 lifted 게이트와 할인(γ 0.99 → 합 ≤ 100·scale·작동점)뿐이다 — 크기는 트랙 cfg 가 Check 1 로 잰다.
        "cmd_rate": -cfg.cmd_rate_scale * cmd_rate.clamp(min=0.0) * lifted_f,
    }
    if tuple(terms) != PROGRESS_REWARD_TERMS:
        raise RuntimeError(f"term order drifted: {tuple(terms)} != {PROGRESS_REWARD_TERMS}")

    # 왜: NaN 물리값(폭발 env)이 total 을 오염시켜 PPO 전체를 죽이지 않게 — abnormal 종료는 env 가 따로 한다.
    total = torch.nan_to_num(torch.stack(list(terms.values()), dim=0).sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
    out = {"lifted": lifted, "just_lifted": just_lifted, "closest_ft": new_closest_ft, "closest_kp": new_closest_kp}
    return total, terms, out
