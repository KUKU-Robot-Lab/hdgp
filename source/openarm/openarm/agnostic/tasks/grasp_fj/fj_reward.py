"""B(full-joint) 전용 보상 — `modules/progress_reward.py` 의 **포크**(순수 torch, isaaclab 금지).

★왜 공유하지 않나(09.08 사용자 확정): A(`grasp_kp`, fabric/palm 공간)와 B(`grasp_fj`,
관절 직접 제어)는 **제어 방식이 다르다**. 같은 모듈에 스위치를 쌓으면 한쪽을 고칠 때마다
다른 쪽의 산술을 안 건드렸음을 매번 증명해야 한다. 트랙별로 소유하고 갈라지게 둔다.

**포크 시점(09.08)의 divergence 는 `goal_bonus` 한 항뿐이다.**

    A: (goal_bonus / success_steps) · near_goal      near_goal 스텝마다 나눠 지급
    B:  goal_bonus · is_success                      성공 순간 **1회 전액**

근거: SimToolReal `env.py:2656-2659` 의 `forceConsecutiveNearGoalSteps` 분기가 그렇다.
목표당 총액은 같다(1000 = 10 × 100). 바뀌는 것은 분포다 — 더 뾰족하고, 연속 10회에
도달하지 못한 near_goal 스텝에는 **한 푼도 안 준다**. 그래서 이 항은 `goal_force_consecutive`
와 **짝**이다(누적 판정에서는 성공 순간이 목표 이탈 뒤에 올 수 있어 의미가 없다).
cfg 가 부팅에서 그 짝을 대조한다(`grasp_fj_env_cfg._validate_fj_fields`).

나머지 8항은 포크 시점에 A 와 **수치가 같아야** 한다 — `tests/test_fj_reward.py` 가 고정한다.

    total, terms, out = compute_fj_reward(obj_z=..., is_success=..., cfg=FJRewardCfg())

`terms` 는 `FJ_REWARD_TERMS` 순서 그대로의 dict(로깅용), `out` 은 env 가 다음 스텝에
되먹여야 하는 상태(lifted 래치·최소거리)다. 모듈은 상태를 갖지 않는다 — 되먹임은 env 몫.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FJ_REWARD_TERMS = (
    "fingertip_progress",
    "lift",
    "lift_bonus",
    "keypoint_progress",
    "goal_bonus",
    "arm_vel",
    "hand_vel",
    "hand_floor",
    "cmd_rate",
    "wrap",
)

# 왜: 손끝 진행량 clamp 상한(10 m) — 사실상 무한이지만 NaN/inf 값이 폭주하는 것만 막는다.
FT_PROGRESS_CLIP = 10.0
# 왜: 키포인트 진행량 clamp 상한(100 m) — 위와 동일 목적(SimToolReal 상수 그대로).
KP_PROGRESS_CLIP = 100.0


@dataclass(frozen=True)
class FJRewardCfg:
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
    # ★09.08 지급 방식은 **성공 술어와 짝**이다. cfg 가 `goal_force_consecutive` 에서 유도한다.
    #   True(연속)  → 성공 순간 1회 전액. SimToolReal env.py:2656-2659 의 forceConsecutive 분기.
    #   False(누적) → A 와 같은 분할. 누적 판정에서 1회성을 쓰면 10회를 채우기 전까지 near_goal
    #     보상이 **0** 이라 신호가 훨씬 성겨진다 — 술어만 보고 지급을 안 맞추면 조용히 그렇게 된다.
    #   ★이 짝이 필요한 이유(09.08 감사): 형제 트랙 `grasp_fj_rh` 가 이 모듈을 상속하는데
    #     `goal_force_consecutive` 는 A 기본값(False)을 쓴다. 짝이 없으면 그 트랙의 보상 밀도가
    #     조용히 바뀐다. 방화벽 테스트는 **필드 배치**만 봐서 이 메서드 포획을 못 잡았다.
    goal_one_shot: bool = False
    arm_vel_scale: float = 0.03
    hand_vel_scale: float = 0.003
    hand_floor_penalty: float = 10.0
    hand_floor_z: float = 0.215
    hand_floor_max: float = 5.0
    # ★09.07 리프트 후 지령 변화 벌점 = −scale · cmd_rate · lifted. 0 = 끔. 측도(cmd_rate)는 env 가 준다 —
    #   B: 팔 액션 1차 차분 RMS / 2 ∈ [0, 1]. 크기는 트랙 cfg(`rw_cmd_rate_scale`)가 작동점에 맞춰 정한다.
    #   리프트 전엔 0 — 억제 항을 처음부터 켜면 탐색이 죽는다(fab_test14: σ −41%, 리프트 350 epoch 지연).
    # ★★09.09 **감쌈 보상**(기본 0.0 = 끔 — 켜지 않은 런은 현행과 비트 동일).
    #   왜 필요한가(09.09 ep_3200 계측): 중간마디 `_3` 는 **지령 자체가** 0.387 이고 상한 근처까지
    #   시킨 시간이 index_3 1.0% · middle_3 1.2% 뿐이다. 보상 10항 중 접촉·감쌈 항이 0개라
    #   쐐기로 끼워 들어도 goal_bonus 는 똑같이 나오고, 손을 움직이면 hand_vel 벌점만 붙는다.
    #   즉 **감싸면 손해**인 구조였다.
    #   왜 접촉 센서를 안 쓰나(09.09 사용자 확정): obs 를 바꾸는 접촉 센서는 쓰지 않는다.
    #   fj 는 접촉 센서를 아예 만들지 않으므로(09.06 확정) 순수 **기하**로 간다 —
    #   실측 굴곡(뿌리 `_2` + 중간 `_3` 의 정규화 관절각)에 준다.
    #   ★**실측**에 준다(지령 아님). 지령에 주면 정책이 "시키기만" 하고 끝난다 — 이번 실패의 재현이다.
    #   ★근접 게이트 필수(reward-audit Check 2): 없으면 **허공에서 주먹만 쥐어도** 만점이다
    #     (`_hand_blocked` 주석이 같은 상황을 적어 뒀다). 접근 구간 파괴(Check 4)도 이걸로 막는다.
    #   ★상한: curl ∈ [0,1] 이라 scale 이 곧 스텝당 상한이다. 2.0 이면 goal_bonus 33.7/step 의 6%.
    wrap_scale: float = 0.0
    # ★★09.11 — 게이트+굴곡에서 **표면 기준 exp 커널**로 교체.
    #   옛 정의 `curl × (ft_dist.mean < 0.06)` 은 두 군데가 틀렸다:
    #     ① `ft_dist` 는 손끝→물체 **원점** 거리다. 그 기울기는 표면 법선(벽을 밀어넣는
    #        쪽)을 가리킨다 — 원통을 감싸는 것은 접선 방향이라 **중심 거리로는 감쌈을
    #        표현할 수 없다**. fj_g1/g2 283 epoch 동안 게이트가 한 번도 안 열렸다(항 기여 0.0002%).
    #     ② `curl` 은 손 자세만 본다 — 물체가 손 **밖**에 있어도 만점이다.
    #   새 정의는 마디마다 표면까지의 여유를 재고 exp 로 감쇠한다. 표면에서 1.0 으로
    #   포화하므로 **밀어넣을 이득이 없다**.
    #   ★z 는 기울기를 주지 않는다(09.11 사용자 확정): z 까지 당기면 마디가 전부 물체
    #     중간 높이로 모여 인벨롭이 무너진다. 파지 띠(±H) 안에서는 z 항이 정확히 1.0 이고,
    #     띠 **밖으로 나간 양**에만 감쇠가 걸린다.
    wrap_tau_xy: float = 0.02        # m — 표면까지 수평 여유의 감쇠 길이
    wrap_tau_z: float = 0.03         # m — 파지 띠 밖으로 벗어난 양의 감쇠 길이
    cmd_rate_scale: float = 0.1
    # ★09.07 kp_a7: 래치는 sticky 라 튕겨 올라갔다 상판에 놓인 컵도 lifted 다 — 그 env 에 벌점이 500 스텝 붙어
    #   접근 자체가 죽었다. **들고 있을 때**(dz > hold_dz)만 벌한다. env 의 drop_frac 판정선과 같은 값.
    cmd_rate_hold_dz: float = 0.03

    def __post_init__(self):
        if self.success_steps < 1:
            raise ValueError(f"success_steps must be ≥ 1, got {self.success_steps}")
        if self.hand_floor_max < 0.0 or self.lift_clip < 0.0:
            raise ValueError("hand_floor_max / lift_clip must be non-negative")
        if self.cmd_rate_scale < 0.0:
            raise ValueError(f"cmd_rate_scale must be non-negative, got {self.cmd_rate_scale}")
        if self.cmd_rate_hold_dz < 0.0:
            raise ValueError(f"cmd_rate_hold_dz must be non-negative, got {self.cmd_rate_hold_dz}")


def _progress_delta(curr: torch.Tensor, closest: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """closest < 0 (센티널) → delta 0, new = curr. 아니면 delta = clamp(closest − curr, min 0),
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
    near_goal, is_success, arm_qd, hand_qd, hand_z_min, cmd_rate,
) -> None:
    """부팅 시 차원 불일치를 시끄럽게 잡는다(조용한 브로드캐스트 금지)."""
    vec = dict(obj_z=obj_z, settled_z=settled_z, lifted_prev=lifted_prev, kp_dist=kp_dist,
               closest_kp=closest_kp, near_goal=near_goal, is_success=is_success,
               hand_z_min=hand_z_min, cmd_rate=cmd_rate)
    for name, t in vec.items():
        if t.shape != (n,):
            raise ValueError(f"{name} must be ({n},), got {tuple(t.shape)}")
    for name, t in dict(ft_dist=ft_dist, closest_ft=closest_ft, arm_qd=arm_qd, hand_qd=hand_qd).items():
        if t.dim() != 2 or t.shape[0] != n:
            raise ValueError(f"{name} must be ({n}, K), got {tuple(t.shape)}")
    if ft_dist.shape != closest_ft.shape:
        raise ValueError(f"ft_dist {tuple(ft_dist.shape)} vs closest_ft {tuple(closest_ft.shape)}")
    for name, t in dict(lifted_prev=lifted_prev, near_goal=near_goal, is_success=is_success).items():
        if t.dtype != torch.bool:
            raise TypeError(f"{name} must be a bool tensor, got {t.dtype}")


def compute_fj_reward(
    *,
    obj_z: torch.Tensor,
    settled_z: torch.Tensor,
    lifted_prev: torch.Tensor,
    ft_dist: torch.Tensor,
    closest_ft: torch.Tensor,
    kp_dist: torch.Tensor,
    closest_kp: torch.Tensor,
    near_goal: torch.Tensor,
    is_success: torch.Tensor,
    arm_qd: torch.Tensor,
    hand_qd: torch.Tensor,
    hand_z_min: torch.Tensor,
    cmd_rate: torch.Tensor,
    wrap_frac: torch.Tensor | None = None,
    cfg: FJRewardCfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """B 보상. 반환 (total (N,), terms dict, out dict).

    - lifted 는 sticky 래치(에피소드 리셋에서만 해제 — env 가 lifted_prev 를 False 로 준다).
    - 리프트 전 항(fingertip_progress·lift)은 lifted 에서 0, keypoint_progress 는 lifted 전 0.
    - `goal_bonus` 는 **성공 순간 1회 전액**(A 와 유일하게 다른 항) — `is_success` 는 **필수**다.
      선택 인자로 두면 안 넘겼을 때 조용히 0 이 되어 보상 하나가 통째로 사라진다.
    - cmd_rate(N,) ≥ 0 는 env 가 준 정규화 지령 변화율 — lifted 이고 **들고 있을 때**만 벌한다.
    - hand_curl(N,) ∈ [0,1] 은 env 가 준 **실측** 뿌리·중간 마디 정규화 굴곡. `wrap_scale`
      이 0 이거나 안 넘기면 `wrap` 항은 0 이라, 켜지 않은 런은 현행과 비트 동일하다.
    """
    n = obj_z.shape[0]
    _check_shapes(n, obj_z=obj_z, settled_z=settled_z, lifted_prev=lifted_prev, ft_dist=ft_dist,
                  closest_ft=closest_ft, kp_dist=kp_dist, closest_kp=closest_kp, near_goal=near_goal,
                  is_success=is_success, arm_qd=arm_qd, hand_qd=hand_qd, hand_z_min=hand_z_min,
                  cmd_rate=cmd_rate)

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
        # ★술어와 짝을 이룬 지급(위 cfg 주석). 연속=1회 전액 · 누적=A 와 같은 분할.
        "goal_bonus": (cfg.goal_bonus * is_success.float()) if cfg.goal_one_shot
                      else (cfg.goal_bonus / cfg.success_steps) * near_goal.float(),
        "arm_vel": -cfg.arm_vel_scale * arm_qd.abs().sum(dim=-1),
        "hand_vel": -cfg.hand_vel_scale * hand_qd.abs().sum(dim=-1),
        # 왜: 센서 없이 상판 관통을 벌하는 기하 항 — 상판(hand_floor_z) 아래 깊이에 비례, 상한 hand_floor_max.
        "hand_floor": -(cfg.hand_floor_penalty * torch.relu(cfg.hand_floor_z - hand_z_min)).clamp(max=cfg.hand_floor_max),
        # 왜 상한이 없나: 작동점에서 clamp 되면 항이 상수가 되어 μ 에 기울기가 없다(reward-clamp-kills-gradient).
        "cmd_rate": -cfg.cmd_rate_scale * cmd_rate.clamp(min=0.0) * (lifted & (dz > cfg.cmd_rate_hold_dz)).float(),
        # 왜 별도 게이트가 없나: 척도가 **물체 표면에 고정**돼 있어 기하가 곧 게이트다 —
        #   허공 주먹은 수평 여유가 커서 exp 가 0 으로 죽는다(옛 근접 게이트의 역할을 대신한다).
        "wrap": (cfg.wrap_scale * wrap_frac.clamp(0.0, 1.0))
                if (cfg.wrap_scale > 0.0 and wrap_frac is not None)
                else torch.zeros_like(dz),
    }
    if tuple(terms) != FJ_REWARD_TERMS:
        raise RuntimeError(f"term order drifted: {tuple(terms)} != {FJ_REWARD_TERMS}")

    # 왜: NaN 물리값(폭발 env)이 total 을 오염시켜 PPO 전체를 죽이지 않게 — abnormal 종료는 env 가 따로 한다.
    total = torch.nan_to_num(torch.stack(list(terms.values()), dim=0).sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
    out = {"lifted": lifted, "just_lifted": just_lifted, "closest_ft": new_closest_ft, "closest_kp": new_closest_kp}
    return total, terms, out
