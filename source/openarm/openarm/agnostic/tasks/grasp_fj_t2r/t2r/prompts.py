"""프롬프트 렌더링 (grasp_fj_t2r 포크) — text2reward 형식:
역할 → 로봇·액션 설명 → 보상 구성 안내 → 환경 클래스 스텁(RewardContext 소스 그대로) → 추가 지식 → 과제·출력 규칙.

★추가 지식에는 **환경 사실만** 넣는다(단위·판정 규칙·기하 계산법). 설계 처방은 넣지 않는다 —
  생성기는 이 세션의 설계 의견 없이 과제 문장과 환경 설명만으로 쓴다(09.14 사용자 결정).
★수치 출처(09.14 코드 대조): 제어 dt 1/120×2 · k_arm 0.025 · arm/hand EMA 0.1 (`grasp_fj_env_cfg`) ·
  에피소드 600 스텝 (`fj_core_cfg.episode_length_s` 10) · 종료 out_x/out_y/object_min_z 0.15/tilt 60°/
  abnormal_qd 20 (`fj_core_cfg`) · 손 바닥 3cm (`fj_kp_cfg.hand_floor_terminate_depth`) · 목표 z 0.2125~0.28 ·
  xy ±0.05 · 기울임 0 · Δ0 (`grasp_fj_env_cfg` leaf · `fj_kp_cfg`) · 반경 29.2~43.8mm (`object_bank.shaker_sweep`) ·
  시작 손바닥→물체 150.4mm (`grasp_fj_env_cfg.start_palm_dist_band_m` 주석, 09.10 FK) ·
  액션 지연 = 최근 3개 중 env·스텝마다 무작위 1개(`fj_kp_cfg.action_delay_steps` 3 · `DelayQueue.push` randint(0, 3)).
★09.14 iter_00 은 액션 지연 문장이 들어가기 **전** 에 렌더됐다(생성기가 본 판본은 iter_00/prompt.md 그대로).
"""

from __future__ import annotations

from dataclasses import dataclass

from .context import context_stub_source
from .loader import ENTRY_NAME

#: 손 관절 순서·액션 범위 — 프로필 `tesollo_right_short_tl` 의 `hand_joint_names` 순서, 범위는
#: 09.14 부팅 로그(`[grasp_fj] 손 액션한계 full-joint`, soft limit ∩ override). 이름 순서는 테스트가 프로필과 대조한다.
HAND_JOINT_RANGES: tuple[tuple[str, float, float], ...] = (
    ("r_hj_thumb_2", -1.581, -1.561), ("r_hj_thumb_3", 0.000, 1.571), ("r_hj_thumb_4", 0.000, 1.571),
    ("r_hj_index_1", -0.010, 0.010), ("r_hj_index_2", 0.000, 2.007),
    ("r_hj_index_3", 0.000, 1.571), ("r_hj_index_4", 0.000, 1.571),
    ("r_hj_middle_1", -0.010, 0.010), ("r_hj_middle_2", 0.000, 2.007),
    ("r_hj_middle_3", 0.000, 1.571), ("r_hj_middle_4", 0.000, 1.571),
    ("r_hj_ring_1", -0.010, 0.010), ("r_hj_ring_2", 0.000, 1.920),
    ("r_hj_ring_3", 0.000, 1.571), ("r_hj_ring_4", 0.000, 1.571),
    ("r_hj_pinky_1", 0.000, 0.010), ("r_hj_pinky_2", -0.010, 0.010),
    ("r_hj_pinky_3", 0.000, 1.571), ("r_hj_pinky_4", 0.000, 1.571),
)
HAND_JOINT_NAMES: tuple[str, ...] = tuple(n for n, _, _ in HAND_JOINT_RANGES)
#: 이 폭 이하의 관절은 설계상 고정(locked)이다 — grasp_fj `_CURL_MIN_SPAN_RAD` 와 같은 값.
LOCKED_SPAN_RAD = 0.05


def _joint_table() -> str:
    rows = []
    for i, (name, lo, hi) in enumerate(HAND_JOINT_RANGES):
        tag = "locked" if hi - lo <= LOCKED_SPAN_RAD else "movable"
        rows.append(f"    actions[{7 + i:2d}] = hand index {i:2d}: {name:14s} range [{lo:+.3f}, {hi:+.3f}] rad  {tag}")
    return "\n".join(rows)


ROBOT_DESCRIPTION = """\
We control one 7-DOF OpenArm robot arm (the right arm) carrying a five-finger Tesollo DG-5F hand, \
standing at a table. An upright cylindrical cup stands on the table in front of the hand; its radius \
differs between parallel environments (29 mm to 44 mm) and is given per environment. At the start of \
an episode the palm is roughly 0.15 m from the cup. Positions are in metres in each environment's local \
frame, with +z pointing up; the table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (26,), float32)` with direct joint control. There is no \
grasp primitive, no hand synergy and no automatic finger stopping:
    actions[0:7]  = arm joint increments: each arm joint target moves by 0.025 * a rad per step and then \
passes a first-order filter (factor 0.1), so each arm joint moves at most about 0.15 rad/s.
    actions[7:26] = finger joint targets: each value is mapped linearly onto that joint's commandable \
range (a = -1 gives the lower limit, a = +1 the upper limit) and then low-pass filtered (factor 0.1). \
The order, which is also the order of ctx.hand_q / hand_q_norm / hand_target_norm / hand_qd, is:
{joint_table}
Joints marked "locked" have a range of 0.05 rad or less and effectively do not move. The thumb's first \
joint is welded in this hand; `thumb_2`, which rotates the thumb into opposition, is locked at the \
opposed angle, and the thumb flexes with `thumb_3` and `thumb_4`. On the index, middle and ring fingers \
`_1` spreads the finger sideways (locked), `_2` flexes the knuckle, and `_3` and `_4` flex the two outer \
joints. On the pinky, `pinky_1` and `pinky_2` are locked and the finger flexes with `pinky_3` and \
`pinky_4`. For every movable joint a larger angle means a more flexed (more closed) finger. The finger \
joints are position-controlled (PD), so a finger that meets the cup stops there and presses with a force \
that grows with the gap between its target angle and its actual angle. One control step is 1/60 s.
"""

REWARD_STRUCTURE = """\
Typically, the reward function of a manipulation task consists of these parts (some are optional — \
include them only if really necessary):
1. the distance between the robot's hand and the target object
2. the difference between the object's current state and its goal state
3. regularization of the robot's action
4. [optional] extra constraints on the target object implied by the task
5. [optional] extra constraints on the robot implied by the task
"""

ADDITIONAL_KNOWLEDGE = """\
Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel \
environments). The function must return a reward tensor of shape (N,) — never a Python float — and a \
dict of named component tensors, each of shape (N,). Use only `torch` and `math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors, e.g. \
`staged = torch.where(ctx.lifted, r_hold, torch.zeros_like(r_hold))`.
3. Prefer bounded, smooth shaping such as `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead \
of raw negative distances, so that no single term dominates.
4. Contact: `ctx.link_cup_force[:, f, k]` is the contact force between link k (0 the link moved by joint \
`_3`, 1 the link moved by joint `_4`, 2 the fingertip) of finger f (0 thumb, 1 index, 2 middle, 3 ring, \
4 pinky) and the cup, and `ctx.palm_cup_force` the force between the palm and the cup. Contact is measured \
only on these links and the palm — the finger links nearest the palm are not measured. Contacts with \
anything else (the table, the hand itself) are not included. A value of 0 means that link is not touching \
the cup. Contact forces come from the physics engine each step and can spike. These forces are only \
available to the reward; the policy itself does not observe contact.
5. Cylinder geometry: for a point p (e.g. `ctx.link_pos[:, f, k]`), with `v = p - ctx.cup_pos`, the axial \
coordinate is `h = (v * ctx.cup_axis).sum(-1)` and the radial vector is `v - h.unsqueeze(-1) * ctx.cup_axis`; \
its norm is the distance from the cup axis. The cup surface is at `ctx.cup_radius` and the graspable \
band is `|h| <= ctx.cup_half_height`. The palm normal `ctx.palm_normal` points out of the palm.
6. Height: `ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]` is how far the cup has been raised above its \
starting height (0 while it rests on the table). `ctx.lifted` becomes True once that height exceeds \
`ctx.lift_latch_height` and stays True for the rest of the episode, even if the cup is dropped again.
7. Goal and success are computed by the environment and cannot be redefined by the reward. The goal \
position is 0.21 m to 0.28 m above the cup's starting position and at most 5 cm away from it \
horizontally, with the cup upright. A success is counted when `ctx.goal_dist <= ctx.success_tol` for \
`ctx.success_hold_steps` consecutive steps, and `ctx.success` is True on that step. After a success the \
next goal is at the same place, so holding the cup still there keeps producing successes until \
`ctx.max_successes`, which ends the episode. The keypoints are fixed on the cup, so tilting the cup also \
increases `goal_dist`. `ctx.success_tol` starts at 0.1125 m and shrinks towards 0.015 m as the policy \
succeeds more often during training. You may add a bonus on `ctx.success`.
8. An episode lasts at most 600 steps (10 s), and the step budget restarts after every success. The \
episode ends early when the cup falls below z = 0.15 (off the table), leaves the table area, or tilts more \
than 60 degrees; when any hand link goes more than 3 cm below the table top; or when an arm joint goes \
past its limit or moves faster than 20 rad/s. On that last kind of physics violation the environment \
also adds a fixed -1 to the reward, outside your function.
9. Do not keep any state between calls (no globals, no attributes); the function must be pure.
10. Each component you return is logged separately during training and may be shown back to you after \
training, so name components meaningfully (e.g. "approach", "lift", "success_bonus").
"""

OUTPUT_RULES = """\
I want it to fulfil the following task: {task}
1. Please think step by step and tell me what this task means and which stages the robot must go through.
2. Then write a function with exactly this signature:
```python
def {entry}(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, {{"component_name": component_tensor, ...}}
```
   where `reward` has shape (N,) and every dict value has shape (N,).
3. Take care of tensor shapes and types; never access a field or method that is not listed in the class \
definition above.
4. Put the final code in ONE ```python fenced block that starts with `import torch` and `import math` \
and contains only the function (plus small helper functions if needed). Add short comments explaining \
the weights.
"""


@dataclass(frozen=True)
class PromptSpec:
    task: str
    previous_code: str | None = None
    feedback: str | None = None       # 렌더된 학습 지표 표(이후 라운드)
    user_notes: str | None = None     # 사용자 관찰(선택)


def render_prompt(spec: PromptSpec) -> str:
    parts = [
        "You are an expert in robotics, reinforcement learning and code generation.",
        ROBOT_DESCRIPTION.format(joint_table=_joint_table()),
        "Now I want you to help me write a reward function for reinforcement learning.",
        REWARD_STRUCTURE,
        "The reward function receives a single argument `ctx`, an instance of this class "
        "(all positions env-local, metres; angles rad; forces N):",
        "```python\n" + context_stub_source() + "```",
        ADDITIONAL_KNOWLEDGE,
        OUTPUT_RULES.format(task=spec.task, entry=ENTRY_NAME),
    ]
    if spec.previous_code:
        parts.append("The previous reward function was:\n```python\n" + spec.previous_code.rstrip() + "\n```")
    if spec.feedback:
        parts.append(spec.feedback)
    if spec.user_notes:
        parts.append("Observations from watching the trained policy:\n" + spec.user_notes.rstrip())
    return "\n\n".join(p.rstrip() for p in parts) + "\n"
