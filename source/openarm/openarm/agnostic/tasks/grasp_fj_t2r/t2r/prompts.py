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


@dataclass(frozen=True)
class EnvFacts:
    """변종(= reward_gen 트랙의 env)별 환경 사실 — 부팅 로그·IK·물체 뱅크로 확인한 값만 적는다."""

    scene: str
    k_arm: float
    arm_slew: float
    episode_steps: int
    episode_s: float


VARIANTS: dict[str, EnvFacts] = {
    # grasp_fj_t2r(B leaf): 시작 손바닥→컵 159.8/163.4 mm(09.14 부팅 가드 로그) · shaker_sweep 반경 29.2–43.8 mm ·
    #   k_arm 0.025 · slew 0.15 · 600 스텝.
    "envelope": EnvFacts(
        scene=("An upright cylindrical cup stands on the table in front of the hand; its radius differs between "
               "parallel environments (29 mm to 44 mm) and is given per environment. At the start of an episode "
               "the palm is roughly 0.16 m from the cup."),
        k_arm=0.025, arm_slew=0.15, episode_steps=600, episode_s=10.0),
    # grasp_fj_t2r_reach(09.14 사용자 최종 목표) — 서버 GPU0 스냅샷(13:55): palm (0.050, −0.300, 0.450) · 손바닥 법선 +y ·
    #   손가락 +x 로 상판(x 0.07–0.47) 위 x ≤0.25 · 시작 거리 가드 380.5 mm · 로봇 루트 x 0 → +x 가 테이블 쪽 ·
    #   cup_family 반경 44–80.6 mm · 반높이 42.5–65 mm · k_arm 0.05 · slew 0.3 · 900 스텝.
    "reach": EnvFacts(
        scene=("+x points from the robot toward the table. Each episode starts with the arm raised beside the "
               "robot: the palm is just outside the table edge nearest the robot, about 0.25 m above the table "
               "top, turned sideways, with the fingers pointing forward over the table edge; the palm is roughly "
               "0.38 m from the cup. A cup stands upright on the table; parallel environments use different cups "
               "(open cups of several sizes and a closed shaker), so the graspable radius (44 mm to 81 mm) and "
               "half height (42 mm to 65 mm) differ between environments and are given per environment. "
               # ★09.15 시작 상태 커리큘럼(reach leaf `near_start_*`, IK·FK 대조 테스트) — 두 출발을 생성기가 알아야 한다
               "From the second episode on, each episode starts with probability one half from a second pose "
               "instead: the hand already beside the cup on its -y side, in the same orientation and default hand "
               "pose as the raised start, with the palm about 3 cm from the cup's side, the cup axis about 2.5 cm "
               "further along the fingers than in the position that sets approach_done, and the palm centre in the "
               "upper part of the graspable band; the cup position still varies by up to 2 cm."),
        k_arm=0.05, arm_slew=0.3, episode_steps=900, episode_s=15.0),
}


ROBOT_DESCRIPTION = """\
We control one 7-DOF OpenArm robot arm (the right arm) carrying a five-finger Tesollo DG-5F hand, \
standing at a table. {scene} Positions are in metres in each environment's local \
frame, with +z pointing up; the table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (26,), float32)` with direct joint control. There is no \
grasp primitive, no hand synergy and no automatic finger stopping:
    actions[0:7]  = arm joint increments: each arm joint target moves by {k_arm:g} * a rad per step and then \
passes a first-order filter (factor 0.1), so each arm joint moves at most about {arm_slew:g} rad/s.
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
5. Cylinder geometry (the cups are roughly cylindrical): for a point p (e.g. `ctx.link_pos[:, f, k]`), with `v = p - ctx.cup_pos`, the axial \
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
8. An episode lasts at most {episode_steps} steps ({episode_s:g} s), and the step budget restarts after every success. The \
episode ends early when the cup falls below z = 0.15 (off the table), leaves the allowed area around the table, or \
tilts more than 60 degrees; when any hand link goes below z = table_z - 0.03 (wherever the hand is); or when an arm joint goes \
past its limit or moves faster than 20 rad/s. On that last kind of physics violation the environment \
replaces the reward of that step with a fixed -1 (your function's value is not used on that step).
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


#: 피드백 표 머리말 — 지표의 **뜻만** 적는다(측정 정의 = env 코드 사실). 설계 처방은 넣지 않는다.
#:   contact/* 정의: `grasp_fj_t2r_env._log_fabric_metrics`(0.1 N 초과를 닿음으로 센다) · done/* 는 스텝별 env 비율.
FEEDBACK_HEADER = """\
For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward \
components and some task metrics at {n_points} evenly spaced points during training, plus the \
min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over \
environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup \
(force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links \
touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; \
contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean \
link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted \
(a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, \
averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: \
fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: \
cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, \
hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.
"""

FEEDBACK_TAIL = """\
Please carefully analyse the policy feedback and provide a new, improved reward function. \
Some helpful tips:
(1) If a task metric (e.g. successes) stays near zero, the reward is not giving enough signal for that \
stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its \
scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages \
the policy has not yet reached are still worth pursuing.
(4) Look for exploits: a component that keeps rising while the task metrics it should support do not move \
is being earned some other way — gate or reshape it so it only pays in the states where it is meaningful.
Then write the improved function following the same output rules as before.
"""


#: ★원본 text2reward interactive 문구 그대로 — `skill_gen/text2reward/code_generation/interactive/classlike_prompt/
#:   feedback_prompt.py` 의 예시 형식과 `basic/generation.py` 의 suffix. 관찰(description)·개선 피드백(feedback)은
#:   학습한 로봇을 영상으로 보고 쓴다(09.14 사용자 결정: Claude 가 영상·지표로 초안 → 사용자 승인).
T2R_FEEDBACK_TEMPLATE = """\
Generated code shown as below:
```python
{code}
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, \
I can see from the robot that:
{description}

To make the code more accurate and train better robot, the feedback for improvement is:
{feedback}"""
T2R_SUFFIX = "Re-imagine which steps is missed or wrong.\nShow me the improved code as below:"


@dataclass(frozen=True)
class PromptSpec:
    task: str
    previous_code: str | None = None
    feedback: str | None = None       # 렌더된 학습 지표 표(Eureka 형 — render --feedback 전용)
    user_notes: str | None = None     # 사용자 관찰(선택)
    history: tuple = ()               # ★t2r interactive: ({"code","description","feedback"}, …) 라운드 순서
    metrics: str | None = None        # 이력 뒤에 붙는 참고 지표 표(선택)
    variant: str = "envelope"         # ★환경 사실 묶음(VARIANTS) — 트랙의 env 와 맞춘다(09.14 reach 추가)


def render_prompt(spec: PromptSpec) -> str:
    if spec.variant not in VARIANTS:
        raise KeyError(f"모르는 환경 변종 '{spec.variant}' (있는 것: {sorted(VARIANTS)})")
    facts = VARIANTS[spec.variant]
    parts = [
        "You are an expert in robotics, reinforcement learning and code generation.",
        ROBOT_DESCRIPTION.format(joint_table=_joint_table(), scene=facts.scene, k_arm=facts.k_arm,
                                 arm_slew=facts.arm_slew),
        "Now I want you to help me write a reward function for reinforcement learning.",
        REWARD_STRUCTURE,
        "The reward function receives a single argument `ctx`, an instance of this class "
        "(all positions env-local, metres; angles rad; forces N):",
        "```python\n" + context_stub_source() + "```",
        ADDITIONAL_KNOWLEDGE.format(episode_steps=facts.episode_steps, episode_s=facts.episode_s),
        OUTPUT_RULES.format(task=spec.task, entry=ENTRY_NAME),
    ]
    # ★원본 text2reward interactive: 지난 (코드 · 로봇 관찰 · 개선 피드백) 전 이력 → "Re-imagine …" 로 새 코드.
    for h in spec.history:
        parts.append(T2R_FEEDBACK_TEMPLATE.format(code=h["code"].rstrip(), description=h["description"].rstrip(),
                                                  feedback=h["feedback"].rstrip()))
    if spec.metrics:
        parts.append(spec.metrics)
    if spec.history:
        parts.append(T2R_SUFFIX)
    if spec.previous_code:
        parts.append("The previous reward function was:\n```python\n" + spec.previous_code.rstrip() + "\n```")
    if spec.feedback:
        parts.append(spec.feedback)
        parts.append(FEEDBACK_TAIL)
    if spec.user_notes:
        parts.append("Observations from watching the trained policy:\n" + spec.user_notes.rstrip())
    return "\n\n".join(p.rstrip() for p in parts) + "\n"


def render_feedback_table(series: dict[str, list[float]], n_points: int = 10) -> str:
    """{태그: 값열} → Eureka 형 표. 각 태그를 n_points 로 균등 표본 + min/mean/max (붓기 트랙과 같은 형식)."""
    lines = [FEEDBACK_HEADER.format(n_points=n_points)]
    for tag in sorted(series):
        vals = [v for v in series[tag] if v == v]   # NaN 제거
        if not vals:
            continue
        step = max(len(vals) // n_points, 1)
        sampled = vals[::step][:n_points]
        lines.append(f"{tag}: [{', '.join(f'{v:.3g}' for v in sampled)}]  "
                     f"min {min(vals):.3g} · mean {sum(vals) / len(vals):.3g} · max {max(vals):.3g}")
    return "\n".join(lines) + "\n"
