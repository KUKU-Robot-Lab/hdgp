"""프롬프트 렌더링 — text2reward 형식(역할·로봇·환경 클래스 스텁·추가 지식·과제 문장·출력 규칙)
+ Eureka 형 함수 시그니처와 학습 피드백 섹션.

★환경 설명은 `RewardContext` 소스에서 **그대로** 뽑는다(`context_stub_source`). 손으로 옮겨
  적은 사본이 없으므로 필드가 바뀌면 프롬프트도 같이 바뀐다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .context import context_stub_source
from .loader import ENTRY_NAME

ROBOT_DESCRIPTION = """\
We control a bimanual robot: two 7-DOF OpenArm arms, each carrying a 20-DOF five-finger \
Tesollo DG-5F hand, standing at a table. The RIGHT arm is the "source" arm (src_*), the LEFT \
arm is the "receiver" arm (rcv_*). Two identical cups stand on the table: the source cup \
(in front of the right hand, filled with {num_beads} small beads) and the receiver cup \
(in front of the left hand, empty). Positions are in metres in a frame whose origin is at \
the robot base; +z is up, +x is forward (away from the robot), +y is to the robot's left. \
The table top is at z = table_z.

The action space is a normalized `Box(-1, 1, ({num_actions},), float32)`:
  actions[0:6]   = source palm 6-DoF target offset (xyz + yaw/pitch/roll) — the arm is moved \
by a geometric-fabrics controller toward this target
  actions[6:21]  = source hand closure commands (5 fingers × 3 channels, 0 = open, 1 = closed)
  actions[21:27] = receiver palm 6-DoF target offset
  actions[27:42] = receiver hand closure commands
The hand controller stops a finger joint automatically once that finger link touches its own \
cup (contact freeze), so a closing command produces a wrap-around power grasp; opening is \
always allowed. Fingers can only close when the palm is near its cup.
"""

REWARD_STRUCTURE = """\
Typically, the reward function of a manipulation task consists of these parts (some are \
optional — include them only if really necessary):
1. the distance between the robot's hand and the target object (one term per arm)
2. the difference between the object's current state and its goal state
3. regularization of the robot's action
4. [optional] extra constraints on the target objects implied by the task (e.g. keep the \
receiver cup upright, do not spill)
5. [optional] extra constraints on the robot implied by the task
"""

ADDITIONAL_KNOWLEDGE = """\
Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel \
environments). The function must return a reward tensor of shape (N,) — never a Python \
float — and a dict of named component tensors, each of shape (N,). Use only `torch` and \
`math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors: \
`stage2 = torch.where(ctx.src_grasped, r_lift, torch.zeros_like(r_lift))`.
3. Prefer bounded, smooth shaping: `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` \
instead of raw negative distances, so that no single term dominates.
4. `ctx.src_grasped` / `ctx.rcv_grasped` are True when the thumb AND another finger both \
press on that hand's own cup — this is the grasp-established signal. \
`ctx.*_hand_closure` is the measured closure (0 = open, 1 = fully closed).
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z \
(0 = upright, π/2 = horizontal). Beads leave the source cup once it is tilted past roughly \
110° (1.9 rad) with its mouth over the receiver cup's mouth. `ctx.src_cup_mouth_pos` and \
`ctx.rcv_cup_mouth_pos` are the rim centres.
6. Bead bookkeeping: `bead_in_target_frac` rises as beads land in the receiver cup; \
`bead_spill_frac` counts beads lost outside both cups (permanent). `d_in_target` and \
`d_spill` are this step's increments — reward INCREMENTS of beads transferred rather than \
the level, otherwise the policy is paid for standing still with a filled cup.
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little \
spill, cups close together). You may add a bonus on it but you cannot redefine it.
8. Height above the table: `ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]` is how far \
the source cup has been lifted (0 while it rests on the table).
9. Do not keep any state between calls (no globals, no attributes); the function must be pure.
10. Each component you return is logged separately during training and shown back to you \
after training, so name them meaningfully (e.g. "approach_src", "grasp_rcv", "pour_delta").
"""

OUTPUT_RULES = """\
I want it to fulfil the following task: {task}
1. Please think step by step and tell me what this task means and which stages the robot \
must go through.
2. Then write a function with exactly this signature:
```python
def {entry}(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, {{"component_name": component_tensor, ...}}
```
   where `reward` has shape (N,) and every dict value has shape (N,).
3. Take care of tensor shapes and types; never call methods that are not listed in the \
class definition above.
4. Put the final code in ONE ```python fenced block that starts with `import torch` and \
`import math` and contains only the function (plus small helper functions if needed). \
Add short comments explaining weights, e.g. \
`# weight 0.5 works together with the pour term 5.0 — pouring must dominate once grasped`.
"""

FEEDBACK_HEADER = """\
We trained an RL policy (PPO) using the reward function below and tracked the values of the \
individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, \
beads transferred, spill, success rate, episode length) at {n_points} evenly spaced points \
during training, plus the min / mean / max encountered:
"""

FEEDBACK_TAIL = """\
Please carefully analyse the policy feedback and provide a new, improved reward function. \
Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving \
enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not \
optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), \
or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; \
rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the \
policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.
"""


@dataclass(frozen=True)
class PromptSpec:
    task: str
    num_actions: int = 42
    num_beads: int = 20
    previous_code: str | None = None
    feedback: str | None = None       # 렌더된 지표 표(reflect 가 만든다)
    user_notes: str | None = None     # 사용자 관찰(선택)


def render_prompt(spec: PromptSpec) -> str:
    parts = [
        "You are an expert in robotics, reinforcement learning and code generation.",
        ROBOT_DESCRIPTION.format(num_beads=spec.num_beads, num_actions=spec.num_actions),
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
        parts.append(FEEDBACK_TAIL)
    if spec.user_notes:
        parts.append("Observations from watching the trained policy:\n" + spec.user_notes.rstrip())
    return "\n\n".join(p.rstrip() for p in parts) + "\n"


def render_feedback_table(series: dict[str, list[float]], n_points: int = 10) -> str:
    """{태그: 값열} → Eureka 형 표. 각 태그를 n_points 로 균등 표본 + min/mean/max."""
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
