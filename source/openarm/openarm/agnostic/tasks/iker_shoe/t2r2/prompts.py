"""Prompt rendering of the IKER stage-2 t2r reward generator (spec 2026-09-16-iker-stage2-t2r §7).

text2reward layout: role -> robot and action description -> reward structure -> the RewardContext source -> additional
knowledge -> task and output rules -> (later rounds) previous code, feedback table and tips, notes. Same section order
as the stage-1 fork (this file mirrors its structure, not its field list).

Environment facts only, no design advice (zero-shot, same policy as stage-1). Numbers checked against the code on
2026-09-16: action scales and noise `iker_shoe_env_cfg.IkerShoeEnvCfg` (observation_noise 0.02, action_noise 0.05,
quat_noise_rad 0.2, EPISODE_STEPS 200, CONTROL_DT 0.1); rack geometry `layout.RACK_X_RANGE`/`RACK_Y_RANGE`/`RACK_TOP_Z`;
grip EMA law `grasp_stage.hand_targets`/`HAND_EMA_ALPHA`; success predicate `place_stage.place_step`/`PlaceRewardCfg`
(spec §3).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..place_stage import PlaceRewardCfg
from .. import grasp_stage as gs
from .. import layout
from .context import context_stub_source
from .loader import ENTRY_NAME

ACTION_POS_SCALE_M, ACTION_ROT_SCALE_RAD = 0.02, 0.05
CONTROL_DT_S = 0.1
EPISODE_STEPS = 200
OBSERVATION_NOISE, ACTION_NOISE, QUAT_NOISE_RAD = 0.02, 0.05, 0.2


ROBOT_DESCRIPTION = """\
We control one 7-DOF OpenArm robot arm (the left arm) carrying a five-finger Tesollo DG-5F hand, standing at a table. \
The episode starts with the shoe already held in the hand (its recorded grasp pose); a second shoe stands on a raised \
rack beside the table and the hand must place its held shoe on the rack next to it. The rack's footprint is x in \
[{rack_x_min:.2f}, {rack_x_max:.2f}] m and y in [{rack_y_min:.2f}, {rack_y_max:.2f}] m, and its top is at z = {rack_top_z:.3f} m. \
Positions are in metres in each environment's local frame with +z up; the table top is at z = table_top_z.

The action space is a normalized `Box(-1, 1, (7,), float32)`, one action every {control_dt} s, {episode_steps} steps per \
episode:
    actions[0:3] = change of the palm position in the robot base frame, {pos_scale} * a metres per step;
    actions[3:6] = change of the palm orientation in the robot base frame (axis-angle), {rot_scale} * a radians per step.
The palm pose change goes through damped least-squares inverse kinematics to arm joint position targets (PD, gravity \
compensated), clamped to the arm's joint limits.
    actions[6] = grip axis: a = -1 keeps the hand at its recorded grip pose (holding the shoe), a = +1 moves it toward \
the hand profile's open pose. Each step the finger joint targets move {alpha:.4f} of the way from their previous value \
towards that point (the same EMA law as the arm's other commanded joints), clamped so a finger cannot bend back past its \
open pose. The episode starts at a = -1 (holding). `ctx.grip_norm` reports this filtered state, normalised to [-1, 1], \
where -1 is the grip pose and +1 the open pose.

During training the policy's observations carry uniform noise of +-{obs_noise} (orientations up to {quat_noise} rad) and \
its actions uniform noise of +-{act_noise} before they are executed.
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
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel environments). The function \
must return a reward tensor of shape (N,) — never a Python float — and a dict of named component tensors, each of shape (N,). \
Use only `torch` and `math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors, e.g. \
`staged = torch.where(ctx.released, r_withdraw, torch.zeros_like(r_withdraw))`.
3. Prefer bounded, smooth shaping such as `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative \
distances, so that no single term dominates.
4. Goal: `ctx.target_keypoints` holds 4 fixed target points on the rack, set once for the whole run by a vision model (not \
part of the reward). `ctx.keypoints` are the shoe's own 4 keypoints in its current pose, `ctx.init_keypoints` the same at \
the start of the episode (already in the grasped pose), `ctx.keypoint_err` the per-keypoint distance to the target and \
`ctx.keypoint_dist` their mean.
5. Placement success is computed by the environment and cannot be redefined by the reward. A step counts toward it when \
all four of these hold: `ctx.keypoint_dist <= {place_tol}` m (`ctx.placed`), `ctx.palm_shoe_dist > {release_radius}` m \
(`ctx.released`, the hand has let go), `abs(ctx.shoe_bottom_z - ctx.rack_top_z) <= ctx.resting_tol` (`ctx.resting`, the \
shoe rests on the rack rather than floating or sinking), and `ctx.shoe_lin_vel.norm(dim=-1) < {still_speed}` m/s \
(`ctx.still`). `ctx.stable_count` counts consecutive steps where all four hold (it drops to 0 the moment any one breaks, \
it is not cumulative) and `ctx.success` is True on the step `ctx.stable_count` reaches {stable_steps}. You may add a \
bonus on `ctx.success` or on the individual conditions such as `ctx.placed` or `ctx.released`.
6. The episode ends on a success, when the shoe falls off its support, or after `ctx.episode_steps - 1` steps. Nothing is \
added to the reward outside your function.
7. Do not keep any state between calls (no globals, no attributes); the function must be pure. `ctx` is read-only — never \
assign to one of its fields (no `ctx.x = ...`) and never call an in-place op on one (no `.clamp_()`, no `x[:, 0] = ...`).
8. Each component you return is logged separately during training (as `t2r_reward/<name>`, the sum as `t2r_reward/total`) \
and may be shown back to you after training, so name components meaningfully (e.g. "approach", "align", "release", "withdraw").
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

FEEDBACK_HEADER = """\
We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components \
(t2r_reward/*) as well as task metrics computed by the environment (success and latch rates of finished episodes, the \
fraction of held steps, lift height, horizontal shift, thumb curl, slip speed, contact counts, arm and hand motion, episode \
length) at {n_points} evenly spaced points during training, plus the min / mean / max encountered:
"""

FEEDBACK_TAIL = """\
Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. the success rate) is always near zero, the reward is not giving enough signal for that stage; \
rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its \
temperature/sharpness, or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not \
yet reached are still worth pursuing.
(4) Look for exploits: a component that grows while the task metrics do not may be paid for a behaviour that does not \
serve the task — gate it on the state that makes it meaningful.
Then write the improved function following the same output rules as before.
"""


@dataclass(frozen=True)
class PromptSpec:
    task: str
    previous_code: str | None = None
    feedback: str | None = None  # rendered feedback table (later rounds)
    user_notes: str | None = None  # verified observations written by a person (optional)


FEEDBACK_TAG_PREFIXES = (
    "t2r_reward/", "place/placed", "place/released", "place/resting",
    "iker/success_5cm", "iker/keypoint_distance_m", "iker/dropped", "episode_lengths", "rewards",
)


def task_text(cfg: PlaceRewardCfg) -> str:
    return ("Place the shoe on the rack next to the other shoe: align it with the target keypoints, "
            "set it down, let go, and withdraw the hand.")


def render_prompt(spec: PromptSpec) -> str:
    cfg = PlaceRewardCfg()
    robot = ROBOT_DESCRIPTION.format(
        rack_x_min=layout.RACK_X_RANGE[0], rack_x_max=layout.RACK_X_RANGE[1], rack_y_min=layout.RACK_Y_RANGE[0],
        rack_y_max=layout.RACK_Y_RANGE[1], rack_top_z=layout.RACK_TOP_Z, control_dt=CONTROL_DT_S,
        episode_steps=EPISODE_STEPS, pos_scale=ACTION_POS_SCALE_M, rot_scale=ACTION_ROT_SCALE_RAD,
        alpha=gs.HAND_EMA_ALPHA, obs_noise=OBSERVATION_NOISE, quat_noise=QUAT_NOISE_RAD, act_noise=ACTION_NOISE,
    )
    knowledge = ADDITIONAL_KNOWLEDGE.format(
        place_tol=cfg.place_tolerance, release_radius=cfg.release_radius, still_speed=cfg.still_speed,
        stable_steps=cfg.stable_steps,
    )
    parts = [
        "You are an expert in robotics, reinforcement learning and code generation.",
        robot,
        "Now I want you to help me write a reward function for reinforcement learning.",
        REWARD_STRUCTURE,
        "The reward function receives a single argument `ctx`, an instance of this class "
        "(all positions env-local, metres; angles rad; speeds m/s):",
        "```python\n" + context_stub_source() + "```",
        knowledge,
        OUTPUT_RULES.format(task=spec.task, entry=ENTRY_NAME),
    ]
    if spec.previous_code:
        parts.append("The previous reward function was:\n```python\n" + spec.previous_code.rstrip() + "\n```")
    if spec.feedback:
        parts.append(spec.feedback)
        parts.append(FEEDBACK_TAIL)
    if spec.user_notes:
        parts.append("Observations from watching the trained policy:\n" + spec.user_notes.rstrip())
    return "\n\n".join(part.rstrip() for part in parts) + "\n"


def render_feedback_table(series: dict[str, list[float]], n_points: int = 10) -> str:
    """{tag: values} -> Eureka-style table: each tag sampled at n_points even strides plus min / mean / max."""
    lines = [FEEDBACK_HEADER.format(n_points=n_points)]
    for tag in sorted(series):
        values = [v for v in series[tag] if v == v]  # drop NaN
        if not values:
            continue
        step = max(len(values) // n_points, 1)
        sampled = values[::step][:n_points]
        lines.append(f"{tag}: [{', '.join(f'{v:.3g}' for v in sampled)}]  "
                     f"min {min(values):.3g} · mean {sum(values) / len(values):.3g} · max {max(values):.3g}")
    return "\n".join(lines) + "\n"
