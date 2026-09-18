"""Prompt rendering of the IKER UNIFIED t2r reward generator (one policy: pick up, carry, align, place, let go).

text2reward layout, same section order as the stage-1 and stage-2 forks: role -> robot and action description ->
reward structure -> the RewardContext source -> additional knowledge -> task and output rules -> (later rounds)
previous code, feedback table and tips, notes.

Environment facts only, no design advice. Numbers checked against the code on 2026-09-18: action scales, noise and
domain randomisation `iker_shoe_env_cfg.IkerShoeEnvCfg` / `iker_shoe_grasp_env_cfg.IkerShoeGraspEnvCfg`; hand joint
ranges the boot log of iker_grasp_c00_r8_a (2026-09-15); rack geometry `layout`; hold predicate
`grasp_stage.stage1_step`; placement predicate and scripted return `place_stage.place_step` / `PlaceRewardCfg`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import grasp_stage as gs
from .. import layout
from ..place_stage import PlaceRewardCfg
from openarm.agnostic.modules.robot_profiles import TESOLLO_LEFT_SHORT
from .context import context_stub_source
from .loader import ENTRY_NAME

ARM_ACTION_DIM, HAND_ACTION_DIM = 6, 20
ACTION_POS_SCALE_M, ACTION_ROT_SCALE_RAD = 0.02, 0.05
CONTROL_DT_S = 0.1
OBSERVATION_NOISE, ACTION_NOISE, QUAT_NOISE_RAD = 0.02, 0.05, 0.2
MASS_SCALE_RANGE, FRICTION_RANGE, COM_OFFSET_M = (0.3, 2.0), (0.3, 1.8), 0.025
WRENCH_PROB_RANGE, WRENCH_FORCE_PER_KG, WRENCH_TORQUE_PER_KG = (0.006, 0.6), 2.7, 0.27
DROP_Z_M = 0.10
#: mirrors iker_shoe_unified_env_cfg.UNIFIED_EPISODE_STEPS (pinned by a test); kept here so the CLI can render the
#: prompt without importing isaaclab
DEFAULT_EPISODE_STEPS = 360
LOCKED_SPAN_RAD = 0.05
#: commandable range [rad] of each hand joint in TESOLLO_LEFT_SHORT.hand_joint_names order (boot log of iker_grasp_c00_r8_a)
HAND_ACTION_RANGES: tuple[tuple[float, float], ...] = (
    (-0.010, 0.010), (1.560, 1.580), (-1.571, 0.000), (-1.571, 0.000),
    (-0.010, 0.010), (0.000, 2.007), (0.000, 1.571), (0.000, 1.571),
    (-0.010, 0.010), (0.000, 2.007), (0.000, 1.571), (0.000, 1.571),
    (-0.010, 0.010), (0.000, 1.920), (0.000, 1.571), (0.000, 1.571),
    (-0.010, 0.000), (-0.010, 0.000), (0.000, 1.571), (0.000, 1.571),
)


def _joint_table() -> str:
    prof = TESOLLO_LEFT_SHORT
    rows = []
    for index, (name, (lo, hi)) in enumerate(zip(prof.hand_joint_names, HAND_ACTION_RANGES, strict=True)):
        tag = "locked" if hi - lo <= LOCKED_SPAN_RAD else "movable"
        rows.append(f"    actions[{ARM_ACTION_DIM + index:2d}] = hand joint {index:2d}: {name} range [{lo:+.3f}, {hi:+.3f}] rad  "
                    f"open {float(prof.hand_open_pose[index]):+.3f}  grip {float(prof.hand_grip_pose[index]):+.3f}  {tag}")
    return "\n".join(rows)


ROBOT_DESCRIPTION = """\
We control one 7-DOF OpenArm robot arm (the left arm) carrying a five-finger Tesollo DG-5F hand, standing at a table. \
A shoe lies on the table in front of the hand; a second shoe already stands on a raised rack beside the table, and the \
task is to put the loose shoe on the rack next to it. The rack's footprint is x in [{rack_x_min:.2f}, {rack_x_max:.2f}] m \
and y in [{rack_y_min:.2f}, {rack_y_max:.2f}] m, and its top is at z = {rack_top_z:.3f} m. Positions are in metres in each \
environment's local frame with +z up; the table top is at z = table_top_z.

At the start of an episode the arm stands in its rest posture and the hand is open; the shoe lies somewhere in the \
reachable area of the table, its position drawn fresh every episode (its resting orientation does not change). Some \
episodes instead start from a recorded later state — the shoe already in the hand, or already set down on the rack — so \
that the later parts of the task are practised too; `ctx.start_held` is True in those episodes.

The action space is a normalized `Box(-1, 1, ({action_dim},), float32)`, one action every {control_dt} s, \
{episode_steps} steps per episode:
    actions[0:3] = change of the palm position in the robot base frame, {pos_scale} * a metres per step;
    actions[3:6] = change of the palm orientation in the robot base frame (axis-angle), {rot_scale} * a radians per step;
{joint_table}
The palm pose change goes through damped least-squares inverse kinematics to arm joint position targets (PD, gravity \
compensated), clamped to the arm's joint limits. Each hand action is mapped to its joint's commandable range above \
(a = -1 the lower end, a = +1 the upper end) and then filtered: every step the joint target moves {alpha:.4f} of the way \
from its previous value towards the commanded one.

Each environment draws once at startup a shoe mass scale of {mass_lo} to {mass_hi}, a friction coefficient of {fric_lo} \
to {fric_hi}, a restitution of 0 to 1 and a centre-of-mass offset of up to {com_cm:.1f} cm per axis. During training the \
policy's observations carry uniform noise of +-{obs_noise} (orientations up to {quat_noise} rad) and its actions uniform \
noise of +-{act_noise} before they are executed. Once the grasp has latched, random force and torque pulses act on the \
shoe: each step an environment fires with its own probability between {p_lo} and {p_hi}, and a pulse has a normal random \
component per axis with a standard deviation of {force_per_kg} N ({torque_per_kg} N m) per kg of shoe mass.
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
`staged = torch.where(ctx.held, r_carry, torch.zeros_like(r_carry))`.
3. Prefer bounded, smooth shaping such as `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative \
distances, so that no single term dominates.
4. THE TASK HAS TWO HALVES IN ONE EPISODE and the policy must do both with the same action vector: first pick the shoe \
up off the table, then carry it to the rack, align it with the target keypoints, set it down and let go.
5. Grasp status computed by the environment: `ctx.held` is True on a step where the shoe is lifted at least \
{lift_height} m clear of where it lay, the thumb has closed at least {thumb_curl_min} rad, the palm is within \
{hold_radius} m of the shoe and the shoe slips less than {hold_slip_speed} m/s relative to the palm. `ctx.hold_count` \
counts consecutive held steps, `ctx.latched` turns True once it reaches {latch_steps} and stays True, and `ctx.carried` \
stays True once the shoe has been lifted clear of the table at all. Carrying the shoe away from where it was picked up \
makes `ctx.dz_free` read 0 (it only measures a free lift near the pick spot), so `ctx.held` does NOT stay true across \
the transport — use the contact, slip and gap fields to tell whether the shoe is still in the hand.
6. Goal: `ctx.target_keypoints` holds 4 fixed target points on the rack, set once for the whole run by a vision model \
(not part of the reward). `ctx.keypoints` are the shoe's own 4 keypoints in its current pose, `ctx.init_keypoints` the \
same at the start of the episode, `ctx.keypoint_err` the per-keypoint distance to the target and `ctx.keypoint_dist` \
their mean.
7. Placement success is computed by the environment and cannot be redefined by the reward. A step counts toward it when \
all five of these hold: `ctx.keypoint_dist <= {place_tol}` m (`ctx.placed`), `ctx.palm_shoe_dist > {release_radius}` m \
(`ctx.released`, the hand has let go), `abs(ctx.shoe_bottom_z - ctx.rack_top_z) <= ctx.resting_tol` (`ctx.resting`), \
`ctx.shoe_lin_vel.norm(dim=-1) < {still_speed}` m/s (`ctx.still`), and `ctx.arm_home_err <= {home_joint_tol}` rad \
(`ctx.home`, every arm joint back in the rest posture). `ctx.stable_count` is how many of the last {window_steps} steps \
had all five true — one bad step costs one count and does NOT reset it — and `ctx.success` is True on the step \
`ctx.stable_count` reaches {stable_steps}.
   THE RETURN TO THE REST POSTURE IS NOT THE POLICY'S JOB. The first step the shoe is placed, resting and still while \
the hand has opened at least {retract_open_min} of the way to its open pose, the environment takes over the arm: it \
opens the hand fully and moves the arm joints to the rest posture over {retract_steps} steps, and from then on the \
policy's actions have no effect (`ctx.retracting` is True). Any reward paid while `ctx.retracting` is True cannot be \
influenced by the policy.
8. The episode ends on a success, when the shoe falls below z = {drop_z} m, or after `ctx.episode_steps - 1` steps. \
Nothing is added to the reward outside your function.
9. Do not keep any state between calls (no globals, no attributes); the function must be pure. `ctx` is read-only — never \
assign to one of its fields (no `ctx.x = ...`) and never call an in-place op on one (no `.clamp_()`, no `x[:, 0] = ...`).
10. Each component you return is logged separately during training (as `t2r_reward/<name>`, the sum as \
`t2r_reward/total`) and may be shown back to you after training, so name components meaningfully (e.g. "approach", \
"grasp", "carry", "align", "release").
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
We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward \
components (t2r_reward/*) as well as task metrics computed by the environment (the grasp flags grasp/held_frac, \
grasp/lost_frac, grasp/dz_free, grasp/thumb_curl, grasp/rel_speed, the placement flags place/placed, place/released, \
place/resting, place/still, place/home, the success rates place/success_table_start and place/success_held_start, the \
keypoint distance iker/keypoint_distance_m, the drop rate iker/dropped, episode length and total reward) at {n_points} \
evenly spaced points during training, plus the min / mean / max encountered:
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
    "t2r_reward/", "grasp/held_frac", "grasp/lost_frac", "grasp/dz_free", "grasp/thumb_curl", "grasp/rel_speed",
    "place/placed", "place/released", "place/resting", "place/still", "place/home", "place/retracting",
    "place/success_table_start", "place/success_held_start", "place/carried",
    "iker/keypoint_distance_m", "iker/dropped", "episode_lengths", "rewards",
)


def task_text() -> str:
    return ("Pick the shoe up from the table, carry it to the rack, align it with the target keypoints next to the "
            "other shoe, set it down and let go.")


def render_prompt(spec: PromptSpec, cfg: PlaceRewardCfg | None = None, grasp_cfg: gs.Stage1RewardCfg | None = None,
                  episode_steps: int = DEFAULT_EPISODE_STEPS) -> str:
    cfg = cfg if cfg is not None else PlaceRewardCfg()
    grasp_cfg = grasp_cfg if grasp_cfg is not None else gs.Stage1RewardCfg()
    robot_text = ROBOT_DESCRIPTION.format(
        rack_x_min=layout.RACK_X_RANGE[0], rack_x_max=layout.RACK_X_RANGE[1], rack_y_min=layout.RACK_Y_RANGE[0],
        rack_y_max=layout.RACK_Y_RANGE[1], rack_top_z=layout.RACK_TOP_Z, control_dt=CONTROL_DT_S,
        episode_steps=episode_steps, action_dim=ARM_ACTION_DIM + HAND_ACTION_DIM, pos_scale=ACTION_POS_SCALE_M,
        rot_scale=ACTION_ROT_SCALE_RAD, joint_table=_joint_table(), alpha=gs.HAND_EMA_ALPHA,
        mass_lo=MASS_SCALE_RANGE[0], mass_hi=MASS_SCALE_RANGE[1], fric_lo=FRICTION_RANGE[0], fric_hi=FRICTION_RANGE[1],
        com_cm=COM_OFFSET_M * 100, obs_noise=OBSERVATION_NOISE, quat_noise=QUAT_NOISE_RAD, act_noise=ACTION_NOISE,
        p_lo=WRENCH_PROB_RANGE[0], p_hi=WRENCH_PROB_RANGE[1], force_per_kg=WRENCH_FORCE_PER_KG,
        torque_per_kg=WRENCH_TORQUE_PER_KG,
    )
    knowledge = ADDITIONAL_KNOWLEDGE.format(
        lift_height=grasp_cfg.lift_height_m, thumb_curl_min=grasp_cfg.thumb_curl_min_rad,
        hold_radius=grasp_cfg.hold_radius_m, hold_slip_speed=grasp_cfg.hold_rel_speed,
        latch_steps=grasp_cfg.latch_steps, place_tol=cfg.place_tolerance, release_radius=cfg.release_radius,
        still_speed=cfg.still_speed, stable_steps=cfg.stable_steps, window_steps=cfg.window_steps,
        home_joint_tol=cfg.home_joint_tol, retract_steps=cfg.retract_steps, retract_open_min=cfg.retract_open_min,
        drop_z=DROP_Z_M,
    )
    parts = [
        "You are an expert in robotics, reinforcement learning and code generation.",
        robot_text,
        "Now I want you to help me write a reward function for reinforcement learning.",
        REWARD_STRUCTURE,
        "The reward function receives a single argument `ctx`, an instance of this class "
        "(all positions env-local, metres; angles rad; speeds m/s; forces N):",
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
