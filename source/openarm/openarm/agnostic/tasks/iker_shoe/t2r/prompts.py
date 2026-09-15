"""Prompt rendering of the IKER stage-1 t2r reward generator (spec 2026-09-15-iker-stage1-t2r §6).

text2reward layout: role -> robot and action description -> reward structure -> the RewardContext source -> additional
knowledge -> task and output rules -> (later rounds) previous code, feedback table and tips, notes.

Environment facts only, no design advice (user decision 2026-09-15: zero-shot). Numbers checked against the code on
2026-09-15: action scales and noise `iker_shoe_env_cfg.IkerShoeEnvCfg`; 10 Hz = `DECIMATION` 12 x `PHYSICS_DT` 1/120;
hand law `grasp_stage.hand_targets` with `HAND_EMA_ALPHA`; hand action ranges = the boot log of `iker_grasp_c00_r8_a`
("[iker_grasp] hand action range", profile order); start palm gap measured 105-142 mm (`scripts/iker/grasp_smoke.py`);
startup randomisation `IkerShoeEventCfg`; disturbance `object_wrench.WrenchDR.step` gated on the latch with
`IkerShoeGraspEnvCfg.wrench_*`; terminations `IkerShoeGraspEnv._get_dones`; hold predicate `grasp_stage.stage1_step`.
"""

from __future__ import annotations

from dataclasses import dataclass

from openarm.agnostic.modules.robot_profiles import TESOLLO_LEFT_SHORT

from .. import grasp_stage as gs
from .. import layout
from .context import context_stub_source
from .loader import ENTRY_NAME

ARM_ACTION_DIM, HAND_ACTION_DIM = 6, 20
ACTION_POS_SCALE_M, ACTION_ROT_SCALE_RAD = 0.02, 0.05
CONTROL_DT_S = 0.1
START_PALM_GAP_MM = (105, 142)
START_NOISE_XY_M = 0.02
MASS_SCALE_RANGE, FRICTION_RANGE, COM_OFFSET_M = (0.3, 2.0), (0.3, 1.8), 0.025
OBSERVATION_NOISE, ACTION_NOISE, QUAT_NOISE_RAD = 0.02, 0.05, 0.2
WRENCH_PROB_RANGE, WRENCH_FORCE_PER_KG, WRENCH_TORQUE_PER_KG = (0.006, 0.6), 2.7, 0.27
DROP_Z_M = 0.10
LOCKED_SPAN_RAD = 0.05
#: commandable range [rad] of each hand joint in TESOLLO_LEFT_SHORT.hand_joint_names order (boot log of iker_grasp_c00_r8_a, 2026-09-15)
HAND_ACTION_RANGES: tuple[tuple[float, float], ...] = (
    (-0.010, 0.010), (1.560, 1.580), (-1.571, 0.000), (-1.571, 0.000),
    (-0.010, 0.010), (0.000, 2.007), (0.000, 1.571), (0.000, 1.571),
    (-0.010, 0.010), (0.000, 2.007), (0.000, 1.571), (0.000, 1.571),
    (-0.010, 0.010), (0.000, 1.920), (0.000, 1.571), (0.000, 1.571),
    (-0.010, 0.000), (-0.010, 0.000), (0.000, 1.571), (0.000, 1.571),
)
FEEDBACK_TAG_PREFIXES = (
    # finding 3: "grasp_episode/best_lift_m" (the hand-written reward's in-zone lift level, clipped at 4 cm) is an internal
    # of that reward, not a fact about this env — showing it as "lift height" in the feedback header would mislead the generator.
    "t2r_reward/", "grasp_episode/success", "grasp_episode/latched", "grasp/held_frac",
    "grasp/dz_free", "grasp/shift_xy", "grasp/thumb_curl", "grasp/rel_speed", "grasp/lost_frac", "grasp/over_rack_raised_frac",
    "grasp/arm_speed_sum", "grasp/hand_command_rate", "contact/", "episode_lengths", "rewards",
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
A shoe lies on the table in front of the hand; a second shoe stands on a raised rack beside it. The rack's footprint is \
x in [{rack_x_min:.2f}, {rack_x_max:.2f}] m and y in [{rack_y_min:.2f}, {rack_y_max:.2f}] m, and its top is at z = {rack_top_z:.3f} m. \
Positions are in metres in each environment's local frame with +z up; the table top is at z = table_top_z.

At the start of an episode the arm is at a pre-grasp pose above the shoe with the hand open; the palm is about {gap_lo} to \
{gap_hi} mm from the nearest point of the shoe's surface. The shoe starts at one of a set of recorded rest poses, shifted by \
up to {start_noise_cm:.0f} cm along x and y. Each environment draws once at startup a shoe mass scale of {mass_lo} to {mass_hi}, \
a friction coefficient of {fric_lo} to {fric_hi}, a restitution of 0 to 1 and a centre-of-mass offset of up to {com_cm:.1f} cm \
per axis. During training the policy's observations carry uniform noise of +-{obs_noise} (orientations up to {quat_noise} rad) \
and its actions uniform noise of +-{act_noise} before they are executed. Once the hold has latched (see below), random force \
and torque pulses act on the shoe: each step an environment fires with its own probability between {p_lo} and {p_hi}, and a \
pulse has a normal random component per axis with a standard deviation of {force_per_kg} N ({torque_per_kg} N m) per kg of \
shoe mass.

The action space is a normalized `Box(-1, 1, (26,), float32)`, one action every {control_dt} s:
    actions[0:3] = change of the palm position in the robot base frame, {pos_scale} * a metres per step;
    actions[3:6] = change of the palm orientation in the robot base frame (axis-angle), {rot_scale} * a radians per step.
The palm pose change goes through damped least-squares inverse kinematics to arm joint position targets (PD, gravity \
compensated), clamped to the arm's joint limits.
    actions[6:26] = finger joint targets: each value is mapped linearly onto that joint's range (a = -1 the lower limit, \
a = +1 the upper limit), and each step the joint target moves {alpha:.4f} of the way from its previous value to that point. \
The order, which is also the order of ctx.hand_q / hand_qd / hand_q_norm / hand_target_norm, is:
{joint_table}
Joints marked "locked" have a range of 0.05 rad or less and effectively do not move. "open" is the joint angle of the open \
hand the episode starts with and "grip" the joint angle of a closed grasp: a finger closes as its joint angles move from \
"open" towards "grip". The finger joints are position-controlled (PD), so a finger that meets the shoe stops there and \
presses with a force that grows with the gap between its target and its actual angle.
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
`staged = torch.where(ctx.latched, r_hold, torch.zeros_like(r_hold))`.
3. Prefer bounded, smooth shaping such as `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative \
distances, so that no single term dominates.
4. Contact: `ctx.link_shoe_force[:, f, k]` is the contact force between link k (0 the link moved by joint `_3`, 1 the link \
moved by joint `_4`, 2 the fingertip) of finger f (0 thumb, 1 index, 2 middle, 3 ring, 4 pinky) and the shoe to grasp, and \
`ctx.palm_shoe_force` the force between the palm and that shoe. Contact is measured only on these links and the palm; contacts \
with anything else (the table, the rack, the other shoe, the hand itself) are not included. A value of 0 means that link is \
not touching the shoe. Contact forces come from the physics engine each step and can spike. These forces are only available \
to the reward; the policy itself does not observe contact.
5. Geometry: `ctx.shoe_surface` holds {surface_points} points on the shoe's outer surface. `ctx.palm_gap` is the distance from \
the palm frame origin to the nearest of them and `ctx.link_shoe_gap[:, f, k]` the same for each finger link. `ctx.palm_normal` \
points out of the palm's grasping side. `ctx.dz_free` is how far the lowest surface point has risen above its height at the \
start of the episode, and it is 0 whenever any surface point is above the rack's footprint widened by {rack_margin_cm:.0f} cm, \
so pushing the shoe onto the rack is not a lift. `ctx.shoe_shift_xy` is the horizontal distance of the shoe from its starting \
position. `ctx.slip_speed` is the speed of the shoe's centre of mass relative to the palm body frame (0 while the shoe moves \
rigidly with the hand, also when the hand rotates). `ctx.thumb_curl` is how far the thumb's `_3` joint has moved from its \
open angle towards its grip angle (negative means bent back past the open angle).
6. Hold and success are computed by the environment and cannot be redefined by the reward. A step is held when \
`ctx.lift_height <= ctx.dz_free <= ctx.lift_max`, `ctx.shoe_shift_xy <= ctx.hold_xy_radius`, \
`ctx.thumb_curl >= ctx.thumb_curl_min`, `ctx.palm_shoe_dist <= ctx.hold_radius` and `ctx.slip_speed < ctx.hold_slip_speed` \
are all true; `ctx.held` is that flag and `ctx.hold_count` counts consecutive held steps (it drops to 0 on a step that is not \
held). `ctx.latched` becomes True once `ctx.hold_count` reaches `ctx.latch_steps` and stays True for the rest of the episode. \
A success is counted on the step where `ctx.hold_count >= ctx.success_steps` and the shoe's speed \
`ctx.shoe_lin_vel.norm(dim=-1)` is below `ctx.success_speed`; `ctx.success` is True on that step. You may add a bonus on \
`ctx.success` or `ctx.latched`.
7. The episode ends on a success; when `ctx.shoe_pos[:, 2]` falls below {drop_z:.2f} m (the shoe fell off the table); when \
the episode has latched, has not succeeded and `ctx.dz_free` drops below {lost_cm:.0f} cm (the shoe went back down after the \
hold latched); or after `ctx.episode_steps - 1` steps. Nothing is added to the reward outside your function.
8. Do not keep any state between calls (no globals, no attributes); the function must be pure.
9. Each component you return is logged separately during training (as `t2r_reward/<name>`, the sum as `t2r_reward/total`) and \
may be shown back to you after training, so name components meaningfully (e.g. "approach", "lift", "success_bonus").
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


def task_text(cfg: gs.Stage1RewardCfg) -> str:
    return (f"Grasp the shoe with the left hand, lift it between {cfg.lift_height_m * 100:.0f} and {cfg.lift_max_m * 100:.0f} cm "
            "above where it rested, and hold it steady near where it was picked up until the environment counts a success.")


def render_prompt(spec: PromptSpec) -> str:
    cfg = gs.Stage1RewardCfg()
    robot = ROBOT_DESCRIPTION.format(
        rack_x_min=layout.RACK_X_RANGE[0], rack_x_max=layout.RACK_X_RANGE[1], rack_y_min=layout.RACK_Y_RANGE[0],
        rack_y_max=layout.RACK_Y_RANGE[1], rack_top_z=layout.RACK_TOP_Z, gap_lo=START_PALM_GAP_MM[0], gap_hi=START_PALM_GAP_MM[1],
        start_noise_cm=START_NOISE_XY_M * 100, mass_lo=MASS_SCALE_RANGE[0], mass_hi=MASS_SCALE_RANGE[1], fric_lo=FRICTION_RANGE[0],
        fric_hi=FRICTION_RANGE[1], com_cm=COM_OFFSET_M * 100, obs_noise=OBSERVATION_NOISE, quat_noise=QUAT_NOISE_RAD,
        act_noise=ACTION_NOISE, p_lo=WRENCH_PROB_RANGE[0], p_hi=WRENCH_PROB_RANGE[1], force_per_kg=WRENCH_FORCE_PER_KG,
        torque_per_kg=WRENCH_TORQUE_PER_KG, control_dt=CONTROL_DT_S, pos_scale=ACTION_POS_SCALE_M, rot_scale=ACTION_ROT_SCALE_RAD,
        alpha=gs.HAND_EMA_ALPHA, joint_table=_joint_table(),
    )
    knowledge = ADDITIONAL_KNOWLEDGE.format(surface_points=gs.SURFACE_POINT_COUNT, rack_margin_cm=gs.RACK_MARGIN_M * 100,
                                            drop_z=DROP_Z_M, lost_cm=cfg.lift_deadband_m * 100)
    parts = [
        "You are an expert in robotics, reinforcement learning and code generation.",
        robot,
        "Now I want you to help me write a reward function for reinforcement learning.",
        REWARD_STRUCTURE,
        "The reward function receives a single argument `ctx`, an instance of this class "
        "(all positions env-local, metres; angles rad; forces N):",
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
