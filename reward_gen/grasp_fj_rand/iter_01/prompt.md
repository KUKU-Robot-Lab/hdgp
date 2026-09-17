You are an expert in robotics, reinforcement learning and code generation.

We control one 7-DOF OpenArm robot arm (the right arm) carrying a five-finger Tesollo DG-5F hand, standing at a table. +x points from the robot toward the table. Every episode starts with the arm raised beside the robot: the palm is just outside the table edge nearest the robot, about 0.25 m above the table top, turned sideways, with the fingers pointing forward over the table edge. A cup stands upright on the table at a position drawn uniformly at random for every episode, with x between 0.10 m and 0.40 m and y between -0.30 m and 0.00 m, so at the start the palm is anywhere from about 0.15 m to 0.49 m from the cup (0.31 m on average) and the hand has to go to wherever the cup is; ctx.cup_pos gives its position. Near the far corners of that region (y close to -0.30 m, or x close to 0.40 m) the arm may need the hand turned by up to about 40 degrees from its start orientation to reach the cup's -y side. Parallel environments use different cups (open cups of several sizes and a closed shaker), so the graspable radius (44 mm to 81 mm) and half height (42 mm to 65 mm) differ between environments and are given per environment. There is only this one start pose: approach_done is never set when an episode starts. Positions are in metres in each environment's local frame, with +z pointing up; the table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (26,), float32)` with direct joint control. There is no grasp primitive, no hand synergy and no automatic finger stopping:
    actions[0:7]  = arm joint increments: each arm joint target moves by 0.05 * a rad per step and then passes a first-order filter (factor 0.1), so each arm joint moves at most about 0.3 rad/s.
    actions[7:26] = finger joint targets: each value is mapped linearly onto that joint's commandable range (a = -1 gives the lower limit, a = +1 the upper limit) and then low-pass filtered (factor 0.1). The order, which is also the order of ctx.hand_q / hand_q_norm / hand_target_norm / hand_qd, is:
    actions[ 7] = hand index  0: r_hj_thumb_2   range [-1.581, -1.561] rad  locked
    actions[ 8] = hand index  1: r_hj_thumb_3   range [+0.000, +1.571] rad  movable
    actions[ 9] = hand index  2: r_hj_thumb_4   range [+0.000, +1.571] rad  movable
    actions[10] = hand index  3: r_hj_index_1   range [-0.010, +0.010] rad  locked
    actions[11] = hand index  4: r_hj_index_2   range [+0.000, +2.007] rad  movable
    actions[12] = hand index  5: r_hj_index_3   range [+0.000, +1.571] rad  movable
    actions[13] = hand index  6: r_hj_index_4   range [+0.000, +1.571] rad  movable
    actions[14] = hand index  7: r_hj_middle_1  range [-0.010, +0.010] rad  locked
    actions[15] = hand index  8: r_hj_middle_2  range [+0.000, +2.007] rad  movable
    actions[16] = hand index  9: r_hj_middle_3  range [+0.000, +1.571] rad  movable
    actions[17] = hand index 10: r_hj_middle_4  range [+0.000, +1.571] rad  movable
    actions[18] = hand index 11: r_hj_ring_1    range [-0.010, +0.010] rad  locked
    actions[19] = hand index 12: r_hj_ring_2    range [+0.000, +1.920] rad  movable
    actions[20] = hand index 13: r_hj_ring_3    range [+0.000, +1.571] rad  movable
    actions[21] = hand index 14: r_hj_ring_4    range [+0.000, +1.571] rad  movable
    actions[22] = hand index 15: r_hj_pinky_1   range [+0.000, +0.010] rad  locked
    actions[23] = hand index 16: r_hj_pinky_2   range [-0.010, +0.010] rad  locked
    actions[24] = hand index 17: r_hj_pinky_3   range [+0.000, +1.571] rad  movable
    actions[25] = hand index 18: r_hj_pinky_4   range [+0.000, +1.571] rad  movable
Joints marked "locked" have a range of 0.05 rad or less and effectively do not move. The thumb's first joint is welded in this hand; `thumb_2`, which rotates the thumb into opposition, is locked at the opposed angle, and the thumb flexes with `thumb_3` and `thumb_4`. On the index, middle and ring fingers `_1` spreads the finger sideways (locked), `_2` flexes the knuckle, and `_3` and `_4` flex the two outer joints. On the pinky, `pinky_1` and `pinky_2` are locked and the finger flexes with `pinky_3` and `pinky_4`. For every movable joint a larger angle means a more flexed (more closed) finger. The finger joints are position-controlled (PD), so a finger that meets the cup stops there and presses with a force that grows with the gap between its target angle and its actual angle. One control step is 1/60 s.

Now I want you to help me write a reward function for reinforcement learning.

Typically, the reward function of a manipulation task consists of these parts (some are optional — include them only if really necessary):
1. the distance between the robot's hand and the target object
2. the difference between the object's current state and its goal state
3. regularization of the robot's action
4. [optional] extra constraints on the target object implied by the task
5. [optional] extra constraints on the robot implied by the task

The reward function receives a single argument `ctx`, an instance of this class (all positions env-local, metres; angles rad; forces N):

```python
@dataclass(frozen=True)
class RewardContext:
    # ---- constants (python numbers, fixed for the whole run) ---------------------------
    table_z: float                 # height of the table top [m]
    lift_latch_height: float       # the env sets `lifted` once the cup has risen this far above its starting height [m]
    success_hold_steps: int        # consecutive steps with goal_dist <= success_tol needed to count one success; a success also requires, on that step, that the fingers are wrapped around the cup (the env measures how far the finger links enclose the cup body and requires it above a threshold that starts low and rises as the policy succeeds more often)
    max_successes: int             # the episode ends after this many successes

    # ---- hand: right Tesollo DG-5F, finger index 0 thumb, 1 index, 2 middle, 3 ring, 4 pinky ----
    palm_pos: torch.Tensor         # (N,3) centre of the palm (a virtual point on the palm, not a collision surface)
    palm_normal: torch.Tensor      # (N,3) unit vector pointing out of the palm surface, towards an object held in the hand
    palm_side: torch.Tensor        # (N,3) unit vector lying in the palm plane (palm frame y axis)
    palm_finger_dir: torch.Tensor  # (N,3) unit vector lying in the palm plane, pointing from the palm towards the fingers (palm frame z axis); palm_normal, palm_side, palm_finger_dir form a right-handed frame. In the start pose palm_normal points along +y and palm_finger_dir along +x
    link_pos: torch.Tensor         # (N,5,3,3) finger link positions: [:, f, 0] link moved by joint _3, [:, f, 1] link moved by joint _4, [:, f, 2] fingertip
    link_cup_force: torch.Tensor   # (N,5,3) contact force magnitude between each of those links and the cup only [N] (0 = not touching the cup)
    palm_cup_force: torch.Tensor   # (N,) contact force magnitude between the palm and the cup only [N]
    hand_q: torch.Tensor           # (N,19) finger joint angles [rad], order = the hand joint table in the robot description
    hand_q_norm: torch.Tensor      # (N,19) joint angles normalised to each joint's commandable range: 0 = lower limit (straight), 1 = upper limit (most flexed)
    hand_target_norm: torch.Tensor  # (N,19) commanded finger joint targets (after filtering), same normalisation as hand_q_norm
    hand_default_q_norm: torch.Tensor  # (N,19) the hand's default pose: the finger joint angles every episode starts with, same normalisation as hand_q_norm (the same in every environment); in this pose the four fingers are straight, and the thumb, rotated into opposition, is straight and points along palm_normal, reaching about 0.12 m out from the palm surface at the wrist end of the palm (behind palm_pos along palm_finger_dir)
    hand_qd: torch.Tensor          # (N,19) finger joint velocities [rad/s]
    hand_z_min: torch.Tensor       # (N,) height of the lowest finger/thumb link [m]; the palm is not included in this one
    palm_clearance: torch.Tensor   # (N,) height of the lowest point of the palm itself above the table top [m]: 0 means the palm is resting on the table, negative means it is pressed into it. The palm is a large flat body and palm_pos is a virtual point up to 0.065 m away from its surface, so palm_pos alone cannot tell you whether the palm is on the table; this is the measured distance. Note that the episode-ending floor check looks only at the finger and thumb links, so the palm resting on the table does not end the episode by itself

    # ---- arm: 7-DOF ------------------------------------------------------------------------
    arm_q: torch.Tensor            # (N,7) arm joint angles [rad]
    arm_qd: torch.Tensor           # (N,7) arm joint velocities [rad/s]

    # ---- cup: a roughly cylindrical cup standing upright on the table at the start ----------
    cup_pos: torch.Tensor          # (N,3) cup reference point on its axis; the graspable band (cup_half_height above and below) is centred on it
    cup_quat: torch.Tensor         # (N,4) cup orientation quaternion (w,x,y,z)
    cup_axis: torch.Tensor         # (N,3) unit vector along the cylinder axis towards its top; (0,0,1) when upright
    cup_tilt: torch.Tensor         # (N,) angle between cup_axis and world +z [rad]
    cup_lin_vel: torch.Tensor      # (N,3) cup linear velocity [m/s]
    cup_ang_vel: torch.Tensor      # (N,3) cup angular velocity [rad/s]
    cup_spawn_pos: torch.Tensor    # (N,3) cup_pos at the start of the episode, resting on the table
    cup_radius: torch.Tensor       # (N,) radius of the cup's outer surface in the graspable band [m]; differs between environments
    cup_half_height: torch.Tensor  # (N,) half height of the graspable band, centred on cup_pos along cup_axis [m]

    # ---- goal and task status: computed by the environment --------------------------------
    goal_pos: torch.Tensor         # (N,3) position where cup_pos must be held (with the cup upright)
    goal_dist: torch.Tensor        # (N,) largest distance between keypoints fixed on the cup and the same keypoints at the goal pose [m]
    success_tol: torch.Tensor      # (N,) current success tolerance on goal_dist [m]; shrinks as training succeeds
    lifted: torch.Tensor           # (N,) bool, True once the cup has risen above lift_latch_height in this episode (stays True)
    success: torch.Tensor          # (N,) bool, True on the step a success is counted
    num_successes: torch.Tensor    # (N,) number of successes counted so far in this episode
    episode_progress: torch.Tensor  # (N,) elapsed fraction [0,1] of the step budget; the budget restarts after every success

    # ---- stage completion flags: kept by the environment, stay True until the episode ends ----
    approach_done: torch.Tensor    # (N,) bool, set on the first step on which all of these held at once, i.e. the cup sat between the extended thumb and the four fingers in front of the palm: the palm plane was within 2 cm of the cup's side with the cup in front of the palm (the distance from palm_pos to the cup axis along palm_normal, minus cup_radius, was between -0.01 m and 0.02 m); the cup axis was ahead of palm_pos along palm_finger_dir by between cup_radius - 0.005 m and cup_radius + 0.02 m; palm_pos was within the height of the graspable band (at most cup_half_height from cup_pos along cup_axis); the hand kept its start-pose orientation (palm_normal within about 45 degrees of +y and palm_finger_dir within about 45 degrees of +x); every movable finger joint was within 0.3 of hand_default_q_norm (joints with a range of 0.05 rad or less are ignored); and no finger or thumb link touched the cup (contact force above 0.1 N). Episodes that start beside the cup (see the scene description) have approach_done already set on the first step
    envelope_done: torch.Tensor    # (N,) bool, set once, after approach_done, the palm, the thumb and at least 4 digits in total (the thumb included) touched the cup for 5 consecutive steps; the palm or a finger counts as touching when its contact force with the cup is above 0.1 N

    # ---- actions ---------------------------------------------------------------------------
    actions: torch.Tensor          # (N,26) policy action of this step, clipped to [-1,1]; each step the joints receive one of the last 3 policy actions picked at random (a 0-2 step delay)
    prev_actions: torch.Tensor     # (N,26) policy action of the previous step (zeros right after a reset)
```

Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel environments). The function must return a reward tensor of shape (N,) — never a Python float — and a dict of named component tensors, each of shape (N,). Use only `torch` and `math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors, e.g. `staged = torch.where(ctx.lifted, r_hold, torch.zeros_like(r_hold))`.
3. Prefer bounded, smooth shaping such as `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative distances, so that no single term dominates.
4. Contact: `ctx.link_cup_force[:, f, k]` is the contact force between link k (0 the link moved by joint `_3`, 1 the link moved by joint `_4`, 2 the fingertip) of finger f (0 thumb, 1 index, 2 middle, 3 ring, 4 pinky) and the cup, and `ctx.palm_cup_force` the force between the palm and the cup. Contact is measured only on these links and the palm — the finger links nearest the palm are not measured. Contacts with anything else (the table, the hand itself) are not included. A value of 0 means that link is not touching the cup. Contact forces come from the physics engine each step and can spike. These forces are only available to the reward; the policy itself does not observe contact.
5. Cylinder geometry (the cups are roughly cylindrical): for a point p (e.g. `ctx.link_pos[:, f, k]`), with `v = p - ctx.cup_pos`, the axial coordinate is `h = (v * ctx.cup_axis).sum(-1)` and the radial vector is `v - h.unsqueeze(-1) * ctx.cup_axis`; its norm is the distance from the cup axis. The cup surface is at `ctx.cup_radius` and the graspable band is `|h| <= ctx.cup_half_height`. The palm normal `ctx.palm_normal` points out of the palm.
6. Height: `ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]` is how far the cup has been raised above its starting height (0 while it rests on the table). `ctx.lifted` becomes True once that height exceeds `ctx.lift_latch_height` and stays True for the rest of the episode, even if the cup is dropped again.
7. Goal and success are computed by the environment and cannot be redefined by the reward. The goal position is 0.21 m to 0.28 m above the cup's starting position and at most 5 cm away from it horizontally, with the cup upright. A success is counted when `ctx.goal_dist <= ctx.success_tol` for `ctx.success_hold_steps` consecutive steps and, on that same step, the fingers are wrapped around the cup: the environment measures how far the finger links enclose the cup body and requires that measure to be above a threshold, which starts low and rises as the policy succeeds more often. A cup that reaches the goal without being held that way does not count as a success. `ctx.success` is True on the step a success is counted. After a success the next goal is at the same place, so holding the cup still there keeps producing successes until `ctx.max_successes`, which ends the episode. The keypoints are fixed on the cup, so tilting the cup also increases `goal_dist`. `ctx.success_tol` starts at 0.1125 m and shrinks towards 0.015 m as the policy succeeds more often during training. You may add a bonus on `ctx.success`.
8. An episode lasts at most 900 steps (15 s), and the step budget restarts after every success. The episode ends early when the cup falls below z = 0.15 (off the table), leaves the allowed area around the table, or tilts more than 60 degrees; when any hand link goes below z = table_z - 0.005 (wherever the hand is; this check does not include the palm, whose own clearance is ctx.palm_clearance); or when an arm joint goes past its limit or moves faster than 20 rad/s. On that last kind of physics violation the environment replaces the reward of that step with a fixed -1 (your function's value is not used on that step).
9. Do not keep any state between calls (no globals, no attributes); the function must be pure.
10. Each component you return is logged separately during training and may be shown back to you after training, so name components meaningfully (e.g. "approach", "lift", "success_bonus").

I want it to fulfil the following task: Pick up the upright cup in three stages, strictly in this order. The cup is placed at a different position on the table in every episode.
Stage 1, approach: starting from the raised pose beside the table, go to wherever the cup has been placed and bring the hand to it without turning it more than the cup's position requires: keep the hand's start orientation (ctx.palm_normal along +y, towards the cup, and ctx.palm_finger_dir along +x) as closely as the reach allows; the approach accepts up to about 45 degrees of turn. Keep the hand in its default pose (ctx.hand_default_q_norm), so the fingers and the thumb stay open. In the default pose the thumb sticks straight out of the palm towards the cup at the wrist end of the palm, so the hand has to come to the cup's -y side with the cup between the thumb and the four fingers: the palm close to the cup's side, the cup further along the fingers than the thumb, and neither the thumb nor any finger touching the cup yet. The approach is complete when the environment sets ctx.approach_done, which then stays set for the rest of the episode.
Stage 2, envelope grasp: only after the approach is complete, from beside the cup and with the same hand orientation, move the palm against the cup's side and close the hand around the cup into an envelope grasp (a power grasp): the four fingers wrapping around the cup body on the far side with their inner surfaces, and the thumb closing on the near side, opposite the fingers. The environment sets ctx.envelope_done once the palm, the thumb and enough fingers hold the cup for several consecutive steps.
Stage 3, lift: only after the envelope grasp is formed, keep that grasp and lift the cup straight up to the goal position, then hold it there, upright and still, so that the environment keeps counting successes.
No stage may earn reward before the previous stage is complete, and behaviour that skips or reorders the stages must not earn reward. The cups differ in shape and size between environments. Do not knock the cup over, do not drop it, and do not push the hand into the table.
1. Please think step by step and tell me what this task means and which stages the robot must go through.
2. Then write a function with exactly this signature:
```python
def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, {"component_name": component_tensor, ...}
```
   where `reward` has shape (N,) and every dict value has shape (N,).
3. Take care of tensor shapes and types; never access a field or method that is not listed in the class definition above.
4. Put the final code in ONE ```python fenced block that starts with `import torch` and `import math` and contains only the function (plus small helper functions if needed). Add short comments explaining the weights.

Generated code shown as below:
```python
import torch
import math


# Movable finger joints (range > 0.05 rad), grouped per digit in hand-joint order:
# thumb (3,4), index (2,3,4), middle (2,3,4), ring (2,3,4), pinky (3,4).
_DIGIT_JOINTS = ([1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18])
_MOVABLE = [j for group in _DIGIT_JOINTS for j in group]


def _sat(x: torch.Tensor, full: float) -> torch.Tensor:
    # 0 at/below 0, 1 at/above `full`; the clamp also bounds physics force spikes
    return torch.clamp(x / full, 0.0, 1.0)


def _win(x: torch.Tensor, lo, hi, soft: float) -> torch.Tensor:
    # 1 inside [lo, hi], smooth gaussian decay outside. lo/hi may be tensors (per-cup windows).
    d = torch.relu(lo - x) + torch.relu(x - hi)
    return torch.exp(-(d / soft) ** 2)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Global scale unchanged from the round that produced the working approach + envelope + lift.
    # Per-step totals: good approach ~0.3-0.5, envelope parked on the table ~2.6, the same grasp
    # hovering at height ~2.2, that grasp carried to the goal and held still ~8.9, +15 per success.
    scale = 0.05

    r = ctx.cup_radius                 # (N,)
    hh = ctx.cup_half_height           # (N,)
    n = ctx.palm_normal
    fd = ctx.palm_finger_dir
    axis = ctx.cup_axis

    # ---------------- stage masks (strict order, flags are latched by the env) ----------------
    approach = ctx.approach_done.bool()
    envelope = ctx.envelope_done.bool() & approach
    s1 = (~approach).float()
    s2 = (approach & ~envelope).float()
    s3 = envelope.float()
    progress = torch.clamp(ctx.episode_progress, 0.0, 1.0)

    # ---------------- palm pose relative to the cup (the quantities approach_done checks) -----
    v = ctx.cup_pos - ctx.palm_pos
    v_ax = (v * axis).sum(-1)
    v_perp = v - v_ax.unsqueeze(-1) * axis
    gap_n = (v_perp * n).sum(-1) - r    # palm plane -> cup surface; NEGATIVE = palm pressed in
    gap_f = (v_perp * fd).sum(-1) - r   # cup axis ahead of the palm along the fingers, minus r
    h_palm = -v_ax                      # palm height above the cup centre along the cup axis

    # start orientation kept (palm_normal +y, palm_finger_dir +x); ~0.46 at 20 deg on both axes
    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))
    ang_f = torch.acos(torch.clamp(fd[:, 0], -1.0, 1.0))
    ori = torch.exp(-(ang_n ** 2 + ang_f ** 2) / (2.0 * 0.4 ** 2))

    # open default hand pose kept (the env needs every movable joint within 0.3 of default)
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    dev_max = dev.amax(-1)
    open_pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev_max / 0.2) ** 2)
    keep = ori * open_pose

    # ---------------- measured contact --------------------------------------------------------
    touch = _sat(ctx.link_cup_force, 0.5)         # (N,5,3)
    digit_touch = touch.amax(-1)                  # (N,5)
    thumb_touch = digit_touch[:, 0]
    # sensitive version for the env's "no digit may touch" approach condition (env threshold 0.1 N)
    any_touch = _sat(ctx.link_cup_force, 0.2).amax(-1).amax(-1)      # (N,)
    palm_touch = _sat(ctx.palm_cup_force, 0.5)    # (N,)
    palm_firm = _sat(ctx.palm_cup_force, 2.0)     # rewards actually pressing, not grazing

    # ---------------- table clearance: unchanged - this requirement is met and is not given back
    # `hand_z_min` covers the finger/thumb links only; the palm is charged separately from the
    # MEASURED ctx.palm_clearance (the virtual palm point paid exactly 0 two rounds ago).
    clear = ctx.hand_z_min - ctx.table_z
    near_tab = torch.clamp((0.025 - clear) / 0.025, 0.0, 1.0)   # 0 at 2.5 cm, 1 at contact
    deep_tab = torch.clamp((0.008 - clear) / 0.013, 0.0, 1.0)   # 0 at 8 mm, 1 at the -5 mm cutoff
    table_pen = -(8.0 * near_tab ** 2 + 30.0 * deep_tab ** 2)
    pc = ctx.palm_clearance
    # free above 8 mm only: for the smallest cup a correct mid-body grasp puts the palm's lowest
    # point millimetres above the table, so a higher threshold would punish the working envelope
    palm_low = torch.clamp((0.008 - pc) / 0.008, 0.0, 1.0)
    palm_dig = torch.clamp(-pc / 0.010, 0.0, 1.0)
    palm_table_pen = -(8.0 * palm_low ** 2 + 40.0 * palm_dig ** 2)
    # second, multiplicative pressure: the grasp ladder is halved on the surface
    air = (0.5 + 0.5 * torch.clamp(clear / 0.020, 0.0, 1.0)) \
        * (0.7 + 0.3 * torch.clamp(pc / 0.010, 0.0, 1.0))

    # ================ STAGE 1: approach - KEPT EXACTLY (0.000 -> 0.742 far-start, no collapse) ==
    # (i) the six approach_done conditions, scored individually: both the mean (dense partial
    # credit) and the product (they must all hold on the SAME step for the env to latch).
    c_gap = _win(gap_n, -0.010, 0.020, 0.015)
    c_along = _win(gap_f, -0.005, 0.020, 0.015)
    c_height = _win(h_palm, -0.9 * hh, 0.9 * hh, 0.020)
    c_ori = _win(ang_n, 0.0, 0.60, 0.30) * _win(ang_f, 0.0, 0.60, 0.30)   # env allows ~45 deg
    c_pose = _win(dev_max, 0.0, 0.22, 0.12)                               # env allows 0.3
    c_free = 1.0 - any_touch                                              # nothing may touch yet
    conds = torch.stack([c_gap, c_along, c_height, c_ori, c_pose, c_free], dim=-1)   # (N,6)
    form = conds.mean(-1)
    lock = conds.prod(-1)

    # (ii) distance to the pre-grasp window, aimed at the MIDDLE of the cup body
    a_n = torch.relu(gap_n - 0.018) + torch.relu(-0.008 - gap_n)
    a_f = torch.relu(gap_f - 0.018) + torch.relu(-0.003 - gap_f)
    a_h = torch.relu(h_palm - 0.025) + torch.relu(-0.012 - h_palm)
    d_app = torch.sqrt(a_n ** 2 + a_f ** 2 + a_h ** 2 + 1e-12)
    prox = torch.exp(-(d_app / 0.12) ** 2)
    fine_app = torch.exp(-(d_app / 0.05) ** 2)

    # (iii) long-range travel; k = 1.8 keeps the 0.38 m raised start on a real slope
    off_z = torch.clamp(0.30 * hh, max=0.020)
    offset = torch.stack([-(r + 0.006), -(r + 0.008), off_z], dim=-1)
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(1.8 * d_world)

    dwell1 = 1.0 - 0.3 * progress          # dawdling in stage 1 is mildly worse than finishing it
    shape_k = 0.15 + 0.85 * keep           # reaching the cup by rotating the wrist forfeits 85 %
    approach_travel = 4.0 * s1 * dwell1 * (0.55 * coarse + 0.45 * fine_app) * shape_k
    approach_form = 3.0 * s1 * dwell1 * form * (0.3 + 0.7 * prox)
    approach_lock = 2.0 * s1 * lock        # paid only when all six latch conditions hold at once
    approach_keep = 1.0 * s1 * keep * (0.2 + 0.8 * prox)
    # stage 1 max = 10.0, reachable only in the configuration that makes the env latch the flag

    # ---------------- grasp pose (wider than the latch window: the palm may press in) ----------
    e_n = torch.relu(gap_n - 0.010) + torch.relu(-0.015 - gap_n)
    e_f = torch.relu((gap_f - 0.006).abs() - 0.010)
    e_h = torch.relu(h_palm - 0.025) + torch.relu(-0.012 - h_palm)   # middle of the cup body
    d_pose = torch.sqrt(e_n ** 2 + e_f ** 2 + e_h ** 2 + 1e-12)
    pos = torch.exp(-(d_pose / 0.05) ** 2)

    # ---------------- link geometry in the cup / palm frame -----------------------------------
    lv = ctx.link_pos - ctx.cup_pos[:, None, None, :]      # (N,5,3,3)
    ax = axis[:, None, None, :]
    lh = (lv * ax).sum(-1)                                 # (N,5,3) height along the cup axis
    lrad_vec = lv - lh.unsqueeze(-1) * ax
    lrad = torch.norm(lrad_vec, dim=-1)
    u_n = (lrad_vec * n[:, None, None, :]).sum(-1)         # radial component along the palm normal
    u_f = (lrad_vec * fd[:, None, None, :]).sum(-1)        # radial component along the fingers
    in_band = torch.clamp(1.0 - torch.relu(lh.abs() - hh[:, None, None]) / 0.02, 0.0, 1.0)
    near = torch.exp(-torch.relu(lrad - r[:, None, None] - 0.012) / 0.02) * in_band
    engage = torch.maximum(touch, near).amax(-1)           # (N,5) digit is at the cup body

    # wrap angle around the cup: 0 = palm side, pi/2 = forward side, pi = far side
    theta = torch.atan2(u_f, -u_n)
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)  # saturates ~108 deg, round the far side
    finger_wrap = (touch[:, 1:, :] * wrap[:, 1:, :]).amax(-1)        # (N,4) contact-weighted
    wrap_q = torch.clamp(finger_wrap.sum(-1) / 3.0, 0.0, 1.0)        # 3 wrapped fingers saturate
    wrap_deep = finger_wrap.mean(-1)                                 # all four, no saturation
    # thumb must close on the -fd side, opposite the fingers: 1 at u_f=-r, 0.5 at the side, 0 at +r
    thumb_dir = torch.clamp(0.5 - u_f[:, 0, :] / (2.0 * r[:, None]), 0.0, 1.0)
    thumb_q = (touch[:, 0, :] * (0.4 + 0.6 * thumb_dir)).amax(-1)
    fingers_q = torch.clamp(digit_touch[:, 1:].sum(-1) / 3.0, 0.0, 1.0)  # 3 of 4; pinky may substitute

    # ---------------- finger closure (flexion past the open default pose) ---------------------
    flex = torch.clamp((ctx.hand_q_norm - ctx.hand_default_q_norm) / 0.5, 0.0, 1.0)
    digit_flex = torch.stack([flex[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)  # PD command leads the angle on a touching digit

    # ---------------- palm progress (smooth over the whole remaining travel) ------------------
    press_geo = torch.exp(-torch.relu(gap_n + 0.010) / 0.025)   # 1 at -1 cm, 0.67 at 0, 0.30 at +2 cm
    press = torch.maximum(palm_touch, 0.7 * press_geo)          # geometry alone caps at 0.7
    # CHANGE C: has the palm actually arrived? 0 while it is still >= 2 cm out, 1 once it is on the
    # cup. The pre-grasp shape is paid, and closing is discounted, exactly while this is 0.
    arrived = torch.maximum(palm_touch, torch.clamp((0.020 - gap_n) / 0.020, 0.0, 1.0))
    not_arrived = 1.0 - arrived
    close_gate = 0.25 + 0.75 * arrived     # closing from the approached pose pays 4x diving in

    # ---------------- stage gates -------------------------------------------------------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    upright = torch.clamp(1.0 - (ctx.cup_tilt - 0.35) / 0.35, 0.0, 1.0)   # 1 below 20 deg, 0 at 40
    # carrying uprightness: 1 below 5 deg, 0.78 at 10, 0.35 at 15, 0.002 at the 30 deg seen on video
    upright_c = torch.exp(-(torch.relu(ctx.cup_tilt - 0.09) / 0.17) ** 2)
    resting = torch.clamp(1.0 - (dz - 0.03) / 0.03, 0.0, 1.0)             # no stage-2 pay once airborne
    ori_k = 0.5 + 0.5 * ori                     # turning the hand halves the grasp, in stage 3 too
    dwell2 = 1.0 - 0.5 * progress               # anti-farming: parking in stage 2 decays
    dwell3 = 1.0 - 0.5 * progress               # CHANGE A: and so does parking in the air
    g2 = s2 * ori_k * upright * resting * air * dwell2
    g3 = s3 * ori_k * upright * air     # the same ladder keeps paying while lifting -> releasing loses it
    g23 = g2 + g3

    # CHANGE C: hold the pre-grasp shape from the latch until the palm is against the cup. Fades
    # out exactly as the ladder turns on, so there is no step at which breaking it early is free.
    pregrasp_shape = 5.0 * s2 * air * dwell2 * not_arrived * keep * pos
    # no digit may touch before the flag, nor before the palm arrives after it (-0.0250 was paid
    # willingly last round while per-step no-touch fell 0.968 -> 0.240)
    early_contact_pen = -5.0 * s1 * any_touch - 4.0 * s2 * not_arrived * any_touch

    # ================ STAGE 2/3: the grasp ladder (unchanged, CAPPED at ~51 raw) ================
    # rung 0 is set to 10.0 = stage 1's ceiling, so the step the flag latches the reward steps UP.
    grasp_pose = 10.0 * g23 * pos
    # rung 1: the last centimetres of palm travel
    palm_reach = 6.0 * g23 * pos * press_geo
    # rung 2: measured palm contact, the condition that blocked the envelope for two rounds
    palm_contact = g23 * (0.3 + 0.7 * pos) * (5.0 * palm_touch + 2.0 * palm_firm)
    # rung 3: curling the digits round the cup - now also gated on the palm having arrived
    finger_curl = 5.0 * g23 * pos * close_gate * (digit_flex[:, 1:] * engage[:, 1:]).mean(-1)
    thumb_curl = 2.5 * g23 * pos * close_gate * digit_flex[:, 0] * engage[:, 0]
    light_contact = 1.0 * g23 * torch.clamp(digit_touch.sum(-1) / 4.0, 0.0, 1.0)   # keep-alive
    # rung 4: contact quality, all multiplied by palm progress -> a fingertip grasp gets a fraction
    grasp_wrap = g23 * press * (3.5 * wrap_q + 1.5 * wrap_deep)   # wrap_deep needs all four fingers
    grasp_thumb = 3.0 * g23 * press * thumb_q
    grasp_fingers = 3.0 * g23 * press * fingers_q
    grip_squeeze = 1.5 * g23 * press * squeeze
    pinky_join = 1.2 * g23 * press * digit_touch[:, 4] * digit_flex[:, 4]   # averaged away elsewhere
    # rung 5: the envelope_done condition itself (palm AND thumb AND >= 3 fingers)
    envelope_quality = 6.0 * g23 * palm_touch * thumb_touch * fingers_q
    # the ladder cannot grow past ~51 raw: grasp progress can never substitute for lift progress

    # ================ STAGE 3: get clear, then CARRY TO THE GOAL AND STOP ======================
    firm = _sat(ctx.link_cup_force, 0.3).amax(-1)                 # (N,5)
    grip = (0.3 + 0.7 * palm_touch) * firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)
    # CHANGE A: height income is now small and saturates at 5 cm, and both hover terms decay with
    # the step budget (which restarts on every success, so carrying-and-holding never decays).
    # CHANGE B: upright_c MULTIPLIES every payment below - a cup carried on its side earns ~0.
    lift_hold = 2.0 * s3 * grip * dwell3
    lift_clear = 6.0 * s3 * grip * upright_c * dwell3 * torch.clamp(dz / 0.05, 0.0, 1.0)
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    # the carrying gradient is now distance-to-goal, not height: monotone from the spawn distance
    # (~0.29 m -> 0.5 raw) through 15 cm (15.5) to the goal (30), and it peaks ONLY at the goal
    goal_close = 30.0 * s3 * grip * upright_c * (1.0 - torch.tanh(gd / 0.12))
    goal_near = 45.0 * s3 * grip * upright_c * torch.exp(-(gd / 0.08) ** 2)   # 6 at the start tol, 45 at 0
    goal_in_tol = 20.0 * s3 * grip * upright_c * in_tol
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) \
        * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 40.0 * s3 * grip * upright_c * torch.exp(-(gd / 0.08) ** 2) * calm   # arrive AND stop
    success_bonus = 300.0 * s3 * ctx.success.float()
    # ordering: stage 1 <= 10 < stage-2 entry ~11 < full envelope ~51 < hovering with it ~45 (and
    # decaying) << the same grasp carried upright to the goal and held ~178, +300 per success.

    # ---------------- constraints and regularization -------------------------------------------
    # nudges while closing are free up to 4 cm; only shoving the cup away costs more than grasping
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.04) / 0.10, 0.0, 1.0)
    # CHANGE B: tilt free below ~10 deg (was 15), -8 at the 60 deg termination (was -6). Against the
    # reduced hover income this is ~1:6, not the 1:78 that made carrying it sideways free.
    cup_tilt_pen = -8.0 * torch.clamp((ctx.cup_tilt - 0.17) / (math.pi / 3.0 - 0.17), 0.0, 1.0)
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.02 * (ctx.arm_qd ** 2).sum(-1)   # small: must not discourage carrying

    raw = {
        "approach_travel": approach_travel,
        "approach_form": approach_form,
        "approach_lock": approach_lock,
        "approach_keep": approach_keep,
        "pregrasp_shape": pregrasp_shape,
        "early_contact_pen": early_contact_pen,
        "grasp_pose": grasp_pose,
        "palm_reach": palm_reach,
        "palm_contact": palm_contact,
        "finger_curl": finger_curl,
        "thumb_curl": thumb_curl,
        "light_contact": light_contact,
        "grasp_wrap": grasp_wrap,
        "grasp_thumb": grasp_thumb,
        "grasp_fingers": grasp_fingers,
        "grip_squeeze": grip_squeeze,
        "pinky_join": pinky_join,
        "envelope_quality": envelope_quality,
        "lift_hold": lift_hold,
        "lift_clear": lift_clear,
        "goal_close": goal_close,
        "goal_near": goal_near,
        "goal_in_tol": goal_in_tol,
        "hold_still": hold_still,
        "success_bonus": success_bonus,
        "cup_disturb_pen": cup_disturb_pen,
        "cup_tilt_pen": cup_tilt_pen,
        "table_pen": table_pen,
        "palm_table_pen": palm_table_pen,
        "action_rate_pen": action_rate_pen,
        "arm_vel_pen": arm_vel_pen,
    }
    components = {name: scale * value for name, value in raw.items()}
    reward = torch.stack(list(components.values()), dim=0).sum(0)
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 1 of the random-placement track (`fj_rand_i00`) ran to epoch 2378 over 4.18 hours and ended at the hour cap. The reward is the unchanged round-9 reward from the fixed-placement track. The only change is the environment: the cup now appears anywhere in x 0.10-0.40 m, y -0.30-0.00 m every episode, and every episode starts far from the cup.

**Finding a cup that moves every episode was learned.** The approach latch went from zero at epoch 300 to 0.94 of far-start episodes by epoch 750. The palm-to-cup gap fell from 0.21 m to 0.04 m. Hand orientation passed 0.97 of the time and height 0.83, so turning the hand towards cups at the far corners is not what fails.

**The envelope grasp was learned on top of it.** The envelope latch jumped from zero to 0.76 between epochs 600 and 900. By the end, 3.2 digits and the palm (0.48 of the time) were in contact on average, 4.85 digits and the palm (0.41) at the moment of success, and all five digits were touching at success in almost every case. The recording shows the same: the hand arrives from the start pose, stops beside the cup with the palm facing it, and closes four fingers and the thumb around the lower body of the cup by about four seconds.

**Lifting barely happens, and after the grasp the hand moves down, not up.** This is the main failure. Over the last two hundred epochs the cup was airborne only 0.021 of the time, episodes that lifted at all were 0.07, mean lift height was 0.005 m, success was 0.0025 of episodes, and the tolerance never tightened from 0.1125. The lift fraction peaked at about 0.09 around epoch 1350 and drifted down over the following thousand epochs. In the recording the hand closes around the cup by about four seconds and then, instead of raising it, sinks steadily towards the table for the rest of the fifteen-second episode, pressing the wrapped cup down against the tabletop. The hand never starts upward. Consistent with this, the lowest point of the hand fell from 0.262 m to 0.249 m over the round, the table penalty grew from -0.028 to -0.033 and the palm-near-table penalty from -0.0003 to -0.0029.

**The reward explains why.** Over the last two hundred epochs, grasping the cup while it stands on the table paid about 0.70 per step. The largest terms were grasp pose 0.167, envelope quality 0.102, palm reach 0.098, grasp wrap 0.093, palm contact 0.090, finger and thumb grasp 0.130 combined, and finger curl 0.050. Lifting it added only lift hold 0.036, lift clearance 0.006 and goal closeness 0.008. Lifting also exposed the policy to more of the penalties that were already active: tilt -0.032, table -0.033 and arm speed -0.008 (the arm speed penalty grew from -0.003 at epoch 1000). Almost all of the income is available without ever lifting the cup.

**The approach latch settled at about 0.6 instead of 0.9.** After the envelope was learned, far-start episodes that fired the approach latch fell from 0.94 at epoch 750 to 0.59-0.66 from epoch 1050 onward, and the envelope latch fell with it to 0.55-0.60. The per-step condition logs cannot separate the cause, because most of every episode is spent already holding the cup. There is no per-placement breakdown. The one recorded episode completed the approach, so the recording does not show what the other forty percent do.

**Lifting motions are violent when they occur.** The 99th percentile of arm joint speed while the cup was airborne rose from 3.4 rad/s at epoch 750 to 6.0-7.1 rad/s at the end, and the action rate while airborne stayed around 0.21-0.30. Two separate windows (around epochs 1650 and 1950) show mean cup speed while airborne of 7.2 and 2.3 m/s against a normal 0.25 m/s, which means the cup was occasionally thrown. The cup tipped in only 0.0003 of steps, and mean cup tilt rose from 7.4 to 10.1 degrees over the round.

**Placement rejection behaved as designed.** 0.42 of resets resampled a placement to avoid the hand's shadow and the tabletop holes, and no reset ever failed to find a valid one.

The hand never rested on the table: the floor termination never fired, and palm-on-table stayed at 0.003.

To make the code more accurate and train better robot, the feedback for improvement is:
Three things went right and must be kept.

- The policy finds the cup wherever it is placed. It approaches in the default open hand pose, with the palm facing the cup and the correct height, from a far start, and it turns the hand for cups at the far corners. The approach latch reached 0.94 of episodes under full placement randomisation.
- The grasp is a true envelope grasp: palm against the cup, all five digits in contact at the moment of success.
- The approach-then-envelope ordering held. The envelope was learned only after the approach, and both stayed on for fifteen hundred epochs.

The rest is what has to change.

**After the envelope, the hand must go up, and lifting must pay clearly more than holding the cup on the table.** This is the main failure. In the recording the hand closes around the cup and then sinks towards the table and presses the cup down instead of raising it. Holding the cup wrapped on the table earns about 0.70 per step, and lifting it adds about 0.05, while lifting also brings on tilt, table and arm-speed penalties. The policy therefore closes its hand and stays. Once the envelope is complete, the grasp terms should stop growing and should not by themselves be worth staying for. They should act as the entry ticket to lifting, not as the income. The income after the envelope latch should come from the cup's height above its starting position, rising steadily as it rises, and from carrying it towards the goal. Upward motion of the hand and cup right after the envelope completes should be rewarded immediately and densely, so that lifting is learned quickly. Moving the hand or cup downward towards the table while holding the cup should earn less than keeping it still, and clearly less than raising it. A policy that closes around the cup and never lifts it should end up clearly worse off than one that lifts it. The cup's starting height now differs every episode because the cup model differs, so lift should be measured relative to where the cup started, not to a fixed height.

**Lifting must be smooth, and smoothness must not be the reason not to lift.** When the cup is lifted, the arm moves at up to 6-7 rad/s and the cup is occasionally thrown. Penalise fast arm motion and cup speed while the cup is held in a way that shapes how it is lifted. The penalty must stay small against the lift income, so that a slow, steady lift is always better than no lift.

**The approach should keep firing near 0.9 after the grasp is learned.** It fell from 0.94 to about 0.6 once the envelope was learned, and the envelope latch fell with it. If some episodes are now closing on the cup without first completing the open-hand approach, that path should earn nothing from the grasp or lift terms. The approach income should also not become worthless once grasp income appears. Keep paying for reaching the approached pose at the same rate late in training as early.

One caution. The success count this round is too small (0.0025) to read anything from it, and the tolerance never moved, so do not loosen the success condition to make this round look better. The failure is that the cup is not lifted, not that success is too hard to reach once it is.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.00317, 0.00342, 0.385, 0.621, 0.666, 0.706, 0.672, 0.683, 0.694]  min 0 · mean 0.464 · max 0.725
contact/finger_index_at_success: [-1, -1, 0.783, 0.783, 0.756, 0.868, 0.838, 0.997, 1, 1]  min -1 · mean 0.605 · max 1
contact/finger_middle: [0.00146, 0.00342, 0.000244, 0.336, 0.594, 0.662, 0.692, 0.674, 0.696, 0.705]  min 0 · mean 0.46 · max 0.739
contact/finger_middle_at_success: [-1, -1, 0.955, 0.955, 0.912, 0.922, 0.74, 0.928, 0.885, 0.931]  min -1 · mean 0.63 · max 1
contact/finger_pinky: [0.00146, 0, 0, 0.252, 0.345, 0.474, 0.468, 0.477, 0.533, 0.515]  min 0 · mean 0.316 · max 0.608
contact/finger_pinky_at_success: [-1, -1, 0, 0, 0, 0.351, 0.77, 0.928, 0.93, 0.911]  min -1 · mean 0.297 · max 0.999
contact/finger_ring: [0.0083, 0.00171, 0.000244, 0.299, 0.529, 0.661, 0.652, 0.635, 0.63, 0.677]  min 0 · mean 0.432 · max 0.709
contact/finger_ring_at_success: [-1, -1, 1, 1, 0.952, 0.974, 0.848, 0.997, 1, 1]  min -1 · mean 0.681 · max 1
contact/finger_thumb: [0, 0.00635, 0.00439, 0.374, 0.615, 0.686, 0.727, 0.688, 0.708, 0.713]  min 0 · mean 0.476 · max 0.752
contact/finger_thumb_at_success: [-1, -1, 1, 1, 1, 1, 0.984, 0.948, 1, 1]  min -1 · mean 0.704 · max 1
contact/fingers_touching: [0.0112, 0.0146, 0.0083, 1.65, 2.7, 3.15, 3.25, 3.15, 3.25, 3.3]  min 0 · mean 2.15 · max 3.46
contact/fingers_touching_at_success: [-1, -1, 3.74, 3.74, 3.62, 4.12, 4.18, 4.8, 4.82, 4.84]  min -1 · mean 3.5 · max 5
contact/link_force_mean: [0.00415, 0.0113, 0.00236, 0.844, 1.57, 2.25, 2.26, 2.22, 2.23, 2.47]  min 0 · mean 1.49 · max 13.7
contact/links_touching: [0.0122, 0.0149, 0.00854, 1.9, 3.1, 3.83, 3.87, 3.76, 3.86, 3.94]  min 0 · mean 2.55 · max 4.21
contact/links_touching_at_success: [-1, -1, 3.74, 3.74, 3.62, 4.58, 4.92, 4.98, 5.04, 5.11]  min -1 · mean 3.79 · max 6.4
contact/palm_touching: [0, 0.000244, 0.203, 0.237, 0.319, 0.463, 0.493, 0.48, 0.488, 0.528]  min 0 · mean 0.314 · max 0.554
contact/palm_touching_at_success: [-1, -1, 0, 0, 0, 0.157, 0.342, 0.488, 0.383, 0.563]  min -1 · mean 0.117 · max 0.966
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0.000244, 0.00122, 0.00146, 0.00952, 0.000244, 0]  min 0 · mean 0.00191 · max 0.0359
done/abnormal: [0, 0, 0, 0.000488, 0.000732, 0.000977, 0.000488, 0.000977, 0.000732, 0]  min 0 · mean 0.000509 · max 0.00317
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.66e-07 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0.000244, 0]  min 0 · mean 2.15e-05 · max 0.000732
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.31e-07 · max 0.000244
done/out_xy: [0, 0.000244, 0, 0.000244, 0.000244, 0, 0.000244, 0, 0, 0.000732]  min 0 · mean 0.00022 · max 0.0022
done/tipped: [0, 0.000244, 0.000488, 0.00146, 0.000732, 0.000732, 0.000488, 0.000488, 0.000977, 0.000244]  min 0 · mean 0.000362 · max 0.00317
episode_lengths/step: [25.7, 854, 814, 575, 483, 412, 448, 419, 519, 486]  min 25.7 · mean 570 · max 899
reward/action_rate_pen: [-0.00266, -0.00223, -0.00124, -0.000845, -0.000694, -0.000649, -0.000499, -0.000483, -0.00045, -0.000452]  min -0.00269 · mean -0.00092 · max -0.000423
reward/approach_form: [0.0209, 0.0652, 0.0194, 0.0262, 0.0194, 0.0169, 0.0145, 0.0177, 0.0163, 0.0157]  min 0.0118 · mean 0.0226 · max 0.0879
reward/approach_keep: [0.00106, 0.00128, 0.003, 0.0074, 0.00367, 0.00222, 0.00192, 0.00267, 0.00283, 0.00261]  min 0.000103 · mean 0.00321 · max 0.0141
reward/approach_lock: [1.76e-14, 2.31e-08, 0.00373, 0.00498, 0.00284, 0.00143, 0.00166, 0.00212, 0.00186, 0.00182]  min 6.04e-22 · mean 0.00242 · max 0.0221
reward/approach_travel: [0.0147, 0.0208, 0.0131, 0.0279, 0.0161, 0.0111, 0.00966, 0.0123, 0.0124, 0.0117]  min 0.00781 · mean 0.0154 · max 0.053
reward/arm_vel_pen: [-0.000146, -0.000242, -0.0002, -0.000869, -0.00307, -0.00454, -0.00614, -0.00661, -0.00667, -0.00705]  min -0.00955 · mean -0.0039 · max -0.000136
reward/cup_disturb_pen: [-8.07e-06, -0.000466, -0.000133, -0.00289, -0.00717, -0.00408, -0.00532, -0.00344, -0.00423, -0.00315]  min -0.0117 · mean -0.00373 · max -1.84e-06
reward/cup_tilt_pen: [-0.00115, -0.00293, -0.00318, -0.018, -0.0285, -0.0299, -0.0261, -0.0321, -0.0368, -0.0211]  min -0.0444 · mean -0.0209 · max -0.000105
reward/early_contact_pen: [-0.00259, -0.0023, -0.000843, -0.00787, -0.0141, -0.00812, -0.0101, -0.0089, -0.0102, -0.00682]  min -0.0242 · mean -0.00847 · max 0
reward/envelope_quality: [0, 0, 0, 0.0377, 0.0597, 0.0985, 0.0977, 0.103, 0.104, 0.115]  min 0 · mean 0.0627 · max 0.125
reward/finger_curl: [0, 0, 0.00379, 0.0236, 0.04, 0.0468, 0.0466, 0.0499, 0.0509, 0.052]  min 0 · mean 0.0326 · max 0.0554
reward/goal_close: [0, 0, 0, 0.00749, 0.00624, 0.00844, 0.00974, 0.00855, 0.00758, 0.0116]  min 0 · mean 0.00591 · max 0.0125
reward/goal_in_tol: [0, 0, 0, 0, 0, 0, 0, 2.63e-09, 0, 8.83e-13]  min 0 · mean 4.37e-06 · max 0.000506
reward/goal_near: [0, 0, 0, 4.21e-05, 2.6e-05, 5.39e-05, 3.69e-05, 3.35e-05, 2.97e-05, 4.1e-05]  min 0 · mean 3.17e-05 · max 0.000595
reward/grasp_fingers: [0, 0, 3.05e-05, 0.0246, 0.0468, 0.0644, 0.0637, 0.0653, 0.0675, 0.0701]  min 0 · mean 0.0418 · max 0.0762
reward/grasp_pose: [0, 0, 0.139, 0.122, 0.154, 0.155, 0.155, 0.169, 0.172, 0.171]  min 0 · mean 0.134 · max 0.269
reward/grasp_thumb: [0, 0, 5.07e-05, 0.0249, 0.0459, 0.0619, 0.0623, 0.0636, 0.0656, 0.0675]  min 0 · mean 0.0407 · max 0.0729
reward/grasp_wrap: [0, 0, 2.48e-06, 0.0311, 0.0577, 0.0926, 0.0901, 0.0923, 0.0966, 0.1]  min 0 · mean 0.0579 · max 0.111
reward/grip_squeeze: [0, 0, 2.38e-06, 0.00729, 0.0129, 0.0175, 0.0167, 0.0179, 0.0186, 0.0192]  min 0 · mean 0.0113 · max 0.0205
reward/hold_still: [0, 0, 0, 1.39e-05, 7.42e-06, 8.49e-06, 9.39e-06, 7.95e-06, 6.82e-06, 1.12e-05]  min 0 · mean 6.7e-06 · max 3.9e-05
reward/lift_clear: [0, 0, 0, 0.00217, 0.00299, 0.00521, 0.00623, 0.00649, 0.00577, 0.0083]  min 0 · mean 0.00387 · max 0.01
reward/lift_hold: [0, 0, 0, 0.0183, 0.0261, 0.035, 0.0371, 0.0366, 0.0359, 0.0389]  min 0 · mean 0.0232 · max 0.0412
reward/light_contact: [0, 0, 2.08e-05, 0.00939, 0.0184, 0.0238, 0.0236, 0.0239, 0.0248, 0.0253]  min 0 · mean 0.0156 · max 0.0277
reward/palm_contact: [0, 0, 0.0135, 0.035, 0.0553, 0.0828, 0.0848, 0.0923, 0.0916, 0.101]  min 0 · mean 0.0555 · max 0.107
reward/palm_reach: [0, 0, 0.0801, 0.0665, 0.0883, 0.0899, 0.0914, 0.0992, 0.101, 0.1]  min 0 · mean 0.0769 · max 0.147
reward/palm_table_pen: [0, 0, 0, -8.23e-06, -0.000293, -0.00221, -0.00243, -0.00324, -0.00139, -0.00295]  min -0.00958 · mean -0.00121 · max 0
reward/pinky_join: [0, 0, 0, 0.00503, 0.00738, 0.0135, 0.0119, 0.0121, 0.0132, 0.0146]  min 0 · mean 0.00793 · max 0.0194
reward/pregrasp_shape: [0, 0, 0.000456, 0.00172, 0.00062, 0.000589, 0.00025, 0.000274, 0.000615, 0.000528]  min 0 · mean 0.000742 · max 0.00784
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.23e-05 · max 0.00732
reward/table_pen: [0, -0.000442, -0.0015, -0.015, -0.0316, -0.0298, -0.0458, -0.0385, -0.0323, -0.0424]  min -0.0499 · mean -0.0269 · max 0
reward/thumb_curl: [0, 0, 0.00493, 0.00968, 0.0183, 0.0181, 0.0188, 0.0209, 0.0207, 0.021]  min 0 · mean 0.0138 · max 0.0236
reward/total: [0.03, 0.0787, 0.274, 0.447, 0.596, 0.765, 0.748, 0.801, 0.818, 0.865]  min 0.0153 · mean 0.562 · max 0.925
rewards/step: [-1.31, 57.5, 191, 258, 244, 303, 286, 331, 432, 397]  min -6.84 · mean 279 · max 527
task/lifted_frac: [0, 0, 0, 0.00391, 0.00732, 0.0457, 0.0249, 0.0193, 0.0227, 0.0139]  min 0 · mean 0.014 · max 0.0562
task/successes_mean: [0, 0, 0, 0, 0, 0.000244, 0.000244, 0.0022, 0, 0]  min 0 · mean 0.000374 · max 0.00806
task/tilt_deg: [0.458, 0.548, 3.19, 5.45, 9.25, 9.92, 9.28, 10.1, 11.1, 8.37]  min 0.0796 · mean 7.01 · max 11.8
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
