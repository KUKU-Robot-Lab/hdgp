You are an expert in robotics, reinforcement learning and code generation.

We control one 7-DOF OpenArm robot arm (the right arm) carrying a five-finger Tesollo DG-5F hand, standing at a table. +x points from the robot toward the table. Each episode starts with the arm raised beside the robot: the palm is just outside the table edge nearest the robot, about 0.25 m above the table top, turned sideways, with the fingers pointing forward over the table edge; the palm is roughly 0.38 m from the cup. A cup stands upright on the table; parallel environments use different cups (open cups of several sizes and a closed shaker), so the graspable radius (44 mm to 81 mm) and half height (42 mm to 65 mm) differ between environments and are given per environment. From the second episode on, each episode starts with probability one half from a second pose instead: the hand already beside the cup on its -y side, in the same orientation and default hand pose as the raised start, with the palm about 4.5 cm from the cup's side, the cup axis about cup_radius + 2.5 cm ahead of the palm centre along the fingers, and the palm centre in the upper part of the graspable band (the cup position still varies by up to 2 cm). Such an episode starts with approach_done already set, as if the approach had just been completed. Positions are in metres in each environment's local frame, with +z pointing up; the table top is at z = table_z.

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

I want it to fulfil the following task: Pick up the upright cup in three stages, strictly in this order.
Stage 1, approach: starting from the raised pose beside the table, bring the hand to the cup without turning it: keep the hand's start orientation the whole way (ctx.palm_normal along +y, towards the cup, and ctx.palm_finger_dir along +x) and keep its default pose (ctx.hand_default_q_norm), so the fingers and the thumb stay open. In the default pose the thumb sticks straight out of the palm towards the cup at the wrist end of the palm, so the hand has to come to the cup's -y side with the cup between the thumb and the four fingers: the palm close to the cup's side, the cup further along the fingers than the thumb, and neither the thumb nor any finger touching the cup yet. The approach is complete when the environment sets ctx.approach_done, which then stays set for the rest of the episode. In some episodes the hand instead starts beside the cup with ctx.approach_done already set; those episodes begin at Stage 2.
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


def _soft_touch(force: torch.Tensor, full_force: float) -> torch.Tensor:
    # 0 when not touching, 1 at/above full_force; clamping also bounds physics force spikes
    return torch.clamp(force / full_force, 0.0, 1.0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale = 0.1  # global scale so a typical per-step reward is O(1)

    r = ctx.cup_radius
    hh = ctx.cup_half_height
    n = ctx.palm_normal
    fd = ctx.palm_finger_dir
    axis = ctx.cup_axis

    # ---------------- stage masks (strict order) ----------------
    approach = ctx.approach_done.bool()
    envelope = ctx.envelope_done.bool() & approach
    s1 = (~approach).float()
    s2 = (approach & ~envelope).float()
    s3 = envelope.float()

    # ---------------- palm relative to the cup (same quantities approach_done checks) ----------------
    v = ctx.cup_pos - ctx.palm_pos
    v_ax = (v * axis).sum(-1)
    v_perp = v - v_ax.unsqueeze(-1) * axis
    gap_n = (v_perp * n).sum(-1) - r    # env window [-0.01, 0.02]
    gap_f = (v_perp * fd).sum(-1) - r   # env window [-0.005, 0.02]
    h_palm = -v_ax                      # palm height along the cup axis; env needs |h| <= hh

    # orientation kept at the start orientation (normal +y, fingers +x); ~0.05 at 45 deg on both axes
    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))
    ang_f = torch.acos(torch.clamp(fd[:, 0], -1.0, 1.0))
    ori = torch.exp(-(ang_n ** 2 + ang_f ** 2) / (2.0 * 0.45 ** 2))

    # default hand pose kept (env needs every movable joint within 0.3); mean term keeps a gradient far away
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev.amax(-1) / 0.2) ** 2)

    # ---------------- stage 1: approach ----------------
    # coarse world-frame target on the cup's -y side, slightly behind the axis along +x, upper band
    offset = torch.stack([-(r + 0.0075), -(r + 0.008), 0.25 * hh], dim=-1)
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(3.0 * d_world)
    # fine palm-frame precision centred in the approach_done window
    fine_err2 = (gap_n - 0.008) ** 2 + (gap_f - 0.0075) ** 2 + torch.relu(h_palm.abs() - 0.5 * hh) ** 2
    fine = torch.exp(-fine_err2 / (2.0 * 0.012 ** 2))
    keep_factor = 0.3 + 0.7 * ori * pose  # turning the hand or closing fingers cuts approach progress
    approach_reach = s1 * (1.0 * coarse + 1.5 * fine) * keep_factor  # max 2.5
    approach_keep = s1 * (0.25 * ori + 0.25 * pose)                  # max 0.5 -> stage 1 max 3.0

    # ---------------- contacts ----------------
    link_touch = _soft_touch(ctx.link_cup_force, 0.5)      # (N,5,3)
    digit_touch = link_touch.amax(-1)                       # (N,5)
    palm_touch = _soft_touch(ctx.palm_cup_force, 0.5)       # (N,)
    thumb_touch = digit_touch[:, 0]
    finger_count = digit_touch[:, 1:].sum(-1)               # 0..4
    early_contact_pen = -2.0 * s1 * digit_touch.amax(-1)    # no thumb/finger contact before the approach is done

    # ---------------- link geometry around the cup ----------------
    lv = ctx.link_pos - ctx.cup_pos[:, None, None, :]       # (N,5,3,3)
    ax = axis[:, None, None, :]
    lh = (lv * ax).sum(-1)                                  # (N,5,3)
    lrad_vec = lv - lh.unsqueeze(-1) * ax
    lrad = torch.norm(lrad_vec, dim=-1)
    surf_gap = torch.relu(lrad - r[:, None, None] - 0.01)   # 1 cm allowance for link thickness
    in_band = torch.clamp(1.0 - torch.relu(lh.abs() - hh[:, None, None]) / 0.02, 0.0, 1.0)
    near_surf = torch.exp(-surf_gap / 0.015) * in_band      # (N,5,3)

    # fingertip wrap angle around the axis: 0 = palm side, pi/2 = far side along the fingers
    tip_vec = lrad_vec[:, 1:, 2, :]                         # (N,4,3)
    along_f = (tip_vec * fd[:, None, :]).sum(-1)
    toward_palm = -(tip_vec * n[:, None, :]).sum(-1)
    theta = torch.atan2(along_f, toward_palm)
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)  # saturates at ~108 deg (wrapped past the far side)
    finger_wrap = (wrap * near_surf[:, 1:, 2]).mean(-1)
    thumb_near = near_surf[:, 0, :].amax(-1)

    # palm closing the last centimetre while staying aligned along the fingers and in the band
    align2 = torch.exp(-((gap_f - 0.0075) ** 2 + torch.relu(h_palm.abs() - hh) ** 2) / (2.0 * 0.02 ** 2))
    palm_close = torch.exp(-torch.relu(gap_n) / 0.015) * align2

    # squeeze: command leads the measured angle on touching digits, so the PD fingers press
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, j].mean(-1) for j in _DIGIT_JOINTS], dim=-1)  # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)

    # ---------------- stage 2: envelope grasp ----------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    resting = torch.clamp(1.0 - (dz - 0.02) / 0.02, 0.0, 1.0)  # no stage-2 progress once the cup is lifted >4 cm
    fac2 = s2 * (0.3 + 0.7 * ori) * resting
    grasp_stage_base = 3.5 * s2                                  # > stage 1 max (3.0)
    grasp_palm = fac2 * (0.5 * palm_close + 1.0 * palm_touch)
    grasp_contacts = fac2 * (1.0 * thumb_touch + 1.5 * finger_count / 4.0)
    grasp_wrap = fac2 * (1.0 * finger_wrap + 0.5 * thumb_near)
    grip_squeeze = 0.5 * squeeze * (fac2 + s3)                   # stage 2 max total 3.5 + 6.0 = 9.5

    # ---------------- stage 3: lift and hold ----------------
    firm = _soft_touch(ctx.link_cup_force, 0.25).amax(-1)       # (N,5)
    grip_gate = firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)  # thumb + >=2 fingers pressing
    lift_stage_base = 10.0 * s3                                  # > stage 2 max (9.5)
    hold_contacts = s3 * (1.0 * palm_touch + 1.0 * thumb_touch + 1.5 * finger_count / 4.0)  # releasing loses this
    rise_needed = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)
    lift_frac = torch.clamp(dz / rise_needed, 0.0, 1.0)
    lift_height = 3.0 * s3 * grip_gate * lift_frac
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    lift_goal = s3 * grip_gate * (2.0 * (1.0 - torch.tanh(gd / 0.2)) + 2.0 * torch.exp(-gd / 0.03) + 1.0 * in_tol)
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 1.0 * s3 * grip_gate * torch.exp(-gd / 0.03) * calm
    success_bonus = 20.0 * s3 * ctx.success.float()

    # ---------------- constraints and regularization (all stages) ----------------
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.01) / 0.04, 0.0, 1.0)  # do not shove the cup
    cup_tilt_pen = -2.0 * torch.clamp(ctx.cup_tilt / (math.pi / 3.0), 0.0, 1.0) ** 2   # do not knock it over
    table_pen = -2.0 * torch.clamp((ctx.table_z + 0.002 - ctx.hand_z_min) / 0.02, 0.0, 1.0)  # do not press into the table
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.05 * (ctx.arm_qd ** 2).sum(-1)

    raw = {
        "approach_reach": approach_reach,
        "approach_keep": approach_keep,
        "early_contact_pen": early_contact_pen,
        "grasp_stage_base": grasp_stage_base,
        "grasp_palm": grasp_palm,
        "grasp_contacts": grasp_contacts,
        "grasp_wrap": grasp_wrap,
        "grip_squeeze": grip_squeeze,
        "lift_stage_base": lift_stage_base,
        "hold_contacts": hold_contacts,
        "lift_height": lift_height,
        "lift_goal": lift_goal,
        "hold_still": hold_still,
        "success_bonus": success_bonus,
        "cup_disturb_pen": cup_disturb_pen,
        "cup_tilt_pen": cup_tilt_pen,
        "table_pen": table_pen,
        "action_rate_pen": action_rate_pen,
        "arm_vel_pen": arm_vel_pen,
    }
    components = {name: scale * value for name, value in raw.items()}
    reward = torch.stack(list(components.values()), dim=0).sum(0)
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
The round was ended early, at about epoch 500 of 3000, because the hand had stopped interacting with the cup; the playback uses the latest checkpoint and shows three consecutive episodes of one environment.
Episodes 1 and 2 start from the raised pose. The hand moves down beside the robot column near the table edge closest to the robot, turns away from its start orientation with the fingers spread, drifts around there and rises again. It never moves towards the cup and never touches it.
Episode 3 starts beside the cup, with approach_done already set: the fingers point forward along +x, the palm faces the cup's side and the cup is just in front of the hand. Within 0.25 s the fingers are still open and the hand is already pulling back; by 0.7 s it is clearly away from the cup with the fingers spread; later it rises high and moves to the table edge. It never closes the fingers and never touches the cup.
Metrics (up to epoch 498):
- approach_done was set in every near-start episode (as the environment does) and in no far-start episode; envelope_done was never set in either group.
- In near-start episodes, the share in which at least three digits touched the cup at some step rose to 0.60 at epoch 120, was 0.51 at epoch 338 and fell to 0.00 at epoch 498.
- The mean palm-to-band distance grew from 0.19 m at epoch 60 to 0.41 m at epoch 498.
- The total reward was 0.182 at epoch 498, of which grasp_stage_base was 0.178. grasp_palm was 0 at every logged epoch; grasp_contacts fell from 0.016 (epoch 226) to 0.000, grasp_wrap from 0.012 to 0.000 and grip_squeeze stayed at about 0.
- cup_disturb_pen went from -0.018 at epoch 60 to -0.001 at epoch 498; early_contact_pen stayed at about 0.
- approach_reach stayed at about 0.006; the share of steps with the hand in its start orientation fell from 1.00 to 0.29.

To make the code more accurate and train better robot, the feedback for improvement is:
The policy learned to stay away from the cup: the reward pays almost all of its value for the stage flags alone.
- After approach_done is set, the stage-2 base amount is paid on every step whatever the hand does, and it was nearly the whole reward. Staying beside the cup, closing the hand and moving away all earn that same amount, so the policy pulled the hand back at the start of near-start episodes and kept the reward. Stage progress must only pay while the hand keeps doing that stage's work: after the approach, the reward has to fall clearly when the palm leaves the grasp position beside the cup, and rise as the palm reaches the cup's side and the fingers and thumb close on it.
- Early in training the hand did touch the cup with several digits in near-start episodes, but those touches earned very little while nudging the cup was penalised, and the touching disappeared. Closing the hand around the cup, palm contact and thumb and finger contacts must be worth clearly more than the small cup movements that happen while closing; only knocking the cup over or pushing it away should cost more than grasping gains.
- The palm never touched the cup. In near-start episodes the cup already sits between the thumb and the four fingers in front of the palm; the palm should first move the last few centimetres to the cup's side, then the fingers and thumb close.
- From the raised start the hand never approached, and the approach reward was too small to matter next to what near-start episodes paid. Moving the palm towards the cup's -y side in the start orientation has to pay clearly more than drifting.
- The hand turned away from its start orientation in both kinds of episodes; keeping the orientation should stay part of the approach and of the grasp.
- Stage 3 was never reached, so there is no feedback on lifting yet.

Generated code shown as below:
```python
import torch
import math


# Movable finger joints (range > 0.05 rad), grouped per digit in hand-joint order:
# thumb (3,4), index (2,3,4), middle (2,3,4), ring (2,3,4), pinky (3,4).
_DIGIT_JOINTS = ([1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18])
_MOVABLE = [j for group in _DIGIT_JOINTS for j in group]


def _soft_touch(force: torch.Tensor, full_force: float) -> torch.Tensor:
    # 0 when not touching, 1 at/above full_force; clamping also bounds physics force spikes
    return torch.clamp(force / full_force, 0.0, 1.0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale = 0.1  # global scale so a typical per-step reward is O(0.1-3)

    r = ctx.cup_radius
    hh = ctx.cup_half_height
    n = ctx.palm_normal
    fd = ctx.palm_finger_dir
    axis = ctx.cup_axis

    # ---------------- stage masks (strict order) ----------------
    approach = ctx.approach_done.bool()
    envelope = ctx.envelope_done.bool() & approach
    s1 = (~approach).float()
    s2 = (approach & ~envelope).float()
    s3 = envelope.float()

    # ---------------- palm relative to the cup (same quantities approach_done checks) ----------------
    v = ctx.cup_pos - ctx.palm_pos
    v_ax = (v * axis).sum(-1)
    v_perp = v - v_ax.unsqueeze(-1) * axis
    gap_n = (v_perp * n).sum(-1) - r    # palm plane to cup surface; env window [-0.01, 0.02]
    gap_f = (v_perp * fd).sum(-1) - r   # cup axis ahead along fingers minus radius; env window [-0.005, 0.02]
    h_palm = -v_ax                      # palm height along the cup axis; env needs |h| <= hh

    # start orientation (normal +y, fingers +x): 1.0 aligned, ~0.47 at 20 deg on both axes, ~0.02 at 45 deg
    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))
    ang_f = torch.acos(torch.clamp(fd[:, 0], -1.0, 1.0))
    ori = torch.exp(-(ang_n ** 2 + ang_f ** 2) / (2.0 * 0.4 ** 2))

    # default (open) hand pose kept; env needs every movable joint within 0.3
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev.amax(-1) / 0.2) ** 2)

    # ---------------- cup state relative to its spawn ----------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    cup_home = torch.clamp(1.0 - (disp_xy - 0.04) / 0.06, 0.0, 1.0)  # 1 up to 4 cm of shove, 0 beyond 10 cm
    resting = torch.clamp(1.0 - (dz - 0.02) / 0.02, 0.0, 1.0)       # 0 once lifted >4 cm (no lifting before envelope)

    # ---------------- stage 1: approach ----------------
    # coarse world-frame target on the cup's -y side, behind the axis along +x, upper part of the band
    offset = torch.stack([-(r + 0.0075), -(r + 0.008), 0.25 * hh], dim=-1)
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(2.5 * d_world)  # ~0.26 at the raised start (0.38 m), 0.75 at 0.1 m
    # fine palm-frame precision centred in the approach_done window
    fine_err2 = (gap_n - 0.005) ** 2 + (gap_f - 0.0075) ** 2 + torch.relu(h_palm.abs() - 0.5 * hh) ** 2
    fine = torch.exp(-fine_err2 / (2.0 * 0.015 ** 2))
    keep = ori * pose
    # turning the hand / closing fingers cuts approach progress to 20 %; shoving the cup away removes it
    approach_reach = s1 * cup_home * (2.0 * coarse + 2.0 * fine) * (0.2 + 0.8 * keep)  # max 4.0
    approach_keep = s1 * 0.5 * keep                                                      # max 0.5 -> stage 1 max 4.5

    # ---------------- contacts ----------------
    link_touch = _soft_touch(ctx.link_cup_force, 0.5)      # (N,5,3)
    digit_touch = link_touch.amax(-1)                       # (N,5)
    palm_touch = _soft_touch(ctx.palm_cup_force, 0.5)       # (N,)
    thumb_touch = digit_touch[:, 0]
    finger_count = digit_touch[:, 1:].sum(-1)               # 0..4
    early_contact_pen = -2.0 * s1 * digit_touch.amax(-1)    # no thumb/finger contact before the approach is done

    # ---------------- link geometry around the cup ----------------
    lv = ctx.link_pos - ctx.cup_pos[:, None, None, :]       # (N,5,3,3)
    ax = axis[:, None, None, :]
    lh = (lv * ax).sum(-1)                                  # (N,5,3)
    lrad_vec = lv - lh.unsqueeze(-1) * ax
    lrad = torch.norm(lrad_vec, dim=-1)
    surf_gap = torch.relu(lrad - r[:, None, None] - 0.01)   # 1 cm allowance for link thickness
    in_band = torch.clamp(1.0 - torch.relu(lh.abs() - hh[:, None, None]) / 0.02, 0.0, 1.0)
    near_surf = torch.exp(-surf_gap / 0.015) * in_band      # (N,5,3)

    # fingertip wrap angle around the axis: 0 = palm side, pi/2 = side along the fingers
    tip_vec = lrad_vec[:, 1:, 2, :]                         # (N,4,3)
    along_f = (tip_vec * fd[:, None, :]).sum(-1)
    toward_palm = -(tip_vec * n[:, None, :]).sum(-1)
    theta = torch.atan2(along_f, toward_palm)
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)  # saturates at ~108 deg (curled past the finger-side)
    finger_wrap = (wrap * near_surf[:, 1:, 2]).mean(-1)
    thumb_near = near_surf[:, 0, :].amax(-1)

    # ---------------- grasp position: palm against the cup's side (cup-relative, so it also holds while lifting) ----------------
    gn_err = torch.relu(gap_n - 0.003) + torch.relu(-0.012 - gap_n)      # palm plane on the cup surface
    gf_err = torch.relu((gap_f - 0.0075).abs() - 0.008)                  # cup between thumb and fingers
    h_err = torch.relu(h_palm.abs() - 0.7 * hh)                          # palm inside the band
    d_grasp = torch.sqrt(gn_err ** 2 + gf_err ** 2 + h_err ** 2 + 1e-12)
    stay = torch.exp(-(d_grasp / 0.06) ** 2)   # ~0.6 in the near-start pose (4.5 cm), 1 at the cup, ~0 beyond 12 cm
    close = torch.exp(-d_grasp / 0.015)        # sharp: pulls the palm the last few centimetres onto the cup
    palm_ready = torch.maximum(close, palm_touch)
    close_gate = 0.25 + 0.75 * palm_ready      # finger/thumb closing pays most once the palm is at the cup

    # squeeze: command leads the measured angle on touching digits, so the PD fingers press
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, j].mean(-1) for j in _DIGIT_JOINTS], dim=-1)  # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)

    # ---------------- stages 2+3: grasp terms (no flag-only reward; everything needs the hand at the cup) ----------------
    ori_k = 0.3 + 0.7 * ori
    g2 = s2 * resting * cup_home * ori_k       # stage 2: cup still on the table, near its spawn, hand not turned
    g3 = s3 * ori_k                            # stage 3: same grasp terms keep paying while lifting
    g23 = g2 + g3
    fingers_enough = torch.clamp(finger_count / 3.0, 0.0, 1.0)  # envelope_done needs thumb + >=3 fingers
    grasp_stay = 4.0 * g23 * stay                                # leaving the cup's side loses this
    grasp_palm_close = 2.0 * g23 * close
    grasp_palm_touch = 2.5 * g23 * palm_touch
    grasp_thumb = 2.0 * g23 * stay * close_gate * thumb_touch
    grasp_fingers = 3.0 * g23 * stay * close_gate * (0.75 * fingers_enough + 0.25 * finger_count / 4.0)
    grasp_wrap = g23 * stay * close_gate * (1.0 * finger_wrap + 0.5 * thumb_near)
    grip_squeeze = 1.0 * g23 * squeeze
    envelope_quality = 3.0 * g23 * palm_touch * thumb_touch * fingers_enough  # the envelope_done condition itself
    # grasp terms max ~18 (x0.1); at approach completion stay~1 so stage 2 (>=4) exceeds stage 1 (<=4.5 only at its optimum)

    # ---------------- stage 3: lift and hold (all gated on a firm grip) ----------------
    firm = _soft_touch(ctx.link_cup_force, 0.25).amax(-1)       # (N,5)
    grip_gate = firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)  # thumb + >=2 fingers pressing
    lift_grip_hold = 3.0 * s3 * grip_gate                        # keeping the grasp after envelope_done
    rise_needed = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)
    lift_frac = torch.clamp(dz / rise_needed, 0.0, 1.0)
    lift_height = 5.0 * s3 * grip_gate * lift_frac
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    lift_goal = s3 * grip_gate * (3.0 * (1.0 - torch.tanh(gd / 0.15)) + 3.0 * torch.exp(-gd / 0.03) + 1.0 * in_tol)
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 2.0 * s3 * grip_gate * torch.exp(-gd / 0.03) * calm
    success_bonus = 30.0 * s3 * ctx.success.float()

    # ---------------- constraints and regularization ----------------
    # small nudges while closing (<4 cm) are free; shoving the cup away costs up to 3 (plus the lost grasp terms)
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.04) / 0.08, 0.0, 1.0)
    # tilt below ~10 deg is free; knocking it over costs up to 4 (and >60 deg ends the episode)
    cup_tilt_pen = -4.0 * torch.clamp((ctx.cup_tilt - 0.17) / (math.pi / 3.0 - 0.17), 0.0, 1.0)
    table_pen = -2.0 * torch.clamp((ctx.table_z + 0.002 - ctx.hand_z_min) / 0.02, 0.0, 1.0)  # do not press into the table
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.05 * (ctx.arm_qd ** 2).sum(-1)

    raw = {
        "approach_reach": approach_reach,
        "approach_keep": approach_keep,
        "early_contact_pen": early_contact_pen,
        "grasp_stay": grasp_stay,
        "grasp_palm_close": grasp_palm_close,
        "grasp_palm_touch": grasp_palm_touch,
        "grasp_thumb": grasp_thumb,
        "grasp_fingers": grasp_fingers,
        "grasp_wrap": grasp_wrap,
        "grip_squeeze": grip_squeeze,
        "envelope_quality": envelope_quality,
        "lift_grip_hold": lift_grip_hold,
        "lift_height": lift_height,
        "lift_goal": lift_goal,
        "hold_still": hold_still,
        "success_bonus": success_bonus,
        "cup_disturb_pen": cup_disturb_pen,
        "cup_tilt_pen": cup_tilt_pen,
        "table_pen": table_pen,
        "action_rate_pen": action_rate_pen,
        "arm_vel_pen": arm_vel_pen,
    }
    components = {name: scale * value for name, value in raw.items()}
    reward = torch.stack(list(components.values()), dim=0).sum(0)
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
The round was judged over at the four-hour limit, at epoch 2788 of 3000. The playback uses a snapshot of the checkpoint at that point and shows two consecutive episodes of one environment (30 s). The run itself was left training while the recording was made and has since reached epoch 2989; the figures below are given for both points where they differ.

What the robot does in the recording:
- In both episodes the hand is at the cup's side within the first three seconds and stays there for the rest of the episode. It does not drift away and it does not turn away from its start orientation: the palm keeps facing the cup's side and the fingers keep pointing forward across it.
- The fingers stay extended for the whole recording. They reach across the near side of the cup and touch it with their tips and middle segments, but they never curl around it. At no point in either episode does a fingertip pass the far side of the cup, and the thumb hangs beside and below the cup instead of opposing the fingers around it.
- A gap between the palm and the cup's side is visible in every frame. The palm never rests against the cup.
- The cup stands upright at its spawn position in every frame of both episodes. It is never tipped, never pushed noticeably and never lifted off the table.
- Which of the two episodes starts beside the cup and which starts from the raised pose cannot be told apart in the recording; after the first three seconds both look the same.

Metrics (mean over the last 150 epochs, at epoch 2788):
- approach_done was set in 0.97 of far-start episodes and in 1.00 of near-start episodes. envelope_done was set in 0.0097 of far-start and 0.0033 of near-start episodes, and the episode funnel value for envelope is 0.0000.
- Episode funnel: reach 1.00, grasp 1.00 (at least three digits touched at some step), envelope 0.0000, lift 0.0017, success 0.0000.
- On an average step 3.2 to 3.3 of the five digits touch the cup (thumb 0.85, middle 0.85, ring 0.82, index 0.81, little finger 0.0003), while the palm touches on between 0.001 and 0.013 of steps depending on the window (0.0013 at epoch 2788, 0.013 at epoch 2989).
- The mean palm-to-cup gap is 0.046 m.
- The total reward is 0.958. Its largest parts are grasp_stay 0.330, grasp_fingers 0.210, grasp_palm_close 0.155, grasp_thumb 0.155, grasp_wrap 0.077 and grip_squeeze 0.026. envelope_quality is 0.0001 and grasp_palm_touch is 0.0002; every lift term and the success bonus are 0.
- cup_disturb_pen is -0.002 and cup_tilt_pen is -0.005; no episode ended with the cup tipped.
- There were no successes in the last 600 epochs. The contact values recorded at the moment of success stopped changing after about epoch 2300 and still carry the values from the successes before that: palm touching 0.80 and 2.11 fingers touching. Those were the only grasps in the whole round in which the palm touched the cup, and they stopped.
- The total reward rose through the entire round, from 0.73 at epoch 728 to 0.87 at 1757 and 0.90 to 0.96 at the end, while over the same period palm contact stayed inside a band between about 0.001 and 0.013 of steps with no sustained direction, and the envelope was never completed in a measurable share of episodes.

To make the code more accurate and train better robot, the feedback for improvement is:
Three things are working and should be kept.
- The approach is solved. From both starting states the hand reaches the cup's side in about three seconds, keeps its start orientation and holds that position for the rest of the episode. It no longer drifts away or turns away, which were the failures of the previous rounds.
- The hand closes on the cup. Three or four digits touch it on an average step, and that has been rising steadily.
- The cup is handled gently. It stays upright at its spawn position and is never knocked over, so the penalties for disturbing it are doing their job without suppressing contact.

The rest is what has to change.

The grasp that was learned is a fingertip grasp, not an envelope. The fingers reach across the near side of the cup and touch it with their tips while staying extended, the thumb hangs beside the cup instead of opposing the fingers, and the palm stays a few centimetres away. The reward currently pays its largest amounts for exactly this shape: holding position beside the cup and touching it with the fingers. Touching the cup with extended fingers while the palm stays away must be worth clearly less than a grasp in which the palm is against the cup's side and the fingers are curled around it.

Palm contact is the one missing condition, and it is the whole blockage. Everything else the envelope needs is already there on most steps: the thumb touches, and three or more fingers touch. The palm touched on somewhere between one step in a thousand and one step in a hundred, drifting up and down inside that band all round without ever climbing out of it, and as a result the envelope was completed in far less than one episode in a hundred. The distance still to cover is a few centimetres of palm travel. Those last centimetres, and the palm contact at the end of them, have to become the most valuable thing available in this stage, worth more than any amount of time spent holding position or touching with the fingertips.

The clearest evidence that the present shape is a dead end: the total reward rose for the entire round while palm contact never left a band far below what the envelope needs. Terms that pay for staying beside the cup and for finger contact can be driven up indefinitely without the grasp ever being completed. No term should be able to keep growing while the stage it belongs to goes backwards.

The policy did find the right grasp once. In the middle of the round there was a short period with rare successes in which the palm was touching and two fingers were on the cup - the only palm-contact grasps of the round. They disappeared and never returned, which means what the reward offered for them was too small, or too easily matched by the fingertip shape, for the policy to hold on to. A completed envelope, and the lift that follows it, must be worth far more than the fingertip grasp can ever accumulate, so that finding it once is enough to keep it.

Two smaller points. The little finger never touches the cup in any episode; if a five-finger wrap is intended it needs its own reason to close, and if it is not intended nothing has to change. And nothing has been learned about lifting, because the envelope was never completed - lifting feedback will only become meaningful once the envelope happens.

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


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale = 0.1  # global scale: fingertip touching ~0.4, a real envelope ~3.8, a held success ~6.5

    r = ctx.cup_radius
    hh = ctx.cup_half_height
    n = ctx.palm_normal
    fd = ctx.palm_finger_dir
    axis = ctx.cup_axis

    # ---------------- stage masks (strict order, flags are latched by the env) ----------------
    approach = ctx.approach_done.bool()
    envelope = ctx.envelope_done.bool() & approach
    s1 = (~approach).float()
    s2 = (approach & ~envelope).float()
    s3 = envelope.float()

    # ---------------- palm pose relative to the cup (same quantities approach_done checks) ----
    v = ctx.cup_pos - ctx.palm_pos
    v_ax = (v * axis).sum(-1)
    v_perp = v - v_ax.unsqueeze(-1) * axis
    gap_n = (v_perp * n).sum(-1) - r    # palm plane -> cup surface; NEGATIVE = palm pressed into the cup
    gap_f = (v_perp * fd).sum(-1) - r   # cup axis ahead of the palm along the fingers, minus r
    h_palm = -v_ax                      # palm offset along the cup axis (env needs |h| <= hh)

    # start orientation kept (palm_normal +y, palm_finger_dir +x); ~0.46 at 20 deg on both axes
    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))
    ang_f = torch.acos(torch.clamp(fd[:, 0], -1.0, 1.0))
    ori = torch.exp(-(ang_n ** 2 + ang_f ** 2) / (2.0 * 0.4 ** 2))

    # distance to the grasp pose; flat inside the approach_done window so the palm may press in
    e_n = torch.relu(gap_n - 0.010) + torch.relu(-0.015 - gap_n)
    e_f = torch.relu((gap_f - 0.006).abs() - 0.010)
    e_h = torch.relu(h_palm.abs() - 0.45 * hh)
    d_pose = torch.sqrt(e_n ** 2 + e_f ** 2 + e_h ** 2 + 1e-12)
    pos = torch.exp(-(d_pose / 0.05) ** 2)    # ~0.6 at the near start (4.5 cm), 1 at the cup
    fine = torch.exp(-(d_pose / 0.018) ** 2)  # sharp version used by the approach

    # ---------------- stage 1: approach (solved last round; kept, only rescaled) ---------------
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    open_pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev.amax(-1) / 0.2) ** 2)
    keep = ori * open_pose
    offset = torch.stack([-(r + 0.0075), -(r + 0.008), 0.25 * hh], dim=-1)  # cup's -y side, behind in +x
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(2.5 * d_world)  # ~0.26 at the raised start (0.38 m), 0.75 at 0.1 m
    approach_reach = s1 * (1.2 * coarse + 1.3 * fine) * (0.2 + 0.8 * keep)  # max 2.5
    approach_keep = s1 * 0.5 * keep                                        # stage 1 max 3.0

    # ---------------- measured contact ---------------------------------------------------------
    touch = _sat(ctx.link_cup_force, 0.5)         # (N,5,3)
    digit_touch = touch.amax(-1)                  # (N,5)
    thumb_touch = digit_touch[:, 0]
    palm_touch = _sat(ctx.palm_cup_force, 0.5)    # (N,)
    palm_firm = _sat(ctx.palm_cup_force, 2.0)     # rewards actually pressing, not grazing
    early_contact_pen = -2.0 * s1 * digit_touch.amax(-1)  # no digit may touch before approach_done

    # ---------------- link geometry in the cup / palm frame ------------------------------------
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

    # wrap angle around the cup: 0 = palm side (a tangential fingertip touch), pi/2 = forward
    # side, pi = far side. An extended finger lying in the palm plane scores ~0 here.
    theta = torch.atan2(u_f, -u_n)
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)  # saturates ~108 deg, round the far side
    finger_wrap = (touch[:, 1:, :] * wrap[:, 1:, :]).amax(-1)        # (N,4) contact-weighted
    wrap_q = torch.clamp(finger_wrap.sum(-1) / 3.0, 0.0, 1.0)        # 3 wrapped fingers saturate
    # thumb must close on the -fd side, opposite the fingers: 1 at u_f=-r, 0.5 at the side, 0 at +r
    thumb_dir = torch.clamp(0.5 - u_f[:, 0, :] / (2.0 * r[:, None]), 0.0, 1.0)
    thumb_q = (touch[:, 0, :] * (0.4 + 0.6 * thumb_dir)).amax(-1)
    fingers_q = torch.clamp(digit_touch[:, 1:].sum(-1) / 3.0, 0.0, 1.0)  # 3 of 4; pinky may substitute

    # ---------------- finger closure (flexion past the open default pose) ----------------------
    flex = torch.clamp((ctx.hand_q_norm - ctx.hand_default_q_norm) / 0.5, 0.0, 1.0)
    digit_flex = torch.stack([flex[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)  # PD command leads the angle on a touching digit

    # ---------------- palm progress: the one blocked condition ---------------------------------
    # NOT flat at the nominal surface: palm_pos is a virtual point and for a wide cup the tangent
    # point lies past the palm's front edge, so contact needs gap_n driven ~1 cm negative.
    press_geo = torch.clamp((0.008 - gap_n) / 0.018, 0.0, 1.0)   # 0 at +8 mm, 1 at -10 mm
    press = torch.maximum(palm_touch, 0.75 * press_geo)          # geometry alone caps at 0.75

    # ---------------- stage gates --------------------------------------------------------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    upright = torch.clamp(1.0 - (ctx.cup_tilt - 0.35) / 0.35, 0.0, 1.0)   # 1 below 20 deg, 0 at 40
    resting = torch.clamp(1.0 - (dz - 0.03) / 0.03, 0.0, 1.0)             # no stage-2 pay once airborne
    ori_k = 0.5 + 0.5 * ori                                               # turning the hand halves the grasp
    g2 = s2 * ori_k * upright * resting
    g3 = s3 * ori_k * upright     # the same grasp terms keep paying while lifting -> releasing loses them
    g23 = g2 + g3

    # ---------------- stage 2/3: the grasp ladder ----------------------------------------------
    # rung 0: being in the grasp pose is worth little on its own (this replaces the old position rent)
    grasp_pose = 3.0 * g23 * pos
    # rung 1: the last centimetres of palm travel - the largest gradient available in this stage
    palm_reach = 6.0 * g23 * pos * press_geo
    # rung 2: measured palm contact, the single missing condition of the envelope
    palm_contact = g23 * (0.3 + 0.7 * pos) * (6.0 * palm_touch + 2.0 * palm_firm)
    # rung 3: curling the digits around the cup (extended digits that merely graze it score 0)
    finger_curl = 2.5 * g23 * pos * (digit_flex[:, 1:] * engage[:, 1:]).mean(-1)
    thumb_curl = 1.5 * g23 * pos * digit_flex[:, 0] * engage[:, 0]
    # small keep-alive so the contact behaviour already learned is not thrown away
    light_contact = 1.0 * g23 * torch.clamp(digit_touch.sum(-1) / 4.0, 0.0, 1.0)
    # rung 4: contact quality, all multiplied by palm progress -> a fingertip grasp gets a fraction
    grasp_wrap = 4.0 * g23 * press * wrap_q
    grasp_thumb = 3.0 * g23 * press * thumb_q
    grasp_fingers = 3.0 * g23 * press * fingers_q
    grip_squeeze = 1.5 * g23 * press * squeeze
    # rung 5: the envelope_done condition itself (palm AND thumb AND >= 3 fingers)
    envelope_quality = 5.0 * g23 * palm_touch * thumb_touch * fingers_q
    # stage 2: <= 8.0 without palm progress, ~21 with the palm pressed in, ~38.5 with the envelope

    # ---------------- stage 3: lift and hold (every term needs the grip to be kept) -------------
    firm = _sat(ctx.link_cup_force, 0.3).amax(-1)                 # (N,5)
    grip_gate = (0.4 + 0.6 * palm_touch) * firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)
    lift_hold = 5.0 * s3 * grip_gate                              # not a flag bonus: needs the grip
    rise_needed = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)
    lift_frac = torch.clamp(dz / rise_needed, 0.0, 1.0)
    lift_height = 8.0 * s3 * grip_gate * lift_frac
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    lift_goal = s3 * grip_gate * (4.0 * (1.0 - torch.tanh(gd / 0.15))
                                  + 5.0 * torch.exp(-gd / 0.03) + 2.0 * in_tol)
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) \
        * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 4.0 * s3 * grip_gate * torch.exp(-gd / 0.03) * calm
    success_bonus = 60.0 * s3 * ctx.success.float()
    # stage 3 peak ~65 raw > stage 2 peak 38.5 > stage 1 peak 3.0, with no drop at either boundary

    # ---------------- constraints and regularization -------------------------------------------
    # nudges while closing are free up to 5 cm; only shoving the cup away costs more than grasping
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.05) / 0.10, 0.0, 1.0)
    # tilt free below ~15 deg, -4 at the 60 deg termination
    cup_tilt_pen = -4.0 * torch.clamp((ctx.cup_tilt - 0.26) / (math.pi / 3.0 - 0.26), 0.0, 1.0)
    table_pen = -3.0 * torch.clamp((ctx.table_z + 0.005 - ctx.hand_z_min) / 0.02, 0.0, 1.0)
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.02 * (ctx.arm_qd ** 2).sum(-1)  # halved: must not discourage closing in

    raw = {
        "approach_reach": approach_reach,
        "approach_keep": approach_keep,
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
        "envelope_quality": envelope_quality,
        "lift_hold": lift_hold,
        "lift_height": lift_height,
        "lift_goal": lift_goal,
        "hold_still": hold_still,
        "success_bonus": success_bonus,
        "cup_disturb_pen": cup_disturb_pen,
        "cup_tilt_pen": cup_tilt_pen,
        "table_pen": table_pen,
        "action_rate_pen": action_rate_pen,
        "arm_vel_pen": arm_vel_pen,
    }
    components = {name: scale * value for name, value in raw.items()}
    reward = torch.stack(list(components.values()), dim=0).sum(0)
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
The round was ended by the user at about epoch 1750 of 3000 (2.7 h), before the time limit, because the policy had settled into pressing the hand into the table - a behaviour that cannot be used on the real robot. The recording available for this round was made from the epoch-1290 checkpoint, about 460 epochs before the end; the behaviour it shows matches the metrics measured at the end of the round.

What the robot does in the recording:
- The hand reaches the cup and closes on it. The palm comes against the cup's side and all four fingers curl around it. This is a real envelope grasp and it is the first time this track has produced one; in the previous round the fingers stayed extended and only grazed the cup's surface.
- The hand travels in low, along the table surface, and presses down into it while approaching and while grasping.
- The grasp is taken low on the cup, near its base, with the hand coming in from underneath rather than against the middle of the cup's body.
- The little finger stays out of the grasp. The thumb and the other three fingers close on the cup.
- The cup stays upright at its spawn position. It is cradled at a slight angle at times but is never knocked over and is never lifted clear of the table.

Metrics over the last 150 epochs of the round:
- approach_done was set in 0.68 of far-start episodes and 1.00 of near-start episodes; envelope_done in 0.59 of far-start and 0.70 of near-start episodes.
- Episode funnel: reach 0.81, grasp 0.76, envelope 0.48, lift 0.11, success 0.0001.
- The palm touched the cup on 0.55 of steps and the mean palm-to-cup gap was 0.036 m. In the previous round palm contact never left the range 0.001 to 0.013.
- On an average step 2.98 of the five digits touched the cup: thumb 0.76, index 0.73, middle 0.72, ring 0.70, little finger 0.059. Mean contact force 4.1 N.
- The cup rose 0.006 m on average. Its distance to the goal stayed at 0.288 m and was never inside the success tolerance; near_goal was 0.0000 and there were no successes in the last 600 epochs.
- Mean cup tilt 8.8 degrees; the cup was tipped over in 0.0003 of episodes and never fell.
- The hand penetrated the table. The maximum penetration depth grew through the round to 0.028 m, and the lowest hand point reached 0.187 while the table surface is at 0.205. The reward's table term paid -0.0002, so this cost almost nothing.
- Total reward was 1.91 and still rising at the end of the round.
- Around epoch 950 there was a short period with the best lifting of the round: lift in 0.296 of episodes, cup rise 0.019 m, successes 0.0080. In that same period table penetration was at its lowest of the whole round, 0.0012 m. As penetration climbed back to 0.028 m, lifting fell to about a third of that peak and never recovered, while the total reward went on rising.
- Success in this round required only that the cup's keypoints be within 0.1125 m of the goal for ten steps. It did not require that the cup be grasped, lifted, upright, or that the hand stay off the table.

To make the code more accurate and train better robot, the feedback for improvement is:
The big thing went right and must be preserved. The hand now reaches the cup, brings its palm against the cup's side and curls the fingers around it. That envelope grasp is what the previous rounds could never produce, and whatever pays for it should keep paying for it.

Everything below is about what happens before and after that grasp.

The hand must not touch the table. A policy that presses into the table cannot be used on the real robot, so this is a hard requirement and not a preference. At the moment the hand travels in low along the surface and pushes down into it, the penetration grew steadily through the round, and what the reward charged for it was so small it may as well have been nothing. Table contact has to be expensive from the first step of the episode - expensive enough that approaching over the table and settling onto it is never the cheapest way to reach the cup. It should cost more than any amount of progress that pressing down buys.

There is direct evidence that the table contact is what blocks the lift. The best lifting of the whole round happened in the one period when the hand was barely touching the table; as the hand sank back onto the surface, lifting fell away and never recovered even though the grasp itself kept improving. Getting the hand off the table is likely to unblock the lift by itself.

The grasp is taken too low. The hand comes in beneath the cup and grips near its base, which both drives the hand into the table and leaves the arm stretched out with nowhere to go when it should lift. The palm should meet the cup on the middle of its body, within about two centimetres above or below the cup's centre, and the approach should arrive at that height rather than climbing up from the table.

The lift is the blocked stage. The cup rises a few millimetres and then stays there, a quarter of a metre from the goal, for the rest of the episode. Nothing in the round ever carried the cup. Once the envelope grasp is held, raising the cup clear of the table and moving it toward the goal has to become the most valuable thing available, and holding a good grasp in place while going nowhere must not be able to pay more than actually carrying it.

The little finger never joins the grasp. The other four digits close and it stays out. If a five-finger wrap is wanted it needs its own reason to close; if four digits are considered enough, nothing has to change here.

Finally, the same warning as last round. Total reward rose all the way to the end while lifting and success went backwards. No term should be able to keep growing while the stage it belongs to regresses; progress on the grasp must not be able to substitute for progress on the lift.

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


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # global scale: approach ~0.1, a perfect envelope on the table ~1.7,
    # the same envelope carried to the goal ~8.2, each success +15.
    scale = 0.1

    r = ctx.cup_radius
    hh = ctx.cup_half_height
    n = ctx.palm_normal
    fd = ctx.palm_finger_dir
    axis = ctx.cup_axis

    # ---------------- stage masks (strict order, flags are latched by the env) ----------------
    approach = ctx.approach_done.bool()
    envelope = ctx.envelope_done.bool() & approach
    s1 = (~approach).float()
    s2 = (approach & ~envelope).float()
    s3 = envelope.float()

    # ---------------- table clearance: a hard requirement, charged in every stage -------------
    # `hand_z_min` excludes the palm, and the last round came in from underneath, so the palm
    # height is charged separately. The corridor starts 2.5 cm up: settling onto the surface
    # must never be the cheap way in.
    clear = ctx.hand_z_min - ctx.table_z
    near_tab = torch.clamp((0.025 - clear) / 0.025, 0.0, 1.0)   # 0 at 2.5 cm, 1 at contact
    deep_tab = torch.clamp((0.008 - clear) / 0.013, 0.0, 1.0)   # 0 at 8 mm, 1 at the -5 mm cutoff
    # -2.2 at 1 cm, -12 at contact, -22 at the termination depth: more than the whole ladder (16.6)
    table_pen = -(6.0 * near_tab ** 2 + 16.0 * deep_tab ** 2)
    palm_clear = ctx.palm_pos[:, 2] - ctx.table_z
    # free once the palm is 2 cm up (the grasp height is ~cup_half_height up), -8 if it sinks
    palm_low_pen = -8.0 * torch.clamp((0.020 - palm_clear) / 0.025, 0.0, 1.0) ** 2
    # second, multiplicative pressure: the grasp ladder itself is halved on the surface
    air = 0.5 + 0.5 * torch.clamp(clear / 0.02, 0.0, 1.0)

    # ---------------- palm pose relative to the cup (same quantities approach_done checks) ----
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

    # distance to the grasp pose. Flat inside the window so the palm may still press in, but the
    # axial window is now the MIDDLE of the cup body (-1.5 cm .. +2.5 cm about the centre):
    # gripping near the base is what drove the hand into the table last round.
    e_n = torch.relu(gap_n - 0.010) + torch.relu(-0.015 - gap_n)
    e_f = torch.relu((gap_f - 0.006).abs() - 0.010)
    e_h = torch.relu(h_palm - 0.025) + torch.relu(-0.015 - h_palm)
    d_pose = torch.sqrt(e_n ** 2 + e_f ** 2 + e_h ** 2 + 1e-12)
    pos = torch.exp(-(d_pose / 0.05) ** 2)    # ~0.6 at the near start (4.5 cm), 1 in the window
    fine = torch.exp(-(d_pose / 0.018) ** 2)  # sharp version used by the approach

    # ---------------- stage 1: approach (solved in the last rounds; kept, only rescaled) ------
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    open_pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev.amax(-1) / 0.2) ** 2)
    keep = ori * open_pose
    # coarse target: the cup's -y side, behind the axis in +x, at the cup's mid-height, so the
    # approach ARRIVES at grasp height instead of climbing up off the table
    off_z = torch.clamp(0.25 * hh, max=0.020)
    offset = torch.stack([-(r + 0.006), -(r + 0.008), off_z], dim=-1)
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(2.5 * d_world)   # ~0.26 at the raised start (0.38 m), 0.75 at 0.1 m
    approach_reach = s1 * (0.5 * coarse + 0.5 * fine) * (0.2 + 0.8 * keep)   # max 1.0
    approach_keep = s1 * 0.25 * keep                                        # stage 1 max 1.25

    # ---------------- measured contact ---------------------------------------------------------
    touch = _sat(ctx.link_cup_force, 0.5)         # (N,5,3)
    digit_touch = touch.amax(-1)                  # (N,5)
    thumb_touch = digit_touch[:, 0]
    palm_touch = _sat(ctx.palm_cup_force, 0.5)    # (N,)
    palm_firm = _sat(ctx.palm_cup_force, 2.0)     # rewards actually pressing, not grazing
    early_contact_pen = -2.0 * s1 * digit_touch.amax(-1)  # no digit may touch before approach_done

    # ---------------- link geometry in the cup / palm frame ------------------------------------
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

    # wrap angle around the cup: 0 = palm side (a tangential fingertip touch), pi/2 = forward
    # side, pi = far side. An extended finger lying in the palm plane scores ~0 here.
    theta = torch.atan2(u_f, -u_n)
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)  # saturates ~108 deg, round the far side
    finger_wrap = (touch[:, 1:, :] * wrap[:, 1:, :]).amax(-1)        # (N,4) contact-weighted
    wrap_q = torch.clamp(finger_wrap.sum(-1) / 3.0, 0.0, 1.0)        # 3 wrapped fingers saturate
    # thumb must close on the -fd side, opposite the fingers: 1 at u_f=-r, 0.5 at the side, 0 at +r
    thumb_dir = torch.clamp(0.5 - u_f[:, 0, :] / (2.0 * r[:, None]), 0.0, 1.0)
    thumb_q = (touch[:, 0, :] * (0.4 + 0.6 * thumb_dir)).amax(-1)
    fingers_q = torch.clamp(digit_touch[:, 1:].sum(-1) / 3.0, 0.0, 1.0)  # 3 of 4; pinky may substitute

    # ---------------- finger closure (flexion past the open default pose) ----------------------
    flex = torch.clamp((ctx.hand_q_norm - ctx.hand_default_q_norm) / 0.5, 0.0, 1.0)
    digit_flex = torch.stack([flex[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)  # PD command leads the angle on a touching digit

    # ---------------- palm progress ------------------------------------------------------------
    # smooth over the whole remaining travel (palm_pos is a virtual point, and the measured gap
    # can read several centimetres while the palm surface is already touching), so the pull to
    # the cup's side never goes flat at any distance
    press_geo = torch.exp(-torch.relu(gap_n + 0.010) / 0.025)   # 1 at -1 cm, 0.67 at 0, 0.30 at +2 cm
    press = torch.maximum(palm_touch, 0.7 * press_geo)          # geometry alone caps at 0.7

    # ---------------- stage gates --------------------------------------------------------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    upright = torch.clamp(1.0 - (ctx.cup_tilt - 0.35) / 0.35, 0.0, 1.0)   # 1 below 20 deg, 0 at 40
    resting = torch.clamp(1.0 - (dz - 0.03) / 0.03, 0.0, 1.0)             # no stage-2 pay once airborne
    ori_k = 0.5 + 0.5 * ori                                               # turning the hand halves it
    # anti-farming: sitting in stage 2 is worth less the longer it lasts (the budget restarts on
    # every success, so stage 3 never decays). Relative ordering inside stage 2 is unchanged.
    dwell = 1.0 - 0.5 * torch.clamp(ctx.episode_progress, 0.0, 1.0)
    g2 = s2 * ori_k * upright * resting * air * dwell
    g3 = s3 * ori_k * upright * air     # the same ladder keeps paying while lifting -> releasing loses it
    g23 = g2 + g3

    # ---------------- stage 2/3: the grasp ladder (CAPPED at ~16.6 raw; it cannot grow further) -
    # rung 0: standing in the grasp pose is worth little on its own
    grasp_pose = 1.5 * g23 * pos
    # rung 1: the last centimetres of palm travel
    palm_reach = 2.5 * g23 * pos * press_geo
    # rung 2: measured palm contact - the condition that blocked the envelope two rounds ago
    palm_contact = g23 * (0.3 + 0.7 * pos) * (2.0 * palm_touch + 1.0 * palm_firm)
    # rung 3: curling the digits around the cup (extended digits that merely graze it score 0)
    finger_curl = 1.5 * g23 * pos * (digit_flex[:, 1:] * engage[:, 1:]).mean(-1)
    thumb_curl = 0.8 * g23 * pos * digit_flex[:, 0] * engage[:, 0]
    # small keep-alive so the contact behaviour already learned is not thrown away
    light_contact = 0.4 * g23 * torch.clamp(digit_touch.sum(-1) / 4.0, 0.0, 1.0)
    # rung 4: contact quality, all multiplied by palm progress -> a fingertip grasp gets a fraction
    grasp_wrap = 1.5 * g23 * press * wrap_q
    grasp_thumb = 1.2 * g23 * press * thumb_q
    grasp_fingers = 1.2 * g23 * press * fingers_q
    grip_squeeze = 0.6 * g23 * press * squeeze
    # the little finger is averaged away everywhere else, so it gets its own reason to close
    pinky_join = 0.4 * g23 * press * digit_touch[:, 4] * digit_flex[:, 4]
    # rung 5: the envelope_done condition itself (palm AND thumb AND >= 3 fingers)
    envelope_quality = 2.0 * g23 * palm_touch * thumb_touch * fingers_q

    # ---------------- stage 3: lift and hold - the largest payments in the function -------------
    firm = _sat(ctx.link_cup_force, 0.3).amax(-1)                 # (N,5)
    grip = (0.3 + 0.7 * palm_touch) * firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)
    # deliberately small: holding the grasp while going nowhere must not outpay carrying the cup
    lift_hold = 2.0 * s3 * grip
    lift_clear = 8.0 * s3 * grip * torch.clamp(dz / 0.05, 0.0, 1.0)   # buys the first 5 cm
    rise_needed = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)
    lift_frac = torch.clamp(dz / rise_needed, 0.0, 1.0)
    lift_height = 20.0 * s3 * grip * lift_frac                        # the main carrying gradient
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    lift_goal = s3 * grip * (8.0 * (1.0 - torch.tanh(gd / 0.15))
                             + 12.0 * torch.exp(-gd / 0.04) + 5.0 * in_tol)
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) \
        * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 10.0 * s3 * grip * torch.exp(-gd / 0.04) * calm      # only pays AT the goal
    success_bonus = 150.0 * s3 * ctx.success.float()
    # perfect envelope parked on the table ~16.6 raw/step; the same grasp at the goal ~82 raw/step

    # ---------------- constraints and regularization -------------------------------------------
    # nudges while closing are free up to 4 cm; only shoving the cup away costs more than grasping
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.04) / 0.10, 0.0, 1.0)
    # tilt free below ~15 deg, -5 at the 60 deg termination
    cup_tilt_pen = -5.0 * torch.clamp((ctx.cup_tilt - 0.26) / (math.pi / 3.0 - 0.26), 0.0, 1.0)
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.02 * (ctx.arm_qd ** 2).sum(-1)   # small: must not discourage carrying

    raw = {
        "approach_reach": approach_reach,
        "approach_keep": approach_keep,
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
        "lift_height": lift_height,
        "lift_goal": lift_goal,
        "hold_still": hold_still,
        "success_bonus": success_bonus,
        "cup_disturb_pen": cup_disturb_pen,
        "cup_tilt_pen": cup_tilt_pen,
        "table_pen": table_pen,
        "palm_low_pen": palm_low_pen,
        "action_rate_pen": action_rate_pen,
        "arm_vel_pen": arm_vel_pen,
    }
    components = {name: scale * value for name, value in raw.items()}
    reward = torch.stack(list(components.values()), dim=0).sum(0)
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
The round was stopped by the user at epoch 2041 of 3000 (3.35 h), before the time limit, because the recording showed the far-start approach being made in the wrong way and the user judged that both the reward and the environment settings need changing. The recording was made from the epoch-1500 checkpoint and shows two consecutive episodes of one environment (30 s); the user identified the first episode as a far start and the second as a near start.

What the robot does in the recording:
- First episode (far start): the hand does reach the cup and ends up gripping it, but it gets there by rotating the wrist rather than by carrying the hand to the cup's side in its start orientation. The hand arrives low and turned, so the cup is reached without the C-shaped pre-grasp pose the approach is supposed to hold.
- Second episode (near start): the hand is already beside the cup, closes on it immediately and lifts it. There is no approach phase to speak of.
- In both episodes the grasp itself is a real envelope: the palm is against the cup and the fingers wrap around it, and the cup is carried well clear of the table, up to about the height of the robot's column.
- The cup is visibly tilted while it is carried, by roughly thirty degrees in several frames. It is never dropped and never knocked over.
- The hand goes in low but does not sink into the table.

Metrics over the last 150 epochs, at epoch 2041:
- approach_done was set in 0.0014 of far-start episodes and 1.0000 of near-start episodes. Its highest value for far starts in the whole round was 0.669, reached at about epoch 530, after which it fell away and did not return.
- envelope_done was set in 0.0022 of far-start and 0.9288 of near-start episodes; lift in 0.0040 and 0.9350; success in 0.0000 and 0.7549. Essentially every success in this round came from a near start.
- Episode funnel: reach 0.45, grasp 0.42, envelope 0.40, lift 0.40, success 0.34. Successes per finished episode 1.71, with a maximum of 2.84 during the round.
- Of the six approach conditions, orientation 0.855 and no-touch 0.735 pass often, while gap 0.083, along 0.046, height 0.363 and pose 0.208 rarely do; all six must hold on the same step for approach_done to latch.
- The cup was raised 0.056 m on average, against 0.006 m in the previous round, and its distance to the goal fell to 0.210 m from 0.288 m. Mean cup tilt 3.7 degrees at the end, having peaked at 23.9 degrees mid-round; the cup was tipped over in 0.0002 of episodes.
- The hand stayed off the table for the whole round: the episode-ending floor condition never fired once, the deepest penetration was 0.006 m against 0.028 m in the previous round, and the lowest hand point stayed at 0.2145 while the table is at 0.205.
- The little finger touched the cup on 0.242 of steps, against 0.059 in the previous round.
- Both curricula tightened continuously: the success tolerance from 0.1125 m to 0.0664 m, and the wrap threshold that success also requires from 0.150 to 0.689, its ceiling being 0.85. The measured wrap fraction at the end was 0.189, far below that threshold.
- Total reward 1.071 at the end, with the lift terms now the largest positive parts: lift_height 0.394, lift_goal 0.212, lift_clear 0.180.
- The table measure used during this round did not include the palm. It was built from the finger and thumb links only, 28 of the hand's 30 bodies, leaving out the palm and the virtual palm point. The palm collision body measures 64 by 82 by 99 mm and its lowest extent sits 40 mm from the virtual palm point along one palm axis and up to 65 mm along another, so the palm can be on the table while that point is well above it. This is why the floor condition never fired and why the reward term meant to keep the palm up, which used the virtual point with a 20 mm threshold, paid exactly zero for the whole round.
- Per-joint arm action saturation does not single out the wrist: j1 0.554, j6 0.515, j4 0.487, j5 0.432, j7 0.396. No arm joint went past its limit at any point. The recording shows the wrist-led approach clearly, but the logged telemetry does not measure which joint closes the distance, so this is an observation from the video only.

To make the code more accurate and train better robot, the feedback for improvement is:
Three things went right and must be kept.
- The grasp and the lift both work now. The palm comes against the cup, the fingers wrap around it, and the cup is carried well clear of the table instead of rising a few millimetres. Whatever pays for that should keep paying for it.
- The hand stays off the table for the whole round. The episode-ending floor condition never fired once and the deepest penetration was a fifth of what it was before. This requirement is met and must not be given back.
- The little finger now joins the grasp instead of trailing beside it.

The rest is what has to change.

The approach from the raised start has been abandoned, and that is the main failure. Almost every success in this round came from episodes that begin beside the cup. Far-start episodes did learn the approach for a while in the middle of the round and then lost it, and by the end they almost never complete it. Starting beside the cup is a training aid, not the task; the task is to come from the raised pose outside the table and grasp the cup. Reaching the cup from the raised start has to be worth enough that it is never cheaper to give up on it, and progress made while travelling towards the cup from far away must keep paying all the way in rather than only once the hand is already there.

The approach that does happen is made in the wrong way. Instead of carrying the hand to the cup's side while holding its start orientation and its open default pose, the arm turns the wrist so the hand arrives low and rotated. That reaches the cup but skips the shape the approach is supposed to have, which is why the approach almost never registers as complete even in episodes that go on to grasp. Arriving at the cup's side in the start orientation with the hand still open should be what pays; getting within reach by rotating the wrist should not.

The cup is carried tilted. It is gripped and lifted, but it leans noticeably while being moved, at times by about thirty degrees. It is never dropped, so nothing corrects this. Carrying the cup upright should be worth clearly more than carrying it at an angle, from the moment it leaves the table rather than only at the goal.

The hand must not rest on the table, and this round could not see when it did. The measure the environment used for the table looked only at the finger and thumb links and left the palm out of it, and the palm is the largest flat part of the hand. The recording shows the hand skimming the table with the palm while the arm turned the wrist to bring the hand to the cup, and none of that registered. The environment now measures the palm itself and gives its clearance above the table to the reward directly, so that distance no longer has to be guessed from the virtual palm point, which is up to 0.065 m away from the palm surface and moves as the wrist turns. The episode-ending floor check still looks only at the fingers, so keeping the palm off the table is entirely the reward's job: resting or sliding the palm on the surface has to cost more than the approach it buys.

One caution about the middle of the round. The far-start approach reached its best value around epoch 530 and collapsed over the following two hundred epochs, exactly as near-start success took off. A reward that lets the easy start dominate will keep producing this, so the far start needs its own reason to improve that does not disappear once the near start is succeeding.

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
    # Global scale. The raw ladder is much taller than last round (stage 1 is 8x bigger), so the
    # scale is halved to keep per-step totals in the range that trained well: a good approach
    # ~0.3-0.5, a perfect envelope parked on the table ~2.6, that grasp at the goal ~9.6, +15/success.
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

    # ---------------- table clearance: a hard requirement, charged in every stage --------------
    # `hand_z_min` covers the finger/thumb links only. The palm - the largest flat part of the
    # hand, and the part that was skimming the table last round - is charged separately from the
    # MEASURED ctx.palm_clearance, not from the virtual palm point (which paid exactly 0).
    clear = ctx.hand_z_min - ctx.table_z
    near_tab = torch.clamp((0.025 - clear) / 0.025, 0.0, 1.0)   # 0 at 2.5 cm, 1 at contact
    deep_tab = torch.clamp((0.008 - clear) / 0.013, 0.0, 1.0)   # 0 at 8 mm, 1 at the -5 mm cutoff
    # -2.9 at 1 cm, -16 at contact, -38 at the termination depth: more than the whole grasp ladder
    table_pen = -(8.0 * near_tab ** 2 + 30.0 * deep_tab ** 2)
    pc = ctx.palm_clearance
    # Free above 8 mm. NOT higher: for the smallest cup (hh = 42 mm) a correct mid-body grasp puts
    # the palm's lowest point only millimetres above the table, and a 2 cm threshold would punish
    # the envelope grasp that finally works. Resting/sliding the palm still costs -8, digging -48.
    palm_low = torch.clamp((0.008 - pc) / 0.008, 0.0, 1.0)
    palm_dig = torch.clamp(-pc / 0.010, 0.0, 1.0)
    palm_table_pen = -(8.0 * palm_low ** 2 + 40.0 * palm_dig ** 2)
    # second, multiplicative pressure: the grasp ladder is halved on the surface
    air = (0.5 + 0.5 * torch.clamp(clear / 0.020, 0.0, 1.0)) \
        * (0.7 + 0.3 * torch.clamp(pc / 0.010, 0.0, 1.0))

    # ================ STAGE 1: approach =========================================================
    # Last round this whole stage was capped at 1.25 raw and logged 0.004, against 1.03 total for
    # near-start episodes - far starts were abandoned because their signal was invisible. The cap
    # is now 10.0, on the same order as a near-start step. These terms carry `s1`, so they exist
    # ONLY in far-start episodes: the easy start can never take this reason to improve away.

    # (i) the six approach_done conditions, scored individually. The env reports gap 0.083,
    # along 0.046, pose 0.208 and height 0.363 as the ones that fail, and all six must hold on the
    # SAME step, so both the mean (dense partial credit) and the product (simultaneity) are paid.
    c_gap = _win(gap_n, -0.010, 0.020, 0.015)
    c_along = _win(gap_f, -0.005, 0.020, 0.015)
    c_height = _win(h_palm, -0.9 * hh, 0.9 * hh, 0.020)
    c_ori = _win(ang_n, 0.0, 0.60, 0.30) * _win(ang_f, 0.0, 0.60, 0.30)   # env allows ~45 deg
    c_pose = _win(dev_max, 0.0, 0.22, 0.12)                               # env allows 0.3
    c_free = 1.0 - any_touch                                              # nothing may touch yet
    conds = torch.stack([c_gap, c_along, c_height, c_ori, c_pose, c_free], dim=-1)   # (N,6)
    form = conds.mean(-1)
    lock = conds.prod(-1)

    # (ii) distance to the pre-grasp window, aimed at the MIDDLE of the cup body so the approach
    # arrives at grasp height instead of climbing up off the table
    a_n = torch.relu(gap_n - 0.018) + torch.relu(-0.008 - gap_n)
    a_f = torch.relu(gap_f - 0.018) + torch.relu(-0.003 - gap_f)
    a_h = torch.relu(h_palm - 0.025) + torch.relu(-0.012 - h_palm)
    d_app = torch.sqrt(a_n ** 2 + a_f ** 2 + a_h ** 2 + 1e-12)
    prox = torch.exp(-(d_app / 0.12) ** 2)
    fine_app = torch.exp(-(d_app / 0.05) ** 2)

    # (iii) long-range travel. k = 1.8 (was 2.5) so the 0.38 m raised start still sits on a real
    # slope: 0.41 at the start, 0.60 at 0.25 m, 0.82 at 0.1 m - progress pays all the way in.
    off_z = torch.clamp(0.30 * hh, max=0.020)
    offset = torch.stack([-(r + 0.006), -(r + 0.008), off_z], dim=-1)
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(1.8 * d_world)

    dwell1 = 1.0 - 0.3 * progress          # dawdling in stage 1 is mildly worse than finishing it
    # shape floor 0.15: a turned/closed hand forfeits 85 % of the approach but keeps a gradient to
    # climb back on. Reaching the cup by rotating the wrist is what this multiplier is aimed at.
    shape_k = 0.15 + 0.85 * keep
    approach_travel = 4.0 * s1 * dwell1 * (0.55 * coarse + 0.45 * fine_app) * shape_k
    approach_form = 3.0 * s1 * dwell1 * form * (0.3 + 0.7 * prox)
    approach_lock = 2.0 * s1 * lock        # paid only when all six latch conditions hold at once
    approach_keep = 1.0 * s1 * keep * (0.2 + 0.8 * prox)   # holding the shape pays where it counts
    early_contact_pen = -3.0 * s1 * any_touch              # no digit may touch before the flag
    # stage 1 max = 10.0, reachable only in the configuration that makes the env latch the flag

    # ---------------- grasp pose (wider than the latch window: the palm may press in) ----------
    e_n = torch.relu(gap_n - 0.010) + torch.relu(-0.015 - gap_n)
    e_f = torch.relu((gap_f - 0.006).abs() - 0.010)
    # middle of the cup body; gripping near the base drove the hand into the table two rounds ago
    e_h = torch.relu(h_palm - 0.025) + torch.relu(-0.012 - h_palm)
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

    # wrap angle around the cup: 0 = palm side (a tangential fingertip touch), pi/2 = forward side,
    # pi = far side. An extended finger lying in the palm plane scores ~0 here.
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

    # ---------------- stage gates -------------------------------------------------------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    upright = torch.clamp(1.0 - (ctx.cup_tilt - 0.35) / 0.35, 0.0, 1.0)   # 1 below 20 deg, 0 at 40
    # carrying uprightness: 1 below 5 deg, 0.78 at 10, 0.35 at 15, 0.002 at the 30 deg seen on video
    upright_c = torch.exp(-(torch.relu(ctx.cup_tilt - 0.09) / 0.17) ** 2)
    resting = torch.clamp(1.0 - (dz - 0.03) / 0.03, 0.0, 1.0)             # no stage-2 pay once airborne
    ori_k = 0.5 + 0.5 * ori                     # turning the hand halves the grasp, in stage 3 too
    dwell2 = 1.0 - 0.5 * progress               # anti-farming: parking in stage 2 decays
    g2 = s2 * ori_k * upright * resting * air * dwell2
    g3 = s3 * ori_k * upright * air     # the same ladder keeps paying while lifting -> releasing loses it
    g23 = g2 + g3

    # ================ STAGE 2/3: the grasp ladder (CAPPED at ~51 raw) ===========================
    # rung 0 is set to 10.0 = stage 1's ceiling, so the step the flag latches the reward steps UP.
    grasp_pose = 10.0 * g23 * pos
    # rung 1: the last centimetres of palm travel
    palm_reach = 6.0 * g23 * pos * press_geo
    # rung 2: measured palm contact, the condition that blocked the envelope for two rounds
    palm_contact = g23 * (0.3 + 0.7 * pos) * (5.0 * palm_touch + 2.0 * palm_firm)
    # rung 3: curling the digits round the cup. Raised 1.5 -> 5.0 / 0.8 -> 2.5 because the success
    # curriculum now demands a wrap measure of 0.689 (ceiling 0.85) while only 0.189 is achieved.
    finger_curl = 5.0 * g23 * pos * (digit_flex[:, 1:] * engage[:, 1:]).mean(-1)
    thumb_curl = 2.5 * g23 * pos * digit_flex[:, 0] * engage[:, 0]
    light_contact = 1.0 * g23 * torch.clamp(digit_touch.sum(-1) / 4.0, 0.0, 1.0)   # keep-alive
    # rung 4: contact quality, all multiplied by palm progress -> a fingertip grasp gets a fraction
    grasp_wrap = g23 * press * (3.5 * wrap_q + 1.5 * wrap_deep)   # wrap_deep needs all four fingers
    grasp_thumb = 3.0 * g23 * press * thumb_q
    grasp_fingers = 3.0 * g23 * press * fingers_q
    grip_squeeze = 1.5 * g23 * press * squeeze
    pinky_join = 1.2 * g23 * press * digit_touch[:, 4] * digit_flex[:, 4]   # it is averaged away elsewhere
    # rung 5: the envelope_done condition itself (palm AND thumb AND >= 3 fingers)
    envelope_quality = 6.0 * g23 * palm_touch * thumb_touch * fingers_q
    # ladder max ~51 raw and it CANNOT grow further: grasp progress can never substitute for lift
    # progress, which is worth 141 more on top of it.

    # ================ STAGE 3: lift, carry upright, hold ========================================
    firm = _sat(ctx.link_cup_force, 0.3).amax(-1)                 # (N,5)
    grip = (0.3 + 0.7 * palm_touch) * firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)
    # deliberately small: holding the grasp while going nowhere must not outpay carrying the cup
    lift_hold = 3.0 * s3 * grip
    airborne = torch.clamp(dz / 0.03, 0.0, 1.0)
    # carrying it upright is paid from the moment it leaves the table, not only at the goal
    carry_upright = 8.0 * s3 * grip * airborne * upright_c
    lift_clear = 15.0 * s3 * grip * upright_c * torch.clamp(dz / 0.05, 0.0, 1.0)   # the first 5 cm
    rise_needed = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)
    lift_frac = torch.clamp(dz / rise_needed, 0.0, 1.0)
    lift_height = 45.0 * s3 * grip * upright_c * lift_frac         # the main carrying gradient
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    lift_goal = s3 * grip * upright_c * (15.0 * (1.0 - torch.tanh(gd / 0.15))
                                         + 25.0 * torch.exp(-gd / 0.04) + 8.0 * in_tol)
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) \
        * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 22.0 * s3 * grip * upright_c * torch.exp(-gd / 0.04) * calm   # only pays AT the goal
    success_bonus = 300.0 * s3 * ctx.success.float()
    # ordering: stage 1 <= 10 < stage-2 boundary ~13 < parked envelope ~51 < carried to goal ~192

    # ---------------- constraints and regularization -------------------------------------------
    # nudges while closing are free up to 4 cm; only shoving the cup away costs more than grasping
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.04) / 0.10, 0.0, 1.0)
    # tilt free below ~15 deg, -6 at the 60 deg termination (additive, on top of the multiplicative
    # upright_c that removes the carrying payments themselves)
    cup_tilt_pen = -6.0 * torch.clamp((ctx.cup_tilt - 0.26) / (math.pi / 3.0 - 0.26), 0.0, 1.0)
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.02 * (ctx.arm_qd ** 2).sum(-1)   # small: must not discourage travelling in

    raw = {
        "approach_travel": approach_travel,
        "approach_form": approach_form,
        "approach_lock": approach_lock,
        "approach_keep": approach_keep,
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
        "lift_height": lift_height,
        "lift_goal": lift_goal,
        "carry_upright": carry_upright,
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
Round 8 (`fj_stage_i07`) ran to epoch 1712 over 3.14 hours and was stopped by the user. It is the round that fixed the failure of the previous one.

**The raised start learned to approach.** This was the single goal of the round and it succeeded. The fraction of far-start episodes that complete the approach was 0.000 for the first three hundred epochs, crossed zero around epoch 330, and reached 0.742 by epoch 800. It settled at 0.625. In the previous round this figure peaked at 0.669 around epoch 530 and then collapsed to 0.001; here it did not collapse.

**The whole chain now works from the raised start.** By the end of the round, far-start episodes reached the cup 0.943 of the time, closed a grasp 0.889, completed the envelope 0.769, and lifted 0.780. The previous round produced no far-start successes at all; this one peaked at 0.370.

**The grasp is a real envelope grasp.** At the moment of success the palm was on the cup 0.914 of the time and 4.35 digits were in contact. Grasp quality at lift was 0.648. The recording confirms it: the palm lies against the side of the cup with four fingers wrapped around the body and the thumb opposing them. This is not a fingertip pinch.

**The cup is carried much higher than before.** Lift height reached 0.208 m against 0.056 m in the previous round, and 0.62 of episodes had the cup airborne.

**The hand stayed off the table for the whole round.** Palm clearance ended at 0.276 m, the palm-on-table fraction was zero, and the episode-ending floor condition never fired.

Three things went wrong, and they are the substance of the next round.

**The round peaked at epoch 1000-1200 and declined for the remaining five hundred epochs.** Far-start success went 0.370, then 0.215, then 0.066. The mean episode success count went 1.257, then 0.655, then 0.200. Meanwhile the total reward kept climbing the whole time — 1.455, 1.387, 1.705, its maximum at the very end. Reward rose while the task went backwards. This is the same signature as round 5, where the total climbed from 0.73 to 0.96 while palm contact fell away.

The cause is visible in the individual terms. Income for raising the cup rose from 0.489 to 0.622 and income for clearing the table from 0.195 to 0.234, while the success bonus fell from 0.0140 to 0.0014 — a tenfold drop — and the term that pays for holding still at the goal never left 0.0001 in more than a thousand epochs. Income for arriving at the goal also fell, from 0.0290 to 0.0139. The policy found that holding the cup up pays better than carrying it to the goal and stopping there, and moved its income accordingly.

**The cup is carried lying over.** The recording shows it held at roughly thirty to forty-five degrees from upright through most of the carry. The logged mean tilt of 9.6 degrees hides this, because it averages every environment and every step including the long stretches before the lift. The reason tilting is free is arithmetic: the tilt penalty was 0.0146 against 0.622 of lift income, a ratio of one to forty-three, and the upright term is added alongside the lift income rather than gating it, so a cup raised on its side still collects the raising income in full.

**The approach posture is abandoned the instant the latch fires.** The approach gate is a latch — six conditions true in a single step, and it stays on for the rest of the episode. The per-step condition rates show what happens after that: not touching the cup fell from 0.968 to 0.240, the along-axis window from 0.217 to 0.046, the plane gap from 0.292 to 0.169. The penalty for touching the cup during the approach grew fifteenfold, from 0.0017 to 0.0250, and the policy paid it willingly. The episode-level latch stayed at 0.625 through all of this, so the funnel never showed it. In the recording the approach itself still looks reasonable — the hand comes in with the fingers extended — but the shape is held for an instant rather than maintained into the grasp.

Two notes for the record. The wrap ratchet never moved: the mean episode success count peaked at 1.697 and never reached the threshold of 2.0, so the tolerance stayed at its starting value of 0.150 for the entire round and the new ceiling of 0.35 introduced this round is still untested. And a maximum hand joint tracking error of 1051 and later 341 appeared in two buckets; the arm velocity maximum spiked with it and the runaway termination fired, so these were isolated diverging environments that were terminated, not a control fault — the value returned to 2.9 afterwards.

To make the code more accurate and train better robot, the feedback for improvement is:
Four things went right and must be kept.

- The approach from the raised start works now, and it does not collapse. This was the failure of the previous round and it is fixed. Whatever pays for travelling in from far away and arriving in the pre-grasp shape must keep paying exactly as it does.
- The grasp is a true envelope grasp: the palm against the side of the cup, four fingers wrapped around the body, the thumb opposing. At the moment of success the palm is in contact 0.91 of the time. Keep whatever produces this.
- The cup is lifted well clear of the table — 0.208 m, almost four times the previous round.
- The hand never rests on the table. Palm clearance ends at 0.276 m and the floor condition never fires. This requirement is met and must not be given back.

The rest is what has to change.

**Carrying the cup upward must stop being a substitute for finishing the task.** This is the main failure. Income for raising the cup grew to 0.622 and income for clearing the table to 0.234, while the bonus for succeeding fell to 0.0014 and the term that pays for settling at the goal stayed at 0.0001 for more than a thousand epochs. Success peaked at epoch 1000-1200 and then fell by a factor of five, and the total reward rose the whole time it was falling. Holding the cup in the air is currently the most profitable thing the policy can do, and it is not the task. Raising the cup should be worth something only as a step towards the goal: the income for height should saturate quickly once the cup is clear of the table, and the great majority of the remaining value should sit at the goal — arriving there, slowing down, and staying. A policy that lifts the cup and hovers must earn clearly less than one that carries it to the goal and stops.

**Carrying the cup on its side must not collect the carrying income.** The cup is held at thirty to forty-five degrees from upright through most of the carry. The tilt penalty is 0.0146 against 0.622 of lift income — one part in forty-three — and being upright is paid as a separate additive term, so a cup raised sideways still earns the raising income in full. Being upright should multiply the carrying and goal income rather than sit beside it, so that a cup carried lying over earns close to nothing for being carried, from the moment it leaves the table rather than only at the goal.

**Holding the approach shape must pay until the grasp actually begins.** The approach gate latches on a single step and then stays on, and the policy has learned to satisfy it for an instant and immediately drive into the cup. Across the round, not touching the cup during the approach fell from 0.968 to 0.240 and the penalty for touching it early grew fifteenfold while being paid willingly. The episode-level latch never showed this because it only needs one step. Maintaining the pre-grasp shape — the hand open, the palm facing the cup, no digit contact — from the moment the approach completes until the palm is actually against the cup should be worth enough that breaking it early is not profitable. The intended order is approach, then close from the approached pose; right now it is approach for one step, then dive.

One caution about the shape of this round. Everything good happened between epoch 300 and epoch 1200, and the last five hundred epochs undid a large part of it while the total reward reached its maximum. A reward that lets a mid-task behaviour pay more than finishing will keep producing this, so the value has to be concentrated at the end of the task rather than along the way to it.

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
Round 9 (`fj_stage_i08`) ran to epoch 4048 over 6.47 hours and was stopped by the user. It is the round in which the task, as the user defined it, started working from the raised start.

**The whole chain works from the raised start.** Over the last two hundred epochs, far-start episodes completed the approach 0.882 of the time, closed the envelope 0.919, lifted 0.934, and succeeded 0.764. The previous round ended with far-start success at 0.042 after collapsing from a peak of 0.370. This round did not collapse; success rose from zero at epoch 800 to 0.84 by epoch 2900 and held between 0.75 and 0.89 for the remaining twelve hundred epochs.

**The cup is now carried upright.** This was the clearest failure of the previous round, where the recording showed the cup held at thirty to forty-five degrees through most of the carry. Mean tilt ended at 5.45 degrees against 10.44 in the previous round, and the tilt penalty shrank from 0.046 to 0.007 as the policy stopped incurring it. The recording confirms it: in the frames where the cup is off the table it stands close to vertical in the hand, with the palm against its side and four fingers wrapped around the body.

**The grasp is a true envelope grasp and it improved.** At the moment of success the palm was in contact 0.767 of the time and 4.70 digits were touching. Grasp quality rose through the round to 0.439. Lift height reached 0.169 m.

**The reward moved to the end of the task, which is what the round was for.** The terms that pay for arriving at the goal and settling there went from nothing to real income: goal proximity 0.611, goal closeness 0.387, and the settling term 0.068. In the previous round that settling term never left 0.0001 across 1712 epochs; here it is 678 times larger. The success bonus ended at 0.077 against 0.0009.

**The hand never touched the table.** Palm clearance ended at 0.253 m, the palm-on-table fraction was zero from epoch 1000 onward, and the floor termination never fired.

**The wrap ratchet was tested and held.** Mean episode success crossed the threshold of 2.0 for the first time in this track, the tolerance ratchet ran 0.150 → 0.192 → 0.288 → 0.350, and it stopped exactly at the ceiling of 0.350 and stayed there for the last two thousand epochs. Two rounds ago the same ratchet ran to 0.689 and locked success out.

Four things are wrong, and they are the substance of the next round.

**The cup is set back down and picked up again within one episode.** The recording shows the cup airborne and upright in several frames and back on the table in others, with the hand beside or above it, across what appears to be a single episode window. The metrics are consistent with this: the airborne fraction ended at 0.692 while success is 0.764 and the settling term is only 0.068, so episodes are spending a large part of their time holding or re-approaching rather than finishing. Carrying the cup to the goal once and stopping should be worth more than a sequence of lifts.

**The approach posture is still abandoned the moment the latch fires, and worse than before.** The approach gate is satisfied for a single step and then the policy drives into the cup. Not touching the cup during the approach fell from 0.982 to 0.258 over the round, and the open-pose condition fell from a mid-round 0.371 to 0.063 by the end. Both are lower than the previous round finished at. The episode-level latch stays at 0.880 through all of this because it only needs one step, so the funnel never shows it. The new pre-grasp term introduced this round pays only 0.0009, which is too little to hold the shape against the grasp income waiting on the other side.

**Stage-1 income decays as the round goes on.** Travel income fell from 0.022 at epoch 1800 to 0.007 at the end, and the form term from 0.022 to 0.012. The approach still latches, so this is not yet breaking anything, but the trend is the same one that preceded the previous round's approach collapse.

**The round could not end by itself.** The two exit conditions — mean success at or above 4.0 and tolerance at or below 0.03 — moved in opposite directions: as the tolerance curriculum tightened, success fell, and the pair was never satisfied at the same time. The round ran 6.47 hours against a 4.0 hour cap because the success branch in the round policy returns before the hour check is reached. This is a loop-policy problem, not a reward problem, but it cost two hours of GPU time and needs to be recorded.

One note for the record. Tolerance tightened from 0.1125 to 0.0273 over the round, a 76 percent reduction, and the far-start success rate rose across the same span. Falls in the success count late in the round track the tightening criterion, not a regression in behaviour: grasp quality, lift height, airborne fraction and palm contact at success all rose while the success count fell.

To make the code more accurate and train better robot, the feedback for improvement is:
Five things went right and must be kept exactly as they are.

- The approach from the raised start works and does not collapse. Far-start episodes complete it 0.88 of the time and have done so for two thousand epochs.
- The cup is carried upright. Mean tilt is 5.45 degrees against 10.44 in the previous round, and the recording shows the cup close to vertical in the hand. Making uprightness multiply the carrying income instead of sitting beside it is what did this. Keep it.
- The grasp is a true envelope grasp: palm against the side of the cup, four fingers wrapped, thumb opposing, palm in contact 0.77 of the time at the moment of success.
- The value now sits at the end of the task. Goal proximity, goal closeness and the settling term all earn real income where previously the settling term never left 0.0001. This was the whole point of the last revision and it worked.
- The hand never rests on the table, and the wrap tolerance stopped at its ceiling instead of running away.

The rest is what has to change.

**Finishing once must be worth more than lifting repeatedly.** This is the main remaining failure. The recording shows the cup lifted, set back down, and picked up again inside a single episode. The numbers agree: the cup is airborne 0.69 of the time but the settling term earns only 0.068, so episodes spend their length holding and re-approaching rather than completing. Carrying the cup to the goal and coming to rest there should be worth clearly more than the same time spent lifting it over and over. Income that accrues simply from having the cup off the table should saturate early and stop growing; what remains should be paid for arriving, slowing, and staying until the episode ends. Putting the cup down after having reached the goal should not be a way to start earning the lift income again.

**Holding the pre-grasp shape must pay enough to survive the latch.** The approach gate is satisfied for one step and then abandoned: not touching the cup during the approach fell from 0.98 to 0.26 across the round and the open-pose condition ended at 0.06, both worse than the previous round. The term added last round to hold that shape earns 0.0009, which is nothing against the grasp income waiting on the other side, so the policy pays the early-contact penalty and dives. Maintaining the open hand, the palm facing the cup, and no digit contact from the moment the approach completes until the palm is actually against the cup has to be worth enough that breaking it early is a loss. The intended order is approach, then close from the approached pose.

**Travelling in from far away must not get cheaper as the policy improves.** Income for travelling fell from 0.022 to 0.007 and the form term from 0.022 to 0.012 over the round. Nothing is broken yet because the latch still fires, but this is the same decay that preceded the approach collapse two rounds ago. Whatever pays for closing the distance in the correct shape should keep paying at the same rate late in training as early.

One caution about reading this round. The tolerance tightened by 76 percent while the round ran, so the success count falling late in the round is the criterion moving, not the behaviour getting worse — grasp quality, lift height, airborne fraction and palm contact at success all rose over the same span. Do not respond to that fall by making success easier to obtain.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.02, 0.399, 0.595, 0.508, 0.597, 0.618, 0.673, 0.584, 0.67]  min 0 · mean 0.477 · max 0.807
contact/finger_index_at_success: [-1, 0.54, 0.873, 0.818, 0.98, 0.985, 1, 0.999, 0.997, 0.993]  min -1 · mean 0.824 · max 1
contact/finger_middle: [0, 0.019, 0.385, 0.599, 0.449, 0.492, 0.502, 0.518, 0.4, 0.489]  min 0 · mean 0.378 · max 0.683
contact/finger_middle_at_success: [-1, 0.926, 0.87, 0.842, 0.864, 0.815, 0.863, 0.802, 0.802, 0.833]  min -1 · mean 0.739 · max 1
contact/finger_pinky: [0, 0.0232, 0.169, 0.642, 0.497, 0.585, 0.534, 0.598, 0.548, 0.645]  min 0 · mean 0.433 · max 0.799
contact/finger_pinky_at_success: [-1, 0.224, 0.188, 0.861, 0.979, 0.989, 0.974, 0.996, 0.974, 0.994]  min -1 · mean 0.704 · max 1
contact/finger_ring: [0, 0.0195, 0.362, 0.655, 0.478, 0.602, 0.607, 0.666, 0.585, 0.668]  min 0 · mean 0.476 · max 0.811
contact/finger_ring_at_success: [-1, 0.828, 0.913, 0.922, 0.929, 0.977, 0.992, 0.999, 0.998, 0.998]  min -1 · mean 0.863 · max 1
contact/finger_thumb: [0, 0.0212, 0.467, 0.696, 0.527, 0.63, 0.627, 0.68, 0.6, 0.684]  min 0 · mean 0.504 · max 0.82
contact/finger_thumb_at_success: [-1, 1, 0.968, 0.977, 0.995, 0.998, 1, 1, 1, 1]  min -1 · mean 0.91 · max 1
contact/fingers_touching: [0, 0.103, 1.78, 3.19, 2.46, 2.91, 2.89, 3.13, 2.72, 3.16]  min 0 · mean 2.27 · max 3.8
contact/fingers_touching_at_success: [-1, 3.52, 3.81, 4.42, 4.75, 4.76, 4.83, 4.79, 4.77, 4.82]  min -1 · mean 4.21 · max 4.88
contact/link_force_mean: [0, 0.0809, 1.46, 2.26, 1.65, 2.12, 2.17, 2.24, 1.9, 2.26]  min 0 · mean 3.08 · max 5.85e+03
contact/links_touching: [0, 0.118, 2.11, 4.13, 3.1, 3.69, 3.58, 3.91, 3.4, 4.01]  min 0 · mean 2.88 · max 4.95
contact/links_touching_at_success: [-1, 4.64, 4.78, 5.87, 5.9, 6.2, 6.04, 5.92, 6.02, 6.12]  min -1 · mean 5.42 · max 6.58
contact/palm_touching: [0, 0.00537, 0.309, 0.473, 0.428, 0.413, 0.425, 0.418, 0.456, 0.502]  min 0 · mean 0.351 · max 0.586
contact/palm_touching_at_success: [-1, 0.367, 0.874, 0.776, 0.758, 0.65, 0.752, 0.582, 0.76, 0.722]  min -1 · mean 0.611 · max 0.999
ctrl/prev_ep_successes_mean: [0, 0.000244, 0.00269, 0.696, 3.69, 3.44, 3.99, 4, 4.35, 3.95]  min 0 · mean 2.58 · max 4.55
done/abnormal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.3e-05 · max 0.00195
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.53e-06 · max 0.000488
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.65e-06 · max 0.000488
done/max_goals: [0, 0, 0, 0.000488, 0.000977, 0.00195, 0.00146, 0.00122, 0.00171, 0.000732]  min 0 · mean 0.000976 · max 0.00415
done/out_xy: [0, 0, 0.000244, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.56e-05 · max 0.000977
done/tipped: [0, 0, 0.000244, 0.000977, 0.000244, 0.000244, 0.000488, 0, 0.000244, 0.000244]  min 0 · mean 0.00037 · max 0.00464
episode_lengths/step: [43.5, 654, 552, 700, 467, 517, 473, 477, 471, 617]  min 43.5 · mean 551 · max 899
reward/action_rate_pen: [-0.00268, -0.00204, -0.0016, -0.00138, -0.000995, -0.000844, -0.000694, -0.000634, -0.000515, -0.000483]  min -0.00268 · mean -0.00109 · max -0.000462
reward/approach_form: [0.0177, 0.0315, 0.0205, 0.0176, 0.024, 0.0203, 0.0225, 0.0205, 0.0171, 0.0155]  min 0.00725 · mean 0.0218 · max 0.0682
reward/approach_keep: [0.000838, 0.000651, 0.00563, 0.00451, 0.00638, 0.00417, 0.00327, 0.00165, 0.00161, 0.00173]  min 1.67e-05 · mean 0.00304 · max 0.0151
reward/approach_lock: [0, 1.69e-13, 0.00209, 0.00128, 0.00227, 0.00141, 0.00218, 0.00153, 0.00154, 0.00107]  min 0 · mean 0.0014 · max 0.00624
reward/approach_travel: [0.0135, 0.0127, 0.0221, 0.0178, 0.0249, 0.0189, 0.0162, 0.011, 0.01, 0.00982]  min 0.00403 · mean 0.0155 · max 0.0521
reward/arm_vel_pen: [-0.000144, -0.000257, -0.00162, -0.00147, -0.00113, -0.000996, -0.001, -0.000903, -0.000884, -0.00105]  min -0.00251 · mean -0.000995 · max -0.000136
reward/cup_disturb_pen: [0, -0.00116, -0.00682, -0.0103, -0.00435, -0.00591, -0.0119, -0.00947, -0.00203, -0.00375]  min -0.0259 · mean -0.00498 · max 0
reward/cup_tilt_pen: [0, -0.00578, -0.0353, -0.0551, -0.0154, -0.0249, -0.0209, -0.0223, -0.00811, -0.00994]  min -0.067 · mean -0.0172 · max 0
reward/early_contact_pen: [0, -0.00549, -0.00969, -0.021, -0.00808, -0.0197, -0.0247, -0.0292, -0.00708, -0.0144]  min -0.0489 · mean -0.0118 · max 0
reward/envelope_quality: [0, 0.000406, 0.0351, 0.0647, 0.0917, 0.0856, 0.0859, 0.0878, 0.0951, 0.105]  min 0 · mean 0.0706 · max 0.134
reward/finger_curl: [0, 0.00178, 0.024, 0.0351, 0.0442, 0.0377, 0.0404, 0.045, 0.0423, 0.0508]  min 0 · mean 0.0349 · max 0.0671
reward/goal_close: [0, 2.9e-05, 0.00443, 0.0164, 0.0763, 0.0938, 0.113, 0.172, 0.262, 0.328]  min 0 · mean 0.13 · max 0.448
reward/goal_in_tol: [0, 0, 0.000241, 0.00219, 0.0473, 0.0395, 0.0462, 0.0574, 0.0948, 0.0873]  min 0 · mean 0.0436 · max 0.124
reward/goal_near: [0, 8.6e-08, 0.00029, 0.0038, 0.0627, 0.093, 0.132, 0.247, 0.394, 0.511]  min 0 · mean 0.182 · max 0.712
reward/grasp_fingers: [0, 0.000633, 0.0243, 0.0431, 0.0532, 0.0551, 0.0493, 0.0562, 0.0556, 0.0639]  min 0 · mean 0.0438 · max 0.0838
reward/grasp_pose: [0, 0.01, 0.109, 0.114, 0.15, 0.12, 0.135, 0.148, 0.147, 0.165]  min 0 · mean 0.118 · max 0.207
reward/grasp_thumb: [0, 0.000554, 0.0258, 0.0429, 0.0533, 0.0553, 0.0495, 0.0563, 0.0561, 0.0641]  min 0 · mean 0.0441 · max 0.0836
reward/grasp_wrap: [0, 0.000657, 0.0346, 0.0621, 0.0756, 0.0736, 0.0635, 0.0686, 0.0705, 0.0853]  min 0 · mean 0.0586 · max 0.115
reward/grip_squeeze: [0, 0.000105, 0.00548, 0.0098, 0.0115, 0.0125, 0.0113, 0.0129, 0.0125, 0.0147]  min 0 · mean 0.01 · max 0.0193
reward/hold_still: [0, 2.13e-08, 1.1e-05, 0.00027, 0.00453, 0.00897, 0.013, 0.0262, 0.0359, 0.0543]  min 0 · mean 0.0181 · max 0.0831
reward/lift_clear: [0, 1.56e-05, 0.00729, 0.0275, 0.0652, 0.0599, 0.0618, 0.0692, 0.0953, 0.105]  min 0 · mean 0.0567 · max 0.132
reward/lift_hold: [0, 0.000387, 0.0187, 0.0322, 0.034, 0.0323, 0.033, 0.0339, 0.0396, 0.0422]  min 0 · mean 0.0283 · max 0.0505
reward/light_contact: [0, 0.000595, 0.00958, 0.0161, 0.0189, 0.0204, 0.0175, 0.0208, 0.02, 0.0232]  min 0 · mean 0.0161 · max 0.0316
reward/palm_contact: [0, 0.000493, 0.0324, 0.0568, 0.0833, 0.0689, 0.0772, 0.0778, 0.0834, 0.0945]  min 0 · mean 0.0621 · max 0.12
reward/palm_reach: [0, 0.00296, 0.0567, 0.0635, 0.0845, 0.0683, 0.0753, 0.084, 0.0828, 0.0946]  min 0 · mean 0.0659 · max 0.119
reward/palm_table_pen: [0, -3.85e-08, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0404 · mean -0.000242 · max 0
reward/pinky_join: [0, 0.000149, 0.00439, 0.0126, 0.0158, 0.0148, 0.0133, 0.0146, 0.0153, 0.0175]  min 0 · mean 0.0118 · max 0.0238
reward/pregrasp_shape: [0, 0.000446, 0.00285, 0.00177, 0.00223, 0.00138, 0.00171, 0.00126, 0.00147, 0.00092]  min 0 · mean 0.00141 · max 0.0138
reward/success_bonus: [0, 0, 0, 0.0256, 0.114, 0.0696, 0.0952, 0.0916, 0.11, 0.0769]  min 0 · mean 0.0717 · max 0.209
reward/table_pen: [0, -0.00524, -0.0195, -0.000512, -0.00141, -3.66e-05, -0.000171, -0.000789, -2.93e-05, -0.000131]  min -0.0898 · mean -0.00256 · max 0
reward/thumb_curl: [0, 0.000568, 0.0109, 0.0129, 0.0162, 0.014, 0.0149, 0.0172, 0.0165, 0.0194]  min 0 · mean 0.0133 · max 0.0254
reward/total: [0.0293, 0.0446, 0.382, 0.595, 1.13, 1.02, 1.11, 1.36, 1.74, 2]  min -0.0698 · mean 1.08 · max 2.61
rewards/step: [1.36, 10.3, 168, 333, 525, 594, 469, 681, 847, 1.2e+03]  min -11 · mean 564 · max 1.82e+03
task/lifted_frac: [0, 0.000732, 0.0945, 0.469, 0.413, 0.529, 0.543, 0.618, 0.524, 0.63]  min 0 · mean 0.391 · max 0.787
task/successes_mean: [0, 0, 0.00171, 0.183, 0.513, 0.599, 0.576, 0.439, 0.446, 0.657]  min 0 · mean 0.381 · max 1.01
task/tilt_deg: [0.00108, 1.39, 8.9, 13.2, 6.33, 8.32, 7.74, 8.18, 5.34, 6.06]  min 0.000831 · mean 6.22 · max 15.6
task/tol: [0.112, 0.112, 0.112, 0.112, 0.101, 0.082, 0.0664, 0.0484, 0.0392, 0.0318]  min 0.0257 · mean 0.0774 · max 0.112

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
