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
    success_hold_steps: int        # consecutive steps with goal_dist <= success_tol needed to count one success
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
    hand_z_min: torch.Tensor       # (N,) height of the lowest hand link, palm excluded [m]; below table_z - 0.03 ends the episode wherever the hand is

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
7. Goal and success are computed by the environment and cannot be redefined by the reward. The goal position is 0.21 m to 0.28 m above the cup's starting position and at most 5 cm away from it horizontally, with the cup upright. A success is counted when `ctx.goal_dist <= ctx.success_tol` for `ctx.success_hold_steps` consecutive steps, and `ctx.success` is True on that step. After a success the next goal is at the same place, so holding the cup still there keeps producing successes until `ctx.max_successes`, which ends the episode. The keypoints are fixed on the cup, so tilting the cup also increases `goal_dist`. `ctx.success_tol` starts at 0.1125 m and shrinks towards 0.015 m as the policy succeeds more often during training. You may add a bonus on `ctx.success`.
8. An episode lasts at most 900 steps (15 s), and the step budget restarts after every success. The episode ends early when the cup falls below z = 0.15 (off the table), leaves the allowed area around the table, or tilts more than 60 degrees; when any hand link goes below z = table_z - 0.03 (wherever the hand is); or when an arm joint goes past its limit or moves faster than 20 rad/s. On that last kind of physics violation the environment replaces the reward of that step with a fixed -1 (your function's value is not used on that step).
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

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.000244, 0.196, 0.718, 0.726, 0.719, 0.665, 0.809, 0.756, 0.839]  min 0 · mean 0.59 · max 0.841
contact/finger_index_at_success: [-1, -1, -1, 0, 0.258, 0.245, 0.245, 0.18, 0.0407, 0.0407]  min -1 · mean -0.113 · max 0.258
contact/finger_middle: [0, 0.000488, 0.421, 0.689, 0.737, 0.803, 0.831, 0.827, 0.815, 0.866]  min 0 · mean 0.653 · max 0.873
contact/finger_middle_at_success: [-1, -1, -1, 0, 0.326, 0.31, 0.31, 0.454, 0.877, 0.877]  min -1 · mean 0.152 · max 0.877
contact/finger_pinky: [0, 0.00146, 0.00146, 0.000977, 0.0022, 0.000488, 0.00146, 0.000488, 0, 0]  min 0 · mean 0.00362 · max 0.17
contact/finger_pinky_at_success: [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0]  min -1 · mean -0.207 · max 0
contact/finger_ring: [0, 0.00757, 0.333, 0.756, 0.774, 0.781, 0.759, 0.817, 0.765, 0.854]  min 0 · mean 0.638 · max 0.858
contact/finger_ring_at_success: [-1, -1, -1, 0, 0.224, 0.213, 0.213, 0.157, 0.355, 0.355]  min -1 · mean -0.0408 · max 0.355
contact/finger_thumb: [0, 0.01, 0.533, 0.714, 0.786, 0.834, 0.834, 0.839, 0.811, 0.874]  min 0 · mean 0.684 · max 0.879
contact/finger_thumb_at_success: [-1, -1, -1, 0, 0.431, 0.41, 0.41, 0.527, 0.839, 0.839]  min -1 · mean 0.176 · max 0.839
contact/fingers_touching: [0, 0.0198, 1.48, 2.88, 3.03, 3.14, 3.09, 3.29, 3.15, 3.43]  min 0 · mean 2.57 · max 3.45
contact/fingers_touching_at_success: [-1, -1, -1, 0, 1.24, 1.18, 1.18, 1.32, 2.11, 2.11]  min -1 · mean 0.797 · max 2.11
contact/link_force_mean: [0, 0.0186, 0.553, 0.486, 0.658, 0.828, 0.864, 0.795, 0.832, 0.927]  min 0 · mean 0.701 · max 43.6
contact/links_touching: [0, 0.0225, 1.63, 3.05, 3.27, 3.42, 3.44, 3.67, 3.51, 3.88]  min 0 · mean 2.85 · max 3.9
contact/links_touching_at_success: [-1, -1, -1, 0, 1.36, 1.29, 1.29, 1.4, 2.46, 2.46]  min -1 · mean 0.93 · max 2.46
contact/palm_touching: [0, 0.000732, 0.0344, 0.000488, 0.000977, 0.00903, 0.011, 0.000488, 0.00269, 0.000977]  min 0 · mean 0.00843 · max 0.122
contact/palm_touching_at_success: [-1, -1, -1, 0, 0, 0, 0, 0.181, 0.803, 0.803]  min -1 · mean 0.0235 · max 0.803
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0, 0.000244, 0, 0, 0, 0]  min 0 · mean 0.000228 · max 0.00732
done/abnormal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.19e-05 · max 0.000977
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.32e-07 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.69e-07 · max 0.000244
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.76e-06 · max 0.000244
done/tipped: [0, 0.000244, 0.000244, 0, 0, 0.000244, 0, 0, 0.000244, 0]  min 0 · mean 0.000166 · max 0.00659
episode_lengths/step: [34.5, 803, 776, 843, 869, 873, 872, 885, 858, 885]  min 34.5 · mean 820 · max 886
reward/action_rate_pen: [-0.00537, -0.0029, -0.00264, -0.00189, -0.00175, -0.00158, -0.0015, -0.0013, -0.00126, -0.00112]  min -0.00537 · mean -0.00188 · max -0.00104
reward/approach_keep: [0.00382, 0.00742, 0.0104, 0.00658, 0.00512, 0.0049, 0.00432, 0.00432, 0.00472, 0.0038]  min 0.00175 · mean 0.00604 · max 0.0271
reward/approach_reach: [0.023, 0.0239, 0.0336, 0.0203, 0.0156, 0.0143, 0.0142, 0.0126, 0.0154, 0.0109]  min 0.0105 · mean 0.0186 · max 0.0862
reward/arm_vel_pen: [-0.000719, -0.000999, -0.00177, -0.00108, -0.000991, -0.0011, -0.00121, -0.00151, -0.000951, -0.000875]  min -0.00393 · mean -0.00118 · max -0.000719
reward/cup_disturb_pen: [0, -0.00114, -0.017, -0.00282, -0.00578, -0.00412, -0.00332, -0.00139, -0.00338, -0.00186]  min -0.0298 · mean -0.00515 · max 0
reward/cup_tilt_pen: [0, -0.00198, -0.0135, -0.00605, -0.00577, -0.00296, -0.00413, -0.00347, -0.00752, -0.00402]  min -0.0283 · mean -0.00613 · max 0
reward/early_contact_pen: [0, -0.00195, -0.0112, -0.00403, -0.00269, -0.00189, -0.00537, -0.000879, -0.00493, -0.00186]  min -0.0255 · mean -0.00361 · max 0
reward/envelope_quality: [0, 0, 0.000402, 0, 1.31e-05, 0.000518, 0.000675, 2.56e-05, 0.000121, 8.3e-05]  min 0 · mean 0.000285 · max 0.00537
reward/grasp_fingers: [0, 0.000427, 0.0581, 0.145, 0.17, 0.187, 0.179, 0.209, 0.189, 0.22]  min 0 · mean 0.148 · max 0.222
reward/grasp_palm_close: [0, 0.081, 0.0702, 0.116, 0.134, 0.145, 0.141, 0.153, 0.141, 0.16]  min 0 · mean 0.124 · max 0.162
reward/grasp_palm_touch: [0, 5.61e-05, 0.0036, 1.32e-05, 0.000147, 0.00143, 0.00181, 8.08e-05, 0.000324, 0.000154]  min 0 · mean 0.00127 · max 0.0206
reward/grasp_stay: [0, 0.256, 0.206, 0.294, 0.314, 0.322, 0.316, 0.331, 0.313, 0.338]  min 0 · mean 0.29 · max 0.341
reward/grasp_thumb: [0, 0.000253, 0.0657, 0.109, 0.13, 0.145, 0.142, 0.153, 0.141, 0.16]  min 0 · mean 0.114 · max 0.162
reward/grasp_wrap: [0, 0.0195, 0.0339, 0.0581, 0.0676, 0.0713, 0.0691, 0.0754, 0.0671, 0.0794]  min 0 · mean 0.0594 · max 0.08
reward/grip_squeeze: [0, 1.25e-05, 0.00821, 0.0164, 0.0191, 0.0223, 0.0221, 0.0246, 0.0232, 0.0269]  min 0 · mean 0.0179 · max 0.0278
reward/hold_still: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.95e-08]  min 0 · mean 8.5e-09 · max 2.14e-07
reward/lift_goal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.23e-05]  min 0 · mean 1.33e-05 · max 0.00039
reward/lift_grip_hold: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000146]  min 0 · mean 0.000268 · max 0.00952
reward/lift_height: [0, 0, 0, 0, 0, 0, 0, 0, 0, 8.03e-06]  min 0 · mean 1.86e-05 · max 0.000725
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/table_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.000504 · mean -3.7e-06 · max 0
reward/total: [0.0208, 0.38, 0.445, 0.749, 0.838, 0.902, 0.874, 0.954, 0.877, 0.99]  min 0.0146 · mean 0.762 · max 0.997
rewards/step: [0.73, 306, 414, 659, 717, 777, 783, 827, 748, 830]  min 0.0362 · mean 636 · max 860
task/lifted_frac: [0, 0, 0.000244, 0, 0.000244, 0.00146, 0, 0.000488, 0.000732, 0]  min 0 · mean 0.000281 · max 0.00806
task/successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.8e-06 · max 0.00195
task/tilt_deg: [0.00108, 0.625, 6.02, 4.98, 5.4, 3.85, 4.41, 4.37, 5.66, 4.9]  min 0.000912 · mean 4.66 · max 9.62
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
