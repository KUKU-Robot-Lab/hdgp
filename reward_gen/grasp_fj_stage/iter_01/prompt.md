You are an expert in robotics, reinforcement learning and code generation.

We control one 7-DOF OpenArm robot arm (the right arm) carrying a five-finger Tesollo DG-5F hand, standing at a table. +x points from the robot toward the table. Each episode starts with the arm raised beside the robot: the palm is just outside the table edge nearest the robot, about 0.25 m above the table top, turned sideways, with the fingers pointing forward over the table edge; the palm is roughly 0.38 m from the cup. A cup stands upright on the table; parallel environments use different cups (open cups of several sizes and a closed shaker), so the graspable radius (44 mm to 81 mm) and half height (42 mm to 65 mm) differ between environments and are given per environment. Positions are in metres in each environment's local frame, with +z pointing up; the table top is at z = table_z.

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
    hand_default_q_norm: torch.Tensor  # (N,19) the hand's default pose: the finger joint angles every episode starts with, same normalisation as hand_q_norm (the same in every environment)
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
    approach_done: torch.Tensor    # (N,) bool, set on the first step on which all of these held at once: the palm centre was within 2 cm of the surface of the cup's graspable band; the palm faced the cup axis (palm_normal within about 45 degrees of the direction from the palm to the axis); the palm was on the cup's -y side (the direction from the cup axis to the palm centre, perpendicular to the axis, within about 45 degrees of -y); the hand kept its start-pose orientation (palm_normal within about 45 degrees of +y and palm_finger_dir within about 45 degrees of +x); and every movable finger joint was within 0.15 of hand_default_q_norm (joints with a range of 0.05 rad or less are ignored)
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
Stage 1, approach: starting from the raised pose beside the table, move the hand to the cup's -y side without turning it. The hand already has the right orientation at the start: keep that orientation the whole way (ctx.palm_normal along +y, towards the cup, and ctx.palm_finger_dir along +x, as in the start pose), so the approach is a movement of the palm towards the cup, not a rotation of the hand. Keep the hand's default pose as well (ctx.hand_default_q_norm): the fingers and the thumb must not change shape and must not touch the cup during the approach. The approach is complete when the palm centre is at the cup's graspable band on the cup's -y side, with the palm facing the cup and the hand still in its start-pose orientation; the environment then sets ctx.approach_done for the rest of the episode.
Stage 2, envelope grasp: only after the approach is complete, at that position and with that hand orientation, close the hand into an envelope grasp (a power grasp): the palm against the side of the cup and all five fingers wrapped around the cup body with their inner surfaces, the thumb on the opposite side of the cup from the other four fingers. The environment sets ctx.envelope_done once the palm, the thumb and enough fingers hold the cup for several consecutive steps.
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

# Movable finger joints (range > 0.05 rad), in ctx.hand_q order.
# Locked: 0 thumb_2, 3 index_1, 7 middle_1, 11 ring_1, 15 pinky_1, 16 pinky_2.
MOVABLE_HAND_IDX = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18]
CONTACT_FORCE_SCALE = 0.3  # N; 0.1 N (env contact threshold) -> 0.28, 1 N -> 0.96


def _contact_level(force: torch.Tensor) -> torch.Tensor:
    """Bounded, spike-robust contact level in [0, 1)."""
    return 1.0 - torch.exp(-force.clamp(min=0.0) / CONTACT_FORCE_SCALE)


def _cylinder_coords(p: torch.Tensor, cup_pos: torch.Tensor, cup_axis: torch.Tensor):
    """Axial coordinate h, radial vector and radial distance of point(s) p w.r.t. the cup axis."""
    v = p - cup_pos
    h = (v * cup_axis).sum(-1)
    radial_vec = v - h.unsqueeze(-1) * cup_axis
    radial = radial_vec.norm(dim=-1)
    return h, radial_vec, radial


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.palm_pos.device
    dtype = ctx.palm_pos.dtype
    R = ctx.cup_radius
    hh = ctx.cup_half_height
    idx = torch.tensor(MOVABLE_HAND_IDX, device=device, dtype=torch.long)

    # ---------------- stage masks (strict order kept by the env's latched flags) ----------------
    s1 = (~ctx.approach_done).to(dtype)
    s2 = (ctx.approach_done & ~ctx.envelope_done).to(dtype)
    s3 = ctx.envelope_done.to(dtype)

    # ---------------- palm geometry relative to the cup ----------------
    palm_h, palm_rvec, palm_r = _cylinder_coords(ctx.palm_pos, ctx.cup_pos, ctx.cup_axis)
    palm_r_safe = palm_r.clamp(min=1e-6)
    palm_u = palm_rvec / palm_r_safe.unsqueeze(-1)            # outward radial unit vector at the palm
    face_align = (ctx.palm_normal * (-palm_u)).sum(-1).clamp(0.0, 1.0)  # palm normal toward the axis

    # Distance of the palm centre to its target: the band surface (radius R + 5 mm, 1 cm dead band),
    # near the band centre (|h| <= 0.25 * half height). Well inside the env's 2 cm approach test.
    radial_err = torch.relu((palm_r - R - 0.005).abs() - 0.01)
    height_err = torch.relu(palm_h.abs() - 0.25 * hh)
    d_palm = torch.sqrt(radial_err ** 2 + height_err ** 2 + 1e-12)

    # ---------------- finger link geometry and contacts ----------------
    cup_pos_l = ctx.cup_pos[:, None, None, :]
    axis_l = ctx.cup_axis[:, None, None, :]
    link_h, _, link_r = _cylinder_coords(ctx.link_pos, cup_pos_l, axis_l)          # (N,5,3)
    link_c = _contact_level(ctx.link_cup_force)                                      # (N,5,3)
    digit_c = link_c.amax(-1)                                                        # (N,5)
    palm_c = _contact_level(ctx.palm_cup_force)                                      # (N,)
    digit_w = torch.tensor([0.3, 0.175, 0.175, 0.175, 0.175], device=device, dtype=dtype)  # thumb weighted most

    # ---------------- cup state ----------------
    raise_h = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)

    # =========================== STAGE 1: approach ===========================
    # Coarse term gives gradient from the 0.38 m start, fine term sharpens the last centimetres.
    reach = 0.5 * (1.0 - torch.tanh(d_palm / 0.25)) + 0.5 * torch.exp(-d_palm / 0.03)
    # Hand must keep its default shape: worst movable-joint deviation of both actual and commanded angles.
    dev_q = (ctx.hand_q_norm[:, idx] - ctx.hand_default_q_norm[:, idx]).abs()
    dev_t = (ctx.hand_target_norm[:, idx] - ctx.hand_default_q_norm[:, idx]).abs()
    pose_dev = torch.maximum(dev_q, dev_t).amax(-1)
    pose_keep = 1.0 - torch.tanh(torch.relu(pose_dev - 0.03) / 0.06)  # ~0.04 at the env's 0.15 limit
    finger_touch = torch.tanh(link_c.sum(dim=(1, 2)))                  # fingers/thumb must not touch yet

    approach_reach = 1.0 * reach * s1            # max 1.0
    approach_face = 0.5 * face_align * s1        # max 0.5
    approach_pose_keep = 0.5 * pose_keep * s1    # max 0.5  -> stage 1 total <= 2.0
    approach_finger_contact = -0.5 * finger_touch * s1

    # Stages 1-2: the cup must stay where it stands (no pushing, no early lifting).
    disturb = 0.5 * torch.tanh(torch.relu(disp_xy - 0.005) / 0.05) \
        + 0.5 * torch.tanh(torch.relu(raise_h - 0.005) / 0.02)
    cup_disturb = -1.0 * disturb * (s1 + s2)

    # =========================== STAGE 2: envelope grasp ===========================
    stay_gate = torch.exp(-d_palm / 0.03)        # all grasp shaping only counts at the approach position
    envelope_stay = 0.75 * torch.exp(-d_palm / 0.02) * s2
    envelope_face = 0.25 * face_align * stay_gate * s2

    # Closing progress (a finger stops on the cup, so this only drives closing, not crushing).
    closure = ctx.hand_q_norm[:, idx].mean(-1)
    envelope_closure = 0.25 * closure * stay_gate * s2

    # Wrap: finger links on the cup surface, inside the graspable band, on the cup side of the palm plane.
    near_surface = torch.exp(-torch.relu(link_r - R[:, None, None]) / 0.015)
    in_band = torch.exp(-torch.relu(link_h.abs() - hh[:, None, None]) / 0.01)
    rel_palm = ctx.link_pos - ctx.palm_pos[:, None, None, :]
    cup_side = torch.sigmoid((rel_palm * ctx.palm_normal[:, None, None, :]).sum(-1) / 0.01)
    link_wrap = near_surface * in_band * cup_side                      # (N,5,3)
    digit_wrap = link_wrap.mean(-1)                                    # (N,5)
    envelope_wrap = 0.75 * (digit_wrap * digit_w).sum(-1) * stay_gate * s2

    # Contacts: palm against the cup, thumb and all four fingers touching it.
    envelope_palm_contact = 0.5 * palm_c * stay_gate * s2
    envelope_digit_contact = 1.0 * (digit_c * digit_w).sum(-1) * stay_gate * s2

    # Opposition: thumb and the four fingers on opposite sides of the plane through the cup axis and palm.
    tangent = torch.cross(ctx.cup_axis, palm_u, dim=-1)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    link_s = ((ctx.link_pos - cup_pos_l) * tangent[:, None, None, :]).sum(-1)       # (N,5,3)
    digit_s = link_s.mean(-1) / R.unsqueeze(-1).clamp(min=1e-3)                      # (N,5), radius-normalised
    opposition = torch.tanh(-4.0 * digit_s[:, 0] * digit_s[:, 1:].mean(-1)).clamp(min=0.0)
    opp_gate = torch.minimum(digit_wrap[:, 0], digit_wrap[:, 1:].mean(-1))            # only counts on the cup
    envelope_opposition = 0.25 * opposition * opp_gate * stay_gate * s2
    # stage 2 shaping total <= 3.75

    # =========================== STAGE 3: lift and hold ===========================
    others_c = (digit_c[:, 1:].sum(-1) / 3.0).clamp(max=1.0)          # >= 3 fingers besides the thumb
    grip_quality = (palm_c + digit_c[:, 0] + others_c) / 3.0
    grip_gate = (palm_c * digit_c[:, 0] * others_c).clamp(min=0.0).pow(1.0 / 3.0)  # zero if palm/thumb/fingers let go

    lift_grip_hold = 1.0 * grip_quality * s3

    goal_pos_dist = (ctx.cup_pos - ctx.goal_pos).norm(dim=-1)
    goal_progress = 0.5 * (1.0 - torch.tanh(goal_pos_dist / 0.2)) + 0.5 * torch.exp(-ctx.goal_dist / 0.02)
    lift_goal = 3.0 * grip_gate * goal_progress * s3                  # main task term, max 3.0

    near_goal = torch.exp(-ctx.goal_dist / 0.03)
    stillness = torch.exp(-ctx.cup_lin_vel.norm(dim=-1) / 0.1 - ctx.cup_ang_vel.norm(dim=-1) / 1.0)
    lift_hold_still = 1.0 * grip_gate * near_goal * stillness * s3

    # Lift straight up: sideways deviation from the spawn -> goal line at the current height.
    goal_rise = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=1e-3)
    rise_frac = (raise_h / goal_rise).clamp(0.0, 1.0)
    line_xy = ctx.cup_spawn_pos[:, :2] + rise_frac.unsqueeze(-1) * (ctx.goal_pos[:, :2] - ctx.cup_spawn_pos[:, :2])
    lateral_dev = (ctx.cup_pos[:, :2] - line_xy).norm(dim=-1)
    lift_path_dev = -0.5 * torch.tanh(torch.relu(lateral_dev - 0.01) / 0.03) * s3

    # Success is only reachable by a held cup; paid only after the envelope stage.
    success_bonus = 10.0 * ctx.success.to(dtype) * s3

    # =========================== stage offsets ===========================
    # Stage 2 base (2.5) > stage 1 max (2.0); stage 3 base (6.5) > stage 2 max (2.5 + 3.75 = 6.25):
    # completing a stage never lowers reward, and higher levels only come through the env's ordered flags.
    stage_base = 2.5 * s2 + 6.5 * s3

    # =========================== always-on constraints ===========================
    cup_tilt = -1.0 * torch.tanh(torch.relu(ctx.cup_tilt - 0.05) / 0.3)             # keep the cup upright
    table_clearance = -1.0 * torch.tanh(torch.relu(ctx.table_z + 0.005 - ctx.hand_z_min) / 0.01)  # no hand in table
    arm_rate = ((ctx.actions[:, :7] - ctx.prev_actions[:, :7]) ** 2).sum(-1)
    hand_rate = ((ctx.actions[:, 7:] - ctx.prev_actions[:, 7:]) ** 2).sum(-1)
    action_rate = -(0.01 * arm_rate + 0.002 * hand_rate)                              # small smoothness prior

    components = {
        "stage_base": stage_base,
        "approach_reach": approach_reach,
        "approach_face": approach_face,
        "approach_pose_keep": approach_pose_keep,
        "approach_finger_contact": approach_finger_contact,
        "cup_disturb": cup_disturb,
        "envelope_stay": envelope_stay,
        "envelope_face": envelope_face,
        "envelope_closure": envelope_closure,
        "envelope_wrap": envelope_wrap,
        "envelope_palm_contact": envelope_palm_contact,
        "envelope_digit_contact": envelope_digit_contact,
        "envelope_opposition": envelope_opposition,
        "lift_grip_hold": lift_grip_hold,
        "lift_goal": lift_goal,
        "lift_hold_still": lift_hold_still,
        "lift_path_dev": lift_path_dev,
        "success_bonus": success_bonus,
        "cup_tilt": cup_tilt,
        "table_clearance": table_clearance,
        "action_rate": action_rate,
    }
    reward = torch.zeros_like(ctx.cup_tilt)
    for value in components.values():
        reward = reward + value
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Training was stopped early, after about 180 of 20,000 epochs, because the hand was not approaching the cup; the playback uses the best checkpoint of that run.
In the viewed environment the hand never moves towards the cup. Within the first 1.5 s the arm lowers the hand beside the robot column, next to the table edge closest to the robot, and turns the hand away from its start orientation: the fingers no longer point forward over the table (+x) but down towards the table, and the wrist keeps rotating there for the rest of the episode. At the end of the episode the hand is still next to the robot, far from the cup.
The fingers and the thumb curl in from the first second, so the hand does not keep its default pose during what should be the approach.
The cup is never touched and stays upright.
Metrics:
- The approach flag (ctx.approach_done) was set in 0 of the finished episodes up to epoch 110.
- The palm-to-band distance averaged 0.26 m at epoch 1, 0.28 m at epoch 17, 0.30 m at epoch 55 and 0.23 m at epoch 110. In 56 % of the finished episodes the palm centre came within 5 cm of the band at some step, but none of them met the approach flag.
- approach_face rose from 0.19 at epoch 1 to 0.36 at epoch 17, while approach_reach fell from 0.111 to 0.098 over the same epochs; at epoch 110 they were 0.37 and 0.16.
- approach_pose_keep was 0.000 at every logged epoch.
- In the start pose the palm normal points along +y and the fingers along +x; the horizontal direction from the palm centre to the cup axis is about 68 degrees away from the palm normal there, so approach_face paid about 0.19 before the hand moved.

To make the code more accurate and train better robot, the feedback for improvement is:
The approach went wrong from the first steps: the policy turned the hand instead of moving it to the cup.
- The hand already starts with the orientation it needs for the grasp: palm normal along +y towards the cup's side and fingers along +x. The approach should be a movement of the palm to the cup's -y side while this start orientation is kept the whole way. The approach reward asked the palm to face the cup axis from wherever the hand was; from the start pose that means turning the wrist by about 68 degrees, and the policy learned that turn first while the hand moved away from the cup. Reward keeping the start orientation (ctx.palm_normal along +y, ctx.palm_finger_dir along +x) during the approach rather than turning the palm towards the cup axis from a distance.
- The approach target was the nearest point of the cup's band, which from the start pose lies on a diagonal side of the cup. The target should be the cup's -y side of the band. The environment's approach flag now also requires the palm on the cup's -y side and the start orientation of the hand.
- The default hand pose was never kept: approach_pose_keep gave no reward at any point of training, and the fingers curled from the first second. Keeping the default pose needs a signal that changes over the finger deviations the policy actually produces, instead of one that is already at zero as soon as any finger joint, actual or commanded, moves a little.
- Getting closer to the cup earned less than turning the palm (approach_reach 0.10 to 0.16 against approach_face 0.19 to 0.37), so moving the palm to the cup's side must be the part of the approach that pays most.
- Stages 2 and 3 were never reached, so there is no feedback on them yet; keep their order and their dependence on the stage flags.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.28e-05 · max 0.000488
contact/finger_index_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_middle: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000244]  min 0 · mean 6.42e-06 · max 0.000244
contact/finger_middle_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_pinky: [0, 0, 0, 0, 0, 0, 0, 0.0479, 0.00122, 0.00244]  min 0 · mean 0.00509 · max 0.062
contact/finger_pinky_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_ring: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000488]  min 0 · mean 0.000108 · max 0.00171
contact/finger_ring_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_thumb: [0, 0, 0, 0, 0, 0, 0, 0.000244, 0, 0.00122]  min 0 · mean 0.000278 · max 0.00391
contact/finger_thumb_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/fingers_touching: [0, 0, 0, 0, 0, 0, 0, 0.0481, 0.00122, 0.00439]  min 0 · mean 0.0055 · max 0.0625
contact/fingers_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/link_force_mean: [0, 0, 0, 0, 0, 0, 0, 0.0165, 0.002, 0.025]  min 0 · mean 0.00248 · max 0.0276
contact/links_touching: [0, 0, 0, 0, 0, 0, 0, 0.0491, 0.00122, 0.00464]  min 0 · mean 0.00567 · max 0.064
contact/links_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/palm_touching: [0, 0, 0, 0, 0, 0, 0, 0.00146, 0, 0]  min 0 · mean 0.000149 · max 0.00391
contact/palm_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/abnormal: [0, 0.000244, 0.000488, 0, 0, 0, 0, 0, 0.000488, 0.000488]  min 0 · mean 0.000315 · max 0.00317
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/hand_floor: [0, 0, 0, 0.000732, 0, 0, 0.000244, 0, 0.000244, 0]  min 0 · mean 0.000104 · max 0.00122
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.61e-06 · max 0.000244
done/tipped: [0, 0, 0, 0, 0, 0, 0, 0.00586, 0, 0.000244]  min 0 · mean 0.000397 · max 0.00659
episode_lengths/step: [61, 184, 294, 530, 899, 890, 878, 839, 817, 707]  min 61 · mean 627 · max 899
reward/action_rate: [-0.111, -0.109, -0.102, -0.0991, -0.105, -0.0901, -0.0922, -0.0863, -0.0829, -0.0802]  min -0.112 · mean -0.0936 · max -0.075
reward/approach_face: [0.189, 0.34, 0.487, 0.228, 0.23, 0.472, 0.278, 0.269, 0.478, 0.432]  min 0.0597 · mean 0.349 · max 0.491
reward/approach_finger_contact: [0, 0, 0, 0, 0, 0, 0, -0.0175, -0.000453, -0.00145]  min -0.0222 · mean -0.00197 · max 0
reward/approach_pose_keep: [4.17e-08, 2.09e-07, 1.14e-08, 0.000117, 6.37e-09, 2.72e-05, 5.35e-05, 0.00041, 0.000114, 0.000146]  min 7.28e-12 · mean 8.49e-05 · max 0.000717
reward/approach_reach: [0.111, 0.1, 0.0609, 0.0627, 0.0937, 0.0611, 0.136, 0.227, 0.158, 0.216]  min 0.0549 · mean 0.136 · max 0.324
reward/cup_disturb: [0, 0, 0, 0, 0, -9.52e-05, -7.79e-05, -0.0429, -0.000682, -0.00334]  min -0.0565 · mean -0.00453 · max 0
reward/cup_tilt: [0, 0, 0, 0, 0, -0.000232, -0.000248, -0.113, -0.000871, -0.00671]  min -0.158 · mean -0.0107 · max 0
reward/envelope_closure: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_digit_contact: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_face: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_opposition: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_palm_contact: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_stay: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_wrap: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_goal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_grip_hold: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_hold_still: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_path_dev: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/stage_base: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/table_clearance: [0, 0, 0, -0.0358, 0, -0.000233, -0.00476, -0.000563, -0.00222, -0.00215]  min -0.0476 · mean -0.00491 · max 0
reward/total: [0.189, 0.331, 0.446, 0.156, 0.219, 0.443, 0.316, 0.236, 0.549, 0.554]  min 0.0293 · mean 0.369 · max 0.605
rewards/step: [7.69, 37.8, 77.5, 157, 251, 248, 246, 243, 251, 235]  min 7.69 · mean 188 · max 260
task/lifted_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.12e-05 · max 0.000488
task/successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/tilt_deg: [0.00108, 0.00106, 0.00118, 0.00108, 0.00111, 0.0094, 0.00753, 4.1, 0.029, 0.25]  min 0.000912 · mean 0.377 · max 5.45
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
