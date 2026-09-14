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
    link_pos: torch.Tensor         # (N,5,3,3) finger link positions: [:, f, 0] link moved by joint _3, [:, f, 1] link moved by joint _4, [:, f, 2] fingertip
    link_cup_force: torch.Tensor   # (N,5,3) contact force magnitude between each of those links and the cup only [N] (0 = not touching the cup)
    palm_cup_force: torch.Tensor   # (N,) contact force magnitude between the palm and the cup only [N]
    hand_q: torch.Tensor           # (N,19) finger joint angles [rad], order = the hand joint table in the robot description
    hand_q_norm: torch.Tensor      # (N,19) joint angles normalised to each joint's commandable range: 0 = lower limit (straight), 1 = upper limit (most flexed)
    hand_target_norm: torch.Tensor  # (N,19) commanded finger joint targets (after filtering), same normalisation as hand_q_norm
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

I want it to fulfil the following task: Starting from the raised pose beside the table, move the hand to the cup standing upright on the table and grasp it with an envelope grasp (a power grasp): bring the palm against the side of the cup and wrap all five fingers around it, with the thumb on the opposite side of the cup from the other four fingers, so that the palm and the links of every finger are in contact with the cup. Then lift the cup to the goal position and hold it there, upright and still, so that the environment keeps counting successes. The cups differ in shape and size between environments. Do not knock the cup over, do not drop it, and do not push the hand into the table.
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


def _cylinder_coords(p, cup_pos, cup_axis):
    """Axial coordinate h, radial vector and distance to the cup axis for points p.

    p: (N, ..., 3); cup_pos, cup_axis: (N, 3).
    Returns h (N, ...), radial (N, ..., 3), r (N, ...).
    """
    shape = (cup_pos.shape[0],) + (1,) * (p.dim() - 2) + (3,)
    c = cup_pos.reshape(shape)
    a = cup_axis.reshape(shape)
    v = p - c
    h = (v * a).sum(-1)
    radial = v - h.unsqueeze(-1) * a
    r = radial.norm(dim=-1)
    return h, radial, r


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eps = 1e-6

    # ------------------------------------------------------------------ weights
    W_REACH = 1.0            # palm to the grasp spot beside the cup, bounded [0,1]
    W_ORIENT = 0.5           # palm normal horizontal and facing the cup axis
    W_PRESHAPE = 0.3         # open fingers while far, closed targets once the palm is at the cup
    W_WRAP = 1.0             # finger links lying on the cup surface inside the graspable band
    W_OPPOSITION = 0.5       # thumb on the other side of the cup from the four fingers
    W_PALM_CONTACT = 0.75    # palm touching the cup side
    W_FINGER_CONTACT = 1.5   # mean over 5 fingers of "some measured link touches"
    W_LINK_CONTACT = 0.75    # mean over all 15 measured links (how much of each finger wraps)
    W_FULL_ENVELOPE = 1.0    # palm AND all five fingers touching at once
    W_LIFT = 3.0             # lift progress; larger than all grasp terms so lifting always pays
    W_GOAL_COARSE = 2.0      # cup towards the goal (wide basin)
    W_GOAL_FINE = 2.0        # cup precisely at the goal (narrow basin, success_tol shrinks to 1.5 cm)
    W_STILL = 1.0            # cup not moving while at the goal
    W_IN_TOL = 1.0           # dense per-step version of the success condition
    W_SUCCESS = 15.0         # bonus on every counted success
    W_TABLE = 1.0            # hand pushing into the table (ramp up to the termination height)
    W_PUSH = 0.3             # shoving the cup across the table before it is lifted
    W_ACTION_RATE = 0.01     # smooth actions (finger actions are absolute targets)
    W_JOINT_VEL = 0.01       # small joint velocity regularisation

    cup_r = ctx.cup_radius
    cup_hh = ctx.cup_half_height
    axis = ctx.cup_axis
    lifted = ctx.lifted.float()

    # uprightness factor: 1 upright, ~0.89 at 20 deg, 0 at the 60 deg termination limit
    upright = torch.clamp(1.0 - (ctx.cup_tilt / (math.pi / 3.0)) ** 2, 0.0, 1.0)

    # ------------------------------------------------------------------ stage A: palm reach / orientation
    h_p, rad_p, r_p = _cylinder_coords(ctx.palm_pos, ctx.cup_pos, axis)
    u_p = rad_p / r_p.clamp(min=eps).unsqueeze(-1)            # unit radial direction: cup axis -> palm
    # radial dead band: palm origin 0..3 cm outside the cup surface (palm frame offset is not exactly known)
    e_r = torch.relu((r_p - cup_r - 0.015).abs() - 0.015)
    # axial dead band: palm between 0.2*H below and 0.5*H above the band centre (keeps lower fingers off the table)
    e_h = torch.relu(-h_p - 0.2 * cup_hh) + torch.relu(h_p - 0.5 * cup_hh)
    d_palm = torch.sqrt(e_r ** 2 + e_h ** 2 + eps)

    # coarse scale (0.25 m) gives gradient from the start pose, fine scale (4 cm) sharpens the final approach
    reach = 0.5 * (1.0 - torch.tanh(d_palm / 0.25)) + 0.5 * (1.0 - torch.tanh(d_palm / 0.04))

    n = ctx.palm_normal
    facing = -(n * u_p).sum(-1)                               # +1 when the palm normal points at the cup axis
    level = 1.0 - (n * axis).sum(-1).abs()                    # 1 when the palm normal is perpendicular to the axis
    orient = 0.5 * (facing + 1.0) * level
    palm_orient = orient * (1.0 - torch.tanh(d_palm / 0.20))  # matters more as the palm gets close

    near = torch.exp(-(d_palm / 0.03) ** 2) * facing.clamp(0.0, 1.0)  # palm at the grasp spot and facing the cup
    near_soft = torch.exp(-(d_palm / 0.06) ** 2)                      # palm roughly at the cup

    # finger pre-shape: open targets while far, closed targets once at the cup (movable joints only)
    movable = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18]
    close_cmd = ctx.hand_target_norm[:, movable].mean(-1)
    preshape = near * close_cmd + (1.0 - near_soft) * (1.0 - close_cmd)

    # ------------------------------------------------------------------ stage B: wrap, opposition, contact
    h_l, rad_l, r_l = _cylinder_coords(ctx.link_pos, ctx.cup_pos, axis)   # (N,5,3)
    R_l = cup_r.reshape(-1, 1, 1)
    H_l = cup_hh.reshape(-1, 1, 1)
    gap_l = (r_l - R_l - 0.01).abs()               # link frames sit ~1 cm (half a finger) outside the surface
    band_l = torch.relu(h_l.abs() - H_l)           # distance outside the graspable band
    d_l = torch.sqrt(gap_l ** 2 + band_l ** 2 + eps)
    link_prox = 1.0 - torch.tanh(d_l / 0.04)       # (N,5,3)
    finger_prox = link_prox.mean(-1)               # (N,5)
    wrap = finger_prox.mean(-1) * near_soft        # only once the palm is at the cup

    # thumb opposition: tips on opposite sides of the plane through the cup axis and the palm
    tangent = torch.cross(axis, u_p, dim=-1)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=eps)
    tip_rad = rad_l[:, :, 2, :]                                            # (N,5,3)
    tip_u = tip_rad / tip_rad.norm(dim=-1, keepdim=True).clamp(min=eps)
    side = (tip_u * tangent.unsqueeze(1)).sum(-1)                          # (N,5) signed side of each fingertip
    opp = 0.5 * (1.0 - torch.tanh(4.0 * side[:, 0]) * torch.tanh(4.0 * side[:, 1:].mean(-1)))
    opposition = opp * finger_prox[:, 0] * finger_prox[:, 1:].mean(-1) * near_soft

    # contact indicators: saturating (robust to force spikes), counted only on the outer surface inside the band
    F0 = 0.5  # [N] indicator ~0.63 at 0.5 N, ~0.95 at 1.5 N
    valid_l = torch.clamp((r_l - (R_l - 0.015)) / 0.015, 0.0, 1.0) \
        * torch.clamp((H_l + 0.02 - h_l.abs()) / 0.02, 0.0, 1.0)
    link_c = (1.0 - torch.exp(-ctx.link_cup_force.clamp(min=0.0) / F0)) * valid_l   # (N,5,3)
    finger_c = link_c.max(dim=-1).values                                            # (N,5)
    valid_p = torch.clamp((r_p - (cup_r - 0.015)) / 0.015, 0.0, 1.0) \
        * torch.clamp((cup_hh + 0.03 - h_p.abs()) / 0.03, 0.0, 1.0)
    palm_c = (1.0 - torch.exp(-ctx.palm_cup_force.clamp(min=0.0) / F0)) * valid_p  # (N,)

    finger_contact = finger_c.mean(-1)
    link_contact = link_c.mean(dim=(-2, -1))
    full_envelope = palm_c * finger_c.min(dim=-1).values
    grasp_quality = (palm_c + finger_c.sum(-1)) / 6.0     # fraction of {palm, 5 fingers} touching
    hold_q = 0.25 + 0.75 * grasp_quality                  # soft gate: a weak grasp still learns to lift, a full envelope pays 4x

    # ------------------------------------------------------------------ stage C: lift
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    goal_dz = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    lift_prog = torch.clamp(dz / goal_dz, 0.0, 1.0)
    lift = lift_prog * hold_q * upright

    # ------------------------------------------------------------------ stage D: goal and hold
    goal_gate = lifted * hold_q * upright
    goal_coarse = goal_gate * torch.exp(-ctx.goal_dist / 0.08)
    goal_fine = goal_gate * torch.exp(-ctx.goal_dist / 0.02)

    speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    at_goal = lifted * torch.exp(-(ctx.goal_dist / 0.05) ** 2)
    still = at_goal * torch.exp(-speed / 0.1 - spin / 1.0)

    in_tol = (ctx.lifted & (ctx.goal_dist <= ctx.success_tol)).float()
    success_bonus = ctx.success.float() * (0.5 + 0.5 * grasp_quality)

    # ------------------------------------------------------------------ penalties / regularisation
    # 0 while the lowest hand link is >= 5 mm above the table, 1 at the termination height (table_z - 0.03)
    table_pen = torch.clamp((ctx.table_z + 0.005 - ctx.hand_z_min) / 0.035, 0.0, 1.0)
    disp_xy = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    push_pen = torch.tanh(disp_xy / 0.05) * (1.0 - lifted)
    action_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(-1)
    joint_vel = (ctx.arm_qd ** 2).mean(-1) + 0.1 * (ctx.hand_qd ** 2).mean(-1)

    components = {
        "reach": W_REACH * reach,
        "palm_orient": W_ORIENT * palm_orient,
        "finger_preshape": W_PRESHAPE * preshape,
        "finger_wrap": W_WRAP * wrap,
        "thumb_opposition": W_OPPOSITION * opposition,
        "palm_contact": W_PALM_CONTACT * upright * palm_c,
        "finger_contact": W_FINGER_CONTACT * upright * finger_contact,
        "link_contact": W_LINK_CONTACT * upright * link_contact,
        "full_envelope": W_FULL_ENVELOPE * upright * full_envelope,
        "lift": W_LIFT * lift,
        "goal_coarse": W_GOAL_COARSE * goal_coarse,
        "goal_fine": W_GOAL_FINE * goal_fine,
        "hold_still": W_STILL * still,
        "in_tolerance": W_IN_TOL * in_tol,
        "success_bonus": W_SUCCESS * success_bonus,
        "table_penalty": -W_TABLE * table_pen,
        "cup_push_penalty": -W_PUSH * push_pen,
        "action_rate_penalty": -W_ACTION_RATE * action_rate,
        "joint_vel_penalty": -W_JOINT_VEL * joint_vel,
    }
    reward = torch.stack(list(components.values()), dim=0).sum(dim=0)
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
- The arm leaves the raised start pose and reaches the cup within about 2-3 s; in almost every episode the palm reference point of that round (the palm link origin near the wrist, see the feedback) gets within 5 cm of the cup (98% of episodes).
- It does not approach the side of the cup with the palm turned toward it. During the approach (between about 1 s and 3 s) the hand rolls about the forearm: the fingers turn to point down and forward, and the palm ends up facing down and away from the cup instead of toward it.
- In this rolled pose the wrist end of the hand rests against the lower part of the cup on the side nearer the robot, and the straight fingers pass under and behind the cup, sticking out beyond its far side just above the table. The fingers never bend around the cup.
- It keeps this pose for the rest of the episode. At times it pushes the cup so that the cup tilts by roughly 20-25 degrees and then comes back upright; the cup is not knocked over. The cup is never lifted and no success is counted. After a reset the same rolled approach repeats.
- The thumb's placement cannot be seen clearly from this camera angle.
- Training metrics (training stopped at epoch 571): the contact sensor on the palm body registers cup contact in about 61% of steps (up from 21% over the last 200 epochs) even though the palm surface faces away from the cup; finger contact fell from about 0.09 to 0.03 fingers per step and only the thumb still touches occasionally; fewer than 1% of episodes ever had three or more fingers on the cup at once, and fewer than 1% had the palm and all five fingers on the cup together; the lifted fraction is 0. By component over the last 200 epochs: reach rose from about 0.56 to 0.78 per step, palm_contact from 0.13 to 0.41 and finger_wrap (finger links near the cup surface) from 0.14 to 0.26, while finger_contact and link_contact stayed near 0 and palm_orient stayed at about 0.10; the total reward rose from 0.96 to 1.53 while nothing beyond reaching improved.

To make the code more accurate and train better robot, the feedback for improvement is:
- Environment correction for this round: `ctx.palm_pos` is now the centre of the palm (the palm_ee frame). In the previous round it was the palm link origin, which lies about 4 cm from the palm centre toward the wrist and about 3 cm behind the palm surface; `palm_normal` and `palm_side` are unchanged. The previous reward brought that point to the cup, which pulled the wrist end of the hand, not the palm centre, against the cup.
- Keep the fast move from the raised start pose to the cup.
- The approach must bring the palm surface to face the side of the cup with the fingers ready to go around the cup body, instead of rolling the hand so that the palm faces down and the fingers point under the cup. Being near the cup or touching it should only pay when the palm faces the cup; with the palm turned away, approaching and touching should earn little.
- Palm contact must mean the palm surface pressing on the cup. Right now any contact of the palm body counts, including the wrist end or the back of the hand while the palm faces away, and that is what the policy collects.
- Finger links merely being close to the cup surface must not pay on their own: straight fingers sliding along or behind the cup currently earn it. The fingers should be rewarded for bending around the cup body within the graspable band and actually touching it with their links — all five fingers, with the thumb on the opposite side of the cup.
- Rewards that can be collected just by resting next to the cup should not keep growing over the episode. Each next stage should be clearly worth more than staying at the previous one: a correctly oriented approach, then a closed wrap in contact, then lifting the cup upright, then holding it at the goal.
- Pushing or tilting the cup on the table before it is grasped should cost more than it does now; the resting hand pushes the cup and tilts it by 20-25 degrees.

Generated code shown as below:
```python
import torch
import math


def _cylinder_coords(p, cup_pos, cup_axis):
    """Axial coordinate h, radial vector and distance to the cup axis for points p.

    p: (N, ..., 3); cup_pos, cup_axis: (N, 3).
    Returns h (N, ...), radial (N, ..., 3), r (N, ...).
    """
    shape = (cup_pos.shape[0],) + (1,) * (p.dim() - 2) + (3,)
    c = cup_pos.reshape(shape)
    a = cup_axis.reshape(shape)
    v = p - c
    h = (v * a).sum(-1)
    radial = v - h.unsqueeze(-1) * a
    r = radial.norm(dim=-1)
    return h, radial, r


def _unit(v, eps=1e-6):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eps = 1e-6

    # ------------------------------------------------------------------ weights (stage ladder)
    # stage 1, oriented approach: at most ~1.0 per step
    W_REACH_COARSE = 0.25    # ungated, saturates 4 cm from the grasp spot (keeps the fast move)
    W_REACH_FINE = 0.5       # pays only with the palm facing the cup and fingers horizontal
    W_ORIENT = 0.25          # dense orientation shaping, grows as the palm gets close
    # stage 2, wrap in contact: at most ~3.0 per step
    W_CLOSE = 0.3            # commanded finger closing, only with the palm placed and oriented
    W_PALM_CONTACT = 0.5     # palm surface pressing the cup (facing-gated)
    W_FINGER_CONTACT = 1.0   # mean over 5 fingers of valid link contact
    W_LINK_CONTACT = 0.3     # mean over 15 links (how much of each finger wraps)
    W_ENVELOPE = 0.7         # more fingers together with the palm, all five at once
    W_OPPOSITION = 0.2       # thumb on the other side of the cup, both touching
    # stage 3, lift: at most 4.0 per step
    W_LIFT = 4.0
    # stage 4, goal and hold: at most 5.0 per step + success bonus
    W_GOAL_COARSE = 1.5
    W_GOAL_FINE = 1.5
    W_STILL = 1.0
    W_IN_TOL = 1.0
    W_SUCCESS = 20.0
    # penalties / regularisation
    W_TABLE = 1.0            # hand into the table
    W_TILT = 1.5             # tilting the cup before it is lifted (-1.5 at 25 deg)
    W_PUSH = 1.0             # sliding the cup before it is lifted (-1.0 at 6 cm)
    W_ACTION_RATE = 0.01
    W_JOINT_VEL = 0.005

    cup_r = ctx.cup_radius
    cup_hh = ctx.cup_half_height
    axis = ctx.cup_axis
    lifted_f = ctx.lifted.float()

    # uprightness: 0.92 at 5 deg, 0.71 at 10 deg, 0.26 at 20 deg
    upright = torch.exp(-(ctx.cup_tilt / 0.3) ** 2)

    # ------------------------------------------------------------------ stage 1: oriented approach
    h_p, rad_p, r_p = _cylinder_coords(ctx.palm_pos, ctx.cup_pos, axis)
    u_p = rad_p / r_p.clamp(min=eps).unsqueeze(-1)            # unit radial direction: cup axis -> palm centre
    # palm centre 0..2.5 cm outside the cup surface, between 0.1*H below and 0.5*H above the band centre
    e_r = torch.relu((r_p - cup_r - 0.0125).abs() - 0.0125)
    e_h = torch.relu((h_p - 0.2 * cup_hh).abs() - 0.3 * cup_hh)
    d_palm = torch.sqrt(e_r ** 2 + e_h ** 2 + eps)

    n = ctx.palm_normal
    facing = -(n * u_p).sum(-1)                               # +1 when the palm normal points at the cup axis
    palm_gate = torch.clamp((facing - 0.4) / 0.5, 0.0, 1.0)   # 1 within ~26 deg, 0 beyond ~66 deg

    # finger direction from the finger links (index, middle, ring), projected into the palm plane
    base = ctx.link_pos[:, 1:4, 0, :].mean(dim=1) - ctx.palm_pos
    base = base - (base * n).sum(-1, keepdim=True) * n
    d_f = _unit(base)
    f_level = torch.sqrt(torch.clamp(1.0 - (d_f * axis).sum(-1) ** 2, 0.0, 1.0))  # 1 when fingers are horizontal
    finger_gate = torch.clamp((f_level - 0.5) / 0.4, 0.0, 1.0)

    orient_ok = palm_gate * finger_gate

    reach_coarse = 1.0 - torch.tanh(torch.relu(d_palm - 0.04) / 0.2)
    reach_fine = (1.0 - torch.tanh(d_palm / 0.04)) * orient_ok
    orient = 0.5 * (facing + 1.0) * f_level * (1.0 - torch.tanh(d_palm / 0.2))

    near_soft = torch.exp(-(d_palm / 0.04) ** 2)
    gate_g = orient_ok * near_soft                            # palm at the grasp spot and oriented

    # commanded closing of the movable finger joints, only once the palm is placed
    movable = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18]
    close_cmd = ctx.hand_target_norm[:, movable].mean(-1)
    close = close_cmd * orient_ok * torch.exp(-(d_palm / 0.025) ** 2)

    # ------------------------------------------------------------------ stage 2: contact, wrap, opposition
    F0 = 0.5  # [N] saturating indicator, robust to force spikes
    h_l, rad_l, r_l = _cylinder_coords(ctx.link_pos, ctx.cup_pos, axis)   # (N,5,3)
    R_l = cup_r.reshape(-1, 1, 1)
    H_l = cup_hh.reshape(-1, 1, 1)
    valid_l = torch.clamp(1.0 - torch.relu(r_l - R_l - 0.025) / 0.015, 0.0, 1.0) \
        * torch.clamp((H_l + 0.015 - h_l.abs()) / 0.015, 0.0, 1.0)
    link_c = (1.0 - torch.exp(-ctx.link_cup_force.clamp(min=0.0) / F0)) * valid_l
    link_cg = link_c * gate_g.reshape(-1, 1, 1)               # no pay unless the palm is placed and oriented
    finger_cg = link_cg.max(dim=-1).values                    # (N,5)

    # palm contact: the palm surface on the cup (facing, centre near surface, inside the band)
    valid_p = torch.clamp(1.0 - torch.relu(r_p - cup_r - 0.03) / 0.02, 0.0, 1.0) \
        * torch.clamp((cup_hh + 0.02 - h_p.abs()) / 0.02, 0.0, 1.0)
    palm_cg = (1.0 - torch.exp(-ctx.palm_cup_force.clamp(min=0.0) / F0)) * valid_p * palm_gate

    finger_contact = finger_cg.mean(-1)
    link_contact = link_cg.mean(dim=(-2, -1))
    n_frac = finger_cg.sum(-1) / 5.0
    envelope = palm_cg * (0.5 * n_frac ** 2 + 0.5 * finger_cg.min(dim=-1).values)

    # thumb opposition: thumb tip and the other fingertips on opposite sides of the plane (axis, u_p)
    tangent = _unit(torch.cross(axis, u_p, dim=-1))
    tip_u = _unit(rad_l[:, :, 2, :])
    side = (tip_u * tangent.unsqueeze(1)).sum(-1)             # (N,5)
    opp = 0.5 * (1.0 - torch.tanh(4.0 * side[:, 0]) * torch.tanh(4.0 * side[:, 1:].mean(-1)))
    opposition = opp * finger_cg[:, 0] * finger_cg[:, 1:].mean(-1)

    grasp_quality = (palm_cg + finger_cg.sum(-1)) / 6.0      # fraction of {palm, 5 fingers} in valid contact
    hold_q = 0.2 + 0.8 * grasp_quality

    # ------------------------------------------------------------------ stall decay (resting must not keep paying)
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    airborne = torch.clamp(dz / 0.03, 0.0, 1.0) * upright     # tilting the cup does not count as raising it
    stall = airborne + (1.0 - airborne) * (1.0 - 0.75 * ctx.episode_progress.clamp(0.0, 1.0))

    # ------------------------------------------------------------------ stage 3: lift
    goal_dz = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    lift_prog = torch.clamp(dz / goal_dz, 0.0, 1.0)
    lift = lift_prog * hold_q * upright

    # ------------------------------------------------------------------ stage 4: goal and hold
    goal_gate = lifted_f * hold_q * upright
    goal_coarse = goal_gate * torch.exp(-ctx.goal_dist / 0.08)
    goal_fine = goal_gate * torch.exp(-ctx.goal_dist / 0.02)
    speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    still = goal_gate * torch.exp(-(ctx.goal_dist / 0.05) ** 2) * torch.exp(-speed / 0.1 - spin / 1.0)
    in_tol = (ctx.lifted & (ctx.goal_dist <= ctx.success_tol)).float()
    success_bonus = ctx.success.float() * (0.5 + 0.5 * grasp_quality)

    # ------------------------------------------------------------------ penalties
    # 0 while the lowest hand link is >= 5 mm above the table, 1 at the termination height
    table_pen = torch.clamp((ctx.table_z + 0.005 - ctx.hand_z_min) / 0.035, 0.0, 1.0)
    not_lifted = 1.0 - lifted_f
    tilt_pen = torch.clamp((ctx.cup_tilt - 0.087) / 0.35, 0.0, 1.0) * not_lifted     # 5 deg dead band
    disp_xy = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    push_pen = torch.clamp((disp_xy - 0.01) / 0.05, 0.0, 1.0) * not_lifted           # 1 cm dead band
    action_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(-1)
    joint_vel = (ctx.arm_qd ** 2).mean(-1) + 0.1 * (ctx.hand_qd ** 2).mean(-1)

    components = {
        "reach_coarse": W_REACH_COARSE * stall * reach_coarse,
        "reach_oriented": W_REACH_FINE * stall * reach_fine,
        "palm_orient": W_ORIENT * stall * orient,
        "finger_close": W_CLOSE * stall * close,
        "palm_contact": W_PALM_CONTACT * stall * upright * palm_cg,
        "finger_contact": W_FINGER_CONTACT * stall * upright * finger_contact,
        "link_contact": W_LINK_CONTACT * stall * upright * link_contact,
        "envelope": W_ENVELOPE * stall * upright * envelope,
        "thumb_opposition": W_OPPOSITION * stall * upright * opposition,
        "lift": W_LIFT * lift,
        "goal_coarse": W_GOAL_COARSE * goal_coarse,
        "goal_fine": W_GOAL_FINE * goal_fine,
        "hold_still": W_STILL * still,
        "in_tolerance": W_IN_TOL * in_tol,
        "success_bonus": W_SUCCESS * success_bonus,
        "table_penalty": -W_TABLE * table_pen,
        "cup_tilt_penalty": -W_TILT * tilt_pen,
        "cup_push_penalty": -W_PUSH * push_pen,
        "action_rate_penalty": -W_ACTION_RATE * action_rate,
        "joint_vel_penalty": -W_JOINT_VEL * joint_vel,
    }
    reward = torch.stack(list(components.values()), dim=0).sum(dim=0)
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
- The arm leaves the start pose outside the table edge and brings the hand next to the cup within about 2.5-3 s; the palm centre gets within 5 cm of the side of the cup in 99% of episodes.
- The fingers close long before the hand reaches the cup. They start to curl at about 0.5 s, while the hand is still travelling and clearly away from the cup, are fully curled by about 1 s, and never open again. The hand arrives beside the cup with its fingers already bent into a hook instead of open and ready to go around the cup.
- From about 3 s to 12.5 s the hand holds still beside the cup (on its left in the video). The palm does not touch the cup: there is a gap of a few centimetres between the palm and the cup, and bent finger links sit in that gap. The four fingers stay curled in front of the palm next to the lower half of the cup and do not go around the cup body. One digit near the wrist end of the hand rests against the side of the cup; by the contact sensors this is the thumb.
- The camera looks down at the hand from the front, so the hand appears to hang with its fingers pointing down. The orientation measures of this round's reward say otherwise: during the hold the palm faces the cup axis and the finger direction is close to horizontal, so the fingers most likely point horizontally toward the camera along the side of the cup. Which way the thumb points cannot be seen from this angle.
- In the last ~2 s of the episode the hand moves back away from the cup. The cup stays upright where it was placed; it is never lifted, pushed away or tipped.
- Training metrics (the round was ended at epoch 1256 by the stuck rule: reach at or above 0.9 of episodes for the last 600 epochs while grasp stayed near 0; rechecked at epoch 1419 with nothing changed; values are means of the last 150 epochs):
  - Episode funnel: reach 1.00; grasp (three or more fingers on the cup at the same time) 0.00, with a brief peak of 0.12 at epoch 633; palm together with all five fingers 0.00; lifted 0.00, with a brief peak of 0.06 at epoch 318; success 0.
  - Contacts per step: thumb 0.51 (0.41 six hundred epochs earlier); index, middle, ring and pinky 0.003 or less each; palm 0.000. Early in training (around epochs 200-300, while reaching was still unreliable) the palm touched the cup in up to 9% of steps and the index finger in about 4%; both have been zero since about epoch 400-500, when the oriented approach took over.
  - Palm-centre-to-cup gap averaged over all steps, approach included: 0.056 m, flat since epoch 800.
  - Reward components (per step): reach_oriented 0.18, reach_coarse 0.14, palm_orient 0.11, finger_close 0.075, finger_contact 0.034 (thumb only), link_contact 0.003, lift 0.004; palm_contact, envelope, thumb_opposition and all goal terms 0; cup_tilt_penalty -0.007, cup_push_penalty -0.008; total 0.52. reach_oriented and palm_orient have not changed since about epoch 800 and are close to the most that the episode-progress decay allows for a hand held still at the grasp spot.
  - Cup tilt 1.9 degrees on average; episodes ending by tipping, falling or abnormal states are 0.01% or fewer.

To make the code more accurate and train better robot, the feedback for improvement is:
- Keep the fast move from the start pose to the cup, the palm turned toward the side of the cup, and the cup left upright and in place during the approach.
- Hovering beside the cup is currently the best the policy finds. With the palm a few centimetres from the cup and facing it, the approach and orientation rewards are already close to their maximum, and touching the cup adds little by comparison. Being oriented near the cup should earn only a small share; the grasp position should require the palm surface actually pressing on the side of the cup, and the palm touching the cup should be clearly worth more than hovering next to it.
- The fingers must stay open during the approach and arrive open. Right now they are already closed long before the hand gets near the cup, and closing them pays as soon as the palm is near the cup and oriented even when they touch nothing. The curled fingers then end up between the palm and the cup and can no longer go around it. While the palm is not yet on the cup, keeping the fingers extended and spread should be rewarded and closing them should cost. Closing should pay only once the palm is on the cup, and then for fingers closing onto the cup and touching it around the cup body, not for closing on air.
- A single thumb resting on the cup should not collect a meaningful contact reward. Contact should pay more the more fingers touch the cup together with the palm, with the thumb on the opposite side of the cup from the other fingers; the palm plus three or more fingers should be clearly worth more than one finger.
- Each stage should be clearly worth more than staying at the previous one: the palm pressed on the cup above hovering oriented, the palm plus several wrapped fingers above the palm alone, and the cup lifted upright above a wrapped cup still on the table.
- Pressing the palm on the cup and wrapping it will move the cup slightly on the table. That small movement should not cost so much that the policy avoids touching the cup; knocking the cup over or pushing it far away should still cost.

Generated code shown as below:
```python
import torch
import math


def _cylinder_coords(p, cup_pos, cup_axis):
    """Axial coordinate h, radial vector and distance to the cup axis for points p.

    p: (N, ..., 3); cup_pos, cup_axis: (N, 3).
    Returns h (N, ...), radial (N, ..., 3), r (N, ...).
    """
    shape = (cup_pos.shape[0],) + (1,) * (p.dim() - 2) + (3,)
    c = cup_pos.reshape(shape)
    a = cup_axis.reshape(shape)
    v = p - c
    h = (v * a).sum(-1)
    radial = v - h.unsqueeze(-1) * a
    r = radial.norm(dim=-1)
    return h, radial, r


def _unit(v, eps=1e-6):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _touch(force, f0):
    """Saturating contact indicator in [0, 1), robust to force spikes."""
    return 1.0 - torch.exp(-force.clamp(min=0.0) / f0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eps = 1e-6

    # ================================================================== weights (stage ladder)
    # approx. per-step values early in an episode (contact indicators ~0.9):
    #   hover 3-4 cm oriented, fingers open ~0.38 | fingers closed ~0.18
    #   palm pressed ~1.35 | + thumb only ~1.45 | + 3 fingers ~2.15 | palm + 5 fingers opposed ~4.2
    #   cup raised 3 cm ~5.7 | held still at the goal ~11.7 (+ ~19 per success)
    # stage A: oriented approach with an open hand (small share)
    W_REACH_COARSE = 0.15     # ungated, saturates 5 cm from the grasp spot (keeps the fast move)
    W_REACH_ORIENTED = 0.15   # palm facing the cup axis and fingers horizontal
    W_ORIENT = 0.05           # dense orientation shaping
    W_PALM_GAP = 0.10         # sharp (1 cm): the palm centre actually on the cup surface
    W_FINGER_OPEN = 0.10      # +0.1 fully open .. -0.1 fully closed, only while the palm is not on the cup
    # stage B: palm surface pressed on the cup
    W_PALM_CONTACT = 1.0      # ~3.5x hovering
    # stage C: fingers closed onto the cup (everything gated by palm contact)
    W_CURL_ONTO = 0.3         # flexion of fingers whose outer links lie on the cup body
    W_WRAP_FINGERS = 2.5      # (touching fingers / 5)^2: 1 finger 0.1, 3 fingers 0.9, 5 fingers 2.5
    W_WRAP_LINKS = 0.4        # fraction of the 15 measured links touching (real wrap, not fingertip pokes)
    W_ENVELOPE = 0.8          # all five fingers touching, scaled by thumb opposition
    # stage D: lift
    W_LIFT = 4.0              # on top of the grasp terms, so a lifted cup beats a wrapped cup on the table
    # stage E: goal and hold
    W_GOAL_COARSE = 1.5
    W_GOAL_FINE = 1.5         # narrow basin, success_tol shrinks to 1.5 cm
    W_STILL = 1.0
    W_IN_TOL = 1.0
    W_SUCCESS = 20.0
    # penalties / regularisation
    W_TABLE = 1.0             # hand pushing into the table
    W_TILT = 1.5              # 10 deg dead band (pressing may tilt slightly), -1.5 at 35 deg
    W_PUSH = 1.5              # 2 cm dead band (pressing may slide slightly), -1.5 at 8 cm, cup on the table only
    W_ACTION_RATE = 0.01
    W_JOINT_VEL = 0.005

    cup_r = ctx.cup_radius
    cup_hh = ctx.cup_half_height
    axis = ctx.cup_axis
    lifted_f = ctx.lifted.float()
    progress = ctx.episode_progress.clamp(0.0, 1.0)

    # uprightness: strict for lift/goal (0.71 at 10 deg), tolerant for pressing on the table (0.86 at 10 deg)
    upright_hold = torch.exp(-(ctx.cup_tilt / 0.3) ** 2)
    upright_grasp = torch.exp(-(ctx.cup_tilt / 0.45) ** 2)

    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    airborne = torch.clamp(dz / 0.03, 0.0, 1.0) * upright_hold    # tilting the cup does not count as raising it
    on_table = 1.0 - airborne

    # resting on the table must not keep paying: hover terms fall to 40 %, on-table grasp terms to 70 %
    hover_decay = airborne + on_table * (1.0 - 0.6 * progress)
    table_decay = airborne + on_table * (1.0 - 0.3 * progress)

    # ================================================================== stage A: oriented approach
    h_p, rad_p, r_p = _cylinder_coords(ctx.palm_pos, ctx.cup_pos, axis)
    u_p = rad_p / r_p.clamp(min=eps).unsqueeze(-1)            # unit radial direction: cup axis -> palm centre
    gap_p = r_p - cup_r                                       # palm centre distance outside the cup surface
    # grasp spot: palm centre within [-1.0, +1.2] cm of the surface, height between -0.1*H and +0.5*H
    e_r = torch.relu(gap_p - 0.012) + torch.relu(-0.010 - gap_p)
    e_h = torch.relu((h_p - 0.2 * cup_hh).abs() - 0.3 * cup_hh)
    d_palm = torch.sqrt(e_r ** 2 + e_h ** 2 + eps)

    n = ctx.palm_normal
    facing = -(n * u_p).sum(-1)                               # +1 when the palm normal points at the cup axis
    palm_gate = torch.clamp((facing - 0.4) / 0.5, 0.0, 1.0)   # 1 within ~26 deg, 0 beyond ~66 deg

    # finger direction: palm centre -> index/middle/ring links, projected into the palm plane
    base = ctx.link_pos[:, 1:4, 0, :].mean(dim=1) - ctx.palm_pos
    base = base - (base * n).sum(-1, keepdim=True) * n
    d_f = _unit(base)
    f_level = torch.sqrt(torch.clamp(1.0 - (d_f * axis).sum(-1) ** 2, 0.0, 1.0))  # 1 = fingers perpendicular to the cup axis
    finger_gate = torch.clamp((f_level - 0.5) / 0.4, 0.0, 1.0)
    orient_ok = palm_gate * finger_gate

    reach_coarse = 1.0 - torch.tanh(torch.relu(d_palm - 0.05) / 0.2)
    reach_oriented = orient_ok * (1.0 - torch.tanh(d_palm / 0.05))
    palm_orient = 0.5 * (facing + 1.0) * f_level * (1.0 - torch.tanh(d_palm / 0.2))
    palm_gap = orient_ok * torch.exp(-(d_palm / 0.01) ** 2)  # ~0 at a 3 cm hover, full when pressed

    # ================================================================== stage B: palm surface on the cup
    valid_p = torch.clamp(1.0 - torch.relu(gap_p - 0.025) / 0.015, 0.0, 1.0) \
        * torch.clamp(1.0 - torch.relu(-0.020 - gap_p) / 0.015, 0.0, 1.0) \
        * torch.clamp(1.0 - torch.relu(h_p.abs() - cup_hh) / 0.02, 0.0, 1.0)
    # palm-body force counts only with the palm facing the cup, fingers horizontal, palm centre at the surface
    palm_on = _touch(ctx.palm_cup_force, 0.5) * valid_p * orient_ok          # (N,)

    # open hand until the palm is on the cup (never once the cup is raised or has been lifted)
    movable = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18]
    close_tgt = ctx.hand_target_norm[:, movable].mean(-1)
    pre_grasp = (1.0 - palm_on) * (1.0 - lifted_f) * on_table
    finger_open = pre_grasp * (1.0 - 2.0 * close_tgt)

    # ================================================================== stage C: fingers onto the cup
    h_l, rad_l, r_l = _cylinder_coords(ctx.link_pos, ctx.cup_pos, axis)     # (N,5,3)
    R_l = cup_r.reshape(-1, 1, 1)
    H_l = cup_hh.reshape(-1, 1, 1)
    gap_l = r_l - R_l
    in_band_l = torch.clamp(1.0 - torch.relu(h_l.abs() - H_l) / 0.015, 0.0, 1.0)
    outside_l = torch.clamp((gap_l + 0.02) / 0.01, 0.0, 1.0)               # not inside an open cup
    close_l = torch.clamp(1.0 - torch.relu(gap_l - 0.025) / 0.015, 0.0, 1.0)
    valid_l = in_band_l * outside_l * close_l

    link_c = _touch(ctx.link_cup_force, 0.3) * valid_l      # (N,5,3) contact on the cup body
    finger_c = link_c.max(dim=-1).values                    # (N,5)
    n_frac = finger_c.sum(-1) / 5.0

    # closing pays only for fingers whose outer link / fingertip lie on the cup body, and only with the palm on
    finger_joints = [[1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18]]
    curl = torch.stack([ctx.hand_q_norm[:, j].mean(-1) for j in finger_joints], dim=-1)   # (N,5) measured flexion
    near_l = torch.exp(-(torch.relu(gap_l - 0.012) / 0.03) ** 2) * in_band_l * outside_l
    near_f = near_l[:, :, 1:].mean(-1)                      # (N,5) outer link + fingertip
    curl_onto = palm_on * (curl * near_f).mean(-1)

    # thumb opposition: thumb vs. the four fingers on opposite sides of the plane (cup axis, palm direction)
    tangent = _unit(torch.cross(axis, u_p, dim=-1))
    side_l = (_unit(rad_l) * tangent.reshape(-1, 1, 1, 3)).sum(-1)          # (N,5,3) signed side of each link
    w_l = link_c + 0.05                                                     # touching links dominate
    side_f = (side_l * w_l).sum(-1) / w_l.sum(-1)                           # (N,5)
    opp = 0.5 * (1.0 - torch.tanh(4.0 * side_f[:, 0]) * torch.tanh(4.0 * side_f[:, 1:].mean(-1)))

    wrap_fingers = palm_on * n_frac ** 2 * (0.6 + 0.4 * opp)               # superlinear in touching fingers
    wrap_links = palm_on * link_c.mean(dim=(-2, -1))
    envelope = palm_on * finger_c.min(dim=-1).values * (0.4 + 0.6 * opp)

    grasp_quality = (palm_on + finger_c.sum(-1)) / 6.0    # fraction of {palm, 5 fingers} in valid contact
    hold_q = 0.2 + 0.8 * grasp_quality                    # a full envelope lifts for ~2x a weak pinch

    # ================================================================== stage D: lift
    goal_dz = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    # 30 % paid over the first 3 cm so that starting the lift is clearly worth it
    lift_prog = 0.3 * torch.clamp(dz / 0.03, 0.0, 1.0) + 0.7 * torch.clamp(dz / goal_dz, 0.0, 1.0)
    lift = lift_prog * hold_q * upright_hold

    # ================================================================== stage E: goal and hold
    goal_gate = lifted_f * hold_q * upright_hold
    goal_coarse = goal_gate * torch.exp(-ctx.goal_dist / 0.08)
    goal_fine = goal_gate * torch.exp(-ctx.goal_dist / 0.02)
    speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    still = goal_gate * torch.exp(-(ctx.goal_dist / 0.05) ** 2) * torch.exp(-speed / 0.1 - spin / 1.0)
    in_tol = (ctx.lifted & (ctx.goal_dist <= ctx.success_tol)).float()
    success_bonus = ctx.success.float() * (0.5 + 0.5 * grasp_quality)

    # ================================================================== penalties
    # 0 while the lowest hand link is >= 5 mm above the table, 1 at the termination height
    table_pen = torch.clamp((ctx.table_z + 0.005 - ctx.hand_z_min) / 0.035, 0.0, 1.0)
    tilt_pen = torch.clamp((ctx.cup_tilt - math.radians(10.0)) / math.radians(25.0), 0.0, 1.0)
    disp_xy = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    push_pen = torch.clamp((disp_xy - 0.02) / 0.06, 0.0, 1.0) * on_table
    action_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(-1)
    joint_vel = (ctx.arm_qd ** 2).mean(-1) + 0.1 * (ctx.hand_qd ** 2).mean(-1)

    components = {
        "reach_coarse": W_REACH_COARSE * hover_decay * reach_coarse,
        "reach_oriented": W_REACH_ORIENTED * hover_decay * reach_oriented,
        "palm_orient": W_ORIENT * hover_decay * palm_orient,
        "palm_gap": W_PALM_GAP * hover_decay * palm_gap,
        "finger_open": W_FINGER_OPEN * finger_open,
        "palm_contact": W_PALM_CONTACT * table_decay * upright_grasp * palm_on,
        "finger_curl_onto_cup": W_CURL_ONTO * table_decay * upright_grasp * curl_onto,
        "wrap_fingers": W_WRAP_FINGERS * table_decay * upright_grasp * wrap_fingers,
        "wrap_links": W_WRAP_LINKS * table_decay * upright_grasp * wrap_links,
        "envelope": W_ENVELOPE * table_decay * upright_grasp * envelope,
        "lift": W_LIFT * lift,
        "goal_coarse": W_GOAL_COARSE * goal_coarse,
        "goal_fine": W_GOAL_FINE * goal_fine,
        "hold_still": W_STILL * still,
        "in_tolerance": W_IN_TOL * in_tol,
        "success_bonus": W_SUCCESS * success_bonus,
        "table_penalty": -W_TABLE * table_pen,
        "cup_tilt_penalty": -W_TILT * tilt_pen,
        "cup_push_penalty": -W_PUSH * push_pen,
        "action_rate_penalty": -W_ACTION_RATE * action_rate,
        "joint_vel_penalty": -W_JOINT_VEL * joint_vel,
    }
    reward = torch.stack(list(components.values()), dim=0).sum(dim=0)
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
- The video shows the last checkpoint (epoch 2418), which tilts the cup with the palm while pushing down on the table. The wrap-and-lift behaviour that appeared during training (around epochs 1820-2020 and 2330-2390) is not in this checkpoint; it is described below from the training metrics only.
- The arm moves from the start pose to the cup within about 3 s with the hand fully open: the four fingers straight and together, the thumb stretched out ahead of them. The fingers stay open for the whole approach; the early curl of the previous round is gone.
- From about 3.5 s the open hand is low beside the cup, and it does not really press on the cup. The palm leans against the lower side of the cup and tilts the cup away from the hand, while the straight fingers reach past the bottom of the cup and push down on the table. The hand is supported on the table rather than on the cup. The fingers never bend around the cup.
- The hand keeps this pose until the end of the episode, holding the cup tilted a few degrees away from the hand. The cup does not fall over and is never lifted.
- Training metrics (the round was ended at epoch 2418 by the stuck rule on the last point: reach 1.00 at both ends of the 600-epoch window, grasp 0.01; rechecked at epoch 2798 with the same verdict):
  - Two behaviours alternate during training:
    - Tilt the cup and push on the table (epochs 1778-1808, 2137-2267, and from about epoch 2400 to the end): reach 0.96-1.00 of episodes, grasp 0.003-0.013, envelope and lift 0.003 or less. The palm touches the cup in 63-68% of steps, and the mean cup tilt is 4.7-6.0 degrees. Reward per step 0.44-0.50, of which palm_contact 0.23-0.27 and lift 0.023; cup_tilt_penalty -0.003 to -0.021, cup_push_penalty about -0.006, table_penalty 0.0000.
    - Wrap and lift (epochs 1828-1928 and 2327-2387): grasp (three or more fingers on the cup) 0.15-0.32 of episodes, with a peak of 0.69 at epoch 2341; palm plus all five fingers 0.03-0.12, peak 0.28 at epoch 1858; lift 0.03-0.11, peak 0.28. The cup is off the table in 0.5-5% of steps, and successes occur in about 0.1% of episodes (peak 1.4% at epoch 1970). At success 4.8-4.9 fingers were on the cup and the palm in 13-36% of cases.
  - In the wrap-and-lift phases the cup is pushed and tilted much more: cup_push_penalty -0.05 to -0.12, cup_tilt_penalty -0.06 to -0.11, mean tilt 6-8 degrees. Reach drops to 0.57-0.71 and palm_contact to 0.15-0.24. The reward per step is 0.15-0.35, lower than tilting the cup and pushing on the table, and each time training went back to that pose.
  - The table penalty stayed at 0.0000 in every phase even though the fingers push on the table in the video, and no episode ended on the hand-below-table check.
  - The lift term pays about the same in every phase (0.022-0.023 per step), including the tilting phases where the cup is off the table in only 0.01-0.03% of steps.
  - The finger-wrapping terms stay tiny in both behaviours: wrap_fingers 0.001-0.003, finger_curl_onto_cup 0.0006 or less, envelope 0.0002 or less per step.
  - finger_open 0.05-0.06 per step: the fingers are held open during the approach, as intended.
  - Palm-centre-to-cup gap averaged over all steps, approach included: 0.024-0.049 m. Episodes ending by tipping, falling or abnormal states: 0.05% or fewer.

To make the code more accurate and train better robot, the feedback for improvement is:
- Keep: the fast approach with the hand open until the palm is on the cup, the palm turned toward the side of the cup, and the cup not being knocked over.
- The hand must not support itself on the table. In this round the straight fingers push down on the table while the palm tilts the cup, and that costs nothing: the table penalty only looks at how low the hand's link origins go, and those stay above the table while the finger surfaces rest on it. Before the cup is lifted, fingers or the hand coming down onto the table should cost, judged by how close the fingertips and finger links get to the table surface.
- Tilting the cup with the palm must not pay. Palm contact and lift are both collected while the palm holds the cup tilted a few degrees: small tilts are free, and the few millimetres the cup centre rises when it tilts are paid as lift. Palm contact should count only while the cup stays upright. Tilting the cup before it is grasped should start costing at small angles. Lift should pay only when the cup has actually left the table in a grasp.
- Wrapping and lifting must earn clearly more than any resting pose. When the policy closes the fingers and starts to lift, the cup slides and tilts and the palm partly leaves the cup; the movement penalties plus the lost palm and approach rewards outweigh what wrapping and lifting pay. Training found the wrap, the lift and even some successes, and then went back to resting against the cup. Closing the fingers around the cup body while the palm is on it, and then lifting, must pay clearly more per step than resting, including during the transition when the cup moves a little. Resting against the cup with straight fingers should earn less and less the longer it lasts.
- The finger-wrapping rewards are too small to matter (thousandths of a unit per step). Fingers bending around the cup body and touching it while the palm is on the cup need to be a large share of the reward. More fingers touching should be worth clearly more, with the thumb on the opposite side of the cup from the other fingers.
- The cup-movement penalties must tell a grasped cup being lifted apart from a cup being shoved or tipped by the palm. While several fingers are wrapped on the cup, motion and tilt that come from lifting it should not cost as if it were being pushed. Tilting or shoving the cup without a grasp should cost.

Generated code shown as below:
```python
import torch
import math


def _cylinder_coords(p, cup_pos, cup_axis):
    """Axial coordinate h, radial vector and distance to the cup axis for points p.

    p: (N, ..., 3); cup_pos, cup_axis: (N, 3).
    Returns h (N, ...), radial (N, ..., 3), r (N, ...).
    """
    shape = (cup_pos.shape[0],) + (1,) * (p.dim() - 2) + (3,)
    c = cup_pos.reshape(shape)
    a = cup_axis.reshape(shape)
    v = p - c
    h = (v * a).sum(-1)
    radial = v - h.unsqueeze(-1) * a
    r = radial.norm(dim=-1)
    return h, radial, r


def _unit(v, eps=1e-6):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _touch(force, f0):
    """Saturating contact indicator in [0, 1), robust to force spikes."""
    return 1.0 - torch.exp(-force.clamp(min=0.0) / f0)


def _ramp(x, x0, x1):
    """0 for x <= x0, 1 for x >= x1, linear in between (x0 < x1)."""
    return torch.clamp((x - x0) / (x1 - x0), 0.0, 1.0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eps = 1e-6
    deg = math.pi / 180.0

    # ================================================================== weights (stage ladder)
    # approx. per-step values early in an episode (contact indicators ~0.9, thumb opposed):
    #   hovering oriented at the grasp spot ~0.23 | palm pressed on the upright cup ~0.7
    #   palm + thumb + 2 fingers ~1.9 | palm + 5 fingers wrapped ~4.3 (on the table, decays to ~3.3)
    #   same grasp with the cup 3 cm off the table ~6.9 | held still at the goal ~15 (+ ~24 per success)
    #   resting against a 5 deg tilted cup with fingers on the table ~ -1.7
    # stage A: oriented approach with an open hand (small share, decays while resting)
    W_REACH_COARSE = 0.12     # ungated, saturates 5 cm from the grasp spot (keeps the fast move)
    W_REACH_ORIENTED = 0.10   # palm facing the cup axis, fingers horizontal
    W_ORIENT = 0.05           # dense orientation shaping
    W_FINGER_OPEN = 0.10      # +0.1 open .. -0.1 closed, only while the palm is still away from the grasp spot
    # stage B: palm surface on the upright cup
    W_PALM_CONTACT = 0.6      # ~3x hovering; decays while resting without a grasp
    # stage C: wrap (gated by the palm being at the grasp spot, not by palm force)
    W_CURL_ONTO = 1.0         # flexion of fingers lying on / touching the cup body (closing on air pays 0)
    W_WRAP_FINGERS = 3.0      # ((digits touching - 1) / 4)^1.5: 1 digit 0, 2 -> 0.375, 3 -> 1.06, 5 -> 3.0
    W_WRAP_LINKS = 0.8        # fraction of the 15 measured links touching (real wrap, not fingertip pokes)
    W_ENVELOPE = 1.0          # all five digits touching with the thumb opposed, more with the palm on
    # stage D: lift
    W_LIFT = 6.0              # on top of the wrap terms -> a lifted cup clearly beats a wrapped cup on the table
    # stage E: goal and hold
    W_GOAL_COARSE = 2.0
    W_GOAL_FINE = 2.0         # narrow basin, success_tol shrinks to 1.5 cm
    W_STILL = 1.5
    W_IN_TOL = 1.5
    W_SUCCESS = 25.0
    # penalties / regularisation
    W_TABLE = 1.5             # finger links / fingertips / palm coming down onto the table
    W_TILT = 1.5              # ungrasped: 3 deg dead band, full at 13 deg | grasped: 15 .. 35 deg
    W_PUSH = 1.5              # ungrasped on the table: 1 cm dead band, full at 5 cm | grasped: 8 .. 15 cm
    W_ACTION_RATE = 0.01
    W_JOINT_VEL = 0.005

    cup_r = ctx.cup_radius
    cup_hh = ctx.cup_half_height
    axis = ctx.cup_axis
    tilt = ctx.cup_tilt
    progress = ctx.episode_progress.clamp(0.0, 1.0)

    # ================================================================== cup clearance above the table
    # cup_pos starts b above the table. Tipping the cup about its bottom rim raises cup_pos by ~R*sin(tilt)
    # but keeps the lowest rim point on the table, so this clearance stays ~0 when the cup is only tilted.
    cos_t = axis[:, 2].clamp(-1.0, 1.0)
    sin_t = torch.sqrt((1.0 - cos_t ** 2).clamp(min=0.0))
    b = (ctx.cup_spawn_pos[:, 2] - ctx.table_z).clamp(min=0.0)
    clearance = (ctx.cup_pos[:, 2] - ctx.table_z - b * cos_t - cup_r * sin_t).clamp(min=0.0)
    airborne = _ramp(clearance, 0.004, 0.02)                  # 1 once the whole cup is 2 cm off the table
    on_table = 1.0 - airborne

    # ================================================================== palm pose relative to the cup
    h_p, rad_p, r_p = _cylinder_coords(ctx.palm_pos, ctx.cup_pos, axis)
    u_p = rad_p / r_p.clamp(min=eps).unsqueeze(-1)            # unit radial direction: cup axis -> palm centre
    gap_p = r_p - cup_r                                       # palm centre distance outside the cup surface
    # grasp spot: palm centre within [-1.0, +1.2] cm of the surface, height between +0.05*H and +0.55*H
    # (upper part of the band, so the lower fingers stay clear of the table)
    e_r = torch.relu(gap_p - 0.012) + torch.relu(-0.010 - gap_p)
    e_h = torch.relu((h_p - 0.3 * cup_hh).abs() - 0.25 * cup_hh)
    d_palm = torch.sqrt(e_r ** 2 + e_h ** 2 + eps)

    n = ctx.palm_normal
    facing = -(n * u_p).sum(-1)                               # +1 when the palm normal points at the cup axis
    # finger direction: palm centre -> index/middle/ring links, projected into the palm plane
    base = ctx.link_pos[:, 1:4, 0, :].mean(dim=1) - ctx.palm_pos
    base = base - (base * n).sum(-1, keepdim=True) * n
    s_f = (_unit(base) * axis).sum(-1).abs()                  # sin of the finger elevation w.r.t. the cup's cross-section
    # strict gate (approach, palm contact before a grasp): palm within ~26 deg (0 at 53), fingers within ~12 deg (0 at 27)
    orient_ok = _ramp(facing, 0.6, 0.9) * (1.0 - _ramp(s_f, 0.2, 0.45))
    # loose gate (while wrapping / lifting the hand may rotate): palm within ~46 deg, fingers within ~24 deg
    orient_loose = _ramp(facing, 0.3, 0.7) * (1.0 - _ramp(s_f, 0.4, 0.7))

    # ================================================================== hand coming down onto the table
    # link frames sit ~half a finger thickness above the surface when a finger lies on the table
    link_clear = ctx.link_pos[..., 2] - ctx.table_z                                   # (N,5,3)
    link_low = (1.0 - _ramp(link_clear, 0.006, 0.018)).flatten(1).max(dim=-1).values   # 0 at >= 18 mm, 1 at <= 6 mm
    other_low = 1.0 - _ramp(ctx.hand_z_min - ctx.table_z, 0.006, 0.018)               # any other hand link
    palm_low = 1.0 - _ramp(ctx.palm_pos[:, 2] - ctx.table_z, 0.010, 0.030)            # palm lying on the table
    table_near = torch.maximum(torch.maximum(link_low, other_low), palm_low)          # (N,) in [0, 1]

    # ================================================================== finger contacts on the cup body
    h_l, rad_l, r_l = _cylinder_coords(ctx.link_pos, ctx.cup_pos, axis)     # (N,5,3)
    R_l = cup_r.reshape(-1, 1, 1)
    H_l = cup_hh.reshape(-1, 1, 1)
    gap_l = r_l - R_l
    in_band_l = torch.clamp(1.0 - torch.relu(h_l.abs() - H_l) / 0.015, 0.0, 1.0)
    outside_l = _ramp(gap_l, -0.02, -0.01)                                  # not inside an open cup
    close_l = torch.clamp(1.0 - torch.relu(gap_l - 0.03) / 0.02, 0.0, 1.0)
    valid_l = in_band_l * outside_l * close_l

    link_c = _touch(ctx.link_cup_force, 0.3) * valid_l      # (N,5,3) contact on the cup body
    finger_c = link_c.max(dim=-1).values                    # (N,5)
    n_touch = finger_c.sum(-1)                              # (N,) digits touching, 0..5

    # thumb opposition: thumb vs. the four fingers on opposite sides of the plane (cup axis, palm direction)
    tangent = _unit(torch.cross(axis, u_p, dim=-1))
    side_l = (_unit(rad_l) * tangent.reshape(-1, 1, 1, 3)).sum(-1)          # (N,5,3) signed side of each link
    w_l = link_c + 0.05                                                     # touching links dominate
    side_f = (side_l * w_l).sum(-1) / w_l.sum(-1)                           # (N,5)
    opp = 0.5 * (1.0 - torch.tanh(4.0 * side_f[:, 0]) * torch.tanh(4.0 * side_f[:, 1:].mean(-1)))
    opp_q = finger_c[:, 0] * opp                                            # thumb touching on the far side

    # grasp quality: 0 with a single digit, 1 with the thumb opposed + 3 fingers (4 non-thumb fingers alone -> 0.3)
    grasp_q = _ramp(n_touch, 1.0, 4.0) * (0.3 + 0.7 * opp_q)
    relief = torch.sqrt(grasp_q + eps)                      # rises early in the transition to a wrap

    # ================================================================== stage A: oriented approach
    # resting without a grasp must not keep paying: approach and palm terms fall to 20 % over the step budget
    stay = (1.0 - 0.8 * progress) + grasp_q * 0.8 * progress
    tilt_free = _ramp(tilt, 3.0 * deg, 13.0 * deg)          # ungrasped tilt measure
    # tilting the cup or leaning on the table is not an approach
    gate_a = stay * (1.0 - table_near) * (1.0 - (1.0 - relief) * tilt_free)

    reach_coarse = 1.0 - torch.tanh(torch.relu(d_palm - 0.05) / 0.2)
    reach_oriented = orient_ok * (1.0 - torch.tanh(d_palm / 0.05))
    palm_orient = 0.5 * (facing + 1.0) * (1.0 - s_f) * (1.0 - torch.tanh(d_palm / 0.2))

    # open hand while the palm is still away from the grasp spot (0 at the spot, so closing there never costs)
    movable = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18]
    close_tgt = ctx.hand_target_norm[:, movable].mean(-1)
    far = 1.0 - torch.exp(-(d_palm / 0.03) ** 2)            # 0 at the spot, 0.63 at 3 cm, 0.94 at 5 cm
    finger_open = far * (1.0 - grasp_q) * on_table * (1.0 - 2.0 * close_tgt)

    # ================================================================== stage B: palm surface on the upright cup
    valid_p = torch.clamp(1.0 - torch.relu(gap_p - 0.025) / 0.015, 0.0, 1.0) \
        * torch.clamp(1.0 - torch.relu(-0.020 - gap_p) / 0.015, 0.0, 1.0) \
        * torch.clamp(1.0 - torch.relu(h_p.abs() - cup_hh) / 0.02, 0.0, 1.0)
    palm_touch = _touch(ctx.palm_cup_force, 0.5) * valid_p                   # palm force with the centre at the surface
    upright_strict = 1.0 - _ramp(tilt, 2.5 * deg, 5.0 * deg)                # before a grasp: 1 below 2.5 deg, 0 at 5 deg
    upright_grasp = 1.0 - _ramp(tilt, 10.0 * deg, 25.0 * deg)               # in a grasp: 1 below 10 deg, 0 at 25 deg
    up_palm = upright_strict + grasp_q * (upright_grasp - upright_strict)   # relaxes only as the grasp forms
    orient_palm = orient_ok + grasp_q * (orient_loose - orient_ok)
    palm_on = palm_touch * orient_palm * up_palm

    # ================================================================== stage C: fingers wrapped on the cup
    engage = orient_loose * torch.exp(-(torch.relu(d_palm - 0.01) / 0.04) ** 2)   # palm at the grasp spot, no force needed

    finger_joints = [[1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18]]
    curl = torch.stack([ctx.hand_q_norm[:, j].mean(-1) for j in finger_joints], dim=-1)   # (N,5) measured flexion
    near_l = torch.exp(-(torch.relu(gap_l - 0.012) / 0.03) ** 2) * in_band_l * outside_l
    near_f = near_l[:, :, 1:].mean(-1)                      # (N,5) outer link + fingertip on the cup body
    curl_onto = engage * (curl * (0.3 * near_f + 0.7 * finger_c)).mean(-1)

    wrap_fingers = engage * _ramp(n_touch, 1.0, 5.0) ** 1.5 * (0.4 + 0.6 * opp_q)   # superlinear, single digit = 0
    wrap_links = engage * link_c.mean(dim=(-2, -1))
    envelope = engage * finger_c.min(dim=-1).values * opp_q * (0.5 + 0.5 * palm_touch)

    wrap_decay = airborne + on_table * (1.0 - 0.4 * progress)   # a wrap left on the table slowly loses value
    gate_c = wrap_decay * upright_grasp

    # ================================================================== stage D: lift (only a real lift in a grasp)
    upright_hold = torch.exp(-(tilt / 0.25) ** 2)            # 0.89 at 5 deg, 0.61 at 10 deg
    goal_dz = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    # 40 % paid once the whole cup is 2 cm clear, so starting the lift is clearly worth it
    lift_prog = 0.4 * airborne + 0.6 * torch.clamp(clearance / goal_dz, 0.0, 1.0)
    lift = lift_prog * grasp_q * upright_hold

    # ================================================================== stage E: goal and hold
    goal_gate = airborne * grasp_q * upright_hold
    goal_coarse = goal_gate * torch.exp(-ctx.goal_dist / 0.08)
    goal_fine = goal_gate * torch.exp(-ctx.goal_dist / 0.02)
    speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    still = goal_gate * torch.exp(-(ctx.goal_dist / 0.05) ** 2) * torch.exp(-speed / 0.1 - spin / 1.0)
    hold_f = 0.5 + 0.5 * grasp_q
    in_tol = (ctx.lifted & (ctx.goal_dist <= ctx.success_tol)).float() * hold_f
    success_bonus = ctx.success.float() * hold_f

    # ================================================================== penalties
    # cup motion: strict without a grasp, lenient while several digits hold the cup (lifting moves it)
    tilt_pen = (1.0 - relief) * tilt_free + relief * _ramp(tilt, 15.0 * deg, 35.0 * deg)
    disp_xy = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    push_pen = (1.0 - relief) * on_table * _ramp(disp_xy, 0.01, 0.05) + relief * _ramp(disp_xy, 0.08, 0.15)
    action_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(-1)
    joint_vel = (ctx.arm_qd ** 2).mean(-1) + 0.1 * (ctx.hand_qd ** 2).mean(-1)

    components = {
        "reach_coarse": W_REACH_COARSE * gate_a * reach_coarse,
        "reach_oriented": W_REACH_ORIENTED * gate_a * reach_oriented,
        "palm_orient": W_ORIENT * gate_a * palm_orient,
        "finger_open": W_FINGER_OPEN * finger_open,
        "palm_contact": W_PALM_CONTACT * stay * (1.0 - table_near) * palm_on,
        "finger_curl_onto_cup": W_CURL_ONTO * gate_c * curl_onto,
        "wrap_fingers": W_WRAP_FINGERS * gate_c * wrap_fingers,
        "wrap_links": W_WRAP_LINKS * gate_c * wrap_links,
        "envelope": W_ENVELOPE * gate_c * envelope,
        "lift": W_LIFT * lift,
        "goal_coarse": W_GOAL_COARSE * goal_coarse,
        "goal_fine": W_GOAL_FINE * goal_fine,
        "hold_still": W_STILL * still,
        "in_tolerance": W_IN_TOL * in_tol,
        "success_bonus": W_SUCCESS * success_bonus,
        "table_penalty": -W_TABLE * table_near,
        "cup_tilt_penalty": -W_TILT * tilt_pen,
        "cup_push_penalty": -W_PUSH * push_pen,
        "action_rate_penalty": -W_ACTION_RATE * action_rate,
        "joint_vel_penalty": -W_JOINT_VEL * joint_vel,
    }
    reward = torch.stack(list(components.values()), dim=0).sum(dim=0)
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
- The video shows the last checkpoint (epoch 4423).
- The arm brings the hand toward the cup within about 3 s with the hand fully open: the four fingers straight and together, the thumb stretched out ahead of them. The hand ends up beside the upper half of the cup (on its left in the video), not low near the table.
- From about 3.5 s the hand is turned so that its thumb side faces the cup. From the camera the palm is seen edge-on and does not face the cup. The straight fingers hang down beside the cup, clear of the table, and the thumb is stretched out horizontally with its tip resting on the upper side of the cup near the rim. The palm stays a few centimetres away from the cup and never touches it, and the fingers never bend.
- The hand holds this pose, with only the thumb on the cup, until the end of the episode. The cup stays upright and in place and is never lifted. The hand does not touch the table.
- Training metrics (the round was ended at epoch 4423 by the stuck rule: reach at or above 0.9 across the last 600 epochs while grasp stayed at 0):
  - The round went through three phases:
    - Epochs 0-800: the hand stayed far from the cup (reach 0.00 of episodes, palm-centre-to-cup gap 0.25 m, reward 0.06 per step).
    - Epochs 800-2600: it hovered about 10-13 cm from the cup with the hand open (reach 0.20, gap 0.13 m, reward 0.14).
    - Epochs 2600-4423: it came closer (reach 0.81, gap 0.084 m, thumb contact in 5% of steps, reward 0.17).
  - At the end (training kept running while waiting for approval; rechecked at epoch 4973): reach 0.82-0.84 of episodes (1.00 at epoch 4423), grasp 0.00, palm plus all five fingers 0.00, lift 0.00, success 0. The thumb touches the cup in 36-50% of steps and is the only digit in contact; the palm 0.
  - Brief peaks during the round: grasp (three or more fingers on the cup) 0.24 at epoch 3138, palm plus all five fingers 0.03 at epoch 3481, lift 0.05 at epoch 4005; the cup was off the table in at most 0.3% of steps; no successes.
  - Around epoch 1000 the palm touched the cup in up to 12% of steps, but palm_contact never paid more than 0.001 per step during the whole round.
  - Reward components at the end (per step): finger_open 0.09 (its maximum is 0.1), reach_coarse 0.06, palm_orient 0.017, reach_oriented 0.011, palm_contact 0.0000, wrap_fingers 0.0000, lift 0; total 0.18. finger_open was more than half of the total reward from about epoch 800 to the end.
  - Penalties stayed near zero: table_penalty -0.0009 on average after epoch 2600 (its largest value was -0.10 at epoch 574), cup tilt 0.1-0.2 degrees, cup_tilt_penalty and cup_push_penalty -0.0007 or less.

To make the code more accurate and train better robot, the feedback for improvement is:
- Keep: the hand open during the approach, the cup left upright and untouched during the approach, and the hand kept off the table.
- Hovering a few centimetres from the cup with an open hand is currently worth more than moving the palm onto the cup. The open-hand reward is paid only while the palm is away from the grasp position and does not fade over the episode, so it disappears exactly when the palm arrives. Meanwhile the approach rewards fade over the episode and require a strict orientation. Keeping the hand open must never be worth more than getting closer: that reward should not depend on staying away from the grasp position, and it should stay small compared with reaching the grasp position.
- Reaching the grasp position with the palm facing the cup must be clearly worth more than hovering at every point in the episode. Getting closer should keep paying as the palm approaches, even before the orientation is exact; the orientation had to be so exact that the palm-facing approach almost never paid.
- The palm has to face the cup. The hand turned so that only its thumb side faces the cup, with the palm sideways. Approaching thumb first, or resting only the thumb on the cup, should earn little.
- Palm contact must be reachable. When the palm did touch the cup early in training, the palm reward stayed at zero, because every gate had to hold at once: an exact orientation, a cup within a couple of degrees of upright, and the hand clear of the table. The palm on the side of the cup should pay once the palm faces the cup reasonably well and the cup is not being tipped; the strict conditions should shape the reward, not zero it.
- Keep the rest of the previous feedback, which this round could not test because the palm never reached the cup: fingers closing around the cup body with the palm on it must be worth much more than any resting pose, lift must pay only when the cup has left the table in a grasp, and the cup-movement penalties must be relaxed during a real grasp.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0, 0.0061, 0, 0, 0, 0, 0.000488, 0.000488, 0]  min 0 · mean 0.00171 · max 0.113
contact/finger_index_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_middle: [0, 0.000244, 0.000244, 0, 0, 0, 0, 0.000488, 0.000244, 0]  min 0 · mean 0.000402 · max 0.0256
contact/finger_middle_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_pinky: [0, 0, 0, 0.00635, 0.000488, 0.00195, 0, 0.000244, 0, 0]  min 0 · mean 0.00252 · max 0.428
contact/finger_pinky_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_ring: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000341 · max 0.0337
contact/finger_ring_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_thumb: [0, 0, 0, 0, 0, 0, 0.00635, 0.193, 0.0122, 0.335]  min 0 · mean 0.0591 · max 0.683
contact/finger_thumb_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/fingers_touching: [0, 0.000244, 0.00635, 0.00635, 0.000488, 0.00195, 0.00635, 0.194, 0.0129, 0.335]  min 0 · mean 0.0641 · max 0.683
contact/fingers_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/link_force_mean: [0, 1.25e-05, 0.000559, 0.000199, 1.64e-05, 4.89e-05, 0.000231, 0.0136, 0.000422, 0.0101]  min 0 · mean 0.00522 · max 0.893
contact/links_touching: [0, 0.000244, 0.00659, 0.00635, 0.000488, 0.00195, 0.00635, 0.199, 0.0129, 0.336]  min 0 · mean 0.0645 · max 0.685
contact/links_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/palm_touching: [0, 0, 0.0759, 0, 0, 0, 0, 0.000244, 0, 0.0137]  min 0 · mean 0.00132 · max 0.12
contact/palm_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/abnormal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000272 · max 0.0122
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.41e-07 · max 0.000244
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/tipped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.49e-05 · max 0.00684
episode_lengths/step: [57, 843, 880, 809, 890, 885, 884, 893, 898, 899]  min 57 · mean 820 · max 899
reward/action_rate_penalty: [-0.0103, -0.00421, -0.00245, -0.00276, -0.00152, -0.00122, -0.00118, -0.00109, -0.0008, -0.000709]  min -0.0104 · mean -0.00201 · max -0.000509
reward/cup_push_penalty: [0, -0.000191, -3.29e-05, 0, 0, 0, 0, -0.00827, -9.99e-05, 0]  min -0.441 · mean -0.00187 · max 0
reward/cup_tilt_penalty: [0, 0, -0.00233, -6.01e-05, 0, -2.8e-05, -0.000252, -0.00726, -0.00045, -0.000251]  min -0.627 · mean -0.00344 · max 0
reward/envelope: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.84e-08 · max 3.63e-05
reward/finger_curl_onto_cup: [7.93e-23, 3.8e-18, 3.12e-26, 1.07e-08, 6.4e-10, 9.02e-07, 1.89e-06, 4.78e-05, 3.06e-06, 1.09e-06]  min 0 · mean 5.44e-06 · max 0.000902
reward/finger_open: [0.0182, 0.0966, 0.0883, 0.097, 0.0962, 0.0965, 0.0956, 0.0731, 0.0888, 0.0951]  min -0.0866 · mean 0.0854 · max 0.0995
reward/goal_coarse: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.7e-08 · max 1.77e-05
reward/goal_fine: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.89e-12 · max 4.92e-09
reward/hold_still: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.77e-17 · max 6.58e-14
reward/in_tolerance: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/joint_vel_penalty: [-0.00236, -0.00034, -0.000256, -0.000232, -0.000167, -0.00017, -0.000169, -0.000145, -0.0001, -0.000101]  min -0.0104 · mean -0.00042 · max -8.08e-05
reward/lift: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.34e-07 · max 0.000453
reward/palm_contact: [0, 0, 0, 0, 0, 0, 0, 9.53e-06, 0, 1.37e-05]  min 0 · mean 2.91e-06 · max 0.000885
reward/palm_orient: [0.00483, 0.00308, 0.00395, 0.00871, 0.00818, 0.0115, 0.014, 0.0175, 0.0165, 0.0161]  min 0.000166 · mean 0.0105 · max 0.0181
reward/reach_coarse: [0.0272, 0.0243, 0.0475, 0.0502, 0.0539, 0.0571, 0.0585, 0.0599, 0.0601, 0.061]  min 0.00388 · mean 0.049 · max 0.0629
reward/reach_oriented: [0, 2.02e-11, 1.48e-10, 5.84e-09, 0, 1.75e-05, 0.00569, 0.0159, 0.0119, 0.00856]  min 0 · mean 0.00419 · max 0.0171
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/table_penalty: [0, 0, -0.000109, 0, -3.81e-05, 0, 0, -0.00661, 0, 0]  min -0.101 · mean -0.00093 · max 0
reward/total: [0.0375, 0.119, 0.135, 0.153, 0.157, 0.164, 0.172, 0.147, 0.176, 0.18]  min -0.986 · mean 0.141 · max 0.185
reward/wrap_fingers: [0, 0, 0, 0, 0, 0, 0, 7.05e-05, 0, 0]  min 0 · mean 6.64e-06 · max 0.0021
reward/wrap_links: [0, 0, 0, 0, 0, 5.58e-07, 1.12e-05, 0.00395, 0.00012, 0]  min 0 · mean 0.000353 · max 0.00942
rewards/step: [1.55, 87.8, 94.5, 118, 132, 121, 142, 140, 158, 160]  min -88.3 · mean 117 · max 162
task/lifted_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.67e-05 · max 0.00342
task/successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/tilt_deg: [0.00108, 0.00203, 0.151, 0.0112, 0.0031, 0.00636, 0.0125, 0.279, 0.0172, 0.303]  min 0.000912 · mean 0.129 · max 12.8
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
