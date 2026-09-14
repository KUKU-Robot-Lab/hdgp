You are an expert in robotics, reinforcement learning and code generation.

We control one 7-DOF OpenArm robot arm (the right arm) carrying a five-finger Tesollo DG-5F hand, standing at a table. An upright cylindrical cup stands on the table in front of the hand; its radius differs between parallel environments (29 mm to 44 mm) and is given per environment. At the start of an episode the palm is roughly 0.15 m from the cup. Positions are in metres in each environment's local frame, with +z pointing up; the table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (26,), float32)` with direct joint control. There is no grasp primitive, no hand synergy and no automatic finger stopping:
    actions[0:7]  = arm joint increments: each arm joint target moves by 0.025 * a rad per step and then passes a first-order filter (factor 0.1), so each arm joint moves at most about 0.15 rad/s.
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
    palm_pos: torch.Tensor         # (N,3) palm frame origin
    palm_normal: torch.Tensor      # (N,3) unit vector pointing out of the palm surface, towards an object held in the hand
    palm_side: torch.Tensor        # (N,3) unit vector lying in the palm plane (palm frame y axis)
    link_pos: torch.Tensor         # (N,5,3,3) finger link positions: [:, f, 0] link moved by joint _3, [:, f, 1] link moved by joint _4, [:, f, 2] fingertip
    link_cup_force: torch.Tensor   # (N,5,3) contact force magnitude between each of those links and the cup only [N] (0 = not touching the cup)
    palm_cup_force: torch.Tensor   # (N,) contact force magnitude between the palm and the cup only [N]
    hand_q: torch.Tensor           # (N,19) finger joint angles [rad], order = the hand joint table in the robot description
    hand_q_norm: torch.Tensor      # (N,19) joint angles normalised to each joint's commandable range: 0 = lower limit (straight), 1 = upper limit (most flexed)
    hand_target_norm: torch.Tensor  # (N,19) commanded finger joint targets (after filtering), same normalisation as hand_q_norm
    hand_qd: torch.Tensor          # (N,19) finger joint velocities [rad/s]
    hand_z_min: torch.Tensor       # (N,) height of the lowest hand link, palm excluded [m]; below table_z means the hand presses into the table

    # ---- arm: 7-DOF ------------------------------------------------------------------------
    arm_q: torch.Tensor            # (N,7) arm joint angles [rad]
    arm_qd: torch.Tensor           # (N,7) arm joint velocities [rad/s]

    # ---- cup: a cylinder standing upright on the table at the start ------------------------
    cup_pos: torch.Tensor          # (N,3) cup reference point: on the cylinder axis at the cup's centre of mass, which is also the middle of its graspable band
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
5. Cylinder geometry: for a point p (e.g. `ctx.link_pos[:, f, k]`), with `v = p - ctx.cup_pos`, the axial coordinate is `h = (v * ctx.cup_axis).sum(-1)` and the radial vector is `v - h.unsqueeze(-1) * ctx.cup_axis`; its norm is the distance from the cup axis. The cup surface is at `ctx.cup_radius` and the graspable band is `|h| <= ctx.cup_half_height`. The palm normal `ctx.palm_normal` points out of the palm.
6. Height: `ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]` is how far the cup has been raised above its starting height (0 while it rests on the table). `ctx.lifted` becomes True once that height exceeds `ctx.lift_latch_height` and stays True for the rest of the episode, even if the cup is dropped again.
7. Goal and success are computed by the environment and cannot be redefined by the reward. The goal position is 0.21 m to 0.28 m above the cup's starting position and at most 5 cm away from it horizontally, with the cup upright. A success is counted when `ctx.goal_dist <= ctx.success_tol` for `ctx.success_hold_steps` consecutive steps, and `ctx.success` is True on that step. After a success the next goal is at the same place, so holding the cup still there keeps producing successes until `ctx.max_successes`, which ends the episode. The keypoints are fixed on the cup, so tilting the cup also increases `goal_dist`. `ctx.success_tol` starts at 0.1125 m and shrinks towards 0.015 m as the policy succeeds more often during training. You may add a bonus on `ctx.success`.
8. An episode lasts at most 600 steps (10 s), and the step budget restarts after every success. The episode ends early when the cup falls below z = 0.15 (off the table), leaves the table area, or tilts more than 60 degrees; when any hand link goes more than 3 cm below the table top; or when an arm joint goes past its limit or moves faster than 20 rad/s. On that last kind of physics violation the environment also adds a fixed -1 to the reward, outside your function.
9. Do not keep any state between calls (no globals, no attributes); the function must be pure.
10. Each component you return is logged separately during training and may be shown back to you after training, so name components meaningfully (e.g. "approach", "lift", "success_bonus").

I want it to fulfil the following task: Grasp the upright cylindrical cup with an envelope grasp (a power grasp): bring the palm against the side of the cup and wrap all five fingers around it, with the thumb on the opposite side of the cup from the other four fingers, so that the palm and the links of every finger are in contact with the cup. Then lift the cup to the goal position and hold it there, upright and still, so that the environment keeps counting successes. Do not knock the cup over, do not drop it, and do not push the hand into the table.
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


def _unit(v: torch.Tensor) -> torch.Tensor:
    """Normalise the last dimension safely."""
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _cyl_coords(p: torch.Tensor, cup_pos: torch.Tensor, axis: torch.Tensor):
    """Axial coordinate h, radial vector and radial distance of points p w.r.t. the cup axis."""
    v = p - cup_pos
    h = (v * axis).sum(-1)
    rad = v - h.unsqueeze(-1) * axis
    return h, rad, rad.norm(dim=-1)


def _contact_score(force: torch.Tensor, f0: float = 0.5) -> torch.Tensor:
    """Saturating contact score in [0,1]: 0.1 N -> 0.18, 0.5 N -> 0.63, >= 2 N -> ~1 (spike-proof)."""
    return 1.0 - torch.exp(-force.clamp(min=0.0) / f0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ---------------- weights ----------------
    W_APPROACH = 1.0     # stage 1: palm to the side of the cup, facing its axis
    W_REACH = 1.0        # stage 2: all 15 finger links onto the cylinder band
    W_WRAP = 1.0         # stage 2: thumb opposite the four fingers, palm between (enclosure)
    W_CONTACT = 1.5      # stage 3: dense, each touching group (palm + 5 fingers) counts
    W_ENVELOPE = 2.0     # stage 3: geometric mean -> pays only when palm AND all five fingers touch
    W_LINKS = 1.0        # stage 3: several links per finger touching (wrap, not tip pinch)
    W_SQUEEZE = 0.5      # stage 3: fingers in contact keep pressing (PD target beyond actual)
    W_LIFT = 4.0         # stage 4: raise the cup, scaled by grip quality
    W_GOAL = 6.0         # stage 5: bring the cup to the goal (goal_dist includes tilt)
    W_STILL = 2.0        # stage 5: steady hold near the goal so successes keep counting
    W_SUCCESS = 25.0     # per counted success, scaled by envelope quality
    W_TILT = 2.0         # do not knock the cup over
    W_PUSH = 1.0         # do not shove the cup before it is lifted
    W_TABLE = 3.0        # do not press hand links into the table
    W_ARM_RATE = 0.02    # smooth arm actions
    W_HAND_RATE = 0.005  # smooth finger actions (movable joints only)

    # movable finger joints per finger (hand indices): thumb, index, middle, ring, pinky
    finger_joints = [[1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18]]
    movable = [j for group in finger_joints for j in group]

    axis = ctx.cup_axis
    radius = ctx.cup_radius
    half_h = ctx.cup_half_height

    # ---------------- stage 1: palm approach + orientation ----------------
    h_p, rad_p, r_p = _cyl_coords(ctx.palm_pos, ctx.cup_pos, axis)
    u_palm = _unit(rad_p)
    palm_d = torch.relu(r_p - radius - 0.03) + torch.relu(h_p.abs() - half_h)  # 3 cm slack: palm origin vs skin
    s_palm = _contact_score(ctx.palm_cup_force)
    palm_prox = torch.maximum(1.0 - torch.tanh(10.0 * palm_d), s_palm)
    palm_align = (ctx.palm_normal * (-u_palm)).sum(-1)  # 1 = palm faces the cup axis
    orient = ((palm_align + 1.0) * 0.5) ** 2
    approach = palm_prox * (0.3 + 0.7 * orient)

    # ---------------- stage 2: finger links onto the band ----------------
    c = ctx.cup_pos[:, None, None, :]
    a = axis[:, None, None, :]
    h_l, rad_l, r_l = _cyl_coords(ctx.link_pos, c, a)  # (N,5,3)
    link_d = torch.relu((r_l - radius[:, None, None]).abs() - 0.015) + torch.relu(h_l.abs() - half_h[:, None, None])
    s_link = _contact_score(ctx.link_cup_force)  # (N,5,3)
    link_prox = torch.maximum(torch.exp(-20.0 * link_d), s_link)
    finger_reach = link_prox.mean(dim=(1, 2))

    # ---------------- stage 2: envelope geometry (opposition + enclosure) ----------------
    dir_f = _unit(_unit(rad_l).sum(dim=2))           # (N,5,3) angular position of each finger around the axis
    dir_thumb = dir_f[:, 0]
    dir_four = _unit(dir_f[:, 1:].sum(dim=1))        # (N,3) index..pinky as one group
    opposition = 0.5 * (1.0 - (dir_thumb * dir_four).sum(-1))
    enclosure = (1.0 - ((u_palm + dir_thumb + dir_four) / 3.0).norm(dim=-1)).clamp(0.0, 1.0)
    near_f = link_prox.mean(dim=2)                   # (N,5)
    wrap_gate = (palm_prox * near_f[:, 0] * near_f[:, 1:].mean(dim=1)).clamp(min=0.0).pow(1.0 / 3.0)
    wrap = enclosure * opposition * wrap_gate

    # ---------------- stage 3: contact ----------------
    s_finger = s_link.max(dim=2).values              # (N,5)
    groups = torch.cat([s_palm.unsqueeze(1), s_finger], dim=1)  # (N,6) palm, thumb, index, middle, ring, pinky
    contact = groups.mean(dim=1)
    envelope = groups.clamp(min=0.0).prod(dim=1).pow(1.0 / 6.0)
    link_cover = s_link.mean(dim=(1, 2))

    press = ((ctx.hand_target_norm - ctx.hand_q_norm) / 0.2).clamp(0.0, 1.0)  # (N,19)
    press_f = torch.stack([press[:, j].mean(dim=1) for j in finger_joints], dim=1)  # (N,5)
    squeeze = (s_finger * press_f).mean(dim=1)

    # grip factor: two opposing groups needed; full envelope is worth ~4x a pinch
    held = torch.maximum(s_finger[:, 0], s_palm) * s_finger[:, 1:].max(dim=1).values
    grip_quality = 0.3 * contact + 0.7 * envelope
    grip = held * (0.2 + 0.8 * grip_quality)

    # ---------------- stage 4: lift ----------------
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    goal_rise = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    lift = (height / goal_rise).clamp(0.0, 1.0) * grip

    # ---------------- stage 5: goal + still hold ----------------
    zeros = torch.zeros_like(height)
    goal_track = 0.5 * (1.0 - torch.tanh(5.0 * ctx.goal_dist)) + 0.5 * (1.0 - torch.tanh(30.0 * ctx.goal_dist))
    goal = torch.where(ctx.lifted, goal_track * grip, zeros)

    lin_speed = ctx.cup_lin_vel.norm(dim=-1)
    ang_speed = ctx.cup_ang_vel.norm(dim=-1)
    steady = torch.exp(-ctx.goal_dist / 0.03) * torch.exp(-lin_speed / 0.05) * torch.exp(-ang_speed / 0.5)
    hold_still = torch.where(ctx.lifted, steady * grip, zeros)

    success_bonus = ctx.success.float() * (0.25 + 0.75 * grip_quality)

    # ---------------- penalties ----------------
    tilt_pen = (ctx.cup_tilt / (math.pi / 3.0)).clamp(0.0, 1.0) ** 2
    xy_disp = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    push_pen = torch.where(ctx.lifted, zeros, torch.tanh(torch.relu(xy_disp - 0.01) / 0.05))
    table_pen = ((ctx.table_z - ctx.hand_z_min) / 0.03).clamp(0.0, 1.0)
    d_act = ctx.actions - ctx.prev_actions
    arm_rate = (d_act[:, :7] ** 2).sum(-1)
    hand_rate = (d_act[:, 7:][:, movable] ** 2).sum(-1)

    components = {
        "approach": W_APPROACH * approach,
        "finger_reach": W_REACH * finger_reach,
        "wrap": W_WRAP * wrap,
        "contact": W_CONTACT * contact,
        "envelope": W_ENVELOPE * envelope,
        "link_cover": W_LINKS * link_cover,
        "squeeze": W_SQUEEZE * squeeze,
        "lift": W_LIFT * lift,
        "goal": W_GOAL * goal,
        "hold_still": W_STILL * hold_still,
        "success_bonus": W_SUCCESS * success_bonus,
        "tilt_penalty": -W_TILT * tilt_pen,
        "push_penalty": -W_PUSH * push_pen,
        "table_penalty": -W_TABLE * table_pen,
        "arm_rate_penalty": -W_ARM_RATE * arm_rate,
        "hand_rate_penalty": -W_HAND_RATE * hand_rate,
    }
    reward = torch.stack(list(components.values()), dim=0).sum(dim=0)
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
- The hand approaches the cup from the side, places the palm against the side of the cup and closes the fingers around the cup body, so the hand does surround the cup. The fingers mostly press strongly on the cup with their fingertips rather than lying along it.
- While the cup is still on the table it stays upright inside this grasp.
- While lifting, the hand and the cup tilt: the cup is carried strongly tilted — roughly 45 degrees or more from upright, with its opening pointing sideways — and it is held tilted near the goal height. The arm does not rotate to bring the cup back upright. The same cycle (grasp, tilted lift, reset after about 4-5 s) repeats in every episode shown.
- The thumb's placement is not visible from this camera angle.
- Training metrics: successes per episode peaked at about 3.7 and have stayed between about 2.2 and 3.5 while the success tolerance tightened in steps from 0.1125 m to 0.074 m (latest value; recent successes about 2.8); the lifted fraction rose from 0.63 to 0.80; the mean cup tilt rose from about 24 to 27-30 degrees; on average 4.6 of 5 fingers and the palm (0.77 of the time) touch the cup, but only about 5 of the 15 measured links touch, i.e. about one link per finger; the hold_still component stayed at about 0.

To make the code more accurate and train better robot, the feedback for improvement is:
- Keep the grasp that surrounds the cup with the palm and the fingers — that part works.
- The cup must stay upright while it is lifted and held. The goal pose is upright and goal_dist includes the cup's orientation, so a tilted carry stops producing successes as the tolerance tightens. The arm should rotate as needed to keep the cup upright during lifting and holding; make uprightness count throughout lifting and holding, strongly enough that carrying the cup tilted is not worth it.
- Near the goal the cup is almost never held still (hold_still stays about 0); successes need the upright cup held steady at the goal for consecutive steps.
- The fingers mainly press with their fingertips (about one measured link per finger touches); the task asks for the links of every finger to touch the cup, so wrapping with more of each finger should be worth more while holding.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0.641, 0.809, 0.881, 0.861, 0.887, 0.916, 0.921, 0.929, 0.931, 0.932]  min 0.362 · mean 0.884 · max 0.943
contact/finger_middle: [0.66, 0.758, 0.876, 0.905, 0.904, 0.912, 0.937, 0.941, 0.948, 0.947]  min 0.344 · mean 0.892 · max 0.955
contact/finger_pinky: [0.495, 0.711, 0.787, 0.785, 0.756, 0.813, 0.857, 0.866, 0.89, 0.885]  min 0.138 · mean 0.8 · max 0.904
contact/finger_ring: [0.786, 0.772, 0.867, 0.883, 0.887, 0.92, 0.914, 0.933, 0.935, 0.935]  min 0.458 · mean 0.886 · max 0.945
contact/finger_thumb: [0.812, 0.873, 0.926, 0.925, 0.912, 0.929, 0.937, 0.944, 0.949, 0.944]  min 0.696 · mean 0.92 · max 0.956
contact/fingers_touching: [3.39, 3.92, 4.34, 4.36, 4.35, 4.49, 4.57, 4.61, 4.65, 4.64]  min 2.01 · mean 4.38 · max 4.7
contact/link_force_mean: [1.52, 3.5, 4.47, 4.45, 4.53, 4.41, 5.15, 4.71, 4.57, 4.53]  min 1.1 · mean 4.83 · max 508
contact/links_touching: [3.78, 4.68, 5.13, 5.18, 5.11, 5.21, 5.22, 5.11, 5.08, 5.08]  min 2.28 · mean 5.03 · max 5.36
contact/palm_touching: [0, 0.33, 0.569, 0.548, 0.615, 0.661, 0.775, 0.747, 0.758, 0.778]  min 0 · mean 0.627 · max 0.825
ctrl/prev_ep_successes_mean: [0, 8.14e-05, 0.00277, 2.12, 3.48, 3.5, 3.04, 3.21, 2.73, 2.29]  min 0 · mean 2.18 · max 3.67
done/abnormal: [0, 0.000488, 0.000407, 0.000488, 0.000244, 0.000326, 0.000163, 8.14e-05, 0.000244, 8.14e-05]  min 0 · mean 0.00023 · max 0.00179
done/fell: [8.14e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.71e-06 · max 0.000163
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.63e-07 · max 8.14e-05
done/max_goals: [0, 0, 0, 0.000977, 0.00326, 0.00277, 0.00106, 0.00138, 0.000732, 0.00114]  min 0 · mean 0.00104 · max 0.00374
done/out_xy: [8.14e-05, 8.14e-05, 0, 0, 0, 0, 8.14e-05, 0, 0, 0]  min 0 · mean 3.16e-05 · max 0.00057
done/tipped: [0.00806, 0.00293, 0.00163, 0.0013, 0.00138, 0.000814, 0.000732, 0.000244, 0.00138, 0.000977]  min 8.14e-05 · mean 0.00144 · max 0.0128
episode_lengths/step: [21.7, 171, 252, 282, 260, 290, 329, 345, 403, 429]  min 21.7 · mean 301 · max 446
reward/approach: [0.239, 0.496, 0.625, 0.617, 0.63, 0.654, 0.683, 0.682, 0.681, 0.676]  min 0.239 · mean 0.627 · max 0.697
reward/arm_rate_penalty: [-0.145, -0.122, -0.112, -0.101, -0.0786, -0.0657, -0.0634, -0.0554, -0.0582, -0.0591]  min -0.145 · mean -0.0815 · max -0.0542
reward/contact: [0.825, 1.03, 1.2, 1.21, 1.21, 1.27, 1.31, 1.34, 1.34, 1.32]  min 0.483 · mean 1.23 · max 1.36
reward/envelope: [0, 0.509, 0.99, 0.973, 1.1, 1.26, 1.43, 1.45, 1.44, 1.44]  min 0 · mean 1.13 · max 1.56
reward/finger_reach: [0.668, 0.759, 0.782, 0.754, 0.774, 0.796, 0.794, 0.796, 0.794, 0.793]  min 0.565 · mean 0.779 · max 0.804
reward/goal: [0, 0.0175, 0.0603, 0.581, 0.617, 0.755, 0.827, 0.931, 0.934, 0.938]  min 0 · mean 0.616 · max 1.1
reward/hand_rate_penalty: [-0.0668, -0.0452, -0.0317, -0.0282, -0.0234, -0.0213, -0.0189, -0.0165, -0.0157, -0.0145]  min -0.0668 · mean -0.0256 · max -0.0137
reward/hold_still: [0, 8.76e-07, 6.01e-06, 0.00026, 0.000332, 0.00049, 0.000585, 0.000949, 0.00113, 0.00113]  min 0 · mean 0.000535 · max 0.00175
reward/lift: [0.0255, 0.0656, 0.241, 1.8, 1.68, 1.99, 2.4, 2.57, 2.64, 2.63]  min 0.0134 · mean 1.73 · max 2.83
reward/link_cover: [0.243, 0.302, 0.33, 0.337, 0.33, 0.34, 0.341, 0.338, 0.333, 0.33]  min 0.144 · mean 0.326 · max 0.349
reward/push_penalty: [-0.527, -0.591, -0.631, -0.149, -0.172, -0.141, -0.121, -0.0881, -0.0864, -0.0907]  min -0.797 · mean -0.238 · max -0.0792
reward/squeeze: [0.154, 0.19, 0.221, 0.213, 0.205, 0.215, 0.212, 0.214, 0.219, 0.213]  min 0.0799 · mean 0.208 · max 0.236
reward/success_bonus: [0, 0, 0.000883, 0.148, 0.266, 0.217, 0.156, 0.137, 0.124, 0.128]  min 0 · mean 0.126 · max 0.33
reward/table_penalty: [0, -0.000436, 0, 0, -5.55e-05, 0, 0, 0, 0, 0]  min -0.00234 · mean -6.31e-05 · max 0
reward/tilt_penalty: [-0.198, -0.296, -0.162, -0.388, -0.391, -0.468, -0.48, -0.584, -0.591, -0.487]  min -0.639 · mean -0.427 · max -0.151
reward/total: [1.32, 2.71, 4.02, 6.45, 6.64, 7.35, 8.04, 8.27, 8.3, 8.36]  min 0.567 · mean 6.5 · max 8.9
reward/wrap: [0.108, 0.393, 0.509, 0.481, 0.508, 0.542, 0.56, 0.556, 0.551, 0.545]  min 0.108 · mean 0.503 · max 0.568
rewards/step: [-2.16, 418, 899, 1.61e+03, 1.65e+03, 1.98e+03, 2.44e+03, 2.64e+03, 3.2e+03, 3.48e+03]  min -2.16 · mean 2.02e+03 · max 3.58e+03
task/lifted_frac: [0, 0.0801, 0.161, 0.676, 0.6, 0.685, 0.747, 0.814, 0.814, 0.796]  min 0 · mean 0.585 · max 0.831
task/successes_mean: [0, 0.000244, 0.0124, 0.496, 0.52, 0.74, 0.698, 0.748, 0.802, 0.838]  min 0 · mean 0.505 · max 0.894
task/tilt_deg: [14.5, 19.1, 13.7, 23.7, 23.5, 26.4, 27, 30.4, 30.3, 27.3]  min 12 · mean 24.4 · max 31.6
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.101, 0.0911, 0.082, 0.082, 0.082]  min 0.0738 · mean 0.0984 · max 0.112

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
