# Reward design: staged approach, envelope grasp, lift (right OpenArm + DG-5F)

## 1. What the task means

**Geometry of the start orientation.** The palm is vertical: `palm_normal` points along +y
towards the cup and `palm_finger_dir` along +x. So `palm_side` is vertical, and the four
fingers lie horizontally in the palm plane, stacked on top of each other. In the default
pose the opposed thumb sticks straight out of the palm along +y, about 0.12 m, at the wrist
end (the -x end). With this shape the hand forms an "L": the palm and the straight fingers
make one wall, and the thumb makes a second wall at the wrist end. The cup has to sit
inside that corner:

- on the palm's +y side, with the palm plane about 5 mm from the cup surface
  (target `gap = s_n - r = 0.005`, env window [-0.01, 0.02]);
- ahead of `palm_pos` along the fingers by about `r + 0.0075`
  (env window [r - 0.005, r + 0.02]), so the thumb stays behind the cup's -x side;
- at the height of the graspable band.

The corner is open upwards and towards +x/+y. Two paths reach it without touching:

- **Raised start:** first line up horizontally (above the cup), then lower the hand. The
  fingers slide down the cup's -y side and the thumb slides down its -x side.
- **Beside-cup start:** move about 3.5 cm in +y and about 2.5 cm in +x. The thumb has extra
  clearance here, so this path is safe.

Coming in from -x at the wrong height, or overshooting in +x, puts the thumb or fingers into
the cup. That is why the approach needs a contact penalty and a cup-disturbance factor, not
just a distance term.

**Actions.** Arm actions are joint increments, so the arm is slow and smooth (at most about
0.3 rad/s per joint). Finger actions are *absolute* targets: a = -1 means straight, which is
the default pose. So during the approach the policy has to hold every movable finger action
near -1. Only 13 of the 19 finger joints can move (thumb_3/4, index/middle/ring _2.._4,
pinky_3/4). Fingers are PD-controlled, so a finger that reaches the cup stops there and
presses harder the further its target is past its actual angle. That makes "contact force
above a small threshold" the right way to detect that a digit holds the cup.

**The three stages, as the environment sees them**

| stage | active when | goal | what ends it |
|---|---|---|---|
| 1 approach | `~approach_done` | reach the corner pose with the start orientation, the default finger pose and no finger or thumb contact | env latches `approach_done` |
| 2 envelope | `approach_done & ~envelope_done` | stay in place; close the palm, thumb and at least 3 fingers onto the cup (fingers wrap round the +x/far side, thumb presses the near -x side) | env latches `envelope_done` (palm + thumb + at least 4 digits for 5 steps) |
| 3 lift | `approach_done & envelope_done` | keep the grasp, lift the cup 0.21 to 0.28 m straight up, hold it upright and still inside `success_tol` | successes counted; the episode ends at `max_successes` |

## 2. How the stage ordering is enforced

1. **The stage comes only from the environment's latches.** Each stage's shaping term is
   computed only while its stage is active (`torch.where` masks). The lift and goal terms
   and the success bonus exist only in stage 3. The contact and closure terms exist only
   from stage 2 on. Before `approach_done`, closing the fingers or lifting the cup pays
   nothing, and it lowers the stage-1 reward through the pose factor, the disturbance
   factor and the contact penalty.
2. **Stages never lose value.** When a stage is complete, its term is replaced by a
   constant equal to the upper bound of its shaping: `S1 = 1.0` for the approach and
   `S2 = 1.5` for the envelope. Every shaping term is built to stay within
   `[0, its bound]`. So entering the next stage can never lower the reward, and the policy
   has no reason to stall in an earlier stage.
3. **Later stages still check the earlier ones' conditions.** Stage 2 rewards staying in the
   approach pose and orientation (`keep`), and its contact terms are scaled by `keep`.
   Stage 3's lift reward is multiplied by the *current* grasp quality `g` (palm, thumb and
   finger contacts). Lifting the cup by scooping, balancing or pinching earns much less than
   lifting it in the envelope grasp.
4. **The cup must stay put until stage 3.** In stages 1 and 2 the shaping is multiplied by a
   "cup undisturbed" factor. It covers horizontal push, vertical displacement and tilt. The
   stage-2 version has looser dead zones, because closing the hand shifts the cup a little.
   Lifting or pushing the cup before `envelope_done` therefore lowers the reward.

## 3. Terms

### Stage 1: approach (`approach`, at most S1 = 1)

The target palm position uses the *desired* orientation (world x/y), relative to the cup
axis:

- `e_x = rad_x + (r + 0.0075)` and `e_y = rad_y + (r + 0.005)`, where `rad` is the vector
  from the cup axis to the palm;
- height error: zero while the palm's axial coordinate `h` is in `[0, 0.6 * half_height]`,
  otherwise the distance to that range. The target is the upper part of the band, which
  also keeps the lower fingers clear of the table.
- `r_xy = 0.5 * (1 - tanh(d_xy / 0.25)) + 0.5 * exp(-d_xy / 0.02)`: the coarse part gives a
  gradient from 0.38 m away, the fine part gives about 1 cm precision.
- `r_pos = r_xy * (0.25 + 0.75 * r_z)`: height is rewarded mostly once the hand is lined up
  horizontally. This favours "line up, then lower" over diagonal paths that sweep the
  fingers through the cup. It does not push the hand back up in the beside-cup start.
- `r_ori = exp(-((1 - n_y) + (1 - f_x)) / 0.1)`: about 0.74 at 10 degrees error on both
  axes, about 0.30 at 20 degrees.
- `r_pose` from the deviation of the 13 movable joints from `hand_default_q_norm`
  (mean and max; the env limit is 0.15).
- `r_ready = exp(-(d_xy + d_z) / 0.01) * r_ori * pose_ok`: all latch conditions at once.

Formula:
`R1 = S1 * (0.7 * r_pos + 0.3 * r_ready) * (0.3 + 0.7 * r_ori * r_pose) * (0.5 + 0.5 * still1)`.

Orientation and pose *scale* the progress reward instead of being added to it. Staying at
the start therefore earns almost nothing, and turning the hand or bending a finger costs
progress along the whole path.

`touch_penalty = -0.3 * max soft contact of any finger or thumb link`, active before
`approach_done` only.

### Stage 2: envelope grasp (`envelope`, at most S2 = 1.5)

- `keep`: palm-frame version of the latch pose (the palm may press in down to gap -0.015;
  about 1.5 cm tolerance along the fingers; palm inside the band), times `r_ori`.
- `near`: for each digit, the smallest distance from its three measured links to the cup
  surface inside the band, turned into `exp(-d / 0.015)`. The thumb's value is multiplied
  by `thumb_side`, which is 1 when the thumb tip is on the wrist (-x) side of the axis,
  opposite the fingers.
- `wrap`: for each of the four fingertips, how far round the cup it has travelled from the
  palm side: `(1 - cos(phi)) / 2`, where phi is measured around the axis from the palm.
  It is 0.5 at the +x side and 1 at the far side, and is multiplied by fingertip nearness.
  Thick cups cannot reach the far side, so this term is small and additive, never a gate.
- `contact_score = 0.25 * palm + 0.25 * thumb + 0.5 * n_fingers / 4`, with soft contacts
  `clamp(F / 0.5, 0, 1)`. The clamp also removes force spikes.
- `envelope_now`: 1 when the environment's latch condition holds on this step (palm, thumb
  and at least 3 fingers above 0.1 N). Holding it for 5 steps completes the stage.

Formula: `R2 = S2 * (0.2 * keep + 0.2 * near + 0.15 * wrap + (0.3 * contact_score + 0.15 * envelope_now) * (0.4 + 0.6 * keep)) * (0.5 + 0.5 * still2)`.

### Stage 3: lift and hold (`lift`, at most R3_MAX = 2)

- `g = (palm_c + thumb_c + min(n_fingers / 3, 1)) / 3`: the grasp must be kept.
- `lift_frac = clamp((cup_z - spawn_z) / (goal_z - spawn_z), 0, 1)`: a linear ramp, so the
  gradient is the same over the whole lift.
- `r_goal = 1 - tanh(goal_dist / 0.1)` (coarse) and `r_goal_fine = exp(-goal_dist / 0.015)`
  (fine). `goal_dist` already includes tilt.
- `hold = 1[goal_dist <= success_tol] * exp(-|v| / 0.05) * exp(-|w| / 0.5)`: inside the
  tolerance and still.
- `upright = exp(-(tilt / 0.5)^2)`.

Formula: `R3 = R3_MAX * g * upright * (0.3 * lift_frac + 0.3 * r_goal + 0.2 * r_goal_fine + 0.2 * hold)`.

### Bonuses and penalties (every stage)

- `success_bonus = 10 * (0.5 + 0.5 * g)` on `ctx.success`, stage 3 only. A success reached
  by skipping the grasp stage earns nothing.
- `final_success_bonus`: the success that ends the episode (`num_successes >= max_successes - 1`)
  pays `100 * (S1 + S2 + R3)`. That is roughly the value of the per-step reward the policy
  gives up when the episode ends, assuming gamma is about 0.99. Without it, a positive
  per-step reward would teach the policy to hover just outside `success_tol` so the episode
  never ends. If the counter already includes the current success, the bonus is also paid
  on the second-to-last success, which is harmless.
- `table_penalty = -0.5 * clamp((table_z + 0.01 - hand_z_min) / 0.04, 0, 1)`: warns before
  the `table_z - 0.03` termination.
- `action_rate_penalty = -0.02 * mean((a - a_prev)^2)` (at most -0.08): smooth commands,
  which matter with the random 0 to 2 step action delay.
- `arm_vel_penalty = -0.005 * min(sum(arm_qd^2), 10)`: damps arm oscillation; the clamp
  keeps physics spikes out.

**Terminations.** Knocking the cup over, dropping it off the table and pushing the hand
through the table all end the episode. The per-step reward is positive in every stage, so
ending early always loses the rest of the episode's reward and is avoided without an
explicit terminal penalty. The penalties above stay small (about 0.9 at most combined), so
they never make "end the episode" attractive.

**Reward scale per step:** stage 1 up to 1.0; stage 2 up to 2.5; stage 3 up to 4.5; plus
success bonuses.

Diagnostics that are not part of the sum are returned with a `diag_` prefix.

```python
import torch
import math

# Indices (into the 19 hand joints) of the joints whose range is wider than 0.05 rad:
# thumb_3, thumb_4, index_2..4, middle_2..4, ring_2..4, pinky_3, pinky_4
_MOVABLE_HAND_JOINTS = (1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _soft_contact(force: torch.Tensor) -> torch.Tensor:
    # 0 N -> 0, >= 0.5 N -> 1; saturating also removes the effect of physics force spikes
    return (force / 0.5).clamp(0.0, 1.0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ------------------------------------------------------------------ constants
    S1 = 1.0              # value of a completed approach = upper bound of the stage-1 shaping
    S2 = 1.5              # value of a completed envelope grasp = upper bound of the stage-2 shaping
    R3_MAX = 2.0          # upper bound of the stage-3 shaping
    SUCCESS_BONUS = 10.0  # per counted success (success_hold_steps of holding)
    CONT_HORIZON = 100.0  # ~1/(1-gamma), gamma~0.99: continuation value paid at the episode-ending success
    CONTACT_N = 0.1       # the environment's contact threshold [N]

    device = ctx.palm_pos.device
    dtype = ctx.palm_pos.dtype
    zeros = torch.zeros_like(ctx.cup_radius)

    # ------------------------------------------------------------------ stage masks (env latches only)
    app = ctx.approach_done.bool()
    env = ctx.envelope_done.bool()
    stage1 = ~app
    stage2 = app & ~env
    stage3 = app & env

    # ------------------------------------------------------------------ shared geometry
    r = ctx.cup_radius
    hh = ctx.cup_half_height
    axis = ctx.cup_axis
    v_p = ctx.palm_pos - ctx.cup_pos
    h_p = (v_p * axis).sum(-1)                        # palm height along the cup axis
    rad_p = v_p - h_p.unsqueeze(-1) * axis            # cup axis -> palm, perpendicular to the axis

    disp = ctx.cup_pos - ctx.cup_spawn_pos
    disp_xy = disp[:, :2].norm(dim=-1)
    disp_z = disp[:, 2].abs()
    tilt = ctx.cup_tilt

    # orientation: palm_normal along +y, palm_finger_dir along +x
    ori_err = (1.0 - ctx.palm_normal[:, 1]) + (1.0 - ctx.palm_finger_dir[:, 0])
    r_ori = torch.exp(-ori_err / 0.1)                 # ~0.74 at 10 deg on both axes, ~0.30 at 20 deg

    # default finger pose (movable joints only)
    idx = torch.tensor(_MOVABLE_HAND_JOINTS, device=device, dtype=torch.long)
    pose_dev = (ctx.hand_q_norm - ctx.hand_default_q_norm).index_select(1, idx).abs()
    max_dev = pose_dev.max(dim=-1).values
    mean_dev = pose_dev.mean(dim=-1)
    pose_ok = torch.exp(-(max_dev / 0.1) ** 2)        # env limit 0.15 -> 0.11
    r_pose = 0.5 * torch.exp(-mean_dev / 0.05) + 0.5 * pose_ok

    # contacts with the cup
    link_c = _soft_contact(ctx.link_cup_force)        # (N,5,3)
    digit_c = link_c.max(dim=-1).values               # (N,5)
    palm_c = _soft_contact(ctx.palm_cup_force)        # (N,)
    thumb_c = digit_c[:, 0]
    n_fingers_c = digit_c[:, 1:].sum(-1)              # 0..4
    digit_force = ctx.link_cup_force.max(dim=-1).values
    envelope_now = ((ctx.palm_cup_force > CONTACT_N)
                    & (digit_force[:, 0] > CONTACT_N)
                    & ((digit_force[:, 1:] > CONTACT_N).sum(-1) >= 3)).to(dtype)

    # ------------------------------------------------------------------ stage 1: approach
    # target: cup ahead along +x by r+0.0075, palm plane 5 mm from the cup's -y side,
    # palm in the upper part of the graspable band
    e_x = rad_p[:, 0] + (r + 0.0075)
    e_y = rad_p[:, 1] + (r + 0.005)
    d_xy = torch.sqrt(e_x * e_x + e_y * e_y + 1e-12)
    d_z = torch.relu(-h_p) + torch.relu(h_p - 0.6 * hh)
    r_xy = 0.5 * (1.0 - torch.tanh(d_xy / 0.25)) + 0.5 * torch.exp(-d_xy / 0.02)  # far gradient + cm precision
    r_z = 0.5 * (1.0 - torch.tanh(d_z / 0.15)) + 0.5 * torch.exp(-d_z / 0.02)
    r_pos = r_xy * (0.25 + 0.75 * r_z)                # line up horizontally first, then lower
    r_ready = torch.exp(-(d_xy + d_z) / 0.01) * r_ori * pose_ok
    still1 = (torch.exp(-torch.relu(disp_xy - 0.005) / 0.02)
              * torch.exp(-torch.relu(disp_z - 0.005) / 0.01)
              * torch.exp(-torch.relu(tilt - 0.03) / 0.1))
    r1 = S1 * (0.7 * r_pos + 0.3 * r_ready) * (0.3 + 0.7 * r_ori * r_pose) * (0.5 + 0.5 * still1)

    # ------------------------------------------------------------------ stage 2: envelope grasp
    w = -rad_p                                        # palm -> cup axis
    s_n = (w * ctx.palm_normal).sum(-1)
    s_f = (w * ctx.palm_finger_dir).sum(-1)
    gap = s_n - r
    e_keep = (torch.relu(gap - 0.005) + torch.relu(-0.015 - gap)
              + torch.relu((s_f - (r + 0.0075)).abs() - 0.015)
              + torch.relu(h_p.abs() - hh))
    keep = torch.exp(-e_keep / 0.02) * r_ori

    v_l = ctx.link_pos - ctx.cup_pos[:, None, None, :]
    axis_l = axis[:, None, None, :]
    h_l = (v_l * axis_l).sum(-1)                      # (N,5,3)
    rad_l = v_l - h_l.unsqueeze(-1) * axis_l          # (N,5,3,3)
    surf_d = (rad_l.norm(dim=-1) - r[:, None, None]).abs() + torch.relu(h_l.abs() - hh[:, None, None])
    near_digit = torch.exp(-surf_d.min(dim=-1).values / 0.015)   # (N,5)
    near_tip = torch.exp(-surf_d[:, :, 2] / 0.015)                # (N,5)
    u_palm = _unit(rad_p)
    u_tip = _unit(rad_l[:, :, 2, :])                  # (N,5,3)
    cos_phi = (u_tip * u_palm[:, None, :]).sum(-1)    # angle round the axis from the palm side
    wrap = ((1.0 - cos_phi[:, 1:]) * 0.5 * near_tip[:, 1:]).mean(-1)
    thumb_side = (-(u_tip[:, 0] * ctx.palm_finger_dir).sum(-1) / 0.5).clamp(0.0, 1.0)  # thumb on the wrist side
    near_mean = (near_digit[:, 0] * thumb_side + near_digit[:, 1:].sum(-1)) / 5.0
    contact_score = 0.25 * palm_c + 0.25 * thumb_c + 0.5 * n_fingers_c / 4.0
    still2 = (torch.exp(-torch.relu(disp_xy - 0.02) / 0.03)
              * torch.exp(-torch.relu(disp_z - 0.01) / 0.02)
              * torch.exp(-torch.relu(tilt - 0.05) / 0.15))
    r2 = S2 * (0.2 * keep + 0.2 * near_mean + 0.15 * wrap
               + (0.3 * contact_score + 0.15 * envelope_now) * (0.4 + 0.6 * keep)) * (0.5 + 0.5 * still2)

    # ------------------------------------------------------------------ stage 3: lift and hold
    g = (palm_c + thumb_c + (n_fingers_c / 3.0).clamp(max=1.0)) / 3.0
    goal_h = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    lift_frac = ((ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]) / goal_h).clamp(0.0, 1.0)
    r_goal = 1.0 - torch.tanh(ctx.goal_dist / 0.1)
    r_goal_fine = torch.exp(-ctx.goal_dist / 0.015)
    in_tol = (ctx.goal_dist <= ctx.success_tol).to(dtype)
    still_goal = torch.exp(-ctx.cup_lin_vel.norm(dim=-1) / 0.05) * torch.exp(-ctx.cup_ang_vel.norm(dim=-1) / 0.5)
    upright = torch.exp(-(tilt / 0.5) ** 2)
    r3 = R3_MAX * g * upright * (0.3 * lift_frac + 0.3 * r_goal + 0.2 * r_goal_fine + 0.2 * in_tol * still_goal)

    # ------------------------------------------------------------------ reward terms
    approach = torch.where(stage1, r1, torch.full_like(r1, S1))
    envelope = torch.where(stage3, torch.full_like(r2, S2), torch.where(stage2, r2, zeros))
    lift = torch.where(stage3, r3, zeros)

    success = ctx.success.bool() & stage3
    success_bonus = torch.where(success, SUCCESS_BONUS * (0.5 + 0.5 * g), zeros)
    final = success & (ctx.num_successes >= ctx.max_successes - 1)
    final_success_bonus = torch.where(final, CONT_HORIZON * (S1 + S2 + r3), zeros)

    touch_penalty = torch.where(stage1, -0.3 * digit_c.max(dim=-1).values, zeros)
    table_penalty = -0.5 * ((ctx.table_z + 0.01 - ctx.hand_z_min) / 0.04).clamp(0.0, 1.0)
    action_rate_penalty = -0.02 * ((ctx.actions - ctx.prev_actions) ** 2).mean(-1)
    arm_vel_penalty = -0.005 * (ctx.arm_qd ** 2).sum(-1).clamp(max=10.0)

    reward = (approach + envelope + lift + success_bonus + final_success_bonus
              + touch_penalty + table_penalty + action_rate_penalty + arm_vel_penalty)

    stage_idx = stage2.to(dtype) + 2.0 * stage3.to(dtype)
    components = {
        "approach": approach,
        "envelope": envelope,
        "lift": lift,
        "success_bonus": success_bonus,
        "final_success_bonus": final_success_bonus,
        "touch_penalty": touch_penalty,
        "table_penalty": table_penalty,
        "action_rate_penalty": action_rate_penalty,
        "arm_vel_penalty": arm_vel_penalty,
        "diag_stage": stage_idx,
        "diag_approach_pos_err": d_xy + d_z,
        "diag_orientation": r_ori,
        "diag_pose_max_dev": max_dev,
        "diag_cup_still": still1,
        "diag_contact_score": contact_score,
        "diag_envelope_now": envelope_now,
        "diag_grasp_quality": g,
        "diag_lift_frac": lift_frac,
    }
    return reward, components
```
