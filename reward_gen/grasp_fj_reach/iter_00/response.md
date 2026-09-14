## 1. What the task means, step by step

The right arm starts raised beside the table. The palm is outside the near table edge, about 0.25 m above the table top, turned sideways, with the fingers pointing forward (+x) and the cup roughly 0.38 m away. The robot has to pick up an upright cup with a power (envelope) grasp and then hold it still, upright, at a goal 0.21-0.28 m above the cup's starting position. There is no grasp primitive. The policy moves the arm joints in small increments and sets every finger joint target directly, so the reward has to teach the whole sequence.

**Stage A: reach and pre-shape.**
- Move the palm to a spot right next to the cup's side surface, at the height of the graspable band (`|h| <= cup_half_height`).
- Turn the palm so that `palm_normal` is horizontal and points at the cup axis. Given the start pose, the natural way to do this is to slide the sideways-turned hand forward alongside the cup.
- Keep the fingers open while approaching. With `a = 0` the fingers are already half closed, and half-closed fingers would hit and push the cup.
- Do not tip or shove the cup, and keep every hand link above the table.

**Stage B: envelope grasp.**
- Once the palm is against the cup side, flex all movable finger joints so the finger links lie on the cup surface inside the band.
- The four fingers should wrap around one side of the cup and the thumb around the other side. Relative to the plane through the cup axis and the palm, the thumb tip and the four fingertips should be on opposite sides.
- The palm and every finger (the measured links) should be in contact with the cup's outer surface. Contacts from inside an open cup or on its rim do not count.
- Closing the targets past the contact angle produces squeeze force, because the fingers are PD-controlled.

**Stage C: lift.**
- With a secure grasp, raise the cup from the table towards the goal height and keep it upright.
- A lift with a weak grasp (only a pinch, or no contact) should pay much less than a lift with a full envelope.

**Stage D: transport and hold.**
- Bring the cup to `goal_pos`, which is at most 5 cm away horizontally. `goal_dist` also penalises tilt.
- Hold the cup still there so that `goal_dist <= success_tol` holds for `success_hold_steps` steps, over and over. Each time, the environment counts a success.

**Throughout the episode:** don't knock the cup over (the episode ends above 60 deg of tilt), don't drop it, don't push the hand into the table, and avoid jerky actions.

### How the reward follows these stages

Most terms are positive and bounded. Uprightness and grasp quality are used as multiplicative factors instead of large negative penalties, so ending the episode early (for example by knocking the cup over) never becomes attractive.

| Stage | Terms and what they reward |
|---|---|
| A | `reach` pulls the palm to a dead band 0-3 cm outside the cup surface, at band height. It has a coarse and a fine tanh scale, so there is a gradient from 0.38 m away.<br>`palm_orient` rewards a palm normal that is horizontal and faces the axis; it switches on as the palm gets closer.<br>`finger_preshape` rewards open finger targets while far from the cup and closed targets once the palm is at the cup and facing it. |
| B | `finger_wrap` rewards finger links lying on the cup surface inside the band.<br>`thumb_opposition` rewards the thumb tip and the four fingertips being on opposite sides of the axis-palm plane.<br>`palm_contact`, `finger_contact` and `link_contact` are saturating contact indicators. They are robust to force spikes and only count contacts on the outer surface inside the band.<br>`full_envelope` = palm contact x the minimum over the five fingers, so it pays only when the palm and all five fingers touch.<br>All contact terms are scaled by uprightness. |
| C | `lift` = height progress towards the goal height x uprightness x `hold_q` = `0.25 + 0.75 * grasp_quality`, where grasp quality is the fraction of palm plus five fingers touching the cup. |
| D | `goal_coarse`, `goal_fine`, `hold_still` and `in_tolerance` are active after `lifted` and also scaled by grasp quality.<br>`success_bonus` pays on every counted success. |
| Regularisation | `table_penalty`: soft ramp as the lowest hand link approaches the termination height.<br>`cup_push_penalty`: horizontal cup displacement before lifting.<br>`action_rate_penalty` and `joint_vel_penalty`: small smoothness terms. |

A full envelope before lifting pays about 6-7 per step. Lifting and holding at the goal adds about 9 more per step, plus the success bonus. The robot therefore always gains by moving on to the next stage.

## 2. Reward function

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
