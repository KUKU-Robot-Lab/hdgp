## 1. What the task means and the stages

The arm has to go from free-space motion to a power grasp on an upright cylinder whose size changes between environments, and then carry the cup. The stages, in order:

- **A. Oriented approach, hand open.** Move the palm centre from the raised start pose (about 0.38 m away) to the grasp spot. The spot is on the side of the cup, at its surface (radial gap about 0), in the upper part of the graspable band so the lower fingers stay off the table. On the way the palm normal turns to point at the cup axis, and the fingers become horizontal (perpendicular to the axis) so they can later go around the cup body. The four fingers stay open so they arrive around the cup, not between the palm and the cup.
- **B. Palm on the cup.** The palm surface presses lightly on the side of the upright cup. The wrist end, the thumb side or the back of the hand does not count. The cup must not be tipped or shoved while this happens.
- **C. Wrap.** With the palm on the cup, the fingers bend around the cup body inside the graspable band until their links touch it. The thumb closes on the opposite side of the cup from the other four fingers. More digits touching makes a better grasp; palm plus all five digits is the envelope.
- **D. Lift.** Holding that grasp, raise the cup until its whole bottom leaves the table, keeping it upright. Tilting the cup about its rim is not lifting.
- **E. Goal and hold.** Carry the cup to the goal 0.21-0.28 m above its start. Hold it still and upright so that `goal_dist <= success_tol` lasts `success_hold_steps` steps, and repeat until `max_successes`.
- **Throughout:** never knock the cup over or push it away while it is not grasped, never rest the hand or fingers on the table, and keep actions smooth.

## 2. What was missed or wrong in the last reward

1. **Hovering with an open hand was the best option.**
   - `finger_open` (max 0.10) was the largest term: 0.09 of a 0.18 total. It was multiplied by `far`, which drops to 0 exactly at the grasp spot.
   - `reach_coarse` had already saturated 5 cm from the spot (`relu(d - 0.05)`).
   - `reach_oriented` needed facing >= 0.9 and fingers within about 12 degrees at the same time, so it paid only 0.011.
   - Result: moving the palm the last 5-10 cm onto the cup lost up to 0.1 per step and gained almost nothing. The hand hovered 8-13 cm away all round.
2. **The decay made it worse.** The approach terms decayed to 20 % over the episode but `finger_open` did not. Late in the episode, hovering with an open hand was worth even more compared with approaching.
3. **Nothing paid for turning the palm toward the cup.**
   - The only ungated approach term ignored orientation.
   - The orientation-gated terms stayed at zero until the orientation was almost exact.
   - So an edge-on, thumb-first hand earned as much as a hand with the palm facing the cup, and the stretched thumb touched the cup first.
4. **Palm contact could not be reached.** It was a product of hard gates: exact orientation, tilt below 2.5-5 degrees, hand clear of the table, palm centre at the surface, and the decay. In the 12 % of steps where the palm did touch, it paid at most 0.001.
5. **Wrap, lift and the relaxed penalties were never reached, so they were never tested.** They are kept. The move into them is made cheaper: once a real grasp forms, the approach and palm terms count as done, and the cup-movement penalties relax with grasp quality.

## 3. Changes

- **`reach_far`**: no gate, and it keeps rising all the way to the spot (`1 - tanh(d/0.3)`, no 5 cm saturation). This keeps the fast move.
- **`reach_facing`**: a large (0.4), sharp (3 cm) term.
  - Scaled by how well the palm faces the cup: 0 at about 81 degrees, 1 within about 32 degrees.
  - Fingers that are not horizontal only scale it down to 40 %, so they shape the reward without zeroing it.
  - An edge-on or thumb-first hand gets 0.
- **`finger_open`**: small (0.04) and independent of palm position.
  - Open fingers are paid anywhere before the grasp.
  - Closing costs until the palm is seated at the spot, so closing there loses at most 0.04.
  - The thumb has 20 % weight, because the four fingers are what must stay open.
- **One shared decay**: every term that can be collected without a grasp (approach, orientation, open hand, palm contact) falls to 50 % over the step budget.
  - All of them use the same factor, so hovering < palm at the spot < palm pressed holds at every moment.
  - Resting against the cup still loses value over time.
  - A forming grasp removes the decay.
- **`palm_contact`**: palm force × palm centre at the cup surface × soft facing weight × soft upright factor (Gaussian with sigma 6 degrees: 0.78 at 3 degrees, 0.50 at 5 degrees).
  - The table is no longer a gate; it has its own penalty.
  - As a grasp forms, the upright and orientation conditions relax.
  - The palm term keeps half credit if the palm slides during the lift.
- **Wrap terms** (curl onto the cup, fingers touching counted superlinearly, links touching, envelope):
  - Gated by the palm being at the spot and facing the cup; no force is needed for the gate.
  - Weights 1.0 / 3.0 / 1.2 / 1.0. Palm plus thumb plus 2 fingers is worth about 2x palm alone; a full envelope about 4x.
  - A single digit, such as the thumb alone, earns about 0.
- **`grasp_q`** (0 with one digit) also requires the palm to be near the cup and roughly facing it. Fingertip pokes from behind cannot unlock the relaxed penalties or the lift reward.
- **Lift** is paid on the cup bottom's clearance above the table, so tilting is not lifting, and it is multiplied by `grasp_q`.
- **Cup-movement penalties without a grasp:**
  - Tilt costs from 3 degrees (full at 15).
  - Push costs from 1.5 cm (full at 6 cm), so the small slide from pressing the palm is almost free.
  - Both relax as the grasp forms.
- **Table penalty**: unchanged; it still measures how close finger links, fingertips and the palm come to the table.

Approximate per-step values at the start of an episode:

| State | Reward per step |
|---|---|
| Start pose | 0.07 |
| Hover 10 cm, open, facing the cup | 0.22 |
| Hover 5 cm | 0.30 |
| Thumb-first at the spot | 0.28 |
| Palm at the spot, facing the cup | 0.72 |
| Palm pressed on the upright cup | 1.4 |
| Palm + thumb + 2 fingers | about 2.7 |
| Palm + 5 digits | about 5.5 |
| Same grasp, cup 3 cm clear of the table | about 8 |
| Held still at the goal | about 16, plus about 24 per success |
| Palm leaning on a cup tilted 5 degrees, fingers on the table | about -0.2 |

## 4. Improved code

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
    # approx. per-step values at the start of an episode (contact indicators ~0.9, thumb opposed):
    #   start pose 0.07 | hover 10 cm open & facing 0.22 | hover 5 cm 0.30 | thumb-first at the spot 0.28
    #   palm at the spot facing the cup 0.72 | palm pressed on the upright cup 1.4
    #   palm + thumb + 2 fingers ~2.7 | palm + 5 digits ~5.5 | same grasp, cup 3 cm clear ~8
    #   held still at the goal ~16 (+ ~24 per success) | palm leaning on a 5 deg tilted cup, fingers on table ~ -0.2
    # stage A: approach (small share; all resting terms decay with the same factor, so the ordering holds all episode)
    W_REACH_FAR = 0.20        # ungated, keeps rising all the way to the grasp spot (keeps the fast move)
    W_REACH_FACING = 0.40     # sharp (3 cm) term, scaled by how well the palm faces the cup (edge-on -> 0)
    W_ORIENT = 0.08           # dense orientation shaping at any distance
    W_FINGER_OPEN = 0.04      # small and position independent; closing costs only until the palm is seated
    # stage B: palm surface on the upright cup
    W_PALM_CONTACT = 0.8      # ~2x the value of being at the spot without touching
    # stage C: wrap (gated by the palm at the spot and facing the cup, no force required for the gate)
    W_CURL_ONTO = 1.0         # flexion of digits lying on / touching the cup body (closing on air pays 0)
    W_WRAP_FINGERS = 3.0      # ((digits touching - 1) / 4)^1.5: 1 digit 0, 3 -> 1.06, 5 -> 3.0
    W_WRAP_LINKS = 1.2        # fraction of the 15 measured links touching (a real wrap, not fingertip pokes)
    W_ENVELOPE = 1.0          # all five digits touching with the thumb opposed, more with the palm pressing
    # stage D: lift
    W_LIFT = 6.0              # on top of the wrap terms: a lifted cup clearly beats a wrapped cup on the table
    # stage E: goal and hold
    W_GOAL_COARSE = 2.0
    W_GOAL_FINE = 2.0         # narrow basin, success_tol shrinks to 1.5 cm
    W_STILL = 1.5
    W_IN_TOL = 1.5
    W_SUCCESS = 25.0
    # penalties / regularisation
    W_TABLE = 1.0             # finger links / fingertips / palm coming down onto the table
    W_TILT = 1.5              # ungrasped: 3 deg dead band, full at 15 deg | grasped: 15 .. 35 deg
    W_PUSH = 1.0              # ungrasped on the table: 1.5 cm dead band, full at 6 cm | grasped: 8 .. 15 cm
    W_ACTION_RATE = 0.01
    W_JOINT_VEL = 0.005

    cup_r = ctx.cup_radius
    cup_hh = ctx.cup_half_height
    axis = ctx.cup_axis
    tilt = ctx.cup_tilt
    progress = ctx.episode_progress.clamp(0.0, 1.0)
    lifted_f = ctx.lifted.float()

    # ================================================================== cup clearance above the table
    # tipping the cup about its bottom rim raises cup_pos by ~R*sin(tilt) but keeps the rim on the table,
    # so this clearance stays ~0 when the cup is only tilted
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
    # grasp spot: palm centre within [-1.0, +1.2] cm of the surface, height +0.05*H .. +0.55*H
    # (upper part of the band, so the lower fingers stay clear of the table)
    e_r = torch.relu(gap_p - 0.012) + torch.relu(-0.010 - gap_p)
    e_h = torch.relu((h_p - 0.3 * cup_hh).abs() - 0.25 * cup_hh)
    d_palm = torch.sqrt(e_r ** 2 + e_h ** 2 + eps)

    n = ctx.palm_normal
    facing = -(n * u_p).sum(-1)                               # +1 when the palm normal points at the cup axis
    # finger direction: palm centre -> index/middle/ring links, projected into the palm plane
    base = ctx.link_pos[:, 1:4, 0, :].mean(dim=1) - ctx.palm_pos
    base = base - (base * n).sum(-1, keepdim=True) * n
    s_f = (_unit(base) * axis).sum(-1).abs()                  # sin of the finger elevation out of the cup cross-section
    # soft orientation weight: palm facing 0 at ~81 deg, 1 within ~32 deg (edge-on / thumb-first -> 0);
    # fingers not horizontal only scale it down to 40 % (shapes the reward, does not zero it)
    face_w = _ramp(facing, 0.15, 0.85)
    level_w = 1.0 - _ramp(s_f, 0.3, 0.7)
    orient_w = face_w * (0.4 + 0.6 * level_w)
    # loose orientation (while wrapping / lifting the hand may rotate a little)
    orient_loose = _ramp(facing, 0.1, 0.5) * (1.0 - _ramp(s_f, 0.5, 0.85))

    # palm seated at the grasp spot and facing the cup (no force needed): 1 within 1 cm, 0.64 at 3 cm, 0.17 at 5 cm
    spot = torch.exp(-(torch.relu(d_palm - 0.01) / 0.03) ** 2)
    seat = orient_loose * spot
    # looser version used to accept a grasp: palm roughly facing the cup and near its body
    seat_loose = _ramp(facing, 0.0, 0.4) \
        * torch.exp(-(torch.relu(gap_p - 0.02) / 0.05) ** 2) \
        * torch.exp(-(torch.relu(h_p.abs() - cup_hh) / 0.04) ** 2)

    # ================================================================== hand coming down onto the table
    # link frames sit ~half a finger thickness above the surface when a finger lies on the table
    link_clear = ctx.link_pos[..., 2] - ctx.table_z                                   # (N,5,3)
    link_low = (1.0 - _ramp(link_clear, 0.006, 0.018)).flatten(1).max(dim=-1).values   # 0 at >= 18 mm, 1 at <= 6 mm
    other_low = 1.0 - _ramp(ctx.hand_z_min - ctx.table_z, 0.006, 0.018)               # any other hand link
    palm_low = 1.0 - _ramp(ctx.palm_pos[:, 2] - ctx.table_z, 0.010, 0.030)            # palm lying on the table
    table_near = torch.maximum(torch.maximum(link_low, other_low), palm_low)          # (N,) in [0, 1]

    # ================================================================== digit contacts on the cup body
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

    # grasp quality: 0 with a single digit, ~1 with the thumb opposed + 3 fingers (4 fingers without thumb -> 0.3);
    # requires the palm near and roughly facing the cup, so fingertip pokes from behind do not count
    grasp_q = _ramp(n_touch, 1.0, 4.0) * (0.3 + 0.7 * opp_q) * seat_loose
    relief = torch.sqrt(grasp_q + eps)                      # rises early in the transition to a wrap

    # ================================================================== resting decay
    # every term that can be collected without a grasp falls to 50 % over the step budget with the SAME factor,
    # so hover < spot < palm pressed holds at every moment; a forming grasp releases the decay
    rest_decay = 1.0 - 0.5 * progress * (1.0 - grasp_q)
    wrap_decay = 1.0 - 0.25 * progress * on_table            # a wrap left on the table slowly loses value

    # ================================================================== stage A: approach with an open hand
    reach_far = 1.0 - torch.tanh(d_palm / 0.3)               # 0.15 at the start, 0.84 at 5 cm, 1 at the spot
    reach_facing = orient_w * (1.0 - torch.tanh(d_palm / 0.03))
    palm_orient = 0.5 * (facing + 1.0) * (1.0 - s_f) * (1.0 - torch.tanh(d_palm / 0.2))
    # once a real grasp holds the cup, the approach stage counts as complete (the palm may shift while lifting)
    reach_far_c = reach_far + grasp_q * (1.0 - reach_far)
    reach_facing_c = reach_facing + grasp_q * (1.0 - reach_facing)
    palm_orient_c = palm_orient + grasp_q * (1.0 - palm_orient)

    # open hand: open fingers pay anywhere before the grasp (no dependence on staying away from the spot);
    # closing costs until the palm is seated, so at the spot closing only gives up the small open reward
    four = [4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18]
    thumb = [1, 2]
    close_tgt = 0.8 * ctx.hand_target_norm[:, four].mean(-1) + 0.2 * ctx.hand_target_norm[:, thumb].mean(-1)
    pre_grasp = (1.0 - grasp_q) * (1.0 - lifted_f) * on_table
    finger_open = pre_grasp * ((1.0 - close_tgt) - (1.0 - seat) * close_tgt)

    # ================================================================== stage B: palm surface on the upright cup
    valid_p = torch.clamp(1.0 - torch.relu(gap_p - 0.025) / 0.015, 0.0, 1.0) \
        * torch.clamp(1.0 - torch.relu(-0.020 - gap_p) / 0.015, 0.0, 1.0) \
        * torch.clamp(1.0 - torch.relu(h_p.abs() - cup_hh) / 0.02, 0.0, 1.0)
    palm_touch = _touch(ctx.palm_cup_force, 0.5) * valid_p                   # palm force with its centre at the surface
    up_pre = torch.exp(-(tilt / (6.0 * deg)) ** 2)                          # before a grasp: 0.78 at 3 deg, 0.50 at 5 deg
    up_grasp = 1.0 - _ramp(tilt, 10.0 * deg, 25.0 * deg)                    # in a grasp: 1 below 10 deg, 0 at 25 deg
    up_palm = up_pre + grasp_q * (up_grasp - up_pre)                        # relaxes only as the grasp forms
    face_palm = orient_w + grasp_q * (torch.maximum(orient_w, orient_loose) - orient_w)
    palm_on = palm_touch * face_palm * up_palm                              # soft factors, no hard zero gates
    palm_credit = palm_on + (1.0 - palm_on) * 0.5 * grasp_q                 # half credit if the palm slides in a grasp

    # ================================================================== stage C: digits wrapped on the cup body
    seat_c = torch.maximum(seat, airborne * seat_loose)     # once the cup is up, a slightly shifted palm still counts
    wrap_gate = seat_c * up_grasp * wrap_decay

    finger_joints = [[1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18]]
    curl = torch.stack([ctx.hand_q_norm[:, j].mean(-1) for j in finger_joints], dim=-1)   # (N,5) measured flexion
    near_l = torch.exp(-(torch.relu(gap_l - 0.012) / 0.03) ** 2) * in_band_l * outside_l
    near_f = near_l[:, :, 1:].mean(-1)                      # (N,5) outer link + fingertip on the cup body
    curl_onto = (curl * (0.3 * near_f + 0.7 * finger_c)).mean(-1)            # straight fingers or closing on air -> 0
    wrap_fingers = _ramp(n_touch, 1.0, 5.0) ** 1.5 * (0.4 + 0.6 * opp_q)    # superlinear, a single digit = 0
    wrap_links = link_c.mean(dim=(-2, -1))
    envelope = finger_c.min(dim=-1).values * opp_q * (0.5 + 0.5 * palm_touch)

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
    # cup motion: strict without a grasp (tilt from 3 deg, slide from 1.5 cm), lenient while a grasp holds the cup
    tilt_pen = (1.0 - relief) * _ramp(tilt, 3.0 * deg, 15.0 * deg) + relief * _ramp(tilt, 15.0 * deg, 35.0 * deg)
    disp_xy = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    push_pen = (1.0 - relief) * on_table * _ramp(disp_xy, 0.015, 0.06) + relief * _ramp(disp_xy, 0.08, 0.15)
    action_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(-1)
    joint_vel = (ctx.arm_qd ** 2).mean(-1) + 0.1 * (ctx.hand_qd ** 2).mean(-1)

    components = {
        "reach_far": W_REACH_FAR * rest_decay * reach_far_c,
        "reach_facing": W_REACH_FACING * rest_decay * reach_facing_c,
        "palm_orient": W_ORIENT * rest_decay * palm_orient_c,
        "finger_open": W_FINGER_OPEN * rest_decay * finger_open,
        "palm_contact": W_PALM_CONTACT * rest_decay * palm_credit,
        "finger_curl_onto_cup": W_CURL_ONTO * wrap_gate * curl_onto,
        "wrap_fingers": W_WRAP_FINGERS * wrap_gate * wrap_fingers,
        "wrap_links": W_WRAP_LINKS * wrap_gate * wrap_links,
        "envelope": W_ENVELOPE * wrap_gate * envelope,
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
