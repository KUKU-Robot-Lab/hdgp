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
