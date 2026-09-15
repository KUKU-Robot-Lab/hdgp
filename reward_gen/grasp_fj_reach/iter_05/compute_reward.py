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
    # approx. per-step values (contact indicators ~0.9):
    #   hover at the spot 0.7 | palm pressed 1.4 | current grasp (thumb+middle inner, index/ring backs) on table ~1.0
    #   full envelope on table ~7 | current grasp carried 30 cm away ~2 | full envelope carried away ~10
    #   current grasp at the goal ~12 | full envelope still at the goal ~27 (+ ~25 per success)
    # stage A: approach (unchanged)
    W_REACH_FAR = 0.20        # ungated, rises all the way to the grasp spot (keeps the fast move)
    W_REACH_FACING = 0.40     # sharp (3 cm) term, scaled by how well the palm faces the cup
    W_ORIENT = 0.08           # dense orientation shaping
    W_FINGER_OPEN = 0.04      # small, position independent
    # stage B: palm surface on the upright cup before a grasp (no more credit without the palm)
    W_PALM_CONTACT = 0.8
    # stage C: wrap -- only inner-surface contacts count (the back of a finger earns nothing here)
    W_CURL_ONTO = 1.0         # flexion of digits whose inner side lies on / touches the cup body
    W_WRAP_FINGERS = 3.0      # ((inner-wrapped digits - 1) / 4)^1.5: 2 -> 0.375, 3 -> 1.06, 5 -> 3.0
    W_WRAP_LINKS = 1.2        # fraction of the 15 measured links touching with their inner side
    W_PALM_IN_GRASP = 1.2     # palm pressing and facing the cup while digits wrap
    W_ENVELOPE = 1.5          # all five digits inner-wrapped, thumb opposed, palm on
    # stage D: lift (stops growing at goal height; scaled by grasp quality)
    W_LIFT = 5.0
    # stage E: goal and hold (together ~3x the lift near the goal)
    W_GOAL_COARSE = 3.0       # position error, 10 cm scale -> steers the lifted cup to the goal
    W_GOAL_FINE = 5.0         # goal_dist, 2 cm scale -> rises sharply close to the goal
    W_STILL = 4.0             # near the goal and still
    W_IN_TOL = 3.0
    W_SUCCESS = 25.0
    # penalties / regularisation
    W_TABLE = 1.0             # finger links / fingertips / palm coming down onto the table
    W_TILT = 1.5              # ungrasped: 3 .. 15 deg | grasped: 15 .. 35 deg
    W_PUSH = 1.0              # cup on the table only: ungrasped 1.5 .. 6 cm | grasped 8 .. 15 cm
    W_AWAY = 0.5              # lifted cup away from the start->goal segment: -0.5 per 10 cm beyond 3 cm (max -3)
    W_ACTION_RATE = 0.01
    W_JOINT_VEL = 0.005

    cup_r = ctx.cup_radius
    cup_hh = ctx.cup_half_height
    axis = ctx.cup_axis
    tilt = ctx.cup_tilt
    progress = ctx.episode_progress.clamp(0.0, 1.0)
    lifted_f = ctx.lifted.float()

    # ================================================================== cup clearance above the table
    # tipping the cup about its bottom rim raises cup_pos by ~R*sin(tilt) but keeps the rim on the table
    cos_t = axis[:, 2].clamp(-1.0, 1.0)
    sin_t = torch.sqrt((1.0 - cos_t ** 2).clamp(min=0.0))
    b = (ctx.cup_spawn_pos[:, 2] - ctx.table_z).clamp(min=0.0)
    clearance = (ctx.cup_pos[:, 2] - ctx.table_z - b * cos_t - cup_r * sin_t).clamp(min=0.0)
    airborne = _ramp(clearance, 0.004, 0.02)                  # 1 once the whole cup is 2 cm off the table
    on_table = 1.0 - airborne

    # ================================================================== palm pose relative to the cup
    h_p, rad_p, r_p = _cylinder_coords(ctx.palm_pos, ctx.cup_pos, axis)
    u_p = rad_p / r_p.clamp(min=eps).unsqueeze(-1)            # unit radial direction: cup axis -> palm centre
    gap_p = r_p - cup_r
    e_r = torch.relu(gap_p - 0.012) + torch.relu(-0.010 - gap_p)
    e_h = torch.relu((h_p - 0.3 * cup_hh).abs() - 0.25 * cup_hh)
    d_palm = torch.sqrt(e_r ** 2 + e_h ** 2 + eps)

    n = ctx.palm_normal
    facing = -(n * u_p).sum(-1)                               # +1 when the palm normal points at the cup axis
    base = ctx.link_pos[:, 1:4, 0, :].mean(dim=1) - ctx.palm_pos
    base = base - (base * n).sum(-1, keepdim=True) * n
    d_f = _unit(base)                                         # finger direction in the palm plane
    s_f = (d_f * axis).sum(-1).abs()
    face_w = _ramp(facing, 0.15, 0.85)
    level_w = 1.0 - _ramp(s_f, 0.3, 0.7)
    orient_w = face_w * (0.4 + 0.6 * level_w)
    orient_loose = _ramp(facing, 0.1, 0.5) * (1.0 - _ramp(s_f, 0.5, 0.85))
    face_loose = _ramp(facing, 0.1, 0.5)

    spot = torch.exp(-(torch.relu(d_palm - 0.01) / 0.03) ** 2)
    seat = orient_loose * spot
    seat_loose = _ramp(facing, 0.0, 0.4) \
        * torch.exp(-(torch.relu(gap_p - 0.02) / 0.05) ** 2) \
        * torch.exp(-(torch.relu(h_p.abs() - cup_hh) / 0.04) ** 2)

    # ================================================================== hand coming down onto the table
    link_clear = ctx.link_pos[..., 2] - ctx.table_z                                   # (N,5,3)
    link_low = (1.0 - _ramp(link_clear, 0.006, 0.018)).flatten(1).max(dim=-1).values
    other_low = 1.0 - _ramp(ctx.hand_z_min - ctx.table_z, 0.006, 0.018)
    palm_low = 1.0 - _ramp(ctx.palm_pos[:, 2] - ctx.table_z, 0.010, 0.030)
    table_near = torch.maximum(torch.maximum(link_low, other_low), palm_low)          # (N,)

    # ================================================================== digit contacts on the cup body
    lp = ctx.link_pos                                                       # (N,5,3,3)
    h_l, rad_l, r_l = _cylinder_coords(lp, ctx.cup_pos, axis)               # (N,5,3), (N,5,3,3), (N,5,3)
    R_l = cup_r.reshape(-1, 1, 1)
    H_l = cup_hh.reshape(-1, 1, 1)
    gap_l = r_l - R_l
    in_band_l = torch.clamp(1.0 - torch.relu(h_l.abs() - H_l) / 0.015, 0.0, 1.0)
    outside_l = _ramp(gap_l, -0.02, -0.01)
    close_l = torch.clamp(1.0 - torch.relu(gap_l - 0.03) / 0.02, 0.0, 1.0)
    valid_l = in_band_l * outside_l * close_l
    link_touch = _touch(ctx.link_cup_force, 0.3) * valid_l                  # (N,5,3) any side of the link

    # ---- inner surface of each measured link
    # segment directions: link _3 -> link _4, link _4 -> tip (the tip uses the last segment)
    seg0 = _unit(lp[:, :, 1, :] - lp[:, :, 0, :])                           # (N,5,3)
    seg1 = _unit(lp[:, :, 2, :] - lp[:, :, 1, :])
    seg = torch.stack([seg0, seg1, seg1], dim=2)                            # (N,5,3,3)
    # four fingers: flexion axis w = palm_normal x finger_direction; inner normal = seg x w
    # (= palm_normal for a straight finger, rotates toward the wrist as the finger curls)
    w_flex = _unit(torch.cross(n, d_f, dim=-1))
    in_fingers = _unit(torch.cross(seg, w_flex.reshape(-1, 1, 1, 3).expand_as(seg), dim=-1))
    # thumb (opposed, own flexion plane): inner side faces the object centre in front of the palm
    grasp_centre = ctx.palm_pos + n * cup_r.unsqueeze(-1)                   # (N,3)
    ref_t = grasp_centre.unsqueeze(1) - lp[:, 0, :, :]                      # (N,3,3)
    seg_t = seg[:, 0, :, :]
    in_thumb = _unit(ref_t - (ref_t * seg_t).sum(-1, keepdim=True) * seg_t)
    in_n = torch.cat([in_thumb.unsqueeze(1), in_fingers[:, 1:, :, :]], dim=1)   # (N,5,3,3)
    to_axis = -_unit(rad_l)                                                 # link -> cup axis
    inner_l = _ramp((to_axis * in_n).sum(-1), 0.1, 0.5)                     # 1: cup on the inside of the bend

    finger_c = link_touch.max(dim=-1).values                                # (N,5) touching with any side
    n_any = finger_c.sum(-1)
    link_w = link_touch * inner_l                                           # (N,5,3) touching with the inner side
    finger_w = link_w.max(dim=-1).values                                    # (N,5)
    n_wrap = finger_w.sum(-1)

    # thumb opposition: thumb vs. the four fingers on opposite sides of the plane (cup axis, palm direction)
    tangent = _unit(torch.cross(axis, u_p, dim=-1))
    side_l = (_unit(rad_l) * tangent.reshape(-1, 1, 1, 3)).sum(-1)          # (N,5,3)
    w_l = link_touch + 0.05
    side_f = (side_l * w_l).sum(-1) / w_l.sum(-1)                           # (N,5)
    opp = 0.5 * (1.0 - torch.tanh(4.0 * side_f[:, 0]) * torch.tanh(4.0 * side_f[:, 1:].mean(-1)))
    opp_any = finger_c[:, 0] * opp
    opp_w = finger_w[:, 0] * opp                                            # thumb pad on the far side

    # palm surface on the cup
    valid_p = torch.clamp(1.0 - torch.relu(gap_p - 0.025) / 0.015, 0.0, 1.0) \
        * torch.clamp(1.0 - torch.relu(-0.020 - gap_p) / 0.015, 0.0, 1.0) \
        * torch.clamp(1.0 - torch.relu(h_p.abs() - cup_hh) / 0.02, 0.0, 1.0)
    palm_touch = _touch(ctx.palm_cup_force, 0.5) * valid_p
    palm_g = palm_touch * face_loose

    # hold: the cup is held (any side of the digits) -> lift / goal gating and penalty relief
    hold_q = _ramp(n_any, 1.0, 3.0) * (0.6 + 0.4 * opp_any) * (0.5 + 0.5 * seat_loose)
    relief = torch.sqrt(hold_q + eps)
    # envelope quality: palm + inner-wrapped digits, thumb opposed -> multiplies lift and goal terms
    env_q = (palm_g + n_wrap) / 6.0 * (0.5 + 0.5 * opp_w)
    q_mult = 0.4 + 0.6 * env_q

    # ================================================================== resting decay
    rest_decay = 1.0 - 0.5 * progress * (1.0 - hold_q)
    wrap_decay = 1.0 - 0.25 * progress * on_table

    # ================================================================== stage A: approach with an open hand
    reach_far = 1.0 - torch.tanh(d_palm / 0.3)
    reach_facing = orient_w * (1.0 - torch.tanh(d_palm / 0.03))
    palm_orient = 0.5 * (facing + 1.0) * (1.0 - s_f) * (1.0 - torch.tanh(d_palm / 0.2))
    reach_far_c = reach_far + hold_q * (1.0 - reach_far)
    reach_facing_c = reach_facing + hold_q * (1.0 - reach_facing)
    palm_orient_c = palm_orient + hold_q * (1.0 - palm_orient)

    four = [4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18]
    thumb = [1, 2]
    close_tgt = 0.8 * ctx.hand_target_norm[:, four].mean(-1) + 0.2 * ctx.hand_target_norm[:, thumb].mean(-1)
    pre_grasp = (1.0 - hold_q) * (1.0 - lifted_f) * on_table
    finger_open = pre_grasp * ((1.0 - close_tgt) - (1.0 - seat) * close_tgt)

    # ================================================================== stage B: palm on the upright cup
    up_pre = torch.exp(-(tilt / (6.0 * deg)) ** 2)
    up_grasp = 1.0 - _ramp(tilt, 10.0 * deg, 25.0 * deg)
    up_palm = up_pre + hold_q * (up_grasp - up_pre)
    face_palm = orient_w + hold_q * (torch.maximum(orient_w, orient_loose) - orient_w)
    palm_on = palm_touch * face_palm * up_palm             # no half credit without the palm any more

    # ================================================================== stage C: inner-surface wrap
    seat_c = torch.maximum(seat, airborne * seat_loose)
    wrap_gate = seat_c * up_grasp * wrap_decay

    finger_joints = [[1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18]]
    curl = torch.stack([ctx.hand_q_norm[:, j].mean(-1) for j in finger_joints], dim=-1)   # (N,5)
    near_l = torch.exp(-(torch.relu(gap_l - 0.012) / 0.03) ** 2) * in_band_l * outside_l * inner_l
    near_f = near_l[:, :, 1:].mean(-1)                      # outer link + fingertip, inner side toward the cup
    curl_onto = (curl * (0.3 * near_f + 0.7 * finger_w)).mean(-1)
    wrap_fingers = _ramp(n_wrap, 1.0, 5.0) ** 1.5 * (0.4 + 0.6 * opp_w)
    wrap_links = link_w.mean(dim=(-2, -1))
    palm_in_grasp = palm_g * _ramp(n_wrap, 1.0, 3.0)
    envelope = finger_w.min(dim=-1).values * opp_w * (0.4 + 0.6 * palm_g)

    # ================================================================== stage D: lift (real lift, capped at goal height)
    upright_hold = torch.exp(-(tilt / 0.25) ** 2)
    goal_dz = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    lift_prog = 0.4 * airborne + 0.6 * torch.clamp(clearance / goal_dz, 0.0, 1.0)
    lift = lift_prog * hold_q * q_mult * upright_hold

    # ================================================================== stage E: goal and hold
    goal_gate = airborne * hold_q * q_mult * upright_hold
    pos_err = (ctx.cup_pos - ctx.goal_pos).norm(dim=-1)
    goal_coarse = goal_gate * torch.exp(-pos_err / 0.10)
    goal_fine = goal_gate * torch.exp(-ctx.goal_dist / 0.02)
    speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    still = goal_gate * torch.exp(-(ctx.goal_dist / 0.04) ** 2) * torch.exp(-speed / 0.05 - spin / 0.5)
    hold_f = 0.5 + 0.5 * env_q
    in_tol = (ctx.lifted & (ctx.goal_dist <= ctx.success_tol)).float() * airborne * hold_f
    success_bonus = ctx.success.float() * hold_f

    # ================================================================== penalties
    tilt_pen = (1.0 - relief) * _ramp(tilt, 3.0 * deg, 15.0 * deg) + relief * _ramp(tilt, 15.0 * deg, 35.0 * deg)
    disp_xy = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    push_pen = on_table * ((1.0 - relief) * _ramp(disp_xy, 0.015, 0.06) + relief * _ramp(disp_xy, 0.08, 0.15))
    # lifted cup: distance from the straight segment start -> goal (above the goal or to the side costs more)
    seg_g = ctx.goal_pos - ctx.cup_spawn_pos
    t_g = ((ctx.cup_pos - ctx.cup_spawn_pos) * seg_g).sum(-1) / (seg_g * seg_g).sum(-1).clamp(min=eps)
    closest = ctx.cup_spawn_pos + t_g.clamp(0.0, 1.0).unsqueeze(-1) * seg_g
    dev = (ctx.cup_pos - closest).norm(dim=-1)
    away_pen = airborne * torch.clamp(dev - 0.03, 0.0, 0.6) / 0.1
    action_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(-1)
    joint_vel = (ctx.arm_qd ** 2).mean(-1) + 0.1 * (ctx.hand_qd ** 2).mean(-1)

    components = {
        "reach_far": W_REACH_FAR * rest_decay * reach_far_c,
        "reach_facing": W_REACH_FACING * rest_decay * reach_facing_c,
        "palm_orient": W_ORIENT * rest_decay * palm_orient_c,
        "finger_open": W_FINGER_OPEN * rest_decay * finger_open,
        "palm_contact": W_PALM_CONTACT * rest_decay * palm_on,
        "finger_curl_onto_cup": W_CURL_ONTO * wrap_gate * curl_onto,
        "wrap_fingers": W_WRAP_FINGERS * wrap_gate * wrap_fingers,
        "wrap_links": W_WRAP_LINKS * wrap_gate * wrap_links,
        "palm_in_grasp": W_PALM_IN_GRASP * wrap_gate * palm_in_grasp,
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
        "carry_away_penalty": -W_AWAY * away_pen,
        "action_rate_penalty": -W_ACTION_RATE * action_rate,
        "joint_vel_penalty": -W_JOINT_VEL * joint_vel,
    }
    reward = torch.stack(list(components.values()), dim=0).sum(dim=0)
    return reward, components
