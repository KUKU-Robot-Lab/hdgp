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
