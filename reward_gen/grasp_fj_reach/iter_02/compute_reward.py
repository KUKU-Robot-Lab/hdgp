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
