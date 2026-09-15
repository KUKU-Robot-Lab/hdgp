import torch
import math


# Movable finger joints (range > 0.05 rad), grouped per digit in hand-joint order:
# thumb (3,4), index (2,3,4), middle (2,3,4), ring (2,3,4), pinky (3,4).
_DIGIT_JOINTS = ([1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18])
_MOVABLE = [j for group in _DIGIT_JOINTS for j in group]


def _soft_touch(force: torch.Tensor, full_force: float) -> torch.Tensor:
    # 0 when not touching, 1 at/above full_force; clamping also bounds physics force spikes
    return torch.clamp(force / full_force, 0.0, 1.0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale = 0.1  # global scale so a typical per-step reward is O(1)

    r = ctx.cup_radius
    hh = ctx.cup_half_height
    n = ctx.palm_normal
    fd = ctx.palm_finger_dir
    axis = ctx.cup_axis

    # ---------------- stage masks (strict order) ----------------
    approach = ctx.approach_done.bool()
    envelope = ctx.envelope_done.bool() & approach
    s1 = (~approach).float()
    s2 = (approach & ~envelope).float()
    s3 = envelope.float()

    # ---------------- palm relative to the cup (same quantities approach_done checks) ----------------
    v = ctx.cup_pos - ctx.palm_pos
    v_ax = (v * axis).sum(-1)
    v_perp = v - v_ax.unsqueeze(-1) * axis
    gap_n = (v_perp * n).sum(-1) - r    # env window [-0.01, 0.02]
    gap_f = (v_perp * fd).sum(-1) - r   # env window [-0.005, 0.02]
    h_palm = -v_ax                      # palm height along the cup axis; env needs |h| <= hh

    # orientation kept at the start orientation (normal +y, fingers +x); ~0.05 at 45 deg on both axes
    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))
    ang_f = torch.acos(torch.clamp(fd[:, 0], -1.0, 1.0))
    ori = torch.exp(-(ang_n ** 2 + ang_f ** 2) / (2.0 * 0.45 ** 2))

    # default hand pose kept (env needs every movable joint within 0.3); mean term keeps a gradient far away
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev.amax(-1) / 0.2) ** 2)

    # ---------------- stage 1: approach ----------------
    # coarse world-frame target on the cup's -y side, slightly behind the axis along +x, upper band
    offset = torch.stack([-(r + 0.0075), -(r + 0.008), 0.25 * hh], dim=-1)
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(3.0 * d_world)
    # fine palm-frame precision centred in the approach_done window
    fine_err2 = (gap_n - 0.008) ** 2 + (gap_f - 0.0075) ** 2 + torch.relu(h_palm.abs() - 0.5 * hh) ** 2
    fine = torch.exp(-fine_err2 / (2.0 * 0.012 ** 2))
    keep_factor = 0.3 + 0.7 * ori * pose  # turning the hand or closing fingers cuts approach progress
    approach_reach = s1 * (1.0 * coarse + 1.5 * fine) * keep_factor  # max 2.5
    approach_keep = s1 * (0.25 * ori + 0.25 * pose)                  # max 0.5 -> stage 1 max 3.0

    # ---------------- contacts ----------------
    link_touch = _soft_touch(ctx.link_cup_force, 0.5)      # (N,5,3)
    digit_touch = link_touch.amax(-1)                       # (N,5)
    palm_touch = _soft_touch(ctx.palm_cup_force, 0.5)       # (N,)
    thumb_touch = digit_touch[:, 0]
    finger_count = digit_touch[:, 1:].sum(-1)               # 0..4
    early_contact_pen = -2.0 * s1 * digit_touch.amax(-1)    # no thumb/finger contact before the approach is done

    # ---------------- link geometry around the cup ----------------
    lv = ctx.link_pos - ctx.cup_pos[:, None, None, :]       # (N,5,3,3)
    ax = axis[:, None, None, :]
    lh = (lv * ax).sum(-1)                                  # (N,5,3)
    lrad_vec = lv - lh.unsqueeze(-1) * ax
    lrad = torch.norm(lrad_vec, dim=-1)
    surf_gap = torch.relu(lrad - r[:, None, None] - 0.01)   # 1 cm allowance for link thickness
    in_band = torch.clamp(1.0 - torch.relu(lh.abs() - hh[:, None, None]) / 0.02, 0.0, 1.0)
    near_surf = torch.exp(-surf_gap / 0.015) * in_band      # (N,5,3)

    # fingertip wrap angle around the axis: 0 = palm side, pi/2 = far side along the fingers
    tip_vec = lrad_vec[:, 1:, 2, :]                         # (N,4,3)
    along_f = (tip_vec * fd[:, None, :]).sum(-1)
    toward_palm = -(tip_vec * n[:, None, :]).sum(-1)
    theta = torch.atan2(along_f, toward_palm)
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)  # saturates at ~108 deg (wrapped past the far side)
    finger_wrap = (wrap * near_surf[:, 1:, 2]).mean(-1)
    thumb_near = near_surf[:, 0, :].amax(-1)

    # palm closing the last centimetre while staying aligned along the fingers and in the band
    align2 = torch.exp(-((gap_f - 0.0075) ** 2 + torch.relu(h_palm.abs() - hh) ** 2) / (2.0 * 0.02 ** 2))
    palm_close = torch.exp(-torch.relu(gap_n) / 0.015) * align2

    # squeeze: command leads the measured angle on touching digits, so the PD fingers press
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, j].mean(-1) for j in _DIGIT_JOINTS], dim=-1)  # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)

    # ---------------- stage 2: envelope grasp ----------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    resting = torch.clamp(1.0 - (dz - 0.02) / 0.02, 0.0, 1.0)  # no stage-2 progress once the cup is lifted >4 cm
    fac2 = s2 * (0.3 + 0.7 * ori) * resting
    grasp_stage_base = 3.5 * s2                                  # > stage 1 max (3.0)
    grasp_palm = fac2 * (0.5 * palm_close + 1.0 * palm_touch)
    grasp_contacts = fac2 * (1.0 * thumb_touch + 1.5 * finger_count / 4.0)
    grasp_wrap = fac2 * (1.0 * finger_wrap + 0.5 * thumb_near)
    grip_squeeze = 0.5 * squeeze * (fac2 + s3)                   # stage 2 max total 3.5 + 6.0 = 9.5

    # ---------------- stage 3: lift and hold ----------------
    firm = _soft_touch(ctx.link_cup_force, 0.25).amax(-1)       # (N,5)
    grip_gate = firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)  # thumb + >=2 fingers pressing
    lift_stage_base = 10.0 * s3                                  # > stage 2 max (9.5)
    hold_contacts = s3 * (1.0 * palm_touch + 1.0 * thumb_touch + 1.5 * finger_count / 4.0)  # releasing loses this
    rise_needed = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)
    lift_frac = torch.clamp(dz / rise_needed, 0.0, 1.0)
    lift_height = 3.0 * s3 * grip_gate * lift_frac
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    lift_goal = s3 * grip_gate * (2.0 * (1.0 - torch.tanh(gd / 0.2)) + 2.0 * torch.exp(-gd / 0.03) + 1.0 * in_tol)
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 1.0 * s3 * grip_gate * torch.exp(-gd / 0.03) * calm
    success_bonus = 20.0 * s3 * ctx.success.float()

    # ---------------- constraints and regularization (all stages) ----------------
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.01) / 0.04, 0.0, 1.0)  # do not shove the cup
    cup_tilt_pen = -2.0 * torch.clamp(ctx.cup_tilt / (math.pi / 3.0), 0.0, 1.0) ** 2   # do not knock it over
    table_pen = -2.0 * torch.clamp((ctx.table_z + 0.002 - ctx.hand_z_min) / 0.02, 0.0, 1.0)  # do not press into the table
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.05 * (ctx.arm_qd ** 2).sum(-1)

    raw = {
        "approach_reach": approach_reach,
        "approach_keep": approach_keep,
        "early_contact_pen": early_contact_pen,
        "grasp_stage_base": grasp_stage_base,
        "grasp_palm": grasp_palm,
        "grasp_contacts": grasp_contacts,
        "grasp_wrap": grasp_wrap,
        "grip_squeeze": grip_squeeze,
        "lift_stage_base": lift_stage_base,
        "hold_contacts": hold_contacts,
        "lift_height": lift_height,
        "lift_goal": lift_goal,
        "hold_still": hold_still,
        "success_bonus": success_bonus,
        "cup_disturb_pen": cup_disturb_pen,
        "cup_tilt_pen": cup_tilt_pen,
        "table_pen": table_pen,
        "action_rate_pen": action_rate_pen,
        "arm_vel_pen": arm_vel_pen,
    }
    components = {name: scale * value for name, value in raw.items()}
    reward = torch.stack(list(components.values()), dim=0).sum(0)
    return reward, components
