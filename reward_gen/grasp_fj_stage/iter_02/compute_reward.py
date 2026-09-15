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
