import torch
import math

# Movable finger joints (range > 0.05 rad), in ctx.hand_q order.
# Locked: 0 thumb_2, 3 index_1, 7 middle_1, 11 ring_1, 15 pinky_1, 16 pinky_2.
MOVABLE_HAND_IDX = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18]
CONTACT_FORCE_SCALE = 0.3   # N; 0.1 N (env contact threshold) -> 0.28, 1 N -> 0.96

SIDE_CLEARANCE = 0.008      # palm centre target: 8 mm outside the cup surface on the -y side
SIDE_DEADBAND = 0.008       # lateral dead band -> palm centre anywhere in [R, R + 16 mm] (flag allows 2 cm)
X_DEADBAND = 0.01           # palm centre within 1 cm of the cup axis along x
HEIGHT_DEADBAND_FRAC = 0.25  # |h| <= 0.25 * half height of the band counts as "at band height"
STANDOFF_MAX = 0.04         # extra -y distance while the palm is still far from the cup axis along x
STANDOFF_X_SCALE = 0.06     # x offset over which the stand-off fades out
ORIENT_SCALE = 0.4          # (1-n.y)+(1-f.x): 20 deg on both axes -> 0.93, 45 deg on both -> 0.23
POSE_READY_DEV = 0.12       # env approach flag requires every movable joint within 0.15 of default


def _contact_level(force: torch.Tensor) -> torch.Tensor:
    """Bounded, spike-robust contact level in [0, 1)."""
    return 1.0 - torch.exp(-force.clamp(min=0.0) / CONTACT_FORCE_SCALE)


def _cylinder_coords(p: torch.Tensor, cup_pos: torch.Tensor, cup_axis: torch.Tensor):
    """Axial coordinate h, radial vector and radial distance of point(s) p w.r.t. the cup axis."""
    v = p - cup_pos
    h = (v * cup_axis).sum(-1)
    radial_vec = v - h.unsqueeze(-1) * cup_axis
    radial = radial_vec.norm(dim=-1)
    return h, radial_vec, radial


def _side_target_dist(ctx, palm_h: torch.Tensor, standoff: torch.Tensor) -> torch.Tensor:
    """Dead-banded distance of the palm centre to the grasp spot on the cup's -y side."""
    rel = ctx.palm_pos - ctx.cup_pos
    lat = -rel[:, 1]                                   # distance from the axis towards -y (negative = wrong side)
    lat_target = ctx.cup_radius + SIDE_CLEARANCE + standoff
    e_lat = torch.relu((lat - lat_target).abs() - SIDE_DEADBAND)
    e_x = torch.relu(rel[:, 0].abs() - X_DEADBAND)
    e_z = torch.relu(palm_h.abs() - HEIGHT_DEADBAND_FRAC * ctx.cup_half_height)
    return torch.sqrt(e_lat ** 2 + e_x ** 2 + e_z ** 2 + 1e-12)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.palm_pos.device
    dtype = ctx.palm_pos.dtype
    R = ctx.cup_radius
    hh = ctx.cup_half_height
    idx = torch.tensor(MOVABLE_HAND_IDX, device=device, dtype=torch.long)

    # ---------------- stage masks (strict order kept by the env's latched flags) ----------------
    s1 = (~ctx.approach_done).to(dtype)
    s2 = (ctx.approach_done & ~ctx.envelope_done).to(dtype)
    s3 = ctx.envelope_done.to(dtype)

    # ---------------- palm geometry relative to the cup ----------------
    palm_h, palm_rvec, palm_r = _cylinder_coords(ctx.palm_pos, ctx.cup_pos, ctx.cup_axis)
    palm_u = palm_rvec / palm_r.clamp(min=1e-6).unsqueeze(-1)      # outward radial unit vector at the palm

    # Grasp spot on the cup's -y side. While the palm is still far from the cup axis along x
    # (fingers pointing +x would reach the cup first), the target lies up to 4 cm further out in -y.
    rel_x = ctx.palm_pos[:, 0] - ctx.cup_pos[:, 0]
    standoff = STANDOFF_MAX * torch.tanh((rel_x / STANDOFF_X_SCALE) ** 2)
    d_app = _side_target_dist(ctx, palm_h, standoff)               # stage 1 (with stand-off funnel)
    d_side = _side_target_dist(ctx, palm_h, torch.zeros_like(standoff))  # stage 2 (exact grasp spot)

    # Start-pose hand orientation: palm_normal along +y, palm_finger_dir along +x.
    orient_err = (1.0 - ctx.palm_normal[:, 1]) + (1.0 - ctx.palm_finger_dir[:, 0])
    orient_keep = torch.exp(-orient_err.clamp(min=0.0) / ORIENT_SCALE)

    # ---------------- finger link geometry and contacts ----------------
    cup_pos_l = ctx.cup_pos[:, None, None, :]
    axis_l = ctx.cup_axis[:, None, None, :]
    link_h, _, link_r = _cylinder_coords(ctx.link_pos, cup_pos_l, axis_l)          # (N,5,3)
    link_c = _contact_level(ctx.link_cup_force)                                      # (N,5,3)
    digit_c = link_c.amax(-1)                                                        # (N,5)
    palm_c = _contact_level(ctx.palm_cup_force)                                      # (N,)
    digit_w = torch.tensor([0.3, 0.175, 0.175, 0.175, 0.175], device=device, dtype=dtype)  # thumb weighted most

    # ---------------- cup state ----------------
    raise_h = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)

    # =========================== STAGE 1: approach ===========================
    # Default finger pose, per joint: actual angle (what the env flag tests) and the raw policy action
    # (maps linearly onto the target, so the reward follows the action without the filter lag).
    default_n = ctx.hand_default_q_norm[:, idx]
    dev_q = (ctx.hand_q_norm[:, idx] - default_n).abs()
    action_n = (ctx.actions[:, 7:26].clamp(-1.0, 1.0) + 1.0) * 0.5
    dev_a = (action_n[:, idx] - default_n).abs()
    dev = 0.5 * dev_q + 0.5 * dev_a
    # Linear part keeps a gradient at any deviation (random actions give ~0.5); exp part sharpens near 0.
    pose_score = (0.4 * (1.0 - dev).clamp(min=0.0) + 0.6 * torch.exp(-dev / 0.15)).mean(-1)
    pose_ready = torch.sigmoid((POSE_READY_DEV - dev_q) / 0.02).mean(-1)  # fraction of joints inside the flag limit

    finger_touch = torch.tanh(link_c.sum(dim=(1, 2)))                     # fingers/thumb must not touch yet

    # Moving the palm to the -y grasp spot is the largest stage-1 reward (max 2.0 of 3.0).
    approach_reach = 1.0 * (1.0 - torch.tanh(d_app / 0.25)) * s1          # from the 0.38 m start: 0.09 -> 1.0
    approach_arrive = 1.0 * torch.exp(-d_app / 0.03) * orient_keep * pose_ready * s1  # last cm, only in a flag-ready pose
    approach_orient_keep = 0.5 * orient_keep * s1                         # keep the start orientation, max 0.5
    approach_pose_keep = 0.5 * pose_score * s1                            # keep the default finger pose, max 0.5
    approach_finger_contact = -0.5 * finger_touch * s1
    # stage 1 total <= 3.0

    # Stages 1-2: the cup must stay where it stands (no pushing, no early lifting).
    disturb = 0.5 * torch.tanh(torch.relu(disp_xy - 0.005) / 0.05) \
        + 0.5 * torch.tanh(torch.relu(raise_h - 0.005) / 0.02)
    cup_disturb = -1.0 * disturb * (s1 + s2)

    # =========================== STAGE 2: envelope grasp ===========================
    stay_gate = torch.exp(-d_side / 0.03)        # grasp shaping only counts at the -y grasp spot
    envelope_stay = 0.75 * torch.exp(-d_side / 0.02) * s2
    envelope_orient_keep = 0.25 * orient_keep * stay_gate * s2           # same hand orientation as the approach

    # Closing progress (a finger stops on the cup, so this only drives closing, not crushing).
    closure = ctx.hand_q_norm[:, idx].mean(-1)
    envelope_closure = 0.25 * closure * stay_gate * s2

    # Wrap: finger links on the cup surface, inside the graspable band, on the cup side of the palm plane.
    near_surface = torch.exp(-torch.relu(link_r - R[:, None, None]) / 0.015)
    in_band = torch.exp(-torch.relu(link_h.abs() - hh[:, None, None]) / 0.01)
    rel_palm = ctx.link_pos - ctx.palm_pos[:, None, None, :]
    cup_side = torch.sigmoid((rel_palm * ctx.palm_normal[:, None, None, :]).sum(-1) / 0.01)
    link_wrap = near_surface * in_band * cup_side                      # (N,5,3)
    digit_wrap = link_wrap.mean(-1)                                    # (N,5)
    envelope_wrap = 0.75 * (digit_wrap * digit_w).sum(-1) * stay_gate * s2

    # Contacts: palm against the cup, thumb and all four fingers touching it.
    envelope_palm_contact = 0.5 * palm_c * stay_gate * s2
    envelope_digit_contact = 1.0 * (digit_c * digit_w).sum(-1) * stay_gate * s2

    # Opposition: thumb and the four fingers on opposite sides of the plane through the cup axis and palm.
    tangent = torch.cross(ctx.cup_axis, palm_u, dim=-1)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    link_s = ((ctx.link_pos - cup_pos_l) * tangent[:, None, None, :]).sum(-1)       # (N,5,3)
    digit_s = link_s.mean(-1) / R.unsqueeze(-1).clamp(min=1e-3)                      # (N,5), radius-normalised
    opposition = torch.tanh(-4.0 * digit_s[:, 0] * digit_s[:, 1:].mean(-1)).clamp(min=0.0)
    opp_gate = torch.minimum(digit_wrap[:, 0], digit_wrap[:, 1:].mean(-1))            # only counts on the cup
    envelope_opposition = 0.25 * opposition * opp_gate * stay_gate * s2
    # stage 2 shaping total <= 3.75

    # =========================== STAGE 3: lift and hold ===========================
    others_c = (digit_c[:, 1:].sum(-1) / 3.0).clamp(max=1.0)          # >= 3 fingers besides the thumb
    grip_quality = (palm_c + digit_c[:, 0] + others_c) / 3.0
    grip_gate = (palm_c * digit_c[:, 0] * others_c).clamp(min=0.0).pow(1.0 / 3.0)  # zero if palm/thumb/fingers let go

    lift_grip_hold = 1.0 * grip_quality * s3

    goal_pos_dist = (ctx.cup_pos - ctx.goal_pos).norm(dim=-1)
    goal_progress = 0.5 * (1.0 - torch.tanh(goal_pos_dist / 0.2)) + 0.5 * torch.exp(-ctx.goal_dist / 0.02)
    lift_goal = 3.0 * grip_gate * goal_progress * s3                  # main task term, max 3.0

    near_goal = torch.exp(-ctx.goal_dist / 0.03)
    stillness = torch.exp(-ctx.cup_lin_vel.norm(dim=-1) / 0.1 - ctx.cup_ang_vel.norm(dim=-1) / 1.0)
    lift_hold_still = 1.0 * grip_gate * near_goal * stillness * s3

    # Lift straight up: sideways deviation from the spawn -> goal line at the current height.
    goal_rise = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=1e-3)
    rise_frac = (raise_h / goal_rise).clamp(0.0, 1.0)
    line_xy = ctx.cup_spawn_pos[:, :2] + rise_frac.unsqueeze(-1) * (ctx.goal_pos[:, :2] - ctx.cup_spawn_pos[:, :2])
    lateral_dev = (ctx.cup_pos[:, :2] - line_xy).norm(dim=-1)
    lift_path_dev = -0.5 * torch.tanh(torch.relu(lateral_dev - 0.01) / 0.03) * s3

    # Success is only reachable by a held cup; paid only after the envelope stage.
    success_bonus = 10.0 * ctx.success.to(dtype) * s3

    # =========================== stage offsets ===========================
    # Stage 2 base (3.5) > stage 1 max (3.0); stage 3 base (7.5) > stage 2 max (3.5 + 3.75 = 7.25):
    # completing a stage never lowers reward, and higher levels only come through the env's ordered flags.
    stage_base = 3.5 * s2 + 7.5 * s3

    # =========================== always-on constraints ===========================
    cup_tilt = -1.0 * torch.tanh(torch.relu(ctx.cup_tilt - 0.05) / 0.3)             # keep the cup upright
    table_clearance = -1.0 * torch.tanh(torch.relu(ctx.table_z + 0.005 - ctx.hand_z_min) / 0.01)  # no hand in table
    arm_rate = ((ctx.actions[:, :7] - ctx.prev_actions[:, :7]) ** 2).sum(-1)
    hand_rate = ((ctx.actions[:, 7:] - ctx.prev_actions[:, 7:]) ** 2).sum(-1)
    action_rate = -(0.01 * arm_rate + 0.002 * hand_rate)                              # small smoothness prior

    components = {
        "stage_base": stage_base,
        "approach_reach": approach_reach,
        "approach_arrive": approach_arrive,
        "approach_orient_keep": approach_orient_keep,
        "approach_pose_keep": approach_pose_keep,
        "approach_finger_contact": approach_finger_contact,
        "cup_disturb": cup_disturb,
        "envelope_stay": envelope_stay,
        "envelope_orient_keep": envelope_orient_keep,
        "envelope_closure": envelope_closure,
        "envelope_wrap": envelope_wrap,
        "envelope_palm_contact": envelope_palm_contact,
        "envelope_digit_contact": envelope_digit_contact,
        "envelope_opposition": envelope_opposition,
        "lift_grip_hold": lift_grip_hold,
        "lift_goal": lift_goal,
        "lift_hold_still": lift_hold_still,
        "lift_path_dev": lift_path_dev,
        "success_bonus": success_bonus,
        "cup_tilt": cup_tilt,
        "table_clearance": table_clearance,
        "action_rate": action_rate,
    }
    reward = torch.zeros_like(ctx.cup_tilt)
    for value in components.values():
        reward = reward + value
    return reward, components
