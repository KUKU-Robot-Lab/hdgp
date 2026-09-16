import torch
import math


# Movable finger joints (range > 0.05 rad), grouped per digit in hand-joint order:
# thumb (3,4), index (2,3,4), middle (2,3,4), ring (2,3,4), pinky (3,4).
_DIGIT_JOINTS = ([1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18])
_MOVABLE = [j for group in _DIGIT_JOINTS for j in group]


def _sat(x: torch.Tensor, full: float) -> torch.Tensor:
    # 0 at/below 0, 1 at/above `full`; the clamp also bounds physics force spikes
    return torch.clamp(x / full, 0.0, 1.0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale = 0.1  # global scale: fingertip touching ~0.4, a real envelope ~3.8, a held success ~6.5

    r = ctx.cup_radius
    hh = ctx.cup_half_height
    n = ctx.palm_normal
    fd = ctx.palm_finger_dir
    axis = ctx.cup_axis

    # ---------------- stage masks (strict order, flags are latched by the env) ----------------
    approach = ctx.approach_done.bool()
    envelope = ctx.envelope_done.bool() & approach
    s1 = (~approach).float()
    s2 = (approach & ~envelope).float()
    s3 = envelope.float()

    # ---------------- palm pose relative to the cup (same quantities approach_done checks) ----
    v = ctx.cup_pos - ctx.palm_pos
    v_ax = (v * axis).sum(-1)
    v_perp = v - v_ax.unsqueeze(-1) * axis
    gap_n = (v_perp * n).sum(-1) - r    # palm plane -> cup surface; NEGATIVE = palm pressed into the cup
    gap_f = (v_perp * fd).sum(-1) - r   # cup axis ahead of the palm along the fingers, minus r
    h_palm = -v_ax                      # palm offset along the cup axis (env needs |h| <= hh)

    # start orientation kept (palm_normal +y, palm_finger_dir +x); ~0.46 at 20 deg on both axes
    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))
    ang_f = torch.acos(torch.clamp(fd[:, 0], -1.0, 1.0))
    ori = torch.exp(-(ang_n ** 2 + ang_f ** 2) / (2.0 * 0.4 ** 2))

    # distance to the grasp pose; flat inside the approach_done window so the palm may press in
    e_n = torch.relu(gap_n - 0.010) + torch.relu(-0.015 - gap_n)
    e_f = torch.relu((gap_f - 0.006).abs() - 0.010)
    e_h = torch.relu(h_palm.abs() - 0.45 * hh)
    d_pose = torch.sqrt(e_n ** 2 + e_f ** 2 + e_h ** 2 + 1e-12)
    pos = torch.exp(-(d_pose / 0.05) ** 2)    # ~0.6 at the near start (4.5 cm), 1 at the cup
    fine = torch.exp(-(d_pose / 0.018) ** 2)  # sharp version used by the approach

    # ---------------- stage 1: approach (solved last round; kept, only rescaled) ---------------
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    open_pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev.amax(-1) / 0.2) ** 2)
    keep = ori * open_pose
    offset = torch.stack([-(r + 0.0075), -(r + 0.008), 0.25 * hh], dim=-1)  # cup's -y side, behind in +x
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(2.5 * d_world)  # ~0.26 at the raised start (0.38 m), 0.75 at 0.1 m
    approach_reach = s1 * (1.2 * coarse + 1.3 * fine) * (0.2 + 0.8 * keep)  # max 2.5
    approach_keep = s1 * 0.5 * keep                                        # stage 1 max 3.0

    # ---------------- measured contact ---------------------------------------------------------
    touch = _sat(ctx.link_cup_force, 0.5)         # (N,5,3)
    digit_touch = touch.amax(-1)                  # (N,5)
    thumb_touch = digit_touch[:, 0]
    palm_touch = _sat(ctx.palm_cup_force, 0.5)    # (N,)
    palm_firm = _sat(ctx.palm_cup_force, 2.0)     # rewards actually pressing, not grazing
    early_contact_pen = -2.0 * s1 * digit_touch.amax(-1)  # no digit may touch before approach_done

    # ---------------- link geometry in the cup / palm frame ------------------------------------
    lv = ctx.link_pos - ctx.cup_pos[:, None, None, :]      # (N,5,3,3)
    ax = axis[:, None, None, :]
    lh = (lv * ax).sum(-1)                                 # (N,5,3) height along the cup axis
    lrad_vec = lv - lh.unsqueeze(-1) * ax
    lrad = torch.norm(lrad_vec, dim=-1)
    u_n = (lrad_vec * n[:, None, None, :]).sum(-1)         # radial component along the palm normal
    u_f = (lrad_vec * fd[:, None, None, :]).sum(-1)        # radial component along the fingers
    in_band = torch.clamp(1.0 - torch.relu(lh.abs() - hh[:, None, None]) / 0.02, 0.0, 1.0)
    near = torch.exp(-torch.relu(lrad - r[:, None, None] - 0.012) / 0.02) * in_band
    engage = torch.maximum(touch, near).amax(-1)           # (N,5) digit is at the cup body

    # wrap angle around the cup: 0 = palm side (a tangential fingertip touch), pi/2 = forward
    # side, pi = far side. An extended finger lying in the palm plane scores ~0 here.
    theta = torch.atan2(u_f, -u_n)
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)  # saturates ~108 deg, round the far side
    finger_wrap = (touch[:, 1:, :] * wrap[:, 1:, :]).amax(-1)        # (N,4) contact-weighted
    wrap_q = torch.clamp(finger_wrap.sum(-1) / 3.0, 0.0, 1.0)        # 3 wrapped fingers saturate
    # thumb must close on the -fd side, opposite the fingers: 1 at u_f=-r, 0.5 at the side, 0 at +r
    thumb_dir = torch.clamp(0.5 - u_f[:, 0, :] / (2.0 * r[:, None]), 0.0, 1.0)
    thumb_q = (touch[:, 0, :] * (0.4 + 0.6 * thumb_dir)).amax(-1)
    fingers_q = torch.clamp(digit_touch[:, 1:].sum(-1) / 3.0, 0.0, 1.0)  # 3 of 4; pinky may substitute

    # ---------------- finger closure (flexion past the open default pose) ----------------------
    flex = torch.clamp((ctx.hand_q_norm - ctx.hand_default_q_norm) / 0.5, 0.0, 1.0)
    digit_flex = torch.stack([flex[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)  # PD command leads the angle on a touching digit

    # ---------------- palm progress: the one blocked condition ---------------------------------
    # NOT flat at the nominal surface: palm_pos is a virtual point and for a wide cup the tangent
    # point lies past the palm's front edge, so contact needs gap_n driven ~1 cm negative.
    press_geo = torch.clamp((0.008 - gap_n) / 0.018, 0.0, 1.0)   # 0 at +8 mm, 1 at -10 mm
    press = torch.maximum(palm_touch, 0.75 * press_geo)          # geometry alone caps at 0.75

    # ---------------- stage gates --------------------------------------------------------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    upright = torch.clamp(1.0 - (ctx.cup_tilt - 0.35) / 0.35, 0.0, 1.0)   # 1 below 20 deg, 0 at 40
    resting = torch.clamp(1.0 - (dz - 0.03) / 0.03, 0.0, 1.0)             # no stage-2 pay once airborne
    ori_k = 0.5 + 0.5 * ori                                               # turning the hand halves the grasp
    g2 = s2 * ori_k * upright * resting
    g3 = s3 * ori_k * upright     # the same grasp terms keep paying while lifting -> releasing loses them
    g23 = g2 + g3

    # ---------------- stage 2/3: the grasp ladder ----------------------------------------------
    # rung 0: being in the grasp pose is worth little on its own (this replaces the old position rent)
    grasp_pose = 3.0 * g23 * pos
    # rung 1: the last centimetres of palm travel - the largest gradient available in this stage
    palm_reach = 6.0 * g23 * pos * press_geo
    # rung 2: measured palm contact, the single missing condition of the envelope
    palm_contact = g23 * (0.3 + 0.7 * pos) * (6.0 * palm_touch + 2.0 * palm_firm)
    # rung 3: curling the digits around the cup (extended digits that merely graze it score 0)
    finger_curl = 2.5 * g23 * pos * (digit_flex[:, 1:] * engage[:, 1:]).mean(-1)
    thumb_curl = 1.5 * g23 * pos * digit_flex[:, 0] * engage[:, 0]
    # small keep-alive so the contact behaviour already learned is not thrown away
    light_contact = 1.0 * g23 * torch.clamp(digit_touch.sum(-1) / 4.0, 0.0, 1.0)
    # rung 4: contact quality, all multiplied by palm progress -> a fingertip grasp gets a fraction
    grasp_wrap = 4.0 * g23 * press * wrap_q
    grasp_thumb = 3.0 * g23 * press * thumb_q
    grasp_fingers = 3.0 * g23 * press * fingers_q
    grip_squeeze = 1.5 * g23 * press * squeeze
    # rung 5: the envelope_done condition itself (palm AND thumb AND >= 3 fingers)
    envelope_quality = 5.0 * g23 * palm_touch * thumb_touch * fingers_q
    # stage 2: <= 8.0 without palm progress, ~21 with the palm pressed in, ~38.5 with the envelope

    # ---------------- stage 3: lift and hold (every term needs the grip to be kept) -------------
    firm = _sat(ctx.link_cup_force, 0.3).amax(-1)                 # (N,5)
    grip_gate = (0.4 + 0.6 * palm_touch) * firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)
    lift_hold = 5.0 * s3 * grip_gate                              # not a flag bonus: needs the grip
    rise_needed = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)
    lift_frac = torch.clamp(dz / rise_needed, 0.0, 1.0)
    lift_height = 8.0 * s3 * grip_gate * lift_frac
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    lift_goal = s3 * grip_gate * (4.0 * (1.0 - torch.tanh(gd / 0.15))
                                  + 5.0 * torch.exp(-gd / 0.03) + 2.0 * in_tol)
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) \
        * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 4.0 * s3 * grip_gate * torch.exp(-gd / 0.03) * calm
    success_bonus = 60.0 * s3 * ctx.success.float()
    # stage 3 peak ~65 raw > stage 2 peak 38.5 > stage 1 peak 3.0, with no drop at either boundary

    # ---------------- constraints and regularization -------------------------------------------
    # nudges while closing are free up to 5 cm; only shoving the cup away costs more than grasping
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.05) / 0.10, 0.0, 1.0)
    # tilt free below ~15 deg, -4 at the 60 deg termination
    cup_tilt_pen = -4.0 * torch.clamp((ctx.cup_tilt - 0.26) / (math.pi / 3.0 - 0.26), 0.0, 1.0)
    table_pen = -3.0 * torch.clamp((ctx.table_z + 0.005 - ctx.hand_z_min) / 0.02, 0.0, 1.0)
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.02 * (ctx.arm_qd ** 2).sum(-1)  # halved: must not discourage closing in

    raw = {
        "approach_reach": approach_reach,
        "approach_keep": approach_keep,
        "early_contact_pen": early_contact_pen,
        "grasp_pose": grasp_pose,
        "palm_reach": palm_reach,
        "palm_contact": palm_contact,
        "finger_curl": finger_curl,
        "thumb_curl": thumb_curl,
        "light_contact": light_contact,
        "grasp_wrap": grasp_wrap,
        "grasp_thumb": grasp_thumb,
        "grasp_fingers": grasp_fingers,
        "grip_squeeze": grip_squeeze,
        "envelope_quality": envelope_quality,
        "lift_hold": lift_hold,
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
