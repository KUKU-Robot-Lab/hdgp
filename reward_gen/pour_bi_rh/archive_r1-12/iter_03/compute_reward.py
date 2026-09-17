import torch
import math


def _near(d: torch.Tensor, k: float) -> torch.Tensor:
    """Bounded closeness in (0, 1]: 1 at d = 0, ~0.37 at d = 1/k."""
    return torch.exp(-k * d)


def _band_err(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """0 inside [lo, hi], linear distance outside."""
    return torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)


def _cyl_coords(points: torch.Tensor, cup_pos: torch.Tensor, cup_up: torch.Tensor):
    """Cup-cylinder coordinates of (N,F,3) points: axial height, radial distance, radial unit dir."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)                                   # (N,F)
    radial_vec = rel - axial[..., None] * up                          # (N,F,3)
    radial = torch.norm(radial_vec, dim=-1)                           # (N,F)
    radial_dir = radial_vec / (radial[..., None] + 1e-6)
    return axial, radial, radial_dir


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, closure, cup_pos, cup_up, grasped):
    """Per-arm approach / orientation / tip placement / closing / contact / grasp. All (N,)."""
    dtype = cup_pos.dtype
    r_wall = ctx.cup_radius + 0.004          # outer wall ~ inner radius + wall thickness
    lo = ctx.cup_bottom_z + 0.03             # tips >= 3 cm above the cup bottom (table clearance)
    hi = ctx.cup_mouth_z - 0.015             # tips >= 1.5 cm below the rim

    # ---- approach: no flat zone; coarse k=4 from afar + fine k=15 over the last 6 cm (max 1.5)
    d_palm = torch.norm(palm_pos - cup_pos, dim=-1)
    reach = _near(d_palm, 4.0) + 0.5 * _near(torch.clamp(d_palm - 0.06, min=0.0), 15.0)

    # ---- orientation: pads face the cup body (sign-agnostic) and fingertips point toward the cup
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup = cup_pos - palm_pos
    horiz_mask = torch.tensor([1.0, 1.0, 0.0], device=to_cup.device, dtype=dtype)
    to_cup_h = to_cup * horiz_mask
    to_cup_h = to_cup_h / (torch.norm(to_cup_h, dim=-1, keepdim=True) + 1e-6)
    face = torch.abs((normal * to_cup_h).sum(dim=-1))                          # 1 = palm normal points at cup
    d_palm_xy = torch.norm(to_cup[:, :2], dim=-1)
    tip_mid = tips[:, 0:3, :].mean(dim=1)                                      # thumb, index, middle
    d_tip_xy = torch.norm(tip_mid[:, :2] - cup_pos[:, :2], dim=-1)
    ahead = torch.clamp((d_palm_xy - d_tip_xy) / 0.03, 0.0, 1.0)              # tips nearer the cup axis than palm
    orient = _near(torch.clamp(d_palm - 0.08, min=0.0), 5.0) * (0.5 * face + 0.5 * ahead)

    # ---- fingertip placement: distance of each tip to the usable outer wall band
    axial, radial, rdir = _cyl_coords(tips, cup_pos, cup_up)
    rad_out = torch.clamp(radial - r_wall - 0.005, min=0.0)                    # 5 mm radial tolerance
    band = _band_err(axial, lo, hi)
    surf = torch.sqrt(rad_out ** 2 + band ** 2 + 1e-8)                         # (N,F)
    s_thumb = surf[:, 0]
    s_finger = surf[:, 1:3].min(dim=-1).values                                 # thumb+index pinch is fine
    thumb_q = 0.5 * _near(s_thumb, 30.0) + 0.5 * _near(s_thumb, 8.0)
    finger_q = 0.5 * _near(s_finger, 30.0) + 0.5 * _near(s_finger, 8.0)
    tip_q = 0.5 * (thumb_q + finger_q)

    fdir = rdir[:, 1:3, :].mean(dim=1)
    fdir = fdir / (torch.norm(fdir, dim=-1, keepdim=True) + 1e-6)
    opp = torch.clamp(-(rdir[:, 0, :] * fdir).sum(dim=-1), 0.0, 1.0)         # 1 = thumb opposite the fingers
    tip_place = tip_q * (0.5 + 0.5 * opp)

    # ---- dense closing reward, only when the tips are actually at the wall (not when hovering)
    close_near = closure * _near(0.5 * (s_thumb + s_finger), 15.0)

    # ---- hand must be OPEN while the palm is far (> 8 cm); contact does not cancel this
    far = 1.0 - _near(torch.clamp(d_palm - 0.08, min=0.0), 20.0)
    curl_far = torch.clamp((closure - 0.2) / 0.4, 0.0, 1.0) * far

    # ---- contact ramp 0 -> ~0.6 N, ungated; product pays thumb+finger opposition most (max 1.5)
    f_thumb = finger_force[:, 0]
    f_other = finger_force[:, 1:].max(dim=-1).values
    c_t = torch.tanh(f_thumb / 0.3)
    c_o = torch.tanh(f_other / 0.3)
    contact = 0.25 * (c_t + c_o) + c_t * c_o

    # ---- grasp flag (no position gate) and soft hold for lift (flag flickers near 1 N)
    grasped_f = grasped.to(dtype)
    hold = torch.maximum(grasped_f, c_t * c_o)

    # ---- grasp quality: a multiplier on later income, never a gate (0.3 .. 1.0)
    below_soft = torch.clamp(((ctx.cup_mouth_z - 0.015) - axial[:, 0]) / 0.02, 0.0, 1.0)
    quality = below_soft * (0.5 + 0.5 * opp)
    q = 0.3 + 0.7 * quality

    # ---- rim hook: thumb tip at/above rim height over the cup footprint while the palm is near
    near_gate = _near(torch.clamp(d_palm - 0.08, min=0.0), 10.0)
    over_rim = torch.clamp((axial[:, 0] - (ctx.cup_mouth_z - 0.02)) / 0.02, 0.0, 1.0)
    over_footprint = (radial[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = near_gate * over_rim * over_footprint

    return {
        "reach": reach, "orient": orient, "tip_place": tip_place, "close_near": close_near,
        "curl_far": curl_far, "contact": contact, "grasped": grasped_f, "hold": hold,
        "q": q, "rim_hook": rim_hook,
    }


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)
    dtype = zero.dtype

    s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                    ctx.src_hand_closure, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_grasped)
    r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                    ctx.rcv_hand_closure, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_grasped)

    # ------------------------------------------------------------------ stages 0-2: approach, orient, tips, close, grasp
    approach_src = 1.0 * s["reach"]          # weight 1.0 (max 1.5): 12 cm -> 0.78, 6 cm -> 1.29; closing in must pay
    approach_rcv = 1.0 * r["reach"]
    orient_src = 0.4 * s["orient"]           # weight 0.4: pads toward the cup; smaller than approach so it cannot stall it
    orient_rcv = 0.4 * r["orient"]
    tips_src = 0.5 * s["tip_place"]          # weight 0.5: geometry stays below real contact income
    tips_rcv = 0.5 * r["tip_place"]
    close_src = 0.5 * s["close_near"]        # weight 0.5: closing pays only with tips at the wall
    close_rcv = 0.5 * r["close_near"]
    contact_src = 1.0 * s["contact"]         # weight 1.0 (max 1.5): bridge from touching to the 1 N grasp flag
    contact_rcv = 1.0 * r["contact"]
    grasp_src = 2.0 * s["grasped"]           # weight 2.0 > any approach/orient gain: holding must dominate
    grasp_rcv = 2.0 * r["grasped"]
    grasp_both = 1.0 * s["grasped"] * r["grasped"]
    curl_far_pen = -0.4 * (s["curl_far"] + r["curl_far"])    # open hand while far; 0.4 < approach gain
    rim_hook_pen = -1.0 * (s["rim_hook"] + r["rim_hook"])    # separate cost, no longer zeroes contact/grasp

    hold_s, hold_r = s["hold"], r["hold"]
    q_s, q_r = s["q"], r["q"]

    # ------------------------------------------------------------------ stage 3: lift (independent per arm, x quality)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # tanh(h/3cm): 1 cm -> 0.32, 3 cm -> 0.76; weight 3.0 on top of grasp income, x (0.3..1.0) grasp quality
    lift_src = 3.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.03)
    lift_rcv = 3.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.03)
    lifted_src = hold_s * (h_src > 0.03).to(dtype)
    lifted_rcv = hold_r * (h_rcv > 0.03).to(dtype)
    lifted = lifted_src * lifted_rcv

    # ------------------------------------------------------------------ stage 4: align rims (both lifted)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = torch.clamp(0.04 - mouth_dz, min=0.0) + torch.clamp(mouth_dz - 0.12, min=0.0)  # 4..12 cm window
    align_q = (0.5 * _near(mouth_dxy, 5.0) + 0.5 * _near(mouth_dxy, 20.0)) * _near(dz_err, 15.0)
    align = 4.0 * lifted * q_s * align_q     # weight 4.0 > lift 3.0: carrying toward the receiver must pay
    aligned = lifted * (mouth_dxy < 0.05).to(dtype) * (dz_err < 0.015).to(dtype)

    # ------------------------------------------------------------------ stage 5: tilt and pour
    tilt = 3.0 * aligned * q_s * torch.clamp(ctx.src_cup_tilt / 2.0, 0.0, 1.0)   # doorway to pouring only
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads)
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # ------------------------------------------------------------------ stage 6: success
    success = 15.0 * ctx.success.to(dtype)   # 15/step > held+lifted+aligned+tilt income: hold the goal

    # ------------------------------------------------------------------ constraints
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - aligned) * has_beads * torch.tanh(2.0 * torch.clamp(ctx.src_cup_tilt - 0.8, min=0.0))
    # receiver upright only while held (always-on version taught the left arm to stay away)
    rcv_upright_pen = -1.0 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.1, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    knock_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.2, min=0.0))
                        + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.2, min=0.0)))
    # unheld cup pushed across the table: stronger approach must not turn into shoving the cup
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    cup_push_pen = -0.3 * (src_free * torch.tanh(torch.clamp(disp_src - 0.02, min=0.0) / 0.03)
                           + rcv_free * torch.tanh(torch.clamp(disp_rcv - 0.02, min=0.0) / 0.03))
    drop_pen = -1.0 * (src_free * (h_src > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.src_cup_lin_vel[:, 2], min=0.0) / 0.5)
                       + rcv_free * (h_rcv > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.rcv_cup_lin_vel[:, 2], min=0.0) / 0.5))
    cup_speed_pen = -0.2 * (torch.tanh(torch.clamp(torch.norm(ctx.src_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4)
                            + torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4))
    nested_pen = -2.0 * ctx.cups_nested.to(dtype)
    # real-robot safety
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    hand_foreign_pen = -2.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    # palm is not part of a fingertip grasp: pushing the cup with it costs
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ------------------------------------------------------------------ regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "orient_src": orient_src,
        "orient_rcv": orient_rcv,
        "tips_src": tips_src,
        "tips_rcv": tips_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "contact_src": contact_src,
        "contact_rcv": contact_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "grasp_both": grasp_both,
        "curl_far_pen": curl_far_pen,
        "rim_hook_pen": rim_hook_pen,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "align": align,
        "tilt": tilt,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
        "success": success,
        "pre_tilt_pen": pre_tilt_pen,
        "rcv_upright_pen": rcv_upright_pen,
        "knock_pen": knock_pen,
        "cup_push_pen": cup_push_pen,
        "drop_pen": drop_pen,
        "cup_speed_pen": cup_speed_pen,
        "nested_pen": nested_pen,
        "cup_collision_pen": cup_collision_pen,
        "hand_foreign_pen": hand_foreign_pen,
        "palm_push_pen": palm_push_pen,
        "action_rate_pen": action_rate_pen,
        "arm_speed_pen": arm_speed_pen,
    }
    reward = zero
    for v in components.values():
        reward = reward + v
    return reward, components
