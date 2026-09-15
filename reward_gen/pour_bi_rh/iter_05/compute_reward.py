import torch
import math


def _near(d: torch.Tensor, k: float) -> torch.Tensor:
    """Bounded closeness in (0, 1]: 1 at d = 0, ~0.37 at d = 1/k."""
    return torch.exp(-k * d)


def _band_err(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """0 inside [lo, hi], linear distance outside."""
    return torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)


def _cyl(points: torch.Tensor, cup_pos: torch.Tensor, cup_up: torch.Tensor):
    """Cup-cylinder coordinates of (N,K,3) points: axial height (N,K), radial vector (N,K,3), radial distance (N,K)."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)
    rvec = rel - axial[..., None] * up
    radial = torch.norm(rvec, dim=-1)
    return axial, rvec, radial


def _axis_to_segment(rv_a: torch.Tensor, rv_b: torch.Tensor) -> torch.Tensor:
    """Distance (N,) from the cup axis to the segment joining two tips, in the plane perpendicular to the axis.
    ~0 when the axis lies BETWEEN the tips (cup inside the pinch); the nearest tip's radial distance otherwise."""
    seg = rv_b - rv_a
    u = torch.clamp(-(rv_a * seg).sum(dim=-1) / ((seg * seg).sum(dim=-1) + 1e-8), 0.0, 1.0)
    closest = rv_a + u[:, None] * seg
    return torch.norm(closest, dim=-1)


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, closure, cup_pos, cup_up, grasped, a_hand):
    """Per-arm approach / orientation / pinch geometry / squeeze / contact / grasp. All (N,)."""
    dtype = cup_pos.dtype
    mouth = ctx.cup_mouth_z
    r_in = ctx.cup_radius
    r_wall = r_in + 0.004                    # outer wall ~ inner radius + wall thickness
    tip_lo = ctx.cup_bottom_z + 0.03         # tips >= 3 cm above the cup bottom (table clearance)
    tip_hi = mouth - 0.015                   # tips >= 1.5 cm below the rim

    # ---- pre-grasp region for the PALM (kept from iter_04; radial cap 7 -> 10 cm because the side pinch puts the
    #      palm ~8.3 cm from the cup axis). Over-the-mouth palms stay OUTSIDE (radial < wall + 1 cm).
    p_ax, _, p_rad = _cyl(palm_pos[:, None, :], cup_pos, cup_up)
    rad_err = _band_err(p_rad[:, 0], r_wall + 0.01, 0.10)
    ht_err = _band_err(p_ax[:, 0], ctx.cup_bottom_z + 0.03, mouth + 0.01)
    d_pre = torch.sqrt(rad_err ** 2 + ht_err ** 2 + 1e-8)
    reach_lin = torch.clamp(1.0 - d_pre / 0.25, 0.0, 1.0)             # constant slope 4/m, no flat zone
    reach_fine = _near(d_pre, 25.0)                                     # sharp pull over the last few cm
    near_pre = _near(d_pre, 10.0)

    # ---- orientation: pad normal faces the cup horizontally (sign-agnostic), tips nearer the axis than the palm
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup_xy = cup_pos[:, :2] - palm_pos[:, :2]
    d_palm_xy = torch.norm(to_cup_xy, dim=-1)
    u_xy = to_cup_xy / (d_palm_xy[:, None] + 1e-6)
    face = torch.abs((normal[:, :2] * u_xy).sum(dim=-1))
    tip_mid = tips[:, 0:3, :].mean(dim=1)                                        # thumb, index, middle
    d_tip_xy = torch.norm(tip_mid[:, :2] - cup_pos[:, :2], dim=-1)
    ahead = torch.clamp((d_palm_xy - d_tip_xy) / 0.03, 0.0, 1.0)
    orient = near_pre * (0.5 * face + 0.5 * ahead)

    # ---- fingertips on the outer-wall band (below the rim)
    t_ax, t_rv, t_rad = _cyl(tips, cup_pos, cup_up)
    t_dir = t_rv / (t_rad[..., None] + 1e-6)
    rad_out = torch.clamp(t_rad - r_wall - 0.005, min=0.0)                       # 5 mm radial tolerance
    ht_bad = _band_err(t_ax, tip_lo, tip_hi)                                     # (N,F)
    surf = torch.sqrt(rad_out ** 2 + ht_bad ** 2 + 1e-8)
    s_thumb = surf[:, 0]
    s_finger = surf[:, 1:3].min(dim=-1).values                                   # thumb+index pinch is fine
    thumb_q = 0.5 * _near(s_thumb, 30.0) + 0.5 * _near(s_thumb, 8.0)
    finger_q = 0.5 * _near(s_finger, 30.0) + 0.5 * _near(s_finger, 8.0)
    fdir = t_dir[:, 1:3, :].mean(dim=1)
    fdir = fdir / (torch.norm(fdir, dim=-1, keepdim=True) + 1e-6)
    opp = torch.clamp(-(t_dir[:, 0, :] * fdir).sum(dim=-1), 0.0, 1.0)          # 1 = thumb opposite the fingers
    tip_place = 0.5 * (thumb_q + finger_q) * (0.5 + 0.5 * opp)

    # ---- thumb height w.r.t. the rim: low = on the wall (income allowed), over = hooked on the mouth
    thumb_ax = t_ax[:, 0]
    thumb_low = torch.clamp((mouth - thumb_ax) / 0.015, 0.0, 1.0)               # 1 at >= 1.5 cm below rim
    over_rim = torch.clamp((thumb_ax - (mouth - 0.005)) / 0.015, 0.0, 1.0)
    over_footprint = (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = over_rim * over_footprint                                         # flat: independent of palm distance

    # ---- pinch geometry: the cup axis lies BETWEEN the thumb tip and the index (or middle) tip,
    #      both tips at wall height and outside the opening. 1 cm off-axis -> 0.67, tips on the near side (~4.5 cm) -> 0.17
    tip_ok = _near(ht_bad, 40.0) * torch.clamp((t_rad - r_in) / 0.005, 0.0, 1.0)  # (N,F)
    pinch_i = _near(_axis_to_segment(t_rv[:, 0, :], t_rv[:, 1, :]), 40.0) * tip_ok[:, 1]
    pinch_m = _near(_axis_to_segment(t_rv[:, 0, :], t_rv[:, 2, :]), 40.0) * tip_ok[:, 2]
    pinch_geo = torch.maximum(pinch_i, pinch_m) * tip_ok[:, 0] * thumb_low

    # ---- closing on air: thumb tip within 3.5 cm of the nearest index/middle tip is impossible around a 6.4 cm cup
    gap = torch.minimum(torch.norm(tips[:, 0, :] - tips[:, 1, :], dim=-1),
                        torch.norm(tips[:, 0, :] - tips[:, 2, :], dim=-1))
    air_pinch = torch.clamp((0.035 - gap) / 0.02, 0.0, 1.0)

    # ---- squeeze intent: COMMANDED closure (thumb abd, thumb flex, shared 4-finger command) while the cup is in the pinch.
    #      Measured closure is not used: a real pinch stops at ~0.25 (contact freeze), closing on air reaches 0.6.
    cmd = 0.5 * (torch.clamp(a_hand, -1.0, 1.0) + 1.0)                          # (N,6) 0 = open, 1 = closed
    cmd_close = (cmd[:, 0] + cmd[:, 1] + cmd[:, 2:6].mean(dim=-1)) / 3.0
    squeeze = pinch_geo * cmd_close

    # ---- hand must be open while outside the pre-grasp region (unchanged)
    far = 1.0 - _near(torch.clamp(d_pre - 0.03, min=0.0), 20.0)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far

    # ---- forces: keep rising through the 1 N grasp threshold (0.3 N -> 0.39, 1 N -> 0.81, 1.5 N -> 0.92)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].max(dim=-1).values
    s_t = 1.0 - torch.exp(-f_t / 0.6)
    s_o = 1.0 - torch.exp(-f_o / 0.6)
    quality = thumb_low * (0.5 + 0.5 * opp)
    touch_one = (0.3 * s_t + 0.1 * s_o) * pinch_geo                              # single side, only with the cup in the pinch
    pinch_force = s_t * s_o * quality                                            # both sides at once

    # ---- grasp: env flag x quality; soft hold bridges flag flicker around 1 N
    grasped_f = grasped.to(dtype)
    grasp = grasped_f * quality
    hold_soft = torch.clamp((torch.minimum(f_t, f_o) - 0.3) / 0.7, 0.0, 1.0)
    hold = torch.maximum(grasped_f, hold_soft) * thumb_low
    q = 0.3 + 0.7 * quality                                                      # multiplier on lift / carry / tilt

    return {
        "reach_lin": reach_lin, "reach_fine": reach_fine, "orient": orient, "tip_place": tip_place,
        "pinch_geo": pinch_geo, "squeeze": squeeze, "air_pinch": air_pinch, "curl_far": curl_far,
        "touch_one": touch_one, "pinch_force": pinch_force, "grasp": grasp, "hold": hold, "q": q,
        "rim_hook": rim_hook,
    }


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)
    dtype = zero.dtype
    act = ctx.actions

    s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                    ctx.src_hand_closure, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_grasped, act[:, 6:12])
    r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                    ctx.rcv_hand_closure, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_grasped, act[:, 18:24])
    hold_s, hold_r = s["hold"], r["hold"]
    q_s, q_r = s["q"], r["q"]

    # ------------------------------------------------------------------ stage 0: approach (solved - weights kept)
    approach_src = 0.8 * s["reach_lin"] + 0.7 * s["reach_fine"]   # max 1.5 per hand
    approach_rcv = 0.8 * r["reach_lin"] + 0.7 * r["reach_fine"]
    orient_src = 0.3 * s["orient"]           # weight 0.3: helper only
    orient_rcv = 0.3 * r["orient"]
    tips_src = 0.4 * s["tip_place"]          # weight 0.4: helper; pinch_geo below is the real placement signal
    tips_rcv = 0.4 * r["tip_place"]

    # ------------------------------------------------------------------ stages 1-3: pinch, squeeze, grasp
    pinch_geo_src = 1.0 * s["pinch_geo"]     # weight 1.0 > what the air-pinch resting pose collects (~0.2)
    pinch_geo_rcv = 1.0 * r["pinch_geo"]
    squeeze_src = 0.6 * s["squeeze"]         # weight 0.6: command closing only while the cup is between the tips
    squeeze_rcv = 0.6 * r["squeeze"]
    touch_src = 1.0 * s["touch_one"]         # max 0.4: one-sided contact is a step, not a resting place
    touch_rcv = 1.0 * r["touch_one"]
    pinch_force_src = 2.5 * s["pinch_force"] # weight 2.5 (1.6 at 1 N each): both sides pressing must beat any one-sided state
    pinch_force_rcv = 2.5 * r["pinch_force"]
    grasp_src = 3.0 * s["grasp"]             # weight 3.0: the env grasp flag is the stage the operator needs next
    grasp_rcv = 3.0 * r["grasp"]
    grasp_both = 1.5 * s["grasp"] * r["grasp"]
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])
    air_pinch_pen = -0.4 * (s["air_pinch"] * (1.0 - hold_s) + r["air_pinch"] * (1.0 - hold_r))  # flat in palm distance
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])

    # ------------------------------------------------------------------ stage 4: lift (independent per arm, x quality)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # tanh(h/2.5cm): 5 mm -> 0.20, 3 cm -> 0.83; weight 4.0 > grasp 3.0 so holding still on the table is not the end
    lift_src = 4.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.025)
    lift_rcv = 4.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.025)
    lifted = hold_s * (h_src > 0.03).to(dtype) * hold_r * (h_rcv > 0.03).to(dtype)

    # ------------------------------------------------------------------ stage 5: carry together (tilt-aware target)
    tilt = ctx.src_cup_tilt
    # upright: mouths 10 cm apart (bodies side by side, no nesting); the gap closes to 0 as tilt -> ~90 deg,
    # because in a real pour the source ORIGIN stays ~5 cm aside and the tilted mouth swings over the receiver.
    pour_prog = torch.clamp((tilt - 0.35) / 1.2, 0.0, 1.0)
    tgt_dxy = 0.10 * (1.0 - pour_prog)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    dxy_err = torch.abs(mouth_dxy - tgt_dxy)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = _band_err(mouth_dz, 0.04, 0.12)                                     # source mouth 4..12 cm higher
    carry_q = (0.5 * torch.clamp(1.0 - dxy_err / 0.30, 0.0, 1.0) + 0.5 * _near(dxy_err, 30.0)) * _near(dz_err, 15.0)
    align = 5.0 * lifted * q_s * carry_q     # weight 5.0 > lift 4.0: carrying toward the receiver must pay

    # ------------------------------------------------------------------ stage 6: tilt and pour
    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)                            # 1 cm error -> 0.67
    tilt_r = 4.0 * lifted * q_s * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)       # tilt pays only on the schedule
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads)
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # ------------------------------------------------------------------ stage 7: success
    success = 20.0 * ctx.success.to(dtype)   # 20/step on top of all held income: keep the goal state

    # ------------------------------------------------------------------ constraints
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - lifted * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
    # receiver upright only while held; -0.8 max stays far below the receiver's grasp income (a pinch tilts a light cup a little)
    rcv_upright_pen = -0.8 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.15, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    knock_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.3, min=0.0))
                        + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.3, min=0.0)))
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    cup_push_pen = -0.2 * (src_free * torch.tanh(torch.clamp(disp_src - 0.03, min=0.0) / 0.03)
                           + rcv_free * torch.tanh(torch.clamp(disp_rcv - 0.03, min=0.0) / 0.03))
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
        "pinch_geo_src": pinch_geo_src,
        "pinch_geo_rcv": pinch_geo_rcv,
        "squeeze_src": squeeze_src,
        "squeeze_rcv": squeeze_rcv,
        "touch_src": touch_src,
        "touch_rcv": touch_rcv,
        "pinch_force_src": pinch_force_src,
        "pinch_force_rcv": pinch_force_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "grasp_both": grasp_both,
        "curl_far_pen": curl_far_pen,
        "air_pinch_pen": air_pinch_pen,
        "rim_hook_pen": rim_hook_pen,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "align": align,
        "tilt": tilt_r,
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
