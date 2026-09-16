import torch
import math


def _near(d: torch.Tensor, k: float) -> torch.Tensor:
    """Bounded closeness in (0, 1]: 1 at d = 0, never exactly 0 -> a gradient exists at ANY distance."""
    return torch.exp(-k * d)


def _band_err(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """0 inside [lo, hi], linear distance outside."""
    return torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)


def _cyl(points: torch.Tensor, cup_pos: torch.Tensor, cup_up: torch.Tensor):
    """Cup-cylinder coords of (N,K,3) points: axial height (N,K), radial vector (N,K,3), radius (N,K)."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)
    rvec = rel - axial[..., None] * up
    radial = torch.norm(rvec, dim=-1)
    return axial, rvec, radial


def _axis_to_segments(rv_a: torch.Tensor, rv_b: torch.Tensor) -> torch.Tensor:
    """Distance (N,K) from the cup AXIS to each segment joining rv_a (N,1,3) to rv_b (N,K,3), measured in
    the plane normal to the axis. ~0 ONLY when the axis lies BETWEEN the two tips (= cup inside the jaw)."""
    seg = rv_b - rv_a
    u = torch.clamp(-(rv_a * seg).sum(dim=-1) / ((seg * seg).sum(dim=-1) + 1e-8), 0.0, 1.0)
    closest = rv_a + u[..., None] * seg
    return torch.norm(closest, dim=-1)


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, palm_force, closure,
                cup_pos, cup_up, grasped, a_hand):
    """ONE nested pre-grasp ladder + the contact terms for a single hand.

    Round 9 paid `approach` beside `spread` beside `open` as independent incomes, so the policy
    maximised whichever was cheapest and ignored the rest (12 cm away, fingers flat, 95 % of income).
    Here every pre-grasp condition is a FACTOR of one product chain: L_k = L_{k-1} * c_k. In a sum the
    optimum of a partial sum is reached by maxing the cheap terms; in a product the gradient always
    points at the WEAKEST factor, so the cheap-subset strategy that produced both plateaus is gone.
    """
    dtype = cup_pos.dtype
    r_in = ctx.cup_radius
    r_wall = r_in + 0.004                      # outer wall ~ inner radius + wall thickness
    w_cup = max(2.0 * r_wall, 0.055)           # the cup is ~57 mm across; floored so the target can't collapse
    mouth = ctx.cup_mouth_z
    bot = ctx.cup_bottom_z
    tip_lo = bot + 0.025                       # tips >= 2.5 cm above the cup bottom (table clearance)
    tip_hi = mouth - 0.012                     # tips >= 1.2 cm below the rim (no rim hooking)

    t_ax, t_rv, t_rad = _cyl(tips, cup_pos, cup_up)          # (N,F), (N,F,3), (N,F)
    t_dir = t_rv / (t_rad[..., None] + 1e-6)

    # ---- per-tip distance to the GRASPABLE BAND of the outer wall.
    #      NOTE the deliberate absence of any hard "too far" clamp: round 9's `r_ok` hit exactly 0
    #      beyond 8.2 cm, which is why the 12 cm plateau had no gradient at all out of it. Every
    #      "far" falloff here is an exp with global support.
    rad_out = torch.clamp(t_rad - (r_wall + 0.003), min=0.0)          # outside the wall
    rad_in = torch.clamp(r_in - t_rad, min=0.0) * 3.0                 # inside the bore reads as 3x the error
    h_err = _band_err(t_ax, tip_lo, tip_hi)
    surf = torch.sqrt(rad_out ** 2 + rad_in ** 2 + h_err ** 2 + 1e-8)  # (N,F)
    f_out = torch.clamp((t_rad - r_in) / 0.004, 0.0, 1.0)              # 0 for a tip dipped into the bore

    # ---- PAD DIRECTION, SIGNED (kept from round 9 -- a working gate).
    #      palm_axes[:, 0:3] is the palm normal, i.e. the direction the pad faces.
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup = cup_pos - palm_pos
    u3 = to_cup / (torch.norm(to_cup, dim=-1, keepdim=True) + 1e-6)
    facing = (normal * u3).sum(dim=-1)                       # +1 pad at the cup, -1 back of the hand
    pad = torch.clamp(facing / 0.35, 0.0, 1.0)               # HARD gate, no floor: multiplies the WHOLE ladder
    back = torch.clamp(-facing, 0.0, 1.0)                    # 1 = the cup is behind the hand

    # ================= the six factors of the ladder, all per (thumb, finger j) PAIR =================
    th_surf = surf[:, 0:1]                                   # (N,1)
    fg_surf = surf[:, 1:]                                    # (N,F-1)
    # max(), not mean(): BOTH sides of the pinch must arrive. A wrist parked near the cup, or one
    # finger poking at it, leaves the other side far and the pair distance stays large.
    d_pair = torch.maximum(th_surf, fg_surf)                                     # (N,F-1)
    c1 = 0.5 * _near(d_pair, 4.0) + 0.5 * _near(d_pair, 25.0)   # reach: coarse pull + fine pull

    # c2 = pad (scalar per env, broadcast below)

    gap = torch.norm(tips[:, 0:1, :] - tips[:, 1:, :], dim=-1)                   # (N,F-1)
    # OPENING. Cubed sub-threshold ramp: a 45 mm hand keeps a strictly positive d(reward)/d(gap)
    # but collects only ~0.2 of the factor, so it can never be a resting place.
    c3 = (torch.clamp((gap - 0.020) / max(w_cup + 0.006 - 0.020, 1e-3), 0.0, 1.0) ** 3
          * torch.clamp((0.145 - gap) / 0.030, 0.0, 1.0))      # upper clamp: no hyperextension farming

    h_pair = torch.maximum(h_err[:, 0:1], h_err[:, 1:])
    c4 = _near(h_pair, 25.0)                                   # both tips at wall height

    # STRADDLE -- this is `cup_in_pocket`, the metric that was 0 for all of round 9 while the gap sat at
    # 84-93 mm. It is ~0 for an opening beside the cup and 1 only when the cup axis is between the tips.
    # This factor, not a separate `spread` income, is what makes opening pay: it is the geometry that
    # already had to be satisfied, so it cannot become a rung of its own.
    seg = _axis_to_segments(t_rv[:, 0:1, :], t_rv[:, 1:, :])                     # (N,F-1)
    adm = f_out[:, 0:1] * f_out[:, 1:]
    c5 = _near(seg, 45.0) * adm

    opp = -(t_dir[:, 1:, :] * t_dir[:, 0:1, :]).sum(dim=-1)                      # (N,F-1) in [-1,1]
    c6 = torch.clamp((opp - 0.3) / 0.5, 0.0, 1.0)       # HARD, no floor (kept): 107 deg -> 0, 120 deg -> 1
    c7 = (0.35 * _near(d_pair, 12.0) + 0.65 * _near(d_pair, 60.0)) * adm         # tips in contact range

    # ---- a hand laid ON the cup with the thumb tucked in among the fingers is not a jaw, however wide
    #      it measures (kept from round 9). Zeroes the whole ladder there.
    at_cup = _near(surf.min(dim=-1).values, 15.0)
    u_th = t_dir[:, 0, :]
    u_fm = t_dir[:, 1:3, :].mean(dim=1)
    u_fm = u_fm / (torch.norm(u_fm, dim=-1, keepdim=True) + 1e-6)
    opp_mid = -(u_th * u_fm).sum(dim=-1)                       # -1 = thumb among the fingers
    same_side = torch.clamp((-0.1 - opp_mid) / 0.5, 0.0, 1.0)
    jaw_ok = 1.0 - at_cup * same_side

    # ================= ONE nested ladder. L_k requires every condition of L_{k-1}. =================
    p = pad[:, None]
    L1 = c1 * p                      # arrived, pad toward the cup
    L2 = L1 * c3                     # ... opened wider than the cup
    L3 = L2 * c4                     # ... at wall height
    L4 = L3 * c5                     # ... the cup is between the tips   <- cup_in_pocket
    L5 = L4 * c6                     # ... genuinely opposed
    L6 = L5 * c7                     # ... on the wall: one flex from the grasp flag
    # Back-loaded weights (max 2.00). Everything payable WITHOUT the cup in the jaw is capped at 0.50,
    # and 1.10 of the 2.00 needs a fingertip in contact range: the best standing posture the ladder
    # admits is already the grasp precondition.
    j = jaw_ok[:, None]
    lad1 = (0.10 * L1 * j).max(dim=-1).values
    lad2 = (0.15 * L2 * j).max(dim=-1).values
    lad3 = (0.25 * L3 * j).max(dim=-1).values
    lad4 = (0.40 * L4 * j).max(dim=-1).values
    lad5 = (0.50 * L5 * j).max(dim=-1).values
    lad6 = (0.60 * L6 * j).max(dim=-1).values
    ready = (L6 * j).max(dim=-1).values

    # ================= contact: only an OPPOSING finger on the far wall counts =================
    thumb_low = torch.clamp((mouth - t_ax[:, 0]) / 0.015, 0.0, 1.0)
    opp_best = c6.max(dim=-1).values
    f_t = finger_force[:, 0]
    f_o = (finger_force[:, 1:] * c6 * adm).max(dim=-1).values     # force only counts where opposition does
    f_min = torch.minimum(f_t, f_o)
    # force sensors report contact in ANY direction, so without `pad` the back of the fingers reads as
    # grasp progress (round 8's failure). q is hard: no pad, no opposition, no thumb below the rim -> no pay.
    q_grasp = pad * opp_best * thumb_low * jaw_ok
    touch = 0.5 * ((1.0 - torch.exp(-f_t / 0.3)) + (1.0 - torch.exp(-f_o / 0.3))) * q_grasp
    grip = (0.5 * (1.0 - torch.exp(-f_min / 0.25)) + 0.5 * torch.clamp(f_min / 1.0, 0.0, 1.0)) * q_grasp
    grasped_f = grasped.to(dtype)
    grasp = grasped_f * q_grasp                                   # the env flag alone is not enough
    hold_soft = torch.clamp((f_min - 0.3) / 0.7, 0.0, 1.0)
    hold = torch.maximum(grasped_f, hold_soft) * q_grasp
    q = 0.3 + 0.7 * q_grasp

    # squeeze intent from the COMMANDED closure, only inside a pose that can actually trap the cup
    cmd = 0.5 * (torch.clamp(a_hand, -1.0, 1.0) + 1.0)
    cmd_close = (cmd[:, 0] + cmd[:, 1] + cmd[:, 2:6].mean(dim=-1)) / 3.0
    squeeze = ready * cmd_close

    # ---- failure postures
    # air pinch = closing on nothing AT the cup. Measured on the PINCH pair (thumb vs index/middle)
    # only: the thumb-to-pinky distance is naturally small, and taking the min over all four pairs
    # would pay the policy to splay every finger -- which is round 9's exact pathology.
    far = 1.0 - _near(surf.min(dim=-1).values, 20.0)
    gap_pinch = gap[:, 0:2].min(dim=-1).values
    air_pinch = torch.clamp((w_cup - 0.005 - gap_pinch) / 0.025, 0.0, 1.0) * (1.0 - far)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far
    over_rim = torch.clamp((t_ax[:, 0] - (mouth - 0.005)) / 0.015, 0.0, 1.0)
    rim_hook = over_rim * (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    # fingers dipped INSIDE the bore (gripping the cup from the inside is not the task)
    inside = (torch.clamp((r_in - t_rad) / 0.006, 0.0, 1.0)
              * torch.clamp((mouth + 0.01 - t_ax) / 0.01, 0.0, 1.0)
              * torch.clamp((t_ax - bot) / 0.01, 0.0, 1.0))
    bore = inside.max(dim=-1).values
    # pressing a cup with a palm turned away, and loitering near it back-first
    dorsal = back * (0.15 * at_cup + 1.2 * torch.tanh((finger_force.sum(dim=-1) + palm_force) / 1.0))

    return {
        "lad1": lad1, "lad2": lad2, "lad3": lad3, "lad4": lad4, "lad5": lad5, "lad6": lad6,
        "ready": ready, "touch": touch, "grip": grip, "grasp": grasp, "hold": hold, "q": q,
        "squeeze": squeeze, "air_pinch": air_pinch, "curl_far": curl_far, "rim_hook": rim_hook,
        "bore": bore, "dorsal": dorsal,
    }


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)
    dtype = zero.dtype
    act = ctx.actions

    s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                    ctx.src_palm_force, ctx.src_hand_closure, ctx.src_cup_pos, ctx.src_cup_up,
                    ctx.src_grasped, act[:, 6:12])
    r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                    ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                    ctx.rcv_grasped, act[:, 18:24])
    hold_s, hold_r = s["hold"], r["hold"]
    q_s, q_r = s["q"], r["q"]

    # ------------------------------------------- stages 0-2: the single pre-grasp ladder (max 2.0/hand)
    lad1_src, lad1_rcv = s["lad1"], r["lad1"]
    lad2_src, lad2_rcv = s["lad2"], r["lad2"]
    lad3_src, lad3_rcv = s["lad3"], r["lad3"]
    lad4_src, lad4_rcv = s["lad4"], r["lad4"]
    lad5_src, lad5_rcv = s["lad5"], r["lad5"]
    lad6_src, lad6_rcv = s["lad6"], r["lad6"]
    # min(), not sum(): one hand improving while the other goes backwards must not pay.
    ready_both = 1.0 * torch.minimum(s["ready"], r["ready"])

    # ------------------------------------------- stage 3: contact and the grasp flag
    # 1.5 + 3.0 + 5.0 = 9.5 per hand, ~5x the ENTIRE ladder: reaching the top of the ladder is worth
    # far less than the single finger flex that turns it into a grasp.
    touch_src, touch_rcv = 1.5 * s["touch"], 1.5 * r["touch"]
    grip_src, grip_rcv = 3.0 * s["grip"], 3.0 * r["grip"]
    grasp_src, grasp_rcv = 5.0 * s["grasp"], 5.0 * r["grasp"]
    grasp_both = 4.0 * s["grasp"] * r["grasp"]
    squeeze_src, squeeze_rcv = 0.6 * s["squeeze"], 0.6 * r["squeeze"]
    air_pinch_pen = -0.5 * (s["air_pinch"] * (1.0 - hold_s) + r["air_pinch"] * (1.0 - hold_r))
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])
    bore_pen = -0.6 * (s["bore"] + r["bore"])
    dorsal_pen = -1.5 * (s["dorsal"] + r["dorsal"])     # the back of the hand on the cup is never progress

    # ------------------------------------------- stage 4: lift
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_src = 8.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.025)
    lift_rcv = 8.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.025)
    lifted = hold_s * (h_src > 0.03).to(dtype) * hold_r * (h_rcv > 0.03).to(dtype)

    # ------------------------------------------- stage 5: carry together (tilt-aware target)
    tilt = ctx.src_cup_tilt
    pour_prog = torch.clamp((tilt - 0.35) / 1.2, 0.0, 1.0)
    tgt_dxy = 0.10 * (1.0 - pour_prog)         # upright: mouths 10 cm apart, closing as the source tilts
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    dxy_err = torch.abs(mouth_dxy - tgt_dxy)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = _band_err(mouth_dz, 0.04, 0.12)                 # source mouth 4..12 cm higher
    carry_q = (0.5 * torch.clamp(1.0 - dxy_err / 0.30, 0.0, 1.0) + 0.5 * _near(dxy_err, 30.0)) * _near(dz_err, 15.0)
    align = 8.0 * lifted * q_s * carry_q

    # ------------------------------------------- stage 6: tilt and pour
    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)
    tilt_r = 7.0 * lifted * q_s * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)
    pour_delta = 200.0 * ctx.d_in_target       # +10 per transferred bead; increments, never the level
    spill_delta = -100.0 * ctx.d_spill         # -5 per spilled bead: costly, never worth refusing to pour

    # ------------------------------------------- stage 7: success
    success = 30.0 * ctx.success.to(dtype)

    # ------------------------------------------- constraints (guards, not shaping)
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - lifted * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
    rcv_upright_pen = -0.8 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.15, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    disturb_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.3, min=0.0))
                          + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.3, min=0.0))) \
                  - 0.2 * (src_free * torch.tanh(torch.clamp(disp_src - 0.03, min=0.0) / 0.03)
                           + rcv_free * torch.tanh(torch.clamp(disp_rcv - 0.03, min=0.0) / 0.03))
    drop_pen = -1.0 * (src_free * (h_src > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.src_cup_lin_vel[:, 2], min=0.0) / 0.5)
                       + rcv_free * (h_rcv > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.rcv_cup_lin_vel[:, 2], min=0.0) / 0.5))
    cup_speed_pen = -0.2 * (torch.tanh(torch.clamp(torch.norm(ctx.src_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4)
                            + torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4))
    nested_pen = -2.0 * ctx.cups_nested.to(dtype)
    # real-robot safety: the cups never touch each other and each hand touches only its own cup
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    hand_foreign_pen = -2.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    # the palm is not part of a fingertip grasp: pushing the cup with it costs
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ------------------------------------------- regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "lad1_reach_src": lad1_src, "lad1_reach_rcv": lad1_rcv,
        "lad2_open_src": lad2_src, "lad2_open_rcv": lad2_rcv,
        "lad3_height_src": lad3_src, "lad3_height_rcv": lad3_rcv,
        "lad4_straddle_src": lad4_src, "lad4_straddle_rcv": lad4_rcv,
        "lad5_oppose_src": lad5_src, "lad5_oppose_rcv": lad5_rcv,
        "lad6_ready_src": lad6_src, "lad6_ready_rcv": lad6_rcv,
        "ready_both": ready_both,
        "touch_src": touch_src, "touch_rcv": touch_rcv,
        "grip_src": grip_src, "grip_rcv": grip_rcv,
        "grasp_src": grasp_src, "grasp_rcv": grasp_rcv, "grasp_both": grasp_both,
        "squeeze_src": squeeze_src, "squeeze_rcv": squeeze_rcv,
        "air_pinch_pen": air_pinch_pen, "curl_far_pen": curl_far_pen,
        "rim_hook_pen": rim_hook_pen, "bore_pen": bore_pen, "dorsal_pen": dorsal_pen,
        "lift_src": lift_src, "lift_rcv": lift_rcv,
        "align": align, "tilt": tilt_r,
        "pour_delta": pour_delta, "spill_delta": spill_delta, "success": success,
        "pre_tilt_pen": pre_tilt_pen, "rcv_upright_pen": rcv_upright_pen,
        "disturb_pen": disturb_pen, "drop_pen": drop_pen, "cup_speed_pen": cup_speed_pen,
        "nested_pen": nested_pen, "cup_collision_pen": cup_collision_pen,
        "hand_foreign_pen": hand_foreign_pen, "palm_push_pen": palm_push_pen,
        "action_rate_pen": action_rate_pen, "arm_speed_pen": arm_speed_pen,
    }
    reward = zero
    for v in components.values():
        reward = reward + v
    return reward, components
