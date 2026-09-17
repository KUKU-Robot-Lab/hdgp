## 1. What the task means and which stages it needs

Two arms, two light slim cups. The cups are 6 cm in diameter and 11.4 cm tall. From the ctx constants at this cup scale, the rim is 5.4 cm above the cup origin and the bottom is 6.0 cm below it. The right (source) hand must pick up the bead cup and the left (receiver) hand the empty cup. Both cups go up, the source mouth goes over the receiver mouth, and the source cup tilts past about 110° so the beads fall through the air into the receiver. The receiver stays upright, nothing is dropped, nothing collides.

Stages, per arm where applicable:

0. **Approach to a pre-grasp region.** The palm goes *beside* the cup at wall height, with the hand open. This stage has been the bottleneck for four rounds.
1. **Fingertip placement and closing.** The thumb tip goes on the wall opposite the index/middle tips, below the rim. Then the hand closes.
2. **Grasp.** Thumb and another finger press the cup, with the thumb not over the rim.
3. **Lift** both cups a few centimetres.
4. **Carry together.** The source mouth ends up above the receiver mouth *without the cups touching or nesting*.
5. **Tilt and pour.** Pay for beads transferred as increments, and charge for spill as increments.
6. **Success.** Hold the env's success state with the receiver upright.

## 2. Analysis of the feedback

Facts from the feedback, plus formulas and code I checked:

**A. The previous approach term pays the same for coming over the top as for coming from the side.**
- It is a 3-D distance to the cup origin.
- Palm 2 cm above the rim, centred over the mouth: d = 7.4 cm, approach = 1.15.
- Palm beside the cup at mid-height, 6.5 cm from the axis: approach = 1.24.
- The start pose puts the palm 13.7 cm above the cup origin and 16.6 cm to the side (`palm−cup = (−0.08, −0.146, +0.137)`). The straight-line descent therefore heads for the top of the cup, which is where the thumb meets the rim.
- This matches the operator's observations in both rounds: the thumb is caught on the rim, and the left hand hovers above and behind its cup.

**B. The rim-hook penalty grows as the palm comes closer.**
- It is scaled by `exp(−10·max(d−0.08,0))`, which rises 0.41 → 1.0 over the last 9 cm.
- For an env whose thumb is over the rim, that is about 4.1 per metre of extra penalty at 17 cm, while the approach term's slope there is about 3.5/m.
- Net result: moving in from 17 cm earns nothing or loses reward. This is the stall at 16-17 cm, and `rim_hook_pen` grows as the palm moves 21 → 17 cm.
- A penalty whose size depends on palm distance charges for the approach itself, not for the thumb posture.

**C. The closing term pays for closing while far, and "neutral" is already half closed.**
- `close_near` has no palm condition (item 4).
- In the env, action 0 maps to commanded closure `0.5·(a+1) = 0.5`, reached at 0.005/step (`side_rig.synergy_targets`).
- The close gate is fully open at the start distance (`close_gate_radius` 0.30 m).
- So the hands sit 32-42 % closed far away. `close_near` pays for that while `curl_far_pen` charges for it: two terms working against each other on the same variable.

**D. Round-4 history.**
- Contact and grasp income had no thumb-position condition, so thumb-over-rim grasps collected the full contact and grasp reward.
- iter_03 answered that with the distance-ramped penalty from B, which blocked the approach instead.
- The correct lever is to make contact, grasp, hold and lift income *depend on the thumb being below the rim*, and to charge only a small flat cost for a thumb over the mouth.

**E. The later stages have a geometry defect.** No run reached them, but it would block the task.
- iter_03's `aligned` state was mouth xy < 5 cm and source mouth 4-12 cm above the receiver mouth, and `tilt` was paid only in that state.
- With an *upright* source cup, that places the source origin 4-12 cm straight above the receiver origin.
- That is `cups_nested` (< 9 cm), and the 11.4 cm source body would sit inside the receiver cup.
- In a real pour at about 115°, the source origin is about 5 cm sideways and 7-8 cm above the receiver rim level (origin distance about 13 cm). The mouth reaches the receiver *because* the cup is tilted.
- So the horizontal mouth target must depend on tilt: about 10 cm apart while upright, closing to 0 as the tilt reaches about 90°.

**F. Touching the cup is already expensive.**
- A cup knocked flat drops its origin by about the 3 cm drop threshold, which can end the episode and lose all future income.
- The unheld-cup push and knock penalties were 5× the contact income at epoch 0, after which contact went to 0.
- I keep these penalties but make them slightly milder, so learning a delicate fingertip grasp is not scarier than hovering.

## 3. Changes

1. **Approach is now a pre-grasp region in cup-cylinder coordinates.**
   - Palm radial distance must be in [wall + 1 cm, 7 cm]. Palm axial height must be in [bottom + 3 cm, rim + 1 cm]. Error is 0 inside that region.
   - Shape: a linear coarse term over 25 cm (constant slope from the start pose, no flat zone) plus an `exp(−25·d)` fine term. Max 1.5 per hand.
   - Hovering over the mouth now scores about 0.9 instead of the full 1.5.
2. **Rim hook.**
   - The penalty is a flat −0.5 when the thumb tip is at or above the rim over the cup footprint. It no longer depends on palm distance.
   - `thumb_low` (1 when the thumb tip is ≥ 1.5 cm below the rim) multiplies contact, grasp, hold, close and grasp quality. So lift, carry and tilt also need a proper thumb.
3. **Closing** pays only with the tips at the wall (sharpened to k = 40) and the thumb low. `curl_far` starts at closure 0.3 and turns off inside the pre-grasp region.
4. **Grasp quality** = `thumb_low · (0.5 + 0.5·opposition)` scales grasp income (2.0), and `q = 0.3 + 0.7·quality` scales lift, carry and tilt.
5. **Carry and tilt, made tilt-aware.**
   - Horizontal mouth target = `0.10·(1 − clamp((tilt−0.35)/1.2))`. Source mouth height window above the receiver mouth: 4-12 cm.
   - Carry pays for tracking that schedule. Tilt pays only while tracking it closely.
   - `pre_tilt_pen` is waived inside that zone.
6. **Unheld-cup penalties are milder:** push tolerance 3 cm, weight 0.2; knock tilt threshold 0.3 rad.
7. **Unchanged:** safety penalties (cup-cup, hand-foreign, palm push, nesting), bead increments, success bonus, regularisation.

Expected metric movement:
- `task/{src,rcv}_palm_to_cup` should fall below 10 cm, with the palm at wall height beside the cup.
- `task/*_thumb_over_rim_near` should stay low.
- `reward/close_*` should stay near 0 until the palm arrives, and `task/*_closure` far away should fall below 0.3.
- Then `contact/*_max` > 0 → `task/*_grasped` > 0 → `task/*_cup_lift` > 0.

```python
import torch
import math


def _near(d: torch.Tensor, k: float) -> torch.Tensor:
    """Bounded closeness in (0, 1]: 1 at d = 0, ~0.37 at d = 1/k."""
    return torch.exp(-k * d)


def _band_err(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """0 inside [lo, hi], linear distance outside."""
    return torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)


def _cyl_coords(points: torch.Tensor, cup_pos: torch.Tensor, cup_up: torch.Tensor):
    """Cup-cylinder coordinates of (N,K,3) points: axial height, radial distance, radial unit dir."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)                                   # (N,K)
    radial_vec = rel - axial[..., None] * up                          # (N,K,3)
    radial = torch.norm(radial_vec, dim=-1)                           # (N,K)
    radial_dir = radial_vec / (radial[..., None] + 1e-6)
    return axial, radial, radial_dir


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, closure, cup_pos, cup_up, grasped):
    """Per-arm approach / orientation / tip placement / closing / contact / grasp. All (N,)."""
    dtype = cup_pos.dtype
    mouth = ctx.cup_mouth_z
    r_wall = ctx.cup_radius + 0.004          # outer wall ~ inner radius + wall thickness
    tip_lo = ctx.cup_bottom_z + 0.03         # tips >= 3 cm above the cup bottom (table clearance)
    tip_hi = mouth - 0.015                   # tips >= 1.5 cm below the rim

    # ---- pre-grasp region for the PALM: beside the cup (radial wall+1cm .. 7cm) at wall height
    #      (bottom+3cm .. rim+1cm). Error 0 inside. Over-the-mouth palms are OUTSIDE (radial < wall+1cm).
    p_ax, p_rad, _ = _cyl_coords(palm_pos[:, None, :], cup_pos, cup_up)
    rad_err = _band_err(p_rad[:, 0], r_wall + 0.01, 0.07)
    ht_err = _band_err(p_ax[:, 0], ctx.cup_bottom_z + 0.03, mouth + 0.01)
    d_pre = torch.sqrt(rad_err ** 2 + ht_err ** 2 + 1e-8)
    reach_lin = torch.clamp(1.0 - d_pre / 0.25, 0.0, 1.0)            # constant slope 4/m, no flat zone
    reach_fine = _near(d_pre, 25.0)                                    # sharp pull over the last few cm
    near_pre = _near(d_pre, 10.0)

    # ---- orientation: pad normal faces the cup horizontally (sign-agnostic), tips nearer the axis than palm
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup_xy = cup_pos[:, :2] - palm_pos[:, :2]
    d_palm_xy = torch.norm(to_cup_xy, dim=-1)
    u_xy = to_cup_xy / (d_palm_xy[:, None] + 1e-6)
    face = torch.abs((normal[:, :2] * u_xy).sum(dim=-1))
    tip_mid = tips[:, 0:3, :].mean(dim=1)                                      # thumb, index, middle
    d_tip_xy = torch.norm(tip_mid[:, :2] - cup_pos[:, :2], dim=-1)
    ahead = torch.clamp((d_palm_xy - d_tip_xy) / 0.03, 0.0, 1.0)
    orient = near_pre * (0.5 * face + 0.5 * ahead)

    # ---- fingertip placement on the outer-wall band (below the rim)
    t_ax, t_rad, t_dir = _cyl_coords(tips, cup_pos, cup_up)
    rad_out = torch.clamp(t_rad - r_wall - 0.005, min=0.0)                     # 5 mm radial tolerance
    surf = torch.sqrt(rad_out ** 2 + _band_err(t_ax, tip_lo, tip_hi) ** 2 + 1e-8)   # (N,F)
    s_thumb = surf[:, 0]
    s_finger = surf[:, 1:3].min(dim=-1).values                                 # thumb+index pinch is fine
    thumb_q = 0.5 * _near(s_thumb, 30.0) + 0.5 * _near(s_thumb, 8.0)
    finger_q = 0.5 * _near(s_finger, 30.0) + 0.5 * _near(s_finger, 8.0)
    fdir = t_dir[:, 1:3, :].mean(dim=1)
    fdir = fdir / (torch.norm(fdir, dim=-1, keepdim=True) + 1e-6)
    opp = torch.clamp(-(t_dir[:, 0, :] * fdir).sum(dim=-1), 0.0, 1.0)        # 1 = thumb opposite the fingers
    tip_place = 0.5 * (thumb_q + finger_q) * (0.5 + 0.5 * opp)

    # ---- thumb height w.r.t. the rim: low = on the wall (income allowed), over = hooked on the mouth
    thumb_ax = t_ax[:, 0]
    thumb_low = torch.clamp((mouth - thumb_ax) / 0.015, 0.0, 1.0)             # 1 at >= 1.5 cm below rim, 0 at rim
    over_rim = torch.clamp((thumb_ax - (mouth - 0.005)) / 0.015, 0.0, 1.0)    # 0 below rim-0.5cm, 1 at rim+1cm
    over_footprint = (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = over_rim * over_footprint                                       # flat: independent of palm distance

    # ---- closing: only with the tips AT the wall and the thumb low
    s_mean = 0.5 * (s_thumb + s_finger)
    close_near = closure * _near(s_mean, 40.0) * thumb_low                     # 1 cm -> 0.67, 5 cm -> 0.14
    # hand must be open while outside the pre-grasp region (action 0 = 50 % closed in this env)
    far = 1.0 - _near(torch.clamp(d_pre - 0.03, min=0.0), 20.0)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far

    # ---- contact ramp 0 -> ~0.6 N; thumb+finger opposition pays most (max 1.5); void with thumb over the rim
    c_t = torch.tanh(finger_force[:, 0] / 0.3)
    c_o = torch.tanh(finger_force[:, 1:].max(dim=-1).values / 0.3)
    contact = (0.25 * (c_t + c_o) + c_t * c_o) * thumb_low

    # ---- grasp: env flag x quality (thumb below rim, thumb opposite fingers)
    grasped_f = grasped.to(dtype)
    quality = thumb_low * (0.5 + 0.5 * opp)
    grasp = grasped_f * quality
    hold = torch.maximum(grasped_f, c_t * c_o) * thumb_low                    # soft hold (flag flickers near 1 N)
    q = 0.3 + 0.7 * quality                                                    # multiplier on lift / carry / tilt

    return {
        "reach_lin": reach_lin, "reach_fine": reach_fine, "orient": orient, "tip_place": tip_place,
        "close_near": close_near, "curl_far": curl_far, "contact": contact, "grasp": grasp,
        "grasped": grasped_f, "hold": hold, "q": q, "rim_hook": rim_hook,
    }


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)
    dtype = zero.dtype

    s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                    ctx.src_hand_closure, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_grasped)
    r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                    ctx.rcv_hand_closure, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_grasped)

    # ------------------------------------------------------------------ stages 0-2: approach, orient, tips, close, grasp
    # approach max 1.5 per hand: start pose (d_pre ~ 12 cm) -> 0.45, over-the-mouth hover -> 0.90, beside the cup -> 1.5
    approach_src = 0.8 * s["reach_lin"] + 0.7 * s["reach_fine"]
    approach_rcv = 0.8 * r["reach_lin"] + 0.7 * r["reach_fine"]
    orient_src = 0.3 * s["orient"]           # weight 0.3: helper only, must not compete with approach
    orient_rcv = 0.3 * r["orient"]
    tips_src = 0.6 * s["tip_place"]          # weight 0.6: geometry below real contact income (1.5)
    tips_rcv = 0.6 * r["tip_place"]
    close_src = 0.5 * s["close_near"]        # weight 0.5: closing pays only with tips at the wall
    close_rcv = 0.5 * r["close_near"]
    contact_src = 1.0 * s["contact"]         # weight 1.0 (max 1.5): bridge from touching to the 1 N grasp flag
    contact_rcv = 1.0 * r["contact"]
    grasp_src = 2.0 * s["grasp"]             # weight 2.0 > whole approach gain: a proper grasp must dominate
    grasp_rcv = 2.0 * r["grasp"]
    grasp_both = 1.0 * s["grasp"] * r["grasp"]
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])    # open hand while outside the pre-grasp region
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])    # flat cost; the real lever is thumb_low on all income

    hold_s, hold_r = s["hold"], r["hold"]
    q_s, q_r = s["q"], r["q"]

    # ------------------------------------------------------------------ stage 3: lift (independent per arm, x quality)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # tanh(h/3cm): 1 cm -> 0.32, 3 cm -> 0.76; weight 3.0 on top of grasp income
    lift_src = 3.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.03)
    lift_rcv = 3.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.03)
    lifted = hold_s * (h_src > 0.03).to(dtype) * hold_r * (h_rcv > 0.03).to(dtype)

    # ------------------------------------------------------------------ stage 4: carry together (tilt-aware target)
    tilt = ctx.src_cup_tilt
    # upright: mouths 10 cm apart (cup bodies side by side, no nesting); the gap closes to 0 as tilt -> ~90 deg,
    # because in a real pour the source ORIGIN stays ~5 cm aside and the tilted mouth swings over the receiver.
    pour_prog = torch.clamp((tilt - 0.35) / 1.2, 0.0, 1.0)
    tgt_dxy = 0.10 * (1.0 - pour_prog)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    dxy_err = torch.abs(mouth_dxy - tgt_dxy)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = _band_err(mouth_dz, 0.04, 0.12)                                   # source mouth 4..12 cm higher
    carry_q = (0.5 * torch.clamp(1.0 - dxy_err / 0.30, 0.0, 1.0) + 0.5 * _near(dxy_err, 30.0)) * _near(dz_err, 15.0)
    align = 4.0 * lifted * q_s * carry_q     # weight 4.0 > lift 3.0: carrying toward the receiver must pay

    # ------------------------------------------------------------------ stage 5: tilt and pour
    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)                          # 1 cm error -> 0.67
    tilt_r = 3.0 * lifted * q_s * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)     # tilt pays only on the schedule
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads)
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # ------------------------------------------------------------------ stage 6: success
    success = 15.0 * ctx.success.to(dtype)   # 15/step > held+lifted+carry+tilt income: hold the goal

    # ------------------------------------------------------------------ constraints
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    # tilting with beads outside the pour schedule spills them
    pre_tilt_pen = -1.0 * (1.0 - lifted * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
    # receiver upright only while held (always-on version taught the left arm to stay away)
    rcv_upright_pen = -1.0 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.1, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    # unheld cup knocked / shoved: milder than before (a toppled cup already loses all later income)
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
```
