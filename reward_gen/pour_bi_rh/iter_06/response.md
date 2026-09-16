## What the task is, and the stages

The robot must move both cups through a fixed chain of physical states, and every stage is only
meaningful once the previous one holds:

0. **Approach** — each palm parks beside its own cup (outside the wall, below the rim), pad facing
   the wall, fingertips nearer the cup axis than the palm.
1. **Pinch posture** — thumb tip on one wall, index/middle tips on the opposite wall, the cup axis
   *between* them, all tips in the wall band below the rim. (This is the state the current policy
   has reached.)
2. **Close the last centimetres** — the tips travel the remaining radial gap to the wall until the
   thumb AND at least one opposing finger are both pressing above 1 N at the same time, which is the
   environment's grasp flag. **This is the bottleneck of this round.**
3. **Lift** both cups clear of the table without dropping them.
4. **Carry together** — source mouth brought over the receiver mouth, 4–12 cm above it, cups never
   touching each other and never nested.
5. **Tilt and pour** — source cup past ~110° with its mouth over the receiver's, beads leave through
   the air.
6. **Hold the goal state** — receiver upright, little spill, success sustained.

## What the training feedback says, and what I changed

**Diagnosis.** Three facts from the run determine the redesign:

1. *Almost all income is paid for standing still.* Of `reward/total` 3.71, the approach/orient/tips
   plateau pays 3.50; the pinch stages pay 0.35; ten components are exactly 0. The marginal reward
   for risking the posture in order to reach contact was worth less than 10 % of what the policy
   already banks for holding it. → the plateau is cut from ~3.5 to ~1.3 (approach 1.5 → 0.5 per hand,
   orient 0.3 → 0.15, `tips_*` deleted), and the contact stages are scaled far above it.

2. *There is no gradient across the last centimetres.* `pinch_geo`'s radial factor
   `clamp((t_rad − r_in)/0.005)` saturates at 1 for **any** tip further out than `r_in + 5 mm`, so
   the posture pays identically with the tips touching the wall and with the tips 5 cm off it — which
   is exactly what the video shows (fingers curled, stopping short, cup "untouched-looking"). The
   only term that did measure the gap, `tip_place`, was capped at 0.4. → new dominant term
   **`close_*` (weight 3.0)**: a two-scale bounded pull on the tip-to-wall surface distance
   (0.35·e^(−12 s) + 0.65·e^(−50 s): 5 cm → 0.25, 1 cm → 0.71, 3 mm → 0.90, contact → 1.0), gated by
   the pinch posture with a floor so it also survives before the posture forms.

3. *The two-sided force term was a product with an identically-zero factor.*
   `pinch_force = (1−e^(−f_t/0.6))·(1−e^(−f_o/0.6))·quality` stayed at exactly 0.0000 all run
   because the thumb force was always 0 — so it gave no gradient on the finger that *did* reach
   0.20 N either. A product of stage signals cannot teach the stage. → replaced by two terms:
   **`touch_*` (1.0)**, an additive per-side force reward where each side's force is paid only when
   the *opposite* side is already at the wall (so pressing one side and shoving the light cup away
   earns nothing), and **`grip_*` (3.5)** on `min(f_thumb, f_other)`, half saturating
   (1−e^(−f/0.25), dense from the first 0.05 N) and half **linear to 1 N** so the gradient does not
   die before the env's threshold. All quality multipliers are floored at 0.5 so they can never zero
   the force gradient again.

**One further fix found by probing the new function at extreme states.** `pinch_geo`'s radial factor
`clamp((t_rad − r_in)/0.005)` is a one-sided ramp with no upper bound, so a hand holding its tips
20 cm either side of the cup axis at rim height scored a *full* pinch posture without ever
approaching the cup — a do-nothing income of 0.8 per hand at maximum distance, which is exactly what
this round is supposed to remove. `tip_ok` now carries an upper ramp as well: full credit out to
`r_wall + 3 cm` (the posture actually reached this round keeps its score), zero past `r_wall + 5 cm`.

**Kept deliberately.** The pre-grasp palm-band geometry, the orientation test, `pinch_geo`,
`rim_hook_pen`, `air_pinch_pen` and `curl_far_pen` are unchanged in form — the operator's decision is
that the posture reached this round must be preserved; only their weights move. The commanded-closure
term `squeeze_*` is raised 0.6 → 1.0 but now additionally requires the tips to be near the wall, so
commanding closure on air pays nothing. The never-reached downstream weights (lift, align, tilt,
pour 200·Δ, spill −100·Δ, success) are kept in their existing proportions but lifted above the new
grasp income; a component that has never been exercised carries no evidence for re-tuning it. The
near-zero penalties (`drop`, `nested`, `cup_speed`, `cup_collision`, `hand_foreign`, `palm_push`,
`rcv_upright`, `pre_tilt`) are guards for the real robot and for stages not yet reached, not shaping,
so they stay; `knock_pen` and `cup_push_pen` are merged into one `disturb_pen`.

**Resulting ladder per hand** (max): approach 0.5 → orient 0.15 → posture 0.8 → wall contact 3.0 →
squeeze 1.0 → one-sided force 1.0 → two-sided press 3.5 → grasp flag 4.0; both hands 3.0; lift 5.0
each; carry 6.0; tilt 5.0; pour +10/bead; success 25/step. Every step of the chain gains more than
the plateau it leaves behind, and the largest single jump (3.0) now sits exactly on the last
centimetres.

**Expected metric movement:** `contact/*_max` rises past 0.2 N and, critically, the thumb stops being
identically 0; `task/*_grasped` leaves 0 for the first time; `reward/close_*` climbs toward 3.0 while
`reward/total` *falls* at the start of training (the plateau was cut on purpose); `task/*_cup_lift`
follows only after the grasp flag fires.

```python
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
    """Per-arm approach / posture / wall closure / press / grasp signals, all (N,) in [0, 1]."""
    dtype = cup_pos.dtype
    mouth = ctx.cup_mouth_z
    r_in = ctx.cup_radius
    r_wall = r_in + 0.004                    # outer wall ~ inner radius + wall thickness
    tip_lo = ctx.cup_bottom_z + 0.03         # tips >= 3 cm above the cup bottom (table clearance)
    tip_hi = mouth - 0.015                   # tips >= 1.5 cm below the rim

    # ---- pre-grasp region for the PALM. UNCHANGED from iter_05: this posture was reached and must be kept;
    #      only its weight in compute_reward drops (1.5 -> 0.5 per hand) so that standing here is no longer the pay-off.
    p_ax, _, p_rad = _cyl(palm_pos[:, None, :], cup_pos, cup_up)
    rad_err = _band_err(p_rad[:, 0], r_wall + 0.01, 0.10)
    ht_err = _band_err(p_ax[:, 0], ctx.cup_bottom_z + 0.03, mouth + 0.01)
    d_pre = torch.sqrt(rad_err ** 2 + ht_err ** 2 + 1e-8)
    reach_lin = torch.clamp(1.0 - d_pre / 0.25, 0.0, 1.0)             # constant slope 4/m, no flat zone
    reach_fine = _near(d_pre, 25.0)                                    # sharp pull over the last few cm
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

    # ---- fingertip -> outer wall distance (radial gap + height-band error), the quantity the policy must drive to 0.
    #      iter_05 measured this only inside tip_place (cap 0.4); here it carries weight 3.0 and a 3 mm tolerance.
    t_ax, t_rv, t_rad = _cyl(tips, cup_pos, cup_up)
    t_dir = t_rv / (t_rad[..., None] + 1e-6)
    rad_out = torch.clamp(t_rad - r_wall - 0.003, min=0.0)
    ht_bad = _band_err(t_ax, tip_lo, tip_hi)                                     # (N,F)
    surf = torch.sqrt(rad_out ** 2 + ht_bad ** 2 + 1e-8)
    s_thumb = surf[:, 0]
    s_finger = surf[:, 1:3].min(dim=-1).values                                   # thumb + index OR middle is enough
    # two scales: the wide one keeps a pull at 5 cm, the sharp one dominates the last centimetre
    # 5 cm -> 0.25, 3 cm -> 0.39, 1 cm -> 0.71, 3 mm -> 0.90, touching -> 1.00
    wall_t = 0.35 * _near(s_thumb, 12.0) + 0.65 * _near(s_thumb, 50.0)
    wall_f = 0.35 * _near(s_finger, 12.0) + 0.65 * _near(s_finger, 50.0)

    fdir = t_dir[:, 1:3, :].mean(dim=1)
    fdir = fdir / (torch.norm(fdir, dim=-1, keepdim=True) + 1e-6)
    opp = torch.clamp(-(t_dir[:, 0, :] * fdir).sum(dim=-1), 0.0, 1.0)            # 1 = thumb opposite the fingers

    # ---- thumb height w.r.t. the rim: low = on the wall (income allowed), over = hooked on the mouth
    thumb_ax = t_ax[:, 0]
    thumb_low = torch.clamp((mouth - thumb_ax) / 0.015, 0.0, 1.0)                # 1 at >= 1.5 cm below rim
    over_rim = torch.clamp((thumb_ax - (mouth - 0.005)) / 0.015, 0.0, 1.0)
    over_footprint = (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = over_rim * over_footprint                                         # flat: independent of palm distance

    # ---- pinch posture gate: the cup axis lies BETWEEN the thumb tip and the index (or middle) tip,
    #      both tips at wall height and outside the opening. This is a POSTURE test only — it deliberately
    #      says nothing about the remaining gap, which is what `close` below measures.
    # radial window: tips outside the opening (lower ramp) AND not straddling from far away (upper ramp).
    # iter_05 had only the lower ramp, which saturates at r_in + 5 mm, so a wide-open hand whose tips sit
    # 20 cm either side of the axis at rim height scored a FULL pinch posture without approaching the cup.
    # Upper ramp: full credit out to r_wall + 3 cm (the posture actually reached this round), zero past r_wall + 5 cm.
    tip_ok = (_near(ht_bad, 40.0)
              * torch.clamp((t_rad - r_in) / 0.005, 0.0, 1.0)
              * torch.clamp((r_wall + 0.05 - t_rad) / 0.02, 0.0, 1.0))             # (N,F)
    pinch_i = _near(_axis_to_segment(t_rv[:, 0, :], t_rv[:, 1, :]), 40.0) * tip_ok[:, 1]
    pinch_m = _near(_axis_to_segment(t_rv[:, 0, :], t_rv[:, 2, :]), 40.0) * tip_ok[:, 2]
    pinch_geo = torch.maximum(pinch_i, pinch_m) * tip_ok[:, 0] * thumb_low

    # ---- NEW, the round's main term: drive both sides of the pinch onto the wall.
    #      The 0.35 floor keeps the signal alive before the posture exists (it replaces iter_05's ungated tip_place),
    #      the 0.65 share makes the correct straddling posture worth nearly 3x more.
    close = 0.5 * (wall_t + wall_f) * (0.35 + 0.65 * pinch_geo) * (0.5 + 0.5 * opp)

    # ---- closing on air: thumb tip within 3.5 cm of the nearest index/middle tip is impossible around a 6.4 cm cup
    gap = torch.minimum(torch.norm(tips[:, 0, :] - tips[:, 1, :], dim=-1),
                        torch.norm(tips[:, 0, :] - tips[:, 2, :], dim=-1))
    air_pinch = torch.clamp((0.035 - gap) / 0.02, 0.0, 1.0)

    # ---- squeeze intent: COMMANDED closure (measured closure is useless — contact freeze stops a real pinch at ~0.25
    #      while closing on air reaches 0.6). Now also requires the tips to be AT the wall, so commanding closure
    #      from 5 cm away pays ~0.3 of what commanding it in contact pays.
    cmd = 0.5 * (torch.clamp(a_hand, -1.0, 1.0) + 1.0)                          # (N,6) 0 = open, 1 = closed
    cmd_close = (cmd[:, 0] + cmd[:, 1] + cmd[:, 2:6].mean(dim=-1)) / 3.0
    squeeze = pinch_geo * cmd_close * (0.3 + 0.7 * 0.5 * (wall_t + wall_f))

    # ---- hand must be open while outside the pre-grasp region (unchanged)
    far = 1.0 - _near(torch.clamp(d_pre - 0.03, min=0.0), 20.0)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far

    # ---- forces. iter_05 multiplied the two sides, and since the thumb force was identically 0 the product
    #      was 0 everywhere and taught nothing. Here the sides are ADDED, and each side's force is paid only
    #      when the OPPOSITE side is already at the wall — so pressing one side and pushing the light cup away
    #      earns nothing, while squeezing both sides pays immediately.
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].max(dim=-1).values
    f_min = torch.minimum(f_t, f_o)
    s_t = 1.0 - torch.exp(-f_t / 0.3)                                            # 0.2 N -> 0.49, 1 N -> 0.96
    s_o = 1.0 - torch.exp(-f_o / 0.3)
    quality = thumb_low * (0.5 + 0.5 * opp)
    gate_q = 0.5 + 0.5 * quality                                                 # FLOORED: never zeroes a force gradient
    touch = (0.6 * s_t * wall_f + 0.4 * s_o * wall_t) * (0.4 + 0.6 * pinch_geo)  # thumb weighted: it never touched at all
    # half dense near zero (first simultaneous contact pays), half LINEAR to the env's 1 N flag so the
    # gradient does not saturate before the threshold that actually defines a grasp
    grip = (0.5 * (1.0 - torch.exp(-f_min / 0.25)) + 0.5 * torch.clamp(f_min / 1.0, 0.0, 1.0)) * gate_q

    # ---- grasp: env flag x quality (floored); soft hold bridges flag flicker around 1 N
    grasped_f = grasped.to(dtype)
    grasp = grasped_f * gate_q
    hold_soft = torch.clamp((f_min - 0.3) / 0.7, 0.0, 1.0)
    hold = torch.maximum(grasped_f, hold_soft) * thumb_low
    q = 0.3 + 0.7 * quality                                                      # multiplier on lift / carry / tilt

    return {
        "reach_lin": reach_lin, "reach_fine": reach_fine, "orient": orient,
        "pinch_geo": pinch_geo, "close": close, "squeeze": squeeze, "air_pinch": air_pinch,
        "curl_far": curl_far, "touch": touch, "grip": grip, "grasp": grasp, "hold": hold,
        "q": q, "rim_hook": rim_hook,
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

    # --------------------------------------------------------------- stage 0: approach (SOLVED — income cut 1.5 -> 0.5)
    # iter_05 paid 3.50 of its 3.71 total for standing here, which is more than every unreached stage combined.
    approach_src = 0.3 * s["reach_lin"] + 0.2 * s["reach_fine"]   # max 0.5 per hand
    approach_rcv = 0.3 * r["reach_lin"] + 0.2 * r["reach_fine"]
    orient_src = 0.15 * s["orient"]          # weight 0.15: helper only, halved with the rest of the plateau
    orient_rcv = 0.15 * r["orient"]

    # --------------------------------------------------------------- stage 1: pinch posture (keep it, do not pay to rest in it)
    pinch_geo_src = 0.8 * s["pinch_geo"]     # weight 0.8 < close 3.0: the posture is a gate, the gap is the goal
    pinch_geo_rcv = 0.8 * r["pinch_geo"]

    # --------------------------------------------------------------- stage 2: close the last centimetres (THE bottleneck)
    close_src = 3.0 * s["close"]             # weight 3.0 > the whole approach plateau (0.65/hand): moving in must dominate holding still
    close_rcv = 3.0 * r["close"]
    squeeze_src = 1.0 * s["squeeze"]         # weight 1.0 (was 0.6): command the fingers shut, but only in contact range
    squeeze_rcv = 1.0 * r["squeeze"]
    touch_src = 1.0 * s["touch"]             # one side pressing while the other is in place — a step, not a resting place
    touch_rcv = 1.0 * r["touch"]
    grip_src = 3.5 * s["grip"]               # weight 3.5: BOTH sides above 1 N is the state the env calls a grasp
    grip_rcv = 3.5 * r["grip"]
    grasp_src = 4.0 * s["grasp"]             # weight 4.0 > grip 3.5: the flag itself is still the target
    grasp_rcv = 4.0 * r["grasp"]
    grasp_both = 3.0 * s["grasp"] * r["grasp"]
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])
    air_pinch_pen = -0.4 * (s["air_pinch"] * (1.0 - hold_s) + r["air_pinch"] * (1.0 - hold_r))
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])

    # --------------------------------------------------------------- stage 3: lift (independent per arm, x quality)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # tanh(h/2.5cm): 5 mm -> 0.20, 3 cm -> 0.83; weight 5.0 > grasp 4.0 so holding still on the table is not the end
    lift_src = 5.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.025)
    lift_rcv = 5.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.025)
    lifted = hold_s * (h_src > 0.03).to(dtype) * hold_r * (h_rcv > 0.03).to(dtype)

    # --------------------------------------------------------------- stage 4: carry together (tilt-aware target)
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
    align = 6.0 * lifted * q_s * carry_q     # weight 6.0 > lift 5.0: carrying toward the receiver must pay

    # --------------------------------------------------------------- stage 5: tilt and pour
    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)                            # 1 cm error -> 0.67
    tilt_r = 5.0 * lifted * q_s * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)       # tilt pays only on the schedule
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads); increments, never the level
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # --------------------------------------------------------------- stage 6: success
    success = 25.0 * ctx.success.to(dtype)   # 25/step on top of all held income: keep the goal state

    # --------------------------------------------------------------- constraints (guards, not shaping)
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - lifted * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
    # receiver upright only while held; -0.8 max stays far below the receiver's grasp income (a pinch tilts a light cup a little)
    rcv_upright_pen = -0.8 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.15, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    # knocking a cup over and shoving it across the table are the same failure (pressing one side only) — one term
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
    # palm is not part of a fingertip grasp: pushing the cup with it costs
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # --------------------------------------------------------------- regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "orient_src": orient_src,
        "orient_rcv": orient_rcv,
        "pinch_geo_src": pinch_geo_src,
        "pinch_geo_rcv": pinch_geo_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "squeeze_src": squeeze_src,
        "squeeze_rcv": squeeze_rcv,
        "touch_src": touch_src,
        "touch_rcv": touch_rcv,
        "grip_src": grip_src,
        "grip_rcv": grip_rcv,
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
        "disturb_pen": disturb_pen,
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
