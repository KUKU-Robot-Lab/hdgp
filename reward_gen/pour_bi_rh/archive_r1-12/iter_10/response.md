## What the task is, and the stages

The robot must move 20 beads from the right (source) cup into the left (receiver) cup by pouring
through the air. Stages: **(0)** each hand opens wider than its 57 mm cup and brings the fingertips
down beside it with the palm facing it; **(1)** the cup ends up *between* thumb tip and an opposing
fingertip (the straddle), at wall height, outside the bore; **(2)** both sides press until the
environment's grasp flag fires (thumb ≥ 1 N **and** another finger ≥ 1 N); **(3)** both cups lift off
the table; **(4)** the source mouth is carried over the receiver mouth, cups never touching; **(5)**
the source cup tilts past ~70° so the beads fall out through the air; **(6)** success — half the
beads in the receiver, little spill, receiver upright, cups not nested.

## Diagnosis of round 10

Round 10 reached stage 1 and stopped dead there: pocket 0.997, oppose 0.922, thumb 25 mm below the
rim, palm at 9.4 cm — and contact frozen at 0.28 N across 66 extension epochs. Two mechanisms, both
in the reward:

**1. The reward paid the hand to hold its gap open at exactly the width it stalled at.** The opening
factor was `c3 = clamp((gap-0.020)/(w_cup+0.006-0.020), 0, 1)**3`, which saturates at
`gap = w_cup + 0.006 = 70 mm`. The measured settle point was **70.6 mm**. `c3` multiplies rungs
2 - 6, i.e. ~1.51 of the 1.60 the ladder was paying, so closing the 6.6 mm onto the 64 mm wall cost
`(1 - 0.88³)·1.51 ≈ 0.48` of guaranteed income. Meanwhile the gain from squeezing was ~nil, because
`touch` used `1 - exp(-f/0.3)`, which is **already 61 % collected at 0.28 N**, and `grip` used
`min(f_thumb, f_finger)`, which is 0 while only one finger touches. So in the 0.28 → 1 N band the
reward was flat-to-negative. The policy was not failing to find the squeeze; it was being paid to
refuse it. Note *which* knob this is: the ramp's **saturation point**, not its zero-crossing —
lowering it to `w_cup - 0.006 = 58 mm` makes the whole contact band flat-topped, so closing to the
wall is free, while crushing below the cup still falls off.

**2. `ready_both = 2.0·min(src, rcv)` is structurally zero while one hand is at zero,** so the term
meant to couple the hands contributed nothing for 665 epochs and its gradient w.r.t. the lagging
hand was also zero. The receiver's only income was its own rung 1, whose slope at 27 cm was ~0.07 per
metre — below the action-rate and arm-speed costs of moving. It drifted *away*.

## What changed

Kept exactly as the operator required: the nested-product ladder, the floorless opposition gate
`c6`, the signed pad-facing gate `pad` (no floor, multiplies the whole ladder), and dorsal contact
counting as zero progress.

- **`c3` saturates at `w_cup - 0.006` instead of `w_cup + 0.006`.** Closing from 70.6 mm to the wall
  no longer costs ladder income. The "open wider than the cup" duty is carried by `c5` (straddle),
  which is the condition that actually had to hold.
- **New top rung `L7 = L6 · c8`, `c8 = exp(-40·max(gap - w_cup, 0))`** — the geometric pinch. This is
  the one motion contact freeze cannot block, and it is the motion that creates the force.
- **Force terms rebuilt around the 0.28 → 1 N band.** `_press_ramp` is 0.39 at 0.28 N, 0.89 at 1.0 N,
  1.0 at 1.2 N. `press` pays each side *separately* (so a single finger at 0.28 N still has a
  gradient when the thumb is at zero — the old `min` had none), `press_both` pays the min at 5.0.
  `touch` is demoted to 0.8 with a fast 0.12 N scale: first contact is cheap, pressing is where the
  money is. `squeeze` now reads thumb flexion and the index/middle commands only, not thumb abduction.
- **`ready_both` deleted; replaced by `lag_pull`, a self-extinguishing handicap:**
  `3.0 · clamp(prog_other - prog_self, 0, 1) · reach_self`, with `reach` a long-support
  `exp(-2·d)` on the palm-pair distance. It is **zero when the hands are level** (so it is not a
  do-nothing income — verified at the parked state below), and it is largest exactly when one hand
  is far behind. `c1`'s coarse half also moved from `k=4` to `k=2.5` to stretch the long-range pull.
- **Downstream gates opened.** `align` / `tilt` were gated on *both* cups being lifted, which the
  receiver made unreachable; they are now gated on the source being lifted and the receiver cup being
  upright (`rcv_up`), with `lift_both` and `success_bi` keeping the bimanual solution strictly better.
- `air_pinch`'s proximity gate narrowed from `k=20` to `k=40`: at `k=20` it fired 13 cm out and made
  the receiver's approach net-negative.

## Measured on synthetic extreme states

Numbers are totals from a batched run of this exact function (cup r=28 mm, mouth +55 mm, F=5).

| state | total |
|---|---|
| **(z) do-nothing**, both hands parked 27 cm out | **+0.146** |
| **(a) this round's end state** — straddle, oppose 0.95, gap 70.6 mm, 0.28 N on one finger, no flag, receiver idle 27 cm | **+5.551** |
| (a→) gap 68.8 mm, 0.10 / 0.40 N | +7.094 |
| (a→) gap 66.0 mm, 0.30 / 0.50 N | +8.931 |
| (a→) gap 64.0 mm (tips on the wall), 0.60 / 0.60 N, still no flag | +10.654 |
| (a→) gap 64.0 mm, 1.0 / 1.0 N, **flag** | +18.474 |
| **(b) same posture as (a), 1.2 / 1.2 N, flag** | **+18.983  (a + 13.43)** |
| (b2) closed to the wall, 1.2 / 1.2 N, flag | +19.313  (a + 13.76) |
| **(c) palm turned away, back of the fingers on the cup, 1.2 / 1.2 N, flag raised** | **−1.922** |
| **(d) thumb on the same side as the fingers, 1.2 / 1.2 N, flag raised** | **−0.258** |

- **(a) is not a resting place.** Every 2 mm of closing and every 0.2 N of press pays: the chain
  5.551 → 7.094 → 8.931 → 10.654 → 18.474 is strictly monotone with no flat step, and the single
  largest jump (+7.8) is the last one into the flag. Under round 10's reward the equivalent first
  step was *negative*. (a) sits 5.4 above the do-nothing baseline, but the gradient out of it is 13.4
  — 2.5× the whole standing income.
- **(b) dominates (a) by +13.43**, of which `grasp_src` 6.0, `press_both_src` 4.99, `press_src` 2.49.
- **(c) earns nothing and is punished.** `pad = clamp(facing/0.35, 0, 1) = 0` with the palm turned
  away, and `pad` multiplies the entire ladder *and* `q_grasp`, so every rung, `touch`, `press`,
  `press_both` and `grasp` read exactly 0.000 despite 1.2 N on two fingers and the flag raised; only
  `dorsal_pen` −1.99 remains. Real force on the back of the fingers is worth less than doing nothing.
- **(d) earns nothing.** With the thumb among the fingers, `opp ≈ -1` so the floorless `c6 = 0` kills
  `L5..L7` and `q_grasp = pad·opp_best·thumb_low·jaw_ok = 0` kills every contact term; `jaw_ok` zeroes
  the lower rungs too. Residual +0.0008 on rung 1, net −0.258 after the regularisers.

**(e) One hand ready vs both halfway.** (e1) source fully ready and grasped at 1.2 N with the
receiver idle at 27 cm = **+19.313**; (e2) both hands in the (a) straddle at 0.28 N = **+8.596**.
**My design pays (e1) more, deliberately**, and that is not the `min` pathology returning. The failure
of `ready_both = 2.0·min(src, rcv)` was not that it ranked e1 below e2 — it was that its *gradient*
with respect to the lagging hand was identically zero, so nothing rewarded the receiver for moving.
What matters is the marginal return on the receiver's next metre, and in (e1) that is now the
steepest ascent on the board:

```
rcv palm 26.8 cm  +19.313
rcv palm 21.8 cm  +19.518   (+0.204)
rcv palm 16.8 cm  +19.772   (+0.254)
rcv palm 11.8 cm  +20.056   (+0.284)
rcv into the straddle  +21.572
rcv also grasped 1.2/1.2 N  +41.011      <- both hands ready
```

Every 5 cm the receiver closes is worth +0.20 to +0.28 (round 10's reward paid −0.05 per 10 cm — it
was literally cheaper to stand still), and completing the second grasp is worth **+21.7**, more than
the source's entire grasp. Refusing to pay a real one-handed grasp would have thrown away the only
thing round 10 achieved; instead the handicap makes the lagging hand the cheapest remaining
improvement, and it decays to zero on its own once the hands are level, so it can never become an
income.

```python
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
    """Distance (N,K) from the cup AXIS to each segment joining rv_a (N,1,3) to rv_b (N,K,3)."""
    seg = rv_b - rv_a
    u = torch.clamp(-(rv_a * seg).sum(dim=-1) / ((seg * seg).sum(dim=-1) + 1e-8), 0.0, 1.0)
    closest = rv_a + u[..., None] * seg
    return torch.norm(closest, dim=-1)


def _press_ramp(f: torch.Tensor) -> torch.Tensor:
    """Force -> [0,1] with its gradient placed IN the 0.28 N -> 1.0 N band that round 10 never crossed.
    0.28 N -> 0.39, 0.5 N -> 0.55, 1.0 N -> 0.89, 1.2 N -> 1.00.
    The old reward used 1 - exp(-f/0.3), which is already 61 % collected at 0.28 N: the policy had
    banked almost the whole contact income at the force it stalled on, so squeezing harder paid ~nothing."""
    return 0.35 * (1.0 - torch.exp(-f / 0.25)) + 0.65 * torch.clamp(f / 1.2, 0.0, 1.0)


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, palm_force, closure,
                cup_pos, cup_up, grasped, a_hand):
    dtype = cup_pos.dtype
    r_in = ctx.cup_radius
    r_wall = r_in + 0.004
    w_cup = max(2.0 * r_wall, 0.055)
    mouth = ctx.cup_mouth_z
    bot = ctx.cup_bottom_z
    tip_lo = bot + 0.025
    tip_hi = mouth - 0.012

    t_ax, t_rv, t_rad = _cyl(tips, cup_pos, cup_up)
    t_dir = t_rv / (t_rad[..., None] + 1e-6)

    rad_out = torch.clamp(t_rad - (r_wall + 0.003), min=0.0)
    rad_in = torch.clamp(r_in - t_rad, min=0.0) * 3.0
    h_err = _band_err(t_ax, tip_lo, tip_hi)
    surf = torch.sqrt(rad_out ** 2 + rad_in ** 2 + h_err ** 2 + 1e-8)
    f_out = torch.clamp((t_rad - r_in) / 0.004, 0.0, 1.0)

    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup = cup_pos - palm_pos
    u3 = to_cup / (torch.norm(to_cup, dim=-1, keepdim=True) + 1e-6)
    facing = (normal * u3).sum(dim=-1)
    pad = torch.clamp(facing / 0.35, 0.0, 1.0)
    back = torch.clamp(-facing, 0.0, 1.0)

    th_surf = surf[:, 0:1]
    fg_surf = surf[:, 1:]
    d_pair = torch.maximum(th_surf, fg_surf)
    c1 = 0.60 * _near(d_pair, 2.5) + 0.40 * _near(d_pair, 25.0)

    gap = torch.norm(tips[:, 0:1, :] - tips[:, 1:, :], dim=-1)
    open_sat = max(w_cup - 0.006, 0.030)
    c3 = (torch.clamp((gap - 0.020) / max(open_sat - 0.020, 1e-3), 0.0, 1.0) ** 3
          * torch.clamp((0.145 - gap) / 0.030, 0.0, 1.0))

    h_pair = torch.maximum(h_err[:, 0:1], h_err[:, 1:])
    c4 = _near(h_pair, 25.0)

    seg = _axis_to_segments(t_rv[:, 0:1, :], t_rv[:, 1:, :])
    adm = f_out[:, 0:1] * f_out[:, 1:]
    c5 = _near(seg, 45.0) * adm

    opp = -(t_dir[:, 1:, :] * t_dir[:, 0:1, :]).sum(dim=-1)
    c6 = torch.clamp((opp - 0.3) / 0.5, 0.0, 1.0)
    c7 = (0.35 * _near(d_pair, 12.0) + 0.65 * _near(d_pair, 60.0)) * adm
    c8 = _near(torch.clamp(gap - w_cup, min=0.0), 40.0)

    at_cup = _near(surf.min(dim=-1).values, 15.0)
    u_th = t_dir[:, 0, :]
    u_fm = t_dir[:, 1:3, :].mean(dim=1)
    u_fm = u_fm / (torch.norm(u_fm, dim=-1, keepdim=True) + 1e-6)
    opp_mid = -(u_th * u_fm).sum(dim=-1)
    same_side = torch.clamp((-0.1 - opp_mid) / 0.5, 0.0, 1.0)
    jaw_ok = 1.0 - at_cup * same_side

    p = pad[:, None]
    L1 = c1 * p
    L2 = L1 * c3
    L3 = L2 * c4
    L4 = L3 * c5
    L5 = L4 * c6
    L6 = L5 * c7
    L7 = L6 * c8

    j = jaw_ok[:, None]
    lad1 = (0.18 * L1 * j).max(dim=-1).values
    lad2 = (0.24 * L2 * j).max(dim=-1).values
    lad3 = (0.32 * L3 * j).max(dim=-1).values
    lad4 = (0.48 * L4 * j).max(dim=-1).values
    lad5 = (0.62 * L5 * j).max(dim=-1).values
    lad6 = (0.80 * L6 * j).max(dim=-1).values
    lad7 = (1.10 * L7 * j).max(dim=-1).values
    ready = (L6 * j).max(dim=-1).values
    prog = ((L1 + L2 + L3 + L4 + L5 + L6 + L7) * j).max(dim=-1).values / 7.0
    d_best = d_pair.min(dim=-1).values
    reach = (0.60 * _near(d_best, 2.0) + 0.40 * _near(d_best, 8.0)) * pad

    thumb_low = torch.clamp((mouth - t_ax[:, 0]) / 0.015, 0.0, 1.0)
    opp_best = c6.max(dim=-1).values
    f_t = finger_force[:, 0]
    f_o = (finger_force[:, 1:] * c6 * adm).max(dim=-1).values
    f_min = torch.minimum(f_t, f_o)
    q_grasp = pad * opp_best * thumb_low * jaw_ok

    touch = 0.5 * ((1.0 - torch.exp(-f_t / 0.12)) + (1.0 - torch.exp(-f_o / 0.12))) * q_grasp
    press = 0.5 * (_press_ramp(f_t) + _press_ramp(f_o)) * q_grasp
    press_both = _press_ramp(f_min) * q_grasp
    grasped_f = grasped.to(dtype)
    grasp = grasped_f * q_grasp
    hold_soft = torch.clamp((f_min - 0.3) / 0.7, 0.0, 1.0)
    hold = torch.maximum(grasped_f, hold_soft) * q_grasp
    q = 0.3 + 0.7 * q_grasp

    cmd = 0.5 * (torch.clamp(a_hand, -1.0, 1.0) + 1.0)
    cmd_pinch = 0.5 * (cmd[:, 1] + 0.5 * (cmd[:, 2] + cmd[:, 3]))
    squeeze = ready * cmd_pinch

    far = 1.0 - _near(surf.min(dim=-1).values, 20.0)
    gap_pinch = gap[:, 0:2].min(dim=-1).values
    air_pinch = torch.clamp((w_cup - 0.010 - gap_pinch) / 0.025, 0.0, 1.0) * _near(surf.min(dim=-1).values, 40.0)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far
    over_rim = torch.clamp((t_ax[:, 0] - (mouth - 0.005)) / 0.015, 0.0, 1.0)
    rim_hook = over_rim * (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    inside = (torch.clamp((r_in - t_rad) / 0.006, 0.0, 1.0)
              * torch.clamp((mouth + 0.01 - t_ax) / 0.01, 0.0, 1.0)
              * torch.clamp((t_ax - bot) / 0.01, 0.0, 1.0))
    bore = inside.max(dim=-1).values
    dorsal = back * (0.15 * at_cup + 1.2 * torch.tanh((finger_force.sum(dim=-1) + palm_force) / 1.0))

    return {
        "lad1": lad1, "lad2": lad2, "lad3": lad3, "lad4": lad4, "lad5": lad5, "lad6": lad6,
        "lad7": lad7, "ready": ready, "prog": prog, "reach": reach,
        "touch": touch, "press": press, "press_both": press_both, "grasp": grasp, "hold": hold,
        "q": q, "squeeze": squeeze, "air_pinch": air_pinch, "curl_far": curl_far,
        "rim_hook": rim_hook, "bore": bore, "dorsal": dorsal,
    }


def compute_reward(ctx) -> tuple:
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

    rung_s = s["lad1"] + s["lad2"] + s["lad3"] + s["lad4"] + s["lad5"] + s["lad6"] + s["lad7"]
    rung_r = r["lad1"] + r["lad2"] + r["lad3"] + r["lad4"] + r["lad5"] + r["lad6"] + r["lad7"]
    lag_s = torch.clamp(r["prog"] - s["prog"], 0.0, 1.0)
    lag_r = torch.clamp(s["prog"] - r["prog"], 0.0, 1.0)
    lag_pull_src = 3.0 * lag_s * s["reach"]
    lag_pull_rcv = 3.0 * lag_r * r["reach"]

    lad1_src, lad1_rcv = s["lad1"], r["lad1"]
    lad2_src, lad2_rcv = s["lad2"], r["lad2"]
    lad3_src, lad3_rcv = s["lad3"], r["lad3"]
    lad4_src, lad4_rcv = s["lad4"], r["lad4"]
    lad5_src, lad5_rcv = s["lad5"], r["lad5"]
    lad6_src, lad6_rcv = s["lad6"], r["lad6"]
    lad7_src, lad7_rcv = s["lad7"], r["lad7"]

    touch_src, touch_rcv = 0.8 * s["touch"], 0.8 * r["touch"]
    press_src, press_rcv = 2.5 * s["press"], 2.5 * r["press"]
    press_both_src, press_both_rcv = 5.0 * s["press_both"], 5.0 * r["press_both"]
    grasp_src, grasp_rcv = 6.0 * s["grasp"], 6.0 * r["grasp"]
    grasp_both = 5.0 * s["grasp"] * r["grasp"]
    squeeze_src, squeeze_rcv = 1.0 * s["squeeze"], 1.0 * r["squeeze"]
    air_pinch_pen = -0.5 * (s["air_pinch"] * (1.0 - hold_s) + r["air_pinch"] * (1.0 - hold_r))
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])
    bore_pen = -0.6 * (s["bore"] + r["bore"])
    dorsal_pen = -1.5 * (s["dorsal"] + r["dorsal"])

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_src = 9.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.025)
    lift_rcv = 9.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.025)
    lifted_s = hold_s * (h_src > 0.03).to(dtype)
    lifted_r = hold_r * (h_rcv > 0.03).to(dtype)
    lift_both = 4.0 * lifted_s * lifted_r

    tilt = ctx.src_cup_tilt
    rcv_up = _near(ctx.rcv_cup_tilt, 4.0)
    pour_prog = torch.clamp((tilt - 0.35) / 1.2, 0.0, 1.0)
    tgt_dxy = 0.10 * (1.0 - pour_prog)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    dxy_err = torch.abs(mouth_dxy - tgt_dxy)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = _band_err(mouth_dz, 0.04, 0.12)
    carry_q = (0.5 * torch.clamp(1.0 - dxy_err / 0.30, 0.0, 1.0) + 0.5 * _near(dxy_err, 30.0)) * _near(dz_err, 15.0)
    align = 8.0 * lifted_s * q_s * rcv_up * carry_q

    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)
    tilt_r = 7.0 * lifted_s * q_s * rcv_up * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)
    pour_delta = 200.0 * ctx.d_in_target
    spill_delta = -100.0 * ctx.d_spill

    success = 40.0 * ctx.success.to(dtype)
    success_bi = 15.0 * ctx.success.to(dtype) * hold_s * hold_r

    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - lifted_s * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
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
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    hand_foreign_pen = -2.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

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
        "lad7_pinch_src": lad7_src, "lad7_pinch_rcv": lad7_rcv,
        "lag_pull_src": lag_pull_src, "lag_pull_rcv": lag_pull_rcv,
        "touch_src": touch_src, "touch_rcv": touch_rcv,
        "press_src": press_src, "press_rcv": press_rcv,
        "press_both_src": press_both_src, "press_both_rcv": press_both_rcv,
        "grasp_src": grasp_src, "grasp_rcv": grasp_rcv, "grasp_both": grasp_both,
        "squeeze_src": squeeze_src, "squeeze_rcv": squeeze_rcv,
        "air_pinch_pen": air_pinch_pen, "curl_far_pen": curl_far_pen,
        "rim_hook_pen": rim_hook_pen, "bore_pen": bore_pen, "dorsal_pen": dorsal_pen,
        "lift_src": lift_src, "lift_rcv": lift_rcv, "lift_both": lift_both,
        "align": align, "tilt": tilt_r,
        "pour_delta": pour_delta, "spill_delta": spill_delta,
        "success": success, "success_bi": success_bi,
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
```
