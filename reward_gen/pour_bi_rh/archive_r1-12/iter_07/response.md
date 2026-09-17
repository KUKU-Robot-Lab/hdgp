# iter_07 — reward function

## 1. What the task means, and the stages

The robot must move 20 beads from a cup held by the right hand into a cup held by the left hand,
through the air, without knocking anything over. Both cups are 6.5 cm across the outside and light,
and the hand is small, so the only grasp available is a **fingertip pinch: the thumb tip on one wall,
one or more finger tips on the OPPOSITE wall, with the cup body between them.** Everything else in
the task is downstream of that one geometric fact.

Stages:

0. **Approach** — each palm reaches a pre-grasp shell beside its own cup, pad facing the wall, tips
   leading, hand open wider than the cup.
1. **Pocket** — the cup body enters the opening between the thumb tip and the finger tips. This is the
   stage the last four rounds never reached.
2. **Close and grip** — both sides of the pinch travel onto the wall and press; the thumb and at least
   one opposing finger exceed 1 N simultaneously (the environment's grasp flag).
3. **Lift** — both cups leave the table.
4. **Carry** — the source mouth is brought over the receiver mouth, cups never touching.
5. **Tilt and pour** — the source cup rotates past ~110°, beads fall through the air.
6. **Hold the goal state** — receiver upright, little spill, cups not nested.

## 2. Diagnosis of the last round

The operator's finding is that the hand ended in the order *index finger — THUMB — cup*: the thumb
between the fingers and the cup, both on the **same side**. Closing that hand can never trap the cup.
The previous reward paid for it anyway:

- `pinch_geo` used only the distance from the cup axis to the thumb–finger segment. That distance is
  34.5 mm in the video posture, which the exponential still scores 0.19–0.26 — reported as "one fifth
  achieved" while describing an ungraspable hand.
- The opposition factor `opp` existed but only as `0.5 + 0.5·opp`, i.e. a **half-price posture is never
  worthless**, and it was never logged, so no metric could distinguish the two cases.
- `close` had a floor of `0.35 × 0.5 = 0.175` of its weight 3.0, paid for tips merely near the wall with
  no opposition and no pinch at all.
- The force terms were gated on `gate_q = 0.5 + 0.5·quality`, another floor: pressing one wall with both
  the thumb and the fingers and shoving the light cup away still earned.

So the source hand harvested 3.4 of its 4.4 total from posture terms while its contact force stayed at
0.21 N, one fifth of the flag threshold, and `grip_src` fell back to zero. FK on the robot's own URDF
shows the hand *can* close past a 57 mm cup, so this is a reward-shape failure, not a kinematic one.

## 3. What changed

**Every floor under the posture and contact terms is removed, and opposition becomes the gate.**

- **`oppose_*` is now a logged component.** `opp = −cos(angle between the thumb's radial direction and each
  finger's radial direction)`: +1 = far wall, −1 = same side as the thumb. The video posture scores −1.
- **`opp_gate = clamp((opp − 0.3)/0.5, 0, 1)`** — a hard gate with no floor. 90° apart → 0, 120° → 1.
- **`pinch_geo` is replaced by two terms.** `pocket` is the dense, long-range signal: the midpoint of the
  thumb tip and the index/middle tips must sit **on the cup axis**, the opening must be wider than the cup
  (6.5 cm) and narrower than a spread hand (13.5 cm), both sides at wall height, thumb below the rim, times
  a soft opposition factor. Two tips on the same side put the midpoint off the axis *and* score zero
  opposition, so the failure posture cannot earn it. `pose` is the strict per-finger straddle test
  (axis between the two tips × `opp_gate` × both tips admissible), and it is the gate for everything below.
- **`close = pose × wall_closeness`, no additive floor.** Tips near the wall in a hand that cannot trap
  the cup are worth exactly 0.
- **Forces count only from an opposing finger**: `f_o = max_j(finger_force_j × opp_gate_j)`. Pressing the
  same wall with thumb and fingers yields `f_o = 0`, so `grip` and `touch` are 0.
- **`hold` is gated on opposition too.** The environment's grasp flag can fire with thumb and finger on the
  same wall; that must not unlock the lift reward.
- **`pose_both = 2.0 × min(pose_src, pose_rcv)`** — new. The receiver hand went *backwards* last round
  (16.1 cm away, zero contact, `pinch_geo` 0) while the source hand improved; a sum lets the policy
  specialise on one hand. A minimum cannot be raised by the good hand alone.
- **Approach income cut again** (0.5 → 0.3 per hand) and the grasp plateau trimmed (`grip` 3.5→3.0,
  `grasp` 4.0, `grasp_both` 3.0) while `lift` 5→8, `align` 6→8, `tilt` 5→7, `success` 25→30, so that
  standing in a finished double grasp is worth less than what lifting adds.

## 4. Measured on synthetic postures

A 57 mm cup (`cup_radius` 0.0285, rim +0.055, bottom −0.055), five fingertips, both hands set to the same
posture. Raw per-hand term values in [0, 1], and the summed **weighted** posture/contact income
(`oppose + pocket + pose + pose_both + close + squeeze + touch + grip + grasp + grasp_both`, both hands):

| posture | oppose | pocket | pose | close | squeeze | touch | grip | grasp | weighted posture income | total reward |
|---|---|---|---|---|---|---|---|---|---|---|
| true opposition, on the wall, 1.5 N both sides, flag on | 0.983 | 1.000 | 0.887 | 0.884 | 0.530 | 0.990 | 0.999 | 1.000 | **29.45** | 30.23 |
| **thumb between the fingers and the cup (the video posture)** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | 0.57 |
| **same posture with 1.5 N on both sides and the grasp flag ON** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | 0.57 |
| **hand 30 cm away, fingers spread** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | 0.02 |
| closed on air beside the cup (2 cm tip gap) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | −0.21 |
| thumb and index 90° apart on the wall | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.52 |
| thumb and index side by side on the same wall | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | −0.23 |
| thumb hooked over the rim, fingers on the far wall | 0.983 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.59 | 0.38 |
| correct opposition but 2 cm off the wall, no contact | 0.993 | 1.000 | 0.887 | 0.482 | 0.096 | 0.000 | 0.000 | 0.000 | 8.18 | 8.97 |

Every posture that cannot physically trap the cup scores **exactly 0.000 on all eight posture/contact
terms** — including the one with full 1.5 N contact forces and the environment's grasp flag raised, which
the previous reward would have paid in full. The only non-zero score among the impossible postures is the
rim-hooked thumb's `oppose` (0.983 — its fingers genuinely are on the far wall), and it is cancelled by
`rim_hook_pen`, leaving 0.59 weighted against A's 29.45.

The correct-but-not-touching posture keeps a live gradient (8.18 → 29.45 as the tips reach the wall and
press), so removing the floors did not remove the path in.

Downstream still dominates: with the same postures lifted 10 cm, pouring and succeeding, the true-opposition
env reaches 86.96 total (lift 7.99 per hand, pour 10 per bead-batch, success 30) versus 29.45 for standing
in the finished grasp on the table — while the same-side posture with the flag on gets `lift_src = 0.0`.

Checks: 43 components, every one shape (N,), no NaN/Inf, components sum exactly to the returned reward,
batch of 512 finite.

## 5. Expected metric movement

- `task/src_thumb_oppose_near` / `task/rcv_thumb_oppose_near` (new logging): should rise from ~0 toward 0.8+.
  If `pose_*` rises while `*_thumb_oppose_near` stays low, the straddle test has a hole and I was wrong.
- `task/*_cup_in_pocket_near`: should become non-zero before any contact appears.
- `reward/pose_*` and `reward/close_*` can no longer be earned by the iter_05/iter_06 posture, so they start
  near zero and must be *earned*; a repeat of last round's climb with `contact/*_max` flat at 0.2 N is now
  impossible by construction.
- `contact/src_max`, `contact/rcv_max` past 1 N, then `task/*_grasped` off the floor, then `*_cup_lift`.
- `reward/pose_both` is the receiver-engagement canary: it stays 0 while only one hand works.

## 6. Reward function

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
    """Cup-cylinder coords of (N,K,3) points: axial height (N,K), radial vector (N,K,3), radial distance (N,K)."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)
    rvec = rel - axial[..., None] * up
    radial = torch.norm(rvec, dim=-1)
    return axial, rvec, radial


def _axis_to_segments(rv_a: torch.Tensor, rv_b: torch.Tensor) -> torch.Tensor:
    """Distance (N,K) from the cup AXIS to each segment joining rv_a (N,1,3) to rv_b (N,K,3),
    measured in the plane normal to the axis. ~0 only when the axis lies BETWEEN the two tips."""
    seg = rv_b - rv_a
    u = torch.clamp(-(rv_a * seg).sum(dim=-1) / ((seg * seg).sum(dim=-1) + 1e-8), 0.0, 1.0)
    closest = rv_a + u[..., None] * seg
    return torch.norm(closest, dim=-1)


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, closure, cup_pos, cup_up, grasped, a_hand):
    """Per-arm approach / opposition / pocket / contact signals, all (N,) in [0, 1]."""
    dtype = cup_pos.dtype
    mouth = ctx.cup_mouth_z
    r_in = ctx.cup_radius
    r_wall = r_in + 0.004                    # outer wall ~ inner radius + wall thickness
    d_out = 2.0 * r_wall                     # outer diameter ~ 6.5 cm for a 5.7 cm bore
    tip_lo = ctx.cup_bottom_z + 0.03         # tips >= 3 cm above the cup bottom (table clearance)
    tip_hi = mouth - 0.015                   # tips >= 1.5 cm below the rim

    # ---- palm pre-grasp region (unchanged; this stage is solved and stays cheap)
    p_ax, _, p_rad = _cyl(palm_pos[:, None, :], cup_pos, cup_up)
    rad_err = _band_err(p_rad[:, 0], r_wall + 0.01, 0.10)
    ht_err = _band_err(p_ax[:, 0], ctx.cup_bottom_z + 0.03, mouth + 0.01)
    d_pre = torch.sqrt(rad_err ** 2 + ht_err ** 2 + 1e-8)
    reach_lin = torch.clamp(1.0 - d_pre / 0.25, 0.0, 1.0)
    reach_fine = _near(d_pre, 25.0)
    near_pre = _near(d_pre, 10.0)

    # ---- orientation: pad normal faces the cup, tips nearer the axis than the palm
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup_xy = cup_pos[:, :2] - palm_pos[:, :2]
    d_palm_xy = torch.norm(to_cup_xy, dim=-1)
    u_xy = to_cup_xy / (d_palm_xy[:, None] + 1e-6)
    face = torch.abs((normal[:, :2] * u_xy).sum(dim=-1))
    tip_mid_all = tips[:, 0:3, :].mean(dim=1)
    d_tip_xy = torch.norm(tip_mid_all[:, :2] - cup_pos[:, :2], dim=-1)
    ahead = torch.clamp((d_palm_xy - d_tip_xy) / 0.03, 0.0, 1.0)
    orient = near_pre * (0.5 * face + 0.5 * ahead)

    # ---- fingertips in cup-cylinder coordinates
    t_ax, t_rv, t_rad = _cyl(tips, cup_pos, cup_up)
    t_dir = t_rv / (t_rad[..., None] + 1e-6)

    # ---- OPPOSITION (the quantity every previous round was blind to).
    #      +1 = this finger sits on the far wall from the thumb, -1 = same side as the thumb.
    u_th = t_dir[:, 0, :]
    opp_all = (t_dir[:, 1:, :] * u_th[:, None, :]).sum(dim=-1) * (-1.0)      # (N,F-1) in [-1, 1]
    opp_best = torch.clamp(opp_all.max(dim=-1).values, 0.0, 1.0)             # logged as oppose_*
    # HARD gate, no floor: 90 deg apart -> 0, 120 deg -> 1.  A thumb beside or inside the fingers scores 0.
    opp_gate_all = torch.clamp((opp_all - 0.3) / 0.5, 0.0, 1.0)
    opp_gate = opp_gate_all.max(dim=-1).values

    # ---- tip admissibility: at wall height and outside the bore (not dipped into the cup, not far outside)
    ht_bad = _band_err(t_ax, tip_lo, tip_hi)                                  # (N,F)
    h_ok = _near(ht_bad, 40.0)
    r_ok = (torch.clamp((t_rad - r_in) / 0.005, 0.0, 1.0)
            * torch.clamp((r_wall + 0.05 - t_rad) / 0.02, 0.0, 1.0))
    tip_ok = h_ok * r_ok                                                      # (N,F)

    # ---- thumb height w.r.t. the rim: below the rim = on the wall, above = hooked on the mouth
    thumb_ax = t_ax[:, 0]
    thumb_low = torch.clamp((mouth - thumb_ax) / 0.015, 0.0, 1.0)

    # ---- POCKET: is the cup body actually between the thumb tip and the finger tips?
    #      midpoint of the thumb tip and the index/middle mean must sit ON the cup axis, the opening must be
    #      wider than the cup and narrower than a spread hand, and both sides must be at wall height.
    tip_f_mid = tips[:, 1:3, :].mean(dim=1)
    mid = 0.5 * (tips[:, 0, :] + tip_f_mid)
    _, rv_mid, d_pocket = _cyl(mid[:, None, :], cup_pos, cup_up)
    d_pocket = d_pocket[:, 0]
    gap_mid = torch.norm(tips[:, 0, :] - tip_f_mid, dim=-1)
    # full credit for an opening of 6.5..11 cm (cup is 6.5 cm outside, the open hand reaches ~10 cm)
    gap_ok = (torch.clamp((gap_mid - (d_out - 0.015)) / 0.015, 0.0, 1.0)
              * torch.clamp((0.135 - gap_mid) / 0.025, 0.0, 1.0))
    # soft opposition factor: exactly 0 when the thumb is on the same side as the fingers
    u_fm = t_dir[:, 1:3, :].mean(dim=1)
    u_fm = u_fm / (torch.norm(u_fm, dim=-1, keepdim=True) + 1e-6)
    opp_mid = -(u_th * u_fm).sum(dim=-1)
    opp_soft = torch.clamp((opp_mid + 0.1) / 0.6, 0.0, 1.0)
    h_pocket = h_ok[:, 0] * h_ok[:, 1:3].max(dim=-1).values
    pocket = ((0.35 * _near(d_pocket, 15.0) + 0.65 * _near(d_pocket, 45.0))
              * gap_ok * h_pocket * opp_soft * thumb_low)

    # ---- POSE: the strict, per-finger straddle test. The cup axis lies between the thumb tip and THIS
    #      finger's tip, both tips admissible, and the pair genuinely opposed. No floor anywhere.
    seg_d = _axis_to_segments(t_rv[:, 0:1, :], t_rv[:, 1:, :])                # (N,F-1)
    pair_geo = _near(seg_d, 40.0) * opp_gate_all * tip_ok[:, 1:] * tip_ok[:, 0:1]
    pose = pair_geo.max(dim=-1).values

    # ---- distance of each tip to the outer wall (radial gap + height-band error)
    rad_out = torch.clamp(t_rad - r_wall - 0.003, min=0.0)
    surf = torch.sqrt(rad_out ** 2 + ht_bad ** 2 + 1e-8)
    wall_all = 0.35 * _near(surf, 12.0) + 0.65 * _near(surf, 50.0)
    wall_t = wall_all[:, 0]
    wall_f = wall_all[:, 1:].max(dim=-1).values
    wall_pair = 0.5 * (wall_t + wall_f)

    # ---- thumb hooked over the rim instead of pinching the wall
    over_rim = torch.clamp((thumb_ax - (mouth - 0.005)) / 0.015, 0.0, 1.0)
    over_footprint = (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = over_rim * over_footprint

    # ---- close the last centimetres: ONLY inside a pose that can trap the cup
    close = pose * wall_pair

    # ---- closing on air
    gap_min = torch.norm(tips[:, 0:1, :] - tips[:, 1:, :], dim=-1).min(dim=-1).values
    air_pinch = torch.clamp((d_out - 0.02 - gap_min) / 0.02, 0.0, 1.0)

    # ---- squeeze intent from the COMMANDED closure
    cmd = 0.5 * (torch.clamp(a_hand, -1.0, 1.0) + 1.0)
    cmd_close = (cmd[:, 0] + cmd[:, 1] + cmd[:, 2:6].mean(dim=-1)) / 3.0
    squeeze = pose * cmd_close * wall_pair

    # ---- hand must be open while outside the pre-grasp region
    far = 1.0 - _near(torch.clamp(d_pre - 0.03, min=0.0), 20.0)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far

    # ---- forces. Only a finger that OPPOSES the thumb counts as the far side of the pinch.
    f_t = finger_force[:, 0]
    f_o = (finger_force[:, 1:] * opp_gate_all).max(dim=-1).values
    f_min = torch.minimum(f_t, f_o)
    s_t = 1.0 - torch.exp(-f_t / 0.3)
    s_o = 1.0 - torch.exp(-f_o / 0.3)
    q_grasp = opp_gate * thumb_low                                            # hard: no opposition, no pay
    touch = (0.6 * s_t * wall_f + 0.4 * s_o * wall_t) * q_grasp
    grip = (0.5 * (1.0 - torch.exp(-f_min / 0.25)) + 0.5 * torch.clamp(f_min / 1.0, 0.0, 1.0)) * q_grasp
    grasped_f = grasped.to(dtype)
    grasp = grasped_f * q_grasp
    hold_soft = torch.clamp((f_min - 0.3) / 0.7, 0.0, 1.0)
    # opposition gates the HOLD too: the env grasp flag can fire with the thumb and a finger both
    # pressing the SAME wall, which shoves the cup instead of trapping it — that must not pay for lifting.
    hold = torch.maximum(grasped_f, hold_soft) * thumb_low * opp_gate
    q = 0.3 + 0.7 * q_grasp

    return {
        "reach_lin": reach_lin, "reach_fine": reach_fine, "orient": orient,
        "oppose": opp_best * near_pre, "pocket": pocket, "pose": pose,
        "close": close, "squeeze": squeeze, "air_pinch": air_pinch, "curl_far": curl_far,
        "touch": touch, "grip": grip, "grasp": grasp, "hold": hold, "q": q, "rim_hook": rim_hook,
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

    # ---------------------------------------------------- stage 0: approach (solved; kept deliberately cheap)
    approach_src = 0.2 * s["reach_lin"] + 0.1 * s["reach_fine"]      # max 0.3 per hand
    approach_rcv = 0.2 * r["reach_lin"] + 0.1 * r["reach_fine"]
    orient_src = 0.10 * s["orient"]
    orient_rcv = 0.10 * r["orient"]

    # ---------------------------------------------------- stage 1: get the cup BETWEEN thumb and fingers
    oppose_src = 0.3 * s["oppose"]        # logged diagnostic + gentle pull; 0 for a same-side thumb
    oppose_rcv = 0.3 * r["oppose"]
    pocket_src = 0.8 * s["pocket"]        # dense "cup inside the opening" signal
    pocket_rcv = 0.8 * r["pocket"]
    pose_src = 1.2 * s["pose"]            # strict straddle; the gate for everything below
    pose_rcv = 1.2 * r["pose"]
    # min(), not sum(): last round the source hand improved while the receiver went backwards, which a
    # per-hand sum rewards. A minimum cannot be raised by the good hand alone.
    pose_both = 2.0 * torch.minimum(s["pose"], r["pose"])

    # ---------------------------------------------------- stage 2: onto the wall and squeeze
    close_src = 2.0 * s["close"]          # weight 2.0 but NO floor: 0 unless the pose can trap the cup
    close_rcv = 2.0 * r["close"]
    squeeze_src = 0.8 * s["squeeze"]      # command the fingers shut, only inside a valid pose in contact range
    squeeze_rcv = 0.8 * r["squeeze"]
    touch_src = 1.0 * s["touch"]          # one side pressing while the other is in place — a step, not a rest
    touch_rcv = 1.0 * r["touch"]
    grip_src = 3.0 * s["grip"]            # weight 3.0: both sides above 1 N is what the env calls a grasp
    grip_rcv = 3.0 * r["grip"]
    grasp_src = 4.0 * s["grasp"]          # weight 4.0 > grip 3.0: the flag itself is the target
    grasp_rcv = 4.0 * r["grasp"]
    grasp_both = 3.0 * s["grasp"] * r["grasp"]
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])
    air_pinch_pen = -0.5 * (s["air_pinch"] * (1.0 - hold_s) + r["air_pinch"] * (1.0 - hold_r))
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])

    # ---------------------------------------------------- stage 3: lift
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # weight 8.0 per hand > the whole grasp plateau's marginal value: a finished grasp on the table is not the end
    lift_src = 8.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.025)
    lift_rcv = 8.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.025)
    lifted = hold_s * (h_src > 0.03).to(dtype) * hold_r * (h_rcv > 0.03).to(dtype)

    # ---------------------------------------------------- stage 4: carry together (tilt-aware target)
    tilt = ctx.src_cup_tilt
    # upright: mouths 10 cm apart (bodies side by side, no nesting); the gap closes as tilt -> ~90 deg,
    # because in a real pour the source ORIGIN stays aside and the tilted mouth swings over the receiver.
    pour_prog = torch.clamp((tilt - 0.35) / 1.2, 0.0, 1.0)
    tgt_dxy = 0.10 * (1.0 - pour_prog)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    dxy_err = torch.abs(mouth_dxy - tgt_dxy)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = _band_err(mouth_dz, 0.04, 0.12)                     # source mouth 4..12 cm higher
    carry_q = (0.5 * torch.clamp(1.0 - dxy_err / 0.30, 0.0, 1.0) + 0.5 * _near(dxy_err, 30.0)) * _near(dz_err, 15.0)
    align = 8.0 * lifted * q_s * carry_q     # weight 8.0 = lift: carrying toward the receiver must keep paying

    # ---------------------------------------------------- stage 5: tilt and pour
    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)            # 1 cm error -> 0.67
    tilt_r = 7.0 * lifted * q_s * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)   # tilt pays only on the schedule
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads); increments, never the level
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # ---------------------------------------------------- stage 6: success
    success = 30.0 * ctx.success.to(dtype)   # 30/step on top of all held income: keep the goal state

    # ---------------------------------------------------- constraints (guards, not shaping)
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - lifted * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
    rcv_upright_pen = -0.8 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.15, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    # knocking a cup over and shoving it across the table are the same failure (pressing one side only)
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

    # ---------------------------------------------------- regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src, "approach_rcv": approach_rcv,
        "orient_src": orient_src, "orient_rcv": orient_rcv,
        "oppose_src": oppose_src, "oppose_rcv": oppose_rcv,
        "pocket_src": pocket_src, "pocket_rcv": pocket_rcv,
        "pose_src": pose_src, "pose_rcv": pose_rcv, "pose_both": pose_both,
        "close_src": close_src, "close_rcv": close_rcv,
        "squeeze_src": squeeze_src, "squeeze_rcv": squeeze_rcv,
        "touch_src": touch_src, "touch_rcv": touch_rcv,
        "grip_src": grip_src, "grip_rcv": grip_rcv,
        "grasp_src": grasp_src, "grasp_rcv": grasp_rcv, "grasp_both": grasp_both,
        "curl_far_pen": curl_far_pen, "air_pinch_pen": air_pinch_pen, "rim_hook_pen": rim_hook_pen,
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
```
