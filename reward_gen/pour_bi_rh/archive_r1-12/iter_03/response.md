# pour_bi_rh — iter_03 reward response

## 1. What the task means and which stages it needs

The robot has to pour 20 beads from the right-hand (source) cup into the left-hand (receiver) cup, and do it safely enough to run on the real robot. The stages are:

0. **Approach (both arms, in parallel).** Move each palm toward its own cup with the hand **open**. Turn the wrist so the finger pads face the cup body from the side, with the fingertips pointing at the cup. Do not touch the table, the other hand or the other cup.
1. **Fingertip placement.** Put the thumb tip on one side of the slim cup wall and the index and/or middle tip on the opposite side. All tips go between about 3 cm above the cup bottom and just below the rim. Do not hook over the rim, and do not push the cup with the palm.
2. **Close and grasp.** Close the hand once the tips are at the wall. Contact freeze stops each finger when it touches, which gives thumb plus finger contact of about 1 N, so `*_grasped` becomes True.
3. **Lift.** Raise both cups off the table (more than 3 cm) while holding them. The receiver stays upright and the source cup is not tilted yet, so no beads are lost.
4. **Bring together.** Put the source rim 4–12 cm above the receiver rim and horizontally over it. The cups must not touch (no `cup_cup_force`, no nesting).
5. **Tilt and pour.** Tilt the source cup past about 110° while its mouth is over the receiver's mouth, so the beads fall through the air into the receiver.
6. **Hold the goal.** Keep holding until at least half of the beads are in the receiver with little spill (`ctx.success`), with the receiver upright and no cup dropped.

## 2. Diagnosis of iter_02

The policy never got past stage 0/1. `grasped` was 0 on both hands for all 674 epochs, so every term from `grasp_*` onward was exactly 0. The reward crept up only through `reach` and `tips`, then flattened at 0.62. The causes, in order of weight:

| # | Cause | Evidence |
|---|---|---|
| A | **No gradient to close the last few cm.** `reach` is flat inside 8 cm and worth at most 0.3. At 12–16 cm it already pays 0.16–0.20, so the remaining 8 cm is worth about 0.1. | palm–cup distance 0.122 m / 0.159 m; reach_src 0.20, reach_rcv 0.16 and flat |
| B | **Contact and grasp were gated to zero in exactly the pose the policy used.** Contact and grasp are multiplied by a 0/1 "thumb below rim" test, and contact also by 0.5+0.5·tip quality. The hand arrives at rim height with curled fingers, so every accidental touch paid 0. This removed the only bridge from "near the cup" to "holding the cup". | contact 0.008 / 0.011 of a 1.5 maximum; forces 0.06 / 0.07 N; video shows curled fingers hooked over the rim |
| C | **Nothing rewards wrist orientation.** Pads facing away from the cup cost nothing, so the arm settled into a rotated wrist. | operator video, item 1 |
| D | **Nothing makes the hand open far away and close near the cup.** `fist_pen` only fires above closure 0.45 and is cancelled by any contact. Closure 0.36–0.40 at 12–16 cm is free. Closing near the cup had no dense reward before contact. | closure 0.40 / 0.36; fist_pen −0.003 |
| E | Stage gating is the right idea but was too strict. The other track (grasp 0.88, success 0.77) did not gate the grasp on grasp quality. It only scaled later income by 0.3 + 0.7·quality. | operator reference, item 6 |

The mimic blow-ups (item 5) are simulation issues that the environment already terminates. They also explain the absurd `rcv_cup_lift` / `aim_dist` maxima. They are not a reward exploit, so no term targets them.

## 3. Changes (what, why, expected metric movement)

1. **approach**: now `exp(-4·d) + 0.5·exp(-15·max(d-0.06,0))`, weight 1.0, no 8 cm flat zone. At 12 cm it pays about 0.78 and at 6 cm about 1.29, a 0.5 gain per arm for closing in (was about 0.1). Expected: palm–cup distance goes below 0.09 m and `reward/approach_*` rises instead of flattening.
2. **orient (new, 0.4 per arm)**: rewards `|palm_normal · horizontal direction to cup|` (sign-agnostic, so a wrong normal sign cannot invert it) plus "fingertips are ahead of the palm toward the cup axis". Back-of-hand-first scores 0 on the second half. It is shaped from afar (`exp(-5·max(d-0.08,0))`). Expected: wrist no longer rotated; the pads face the cup body.
3. **close_near (new, 0.5 per arm)**: `closure × exp(-15 · tip-to-wall distance)`. It pays closing only when the thumb and best finger tip are actually at the cup wall, so the reference's "reward closing near the cup" cannot pay a curled hand hovering at 12 cm.
4. **curl_far_pen (replaces fist_pen)**: `-0.4 × clamp((closure-0.2)/0.4) × (1 - exp(-20·max(d-0.08,0)))`. The hand must be open while the palm is more than 8 cm from the cup. Contact does not cancel it. Expected: closure during approach drops toward 0–0.2.
5. **contact and grasp ungated**: contact = `0.25·(c_t+c_o) + c_t·c_o` with a 0.3 N ramp (max 1.5). grasp = 2.0 · flag and grasp_both = 1.0 · both flags, with no rim or tip-quality multiplier. `rim_hook_pen` (−1.0) is kept as a separate cost, so hooking stays worse than a side pinch without zeroing the signal. Expected: contact forces rise above 0.3 N, then `task/*_grasped` becomes non-zero.
6. **Grasp quality is a factor, not a gate**: `q = 0.3 + 0.7·below_rim_soft·(0.5+0.5·opposition)` multiplies lift, align and tilt income. A poor grasp still earns 30 % and can improve while lifting.
7. **The soft hold used by lift no longer needs the below-rim indicator**, so lift pays as soon as any real hold exists.
8. Kept unchanged: the lift/align/tilt/pour/success ladder and its weights, the pour/spill increments, all safety penalties (cup–cup 1.0, hand foreign 2.0, palm push 0.5), and the knock/push/drop/nesting penalties. Tip placement now uses a **cylinder-surface distance** (radial and axial combined, 5 mm radial tolerance) instead of separate errors, again without a palm gate.

Income check (per arm): hovering open and oriented at 12 cm earns about 0.78 + 0.3 + 0.1, roughly 1.2. Tips at the wall and closing earns about 1.3 + 0.4 + 0.5 + 0.5, roughly 2.7. A real pinch with contact earns about +1.5 + 2.0, roughly 6 per arm, plus 1.0 when both hold. Holding therefore clearly dominates approaching, lift (3·q) adds on top of holding, and align (4), tilt (3) and success (15) stay dominant later.

Expected movement next run: approach and orient rise quickly; closure during approach drops; contact rises within the first 100–200 epochs, then `grasp_*`. `lift_*` should become non-zero once `grasped` is above about 0.2. Watch `cup_push_pen` and `knock_pen`: if they grow a lot, the stronger approach term is making the hands shove the cups, and the 0.06 m inner offset should move out.

## 4. Reward function

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
```
