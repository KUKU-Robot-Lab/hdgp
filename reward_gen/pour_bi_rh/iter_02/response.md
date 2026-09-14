# iter_02 — reward revision for bimanual RH56F1 bead pouring

## 1. What the task means and the stages the robot must go through

The right (source) hand has to pick up a light, slim cup (about 6 cm in diameter) that holds 20 beads. The left (receiver) hand has to pick up an identical empty cup. Both hands use fingertip pinch grasps: the thumb tip on one side of the cup wall, one or more finger tips on the other side, below the rim. Both cups are then lifted and brought together without touching. The source rim goes a few cm above and over the receiver rim, and the source cup tips past about 110 degrees so the beads fall through the air into the receiver. The receiver has to stay upright, neither cup may fall, and spill should be low. The environment decides success.

Stages (each one is only reachable after the one before it):

0. **Reach** — each palm moves close enough to its cup that the fingers can reach the wall. The hand stays open until the tips are there, because a pre-curled fist cannot pinch anything.
1. **Fingertip placement** — the thumb tip and an index/middle tip sit on opposite sides of the cup wall, below the rim and above the table.
2. **Contact / pinch** — the thumb and the opposing finger actually press on the cup (the grasp flag needs about 1 N on each). The cup must not be pushed away or knocked over.
3. **Lift** — each held cup rises a few cm off the table.
4. **Align** — with both cups lifted, the source rim goes over the receiver rim, 4 to 12 cm higher, and the cups do not touch.
5. **Tilt and pour** — the source cup tips past about 1.9 rad and beads move into the receiver (reward the increments).
6. **Success** — hold that state. The receiver stays upright, there is no nesting or collision, and spill stays low.

## 2. Diagnosis of the previous run (iter_01 function)

**Primary bottleneck: the reward never asks for contact before the grasp flag, and the hover pose already earns almost everything below it.**

- `task/src_grasped` and `task/rcv_grasped` were 0 for all 620 epochs. So `grasp_*`, `lift_*`, `align`, `tilt`, `pour_delta` and `success` were all 0, and nothing past stage 1 was ever trained.
- `reward/total` plateaued at 1.08–1.16 from about epoch 280. Almost all of it came from `approach_*` (0.40 + 0.41) and `tips_*` (0.22 + 0.22). None of these terms needs contact:
  - `tips` is pure geometry.
  - Its palm gate was still about 0.6 at the observed 13 cm palm–cup distance.
  - The 1 cm radial tolerance let tips that only hover near the wall score.
  
  The policy found a local optimum: hover beside the cup at mid height and collect about 1.2 per step for free.
- Going from "hover" to "grasped" meant a reward jump with no ramp in between. Finger contact force was 0.03–0.07 N against a 1 N threshold, and nothing paid for rising force. Pushing a finger into a light cup mainly risks `knock_pen`, so the policy never tried.
- The hands curl into a loose fist (closure 0.60–0.69) with no contact. The finger pads face the palm and the tips pass the cup without touching it. Closing without contact was free, and a pre-curled fist cannot form a pinch.
- The foreign-contact rate rose (src 0.010, rcv 0.007) because the hands are held low beside the cups. At weight 1.0 the penalty was only about −0.014 per step, far too cheap next to a 1.2 hover income.
- The tilt term did not fire without alignment, and `pre_tilt_pen` / `knock_pen` stayed small. Stages 3–6 are fine for now; they are simply unreached.

## 3. Changes (what and why)

| Change | Why | Expected metric movement |
|---|---|---|
| **New `contact_src/rcv`**: dense, bounded contact reward `0.25·(c_thumb + c_finger) + c_thumb·c_finger` with `c = tanh(f/0.5 N)`, scaled by fingertip placement quality and by a thumb-below-rim gate (weight 1.0, max 1.5) | Creates the missing ramp from hover (0 N) to pinch (≥ 1 N). The product term pays most for two opposing contacts, which is also the posture that does not push the cup away. Scaling by tip placement stops knuckles or the back of a fist from farming it. | `task/*_grasped` > 0, finger force rises |
| **Grasp bonus 1.0 → 2.0**, opposition now a soft multiplier (0.5+0.5·opp) instead of a hard `cos < −0.3` gate; **+1.0 `grasp_both`** | A grasp has to be clearly worth more than hovering (hover is now about 0.4 per arm). The hard opposition gate could zero out a valid pinch at an odd angle. | grasp rate ↑ |
| **`reach` replaces `approach`**: band `exp(−10·max(d−0.08, 0))`, weight 0.3 (was a 1.0 two-scale term) | Hovering at 13 cm now earns 0.18, not 0.40. Inside 8 cm the term is flat, so the tip terms (not the palm) decide the final pose, and the palm is not driven into the cup. | palm–cup distance ↓ from 0.13 m |
| **`tips` without the palm gate, tighter** (5 mm radial tolerance, two-scale k = 30/8, best of index/middle instead of their mean), weight 1.0 → 0.5 | The palm gate was what let hovering score. Taking the best single finger allows a thumb + index pinch. The lower weight keeps geometry clearly below contact. | tips land on the wall, not near it |
| **New `fist_pen`**: −0.3 · clamp((closure−0.45)/0.35, 0, 1) · (1 − any_contact) per arm | Directly targets the observed pre-curled fist. It switches off as soon as any finger touches, so closing onto the cup is never penalised. | closure before contact ↓ from 0.6–0.7 |
| **New `cup_push_pen`**: −0.3 · tanh(max(xy_disp−2 cm, 0)/3 cm) per unheld cup | Contact is now rewarded, so a pushed-away cup (the failure the prompt warns about) needs a cost. The 2 cm dead band tolerates the nudge that comes with forming a grasp. | cup xy displacement stays < 2–3 cm |
| **Soft hold** `hold = below_rim · max(grasped, c_thumb·c_finger)` gates lift, the receiver-upright penalty and the knock/drop "free cup" logic | The grasp flag flickers around 1 N. Lift reward would switch on and off, and a held cup would briefly count as "free". A cup can only rise off the table while the thumb and a finger both press on it. | `lift_*` > 0 once any pinch exists |
| **`hand_foreign_pen` weight 1.0 → 2.0** | Real-robot safety. The rate went up because the hands sit low near the table. | foreign rate back ≤ 0.001 |
| **`success` 10 → 15** | The per-step income of a held, lifted, aligned, tilted state is now higher (about 20). Holding the goal must stay the best option. | — |
| Unchanged: lift (3/arm), align (4), tilt (3, gated on aligned), `pour_delta` (200·Δ), `spill_delta` (−100·Δ), pre-tilt, nested, collision, palm push, drop, cup speed, regularisation | Never reached, so there is no evidence against them. | — |

Per-step reward ladder, approximate, per arm unless marked:
- **Hover:** reach 0.18 + tips ≈0.2 − fist 0.2 ≈ **0.2**.
- **Tips on the wall, open hand:** 0.3 + 0.5 ≈ **0.8**.
- **Pinch:** 0.3 + 0.5 + contact ≈1.5 + grasp 2 ≈ **4.3**, plus 1 for both arms.
- **Lift:** +3 per arm.
- **Align:** +4.
- **Tilt:** +3.
- **Pour:** +10 per bead, one-off.
- **Success:** +15 per step.

Every step up the ladder is worth more than the one before.

## 4. Improved reward function

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
    axial = (rel * up).sum(dim=-1)                                   # (N,F) height along cup axis
    radial_vec = rel - axial[..., None] * up                          # (N,F,3)
    radial = torch.norm(radial_vec, dim=-1)                           # (N,F)
    radial_dir = radial_vec / (radial[..., None] + 1e-6)
    return axial, radial, radial_dir


def _hand_terms(ctx, palm_pos, tips, finger_force, closure, cup_pos, cup_up, grasped):
    """Per-arm reach / fingertip placement / contact / grasp. Everything (N,)."""
    dtype = cup_pos.dtype
    r_wall = ctx.cup_radius + 0.004          # outer wall ~ inner radius + wall thickness
    lo = ctx.cup_bottom_z + 0.03             # tips >= 3 cm above the cup bottom (table clearance)
    hi_f = ctx.cup_mouth_z - 0.02            # finger tips >= 2 cm below the rim
    hi_t = ctx.cup_mouth_z - 0.025           # thumb tip >= 2.5 cm below the rim (anti rim-hook)

    # ---- reach: flat inside 8 cm so the tips (not the palm) set the final pose; 13 cm hover -> 0.61
    d_palm = torch.norm(palm_pos - cup_pos, dim=-1)
    reach = _near(torch.clamp(d_palm - 0.08, min=0.0), 10.0)

    # ---- fingertip placement on the wall (no palm gate: the gate is what paid the hover last round)
    axial, radial, rdir = _cyl_coords(tips, cup_pos, cup_up)
    rad_err = torch.clamp(torch.abs(radial - r_wall) - 0.005, min=0.0)      # 5 mm tolerance
    thumb_err = rad_err[:, 0] + _band_err(axial[:, 0], lo, hi_t)
    f_err = rad_err[:, 1:3] + _band_err(axial[:, 1:3], lo, hi_f)           # index, middle
    finger_err = f_err.min(dim=-1).values                                   # best one: thumb+index pinch is fine

    fdir = rdir[:, 1:3, :].mean(dim=1)
    fdir = fdir / (torch.norm(fdir, dim=-1, keepdim=True) + 1e-6)
    cos_opp = (rdir[:, 0, :] * fdir).sum(dim=-1)
    opp = torch.clamp(-cos_opp, 0.0, 1.0)                                    # 1 = thumb opposite the fingers

    thumb_q = 0.5 * _near(thumb_err, 30.0) + 0.5 * _near(thumb_err, 8.0)     # k=8 far gradient, k=30 last cm
    finger_q = 0.5 * _near(finger_err, 30.0) + 0.5 * _near(finger_err, 8.0)
    tip_q = 0.5 * (thumb_q + finger_q)
    tip_place = tip_q * (0.5 + 0.5 * opp)

    # ---- contact ramp 0 N -> ~1 N (the grasp flag threshold); product term pays opposing contacts most
    f_thumb = finger_force[:, 0]
    f_other = finger_force[:, 1:].max(dim=-1).values
    c_t = torch.tanh(f_thumb / 0.5)
    c_o = torch.tanh(f_other / 0.5)
    below = (axial[:, 0] < (ctx.cup_mouth_z - 0.015)).to(dtype)             # thumb below rim (anti hook)
    contact = below * (0.25 * (c_t + c_o) + c_t * c_o) * (0.5 + 0.5 * tip_q)  # knuckle/fist pushes earn half

    # ---- grasp flag and soft hold (flag flickers around 1 N; lift must not flicker with it)
    grasped_f = grasped.to(dtype)
    good_grasp = below * grasped_f * (0.5 + 0.5 * opp)
    hold = below * torch.maximum(grasped_f, c_t * c_o)                        # (N,) in [0,1]

    # ---- pre-curled fist without any contact (observed: closure 0.6-0.7, 0.03 N)
    any_contact = torch.tanh((f_thumb + f_other) / 0.3)
    fist = torch.clamp((closure - 0.45) / 0.35, 0.0, 1.0) * (1.0 - any_contact)

    # ---- rim hook: thumb tip at/above rim height over the cup footprint while the palm is near
    near_gate = _near(torch.clamp(d_palm - 0.08, min=0.0), 10.0)
    over_rim = torch.clamp((axial[:, 0] - (ctx.cup_mouth_z - 0.02)) / 0.02, 0.0, 1.0)
    over_footprint = (radial[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = near_gate * over_rim * over_footprint

    return reach, tip_place, contact, good_grasp, hold, fist, rim_hook


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)

    # ------------------------------------------------------------------ stages 0-2: reach, tips, contact, grasp
    re_s, tp_s, ct_s, gg_s, hold_s, fist_s, hook_s = _hand_terms(
        ctx, ctx.src_palm_pos, ctx.src_tips_pos, ctx.src_finger_force, ctx.src_hand_closure,
        ctx.src_cup_pos, ctx.src_cup_up, ctx.src_grasped)
    re_r, tp_r, ct_r, gg_r, hold_r, fist_r, hook_r = _hand_terms(
        ctx, ctx.rcv_palm_pos, ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_hand_closure,
        ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_grasped)

    reach_src = 0.3 * re_s                   # weight 0.3: hovering at 13 cm now earns only 0.18
    reach_rcv = 0.3 * re_r
    tips_src = 0.5 * tp_s                    # weight 0.5: geometry must stay below any real contact
    tips_rcv = 0.5 * tp_r
    contact_src = 1.0 * ct_s                 # weight 1.0 (max 1.5): the ramp from 0 N to the 1 N grasp flag
    contact_rcv = 1.0 * ct_r
    grasp_src = 2.0 * gg_s                   # weight 2.0 > reach+tips+single-finger contact: pinching must pay
    grasp_rcv = 2.0 * gg_r
    grasp_both = 1.0 * gg_s * gg_r           # weight 1.0: both hands holding is the gateway to everything else
    fist_pen = -0.3 * (fist_s + fist_r)      # weight 0.3 < tips 0.5: open the hand while approaching
    rim_hook_pen = -1.0 * (hook_s + hook_r)

    # ------------------------------------------------------------------ stage 3: lift (independent per arm)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # tanh(h/3cm): 1 cm -> 0.32, 3 cm -> 0.76; weight 3.0 so lifting beats holding on the table
    lift_src = 3.0 * hold_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.03)
    lift_rcv = 3.0 * hold_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.03)
    lifted_src = hold_s * (h_src > 0.03).to(zero.dtype)
    lifted_rcv = hold_r * (h_rcv > 0.03).to(zero.dtype)
    lifted = lifted_src * lifted_rcv

    # ------------------------------------------------------------------ stage 4: align rims (both lifted)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = torch.clamp(0.04 - mouth_dz, min=0.0) + torch.clamp(mouth_dz - 0.12, min=0.0)  # 4..12 cm window
    align_q = (0.5 * _near(mouth_dxy, 5.0) + 0.5 * _near(mouth_dxy, 20.0)) * _near(dz_err, 15.0)
    align = 4.0 * lifted * align_q           # weight 4.0 > lift 3.0: carrying toward the receiver must pay
    aligned = lifted * (mouth_dxy < 0.05).to(zero.dtype) * (dz_err < 0.015).to(zero.dtype)

    # ------------------------------------------------------------------ stage 5: tilt and pour
    tilt = 3.0 * aligned * torch.clamp(ctx.src_cup_tilt / 2.0, 0.0, 1.0)   # doorway to pouring only
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads)
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # ------------------------------------------------------------------ stage 6: success
    success = 15.0 * ctx.success.to(zero.dtype)   # 15/step > held+lifted+aligned+tilt income: hold the goal

    # ------------------------------------------------------------------ constraints
    has_beads = (ctx.bead_in_source_frac > 0.05).to(zero.dtype)
    pre_tilt_pen = -1.0 * (1.0 - aligned) * has_beads * torch.tanh(2.0 * torch.clamp(ctx.src_cup_tilt - 0.8, min=0.0))
    # receiver upright only while held (always-on version taught the left arm to stay away)
    rcv_upright_pen = -1.0 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.1, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    knock_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.2, min=0.0))
                        + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.2, min=0.0)))
    # unheld cup pushed across the table: contact is now rewarded, so shoving the cup away must cost
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    cup_push_pen = -0.3 * (src_free * torch.tanh(torch.clamp(disp_src - 0.02, min=0.0) / 0.03)
                           + rcv_free * torch.tanh(torch.clamp(disp_rcv - 0.02, min=0.0) / 0.03))
    drop_pen = -1.0 * (src_free * (h_src > 0.015).to(zero.dtype) * torch.tanh(torch.clamp(-ctx.src_cup_lin_vel[:, 2], min=0.0) / 0.5)
                       + rcv_free * (h_rcv > 0.015).to(zero.dtype) * torch.tanh(torch.clamp(-ctx.rcv_cup_lin_vel[:, 2], min=0.0) / 0.5))
    cup_speed_pen = -0.2 * (torch.tanh(torch.clamp(torch.norm(ctx.src_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4)
                            + torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4))
    nested_pen = -2.0 * ctx.cups_nested.to(zero.dtype)
    # real-robot safety
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    # weight 2.0 (was 1.0): foreign-contact rate rose 30x with hands held low near the table
    hand_foreign_pen = -2.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ------------------------------------------------------------------ regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "reach_src": reach_src,
        "reach_rcv": reach_rcv,
        "tips_src": tips_src,
        "tips_rcv": tips_rcv,
        "contact_src": contact_src,
        "contact_rcv": contact_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "grasp_both": grasp_both,
        "fist_pen": fist_pen,
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

## 5. What to watch in the next run

- `reward/contact_*` should rise off 0 within the first ~100 epochs. If it stays below 0.05, the palm is still too far away, so check `reward/reach_*` against its 0.18 hover value.
- `task/*_grasped` should become non-zero. If `contact_*` rises but grasped stays at 0, the contact freeze is capping force below 1 N. The next lever would be gating on soft hold (already used for lift), not on the flag.
- `reward/fist_pen` should shrink toward 0 as the hands stay open until contact.
- `reward/cup_push_pen` and `knock_pen` should stay small (< 0.05). If they grow while `contact_*` grows, the fingers are shoving the cups, and the single-finger share of `contact` (0.25) should be lowered.
- `task/*_hand_foreign_rate` should return to ≤ 0.001.
- Once grasps exist, `lift_*` should move. If grasped > 0 but lift stays at 0, the next bottleneck is lifting.
