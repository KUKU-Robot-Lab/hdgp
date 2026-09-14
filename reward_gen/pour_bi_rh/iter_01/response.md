## 1. What the task means and which stages the robot must go through

The task is a bimanual pour. Each hand has to hold its own cup with a fingertip grasp. Both cups go up, the source rim goes above the receiver rim, and the source cup tips past about 110° so the beads fall through the air into the upright receiver cup.

Stages, per arm and then jointly:

| Stage | Source (right) | Receiver (left) | Signal that it is done |
|---|---|---|---|
| 0 reach | palm next to the source cup | palm next to the receiver cup | palm–cup distance a few cm |
| 1 fingertip grasp | thumb tip on the cup **wall**, below the rim, with index/middle tips on the **opposite** side of the wall | same | `*_grasped` **and** thumb below the rim **and** thumb opposed to the fingers |
| 2 lift | cup origin > 3 cm above spawn | same, and the cup stays upright | height above spawn |
| 3 align | source rim 4–12 cm above the receiver rim, horizontally centred on it | hold the cup still and upright | rim xy distance, rim dz window |
| 4 tilt & pour | tilt toward about 2 rad while aligned | stay upright | `d_in_target > 0`, `d_spill ≈ 0` |
| 5 success | hold the pour state | hold | `ctx.success` |

The whole time: cups must not touch each other or be nested, hands touch only their own cup, no drops, slow motion.

### Diagnosis of the previous run

1. **The source "grasp" was a false positive, and the reward paid for it.** `grasp_src` rose to 0.37, but the thumb hooked over the rim. `src_grasped` only needs a thumb link and a finger to both press the cup, anywhere. The reward paid 1.0 per step for this hook, and the hook cannot lift the cup, so `src_cup_lift` stayed at 2 mm. The grasp bonus has to depend on the grasp's **geometry**: thumb below the rim and opposed to the fingers.
2. **Nothing guided the hand's height or where the fingertips landed.** `tips_src` pulled all tips toward the cup **origin**. That is equally satisfied with the thumb on top of the rim. Tips have to be shaped toward the cup **wall**, inside a height band below the rim, and the thumb has to be opposed to the fingers.
3. **Lift gave almost no gradient.** It was a linear ramp to 8 cm with weight 1.5, so a 2 mm lift was worth 0.04, while the constant grasp and approach terms paid about 1.2. New lift term: `tanh(h / 3 cm)` with weight 3.0. One cm is now worth about 1.0.
4. **Receiver avoidance was learned from penalties.** In epoch 0, `rcv_upright_pen` was −0.12 and `drop_pen` −0.06, which is the receiver cup being knocked. Only the receiver had an always-on upright penalty, and it was as large as the whole receiver approach term. The policy learned to hover about 19 cm away. `tips_rcv` also had a hard 15 cm palm gate, so it gave no gradient at all from where the receiver palm sat. Fixes:
   - The strict upright penalty now applies only while the receiver cup is held.
   - A knocked, unheld cup costs a small, symmetric 0.3.
   - The tip gate is soft.
5. **Nothing downstream could ever fire.** Align, tilt and success needed both cups held and lifted, and the receiver never got there. Pour was also badly under-scaled: `30 * Δ` sums to at most 30 per episode, against about 1000 of per-step shaping. Pour is now `200 * Δ`, and success pays 10 per step, more than align + tilt.

## 2. Improved reward function

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


def _hand_terms(ctx, palm_pos, tips, cup_pos, cup_up, grasped):
    """Per-arm reach / fingertip-grasp geometry. Everything (N,)."""
    zero = torch.zeros_like(cup_pos[:, 0])
    r_wall = ctx.cup_radius + 0.004          # outer wall ~ inner radius + wall thickness
    lo = ctx.cup_bottom_z + 0.03             # keep tips >= 3 cm above the cup bottom (table clearance)
    hi_f = ctx.cup_mouth_z - 0.02            # finger tips >= 2 cm below the rim
    hi_t = ctx.cup_mouth_z - 0.025           # thumb tip >= 2.5 cm below the rim (anti rim-hook)

    d_palm = torch.norm(palm_pos - cup_pos, dim=-1)
    # two-scale approach: k=4 keeps gradient at 20-30 cm, k=15 sharpens the last few cm
    approach = 0.5 * _near(d_palm, 4.0) + 0.5 * _near(d_palm, 15.0)

    axial, radial, rdir = _cyl_coords(tips, cup_pos, cup_up)
    rad_err = torch.clamp(torch.abs(radial - r_wall) - 0.01, min=0.0)  # 1 cm tolerance for tip-pad offset
    thumb_err = rad_err[:, 0] + _band_err(axial[:, 0], lo, hi_t)
    finger_err = (rad_err[:, 1:3] + _band_err(axial[:, 1:3], lo, hi_f)).mean(dim=-1)  # index + middle

    # opposition: thumb radial direction vs mean index/middle radial direction (cos -1 = opposite sides)
    fdir = rdir[:, 1:3, :].mean(dim=1)
    fdir = fdir / (torch.norm(fdir, dim=-1, keepdim=True) + 1e-6)
    cos_opp = (rdir[:, 0, :] * fdir).sum(dim=-1)
    opp = torch.clamp(-cos_opp, 0.0, 1.0)

    # soft palm gate: 1 within 8 cm, 0.37 at 18 cm -> the receiver (at ~19 cm) still feels it
    near_gate = _near(torch.clamp(d_palm - 0.08, min=0.0), 10.0)
    tip_place = near_gate * (0.5 * _near(thumb_err, 25.0) + 0.5 * _near(finger_err, 25.0)) * (0.5 + 0.5 * opp)

    thumb_below_rim = axial[:, 0] < (ctx.cup_mouth_z - 0.015)
    good_grasp = (grasped & thumb_below_rim & (cos_opp < -0.3)).to(zero.dtype)
    held = (grasped & thumb_below_rim).to(zero.dtype)                 # used for lift (opposition may flicker)

    # rim hook: thumb tip at/above rim height AND over the cup footprint, while the palm is near
    over_rim = torch.clamp((axial[:, 0] - (ctx.cup_mouth_z - 0.02)) / 0.02, 0.0, 1.0)
    over_footprint = (radial[:, 0] < r_wall + 0.025).to(zero.dtype)
    rim_hook = near_gate * over_rim * over_footprint

    return approach, tip_place, good_grasp, held, rim_hook


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)

    # ------------------------------------------------------------------ stages 0-1: reach + fingertip grasp
    a_s, tp_s, gg_s, held_s, hook_s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_tips_pos,
                                                  ctx.src_cup_pos, ctx.src_cup_up, ctx.src_grasped)
    a_r, tp_r, gg_r, held_r, hook_r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_tips_pos,
                                                  ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_grasped)
    approach_src = 1.0 * a_s                  # weight 1.0 per arm (<= 1.0/step)
    approach_rcv = 1.0 * a_r
    tips_src = 1.0 * tp_s                     # weight 1.0: fingertips onto the wall below the rim, opposed
    tips_rcv = 1.0 * tp_r
    grasp_src = 1.0 * gg_s                    # weight 1.0: only a geometrically correct grasp is paid
    grasp_rcv = 1.0 * gg_r
    rim_hook_pen = -1.0 * (hook_s + hook_r)   # weight 1.0 = the grasp bonus it used to steal

    # ------------------------------------------------------------------ stage 2: lift (independent per arm)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # tanh(h/3cm): 1 cm -> 0.32, 3 cm -> 0.76; weight 3.0 so lifting 1 cm beats any constant pre-grasp term
    lift_src = 3.0 * held_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.03)
    lift_rcv = 3.0 * held_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.03)
    lifted_src = held_s * (h_src > 0.03).to(zero.dtype)
    lifted_rcv = held_r * (h_rcv > 0.03).to(zero.dtype)
    lifted = lifted_src * lifted_rcv

    # ------------------------------------------------------------------ stage 3: align rims (both lifted)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = torch.clamp(0.04 - mouth_dz, min=0.0) + torch.clamp(mouth_dz - 0.12, min=0.0)  # 4..12 cm window
    align_q = (0.5 * _near(mouth_dxy, 5.0) + 0.5 * _near(mouth_dxy, 20.0)) * _near(dz_err, 15.0)
    align = 4.0 * lifted * align_q            # weight 4.0 > lift 3.0: carrying toward the receiver must pay
    aligned = lifted * (mouth_dxy < 0.05).to(zero.dtype) * (dz_err < 0.015).to(zero.dtype)

    # ------------------------------------------------------------------ stage 4: tilt and pour
    tilt = 3.0 * aligned * torch.clamp(ctx.src_cup_tilt / 2.0, 0.0, 1.0)   # doorway to pouring only
    # 20 beads: +10 per transferred bead (full transfer = 200 ~ 50 steps of align) -> pouring dominates
    pour_delta = 200.0 * ctx.d_in_target
    spill_delta = -100.0 * ctx.d_spill        # -5 per spilled bead: costly but never worth refusing to tilt

    # ------------------------------------------------------------------ stage 5: success
    success = 10.0 * ctx.success.to(zero.dtype)   # 10/step > align + tilt (7): holding the goal is best

    # ------------------------------------------------------------------ constraints
    # tilting a bead-filled source cup before alignment spills; < 0.8 rad is harmless while carrying
    has_beads = (ctx.bead_in_source_frac > 0.05).to(zero.dtype)
    pre_tilt_pen = -1.0 * (1.0 - aligned) * has_beads * torch.tanh(2.0 * torch.clamp(ctx.src_cup_tilt - 0.8, min=0.0))
    # receiver upright: strict ONLY while held (the always-on version taught the left arm to stay away)
    rcv_upright_pen = -1.0 * ctx.rcv_grasped.to(zero.dtype) * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.1, min=0.0))
    # knocked-over unheld cup, symmetric for both arms, small (0.3) so it does not deter approaching
    src_free = 1.0 - ctx.src_grasped.to(zero.dtype)
    rcv_free = 1.0 - ctx.rcv_grasped.to(zero.dtype)
    knock_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.2, min=0.0))
                        + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.2, min=0.0)))
    # drop: an unheld cup above the table falling down
    drop_pen = -1.0 * (src_free * (h_src > 0.015).to(zero.dtype) * torch.tanh(torch.clamp(-ctx.src_cup_lin_vel[:, 2], min=0.0) / 0.5)
                       + rcv_free * (h_rcv > 0.015).to(zero.dtype) * torch.tanh(torch.clamp(-ctx.rcv_cup_lin_vel[:, 2], min=0.0) / 0.5))
    # fast cup motion (thrown / swung): only speeds above 0.4 m/s are penalised, weight 0.2 per cup
    cup_speed_pen = -0.2 * (torch.tanh(torch.clamp(torch.norm(ctx.src_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4)
                            + torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4))
    nested_pen = -2.0 * ctx.cups_nested.to(zero.dtype)
    # real-robot safety
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    hand_foreign_pen = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ------------------------------------------------------------------ regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "tips_src": tips_src,
        "tips_rcv": tips_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
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

### Expected metric movement

- `grasp_src` should drop at first. The rim hook no longer pays, and `rim_hook_pen` is negative while the policy still uses it.
- After that, `tips_src` and `grasp_src` should climb again with the thumb on the wall.
- `task/src_cup_lift` should rise above 0.02 m.
- `approach_rcv` should rise above 0.4, meaning the receiver palm gets within about 10 cm. `tips_rcv` should become non-zero, then `task/rcv_grasped` should follow.
- `align` becomes non-zero only once both lifts exceed 3 cm. `tilt`, `pour_delta` and `success` come after that.
- `rcv_upright_pen` should read about 0 until the receiver is grasped. `knock_pen` should stay small.
