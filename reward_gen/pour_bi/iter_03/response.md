# iter_03 — reward redesign

## 1. What the task means and the stages the robot must go through

1. **Approach** — each palm moves to its own cup (right → source cup with beads, left → empty receiver cup).
2. **Grasp** — each hand closes around its cup until thumb + another finger press on it (`*_grasped`).
3. **Lift** — both cups leave the table. The source cup must end up clearly *higher* than the receiver, because it will be tilted and its rim swings downward.
4. **Bring together / pre-pour pose** — the source cup is carried next to (not into) the receiver cup so that, once tilted, its rim will hang over the receiver's mouth. The cups must never touch and the source cup must never be pushed into the receiver (`cups_nested`).
5. **Tilt / pour** — roll the source cup past ~110° with the *pouring lip* (lowest point of the rim) a few cm above the receiver rim, slowly enough that beads fall into the receiver and not on the table.
6. **Hold** — keep the receiver upright, keep both cups held, until ≥ half of the beads are in the receiver with little spill (`success`, judged by the env).

## 2. Analysis of the iter_02 feedback

| stage | evidence | verdict |
|---|---|---|
| approach / grasp | grasp_src 0.97, grasp_rcv 0.96, src/rcv grasped 0.88 / 0.87 | solved, same level as the 0.73-success run |
| lift | lift_src 1.75 / 2.0, lift_rcv 1.7 / 2.0, both_lifted 0.86, src lift 0.13–0.22 m, rcv 0.09–0.18 m | solved, but src and rcv are lifted to *the same* height (both saturate the 0.10 m target) |
| bring together / align | align 1.43 / 3.0, bring_together 0.6 / 1.0, aim_dist stuck at 0.13 m, cups_center_dist 0.18 m from mid-training on | **stalled** — the policy found a fixed point 13 cm short of the gate |
| tilt | tilt ≈ 0.002 (max 0.029), src_tilt 30° all run, tilt_premature → 0 | **never unlocked** — the policy learned to keep the cup below TILT_FREE and stop |
| pour | pour_delta 0, bead/in_target 0, success 0, adr/progress 0 | nothing happens downstream |
| safety | cup_collision 2 %, hand_foreign ~10–20 %, nested 0.05 % | fine, penalties are not what stops progress |

Why the alignment stalls at 13 cm is geometric, not a weight problem. The gate that unlocks `tilt` was
`d_xy(mouth centres) < 1.5·cup_radius` **and** `0 < dz < 8 cm` **and** not nested. With a cup that is still nearly
upright (30°), putting the source *mouth centre* within ~5 cm of the receiver mouth centre while its rim is only
0–8 cm above the receiver rim means the source cup body sits on top of / inside the receiver cup — which is exactly
`cups_nested` (origin distance < 9 cm) or a cup–cup collision (`clear` → 0). So the states that satisfy the tilt
gate are unreachable without nesting, and tilting *before* the gate is (mildly) punished. The policy did the rational
thing: park the source cup side by side with the receiver (centre distance 0.18 m ≈ two cup diameters + clearance),
collect the partial `align`/`bring_together` income (exp(-5·0.13) ≈ 0.5), and never tilt. That is a chicken-and-egg
deadlock created by the reward, and it also explains why `tilt_premature` decays to zero: the policy simply stopped
exploring tilt.

The iteration-1 policy escaped this because its `align_z` was flat up to 12 cm: it lifted the source 30 cm and
poured from ~10 cm above, so the mouth centre could be over the receiver without nesting. Iteration 2 tried to force a
3 cm pour height and thereby closed the only non-nesting route.

What is physically true: during a pour the source cup **origin** is ~10–15 cm away from the receiver origin (higher
and to the side); only the **lowest point of its rim** is over the receiver mouth, and it gets there by *tilting*, not
by translating the cup over the receiver. The alignment metric must therefore be defined on that rim point, so that
tilting and translating are rewarded jointly and there is a monotone path from the current fixed point
(side by side, 30°) to the pour pose (higher, closer, 110°).

Other observations:
- `align_z` peaked at 3 cm with σ = 3 cm was too narrow for delayed/noisy cup poses and fought the nesting gate.
- Both cups saturate the same 0.10 m lift target, so nothing asks the source cup to be higher than the receiver.
- All safety terms are small and stable; keep them unchanged.
- Exploit to guard against once tilt is rewarded softly: tilt with the lip *not* over the receiver, dump all beads on
  the table, and then keep collecting tilt reward with an empty cup. Fix: scale the tilt reward by the fraction of beads
  that are still in play (`bead_in_source_frac + bead_in_target_frac`); spilled beads permanently shrink it.

## 3. Changes

1. **Pouring-lip geometry.** New helper `_rim_low_point`: lowest rim point = mouth centre + cup_radius · unit(−z projected
   onto the rim plane). Upright cup → collapses to the mouth centre; tilted cup → the point that beads actually leave from.
   All pour shaping uses `lip_xy` (horizontal distance lip → receiver mouth centre) and `dz_lip` (lip height above the
   receiver rim).
2. **`aim` (replaces `align`, weight 3.0)** = both lifted · not nested · clear · exp(−6·lip_xy) · aim_z, where
   aim_z = wide "above the rim" ramp (−6 cm → +2 cm) × gentle decay beyond 4 cm (σ 6 cm). No hard xy/z gate anymore.
3. **`bring_together` (weight 1.0)** uses the same lip metric with a wider basin exp(−3·lip_xy) and the same aim_z, so
   the high-and-upright exploit (mouth centre over the receiver from 20 cm up) pays little.
4. **`tilt` (weight 3.0)** = both lifted · not nested · clear · **aim_soft** · **beads_left** · tilt_frac. Soft aim factor
   instead of the hard `aligned` flag → gradient exists from the current state; `beads_left` closes the empty-cup exploit.
5. **`tilt_premature`** now weighted by (1 − aim_soft) instead of (1 − aligned): zero once the lip is over the receiver.
6. **Asymmetric lift targets**: source 0.20 m, receiver 0.10 m — the tilted lip hangs ~5–8 cm below the source origin
   plus the receiver mouth is above its origin, so the source must be ~10 cm+ higher; the `aim_z` gradient fills the rest.
7. Everything else (approach, grasp, lift structure, pour_delta 50 / spill_delta −30, all collision/clearance terms,
   upright, drop, action_rate, success 10·clear) is unchanged — those stages work or are safety terms that are not the
   bottleneck.

Expected movement: `task/src_tilt_deg` should leave 30° within the first third of training, `aim_dist` should fall below
0.05 m, `tilt` and `aim` should rise together, then `pour_delta` and `episode_success` become non-zero and `adr/progress`
starts moving.

```python
import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (N,3),(N,3) -> (N,)
    return torch.norm(a - b, dim=-1)


def _rim_low_point(mouth_pos: torch.Tensor, up: torch.Tensor, radius: float) -> torch.Tensor:
    # Lowest point of the rim circle = the "pouring lip" the beads leave from.
    # Direction = -z projected onto the rim plane (perpendicular to the cup's up axis):
    #   p = -z_hat + (u . z_hat) u ;  lip = mouth + r * p / |p|
    # For an upright cup p -> 0 and the lip collapses to the mouth centre (eps keeps it finite).
    z_hat = torch.cat([torch.zeros_like(up[:, :2]), torch.ones_like(up[:, 2:3])], dim=-1)  # (N,3)
    p = up * up[:, 2:3] - z_hat                                                             # (N,3)
    p_hat = p / (torch.norm(p, dim=-1, keepdim=True) + 1e-6)
    return mouth_pos + radius * p_hat


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET_SRC = 0.20    # [m] source cup must end up clearly higher than the receiver: its tilted lip
    LIFT_TARGET_RCV = 0.10    # [m] hangs ~5-8 cm below its origin and the receiver rim is above the receiver origin
    LIFT_GATE = 0.05          # [m] minimum lift before the pour stack is rewarded
    LIP_DEADBAND = 0.01       # [m] lip within 1 cm of the receiver mouth centre counts as centred (pose noise)
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1 (wide so the gradient is felt early)
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay (10 cm above -> 0.37): high drop-pours spill
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, saturate slightly above
    TILT_FREE = 0.8           # [rad] grasp posture alone tilts the cup 20-35 deg; only charge beyond 46 deg
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties (knowledge item 11)
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes (e.g. finger on table while grasping) are free

    # ------------------------------------------------------------------ clearance factors
    # clear in [0,1]: 1 when the cups do not touch, -> 0 as the cup-cup force grows. Multiplies the whole
    # pour stack (aim + tilt + success) so a bump costs ~6-16/step instead of a small additive penalty.
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ stage 1: approach (unchanged — solved)
    # weight 1.0 per arm, exp(-4d): wide basin, saturates ~0.68 in the grasp posture.
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp (unchanged — solved)
    # 1.0 for the contact-established flag + 0.5 * closure weighted by proximity exp(-6d).
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    grasp_src = 1.0 * g_src + 0.5 * prox_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * prox_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)
    # weight 1.0 bonus for holding BOTH cups: the second grasp is worth more than the first.
    both_grasped_b = ctx.src_grasped & ctx.rcv_grasped
    both_grasped = 1.0 * both_grasped_b.to(dt)

    # ------------------------------------------------------------------ stage 3: lift
    # weight 2.0 per arm — larger than grasp so holding on and lifting beats holding on the table.
    # ASYMMETRIC targets (new): the source saturates at 0.20 m, the receiver at 0.10 m, so the lift stage
    # already produces most of the height difference the pour pose needs (iter_02 lifted both to ~0.14 m).
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    lift_src = 2.0 * g_src * lift_frac_src
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    src_lifted_f = src_lifted_b.to(dt)
    both_lifted_f = both_lifted_b.to(dt)
    # weight 1.0 bonus for both cups in the air
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ pouring-lip geometry (new)
    # iter_02 measured alignment between MOUTH CENTRES; with a near-upright cup that is only reachable by
    # nesting / bumping the cups, so the tilt gate never opened and the policy parked 13 cm away.
    # The point that matters is the lowest rim point ("lip"): it swings over the receiver by TILTING,
    # while the cup body stays outside the receiver. Aim terms are defined on the lip.
    lip = _rim_low_point(ctx.src_cup_mouth_pos, ctx.src_cup_up, ctx.cup_radius)   # (N,3)
    lip_delta = lip - ctx.rcv_cup_mouth_pos                                         # (N,3)
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)                                   # (N,) horizontal miss
    dz_lip = lip_delta[:, 2]                                                        # (N,) lip height above rcv rim
    # above: wide ramp -6 cm -> +2 cm (gradient toward "source higher" is felt long before the pose is right).
    above = torch.clamp((dz_lip - DZ_LOW) / (DZ_HIGH - DZ_LOW), 0.0, 1.0)
    # high: free up to 4 cm above the rim, then gentle decay — a 10 cm drop-pour (iter_01: 7 % spill) pays 0.37.
    high = torch.exp(-(torch.clamp(dz_lip - DZ_FREE, min=0.0) / DZ_SIGMA) ** 2)
    aim_z = above * high
    # aim_xy: exp(-6 d) with a 1 cm deadband — 5 cm miss pays 0.79, 13 cm (the iter_02 fixed point) pays 0.49.
    aim_xy = torch.exp(-6.0 * torch.clamp(lip_xy - LIP_DEADBAND, min=0.0))
    aim_soft = aim_xy * aim_z                                                       # (N,) in [0,1]

    # ------------------------------------------------------------------ stage 4: bring cups together
    # bring_together, weight 1.0: once the SOURCE cup is lifted, pull its lip over the receiver rim with a wide
    # basin exp(-3d); aim_z included so hovering upright 20 cm above the receiver is not a free 1.0.
    bring_together = 1.0 * src_lifted_f * not_nested_f * torch.exp(-3.0 * lip_xy) * aim_z
    # aim, weight 3.0 (replaces align): both lifted, not nested, cups clear. No hard xy / z gate any more —
    # the soft product gives a monotone path from "side by side, 30 deg" to "lip over the rim, 110 deg".
    # Must beat lift (2+2 banked) + bring_together (1.0).
    aim = 3.0 * both_lifted_f * not_nested_f * clear * aim_soft

    # ------------------------------------------------------------------ stage 5: tilt
    # beads_left in [0,1]: beads still in play (in source or already in receiver). Spilled beads shrink the
    # tilt reward permanently, so "dump the beads on the table, then keep tilting" earns nothing.
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    # weight 3.0, scaled by the SOFT aim factor instead of a hard aligned flag: at the iter_02 fixed point
    # (aim ~0.4) tilting from 30 to 60 deg already pays ~+0.4/step and moves the lip closer, so the
    # chicken-and-egg deadlock is gone. At the pour pose aim -> ~0.9 and this pays ~2.7/step.
    tilt = 3.0 * both_lifted_f * not_nested_f * clear * beads_left * aim_soft * tilt_frac
    # premature tilt (weight 0.5): tilting well past grasp posture while the lip is NOT over the receiver.
    # Weighted by (1 - aim_soft) so it vanishes as the lip comes over the rim; spills are priced by spill_delta.
    tilt_premature = -0.5 * (1.0 - aim_soft) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)
    # pour_rate, weight 0.3: gentle cost on fast tilting while aimed — a slow pour spills less.
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    pour_rate = -0.3 * aim_soft * torch.tanh(src_ang_speed / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (unchanged)
    # 50 * dfrac = 2.5 per bead (20 beads) — one-shot payments, must dominate the shaping stack.
    # 30 * dfrac = -1.5 per spilled bead: one landed bead still outweighs one lost.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ collision / clearance (unchanged — real-robot safety)
    # cup_contact, weight 1.0: bounded price on cup-cup force on top of the lost-stage gate.
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    # hand_foreign_*, weight 0.5 each with a 1 N deadband — small versus the ~3.5/step each arm earns from
    # grasp+lift, so the hand is not taught to avoid its own cup (iter_00 failure mode).
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    hand_foreign_src = -0.5 * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -0.5 * torch.tanh(f_rcv / FORCE_SCALE)
    # closing_speed, weight 1.0: rate at which the two cups approach each other inside 30 cm
    # (10 cm/s deadband, 30 cm/s costs ~0.76).
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos                                   # (N,3)
    cup_dist = torch.norm(rel, dim=-1)                                        # (N,)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel                         # (N,3)
    v_close = torch.sum(v_rel * u, dim=-1)                                    # >0 = approaching
    near_f = (cup_dist < NEAR_DIST).to(dt)
    closing_speed = -1.0 * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    # nested, weight 0.5: small explicit price; the gates above already remove all pour income.
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints (unchanged)
    # Receiver-cup penalties stay SMALL (0.3 + 0.5 = 0.8/step worst case) versus the ~7/step the receiver
    # branch earns — larger values taught the left hand to avoid the cup (iter_00).
    upright_rcv = -0.3 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # action-rate regularisation (weight 0.02 — tiny, must never compete with any stage term)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus
    # weight 10.0, withheld while the cups touch: success is defined by the env, the bonus is not.
    success = 10.0 * ctx.success.to(dt) * clear

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "both_grasped": both_grasped,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "both_lifted": both_lifted,
        "bring_together": bring_together,
        "aim": aim,
        "tilt": tilt,
        "tilt_premature": tilt_premature,
        "pour_rate": pour_rate,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
        "cup_contact": cup_contact,
        "hand_foreign_src": hand_foreign_src,
        "hand_foreign_rcv": hand_foreign_rcv,
        "closing_speed": closing_speed,
        "nested": nested,
        "upright_rcv": upright_rcv,
        "drop": drop,
        "action_rate": action_rate,
        "success": success,
    }

    reward = zeros
    for v in components.values():
        reward = reward + v

    return reward, components
```
