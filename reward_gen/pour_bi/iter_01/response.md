# iter_01 — analysis of the previous reward and the improved function

## 1. What the metrics say

### Stages reached

| stage | source arm (right) | receiver arm (left) |
|---|---|---|
| 1 approach | `approach_src` 0.41 → 0.62 (palm ≈ 9.6 cm from cup, i.e. grasp posture) — **reached** | `approach_rcv` 0.40 → **0.35 and falling** (palm ≈ 21 cm away) — **never reached, actually regressed** |
| 2 grasp | `grasp_src` ≈ 1.04, `task/src_grasped` 0.89 — **reached** | `grasp_rcv` ≈ 0, `task/rcv_grasped` ≤ 0.5 % (max 6.7 % during a brief random burst) — **dead** |
| 3 lift | `lift_src` ≈ 1.73 (0.11 m ≥ LIFT_TARGET 0.10 → the term is saturated) — **reached** | `lift_rcv` ≈ 0 — **dead** |
| 4 align | `align` = 0 everywhere (gated on `both_lifted`, which needs the receiver grasp) — **unreachable** | |
| 5 tilt | `tilt` = 0 (gated on `aligned`) — **unreachable**; `src_tilt_deg` ≈ 20–35° is just the grasp posture | |
| 6 pour / success | `pour_delta`, `bead/in_target`, `success` = 0 | |

The whole chain after stage 3 is gated on the receiver arm, and the receiver arm never even approaches its cup. The source arm is done and idles.

### Why the receiver arm went dead — a penalty cliff, not a weak gradient

- First checkpoint (random policy, 105-step episodes): `upright_rcv` = **−0.489** and `drop` = −0.096 — random left-arm motion knocked the receiver cup over in roughly half the envs; `done/drop` 0.9 % ended episodes.
- From the second checkpoint on `upright_rcv` ≈ 0, `drop` ≈ 0 **and `approach_rcv` dropped from 0.40 to 0.33–0.35 and stayed there**. The policy did not fail to find the receiver cup; it learned to keep the left hand *away* from it.
- Arithmetic of the cliff: toppling the receiver cup costs `upright_rcv` up to −1.0 **plus** `rcv_dropped` −2.0 = −3.0 **per step, for the rest of the episode** (a topple is permanent). Episodes are ~880 steps → ≈ −2600 per topple. The entire positive potential of the receiver branch is 1.0 (approach) + 1.5 (grasp) + 2.0 (lift) = 4.5/step and only 0.35 of it is actually collected at 21 cm. A single accidental bump wipes out hundreds of steps of approach income, so "stay away" is the optimum. This is the same contact-cliff failure as a penalised contact term: the state the policy must pass through (hand next to a light, tippable cup) is where the penalty fires.
- Secondary weakness: the closure part of `grasp_rcv` was hard-gated on `d < 0.10 m` and the approach exponent `exp(-5d)` has a gradient of only ≈ 1.7/m at 21 cm, so once the hand retreated nothing pulled it back. The hard gate is also redundant — the environment already only lets fingers close near the cup.

### Secondary observations

- **Idle attractor.** `reward/total` plateaus at 3.7/step, all from the source branch (0.62 + 1.04 + 1.73 + tiny penalties), and `rewards/step` ≈ 3300 = 3.7 × 890 steps. Holding the lifted source cup still is paid for the entire episode; the episode length went 105 → 890 because the policy learned to never drop anything and just sit. `lift_src` is saturated (0.11 m > 0.10 m), so there is no gradient anywhere for the source arm either. Because align/tilt are gated on the receiver, the only thing that can pay more than idling is the receiver branch — which is exactly the branch the cliff forbids.
- **Cups drifting apart.** `task/cups_center_dist` 0.19 → 0.38 m and `aim_dist` ≈ 0.38 m: nothing rewards bringing the cups together before `both_lifted`, and the source arm lifts the cup wherever it happens to be.
- **`tilt_premature` is a dead tax.** It charges −0.01…−0.19 for the 20–35° tilt that is simply the source cup's posture inside a power grasp (`TILT_FREE` = 0.5 rad = 29°). It carries no information (real spills are already priced by `spill_delta`) and slightly punishes a firm grasp.
- `action_rate` ≈ −0.018 constant, harmless. `nested_rate` and `spill` ≈ 0: no nesting exploit yet.
- No component exploit is being farmed (tilt = 0 with beads = 0 is a *gate*, not an exploit). The failure is a **dead signal on the receiver branch caused by a penalty that dominates the reward the branch could ever earn.**

## 2. Changes and why

Primary hypothesis: **the receiver branch must be worth strictly more than the risk of touching the receiver cup, and the penalties for disturbing that cup must be small enough that an accidental bump is recoverable.**

1. **Shrink the receiver-cup penalties by ~6×.** `upright_rcv` weight 1.0 → **0.3**, drop/topple flat penalty 2.0 → **0.5** per cup. Worst case a toppled receiver now costs 0.8/step versus a receiver-branch potential of ≈ 7.5/step (below). Toppling is still strictly worse than a good grasp, but no longer catastrophic. Same logic applied to the source drop so the two arms stay symmetric.
2. **Widen and strengthen the approach pull.** `exp(-5d)` → **`1.0·exp(-4d)`** for both arms (gradient at 21 cm: 1.7 → 1.9/m, and the term still saturates at 0.68 in the grasp posture) — a modest change; the big one is item 1. Approach weight stays 1.0 so the source arm's already-working curriculum is untouched.
3. **Replace the hard `near` gate on closure with a smooth proximity factor** `exp(-8d)` (0.45 at 10 cm, 0.20 at 20 cm). Closing far away is already impossible in the environment, so the only effect is to remove the flat zone that let the receiver hand retreat for free.
4. **Add combinatorial bonuses `both_grasped` (+1.0) and `both_lifted` (+1.0).** The second grasp/lift is now worth more than the first, so the marginal value of the receiver branch once the source is done is 1.0 (approach) + 1.5 (grasp) + 1.0 (both_grasped) + 2.0 (lift) + 1.0 (both_lifted) ≈ 6.5–7.5/step — roughly double the 3.7/step idle income of holding the source cup, which is what breaks the idle attractor.
5. **Add a pre-alignment term `bring_together`** (weight 1.0): once the source cup is grasped and lifted, reward reducing the horizontal rim-to-rim distance while keeping the source rim *above* the receiver rim (`dz > DZ_MIN`), independent of the receiver grasp. This addresses the cups drifting to 0.38 m apart and gives the source arm a gradient again after lift saturation. The z-window gate means it never pays for pushing the source cup *into* the receiver cup (nesting), and its weight (1.0) is below the full `align` term (3.0) so lifting the receiver remains the better path.
6. **Relax `tilt_premature`:** `TILT_FREE` 0.5 → **0.8 rad** and weight 1.0 → 0.5. It now only fires for genuine over-tilting far from the receiver (beads leave past ~1.9 rad), not for grasp posture.
7. Kept unchanged: `lift` (2.0/arm), `align` (3.0), `tilt` (3.0), `pour_delta` (50·Δ = 2.5/bead), `spill_delta` (−20·Δ), `action_rate` (0.02), `success` (10). These stages were either working or never reached, so there is no evidence to retune them; the tilt/pour stack still dominates the shaping stack once aligned.

Expected metric movement: `approach_rcv` ↑ (0.35 → ≥ 0.6), `rcv_grasped` and `lift_rcv` from 0 to non-trivial, `cups_center_dist` ↓, `upright_rcv` temporarily more negative early in training (bumps are now allowed to happen) and then recovering, `align` and `tilt` leaving zero.

## 3. Improved reward function

```python
import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (N,3),(N,3) -> (N,)
    return torch.norm(a - b, dim=-1)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET = 0.10        # [m] cup height above spawn that counts as fully lifted
    LIFT_GATE = 0.05          # [m] minimum lift before alignment / bring_together are rewarded
    ALIGN_XY_GATE = 1.5 * ctx.cup_radius   # [m] rim-over-rim tolerance for the tilt gate
    DZ_MIN, DZ_MAX = 0.03, 0.12            # [m] source rim above receiver rim window
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, so saturate slightly above
    TILT_FREE = 0.8           # [rad] grasp posture alone tilts the cup 20-35 deg; only charge beyond 46 deg
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side

    # ------------------------------------------------------------------ stage 1: approach
    # weight 1.0 per arm, exp(-4d): wider basin than exp(-5d) so a retreated hand (20 cm) is
    # still pulled back; saturates ~0.68 in the grasp posture (palm ~9 cm from cup origin).
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp
    # 1.0 for the contact-established flag + 0.5 * closure weighted by a SMOOTH proximity
    # factor (the env already forbids closing far from the cup, so no hard gate is needed;
    # the old hard gate at 10 cm left a flat zone in which the receiver hand retreated for free).
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    prox_src = torch.exp(-8.0 * d_src)
    prox_rcv = torch.exp(-8.0 * d_rcv)
    grasp_src = 1.0 * g_src + 0.5 * prox_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * prox_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)
    # weight 1.0 bonus for holding BOTH cups: makes the second grasp worth more than the first,
    # so finishing the receiver branch beats idling with the source cup (3.7/step in the last run).
    both_grasped_b = ctx.src_grasped & ctx.rcv_grasped
    both_grasped = 1.0 * both_grasped_b.to(dt)

    # ------------------------------------------------------------------ stage 3: lift
    # weight 2.0 per arm — larger than grasp so holding on and lifting beats holding on the table.
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET, 0.0, 1.0)
    lift_src = 2.0 * g_src * lift_frac_src
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    src_lifted_f = src_lifted_b.to(dt)
    both_lifted_f = both_lifted_b.to(dt)
    # weight 1.0 bonus for both cups in the air (same combinatorial logic as both_grasped)
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ stage 4: bring cups together / align rims
    mouth_delta = ctx.src_cup_mouth_pos - ctx.rcv_cup_mouth_pos          # (N,3)
    d_xy = torch.norm(mouth_delta[:, :2], dim=-1)                          # (N,)
    dz = mouth_delta[:, 2]                                                 # (N,)
    dz_err = torch.clamp(DZ_MIN - dz, min=0.0) + torch.clamp(dz - DZ_MAX, min=0.0)
    align_xy = torch.exp(-8.0 * d_xy)
    align_z = torch.exp(-10.0 * dz_err)

    # bring_together, weight 1.0: once the SOURCE cup is lifted, pull its rim horizontally over the
    # receiver rim while staying ABOVE it (dz > DZ_MIN). Independent of the receiver grasp so the
    # cups stop drifting apart (0.38 m last run) and the source arm keeps a gradient after lift
    # saturation. The height gate means pushing the source cup INTO the receiver cup earns nothing.
    above_rcv = (dz > DZ_MIN).to(dt)
    bring_together = 1.0 * src_lifted_f * above_rcv * torch.exp(-4.0 * d_xy)

    # align, weight 3.0 — full alignment only with both cups lifted; must beat lift (2+2 banked)
    # plus bring_together (1.0) so lifting the receiver remains the better path.
    align = 3.0 * both_lifted_f * align_xy * align_z

    aligned_b = both_lifted_b & (d_xy < ALIGN_XY_GATE) & (dz > 0.0)
    aligned_f = aligned_b.to(dt)

    # ------------------------------------------------------------------ stage 5: tilt
    # weight 3.0, only while aligned; saturates at TILT_TARGET so full pour is the goal.
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    tilt = 3.0 * aligned_f * tilt_frac
    # premature tilt (weight 0.5): tilting well past grasp posture when the rim is NOT over the
    # receiver. Real spills are priced by spill_delta, so this stays small.
    tilt_premature = -0.5 * (1.0 - aligned_f) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)

    # ------------------------------------------------------------------ stage 6: bead transfer
    # 50 * Δfrac = 2.5 per bead (20 beads) — one-shot payments, must dominate the shaping stack.
    # 20 * Δfrac = -1.0 per spilled bead: one bead in the target outweighs two lost.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -20.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ constraints
    # Receiver-cup penalties are deliberately SMALL (0.3 + 0.5 = 0.8/step worst case) versus the
    # ~7/step the receiver branch can earn. Last run they were 1.0 + 2.0 = 3.0/step for the rest
    # of the episode after one accidental bump (~-2600 total), and the policy learned to keep the
    # left hand away from the receiver cup entirely (approach_rcv fell from 0.40 to 0.35).
    upright_rcv = -0.3 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # action-rate regularisation (weight 0.02 — tiny, must never compete with any stage term)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus
    success = 10.0 * ctx.success.to(dt)

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
        "align": align,
        "tilt": tilt,
        "tilt_premature": tilt_premature,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
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
