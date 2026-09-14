# iter_02 — analysis and revised reward

## 1. What the iter_01 metrics say

**Worked (keep):**
- Grasp both hands: `src_grasped` 0.89 / `rcv_grasped` 0.88, `both_grasped` 0.86, `both_lifted` 0.86. The staged grasp → lift → bring_together → align → tilt stack carries the policy to `episode_success` ≈ 0.80 (max 0.82). Drop ≈ 0.
- Nesting exploit is closed: `nested_rate` spiked to 0.24 early (the `dz > DZ_MIN` gate was the only thing fighting it) and then fell below 1 %.
- Tilt reaches ~97°, `bead/in_target` 0.73–0.77, `pour_delta` and `spill_delta` are small per-step deltas as intended. `tilt_premature` stays ~-0.01 (no premature-tilt exploit).

**Reward-structure observations:**
- `episode_lengths/step` ≈ 880 while success ≈ 0.8: the episode is NOT cut at success, so the per-step stack at the success state (~21.6/step: success 8.0 + align 2.2 + tilt 2.4 + lift 3.5 + grasp 1.9 + both_* 1.7 + approach 1.3 + bring 0.75) is a **level** reward paid for the rest of the episode. Every step of delay costs ~21 reward, so the policy is strongly pushed to reach the pour pose as fast as possible. This is the mechanical cause of operator observation 1 (cups dragged together fast and bumping): a transient bump costs nothing in iter_01 (no contact term at all), while arriving 10 steps earlier is worth ~200.
- `src_cup_lift` 0.30 m vs `rcv_cup_lift` 0.14 m (operator observation 2). `lift_*` saturates at 0.10 m, so the extra 0.16 m of source lift is driven by the alignment gate: `align_z` was flat over the whole `dz ∈ [0.03, 0.12]` window and the source sits at the top of it (0.30 − 0.14 ≈ cup_mouth_z + dz with dz ≈ 0.10). Pouring from 10 cm above the receiver rim is what produces the 7 % spill and the long bead trajectories.
- `align` (2.2) and `tilt` (2.4) sit at ~73 % / ~80 % of their maxima — the sharp `exp(-8·d_xy)` never pays full alignment, and with perception noise/curriculum added this round (observation 3) a 1-cm error costing 8 % is too sharp. Tolerances of a few cm are wanted.
- `approach_*` ≈ 0.65 constant: saturated in the grasp posture — it is doing its job (basin) and nothing needs changing.
- `upright_rcv` ≈ -0.10 (receiver tilted ~19° by the grasp posture) — small, keep.

**Hazards remaining:**
1. Cup–cup collision and hand–foreign contacts: unpriced in iter_01; the new fields `cup_cup_force`, `src_hand_foreign_force`, `rcv_hand_foreign_force` make them visible. Since the speed pressure comes from level rewards, a small additive penalty alone cannot compete (a −1/step bump for 5 steps is −5 against +200 for arriving earlier). The collision must therefore cost the **stage rewards themselves**: align, tilt and the success bonus are multiplied by a clearance factor `1 − tanh(cup_cup_force / 5)` — while the cups touch, ~16/step disappears. This is a lost-reward gate, not an avoidance penalty, so it does not re-create the iter_00 failure (heavy receiver-cup penalties taught the left hand to stay away from the cup): the contact-free pour pose still pays in full.
2. Nesting with the loosened `dz` gate: lowering `DZ_MIN` to allow a low pour would let a nested source cup (rim a few cm above the receiver rim) pass the height gate again. `cups_nested` is now an explicit gate on bring_together / align / tilt plus a small penalty.
3. High, fast pour → spill. Addressed by a peaked `align_z` (target 3 cm rim clearance, σ = 3 cm), by capping the tilt gate at `dz < 0.08`, by a gentle angular-speed cost while aligned, and by raising the spill price.

## 2. Changes

| Component | iter_01 | iter_02 | Why |
|---|---|---|---|
| approach_src / approach_rcv | 1.0·exp(-4d) | unchanged | saturated in grasp posture, working |
| grasp_src / grasp_rcv, both_grasped | 1.0 + 0.5·prox·closure, 1.0 | unchanged (prox softened exp(-8d) → exp(-6d)) | few-cm tolerance under noise |
| lift_src / lift_rcv, both_lifted | 2.0, 1.0 | unchanged | working |
| bring_together | 1.0, gate dz>0.03 | 1.0, gate dz>0.01 & not nested & lifted | allow low pour, block nesting explicitly |
| align | 3.0·exp(-8 d_xy)·window[0.03,0.12] | 3.0·exp(-5 d_xy)·exp(-((dz−0.03)/0.03)²)·not_nested·clear | peaked at 3 cm rim clearance instead of flat to 12 cm → lower, gentler pour; softer xy for noise; lost on cup contact |
| tilt | 3.0·aligned | 3.0·aligned·clear, aligned adds dz<0.08 & not nested | no tilt reward from high above or while touching |
| tilt_premature | -0.5 | unchanged | tiny, no exploit seen |
| pour_delta | 50 | unchanged | one-shot 2.5/bead still dominates |
| spill_delta | -20 | **-30** | 1.5/bead: one landed bead (2.5) still beats one lost, but a 10 cm drop-pour becomes unprofitable |
| **cup_contact** (new) | — | -1.0·tanh(f_cc/5) | direct price on cup–cup force (knowledge 11) |
| **hand_foreign_src / hand_foreign_rcv** (new) | — | -0.5·tanh(max(f−1,0)/5) each | hand on other hand / other cup / table; 1 N deadband so incidental brushes during a table-level grasp do not cost, 0.5 ≪ the 3.5/step that arm earns so no avoidance |
| **closing_speed** (new) | — | -1.0·[dist<0.30]·tanh(max(v_close−0.10,0)/0.20) | rate-of-approach of the two cups in the near field; regulariser toward a slow final approach on the real robot |
| **pour_rate** (new) | — | -0.3·aligned·tanh(‖ω_src‖/1.5) | gentle cost on fast tilting once aligned (slower pour spills less) |
| **nested** (new) | — | -0.5·cups_nested | explicit, small; the gates do the real work |
| upright_rcv, drop, action_rate | -0.3, -0.5, -0.02 | unchanged | small, working |
| success | 10.0·success | 10.0·success·clear | bonus withheld while the cups touch |

Expected movement: `cup_cup_force`-based rates and `*_hand_foreign_rate` fall; `src_cup_lift` drops from ~0.30 toward ~0.20 (rcv_lift + cup_mouth_z + 3 cm); `bead/spill` falls below 5 %; `align` rises toward 2.7–3.0 as the peaked/softer terms are actually attainable; `episode_success` should hold ≥ 0.8 (nothing that earned it was removed, only gated on clearance). Watch `nested_rate` (must stay < 1 %) and `both_lifted` (must not fall — if the foreign-force term suppresses grasps, halve its weight).

## 3. Final reward function

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
    DZ_MIN = 0.01             # [m] source rim must stay above the receiver rim (nesting guard)
    DZ_AIM = 0.03             # [m] preferred rim clearance — a low, gentle pour
    DZ_SIGMA = 0.03           # [m] tolerance around DZ_AIM (perception noise is a few cm)
    DZ_TILT_MAX = 0.08        # [m] no tilt reward when pouring from higher than this
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, so saturate slightly above
    TILT_FREE = 0.8           # [rad] grasp posture alone tilts the cup 20-35 deg; only charge beyond 46 deg
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties (knowledge item 11)
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes (e.g. finger on table while grasping) are free

    # ------------------------------------------------------------------ clearance factors
    # clear in [0,1]: 1 when the cups do not touch, -> 0 as the cup-cup force grows.
    # Multiplies align, tilt and the success bonus: a bump costs the whole pour stack
    # (~16/step) instead of a small additive penalty that the level rewards would outrun.
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ stage 1: approach
    # weight 1.0 per arm, exp(-4d): wide basin, saturates ~0.68 in the grasp posture.
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp
    # 1.0 for the contact-established flag + 0.5 * closure weighted by a smooth proximity
    # factor (exp(-6d), softened from -8d: a few cm of cup-pose noise must not zero it).
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
    # weight 1.0 bonus for both cups in the air
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ stage 4: bring cups together / align rims
    mouth_delta = ctx.src_cup_mouth_pos - ctx.rcv_cup_mouth_pos          # (N,3)
    d_xy = torch.norm(mouth_delta[:, :2], dim=-1)                          # (N,)
    dz = mouth_delta[:, 2]                                                 # (N,)
    # xy: exp(-5d) (was -8d) — 2 cm error still pays 0.90, 5 cm pays 0.78; tolerant to noise.
    align_xy = torch.exp(-5.0 * d_xy)
    # z: PEAKED at DZ_AIM (3 cm rim clearance) instead of flat up to 12 cm. iter_01 poured from
    # ~10 cm above the receiver rim (src lift 0.30 vs rcv 0.14) and spilled 7 %; this pulls the
    # source rim down to a few cm. Below the rim (dz < DZ_MIN) the gate below zeroes everything.
    align_z = torch.exp(-((dz - DZ_AIM) / DZ_SIGMA) ** 2)
    above_rcv = (dz > DZ_MIN).to(dt)

    # bring_together, weight 1.0: once the SOURCE cup is lifted, pull its rim horizontally over
    # the receiver rim while staying above it and NOT nested (explicit nesting gate replaces the
    # old 3 cm height margin, so a low pour is allowed without reopening the nesting exploit).
    bring_together = 1.0 * src_lifted_f * above_rcv * not_nested_f * torch.exp(-4.0 * d_xy)

    # align, weight 3.0 — full alignment only with both cups lifted, above, not nested and with
    # the cups NOT touching (clear). Must beat lift (2+2 banked) + bring_together (1.0).
    align = 3.0 * both_lifted_f * above_rcv * not_nested_f * align_xy * align_z * clear

    aligned_b = (both_lifted_b & (~ctx.cups_nested)
                 & (d_xy < ALIGN_XY_GATE) & (dz > 0.0) & (dz < DZ_TILT_MAX))
    aligned_f = aligned_b.to(dt)

    # ------------------------------------------------------------------ stage 5: tilt
    # weight 3.0, only while aligned (now also: not from higher than 8 cm) and cups clear.
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    tilt = 3.0 * aligned_f * tilt_frac * clear
    # premature tilt (weight 0.5): tilting well past grasp posture when the rim is NOT over the
    # receiver. Real spills are priced by spill_delta, so this stays small.
    tilt_premature = -0.5 * (1.0 - aligned_f) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)
    # pour_rate, weight 0.3: gentle cost on fast tilting while aligned — a slow pour spills less.
    # tanh(|w|/1.5): 1 rad/s costs 0.17, saturates at 0.3, never competes with tilt (3.0).
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    pour_rate = -0.3 * aligned_f * torch.tanh(src_ang_speed / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer
    # 50 * Δfrac = 2.5 per bead (20 beads) — one-shot payments, must dominate the shaping stack.
    # 30 * Δfrac = -1.5 per spilled bead (was -1.0): one landed bead still outweighs one lost,
    # but a high drop-pour that loses 1 bead in 2 is no longer profitable.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ collision / clearance (real-robot safety)
    # cup_contact, weight 1.0: bounded price on cup-cup force on top of the lost-stage gate.
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    # hand_foreign_*, weight 0.5 each with a 1 N deadband: hand on the other hand / other cup /
    # table. 0.5 << the ~3.5/step that arm earns from grasp+lift, so the hand is not taught to
    # avoid its own cup (the iter_00 failure mode of heavy receiver-cup penalties).
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    hand_foreign_src = -0.5 * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -0.5 * torch.tanh(f_rcv / FORCE_SCALE)
    # closing_speed, weight 1.0: rate at which the two cups approach each other inside 30 cm.
    # The level rewards pay ~21/step at the pour pose, so speed pressure is inherent; this term
    # only shapes the final approach to be slow (10 cm/s deadband, 30 cm/s costs ~0.76).
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos                                   # (N,3)
    cup_dist = torch.norm(rel, dim=-1)                                        # (N,)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel                         # (N,3)
    v_close = torch.sum(v_rel * u, dim=-1)                                    # >0 = approaching
    near_f = (cup_dist < NEAR_DIST).to(dt)
    closing_speed = -1.0 * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    # nested, weight 0.5: small explicit price; the gates above already remove all pour income.
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints
    # Receiver-cup penalties stay SMALL (0.3 + 0.5 = 0.8/step worst case) versus the ~7/step
    # the receiver branch earns — larger values taught the left hand to avoid the cup (iter_00).
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
        "align": align,
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
