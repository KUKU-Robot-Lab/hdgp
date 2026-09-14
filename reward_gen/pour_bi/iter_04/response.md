# iter_04 — reward analysis and revision

## 1. Task stages (unchanged)

1. **Approach** — each palm moves to its own cup (right → source, left → receiver).
2. **Grasp** — fingers close; for the source cup this must be a *power/wrap* grasp (cup body inside the palm, palm pressed against the body, ≥4 fingers around it), because the cup will later be rotated past 110° while the hand alone holds it.
3. **Lift** — both cups off the table; source higher than receiver.
4. **Bring together** — the source cup's pouring lip is carried over the receiver rim, cups never touching, hands never touching each other or the other cup. This must happen in the middle of the shared workspace where both arms have room.
5. **Tilt / pour** — source cup rotated past ~1.9 rad with its lip over the receiver mouth; beads fall through the air.
6. **Hold** — receiver upright, nothing dropped, little spill; env declares success at ≥ 50 % beads in receiver.

## 2. What the iter_03 feedback says

**What worked (keep):**
- approach/grasp/lift are solved: `src_grasped` 0.91, `rcv_grasped` 0.90, `both_lifted` 0.89, lift_src/lift_rcv saturated (1.8 / 1.79 of 2.0).
- The soft lip-aim stack removed the iter_02 deadlock: `aim` 0 → 2.45 (of 3.0), `tilt` 0 → 1.95, `bring_together` 0 → 0.82, `aim_dist` 0.92 → 0.11 m, `src_tilt_deg` → 86°, and finally `bead/in_target` 0.52, `episode_success` 0.54, spill 4 %. `nested_rate` and `cup_collision_rate` collapsed to ≈ 0 — the lost-stage gate on `cup_cup_force` works.
- `spill_delta`, `drop`, `nested`, `closing_speed` are essentially zero: not binding, no need to touch.

**What is wrong (must change):**

| symptom | evidence | reward cause |
|---|---|---|
| Source hand hooks the rim from above, cup body outside the palm | `task/src_closure` flat at 0.38 all run; `src_grasped` 0.90; `grasp_src` saturated at 1.0 | `src_grasped` is a two-finger flag; `grasp_src = 1.0·flag + 0.5·prox·closure` pays 1.0 for the hook and only +0.1 more for a full wrap. Nothing in the stack (lift, aim, tilt) cares *how* the cup is held. |
| Pour is done by resting the tilted source cup / right hand on the left hand's fingers and the receiver rim | `rcv_hand_foreign_rate` 0.03 → 0.44 (max 0.55) rising together with success; `hand_foreign_rcv` only −0.11/step | The clearance gate `clear` only looks at cup–cup force. Hand–foreign force is a 0.5-weight additive penalty with a 1 N deadband, i.e. ≈ 2 % of the ~5/step that aim+tilt+bring_together pay at the pour pose — the support strategy is cheaper than holding the cup properly. |
| Pour drifts to the far right (receiver carried to y < −0.15 m), right arm cramped | operator video; `cups_center_dist` 0.21 is fine, but no metric constrains *where* | Every pour term is a relative quantity (lip vs rim). The left arm's transport is easy so it does all the moving; nothing prices the right arm's loss of workspace. |
| Source lifted 0.35 m (target 0.20) | `task/src_cup_lift` 0.35 | Consequence of the hook: the cup hangs below the fingers, so the palm has to be much higher to get the lip above the rim. Fixing the grasp fixes this — no lift change. |

Symptoms 1 and 2 are one behaviour: a hooked cup cannot be rotated to 110° by the right hand alone, so the policy borrows the left hand as a support. The operator names the grasp as the root cause; the reward must (a) pay for the wrap itself and (b) make the borrowed support unprofitable.

Constant-component check: `action_rate`, `upright_rcv`, `drop`, `nested`, `closing_speed`, `spill_delta` are small and flat — they are satisfied constraints, left as is. `approach_*` flat at ~0.65 = saturated by design.

## 3. Changes

1. **Wrap-quality measure `wq` (new, both hands)** from the per-finger signals now exposed:
   - `finger_frac` — fraction of the 5 fingers pressing the own cup (> 0.3 N); hook ≈ 0.4, wrap ≈ 1.
   - `palm_c = tanh(palm_force / 2)` — cup body pressed into the palm; hook = 0.
   - `side = 1 − |palm_normal · cup_up|` — palm faces the cup *side* (hook from above gives |·| ≈ 1 → 0). Invariant under tilting because normal and cup axis rotate together, so it does not fight the pour.
   - `level = exp(−(axial/0.05)²)` — cup origin at palm height along the cup axis (hook: origin 8–12 cm below the palm → ≈ 0).
   - `clos` — closure ramp 0.35 → 0.65 (hook 0.38 → 0.1).
   Weighted mean 0.30/0.30/0.20/0.10/0.10 so each ingredient has its own gradient; hook ≈ 0.15, wrap ≈ 1.
   - Direct income: `wrap_src = 1.5·grasped·wq`, `wrap_rcv = 0.5·grasped·wq` (receiver already wraps; small term just stabilises it).
   - `grip = 0.3 + 0.7·wq_src` multiplies `lift_src`, `aim` and `tilt`. With the hook the pour stack is worth ~40 %; with a wrap 100 %. Total incentive for wrapping ≈ +5/step at the pour pose and it is already present on the table, where the grasp is formed. Floor 0.3 keeps the stage reachable if the wrap is slow to appear.

2. **Hand-foreign contact becomes a lost-stage gate.** `clean = 1 − tanh((max(src_foreign, rcv_foreign) − 0.5 N)/5)` multiplies `bring_together`, `aim`, `tilt` and the `success` bonus exactly like `clear` does for cup–cup force. Resting on the left hand at 5 N now forfeits ~4–5/step plus 7.6 of the 10 success bonus instead of costing 0.1. The additive penalties go 0.5 → 1.0 (deadband 1 N kept, still ≪ the 3.5/step each arm earns from grasp+lift, so the iter_00 cup-avoidance failure cannot recur). `pour_delta`/`spill_delta` stay ungated so the bead signal itself is never lost.

3. **Meeting-point shaping (new).** Midline `mid_y = ½(src_spawn_y + rcv_spawn_y)`. Once both cups are lifted: `meet_rcv = −1.0·tanh(max(mid_y − rcv_y − 0.02, 0)/0.10)` (receiver carried into the right half; at the observed −0.15 m it costs −0.9/step), and a milder `meet_src = −0.5·tanh(max(src_y − mid_y − 0.05, 0)/0.10)` for symmetry. The pour geometry is unaffected: the tilted source body naturally sits ~8 cm to the right of the receiver, so the target configuration (receiver ≈ mid, source ≈ mid − 0.06) is penalty-free, and both arms work near their shared centre.

4. Everything else (approach, grasp, lift targets, lip geometry, aim/tilt shapes, bead deltas, closing speed, drop, upright, action-rate, success 10) is unchanged.

**Expected metric movement:** `task/src_closure` 0.38 → > 0.55 and `src_palm_force` > 0 (new `wrap_src` rises toward 1.5); `rcv_hand_foreign_rate` 0.44 → < 0.1; `task/src_cup_lift` 0.35 → ~0.25; receiver cup y stays ≥ −0.02; `aim`/`tilt` dip at first (grip/clean gates) then recover; success should re-reach ≥ 0.5 without hand–hand contact.

## 4. Reward function

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


def _wrap_quality(finger_force: torch.Tensor, palm_force: torch.Tensor, closure: torch.Tensor,
                  palm_axes: torch.Tensor, cup_up: torch.Tensor, cup_pos: torch.Tensor,
                  palm_pos: torch.Tensor) -> torch.Tensor:
    # Power-grasp quality in [0,1] (new in iter_04). iter_03's right hand hooked the rim from above with
    # two fingertips (closure 0.38, palm force 0, palm normal pointing down the cup axis) and satisfied
    # the two-finger `grasped` flag. A wrap has: most fingers pressing, the cup body pressed into the
    # palm, the palm facing the cup SIDE, and the cup origin at palm height. Weighted MEAN (not product)
    # so every ingredient carries its own gradient from the hook posture.
    FINGER_ON = 0.3      # [N] a finger counts as "on the cup" above this force
    PALM_SCALE = 2.0     # [N] palm force tanh scale: 2 N -> 0.76
    LEVEL_SIGMA = 0.05   # [m] cup origin within ~5 cm of palm height along the cup axis
    CLOS_LO, CLOS_HI = 0.35, 0.65   # closure ramp: the measured hook (0.38) -> 0.1, a wrap -> 1
    dt = closure.dtype
    finger_frac = torch.mean((finger_force > FINGER_ON).to(dt), dim=-1)                # (N,)
    palm_c = torch.tanh(torch.clamp(palm_force, min=0.0) / PALM_SCALE)                 # (N,)
    normal = palm_axes[:, 0:3]                                                          # (N,3) palm normal
    # side: 1 when the palm normal is perpendicular to the cup axis (grasp from the side), 0 when the
    # palm sits on top of / under the cup. Rotates with the cup, so tilting does not change it.
    side = 1.0 - torch.clamp(torch.abs(torch.sum(normal * cup_up, dim=-1)), 0.0, 1.0)  # (N,)
    axial = torch.sum((cup_pos - palm_pos) * cup_up, dim=-1)                            # (N,) origin height vs palm
    level = torch.exp(-(axial / LEVEL_SIGMA) ** 2)                                       # (N,)
    clos = torch.clamp((closure - CLOS_LO) / (CLOS_HI - CLOS_LO), 0.0, 1.0)             # (N,)
    return 0.30 * finger_frac + 0.30 * palm_c + 0.20 * side + 0.10 * level + 0.10 * clos


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET_SRC = 0.20    # [m] source cup must end up clearly higher than the receiver
    LIFT_TARGET_RCV = 0.10    # [m]
    LIFT_GATE = 0.05          # [m] minimum lift before the pour stack is rewarded
    LIP_DEADBAND = 0.01       # [m] lip within 1 cm of the receiver mouth centre counts as centred (pose noise)
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay (10 cm above -> 0.37)
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, saturate slightly above
    TILT_FREE = 0.8           # [rad] grasp posture alone tilts the cup 20-35 deg; only charge beyond 46 deg
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties (knowledge item 11)
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes (finger on table while grasping) are free (additive term)
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative hand-foreign gate (pour stage only)
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a hook grasp, 100 % with a wrap
    MEET_RCV_FREE = 0.02      # [m] receiver cup may cross the spawn midline by 2 cm for free
    MEET_SRC_FREE = 0.05      # [m] source cup may cross the midline by 5 cm for free
    MEET_SCALE = 0.10         # [m] tanh scale of the crossing penalties (15 cm over -> 0.9)

    # ------------------------------------------------------------------ clearance factors
    # clear in [0,1]: 1 when the cups do not touch, -> 0 as the cup-cup force grows (unchanged, worked:
    # cup_collision_rate 0.28 -> 0.005).
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    # clean in [0,1] (new): same idea for hand-foreign force. iter_03 poured by resting the tilted source cup
    # on the left hand's fingers (rcv_hand_foreign_rate 0.44 while success rose) because the additive
    # 0.5-weight penalty was ~2 % of the pour-stack income. Now touching the other hand / cup / table with
    # either hand forfeits bring_together + aim + tilt + the success bonus, exactly like a cup-cup bump.
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ stage 1: approach (unchanged — solved)
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp (flag part unchanged)
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    grasp_src = 1.0 * g_src + 0.5 * prox_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * prox_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)
    both_grasped_b = ctx.src_grasped & ctx.rcv_grasped
    both_grasped = 1.0 * both_grasped_b.to(dt)

    # wrap quality (new): hook ~0.15, full power grasp ~1.0
    wq_src = _wrap_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                           ctx.src_palm_axes, ctx.src_cup_up, ctx.src_cup_pos, ctx.src_palm_pos)
    wq_rcv = _wrap_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                           ctx.rcv_palm_axes, ctx.rcv_cup_up, ctx.rcv_cup_pos, ctx.rcv_palm_pos)
    # wrap_src weight 1.5: bigger than the 1.0 grasp flag so that, once the flag is banked, the next thing
    # worth doing on the table is enveloping the cup. wrap_rcv 0.5: the left hand already wraps; keep it.
    wrap_src = 1.5 * g_src * wq_src
    wrap_rcv = 0.5 * g_rcv * wq_rcv
    # grip in [0.3,1]: multiplies lift_src, aim and tilt. With the hook the whole downstream stack is worth
    # ~40 %, with a wrap 100 % -> ~+5/step at the pour pose for wrapping, felt already at grasp time.
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src

    # ------------------------------------------------------------------ stage 3: lift
    # weight 2.0 per arm; asymmetric targets unchanged. lift_src additionally scaled by grip (see above).
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    lift_src = 2.0 * g_src * grip * lift_frac_src
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    src_lifted_f = src_lifted_b.to(dt)
    both_lifted_f = both_lifted_b.to(dt)
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ pouring-lip geometry (unchanged — worked)
    lip = _rim_low_point(ctx.src_cup_mouth_pos, ctx.src_cup_up, ctx.cup_radius)   # (N,3)
    lip_delta = lip - ctx.rcv_cup_mouth_pos                                         # (N,3)
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)                                   # (N,) horizontal miss
    dz_lip = lip_delta[:, 2]                                                        # (N,) lip height above rcv rim
    above = torch.clamp((dz_lip - DZ_LOW) / (DZ_HIGH - DZ_LOW), 0.0, 1.0)
    high = torch.exp(-(torch.clamp(dz_lip - DZ_FREE, min=0.0) / DZ_SIGMA) ** 2)
    aim_z = above * high
    aim_xy = torch.exp(-6.0 * torch.clamp(lip_xy - LIP_DEADBAND, min=0.0))
    aim_soft = aim_xy * aim_z                                                       # (N,) in [0,1]

    # ------------------------------------------------------------------ stage 4: bring cups together
    # weights unchanged (1.0 / 3.0); new gates: clean (hand-foreign) and grip (wrap quality).
    bring_together = 1.0 * src_lifted_f * not_nested_f * clean * torch.exp(-3.0 * lip_xy) * aim_z
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    aim = 3.0 * stack_gate * grip * aim_soft

    # ------------------------------------------------------------------ stage 5: tilt (weights unchanged)
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    tilt = 3.0 * stack_gate * grip * beads_left * aim_soft * tilt_frac
    tilt_premature = -0.5 * (1.0 - aim_soft) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    pour_rate = -0.3 * aim_soft * torch.tanh(src_ang_speed / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (unchanged, ungated)
    # 50 * dfrac = 2.5 per bead; -30 * dfrac = -1.5 per spilled bead. Never gated: the bead signal must survive.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ workspace: meet near the midline (new)
    # iter_03 carried the receiver cup all the way into the right hand's home region (y < -0.15 m) and the
    # right arm ran out of room. Midline from the spawn positions (robust to spawn randomisation).
    # meet_rcv weight 1.0: receiver cup crossing into the source half beyond 2 cm; 15 cm over -> -0.9/step.
    # meet_src weight 0.5: mirror image with a 5 cm allowance (the tilted source body naturally sits ~6-8 cm
    # to the RIGHT of the receiver, so the target pose "receiver at mid, source at mid-0.06" is free).
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    rcv_cross = mid_y - ctx.rcv_cup_pos[:, 1]          # >0: receiver moved toward the source side (-y)
    src_cross = ctx.src_cup_pos[:, 1] - mid_y          # >0: source moved toward the receiver side (+y)
    meet_rcv = -1.0 * both_lifted_f * torch.tanh(torch.clamp(rcv_cross - MEET_RCV_FREE, min=0.0) / MEET_SCALE)
    meet_src = -0.5 * both_lifted_f * torch.tanh(torch.clamp(src_cross - MEET_SRC_FREE, min=0.0) / MEET_SCALE)

    # ------------------------------------------------------------------ collision / clearance (real-robot safety)
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    # hand_foreign_* 0.5 -> 1.0 each, 1 N deadband kept: still small versus the ~3.5/step each arm earns from
    # grasp+lift (no own-cup avoidance, iter_00), while the multiplicative `clean` gate does the real work.
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    hand_foreign_src = -1.0 * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -1.0 * torch.tanh(f_rcv / FORCE_SCALE)
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos                                   # (N,3)
    cup_dist = torch.norm(rel, dim=-1)                                        # (N,)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel                         # (N,3)
    v_close = torch.sum(v_rel * u, dim=-1)                                    # >0 = approaching
    near_f = (cup_dist < NEAR_DIST).to(dt)
    closing_speed = -1.0 * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints (unchanged)
    upright_rcv = -0.3 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus
    # weight 10.0, withheld while the cups touch (clear) OR a hand touches something foreign (clean, new):
    # a pour that leans on the other hand is not a deployable success.
    success = 10.0 * ctx.success.to(dt) * clear * clean

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "wrap_src": wrap_src,
        "wrap_rcv": wrap_rcv,
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
        "meet_rcv": meet_rcv,
        "meet_src": meet_src,
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
