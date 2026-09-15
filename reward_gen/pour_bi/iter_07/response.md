# iter_07 — reward function response

## 1. What the task means (stages)

1. **Approach (both arms)**: the right palm goes to the source cup, which holds the beads. The left palm goes to the empty receiver cup.
2. **Wrap grasp (both hands)**: each hand closes around its own cup. The palm touches the cup and the thumb plus the other fingers press on it (`*_grasped`). Contact freeze stops each finger on the cup, so measured closure tops out at about 0.35–0.43.
3. **Lift (both cups)**: the source cup goes about 20 cm up and the receiver cup about 10 cm up. The receiver cup stays mouth-up.
4. **Bring together**: move the source cup's pouring lip over the receiver cup's mouth, slightly above its rim. The cups must not touch (`cup_cup_force`), neither hand may touch anything but its own cup, and the cups must not nest.
5. **Tilt / pour**: tilt the source cup past about 1.9 rad, towards the receiver, with the lip over the receiver mouth. Beads fall through the air into the receiver cup, which is held with at most 20° of tilt.
6. **Hold after the pour**: keep both wrap grasps, keep the full receiver level, no drop and little spill. Success comes from the environment: at least half the beads in the receiver, little spill, receiver tilt ≤ 20°, cups close but not nested.

## 2. Analysis of the feedback

**What the source side learned (normal):**
- `src_grasped` reached 0.88 and `wrap_src` 1.08.
- `lift_src` reached 1.44, with the source cup about 0.19 m up.
- The source cup is tilted to 53°.
- This matches the earlier run that did learn the receiver.

**What the receiver side did (nothing):**
- `task/rcv_grasped` was 0.000 at every checkpoint.
- `rcv_closure` stayed at 0.01–0.02, so the receiver hand stayed open.
- `rcv_cup_lift` was 0 and `approach_rcv` sat flat at 0.45 all run. That value is just the arm's resting distance; it never improved.
- The earlier run had `grasp_rcv` at 0.73 by epoch 50. This run had 0.01.
- Every stacked term was exactly 0: `both_grasped`, `both_lifted`, `aim`, `tilt`, `pour_pose`, `success`. The cups stayed 0.48 m apart.

**Why the receiver branch never started:**
- In the previous function, `upright_rcv = -1.0 * tanh((rcv_tilt - 0.12)/0.25)` is paid **all the time**, even while the receiver cup stands untouched on the table.
- Random exploration bumps the receiver cup to 15–19° of tilt. That cost −0.36 per step at epoch 1 (earlier run: −0.10) and −0.22 at epoch 5.
- The receiver's early income is tiny by comparison. `grasp_rcv` is about 0.01 and needs a thumb and a finger both pressing on the cup, which is rare at random.
- So the cheapest thing to learn was "never touch the receiver cup". By epoch 20 the receiver cup was untouched (0.1–0.7° tilt) and the upright penalty was about 0.
- From then on the receiver hand has no reason to approach. Any contact tilts the cup and is charged at once, while the grasp and lift income sits behind that contact.
- This is a local optimum made by the penalty itself.

**Secondary contributor with the same cause:**
- `closing_speed` was −0.28 at epoch 1. It is paid whenever the cup centres are within 0.30 m and approach each other at more than 0.1 m/s.
- The cups start about 0.20 m apart, so knocking either cup on the table (including the receiver cup while reaching for it) is charged too.
- The term exists to slow the carried cups before they meet. Before either cup is carried it only discourages touching the cups.
- The real safety signal, `cup_contact` (the actual cup–cup force), is kept unchanged and ungated.

**Upright scaling that runs too early:**
- `lift_rcv` has a `(0.5 + 0.5*rcv_up)` factor.
- `bring_together` and `aim` use `rcv_up_soft`.
- `hold_rcv`, `tilt`, `tilt_dir` and `pour_pose` use `rcv_up`.
- The operator requirement says none of this may act before the receiver cup is grasped **and** lifted.
- `bring_together` is gated only on the source being lifted, so a bumped receiver on the table used to cut it to 30%.

**Possible exploits of the new gating, checked:**
- **Tilt the receiver, grasped but still on the table, to meet the lip.** `aim`, `tilt` and `pour_pose` need `stack_gate`, which needs the receiver above 5 cm, where the upright gating is already fully on. Only `bring_together` (weight 1) could be earned that way. Lifting pays far more (`lift_rcv` 2, `both_lifted` 1, `aim` 3, `tilt` 4, `pour_pose` 3), and success still needs ≤ 20°.
- **A bump raising the cup's origin and switching the phase on.** Tipping a cup on its bottom rim raises its origin by only about 1 cm, even at 45°. The carried phase therefore starts at 1.5 cm of lift and needs the grasp. It is also switched on without the grasp above 6 cm, so a grasp that flickers while carrying cannot switch the penalty off.
- **Releasing the grasp during the pour to avoid the penalty.** Above 6 cm the phase no longer needs the grasp. Releasing also forfeits `lift_rcv`, `both_lifted`, `wrap_rcv`, `hold_rcv` and the whole pour stack.
- **Knocking the receiver over during the approach.** The `drop` term, whose topple threshold is 1.2 rad or about 69°, is kept. That is a lost cup (and an environment `done/drop`), not wobble from hand contact.

## 3. Changes (one hypothesis: phase-dependent receiver-upright handling)

1. **`rcv_carry` phase in [0,1]:**
   - `rcv_carry = max(g_rcv * smoothstep((h_rcv - 0.015)/0.035), smoothstep((h_rcv - 0.06)/0.04))`
   - It is 0 while the cup is on the table (approach, contact, grasp closing, bumps).
   - It reaches 1 once the grasped cup is 5 cm up, or once the cup is 10 cm up whatever the grasp flag says.
2. **`upright_rcv` is multiplied by `rcv_carry`.** During the pour it is also scaled up by a pour phase: `(1.0 + 1.0*pour_phase)`.
   - `pour_phase = max(both_lifted * aim_loose * smoothstep((src_tilt-0.8)/0.8), post_pour)`.
   - Charge at 20° of receiver tilt: −0.73 while carrying, up to −1.46 while pouring or holding the poured beads.
   - Charge at 10°: −0.22 while carrying, −0.44 while pouring.
   - While the cup is on the table the charge is exactly 0.
3. **Upright-dependent scaling is phase-gated:** `rcv_up_eff = 1 - rcv_carry*(1 - rcv_up)`.
   - It replaces `rcv_up` in `lift_rcv`, `bring_together`, `aim`, `tilt`, `tilt_dir`, `pour_pose` and `hold_rcv`.
   - Once the receiver is grasped and lifted it equals the old gate, so the pour stack is exactly as strict as before (27% of the income left at 20°, about 0 at 30°).
4. **`closing_speed` only applies while at least one cup is carried.**
   - It is multiplied by `max(src_carry, rcv_carry)`, and `src_carry` uses the same ramp.
   - Knocking a cup on the table during the approach is no longer charged here. Real cup–cup contact is still charged by `cup_contact` and gates `success`.
5. **Everything else is unchanged:** approach, grasp and wrap for both hands, the lift targets, lip aiming, the bead increments, spill, the post-pour hold, the midline limits, the foreign-contact safety terms, the command smoothness terms, drop and the success bonus. The source side learned correctly with them, and the earlier run learned the receiver with the same approach and grasp terms.

**Expected metric movement:**
- `reward/upright_rcv` is about 0 in early epochs even though `task/rcv_tilt_deg` stays at 10–19° (hand bumps are now allowed).
- `reward/grasp_rcv` should rise to 0.2 or more by epoch 20–50, and `task/rcv_grasped` to 0.5 or more by epoch 50.
- `task/rcv_cup_lift` should pass 0.05 m around epoch 100.
- `both_lifted` and `aim` should become non-zero, and `aim_dist` should fall below 0.25 m later.
- Once the receiver is carried, `upright_rcv` may briefly go negative, then carried receiver tilt should settle below about 10°.

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


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    # x already normalised; clamp to [0,1] then 3x^2 - 2x^3 (smooth ends, steepest in the middle)
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wrap_quality(finger_force: torch.Tensor, palm_force: torch.Tensor, closure: torch.Tensor,
                  palm_axes: torch.Tensor, cup_up: torch.Tensor, cup_pos: torch.Tensor,
                  palm_pos: torch.Tensor) -> torch.Tensor:
    # Power-grasp quality in [0,1]. Weighted MEAN so each ingredient keeps its own gradient.
    FINGER_ON = 0.3      # [N] a finger counts as "on the cup" above this force
    PALM_SCALE = 3.0     # [N] palm contact saturation (a fingertip pinch with ~2 N palm scores clearly lower)
    LEVEL_SIGMA = 0.05   # [m] cup origin within ~5 cm of palm height along the cup axis
    # closure saturates at 0.35-0.43 on this cup under contact freeze whatever the grasp type
    CLOS_LO, CLOS_HI = 0.25, 0.40
    dt = closure.dtype
    finger_frac = torch.mean((finger_force > FINGER_ON).to(dt), dim=-1)                # (N,)
    palm_c = torch.tanh(torch.clamp(palm_force, min=0.0) / PALM_SCALE)                 # (N,)
    normal = palm_axes[:, 0:3]                                                          # (N,3) palm normal
    side = 1.0 - torch.clamp(torch.abs(torch.sum(normal * cup_up, dim=-1)), 0.0, 1.0)  # (N,)
    axial = torch.sum((cup_pos - palm_pos) * cup_up, dim=-1)                            # (N,)
    level = torch.exp(-(axial / LEVEL_SIGMA) ** 2)                                       # (N,)
    clos = torch.clamp((closure - CLOS_LO) / (CLOS_HI - CLOS_LO), 0.0, 1.0)             # (N,)
    return 0.30 * finger_frac + 0.30 * palm_c + 0.20 * side + 0.10 * level + 0.10 * clos


def _carry_phase(grasped_f: torch.Tensor, h: torch.Tensor,
                 grasp_lo: float, grasp_hi: float, free_lo: float, free_hi: float) -> torch.Tensor:
    # (N,) in [0,1]: 0 while the cup stands on the table (approach / contact / grasp closing / bumps),
    # 1 once it is grasped and lifted. Tipping a cup on its bottom rim raises its origin only ~1 cm, so the
    # grasped ramp starts at 1.5 cm; far above the table (>= 6 cm) the cup must be carried, so the second
    # ramp ignores a flickering grasp flag and releasing the grasp mid-pour cannot switch the phase off.
    held_ramp = grasped_f * _smoothstep((h - grasp_lo) / (grasp_hi - grasp_lo))
    high_ramp = _smoothstep((h - free_lo) / (free_hi - free_lo))
    return torch.maximum(held_ramp, high_ramp)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET_SRC = 0.20    # [m] source cup must end up clearly higher than the receiver
    LIFT_TARGET_RCV = 0.10    # [m] receiver lifted moderately (operator requirement)
    LIFT_GATE = 0.05          # [m] minimum lift before the pour stack is rewarded
    LIP_DEADBAND = 0.01       # [m] lip within 1 cm of the receiver mouth centre counts as centred (pose noise)
    LIP_DEADBAND_LOOSE = 0.02 # [m] deadband of the loose aim gate used by the tilt-progress term
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay (10 cm above -> 0.37)
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, saturate slightly above
    POUR_TILT_LO = 1.3        # [rad] start of the release band bonus (~75 deg)
    POUR_TILT_HI = 2.0        # [rad] end of the release band bonus (~115 deg)
    TILT_DIR_SCALE = 1.0      # [rad] direction term fully weighted once tilted >= 57 deg
    TILT_FREE = 1.5           # [rad] unaimed tilt below 86 deg spills nothing -> free
    ROT_SPEED_FREE = 1.5      # [rad/s] controlled pouring rotation is free; only violent flips are charged
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side (a lost cup, not contact wobble)
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes are free (additive term)
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative hand-foreign gate
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a hook grasp, 100 % with a wrap
    MEET_RCV_FREE = 0.02      # [m] receiver cup may cross the spawn midline by 2 cm for free
    MEET_SRC_FREE = 0.05      # [m] source cup may cross the midline by 5 cm for free
    MEET_SCALE = 0.10         # [m] tanh scale of the crossing penalties
    # receiver-upright shape (success requires rcv tilt <= 20 deg = 0.349 rad)
    RCV_TILT_FREE = 0.12      # [rad] ~7 deg of receiver tilt is free (grasp/lift wobble, pose noise)
    RCV_TILT_SIGMA = 0.20     # [rad] gate: 15 deg -> 0.61, 20 deg -> 0.27, 30 deg -> 0.03
    RCV_PEN_SCALE = 0.25      # [rad] additive penalty: 10 deg -> 0.22, 20 deg -> 0.73, 46 deg -> ~1
    RCV_GATE_FLOOR = 0.3      # positioning terms (bring_together / aim) keep 30 % income while rcv is tilted
    # iter_07 phase-dependent receiver upright (operator requirement)
    CARRY_GRASP_LO = 0.015    # [m] grasped-cup carry ramp start (above the ~1 cm origin rise of a tipped cup)
    CARRY_GRASP_HI = 0.05     # [m] grasped-cup carry ramp end (= LIFT_GATE: pour stack starts here)
    CARRY_FREE_LO = 0.06      # [m] above this the cup is carried whatever the grasp flag says
    CARRY_FREE_HI = 0.10      # [m]
    RCV_UP_W = 1.0            # carried receiver: 20 deg -> -0.73 (same as before, but only once carried)
    RCV_UP_POUR_W = 1.0       # extra during pour / post-pour hold: 20 deg -> -1.46 total
    POUR_PHASE_TILT_LO = 0.8  # [rad] pour phase ramps in as the aimed source cup tilts 46 deg ...
    POUR_PHASE_TILT_HI = 1.6  # [rad] ... to 92 deg (beads start leaving ~1.9 rad)
    # post-pour hold
    POST_POUR_FRAC = 0.5      # hold terms ramp in as bead_in_target_frac goes 0 -> 0.5
    # command smoothness (raw actions, before the env EMA)
    PALM_RATE_BASE = 0.02     # per unit of sum-of-squared palm command change, before the grasp (exploration)
    PALM_RATE_HELD = 0.18     # extra once that hand holds its cup (suppresses 4-5 Hz command switching)
    HAND_RATE_W = 0.01        # finger commands: small (grasp closing/opening is legitimate)

    # ------------------------------------------------------------------ clearance factors (unchanged)
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ grasp flags / heights
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]

    # ------------------------------------------------------------------ carry phases (iter_07)
    # rcv_carry = 0 while the receiver cup is on the table: tilt from hand contact during approach and grasping is
    # neither penalised nor used to scale any income (last round a global upright penalty of -0.36/step at epoch 1
    # taught the policy never to touch the receiver cup; rcv_grasped stayed 0.000 for 208 epochs).
    rcv_carry = _carry_phase(g_rcv, h_rcv, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)
    src_carry = _carry_phase(g_src, h_src, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)

    # ------------------------------------------------------------------ receiver upright gate (phase-gated)
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    # upright-dependent scaling acts only in proportion to the carry phase; identical to the old gate once carried
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)                                  # (N,) in [0,1]
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up                 # (N,) in [0.3,1]

    # ------------------------------------------------------------------ stage 1: approach (unchanged)
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp (unchanged)
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    grasp_src = 1.0 * g_src + 0.5 * prox_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * prox_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)
    both_grasped_b = ctx.src_grasped & ctx.rcv_grasped
    both_grasped = 1.0 * both_grasped_b.to(dt)

    wq_src = _wrap_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                           ctx.src_palm_axes, ctx.src_cup_up, ctx.src_cup_pos, ctx.src_palm_pos)
    wq_rcv = _wrap_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                           ctx.rcv_palm_axes, ctx.rcv_cup_up, ctx.rcv_cup_pos, ctx.rcv_palm_pos)
    wrap_src = 1.5 * g_src * wq_src
    wrap_rcv = 0.5 * g_rcv * wq_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src   # in [0.3,1]; multiplies lift_src / aim / tilt / pour_pose

    # ------------------------------------------------------------------ stage 3: lift
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    lift_src = 2.0 * g_src * grip * lift_frac_src
    # receiver lift worth 50 % if carried tilted, 100 % upright; rcv_up is phase-gated, so the first centimetres
    # of lifting (and any contact tilt on the table) are paid in full
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv * (0.5 + 0.5 * rcv_up)

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    src_lifted_f = src_lifted_b.to(dt)
    both_lifted_f = both_lifted_b.to(dt)
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ pouring-lip geometry
    lip = _rim_low_point(ctx.src_cup_mouth_pos, ctx.src_cup_up, ctx.cup_radius)   # (N,3)
    lip_delta = lip - ctx.rcv_cup_mouth_pos                                         # (N,3)
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)                                   # (N,) horizontal miss
    dz_lip = lip_delta[:, 2]                                                        # (N,) lip height above rcv rim
    above = torch.clamp((dz_lip - DZ_LOW) / (DZ_HIGH - DZ_LOW), 0.0, 1.0)
    high = torch.exp(-(torch.clamp(dz_lip - DZ_FREE, min=0.0) / DZ_SIGMA) ** 2)
    aim_z = above * high
    aim_xy = torch.exp(-6.0 * torch.clamp(lip_xy - LIP_DEADBAND, min=0.0))
    aim_soft = aim_xy * aim_z                                                       # precise, in [0,1]
    aim_xy_loose = torch.exp(-3.0 * torch.clamp(lip_xy - LIP_DEADBAND_LOOSE, min=0.0))
    aim_loose = aim_xy_loose * aim_z

    # ------------------------------------------------------------------ stage 4: bring cups together
    # rcv_up_soft is phase-gated: a receiver cup bumped on the table no longer cuts this term; a CARRIED tilted
    # receiver still keeps only 30 % (no tilting the receiver to meet the lip)
    bring_together = 1.0 * src_lifted_f * not_nested_f * clean * torch.exp(-3.0 * lip_xy) * aim_z * rcv_up_soft
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    aim = 3.0 * stack_gate * grip * aim_soft * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt / pour pose
    # stack_gate needs the receiver grasped and > 5 cm up, where rcv_carry = 1 -> exactly as strict as before
    # (20 deg receiver tilt keeps 27 % of the pour-pose income, 30 deg ~0)
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_rad = ctx.src_cup_tilt
    tilt_frac = torch.clamp(tilt_rad / TILT_TARGET, 0.0, 1.0)
    tilt = 4.0 * stack_gate * grip * beads_left * aim_loose * tilt_frac * rcv_up

    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]                  # (N,2)
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    up_xy = ctx.src_cup_up[:, :2]                                                    # (N,2)
    up_xy_n = up_xy / (torch.norm(up_xy, dim=-1, keepdim=True) + 1e-6)
    dir_cos = torch.sum(to_rcv_n * up_xy_n, dim=-1)                                  # (N,) in [-1,1]
    tilt_amt = torch.clamp(tilt_rad / TILT_DIR_SCALE, 0.0, 1.0)
    tilt_dir = 1.0 * stack_gate * beads_left * tilt_amt * dir_cos * rcv_up

    pour_band = _smoothstep((tilt_rad - POUR_TILT_LO) / (POUR_TILT_HI - POUR_TILT_LO))
    pour_pose = 3.0 * stack_gate * grip * beads_left * aim_soft * pour_band * rcv_up

    tilt_premature = -1.0 * (1.0 - aim_soft) * torch.clamp(tilt_rad - TILT_FREE, min=0.0)

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (unchanged, ungated)
    # 50 * dfrac = 2.5 per bead; -30 * dfrac = -1.5 per spilled bead. Never gated: the bead signal must survive.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour (unchanged)
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)
    hold_src = 1.5 * post_pour * g_src * wq_src
    hold_rcv = 0.5 * post_pour * g_rcv * wq_rcv * rcv_up    # keep the filled receiver level and firmly held

    # ------------------------------------------------------------------ workspace: meet near the midline (unchanged)
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    rcv_cross = mid_y - ctx.rcv_cup_pos[:, 1]
    src_cross = ctx.src_cup_pos[:, 1] - mid_y
    meet_rcv = -1.0 * both_lifted_f * torch.tanh(torch.clamp(rcv_cross - MEET_RCV_FREE, min=0.0) / MEET_SCALE)
    meet_src = -0.5 * both_lifted_f * torch.tanh(torch.clamp(src_cross - MEET_SRC_FREE, min=0.0) / MEET_SCALE)

    # ------------------------------------------------------------------ collision / clearance
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)   # unchanged, ungated: real cup-cup contact
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    hand_foreign_src = -1.0 * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -1.0 * torch.tanh(f_rcv / FORCE_SCALE)
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos
    cup_dist = torch.norm(rel, dim=-1)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel
    v_close = torch.sum(v_rel * u, dim=-1)
    near_f = (cup_dist < NEAR_DIST).to(dt)
    # iter_07: only while at least one cup is carried. Cups spawn ~0.20 m apart (< NEAR_DIST), so cups knocked on
    # the table during reaching were charged (-0.28/step at epoch 1) — another "don't touch the cups" signal.
    carry_any = torch.maximum(src_carry, rcv_carry)
    closing_speed = -1.0 * carry_any * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints
    # iter_07 (OPERATOR REQUIREMENT): receiver-upright penalty is phase-dependent.
    #   on the table (approach / grasp):     0 — tilt from hand contact is free
    #   grasped and lifted (carry):          -1.0 * tanh(excess/0.25): 10 deg -> -0.22, 20 deg -> -0.73
    #   during the pour / post-pour hold:    doubled: 10 deg -> -0.44, 20 deg -> -1.46
    # pour_phase needs both cups lifted AND the lip roughly over the receiver mouth AND the source tilted, or beads
    # already in the receiver; weight 2.0 at the pour is below the pour-stack income (aim 3 + tilt 4 + pour_pose 3)
    # it protects, so keeping the receiver level is cheaper than giving up the pour.
    pour_phase = torch.maximum(
        both_lifted_f * aim_loose * _smoothstep((tilt_rad - POUR_PHASE_TILT_LO) / (POUR_PHASE_TILT_HI - POUR_PHASE_TILT_LO)),
        post_pour,
    )
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)
    # drop kept: a cup below its spawn height or a receiver lying on its side (> 69 deg) is a lost cup, not wobble
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # ------------------------------------------------------------------ command smoothness (unchanged)
    da = ctx.actions - ctx.prev_actions                                              # (N,18)
    palm_sq_src = torch.sum(da[:, 0:6] ** 2, dim=-1)
    palm_sq_rcv = torch.sum(da[:, 9:15] ** 2, dim=-1)
    hand_sq = torch.sum(da[:, 6:9] ** 2, dim=-1) + torch.sum(da[:, 15:18] ** 2, dim=-1)
    palm_rate_src = -(PALM_RATE_BASE + PALM_RATE_HELD * g_src) * palm_sq_src
    palm_rate_rcv = -(PALM_RATE_BASE + PALM_RATE_HELD * g_rcv) * palm_sq_rcv
    hand_rate = -HAND_RATE_W * hand_sq

    # ------------------------------------------------------------------ success bonus (unchanged)
    # env success includes rcv tilt <= 20 deg; scaled by source grip quality (0.6..1.0) for a firm post-pour hold
    success = 10.0 * ctx.success.to(dt) * clear * clean * (0.6 + 0.4 * g_src * wq_src)

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
        "tilt_dir": tilt_dir,
        "pour_pose": pour_pose,
        "tilt_premature": tilt_premature,
        "rot_speed": rot_speed,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
        "hold_src": hold_src,
        "hold_rcv": hold_rcv,
        "meet_rcv": meet_rcv,
        "meet_src": meet_src,
        "cup_contact": cup_contact,
        "hand_foreign_src": hand_foreign_src,
        "hand_foreign_rcv": hand_foreign_rcv,
        "closing_speed": closing_speed,
        "nested": nested,
        "upright_rcv": upright_rcv,
        "drop": drop,
        "palm_rate_src": palm_rate_src,
        "palm_rate_rcv": palm_rate_rcv,
        "hand_rate": hand_rate,
        "success": success,
    }

    reward = zeros
    for v in components.values():
        reward = reward + v

    return reward, components
```
