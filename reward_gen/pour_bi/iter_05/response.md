# iter_05 — reward revision

## 1. What the task means / stages

1. **Approach**: each palm goes to its own cup (right palm → source cup, left palm → receiver cup).
2. **Grasp**: a side-on wrap power grasp on each cup (thumb opposite the fingers, palm against the cup body).
3. **Lift**: both cups off the table. The source cup ends up clearly higher than the receiver.
4. **Bring together**: the source cup's pouring lip goes over the receiver's mouth, slightly above its rim. The two cups must not touch, no hand may touch anything but its own cup, and the cups must not be nested.
5. **Tilt / pour**: rotate the source cup *toward the receiver* past ~1.9 rad (110°) while the lip stays over the receiver mouth. The beads fall through the air into the receiver.
6. **Hold / finish**: at least half of the beads in the receiver, little spill, receiver upright, nothing dropped (env `success`).

## 2. Feedback analysis

**Solved stages (stable over the last 6-7 checkpoints)**
- Approach is flat at ~0.66-0.69 per arm, which is saturated.
- Grasp: `src_grasped` 0.88, `rcv_grasped` 0.87, `wrap_src` 1.15 (wq ≈ 0.87).
- The operator confirms that both hands form the intended envelope grasp.
- Lift: source 0.255 m and receiver 0.165 m (`lift_src` 1.58, `lift_rcv` 1.74). `both_lifted` is 0.86.
- Safety: `cup_collision_rate` ~1 %, hand-foreign rates ~2-3 %, nesting 0, drops ~0. The multiplicative `clear` / `clean` gates work, so keep them.

**Moving but plateauing**
- `aim` 0.07 → 1.91 and `bring_together` 0.10 → 0.73. `task/aim_dist` fell 0.43 → 0.125 m, but the last three checkpoints only moved 0.16 → 0.144 → 0.125.
- A 12 cm horizontal lip miss means `aim_xy = exp(-6·0.115) ≈ 0.50`, so the pour stack is paid at about half rate.

**Stuck stage: tilt**
- `src_tilt_deg` sits at 38-44° (max 47.7° in the whole run). Beads need ~110°.
- `bead/in_target` 0.0002, `pour_delta` 0, `success` 0.0002. This is the primary bottleneck.

**Why the previous function stalls the tilt**
- **Anti-tilt term.** `pour_rate = -0.3 · aim_soft · tanh(|ω|/1.5)` grows from -0.005 to -0.13 as aiming improves. It charges any cup rotation, and charges it more the better the policy is aimed. At the exact place where the cup should rotate, rotating is taxed.
- **Flat slope.** `tilt = 3 · gate · grip · beads_left · aim_soft · tilt/2.0` is linear with slope 1.5·gate·grip·aim_soft ≈ 0.5 reward per rad per step at aim_soft ≈ 0.5. Over the whole 47° → 110° transit that is only about +0.5/step. It is easily cancelled by the rotation tax, `tilt_premature` (charged beyond 0.8 rad whenever aim_soft < 1) and any aim loss caused by moving the wrist.
- **No direction signal.** `tilt` is direction-agnostic. Tilting away from the receiver pays the same as tilting toward it, and only the lip geometry notices the difference, through a 12-cm-wide exp. The 47° plateau may simply be tilted sideways or away from the receiver.
- **Nothing special near the release angle.** The tilt band 1.3-2.0 rad, where beads actually start to leave, earns nothing extra. The big bead rewards (`pour_delta` 2.5/bead, `success` 10/step) sit behind an exploration gap the policy never crosses.

**Measurement correction (operator note 2)**
- `hand_closure` saturates at 0.35-0.43 with contact freeze, whatever the grasp type.
- The old wrap-quality closure ramp (0.35 → 0.65) therefore gives a real wrap only ~0.2 on that ingredient. Its ceiling is structurally unreachable, which caps `grip` (and so the whole lift/aim/tilt stack) below 1 for no physical reason.
- Rescale the ramp to 0.25 → 0.40. It is only 10 % of wrap quality, so the grasp behaviour that already works stays untouched.

**Exploit check for the new tilt incentives**
- Tilting an unlifted cup, or tilting while touching the other hand or cup, stays blocked by `stack_gate`.
- Tilting with nothing to pour stays blocked by `beads_left`: spilled beads remove income and cost 1.5 each.
- A tilted lip far below the receiver rim stays blocked by `aim_z`.
- Nesting stays blocked by `not_nested`.
- Loosening the aim gate on the tilt-progress term is deliberate. Once beads actually flow, aim precision is judged by the bead terms themselves (`pour_delta` +2.5 vs `spill_delta` -1.5 per bead), which never fired before.

## 3. Changes

1. **Wrap quality closure ramp: 0.35-0.65 → 0.25-0.40.** A real wrap (0.35-0.43) now scores ~1 on that ingredient, so `grip` can reach 1.
2. **`tilt`: weight 3 → 4, gated by a looser aim.** It now uses `aim_loose = exp(-3·max(lip_xy-0.02, 0)) · aim_z` instead of `aim_soft`. At the current 12 cm miss the gate is ~0.74 instead of ~0.50, which roughly doubles the per-rad slope today.
3. **New `tilt_dir` (weight 1.0).** It rewards the horizontal part of the source cup's up axis pointing from the source cup toward the receiver mouth. It is signed (tilting away is charged), scaled by `clamp(tilt/1.0)` so it matters only when tilted, and gated by `stack_gate · beads_left`. It gives the wrist a direct gradient on the pour direction.
4. **New `pour_pose` (weight 3.0).** It adds a smoothstep over tilt 1.3 → 2.0 rad × precise `aim_soft` × `stack_gate · grip · beads_left`. This puts extra slope exactly in the release band and ties it to precise aim.
   - At the ideal pose, `tilt` + `pour_pose` ≈ 6/step, versus ≈ 1.2/step today at 47°.
   - That is a clear but not dominating gain on a total of ~12.8. Grasp and lift remain prerequisites through the gates.
5. **`pour_rate` → `rot_speed`.** No longer scaled by aim. Only rotation faster than 1.5 rad/s is charged (weight 0.3), so a controlled pour is free and a violent flip is still discouraged for real-robot safety.
6. **`tilt_premature`: free threshold 0.8 → 1.5 rad, weight 0.5 → 1.0.** Tilting while not aimed is harmless below the release angle (beads leave at ~1.9 rad), so the transit is no longer taxed. Tilting past 1.5 rad while not aimed is charged harder, because that is where spills happen.
7. **Unchanged.** Approach, grasp, lift, bring_together, aim (weight 3, precise), meet_* workspace terms, the clearance gates and penalties, bead increments (50 / -30), upright, drop, action_rate, and the success bonus (10 × clear × clean).

**Expected metric movement**
- `task/src_tilt_deg` rises well past 47° toward ~100-115°.
- `reward/tilt_dir` turns positive, and `reward/pour_pose` starts from ~0 and rises.
- `task/aim_dist` keeps falling below 0.10 m.
- `bead/in_target` and `pour_delta` become non-zero, then `success` rises.
- A transient rise in `bead/spill` is possible while aim is learned from bead outcomes.
- Grasp, lift and collision rates should stay roughly where they are.

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
    PALM_SCALE = 2.0     # [N] palm force tanh scale: 2 N -> 0.76
    LEVEL_SIGMA = 0.05   # [m] cup origin within ~5 cm of palm height along the cup axis
    # iter_05: closure saturates at 0.35-0.43 on this cup under contact freeze whatever the grasp type
    # (operator correction). Old ramp 0.35-0.65 capped a real wrap at ~0.2 on this ingredient; 0.25-0.40
    # lets a real wrap score ~1. It no longer tries to separate hook from wrap (forces/palm/side do that).
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
    LIP_DEADBAND_LOOSE = 0.02 # [m] deadband of the loose aim gate used by the tilt-progress term
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay (10 cm above -> 0.37)
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, saturate slightly above
    POUR_TILT_LO = 1.3        # [rad] start of the release band bonus (~75 deg)
    POUR_TILT_HI = 2.0        # [rad] end of the release band bonus (~115 deg)
    TILT_DIR_SCALE = 1.0      # [rad] direction term fully weighted once tilted >= 57 deg
    TILT_FREE = 1.5           # [rad] unaimed tilt below 86 deg spills nothing -> free (was 0.8)
    ROT_SPEED_FREE = 1.5      # [rad/s] controlled pouring rotation is free; only violent flips are charged
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes are free (additive term)
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative hand-foreign gate
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a hook grasp, 100 % with a wrap
    MEET_RCV_FREE = 0.02      # [m] receiver cup may cross the spawn midline by 2 cm for free
    MEET_SRC_FREE = 0.05      # [m] source cup may cross the midline by 5 cm for free
    MEET_SCALE = 0.10         # [m] tanh scale of the crossing penalties

    # ------------------------------------------------------------------ clearance factors (unchanged — worked)
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ stage 1: approach (unchanged — solved)
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp (unchanged — solved)
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
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
    wrap_src = 1.5 * g_src * wq_src   # weights unchanged; the wrap is achieved (operator note 1)
    wrap_rcv = 0.5 * g_rcv * wq_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src   # in [0.3,1]; multiplies lift_src / aim / tilt / pour_pose

    # ------------------------------------------------------------------ stage 3: lift (unchanged — solved)
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
    # loose aim (new): 12 cm miss -> 0.74 (precise gives 0.50). Used only by the tilt-progress term so the
    # wrist can start rotating before the lip is perfectly centred; beads then judge the precision.
    aim_xy_loose = torch.exp(-3.0 * torch.clamp(lip_xy - LIP_DEADBAND_LOOSE, min=0.0))
    aim_loose = aim_xy_loose * aim_z

    # ------------------------------------------------------------------ stage 4: bring cups together (unchanged)
    bring_together = 1.0 * src_lifted_f * not_nested_f * clean * torch.exp(-3.0 * lip_xy) * aim_z
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    aim = 3.0 * stack_gate * grip * aim_soft

    # ------------------------------------------------------------------ stage 5: tilt / pour pose (reworked)
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_rad = ctx.src_cup_tilt
    tilt_frac = torch.clamp(tilt_rad / TILT_TARGET, 0.0, 1.0)
    # weight 3 -> 4 and loose aim gate: slope per rad at today's pose roughly doubles (0.5 -> ~1.1 /step/rad),
    # still below aim (3.0, precise) so centring the lip is never traded for raw tilt.
    tilt = 4.0 * stack_gate * grip * beads_left * aim_loose * tilt_frac

    # direction (new): the cup's up axis, projected on the table plane, should point from the source cup toward
    # the receiver mouth — that is the side the lip must swing to. Signed: tilting away is charged.
    # Scaled by tilt amount so an upright cup (ill-defined horizontal axis) contributes nothing.
    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]                  # (N,2)
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    up_xy = ctx.src_cup_up[:, :2]                                                    # (N,2)
    up_xy_n = up_xy / (torch.norm(up_xy, dim=-1, keepdim=True) + 1e-6)
    dir_cos = torch.sum(to_rcv_n * up_xy_n, dim=-1)                                  # (N,) in [-1,1]
    tilt_amt = torch.clamp(tilt_rad / TILT_DIR_SCALE, 0.0, 1.0)
    # weight 1.0: small next to tilt/pour_pose, but a direct wrist-rotation gradient that the lip geometry
    # only provides through a 12-cm-wide exponential.
    tilt_dir = 1.0 * stack_gate * beads_left * tilt_amt * dir_cos

    # release-band bonus (new): extra slope exactly where beads start to leave (75-115 deg), tied to PRECISE aim.
    # weight 3.0: at the ideal pose tilt + pour_pose ~ 6/step vs ~1.2/step at today's 47 deg plateau.
    pour_band = _smoothstep((tilt_rad - POUR_TILT_LO) / (POUR_TILT_HI - POUR_TILT_LO))
    pour_pose = 3.0 * stack_gate * grip * beads_left * aim_soft * pour_band

    # unaimed tilt is harmless below the release angle -> free up to 1.5 rad; beyond, charge harder (1.0)
    tilt_premature = -1.0 * (1.0 - aim_soft) * torch.clamp(tilt_rad - TILT_FREE, min=0.0)

    # rotation speed (replaces pour_rate): no longer scaled by aim (it taxed rotation most where rotation is
    # needed). Only violent flips above 1.5 rad/s are charged — real-robot safety, weight 0.3.
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (unchanged, ungated)
    # 50 * dfrac = 2.5 per bead; -30 * dfrac = -1.5 per spilled bead. Never gated: the bead signal must survive.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ workspace: meet near the midline (unchanged)
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    rcv_cross = mid_y - ctx.rcv_cup_pos[:, 1]
    src_cross = ctx.src_cup_pos[:, 1] - mid_y
    meet_rcv = -1.0 * both_lifted_f * torch.tanh(torch.clamp(rcv_cross - MEET_RCV_FREE, min=0.0) / MEET_SCALE)
    meet_src = -0.5 * both_lifted_f * torch.tanh(torch.clamp(src_cross - MEET_SRC_FREE, min=0.0) / MEET_SCALE)

    # ------------------------------------------------------------------ collision / clearance (unchanged)
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
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
    closing_speed = -1.0 * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints (unchanged)
    upright_rcv = -0.3 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus (unchanged)
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
        "tilt_dir": tilt_dir,
        "pour_pose": pour_pose,
        "tilt_premature": tilt_premature,
        "rot_speed": rot_speed,
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
