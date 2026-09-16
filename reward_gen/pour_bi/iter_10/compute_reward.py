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
    # iter_09: the hover band needs h_src = h_rcv + 0.11..0.16; with the receiver in its 0.10-0.15 m band that is
    # 0.21-0.31 m, so the absolute source lift keeps its gradient up to 25 cm (was 20)
    LIFT_TARGET_SRC = 0.25    # [m]
    LIFT_TARGET_RCV = 0.10    # [m] receiver lifted moderately (operator requirement)
    RCV_LIFT_FREE = 0.15      # [m] iter_09: receiver lift income is full up to 15 cm ...
    RCV_LIFT_SIGMA = 0.08     # [m] ... then decays (20 cm -> 0.68, 22 cm -> 0.46, 25 cm -> 0.21); it sat at 22 cm
    LIFT_GATE = 0.05          # [m] minimum lift before the pour stack is rewarded
    LIP_DEADBAND = 0.01       # [m] lip within 1 cm of the receiver mouth centre counts as centred (pose noise)
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay (10 cm above -> 0.37)
    # upright-carry hover band (SOURCE MOUTH above the receiver rim). 11 cm keeps the origins > 9 cm apart
    # (no nesting flag) and the source bottom above the receiver rim; decays above 16 cm.
    HOVER_LO = 0.06           # [m] mouth 6 cm above the receiver rim -> 0 (an upright cup here would be nested)
    HOVER_HI = 0.11           # [m] mouth 11 cm above the rim -> 1
    HOVER_FREE = 0.16         # [m] free up to 16 cm ...
    HOVER_SIGMA = 0.06        # [m] ... then Gaussian decay (22 cm -> 0.37)
    # iter_09: relative-height ramp, NOT multiplied by the xy distance, so "source mouth above receiver rim" has a
    # gradient from 4 cm below the rim upward even while the cups are still 30 cm apart
    RAISE_LO = -0.04          # [m] ramp start (mouth 4 cm below the receiver rim -> 0); ends at HOVER_HI -> 1
    RAISE_W = 1.0             # weight: below bring_together (1.5) so converging still pays more than height alone
    BRING_W = 1.5             # iter_09: was 1.0 — the xy transport is the only road to the near-gated pour income
    # near gate on MOUTH-TO-MOUTH xy distance, the quantity the env latch uses (threshold 0.10 m)
    NEAR_FULL = 0.05          # [m] mouths within 5 cm (xy): tilting is fully allowed / fully paid
    NEAR_ZERO = 0.09          # [m] beyond 9 cm (1 cm inside the env latch): no tilt income, full upright penalty
    # source upright while far (OPERATOR REQUIREMENT a) — all gated on the cup being HELD
    SRC_TILT_FREE = 0.20      # [rad] 11.5 deg of source tilt is free (wrap-grasp / carry wobble, pose noise)
    SRC_TILT_SCALE = 0.20     # [rad] tanh scale of the penalty: 20 deg -> 0.63, 30 deg -> 0.92 of the weight
    SRC_TILT_SIGMA = 0.25     # [rad] Gaussian income factor: 20 deg -> 0.70, 30 deg -> 0.19, 45 deg -> 0
    SRC_UP_W_TABLE = 1.0      # penalty weight while grasped on the table (30 deg -> -0.92)
    SRC_UP_W_CARRY = 1.5      # extra once carried: total 2.5 (20 deg -> -1.6, 30 deg -> -2.3)
    LATCH_W = 0.5             # per-step penalty once the env's premature-tilt latch is set (GRASPED cups only)
    # fill-adaptive release angle (measured: first bead ~72 deg full, ~97 deg with 6 beads)
    RELEASE_FULL = 1.25       # [rad] release tilt for a full cup (72 deg)
    RELEASE_SPAN = 0.60       # [rad] + (1 - fill) * span: fill 0.3 -> 1.67 rad (96 deg)
    BAND_BELOW = 0.10         # [rad] pour_pose band starts 6 deg below the release tilt ...
    BAND_ABOVE = 0.45         # [rad] ... and saturates 26 deg above it
    TILT_TARGET_ABOVE = 0.55  # [rad] linear tilt term saturates 32 deg past the release tilt (empties the cup)
    TILT_DIR_SCALE = 1.0      # [rad] direction term fully weighted once tilted >= 57 deg
    ROT_SPEED_FREE = 1.0      # [rad/s] controlled pouring rotation is free; 30 mm beads need a slow pour
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side (a lost cup, not contact wobble)
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes are free (additive term)
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative hand-foreign gate
    # iter_10: the hand-foreign penalty is phase-weighted. At the first checkpoint the fumbling source hand paid
    # -0.088/step for touching the table and learnt to back away from its cup instead of grasping it; a hand that
    # is NOT holding its cup pays 30 %, a hand holding/carrying its cup (the real-robot hazard) pays 100 %.
    FOREIGN_W_FREE = 0.3
    FOREIGN_W_HELD = 1.0
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a hook grasp, 100 % with a wrap
    MEET_RCV_FREE = 0.02      # [m] receiver cup may cross the spawn midline by 2 cm for free
    MEET_SRC_FREE = 0.05      # [m] source cup may cross the midline by 5 cm for free
    MEET_SCALE = 0.10         # [m] tanh scale of the crossing penalties
    # receiver-upright shape (success requires rcv tilt <= 20 deg = 0.349 rad)
    RCV_TILT_FREE = 0.12      # [rad] ~7 deg of receiver tilt is free (grasp/lift wobble, pose noise)
    RCV_TILT_SIGMA = 0.20     # [rad] gate: 15 deg -> 0.61, 20 deg -> 0.27, 30 deg -> 0.03
    RCV_PEN_SCALE = 0.25      # [rad] additive penalty: 10 deg -> 0.22, 20 deg -> 0.73, 46 deg -> ~1
    RCV_GATE_FLOOR = 0.3      # positioning terms keep 30 % income while rcv is tilted
    # carry phases
    CARRY_GRASP_LO = 0.015    # [m] grasped-cup carry ramp start (above the ~1 cm origin rise of a tipped cup)
    CARRY_GRASP_HI = 0.05     # [m] grasped-cup carry ramp end (= LIFT_GATE: pour stack starts here)
    CARRY_FREE_LO = 0.06      # [m] above this the cup is carried whatever the grasp flag says
    CARRY_FREE_HI = 0.10      # [m]
    RCV_UP_W = 1.0            # carried receiver: 20 deg -> -0.73
    RCV_UP_POUR_W = 1.0       # extra during pour / post-pour hold: 20 deg -> -1.46 total
    POUR_PHASE_TILT_LO = 0.8  # [rad] pour phase ramps in as the aimed source cup tilts 46 deg ...
    POUR_PHASE_TILT_HI = 1.6  # [rad] ... to 92 deg
    # receiver held still (OPERATOR REQUIREMENT b)
    RCV_STILL_FREE = 0.03     # [m/s] receiver cup speed below 3 cm/s is free (a slow 10 cm lift costs nothing)
    RCV_STILL_SCALE = 0.15    # [m/s] tanh scale: 0.10 m/s -> 0.44 of the weight, 0.25 m/s -> 0.9
    RCV_STILL_W = 0.5         # while carried (0.10 m/s -> -0.22)
    RCV_STILL_POUR_W = 0.5    # extra while the source mouth is near or beads are already in: total 1.0
    # post-pour hold ramps in over 0 -> 0.8 of the beads: emptying the cup keeps paying
    POST_POUR_FRAC = 0.8
    # bead spawn/settling drops 0.6-6 % of the beads in the first moments of every episode whatever the policy
    # does; the spill penalty ramps in over the first 5 % of the episode so that floor is not charged
    SETTLE_FRAC = 0.05
    # iter_10: reach ramp across the 0.22 m closing radius. Round 9 parked the source palm at 0.21-0.25 m
    # (approach_src still paid 0.44 there vs 0.63 grasped) so the hand could never close; the ramp pays 0 at
    # 26 cm and 1.0 at 14 cm (a wrap grasp sits at ~0.115 m): from the 0.165 m start pose it is worth 0.89
    # and backing off to 0.25 m loses ~0.87/step — far more than any pre-grasp contact penalty.
    REACH_FAR = 0.26          # [m]
    REACH_NEAR = 0.14         # [m]
    REACH_W = 1.0
    # iter_10: closing at the cup is paid 1.0 x exp(-6d) x closure/0.35 (0.5 at the grasp distance; was
    # 0.5 x exp(-6d) x raw closure ~ 0.1). Closure is normalised by its contact-freeze saturation (0.35-0.43).
    CLOSE_W = 1.0
    CLOSE_NORM = 0.35
    # iter_10: command smoothness on RAW actions is a tax on exploration noise and paid the policy to pin the
    # palm commands at +-1 (noise clipped -> zero rate): palm_rate_rcv -0.53 -> -0.12 by saturating, and once the
    # source was held the 0.18 extra cost -0.31/step, more than lift_src (+0.29) paid. The env EMA already removes
    # command chatter physically, so only a small flat rate remains (0.02 + 0.18 held -> 0.01 flat).
    PALM_RATE_W = 0.01        # per unit of sum-of-squared palm command change (both arms, all phases)
    HAND_RATE_W = 0.005       # finger commands: grasp closing/opening is legitimate (was 0.01)
    # iter_10: saturation penalty — palm commands beyond |0.7| ramp to a penalty (all six pinned at +-1 ->
    # -0.15/step per arm). Pinned commands drove the arm to its workspace limit (target-actual error 0.10 m,
    # 144 N cup-cup peaks) and the source palm out of the closing radius. A legitimate 25 cm lift needs far less.
    SAT_LO = 0.7
    SAT_W = 0.15

    # ------------------------------------------------------------------ clearance factors
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ grasp flags / heights
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]

    # ------------------------------------------------------------------ carry phases
    rcv_carry = _carry_phase(g_rcv, h_rcv, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)
    src_carry = _carry_phase(g_src, h_src, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)
    held_src = torch.maximum(g_src, src_carry)      # (N,) source cup held (grasp flag or carried)
    held_rcv = torch.maximum(g_rcv, rcv_carry)      # (N,) receiver cup held

    # ------------------------------------------------------------------ receiver upright gate (phase-gated)
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)                                  # (N,) in [0,1]
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up                 # (N,) in [0.3,1]

    # ------------------------------------------------------------------ near gate (mouth-to-mouth xy)
    # The env latches premature_tilt when the GRASPED source exceeds 30 deg while the mouths are > 0.10 m apart
    # (xy). near = 1 within 5 cm, 0 beyond 9 cm. It multiplies EVERY tilt income and releases the upright penalty.
    mouth_xy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)   # (N,)
    near = 1.0 - _smoothstep((mouth_xy - NEAR_FULL) / (NEAR_ZERO - NEAR_FULL))                     # (N,) in [0,1]
    far = 1.0 - near

    # ------------------------------------------------------------------ source upright while far
    # src_up scales the lift / carry income; upright_src is the additive penalty. Both are gated on the cup being
    # HELD (grasp flag or carried) so a random policy that knocks the source cup over is not taught to avoid the cup.
    src_tilt_excess = torch.clamp(ctx.src_cup_tilt - SRC_TILT_FREE, min=0.0)
    src_up_raw = torch.exp(-(src_tilt_excess / SRC_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    src_up = 1.0 - far * (1.0 - src_up_raw)                                        # (N,) in [0,1]

    # ------------------------------------------------------------------ stage 1: approach + reach
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)
    # iter_10: steep ramp across the closing radius (0 at 26 cm, 1 at 14 cm) — the anchor that keeps the palm
    # inside the 0.22 m radius where the hand is allowed to close
    reach_src = REACH_W * _smoothstep((REACH_FAR - d_src) / (REACH_FAR - REACH_NEAR))
    reach_rcv = REACH_W * _smoothstep((REACH_FAR - d_rcv) / (REACH_FAR - REACH_NEAR))

    # ------------------------------------------------------------------ stage 2: grasp
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    clos_src = torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)   # wrapped hand under contact freeze -> 1
    clos_rcv = torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_src = 1.0 * g_src + CLOSE_W * prox_src * clos_src
    grasp_rcv = 1.0 * g_rcv + CLOSE_W * prox_rcv * clos_rcv
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
    # receiver income band — full up to 15 cm, decaying above (income shaping, never negative)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    # the source lift is paid in full only while the cup is upright (far); 30 deg far -> 19 %
    lift_src = 2.0 * g_src * grip * lift_frac_src * src_up
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv * rcv_band * (0.5 + 0.5 * rcv_up)

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

    # ------------------------------------------------------------------ stage 4: raise, then bring cups together
    # Upright carry is steered to a hover: source mouth 11-16 cm above the receiver rim with the mouths converging
    # in xy; once near, the lip-based aim_z takes over so lowering the lip during the pour does not lose income.
    dz_mouth = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    hover_decay = torch.exp(-(torch.clamp(dz_mouth - HOVER_FREE, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _smoothstep((dz_mouth - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    # relative height paid on its own (no xy factor): lowering the receiver (free above its 0.10 m target) or
    # raising the source both climb this ramp
    hover_wide = _smoothstep((dz_mouth - RAISE_LO) / (HOVER_HI - RAISE_LO)) * hover_decay
    raise_src = RAISE_W * both_lifted_f * not_nested_f * src_up * rcv_up_soft * hover_wide
    bt_z = torch.maximum(hover, near * aim_z)
    bring_together = BRING_W * src_lifted_f * not_nested_f * clean * src_up * torch.exp(-3.0 * mouth_xy) \
        * bt_z * rcv_up_soft
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    # aim is gated by near as well: no income for hovering the lip somewhere beside the receiver
    aim = 3.0 * stack_gate * grip * near * aim_soft * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt / pour pose (all x near)
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_rad = ctx.src_cup_tilt
    # fill-adaptive release angle: full -> 1.25 rad (72 deg), fill 0.3 (6 beads) -> 1.67 rad (96 deg)
    fill = torch.clamp(ctx.bead_fill_level, 0.0, 1.0)
    release_tilt = RELEASE_FULL + RELEASE_SPAN * (1.0 - fill)                        # (N,)
    tilt_target = release_tilt + TILT_TARGET_ABOVE                                   # (N,) full 1.80, 6 beads 2.22
    tilt_frac = torch.clamp(tilt_rad / tilt_target, 0.0, 1.0)
    # weight 4.0 x near: worth nothing until the mouths are within 5 cm; aim_z (lip height) gives the gradient
    # from the upright hover (lip ~11 cm above the rim -> 0.26) down to the pouring lip at the rim (1.0)
    tilt = 4.0 * stack_gate * grip * beads_left * near * aim_z * tilt_frac * rcv_up

    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]                  # (N,2)
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    up_xy = ctx.src_cup_up[:, :2]                                                    # (N,2)
    up_xy_n = up_xy / (torch.norm(up_xy, dim=-1, keepdim=True) + 1e-6)
    dir_cos = torch.sum(to_rcv_n * up_xy_n, dim=-1)                                  # (N,) in [-1,1]
    tilt_amt = torch.clamp(tilt_rad / TILT_DIR_SCALE, 0.0, 1.0)
    tilt_dir = 1.0 * stack_gate * beads_left * near * tilt_amt * dir_cos * rcv_up

    pour_band = _smoothstep((tilt_rad - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = 3.0 * stack_gate * grip * beads_left * near * aim_soft * pour_band * rcv_up

    # iter_10: gated on the source being HELD — a cup knocked over by an exploring hand spins fast and used to
    # charge -0.134/step at the first checkpoint, another "avoid the cup" signal; a held cup's rotation is the pour
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (ungated income)
    # fractions are per-episode: 20 beads -> 2.5 per bead, 6 beads -> 8.3 per bead. Never gated.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    # spill charged only once the beads have settled (first 5 % of the episode ramps 0 -> 1)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_delta = -30.0 * settled * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)   # 50 % -> 0.68, 75 % -> 0.99
    hold_src = 1.5 * post_pour * g_src * wq_src
    hold_rcv = 0.5 * post_pour * g_rcv * wq_rcv * rcv_up    # keep the filled receiver level and firmly held

    # ------------------------------------------------------------------ workspace: meet near the midline
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    rcv_cross = mid_y - ctx.rcv_cup_pos[:, 1]
    src_cross = ctx.src_cup_pos[:, 1] - mid_y
    meet_rcv = -1.0 * both_lifted_f * torch.tanh(torch.clamp(rcv_cross - MEET_RCV_FREE, min=0.0) / MEET_SCALE)
    meet_src = -0.5 * both_lifted_f * torch.tanh(torch.clamp(src_cross - MEET_SRC_FREE, min=0.0) / MEET_SCALE)

    # ------------------------------------------------------------------ collision / clearance
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    # iter_10: 0.3 while the hand is not holding its cup (table brushes during exploration), 1.0 while it holds it
    w_for_src = FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src
    w_for_rcv = FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv
    hand_foreign_src = -w_for_src * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -w_for_rcv * torch.tanh(f_rcv / FORCE_SCALE)
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos
    cup_dist = torch.norm(rel, dim=-1)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel
    v_close = torch.sum(v_rel * u, dim=-1)
    near_f = (cup_dist < NEAR_DIST).to(dt)
    carry_any = torch.maximum(src_carry, rcv_carry)
    closing_speed = -1.0 * carry_any * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints
    # (OPERATOR REQUIREMENT a): source stays upright until its mouth is near the receiver's mouth.
    #   held on the table, far:   -1.0 * tanh(excess/0.2): 20 deg -> -0.63, 30 deg -> -0.92
    #   carried, far:             -2.5 * tanh(excess/0.2): 20 deg -> -1.6,  30 deg -> -2.3, 45 deg -> -2.5
    #   near (mouths <= 5 cm):    0 — the pour is free, and it is the only place any tilt income exists
    # 2.5 exceeds the largest income tilting far could still touch (lift_src 2.0, itself scaled down by src_up).
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src * far \
        * torch.tanh(src_tilt_excess / SRC_TILT_SCALE)
    # The env latch fires only on a GRASPED cup; a constant -0.5/step from the latch moment makes the value drop at
    # the violating step, while the held-cup income (grasp 1.0 + wrap <= 1.5 + lift <= 2.0) still exceeds it.
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)

    # receiver-upright penalty is phase-dependent; pour_phase uses the near gate
    pour_phase = torch.maximum(
        both_lifted_f * near * aim_z
        * _smoothstep((tilt_rad - POUR_PHASE_TILT_LO) / (POUR_PHASE_TILT_HI - POUR_PHASE_TILT_LO)),
        post_pour,
    )
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)

    # (OPERATOR REQUIREMENT b): the carried receiver is held still — 0.5 while carried (a slow lift is inside the
    # 3 cm/s deadband), 1.0 while the source mouth is near or beads are already in (pour / hold):
    # 0.10 m/s -> -0.22 carried / -0.44 during the pour, 0.25 m/s -> -0.45 / -0.9
    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    still_phase = torch.maximum(near, post_pour)
    rcv_still = -(RCV_STILL_W + RCV_STILL_POUR_W * still_phase) * rcv_carry \
        * torch.tanh(torch.clamp(v_rcv - RCV_STILL_FREE, min=0.0) / RCV_STILL_SCALE)

    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # ------------------------------------------------------------------ command smoothness + saturation
    da = ctx.actions - ctx.prev_actions                                              # (N,18)
    palm_sq_src = torch.sum(da[:, 0:6] ** 2, dim=-1)
    palm_sq_rcv = torch.sum(da[:, 9:15] ** 2, dim=-1)
    hand_sq = torch.sum(da[:, 6:9] ** 2, dim=-1) + torch.sum(da[:, 15:18] ** 2, dim=-1)
    palm_rate_src = -PALM_RATE_W * palm_sq_src
    palm_rate_rcv = -PALM_RATE_W * palm_sq_rcv
    hand_rate = -HAND_RATE_W * hand_sq
    # iter_10: palm commands pinned at +-1 (mean of the six per arm, ramp from |0.7| to |1.0|)
    a_abs = torch.abs(ctx.actions)                                                   # (N,18)
    sat_src = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)    # (N,) in [0,1]
    sat_rcv = torch.mean(_smoothstep((a_abs[:, 9:15] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)   # (N,) in [0,1]
    palm_sat_src = -SAT_W * sat_src
    palm_sat_rcv = -SAT_W * sat_rcv

    # ------------------------------------------------------------------ success bonus
    # env success requires no premature tilt, no nesting, receiver <= 20 deg. Scaled by source grip (0.6..1.0) and
    # by the transferred fraction (0.6..1.0; OPERATOR REQUIREMENT c): 50 % of the beads -> 8.0/step, all -> 10/step
    success = 10.0 * ctx.success.to(dt) * clear * clean * (0.6 + 0.4 * g_src * wq_src) \
        * (0.6 + 0.4 * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "reach_src": reach_src,
        "reach_rcv": reach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "wrap_src": wrap_src,
        "wrap_rcv": wrap_rcv,
        "both_grasped": both_grasped,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "both_lifted": both_lifted,
        "raise_src": raise_src,
        "bring_together": bring_together,
        "aim": aim,
        "tilt": tilt,
        "tilt_dir": tilt_dir,
        "pour_pose": pour_pose,
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
        "upright_src": upright_src,
        "premature_latch": premature_latch,
        "upright_rcv": upright_rcv,
        "rcv_still": rcv_still,
        "drop": drop,
        "palm_rate_src": palm_rate_src,
        "palm_rate_rcv": palm_rate_rcv,
        "palm_sat_src": palm_sat_src,
        "palm_sat_rcv": palm_sat_rcv,
        "hand_rate": hand_rate,
        "success": success,
    }

    reward = zeros
    for v in components.values():
        reward = reward + v

    return reward, components
