import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (N,3),(N,3) -> (N,)
    return torch.norm(a - b, dim=-1)


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
    LIFT_TARGET_SRC = 0.25    # [m] source lift keeps its gradient up to 25 cm (hover needs h_src ~ h_rcv + 0.11..0.18)
    LIFT_TARGET_RCV = 0.10    # [m] receiver lifted moderately (operator requirement)
    RCV_LIFT_FREE = 0.15      # [m] receiver lift income is full up to 15 cm ...
    RCV_LIFT_SIGMA = 0.08     # [m] ... then decays (20 cm -> 0.68, 25 cm -> 0.21)
    LIFT_GATE = 0.05          # [m] minimum lift for the both_lifted flag
    # --- lip geometry (iter_13: everything is measured on ctx.src_pour_lip_pos, the point the env latch uses)
    LIP_DEADBAND = 0.02       # [m] lip within 2 cm (xy) of the receiver mouth centre counts as centred
    APPR_FULL = 0.08          # [m] iter_14: 0.10 -> 0.08, full zone now ends at the env latch radius (8.1 cm);
                              #     round 14 parked the lip at 9.9 cm, exactly on the old edge
    APPR_ZERO = 0.24          # [m] ... closed beyond 24 cm (lip xy at the spawn separation is ~26 cm -> 0)
    OVER_FULL = 0.035         # [m] lip inside the receiver mouth (inner radius 4.1 cm): release tilt fully paid
    OVER_ZERO = 0.07          # [m] ... nothing beyond 7 cm, 1.1 cm before the env latch radius 8.1 cm
    LIP_Z_LO = 0.0            # [m] lip at receiver rim height -> pour income 0 (it would hit the rim)
    LIP_Z_HI = 0.02           # [m] lip 2 cm above the rim -> 1
    LIP_Z_FREE = 0.07         # [m] lip up to 7 cm above the rim costs nothing ...
    LIP_Z_SIGMA = 0.08        # [m] ... then gentle decay (9 cm -> 0.94, 14 cm -> 0.47)
    # --- tilt-invariant carry height: dz_eq = source origin z + cup_mouth_z - receiver rim z. For an upright cup
    # this equals the old mouth-height signal; it does not fall by ~5 cm when the cup tilts.
    HOVER_LO = 0.06           # [m]
    HOVER_HI = 0.11           # [m] origin ~5 cm above the receiver rim: lip stays above the rim even at 100 deg
    HOVER_FREE = 0.18         # [m] iter_13: 0.16 -> 0.18, a little more bottom clearance is free
    HOVER_SIGMA = 0.06        # [m] Gaussian decay above that
    RAISE_LO = -0.04          # [m] wide ramp start
    RAISE_W = 1.0
    CONV_FAR = 0.40           # [m] linear lip-xy convergence: 0 at 40 cm, 1 inside the dead band (7.5/m at weight 3)
    CONV_W = 3.0
    BRING_W = 2.0             # iter_13: 3 -> 2; the lip target already pays more at the same pose (exp(-10 * 0.034))
    BT_K = 10.0               # [1/m]
    AIM_K = 15.0              # [1/m] precise lip term: 5.4 cm -> 0.60, 2.4 cm -> 0.94 (tilting in place raises it)
    MEET_RANGE = 0.30         # [m] per-arm meeting-point income, linear ramp
    MEET_RCV_W = 0.75
    MEET_SRC_W = 1.0
    # --- source tilt corridor (OPERATOR REQUIREMENT a + new env latch rule)
    SRC_TILT_FREE = 0.20      # [rad] 11.5 deg free far from the receiver (carry wobble, pose noise)
    PRE_MARGIN = 0.17         # [rad] pre-tilt stops 10 deg below ctx.premature_tilt_limit (measured wobble 9.3 deg)
    SRC_TILT_SCALE = 0.20     # [rad] tanh scale of the penalty on tilt beyond the corridor
    SRC_TILT_SIGMA = 0.25     # [rad] Gaussian income factor on tilt beyond the corridor
    SRC_UP_W_TABLE = 1.0      # penalty weight while grasped on the table
    SRC_UP_W_CARRY = 1.5      # extra once carried: total 2.5 exceeds anything tilting outside the corridor could earn
    LATCH_W = 0.5             # per-step penalty once the env's premature_tilt has latched
    # --- tilt income, two segments
    RELEASE_ABOVE_LIMIT = math.radians(20.0)  # env definition: limit = first-release angle - 20 deg
    TILT_TARGET_ABOVE = 0.45  # [rad] release tilt income saturates 26 deg past the first-release angle
    TILT_TARGET_CAP = 2.0     # [rad] never ask for more than 115 deg
    BAND_BELOW = 0.10         # [rad] pour_pose band starts 6 deg below release ...
    BAND_ABOVE = 0.40         # [rad] ... and saturates 23 deg above it
    PRE_W = 4.0               # iter_14: 3 -> 4; pre-tilt (0 -> limit - 10 deg): cannot spill, cannot latch
    PRE_NEAR_FLOOR = 0.25     # iter_14: pre-tilt keeps 25 % income with the lip far, full only over the mouth
    PRE_NEAR_K = 10.0         # [1/m] lip 10 cm -> 0.59, 5 cm -> 0.81, 2 cm -> 1.0
    TILT_W = 4.0              # release tilt (limit - 10 deg -> release + 26 deg) only with the lip over the mouth
    POSE_W = 3.0              # standing income of the full pour pose; 3 + 4 + 3 = 10 on top of ~20 carry income
    ROT_SPEED_FREE = 1.0      # [rad/s] controlled pouring rotation is free below this
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped
    RCV_TOPPLED = 1.2         # [rad] receiver lying on its side
    NEAR_DIST = 0.15          # [m] cup-centre distance inside which fast closing is charged
    CLOSE_V_FREE = 0.15       # [m/s]
    CLOSE_V_SCALE = 0.20      # [m/s]
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes are free
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative clean gate
    FOREIGN_W_FREE = 0.3      # hand-foreign weight while the hand does not hold its cup
    FOREIGN_W_HELD = 1.5      # iter_14: 1.0 -> 1.5 while held; contact is now charged additively (same cost
                              #     at any distance) instead of wiping out aim/bring income near the receiver
    CLEAN_FLOOR = 0.5         # iter_14: multiplicative clean gate keeps 50 % (was 0 %)
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a poor wrap
    # receiver-upright shape (success requires rcv tilt <= 20 deg)
    RCV_TILT_FREE = 0.12      # [rad] ~7 deg of receiver tilt is free
    RCV_TILT_SIGMA = 0.20     # [rad] gate: 15 deg -> 0.61, 20 deg -> 0.37
    RCV_PEN_SCALE = 0.25      # [rad] additive penalty scale
    RCV_GATE_FLOOR = 0.3      # positioning terms keep 30 % income under a tipped receiver
    # carry phases
    CARRY_GRASP_LO = 0.015    # [m]
    CARRY_GRASP_HI = 0.05     # [m]
    CARRY_FREE_LO = 0.06      # [m]
    CARRY_FREE_HI = 0.10      # [m]
    RCV_UP_W = 1.0            # carried receiver upright penalty
    RCV_UP_POUR_W = 1.0       # extra during pour / post-pour
    # receiver held still (OPERATOR REQUIREMENT b). Round 13 measured the receiver palm at 0.38 m/s (0.14 before),
    # so the weight comes back in steps: 0.25 far, +0.25 once the lip is in the approach zone, +0.5 while pouring
    # and after the pour (total 1.0). Only the receiver's own cup speed is charged, which costs nothing to reduce.
    RCV_STILL_FREE = 0.05     # [m/s]
    RCV_STILL_SCALE = 0.15    # [m/s] tanh scale
    RCV_STILL_W = 0.25
    RCV_STILL_NEAR_W = 0.25
    RCV_STILL_POUR_W = 0.50
    POST_POUR_FRAC = 0.8      # post-pour hold ramps in over 0 -> 0.8 of the beads
    SETTLE_FRAC = 0.05        # spill penalty ramps in over the first 5 % of the episode (bead settling)
    POUR_DELTA_W = 100.0      # per unit of transferred fraction: 12 beads -> 8.3 per bead, 6 beads -> 16.7
    SPILL_W = 50.0            # a spilled bead costs half of what a transferred bead earns
    REACH_FAR = 0.26          # [m]
    REACH_NEAR = 0.14         # [m]
    REACH_W = 0.5             # anchor only (solved stage)
    CLOSE_W = 1.0             # closing at the cup: 1.0 x exp(-6d) x normalised closure
    CLOSE_NORM = 0.35
    PALM_RATE_W = 0.01        # per unit of sum-of-squared palm command change (round 13: sampling noise, left as is)
    HAND_RATE_W = 0.005       # finger commands
    SAT_LO = 0.7              # palm commands pinned beyond |0.7| are taxed ...
    SAT_W = 0.15              # ... up to -0.15/step per arm (0.025 per pinned axis: negligible against the pour stack)

    # ------------------------------------------------------------------ clearance factors
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    # iter_14: soft version for the positioning / pour stack. With the hard gate every touch near the receiver
    # cost ~4/step of aim+bring income but nothing far away, which pushed the cups apart as the tilt grew.
    clean_soft = CLEAN_FLOOR + (1.0 - CLEAN_FLOOR) * clean
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
    # iter_13: the pour stack is gated by the carry phases, not by the boolean grasp flags, so a flickering
    # thumb contact while the hand rotates does not zero the whole stack
    carry_both = src_carry * rcv_carry              # (N,) in [0,1]

    # ------------------------------------------------------------------ receiver upright gate (phase-gated)
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)                                  # (N,) in [0,1]
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up                 # (N,) in [0.3,1]

    # ------------------------------------------------------------------ pouring-lip geometry (env lip point)
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos                        # (N,3)
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)                                   # (N,) horizontal miss of the lip
    lip_dz = lip_delta[:, 2]                                                        # (N,) lip height above the rcv rim
    lip_off = torch.clamp(lip_xy - LIP_DEADBAND, min=0.0)                           # (N,)
    approach_gate = 1.0 - _smoothstep((lip_xy - APPR_FULL) / (APPR_ZERO - APPR_FULL))   # (N,) wide zone
    over = 1.0 - _smoothstep((lip_xy - OVER_FULL) / (OVER_ZERO - OVER_FULL))            # (N,) lip over the mouth

    # ------------------------------------------------------------------ tilt angles
    tilt_mag = ctx.src_cup_tilt                                                     # (N,) any direction (latch uses this)
    limit = ctx.premature_tilt_limit.to(dt)                                         # (N,)
    release_tilt = limit + RELEASE_ABOVE_LIMIT                                      # (N,) first-release angle of this fill
    theta_pre = torch.clamp(limit - PRE_MARGIN, min=0.0)                            # (N,) safe pre-tilt angle
    tilt_target = torch.clamp(release_tilt + TILT_TARGET_ABOVE, max=TILT_TARGET_CAP)  # (N,)
    # toward-receiver tilt: env signal, backed by the same angle from fields of ctx (no folding above 90 deg);
    # never larger than the real tilt, never negative (leaning away or sideways earns nothing)
    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]                  # (N,2)
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    a_own = torch.sum(ctx.src_cup_up[:, :2] * to_rcv_n, dim=-1)                     # (N,) up . d
    theta_own = torch.atan2(a_own, ctx.src_cup_up[:, 2])                            # (N,)
    theta = torch.maximum(ctx.src_tilt_toward_rcv.to(dt), theta_own)                # (N,)
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), tilt_mag)                # (N,)

    # ------------------------------------------------------------------ source tilt corridor
    # allowed tilt: 0.20 rad far away -> theta_pre inside the approach zone -> unlimited with the lip over the mouth
    allowed = SRC_TILT_FREE + approach_gate * torch.clamp(theta_pre - SRC_TILT_FREE, min=0.0)
    tilt_over = torch.clamp(tilt_mag - allowed, min=0.0) * (1.0 - over)             # (N,) tilt beyond the corridor
    src_up = torch.exp(-(tilt_over / SRC_TILT_SIGMA) ** 2)                          # (N,) in [0,1]

    # ------------------------------------------------------------------ stage 1: approach + reach
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)
    reach_src = REACH_W * _smoothstep((REACH_FAR - d_src) / (REACH_FAR - REACH_NEAR))
    reach_rcv = REACH_W * _smoothstep((REACH_FAR - d_rcv) / (REACH_FAR - REACH_NEAR))

    # ------------------------------------------------------------------ stage 2: grasp
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    clos_src = torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
    clos_rcv = torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_src = 1.0 * g_src + CLOSE_W * prox_src * clos_src
    grasp_rcv = 1.0 * g_rcv + CLOSE_W * prox_rcv * clos_rcv
    both_grasped_b = ctx.src_grasped & ctx.rcv_grasped
    both_grasped = 1.0 * both_grasped_b.to(dt)

    wq_src = _wrap_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                           ctx.src_palm_axes, ctx.src_cup_up, ctx.src_cup_pos, ctx.src_palm_pos)
    wq_rcv = _wrap_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                           ctx.rcv_palm_axes, ctx.rcv_cup_up, ctx.rcv_cup_pos, ctx.rcv_palm_pos)
    wrap_src = 1.5 * g_src * wq_src     # the source grasp has to survive a 90 deg rotation: 3x the receiver weight
    wrap_rcv = 0.5 * g_rcv * wq_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src   # in [0.3,1]; multiplies lift_src / aim / pretilt / tilt / pour_pose

    # ------------------------------------------------------------------ stage 3: lift
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_src = 2.0 * g_src * grip * lift_frac_src * src_up
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv * rcv_band * (0.5 + 0.5 * rcv_up)

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    both_lifted = 1.0 * both_lifted_b.to(dt)

    # ------------------------------------------------------------------ heights
    dz_eq = ctx.src_cup_pos[:, 2] + ctx.cup_mouth_z - ctx.rcv_cup_mouth_pos[:, 2]  # (N,) tilt-invariant
    hover_decay = torch.exp(-(torch.clamp(dz_eq - HOVER_FREE, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _smoothstep((dz_eq - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    hover_wide = _smoothstep((dz_eq - RAISE_LO) / (HOVER_HI - RAISE_LO)) * hover_decay
    lip_above = torch.clamp((lip_dz - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)  # lip not below the rcv rim
    lip_high = torch.exp(-(torch.clamp(lip_dz - LIP_Z_FREE, min=0.0) / LIP_Z_SIGMA) ** 2)
    aim_z = lip_above * lip_high
    z_ok = torch.maximum(hover, over * aim_z)       # carry height band OR a good lip height over the mouth

    # ------------------------------------------------------------------ stage 4: raise, converge, bring together, aim
    raise_src = RAISE_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide
    # linear in LIP xy down to the 2 cm dead band. For an upright cup lip_xy = mouth_xy - 4.1 cm, so the target no
    # longer asks to stack the source body over the receiver; tilting in place also moves the lip inward and is paid.
    conv_xy = torch.clamp(1.0 - lip_off / CONV_FAR, 0.0, 1.0)
    converge = CONV_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide * conv_xy
    bring_together = BRING_W * src_carry * not_nested_f * clean_soft * src_up * torch.exp(-BT_K * lip_off) \
        * z_ok * rcv_up_soft
    stack_gate = carry_both * not_nested_f * clear * clean_soft
    aim_xy = torch.exp(-AIM_K * lip_off)
    aim_pose = aim_xy * z_ok
    aim = 3.0 * stack_gate * grip * aim_pose * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: pre-tilt, release tilt, pour pose
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    # segment 1: 0 -> theta_pre, open in the whole approach zone (gradient exists at the round-13 parking pose).
    # No contact gate here: the additive contact penalties still charge every touch.
    pre_frac = torch.clamp(theta_eff / (theta_pre + 1e-3), 0.0, 1.0)
    # iter_14: tilting pays most with the lip over the mouth, so tilting in place no longer competes with closing in
    near_pre = PRE_NEAR_FLOOR + (1.0 - PRE_NEAR_FLOOR) * torch.exp(-PRE_NEAR_K * lip_off)
    pretilt = PRE_W * carry_both * not_nested_f * grip * beads_left * approach_gate * hover_wide \
        * pre_frac * near_pre * rcv_up_soft
    # segment 2: theta_pre -> release + 26 deg, only with the lip over the receiver mouth and above its rim
    fin_den = torch.clamp(tilt_target - theta_pre, min=0.2)
    fin_frac = torch.clamp((theta_eff - theta_pre) / fin_den, 0.0, 1.0)
    tilt = TILT_W * stack_gate * grip * beads_left * over * lip_above * z_ok * fin_frac * rcv_up

    pour_band = _smoothstep((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = POSE_W * stack_gate * grip * beads_left * over * lip_above * aim_pose * pour_band * rcv_up

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (ungated income)
    # increments, never the level. Symmetric: a bead that bounces out again is charged, so in/out chatter nets 0.
    pour_delta = POUR_DELTA_W * torch.clamp(ctx.d_in_target, -1.0, 1.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_delta = -SPILL_W * settled * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)
    hold_src = 1.5 * post_pour * g_src * wq_src
    hold_rcv = 0.5 * post_pour * g_rcv * wq_rcv * rcv_up

    # ------------------------------------------------------------------ workspace: meet near the midline
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    mid_x = 0.5 * (ctx.src_cup_spawn_pos[:, 0] + ctx.rcv_cup_spawn_pos[:, 0])
    meet_xy = torch.stack([mid_x, mid_y], dim=-1)                                    # (N,2)
    # cup ORIGINS (same as the mouths for upright cups, no drift while the source tilts)
    d_meet_src = torch.norm(ctx.src_cup_pos[:, :2] - meet_xy, dim=-1)                # (N,)
    d_meet_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - meet_xy, dim=-1)                # (N,)
    meet_src = MEET_SRC_W * carry_both * not_nested_f * src_up * hover_wide \
        * torch.clamp(1.0 - d_meet_src / MEET_RANGE, 0.0, 1.0)
    meet_rcv = MEET_RCV_W * carry_both * not_nested_f * rcv_up_soft \
        * torch.clamp(1.0 - d_meet_rcv / MEET_RANGE, 0.0, 1.0)

    # ------------------------------------------------------------------ collision / clearance
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
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
    closing_speed = -1.0 * carry_any * near_f \
        * torch.tanh(torch.clamp(v_close - CLOSE_V_FREE, min=0.0) / CLOSE_V_SCALE)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints
    # (OPERATOR REQUIREMENT a): the source stays inside the tilt corridor: upright far away, at most
    # (limit - 10 deg) in the approach zone, free only with the lip over the receiver mouth. The penalty reaches
    # about -1.7 at the latch limit itself, so the policy is warned before the env latch can fire.
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src \
        * torch.tanh(tilt_over / SRC_TILT_SCALE)
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)

    fin_active = _smoothstep((theta_eff - theta_pre) / torch.clamp(release_tilt - theta_pre, min=0.1))
    pour_phase = torch.maximum(carry_both * over * fin_active, post_pour)           # (N,) in [0,1]
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)

    # (OPERATOR REQUIREMENT b): the carried receiver is held still
    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    still_w = RCV_STILL_W + RCV_STILL_NEAR_W * approach_gate * src_carry + RCV_STILL_POUR_W * pour_phase
    rcv_still = -still_w * rcv_carry \
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
    a_abs = torch.abs(ctx.actions)                                                   # (N,18)
    sat_src = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)    # (N,) in [0,1]
    sat_rcv = torch.mean(_smoothstep((a_abs[:, 9:15] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)   # (N,) in [0,1]
    palm_sat_src = -SAT_W * sat_src
    palm_sat_rcv = -SAT_W * sat_rcv

    # ------------------------------------------------------------------ success bonus
    # (OPERATOR REQUIREMENT c): 50 % of the beads -> 8.0/step, all -> 10/step, scaled by source grip.
    # 10/step on top of the 10/step pour stack doubles the carry income (~20/step): pouring dominates standing.
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
        "converge": converge,
        "bring_together": bring_together,
        "aim": aim,
        "pretilt": pretilt,
        "tilt": tilt,
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
