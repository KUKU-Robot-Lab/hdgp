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
    LIFT_TARGET_SRC = 0.25    # [m] source lift keeps its gradient up to 25 cm (hover needs h_src ~ h_rcv + 0.11..0.16)
    LIFT_TARGET_RCV = 0.10    # [m] receiver lifted moderately (operator requirement)
    RCV_LIFT_FREE = 0.15      # [m] receiver lift income is full up to 15 cm ...
    RCV_LIFT_SIGMA = 0.08     # [m] ... then decays (20 cm -> 0.68, 25 cm -> 0.21)
    LIFT_GATE = 0.05          # [m] minimum lift before the pour stack is rewarded
    LIP_DEADBAND = 0.01       # [m] lip within 1 cm of the receiver mouth centre counts as centred (pose noise)
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.08           # [m] beyond that, gentle decay (iter_12: 0.06 -> 0.08, closes the dip between hover and lip bands)
    # upright-carry hover band (source mouth above the receiver rim)
    HOVER_LO = 0.06           # [m] mouth 6 cm above the receiver rim -> 0
    HOVER_HI = 0.11           # [m] mouth 11 cm above the rim -> 1
    HOVER_FREE = 0.16         # [m] free up to 16 cm ...
    HOVER_SIGMA = 0.06        # [m] ... then Gaussian decay
    RAISE_LO = -0.04          # [m] relative-height ramp start (mouth 4 cm below the rim)
    RAISE_W = 1.0
    BRING_W = 3.0             # iter_12: 1.5 -> 3.0, the last 15 cm must out-earn the standing carry income
    # iter_11: dedicated xy convergence ramp (mouth-to-mouth). Round 11 parked at 0.29 m: bring_together's
    # exp(-3d) paid 0.42 there vs 0.86 at 5 cm (0.65/step for the whole gap on a 12/step income). This ramp
    # pays 0.72 at 0.29 m, 2.0 at 0.17 m, 2.9 at 0.10 m and 3.0 at 6 cm: ~2.3/step for closing the gap.
    CONV_FAR = 0.40           # [m]
    CONV_NEAR = 0.06          # [m]
    CONV_W = 3.0
    # iter_12: converge is LINEAR in mouth xy (constant 7.5/m all the way to 0, no flat end) and no longer
    # multiplied by the cup-contact gate (a brush zeroed 2.0/step of income: the policy learned to stay 0.2 m away;
    # the additive cup_contact penalty -1.0 still charges every contact)
    BT_K = 10.0               # [1/m] bring_together kernel exp(-10 d): 0.20 m -> 0.14, 0.10 -> 0.37, 0.05 -> 0.61
    # iter_12: aim opens on a WIDE mouth-xy gate (does not pay tilt, so it cannot cause a premature latch)
    AIM_FULL = 0.06           # [m]
    AIM_ZERO = 0.22           # [m]
    # iter_12: per-arm meeting-point income replaces the meet_* crossing taxes (each arm gets its own gradient,
    # independent of what the other arm does). Meeting point = xy midpoint of the two spawn positions.
    MEET_RANGE = 0.30         # [m] linear ramp 0 at 30 cm from the meeting point, 1 on it
    # near gate on MOUTH-TO-MOUTH xy distance (env premature-tilt latch is at 0.10 m)
    NEAR_FULL = 0.05          # [m] mouths within 5 cm (xy): tilt income fully open
    NEAR_ZERO = 0.09          # [m] beyond 9 cm: closed
    # source upright while far (OPERATOR REQUIREMENT a)
    SRC_TILT_FREE = 0.20      # [rad] 11.5 deg of source tilt is free (carry wobble, pose noise)
    SRC_TILT_SCALE = 0.20     # [rad] tanh scale of the penalty
    SRC_TILT_SIGMA = 0.25     # [rad] Gaussian income factor
    SRC_UP_W_TABLE = 1.0      # penalty weight while grasped on the table
    SRC_UP_W_CARRY = 1.5      # extra once carried: total 2.5 exceeds any far income tilting could touch
    LATCH_W = 0.5             # per-step penalty once the env's premature_tilt has latched
    # fill-adaptive release angle (measured: first bead ~72 deg full, ~90 deg with a few beads)
    RELEASE_FULL = 1.25       # [rad] release tilt for a full cup
    RELEASE_SPAN = 0.60       # [rad] + (1 - fill) * span
    BAND_BELOW = 0.10         # [rad] pour_pose band starts 6 deg below release ...
    BAND_ABOVE = 0.45         # [rad] ... and saturates 26 deg above it
    TILT_TARGET_ABOVE = 0.55  # [rad] linear tilt term saturates 32 deg past release
    TILT_DIR_SCALE = 1.0      # [rad] direction term fully weighted at 57 deg
    ROT_SPEED_FREE = 1.0      # [rad/s] controlled pouring rotation is free below this
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped
    RCV_TOPPLED = 1.2         # [rad] receiver lying on its side
    # iter_11: closing-speed guard moved inside 15 cm (was 30 cm: the policy parked exactly on that boundary) and
    # the free closing speed raised to 0.15 m/s; cup_contact still charges any actual collision
    NEAR_DIST = 0.15          # [m] cup-centre distance inside which fast closing is charged
    CLOSE_V_FREE = 0.15       # [m/s]
    CLOSE_V_SCALE = 0.20      # [m/s]
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes are free
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative clean gate
    FOREIGN_W_FREE = 0.3      # hand-foreign weight while the hand does not hold its cup
    FOREIGN_W_HELD = 1.0      # ... and while it holds / carries it
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a poor wrap
    # iter_11: meet bands widened (receiver 6 cm, source 10 cm across the midline) and lighter, so the hover xy
    # target is reachable without paying the crossing tax; weights 0.5 / 0.3 (were 1.0 / 0.5)
    MEET_RCV_FREE = 0.06      # [m]
    MEET_SRC_FREE = 0.10      # [m]
    MEET_SCALE = 0.10         # [m] tanh scale of the crossing excess
    MEET_RCV_W = 0.75         # iter_12: now POSITIVE per-arm income weights (see MEET_RANGE)
    MEET_SRC_W = 1.0
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
    POUR_PHASE_TILT_LO = 0.8  # [rad]
    POUR_PHASE_TILT_HI = 1.6  # [rad]
    # receiver held still (OPERATOR REQUIREMENT b). iter_11: the carry-far weight is halved (0.5 -> 0.25) because
    # -0.245/step was the largest penalty and it paid the left arm to freeze 0.29 m from the source; during the
    # pour / post-pour the total stays 1.0 (0.25 + 0.75)
    RCV_STILL_FREE = 0.05     # [m/s] iter_12: 0.03 -> 0.05 (measured carry jitter 0.14 m/s)
    RCV_STILL_SCALE = 0.15    # [m/s] tanh scale
    RCV_STILL_W = 0.25        # while carried far (0.10 m/s -> -0.11)
    RCV_STILL_POUR_W = 0.75   # extra while the source mouth is near or beads are in (0.10 m/s -> -0.44)
    POST_POUR_FRAC = 0.8      # post-pour hold ramps in over 0 -> 0.8 of the beads
    SETTLE_FRAC = 0.05        # spill penalty ramps in over the first 5 % of the episode (bead settling)
    # reach ramp across the 0.22 m closing radius: 0 at 26 cm, 1 at 14 cm. Saturated at 0.99 from epoch 0 in
    # round 11, so iter_11 halves it (anchor only, less standing income)
    REACH_FAR = 0.26          # [m]
    REACH_NEAR = 0.14         # [m]
    REACH_W = 0.5
    CLOSE_W = 1.0             # closing at the cup: 1.0 x exp(-6d) x normalised closure
    CLOSE_NORM = 0.35
    PALM_RATE_W = 0.01        # per unit of sum-of-squared palm command change
    HAND_RATE_W = 0.005       # finger commands
    SAT_LO = 0.7              # palm commands pinned beyond |0.7| are taxed ...
    SAT_W = 0.15              # ... up to -0.15/step per arm

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
    mouth_xy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)   # (N,)
    near = 1.0 - _smoothstep((mouth_xy - NEAR_FULL) / (NEAR_ZERO - NEAR_FULL))                     # (N,) in [0,1]
    far = 1.0 - near

    # ------------------------------------------------------------------ source upright while far
    src_tilt_excess = torch.clamp(ctx.src_cup_tilt - SRC_TILT_FREE, min=0.0)
    src_up_raw = torch.exp(-(src_tilt_excess / SRC_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    src_up = 1.0 - far * (1.0 - src_up_raw)                                        # (N,) in [0,1]

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
    wrap_src = 1.5 * g_src * wq_src
    wrap_rcv = 0.5 * g_rcv * wq_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src   # in [0.3,1]; multiplies lift_src / aim / tilt / pour_pose

    # ------------------------------------------------------------------ stage 3: lift
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
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

    # ------------------------------------------------------------------ stage 4: raise, converge, bring together
    dz_mouth = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    hover_decay = torch.exp(-(torch.clamp(dz_mouth - HOVER_FREE, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _smoothstep((dz_mouth - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    hover_wide = _smoothstep((dz_mouth - RAISE_LO) / (HOVER_HI - RAISE_LO)) * hover_decay
    raise_src = RAISE_W * both_lifted_f * not_nested_f * src_up * rcv_up_soft * hover_wide

    # iter_11: the missing gradient — xy convergence of the mouths from 40 cm down to 6 cm, weight 3.0, gated on
    # both cups lifted, no nesting, source upright (far), receiver upright-ish and the source mouth above the
    # receiver rim (hover_wide) so the only way to earn it is the correct carry geometry.
    conv_xy = torch.clamp(1.0 - mouth_xy / CONV_FAR, 0.0, 1.0)                     # iter_12: linear, 0 at 40 cm, 1 at 0
    converge = CONV_W * both_lifted_f * not_nested_f * src_up * rcv_up_soft * hover_wide * conv_xy

    # iter_11: xy factor sharpened exp(-3d) -> exp(-5d): 0.29 m -> 0.23, 0.10 m -> 0.61, 0.05 m -> 0.78
    bt_z = torch.maximum(hover, near * aim_z)
    bring_together = BRING_W * src_lifted_f * not_nested_f * clean * src_up * torch.exp(-BT_K * mouth_xy) \
        * bt_z * rcv_up_soft
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    # iter_12: an UPRIGHT source hovering 11-16 cm above the rim scored aim_z = 0.26, so entering the near zone
    # paid ~0.4 while rcv_still charged ~0.47 there. Height factor is now max(hover, lip band): the hover pose
    # is a full-value aim pose and tilting from it pays at once.
    near_wide = 1.0 - _smoothstep((mouth_xy - AIM_FULL) / (AIM_ZERO - AIM_FULL))
    aim_pose = aim_xy * bt_z
    aim = 3.0 * stack_gate * grip * near_wide * aim_pose * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt / pour pose (all x near)
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_rad = ctx.src_cup_tilt
    fill = torch.clamp(ctx.bead_fill_level, 0.0, 1.0)
    release_tilt = RELEASE_FULL + RELEASE_SPAN * (1.0 - fill)                        # (N,)
    tilt_target = release_tilt + TILT_TARGET_ABOVE                                   # (N,)
    tilt_frac = torch.clamp(tilt_rad / tilt_target, 0.0, 1.0)
    # weight 4.0 x near: worth nothing until the mouths are within 5 cm; aim_z gives the lip-height gradient
    tilt = 4.0 * stack_gate * grip * beads_left * near * bt_z * tilt_frac * rcv_up

    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]                  # (N,2)
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    up_xy = ctx.src_cup_up[:, :2]                                                    # (N,2)
    up_xy_n = up_xy / (torch.norm(up_xy, dim=-1, keepdim=True) + 1e-6)
    dir_cos = torch.sum(to_rcv_n * up_xy_n, dim=-1)                                  # (N,) in [-1,1]
    tilt_amt = torch.clamp(tilt_rad / TILT_DIR_SCALE, 0.0, 1.0)
    tilt_dir = 1.0 * stack_gate * beads_left * near * tilt_amt * dir_cos * rcv_up

    pour_band = _smoothstep((tilt_rad - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = 3.0 * stack_gate * grip * beads_left * near * aim_pose * pour_band * rcv_up

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (ungated income)
    # increments, never the level: 20 beads -> 2.5 per bead, 6 beads -> 8.3 per bead
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_delta = -30.0 * settled * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)
    hold_src = 1.5 * post_pour * g_src * wq_src
    hold_rcv = 0.5 * post_pour * g_rcv * wq_rcv * rcv_up

    # ------------------------------------------------------------------ workspace: meet near the midline
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    mid_x = 0.5 * (ctx.src_cup_spawn_pos[:, 0] + ctx.rcv_cup_spawn_pos[:, 0])
    meet_xy = torch.stack([mid_x, mid_y], dim=-1)                                    # (N,2)
    d_meet_src = torch.norm(ctx.src_cup_mouth_pos[:, :2] - meet_xy, dim=-1)          # (N,)
    d_meet_rcv = torch.norm(ctx.rcv_cup_mouth_pos[:, :2] - meet_xy, dim=-1)          # (N,)
    meet_src = MEET_SRC_W * both_lifted_f * not_nested_f * src_up * hover_wide \
        * torch.clamp(1.0 - d_meet_src / MEET_RANGE, 0.0, 1.0)
    meet_rcv = MEET_RCV_W * both_lifted_f * not_nested_f * rcv_up_soft \
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
    # (OPERATOR REQUIREMENT a): source stays upright until its mouth is near the receiver's mouth.
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src * far \
        * torch.tanh(src_tilt_excess / SRC_TILT_SCALE)
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)

    pour_phase = torch.maximum(
        both_lifted_f * near * bt_z
        * _smoothstep((tilt_rad - POUR_PHASE_TILT_LO) / (POUR_PHASE_TILT_HI - POUR_PHASE_TILT_LO)),
        post_pour,
    )
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)

    # (OPERATOR REQUIREMENT b): the carried receiver is held still — light while carrying far, full near / post-pour
    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    # iter_12: full weight only while actually pouring / after the pour (was: whenever near -> a -0.47 wall at 9 cm)
    still_phase = pour_phase
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
    a_abs = torch.abs(ctx.actions)                                                   # (N,18)
    sat_src = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)    # (N,) in [0,1]
    sat_rcv = torch.mean(_smoothstep((a_abs[:, 9:15] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)   # (N,) in [0,1]
    palm_sat_src = -SAT_W * sat_src
    palm_sat_rcv = -SAT_W * sat_rcv

    # ------------------------------------------------------------------ success bonus
    # (OPERATOR REQUIREMENT c): 50 % of the beads -> 8.0/step, all -> 10/step, scaled by source grip
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
