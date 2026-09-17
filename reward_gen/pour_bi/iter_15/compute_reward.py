import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (N,3),(N,3) -> (N,)
    return torch.norm(a - b, dim=-1)


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    # x already normalised; clamp to [0,1] then 3x^2 - 2x^3
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wrap_quality(finger_force: torch.Tensor, palm_force: torch.Tensor, closure: torch.Tensor,
                  palm_axes: torch.Tensor, cup_up: torch.Tensor, cup_pos: torch.Tensor,
                  palm_pos: torch.Tensor) -> torch.Tensor:
    # Power-grasp quality in [0,1]. Weighted MEAN so each ingredient keeps its own gradient.
    FINGER_ON = 0.3      # [N]
    PALM_SCALE = 3.0     # [N]
    LEVEL_SIGMA = 0.05   # [m]
    CLOS_LO, CLOS_HI = 0.25, 0.40
    dt = closure.dtype
    finger_frac = torch.mean((finger_force > FINGER_ON).to(dt), dim=-1)
    palm_c = torch.tanh(torch.clamp(palm_force, min=0.0) / PALM_SCALE)
    normal = palm_axes[:, 0:3]
    side = 1.0 - torch.clamp(torch.abs(torch.sum(normal * cup_up, dim=-1)), 0.0, 1.0)
    axial = torch.sum((cup_pos - palm_pos) * cup_up, dim=-1)
    level = torch.exp(-(axial / LEVEL_SIGMA) ** 2)
    clos = torch.clamp((closure - CLOS_LO) / (CLOS_HI - CLOS_LO), 0.0, 1.0)
    return 0.30 * finger_frac + 0.30 * palm_c + 0.20 * side + 0.10 * level + 0.10 * clos


def _carry_phase(grasped_f: torch.Tensor, h: torch.Tensor,
                 grasp_lo: float, grasp_hi: float, free_lo: float, free_hi: float) -> torch.Tensor:
    # (N,) in [0,1]: 0 on the table, 1 once grasped and lifted; far above the table the flag is not needed
    held_ramp = grasped_f * _smoothstep((h - grasp_lo) / (grasp_hi - grasp_lo))
    high_ramp = _smoothstep((h - free_lo) / (free_hi - free_lo))
    return torch.maximum(held_ramp, high_ramp)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET_SRC = 0.25
    LIFT_TARGET_RCV = 0.10
    RCV_LIFT_FREE = 0.15
    RCV_LIFT_SIGMA = 0.08
    LIP_DEADBAND = 0.02       # [m]
    APPR_FULL = 0.08          # [m] approach zone full up to the latch radius
    APPR_ZERO = 0.24
    # iter_15: one "safe" gate replaces `over` (3.5 -> 7 cm). Full within 4.5 cm: round 15 parked the lip at
    # 5.3 cm where `over` was 0.48 and tilt past theta_pre lost more to the corridor penalty than it earned.
    SAFE_FULL = 0.045         # [m]
    SAFE_ZERO = 0.07          # [m] 1.1 cm inside the env latch radius 8.1 cm
    LIP_Z_LO = 0.0
    LIP_Z_HI = 0.02
    LIP_Z_FREE = 0.07
    LIP_Z_SIGMA = 0.08
    HOVER_LO = 0.06
    HOVER_HI = 0.11
    HOVER_FREE = 0.18
    HOVER_SIGMA = 0.06
    RAISE_LO = -0.04
    RAISE_W = 1.0
    CONV_FAR = 0.40
    CONV_W = 3.0
    BRING_W = 2.0
    BT_K = 10.0
    AIM_K = 15.0
    MEET_RANGE = 0.30
    MEET_RCV_W = 0.75
    MEET_SRC_W = 1.0
    SRC_TILT_FREE = 0.20
    PRE_MARGIN = 0.17         # [rad] theta_pre = limit - 10 deg
    SRC_TILT_SCALE = 0.20
    SRC_TILT_SIGMA = 0.25
    SRC_UP_W_TABLE = 1.0
    SRC_UP_W_CARRY = 1.5
    LATCH_W = 0.5
    RELEASE_ABOVE_LIMIT = math.radians(20.0)
    TILT_TARGET_ABOVE = 0.45
    TILT_TARGET_CAP = 2.0
    BAND_BELOW = 0.10
    BAND_ABOVE = 0.40
    # iter_15: single continuous tilt income (was PRE_W 4 over 0 -> theta_pre + TILT_W 4 beyond it).
    # 10 x theta/tilt_target: same value at the round-15 pose (~2.2 at 32 deg), ~4.7 at theta_pre, 10 at the
    # release target. With pour_pose 3 the pour pose is worth +11 over the parked pose (vs ~20 carry income).
    TILT_W = 10.0
    PRE_NEAR_FLOOR = 0.25
    PRE_NEAR_K = 10.0
    POSE_W = 3.0
    ROT_SPEED_FREE = 1.0
    DROP_DEPTH = 0.03
    RCV_TOPPLED = 1.2
    NEAR_DIST = 0.15
    CLOSE_V_FREE = 0.15
    CLOSE_V_SCALE = 0.20
    FORCE_SCALE = 5.0
    FOREIGN_DEADBAND = 1.0
    CLEAN_DEADBAND = 0.5
    FOREIGN_W_FREE = 0.3
    FOREIGN_W_HELD = 1.5
    CLEAN_FLOOR = 0.5
    GRIP_FLOOR = 0.3
    RCV_TILT_FREE = 0.12
    RCV_TILT_SIGMA = 0.20
    RCV_PEN_SCALE = 0.25
    RCV_GATE_FLOOR = 0.3
    CARRY_GRASP_LO = 0.015
    CARRY_GRASP_HI = 0.05
    CARRY_FREE_LO = 0.06
    CARRY_FREE_HI = 0.10
    RCV_UP_W = 1.0
    RCV_UP_POUR_W = 1.0
    RCV_STILL_FREE = 0.05
    RCV_STILL_SCALE = 0.15
    RCV_STILL_W = 0.25
    RCV_STILL_NEAR_W = 0.25
    RCV_STILL_POUR_W = 0.50
    POST_POUR_FRAC = 0.8
    SETTLE_FRAC = 0.05
    POUR_DELTA_W = 100.0
    SPILL_W = 50.0
    REACH_FAR = 0.26
    REACH_NEAR = 0.14
    REACH_W = 0.5
    CLOSE_W = 1.0
    CLOSE_NORM = 0.35
    PALM_RATE_W = 0.01
    HAND_RATE_W = 0.005
    SAT_LO = 0.7
    SAT_W = 0.15

    # ------------------------------------------------------------------ clearance factors
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
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
    # iter_15: held state = grasp flag OR carried. Used for every grasp-dependent income, so a thumb contact
    # that flickers while the cup rotates no longer costs ~6/step (round 15: src_grasped 0.98 -> 0.76 as tilt rose)
    held_src = torch.maximum(g_src, src_carry)
    held_rcv = torch.maximum(g_rcv, rcv_carry)
    carry_both = src_carry * rcv_carry

    # ------------------------------------------------------------------ receiver upright gate
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up

    # ------------------------------------------------------------------ pouring-lip geometry
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    lip_off = torch.clamp(lip_xy - LIP_DEADBAND, min=0.0)
    approach_gate = 1.0 - _smoothstep((lip_xy - APPR_FULL) / (APPR_ZERO - APPR_FULL))
    safe = 1.0 - _smoothstep((lip_xy - SAFE_FULL) / (SAFE_ZERO - SAFE_FULL))   # (N,) lip close enough to release

    # ------------------------------------------------------------------ tilt angles
    tilt_mag = ctx.src_cup_tilt
    limit = ctx.premature_tilt_limit.to(dt)
    release_tilt = limit + RELEASE_ABOVE_LIMIT
    theta_pre = torch.clamp(limit - PRE_MARGIN, min=0.0)
    tilt_target = torch.clamp(release_tilt + TILT_TARGET_ABOVE, max=TILT_TARGET_CAP)
    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    a_own = torch.sum(ctx.src_cup_up[:, :2] * to_rcv_n, dim=-1)
    theta_own = torch.atan2(a_own, ctx.src_cup_up[:, 2])
    theta = torch.maximum(ctx.src_tilt_toward_rcv.to(dt), theta_own)
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), tilt_mag)

    # ------------------------------------------------------------------ source tilt corridor
    # allowed: 0.20 rad far away -> theta_pre in the approach zone -> unlimited once the lip is within 4.5 cm
    allowed = SRC_TILT_FREE + approach_gate * torch.clamp(theta_pre - SRC_TILT_FREE, min=0.0)
    tilt_over = torch.clamp(tilt_mag - allowed, min=0.0) * (1.0 - safe)
    src_up = torch.exp(-(tilt_over / SRC_TILT_SIGMA) ** 2)

    # ------------------------------------------------------------------ stage 1: approach + reach
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)
    reach_src = REACH_W * _smoothstep((REACH_FAR - d_src) / (REACH_FAR - REACH_NEAR))
    reach_rcv = REACH_W * _smoothstep((REACH_FAR - d_rcv) / (REACH_FAR - REACH_NEAR))

    # ------------------------------------------------------------------ stage 2: grasp (held state)
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    clos_src = torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
    clos_rcv = torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_src = 1.0 * held_src + CLOSE_W * prox_src * clos_src
    grasp_rcv = 1.0 * held_rcv + CLOSE_W * prox_rcv * clos_rcv
    both_grasped = 1.0 * held_src * held_rcv

    wq_src = _wrap_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                           ctx.src_palm_axes, ctx.src_cup_up, ctx.src_cup_pos, ctx.src_palm_pos)
    wq_rcv = _wrap_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                           ctx.rcv_palm_axes, ctx.rcv_cup_up, ctx.rcv_cup_pos, ctx.rcv_palm_pos)
    wrap_src = 1.5 * held_src * wq_src   # wq still falls if fingers slip, so wrap keeps its gradient
    wrap_rcv = 0.5 * held_rcv * wq_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src

    # ------------------------------------------------------------------ stage 3: lift
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_src = 2.0 * held_src * grip * lift_frac_src * src_up
    lift_rcv = 2.0 * held_rcv * lift_frac_rcv * rcv_band * (0.5 + 0.5 * rcv_up)
    both_lifted = 1.0 * carry_both

    # ------------------------------------------------------------------ heights
    dz_eq = ctx.src_cup_pos[:, 2] + ctx.cup_mouth_z - ctx.rcv_cup_mouth_pos[:, 2]
    hover_decay = torch.exp(-(torch.clamp(dz_eq - HOVER_FREE, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _smoothstep((dz_eq - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    hover_wide = _smoothstep((dz_eq - RAISE_LO) / (HOVER_HI - RAISE_LO)) * hover_decay
    lip_above = torch.clamp((lip_dz - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)
    lip_high = torch.exp(-(torch.clamp(lip_dz - LIP_Z_FREE, min=0.0) / LIP_Z_SIGMA) ** 2)
    aim_z = lip_above * lip_high
    z_ok = torch.maximum(hover, safe * aim_z)

    # ------------------------------------------------------------------ stage 4: raise, converge, bring, aim
    raise_src = RAISE_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide
    conv_xy = torch.clamp(1.0 - lip_off / CONV_FAR, 0.0, 1.0)
    converge = CONV_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide * conv_xy
    bring_together = BRING_W * src_carry * not_nested_f * clean_soft * src_up * torch.exp(-BT_K * lip_off) \
        * z_ok * rcv_up_soft
    stack_gate = carry_both * not_nested_f * clear * clean_soft
    aim_xy = torch.exp(-AIM_K * lip_off)
    aim_pose = aim_xy * z_ok
    aim = 3.0 * stack_gate * grip * aim_pose * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt (one continuous income) + pour pose
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    near_pre = PRE_NEAR_FLOOR + (1.0 - PRE_NEAR_FLOOR) * torch.exp(-PRE_NEAR_K * lip_off)
    tgt = torch.clamp(tilt_target, min=0.3)
    # part below theta_pre: cannot spill or latch, open in the whole approach zone
    part_pre = torch.minimum(theta_eff, theta_pre) / tgt * near_pre * approach_gate * hover_wide
    # part above theta_pre: only with the lip within the safe radius and above the receiver rim
    part_rel = torch.clamp(torch.minimum(theta_eff, tgt) - theta_pre, min=0.0) / tgt * safe * lip_above
    tilt = TILT_W * carry_both * not_nested_f * clean_soft * grip * beads_left * rcv_up_soft \
        * (part_pre + part_rel)

    pour_band = _smoothstep((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = POSE_W * stack_gate * grip * beads_left * safe * lip_above * aim_pose * pour_band * rcv_up

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (increments)
    pour_delta = POUR_DELTA_W * torch.clamp(ctx.d_in_target, -1.0, 1.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_delta = -SPILL_W * settled * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)
    hold_src = 1.5 * post_pour * held_src * wq_src
    hold_rcv = 0.5 * post_pour * held_rcv * wq_rcv * rcv_up

    # ------------------------------------------------------------------ workspace: meet near the midline
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    mid_x = 0.5 * (ctx.src_cup_spawn_pos[:, 0] + ctx.rcv_cup_spawn_pos[:, 0])
    meet_xy = torch.stack([mid_x, mid_y], dim=-1)
    d_meet_src = torch.norm(ctx.src_cup_pos[:, :2] - meet_xy, dim=-1)
    d_meet_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - meet_xy, dim=-1)
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
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src \
        * torch.tanh(tilt_over / SRC_TILT_SCALE)
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)

    fin_active = _smoothstep((theta_eff - theta_pre) / torch.clamp(release_tilt - theta_pre, min=0.1))
    pour_phase = torch.maximum(carry_both * safe * fin_active, post_pour)
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)

    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    still_w = RCV_STILL_W + RCV_STILL_NEAR_W * approach_gate * src_carry + RCV_STILL_POUR_W * pour_phase
    rcv_still = -still_w * rcv_carry \
        * torch.tanh(torch.clamp(v_rcv - RCV_STILL_FREE, min=0.0) / RCV_STILL_SCALE)

    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # ------------------------------------------------------------------ command smoothness + saturation
    da = ctx.actions - ctx.prev_actions
    palm_sq_src = torch.sum(da[:, 0:6] ** 2, dim=-1)
    palm_sq_rcv = torch.sum(da[:, 9:15] ** 2, dim=-1)
    hand_sq = torch.sum(da[:, 6:9] ** 2, dim=-1) + torch.sum(da[:, 15:18] ** 2, dim=-1)
    palm_rate_src = -PALM_RATE_W * palm_sq_src
    palm_rate_rcv = -PALM_RATE_W * palm_sq_rcv
    hand_rate = -HAND_RATE_W * hand_sq
    a_abs = torch.abs(ctx.actions)
    sat_src = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    sat_rcv = torch.mean(_smoothstep((a_abs[:, 9:15] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    palm_sat_src = -SAT_W * sat_src
    palm_sat_rcv = -SAT_W * sat_rcv

    # ------------------------------------------------------------------ success bonus
    success = 10.0 * ctx.success.to(dt) * clear * clean * (0.6 + 0.4 * held_src * wq_src) \
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
