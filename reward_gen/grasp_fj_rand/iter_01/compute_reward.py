import torch
import math


# Movable finger joints (range > 0.05 rad), grouped per digit in hand-joint order:
# thumb (3,4), index (2,3,4), middle (2,3,4), ring (2,3,4), pinky (3,4).
_DIGIT_JOINTS = ([1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18])
_MOVABLE = [j for group in _DIGIT_JOINTS for j in group]


def _sat(x: torch.Tensor, full: float) -> torch.Tensor:
    # 0 at/below 0, 1 at/above `full`; the clamp also bounds physics force spikes
    return torch.clamp(x / full, 0.0, 1.0)


def _win(x: torch.Tensor, lo, hi, soft: float) -> torch.Tensor:
    # 1 inside [lo, hi], smooth gaussian decay outside. lo/hi may be tensors (per-cup windows).
    d = torch.relu(lo - x) + torch.relu(x - hi)
    return torch.exp(-(d / soft) ** 2)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Global scale unchanged. Raw per-step ordering:
    # stage 1 <= 10 <= stage-2 entry ~11-19 < full envelope on the table <= ~51 (decaying)
    # ~ envelope latched, still on the table ~32*hold_q (decaying to 40 %)
    # << same grasp 3 cm up ~+25 (no decay) << carried to the goal and held still ~240, +300 per success.
    scale = 0.05

    r = ctx.cup_radius                 # (N,)
    hh = ctx.cup_half_height           # (N,)
    n = ctx.palm_normal
    fd = ctx.palm_finger_dir
    axis = ctx.cup_axis

    # ---------------- stage masks (strict order, flags are latched by the env) ----------------
    approach = ctx.approach_done.bool()
    envelope = ctx.envelope_done.bool() & approach
    s1 = (~approach).float()
    s2 = (approach & ~envelope).float()
    s3 = envelope.float()
    progress = torch.clamp(ctx.episode_progress, 0.0, 1.0)

    # ---------------- palm pose relative to the cup (the quantities approach_done checks) -----
    v = ctx.cup_pos - ctx.palm_pos
    v_ax = (v * axis).sum(-1)
    v_perp = v - v_ax.unsqueeze(-1) * axis
    gap_n = (v_perp * n).sum(-1) - r    # palm plane -> cup surface; NEGATIVE = palm pressed in
    gap_f = (v_perp * fd).sum(-1) - r   # cup axis ahead of the palm along the fingers, minus r
    h_palm = -v_ax                      # palm height above the cup centre along the cup axis

    # start orientation kept (palm_normal +y, palm_finger_dir +x); ~0.46 at 20 deg on both axes
    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))
    ang_f = torch.acos(torch.clamp(fd[:, 0], -1.0, 1.0))
    ori = torch.exp(-(ang_n ** 2 + ang_f ** 2) / (2.0 * 0.4 ** 2))

    # open default hand pose kept (the env needs every movable joint within 0.3 of default)
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    dev_max = dev.amax(-1)
    open_pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev_max / 0.2) ** 2)
    keep = ori * open_pose

    # arm calmness: 1 when the arm is still, ~0.14 at 1 rad/s on one joint
    arm_speed2 = (ctx.arm_qd ** 2).sum(-1)
    arm_calm = torch.exp(-arm_speed2 / 0.5)

    # ---------------- measured contact --------------------------------------------------------
    touch = _sat(ctx.link_cup_force, 0.5)         # (N,5,3)
    digit_touch = touch.amax(-1)                  # (N,5)
    thumb_touch = digit_touch[:, 0]
    # sensitive version for the env's "no digit may touch" approach condition (env threshold 0.1 N)
    any_touch = _sat(ctx.link_cup_force, 0.2).amax(-1).amax(-1)      # (N,)
    palm_touch = _sat(ctx.palm_cup_force, 0.5)    # (N,)
    palm_firm = _sat(ctx.palm_cup_force, 2.0)     # rewards actually pressing, not grazing

    # ---------------- table clearance (unchanged: met last round) -----------------------------
    clear = ctx.hand_z_min - ctx.table_z
    near_tab = torch.clamp((0.025 - clear) / 0.025, 0.0, 1.0)   # 0 at 2.5 cm, 1 at contact
    deep_tab = torch.clamp((0.008 - clear) / 0.013, 0.0, 1.0)   # 0 at 8 mm, 1 at the -5 mm cutoff
    table_pen = -(8.0 * near_tab ** 2 + 30.0 * deep_tab ** 2)
    pc = ctx.palm_clearance
    palm_low = torch.clamp((0.008 - pc) / 0.008, 0.0, 1.0)
    palm_dig = torch.clamp(-pc / 0.010, 0.0, 1.0)
    palm_table_pen = -(8.0 * palm_low ** 2 + 40.0 * palm_dig ** 2)
    air = (0.5 + 0.5 * torch.clamp(clear / 0.020, 0.0, 1.0)) \
        * (0.7 + 0.3 * torch.clamp(pc / 0.010, 0.0, 1.0))

    # ================ STAGE 1: approach (kept; dwell decay removed, calm arrival preferred) =====
    c_gap = _win(gap_n, -0.010, 0.020, 0.015)
    c_along = _win(gap_f, -0.005, 0.020, 0.015)
    c_height = _win(h_palm, -0.9 * hh, 0.9 * hh, 0.020)
    c_ori = _win(ang_n, 0.0, 0.60, 0.30) * _win(ang_f, 0.0, 0.60, 0.30)   # env allows ~45 deg
    c_pose = _win(dev_max, 0.0, 0.22, 0.12)                               # env allows 0.3
    c_free = 1.0 - any_touch                                              # nothing may touch yet
    conds = torch.stack([c_gap, c_along, c_height, c_ori, c_pose, c_free], dim=-1)   # (N,6)
    form = conds.mean(-1)
    lock = conds.prod(-1)

    a_n = torch.relu(gap_n - 0.018) + torch.relu(-0.008 - gap_n)
    a_f = torch.relu(gap_f - 0.018) + torch.relu(-0.003 - gap_f)
    a_h = torch.relu(h_palm - 0.025) + torch.relu(-0.012 - h_palm)
    d_app = torch.sqrt(a_n ** 2 + a_f ** 2 + a_h ** 2 + 1e-12)
    prox = torch.exp(-(d_app / 0.12) ** 2)
    fine_app = torch.exp(-(d_app / 0.05) ** 2)

    off_z = torch.clamp(0.30 * hh, max=0.020)
    offset = torch.stack([-(r + 0.006), -(r + 0.008), off_z], dim=-1)
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(1.8 * d_world)

    shape_k = 0.15 + 0.85 * keep           # reaching the cup by rotating the wrist forfeits 85 %
    # CHANGE: no dwell decay - reaching the approached pose pays the same however long it took
    approach_travel = 4.0 * s1 * (0.55 * coarse + 0.45 * fine_app) * shape_k
    approach_form = 3.0 * s1 * form * (0.3 + 0.7 * prox)
    # CHANGE: the latch configuration pays full only when reached calmly (no overshoot into the cup)
    approach_lock = 2.0 * s1 * lock * (0.5 + 0.5 * arm_calm)
    approach_keep = 1.0 * s1 * keep * (0.2 + 0.8 * prox)
    # stage 1 max = 10.0

    # ---------------- grasp pose (wider than the latch window: the palm may press in) ----------
    e_n = torch.relu(gap_n - 0.010) + torch.relu(-0.015 - gap_n)
    e_f = torch.relu((gap_f - 0.006).abs() - 0.010)
    e_h = torch.relu(h_palm - 0.025) + torch.relu(-0.012 - h_palm)   # middle of the cup body
    d_pose = torch.sqrt(e_n ** 2 + e_f ** 2 + e_h ** 2 + 1e-12)
    pos = torch.exp(-(d_pose / 0.05) ** 2)

    # ---------------- link geometry in the cup / palm frame -----------------------------------
    lv = ctx.link_pos - ctx.cup_pos[:, None, None, :]      # (N,5,3,3)
    ax = axis[:, None, None, :]
    lh = (lv * ax).sum(-1)                                 # (N,5,3)
    lrad_vec = lv - lh.unsqueeze(-1) * ax
    lrad = torch.norm(lrad_vec, dim=-1)
    u_n = (lrad_vec * n[:, None, None, :]).sum(-1)
    u_f = (lrad_vec * fd[:, None, None, :]).sum(-1)
    in_band = torch.clamp(1.0 - torch.relu(lh.abs() - hh[:, None, None]) / 0.02, 0.0, 1.0)
    near = torch.exp(-torch.relu(lrad - r[:, None, None] - 0.012) / 0.02) * in_band
    engage = torch.maximum(touch, near).amax(-1)           # (N,5)

    theta = torch.atan2(u_f, -u_n)                         # 0 palm side, pi far side
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)
    finger_wrap = (touch[:, 1:, :] * wrap[:, 1:, :]).amax(-1)        # (N,4)
    wrap_q = torch.clamp(finger_wrap.sum(-1) / 3.0, 0.0, 1.0)
    wrap_deep = finger_wrap.mean(-1)
    thumb_dir = torch.clamp(0.5 - u_f[:, 0, :] / (2.0 * r[:, None]), 0.0, 1.0)
    thumb_q = (touch[:, 0, :] * (0.4 + 0.6 * thumb_dir)).amax(-1)
    fingers_q = torch.clamp(digit_touch[:, 1:].sum(-1) / 3.0, 0.0, 1.0)

    # ---------------- finger closure ----------------------------------------------------------
    flex = torch.clamp((ctx.hand_q_norm - ctx.hand_default_q_norm) / 0.5, 0.0, 1.0)
    digit_flex = torch.stack([flex[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)

    # ---------------- palm progress -----------------------------------------------------------
    press_geo = torch.exp(-torch.relu(gap_n + 0.010) / 0.025)
    press = torch.maximum(palm_touch, 0.7 * press_geo)
    arrived = torch.maximum(palm_touch, torch.clamp((0.020 - gap_n) / 0.020, 0.0, 1.0))
    not_arrived = 1.0 - arrived
    close_gate = 0.25 + 0.75 * arrived

    # ---------------- cup state ---------------------------------------------------------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]       # lift relative to THIS cup's start height
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    upright = torch.clamp(1.0 - (ctx.cup_tilt - 0.35) / 0.35, 0.0, 1.0)   # 1 below 20 deg, 0 at 40
    upright_c = torch.exp(-(torch.relu(ctx.cup_tilt - 0.09) / 0.17) ** 2)  # carrying: 1 below 5 deg
    resting = torch.clamp(1.0 - (dz - 0.03) / 0.03, 0.0, 1.0)
    ori_k = 0.5 + 0.5 * ori
    dwell2 = 1.0 - 0.5 * progress

    # ================ STAGE 2 ONLY: the grasp ladder (CHANGE: g2, no longer paid in stage 3) ====
    g2 = s2 * ori_k * upright * resting * air * dwell2

    pregrasp_shape = 5.0 * s2 * air * dwell2 * not_arrived * keep * pos
    early_contact_pen = -5.0 * s1 * any_touch - 4.0 * s2 * not_arrived * any_touch

    grasp_pose = 10.0 * g2 * pos                   # rung 0 = stage-1 ceiling: latching steps up
    palm_reach = 6.0 * g2 * pos * press_geo
    palm_contact = g2 * (0.3 + 0.7 * pos) * (5.0 * palm_touch + 2.0 * palm_firm)
    finger_curl = 5.0 * g2 * pos * close_gate * (digit_flex[:, 1:] * engage[:, 1:]).mean(-1)
    thumb_curl = 2.5 * g2 * pos * close_gate * digit_flex[:, 0] * engage[:, 0]
    light_contact = 1.0 * g2 * torch.clamp(digit_touch.sum(-1) / 4.0, 0.0, 1.0)
    grasp_wrap = g2 * press * (3.5 * wrap_q + 1.5 * wrap_deep)
    grasp_thumb = 3.0 * g2 * press * thumb_q
    grasp_fingers = 3.0 * g2 * press * fingers_q
    grip_squeeze = 1.5 * g2 * press * squeeze
    pinky_join = 1.2 * g2 * press * digit_touch[:, 4] * digit_flex[:, 4]
    envelope_quality = 6.0 * g2 * palm_touch * thumb_touch * fingers_q
    # ladder max ~51 raw, only while the cup still rests on the table and before the envelope latch

    # ================ STAGE 3: grasp is the TICKET, height and goal are the INCOME ==============
    firm = _sat(ctx.link_cup_force, 0.3).amax(-1)                     # (N,5)
    hold_q = firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0) \
        * (0.6 + 0.4 * palm_touch) * (0.6 + 0.4 * wrap_q)             # [0,1]; 0 = cup not held
    h3 = s3 * hold_q                                                  # every stage-3 payment uses it

    goal_h = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)   # per-env lift span
    h_frac = torch.clamp(dz / goal_h, 0.0, 1.0)                       # 0 on the table, 1 at goal height
    air_k = torch.clamp(dz / 0.03, 0.0, 1.0)                          # first 3 cm

    # ticket: ~ the stage-2 value at the latch (no step down), but it decays while the cup rests
    # (to 40 % at the end of the budget); 3 cm of lift restores it fully
    dwell_rest = 1.0 - 0.6 * progress
    grasp_hold = 32.0 * h3 * ori_k * upright * air * (dwell_rest + (1.0 - dwell_rest) * air_k)
    # dense, immediate: +20 for the first 3 cm (the step that was never taken last round)
    lift_start = 20.0 * h3 * upright_c * air_k
    # steady: +50 linear in height from the cup's own start to the goal height
    lift_height = 50.0 * h3 * upright_c * h_frac
    # upward cup velocity while held; downward is NEGATIVE (worse than still); fades at goal height
    vz = ctx.cup_lin_vel[:, 2]
    lift_vel = 12.0 * h3 * upright_c * torch.clamp(vz / 0.08, -1.0, 1.0) * (1.0 - h_frac)

    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    goal_close = 30.0 * h3 * upright_c * (1.0 - torch.tanh(gd / 0.12))
    goal_near = 45.0 * h3 * upright_c * torch.exp(-(gd / 0.08) ** 2)
    goal_in_tol = 20.0 * h3 * upright_c * in_tol
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) \
        * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 40.0 * h3 * upright_c * torch.exp(-(gd / 0.08) ** 2) * calm
    success_bonus = 300.0 * s3 * ctx.success.float()          # unchanged; success condition untouched

    # hand sinking down the cup while it rests (pressing it into the table): up to -6
    hand_sink_pen = -6.0 * (s2 + s3) * resting * torch.clamp(torch.relu(-0.012 - h_palm) / 0.03, 0.0, 1.0)

    # ---------------- smooth carrying: bounded (<= -8) vs >= ~20 income for any 1 cm lift -----------
    held_air = h3 * torch.clamp(dz / 0.01, 0.0, 1.0)
    qd_excess = torch.relu(ctx.arm_qd.abs() - 1.0)             # free below 1 rad/s per joint
    carry_arm_pen = -4.0 * held_air * torch.clamp((qd_excess ** 2).sum(-1) / 9.0, 0.0, 1.0)
    cup_speed = torch.norm(ctx.cup_lin_vel, dim=-1)
    cup_spin = torch.norm(ctx.cup_ang_vel, dim=-1)
    carry_cup_pen = -3.0 * held_air * torch.clamp((cup_speed - 0.25) / 0.75, 0.0, 1.0) \
        - 1.0 * held_air * torch.clamp((cup_spin - 2.0) / 4.0, 0.0, 1.0)

    # ---------------- constraints and regularization -------------------------------------------
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.04) / 0.10, 0.0, 1.0)
    cup_tilt_pen = -8.0 * torch.clamp((ctx.cup_tilt - 0.17) / (math.pi / 3.0 - 0.17), 0.0, 1.0)
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    # CHANGE: only joint speed above 0.5 rad/s is charged, so slow useful motion is free
    arm_vel_pen = -0.02 * (torch.relu(ctx.arm_qd.abs() - 0.5) ** 2).sum(-1)

    raw = {
        "approach_travel": approach_travel,
        "approach_form": approach_form,
        "approach_lock": approach_lock,
        "approach_keep": approach_keep,
        "pregrasp_shape": pregrasp_shape,
        "early_contact_pen": early_contact_pen,
        "grasp_pose": grasp_pose,
        "palm_reach": palm_reach,
        "palm_contact": palm_contact,
        "finger_curl": finger_curl,
        "thumb_curl": thumb_curl,
        "light_contact": light_contact,
        "grasp_wrap": grasp_wrap,
        "grasp_thumb": grasp_thumb,
        "grasp_fingers": grasp_fingers,
        "grip_squeeze": grip_squeeze,
        "pinky_join": pinky_join,
        "envelope_quality": envelope_quality,
        "grasp_hold": grasp_hold,
        "lift_start": lift_start,
        "lift_height": lift_height,
        "lift_vel": lift_vel,
        "goal_close": goal_close,
        "goal_near": goal_near,
        "goal_in_tol": goal_in_tol,
        "hold_still": hold_still,
        "success_bonus": success_bonus,
        "hand_sink_pen": hand_sink_pen,
        "carry_arm_pen": carry_arm_pen,
        "carry_cup_pen": carry_cup_pen,
        "cup_disturb_pen": cup_disturb_pen,
        "cup_tilt_pen": cup_tilt_pen,
        "table_pen": table_pen,
        "palm_table_pen": palm_table_pen,
        "action_rate_pen": action_rate_pen,
        "arm_vel_pen": arm_vel_pen,
    }
    components = {name: scale * value for name, value in raw.items()}
    reward = torch.stack(list(components.values()), dim=0).sum(0)
    return reward, components
