import torch
import math


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted)."""
    ARRIVE_RISE = 0.045    # arrive 45 mm above grasp height (0/336 knock-overs at +45 mm) ...
    FUNNEL_R0 = 0.01       # ... and be at grasp height once the cup is centred within 1 cm
    FUNNEL_R1 = 0.04
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # pinch height above cup origin (~13 mm, inside the 0-25 mm lift group)
    thumb_hi = 0.5 * cup_mouth_z        # thumb band 0..27 mm above origin (no lift with thumb >= 50 mm)

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F) height along the cup axis
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
    # gap is zero only at the measured contact radius (tip frame 26-28 mm, cup outer 28.5 mm);
    # the old +10 mm pad made 33-37 mm count as "on the wall" and killed the gradient
    gap = torch.clamp(rad - tip_r, min=0.0)                    # (N,F)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)        # (N,)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values   # nearest non-thumb tip, (N,)

    # opposition: 1 when the thumb and the other fingers sit on opposite sides of the cup axis
    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))                # (N,)

    # pinch centre and the descending funnel target
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp,
            "d_funnel": d_funnel, "d_grip": d_grip, "thumb_band": thumb_band}


def _hand_cup_terms(tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd, cup_pos, cup_up,
                    cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok, lift_target):
    """Unweighted per-arm terms, each in [0, 1]. pose_ok (N,) = cup attitude is acceptable for this stage."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    # A. reach: long-range funnel term + per-side fingertip-on-wall term.
    #    Each side is scored on its own (sum form) so the thumb keeps its full gradient while the finger
    #    side is already done, and measured force completes a side: reach saturates only with contact
    #    force on BOTH sides (the parked pose of i01 had thumb 0.00 N and reach 0.97).
    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach = 0.4 * reach_far + 0.6 * reach_tips

    # B. grasp: balanced two-sided pinch. Before the flag, finger-first touch earns 0.15 and any thumb force
    #    added to it can only raise the value (0.5 N -> +0.19); thumb-only still earns nothing.
    #    Flag boundary (1 N / 1 N): pre_grasp 0.355 -> flagged 0.75, a step up, never a valley.
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)          # lifts had thumb ~2.3 N vs fingers ~2.7 N
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, pre_grasp)
    grasp = contact_q * (0.1 + 0.9 * geo["thumb_band"]) * pose_ok

    # closure shaping: measured grasps close to 0.68-0.98, the parked hand sat at 0.19 with the closure
    # channels driven open; pays only once the pinch centre is at the grasp pose (contact freeze makes
    # closing there harmless), saturates at 0.7 so over-squeezing is not rewarded
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # C. lift: measured cup elevation only, clamped, and only while really held (explosion-proof)
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)                # 5 mm dead band: tipping on the rim is not lifting
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_f * torch.exp(-(geo["d_grip"] / 0.08) ** 2) * calm
    lift_prog = 0.5 * torch.tanh(h_pos / 0.02) + 0.5 * torch.clamp(h_pos / lift_target, max=1.0)
    lift = held * pose_ok * lift_prog
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)               # 0 on the table, 1 from 2.5 cm up
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)            # "held and at least 5 cm up"

    # penalties (unweighted, bounded)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)       # 1 N dead band: a slightly stronger thumb is free
    finger_excess = torch.clamp(f_o - f_t, min=0.0)             # finger-only contact: adding thumb force lowers this
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)   # knock-overs came at ~0.09 m/s hand speed
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "grasp": grasp, "close": close, "lift": lift, "up_w": up_w, "slide": slide,
            "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling,
            "held": held, "far": torch.tanh(4.0 * geo["d_funnel"])}


def _pour_geometry(ctx, outer_r):
    """Distance of the source pouring lip (lowest rim point) from a point above the receiver rim centre."""
    POUR_CLEARANCE = 0.05   # lip 5 cm above the receiver rim: no cup-cup contact, origins stay > 9 cm apart
    ez = torch.tensor([0.0, 0.0, 1.0], device=ctx.src_cup_up.device, dtype=ctx.src_cup_up.dtype)
    # (up * up_z - ez) has norm sin(tilt): zero when upright, straight down by one radius when horizontal
    lip = ctx.src_cup_mouth_pos + outer_r * (ctx.src_cup_up * ctx.src_cup_up[:, 2:3] - ez)
    delta = lip - (ctx.rcv_cup_mouth_pos + POUR_CLEARANCE * ez)
    d_xy = delta[:, :2].norm(dim=-1)
    d_align = torch.sqrt(d_xy ** 2 + (0.5 * delta[:, 2]) ** 2)          # height error counts half
    align_gate = torch.exp(-(d_align / 0.06) ** 2)                       # wide: tilting is allowed here
    pour_gate = torch.exp(-(d_align / max(ctx.cup_radius, 0.015)) ** 2)  # narrow: beads would land inside
    return d_align, align_gate, pour_gate


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Stage ladder: reach < grasp < lift per arm, then carry < align < tilt. New in iter_04: the source
    # lift pay is scaled by receiver progress, and the receiver pays for being away from its cup.
    W_REACH_SRC, W_GRASP, W_LIFT = 1.0, 2.0, 3.0   # source formulation and weights unchanged (grasp_src 1.61, lift_src 2.09)
    W_REACH_RCV = 1.5     # near 0.7 vs away 0.2 measured: staying now earns 0.75 more from reach alone (was 0.5)
    W_CLOSE_SRC = 0.3
    W_CLOSE_RCV = 0.6     # close_rcv peaked at 0.10 of 0.3 with the tip gap stuck at 66-77 mm; doubled
    W_AWAY_RCV = 0.8      # x tanh(4 d): ~0.16 at the grasp pose, ~0.66 at 0.3 m -> leaving costs a further 0.5
    LIFT_FLOOR = 0.15     # source lift pay with the receiver away: 2.09 -> ~0.31, about the old 0.3 net loss of leaving
    PROG_REACH, PROG_GRASP = 0.35, 0.65   # parked-near receiver unlocks 0.45 of lift_src, a grasping one all of it
    REACH_AWAY, REACH_NEAR = 0.2, 0.7     # measured reach_rcv after the retreat / while near the cup
    W_CARRY = 2.0         # new stage between lift (3) and align (4): source mouth toward the receiver
    W_ALIGN, W_TILT = 4.0, 5.0
    W_POUR = 150.0        # x d_in_target: 7.5 per bead with 20 beads
    W_SPILL = 60.0
    W_SUCCESS = 20.0
    W_ALIVE = 1.0
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25  # rcv halved: cup_disturb_rcv was -0.13 against grasp_rcv 0.01
    W_TILT_PEN = 0.3                      # knock-over protection kept for both cups (tips at 0.3-0.4 N one-sided)
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05    # rush was the largest near-cup cost (-0.23 summed at e2000)
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05
    CUP_OUTER_R = 0.0285
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35
    TILT_FULL = math.radians(100.0)
    LIFT_TARGET_SRC, LIFT_TARGET_RCV = 0.15, 0.08

    dtype = ctx.src_cup_tilt.dtype
    d_align, align_gate, pour_gate = _pour_geometry(ctx, CUP_OUTER_R)
    upright_src = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = align_gate + (1.0 - align_gate) * upright_src
    pose_ok_rcv = upright_rcv

    src = _hand_cup_terms(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                          ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up,
                          ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_src, LIFT_TARGET_SRC)
    rcv = _hand_cup_terms(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                          ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                          ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_rcv, LIFT_TARGET_RCV)

    # receiver progress in [0, 1]: 0 at the retreated pose, 0.35 parked at the cup, 1 with a flagged grasp
    # (grasp term is >= 0.4 once flagged with the thumb in its band)
    prog_reach = torch.clamp((rcv["reach"] - REACH_AWAY) / (REACH_NEAR - REACH_AWAY), 0.0, 1.0)
    prog_grasp = torch.clamp(rcv["grasp"] / 0.4, max=1.0)
    rcv_prog = PROG_REACH * prog_reach + PROG_GRASP * torch.maximum(prog_grasp, rcv["up_w"])
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog

    nested = ctx.cups_nested.to(dtype)
    # carry: needs the source held and up and the receiver merely HELD (not yet lifted), so the gradient
    # exists at the measured 0.34 m: linear part keeps a constant slope out to 0.45 m
    carry_w = src["up_w"] * rcv["held"] * pose_ok_rcv * (1.0 - nested)
    carry = carry_w * (0.5 * (1.0 - torch.tanh(2.5 * d_align)) + 0.5 * torch.clamp(1.0 - d_align / 0.45, 0.0, 1.0))
    both_up = src["up_w"] * rcv["up_w"] * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(4.0 * d_align)) + 0.5 * torch.exp(-25.0 * d_align))
    pour_tilt = both_up * pour_gate * torch.clamp(ctx.src_cup_tilt / TILT_FULL, max=1.0)

    tilt_pen_src = (1.0 - align_gate) * torch.tanh(ctx.src_cup_tilt / UPRIGHT_TOL)
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)

    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        "reach_src": W_REACH_SRC * src["reach"],
        "reach_rcv": W_REACH_RCV * rcv["reach"],
        "away_rcv": -W_AWAY_RCV * rcv["far"],
        "close_src": W_CLOSE_SRC * src["close"],
        "close_rcv": W_CLOSE_RCV * rcv["close"],
        "grasp_src": W_GRASP * src["grasp"],
        "grasp_rcv": W_GRASP * rcv["grasp"],
        "lift_src": W_LIFT * lift_gate * src["lift"],
        "lift_rcv": W_LIFT * rcv["lift"],
        "carry_src": W_CARRY * carry,
        "align_mouths": W_ALIGN * align,
        "pour_tilt": W_TILT * pour_tilt,
        "pour_delta": W_POUR * ctx.d_in_target,
        "spill_delta": -W_SPILL * ctx.d_spill,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype),
        "cup_disturb_src": -(W_SLIDE_SRC * src["slide"] + W_TILT_PEN * tilt_pen_src),
        "cup_disturb_rcv": -(W_SLIDE_RCV * rcv["slide"] + W_TILT_PEN * tilt_pen_rcv),
        "contact_push": -(W_ONE_SIDED_SRC * src["one_sided"] + W_ONE_SIDED_RCV * rcv["one_sided"]
                          + W_PALM_SRC * src["palm_push"] + W_PALM_RCV * rcv["palm_push"]),
        "rush_near_cup": -(W_RUSH_SRC * src["rush"] + W_RUSH_RCV * rcv["rush"]),
        "cup_fling": -W_FLING * (src["fling"] + rcv["fling"]),
        "cup_collision": -W_CUP_CUP * torch.tanh(ctx.cup_cup_force / 5.0),
        "hand_foreign": -W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                      + torch.tanh(ctx.rcv_hand_foreign_force / 5.0)),
        "nested": -W_NESTED * nested,
        "action_rate": -W_ACT_RATE * act_rate,
        "rcv_progress": torch.zeros_like(rcv_prog),
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    comps["rcv_progress"] = rcv_prog   # diagnostic only (added after the sum, not part of the reward)
    return reward, comps
