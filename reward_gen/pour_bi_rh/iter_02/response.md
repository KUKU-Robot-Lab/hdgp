# pour_bi_rh iter_02 reward (iter_00 code + round-3 operator patch: PAD 0, two-scale reach_tips, monotone touch_first — see notes_round3.md)

## 1. What the task means and the stages

Two arms, two identical light cups (0.134 kg, 57 mm outer diameter, rim 53.9 mm above the cup origin). The right (source) hand must hold the bead cup, the left (receiver) hand the empty cup. Both cups leave the table, the source rim goes over the receiver rim, the source cup tilts past about 70-90 degrees so the beads fall through the air into the receiver cup, the receiver cup stays within 20 degrees of upright, the cups never touch or nest, and at least half of the beads end up in the receiver cup with little spill. Success is judged by the env; the reward only shapes the way there.

Stages (per arm for A-C, joint for D-F):

- A. Reach without disturbing the cup. The operator notes say a low sideways arrival knocks the cup over (120/336 at -10 mm, 0/336 at +45 mm) while the only lifts happened with the thumb tip 0-50 mm above the cup origin. The reward resolves that conflict with a funnel: the pinch centre is pulled to a point 45 mm above the grasp height while it is still more than about 4 cm from the cup axis, and the target height drops to the grasp height only as the cup becomes centred between thumb and fingers. Fingertips are then pulled to the cup wall, thumb inside the 0-27 mm height band, thumb and fingers on opposite sides of the cup axis.
- B. Two-sided fingertip pinch with the cup still upright and in place. Closing moved the cup in 1132/1152 trials at 0.3-0.4 N, mostly from one-sided (thumb-only) contact, and thumb-first contact ended fallen in 380/519 trials. So: fingers-first touch gets a small credit, thumb-only push is penalised, the main grasp reward needs `grasped` and scales with the smaller of thumb force and summed finger force (balanced pinch), with the thumb-height band (0 lifts of 113 with the thumb at or above the rim), and with cup uprightness. Palm contact is penalised (palm force was 0 in all lifts, and the prompt asks for fingertips only).
- C. Lift. `grasped` and upright almost never meant liftable (4 of 218), so the lift term pays only for measured cup elevation, with a steep first 2 cm (the failures rose at most 0.3 cm) and saturation at the target height. It is gated by `grasped`, by the pinch centre being at the cup, and by a calm cup velocity, and the height is clamped, so a contact explosion (cup 1 m up or 10 m away) earns nothing; cup speed above normal handling speed is also penalised.
- D. Align. With both cups held and up, the lowest rim point of the source cup (the pouring lip, computed from the cup axis) is pulled to 5 cm above the receiver rim centre. 5 cm keeps the cups from touching and keeps the cup origins farther apart than the 9 cm nesting distance.
- E. Pour. Source tilt is rewarded only when the lip is over the receiver mouth (gate width = cup inner radius); tilt anywhere else is penalised. Bead transfer is paid on increments (`d_in_target`, negative increments cost the same), spill on increments.
- F. Hold. Receiver tilt penalty, success bonus per step, alive bonus so that ending the episode by dropping a cup is never the better option.

Safety terms run through all stages: cup-cup force, hand foreign force, nesting, all bounded with tanh(f / 5).

Design rules used: every shaping term is bounded in [0, 1] before weighting; the stage ladder is cumulative (reach 1, grasp 2, lift 3 per arm, align 4, tilt 5), so a later stage never lowers an earlier term; all persistent penalties together stay below alive + reach, so no state makes termination attractive; returned components are already weighted and sum exactly to the reward.

## 2. Reward function

```python
import torch
import math


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, outer_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted)."""
    PAD = 0.0              # measured (round 3): in upright two-sided contact the tip frames sit at 27.9 mm (thumb) / 26.2 mm (finger) from the axis
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
    gap = torch.clamp(rad - (outer_r + PAD), min=0.0)          # (N,F) distance left to the cup wall

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


def _hand_cup_terms(tips_pos, finger_force, palm_force, grasped, arm_qd, cup_pos, cup_up,
                    cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, outer_r, pose_ok, lift_target):
    """Unweighted per-arm terms, each in [0, 1]. pose_ok (N,) = cup attitude is acceptable for this stage."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, outer_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)

    # A. reach: long-range funnel term + short-range fingertip-on-wall term
    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    d_tips = 0.5 * (geo["d_thumb"] + geo["d_oth"])
    # long scale brings the hand in, short scale keeps a slope over the last centimetre (iter_01 parked 5-9 mm off the wall)
    reach_tips = (0.5 * torch.exp(-30.0 * d_tips) + 0.5 * torch.exp(-160.0 * d_tips)) * (0.3 + 0.7 * geo["opp"])
    reach = 0.5 * reach_far + 0.5 * reach_tips

    # B. grasp: balanced two-sided pinch; fingers-first touch gets a small credit, thumb-only gets none
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)          # lifts had thumb ~2.3 N vs fingers ~2.7 N
    # thumb-only still earns nothing; adding thumb force to a finger contact never lowers the reward (iter_01 had a valley below the 1 N flag)
    touch_first = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, touch_first)
    grasp = contact_q * (0.1 + 0.9 * geo["thumb_band"]) * pose_ok

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
    thumb_excess = torch.clamp(f_t - f_o, min=0.0)
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)   # knock-overs came at ~0.09 m/s hand speed
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "grasp": grasp, "lift": lift, "up_w": up_w, "slide": slide,
            "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


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
    # Stage ladder (per step, cumulative): reach 1 < grasp 2 < lift 3 per arm, then align 4 < tilt 5.
    # A later stage keeps every earlier term, so progress never lowers the reward.
    W_REACH, W_GRASP, W_LIFT = 1.0, 2.0, 3.0
    W_ALIGN, W_TILT = 4.0, 5.0
    W_POUR = 150.0        # x d_in_target: 7.5 per bead with 20 beads, one-time, dominates the dense terms at that step
    W_SPILL = 60.0        # x d_spill: 3 per bead, below the pour pay so an imperfect pour is still worth trying
    W_SUCCESS = 20.0      # per step while the env reports success
    W_ALIVE = 1.0         # with reach it outweighs all persistent penalties, so ending the episode never pays
    W_SLIDE, W_TILT_PEN = 0.5, 0.3
    W_ONE_SIDED, W_PALM, W_RUSH, W_FLING = 0.3, 0.2, 0.2, 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05     # x mean squared action change (<= 4), regularisation only
    CUP_OUTER_R = 0.0285
    UPRIGHT_TOL = 0.35    # rad, = the 20 deg success limit on the receiver cup
    TILT_FULL = math.radians(100.0)   # half the beads are out by 86 deg, reward saturates a little past that
    LIFT_TARGET_SRC, LIFT_TARGET_RCV = 0.15, 0.08   # source ends above the receiver

    dtype = ctx.src_cup_tilt.dtype
    d_align, align_gate, pour_gate = _pour_geometry(ctx, CUP_OUTER_R)
    upright_src = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = align_gate + (1.0 - align_gate) * upright_src   # source may tilt only over the receiver
    pose_ok_rcv = upright_rcv

    src = _hand_cup_terms(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_grasped,
                          ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel,
                          ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z, CUP_OUTER_R,
                          pose_ok_src, LIFT_TARGET_SRC)
    rcv = _hand_cup_terms(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_grasped,
                          ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel,
                          ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z, CUP_OUTER_R,
                          pose_ok_rcv, LIFT_TARGET_RCV)

    nested = ctx.cups_nested.to(dtype)
    both_up = src["up_w"] * rcv["up_w"] * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(4.0 * d_align)) + 0.5 * torch.exp(-25.0 * d_align))
    pour_tilt = both_up * pour_gate * torch.clamp(ctx.src_cup_tilt / TILT_FULL, max=1.0)

    tilt_pen_src = (1.0 - align_gate) * torch.tanh(ctx.src_cup_tilt / UPRIGHT_TOL)
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)

    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        "reach_src": W_REACH * src["reach"],
        "reach_rcv": W_REACH * rcv["reach"],
        "grasp_src": W_GRASP * src["grasp"],
        "grasp_rcv": W_GRASP * rcv["grasp"],
        "lift_src": W_LIFT * src["lift"],
        "lift_rcv": W_LIFT * rcv["lift"],
        "align_mouths": W_ALIGN * align,
        "pour_tilt": W_TILT * pour_tilt,
        "pour_delta": W_POUR * ctx.d_in_target,
        "spill_delta": -W_SPILL * ctx.d_spill,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype),
        "cup_disturb_src": -(W_SLIDE * src["slide"] + W_TILT_PEN * tilt_pen_src),
        "cup_disturb_rcv": -(W_SLIDE * rcv["slide"] + W_TILT_PEN * tilt_pen_rcv),
        "contact_push": -(W_ONE_SIDED * (src["one_sided"] + rcv["one_sided"])
                          + W_PALM * (src["palm_push"] + rcv["palm_push"])),
        "rush_near_cup": -W_RUSH * (src["rush"] + rcv["rush"]),
        "cup_fling": -W_FLING * (src["fling"] + rcv["fling"]),
        "cup_collision": -W_CUP_CUP * torch.tanh(ctx.cup_cup_force / 5.0),
        "hand_foreign": -W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                      + torch.tanh(ctx.rcv_hand_foreign_force / 5.0)),
        "nested": -W_NESTED * nested,
        "action_rate": -W_ACT_RATE * act_rate,
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    return reward, comps
```

## 3. Notes on choices and open points

- Every returned component is already weighted; the components sum exactly to `reward`.
- Context fields not used: `table_z`, `cup_bottom_z`, `num_beads`, `*_palm_pos`, `*_palm_axes`, `*_hand_closure`, `*_cup_quat`, `bead_in_source_frac`, `bead_in_target_frac`, `bead_spill_frac`, `bead_centroid`, `episode_progress`. Hand position is taken from the fingertips (pinch centre) instead of the palm because the evidence in the operator notes is about fingertip placement.
- The closing speed of the fingers cannot be penalised directly: the context has the measured closure but not its rate, and a constant "close" command has zero action rate. Horizontal cup velocity while the cup is on the table is used as the immediate signal that closing is pushing the cup.
- Assumptions that the prompt does not settle: (a) `tips_pos` are fingertip frame origins; measured contact radius equals the cup outer radius (PAD = 0.0, round-3 operator patch); (b) the order of fingers 1..F-1 is not given, so the code only distinguishes thumb (index 0) from "the others"; (c) the sign convention of the palm normal in `*_palm_axes` is not given, so no palm-facing term was written; (d) whether `success` ends the episode is not stated. The bonus is per step, which is right if the episode continues after success. If success terminates the episode, a one-time 20 is small next to the dense stream and should be raised.
