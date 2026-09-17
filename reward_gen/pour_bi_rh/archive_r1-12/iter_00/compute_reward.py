import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance along the last dim -> (...,)."""
    return torch.norm(a - b, dim=-1)


def _near(d: torch.Tensor, k: float) -> torch.Tensor:
    """Bounded closeness in (0, 1]: 1 at d = 0, ~0.37 at d = 1/k."""
    return torch.exp(-k * d)


def _lift_progress(cup_pos: torch.Tensor, spawn_pos: torch.Tensor, full_h: float) -> torch.Tensor:
    """Height above the spawn height, clamped to [0, 1] at full_h metres."""
    h = cup_pos[:, 2] - spawn_pos[:, 2]
    return torch.clamp(h / full_h, 0.0, 1.0)


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)          # (N,) float template, never touches ctx
    src_g = ctx.src_grasped.to(zero.dtype)              # (N,) 1.0 when thumb + finger on source cup
    rcv_g = ctx.rcv_grasped.to(zero.dtype)              # (N,)
    both_g = src_g * rcv_g

    # ------------------------------------------------------------------ stage 0: reach
    # palm -> own cup centre. k = 8: 1.0 at contact, 0.45 at 10 cm, 0.2 at 20 cm.
    d_src_palm = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv_palm = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * _near(d_src_palm, 8.0)         # weight 1.0 each arm
    approach_rcv = 1.0 * _near(d_rcv_palm, 8.0)

    # ------------------------------------------------------------------ stage 1: fingertip grasp
    # mean fingertip distance to the cup centre; only paid once the palm is near (< ~15 cm),
    # so fingers are not rewarded for curling in mid-air far from the cup. weight 0.5 (helper).
    tips_src_d = _dist(ctx.src_tips_pos, ctx.src_cup_pos[:, None, :]).mean(dim=-1)
    tips_rcv_d = _dist(ctx.rcv_tips_pos, ctx.rcv_cup_pos[:, None, :]).mean(dim=-1)
    palm_near_src = (d_src_palm < 0.15).to(zero.dtype)
    palm_near_rcv = (d_rcv_palm < 0.15).to(zero.dtype)
    tips_src = 0.5 * palm_near_src * _near(tips_src_d, 12.0)
    tips_rcv = 0.5 * palm_near_rcv * _near(tips_rcv_d, 12.0)
    # grasp-established bonus: weight 1.0 each (constant while held -> pays for keeping the grasp)
    grasp_src = 1.0 * src_g
    grasp_rcv = 1.0 * rcv_g

    # ------------------------------------------------------------------ stage 2: lift
    # full credit at 8 cm above spawn; gated by grasp so a knocked-up cup earns nothing.
    lift_p_src = _lift_progress(ctx.src_cup_pos, ctx.src_cup_spawn_pos, 0.08)
    lift_p_rcv = _lift_progress(ctx.rcv_cup_pos, ctx.rcv_cup_spawn_pos, 0.08)
    lift_src = 1.5 * src_g * lift_p_src                  # weight 1.5 each: lifting must beat approach
    lift_rcv = 1.5 * rcv_g * lift_p_rcv
    # "lifted" gate: both held and both at least 3 cm up
    lifted = both_g * (lift_p_src > 0.375).to(zero.dtype) * (lift_p_rcv > 0.375).to(zero.dtype)

    # ------------------------------------------------------------------ stage 3: align rims
    mouth_dxy = _dist(ctx.src_cup_mouth_pos[:, :2], ctx.rcv_cup_mouth_pos[:, :2])
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    # vertical window 4..12 cm above the receiver rim (zero error inside, linear outside);
    # too low -> nesting / rim contact, too high -> beads scatter.
    dz_err = torch.clamp(0.04 - mouth_dz, min=0.0) + torch.clamp(mouth_dz - 0.12, min=0.0)
    align_q = _near(mouth_dxy, 10.0) * _near(dz_err, 15.0)   # (N,) in (0,1]
    align = 2.0 * lifted * align_q                        # weight 2.0: > lift so carrying pays
    # alignment gate for tilting: rims within ~4 cm horizontally and inside the height window
    aligned = lifted * (mouth_dxy < 0.04).to(zero.dtype) * (dz_err < 0.01).to(zero.dtype)

    # ------------------------------------------------------------------ stage 4: tilt and pour
    # tilt progress towards ~2.0 rad (beads leave past ~1.9 rad); only paid when aligned.
    tilt_p = torch.clamp(ctx.src_cup_tilt / 2.0, 0.0, 1.0)
    tilt = 2.0 * aligned * tilt_p                         # weight 2.0 works with pour 30: tilt is only the doorway
    # pay the INCREMENT of beads transferred (episode sum <= 30) -> pouring dominates everything
    pour_delta = 30.0 * ctx.d_in_target
    spill_delta = -20.0 * ctx.d_spill                     # spilled beads are permanent: nearly as costly as a transfer is valuable

    # ------------------------------------------------------------------ stage 5: success
    success = 5.0 * ctx.success.to(zero.dtype)            # per-step bonus for reaching and holding the goal

    # ------------------------------------------------------------------ constraints (always on)
    # tilting the source cup before the rims are aligned spills on the table; tilts below 0.8 rad
    # are harmless while carrying, so only the excess is penalised.
    pre_tilt_pen = -1.0 * (1.0 - aligned) * torch.tanh(2.0 * torch.clamp(ctx.src_cup_tilt - 0.8, min=0.0))
    # receiver must stay upright the whole time
    rcv_upright_pen = -1.0 * torch.tanh(3.0 * ctx.rcv_cup_tilt)
    # dropped / thrown cups: bounded speed penalty on both cups (0.3 m/s -> ~0.29 each)
    drop_pen = -0.5 * (torch.tanh(_dist(ctx.src_cup_lin_vel, torch.zeros_like(ctx.src_cup_lin_vel)) / 1.0)
                       + torch.tanh(_dist(ctx.rcv_cup_lin_vel, torch.zeros_like(ctx.rcv_cup_lin_vel)) / 1.0))
    # nested cups transfer nothing and never succeed
    nested_pen = -2.0 * ctx.cups_nested.to(zero.dtype)
    # real-robot safety: cups touching each other, hands touching anything but their own cup
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    hand_foreign_pen = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                               + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    # fingertip grasp wanted: palm pressing on the cup means a shove / power grasp attempt
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ------------------------------------------------------------------ regularisation
    # action rate: mean squared change, bounded by 4 since actions are in [-1, 1]; weight 0.1
    action_rate_pen = -0.1 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    # arm joint speed: slow, smooth motion near light cups; tanh keeps it <= 0.05 per arm
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "tips_src": tips_src,
        "tips_rcv": tips_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "align": align,
        "tilt": tilt,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
        "success": success,
        "pre_tilt_pen": pre_tilt_pen,
        "rcv_upright_pen": rcv_upright_pen,
        "drop_pen": drop_pen,
        "nested_pen": nested_pen,
        "cup_collision_pen": cup_collision_pen,
        "hand_foreign_pen": hand_foreign_pen,
        "palm_push_pen": palm_push_pen,
        "action_rate_pen": action_rate_pen,
        "arm_speed_pen": arm_speed_pen,
    }
    reward = zero
    for v in components.values():
        reward = reward + v
    return reward, components
