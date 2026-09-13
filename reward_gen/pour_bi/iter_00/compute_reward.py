import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (N,3),(N,3) -> (N,)
    return torch.norm(a - b, dim=-1)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    zeros = torch.zeros(N, device=dev, dtype=ctx.src_palm_pos.dtype)
    ones = torch.ones_like(zeros)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET = 0.10        # [m] cup height above spawn that counts as fully lifted
    LIFT_GATE = 0.05          # [m] minimum lift before alignment is rewarded
    NEAR_PALM = 0.10          # [m] palm-cup distance below which closure is rewarded
    ALIGN_XY_GATE = 1.5 * ctx.cup_radius   # [m] rim-over-rim tolerance for the tilt gate
    DZ_MIN, DZ_MAX = 0.03, 0.12            # [m] source rim above receiver rim window
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, so saturate slightly above
    TILT_FREE = 0.5           # [rad] tilt allowed before "premature tilt" is charged
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side

    # ------------------------------------------------------------------ stage 1: approach
    # weight 1.0 per arm — smallest stage, only has to pull the palms toward the cups
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-5.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-5.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp
    # 1.0 for the contact-established flag + 0.5 for measured closure while the palm is near.
    # Closure is gated on proximity so closing in free air earns nothing.
    near_src = (d_src < NEAR_PALM).to(zeros.dtype)
    near_rcv = (d_rcv < NEAR_PALM).to(zeros.dtype)
    g_src = ctx.src_grasped.to(zeros.dtype)
    g_rcv = ctx.rcv_grasped.to(zeros.dtype)
    grasp_src = 1.0 * g_src + 0.5 * near_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * near_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)

    # ------------------------------------------------------------------ stage 3: lift
    # weight 2.0 per arm — larger than grasp so holding on and lifting beats holding on the table.
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET, 0.0, 1.0)
    lift_src = 2.0 * g_src * lift_frac_src
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv

    both_lifted = (
        ctx.src_grasped & ctx.rcv_grasped & (h_src > LIFT_GATE) & (h_rcv > LIFT_GATE)
    )
    both_lifted_f = both_lifted.to(zeros.dtype)

    # ------------------------------------------------------------------ stage 4: align rims
    # weight 3.0 — must beat lift (2+2 already banked) so the arms keep moving after lifting.
    mouth_delta = ctx.src_cup_mouth_pos - ctx.rcv_cup_mouth_pos          # (N,3)
    d_xy = torch.norm(mouth_delta[:, :2], dim=-1)                          # (N,)
    dz = mouth_delta[:, 2]                                                 # (N,)
    dz_err = torch.clamp(DZ_MIN - dz, min=0.0) + torch.clamp(dz - DZ_MAX, min=0.0)
    align_xy = torch.exp(-8.0 * d_xy)
    align_z = torch.exp(-10.0 * dz_err)
    align = 3.0 * both_lifted_f * align_xy * align_z

    aligned = both_lifted & (d_xy < ALIGN_XY_GATE) & (dz > 0.0)
    aligned_f = aligned.to(zeros.dtype)

    # ------------------------------------------------------------------ stage 5: tilt
    # weight 3.0, only while aligned; saturates at TILT_TARGET so full pour is the goal.
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    tilt = 3.0 * aligned_f * tilt_frac
    # premature tilt: tilting when the rim is NOT over the receiver spills beads on the table
    tilt_premature = -1.0 * (1.0 - aligned_f) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)

    # ------------------------------------------------------------------ stage 6: bead transfer
    # 50 * Δfrac = 2.5 per bead (20 beads) — one-shot payments, must dominate the shaping stack.
    # 20 * Δfrac = -1.0 per spilled bead: one bead in the target outweighs two lost.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -20.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ constraints
    # receiver cup upright (weight 1.0, saturates at 1 rad)
    upright_rcv = -1.0 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    # dropped / knocked-off cup or toppled receiver cup (flat -2.0 each while it lasts)
    src_dropped = (h_src < -DROP_DEPTH).to(zeros.dtype)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(zeros.dtype)
    drop = -2.0 * (src_dropped + rcv_dropped)

    # action-rate regularisation (weight 0.02 — tiny, must never compete with any stage term)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus
    success = 10.0 * ctx.success.to(zeros.dtype)

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "align": align,
        "tilt": tilt,
        "tilt_premature": tilt_premature,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
        "upright_rcv": upright_rcv,
        "drop": drop,
        "action_rate": action_rate,
        "success": success,
    }

    reward = zeros
    for v in components.values():
        reward = reward + v

    return reward, components
