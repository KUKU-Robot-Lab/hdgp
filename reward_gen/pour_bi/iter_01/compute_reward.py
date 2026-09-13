import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (N,3),(N,3) -> (N,)
    return torch.norm(a - b, dim=-1)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET = 0.10        # [m] cup height above spawn that counts as fully lifted
    LIFT_GATE = 0.05          # [m] minimum lift before alignment / bring_together are rewarded
    ALIGN_XY_GATE = 1.5 * ctx.cup_radius   # [m] rim-over-rim tolerance for the tilt gate
    DZ_MIN, DZ_MAX = 0.03, 0.12            # [m] source rim above receiver rim window
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, so saturate slightly above
    TILT_FREE = 0.8           # [rad] grasp posture alone tilts the cup 20-35 deg; only charge beyond 46 deg
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side

    # ------------------------------------------------------------------ stage 1: approach
    # weight 1.0 per arm, exp(-4d): wider basin than exp(-5d) so a retreated hand (20 cm) is
    # still pulled back; saturates ~0.68 in the grasp posture (palm ~9 cm from cup origin).
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp
    # 1.0 for the contact-established flag + 0.5 * closure weighted by a SMOOTH proximity
    # factor (the env already forbids closing far from the cup, so no hard gate is needed;
    # the old hard gate at 10 cm left a flat zone in which the receiver hand retreated for free).
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    prox_src = torch.exp(-8.0 * d_src)
    prox_rcv = torch.exp(-8.0 * d_rcv)
    grasp_src = 1.0 * g_src + 0.5 * prox_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * prox_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)
    # weight 1.0 bonus for holding BOTH cups: makes the second grasp worth more than the first,
    # so finishing the receiver branch beats idling with the source cup (3.7/step in the last run).
    both_grasped_b = ctx.src_grasped & ctx.rcv_grasped
    both_grasped = 1.0 * both_grasped_b.to(dt)

    # ------------------------------------------------------------------ stage 3: lift
    # weight 2.0 per arm — larger than grasp so holding on and lifting beats holding on the table.
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET, 0.0, 1.0)
    lift_src = 2.0 * g_src * lift_frac_src
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    src_lifted_f = src_lifted_b.to(dt)
    both_lifted_f = both_lifted_b.to(dt)
    # weight 1.0 bonus for both cups in the air (same combinatorial logic as both_grasped)
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ stage 4: bring cups together / align rims
    mouth_delta = ctx.src_cup_mouth_pos - ctx.rcv_cup_mouth_pos          # (N,3)
    d_xy = torch.norm(mouth_delta[:, :2], dim=-1)                          # (N,)
    dz = mouth_delta[:, 2]                                                 # (N,)
    dz_err = torch.clamp(DZ_MIN - dz, min=0.0) + torch.clamp(dz - DZ_MAX, min=0.0)
    align_xy = torch.exp(-8.0 * d_xy)
    align_z = torch.exp(-10.0 * dz_err)

    # bring_together, weight 1.0: once the SOURCE cup is lifted, pull its rim horizontally over the
    # receiver rim while staying ABOVE it (dz > DZ_MIN). Independent of the receiver grasp so the
    # cups stop drifting apart (0.38 m last run) and the source arm keeps a gradient after lift
    # saturation. The height gate means pushing the source cup INTO the receiver cup earns nothing.
    above_rcv = (dz > DZ_MIN).to(dt)
    bring_together = 1.0 * src_lifted_f * above_rcv * torch.exp(-4.0 * d_xy)

    # align, weight 3.0 — full alignment only with both cups lifted; must beat lift (2+2 banked)
    # plus bring_together (1.0) so lifting the receiver remains the better path.
    align = 3.0 * both_lifted_f * align_xy * align_z

    aligned_b = both_lifted_b & (d_xy < ALIGN_XY_GATE) & (dz > 0.0)
    aligned_f = aligned_b.to(dt)

    # ------------------------------------------------------------------ stage 5: tilt
    # weight 3.0, only while aligned; saturates at TILT_TARGET so full pour is the goal.
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    tilt = 3.0 * aligned_f * tilt_frac
    # premature tilt (weight 0.5): tilting well past grasp posture when the rim is NOT over the
    # receiver. Real spills are priced by spill_delta, so this stays small.
    tilt_premature = -0.5 * (1.0 - aligned_f) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)

    # ------------------------------------------------------------------ stage 6: bead transfer
    # 50 * Δfrac = 2.5 per bead (20 beads) — one-shot payments, must dominate the shaping stack.
    # 20 * Δfrac = -1.0 per spilled bead: one bead in the target outweighs two lost.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -20.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ constraints
    # Receiver-cup penalties are deliberately SMALL (0.3 + 0.5 = 0.8/step worst case) versus the
    # ~7/step the receiver branch can earn. Last run they were 1.0 + 2.0 = 3.0/step for the rest
    # of the episode after one accidental bump (~-2600 total), and the policy learned to keep the
    # left hand away from the receiver cup entirely (approach_rcv fell from 0.40 to 0.35).
    upright_rcv = -0.3 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # action-rate regularisation (weight 0.02 — tiny, must never compete with any stage term)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus
    success = 10.0 * ctx.success.to(dt)

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "both_grasped": both_grasped,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "both_lifted": both_lifted,
        "bring_together": bring_together,
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
