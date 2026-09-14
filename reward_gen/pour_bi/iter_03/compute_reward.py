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


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET_SRC = 0.20    # [m] source cup must end up clearly higher than the receiver: its tilted lip
    LIFT_TARGET_RCV = 0.10    # [m] hangs ~5-8 cm below its origin and the receiver rim is above the receiver origin
    LIFT_GATE = 0.05          # [m] minimum lift before the pour stack is rewarded
    LIP_DEADBAND = 0.01       # [m] lip within 1 cm of the receiver mouth centre counts as centred (pose noise)
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1 (wide so the gradient is felt early)
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay (10 cm above -> 0.37): high drop-pours spill
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, saturate slightly above
    TILT_FREE = 0.8           # [rad] grasp posture alone tilts the cup 20-35 deg; only charge beyond 46 deg
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties (knowledge item 11)
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes (e.g. finger on table while grasping) are free

    # ------------------------------------------------------------------ clearance factors
    # clear in [0,1]: 1 when the cups do not touch, -> 0 as the cup-cup force grows. Multiplies the whole
    # pour stack (aim + tilt + success) so a bump costs ~6-16/step instead of a small additive penalty.
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ stage 1: approach (unchanged — solved)
    # weight 1.0 per arm, exp(-4d): wide basin, saturates ~0.68 in the grasp posture.
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp (unchanged — solved)
    # 1.0 for the contact-established flag + 0.5 * closure weighted by proximity exp(-6d).
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    grasp_src = 1.0 * g_src + 0.5 * prox_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * prox_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)
    # weight 1.0 bonus for holding BOTH cups: the second grasp is worth more than the first.
    both_grasped_b = ctx.src_grasped & ctx.rcv_grasped
    both_grasped = 1.0 * both_grasped_b.to(dt)

    # ------------------------------------------------------------------ stage 3: lift
    # weight 2.0 per arm — larger than grasp so holding on and lifting beats holding on the table.
    # ASYMMETRIC targets (new): the source saturates at 0.20 m, the receiver at 0.10 m, so the lift stage
    # already produces most of the height difference the pour pose needs (iter_02 lifted both to ~0.14 m).
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    lift_src = 2.0 * g_src * lift_frac_src
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    src_lifted_f = src_lifted_b.to(dt)
    both_lifted_f = both_lifted_b.to(dt)
    # weight 1.0 bonus for both cups in the air
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ pouring-lip geometry (new)
    # iter_02 measured alignment between MOUTH CENTRES; with a near-upright cup that is only reachable by
    # nesting / bumping the cups, so the tilt gate never opened and the policy parked 13 cm away.
    # The point that matters is the lowest rim point ("lip"): it swings over the receiver by TILTING,
    # while the cup body stays outside the receiver. Aim terms are defined on the lip.
    lip = _rim_low_point(ctx.src_cup_mouth_pos, ctx.src_cup_up, ctx.cup_radius)   # (N,3)
    lip_delta = lip - ctx.rcv_cup_mouth_pos                                         # (N,3)
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)                                   # (N,) horizontal miss
    dz_lip = lip_delta[:, 2]                                                        # (N,) lip height above rcv rim
    # above: wide ramp -6 cm -> +2 cm (gradient toward "source higher" is felt long before the pose is right).
    above = torch.clamp((dz_lip - DZ_LOW) / (DZ_HIGH - DZ_LOW), 0.0, 1.0)
    # high: free up to 4 cm above the rim, then gentle decay — a 10 cm drop-pour (iter_01: 7 % spill) pays 0.37.
    high = torch.exp(-(torch.clamp(dz_lip - DZ_FREE, min=0.0) / DZ_SIGMA) ** 2)
    aim_z = above * high
    # aim_xy: exp(-6 d) with a 1 cm deadband — 5 cm miss pays 0.79, 13 cm (the iter_02 fixed point) pays 0.49.
    aim_xy = torch.exp(-6.0 * torch.clamp(lip_xy - LIP_DEADBAND, min=0.0))
    aim_soft = aim_xy * aim_z                                                       # (N,) in [0,1]

    # ------------------------------------------------------------------ stage 4: bring cups together
    # bring_together, weight 1.0: once the SOURCE cup is lifted, pull its lip over the receiver rim with a wide
    # basin exp(-3d); aim_z included so hovering upright 20 cm above the receiver is not a free 1.0.
    bring_together = 1.0 * src_lifted_f * not_nested_f * torch.exp(-3.0 * lip_xy) * aim_z
    # aim, weight 3.0 (replaces align): both lifted, not nested, cups clear. No hard xy / z gate any more —
    # the soft product gives a monotone path from "side by side, 30 deg" to "lip over the rim, 110 deg".
    # Must beat lift (2+2 banked) + bring_together (1.0).
    aim = 3.0 * both_lifted_f * not_nested_f * clear * aim_soft

    # ------------------------------------------------------------------ stage 5: tilt
    # beads_left in [0,1]: beads still in play (in source or already in receiver). Spilled beads shrink the
    # tilt reward permanently, so "dump the beads on the table, then keep tilting" earns nothing.
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    # weight 3.0, scaled by the SOFT aim factor instead of a hard aligned flag: at the iter_02 fixed point
    # (aim ~0.4) tilting from 30 to 60 deg already pays ~+0.4/step and moves the lip closer, so the
    # chicken-and-egg deadlock is gone. At the pour pose aim -> ~0.9 and this pays ~2.7/step.
    tilt = 3.0 * both_lifted_f * not_nested_f * clear * beads_left * aim_soft * tilt_frac
    # premature tilt (weight 0.5): tilting well past grasp posture while the lip is NOT over the receiver.
    # Weighted by (1 - aim_soft) so it vanishes as the lip comes over the rim; spills are priced by spill_delta.
    tilt_premature = -0.5 * (1.0 - aim_soft) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)
    # pour_rate, weight 0.3: gentle cost on fast tilting while aimed — a slow pour spills less.
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    pour_rate = -0.3 * aim_soft * torch.tanh(src_ang_speed / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (unchanged)
    # 50 * dfrac = 2.5 per bead (20 beads) — one-shot payments, must dominate the shaping stack.
    # 30 * dfrac = -1.5 per spilled bead: one landed bead still outweighs one lost.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ collision / clearance (unchanged — real-robot safety)
    # cup_contact, weight 1.0: bounded price on cup-cup force on top of the lost-stage gate.
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    # hand_foreign_*, weight 0.5 each with a 1 N deadband — small versus the ~3.5/step each arm earns from
    # grasp+lift, so the hand is not taught to avoid its own cup (iter_00 failure mode).
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    hand_foreign_src = -0.5 * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -0.5 * torch.tanh(f_rcv / FORCE_SCALE)
    # closing_speed, weight 1.0: rate at which the two cups approach each other inside 30 cm
    # (10 cm/s deadband, 30 cm/s costs ~0.76).
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos                                   # (N,3)
    cup_dist = torch.norm(rel, dim=-1)                                        # (N,)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel                         # (N,3)
    v_close = torch.sum(v_rel * u, dim=-1)                                    # >0 = approaching
    near_f = (cup_dist < NEAR_DIST).to(dt)
    closing_speed = -1.0 * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    # nested, weight 0.5: small explicit price; the gates above already remove all pour income.
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints (unchanged)
    # Receiver-cup penalties stay SMALL (0.3 + 0.5 = 0.8/step worst case) versus the ~7/step the receiver
    # branch earns — larger values taught the left hand to avoid the cup (iter_00).
    upright_rcv = -0.3 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # action-rate regularisation (weight 0.02 — tiny, must never compete with any stage term)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus
    # weight 10.0, withheld while the cups touch: success is defined by the env, the bonus is not.
    success = 10.0 * ctx.success.to(dt) * clear

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
        "aim": aim,
        "tilt": tilt,
        "tilt_premature": tilt_premature,
        "pour_rate": pour_rate,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
        "cup_contact": cup_contact,
        "hand_foreign_src": hand_foreign_src,
        "hand_foreign_rcv": hand_foreign_rcv,
        "closing_speed": closing_speed,
        "nested": nested,
        "upright_rcv": upright_rcv,
        "drop": drop,
        "action_rate": action_rate,
        "success": success,
    }

    reward = zeros
    for v in components.values():
        reward = reward + v

    return reward, components
