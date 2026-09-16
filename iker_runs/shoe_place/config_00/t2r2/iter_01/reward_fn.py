import torch
import math


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Place the held shoe on the rack: align -> seat -> LET GO -> withdraw -> settle.

    Previous policy reached `placed & resting` on 30% of steps but held the grip shut
    (open_frac 0.000 there), so `released`/`still`/`stable_count` were unreachable.
    This version pays heavily for the grip opening in exactly that state, removes the
    permanent `dropped` penalty that made any imperfect release catastrophic, and
    un-gates `withdraw` from a condition that could never be satisfied.
    """

    # ---- shared quantities --------------------------------------------------------------
    dist = ctx.keypoint_dist                       # (N,) mean keypoint error to the goal pose
    zero = torch.zeros_like(dist)

    # python-side constants, guarded against degenerate values
    tol = max(float(ctx.place_tolerance), 1e-3)
    rest_tol = max(float(ctx.resting_tol), 1e-3)
    still_sp = max(float(ctx.still_speed), 1e-3)
    rel_r = max(float(ctx.release_radius), 1e-3)
    n_stable = max(int(ctx.stable_steps), 1)

    # grip axis as an "open fraction": 0.0 = recorded grip pose (holding), 1.0 = open pose
    open_frac = torch.clamp((ctx.grip_norm + 1.0) * 0.5, min=0.0, max=1.0)
    ep = torch.clamp(ctx.episode_progress, min=0.0, max=1.0)

    # the state in which letting go is the right thing to do: shoe at the goal pose,
    # sitting at rack height. Reached on 30% of steps by the previous policy.
    seated = ctx.placed & ctx.resting
    seated_f = seated.float()
    handed_over = seated & ctx.released            # ... and the hand is actually off it
    handed_f = handed_over.float()

    # ---- 1. align: carry the shoe onto the 4 target keypoints ---------------------------
    # unchanged: this worked (0.13 -> 0.555 over the previous run). Long-range tanh pull
    # plus a sharp gaussian inside the tolerance for the final gradient.
    coarse = 1.0 - torch.tanh(dist / 0.20)
    fine = torch.exp(-((dist / tol) ** 2))
    align = 2.0 * (0.6 * coarse + 0.4 * fine)          # w=2.0, the transport shaping term

    # ---- 2. progress: same error, normalised by this episode's own starting error -------
    init_err = (ctx.init_keypoints - ctx.target_keypoints).norm(dim=-1).mean(dim=-1)
    progress = 1.0 * torch.clamp(
        (init_err - dist) / torch.clamp(init_err, min=1e-3), min=0.0, max=1.0
    )                                                   # w=1.0, bounded [0,1]

    # ---- 3. seat: put the shoe's lowest point onto the rack top -------------------------
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat_shape = 1.0 - torch.tanh(height_err / rest_tol)
    near_goal = dist <= (2.0 * tol)                     # only near the target, else the
    seat = 1.0 * torch.where(near_goal, seat_shape, zero)   # policy is paid to hover

    # ---- 4. release: THE FIX. w 1.5 -> 4.0, dense in the grip EMA -----------------------
    # one step of a=+1 moves open_frac by 0.4686, i.e. ~+1.9 reward immediately.
    release = 4.0 * torch.where(seated, open_frac, zero)

    # ---- 5. hold_cost: price the local optimum of never letting go ----------------------
    # -1.0 max, ramped with elapsed time so a careful early placement is not punished.
    # Together with `release` the grip axis is worth a 5.0/step swing while seated, yet
    # stays far below the ~4.6/step the policy already earns there, so `seated` stays
    # attractive and the policy has no reason to avoid reaching it.
    hold_cost = -1.0 * torch.where(seated, (1.0 - open_frac) * (0.5 + 0.5 * ep), zero)

    # ---- 6. handsfree: flat dense pay once the shoe stands on the rack alone ------------
    handsfree = 3.0 * handed_f                          # w=3.0, the state the task wants

    # ---- 7. withdraw: back the palm off; gated on `seated` only, NOT on the grip --------
    # the old `open_frac > 0.5` gate was never satisfiable, so this term paid exactly 0
    # for the whole previous run. Gating on `seated` is self-protecting: a hand that still
    # grips cannot retreat without dragging the shoe, which breaks placed/resting.
    withdraw_frac = torch.clamp(ctx.palm_shoe_dist / (rel_r + 0.05), min=0.0, max=1.0)
    clearance = torch.tanh(ctx.palm_gap / 0.12)         # finer, earlier signal than the above
    withdraw = 2.0 * torch.where(seated, 0.5 * withdraw_frac + 0.5 * clearance, zero)

    # ---- 8. settle: let the shoe come to rest once the hand is off it -------------------
    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)
    calm = 1.0 - torch.tanh(lin_sp / still_sp + 0.5 * ang_sp)
    settle = 2.0 * torch.where(handed_over, calm, zero)  # w=2.0, only meaningful after release

    # ---- 9. gates: smooth staircase over the four success predicates --------------------
    # superlinear so the 4th (missing) condition is worth more than the first three:
    # 2 gates -> 0.625, 3 -> 1.41, 4 -> 2.5.
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.released.float() + ctx.still.float()
    )
    gates = 2.5 * (n_gates / 4.0) ** 2

    # ---- 10. stability: dense credit for the 20-step success counter, w 1.0 -> 3.0 ------
    stability = 3.0 * torch.clamp(ctx.stable_count.float() / n_stable, min=0.0, max=1.0)

    # ---- 11. success: terminal bonus, 20 -> 30, scaled up for finishing early -----------
    time_left = torch.clamp(1.0 - ep, min=0.0, max=1.0)
    success = 30.0 * torch.where(ctx.success, 1.0 + 0.5 * time_left, zero)

    # ---- 12. mishandle: the ONLY penalty on the grip axis, deliberately narrow ----------
    # the old -0.5*early_open froze actions[6] entirely (released 0.643 -> 0.098). This
    # fires only on opening that is clearly not a placement: off rack height AND far from
    # the goal pose. Opening near the rack is free to explore.
    bad_open = (~ctx.resting).float() * (dist > (2.0 * tol)).float()
    mishandle = -0.4 * open_frac * bad_open

    # ---- 13. fall: bounded cost for dropping the shoe below the TABLE (not the rack) ----
    # the transport path legitimately passes below rack height, so the table is the
    # reference. No permanent `released & ~placed` bleed any more: that cliff is what made
    # any imperfect release irrational compared with simply never opening the hand.
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -2.0 * torch.tanh(fall / 0.05)

    # ---- 14. action regularisation: small, or the +-0.05 action noise kills exploration -
    d_act = ctx.actions - ctx.prev_actions
    rate = torch.mean(d_act ** 2, dim=-1)               # jitter is what breaks stable_count
    mag = torch.mean(ctx.actions ** 2, dim=-1)
    arm_motion = torch.tanh(torch.mean(ctx.arm_qd ** 2, dim=-1) / 4.0)
    action_reg = -(0.05 * rate + 0.02 * mag + 0.02 * arm_motion)

    components = {
        "align": align,
        "progress": progress,
        "seat": seat,
        "release": release,
        "hold_cost": hold_cost,
        "handsfree": handsfree,
        "withdraw": withdraw,
        "settle": settle,
        "gates": gates,
        "stability": stability,
        "success": success,
        "mishandle": mishandle,
        "fall_penalty": fall_penalty,
        "action_reg": action_reg,
    }

    reward = torch.zeros_like(dist)
    for value in components.values():
        reward = reward + value

    return reward, components
