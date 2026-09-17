import torch
import math


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Place the held shoe on the rack: transport -> seat -> let go -> withdraw -> hold quiet.

    Changes vs the previous reward, driven by the measured rollouts:
      * the release/withdraw chain is learned (open_frac 0.97 where placed & resting), so its
        weights are cut hard -- they now only maintain the behaviour;
      * `align` gains a middle length scale at 2*tol and a sharper fine scale, because the
        previous shaping was nearly flat between 0.03 and 0.10 m, which is where the policy lives
        and where the newly tightened 0.03 m tolerance must now be won;
      * `settle` is re-gated and its temperature sharpened 2.5x -- the old one tracked how often
        its gate was open, not whether the shoe was still (it rose while `still` occupancy fell);
      * success terminates the episode, so the terminal bonus is paid PER REMAINING STEP at a rate
        above the dense per-step ceiling; otherwise hovering one condition short of the counter
        strictly beats finishing, which is what the previous logs show happened.
    """

    # ---- shared quantities ---------------------------------------------------------------
    dist = ctx.keypoint_dist                       # (N,) mean keypoint error to the goal pose
    zero = torch.zeros_like(dist)

    # python-side constants, guarded against degenerate values
    tol = max(float(ctx.place_tolerance), 1e-3)        # 0.03 m: the placement tolerance
    rest_tol = max(float(ctx.resting_tol), 1e-3)
    still_sp = max(float(ctx.still_speed), 1e-3)
    rel_r = max(float(ctx.release_radius), 1e-3)
    n_stable = max(int(ctx.stable_steps), 1)
    ep_steps = max(int(ctx.episode_steps), 1)

    # grip axis as an "open fraction": 0.0 = recorded grip pose (holding), 1.0 = open pose
    open_frac = torch.clamp((ctx.grip_norm + 1.0) * 0.5, 0.0, 1.0)
    ep = torch.clamp(ctx.episode_progress, 0.0, 1.0)

    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)

    seated = ctx.placed & ctx.resting              # shoe at the goal pose, at rack height
    handed = seated & ctx.released                 # ... and the hand is off it
    near_goal = dist <= (3.0 * tol)                # 0.09 m funnel for the height/stillness terms

    # ---- 1. align: three length scales, w=1.5 ---------------------------------------------
    # The previous version had only 0.20 m and tol. Between 3 and 10 cm the 0.20 m term moves
    # just 0.10 total and the Gaussian is numerically dead, so there was almost no gradient in
    # the band the policy actually occupies (measured median 0.046 m). The 2*tol term supplies
    # it; the 0.5*tol term supplies the final centimetre.
    coarse = 1.0 - torch.tanh(dist / 0.20)             # long-range transport pull
    mid = 1.0 - torch.tanh(dist / (2.0 * tol))         # ~0.06 m: the band that must be closed
    fine = torch.exp(-((dist / (0.5 * tol)) ** 2))     # ~0.015 m: margin inside the tolerance
    align = 1.5 * (0.30 * coarse + 0.35 * mid + 0.35 * fine)

    # ---- 2. precision: the WORST keypoint, not the mean, w=0.8 -----------------------------
    # `placed` uses the mean, which can hide one badly rotated corner. Paying the worst corner
    # buys orientation as well as position -- the difference between "next to the target" and
    # "on it".
    worst_err = ctx.keypoint_err.max(dim=-1).values
    precision = 0.8 * torch.exp(-((worst_err / (1.5 * tol)) ** 2))

    # ---- 3. seat: put the shoe's lowest point onto the rack top, w=0.5 ---------------------
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat = 0.5 * torch.where(near_goal, 1.0 - torch.tanh(height_err / rest_tol), zero)

    # ---- 4. release: maintenance only, w 4.0 -> 0.8 ---------------------------------------
    # open_frac is already 0.97 in this state; this term no longer has to discover anything.
    release = 0.8 * torch.where(seated, open_frac, zero)

    # ---- 5. handsfree: paid only in proportion to the margin inside the tolerance, w=0.6 ---
    # Flat pay for `placed & resting & released` goes dead the instant the shoe is nominally
    # inside 0.03 m. With +-0.02 m observation noise the policy needs margin, so this varies
    # from 0.13 to 1.0 WITHIN the tolerance band and keeps a gradient where the old one had none.
    margin = torch.exp(-((dist / (0.7 * tol)) ** 2))
    handsfree = 0.6 * torch.where(handed, margin, zero)

    # ---- 6. withdraw: maintenance only, saturating at the release radius, w 2.0 -> 0.4 -----
    # Capped at 1.0 so there is no pay for fleeing further than `released` requires.
    withdraw = 0.4 * torch.where(
        seated, torch.clamp(ctx.palm_shoe_dist / rel_r, 0.0, 1.0), zero
    )

    # ---- 7. settle: THE FIX for `still`, w=1.2 --------------------------------------------
    # Old shape paid 0.24 at exactly the failure threshold and was gated on `released`, so its
    # logged mean tracked gate occupancy rather than stillness (it rose 0.126 -> 0.699 while
    # measured `still` fell 0.811 -> 0.614). Temperature is now 0.4*still_speed: ~0.01 at the
    # threshold, 0.24 at 40% of it, 0.76 at 10% of it -- it only pays for real margin. Gated on
    # the shoe resting near the goal, not on the hand being away.
    calm = 1.0 - torch.tanh(lin_sp / (0.4 * still_sp) + ang_sp / 3.0)
    settle = 1.2 * torch.where(ctx.resting & near_goal, calm, zero)

    # ---- 8. gates: smooth superlinear staircase over the four predicates, w=1.0 ------------
    # 2 gates -> 0.25, 3 -> 0.56, 4 -> 1.0, so the last missing condition is worth the most.
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.released.float() + ctx.still.float()
    )
    gates = 1.0 * (n_gates / 4.0) ** 2

    # ---- 9. stability: the direct success proxy, now the largest dense term, max 3.0 -------
    # Previously 3.0 weight but only 0.138 realised. Linear part gives gradient from the first
    # counted step; the quadratic part pulls the last few counts toward 20/30.
    frac = torch.clamp(ctx.stable_count.float() / n_stable, 0.0, 1.0)
    stability = 2.2 * frac + 0.8 * frac ** 2

    # ---- 10. success: paid PER REMAINING STEP -------------------------------------------
    # Success ends the episode, so a flat bonus makes finishing irrational: at ~8.35/step the
    # old policy forfeited ~670 reward to collect 30. The dense ceiling above is ~9.8/step, so
    # 12.0 per remaining step makes finishing strictly better than hovering, and finishing
    # sooner strictly better than finishing later.
    steps_left = torch.clamp(1.0 - ep, 0.0, 1.0) * float(ep_steps)
    success = torch.where(ctx.success, 40.0 + 12.0 * steps_left, zero)

    # ---- 11. fall: bounded safety guard against dropping the shoe below the TABLE ----------
    # Kept although small (-0.017): it insures the ~9% drop rate, it is not a shaping term.
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -1.5 * torch.tanh(fall / 0.05)

    # ---- 12. action regularisation: rate term raised, jitter is what breaks `still` --------
    # Still small overall, or the +-0.05 action noise would dominate and kill exploration.
    d_act = ctx.actions - ctx.prev_actions
    rate = torch.mean(d_act ** 2, dim=-1)
    mag = torch.mean(ctx.actions ** 2, dim=-1)
    arm_motion = torch.tanh(torch.mean(ctx.arm_qd ** 2, dim=-1) / 4.0)
    action_reg = -(0.08 * rate + 0.02 * mag + 0.03 * arm_motion)

    components = {
        "align": align,
        "precision": precision,
        "seat": seat,
        "release": release,
        "handsfree": handsfree,
        "withdraw": withdraw,
        "settle": settle,
        "gates": gates,
        "stability": stability,
        "success": success,
        "fall_penalty": fall_penalty,
        "action_reg": action_reg,
    }

    reward = torch.zeros_like(dist)
    for value in components.values():
        reward = reward + value

    return reward, components
