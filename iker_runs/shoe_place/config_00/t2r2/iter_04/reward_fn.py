import torch
import math


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Place the held shoe on the rack: transport -> seat -> let go -> return home -> hold quiet.

    Changes vs the previous reward (only the return stage; learned terms are untouched):
      * home_return shape: a constant-slope linear ramp over 0.9 m dominates, so a 0.02 m step
        toward home pays ~0.037 at 0.6 m (was 0.003); tanh(0.15) + Gaussian(home_radius) finish;
      * home_return gate: `released` dropped (true even while holding), let-go ramp from 30% to 70%
        open, softer placement-quality factor (0.89 at 3 cm instead of 0.64).
    """

    # ---- shared quantities ---------------------------------------------------------------
    dist = ctx.keypoint_dist                       # (N,) mean keypoint error to the goal pose
    zero = torch.zeros_like(dist)

    tol = max(float(ctx.place_tolerance), 1e-3)        # 0.03 m placement tolerance
    rest_tol = max(float(ctx.resting_tol), 1e-3)
    still_sp = max(float(ctx.still_speed), 1e-3)
    rel_r = max(float(ctx.release_radius), 1e-3)
    home_r = max(float(ctx.home_radius), 1e-3)         # 0.05 m
    n_stable = max(int(ctx.stable_steps), 1)
    ep_steps = max(int(ctx.episode_steps), 1)

    # grip axis as an "open fraction": 0.0 = grip pose (holding), 1.0 = open pose
    open_frac = torch.clamp((ctx.grip_norm + 1.0) * 0.5, 0.0, 1.0)
    ep = torch.clamp(ctx.episode_progress, 0.0, 1.0)

    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)

    seated = ctx.placed & ctx.resting              # shoe at the goal pose, at rack height
    handed = seated & ctx.released                 # ... and the palm is away from it
    near_goal = dist <= (3.0 * tol)                # 0.09 m funnel for height/stillness/home terms

    # ---- 1. align: three length scales, w=1.5 (unchanged) ---------------------------------
    coarse = 1.0 - torch.tanh(dist / 0.20)
    mid = 1.0 - torch.tanh(dist / (2.0 * tol))
    fine = torch.exp(-((dist / (0.5 * tol)) ** 2))
    align = 1.5 * (0.30 * coarse + 0.35 * mid + 0.35 * fine)

    # ---- 2. precision: worst keypoint, w=0.8 (unchanged) ----------------------------------
    worst_err = ctx.keypoint_err.max(dim=-1).values
    precision = 0.8 * torch.exp(-((worst_err / (1.5 * tol)) ** 2))

    # ---- 3. seat: lowest point onto the rack top, w=0.5 (unchanged) -----------------------
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat = 0.5 * torch.where(near_goal, 1.0 - torch.tanh(height_err / rest_tol), zero)

    # ---- 4. release: maintenance, w=0.8 (unchanged) ---------------------------------------
    release = 0.8 * torch.where(seated, open_frac, zero)

    # ---- 5. handsfree: margin inside tolerance once hand is off, w=0.6 (unchanged) ---------
    margin = torch.exp(-((dist / (0.7 * tol)) ** 2))
    handsfree = 0.6 * torch.where(handed, margin, zero)

    # ---- 6. withdraw: saturating at the release radius, w=0.4 (unchanged) -----------------
    withdraw = 0.4 * torch.where(
        seated, torch.clamp(ctx.palm_shoe_dist / rel_r, 0.0, 1.0), zero
    )

    # ---- 7. settle: real stillness margin near the goal, w=1.2 (unchanged) ----------------
    calm = 1.0 - torch.tanh(lin_sp / (0.4 * still_sp) + ang_sp / 3.0)
    settle = 1.2 * torch.where(ctx.resting & near_goal, calm, zero)

    # ---- 8. home_return: REWORKED, max 3.0 -------------------------------------------------
    # Gate: shoe resting within 9 cm of the goal (hard) x grip let go (ramp 30%..70% open)
    # x soft placement quality (0.89 at 3 cm, 0.73 at 5 cm, 0.37 at 9 cm).
    # `released` is not used: it is true even while holding (palm > 0.15 m from shoe origin).
    # Carrying the shoe home drags it out of the funnel and closes the gate by itself.
    hd = ctx.palm_home_dist
    let_go = torch.clamp((open_frac - 0.3) / 0.4, 0.0, 1.0)
    quality = torch.exp(-((dist / (3.0 * tol)) ** 2))
    home_gate = (ctx.resting & near_goal).float() * let_go * quality
    # Shape: 0.55 linear ramp over 0.9 m (constant slope where the return starts, 0.3..0.6 m),
    # 0.25 tanh at 0.15 m (last stretch), 0.20 Gaussian at home_radius (margin vs +-0.02 noise).
    # Values: 0.6 m -> 0.11, 0.45 m -> 0.22, 0.3 m -> 0.34, 0.1 m -> 0.58, 0 m -> 1.0 (x3.0).
    home_shape = (
        0.55 * torch.clamp(1.0 - hd / 0.9, 0.0, 1.0)
        + 0.25 * (1.0 - torch.tanh(hd / 0.15))
        + 0.20 * torch.exp(-((hd / home_r) ** 2))
    )
    home_return = 3.0 * home_gate * home_shape

    # ---- 9. home_hold: direct dense proxy of the fifth condition, w=1.0 (unchanged) --------
    home_hold = 1.0 * (ctx.placed & ctx.resting & ctx.released & ctx.home).float()

    # ---- 10. gates: smooth superlinear staircase over FIVE predicates, w=1.0 (unchanged) ---
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.released.float()
        + ctx.still.float() + ctx.home.float()
    )
    gates = 1.0 * (n_gates / 5.0) ** 2

    # ---- 11. stability: direct success proxy, max 3.0 (unchanged) --------------------------
    frac = torch.clamp(ctx.stable_count.float() / n_stable, 0.0, 1.0)
    stability = 2.2 * frac + 0.8 * frac ** 2

    # ---- 12. success: paid PER REMAINING STEP, rate 16 (unchanged) --------------------------
    # Dense ceiling is ~13.8/step (home_return 2.5 -> 3.0); 16 per remaining step keeps
    # finishing strictly better than hovering, and sooner better than later.
    steps_left = torch.clamp(1.0 - ep, 0.0, 1.0) * float(ep_steps)
    success = torch.where(ctx.success, 40.0 + 16.0 * steps_left, zero)

    # ---- 13. fall: bounded guard against dropping the shoe below the table (unchanged) -----
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -1.5 * torch.tanh(fall / 0.05)

    # ---- 14. action regularisation (unchanged; small so +-0.05 action noise cannot dominate) ---
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
        "home_return": home_return,
        "home_hold": home_hold,
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
