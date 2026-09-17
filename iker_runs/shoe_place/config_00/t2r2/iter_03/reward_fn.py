import torch
import math


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Place the held shoe on the rack: transport -> seat -> let go -> return home -> hold quiet.

    Changes vs the previous reward (driven by the measured rollouts and the new `home` condition):
      * NEW home_return: after the shoe rests near the goal AND the grip is open, pull the palm back
        to home_palm_pos (previously nothing depended on where the hand went; the arm was lifted);
      * NEW home_hold: flat pay when placed & resting & released & home hold together;
      * gates counts five predicates (home added);
      * success per remaining step raised 12 -> 16 to stay above the raised dense ceiling.
    Transport / seat / release / settle terms are kept as they were: they are learned.
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

    # ---- 8. home_return: NEW, the fix for the lifted arm, max 2.5 --------------------------
    # `released` alone is NOT proof of letting go (it was 0.995 at epoch 1, before any shoe
    # reached the rack: the palm is >0.15 m from the shoe origin even while holding). So the gate
    # also needs the grip opened past half and the shoe resting within 9 cm of the goal. Carrying
    # the shoe home would pull it off the rack and close its own gate.
    # Scaled by placement quality so "drop it roughly and go home" pays little:
    # 0.64 at 3 cm, 0.29 at 5 cm, ~0.02 at 9 cm.
    # Three length scales over the ~0.45 m return: 0.25 m long pull, 0.10 m middle, and a
    # Gaussian at home_radius for margin against +-0.02 m observation noise.
    hd = ctx.palm_home_dist
    let_go = torch.clamp((open_frac - 0.5) / 0.4, 0.0, 1.0)
    quality = torch.exp(-((dist / (1.5 * tol)) ** 2))
    home_gate = (ctx.resting & near_goal & ctx.released).float() * let_go * quality
    home_shape = (
        0.40 * (1.0 - torch.tanh(hd / 0.25))
        + 0.30 * (1.0 - torch.tanh(hd / 0.10))
        + 0.30 * torch.exp(-((hd / home_r) ** 2))
    )
    home_return = 2.5 * home_gate * home_shape

    # ---- 9. home_hold: NEW, direct dense proxy of the fifth condition, w=1.0 ---------------
    home_hold = 1.0 * (ctx.placed & ctx.resting & ctx.released & ctx.home).float()

    # ---- 10. gates: smooth superlinear staircase over FIVE predicates, w=1.0 ---------------
    # 3 -> 0.36, 4 -> 0.64, 5 -> 1.0: the last missing condition (now usually home) is worth most.
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.released.float()
        + ctx.still.float() + ctx.home.float()
    )
    gates = 1.0 * (n_gates / 5.0) ** 2

    # ---- 11. stability: direct success proxy, max 3.0 (stable_count now includes home) -----
    frac = torch.clamp(ctx.stable_count.float() / n_stable, 0.0, 1.0)
    stability = 2.2 * frac + 0.8 * frac ** 2

    # ---- 12. success: paid PER REMAINING STEP, rate 12 -> 16 --------------------------------
    # Dense ceiling is now ~13.3/step (added home_return 2.5 + home_hold 1.0). 16 per remaining
    # step keeps finishing strictly better than hovering, and sooner better than later.
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
