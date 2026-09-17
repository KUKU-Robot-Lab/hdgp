import torch
import math


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Place the held shoe on the rack: transport -> align -> seat -> open hand (env retracts arm)."""

    # ---- shared quantities ---------------------------------------------------------------
    dist = ctx.keypoint_dist                       # (N,) mean keypoint error to the goal pose
    zero = torch.zeros_like(dist)

    tol = max(float(ctx.place_tolerance), 1e-3)        # 0.03 m placement tolerance
    rest_tol = max(float(ctx.resting_tol), 1e-3)
    still_sp = max(float(ctx.still_speed), 1e-3)
    open_min = min(max(float(ctx.retract_open_min), 1e-3), 1.0)  # 0.9 takeover threshold
    n_stable = max(int(ctx.stable_steps), 1)
    ep_steps = max(int(ctx.episode_steps), 1)

    # grip axis as an "open fraction": 0.0 = grip pose (holding), 1.0 = open pose
    open_frac = torch.clamp((ctx.grip_norm + 1.0) * 0.5, 0.0, 1.0)
    open_enough = open_frac >= open_min
    ep = torch.clamp(ctx.episode_progress, 0.0, 1.0)

    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)

    seated = ctx.placed & ctx.resting              # shoe at the goal pose, at rack height
    ready = seated & ctx.still & open_enough       # exactly the environment's takeover trigger
    near_goal = dist <= (3.0 * tol)                # 0.09 m funnel for height/stillness terms

    # ---- 1. align: three length scales, w=1.5 (unchanged, learned) -----------------------
    coarse = 1.0 - torch.tanh(dist / 0.20)
    mid = 1.0 - torch.tanh(dist / (2.0 * tol))
    fine = torch.exp(-((dist / (0.5 * tol)) ** 2))
    align = 1.5 * (0.30 * coarse + 0.35 * mid + 0.35 * fine)

    # ---- 2. precision: worst keypoint, w=0.8 (unchanged) ----------------------------------
    worst_err = ctx.keypoint_err.max(dim=-1).values
    precision = 0.8 * torch.exp(-((worst_err / (1.5 * tol)) ** 2))

    # ---- 3. lock: NEW, max 1.2 -------------------------------------------------------------
    # 0.7 step on `placed` (value jump at the 3 cm boundary where the median policy stops)
    # + 0.5 narrow Gaussian (1.2 cm scale) for gradient well inside the tolerance.
    lock = 0.7 * ctx.placed.float() + 0.5 * torch.exp(-((dist / (0.4 * tol)) ** 2))

    # ---- 4. seat: lowest point onto the rack top, w=0.5 (unchanged) -----------------------
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat = 0.5 * torch.where(near_goal, 1.0 - torch.tanh(height_err / rest_tol), zero)

    # ---- 5. settle: stillness margin near the goal, w=1.2 (unchanged) ---------------------
    calm = 1.0 - torch.tanh(lin_sp / (0.4 * still_sp) + ang_sp / 3.0)
    settle = 1.2 * torch.where(ctx.resting & near_goal, calm, zero)

    # ---- 6. release: RESHAPED, w=1.2 -------------------------------------------------------
    # Only when the shoe is seated on target; saturates at the takeover threshold (0.9 open),
    # so opening off-target earns nothing and the last bit to 0.9 is fully paid.
    release = 1.2 * torch.where(seated, torch.clamp(open_frac / open_min, 0.0, 1.0), zero)

    # ---- 7. ready: NEW, max 3.0 ------------------------------------------------------------
    # Paid on the takeover condition itself (placed & resting & still & open >= 0.9).
    # Cannot be hovered before the trigger (meeting it starts the takeover); afterwards it keeps
    # the value of triggering continuous while the shoe stays placed. The margin part (1.0)
    # favours shoes set well inside 3 cm, which stay placed while the arm withdraws.
    margin = torch.exp(-((dist / (0.7 * tol)) ** 2))
    ready_r = 3.0 * torch.where(ready, (2.0 + margin) / 3.0, zero)

    # ---- 8. gates: smooth superlinear staircase, w=1.0 -------------------------------------
    # `released` replaced by `open_enough` (released is true even while holding);
    # `home` stays since it is part of success and only reachable via the takeover.
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.still.float()
        + open_enough.float() + ctx.home.float()
    )
    gates = 1.0 * (n_gates / 5.0) ** 2

    # ---- 9. stability: direct success proxy, max 3.0 (unchanged) --------------------------
    frac = torch.clamp(ctx.stable_count.float() / n_stable, 0.0, 1.0)
    stability = 2.2 * frac + 0.8 * frac ** 2

    # ---- 10. success: paid PER REMAINING STEP, rate 16 (unchanged) --------------------------
    # Dense ceiling is now ~13.4/step (1.5+0.8+1.2+0.5+1.2+1.2+3.0+1.0+3.0); 16 per remaining
    # step keeps finishing strictly better than hovering, and sooner better than later.
    steps_left = torch.clamp(1.0 - ep, 0.0, 1.0) * float(ep_steps)
    success = torch.where(ctx.success, 40.0 + 16.0 * steps_left, zero)

    # ---- 11. fall: bounded guard against dropping the shoe below the table (unchanged) -----
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -1.5 * torch.tanh(fall / 0.05)

    # ---- 12. action regularisation: small; masked while the env drives the arm -------------
    d_act = ctx.actions - ctx.prev_actions
    rate = torch.mean(d_act ** 2, dim=-1)
    mag = torch.mean(ctx.actions ** 2, dim=-1)
    arm_motion = torch.tanh(torch.mean(ctx.arm_qd ** 2, dim=-1) / 4.0)
    reg = -(0.08 * rate + 0.02 * mag + 0.03 * arm_motion)
    action_reg = torch.where(ctx.retracting, zero, reg)

    components = {
        "align": align,
        "precision": precision,
        "lock": lock,
        "seat": seat,
        "settle": settle,
        "release": release,
        "ready": ready_r,
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
