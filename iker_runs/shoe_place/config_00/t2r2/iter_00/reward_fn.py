import torch
import math


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Place the held shoe on the rack: align -> seat -> release -> withdraw -> hold still."""

    # ---- shared quantities --------------------------------------------------------------
    dist = ctx.keypoint_dist                       # (N,) mean keypoint error to the goal pose
    zero = torch.zeros_like(dist)
    one = torch.ones_like(dist)

    # python-side constants, guarded against degenerate values
    tol = max(float(ctx.place_tolerance), 1e-3)
    rest_tol = max(float(ctx.resting_tol), 1e-3)
    still_sp = max(float(ctx.still_speed), 1e-3)
    n_stable = max(int(ctx.stable_steps), 1)

    # grip axis as an "open fraction": 0.0 = recorded grip pose (holding), 1.0 = open pose
    open_frac = torch.clamp((ctx.grip_norm + 1.0) * 0.5, min=0.0, max=1.0)

    # the state in which letting go is actually the right thing to do
    ready = ctx.placed & ctx.resting

    # ---- 1. align: carry the shoe onto the 4 target keypoints ---------------------------
    # two length scales: a long-range tanh pull (half value ~0.22 m) plus a sharp gaussian
    # that only lives inside the placement tolerance and supplies the final gradient.
    coarse = 1.0 - torch.tanh(dist / 0.20)
    fine = torch.exp(-((dist / tol) ** 2))
    align = 2.0 * (0.6 * coarse + 0.4 * fine)          # w=2.0: the dominant shaping term

    # ---- 2. progress: same error, normalised by this episode's own starting error -------
    init_err = (ctx.init_keypoints - ctx.target_keypoints).norm(dim=-1).mean(dim=-1)
    progress = 1.0 * torch.clamp(
        (init_err - dist) / torch.clamp(init_err, min=1e-3), min=0.0, max=1.0
    )                                                   # w=1.0, bounded [0,1]

    # ---- 3. seat: put the shoe's lowest point onto the rack top -------------------------
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat_shape = 1.0 - torch.tanh(height_err / rest_tol)
    near_goal = dist <= (2.0 * tol)                     # only counts near the target, else the
    seat = 1.0 * torch.where(near_goal, seat_shape, zero)   # policy is paid to hover at rack height

    # ---- 4. release: open the fingers, but only once the shoe is placed and resting -----
    release = 1.5 * torch.where(ready, open_frac, zero)  # w=1.5, dense in the EMA opening

    # ---- 4b. premature release: the main degenerate strategy to price out ---------------
    early_open = torch.where(ready, zero, open_frac)     # opening anywhere else
    dropped = torch.where(ctx.released & (~ctx.placed), one, zero)   # shoe away from palm, not placed
    premature_release = -(0.5 * early_open + 1.0 * dropped)

    # ---- 5. withdraw: back the palm off past release_radius, after letting go -----------
    withdraw_frac = torch.clamp(
        ctx.palm_shoe_dist / (float(ctx.release_radius) + 0.10), min=0.0, max=1.0
    )
    can_withdraw = ready & (open_frac > 0.5)             # never rewarded while still gripping
    withdraw = 1.5 * torch.where(can_withdraw, withdraw_frac, zero)

    # ---- 6. still: set it down gently instead of throwing it ----------------------------
    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)
    calm = 1.0 - torch.tanh(lin_sp / still_sp + 0.5 * ang_sp)
    still = 0.75 * torch.where(ctx.placed, calm, zero)   # w=0.75, secondary to alignment

    # ---- 7. stability: dense credit for the 20-step success counter ---------------------
    stability = 1.0 * torch.clamp(ctx.stable_count.float() / n_stable, min=0.0, max=1.0)

    # ---- 8. success: the terminal bonus, scaled up for finishing early ------------------
    time_left = torch.clamp(1.0 - ctx.episode_progress, min=0.0, max=1.0)
    success = 20.0 * torch.where(ctx.success, 1.0 + 0.5 * time_left, zero)

    # ---- 9. fall: bounded cost for dropping the shoe below the TABLE (not the rack) -----
    # the transport path legitimately passes below rack height, so the table is the reference.
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -2.0 * torch.tanh(fall / 0.05)

    # ---- 10. action regularisation: small, or the +-0.05 action noise kills exploration -
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
        "premature_release": premature_release,
        "withdraw": withdraw,
        "still": still,
        "stability": stability,
        "success": success,
        "fall_penalty": fall_penalty,
        "action_reg": action_reg,
    }

    reward = torch.zeros_like(dist)
    for value in components.values():
        reward = reward + value

    return reward, components
