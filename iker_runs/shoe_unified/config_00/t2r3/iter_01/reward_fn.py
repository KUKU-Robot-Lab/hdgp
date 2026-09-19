import torch
import math


def _unit(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise unit vector; safe for zero rows."""
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def compute_reward(ctx) -> "tuple[torch.Tensor, dict[str, torch.Tensor]]":
    dev = ctx.palm_pos.device
    dt = ctx.palm_pos.dtype
    n = ctx.palm_pos.shape[0]
    zero = torch.zeros(n, device=dev, dtype=dt)

    # ---------------- weights -------------------------------------------------------------
    W_APPROACH = 0.8    # reaching is already learned; kept small so it cannot anchor the policy
    W_GRASP = 1.5       # wrap-around grasp quality (top-of-shoe contact earns ~0.1 of this)
    W_LIFT = 3.0        # raised from 1.5: getting the shoe off the table is the missing skill
    W_HOLD = 1.0        # anti-drop insurance during the carry
    W_TRANSPORT = 2.5   # coarse travel to the rack (unchanged, already learned)
    W_ALIGN = 2.0       # fine 6-DOF keypoint match (unchanged, already learned)
    W_SETTLE = 1.5      # at rack height and not moving
    W_PLACED = W_GRASP + W_HOLD   # replaces grasp+hold once placed, so placing never loses grip money
    W_OPEN = 3.0        # opening the hand (ctx.open_frac, the takeover's own measure) while placed
    W_TRIGGER = 2.0     # every step the takeover condition holds (or the takeover is running)
    W_STABLE = 4.0      # every step that counts toward stable_count
    W_SUCCESS = 60.0    # terminal bonus floor
    # Upper bound of the per-step positive reward; the success bonus pays this for every step the
    # episode is cut short, so succeeding is never worse than lingering on the rack.
    R_RATE = (W_LIFT + W_TRANSPORT + W_ALIGN + W_SETTLE + W_PLACED + W_OPEN
              + W_TRIGGER + W_STABLE)
    W_PEN = 1.0         # drop / shove / table / rack penalties (each sub-term <= ~0.6)

    busy = ~ctx.retracting                  # the policy only controls the robot here
    shoe_speed = ctx.shoe_lin_vel.norm(dim=-1)

    # ---------------- contact facts --------------------------------------------------------
    force = ctx.link_shoe_force                                   # (N,5,3)
    touch = force > 0.2                                           # (N,5,3) bool
    n_links = touch.to(dt).sum(dim=(1, 2))                        # 0..15 touching finger links
    palm_touch = ctx.palm_shoe_force > 0.2
    thumb_touch = touch[:, 0].any(dim=-1)
    other_fingers = touch[:, 1:].any(dim=-1).to(dt).sum(dim=-1)   # 0..4 non-thumb fingers touching
    oppose = (thumb_touch & (other_fingers >= 2.0)).to(dt)        # thumb against >= 2 fingers

    # A real grasp of this shoe touches 6-10 links; convex so 1-3 tip contacts are worth little.
    contact_q = (n_links / 8.0).clamp(0.0, 1.0) ** 1.5

    # Wrap: finger links close to the shoe AND below its mid-height, i.e. on the sides/underneath
    # rather than resting on top (the measured failure).
    shoe_top_z = ctx.shoe_surface[..., 2].max(dim=-1).values      # (N,)
    mid_z = 0.5 * (ctx.shoe_bottom_z + shoe_top_z)                # (N,)
    link_z = ctx.link_pos[..., 2]                                 # (N,5,3)
    near_s = torch.exp(-ctx.link_shoe_gap.clamp_min(0.0) / 0.01)  # 1 in contact, ~0.37 at 1 cm
    below_s = torch.sigmoid((mid_z[:, None, None] - link_z) / 0.01)
    wrap = ((near_s * below_s).sum(dim=(1, 2)) / 6.0).clamp(0.0, 1.0)

    tip_gap = ctx.link_shoe_gap[:, :, 2].mean(dim=-1)

    # Own "still in the hand" predicate (held/latched go false away from the pick spot because
    # dz_free reads 0 there). Needs the thumb plus >= 3 contacts, so tips on top do not count.
    secure = thumb_touch \
        & ((n_links + palm_touch.to(dt)) >= 3.0) \
        & (ctx.palm_shoe_dist < ctx.hold_radius) \
        & (ctx.slip_speed < 3.0 * ctx.hold_slip_speed)
    in_hand = secure | ctx.held
    lifted = ctx.shoe_bottom_z > (ctx.table_top_z + 0.02)

    placed_ok = ctx.placed & ctx.resting        # a genuinely finished placement on the rack

    # ---------------- 1. approach ----------------------------------------------------------
    face = 0.5 * (1.0 + (ctx.palm_normal * _unit(ctx.shoe_pos - ctx.palm_pos)).sum(dim=-1))
    r_app = 0.45 * (1.0 - torch.tanh(5.0 * ctx.palm_gap)) \
        + 0.30 * (1.0 - torch.tanh(5.0 * tip_gap)) \
        + 0.25 * face.clamp(0.0, 1.0)
    m_app = (~in_hand) & (~placed_ok) & busy
    approach = W_APPROACH * torch.where(m_app, r_app, zero)

    # ---------------- 2. grasp (wrap-around, not on top) -----------------------------------
    thumb = (ctx.thumb_curl / 0.6).clamp(0.0, 1.0)
    r_grasp = 0.40 * wrap + 0.30 * contact_q + 0.15 * oppose + 0.15 * thumb * contact_q
    m_grasp = (ctx.palm_gap < 0.12) & (~placed_ok) & busy
    grasp = W_GRASP * torch.where(m_grasp, r_grasp, zero)

    # ---------------- 3. lift ----------------------------------------------------------------
    # Height above the table while in the hand; also paid while placed on the (raised) rack so
    # that setting the shoe down does not remove it.
    h_tab = ((ctx.shoe_bottom_z - ctx.table_top_z) / 0.10).clamp(0.0, 1.0)
    h_free = (ctx.dz_free / max(ctx.lift_height, 1e-3)).clamp(0.0, 1.0)
    r_lift = torch.maximum(0.7 * h_tab + 0.3 * h_free, h_tab)
    lift = W_LIFT * torch.where(in_hand | placed_ok, r_lift, zero)

    # ---------------- 4. hold (carry insurance; OFF once placed so it never rewards gripping on the rack)
    r_hold = 0.40 * contact_q \
        + 0.35 * (1.0 - torch.tanh(ctx.slip_speed / max(ctx.hold_slip_speed, 1e-3))) \
        + 0.25 * (1.0 - torch.tanh(8.0 * ctx.palm_gap))
    hold = W_HOLD * torch.where(in_hand & lifted & (~placed_ok) & busy, r_hold, zero)

    # ---------------- 5. transport (unchanged) ---------------------------------------------
    transport = W_TRANSPORT * torch.where(
        lifted, 1.0 - torch.tanh(2.5 * ctx.keypoint_dist), zero)

    # ---------------- 6. align (unchanged) -------------------------------------------------
    worst = ctx.keypoint_err.max(dim=-1).values
    r_align = 0.6 * (1.0 - torch.tanh(12.0 * ctx.keypoint_dist)) \
        + 0.4 * (1.0 - torch.tanh(8.0 * worst))
    align = W_ALIGN * torch.where(lifted, r_align, zero)

    # ---------------- 7. settle ------------------------------------------------------------
    dz_rest = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    r_settle = 0.5 * (1.0 - torch.tanh(dz_rest / max(ctx.resting_tol, 1e-3))) \
        + 0.5 * (1.0 - torch.tanh(shoe_speed / max(ctx.still_speed, 1e-3)))
    m_settle = ctx.keypoint_dist < (3.0 * ctx.place_tolerance)
    settle = W_SETTLE * torch.where(m_settle, r_settle, zero)

    # ---------------- 8. placed base + open the hand + takeover trigger --------------------
    # placed: constant that replaces grasp+hold (same size) once the shoe sits on target.
    placed = W_PLACED * placed_ok.to(dt)
    # open: ctx.open_frac is EXACTLY what the takeover compares with retract_open_min; convex so
    # the last stretch to 0.9 is worth the most. Maximal while the environment is retracting.
    of = ctx.open_frac.clamp(0.0, 1.0)
    r_open = 0.4 * of + 0.6 * (of / max(ctx.retract_open_min, 1e-3)).clamp(0.0, 1.0) ** 2
    r_open = torch.where(ctx.retracting, torch.ones_like(r_open), r_open)
    release = W_OPEN * torch.where(placed_ok, r_open, zero)
    # trigger: the takeover condition itself (placed, resting, still, open enough), or the
    # takeover already running. Paying the maximum during retracting makes the hand-over the
    # best reachable state for the policy.
    cond = placed_ok & ctx.still & (ctx.open_frac >= ctx.retract_open_min)
    trigger = W_TRIGGER * ((cond | ctx.retracting) & placed_ok).to(dt)

    # ---------------- 9. stable + success ---------------------------------------------------
    good_step = ctx.placed & ctx.released & ctx.resting & ctx.still & ctx.home
    stable = W_STABLE * good_step.to(dt)
    remaining = (1.0 - ctx.episode_progress).clamp(0.0, 1.0) * float(ctx.episode_steps)
    success = torch.where(ctx.success, W_SUCCESS + R_RATE * remaining, zero)

    # ---------------- 10. penalties ---------------------------------------------------------
    # (a) dropping a shoe that had already been carried
    dropped = ctx.carried & (~in_hand) & (~placed_ok) \
        & (ctx.shoe_bottom_z < (ctx.table_top_z + 0.03))
    p_drop = torch.where(dropped, torch.full_like(zero, 0.6), zero)
    # (b) shoving the shoe around the table instead of grasping it
    m_dist = (~ctx.carried) & (~in_hand) & (~ctx.start_held)
    p_dist = torch.where(m_dist, 0.4 * (ctx.shoe_shift_xy / 0.08).clamp(0.0, 1.0), zero)
    # (c) pressing the fingers into the table top (the wrap term reaches low, keep this guard)
    p_tab = 0.4 * ((ctx.table_top_z - ctx.hand_z_min) / 0.05).clamp(0.0, 1.0)
    # (d) driving the shoe into the side of the rack instead of down onto its top
    inside = (ctx.shoe_pos[:, 0] > ctx.rack_x_min - 0.05) & (ctx.shoe_pos[:, 0] < ctx.rack_x_max + 0.05) \
        & (ctx.shoe_pos[:, 1] > ctx.rack_y_min - 0.05) & (ctx.shoe_pos[:, 1] < ctx.rack_y_max + 0.05)
    m_ram = inside & in_hand & ctx.carried & (ctx.keypoint_dist < 0.25)
    p_ram = torch.where(
        m_ram, 0.5 * (((ctx.rack_top_z - 0.01) - ctx.shoe_bottom_z) / 0.10).clamp(0.0, 1.0), zero)
    penalty = -W_PEN * torch.where(busy, p_drop + p_dist + p_tab + p_ram, zero)

    # ---------------- 11. action regularization ---------------------------------------------
    # Small (<= ~0.35/step); off while the environment drives the arm.
    rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    mag = (ctx.actions[:, 0:6] ** 2).mean(dim=-1)
    armv = (ctx.arm_qd ** 2).mean(dim=-1).clamp(0.0, 25.0)
    action_reg = -torch.where(busy, 0.05 * rate + 0.02 * mag + 0.005 * armv, zero)

    reward = (approach + grasp + lift + hold + transport + align + settle
              + placed + release + trigger + stable + success + penalty + action_reg)

    return reward, {
        "approach": approach,
        "grasp": grasp,
        "lift": lift,
        "hold": hold,
        "transport": transport,
        "align": align,
        "settle": settle,
        "placed": placed,
        "release": release,
        "trigger": trigger,
        "stable": stable,
        "success": success,
        "penalty": penalty,
        "action_reg": action_reg,
    }
