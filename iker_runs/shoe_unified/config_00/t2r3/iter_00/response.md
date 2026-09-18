# 1. What the task means and which stages the robot must go through

## Reading of the task

One policy, one 26-D action vector, 360 steps, must do the whole job: find the shoe wherever it was
spawned on the table, close the hand on it, lift it, carry it across to the rack, put it down so its own
4 keypoints land on the 4 target keypoints (that is a full 6-DOF pose match, not just a position match),
open the hand, and then let the environment take the arm home. The episode only counts as a success if,
for 20 of the last 30 steps, all five environment conditions hold at once: `placed` (keypoint_dist ≤ 3 cm),
`released` (palm > 15 cm from the shoe), `resting` (shoe bottom at rack height), `still`, and `home`.

Three facts from the measurements drive the whole design:

1. **Keeping the grasp is the bottleneck, not making it.** 62 % of stage-1 episodes reached a proper hold
   at least once and only 34 % sustained it for 20 steps. Here the shoe must stay in the hand for the
   entire carry, which is far longer than 20 steps, and force/torque pulses hit the shoe once the grasp
   latches. So a *persistent per-step* payment for a secure grip (multi-finger contact, no slip, palm
   close) is the single most important shaping term, and it must be paid for the whole carry — not once.
2. **The pick-up status fields go dark during the carry.** `dz_free` only measures a free lift near the
   pick spot, so `held`, `hold_count` and `latched` describe the pick, not the transport. During the
   transport the only honest facts are `link_shoe_force`, `palm_shoe_force`, `palm_gap`,
   `palm_shoe_dist` and `slip_speed`. I therefore build my own contact-based `in_hand` mask and use the
   environment's `held` only as an extra OR.
3. **An off-target set-down is a dead end.** The placement policy never learned to go back and touch a
   shoe it had already put down; two generated rewards that paid for the return stayed flat for 200
   epochs. The conclusion is not to pay more for recovery but to make the *release itself* impossible to
   earn until the placement is already correct: the release term is gated hard on `placed & resting`, so
   dropping the shoe 5 cm off target earns nothing at all and the only profitable route is to align
   before opening the hand. The fine alignment term uses both the mean and the worst keypoint error so a
   tilted shoe (right position, wrong orientation) cannot collect it.

## Stages (all expressed as masks over state, never over time — a fraction of episodes starts mid-task
with `start_held` and must see the same reward)

| stage | mask | what is paid |
|---|---|---|
| approach | not in hand, shoe not already correctly placed | palm gap, fingertip gaps, palm facing the shoe |
| grasp | palm near the shoe | multi-finger contact force, thumb curl, overall closure |
| lift | in hand | shoe bottom rising above the table, plus `dz_free` toward `lift_height` |
| hold (whole carry) | in hand and clear of the table | contact quality, no slip, palm close — the anti-drop insurance |
| transport | shoe clear of the table | coarse `keypoint_dist` shaping toward the rack |
| align | shoe clear of the table | fine mean + worst-keypoint shaping, only meaningful under ~15 cm |
| settle | near the target | shoe at rack height and slow |
| release | **only** when `placed & resting` | commanded hand opening + palm retreat past `release_radius` |
| stable / success | environment booleans | per-step payment while all five success conditions hold, plus a one-shot success bonus |

The return of the arm to the rest posture is *not* shaped: it is the environment's scripted 20-step
retract. I do not zero the shaping during `retracting` (that would create a cliff that makes triggering
the handover locally unattractive); instead the terms that remain true there — transport, align, settle,
release, stable — form a high constant plateau, which is exactly a terminal bonus for having triggered
the handover as early as possible. The only terms explicitly switched off during `retracting` are the
ones the policy could otherwise be blamed for: approach, grasp, the penalties and the action
regularization, since the policy's actions are ignored there.

Two things I deliberately do **not** do: no reward for the arm going home (not the policy's job), and no
reward that can be collected by a shoe that reaches the rack without ever being held — transport/align
are gated on the shoe being clear of the table, and the large money is behind `placed & resting`, which a
thrown shoe will not satisfy for 20 steps.

Penalties are kept small (≤ ~0.35/step against ~10/step of shaping at the end) and cover the four ways
this setup is known to go wrong: dropping the shoe after it has been carried, shoving the shoe around the
table before grasping it, pressing the fingers through the table top, and ramming the shoe into the side
of the rack instead of coming down onto its top surface.

# 2. Reward function

```python
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
    # Ordered so the task is strictly monotone in progress: every stage pays more than the one
    # before it, and the release/stable money is unreachable without a correct placement.
    W_APPROACH = 1.0    # cheapest: reaching was already solved by the predecessor policy
    W_GRASP = 1.5       # closing on the shoe
    W_LIFT = 1.5        # getting it off the table
    W_HOLD = 1.0        # paid EVERY carry step: the measured bottleneck is losing the grip
    W_TRANSPORT = 2.5   # coarse travel to the rack
    W_ALIGN = 2.0       # fine 6-DOF keypoint match (mean + worst keypoint)
    W_SETTLE = 1.5      # sitting at rack height, not moving
    W_RELEASE = 3.0     # opening and backing off, ONLY once the placement is already correct
    W_STABLE = 4.0      # every step that counts toward stable_count
    W_SUCCESS = 40.0    # one-shot terminal bonus
    W_PEN = 1.0         # drop / disturb / table / rack penalties (each sub-term <= ~0.5)

    # ---------------- hand closure from the normalised joint angles -----------------------
    # hand_q_norm is 0 at the lower limit and 1 at the upper limit. For the movable joints the
    # open pose sits at the LOWER limit, except the two thumb joints (_3, _4) whose open angle
    # 0.0 rad is the UPPER end of [-1.571, 0]; those two are flipped.
    movable = torch.tensor([2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 18, 19],
                           device=dev, dtype=torch.long)
    flip = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        device=dev, dtype=dt)
    q_n = ctx.hand_q_norm.index_select(1, movable).clamp(0.0, 1.0)
    t_n = ctx.hand_target_norm.index_select(1, movable).clamp(0.0, 1.0)
    curl = flip * (1.0 - q_n) + (1.0 - flip) * q_n          # 0 open, 1 fully gripped
    curl_cmd = flip * (1.0 - t_n) + (1.0 - flip) * t_n      # same, for the filtered targets
    close_frac = curl.mean(dim=-1)                          # (N,)
    open_cmd = 1.0 - curl_cmd.mean(dim=-1)                  # commanded open fraction (N,)

    # ---------------- contact / grasp facts (valid during the whole carry) -----------------
    finger_force = ctx.link_shoe_force.sum(dim=-1)                       # (N,5) per finger
    contact_score = torch.tanh(finger_force / 2.0).mean(dim=-1)          # bounded contact quality
    n_touch = (finger_force > 0.2).to(dt).sum(dim=-1) + (ctx.palm_shoe_force > 0.2).to(dt)
    tip_gap = ctx.link_shoe_gap[:, :, 2].mean(dim=-1)                    # fingertips to shoe
    all_gap = ctx.link_shoe_gap.mean(dim=(1, 2))                         # all finger links

    # Own "still in the hand" predicate: dz_free (and therefore held/latched) reads 0 once the
    # shoe is carried away from the pick spot, so the carry must be judged from contact,
    # distance and slip only. ctx.held is OR-ed in, it is never relied on alone.
    secure = (n_touch >= 2.0) & (ctx.palm_shoe_dist < ctx.hold_radius) \
        & (ctx.slip_speed < 3.0 * ctx.hold_slip_speed)
    in_hand = secure | ctx.held
    lifted = ctx.shoe_bottom_z > (ctx.table_top_z + 0.02)

    placed_ok = ctx.placed & ctx.resting          # a genuinely finished placement
    busy = ~ctx.retracting                        # the policy only controls the robot here
    shoe_speed = ctx.shoe_lin_vel.norm(dim=-1)

    # ---------------- 1. approach ----------------------------------------------------------
    # Off while the shoe is already correctly placed, so a good placement is never disturbed,
    # and off while the environment drives the arm.
    face = 0.5 * (1.0 + (ctx.palm_normal * _unit(ctx.shoe_pos - ctx.palm_pos)).sum(dim=-1))
    r_app = 0.45 * (1.0 - torch.tanh(5.0 * ctx.palm_gap)) \
        + 0.30 * (1.0 - torch.tanh(5.0 * tip_gap)) \
        + 0.25 * face.clamp(0.0, 1.0)
    m_app = (~in_hand) & (~placed_ok) & busy
    approach = W_APPROACH * torch.where(m_app, r_app, zero)

    # ---------------- 2. grasp -------------------------------------------------------------
    # Only within reach of the shoe, so the hand cannot farm it by closing in mid-air.
    thumb = (ctx.thumb_curl / 0.6).clamp(0.0, 1.0)          # deep curl, well past thumb_curl_min
    r_grasp = 0.50 * contact_score + 0.30 * thumb \
        + 0.20 * close_frac * (1.0 - torch.tanh(8.0 * all_gap))
    m_grasp = (ctx.palm_gap < 0.12) & (~placed_ok) & busy
    grasp = W_GRASP * torch.where(m_grasp, r_grasp, zero)

    # ---------------- 3. lift --------------------------------------------------------------
    # Height above the table works everywhere; dz_free only pays at the pick spot, which is
    # exactly where the free-lift signal is meaningful.
    h_tab = ((ctx.shoe_bottom_z - ctx.table_top_z) / 0.12).clamp(0.0, 1.0)
    h_free = (ctx.dz_free / max(ctx.lift_height, 1e-3)).clamp(0.0, 1.0)
    lift = W_LIFT * torch.where(in_hand, 0.6 * h_tab + 0.4 * h_free, zero)

    # ---------------- 4. hold (the anti-drop insurance, paid every carry step) -------------
    r_hold = 0.45 * contact_score \
        + 0.35 * (1.0 - torch.tanh(ctx.slip_speed / max(ctx.hold_slip_speed, 1e-3))) \
        + 0.20 * (1.0 - torch.tanh(8.0 * ctx.palm_gap))
    hold = W_HOLD * torch.where(in_hand & lifted, r_hold, zero)

    # ---------------- 5. transport ---------------------------------------------------------
    # Gated on the shoe being clear of the table (not on in_hand) so that letting go on target
    # does not knock this term out and create a cliff at the moment of release.
    transport = W_TRANSPORT * torch.where(
        lifted, 1.0 - torch.tanh(2.5 * ctx.keypoint_dist), zero)

    # ---------------- 6. align (position AND orientation) ----------------------------------
    # keypoint_dist is the mean; the worst keypoint punishes a shoe at the right spot but tilted.
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

    # ---------------- 8. release (hard-gated on an already correct placement) --------------
    # The measured failure of the predecessor was setting the shoe down off target and never
    # returning. Paying nothing at all for an off-target release removes that whole branch:
    # the only way to earn it is to be within place_tolerance and resting first.
    r_rel = 0.55 * open_cmd.clamp(0.0, 1.0) \
        + 0.45 * (ctx.palm_shoe_dist / max(ctx.release_radius, 1e-3)).clamp(0.0, 1.0)
    release = W_RELEASE * torch.where(placed_ok, r_rel, zero)

    # ---------------- 9. stable + success ---------------------------------------------------
    good_step = ctx.placed & ctx.released & ctx.resting & ctx.still & ctx.home
    stable = W_STABLE * good_step.to(dt)
    success = W_SUCCESS * ctx.success.to(dt)

    # ---------------- 10. penalties ---------------------------------------------------------
    # (a) dropping a shoe that had already been carried
    dropped = ctx.carried & (~in_hand) & (~placed_ok) \
        & (ctx.shoe_bottom_z < (ctx.table_top_z + 0.03))
    p_drop = torch.where(dropped, torch.full_like(ctx.shoe_bottom_z, 0.6), zero)
    # (b) shoving the shoe around the table instead of grasping it (not for start_held resets,
    #     where shoe_start_xy is where it started inside the hand)
    m_dist = (~ctx.carried) & (~in_hand) & (~ctx.start_held)
    p_dist = torch.where(m_dist, 0.4 * (ctx.shoe_shift_xy / 0.08).clamp(0.0, 1.0), zero)
    # (c) pressing the fingers into / through the table top
    p_tab = 0.4 * ((ctx.table_top_z - ctx.hand_z_min) / 0.05).clamp(0.0, 1.0)
    # (d) driving the shoe into the side of the rack instead of down onto its top surface
    inside = (ctx.shoe_pos[:, 0] > ctx.rack_x_min - 0.05) & (ctx.shoe_pos[:, 0] < ctx.rack_x_max + 0.05) \
        & (ctx.shoe_pos[:, 1] > ctx.rack_y_min - 0.05) & (ctx.shoe_pos[:, 1] < ctx.rack_y_max + 0.05)
    m_ram = inside & in_hand & ctx.carried & (ctx.keypoint_dist < 0.25)
    p_ram = torch.where(
        m_ram, 0.5 * (((ctx.rack_top_z - 0.01) - ctx.shoe_bottom_z) / 0.10).clamp(0.0, 1.0), zero)
    penalty = -W_PEN * torch.where(busy, p_drop + p_dist + p_tab + p_ram, zero)

    # ---------------- 11. action regularization ---------------------------------------------
    # Small on purpose (<= ~0.35/step against ~10/step of shaping at the end) and switched off
    # while the environment drives the arm, where the actions have no effect anyway.
    rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    mag = (ctx.actions[:, 0:6] ** 2).mean(dim=-1)
    armv = (ctx.arm_qd ** 2).mean(dim=-1).clamp(0.0, 25.0)
    action_reg = -torch.where(busy, 0.05 * rate + 0.02 * mag + 0.005 * armv, zero)

    reward = (approach + grasp + lift + hold + transport + align + settle
              + release + stable + success + penalty + action_reg)

    return reward, {
        "approach": approach,
        "grasp": grasp,
        "lift": lift,
        "hold": hold,
        "transport": transport,
        "align": align,
        "settle": settle,
        "release": release,
        "stable": stable,
        "success": success,
        "penalty": penalty,
        "action_reg": action_reg,
    }
```
