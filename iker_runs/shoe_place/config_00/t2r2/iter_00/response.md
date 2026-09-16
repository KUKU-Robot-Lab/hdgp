# Reward design: place the held shoe on the rack

## 1. What the task actually asks for

The episode does **not** start with a reach-and-grasp problem: the shoe is already in the hand in its
recorded grasp pose, and `ctx.grip_norm` starts at `-1` (holding). So the whole task is the *second half*
of a pick-and-place — a **transport → seat → let go → retreat → stay put** problem.

Success is defined by the environment and I cannot redefine it. It needs **all four** of these to hold for
`ctx.stable_steps` (20) *consecutive* steps:

| condition | field | meaning |
|---|---|---|
| `keypoint_dist <= place_tolerance` | `ctx.placed` | the shoe's 4 keypoints sit on the 4 target keypoints — this encodes **both** position and orientation, so I need no quaternion math of my own |
| `palm_shoe_dist > release_radius` | `ctx.released` | the hand has physically backed off |
| `abs(shoe_bottom_z - rack_top_z) <= resting_tol` | `ctx.resting` | the shoe's hull rests *on* the rack top, not floating above it and not sunk into it |
| `shoe_lin_vel.norm() < still_speed` | `ctx.still` | it is not sliding, rocking or tipping |

`ctx.stable_count` resets to 0 the instant any one breaks, so the reward must make the policy reach a
configuration that survives the hand leaving. Two failure modes follow directly from that structure and
have to be priced in explicitly:

- **Letting go too early.** `released` is trivially satisfiable by just dropping the shoe; the agent would
  collect any un-gated release/withdraw reward and lose only the alignment term. Release and withdraw must
  therefore be *gated* on `placed & resting`, and opening the grip elsewhere must cost something.
- **Yanking the hand away while still gripping.** This drags the shoe out of place. Gating withdraw on
  `placed & resting` *and* on the grip already being open handles it.

## 2. Stages the robot must go through

1. **Transport / align** — carry the shoe from its grasp pose to the target pose on the rack
   (`keypoint_dist` from `init_keypoints` down to `<= place_tolerance`). This is one continuous SE(3) motion
   of the palm; the arm's IK is handled by the action space, so I only shape the resulting shoe pose.
2. **Seat** — bring the shoe's lowest hull point onto `rack_top_z` (the rack top is at 0.325 m, above the
   table, so the shoe must be *lifted over* the rack rim and set down, not slid across).
3. **Release** — drive the grip axis from `-1` toward `+1`. The EMA (0.4686 per step) means the fingers take
   several steps to open, so the reward has to pay for *partial* opening, not just the end state.
4. **Withdraw** — move the palm past `release_radius` without disturbing the shoe.
5. **Settle / hold** — the shoe must stay placed, resting and still for 20 consecutive steps while the hand
   is away. `stable_count` gives me a dense ramp for this.

Throughout: keep the arm smooth (the observations and actions are noisy — ±0.02 / ±0.05 — so a jittery
policy will keep knocking `stable_count` back to zero), and never let the shoe fall off its support.

## 3. Term-by-term rationale

- **`align`** — two length scales on `keypoint_dist`: a `tanh` with a 0.20 m scale that pulls from anywhere
  in the workspace, plus a Gaussian whose width is `place_tolerance` that only fires in the last few
  centimetres. The coarse term alone saturates near the goal and stops discriminating "almost placed" from
  "placed"; the fine term supplies that last gradient. Biggest weight (2.0) because everything else is
  conditional on it.
- **`progress`** — the same distance normalised by the episode's own starting error, clamped to `[0,1]`.
  Makes the early signal independent of how far the vision model happened to put the target, and is bounded
  so it cannot dominate.
- **`seat`** — `|shoe_bottom_z - rack_top_z|` on the `resting_tol` scale, **masked to `keypoint_dist <= 2 ×
  place_tolerance`**. Without that mask the policy gets paid for holding the shoe at rack height anywhere in
  the room, which competes with transport.
- **`release`** — the normalised open fraction of the grip axis, gated on `placed & resting`. Dense in the
  EMA opening, so the policy is paid per step of finger opening once (and only once) the shoe is in place.
- **`premature_release`** — negative. Two parts: a mild cost for opening the hand when *not* in the placed &
  resting state, and a larger flat cost for `released & ~placed`, i.e. the shoe is away from the palm but not
  at the target — the signature of a drop. This is what stops the degenerate "open immediately" policy.
- **`withdraw`** — palm-to-shoe distance normalised against `release_radius + 0.10 m`, gated on
  `placed & resting` *and* the grip being mostly open, so retreat is only rewarded after letting go.
- **`still`** — combines linear speed (on the `still_speed` scale) and angular speed, gated on `placed`.
  Rewards setting the shoe down gently rather than tossing it into the target pose.
- **`stability`** — `stable_count / stable_steps`, clamped to 1. The one term that directly tracks the
  success counter, giving 20 steps of dense credit before the bonus lands.
- **`success`** — the large terminal bonus (20), scaled up to ×1.5 by the fraction of the episode remaining
  so that finishing earlier is worth more. This is the only term allowed to dominate.
- **`fall_penalty`** — bounded `tanh` cost on the shoe's hull dropping below `table_top_z`. Deliberately
  referenced to the *table*, not the rack, because the transport path legitimately passes below rack height.
- **`action_reg`** — action-rate (dominant), action magnitude, and a bounded arm-joint-velocity term. Small
  weights only: with ±0.05 action noise, heavy suppression would kill exploration, but some smoothing is
  needed or the hand shakes the rack during the 20-step hold.

All shaping is bounded (`tanh`, `exp`, clamped ratios), no raw negative distances, everything is a masked
`torch.where` rather than Python control flow, and the function is pure — it reads `ctx` only and keeps no
state between calls.

```python
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
```
