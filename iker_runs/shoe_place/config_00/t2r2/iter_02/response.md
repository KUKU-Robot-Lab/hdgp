# Analysis

## What the task means, stage by stage

The shoe starts in the hand. To finish, the environment must see four predicates true together for 20 of
the last 30 steps:

1. `placed` — mean keypoint error ≤ 0.03 m (tightened from the previous run),
2. `resting` — the shoe's lowest hull point within `resting_tol` of the rack top,
3. `released` — palm more than 0.15 m from the shoe origin,
4. `still` — shoe speed below 0.05 m/s.

So the stages are: **transport** (carry the shoe onto the 4 target keypoints), **seat** (lower it until its
sole is at rack height), **let go** (open the grip axis), **withdraw** (back the palm out past the release
radius), and **hold quiet** (do nothing that re-excites the shoe for the rest of the window). Because a
single bad step now only costs one count instead of resetting it, a correcting nudge in the middle of the
window is legal.

## Reading the previous run

The measured rollouts say plainly which parts are done and which are not.

**Done — do not rebuild.** `open_frac` on `placed & resting` steps is 0.97 (it was 0.00 before that reward).
`released` sits at 0.700 of steps and `resting` at 0.711. Setting down, opening, and backing off are learned
behaviours. The corresponding terms (`release`, `handsfree`, `withdraw`) therefore no longer need large
weights; their job is now only to *maintain* the behaviour, not to discover it. In the previous reward they
were carrying 59.7 % of the total while the success metric sat at 0.23 % of it — that is mass spent on
already-solved stages.

**Failure 1 — the last three centimetres.** Median final keypoint error was 0.046 m against a tolerance that
is now 0.03 m. That median would now fail outright. The old `align` had only two length scales, 0.20 m and
`tol`; between 0.03 m and 0.10 m the coarse term is nearly flat (`1 - tanh(d/0.2)` moves only 0.10 across
that whole band) and the Gaussian is numerically dead (at 0.046 m with tol 0.03 it is `exp(-2.35) ≈ 0.10`,
and it was dead entirely at the old tolerance). There was almost no gradient in exactly the band where the
policy lives. The new `align` adds a middle scale at `2·tol` and sharpens the fine scale to `0.5·tol`, so the
strongest gradient sits at 0.02–0.08 m. I also add a separate term on the **worst** of the four keypoints
rather than their mean, because a mean of 0.03 m can hide one badly rotated corner, and orientation error is
what puts the shoe next to the target instead of on it.

**Failure 2 — `still` is the binding constraint and it went the wrong way.** It is the least satisfied
predicate (0.491 of steps) and it *fell* over training, 0.811 → 0.614, while the other three rose. The old
`settle` used `1 - tanh(v/still_speed)`, which still pays 0.24 at exactly the failure threshold — a very soft
temperature — and was gated on `handed_over`. As the occupancy of that gate grew, the logged mean grew with
it even as per-step stillness got worse: the component tracked *how often the gate was open*, not how still
the shoe was. That is the contradiction noted in the observations. The fix is two-fold: sharpen the
temperature by 2.5× (`0.4·still_speed`, so the shape is essentially zero at the threshold and only pays for
real margin below it), and gate it on `resting & near-goal` instead of on `released`, so it measures the shoe
being at rest where it belongs rather than the hand being away.

**Failure 3 — the structural one: finishing is punished.** Success *terminates the episode*. At 8.35 dense
reward per step, a policy that succeeds at step 120 forfeits ~80 × 8.35 ≈ 670 of future reward to collect a
30-point bonus. Hovering one condition short of the counter was strictly the better policy, and the logs show
exactly that signature: every post-release component rose monotonically to the end of training while
`iker/success_5cm` did not. Two changes follow from this, and they are the core of this revision:

- **Cut the dense ceiling.** With the weights below, the maximum per-step sum in the near-success state is
  about 9.8 (was ~8.7 but with most of it payable while short of the goal).
- **Pay the terminal bonus per remaining step.** `40 + 12 · steps_remaining` strictly exceeds the dense
  reward the policy gives up by terminating (12 > 9.8 per step), so finishing early is never worse than
  hovering, and finishing *earlier* is strictly better than finishing late. This is the one place I deliberately
  use a large number.

**Failure 4 — paying for post-release states that are not accurate enough.** `handsfree` was a flat 3.0 for
`placed & resting & released`. With the tolerance at 0.03 m and observation noise at ±0.02 m, sitting *just*
inside the tolerance is not good enough — the policy needs margin. So `handsfree` is now multiplied by a
Gaussian on the error at `0.7·tol`, which varies from 0.13 to 1.0 *inside* the tolerance band. That keeps a
gradient alive after the shoe is already nominally placed, which is precisely where the previous reward went
flat.

## What I removed and why

Following the "nearly constant ⇒ the policy is not optimising it" rule:

- `progress` (0.607 at the end) is a monotone function of `keypoint_dist`, duplicating `align` with a
  redundant normalisation — dropped, its budget folded into the sharper `align`.
- `hold_cost` averaged −0.002 over the whole run. The behaviour it was insuring against (never opening the
  hand) is solved — dropped.
- `mishandle` ended at −0.006 and its target behaviour is gone — dropped.
- `fall_penalty` is kept despite being small: it is a safety guard against the 9 % drop rate, not a shaping
  term, and removing it risks that number climbing.

`action_reg` keeps a slightly higher rate coefficient (0.05 → 0.08) because step-to-step jitter is the most
direct mechanical cause of the `still` predicate breaking, but it stays small so the ±0.05 action noise does
not swamp exploration.

## Expected movement

`iker/keypoint_distance_m` should fall below the old 0.13 floor (the mid-scale term is the new gradient);
`place/still` should stop declining and rise, since it is now paid with a sharp temperature and it is the
only predicate still missing; `t2r_reward/stability` (0.138 of 3.0 previously) is the component to watch —
it is the direct proxy for the success counter, and it is now the largest dense term. If `t2r_reward/success`
still does not move while `stability` saturates near 19/20, the terminating-early compensation coefficient
is still too small and should be raised again.

```python
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
```
