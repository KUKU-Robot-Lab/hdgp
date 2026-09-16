# Analysis and new reward function

## 1. What the task means, and the stages it decomposes into

The shoe starts *already in the hand*. Nothing has to be grasped. The whole task is a
**hand-off**: carry the held shoe to a pose on the rack, seat it there, let go of it, get the
hand out of the way, and let it come to rest — and then keep all of that true for 20
consecutive steps.

Stages:

1. **Transport / align** — move the palm so that the shoe's 4 keypoints converge on
   `target_keypoints` (`keypoint_dist` → 0). This is a pose problem, not just a position
   problem: the keypoints encode orientation too.
2. **Seat** — bring the shoe's lowest point down onto the rack top, i.e.
   `|shoe_bottom_z - rack_top_z| <= resting_tol` (`resting`).
3. **Release** — drive the grip axis (`actions[6]`) toward +1 so the fingers open and the
   shoe is left standing on the rack rather than held.
4. **Withdraw** — move the palm away until `palm_shoe_dist > 0.15` (`released`).
5. **Settle / hold** — the shoe stops moving (`still`), and `placed & released & resting &
   still` holds for `stable_steps = 20` consecutive steps, which is what `success` means.

## 2. What the previous run actually produced

The training curves and the deterministic rollout agree on one diagnosis, and it is not the
one a reader would guess from the metrics alone.

The rollout numbers are decisive:

| condition | fraction of steps | fraction of envs reaching it |
|---|---|---|
| `placed` | 0.330 | 0.719 |
| `resting` | 0.486 | 0.984 |
| `placed & resting` | 0.299 | 0.719 |
| `released` | 0.043 | 0.219 |
| `still` | 0.017 | 0.047 |
| all four | **0.000** | **0.000** |

and the grip axis, in exactly the state the previous reward paid for opening it:

```
open_frac mean over all steps                 0.014
open_frac mean over steps where placed&resting 0.000
```

So the previous policy solved stages 1 and 2 well — `align` rose 0.13 → 0.555, `seat` 0.001 →
0.245, `resting` was reached by 98.4% of environments — and then **never executed stage 3**.
It reached the release condition constantly (30% of all steps) and held the hand shut at
open_frac = 0.000 in every one of those steps. `t2r_reward/release` ended at 1.9e-4 out of a
possible 1.5, and `t2r_reward/withdraw` was *exactly* 0.0 for nine of the ten sample points.
`stable_count` never left zero in any of the 64 environments.

This is not "not enough signal to find the release state". The release state was found and
occupied. The policy declined to open. Three things made declining correct:

**(a) The premature-release penalty taught the grip axis to be dead.** `-0.5 * early_open`
fired for *any* opening outside `placed & resting`, which is where essentially all early
exploration happens. `place/released` collapsed from 0.643 at the first sample to 0.098 at the
last, and `premature_release` shrank monotonically from -0.894 to -0.102 — the policy did not
learn *when* to open, it learned to stop touching `actions[6]` at all. With entropy falling
9.87 → 1.83, that dimension was frozen long before the policy could ever have been in a
`placed & resting` state to be paid for using it.

**(b) The `dropped` term made any imperfect release catastrophic and permanent.**
`dropped = released & ~placed` paid -1.0 **every step** it held. A release that leaves the shoe
even slightly off-target costs -1.0 for the whole remainder of the episode — on the order of
-100 against a release reward whose maximum was +1.5 per step and which is itself lost the
moment the shoe shifts out of tolerance. Holding was a guaranteed +1.1/step; releasing was a
gamble on a cliff. The expected-value calculation favoured holding by a wide margin, and PPO
found that.

**(c) `withdraw` was gated behind the very thing that never happened.**
`can_withdraw = ready & (open_frac > 0.5)` — since open_frac never exceeded 0.0 in the ready
state, the gate was never satisfiable and the term contributed literally nothing all run. The
same is true of `release` in practice. Two of the eleven components were structurally dead.

The structural notes in the feedback close the loop: `released` is a *consequence* of opening
the grip, not an independent condition; and a shoe in a moving hand is almost never `still`
(0.017 vs `resting` 0.486). So the four-way conjunction has **no path** to becoming true while
the hand stays shut, and the 20-step counter can never start. Everything downstream of the grip
— `released`, `still`, `stable_count`, `success` — was unreachable, which is exactly what the
component log shows: `release` 2e-4, `withdraw` 0, `stability` 2e-4, `success` 2e-4, all flat
across the whole run while `align` and `seat` climbed steadily. That is the classic signature
of a reward whose dense income is earned by *not* finishing the task.

## 3. What I changed, and why

The single design goal: **make letting go strictly and immediately more profitable than
holding, in the state the policy already occupies 30% of the time, and stop punishing the
exploration that gets there.**

1. **`release` weight 1.5 → 4.0**, still dense in `open_frac` and gated on `placed & resting`.
   The grip EMA moves 0.4686 of the way per step, so one step of `a = +1` already yields
   open_frac ≈ 0.47 → ≈ +1.9 reward. The gradient is felt on the very first exploratory step.

2. **New `hold_cost`**, the direct attack on the local optimum: `-1.0 * (1 - open_frac)` while
   `placed & resting`, ramped by `episode_progress` so an early, careful placement is not
   punished but sitting there holding is. Combined with `release`, the grip axis now carries a
   **5.0-per-step swing** in precisely the state that matters, versus 1.5 before. Crucially the
   penalty (max -1.0) stays well below the ≈ +4.6/step the policy already earns in that state,
   so it cannot make `placed & resting` net-negative and cause the policy to avoid it.

3. **The `dropped` cliff is gone.** No permanent per-step penalty for a shoe that is away from
   the palm and not in tolerance. Losing `align`/`seat`/`gates` is already the opportunity cost
   of a bad release; doubling it with a -1.0/step bleed is what made the gamble irrational.
   What remains is `mishandle`, weight 0.4, and it only fires on opening that is clearly not a
   placement attempt (not `resting` **and** `keypoint_dist > 2 * place_tolerance`). Opening
   anywhere near the rack at rack height is now free to explore.

4. **`withdraw` is no longer gated on the grip.** It is gated on `placed & resting` alone, and
   pays on two length scales (`palm_gap` first, `palm_shoe_dist` second) so there is gradient
   from the very first centimetre of separation. This gate is self-protecting: a hand that is
   still gripping cannot move away without dragging the shoe, which breaks `placed`/`resting`
   and cancels the term. No `open_frac > 0.5` precondition that can never be met.

5. **New `handsfree` (3.0)** — a flat, dense payment for `placed & resting & released`, and
   **`settle` (2.0)** scaled by the shoe's stillness in that same state. These make the chain
   after opening monotonically increasing: open → +4.0, separate → +2.0 withdraw and +3.0
   handsfree, shoe stops → +2.0 settle, then `gates` and `stability` ramp. Post-release income
   is ≈ 18/step against ≈ 4/step for holding — a 4.5× gap, where before releasing was a net
   loss.

6. **New `gates`** — `2.5 * (n_satisfied / 4)**2` over the four predicates. Superlinear, so the
   fourth condition is worth more than the first three, giving a smooth staircase toward a
   conjunction that the previous reward only ever paid for all-or-nothing.

7. **`stability` 1.0 → 3.0** and **`success` 20 → 30**. `stable_count` was pinned at 0, so
   these cost nothing until the chain above starts working, and then they are what carries the
   policy across the 20 consecutive steps.

8. **`align` (2.0), `progress` (1.0), `seat` (1.0), `fall_penalty`, `action_reg` are kept as
   they were.** They are the terms that demonstrably worked — transport and seating were
   solved. Note that they keep paying *after* a good release (the shoe is still at the goal
   pose on the rack), so releasing does not forfeit them; that was already true and is worth
   preserving, because it means the only thing standing between the policy and the success
   bonus is the grip axis.

Expected metric movement: `place/released` and the grip's `open_frac` should rise first (within
the state `placed & resting`, which is already reached by 72% of environments), followed by
`place/retreated`, then `still`, then `stable_count` leaving zero for the first time, then
`iker/success_5cm`. If `place/placed` and `place/resting` *fall* while `released` rises, the
hold/release balance is too aggressive and `hold_cost` should be reduced.

```python
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
```
