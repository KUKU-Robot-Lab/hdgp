## 1. What the task means and its stages

The hand starts holding the shoe. It has to put the shoe down on the rack so that the shoe's 4 keypoints match the 4 target keypoints within 3 cm. The shoe must rest on the rack top and be still, and the hand must open to at least 0.9. Once all of that is true, the environment takes over, opens the hand fully and moves the arm back to its rest posture. Success needs the shoe to stay placed and still for 20 of the last 30 steps while that happens.

Stages:
1. **Transport and align.** Carry the shoe over the rack and bring its keypoints onto the targets.
2. **Seat.** Lower the shoe until its lowest point is on the rack top.
3. **Settle and open.** Let the shoe come to rest, then open the hand to at least 0.9. This triggers the takeover.
4. **Correct (new, only when needed).** If the shoe was set down more than 3 cm off target, the policy still controls the arm. It must go back to the shoe and push it or grasp it again to move it onto the target, then do stage 3 again.
5. **Hold (done by the environment).** The shoe stays placed while the arm withdraws, and `stable_count` climbs to 20.

## 2. Analysis of the feedback

- Stages 1 to 3 and 5 are learned. 0.625 of episodes were placed at some point and 0.578 succeeded. Almost every episode that was ever placed went on to succeed. Those parts stay as they are.
- The main failure (0.31 of episodes) is a shoe that rests on the rack, released, 3 to 8 cm off target (median 4 cm), and is never touched again until the time limit.
- The reason is that doing nothing pays well in that state:
  - `settle` (1.2) and `seat` (0.5) were paid whenever `keypoint_dist <= 0.09` and the shoe rested, whether it was placed or not.
  - At 4 cm, a motionless shoe earned about align 0.58 + seat 0.5 + settle about 1.2 + gates about 0.3, or roughly 2.5 per step.
  - Touching the shoe makes it move. That lowers `settle`, and it is the only way to reach the target.
  - Nothing penalised staying off target, and nothing rewarded moving the shoe toward the target.
- `settle` averaged 0.61 per step, the largest term after `success`. That confirms it was being collected in the stranded state.

## 3. Changes (everything else unchanged)

1. **`seat` and `settle` pay only near the target.** Both are multiplied by a closeness factor: 1 inside the 3 cm tolerance, about 0.29 at 4 cm, and effectively 0 beyond 5 cm. A still shoe off target no longer earns them.
2. **`off_target`, a new penalty (max -0.6 per step).**
   - It is `-0.6 * tanh(max(dist - tol, 0) / tol)`, paid while the shoe is not placed and the arm is not retracting.
   - It rises steadily with distance and has no step or cutoff zone. Moving the shoe farther away never helps.
   - During the short carry phase it only acts as a small time cost.
   - It is 0 once the shoe is placed, so reaching the target removes it.
3. **`push`, a new term (range -1 to +1).**
   - It is `tanh(v_toward / 0.05)`, where `v_toward` is the shoe's horizontal speed toward the target keypoint centroid.
   - It is paid only while the shoe rests on the rack, is not placed, is within 0.15 m, and the arm is not retracting.
   - The term is signed, so pushing back and forth nets about zero.
   - This is the gradient that was missing: it pays moving the shoe, not keeping it still.
4. **`reach`, a new term (max 0.5).**
   - It pays the palm for being near the shoe surface (`palm_gap`) when the shoe rests off target within 0.12 m.
   - It helps the policy find its way back to the shoe.
   - It is smaller than the -0.6 `off_target` penalty, so hovering without fixing the shoe still loses reward.
5. **`lock` step raised from 0.7 to 1.2.** Crossing the 3 cm boundary is worth more.
6. **`fall_penalty` raised from 1.5 to 5.** Pushing must not make knocking the shoe off look cheap. A fall also ends the episode, so all later reward is lost.
7. **`success` rate per remaining step raised from 16 to 18.** The dense ceiling is now about 14.9 per step, and finishing must stay strictly better than hovering.

Expected movement:
- The share of episodes that rest off target should drop, and `place/placed` and success should rise.
- `settle` and `seat` should fall at first, because they are no longer paid off target.
- `push` should be slightly positive on average.
- `off_target` should shrink toward 0 as correction is learned.

```python
import torch
import math


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Place the held shoe on the rack: transport -> align -> seat -> open hand (env retracts arm).
    If the shoe was set down off target, go back and move it onto the target."""

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
    active = ~ctx.retracting                       # policy still controls the arm

    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)

    seated = ctx.placed & ctx.resting              # shoe at the goal pose, at rack height
    ready = seated & ctx.still & open_enough       # exactly the environment's takeover trigger

    # closeness factor: 1 inside tolerance, ~0.29 at 4 cm, ~0 beyond 5 cm
    excess = torch.clamp(dist - tol, min=0.0)
    closeness = torch.exp(-((excess / (0.3 * tol)) ** 2))

    # ---- 1. align: three length scales, w=1.5 (unchanged, learned) -----------------------
    coarse = 1.0 - torch.tanh(dist / 0.20)
    mid = 1.0 - torch.tanh(dist / (2.0 * tol))
    fine = torch.exp(-((dist / (0.5 * tol)) ** 2))
    align = 1.5 * (0.30 * coarse + 0.35 * mid + 0.35 * fine)

    # ---- 2. precision: worst keypoint, w=0.8 (unchanged) ----------------------------------
    worst_err = ctx.keypoint_err.max(dim=-1).values
    precision = 0.8 * torch.exp(-((worst_err / (1.5 * tol)) ** 2))

    # ---- 3. lock: step on `placed` raised 0.7 -> 1.2, + 0.5 narrow Gaussian ----------------
    lock = 1.2 * ctx.placed.float() + 0.5 * torch.exp(-((dist / (0.4 * tol)) ** 2))

    # ---- 4. seat: height onto rack top, w=0.5, now scaled by closeness ---------------------
    # (was paid up to 9 cm off target -> paid the stranded state)
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat = 0.5 * closeness * (1.0 - torch.tanh(height_err / rest_tol))

    # ---- 5. settle: stillness margin, w=1.2, now only near/at the target -------------------
    # (was the main reward for leaving a shoe motionless 4 cm off target)
    calm = 1.0 - torch.tanh(lin_sp / (0.4 * still_sp) + ang_sp / 3.0)
    settle = 1.2 * torch.where(ctx.resting, closeness * calm, zero)

    # ---- 6. release: w=1.2 (unchanged) -----------------------------------------------------
    release = 1.2 * torch.where(seated, torch.clamp(open_frac / open_min, 0.0, 1.0), zero)

    # ---- 7. ready: takeover condition, max 3.0 (unchanged) --------------------------------
    margin = torch.exp(-((dist / (0.7 * tol)) ** 2))
    ready_r = 3.0 * torch.where(ready, (2.0 + margin) / 3.0, zero)

    # ---- 8. gates: smooth superlinear staircase, w=1.0 (unchanged) -------------------------
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.still.float()
        + open_enough.float() + ctx.home.float()
    )
    gates = 1.0 * (n_gates / 5.0) ** 2

    # ---- 9. stability: direct success proxy, max 3.0 (unchanged) --------------------------
    frac = torch.clamp(ctx.stable_count.float() / n_stable, 0.0, 1.0)
    stability = 2.2 * frac + 0.8 * frac ** 2

    # ---- 10. off_target: NEW, max -0.6 -----------------------------------------------------
    # Monotone in distance beyond tolerance (no cutoff to escape from), zero once placed,
    # masked during the scripted retract. Makes a stranded shoe cost reward every step.
    not_placed = active & ~ctx.placed
    off_target = -0.6 * torch.where(not_placed, torch.tanh(excess / tol), zero)

    # ---- 11. push: NEW, range [-1, 1] ------------------------------------------------------
    # Signed horizontal shoe velocity toward the target keypoint centroid while the shoe rests
    # on the rack off target. Signed, so back-and-forth nets ~0; pays moving it the right way.
    tgt_c = ctx.target_keypoints.mean(dim=1)       # (N,3)
    cur_c = ctx.keypoints.mean(dim=1)              # (N,3)
    to_tgt = (tgt_c - cur_c)[:, :2]
    to_tgt_dir = to_tgt / (to_tgt.norm(dim=-1, keepdim=True) + 1e-6)
    v_toward = (ctx.shoe_lin_vel[:, :2] * to_tgt_dir).sum(dim=-1)
    stranded = not_placed & ctx.resting & (dist < 0.15)
    push = 1.0 * torch.where(stranded, torch.tanh(v_toward / 0.05), zero)

    # ---- 12. reach: NEW, max 0.5 -----------------------------------------------------------
    # Palm back to the shoe surface when it rests off target (<0.12 m). Smaller than the
    # off_target penalty, so hovering without correcting never pays.
    reach_zone = stranded & (dist < 0.12)
    reach = 0.5 * torch.where(reach_zone, 1.0 - torch.tanh(ctx.palm_gap / 0.05), zero)

    # ---- 13. success: paid PER REMAINING STEP, rate 18 -------------------------------------
    # Dense ceiling ~14.9/step; 18 per remaining step keeps finishing strictly better.
    steps_left = torch.clamp(1.0 - ep, 0.0, 1.0) * float(ep_steps)
    success = torch.where(ctx.success, 40.0 + 18.0 * steps_left, zero)

    # ---- 14. fall: stronger guard now that the policy pushes the shoe (1.5 -> 5.0) ---------
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -5.0 * torch.tanh(fall / 0.05)

    # ---- 15. action regularisation: small; masked while the env drives the arm -------------
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
        "off_target": off_target,
        "push": push,
        "reach": reach,
        "success": success,
        "fall_penalty": fall_penalty,
        "action_reg": action_reg,
    }

    reward = torch.zeros_like(dist)
    for value in components.values():
        reward = reward + value

    return reward, components
```
