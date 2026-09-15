## 1. What the task means, step by step

The robot has to pick up an upright cup with a power grasp, using a strict three-stage order. The environment tracks the order itself with two flags that never reset, `approach_done` and `envelope_done`. These flags split every step into exactly one stage:

| stage | condition on the step | what the robot must do |
|---|---|---|
| 1 approach | `not approach_done` | Move the palm to the cup's -y side without rotating the hand or changing the finger shape. Stay out of contact. |
| 2 envelope | `approach_done and not envelope_done` | Stay at that spot and close the hand around the cup: palm, thumb and at least 3 fingers on the cup. |
| 3 lift | `envelope_done` | Keep the grasp, raise the cup straight up to the goal and hold it still and upright. |

**Stage 1, approach.** The hand starts in the right orientation (`palm_normal` ≈ +y, `palm_finger_dir` ≈ +x) and the right shape (the default pose). That means the stage is mainly a position problem with two constraints: keep the orientation and keep the pose. The target comes straight from the `approach_done` definition, measured in the palm frame relative to the cup:
- along `palm_normal`: palm surface 1 cm from the cup side, so the distance to the axis is `cup_radius + 0.01` (the window allows up to +0.02);
- along `palm_finger_dir`: cup axis ahead of the palm by `cup_radius + 0.0075` (the centre of the window `[r-0.005, r+0.02]`);
- along `cup_axis`: palm within the graspable band. I use a dead band of half the band height.

The thumb sticks about 12 cm out along `palm_normal` at the wrist end. So the dangerous motion is sliding along the fingers (+x) at cup height while the thumb overlaps the cup in y. To make the hand fix its along-finger offset first, the along-finger error gets weight 1.5. The safe paths are then a vertical descent, or a final closing motion along +y.

Constraints in this stage, all applied as factors so the reward never goes negative:
- orientation kept;
- movable finger joints near `hand_default_q_norm`, since `approach_done` needs every joint within 0.15;
- no thumb or finger contact with the cup (the palm is allowed to touch);
- the cup not moved or tilted.

**Stage 2, envelope grasp.** The palm must stay at the cup's side, with the gap closed to about 5 mm, and the hand orientation must stay the same. The fingers and thumb then close until they press on the cup:
- the thumb presses the near (-x) side;
- the index, middle, ring and pinky wrap around the +x side toward the far side, ideally with their inner links (`_3` and `_4`) rather than only the fingertips.

The env needs palm, thumb and at least 4 digits in total (so at least 3 fingers) above 0.1 N for 5 consecutive steps. I reward soft contact scores that saturate at 0.3 N. That asks for a firm grasp without letting force spikes dominate. A small proximity term (finger links near the cup surface) gives a signal before contact happens. The cup must not be pushed away, tipped or lifted yet.

**Stage 3, lift and hold.** Keep the contacts, raise the cup straight up by 0.21 to 0.28 m, and bring `goal_dist` under `success_tol`. Then keep the cup upright and still so that successes keep being counted.

**Rules that shape the reward structure**

- **Stage ladder, no hovering.** Every stage has a constant base paid on every step once the stage is reached. Each base is larger than the largest reward the previous stage can pay:
  - stage 1 pays at most 1.0;
  - stage 2 pays 1.2 to 2.2;
  - stage 3 pays 2.5 to 4.5.

  Each stage's shaping peaks exactly where the environment sets the next flag. So staying just short of a stage boundary always pays less than crossing it, and no bonus is paid for being "almost there".
- **No skipping or reordering.** Lift and success terms exist only once `envelope_done` is set. Grasp terms exist only once `approach_done` is set. Closing the fingers during the approach removes the approach reward through the pose factor. Lifting during stage 2 lowers the stage-2 reward through the cup-displacement factor.
- **Every task term is ≥ 0.** Every early termination is a failure: cup knocked over, dropped, or off the table, or the hand into the table. Because the reward is never negative, termination always forfeits future reward, and the policy never gains by ending an episode. Regularization is a small bounded smoothness bonus in [0, 0.05], not a penalty. There is also a soft table factor that halves the task reward when a hand link goes more than 1 cm below the table top, before the env terminates at 3 cm.
- **Successes.** Each counted success pays a bonus, but only in stage 3. The success that reaches `max_successes` ends the episode and so cuts off a stream of about 4.5 per step. If nothing replaced that stream, hovering just outside the tolerance would pay more. The final success therefore pays an extra bonus of about 100 steps of the maximum stage-3 reward. Counting may or may not include the current step, so the check fires at `num_successes >= max_successes - 1`.
- **Smoothness.** A bounded smoothness bonus discourages action jitter (there is a random 0 to 2 step action delay) and fast arm motion. It is far smaller than any stage gap.

## 2. Reward function

```python
import torch
import math

# Hand joints that actually move (range > 0.05 rad), from the joint table.
_MOVABLE = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18]

# ---- stage ladder (per step) ------------------------------------------------------------
W_APPROACH = 1.0    # stage-1 shaping, in [0, 1]
BASE_GRASP = 1.2    # paid every step once approach_done; above the stage-1 maximum (1.0)
W_GRASP = 1.0       # stage-2 shaping; stage-2 total in [1.2, 2.2]
BASE_LIFT = 2.5     # paid every step once envelope_done; above the stage-2 maximum (2.2)
W_LIFT = 2.0        # stage-3 shaping; stage-3 total in [2.5, 4.5]
W_SMOOTH = 0.05     # bounded smoothness bonus, far below every stage gap
SUCCESS_BONUS = 20.0
# The success that reaches max_successes ends the episode; this replaces the ~100 steps of
# maximum stage-3 reward it forfeits, so hovering just outside the tolerance never pays more.
FINAL_SUCCESS_BONUS = 100.0 * (BASE_LIFT + W_LIFT + W_SMOOTH)


def _palm_target_error(ctx, gap_target):
    # Palm offset from the approach_done target, measured in the palm frame relative to the cup axis.
    v = ctx.cup_pos - ctx.palm_pos                               # palm -> cup reference point
    h_cup = (v * ctx.cup_axis).sum(-1)
    v_rad = v - h_cup.unsqueeze(-1) * ctx.cup_axis               # palm -> cup axis, perpendicular to it
    d_n = (v_rad * ctx.palm_normal).sum(-1)                      # axis ahead of palm along palm normal
    d_f = (v_rad * ctx.palm_finger_dir).sum(-1)                  # axis ahead of palm along the fingers
    e_n = (d_n - ctx.cup_radius) - gap_target                    # palm-surface gap to the cup side
    e_f = d_f - (ctx.cup_radius + 0.0075)                        # centre of the [r-0.005, r+0.02] window
    e_h = torch.relu(h_cup.abs() - 0.5 * ctx.cup_half_height)    # stay in the middle of the band
    # along-finger error weighted 1.5: align it first, so the protruding thumb does not slide into the cup
    return torch.sqrt(e_n ** 2 + (1.5 * e_f) ** 2 + e_h ** 2 + 1e-12)


def _orientation_score(ctx):
    # start orientation: palm_normal along +y, palm_finger_dir along +x (about 0.17 at 45 deg on both)
    cos_n = ctx.palm_normal[:, 1]
    cos_f = ctx.palm_finger_dir[:, 0]
    return torch.exp(-3.0 * (1.0 - cos_n)) * torch.exp(-3.0 * (1.0 - cos_f))


def _default_pose_score(ctx):
    # movable finger joints must stay near the default pose (approach_done needs every one within 0.15)
    dev = (ctx.hand_q_norm - ctx.hand_default_q_norm)[:, _MOVABLE].abs()
    return torch.exp(-dev.mean(-1) / 0.08) * torch.exp(-torch.relu(dev.max(-1).values - 0.10) / 0.03)


def _contact_terms(ctx):
    c = torch.clamp(ctx.link_cup_force / 0.3, 0.0, 1.0)                       # (N,5,3), saturates at 0.3 N
    # inner links (_3, _4) count fully, fingertip-only contact counts 0.6 (a wrap, not a poke)
    digit = torch.maximum(torch.maximum(c[:, :, 0], c[:, :, 1]), 0.6 * c[:, :, 2])  # (N,5)
    thumb = digit[:, 0]
    fingers = torch.clamp(digit[:, 1:].sum(-1) / 3.0, max=1.0)                  # 3 fingers + thumb = 4 digits
    palm = torch.clamp(ctx.palm_cup_force / 0.3, 0.0, 1.0)
    digit_touch = torch.clamp(ctx.link_cup_force.flatten(1).max(-1).values / 0.1, 0.0, 1.0)
    return palm, thumb, fingers, digit_touch


def _wrap_score(ctx):
    # pre-contact signal: nearest link of each digit close to the cup surface inside the band
    axis = ctx.cup_axis[:, None, None, :]
    v = ctx.link_pos - ctx.cup_pos[:, None, None, :]
    h = (v * axis).sum(-1)
    radial = torch.norm(v - h.unsqueeze(-1) * axis, dim=-1)
    gap = (torch.relu(radial - ctx.cup_radius[:, None, None] - 0.01)            # 1 cm: link origin inside the finger
           + torch.relu(h.abs() - ctx.cup_half_height[:, None, None]))
    return torch.exp(-gap.min(-1).values / 0.02).mean(-1)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    approach_done = ctx.approach_done.bool()
    envelope_done = approach_done & ctx.envelope_done.bool()
    m1 = (~approach_done).float()
    m2 = (approach_done & ~envelope_done).float()
    m3 = envelope_done.float()

    orient = _orientation_score(ctx)
    palm_c, thumb_c, finger_c, digit_touch = _contact_terms(ctx)
    cup_disp = torch.norm(ctx.cup_pos - ctx.cup_spawn_pos, dim=-1)
    # halves the task reward when a hand link is more than 1 cm into the table (termination at 3 cm)
    table_ok = 0.5 + 0.5 * torch.clamp((ctx.hand_z_min - (ctx.table_z - 0.03)) / 0.02, 0.0, 1.0)

    # ---- stage 1: approach with fixed orientation and default pose, no digit contact ------------
    err1 = _palm_target_error(ctx, gap_target=0.01)
    place1 = 0.5 * (1.0 - torch.tanh(err1 / 0.5)) + 0.5 * torch.exp(-err1 / 0.03)  # broad pull + fine finish
    calm1 = torch.exp(-cup_disp / 0.05) * torch.exp(-ctx.cup_tilt / 0.3)          # do not push or tip the cup
    s1 = place1 * orient * _default_pose_score(ctx) * (1.0 - 0.5 * digit_touch) * calm1

    # ---- stage 2: envelope grasp from the approach position ---------------------------------------
    err2 = _palm_target_error(ctx, gap_target=0.005)                               # palm against the cup side
    place2 = 0.3 * (1.0 - torch.tanh(err2 / 0.1)) + 0.7 * torch.exp(-err2 / 0.03)
    grip2 = 0.15 * _wrap_score(ctx) + 0.2 * palm_c + 0.25 * thumb_c + 0.4 * finger_c
    calm2 = torch.exp(-cup_disp / 0.05) * torch.exp(-ctx.cup_tilt / 0.25)          # no push, tip or early lift
    s2 = place2 * orient * calm2 * (0.3 + 0.7 * grip2)

    # ---- stage 3: keep the grasp, lift to the goal, hold upright and still -----------------------
    grip3 = 0.2 * palm_c + 0.3 * thumb_c + 0.5 * finger_c
    lift_h = torch.clamp(ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.0)
    lift_prog = torch.clamp(lift_h / 0.21, max=1.0)                                # up to the lowest goal height
    goal_broad = 1.0 - torch.tanh(ctx.goal_dist / 0.25)
    goal_fine = torch.exp(-ctx.goal_dist / 0.02)
    still = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1 - torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    in_tol = (ctx.goal_dist <= ctx.success_tol).float()
    upright = torch.exp(-ctx.cup_tilt / 0.2)
    s3 = (0.25 + 0.75 * grip3) * upright * (
        0.25 * lift_prog + 0.25 * goal_broad + 0.25 * goal_fine + 0.25 * in_tol * still)

    approach = m1 * W_APPROACH * s1 * table_ok
    grasp = m2 * W_GRASP * s2 * table_ok
    lift = m3 * W_LIFT * s3 * table_ok
    stage_base = (m2 * BASE_GRASP + m3 * BASE_LIFT) * table_ok

    success = (ctx.success.bool() & envelope_done).float()
    final = (ctx.num_successes >= (ctx.max_successes - 1)).float()                 # robust to either count convention
    success_bonus = success * (SUCCESS_BONUS + FINAL_SUCCESS_BONUS * final)

    # ---- regularization: bounded smoothness bonus (jitter under the 0-2 step delay, fast arm motion) --
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(-1)
    arm_speed = (ctx.arm_qd ** 2).mean(-1)
    smoothness = W_SMOOTH * torch.exp(-act_rate / 0.5 - arm_speed / 0.1)

    reward = approach + grasp + lift + stage_base + success_bonus + smoothness
    return reward, {
        "approach": approach,
        "grasp": grasp,
        "lift": lift,
        "stage_base": stage_base,
        "success_bonus": success_bonus,
        "smoothness": smoothness,
    }
```
