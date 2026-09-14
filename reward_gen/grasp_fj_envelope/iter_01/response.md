## 1. What the task means and which stages it has

The robot has to pick up a cup standing on the table and hold it at a goal about 21–28 cm higher, upright and still. It has to use a power grasp: the palm flat against the side of the cup, the four fingers curled around one side and the thumb around the other, so that the palm and the finger links touch the cup, not only the fingertips.

The environment counts a success each time the cup stays within `success_tol` of the upright goal pose for `success_hold_steps` consecutive steps. The keypoints are fixed on the cup, so both a position error and a tilt increase `goal_dist`. The tolerance shrinks from 11 cm to 1.5 cm during training. By the end, only a cup that is at the goal *and* within a few degrees of upright produces successes.

Stages:

1. **Reach.** Move the palm to the side of the cup at the height of the graspable band, with the palm normal pointing at the cup axis. Do not shove the cup.
2. **Enclose.** Bring the fingers around the cylinder: the four fingers on one side, the thumb on the opposite side, the palm in between.
3. **Close and wrap.** The palm and all five fingers touch the cup. Ideally the two outer links of each finger lie along the cup, and the fingers press moderately. The cup stays upright on the table.
4. **Lift.** Raise the cup and keep it upright *while* raising it. Lifting with the shoulder and elbow also rotates the hand, so the wrist joints must counter-rotate during the lift. The tilt has to be prevented as it starts, because fixing it later is too slow. Each arm joint moves at most about 0.15 rad/s, so undoing a 45° tilt (0.79 rad) takes more than 5 s of the 10 s budget.
5. **Hold.** Bring the cup inside the tolerance at the goal and keep it there, upright and still, for consecutive steps. The arm actions must be close to 0 because they are increments, and the fingers must stay steady. This repeats until `max_successes`.

At every stage the cup must not tip (more than 60° ends the episode), must not be dropped, and no hand link may be pushed into the table.

## 2. What was missed or wrong in the previous reward

### 2.1 Carrying the cup tilted paid off

- Every grasp term (approach, finger_reach, wrap, contact, envelope, link_cover, squeeze) is measured in the cup's own frame. These terms do not change when the hand and the cup tilt together.
- The two largest positive terms ignored tilt. `lift` (W=4, 2.6 per step at the end) uses height only. `goal` (W=6) sees tilt only through `goal_dist`. With a tolerance of 7–11 cm, a cup tilted by about 45° still falls inside it, so successes kept being counted.
- The tilt penalty `2·(θ/60°)²` was only −1.1 at 45°.
- Rough per-step values, for the same grasp and relative to leaving the cup on the table (`goal_dist` ≈ 1 cm upright, ≈ 5 cm at 45°): upright carry at the goal about +6.2, 45° carry about +3.4. A tilted carry earned more than half of an upright carry, needed no wrist coordination, and avoided the slow (> 5 s) rotation back to upright. So the policy learned to lift tilted.
- The metrics show the cost:
  - Mean tilt rose from 24° to 27–30°.
  - Successes fell from 3.5 to 2.3 while the tolerance went from 11.25 cm to 7.4–8.2 cm.
  - `done/tipped` (≈ 0.00098 per step) is almost as frequent as `done/max_goals` (≈ 0.00114). With episodes of about 430 steps, roughly 40 % of all episodes end because the cup tips past 60°. That matches the "reset after about 4–5 s" in the video.

### 2.2 `hold_still` could never pay

- It was the product `exp(-goal_dist/0.03) · exp(-v/0.05) · exp(-ω/0.5) · grip`.
- With a tilted cup, `goal_dist` is several centimetres, so the first factor is about 0.1–0.2. A cup held in a jittering hand has ω of about 1 rad/s (factor about 0.1) and v of a few cm/s (factor about 0.4). The product is about 0.001, which is exactly the logged value.
- A term that is close to zero everywhere gives no gradient.
- Nothing rewarded the one thing that stops the arm. Arm actions are increments, so the arm is still only when `actions[0:7]` ≈ 0. PPO exploration noise keeps the arm target drifting, and the cup keeps moving.

### 2.3 One link per finger was enough

- `contact`, `envelope` and the grip factor `held` all take the maximum over a finger's links, so a single fingertip already scores full marks.
- `link_cover` paid only 1/15 for each extra link and had no effect on lift, goal or success.
- `squeeze` paid for pressing wherever the finger touched. The result was hard fingertip presses: mean link force 4.5 N, with spikes up to 508 N.
- A grasp on the fingertips alone has few contact points to resist the cup's torque. Tilting the hand so that the cup rests on the palm and fingers is an easy workaround. The missing wrap and the tilted carry therefore reinforce each other.

## 3. What changes

**Kept unchanged:** approach, finger_reach, wrap (opposition and enclosure), contact, envelope, and the push, table and action-rate penalties. The grasp that surrounds the cup works.

1. **Upright factor.** `up(θ) = 0.5·exp(-(θ/0.5)²) + 0.5·exp(-(θ/0.15)²)`.
   - Values: 5° → 0.84, 10° → 0.57, 20° → 0.31, 30° → 0.17, 45° → 0.04.
   - The coarse part keeps a gradient even from large tilts. The fine part asks for the few degrees that the final 1.5 cm tolerance needs.
   - `up` multiplies every reward earned by a raised cup: `lift`, `goal`, `hold_still`, the new `hold_wrap`, and `success_bonus` (as `0.2 + 0.8·up`).
2. **Carry-phase tilt penalty.**
   - `carry = max(clamp(height / lift_latch_height, 0, 1), lifted)` ramps in from the first millimetres of the lift. Uprightness therefore counts during the lift itself, not only after the latch.
   - The tilt weight rises from 2 (cup on the table) to 5 (cup raised).
   - The cost is `0.5·x + 0.5·x²` with `x = θ/60°`. The linear part keeps a slope close to upright.
   - Rough per-step values for a fingertip grasp at the goal, relative to leaving the cup on the table:

     | tilt while carried | old reward | new reward |
     |---|---|---|
     | 0° | +6.2 | +6.0 |
     | 10° | – | +2.7 |
     | 20° | – | +0.5 |
     | 30° | – | −1.0 |
     | 45° | +3.4 | −3.1 |

   - A tilted carry is now worse than not lifting at all. The reward rises steadily from 45° down to 0°.
   - While the grasp is held, the grasp terms (about +5) keep the total per-step reward positive even near 60°. Tipping the cup on purpose to end the episode is therefore not attractive.
3. **Steady hold.**
   - `hold_still = lifted · in_tol · grip · up · (0.3 + 0.7·still)`. The weight goes from 2 to 4.
   - `in_tol = sigmoid((success_tol − goal_dist) / (0.2·success_tol))` measures being inside the current tolerance. Because it is relative to the tolerance, it stays meaningful while the tolerance shrinks. It is a dense version of what a success needs: staying inside the tolerance on consecutive steps.
   - `still = exp(−v/0.1 − ω/1.0)` uses realistic velocity scales, so it is not always close to zero.
   - The 0.3 base pays for being inside the tolerance even while the cup still moves, which keeps a gradient.
   - A new `hold_arm_action_penalty` (0.3 · in_tol · mean of the squared arm actions) rewards zero arm increments at the goal, so the arm target stops drifting. It is quadratic, so small corrections are cheap.
4. **Wrapping with more of each finger.**
   - Link coverage uses a light-touch contact score with f0 = 0.25 N (0.1 N → 0.33). The task asks for links to *touch* the cup, not to press on it.
   - `extra_links` is, averaged over the fingers, the mean score of each finger's two links other than its best one. It is 0 for a press with the fingertip alone and 1 when all three measured links touch.
   - `extra_links` enters `grip_quality = 0.2·contact + 0.5·envelope + 0.3·extra_links`, so it scales lift, goal, hold and success.
   - A new `hold_wrap = carry · held · extra_links · up` (W = 3) pays for wrapping while the cup is carried upright.
   - `squeeze` now uses each finger's link coverage instead of its maximum, and it saturates at a normalised gap of 0.15. Pressing hard with one fingertip no longer earns the full squeeze reward.
   - When the cup is held upright at the goal, going from fingertips only to almost all links touching raises the per-step reward above the table baseline from about +6 to about +10.

## 4. Expected changes in the metrics

- `task/tilt_deg` falls, with a lifted cup below about 10°.
- `done/tipped` falls.
- `ctrl/prev_ep_successes_mean` stops dropping as the tolerance shrinks, and the tolerance can keep tightening.
- `reward/hold_still` moves well above 0, to about 1–3 per step while the cup is held.
- `reward/tilt_penalty` may dip at first, then shrink towards 0.
- `contact/links_touching` rises from about 5 towards 8–10, and `contact/link_force_mean` may drop.
- `task/lifted_frac` may dip for a while as the policy learns the coordinated upright lift. If it collapses towards 0 and stays there, lower `W_TILT_CARRY` to 3–4.

```python
import torch
import math


def _unit(v: torch.Tensor) -> torch.Tensor:
    """Normalise the last dimension safely."""
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _cyl_coords(p: torch.Tensor, cup_pos: torch.Tensor, axis: torch.Tensor):
    """Axial coordinate h, radial vector and radial distance of points p w.r.t. the cup axis."""
    v = p - cup_pos
    h = (v * axis).sum(-1)
    rad = v - h.unsqueeze(-1) * axis
    return h, rad, rad.norm(dim=-1)


def _contact_score(force: torch.Tensor, f0: float = 0.5) -> torch.Tensor:
    """Saturating contact score in [0,1] (spike-proof). f0=0.5: 0.1 N -> 0.18, 0.5 N -> 0.63, 2 N -> 0.98."""
    return 1.0 - torch.exp(-force.clamp(min=0.0) / f0)


def _upright_factor(tilt: torch.Tensor) -> torch.Tensor:
    """1 when upright. Coarse part (0.5 rad) keeps a gradient from large tilts, fine part (0.15 rad)
    asks for the few degrees the final 1.5 cm tolerance needs.
    5 deg -> 0.84, 10 deg -> 0.57, 20 deg -> 0.31, 30 deg -> 0.17, 45 deg -> 0.04."""
    return 0.5 * torch.exp(-(tilt / 0.5) ** 2) + 0.5 * torch.exp(-(tilt / 0.15) ** 2)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ---------------- weights ----------------
    W_APPROACH = 1.0      # stage 1: palm to the side of the cup, facing its axis (unchanged)
    W_REACH = 1.0         # stage 2: all 15 finger links onto the cylinder band (unchanged)
    W_WRAP = 1.0          # stage 2: thumb opposite the four fingers, palm between (unchanged)
    W_CONTACT = 1.5       # stage 3: each touching group (palm + 5 fingers) counts (unchanged)
    W_ENVELOPE = 2.0      # stage 3: pays only when palm AND all five fingers touch (unchanged)
    W_LINKS = 1.0         # stage 3: share of the 15 measured links touching (light touch counts)
    W_SQUEEZE = 0.5       # stage 3: pressing, now scaled by how much of each finger touches
    W_HOLD_WRAP = 3.0     # NEW stage 4-5: 2nd/3rd link of every finger on the cup while carrying upright
    W_LIFT = 4.0          # stage 4: raise the cup, x grip x upright factor
    W_GOAL = 6.0          # stage 5: bring the cup to the goal, x grip x upright factor
    W_HOLD = 4.0          # stage 5: inside the success tolerance, upright and still (was 2 and ~0)
    W_SUCCESS = 25.0      # per counted success, x grasp quality x uprightness
    W_TILT_TABLE = 2.0    # tilt cost while the cup stands on the table (as before)
    W_TILT_CARRY = 5.0    # NEW: tilt cost once the cup is raised -> 45 deg carry costs ~3.3/step
    W_PUSH = 1.0          # do not shove the cup before it is lifted
    W_TABLE = 3.0         # do not press hand links into the table
    W_HOLD_ACT = 0.3      # NEW: arm increments while inside the tolerance (arm target stops drifting)
    W_ARM_RATE = 0.02     # smooth arm actions
    W_HAND_RATE = 0.005   # smooth finger actions (movable joints only)

    # movable finger joints per finger (hand indices): thumb, index, middle, ring, pinky
    finger_joints = [[1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18]]
    movable = [j for group in finger_joints for j in group]

    axis = ctx.cup_axis
    radius = ctx.cup_radius
    half_h = ctx.cup_half_height
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    zeros = torch.zeros_like(height)
    lifted_f = ctx.lifted.to(height.dtype)

    # ---------------- stage 1: palm approach + orientation (unchanged) ----------------
    h_p, rad_p, r_p = _cyl_coords(ctx.palm_pos, ctx.cup_pos, axis)
    u_palm = _unit(rad_p)
    palm_d = torch.relu(r_p - radius - 0.03) + torch.relu(h_p.abs() - half_h)  # 3 cm slack: palm origin vs skin
    s_palm = _contact_score(ctx.palm_cup_force)
    palm_prox = torch.maximum(1.0 - torch.tanh(10.0 * palm_d), s_palm)
    palm_align = (ctx.palm_normal * (-u_palm)).sum(-1)  # 1 = palm faces the cup axis
    orient = ((palm_align + 1.0) * 0.5) ** 2
    approach = palm_prox * (0.3 + 0.7 * orient)

    # ---------------- stage 2: finger links onto the band (unchanged) ----------------
    c = ctx.cup_pos[:, None, None, :]
    a = axis[:, None, None, :]
    h_l, rad_l, r_l = _cyl_coords(ctx.link_pos, c, a)  # (N,5,3)
    link_d = torch.relu((r_l - radius[:, None, None]).abs() - 0.015) + torch.relu(h_l.abs() - half_h[:, None, None])
    s_link = _contact_score(ctx.link_cup_force)  # (N,5,3)
    link_prox = torch.maximum(torch.exp(-20.0 * link_d), s_link)
    finger_reach = link_prox.mean(dim=(1, 2))

    # ---------------- stage 2: envelope geometry, opposition + enclosure (unchanged) ----------------
    dir_f = _unit(_unit(rad_l).sum(dim=2))           # (N,5,3) angular position of each finger around the axis
    dir_thumb = dir_f[:, 0]
    dir_four = _unit(dir_f[:, 1:].sum(dim=1))        # (N,3) index..pinky as one group
    opposition = 0.5 * (1.0 - (dir_thumb * dir_four).sum(-1))
    enclosure = (1.0 - ((u_palm + dir_thumb + dir_four) / 3.0).norm(dim=-1)).clamp(0.0, 1.0)
    near_f = link_prox.mean(dim=2)                   # (N,5)
    wrap_gate = (palm_prox * near_f[:, 0] * near_f[:, 1:].mean(dim=1)).clamp(min=0.0).pow(1.0 / 3.0)
    wrap = enclosure * opposition * wrap_gate

    # ---------------- stage 3: contact and wrap coverage ----------------
    s_finger = s_link.max(dim=2).values              # (N,5)
    groups = torch.cat([s_palm.unsqueeze(1), s_finger], dim=1)  # (N,6) palm, thumb, index, middle, ring, pinky
    contact = groups.mean(dim=1)
    envelope = groups.clamp(min=0.0).prod(dim=1).pow(1.0 / 6.0)

    s_touch = _contact_score(ctx.link_cup_force, 0.25)  # (N,5,3) light touch counts: 0.1 N -> 0.33
    link_cover = s_touch.mean(dim=(1, 2))
    finger_cover = s_touch.mean(dim=2)                  # (N,5) share of each finger's links touching
    # links beyond each finger's best one: 0 for a fingertip-only press, 1 when all three links touch
    extra_links = ((s_touch.sum(dim=2) - s_touch.max(dim=2).values) / 2.0).mean(dim=1)

    press = ((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15).clamp(0.0, 1.0)  # (N,19) moderate press saturates
    press_f = torch.stack([press[:, j].mean(dim=1) for j in finger_joints], dim=1)  # (N,5)
    squeeze = (finger_cover * press_f).mean(dim=1)   # a single pressing fingertip earns only 1/3

    # grip factor: two opposing groups needed; quality now also asks for several links per finger
    held = torch.maximum(s_finger[:, 0], s_palm) * s_finger[:, 1:].max(dim=1).values
    grip_quality = 0.2 * contact + 0.5 * envelope + 0.3 * extra_links
    grip = held * (0.2 + 0.8 * grip_quality)

    # ---------------- uprightness: counts from the first millimetre of lift ----------------
    up = _upright_factor(ctx.cup_tilt)
    raised = (height / max(float(ctx.lift_latch_height), 0.01)).clamp(0.0, 1.0)
    carry = torch.maximum(raised, lifted_f)          # 0 on the table -> 1 once raised / lifted

    # ---------------- stage 4: lift (upright only) ----------------
    goal_rise = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    lift = (height / goal_rise).clamp(0.0, 1.0) * grip * up
    hold_wrap = carry * held * extra_links * up

    # ---------------- stage 5: goal, steady hold inside the tolerance, success ----------------
    goal_track = 0.5 * (1.0 - torch.tanh(5.0 * ctx.goal_dist)) + 0.5 * (1.0 - torch.tanh(30.0 * ctx.goal_dist))
    goal = torch.where(ctx.lifted, goal_track * grip * up, zeros)

    tol = ctx.success_tol.clamp(min=1e-3)
    in_tol = torch.sigmoid((tol - ctx.goal_dist) / (0.2 * tol))  # ~1 inside the current tolerance, 0.5 at its edge
    lin_speed = ctx.cup_lin_vel.norm(dim=-1)
    ang_speed = ctx.cup_ang_vel.norm(dim=-1)
    still = torch.exp(-lin_speed / 0.1 - ang_speed / 1.0)        # 5 cm/s + 0.5 rad/s -> 0.37
    hold_still = torch.where(ctx.lifted, in_tol * grip * up * (0.3 + 0.7 * still), zeros)

    success_bonus = ctx.success.to(height.dtype) * (0.25 + 0.75 * grip_quality) * (0.2 + 0.8 * up)

    # ---------------- penalties ----------------
    x_tilt = (ctx.cup_tilt / (math.pi / 3.0)).clamp(0.0, 1.0)    # 1 at the 60 deg termination angle
    tilt_cost = 0.5 * x_tilt + 0.5 * x_tilt ** 2                 # linear part keeps a slope near upright
    tilt_w = W_TILT_TABLE + (W_TILT_CARRY - W_TILT_TABLE) * carry
    xy_disp = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    push_pen = torch.where(ctx.lifted, zeros, torch.tanh(torch.relu(xy_disp - 0.01) / 0.05))
    table_pen = ((ctx.table_z - ctx.hand_z_min) / 0.03).clamp(0.0, 1.0)
    d_act = ctx.actions - ctx.prev_actions
    arm_rate = (d_act[:, :7] ** 2).sum(-1)
    hand_rate = (d_act[:, 7:][:, movable] ** 2).sum(-1)
    hold_act = torch.where(ctx.lifted, in_tol * (ctx.actions[:, :7] ** 2).mean(dim=-1), zeros)

    components = {
        "approach": W_APPROACH * approach,
        "finger_reach": W_REACH * finger_reach,
        "wrap": W_WRAP * wrap,
        "contact": W_CONTACT * contact,
        "envelope": W_ENVELOPE * envelope,
        "link_cover": W_LINKS * link_cover,
        "squeeze": W_SQUEEZE * squeeze,
        "hold_wrap": W_HOLD_WRAP * hold_wrap,
        "lift": W_LIFT * lift,
        "goal": W_GOAL * goal,
        "hold_still": W_HOLD * hold_still,
        "success_bonus": W_SUCCESS * success_bonus,
        "tilt_penalty": -tilt_w * tilt_cost,
        "push_penalty": -W_PUSH * push_pen,
        "table_penalty": -W_TABLE * table_pen,
        "hold_arm_action_penalty": -W_HOLD_ACT * hold_act,
        "arm_rate_penalty": -W_ARM_RATE * arm_rate,
        "hand_rate_penalty": -W_HAND_RATE * hand_rate,
    }
    reward = torch.stack(list(components.values()), dim=0).sum(dim=0)
    return reward, components
```
