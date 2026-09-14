## 1. What the task means

The right arm with the DG-5F hand has to do a **power (envelope) grasp** on an upright cylinder, then carry it to a goal above its spawn point and hold it there. An envelope grasp is not a fingertip pinch. The palm presses against the side of the cup, and the fingers curl around the cylinder so that several links of each finger touch it, not only the tips. The thumb wraps the other way, so the thumb and the four fingers end up on opposite sides of the cup with the palm behind them. This closes the grip geometrically: the cup cannot slide out horizontally in any direction, and friction on many contacts carries its weight.

Things that make this hard with this action space:
* There is no grasp primitive. The policy sets 13 movable finger joints directly, plus 7 arm joint increments. The arm is slow (about 0.15 rad/s per joint), so reaching the cup takes a while.
* Cup radius varies from 29 to 44 mm, so the geometry terms must use `ctx.cup_radius` and never a fixed number.
* The policy cannot see contact, so the reward alone has to shape the contacts. Contact forces can spike, so every force term must saturate.
* A pinch lift (thumb plus one finger) is a much easier local optimum than a true envelope. The reward has to make the envelope clearly worth more **during** lift and hold, not only before.

### Stages the robot goes through

1. **Approach and orient the palm.** Move the palm up to the side of the cup, at the height of the graspable band (`|h| <= cup_half_height`), with `palm_normal` pointing at the cylinder axis (horizontal when the cup is upright).
2. **Pre-wrap.** Bring the links of all five fingers onto the cylinder surface inside the band. The thumb goes to the opposite side from the index, middle, ring and pinky, and the palm sits between the two groups.
3. **Close and make contact.** Flex the fingers until the palm, the thumb and every other finger touch the cup with several links. Then keep commanding past the contact angle so each finger actually presses (PD target above the measured angle).
4. **Lift.** Raise the cup while the grip holds.
5. **Go to the goal and hold.** Reduce `goal_dist` (position plus uprightness, because the keypoints are fixed on the cup), keep the cup still, and let the environment count successes until `max_successes`.

Across all stages: don't tip the cup over, don't shove it across the table before grasping, don't push hand links into the table, and don't make jerky actions.

## 2. Design of the reward

All shaping terms are bounded in [0, 1] and multiplied by a weight. Penalties are bounded too, so no single term can dominate.

**Cylinder geometry (helper).** For any point, the reward computes the axial coordinate `h`, the radial vector and the radial distance, exactly as described in the spec. It does this for the palm and for all 15 measured finger links.

**Contact score (helper).** `s = 1 - exp(-F / 0.5 N)`. For example, 0.1 N gives 0.18, 0.5 N gives 0.63 and 2 N or more gives about 1. A force spike cannot inflate the reward.

**Stage 1: `approach` (w = 1).**
* Palm surface distance `d = relu(r_palm - R - 3 cm) + relu(|h_palm| - half_height)`. The 3 cm slack allows for the unknown offset between the palm frame origin and the palm skin.
* Proximity is `1 - tanh(10 d)`. When the palm actually touches the cup, proximity is replaced by the palm contact score.
* Proximity is multiplied by an orientation factor `0.3 + 0.7 * ((cos + 1) / 2)^2`, where `cos = palm_normal · (-radial unit)`. So the palm faces the axis.

**Stage 2 part 1: `finger_reach` (w = 1).** For each of the 15 links, the distance to the cylinder surface inside the band is `relu(|r - R| - 1.5 cm) + relu(|h| - half_height)`, scored with `exp(-20 d)`. If the link touches the cup, its contact score is used instead. The term is the mean over all links. This pulls every finger link, not only the tips, onto the band, which is what a wrap needs.

**Stage 2 part 2: `wrap` (w = 1).** This is the geometric definition of the envelope.
* Each finger's angular position around the axis is the normalised sum of the radial unit vectors of its three links.
* The four non-thumb fingers are merged into one direction.
* `opposition = (1 - thumb · four) / 2` is 1 when the thumb is diametrically opposite the four fingers.
* `enclosure = 1 - |mean(palm dir, thumb dir, four-finger dir)|` is 1 when the three contact groups surround the cup, and 0 when they are all on one side.
* The product `enclosure * opposition` peaks when the thumb and four fingers sit roughly 100 to 120 degrees on either side of the palm. That is the classic power-grasp layout, and it does not depend on cup radius or hand orientation.
* The term is gated by how close the palm, thumb and other fingers are to the surface, so it cannot pay out while the hand is far away.

**Stage 3: contact (all these scores come from the cup-only contact forces).**
* `contact` (w = 1.5) is the mean of 6 group scores: palm, and each of the 5 fingers (the maximum over its links). It is dense and pays for every new group that touches.
* `envelope` (w = 2) is the **geometric mean** of the same 6 scores. It is close to zero unless the palm **and all five fingers** touch, so it is the term that asks for the complete envelope.
* `link_cover` (w = 1) is the mean contact score over all 15 links. It pays for wrapping with the outer links, not only touching with the tips.
* `squeeze` (w = 0.5) applies per finger. It is the finger's contact score times its normalised PD "press" (`hand_target_norm - hand_q_norm`, saturating at 0.2). A finger stopped by the cup that keeps pressing makes a firm grip. A finger closing in free air (no cup contact) or pressing on the table or on itself earns nothing.

**Grip factor used by all later stages.**
* `held = max(thumb, palm) * max(index..pinky)` requires two opposing contact groups, so pushing or tossing the cup is not rewarded as lifting.
* `grip_quality = 0.3 * contact + 0.7 * envelope`.
* `grip = held * (0.2 + 0.8 * grip_quality)`.
* A pinch lift receives only about 25% of the lift, goal and hold reward. A full envelope receives 100%. The pinch optimum therefore stays clearly worse for the whole episode, not only during the grasp phase. The 0.3 share of the dense `contact` mean keeps the factor from collapsing to its floor when one link loses contact for a single step.

**Stage 4: `lift` (w = 4).** `clamp(height / goal_rise, 0, 1) * grip`, where height is `cup_z - spawn_z` and `goal_rise` is the goal's height above the spawn point. The term stops increasing at goal height, so there is no reason to overshoot.

**Stage 5: goal and hold.**
* `goal` (w = 6): `0.5 * (1 - tanh(5 * goal_dist)) + 0.5 * (1 - tanh(30 * goal_dist))`, multiplied by `grip` and by the `lifted` mask. The coarse part guides from about 25 cm away. The fine part keeps a useful gradient near the tolerance, which shrinks toward 1.5 cm. `goal_dist` already contains tilt, so this also rewards keeping the cup upright.
* `hold_still` (w = 2): only near the goal (`exp(-goal_dist / 3 cm)`), times `exp(-|v_cup| / 0.05) * exp(-|w_cup| / 0.5)`, times `grip` and `lifted`. It rewards a steady hold so that the success counter keeps running.
* `success_bonus` (w = 25 per counted success): scaled by `0.25 + 0.75 * grip_quality`, so successes reached with the real envelope pay the most.

**Penalties (constraints on object and robot).**
* `tilt_penalty` (w = 2): `(tilt / 60 deg)^2`. It is gentle for the small tilts of normal handling and grows toward the 60-degree termination.
* `push_penalty` (w = 1): only before `lifted`. It is `tanh(relu(xy_displacement - 1 cm) / 5 cm)` and discourages shoving or knocking the cup across the table during approach and closing. The 1 cm dead zone allows normal contact.
* `table_penalty` (w = 3): `clamp((table_z - hand_z_min) / 3 cm, 0, 1)`. It rises linearly from 0 at the table top to its maximum at the 3 cm termination depth.
* `arm_rate_penalty` (w = 0.02) and `hand_rate_penalty` (w = 0.005): squared action differences. The hand penalty only counts the 13 movable joints, because the locked ones do nothing.

**Scale check.**
* Before the grasp, the positive terms add up to at most about 8 per step.
* A full-envelope lift and hold adds up to about 12 more per step.
* Each success adds 25.
* All penalties together stay within a few units and are zero in normal operation.
* The reward is positive for normal behaviour, so early termination (cup knocked over or dropped, hand in the table) always loses future reward and is never a shortcut.
* A drop removes contact, so `grip`, and with it lift, goal and hold, vanishes immediately, even though `lifted` stays latched.

Fields not used: `palm_side` (its orientation in the palm is not specified), `cup_quat` (`cup_tilt` and `cup_axis` already cover it), `arm_q`, `hand_q`, `hand_qd`, `arm_qd` and `episode_progress`. The function keeps no state. It returns weighted component tensors whose sum is exactly the reward.

## 3. Reward function

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
    """Saturating contact score in [0,1]: 0.1 N -> 0.18, 0.5 N -> 0.63, >= 2 N -> ~1 (spike-proof)."""
    return 1.0 - torch.exp(-force.clamp(min=0.0) / f0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ---------------- weights ----------------
    W_APPROACH = 1.0     # stage 1: palm to the side of the cup, facing its axis
    W_REACH = 1.0        # stage 2: all 15 finger links onto the cylinder band
    W_WRAP = 1.0         # stage 2: thumb opposite the four fingers, palm between (enclosure)
    W_CONTACT = 1.5      # stage 3: dense, each touching group (palm + 5 fingers) counts
    W_ENVELOPE = 2.0     # stage 3: geometric mean -> pays only when palm AND all five fingers touch
    W_LINKS = 1.0        # stage 3: several links per finger touching (wrap, not tip pinch)
    W_SQUEEZE = 0.5      # stage 3: fingers in contact keep pressing (PD target beyond actual)
    W_LIFT = 4.0         # stage 4: raise the cup, scaled by grip quality
    W_GOAL = 6.0         # stage 5: bring the cup to the goal (goal_dist includes tilt)
    W_STILL = 2.0        # stage 5: steady hold near the goal so successes keep counting
    W_SUCCESS = 25.0     # per counted success, scaled by envelope quality
    W_TILT = 2.0         # do not knock the cup over
    W_PUSH = 1.0         # do not shove the cup before it is lifted
    W_TABLE = 3.0        # do not press hand links into the table
    W_ARM_RATE = 0.02    # smooth arm actions
    W_HAND_RATE = 0.005  # smooth finger actions (movable joints only)

    # movable finger joints per finger (hand indices): thumb, index, middle, ring, pinky
    finger_joints = [[1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18]]
    movable = [j for group in finger_joints for j in group]

    axis = ctx.cup_axis
    radius = ctx.cup_radius
    half_h = ctx.cup_half_height

    # ---------------- stage 1: palm approach + orientation ----------------
    h_p, rad_p, r_p = _cyl_coords(ctx.palm_pos, ctx.cup_pos, axis)
    u_palm = _unit(rad_p)
    palm_d = torch.relu(r_p - radius - 0.03) + torch.relu(h_p.abs() - half_h)  # 3 cm slack: palm origin vs skin
    s_palm = _contact_score(ctx.palm_cup_force)
    palm_prox = torch.maximum(1.0 - torch.tanh(10.0 * palm_d), s_palm)
    palm_align = (ctx.palm_normal * (-u_palm)).sum(-1)  # 1 = palm faces the cup axis
    orient = ((palm_align + 1.0) * 0.5) ** 2
    approach = palm_prox * (0.3 + 0.7 * orient)

    # ---------------- stage 2: finger links onto the band ----------------
    c = ctx.cup_pos[:, None, None, :]
    a = axis[:, None, None, :]
    h_l, rad_l, r_l = _cyl_coords(ctx.link_pos, c, a)  # (N,5,3)
    link_d = torch.relu((r_l - radius[:, None, None]).abs() - 0.015) + torch.relu(h_l.abs() - half_h[:, None, None])
    s_link = _contact_score(ctx.link_cup_force)  # (N,5,3)
    link_prox = torch.maximum(torch.exp(-20.0 * link_d), s_link)
    finger_reach = link_prox.mean(dim=(1, 2))

    # ---------------- stage 2: envelope geometry (opposition + enclosure) ----------------
    dir_f = _unit(_unit(rad_l).sum(dim=2))           # (N,5,3) angular position of each finger around the axis
    dir_thumb = dir_f[:, 0]
    dir_four = _unit(dir_f[:, 1:].sum(dim=1))        # (N,3) index..pinky as one group
    opposition = 0.5 * (1.0 - (dir_thumb * dir_four).sum(-1))
    enclosure = (1.0 - ((u_palm + dir_thumb + dir_four) / 3.0).norm(dim=-1)).clamp(0.0, 1.0)
    near_f = link_prox.mean(dim=2)                   # (N,5)
    wrap_gate = (palm_prox * near_f[:, 0] * near_f[:, 1:].mean(dim=1)).clamp(min=0.0).pow(1.0 / 3.0)
    wrap = enclosure * opposition * wrap_gate

    # ---------------- stage 3: contact ----------------
    s_finger = s_link.max(dim=2).values              # (N,5)
    groups = torch.cat([s_palm.unsqueeze(1), s_finger], dim=1)  # (N,6) palm, thumb, index, middle, ring, pinky
    contact = groups.mean(dim=1)
    envelope = groups.clamp(min=0.0).prod(dim=1).pow(1.0 / 6.0)
    link_cover = s_link.mean(dim=(1, 2))

    press = ((ctx.hand_target_norm - ctx.hand_q_norm) / 0.2).clamp(0.0, 1.0)  # (N,19)
    press_f = torch.stack([press[:, j].mean(dim=1) for j in finger_joints], dim=1)  # (N,5)
    squeeze = (s_finger * press_f).mean(dim=1)

    # grip factor: two opposing groups needed; full envelope is worth ~4x a pinch
    held = torch.maximum(s_finger[:, 0], s_palm) * s_finger[:, 1:].max(dim=1).values
    grip_quality = 0.3 * contact + 0.7 * envelope
    grip = held * (0.2 + 0.8 * grip_quality)

    # ---------------- stage 4: lift ----------------
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    goal_rise = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    lift = (height / goal_rise).clamp(0.0, 1.0) * grip

    # ---------------- stage 5: goal + still hold ----------------
    zeros = torch.zeros_like(height)
    goal_track = 0.5 * (1.0 - torch.tanh(5.0 * ctx.goal_dist)) + 0.5 * (1.0 - torch.tanh(30.0 * ctx.goal_dist))
    goal = torch.where(ctx.lifted, goal_track * grip, zeros)

    lin_speed = ctx.cup_lin_vel.norm(dim=-1)
    ang_speed = ctx.cup_ang_vel.norm(dim=-1)
    steady = torch.exp(-ctx.goal_dist / 0.03) * torch.exp(-lin_speed / 0.05) * torch.exp(-ang_speed / 0.5)
    hold_still = torch.where(ctx.lifted, steady * grip, zeros)

    success_bonus = ctx.success.float() * (0.25 + 0.75 * grip_quality)

    # ---------------- penalties ----------------
    tilt_pen = (ctx.cup_tilt / (math.pi / 3.0)).clamp(0.0, 1.0) ** 2
    xy_disp = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    push_pen = torch.where(ctx.lifted, zeros, torch.tanh(torch.relu(xy_disp - 0.01) / 0.05))
    table_pen = ((ctx.table_z - ctx.hand_z_min) / 0.03).clamp(0.0, 1.0)
    d_act = ctx.actions - ctx.prev_actions
    arm_rate = (d_act[:, :7] ** 2).sum(-1)
    hand_rate = (d_act[:, 7:][:, movable] ** 2).sum(-1)

    components = {
        "approach": W_APPROACH * approach,
        "finger_reach": W_REACH * finger_reach,
        "wrap": W_WRAP * wrap,
        "contact": W_CONTACT * contact,
        "envelope": W_ENVELOPE * envelope,
        "link_cover": W_LINKS * link_cover,
        "squeeze": W_SQUEEZE * squeeze,
        "lift": W_LIFT * lift,
        "goal": W_GOAL * goal,
        "hold_still": W_STILL * hold_still,
        "success_bonus": W_SUCCESS * success_bonus,
        "tilt_penalty": -W_TILT * tilt_pen,
        "push_penalty": -W_PUSH * push_pen,
        "table_penalty": -W_TABLE * table_pen,
        "arm_rate_penalty": -W_ARM_RATE * arm_rate,
        "hand_rate_penalty": -W_HAND_RATE * hand_rate,
    }
    reward = torch.stack(list(components.values()), dim=0).sum(dim=0)
    return reward, components
```

