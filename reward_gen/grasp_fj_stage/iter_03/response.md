## 1. What the task means, step by step

The right arm starts high and outside the table edge. The hand is turned sideways: the palm faces +y and the fingers point +x. In the default hand pose the thumb sticks straight out of the palm, 0.12 m along +y, at the wrist end of the palm. The robot has to finish with the cup held upright and still at a goal 0.21–0.28 m above where the cup started. The environment enforces three stages with sticky flags, and the reward has to follow the same order.

**Stage 1 – approach (`approach_done` is False).**
- **Where the palm must end up.** It goes to the cup's −y side and stays in the band that `approach_done` checks, in the palm frame:
  - Normal gap: the distance to the cup axis along `palm_normal`, minus the radius, must be in [−0.01, 0.02]. I aim for about 0.008 m, which is close but not touching.
  - Along the fingers: the cup axis must be ahead along `palm_finger_dir` by the radius plus a value in [−0.005, 0.02]. I aim for +0.0075 m. This keeps the thumb, at the wrist end, behind the cup, so the cup lies between the thumb and the four fingers.
  - Height: the palm must be inside the graspable band (|h| ≤ cup_half_height). I aim a little above the band centre.
- **What must hold on the way.** The orientation stays at the start orientation (normal ≈ +y, fingers ≈ +x). Every movable finger joint stays near `hand_default_q_norm`. No thumb or finger link may touch the cup, and the cup must not be pushed or tipped.
- **Why the geometry is safe.** Coming in with the palm plane just outside the cup's side and the palm not past its target along +x, the thumb never enters the cup's circle. So a smooth distance-to-target reward, plus a contact penalty, gives a collision-free path.
- **Size and generality.** Cup radius and half height differ per environment, so every target is written in terms of `cup_radius` and `cup_half_height`.

**Stage 2 – envelope grasp (`approach_done` True, `envelope_done` False).**
- **What the hand does.** It keeps the same orientation. The palm closes the last centimetre onto the cup side. The four fingers flex around the far side of the cup (the +finger direction side) and wrap back towards +y. The thumb flexes onto the near (−finger direction) side, opposite the fingers.
- **Signals I reward:**
  - palm contact and palm gap;
  - thumb contact and thumb proximity to the cup surface;
  - the number of fingers touching;
  - how far each fingertip has wrapped around the cup axis. This is the angle from the palm side towards the far side, and it works for any cup size.
  - a small "squeeze" term: once a digit touches, its command should lead its measured angle, so the PD fingers actually press.
- **What stays off-limits.** The cup should still rest on the table; lifting before the envelope exists earns no stage-2 progress. The cup should not be shoved sideways. `envelope_done` is set once the palm, the thumb and at least three fingers hold for 5 steps.

**Stage 3 – lift and hold (`envelope_done` True).**
- **What the hand does.** It keeps the grasp: palm, thumb and fingers stay in contact. It lifts the cup straight up towards `goal_pos`, keeps it upright, and holds it still so that `goal_dist <= success_tol` keeps producing successes.
- **Grip requirement.** All lift, goal and hold terms are multiplied by a grip gate: the thumb plus at least two fingers pressing. A cup that is dropped, or carried without the grasp, therefore earns nothing from them.
- **Bonus.** Each counted success gets a bonus.

**Ordering and anti-skip design.**
- **Masks.** Each stage's positive terms are masked by the flags: stage 1 by `~approach_done`, stage 2 by `approach_done & ~envelope_done`, stage 3 by `envelope_done`. Grasping or lifting without the earlier flags earns nothing positive from the later stages.
- **Stage bases.** Each stage adds a constant base larger than the most the previous stage can pay (stage 1 ≤ 3.0 < 3.5; stage 2 ≤ 9.5 < 10). Completing a stage therefore never lowers the reward, and there is no incentive to hover just before a flag.
- **Stage-3 contact terms.** Stage 3 repeats the stage-2 contact terms, so releasing after `envelope_done` loses reward, and so does any lift without the grip.
- **Penalties in every stage.** Cup tilt, hand below the table top, action rate and arm speed are penalised throughout. In stages 1–2 the cup's horizontal displacement from its spawn point is also penalised. In stage 1 any thumb or finger contact with the cup is penalised.
- **Scale.** All components are multiplied by 0.1 so that a typical per-step reward is O(1).

## 2. Reward function

```python
import torch
import math


# Movable finger joints (range > 0.05 rad), grouped per digit in hand-joint order:
# thumb (3,4), index (2,3,4), middle (2,3,4), ring (2,3,4), pinky (3,4).
_DIGIT_JOINTS = ([1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18])
_MOVABLE = [j for group in _DIGIT_JOINTS for j in group]


def _soft_touch(force: torch.Tensor, full_force: float) -> torch.Tensor:
    # 0 when not touching, 1 at/above full_force; clamping also bounds physics force spikes
    return torch.clamp(force / full_force, 0.0, 1.0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale = 0.1  # global scale so a typical per-step reward is O(1)

    r = ctx.cup_radius
    hh = ctx.cup_half_height
    n = ctx.palm_normal
    fd = ctx.palm_finger_dir
    axis = ctx.cup_axis

    # ---------------- stage masks (strict order) ----------------
    approach = ctx.approach_done.bool()
    envelope = ctx.envelope_done.bool() & approach
    s1 = (~approach).float()
    s2 = (approach & ~envelope).float()
    s3 = envelope.float()

    # ---------------- palm relative to the cup (same quantities approach_done checks) ----------------
    v = ctx.cup_pos - ctx.palm_pos
    v_ax = (v * axis).sum(-1)
    v_perp = v - v_ax.unsqueeze(-1) * axis
    gap_n = (v_perp * n).sum(-1) - r    # env window [-0.01, 0.02]
    gap_f = (v_perp * fd).sum(-1) - r   # env window [-0.005, 0.02]
    h_palm = -v_ax                      # palm height along the cup axis; env needs |h| <= hh

    # orientation kept at the start orientation (normal +y, fingers +x); ~0.05 at 45 deg on both axes
    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))
    ang_f = torch.acos(torch.clamp(fd[:, 0], -1.0, 1.0))
    ori = torch.exp(-(ang_n ** 2 + ang_f ** 2) / (2.0 * 0.45 ** 2))

    # default hand pose kept (env needs every movable joint within 0.3); mean term keeps a gradient far away
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev.amax(-1) / 0.2) ** 2)

    # ---------------- stage 1: approach ----------------
    # coarse world-frame target on the cup's -y side, slightly behind the axis along +x, upper band
    offset = torch.stack([-(r + 0.0075), -(r + 0.008), 0.25 * hh], dim=-1)
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(3.0 * d_world)
    # fine palm-frame precision centred in the approach_done window
    fine_err2 = (gap_n - 0.008) ** 2 + (gap_f - 0.0075) ** 2 + torch.relu(h_palm.abs() - 0.5 * hh) ** 2
    fine = torch.exp(-fine_err2 / (2.0 * 0.012 ** 2))
    keep_factor = 0.3 + 0.7 * ori * pose  # turning the hand or closing fingers cuts approach progress
    approach_reach = s1 * (1.0 * coarse + 1.5 * fine) * keep_factor  # max 2.5
    approach_keep = s1 * (0.25 * ori + 0.25 * pose)                  # max 0.5 -> stage 1 max 3.0

    # ---------------- contacts ----------------
    link_touch = _soft_touch(ctx.link_cup_force, 0.5)      # (N,5,3)
    digit_touch = link_touch.amax(-1)                       # (N,5)
    palm_touch = _soft_touch(ctx.palm_cup_force, 0.5)       # (N,)
    thumb_touch = digit_touch[:, 0]
    finger_count = digit_touch[:, 1:].sum(-1)               # 0..4
    early_contact_pen = -2.0 * s1 * digit_touch.amax(-1)    # no thumb/finger contact before the approach is done

    # ---------------- link geometry around the cup ----------------
    lv = ctx.link_pos - ctx.cup_pos[:, None, None, :]       # (N,5,3,3)
    ax = axis[:, None, None, :]
    lh = (lv * ax).sum(-1)                                  # (N,5,3)
    lrad_vec = lv - lh.unsqueeze(-1) * ax
    lrad = torch.norm(lrad_vec, dim=-1)
    surf_gap = torch.relu(lrad - r[:, None, None] - 0.01)   # 1 cm allowance for link thickness
    in_band = torch.clamp(1.0 - torch.relu(lh.abs() - hh[:, None, None]) / 0.02, 0.0, 1.0)
    near_surf = torch.exp(-surf_gap / 0.015) * in_band      # (N,5,3)

    # fingertip wrap angle around the axis: 0 = palm side, pi/2 = far side along the fingers
    tip_vec = lrad_vec[:, 1:, 2, :]                         # (N,4,3)
    along_f = (tip_vec * fd[:, None, :]).sum(-1)
    toward_palm = -(tip_vec * n[:, None, :]).sum(-1)
    theta = torch.atan2(along_f, toward_palm)
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)  # saturates at ~108 deg (wrapped past the far side)
    finger_wrap = (wrap * near_surf[:, 1:, 2]).mean(-1)
    thumb_near = near_surf[:, 0, :].amax(-1)

    # palm closing the last centimetre while staying aligned along the fingers and in the band
    align2 = torch.exp(-((gap_f - 0.0075) ** 2 + torch.relu(h_palm.abs() - hh) ** 2) / (2.0 * 0.02 ** 2))
    palm_close = torch.exp(-torch.relu(gap_n) / 0.015) * align2

    # squeeze: command leads the measured angle on touching digits, so the PD fingers press
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, j].mean(-1) for j in _DIGIT_JOINTS], dim=-1)  # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)

    # ---------------- stage 2: envelope grasp ----------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    resting = torch.clamp(1.0 - (dz - 0.02) / 0.02, 0.0, 1.0)  # no stage-2 progress once the cup is lifted >4 cm
    fac2 = s2 * (0.3 + 0.7 * ori) * resting
    grasp_stage_base = 3.5 * s2                                  # > stage 1 max (3.0)
    grasp_palm = fac2 * (0.5 * palm_close + 1.0 * palm_touch)
    grasp_contacts = fac2 * (1.0 * thumb_touch + 1.5 * finger_count / 4.0)
    grasp_wrap = fac2 * (1.0 * finger_wrap + 0.5 * thumb_near)
    grip_squeeze = 0.5 * squeeze * (fac2 + s3)                   # stage 2 max total 3.5 + 6.0 = 9.5

    # ---------------- stage 3: lift and hold ----------------
    firm = _soft_touch(ctx.link_cup_force, 0.25).amax(-1)       # (N,5)
    grip_gate = firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)  # thumb + >=2 fingers pressing
    lift_stage_base = 10.0 * s3                                  # > stage 2 max (9.5)
    hold_contacts = s3 * (1.0 * palm_touch + 1.0 * thumb_touch + 1.5 * finger_count / 4.0)  # releasing loses this
    rise_needed = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)
    lift_frac = torch.clamp(dz / rise_needed, 0.0, 1.0)
    lift_height = 3.0 * s3 * grip_gate * lift_frac
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    lift_goal = s3 * grip_gate * (2.0 * (1.0 - torch.tanh(gd / 0.2)) + 2.0 * torch.exp(-gd / 0.03) + 1.0 * in_tol)
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 1.0 * s3 * grip_gate * torch.exp(-gd / 0.03) * calm
    success_bonus = 20.0 * s3 * ctx.success.float()

    # ---------------- constraints and regularization (all stages) ----------------
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.01) / 0.04, 0.0, 1.0)  # do not shove the cup
    cup_tilt_pen = -2.0 * torch.clamp(ctx.cup_tilt / (math.pi / 3.0), 0.0, 1.0) ** 2   # do not knock it over
    table_pen = -2.0 * torch.clamp((ctx.table_z + 0.002 - ctx.hand_z_min) / 0.02, 0.0, 1.0)  # do not press into the table
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.05 * (ctx.arm_qd ** 2).sum(-1)

    raw = {
        "approach_reach": approach_reach,
        "approach_keep": approach_keep,
        "early_contact_pen": early_contact_pen,
        "grasp_stage_base": grasp_stage_base,
        "grasp_palm": grasp_palm,
        "grasp_contacts": grasp_contacts,
        "grasp_wrap": grasp_wrap,
        "grip_squeeze": grip_squeeze,
        "lift_stage_base": lift_stage_base,
        "hold_contacts": hold_contacts,
        "lift_height": lift_height,
        "lift_goal": lift_goal,
        "hold_still": hold_still,
        "success_bonus": success_bonus,
        "cup_disturb_pen": cup_disturb_pen,
        "cup_tilt_pen": cup_tilt_pen,
        "table_pen": table_pen,
        "action_rate_pen": action_rate_pen,
        "arm_vel_pen": arm_vel_pen,
    }
    components = {name: scale * value for name, value in raw.items()}
    reward = torch.stack(list(components.values()), dim=0).sum(0)
    return reward, components
```
