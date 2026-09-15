## 1. What the task means, stage by stage

The right arm holds a five-finger hand whose thumb is locked in opposition and sticks straight out of the palm (along `palm_normal`) at the wrist end. The cup has to end up sitting between that thumb and the four fingers, pressed against the palm. Only then can the hand close into a power grasp and lift.

- **Stage 1: approach** (`approach_done` False). Start from the raised pose and keep the start orientation the whole way: `palm_normal` along +y and `palm_finger_dir` along +x. The fingers stay in the default open pose. Bring the palm to the cup's -y side: the palm plane within about 2 cm of the cup surface, the cup axis about `cup_radius + 0.5..2 cm` ahead of `palm_pos` along the fingers, and the palm inside the graspable band. No thumb or finger may touch the cup. The environment then latches `approach_done`. Near-start episodes skip this stage.
- **Stage 2: envelope grasp** (`approach_done` True, `envelope_done` False). Keep the orientation and first close the last few centimetres between the palm and the cup's side until the palm touches. Then flex the thumb (near side) and the index, middle, ring and pinky fingers (far side) around the cup body. The environment latches `envelope_done` when the palm, the thumb and at least 3 fingers touch for 5 consecutive steps. The cup must stay on the table, upright and roughly where it was.
- **Stage 3: lift** (`envelope_done` True). Keep the grasp (palm, thumb and fingers pressing), raise the cup straight up by 0.21 to 0.28 m to `goal_pos`, and hold it there upright and still so successes keep being counted.

Throughout: do not tip or drop the cup, do not push it away, and do not push the hand into the table.

## 2. What went wrong in the last round

1. **The stage-2 reward was a flag reward.** `grasp_stage_base = 3.5 * s2` was paid on every step after `approach_done`, whatever the hand did, and it was about 98 % of the total. Near-start episodes begin with the flag already set. Staying, grasping and leaving therefore all earned the same 0.35 per step. Leaving also avoided the tilt and disturbance penalties, so the policy pulled back (palm-to-band distance 0.19 → 0.41 m, digit touches 0.60 → 0.00).
2. **Grasping paid almost nothing.** All the grasp terms together were worth under 0.1 per step, and the cup-disturbance penalty was already active from 1 cm of cup movement. The small nudges that closing a hand around a cup causes cancelled the contact gains, so contact was unlearned.
3. **Nothing pulled the palm the last few centimetres.** `palm_close` was worth only 0.05 per step. Near-start episodes begin with the palm about 4.5 cm off the cup, so the palm never touched (`grasp_palm` was 0 in the playback).
4. **The approach reward was tiny.** It was worth at most 0.3 per step, the kernel was shallow at 0.38 m, and a flat 30 % of it survived any orientation or pose. From the raised start, drifting with a turned hand paid almost the same as approaching. The retreat learned in near-start episodes also carried over.
5. **Orientation only mattered in stage 1.** The stage-2 flag base ignored orientation, so turning the hand cost nothing once the flag was set.

## 3. Fixes

- **No reward for flags alone.** Every stage-2 and stage-3 term is multiplied by how well the stage's work is being done:
  - `grasp_stay` is a kernel on the distance of the palm from the grasp position beside the cup: palm-plane gap, along-finger offset and band height, all measured relative to the cup. It is about 0.6 in the near-start pose, 1 at the cup's side, and about 0 by 10 to 15 cm away. Pulling back now loses nearly everything.
  - There is no constant stage base. At the approach-completion geometry, stage 2 still pays more than stage 1, because the palm is already inside the stay kernel. The incentive to finish stage 1 is kept without a free flag reward.
- **Palm first, then fingers.** `grasp_palm_close` (sharp kernel) and `grasp_palm_touch` pull the palm onto the cup. Thumb, finger and wrap rewards are scaled by `0.25 + 0.75 * palm_ready`, so closing pays most once the palm is at the cup.
- **Grasp terms are clearly bigger than disturbance.** Contact and wrap terms total about 13 before scaling, and `envelope_quality` rewards palm, thumb and at least 3 fingers together, which is the condition `envelope_done` checks.
  - Cup displacement costs nothing below 4 cm and only ramps to -3 at 12 cm.
  - Stage-2 rewards fade out between 4 and 10 cm of cup displacement (`cup_home`), so pushing the cup away still loses more than grasping gains.
  - Tilt costs nothing below about 10°.
- **Stronger approach.**
  - Weights are raised to a stage-1 maximum of 4.5, with a steeper long-range kernel `1 - tanh(2.5 d)` and a fine kernel centred in the `approach_done` window.
  - Losing the orientation or the open pose scales the approach reward down to 20 %.
  - Pushing the cup away removes the approach reward.
- **Orientation in stages 2 and 3.** All grasp terms are multiplied by `0.3 + 0.7 * ori`.
- **Grasp terms carry into stage 3 without the "cup resting" gate.** Stage 3 therefore strictly adds to stage 2, through `lift_grip_hold`, lift height, goal distance, stillness and the success bonus, all gated on a firm thumb plus at least 2 pressing fingers. In stage 2 the grasp terms still vanish once the cup is lifted more than 4 cm, so lifting before the envelope grasp earns nothing.
- **Strict order.**
  - Stage-1 terms use the `~approach_done` mask, stage-2 terms `approach_done & ~envelope_done`, stage-3 terms `envelope_done`.
  - Thumb or finger contact before `approach_done` is penalised.
  - Lifting before `envelope_done` pays nothing.

Expected metric changes:
- In near-start episodes, `grasp_stay` and `grasp_palm_close` should rise first, followed by `contact/palm_touching`.
- `contact/fingers_touching` should then climb towards 4 and `envelope_done` should start being set.
- In far-start episodes, `approach_reach` should rise well above 0.03 and the start-orientation share should stay near 1.
- `cup_disturb_pen` should stay near 0 while contacts rise.

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
    scale = 0.1  # global scale so a typical per-step reward is O(0.1-3)

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
    gap_n = (v_perp * n).sum(-1) - r    # palm plane to cup surface; env window [-0.01, 0.02]
    gap_f = (v_perp * fd).sum(-1) - r   # cup axis ahead along fingers minus radius; env window [-0.005, 0.02]
    h_palm = -v_ax                      # palm height along the cup axis; env needs |h| <= hh

    # start orientation (normal +y, fingers +x): 1.0 aligned, ~0.47 at 20 deg on both axes, ~0.02 at 45 deg
    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))
    ang_f = torch.acos(torch.clamp(fd[:, 0], -1.0, 1.0))
    ori = torch.exp(-(ang_n ** 2 + ang_f ** 2) / (2.0 * 0.4 ** 2))

    # default (open) hand pose kept; env needs every movable joint within 0.3
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev.amax(-1) / 0.2) ** 2)

    # ---------------- cup state relative to its spawn ----------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    cup_home = torch.clamp(1.0 - (disp_xy - 0.04) / 0.06, 0.0, 1.0)  # 1 up to 4 cm of shove, 0 beyond 10 cm
    resting = torch.clamp(1.0 - (dz - 0.02) / 0.02, 0.0, 1.0)       # 0 once lifted >4 cm (no lifting before envelope)

    # ---------------- stage 1: approach ----------------
    # coarse world-frame target on the cup's -y side, behind the axis along +x, upper part of the band
    offset = torch.stack([-(r + 0.0075), -(r + 0.008), 0.25 * hh], dim=-1)
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(2.5 * d_world)  # ~0.26 at the raised start (0.38 m), 0.75 at 0.1 m
    # fine palm-frame precision centred in the approach_done window
    fine_err2 = (gap_n - 0.005) ** 2 + (gap_f - 0.0075) ** 2 + torch.relu(h_palm.abs() - 0.5 * hh) ** 2
    fine = torch.exp(-fine_err2 / (2.0 * 0.015 ** 2))
    keep = ori * pose
    # turning the hand / closing fingers cuts approach progress to 20 %; shoving the cup away removes it
    approach_reach = s1 * cup_home * (2.0 * coarse + 2.0 * fine) * (0.2 + 0.8 * keep)  # max 4.0
    approach_keep = s1 * 0.5 * keep                                                      # max 0.5 -> stage 1 max 4.5

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

    # fingertip wrap angle around the axis: 0 = palm side, pi/2 = side along the fingers
    tip_vec = lrad_vec[:, 1:, 2, :]                         # (N,4,3)
    along_f = (tip_vec * fd[:, None, :]).sum(-1)
    toward_palm = -(tip_vec * n[:, None, :]).sum(-1)
    theta = torch.atan2(along_f, toward_palm)
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)  # saturates at ~108 deg (curled past the finger-side)
    finger_wrap = (wrap * near_surf[:, 1:, 2]).mean(-1)
    thumb_near = near_surf[:, 0, :].amax(-1)

    # ---------------- grasp position: palm against the cup's side (cup-relative, so it also holds while lifting) ----------------
    gn_err = torch.relu(gap_n - 0.003) + torch.relu(-0.012 - gap_n)      # palm plane on the cup surface
    gf_err = torch.relu((gap_f - 0.0075).abs() - 0.008)                  # cup between thumb and fingers
    h_err = torch.relu(h_palm.abs() - 0.7 * hh)                          # palm inside the band
    d_grasp = torch.sqrt(gn_err ** 2 + gf_err ** 2 + h_err ** 2 + 1e-12)
    stay = torch.exp(-(d_grasp / 0.06) ** 2)   # ~0.6 in the near-start pose (4.5 cm), 1 at the cup, ~0 beyond 12 cm
    close = torch.exp(-d_grasp / 0.015)        # sharp: pulls the palm the last few centimetres onto the cup
    palm_ready = torch.maximum(close, palm_touch)
    close_gate = 0.25 + 0.75 * palm_ready      # finger/thumb closing pays most once the palm is at the cup

    # squeeze: command leads the measured angle on touching digits, so the PD fingers press
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, j].mean(-1) for j in _DIGIT_JOINTS], dim=-1)  # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)

    # ---------------- stages 2+3: grasp terms (no flag-only reward; everything needs the hand at the cup) ----------------
    ori_k = 0.3 + 0.7 * ori
    g2 = s2 * resting * cup_home * ori_k       # stage 2: cup still on the table, near its spawn, hand not turned
    g3 = s3 * ori_k                            # stage 3: same grasp terms keep paying while lifting
    g23 = g2 + g3
    fingers_enough = torch.clamp(finger_count / 3.0, 0.0, 1.0)  # envelope_done needs thumb + >=3 fingers
    grasp_stay = 4.0 * g23 * stay                                # leaving the cup's side loses this
    grasp_palm_close = 2.0 * g23 * close
    grasp_palm_touch = 2.5 * g23 * palm_touch
    grasp_thumb = 2.0 * g23 * stay * close_gate * thumb_touch
    grasp_fingers = 3.0 * g23 * stay * close_gate * (0.75 * fingers_enough + 0.25 * finger_count / 4.0)
    grasp_wrap = g23 * stay * close_gate * (1.0 * finger_wrap + 0.5 * thumb_near)
    grip_squeeze = 1.0 * g23 * squeeze
    envelope_quality = 3.0 * g23 * palm_touch * thumb_touch * fingers_enough  # the envelope_done condition itself
    # grasp terms max ~18 (x0.1); at approach completion stay~1 so stage 2 (>=4) exceeds stage 1 (<=4.5 only at its optimum)

    # ---------------- stage 3: lift and hold (all gated on a firm grip) ----------------
    firm = _soft_touch(ctx.link_cup_force, 0.25).amax(-1)       # (N,5)
    grip_gate = firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)  # thumb + >=2 fingers pressing
    lift_grip_hold = 3.0 * s3 * grip_gate                        # keeping the grasp after envelope_done
    rise_needed = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)
    lift_frac = torch.clamp(dz / rise_needed, 0.0, 1.0)
    lift_height = 5.0 * s3 * grip_gate * lift_frac
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    lift_goal = s3 * grip_gate * (3.0 * (1.0 - torch.tanh(gd / 0.15)) + 3.0 * torch.exp(-gd / 0.03) + 1.0 * in_tol)
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 2.0 * s3 * grip_gate * torch.exp(-gd / 0.03) * calm
    success_bonus = 30.0 * s3 * ctx.success.float()

    # ---------------- constraints and regularization ----------------
    # small nudges while closing (<4 cm) are free; shoving the cup away costs up to 3 (plus the lost grasp terms)
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.04) / 0.08, 0.0, 1.0)
    # tilt below ~10 deg is free; knocking it over costs up to 4 (and >60 deg ends the episode)
    cup_tilt_pen = -4.0 * torch.clamp((ctx.cup_tilt - 0.17) / (math.pi / 3.0 - 0.17), 0.0, 1.0)
    table_pen = -2.0 * torch.clamp((ctx.table_z + 0.002 - ctx.hand_z_min) / 0.02, 0.0, 1.0)  # do not press into the table
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.05 * (ctx.arm_qd ** 2).sum(-1)

    raw = {
        "approach_reach": approach_reach,
        "approach_keep": approach_keep,
        "early_contact_pen": early_contact_pen,
        "grasp_stay": grasp_stay,
        "grasp_palm_close": grasp_palm_close,
        "grasp_palm_touch": grasp_palm_touch,
        "grasp_thumb": grasp_thumb,
        "grasp_fingers": grasp_fingers,
        "grasp_wrap": grasp_wrap,
        "grip_squeeze": grip_squeeze,
        "envelope_quality": envelope_quality,
        "lift_grip_hold": lift_grip_hold,
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
