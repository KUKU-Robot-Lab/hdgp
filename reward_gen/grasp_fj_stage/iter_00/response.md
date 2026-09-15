## 1. What the task means and which stages the robot must go through

The right arm and its five-finger hand start raised beside the table, about 0.38 m from an upright cup. The cup's size changes from one environment to the next. The policy must pick the cup up by an envelope (power) grasp. Three stages must happen strictly in order, and the environment already keeps latched flags for the first two:

**Stage 1: Approach** (`approach_done == False`)
- Move the palm centre to the side of the cup, on the graspable band. The target is the cylinder surface at `radius ≈ cup_radius` with the axial coordinate `|h|` small.
- Turn the palm so that `palm_normal` points at the cup axis.
- Keep every movable finger joint at `hand_default_q_norm`: no pre-shaping and no closing. The fingers and thumb must not touch the cup, and the cup must not be pushed, raised or tilted.
- The environment latches `approach_done` when the palm is within 2 cm of the band surface, faces the axis within about 45°, and every movable joint is within 0.15 of the default pose.
- Shaping: a coarse and a fine palm-to-surface distance term, a palm-facing term, a pose-keeping term, and penalties for finger contact and for disturbing the cup.

**Stage 2: Envelope grasp** (`approach_done & ~envelope_done`)
- Stay at the approach position with the palm against the cup side and still facing the axis.
- Close the fingers so that all five digits wrap the cup body with their inner surfaces. That means finger links on the cup side of the palm plane, on the cup surface and inside the graspable band.
- Touch the cup with the palm, the thumb and the four fingers.
- Put the thumb on the opposite side of the cup from the other four fingers. On a plane through the cup axis and the palm, the thumb and the fingers must lie on different sides.
- Every term except "stay" is multiplied by a soft gate on the palm staying at the cup. Closing the hand somewhere else earns nothing. The cup still must not be pushed, raised or tilted.
- The environment latches `envelope_done`.

**Stage 3: Lift and hold** (`envelope_done`)
- Keep the grasp: palm, thumb and fingers stay in contact.
- Lift the cup straight up (little sideways drift from the spawn→goal line) to `goal_pos`.
- Hold it there, upright and still, so that successes keep being counted.
- Lifting and holding only pay through a grasp gate: the geometric mean of the palm, thumb and other-finger contact levels. A cup that is not really held earns no lift reward.
- A bonus is paid on `ctx.success`.

**Ordering and no skipping**
- Each stage's shaping terms are masked by that stage's flag.
- A stage offset (`stage_base`) is added: 2.5 in stage 2 and 6.5 in stage 3. Each offset is larger than the highest reward the previous stage can give (stage 1 ≤ 2.0, stage 2 ≤ 2.5 + 3.75 = 6.25). Finishing a stage therefore never lowers reward, and the only way to reach a higher level is the environment's ordered flags.
- Raising the cup before `envelope_done` earns no lift reward and is penalised.
- Touching the cup with the fingers during the approach is penalised.
- The latched offsets do not reward leaving the cup. All shaping of the later stage is gated on the palm staying at the cup and on current contact.

**Always-on safety terms**
- Cup tilt penalty: keeps the cup upright and away from the 60° termination.
- Table clearance penalty on `hand_z_min`: keeps the hand from being pushed into the table.
- Small action-rate regularisation.
- The per-step reward stays positive once a stage is complete, so any termination (dropping the cup off the table, tipping it, the hand going into the table) also loses future reward.

**Cup size**
- All geometry is relative: `cup_radius` and `cup_half_height` per environment, `cup_axis`, and `cup_spawn_pos`.
- Contact forces are squashed to bounded levels, `1 - exp(-F / 0.3 N)`, so force spikes cannot dominate.

## 2. Reward function

```python
import torch
import math

# Movable finger joints (range > 0.05 rad), in ctx.hand_q order.
# Locked: 0 thumb_2, 3 index_1, 7 middle_1, 11 ring_1, 15 pinky_1, 16 pinky_2.
MOVABLE_HAND_IDX = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18]
CONTACT_FORCE_SCALE = 0.3  # N; 0.1 N (env contact threshold) -> 0.28, 1 N -> 0.96


def _contact_level(force: torch.Tensor) -> torch.Tensor:
    """Bounded, spike-robust contact level in [0, 1)."""
    return 1.0 - torch.exp(-force.clamp(min=0.0) / CONTACT_FORCE_SCALE)


def _cylinder_coords(p: torch.Tensor, cup_pos: torch.Tensor, cup_axis: torch.Tensor):
    """Axial coordinate h, radial vector and radial distance of point(s) p w.r.t. the cup axis."""
    v = p - cup_pos
    h = (v * cup_axis).sum(-1)
    radial_vec = v - h.unsqueeze(-1) * cup_axis
    radial = radial_vec.norm(dim=-1)
    return h, radial_vec, radial


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.palm_pos.device
    dtype = ctx.palm_pos.dtype
    R = ctx.cup_radius
    hh = ctx.cup_half_height
    idx = torch.tensor(MOVABLE_HAND_IDX, device=device, dtype=torch.long)

    # ---------------- stage masks (strict order kept by the env's latched flags) ----------------
    s1 = (~ctx.approach_done).to(dtype)
    s2 = (ctx.approach_done & ~ctx.envelope_done).to(dtype)
    s3 = ctx.envelope_done.to(dtype)

    # ---------------- palm geometry relative to the cup ----------------
    palm_h, palm_rvec, palm_r = _cylinder_coords(ctx.palm_pos, ctx.cup_pos, ctx.cup_axis)
    palm_r_safe = palm_r.clamp(min=1e-6)
    palm_u = palm_rvec / palm_r_safe.unsqueeze(-1)            # outward radial unit vector at the palm
    face_align = (ctx.palm_normal * (-palm_u)).sum(-1).clamp(0.0, 1.0)  # palm normal toward the axis

    # Distance of the palm centre to its target: the band surface (radius R + 5 mm, 1 cm dead band),
    # near the band centre (|h| <= 0.25 * half height). Well inside the env's 2 cm approach test.
    radial_err = torch.relu((palm_r - R - 0.005).abs() - 0.01)
    height_err = torch.relu(palm_h.abs() - 0.25 * hh)
    d_palm = torch.sqrt(radial_err ** 2 + height_err ** 2 + 1e-12)

    # ---------------- finger link geometry and contacts ----------------
    cup_pos_l = ctx.cup_pos[:, None, None, :]
    axis_l = ctx.cup_axis[:, None, None, :]
    link_h, _, link_r = _cylinder_coords(ctx.link_pos, cup_pos_l, axis_l)          # (N,5,3)
    link_c = _contact_level(ctx.link_cup_force)                                      # (N,5,3)
    digit_c = link_c.amax(-1)                                                        # (N,5)
    palm_c = _contact_level(ctx.palm_cup_force)                                      # (N,)
    digit_w = torch.tensor([0.3, 0.175, 0.175, 0.175, 0.175], device=device, dtype=dtype)  # thumb weighted most

    # ---------------- cup state ----------------
    raise_h = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)

    # =========================== STAGE 1: approach ===========================
    # Coarse term gives gradient from the 0.38 m start, fine term sharpens the last centimetres.
    reach = 0.5 * (1.0 - torch.tanh(d_palm / 0.25)) + 0.5 * torch.exp(-d_palm / 0.03)
    # Hand must keep its default shape: worst movable-joint deviation of both actual and commanded angles.
    dev_q = (ctx.hand_q_norm[:, idx] - ctx.hand_default_q_norm[:, idx]).abs()
    dev_t = (ctx.hand_target_norm[:, idx] - ctx.hand_default_q_norm[:, idx]).abs()
    pose_dev = torch.maximum(dev_q, dev_t).amax(-1)
    pose_keep = 1.0 - torch.tanh(torch.relu(pose_dev - 0.03) / 0.06)  # ~0.04 at the env's 0.15 limit
    finger_touch = torch.tanh(link_c.sum(dim=(1, 2)))                  # fingers/thumb must not touch yet

    approach_reach = 1.0 * reach * s1            # max 1.0
    approach_face = 0.5 * face_align * s1        # max 0.5
    approach_pose_keep = 0.5 * pose_keep * s1    # max 0.5  -> stage 1 total <= 2.0
    approach_finger_contact = -0.5 * finger_touch * s1

    # Stages 1-2: the cup must stay where it stands (no pushing, no early lifting).
    disturb = 0.5 * torch.tanh(torch.relu(disp_xy - 0.005) / 0.05) \
        + 0.5 * torch.tanh(torch.relu(raise_h - 0.005) / 0.02)
    cup_disturb = -1.0 * disturb * (s1 + s2)

    # =========================== STAGE 2: envelope grasp ===========================
    stay_gate = torch.exp(-d_palm / 0.03)        # all grasp shaping only counts at the approach position
    envelope_stay = 0.75 * torch.exp(-d_palm / 0.02) * s2
    envelope_face = 0.25 * face_align * stay_gate * s2

    # Closing progress (a finger stops on the cup, so this only drives closing, not crushing).
    closure = ctx.hand_q_norm[:, idx].mean(-1)
    envelope_closure = 0.25 * closure * stay_gate * s2

    # Wrap: finger links on the cup surface, inside the graspable band, on the cup side of the palm plane.
    near_surface = torch.exp(-torch.relu(link_r - R[:, None, None]) / 0.015)
    in_band = torch.exp(-torch.relu(link_h.abs() - hh[:, None, None]) / 0.01)
    rel_palm = ctx.link_pos - ctx.palm_pos[:, None, None, :]
    cup_side = torch.sigmoid((rel_palm * ctx.palm_normal[:, None, None, :]).sum(-1) / 0.01)
    link_wrap = near_surface * in_band * cup_side                      # (N,5,3)
    digit_wrap = link_wrap.mean(-1)                                    # (N,5)
    envelope_wrap = 0.75 * (digit_wrap * digit_w).sum(-1) * stay_gate * s2

    # Contacts: palm against the cup, thumb and all four fingers touching it.
    envelope_palm_contact = 0.5 * palm_c * stay_gate * s2
    envelope_digit_contact = 1.0 * (digit_c * digit_w).sum(-1) * stay_gate * s2

    # Opposition: thumb and the four fingers on opposite sides of the plane through the cup axis and palm.
    tangent = torch.cross(ctx.cup_axis, palm_u, dim=-1)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    link_s = ((ctx.link_pos - cup_pos_l) * tangent[:, None, None, :]).sum(-1)       # (N,5,3)
    digit_s = link_s.mean(-1) / R.unsqueeze(-1).clamp(min=1e-3)                      # (N,5), radius-normalised
    opposition = torch.tanh(-4.0 * digit_s[:, 0] * digit_s[:, 1:].mean(-1)).clamp(min=0.0)
    opp_gate = torch.minimum(digit_wrap[:, 0], digit_wrap[:, 1:].mean(-1))            # only counts on the cup
    envelope_opposition = 0.25 * opposition * opp_gate * stay_gate * s2
    # stage 2 shaping total <= 3.75

    # =========================== STAGE 3: lift and hold ===========================
    others_c = (digit_c[:, 1:].sum(-1) / 3.0).clamp(max=1.0)          # >= 3 fingers besides the thumb
    grip_quality = (palm_c + digit_c[:, 0] + others_c) / 3.0
    grip_gate = (palm_c * digit_c[:, 0] * others_c).clamp(min=0.0).pow(1.0 / 3.0)  # zero if palm/thumb/fingers let go

    lift_grip_hold = 1.0 * grip_quality * s3

    goal_pos_dist = (ctx.cup_pos - ctx.goal_pos).norm(dim=-1)
    goal_progress = 0.5 * (1.0 - torch.tanh(goal_pos_dist / 0.2)) + 0.5 * torch.exp(-ctx.goal_dist / 0.02)
    lift_goal = 3.0 * grip_gate * goal_progress * s3                  # main task term, max 3.0

    near_goal = torch.exp(-ctx.goal_dist / 0.03)
    stillness = torch.exp(-ctx.cup_lin_vel.norm(dim=-1) / 0.1 - ctx.cup_ang_vel.norm(dim=-1) / 1.0)
    lift_hold_still = 1.0 * grip_gate * near_goal * stillness * s3

    # Lift straight up: sideways deviation from the spawn -> goal line at the current height.
    goal_rise = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=1e-3)
    rise_frac = (raise_h / goal_rise).clamp(0.0, 1.0)
    line_xy = ctx.cup_spawn_pos[:, :2] + rise_frac.unsqueeze(-1) * (ctx.goal_pos[:, :2] - ctx.cup_spawn_pos[:, :2])
    lateral_dev = (ctx.cup_pos[:, :2] - line_xy).norm(dim=-1)
    lift_path_dev = -0.5 * torch.tanh(torch.relu(lateral_dev - 0.01) / 0.03) * s3

    # Success is only reachable by a held cup; paid only after the envelope stage.
    success_bonus = 10.0 * ctx.success.to(dtype) * s3

    # =========================== stage offsets ===========================
    # Stage 2 base (2.5) > stage 1 max (2.0); stage 3 base (6.5) > stage 2 max (2.5 + 3.75 = 6.25):
    # completing a stage never lowers reward, and higher levels only come through the env's ordered flags.
    stage_base = 2.5 * s2 + 6.5 * s3

    # =========================== always-on constraints ===========================
    cup_tilt = -1.0 * torch.tanh(torch.relu(ctx.cup_tilt - 0.05) / 0.3)             # keep the cup upright
    table_clearance = -1.0 * torch.tanh(torch.relu(ctx.table_z + 0.005 - ctx.hand_z_min) / 0.01)  # no hand in table
    arm_rate = ((ctx.actions[:, :7] - ctx.prev_actions[:, :7]) ** 2).sum(-1)
    hand_rate = ((ctx.actions[:, 7:] - ctx.prev_actions[:, 7:]) ** 2).sum(-1)
    action_rate = -(0.01 * arm_rate + 0.002 * hand_rate)                              # small smoothness prior

    components = {
        "stage_base": stage_base,
        "approach_reach": approach_reach,
        "approach_face": approach_face,
        "approach_pose_keep": approach_pose_keep,
        "approach_finger_contact": approach_finger_contact,
        "cup_disturb": cup_disturb,
        "envelope_stay": envelope_stay,
        "envelope_face": envelope_face,
        "envelope_closure": envelope_closure,
        "envelope_wrap": envelope_wrap,
        "envelope_palm_contact": envelope_palm_contact,
        "envelope_digit_contact": envelope_digit_contact,
        "envelope_opposition": envelope_opposition,
        "lift_grip_hold": lift_grip_hold,
        "lift_goal": lift_goal,
        "lift_hold_still": lift_hold_still,
        "lift_path_dev": lift_path_dev,
        "success_bonus": success_bonus,
        "cup_tilt": cup_tilt,
        "table_clearance": table_clearance,
        "action_rate": action_rate,
    }
    reward = torch.zeros_like(ctx.cup_tilt)
    for value in components.values():
        reward = reward + value
    return reward, components
```
