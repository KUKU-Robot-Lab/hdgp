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
