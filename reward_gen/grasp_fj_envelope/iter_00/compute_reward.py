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
