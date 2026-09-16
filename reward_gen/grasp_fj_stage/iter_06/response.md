# 1. What the task means and which stages the robot must go through

## The task

One right OpenArm (7 DOF) with a five-finger Tesollo DG-5F hand has to **pick an upright cup off a table and hold it at a goal about 25 cm above its spawn**, using a **power (envelope) grasp**, in **three strictly ordered stages**. There is no grasp primitive: 7 arm increments and 19 finger position targets, low-pass filtered, 60 Hz. The cups differ per environment (radius 44–81 mm, half height 42–65 mm), so every geometric quantity has to be written relative to `cup_radius` / `cup_half_height` / `cup_axis`, never as an absolute offset.

**Stage 1 — approach.** From the raised pose (palm ~0.38 m from the cup, outside the table edge) bring the palm to the cup's **−y side**, keeping the start orientation (`palm_normal` ≈ +y, `palm_finger_dir` ≈ +x) and the **default open hand pose**, so the cup ends up *between* the extended thumb (near side, behind the palm along the fingers) and the four fingers. No digit may touch the cup yet. The env latches `approach_done`. Half of the episodes start with this already done.

**Stage 2 — envelope.** Only after `approach_done`: drive the **palm against the cup's side** and curl thumb + fingers around the body. The env latches `envelope_done` when palm + thumb + ≥4 digits total touch for 5 consecutive steps.

**Stage 3 — lift and hold.** Only after `envelope_done`: keep the grasp, raise the cup to the goal and hold it upright and still so `goal_dist ≤ success_tol` for `success_hold_steps`, repeatedly, until `max_successes`.

Hard constraints throughout: don't tip the cup, don't drop it, **don't put the hand into the table** (this also ends the episode below `table_z − 0.005`, and a table-crawling policy is unusable on the real robot).

## What the three previous rounds proved

* Round 1: paying a **flag-only stage base** (`grasp_stage_base = 3.5 * s2`) let the policy collect stage-2 money while walking away. Fixed by making every term depend on what the hand is doing.
* Round 2: the reward paid its largest amounts for *being beside the cup and touching it with fingertips*, so the policy converged on a fingertip grasp; palm contact never left 0.001–0.013 of steps. Fixed by making palm travel and palm contact the dominant gradient of stage 2 and multiplying contact-quality terms by palm progress.
* Round 3 (the last one): **the envelope grasp finally appeared** — palm on the cup 0.55 of steps, ~3 digits curled, `envelope_done` in 0.48 of episodes. That machinery works and must be preserved almost unchanged. But three things went wrong, and the metrics say exactly what they are.

## What is missed or wrong in the last reward

**(a) The table is free.** `table_pen` paid **−0.0002** per step over the whole round while penetration grew to 0.028 m. The term only started at `table_z + 0.005` and had weight 3, i.e. it was worth less than a rounding error next to the ~2.0 per step the grasp ladder paid. Pressing into the table was therefore the *cheapest* way to reach the cup. Two structural holes made it worse: the ramp began essentially at contact (no clearance corridor at all), and it was evaluated on `hand_z_min`, which **excludes the palm** — the recording shows the hand coming in *from underneath*, so the deepest part was probably the palm, which the penalty could not see. Fix: a **corridor** penalty that starts biting 2.5 cm above the surface and grows quadratically to ≈ −22 raw at the termination depth (more than the entire stage-2 ladder), a **separate palm-height penalty** on `palm_pos[:, 2]`, and an `air` factor that halves the whole grasp ladder while the hand is on the surface. The penalty is unmasked — it applies from the first step of the episode, in every stage.

**(b) The grasp is taken too low.** The old pose window allowed the palm anywhere within `0.45 * cup_half_height` of the cup centre, and symmetrically below it, so gripping near the base scored the same as gripping the middle. Gripping low drives the hand into the table and leaves the arm stretched with nowhere to go when it should lift. Fix: tighten the axial window to roughly **−1.5 cm … +2.5 cm about the cup centre**, slightly biased upwards, and put that window inside `d_pose`, which already gates the entire ladder. The stage-1 coarse target keeps its height at the cup centre so the approach *arrives* at grasp height instead of climbing up from the table.

**(c) The lift cannot compete with the rent.** This is the arithmetic that killed the round. Per step at the end: grasp ladder + `lift_hold` ≈ **2.0**, all lift terms together ≈ **0.04**, success bonus **0.0**. Over a 900-step episode holding still on the table returns ~1800 while a success was worth 6. Worse, `lift_hold = 5.0 * s3 * grip_gate` paid **0.31 per step for keeping the grip and going nowhere** — literally more than every lift term combined (0.042). "Total reward rose all the way to the end while lifting went backwards" is the direct consequence. Fix: the grasp ladder is **capped** (≈16.6 raw at a perfect envelope, and it cannot grow past that no matter how long it is held), the stationary stage-3 payment is cut to 2.0 raw, and the *carrying* terms are made the biggest thing in the function: 8 for clearing the table, 20 for the height fraction, up to 25 for goal proximity, 10 for holding still **at the goal**, and 150 for each counted success. Holding a perfect envelope at the spawn is ~1.7 per step scaled; carrying it to the goal is ~8.2 — about **five times more**, per step, so lifting is never outbid by dwelling. The `lift_clear` term specifically buys the *first five centimetres*, which is where the policy has been stuck at 6 mm.

**(d) Anti-farming.** Nothing in stage 2 may keep growing while the stage is not being completed. Beyond the caps, stage-2 payment is multiplied by `1 − 0.5 * episode_progress`: dwelling in stage 2 is worth progressively less inside an episode (the budget restarts on every success, so stage 3 is untouched by this). Ordering inside stage 2 is preserved, so it does not distort the grasp shaping.

**(e) The little finger.** It touched in 0.06 of steps and 0.03 at success. A five-digit wrap is wanted, so it gets its **own small term** (`pinky_join`, 0.4 raw) instead of being pooled into means where the other four saturate the score.

**(f) Kept unchanged, because they worked.** The stage-1 approach shaping (coarse world target + fine palm-frame window × orientation × open-pose factor), `early_contact_pen`, the ladder order (pose → palm travel → palm contact → curl → contact quality → envelope condition), the wrap angle and the thumb-opposition measure, the gentle cup handling (tilt/disturb penalties that never suppressed contact), and cup-relative geometry so the same terms keep paying while the cup is in the air.

Stage budgets (raw, before the 0.1 scale), with no drop at any boundary:
stage 1 ≤ **1.25** → stage 2 entry ≈ **2.5**, perfect envelope ≈ **16.6** → stage 3 at the goal ≈ **82**, plus **150** per success.

# 2. Improved reward function

```python
import torch
import math


# Movable finger joints (range > 0.05 rad), grouped per digit in hand-joint order:
# thumb (3,4), index (2,3,4), middle (2,3,4), ring (2,3,4), pinky (3,4).
_DIGIT_JOINTS = ([1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14], [17, 18])
_MOVABLE = [j for group in _DIGIT_JOINTS for j in group]


def _sat(x: torch.Tensor, full: float) -> torch.Tensor:
    # 0 at/below 0, 1 at/above `full`; the clamp also bounds physics force spikes
    return torch.clamp(x / full, 0.0, 1.0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # global scale: approach ~0.1, a perfect envelope on the table ~1.7,
    # the same envelope carried to the goal ~8.2, each success +15.
    scale = 0.1

    r = ctx.cup_radius
    hh = ctx.cup_half_height
    n = ctx.palm_normal
    fd = ctx.palm_finger_dir
    axis = ctx.cup_axis

    # ---------------- stage masks (strict order, flags are latched by the env) ----------------
    approach = ctx.approach_done.bool()
    envelope = ctx.envelope_done.bool() & approach
    s1 = (~approach).float()
    s2 = (approach & ~envelope).float()
    s3 = envelope.float()

    # ---------------- table clearance: a hard requirement, charged in every stage -------------
    # `hand_z_min` excludes the palm, and the last round came in from underneath, so the palm
    # height is charged separately. The corridor starts 2.5 cm up: settling onto the surface
    # must never be the cheap way in.
    clear = ctx.hand_z_min - ctx.table_z
    near_tab = torch.clamp((0.025 - clear) / 0.025, 0.0, 1.0)   # 0 at 2.5 cm, 1 at contact
    deep_tab = torch.clamp((0.008 - clear) / 0.013, 0.0, 1.0)   # 0 at 8 mm, 1 at the -5 mm cutoff
    # -2.2 at 1 cm, -12 at contact, -22 at the termination depth: more than the whole ladder (16.6)
    table_pen = -(6.0 * near_tab ** 2 + 16.0 * deep_tab ** 2)
    palm_clear = ctx.palm_pos[:, 2] - ctx.table_z
    # free once the palm is 2 cm up (the grasp height is ~cup_half_height up), -8 if it sinks
    palm_low_pen = -8.0 * torch.clamp((0.020 - palm_clear) / 0.025, 0.0, 1.0) ** 2
    # second, multiplicative pressure: the grasp ladder itself is halved on the surface
    air = 0.5 + 0.5 * torch.clamp(clear / 0.02, 0.0, 1.0)

    # ---------------- palm pose relative to the cup (same quantities approach_done checks) ----
    v = ctx.cup_pos - ctx.palm_pos
    v_ax = (v * axis).sum(-1)
    v_perp = v - v_ax.unsqueeze(-1) * axis
    gap_n = (v_perp * n).sum(-1) - r    # palm plane -> cup surface; NEGATIVE = palm pressed in
    gap_f = (v_perp * fd).sum(-1) - r   # cup axis ahead of the palm along the fingers, minus r
    h_palm = -v_ax                      # palm height above the cup centre along the cup axis

    # start orientation kept (palm_normal +y, palm_finger_dir +x); ~0.46 at 20 deg on both axes
    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))
    ang_f = torch.acos(torch.clamp(fd[:, 0], -1.0, 1.0))
    ori = torch.exp(-(ang_n ** 2 + ang_f ** 2) / (2.0 * 0.4 ** 2))

    # distance to the grasp pose. Flat inside the window so the palm may still press in, but the
    # axial window is now the MIDDLE of the cup body (-1.5 cm .. +2.5 cm about the centre):
    # gripping near the base is what drove the hand into the table last round.
    e_n = torch.relu(gap_n - 0.010) + torch.relu(-0.015 - gap_n)
    e_f = torch.relu((gap_f - 0.006).abs() - 0.010)
    e_h = torch.relu(h_palm - 0.025) + torch.relu(-0.015 - h_palm)
    d_pose = torch.sqrt(e_n ** 2 + e_f ** 2 + e_h ** 2 + 1e-12)
    pos = torch.exp(-(d_pose / 0.05) ** 2)    # ~0.6 at the near start (4.5 cm), 1 in the window
    fine = torch.exp(-(d_pose / 0.018) ** 2)  # sharp version used by the approach

    # ---------------- stage 1: approach (solved in the last rounds; kept, only rescaled) ------
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    open_pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev.amax(-1) / 0.2) ** 2)
    keep = ori * open_pose
    # coarse target: the cup's -y side, behind the axis in +x, at the cup's mid-height, so the
    # approach ARRIVES at grasp height instead of climbing up off the table
    off_z = torch.clamp(0.25 * hh, max=0.020)
    offset = torch.stack([-(r + 0.006), -(r + 0.008), off_z], dim=-1)
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(2.5 * d_world)   # ~0.26 at the raised start (0.38 m), 0.75 at 0.1 m
    approach_reach = s1 * (0.5 * coarse + 0.5 * fine) * (0.2 + 0.8 * keep)   # max 1.0
    approach_keep = s1 * 0.25 * keep                                        # stage 1 max 1.25

    # ---------------- measured contact ---------------------------------------------------------
    touch = _sat(ctx.link_cup_force, 0.5)         # (N,5,3)
    digit_touch = touch.amax(-1)                  # (N,5)
    thumb_touch = digit_touch[:, 0]
    palm_touch = _sat(ctx.palm_cup_force, 0.5)    # (N,)
    palm_firm = _sat(ctx.palm_cup_force, 2.0)     # rewards actually pressing, not grazing
    early_contact_pen = -2.0 * s1 * digit_touch.amax(-1)  # no digit may touch before approach_done

    # ---------------- link geometry in the cup / palm frame ------------------------------------
    lv = ctx.link_pos - ctx.cup_pos[:, None, None, :]      # (N,5,3,3)
    ax = axis[:, None, None, :]
    lh = (lv * ax).sum(-1)                                 # (N,5,3) height along the cup axis
    lrad_vec = lv - lh.unsqueeze(-1) * ax
    lrad = torch.norm(lrad_vec, dim=-1)
    u_n = (lrad_vec * n[:, None, None, :]).sum(-1)         # radial component along the palm normal
    u_f = (lrad_vec * fd[:, None, None, :]).sum(-1)        # radial component along the fingers
    in_band = torch.clamp(1.0 - torch.relu(lh.abs() - hh[:, None, None]) / 0.02, 0.0, 1.0)
    near = torch.exp(-torch.relu(lrad - r[:, None, None] - 0.012) / 0.02) * in_band
    engage = torch.maximum(touch, near).amax(-1)           # (N,5) digit is at the cup body

    # wrap angle around the cup: 0 = palm side (a tangential fingertip touch), pi/2 = forward
    # side, pi = far side. An extended finger lying in the palm plane scores ~0 here.
    theta = torch.atan2(u_f, -u_n)
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)  # saturates ~108 deg, round the far side
    finger_wrap = (touch[:, 1:, :] * wrap[:, 1:, :]).amax(-1)        # (N,4) contact-weighted
    wrap_q = torch.clamp(finger_wrap.sum(-1) / 3.0, 0.0, 1.0)        # 3 wrapped fingers saturate
    # thumb must close on the -fd side, opposite the fingers: 1 at u_f=-r, 0.5 at the side, 0 at +r
    thumb_dir = torch.clamp(0.5 - u_f[:, 0, :] / (2.0 * r[:, None]), 0.0, 1.0)
    thumb_q = (touch[:, 0, :] * (0.4 + 0.6 * thumb_dir)).amax(-1)
    fingers_q = torch.clamp(digit_touch[:, 1:].sum(-1) / 3.0, 0.0, 1.0)  # 3 of 4; pinky may substitute

    # ---------------- finger closure (flexion past the open default pose) ----------------------
    flex = torch.clamp((ctx.hand_q_norm - ctx.hand_default_q_norm) / 0.5, 0.0, 1.0)
    digit_flex = torch.stack([flex[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)  # PD command leads the angle on a touching digit

    # ---------------- palm progress ------------------------------------------------------------
    # smooth over the whole remaining travel (palm_pos is a virtual point, and the measured gap
    # can read several centimetres while the palm surface is already touching), so the pull to
    # the cup's side never goes flat at any distance
    press_geo = torch.exp(-torch.relu(gap_n + 0.010) / 0.025)   # 1 at -1 cm, 0.67 at 0, 0.30 at +2 cm
    press = torch.maximum(palm_touch, 0.7 * press_geo)          # geometry alone caps at 0.7

    # ---------------- stage gates --------------------------------------------------------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    upright = torch.clamp(1.0 - (ctx.cup_tilt - 0.35) / 0.35, 0.0, 1.0)   # 1 below 20 deg, 0 at 40
    resting = torch.clamp(1.0 - (dz - 0.03) / 0.03, 0.0, 1.0)             # no stage-2 pay once airborne
    ori_k = 0.5 + 0.5 * ori                                               # turning the hand halves it
    # anti-farming: sitting in stage 2 is worth less the longer it lasts (the budget restarts on
    # every success, so stage 3 never decays). Relative ordering inside stage 2 is unchanged.
    dwell = 1.0 - 0.5 * torch.clamp(ctx.episode_progress, 0.0, 1.0)
    g2 = s2 * ori_k * upright * resting * air * dwell
    g3 = s3 * ori_k * upright * air     # the same ladder keeps paying while lifting -> releasing loses it
    g23 = g2 + g3

    # ---------------- stage 2/3: the grasp ladder (CAPPED at ~16.6 raw; it cannot grow further) -
    # rung 0: standing in the grasp pose is worth little on its own
    grasp_pose = 1.5 * g23 * pos
    # rung 1: the last centimetres of palm travel
    palm_reach = 2.5 * g23 * pos * press_geo
    # rung 2: measured palm contact - the condition that blocked the envelope two rounds ago
    palm_contact = g23 * (0.3 + 0.7 * pos) * (2.0 * palm_touch + 1.0 * palm_firm)
    # rung 3: curling the digits around the cup (extended digits that merely graze it score 0)
    finger_curl = 1.5 * g23 * pos * (digit_flex[:, 1:] * engage[:, 1:]).mean(-1)
    thumb_curl = 0.8 * g23 * pos * digit_flex[:, 0] * engage[:, 0]
    # small keep-alive so the contact behaviour already learned is not thrown away
    light_contact = 0.4 * g23 * torch.clamp(digit_touch.sum(-1) / 4.0, 0.0, 1.0)
    # rung 4: contact quality, all multiplied by palm progress -> a fingertip grasp gets a fraction
    grasp_wrap = 1.5 * g23 * press * wrap_q
    grasp_thumb = 1.2 * g23 * press * thumb_q
    grasp_fingers = 1.2 * g23 * press * fingers_q
    grip_squeeze = 0.6 * g23 * press * squeeze
    # the little finger is averaged away everywhere else, so it gets its own reason to close
    pinky_join = 0.4 * g23 * press * digit_touch[:, 4] * digit_flex[:, 4]
    # rung 5: the envelope_done condition itself (palm AND thumb AND >= 3 fingers)
    envelope_quality = 2.0 * g23 * palm_touch * thumb_touch * fingers_q

    # ---------------- stage 3: lift and hold - the largest payments in the function -------------
    firm = _sat(ctx.link_cup_force, 0.3).amax(-1)                 # (N,5)
    grip = (0.3 + 0.7 * palm_touch) * firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)
    # deliberately small: holding the grasp while going nowhere must not outpay carrying the cup
    lift_hold = 2.0 * s3 * grip
    lift_clear = 8.0 * s3 * grip * torch.clamp(dz / 0.05, 0.0, 1.0)   # buys the first 5 cm
    rise_needed = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)
    lift_frac = torch.clamp(dz / rise_needed, 0.0, 1.0)
    lift_height = 20.0 * s3 * grip * lift_frac                        # the main carrying gradient
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    lift_goal = s3 * grip * (8.0 * (1.0 - torch.tanh(gd / 0.15))
                             + 12.0 * torch.exp(-gd / 0.04) + 5.0 * in_tol)
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) \
        * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 10.0 * s3 * grip * torch.exp(-gd / 0.04) * calm      # only pays AT the goal
    success_bonus = 150.0 * s3 * ctx.success.float()
    # perfect envelope parked on the table ~16.6 raw/step; the same grasp at the goal ~82 raw/step

    # ---------------- constraints and regularization -------------------------------------------
    # nudges while closing are free up to 4 cm; only shoving the cup away costs more than grasping
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.04) / 0.10, 0.0, 1.0)
    # tilt free below ~15 deg, -5 at the 60 deg termination
    cup_tilt_pen = -5.0 * torch.clamp((ctx.cup_tilt - 0.26) / (math.pi / 3.0 - 0.26), 0.0, 1.0)
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.02 * (ctx.arm_qd ** 2).sum(-1)   # small: must not discourage carrying

    raw = {
        "approach_reach": approach_reach,
        "approach_keep": approach_keep,
        "early_contact_pen": early_contact_pen,
        "grasp_pose": grasp_pose,
        "palm_reach": palm_reach,
        "palm_contact": palm_contact,
        "finger_curl": finger_curl,
        "thumb_curl": thumb_curl,
        "light_contact": light_contact,
        "grasp_wrap": grasp_wrap,
        "grasp_thumb": grasp_thumb,
        "grasp_fingers": grasp_fingers,
        "grip_squeeze": grip_squeeze,
        "pinky_join": pinky_join,
        "envelope_quality": envelope_quality,
        "lift_hold": lift_hold,
        "lift_clear": lift_clear,
        "lift_height": lift_height,
        "lift_goal": lift_goal,
        "hold_still": hold_still,
        "success_bonus": success_bonus,
        "cup_disturb_pen": cup_disturb_pen,
        "cup_tilt_pen": cup_tilt_pen,
        "table_pen": table_pen,
        "palm_low_pen": palm_low_pen,
        "action_rate_pen": action_rate_pen,
        "arm_vel_pen": arm_vel_pen,
    }
    components = {name: scale * value for name, value in raw.items()}
    reward = torch.stack(list(components.values()), dim=0).sum(0)
    return reward, components
```

## Expected movement of the metrics

| metric | now | expected |
|---|---|---|
| table penetration (max) | 0.028 m | ≈ 0, `reward/table_pen` clearly negative early then → 0 |
| `reward/palm_low_pen` | — | strongly negative early, → 0 as the hand comes in at cup height |
| palm height on the cup (`h_palm`) | near the base | −1.5 … +2.5 cm about the cup centre |
| `contact/palm_touching` | 0.55 | keep ≥ 0.5 |
| `envelope_done` | 0.48 / 0.70 | keep or rise |
| `task/lifted_frac` | 0.048 mean | rise well past the 0.225 peak |
| `reward/lift_height` + `lift_clear` + `lift_goal` | 0.042 | become the largest block of the total |
| `contact/finger_pinky` | 0.06 | rise |
| `reward/total` | 1.91 rising while lift fell | rises only together with `task/lifted_frac` |
