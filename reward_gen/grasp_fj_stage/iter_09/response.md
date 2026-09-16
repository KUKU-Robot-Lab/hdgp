# Stage analysis

The task is three latched stages in a fixed order.

1. **Approach** — from the raised pose (or already satisfied in near-start episodes): carry the palm to the cup's -y side in the *start orientation* (`palm_normal` ≈ +y, `palm_finger_dir` ≈ +x) with the hand in its open default pose, palm plane on the cup's surface, cup axis a radius ahead along the fingers, palm inside the graspable band, and **nothing touching the cup yet**. The env latches `approach_done` on the single step all six conditions hold.
2. **Envelope** — from that approached pose, drive the palm the last centimetres onto the cup's side and close the thumb and four fingers around the body. The env latches `envelope_done` on palm + thumb + ≥4 digits for 5 steps.
3. **Lift, carry, settle** — keep the grasp, raise the cup, carry it *upright* to the goal, slow down and stay there so successes keep counting until `max_successes`.

# What I keep from round 9 (`fj_stage_i08`) and why

Round 9 fixed the three things the previous rounds could not: far-start approach 0.882 with no collapse over 1200 epochs, mean tilt 5.45° (uprightness *multiplying* carry income), palm-at-success 0.767 with 4.70 digits, and the value having moved to the end of the task (`goal_near` 0.511, `goal_close` 0.328, `hold_still` 0.068 against 0.0001 the round before). The table requirement is met (`palm_table_pen` 0, `table_pen` −0.0001, `done/hand_floor` 0).

So the following are carried over **byte-for-byte**: the whole stage-1 geometry (six scored latch conditions, `form`/`lock`, `prox`/`fine_app`, `coarse` with k = 1.8, `shape_k`), the table terms and the multiplicative `air`, the grasp-pose window aimed at the middle of the cup body, the link/wrap/thumb-direction geometry, `press_geo`/`press`, `upright_c` multiplying every carry and goal payment, and the goal ladder shape (`goal_close`, `goal_near`, `goal_in_tol`, `hold_still`, success bonus).

# What I change, and the number each change answers

**A. Stage-1 income must not decay (`approach_travel` 0.022 → 0.007, `approach_form` 0.022 → 0.012).**
The only thing in the function that literally makes travelling cheaper is `dwell1 = 1 − 0.3·progress`. It is removed, so the per-step rate of `approach_travel`, `approach_form`, `approach_lock` and `approach_keep` is now constant for the whole episode and the whole run. Urgency is not lost: stage-2 entry now pays ~24 raw against stage 1's ceiling of 10, so the latch is still a large step up. Weights are otherwise untouched — this stage works and is not being redesigned.

**B. The pre-grasp shape must survive the latch (no-touch 0.98 → 0.26, open-pose 0.063, `pregrasp_shape` = 0.0009).**
0.0009 is nothing against the ~0.7 of grasp-ladder income waiting on the other side, so the policy latches for one step and dives. Three coupled fixes:
- `pregrasp_shape` 5.0 → 14.0 with a softer position factor (0.3 + 0.7·pos) so it is real income, not a rounding error, and it is the largest single term available while the palm is still out.
- Every closing rung is now multiplied by `arrived` instead of leaking through `press`: at gap_n = 2 cm, `press` is still 0.21, which paid `grasp_wrap`/`grasp_thumb`/`grasp_fingers`/`grip_squeeze` for fingertip contact made *before* the palm arrives. `close_gate` floor drops 0.25 → 0.10, and `light_contact` is gated too.
- The stage-2 early-contact penalty doubles (−4 → −8); it was being paid willingly at −0.025.
Diving now loses up to 14 raw of shape income, earns ~0 of quality income, and pays −8. Holding the shape and then closing from it pays ~50. Ordering is preserved, so there is no trap in the pre-grasp pose.

**C. Parking with the grasp decays exactly as parking without it does.**
`g3` had no dwell, so a held envelope paid its full ~51 raw for 900 steps whether or not the cup moved. `g2` and `g3` now share `dwell = 1 − 0.5·progress` (identical, so there is no discontinuity at the envelope latch). The budget restarts on every success, so a policy that carries and succeeds never decays; only one that holds and re-approaches does.

**D. Lift income cannot be re-earned (airborne 0.69, `hold_still` 0.068, cup set down and picked up inside one episode).**
`lift_hold` and `lift_clear` are now multiplied by `pre_success = 1 − min(num_successes, 1)`: the income for simply getting the cup off the table is available once per episode, saturates at 5 cm as before, and is gone for good after the first success. A new `setdown_pen` (−6 raw, active only after a success and only while the cup is back down) makes putting it down an explicit loss rather than a fresh start.

**E. The remaining value sits on arriving, slowing and staying.**
`hold_still` 40 → 70 and a new `goal_stay` (25 raw, needs `in_tol` AND `calm` AND at least one banked success) pay only for remaining at rest at the goal — behaviour that a sequence of lifts cannot produce. `success_bonus` 300 → 400. Per-step ordering is now: stage 1 ≤ 10 < stage-2 entry ~24 < envelope parked on the table ~51 (and decaying) < hovering with it ~45 (decaying, and once-only) ≪ carried upright to the goal and held still ~215, plus 400 per success.

I deliberately do **not** touch the tilt gating, the table terms, the wrap/thumb geometry or the success criterion — the round's note is explicit that the late fall in success count is the tolerance curriculum tightening 76 %, not a regression, and must not be answered by making success easier.

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


def _win(x: torch.Tensor, lo, hi, soft: float) -> torch.Tensor:
    # 1 inside [lo, hi], smooth gaussian decay outside. lo/hi may be tensors (per-cup windows).
    d = torch.relu(lo - x) + torch.relu(x - hi)
    return torch.exp(-(d / soft) ** 2)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Global scale unchanged from the round that produced approach + envelope + upright carry +
    # success from the raised start. Per-step totals: approach ~0.3-0.5, envelope parked on the
    # table ~2.6 (decaying), the same grasp carried upright to the goal and held ~10.7, +20/success.
    scale = 0.05

    r = ctx.cup_radius                 # (N,)
    hh = ctx.cup_half_height           # (N,)
    n = ctx.palm_normal
    fd = ctx.palm_finger_dir
    axis = ctx.cup_axis

    # ---------------- stage masks (strict order, flags are latched by the env) ----------------
    approach = ctx.approach_done.bool()
    envelope = ctx.envelope_done.bool() & approach
    s1 = (~approach).float()
    s2 = (approach & ~envelope).float()
    s3 = envelope.float()
    progress = torch.clamp(ctx.episode_progress, 0.0, 1.0)
    # has this episode already delivered the cup once? (CHANGE D)
    banked = torch.clamp(ctx.num_successes.float(), 0.0, 1.0)
    pre_success = 1.0 - banked

    # ---------------- palm pose relative to the cup (the quantities approach_done checks) -----
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

    # open default hand pose kept (the env needs every movable joint within 0.3 of default)
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    dev_max = dev.amax(-1)
    open_pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev_max / 0.2) ** 2)
    keep = ori * open_pose

    # ---------------- measured contact --------------------------------------------------------
    touch = _sat(ctx.link_cup_force, 0.5)         # (N,5,3)
    digit_touch = touch.amax(-1)                  # (N,5)
    thumb_touch = digit_touch[:, 0]
    # sensitive version for the env's "no digit may touch" approach condition (env threshold 0.1 N)
    any_touch = _sat(ctx.link_cup_force, 0.2).amax(-1).amax(-1)      # (N,)
    palm_touch = _sat(ctx.palm_cup_force, 0.5)    # (N,)
    palm_firm = _sat(ctx.palm_cup_force, 2.0)     # rewards actually pressing, not grazing

    # ---------------- table clearance: unchanged - this requirement is met and is not given back
    # `hand_z_min` covers the finger/thumb links only; the palm is charged separately from the
    # MEASURED ctx.palm_clearance (the virtual palm point paid exactly 0 three rounds ago).
    clear = ctx.hand_z_min - ctx.table_z
    near_tab = torch.clamp((0.025 - clear) / 0.025, 0.0, 1.0)   # 0 at 2.5 cm, 1 at contact
    deep_tab = torch.clamp((0.008 - clear) / 0.013, 0.0, 1.0)   # 0 at 8 mm, 1 at the -5 mm cutoff
    table_pen = -(8.0 * near_tab ** 2 + 30.0 * deep_tab ** 2)
    pc = ctx.palm_clearance
    # free above 8 mm only: for the smallest cup a correct mid-body grasp puts the palm's lowest
    # point millimetres above the table, so a higher threshold would punish the working envelope
    palm_low = torch.clamp((0.008 - pc) / 0.008, 0.0, 1.0)
    palm_dig = torch.clamp(-pc / 0.010, 0.0, 1.0)
    palm_table_pen = -(8.0 * palm_low ** 2 + 40.0 * palm_dig ** 2)
    # second, multiplicative pressure: the grasp ladder is halved on the surface
    air = (0.5 + 0.5 * torch.clamp(clear / 0.020, 0.0, 1.0)) \
        * (0.7 + 0.3 * torch.clamp(pc / 0.010, 0.0, 1.0))

    # ================ STAGE 1: approach - geometry KEPT (0.88 far-start, no collapse) ===========
    # CHANGE A: the within-episode decay `dwell1` is REMOVED. Travel income fell 0.022 -> 0.007
    # and form income 0.022 -> 0.012 over the last round; `dwell1` was the one factor that made
    # closing the distance literally cheaper the longer the hand took. The per-step rate is now
    # constant. Urgency comes from the step UP at the latch (stage-2 entry ~24 vs stage-1 max 10).

    # (i) the six approach_done conditions, scored individually: both the mean (dense partial
    # credit) and the product (they must all hold on the SAME step for the env to latch).
    c_gap = _win(gap_n, -0.010, 0.020, 0.015)
    c_along = _win(gap_f, -0.005, 0.020, 0.015)
    c_height = _win(h_palm, -0.9 * hh, 0.9 * hh, 0.020)
    c_ori = _win(ang_n, 0.0, 0.60, 0.30) * _win(ang_f, 0.0, 0.60, 0.30)   # env allows ~45 deg
    c_pose = _win(dev_max, 0.0, 0.22, 0.12)                               # env allows 0.3
    c_free = 1.0 - any_touch                                              # nothing may touch yet
    conds = torch.stack([c_gap, c_along, c_height, c_ori, c_pose, c_free], dim=-1)   # (N,6)
    form = conds.mean(-1)
    lock = conds.prod(-1)

    # (ii) distance to the pre-grasp window, aimed at the MIDDLE of the cup body
    a_n = torch.relu(gap_n - 0.018) + torch.relu(-0.008 - gap_n)
    a_f = torch.relu(gap_f - 0.018) + torch.relu(-0.003 - gap_f)
    a_h = torch.relu(h_palm - 0.025) + torch.relu(-0.012 - h_palm)
    d_app = torch.sqrt(a_n ** 2 + a_f ** 2 + a_h ** 2 + 1e-12)
    prox = torch.exp(-(d_app / 0.12) ** 2)
    fine_app = torch.exp(-(d_app / 0.05) ** 2)

    # (iii) long-range travel; k = 1.8 keeps the 0.38 m raised start on a real slope
    off_z = torch.clamp(0.30 * hh, max=0.020)
    offset = torch.stack([-(r + 0.006), -(r + 0.008), off_z], dim=-1)
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(1.8 * d_world)

    shape_k = 0.15 + 0.85 * keep           # reaching the cup by rotating the wrist forfeits 85 %
    approach_travel = 4.0 * s1 * (0.55 * coarse + 0.45 * fine_app) * shape_k
    approach_form = 3.0 * s1 * form * (0.3 + 0.7 * prox)
    approach_lock = 2.0 * s1 * lock        # paid only when all six latch conditions hold at once
    approach_keep = 1.0 * s1 * keep * (0.2 + 0.8 * prox)
    # stage 1 max = 10.0, reachable only in the configuration that makes the env latch the flag

    # ---------------- grasp pose (wider than the latch window: the palm may press in) ----------
    e_n = torch.relu(gap_n - 0.010) + torch.relu(-0.015 - gap_n)
    e_f = torch.relu((gap_f - 0.006).abs() - 0.010)
    e_h = torch.relu(h_palm - 0.025) + torch.relu(-0.012 - h_palm)   # middle of the cup body
    d_pose = torch.sqrt(e_n ** 2 + e_f ** 2 + e_h ** 2 + 1e-12)
    pos = torch.exp(-(d_pose / 0.05) ** 2)

    # ---------------- link geometry in the cup / palm frame -----------------------------------
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

    # wrap angle around the cup: 0 = palm side, pi/2 = forward side, pi = far side
    theta = torch.atan2(u_f, -u_n)
    wrap = torch.clamp(theta / (0.6 * math.pi), 0.0, 1.0)  # saturates ~108 deg, round the far side
    finger_wrap = (touch[:, 1:, :] * wrap[:, 1:, :]).amax(-1)        # (N,4) contact-weighted
    wrap_q = torch.clamp(finger_wrap.sum(-1) / 3.0, 0.0, 1.0)        # 3 wrapped fingers saturate
    wrap_deep = finger_wrap.mean(-1)                                 # all four, no saturation
    # thumb must close on the -fd side, opposite the fingers: 1 at u_f=-r, 0.5 at the side, 0 at +r
    thumb_dir = torch.clamp(0.5 - u_f[:, 0, :] / (2.0 * r[:, None]), 0.0, 1.0)
    thumb_q = (touch[:, 0, :] * (0.4 + 0.6 * thumb_dir)).amax(-1)
    fingers_q = torch.clamp(digit_touch[:, 1:].sum(-1) / 3.0, 0.0, 1.0)  # 3 of 4; pinky may substitute

    # ---------------- finger closure (flexion past the open default pose) ---------------------
    flex = torch.clamp((ctx.hand_q_norm - ctx.hand_default_q_norm) / 0.5, 0.0, 1.0)
    digit_flex = torch.stack([flex[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    lead = torch.clamp((ctx.hand_target_norm - ctx.hand_q_norm) / 0.15, 0.0, 1.0)
    digit_lead = torch.stack([lead[:, g].mean(-1) for g in _DIGIT_JOINTS], dim=-1)   # (N,5)
    squeeze = (digit_lead * digit_touch).mean(-1)  # PD command leads the angle on a touching digit

    # ---------------- palm progress (smooth over the whole remaining travel) ------------------
    press_geo = torch.exp(-torch.relu(gap_n + 0.010) / 0.025)   # 1 at -1 cm, 0.67 at 0, 0.30 at +2 cm
    press = torch.maximum(palm_touch, 0.7 * press_geo)          # geometry alone caps at 0.7
    # has the palm actually arrived? 0 while it is still >= 2 cm out, 1 once it is on the cup.
    arrived = torch.maximum(palm_touch, torch.clamp((0.020 - gap_n) / 0.020, 0.0, 1.0))
    not_arrived = 1.0 - arrived
    # CHANGE B: the closing gate floor drops 0.25 -> 0.10, and `arrived` now multiplies every
    # contact-quality rung as well. At gap_n = 2 cm `press` is still 0.21, and that residue was
    # paying for fingertip contact made before the palm arrives - the leak that let the policy
    # satisfy the latch for one step and dive (no-touch fell 0.98 -> 0.26 across the round).
    close_gate = 0.10 + 0.90 * arrived

    # ---------------- stage gates -------------------------------------------------------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    upright = torch.clamp(1.0 - (ctx.cup_tilt - 0.35) / 0.35, 0.0, 1.0)   # 1 below 20 deg, 0 at 40
    # carrying uprightness: 1 below 5 deg, 0.78 at 10, 0.35 at 15, 0.002 at 30 deg. KEPT - making
    # this MULTIPLY the carry income is what brought mean tilt from 10.4 to 5.45 deg.
    upright_c = torch.exp(-(torch.relu(ctx.cup_tilt - 0.09) / 0.17) ** 2)
    resting = torch.clamp(1.0 - (dz - 0.03) / 0.03, 0.0, 1.0)             # no stage-2 pay once airborne
    ori_k = 0.5 + 0.5 * ori                     # turning the hand halves the grasp, in stage 3 too
    # CHANGE C: stage 2 AND stage 3 now share the same dwell factor. Last round the stage-3 ladder
    # had none, so a held envelope paid its full ~51 raw for the whole budget whether or not the
    # cup moved - which is what makes lifting and setting down affordable. Identical factors mean
    # no discontinuity at the envelope latch, and the budget restarts on every success, so a
    # policy that carries and succeeds never decays; only one that holds and re-approaches does.
    dwell = 1.0 - 0.5 * progress
    g2 = s2 * ori_k * upright * resting * air * dwell
    g3 = s3 * ori_k * upright * air * dwell
    g23 = g2 + g3

    # CHANGE B: hold the pre-grasp shape from the latch until the palm is against the cup. Raised
    # 5.0 -> 14.0 (it earned 0.0009 last round, which is nothing against the grasp income on the
    # other side) and the position dependence softened, so this is the largest single payment
    # available while the palm is still out. It fades to 0 exactly as the closing rungs turn on.
    pregrasp_shape = 14.0 * s2 * air * dwell * not_arrived * keep * (0.3 + 0.7 * pos)
    # no digit may touch before the flag, nor before the palm arrives after it. The stage-2 half
    # doubles: -0.0250 was being paid willingly last round.
    early_contact_pen = -5.0 * s1 * any_touch - 8.0 * s2 * not_arrived * any_touch

    # ================ STAGE 2/3: the grasp ladder (CAPPED at ~51 raw) ===========================
    # rung 0 is set to 10.0 = stage 1's ceiling, so the step the flag latches the reward steps UP.
    grasp_pose = 10.0 * g23 * pos
    # rung 1: the last centimetres of palm travel
    palm_reach = 6.0 * g23 * pos * press_geo
    # rung 2: measured palm contact, the condition that blocked the envelope for two rounds
    palm_contact = g23 * (0.3 + 0.7 * pos) * (5.0 * palm_touch + 2.0 * palm_firm)
    # rung 3: curling the digits round the cup, gated on the palm having arrived
    finger_curl = 5.0 * g23 * pos * close_gate * (digit_flex[:, 1:] * engage[:, 1:]).mean(-1)
    thumb_curl = 2.5 * g23 * pos * close_gate * digit_flex[:, 0] * engage[:, 0]
    light_contact = 1.0 * g23 * arrived * torch.clamp(digit_touch.sum(-1) / 4.0, 0.0, 1.0)
    # rung 4: contact quality - now needs BOTH palm progress and the palm having arrived, so a
    # fingertip grasp taken on the way in scores ~0 instead of 30 % of the full amount
    grasp_wrap = g23 * arrived * press * (3.5 * wrap_q + 1.5 * wrap_deep)
    grasp_thumb = 3.0 * g23 * arrived * press * thumb_q
    grasp_fingers = 3.0 * g23 * arrived * press * fingers_q
    grip_squeeze = 1.5 * g23 * arrived * press * squeeze
    pinky_join = 1.2 * g23 * arrived * press * digit_touch[:, 4] * digit_flex[:, 4]
    # rung 5: the envelope_done condition itself (palm AND thumb AND >= 3 fingers)
    envelope_quality = 6.0 * g23 * palm_touch * thumb_touch * fingers_q

    # ================ STAGE 3: get clear ONCE, then carry to the goal and STAY =================
    firm = _sat(ctx.link_cup_force, 0.3).amax(-1)                 # (N,5)
    grip = (0.3 + 0.7 * palm_touch) * firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)
    # CHANGE D: height income is small, saturates at 5 cm, AND is available only until the first
    # success of the episode. Last round the cup was airborne 0.69 of the time while the settling
    # term earned 0.068: lifting repeatedly paid, finishing did not. Setting the cup down after a
    # delivery can no longer restart this income.
    lift_hold = 2.0 * s3 * grip * pre_success * dwell
    lift_clear = 6.0 * s3 * grip * upright_c * pre_success * dwell * torch.clamp(dz / 0.05, 0.0, 1.0)
    # and putting it back down after a delivery is an explicit loss, not a fresh start
    setdown_pen = -6.0 * s3 * banked * (1.0 - torch.clamp(dz / 0.05, 0.0, 1.0))

    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) \
        * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    # the carrying gradient is distance-to-goal, not height, and upright_c multiplies all of it
    goal_close = 30.0 * s3 * grip * upright_c * (1.0 - torch.tanh(gd / 0.12))
    goal_near = 45.0 * s3 * grip * upright_c * torch.exp(-(gd / 0.08) ** 2)
    goal_in_tol = 20.0 * s3 * grip * upright_c * in_tol
    # CHANGE E: arriving, slowing and STAYING carry the rest of the value. hold_still 40 -> 70,
    # and goal_stay pays only for remaining at rest at the goal after a delivery - something a
    # sequence of lifts cannot produce, and it does not decay with the budget.
    hold_still = 70.0 * s3 * grip * upright_c * torch.exp(-(gd / 0.08) ** 2) * calm
    goal_stay = 25.0 * s3 * grip * upright_c * banked * in_tol * calm
    success_bonus = 400.0 * s3 * ctx.success.float()
    # ordering: stage 1 <= 10 < stage-2 entry ~24 < full envelope ~51 (decaying) < hovering with it
    # ~45 (decaying, once per episode) << carried upright to the goal and held ~215, +400/success.

    # ---------------- constraints and regularization -------------------------------------------
    # nudges while closing are free up to 4 cm; only shoving the cup away costs more than grasping
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.04) / 0.10, 0.0, 1.0)
    # tilt free below ~10 deg, -8 at the 60 deg termination (on top of the multiplicative upright_c)
    cup_tilt_pen = -8.0 * torch.clamp((ctx.cup_tilt - 0.17) / (math.pi / 3.0 - 0.17), 0.0, 1.0)
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.02 * (ctx.arm_qd ** 2).sum(-1)   # small: must not discourage carrying

    raw = {
        "approach_travel": approach_travel,
        "approach_form": approach_form,
        "approach_lock": approach_lock,
        "approach_keep": approach_keep,
        "pregrasp_shape": pregrasp_shape,
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
        "setdown_pen": setdown_pen,
        "goal_close": goal_close,
        "goal_near": goal_near,
        "goal_in_tol": goal_in_tol,
        "hold_still": hold_still,
        "goal_stay": goal_stay,
        "success_bonus": success_bonus,
        "cup_disturb_pen": cup_disturb_pen,
        "cup_tilt_pen": cup_tilt_pen,
        "table_pen": table_pen,
        "palm_table_pen": palm_table_pen,
        "action_rate_pen": action_rate_pen,
        "arm_vel_pen": arm_vel_pen,
    }
    components = {name: scale * value for name, value in raw.items()}
    reward = torch.stack(list(components.values()), dim=0).sum(0)
    return reward, components
```
