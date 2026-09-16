# Stage-by-stage reading of the task

The robot must do three things in a fixed order, and the environment latches a flag for each:

1. **Approach** (`approach_done`): carry the hand from the raised start to the cup's -y side **without
   turning it** — `palm_normal` on +y, `palm_finger_dir` on +x, every movable finger joint within 0.3 of
   `hand_default_q_norm`, the palm plane within [-0.01, 0.02] m of the cup surface, the cup axis ahead of
   the palm along the fingers, the palm inside the graspable band, and **nothing touching the cup yet**.
   Half the episodes start with this flag already set.
2. **Envelope** (`envelope_done`): from that pose, drive the palm the last centimetres onto the cup's side
   and close the four fingers around the body with the thumb opposing, until palm + thumb + ≥4 digits hold
   for 5 consecutive steps.
3. **Lift**: carry the cup, upright, to the goal (0.21–0.28 m above spawn, ≤5 cm horizontally) and hold it
   there so successes keep counting.

# What round 8 (`fj_stage_i07`) says

Four things are declared solved and I am told to keep them paying **exactly as they do**:

* far-start approach 0.000 → 0.742, settled 0.625, no collapse (`approach_travel/form/lock/keep`, the
  six-condition `form`/`lock` scoring, the k=1.8 long-range slope);
* a true envelope grasp — palm on the cup 0.914 at success, 4.35 digits — i.e. the whole grasp ladder;
* the cup lifted 0.208 m clear;
* the hand off the table for the entire round (`table_pen`, `palm_table_pen`, the `air` multiplier,
  `palm_clearance` instead of the virtual point). Floor condition never fired.

So stage 1 and the ladder are carried over unchanged, and the table block is carried over byte-for-byte.

Three failures, all in stage 3 or at the stage-2 entry, and all visible in the component table:

* **Hovering outpays finishing.** `lift_height` (weight 45, linear in `dz/rise_needed`) grew to 0.787 and
  `lift_clear` to 0.290 while `success_bonus` fell 0.0140 → 0.0014 and `hold_still` never left 1e-4 in a
  thousand epochs. Height income is ~1.1 of a 2.07 total; the terms that pay for *arriving and stopping*
  are together ~0.016. Success peaked at epoch 1000–1200 (0.370 far-start, 1.257 successes/episode) and
  fell five-fold while `reward/total` climbed to its maximum. Raising the cup is a mid-task behaviour that
  can be farmed to saturation without ever finishing.
* **The cup is carried lying over** (30–45° on video; `task/tilt_deg` averages it away at 8.5), and
  `cup_tilt_pen` is -0.0101 against 0.787 of height income — 1 : 78. `carry_upright` (weight 8) pays for
  being airborne *additively*, so hovering with a good grip collects it regardless.
* **The approach shape is dropped the instant the latch fires.** The latch needs one step and stays on;
  afterwards no term pays for holding the open, untouched pre-grasp shape, so the policy satisfies it for
  an instant and dives. Per-step no-touch fell 0.968 → 0.240 and `early_contact_pen` grew fifteen-fold
  (-0.0017 → -0.0250) and was paid willingly.

# What I am changing

**A. Height income saturates; the value moves to the goal.** `lift_height` (45, linear) and `lift_goal`
are deleted. What is left of height is `lift_clear` = 6 · grip · upright · min(dz/0.05, 1) — it buys the
first 5 cm off the table and then goes flat — plus `lift_hold` = 2. Both now decay with
`dwell3 = 1 - 0.5·episode_progress`, so parking in stage 3 loses value while the budget runs (it restarts
on every success, so a policy that keeps succeeding never decays). In their place, three goal terms that
are maximal only at the goal: `goal_close` = 30 · (1 - tanh(gd/0.12)) gives a live gradient from
gd = 0.29 all the way in, `goal_near` = 45 · exp(-(gd/0.08)²) concentrates value inside the tolerance,
`goal_in_tol` = 20, and `hold_still` = 40 · exp(-(gd/0.08)²) · calm pays for stopping there.
Arithmetic: a perfect envelope hovering at height earns ladder (~35–51) + 6 + 2 + ~2 ≈ 45 raw and falling
with dwell; the same grasp carried to the goal and held still earns ladder + 135 ≈ 178 raw, plus 300 per
success. Hovering is roughly a quarter of finishing, where last round it was the best thing available.
Removing the 45-weight height term leaves no gradient gap: the goal sits 0.21–0.28 m above spawn and
≤5 cm sideways, so `goal_close` rises monotonically with the same lift (2.1 raw at dz = 5 cm, 15.5 at
15 cm, 25 at 24 cm) — it is the same motion, paid for the part that matters.

**B. Upright is purely multiplicative.** The additive `carry_upright` is deleted. `upright_c`
(1 below 5°, 0.35 at 15°, 0.002 at 30°) now multiplies *every* stage-3 income — clearance, all three goal
terms, hold_still — so a cup carried on its side earns essentially nothing for being carried, from the
moment it leaves the table. `cup_tilt_pen` is raised from -6 to -8 and starts at 10° instead of 15°, which
against the reduced hover income (~45 raw, not 71) is a ratio of 1 : 6 rather than 1 : 78.

**C. The pre-grasp shape must be held until the palm is actually on the cup.** New `pregrasp_shape` =
5 · s2 · (1 - arrived) · keep · pos, where `arrived` = max(palm_touch, clamp((0.02 - gap_n)/0.02)) is 0
while the palm is still ≥2 cm out and 1 once it is on the cup: the orientation-and-open-hand income of
stage 1 now continues into stage 2 and fades out exactly as the grasp ladder turns on, so there is no step
at which breaking the shape early is free. `early_contact_pen` is raised -3 → -5 in stage 1 and extended
into stage 2 at -4 while the palm has not arrived, which is precisely the "approach for one step, then
dive" behaviour. `finger_curl`/`thumb_curl` — the only rungs not already gated on palm progress — get
`close_gate = 0.25 + 0.75·arrived`, so closing from the approached pose pays four times what closing on
the way in does. The rungs already multiplied by `press` are untouched, to avoid double-gating the grasp
that works.

Everything else — stage 1 in full, the six-condition scoring, the whole grasp ladder and its 51-raw cap,
`air`, `table_pen`, `palm_table_pen`, `cup_disturb_pen`, the regularizers and `scale = 0.05` — is carried
over unchanged. Stage ordering still steps up and never down: stage 1 ≤ 10 raw, stage 2 at the latch
≈ 11 (grasp_pose·pos + pregrasp_shape) rising to ~51, stage 3 hovering ~45, at the goal ~178, +300 per
success.

Expected movement: `reward/lift_height` gone; `reward/goal_close` and `reward/lift_clear` become the
stage-3 income early, `goal_near`/`hold_still`/`success_bonus` late; `task/tilt_deg` down during carries;
`ctrl/prev_ep_successes_mean` should stop decaying after epoch ~1200; per-step no-touch during the
approach should stop falling. If total reward rises again while successes fall, the remaining suspect is
the ladder being paid in full during stage 3.

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
    # Global scale unchanged from the round that produced the working approach + envelope + lift.
    # Per-step totals: good approach ~0.3-0.5, envelope parked on the table ~2.6, the same grasp
    # hovering at height ~2.2, that grasp carried to the goal and held still ~8.9, +15 per success.
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
    # MEASURED ctx.palm_clearance (the virtual palm point paid exactly 0 two rounds ago).
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

    # ================ STAGE 1: approach - KEPT EXACTLY (0.000 -> 0.742 far-start, no collapse) ==
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

    dwell1 = 1.0 - 0.3 * progress          # dawdling in stage 1 is mildly worse than finishing it
    shape_k = 0.15 + 0.85 * keep           # reaching the cup by rotating the wrist forfeits 85 %
    approach_travel = 4.0 * s1 * dwell1 * (0.55 * coarse + 0.45 * fine_app) * shape_k
    approach_form = 3.0 * s1 * dwell1 * form * (0.3 + 0.7 * prox)
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
    # CHANGE C: has the palm actually arrived? 0 while it is still >= 2 cm out, 1 once it is on the
    # cup. The pre-grasp shape is paid, and closing is discounted, exactly while this is 0.
    arrived = torch.maximum(palm_touch, torch.clamp((0.020 - gap_n) / 0.020, 0.0, 1.0))
    not_arrived = 1.0 - arrived
    close_gate = 0.25 + 0.75 * arrived     # closing from the approached pose pays 4x diving in

    # ---------------- stage gates -------------------------------------------------------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    upright = torch.clamp(1.0 - (ctx.cup_tilt - 0.35) / 0.35, 0.0, 1.0)   # 1 below 20 deg, 0 at 40
    # carrying uprightness: 1 below 5 deg, 0.78 at 10, 0.35 at 15, 0.002 at the 30 deg seen on video
    upright_c = torch.exp(-(torch.relu(ctx.cup_tilt - 0.09) / 0.17) ** 2)
    resting = torch.clamp(1.0 - (dz - 0.03) / 0.03, 0.0, 1.0)             # no stage-2 pay once airborne
    ori_k = 0.5 + 0.5 * ori                     # turning the hand halves the grasp, in stage 3 too
    dwell2 = 1.0 - 0.5 * progress               # anti-farming: parking in stage 2 decays
    dwell3 = 1.0 - 0.5 * progress               # CHANGE A: and so does parking in the air
    g2 = s2 * ori_k * upright * resting * air * dwell2
    g3 = s3 * ori_k * upright * air     # the same ladder keeps paying while lifting -> releasing loses it
    g23 = g2 + g3

    # CHANGE C: hold the pre-grasp shape from the latch until the palm is against the cup. Fades
    # out exactly as the ladder turns on, so there is no step at which breaking it early is free.
    pregrasp_shape = 5.0 * s2 * air * dwell2 * not_arrived * keep * pos
    # no digit may touch before the flag, nor before the palm arrives after it (-0.0250 was paid
    # willingly last round while per-step no-touch fell 0.968 -> 0.240)
    early_contact_pen = -5.0 * s1 * any_touch - 4.0 * s2 * not_arrived * any_touch

    # ================ STAGE 2/3: the grasp ladder (unchanged, CAPPED at ~51 raw) ================
    # rung 0 is set to 10.0 = stage 1's ceiling, so the step the flag latches the reward steps UP.
    grasp_pose = 10.0 * g23 * pos
    # rung 1: the last centimetres of palm travel
    palm_reach = 6.0 * g23 * pos * press_geo
    # rung 2: measured palm contact, the condition that blocked the envelope for two rounds
    palm_contact = g23 * (0.3 + 0.7 * pos) * (5.0 * palm_touch + 2.0 * palm_firm)
    # rung 3: curling the digits round the cup - now also gated on the palm having arrived
    finger_curl = 5.0 * g23 * pos * close_gate * (digit_flex[:, 1:] * engage[:, 1:]).mean(-1)
    thumb_curl = 2.5 * g23 * pos * close_gate * digit_flex[:, 0] * engage[:, 0]
    light_contact = 1.0 * g23 * torch.clamp(digit_touch.sum(-1) / 4.0, 0.0, 1.0)   # keep-alive
    # rung 4: contact quality, all multiplied by palm progress -> a fingertip grasp gets a fraction
    grasp_wrap = g23 * press * (3.5 * wrap_q + 1.5 * wrap_deep)   # wrap_deep needs all four fingers
    grasp_thumb = 3.0 * g23 * press * thumb_q
    grasp_fingers = 3.0 * g23 * press * fingers_q
    grip_squeeze = 1.5 * g23 * press * squeeze
    pinky_join = 1.2 * g23 * press * digit_touch[:, 4] * digit_flex[:, 4]   # averaged away elsewhere
    # rung 5: the envelope_done condition itself (palm AND thumb AND >= 3 fingers)
    envelope_quality = 6.0 * g23 * palm_touch * thumb_touch * fingers_q
    # the ladder cannot grow past ~51 raw: grasp progress can never substitute for lift progress

    # ================ STAGE 3: get clear, then CARRY TO THE GOAL AND STOP ======================
    firm = _sat(ctx.link_cup_force, 0.3).amax(-1)                 # (N,5)
    grip = (0.3 + 0.7 * palm_touch) * firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)
    # CHANGE A: height income is now small and saturates at 5 cm, and both hover terms decay with
    # the step budget (which restarts on every success, so carrying-and-holding never decays).
    # CHANGE B: upright_c MULTIPLIES every payment below - a cup carried on its side earns ~0.
    lift_hold = 2.0 * s3 * grip * dwell3
    lift_clear = 6.0 * s3 * grip * upright_c * dwell3 * torch.clamp(dz / 0.05, 0.0, 1.0)
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    # the carrying gradient is now distance-to-goal, not height: monotone from the spawn distance
    # (~0.29 m -> 0.5 raw) through 15 cm (15.5) to the goal (30), and it peaks ONLY at the goal
    goal_close = 30.0 * s3 * grip * upright_c * (1.0 - torch.tanh(gd / 0.12))
    goal_near = 45.0 * s3 * grip * upright_c * torch.exp(-(gd / 0.08) ** 2)   # 6 at the start tol, 45 at 0
    goal_in_tol = 20.0 * s3 * grip * upright_c * in_tol
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) \
        * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 40.0 * s3 * grip * upright_c * torch.exp(-(gd / 0.08) ** 2) * calm   # arrive AND stop
    success_bonus = 300.0 * s3 * ctx.success.float()
    # ordering: stage 1 <= 10 < stage-2 entry ~11 < full envelope ~51 < hovering with it ~45 (and
    # decaying) << the same grasp carried upright to the goal and held ~178, +300 per success.

    # ---------------- constraints and regularization -------------------------------------------
    # nudges while closing are free up to 4 cm; only shoving the cup away costs more than grasping
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.04) / 0.10, 0.0, 1.0)
    # CHANGE B: tilt free below ~10 deg (was 15), -8 at the 60 deg termination (was -6). Against the
    # reduced hover income this is ~1:6, not the 1:78 that made carrying it sideways free.
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
        "goal_close": goal_close,
        "goal_near": goal_near,
        "goal_in_tol": goal_in_tol,
        "hold_still": hold_still,
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
