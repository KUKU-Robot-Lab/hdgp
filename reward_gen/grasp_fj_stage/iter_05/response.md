# 1. What the task means and which stages the robot must go through

## The scene, read in the palm frame

Everything about this task is easier to see in the palm frame `(palm_normal n, palm_side, palm_finger_dir fd)`,
because the hand must keep its start orientation the whole time (`n` along +y, `fd` along +x, `palm_side`
therefore along the world +z, i.e. along the upright cup axis).

Put the origin at `palm_pos` and look at the `(fd, n)` plane (the plane the cup's cross-section lives in):

* the cup axis has to end up at about `fd = +r`, `n = +r` (that is exactly what `approach_done` checks:
  `gap_f = v·fd - r ∈ [-0.005, +0.02]` and `gap_n = v·n - r ∈ [-0.01, +0.02]`);
* so the cup's cross-section circle is **tangent to the palm plane** (`n = 0`) and **tangent to the line
  through `palm_pos` across the palm** (`fd = 0`);
* the four fingers leave the front edge of the palm and run along `+fd` inside the palm plane, so when they are
  **extended** they graze the cup's palm-side surface tangentially — they touch it without holding anything;
* the thumb is welded into opposition and, in the default pose, sticks straight out along `+n` from the **wrist
  end** of the palm (`fd ≈ -0.05`), so it passes the cup on its `-fd` side.

The envelope grasp that the task asks for is therefore a three-sided hold in that plane:

* **palm** presses the cup from `-n`,
* **four fingers** run past the cup along `+fd`, flex, and come around the `+fd → +n` quadrant
  (finger flexion rotates a fingertip from `+fd` towards `+n` and then back towards `-fd`),
* **thumb** flexes towards `+fd` and closes on the cup's `-fd` side, opposing the fingers across the cup.

The three stages:

| Stage | Entry condition | What the robot must do | Exit |
|---|---|---|---|
| 1 approach | episode start (far start) | translate the palm ~0.38 m to the cup's -y side **without turning the hand and without closing the fingers**, park it so the cup is between the extended thumb and the open fingers, touching nothing | env sets `approach_done` (latched) |
| 2 envelope | `approach_done` | push the palm the last centimetres onto the cup's side, then flex the fingers around the `+fd/+n` side and the thumb onto the `-fd` side until palm + thumb + ≥3 fingers hold for 5 steps | env sets `envelope_done` (latched) |
| 3 lift | `envelope_done` | keep that grasp, raise the cup 0.21–0.28 m to the goal, hold it upright and still so successes keep counting | `max_successes` |

Half the episodes (from the second on) start already at Stage 2 with `approach_done` set; the reward must make
those episodes pay for *grasp work*, not for the flag.

# 2. What the last reward got wrong (mechanism, not symptoms)

The metrics say the run converged to a **tangential fingertip grasp** and then kept climbing inside it:
`grasp_stay 0.330 + grasp_fingers 0.210 + grasp_palm_close 0.155 + grasp_thumb 0.155 + grasp_wrap 0.077` = 0.93 of
a 0.958 total, with `contact/palm_touching` stuck between 0.001 and 0.013 for 2800 epochs.

Five concrete faults:

1. **`grasp_stay` (4.0 · stay) is a position rent.** It pays the largest single amount for *being* beside the
   cup, with no requirement on the hand. It is what the reward curve was climbing while the stage went nowhere.
   A pose term must be a *gate on other terms*, not a large additive term of its own.
2. **Finger and thumb contact were not conditioned on the palm.** `grasp_thumb`/`grasp_fingers` used
   `close_gate = 0.25 + 0.75·palm_ready`, and `palm_ready = max(close, palm_touch)` — where `close` is a purely
   geometric function that already saturates at `gap_n ≈ 0`. So the fingertip shape collected ~78 % of the
   contact terms **without ever touching with the palm**. The gate never bit.
3. **The geometric palm term saturates before contact happens.** `gn_err = relu(gap_n - 0.003) + relu(-0.012 - gap_n)`
   is flat over `gap_n ∈ [-0.012, 0.003]`, so once `palm_pos`'s plane is level with the nominal cup surface the
   gradient dies. But `palm_pos` is a virtual point and the palm is a *plate*: with the cup axis `r` ahead along
   `fd`, the cup's tangent point is `r` ahead of `palm_pos`, which for the big cups (`r` up to 0.081) is **past
   the front edge of the palm**. Real palm contact needs `gap_n` driven clearly *negative*. The old reward had
   no reason to go there — exactly the "few centimetres of palm travel" that never happened.
4. **Nothing paid for finger flexion.** Every finger term was a function of *contact* or of *tip position*, and
   an extended finger lying in the palm plane already touches the cup tangentially. So "touch" was reachable
   with `hand_q_norm ≈ hand_default_q_norm` and the fingers never had a reason to curl. `grasp_wrap` used a
   geometric `near_surf` on the tips, which for extended fingers sitting 3–10 cm outboard of the surface was
   nearly zero and gave no usable gradient either.
5. **`cup_home` multiplied every grasp term.** Closing a hand on a free-standing cup *always* nudges it a
   centimetre or two; the reward shrank all grasp income for exactly the motion it wanted. Combined with (2)
   the safest policy was "hold still and touch with the tips" — which is what was learned.

Plus the one-line structural rule the feedback states: *no term may keep growing while its stage goes backwards*.
The fix is to make the stage-2 total a **nested ladder**: everything after the palm is multiplied by palm
progress, so the achievable reward without palm contact is hard-capped well below the grasp.

# 3. What the new reward does

**Stage-2 ladder (raw, before the 0.1 scale).** Cap without any palm progress = **8.0**; with the palm pressed
in but not yet touching ≈ 21; with a real envelope ≈ **38.5**. The fingertip shape currently earns ~9.6 raw, so
it drops to ~4 and the only way back up is the palm.

| rung | term | max | gated on |
|---|---|---|---|
| 0 | `grasp_pose` | 3.0 | pose only (small, bounded — replaces `grasp_stay`) |
| 1 | `palm_reach` | 6.0 | `press_geo`: `gap_n` from +8 mm **down to −10 mm** — the last centimetres, no saturation at the nominal surface |
| 2 | `palm_contact` | 8.0 | measured `palm_cup_force` (6.0 touch + 2.0 firmness) — the single largest stage-2 term |
| 3 | `finger_curl`, `thumb_curl` | 4.0 | measured **flexion beyond the default pose**, weighted by that digit being engaged with the cup |
| 4 | `grasp_wrap`, `grasp_thumb`, `grasp_fingers`, `grip_squeeze` | 11.5 | all multiplied by `press` — fingertip contact with the palm away is worth a small fraction |
| 5 | `envelope_quality` | 5.0 | the literal `envelope_done` condition (palm ∧ thumb ∧ ≥3 fingers) |
| — | `light_contact` | 1.0 | a deliberately small keep-alive so the existing contact behaviour is not forgotten while the palm is being learned |

Other changes:

* **Wrap is measured by contact, not proximity.** `finger_wrap = max_links(touch · wrap(θ))` with
  `θ = atan2(u_f, −u_n)` (0 = palm side, π/2 = forward side, π = far side). A tangential fingertip touch sits at
  θ ≈ 0 and scores **zero**; only contact that has come round the cup scores. This is the term that separates
  "fingertip" from "envelope", and it now cannot be collected by standing still.
* **Thumb opposition has a direction.** `thumb_dir = clamp(0.5 − u_f/2r, 0, 1)` is 1 when the thumb touches the
  cup's `-fd` side (opposite the fingers), 0.5 at the side, 0 when it is on the finger side.
* **`cup_home` removed as a multiplier.** Cup displacement is now only an additive penalty with a 5 cm free band
  (`-3.0` beyond that) plus the tilt penalty (free below 15°, `-4.0` at 60°). Small nudges while closing are
  free; only shoving it away or tipping it costs more than the grasp gains.
* **Stage 3 strictly dominates.** Stage 3 keeps every grasp term (so releasing the cup *loses* them — the latched
  flag never pays by itself) and adds `lift_hold 5 + lift_height 8 + lift_goal 11 + hold_still 4`, all multiplied
  by `grip_gate` (thumb + ≥2 fingers pressing, weighted by palm contact), plus a `60 ×` success bonus. Peak
  stage-3 step value ≈ 65 raw vs 38.5 for stage 2 vs 3.0 for stage 1, so the ordering `1 < 2 < 3` holds at every
  boundary and there is no reward drop when a flag is set.
* **Little finger:** not required. `fingers_q = clamp(Σ touch / 3, 0, 1)` needs three of the four fingers (what
  `envelope_done` needs), and the pinky can substitute for any of them; it also earns curl credit. No term
  demands it, matching the fact that its `_1`/`_2` are locked.
* Stage 1 is kept as it is (it is solved), only rescaled so its maximum (3.0) matches stage 2's entry value.

# 4. Improved code

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
    scale = 0.1  # global scale: fingertip touching ~0.4, a real envelope ~3.8, a held success ~6.5

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

    # ---------------- palm pose relative to the cup (same quantities approach_done checks) ----
    v = ctx.cup_pos - ctx.palm_pos
    v_ax = (v * axis).sum(-1)
    v_perp = v - v_ax.unsqueeze(-1) * axis
    gap_n = (v_perp * n).sum(-1) - r    # palm plane -> cup surface; NEGATIVE = palm pressed into the cup
    gap_f = (v_perp * fd).sum(-1) - r   # cup axis ahead of the palm along the fingers, minus r
    h_palm = -v_ax                      # palm offset along the cup axis (env needs |h| <= hh)

    # start orientation kept (palm_normal +y, palm_finger_dir +x); ~0.46 at 20 deg on both axes
    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))
    ang_f = torch.acos(torch.clamp(fd[:, 0], -1.0, 1.0))
    ori = torch.exp(-(ang_n ** 2 + ang_f ** 2) / (2.0 * 0.4 ** 2))

    # distance to the grasp pose; flat inside the approach_done window so the palm may press in
    e_n = torch.relu(gap_n - 0.010) + torch.relu(-0.015 - gap_n)
    e_f = torch.relu((gap_f - 0.006).abs() - 0.010)
    e_h = torch.relu(h_palm.abs() - 0.45 * hh)
    d_pose = torch.sqrt(e_n ** 2 + e_f ** 2 + e_h ** 2 + 1e-12)
    pos = torch.exp(-(d_pose / 0.05) ** 2)    # ~0.6 at the near start (4.5 cm), 1 at the cup
    fine = torch.exp(-(d_pose / 0.018) ** 2)  # sharp version used by the approach

    # ---------------- stage 1: approach (solved last round; kept, only rescaled) ---------------
    dev = (ctx.hand_q_norm[:, _MOVABLE] - ctx.hand_default_q_norm[:, _MOVABLE]).abs()
    open_pose = 0.5 * (1.0 - torch.tanh(dev.mean(-1) / 0.3)) + 0.5 * torch.exp(-(dev.amax(-1) / 0.2) ** 2)
    keep = ori * open_pose
    offset = torch.stack([-(r + 0.0075), -(r + 0.008), 0.25 * hh], dim=-1)  # cup's -y side, behind in +x
    d_world = torch.norm(ctx.palm_pos - (ctx.cup_pos + offset), dim=-1)
    coarse = 1.0 - torch.tanh(2.5 * d_world)  # ~0.26 at the raised start (0.38 m), 0.75 at 0.1 m
    approach_reach = s1 * (1.2 * coarse + 1.3 * fine) * (0.2 + 0.8 * keep)  # max 2.5
    approach_keep = s1 * 0.5 * keep                                        # stage 1 max 3.0

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

    # ---------------- palm progress: the one blocked condition ---------------------------------
    # NOT flat at the nominal surface: palm_pos is a virtual point and for a wide cup the tangent
    # point lies past the palm's front edge, so contact needs gap_n driven ~1 cm negative.
    press_geo = torch.clamp((0.008 - gap_n) / 0.018, 0.0, 1.0)   # 0 at +8 mm, 1 at -10 mm
    press = torch.maximum(palm_touch, 0.75 * press_geo)          # geometry alone caps at 0.75

    # ---------------- stage gates --------------------------------------------------------------
    dz = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    disp_xy = torch.norm(ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2], dim=-1)
    upright = torch.clamp(1.0 - (ctx.cup_tilt - 0.35) / 0.35, 0.0, 1.0)   # 1 below 20 deg, 0 at 40
    resting = torch.clamp(1.0 - (dz - 0.03) / 0.03, 0.0, 1.0)             # no stage-2 pay once airborne
    ori_k = 0.5 + 0.5 * ori                                               # turning the hand halves the grasp
    g2 = s2 * ori_k * upright * resting
    g3 = s3 * ori_k * upright     # the same grasp terms keep paying while lifting -> releasing loses them
    g23 = g2 + g3

    # ---------------- stage 2/3: the grasp ladder ----------------------------------------------
    # rung 0: being in the grasp pose is worth little on its own (this replaces the old position rent)
    grasp_pose = 3.0 * g23 * pos
    # rung 1: the last centimetres of palm travel - the largest gradient available in this stage
    palm_reach = 6.0 * g23 * pos * press_geo
    # rung 2: measured palm contact, the single missing condition of the envelope
    palm_contact = g23 * (0.3 + 0.7 * pos) * (6.0 * palm_touch + 2.0 * palm_firm)
    # rung 3: curling the digits around the cup (extended digits that merely graze it score 0)
    finger_curl = 2.5 * g23 * pos * (digit_flex[:, 1:] * engage[:, 1:]).mean(-1)
    thumb_curl = 1.5 * g23 * pos * digit_flex[:, 0] * engage[:, 0]
    # small keep-alive so the contact behaviour already learned is not thrown away
    light_contact = 1.0 * g23 * torch.clamp(digit_touch.sum(-1) / 4.0, 0.0, 1.0)
    # rung 4: contact quality, all multiplied by palm progress -> a fingertip grasp gets a fraction
    grasp_wrap = 4.0 * g23 * press * wrap_q
    grasp_thumb = 3.0 * g23 * press * thumb_q
    grasp_fingers = 3.0 * g23 * press * fingers_q
    grip_squeeze = 1.5 * g23 * press * squeeze
    # rung 5: the envelope_done condition itself (palm AND thumb AND >= 3 fingers)
    envelope_quality = 5.0 * g23 * palm_touch * thumb_touch * fingers_q
    # stage 2: <= 8.0 without palm progress, ~21 with the palm pressed in, ~38.5 with the envelope

    # ---------------- stage 3: lift and hold (every term needs the grip to be kept) -------------
    firm = _sat(ctx.link_cup_force, 0.3).amax(-1)                 # (N,5)
    grip_gate = (0.4 + 0.6 * palm_touch) * firm[:, 0] * torch.clamp(firm[:, 1:].sum(-1) / 2.0, 0.0, 1.0)
    lift_hold = 5.0 * s3 * grip_gate                              # not a flag bonus: needs the grip
    rise_needed = torch.clamp(ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2], min=0.05)
    lift_frac = torch.clamp(dz / rise_needed, 0.0, 1.0)
    lift_height = 8.0 * s3 * grip_gate * lift_frac
    gd = ctx.goal_dist
    in_tol = (gd <= ctx.success_tol).float()
    lift_goal = s3 * grip_gate * (4.0 * (1.0 - torch.tanh(gd / 0.15))
                                  + 5.0 * torch.exp(-gd / 0.03) + 2.0 * in_tol)
    calm = torch.exp(-torch.norm(ctx.cup_lin_vel, dim=-1) / 0.1) \
        * torch.exp(-torch.norm(ctx.cup_ang_vel, dim=-1) / 1.0)
    hold_still = 4.0 * s3 * grip_gate * torch.exp(-gd / 0.03) * calm
    success_bonus = 60.0 * s3 * ctx.success.float()
    # stage 3 peak ~65 raw > stage 2 peak 38.5 > stage 1 peak 3.0, with no drop at either boundary

    # ---------------- constraints and regularization -------------------------------------------
    # nudges while closing are free up to 5 cm; only shoving the cup away costs more than grasping
    cup_disturb_pen = -3.0 * (s1 + s2) * torch.clamp((disp_xy - 0.05) / 0.10, 0.0, 1.0)
    # tilt free below ~15 deg, -4 at the 60 deg termination
    cup_tilt_pen = -4.0 * torch.clamp((ctx.cup_tilt - 0.26) / (math.pi / 3.0 - 0.26), 0.0, 1.0)
    table_pen = -3.0 * torch.clamp((ctx.table_z + 0.005 - ctx.hand_z_min) / 0.02, 0.0, 1.0)
    action_rate_pen = -0.002 * ((ctx.actions - ctx.prev_actions) ** 2).sum(-1)
    arm_vel_pen = -0.02 * (ctx.arm_qd ** 2).sum(-1)  # halved: must not discourage closing in

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
        "envelope_quality": envelope_quality,
        "lift_hold": lift_hold,
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

# 5. What should move, and what would falsify this

Expected direction (first ~600 epochs):

* `reward/total` **falls first**, from ~0.96 to ~0.4–0.5, because the fingertip shape is no longer paid. A total
  that stays flat at ~0.9 means the ladder is still collectable without the palm and the gating is wrong.
* `reward/palm_reach` rises first (it is the only large gradient available from the current state), then
  `contact/palm_touching` must leave the 0.001–0.013 band. This is the single decisive metric of the round:
  if `palm_reach` saturates near 0.6 (= `press_geo` at 1) while `palm_touching` stays below 0.05, then the palm
  collision surface cannot reach the cup at `gap_n = -0.010` and the geometry, not the reward, is the blocker —
  in that case measure the palm collider offset against `palm_pos` before touching the reward again.
* `finger_curl` / `thumb_curl` rising together with mean `hand_q_norm` of the movable joints: the fingers must
  actually flex, which they never did. `grasp_wrap` is now near zero for the learned shape, so any rise in it is
  genuine wrapping.
* `envelope_done` share out of 0.003 → a measurable fraction; only then do the lift terms carry information.
* Watch `cup_disturb_pen` and `done/tipped`: the free band was widened to 5 cm on purpose, so some pushing is
  expected. `done/tipped` above ~0.02 means the widened band is too generous.
