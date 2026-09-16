## What the task is, and the stages

The robot must move two cups through a fixed chain of physical states, and every later state is
impossible unless the earlier one holds:

0. **Approach** — each palm arrives in a ring around its own cup: outside the wall, at wall height,
   with the **pad of the hand turned toward the cup** (not the back of the hand).
1. **Open** — while near the cup the thumb-to-finger opening must become *wider than the 57 mm cup*.
   Nothing can be grasped by a hand that is already narrower than the object.
2. **Straddle** — the cup body ends up *between* the thumb tip and at least one finger tip
   (the cup axis lies inside the opening, both tips at wall height, genuinely opposed).
3. **Pinch** — both sides come onto the wall and squeeze until the thumb and one opposing finger
   each press above ~1 N: the environment's `grasped` flag.
4. **Lift** — both cups leave the table while the pinch is maintained.
5. **Carry** — the source mouth is brought over the receiver mouth (bodies apart, source mouth
   4–12 cm higher), without the cups touching each other.
6. **Pour** — the source cup is tilted past ~110° over the receiver mouth, beads fall through the air.
7. **Hold the goal** — receiver upright, nothing dropped, little spilled, cups not nested.

## What the last round proved, and what changed here

Round 7's opposition gate worked exactly as intended: no posture that cannot grasp was paid anything
for 680 epochs. That is kept — **no floors are reintroduced anywhere**. But with every rung above the
approach removed, the policy banked the approach term (88 % of its income), turned the **back** of its
fingers against the cup, and closed the hand to 48–50 mm — narrower than the cup — because nothing
asked it to open and the `|face|` term scored a palm turned away exactly like a palm turned toward.

Three additions, one per operator requirement, plus one penalty:

1. **`orient_*` is now signed** (requirement 1). `facing = n̂ · û(palm→cup)` with **no absolute value**.
   `pad = clamp(facing / 0.35, 0, 1)` is a hard gate that is exactly **0** whenever the pad is not
   turned toward the cup, and it multiplies *every* orientation, posture and contact term of that
   hand. The income part saturates at `facing = 0.5` (60°) so a fingertip grasp, where the cup sits
   ahead of the fingers rather than square on the palm, is not asked for a power-grasp alignment.
2. **`open_*` and `spread_*`** (requirement 2). `spread` pays only once the thumb-to-(nearer of
   index/middle) gap exceeds the cup (`60 mm` onset, full at `75 mm`), and is *not* gated on
   opposition — it is the rung below it. `open` supplies the sub-threshold gradient from the
   commanded/measured closure and from the gap itself, squared so a 48 mm hand collects almost
   nothing while the derivative toward opening stays strictly positive. It decays as `(1 − pose)`
   so it cannot pay once the hand should be closing.
3. **Contact on the back of the hand is not progress** (requirement 3). `touch/grip/grasp/hold` are
   gated on `pad · opp_gate · thumb_low`, so force arriving on a dorsal surface, or from a thumb on
   the same side as the fingers, is worth zero — even with the env grasp flag raised. The new
   `dorsal_pen` additionally *charges* for pressing a cup with a palm turned away (1.2·tanh(f)) and
   for standing near the cup facing away (0.15), which is the gradient out of the observed posture.

Everything from stage 3 upward (pinch → lift → carry → tilt → pour → success) is unchanged from
round 7, since no policy has reached it yet and it is not what this round is testing.

```python
import torch
import math


def _near(d: torch.Tensor, k: float) -> torch.Tensor:
    """Bounded closeness in (0, 1]: 1 at d = 0, ~0.37 at d = 1/k."""
    return torch.exp(-k * d)


def _band_err(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """0 inside [lo, hi], linear distance outside."""
    return torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)


def _cyl(points: torch.Tensor, cup_pos: torch.Tensor, cup_up: torch.Tensor):
    """Cup-cylinder coords of (N,K,3) points: axial height (N,K), radial vector (N,K,3), radial distance (N,K)."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)
    rvec = rel - axial[..., None] * up
    radial = torch.norm(rvec, dim=-1)
    return axial, rvec, radial


def _axis_to_segments(rv_a: torch.Tensor, rv_b: torch.Tensor) -> torch.Tensor:
    """Distance (N,K) from the cup AXIS to each segment joining rv_a (N,1,3) to rv_b (N,K,3),
    measured in the plane normal to the axis. ~0 only when the axis lies BETWEEN the two tips."""
    seg = rv_b - rv_a
    u = torch.clamp(-(rv_a * seg).sum(dim=-1) / ((seg * seg).sum(dim=-1) + 1e-8), 0.0, 1.0)
    closest = rv_a + u[..., None] * seg
    return torch.norm(closest, dim=-1)


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, palm_force, closure,
                cup_pos, cup_up, grasped, a_hand):
    """Per-arm approach / pad-direction / opening / opposition / contact signals, all (N,) in [0, 1]."""
    dtype = cup_pos.dtype
    mouth = ctx.cup_mouth_z
    r_in = ctx.cup_radius
    r_wall = r_in + 0.004                    # outer wall ~ inner radius + wall thickness
    d_out = 2.0 * r_wall                     # outer diameter
    w_cup = max(2.0 * r_in, 0.055)           # the cup is 57 mm across; floored so the target can never collapse
    gap_lo = w_cup + 0.003                   # the opening must CLEAR the cup before it can go around it
    gap_hi = w_cup + 0.018                   # comfortably wider than the cup -> full credit
    tip_lo = ctx.cup_bottom_z + 0.03         # tips >= 3 cm above the cup bottom (table clearance)
    tip_hi = mouth - 0.015                   # tips >= 1.5 cm below the rim

    # ---- palm pre-grasp region (this stage is solved; it stays deliberately cheap)
    p_ax, _, p_rad = _cyl(palm_pos[:, None, :], cup_pos, cup_up)
    rad_err = _band_err(p_rad[:, 0], r_wall + 0.01, 0.10)
    ht_err = _band_err(p_ax[:, 0], ctx.cup_bottom_z + 0.03, mouth + 0.01)
    d_pre = torch.sqrt(rad_err ** 2 + ht_err ** 2 + 1e-8)
    reach_lin = torch.clamp(1.0 - d_pre / 0.25, 0.0, 1.0)
    reach_fine = _near(d_pre, 25.0)
    near_pre = _near(d_pre, 10.0)

    # ---- PAD DIRECTION, SIGNED (round 7 used |.| here, so a palm turned away scored like a palm
    #      turned toward, and the policy laid the BACK of its fingers on the cup for free).
    #      palm_axes[:, 0:3] is the palm normal, i.e. the direction the pad faces.
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup = cup_pos - palm_pos
    u3 = to_cup / (torch.norm(to_cup, dim=-1, keepdim=True) + 1e-6)
    facing = (normal * u3).sum(dim=-1)                       # +1 pad at the cup, -1 back of the hand
    face_lin = torch.clamp(facing / 0.5, 0.0, 1.0)           # income: saturates at 60 deg, a fingertip
    #                                                          grasp does not need the cup square on the pad
    pad = torch.clamp(facing / 0.35, 0.0, 1.0)               # HARD gate, no floor: 0 for any pad not
    #                                                          turned toward the cup. Multiplies every
    #                                                          orientation / posture / contact term below.
    back = torch.clamp(-facing, 0.0, 1.0)                    # 1 = the cup is behind the hand

    # ---- tips ahead of the palm (convention-free: the cup is in front of the fingers, not beside them)
    d_palm_xy = torch.norm(cup_pos[:, :2] - palm_pos[:, :2], dim=-1)
    tip_mid_all = tips[:, 0:3, :].mean(dim=1)
    d_tip_xy = torch.norm(tip_mid_all[:, :2] - cup_pos[:, :2], dim=-1)
    ahead = torch.clamp((d_palm_xy - d_tip_xy) / 0.03, 0.0, 1.0)
    orient_raw = near_pre * pad * (0.6 * face_lin + 0.4 * ahead)

    # ---- fingertips in cup-cylinder coordinates
    t_ax, t_rv, t_rad = _cyl(tips, cup_pos, cup_up)
    t_dir = t_rv / (t_rad[..., None] + 1e-6)

    # ---- OPPOSITION. +1 = this finger sits on the far wall from the thumb, -1 = same side as the thumb.
    u_th = t_dir[:, 0, :]
    opp_all = (t_dir[:, 1:, :] * u_th[:, None, :]).sum(dim=-1) * (-1.0)      # (N,F-1) in [-1, 1]
    opp_best = torch.clamp(opp_all.max(dim=-1).values, 0.0, 1.0)             # logged as oppose_*
    # HARD gate, no floor: 90 deg apart -> 0, 120 deg -> 1.  A thumb beside or inside the fingers scores 0.
    opp_gate_all = torch.clamp((opp_all - 0.3) / 0.5, 0.0, 1.0)
    opp_gate = opp_gate_all.max(dim=-1).values
    u_fm = t_dir[:, 1:3, :].mean(dim=1)
    u_fm = u_fm / (torch.norm(u_fm, dim=-1, keepdim=True) + 1e-6)
    opp_mid = -(u_th * u_fm).sum(dim=-1)                                     # -1 = thumb among the fingers
    opp_soft = torch.clamp((opp_mid + 0.1) / 0.6, 0.0, 1.0)

    # ---- tip admissibility: at wall height and outside the bore (not dipped in, not far outside)
    ht_bad = _band_err(t_ax, tip_lo, tip_hi)                                  # (N,F)
    h_ok = _near(ht_bad, 40.0)
    r_ok = (torch.clamp((t_rad - r_in) / 0.005, 0.0, 1.0)
            * torch.clamp((r_wall + 0.05 - t_rad) / 0.02, 0.0, 1.0))
    tip_ok = h_ok * r_ok                                                      # (N,F)
    thumb_ax = t_ax[:, 0]
    thumb_low = torch.clamp((mouth - thumb_ax) / 0.015, 0.0, 1.0)             # thumb below the rim = on the wall

    # ---- distance of each tip to the outer wall (radial gap + height-band error)
    rad_out = torch.clamp(t_rad - r_wall - 0.003, min=0.0)
    surf = torch.sqrt(rad_out ** 2 + ht_bad ** 2 + 1e-8)
    wall_all = 0.35 * _near(surf, 12.0) + 0.65 * _near(surf, 50.0)
    wall_t = wall_all[:, 0]
    wall_f = wall_all[:, 1:].max(dim=-1).values
    wall_pair = 0.5 * (wall_t + wall_f)
    at_cup = _near(surf.min(dim=-1).values, 15.0)                             # a tip is actually on the cup

    # ---- POSE: strict per-finger straddle. The cup axis lies between the thumb tip and THIS finger's
    #      tip, both tips admissible, the pair genuinely opposed, the pad turned toward the cup. No floor.
    seg_d = _axis_to_segments(t_rv[:, 0:1, :], t_rv[:, 1:, :])                # (N,F-1)
    pair_geo = _near(seg_d, 40.0) * opp_gate_all * tip_ok[:, 1:] * tip_ok[:, 0:1]
    pose = pair_geo.max(dim=-1).values * pad

    # ---- OPENING (requirement 2). gap = thumb tip to the NEARER of the index / middle tips, the same
    #      quantity as task/{src,rcv}_tip_gap_mm_near. The policy settled at 48-50 mm, i.e. narrower
    #      than the 57 mm cup, because nothing paid for opening.
    gap_pair = torch.norm(tips[:, 0:1, :] - tips[:, 1:3, :], dim=-1).min(dim=-1).values
    # spread: pays ONLY once the opening clears the cup. Not gated on opposition - it is the rung below it.
    gap_ok = (torch.clamp((gap_pair - gap_lo) / (gap_hi - gap_lo), 0.0, 1.0)
              * torch.clamp((0.155 - gap_pair) / 0.025, 0.0, 1.0))
    # ... but a hand laid ON the cup with the thumb tucked in among the fingers is not an opening,
    # however wide it measures: it is the round-7 posture with the object outside the jaw. `jaw_ok`
    # zeroes the whole pre-grasp ladder there (it is ~1 whenever the tips are not yet on the cup,
    # so it never blocks the approach).
    same_side = torch.clamp((-0.1 - opp_mid) / 0.5, 0.0, 1.0)
    jaw_ok = 1.0 - at_cup * same_side
    orient = orient_raw * jaw_ok
    spread = near_pre * pad * gap_ok * jaw_ok
    # open: the sub-threshold gradient. Squared, so a 48 mm hand collects ~0.05 of the 0.5 available
    # while d(reward)/d(opening) stays strictly positive all the way from a fist to 75 mm.
    open_gap = torch.clamp((gap_pair - 0.025) / max(gap_lo - 0.025, 1e-3), 0.0, 1.0)
    open_cmd = torch.clamp((0.45 - closure) / 0.40, 0.0, 1.0)
    open_hand = near_pre * pad * (0.5 * (open_gap + open_cmd)) ** 2 * (1.0 - pose) * jaw_ok

    # ---- POCKET: is the cup body actually inside the opening? midpoint of the thumb tip and the
    #      index/middle mean must sit ON the cup axis, the opening wider than the cup, both sides at
    #      wall height, thumb below the rim, and exactly 0 when the thumb is on the fingers' side.
    tip_f_mid = tips[:, 1:3, :].mean(dim=1)
    mid = 0.5 * (tips[:, 0, :] + tip_f_mid)
    _, _, d_pocket = _cyl(mid[:, None, :], cup_pos, cup_up)
    d_pocket = d_pocket[:, 0]
    h_pocket = h_ok[:, 0] * h_ok[:, 1:3].max(dim=-1).values
    pocket = ((0.35 * _near(d_pocket, 15.0) + 0.65 * _near(d_pocket, 45.0))
              * gap_ok * h_pocket * opp_soft * thumb_low * pad)

    # ---- close the last centimetres: ONLY inside a pose that can trap the cup
    close = pose * wall_pair

    # ---- squeeze intent from the COMMANDED closure, only inside a valid pose in contact range
    cmd = 0.5 * (torch.clamp(a_hand, -1.0, 1.0) + 1.0)
    cmd_close = (cmd[:, 0] + cmd[:, 1] + cmd[:, 2:6].mean(dim=-1)) / 3.0
    squeeze = pose * cmd_close * wall_pair

    # ---- closing on air / curling while far / hooking the rim
    gap_min = torch.norm(tips[:, 0:1, :] - tips[:, 1:, :], dim=-1).min(dim=-1).values
    air_pinch = torch.clamp((w_cup - 0.005 - gap_min) / 0.025, 0.0, 1.0)
    far = 1.0 - _near(torch.clamp(d_pre - 0.03, min=0.0), 20.0)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far
    over_rim = torch.clamp((thumb_ax - (mouth - 0.005)) / 0.015, 0.0, 1.0)
    over_footprint = (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = over_rim * over_footprint

    # ---- FORCES. Only a finger that OPPOSES the thumb counts as the far side of the pinch, and only
    #      while the pad faces the cup: the force sensors report any contact in any direction, so
    #      without `pad` the back of the fingers reads as legitimate grasp progress (round 7's failure).
    f_t = finger_force[:, 0]
    f_o = (finger_force[:, 1:] * opp_gate_all).max(dim=-1).values
    f_min = torch.minimum(f_t, f_o)
    s_t = 1.0 - torch.exp(-f_t / 0.3)
    s_o = 1.0 - torch.exp(-f_o / 0.3)
    q_grasp = opp_gate * thumb_low * pad                                      # hard: no opposition or no pad, no pay
    touch = (0.6 * s_t * wall_f + 0.4 * s_o * wall_t) * q_grasp
    grip = (0.5 * (1.0 - torch.exp(-f_min / 0.25)) + 0.5 * torch.clamp(f_min / 1.0, 0.0, 1.0)) * q_grasp
    grasped_f = grasped.to(dtype)
    grasp = grasped_f * q_grasp                                               # the env flag alone is not enough
    hold_soft = torch.clamp((f_min - 0.3) / 0.7, 0.0, 1.0)
    hold = torch.maximum(grasped_f, hold_soft) * thumb_low * opp_gate * pad
    q = 0.3 + 0.7 * q_grasp

    # ---- pressing a cup with a palm turned away, and loitering near it back-first
    dorsal = back * (0.15 * near_pre
                     + 1.2 * torch.tanh((finger_force.sum(dim=-1) + palm_force) / 1.0))

    return {
        "reach_lin": reach_lin, "reach_fine": reach_fine, "orient": orient,
        "open": open_hand, "spread": spread, "oppose": opp_best * near_pre * pad * jaw_ok,
        "pocket": pocket, "pose": pose, "close": close, "squeeze": squeeze,
        "air_pinch": air_pinch, "curl_far": curl_far, "rim_hook": rim_hook, "dorsal": dorsal,
        "touch": touch, "grip": grip, "grasp": grasp, "hold": hold, "q": q,
    }


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)
    dtype = zero.dtype
    act = ctx.actions

    s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                    ctx.src_palm_force, ctx.src_hand_closure, ctx.src_cup_pos, ctx.src_cup_up,
                    ctx.src_grasped, act[:, 6:12])
    r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                    ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                    ctx.rcv_grasped, act[:, 18:24])
    hold_s, hold_r = s["hold"], r["hold"]
    q_s, q_r = s["q"], r["q"]

    # ---------------------------------------------------- stage 0: approach (solved; kept cheap)
    approach_src = 0.2 * s["reach_lin"] + 0.1 * s["reach_fine"]      # max 0.3 per hand
    approach_rcv = 0.2 * r["reach_lin"] + 0.1 * r["reach_fine"]
    # weight 0.3: the SIGNED pad direction is now a prerequisite, but it must stay far below the
    # opening terms below, or facing the cup with a closed fist becomes the next plateau.
    orient_src = 0.3 * s["orient"]
    orient_rcv = 0.3 * r["orient"]

    # ---------------------------------------------------- stage 1: OPEN the hand wider than the cup
    open_src = 0.5 * s["open"]            # sub-threshold gradient only; ~0.05 for the 48 mm hand
    open_rcv = 0.5 * r["open"]
    spread_src = 1.5 * s["spread"]        # weight 1.5 > orient 0.3 + approach 0.3: opening beats standing
    spread_rcv = 1.5 * r["spread"]

    # ---------------------------------------------------- stage 2: get the cup BETWEEN thumb and fingers
    oppose_src = 0.3 * s["oppose"]        # logged diagnostic + gentle pull; 0 for a same-side thumb
    oppose_rcv = 0.3 * r["oppose"]
    pocket_src = 1.0 * s["pocket"]        # dense "cup inside the opening" signal
    pocket_rcv = 1.0 * r["pocket"]
    pose_src = 1.5 * s["pose"]            # strict straddle; the gate for everything below
    pose_rcv = 1.5 * r["pose"]
    # min(), not sum(): one hand improving while the other goes backwards must not pay.
    pose_both = 2.0 * torch.minimum(s["pose"], r["pose"])

    # ---------------------------------------------------- stage 3: onto the wall and squeeze
    close_src = 2.0 * s["close"]          # weight 2.0 but NO floor: 0 unless the pose can trap the cup
    close_rcv = 2.0 * r["close"]
    squeeze_src = 0.8 * s["squeeze"]
    squeeze_rcv = 0.8 * r["squeeze"]
    touch_src = 1.0 * s["touch"]          # one side pressing while the other is in place — a step, not a rest
    touch_rcv = 1.0 * r["touch"]
    grip_src = 3.0 * s["grip"]            # weight 3.0: both sides above 1 N is what the env calls a grasp
    grip_rcv = 3.0 * r["grip"]
    grasp_src = 4.0 * s["grasp"]          # weight 4.0 > grip 3.0: the flag itself is the target
    grasp_rcv = 4.0 * r["grasp"]
    grasp_both = 3.0 * s["grasp"] * r["grasp"]
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])
    air_pinch_pen = -0.5 * (s["air_pinch"] * (1.0 - hold_s) + r["air_pinch"] * (1.0 - hold_r))
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])
    # the back of the hand on the cup is not contact progress — it costs
    dorsal_pen = -1.0 * (s["dorsal"] + r["dorsal"])

    # ---------------------------------------------------- stage 4: lift
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # weight 8.0 per hand > the whole grasp plateau's marginal value: a finished grasp on the table is not the end
    lift_src = 8.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.025)
    lift_rcv = 8.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.025)
    lifted = hold_s * (h_src > 0.03).to(dtype) * hold_r * (h_rcv > 0.03).to(dtype)

    # ---------------------------------------------------- stage 5: carry together (tilt-aware target)
    tilt = ctx.src_cup_tilt
    # upright: mouths 10 cm apart (bodies side by side, no nesting); the gap closes as tilt -> ~90 deg,
    # because in a real pour the source ORIGIN stays aside and the tilted mouth swings over the receiver.
    pour_prog = torch.clamp((tilt - 0.35) / 1.2, 0.0, 1.0)
    tgt_dxy = 0.10 * (1.0 - pour_prog)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    dxy_err = torch.abs(mouth_dxy - tgt_dxy)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = _band_err(mouth_dz, 0.04, 0.12)                     # source mouth 4..12 cm higher
    carry_q = (0.5 * torch.clamp(1.0 - dxy_err / 0.30, 0.0, 1.0) + 0.5 * _near(dxy_err, 30.0)) * _near(dz_err, 15.0)
    align = 8.0 * lifted * q_s * carry_q     # weight 8.0 = lift: carrying toward the receiver must keep paying

    # ---------------------------------------------------- stage 6: tilt and pour
    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)            # 1 cm error -> 0.67
    tilt_r = 7.0 * lifted * q_s * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)   # tilt pays only on the schedule
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads); increments, never the level
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # ---------------------------------------------------- stage 7: success
    success = 30.0 * ctx.success.to(dtype)   # 30/step on top of all held income: keep the goal state

    # ---------------------------------------------------- constraints (guards, not shaping)
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - lifted * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
    rcv_upright_pen = -0.8 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.15, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    # knocking a cup over and shoving it across the table are the same failure (pressing one side only)
    disturb_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.3, min=0.0))
                          + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.3, min=0.0))) \
                  - 0.2 * (src_free * torch.tanh(torch.clamp(disp_src - 0.03, min=0.0) / 0.03)
                           + rcv_free * torch.tanh(torch.clamp(disp_rcv - 0.03, min=0.0) / 0.03))
    drop_pen = -1.0 * (src_free * (h_src > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.src_cup_lin_vel[:, 2], min=0.0) / 0.5)
                       + rcv_free * (h_rcv > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.rcv_cup_lin_vel[:, 2], min=0.0) / 0.5))
    cup_speed_pen = -0.2 * (torch.tanh(torch.clamp(torch.norm(ctx.src_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4)
                            + torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4))
    nested_pen = -2.0 * ctx.cups_nested.to(dtype)
    # real-robot safety: the cups never touch each other and each hand touches only its own cup
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    hand_foreign_pen = -2.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    # the palm is not part of a fingertip grasp: pushing the cup with it costs
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ---------------------------------------------------- regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src, "approach_rcv": approach_rcv,
        "orient_src": orient_src, "orient_rcv": orient_rcv,
        "open_src": open_src, "open_rcv": open_rcv,
        "spread_src": spread_src, "spread_rcv": spread_rcv,
        "oppose_src": oppose_src, "oppose_rcv": oppose_rcv,
        "pocket_src": pocket_src, "pocket_rcv": pocket_rcv,
        "pose_src": pose_src, "pose_rcv": pose_rcv, "pose_both": pose_both,
        "close_src": close_src, "close_rcv": close_rcv,
        "squeeze_src": squeeze_src, "squeeze_rcv": squeeze_rcv,
        "touch_src": touch_src, "touch_rcv": touch_rcv,
        "grip_src": grip_src, "grip_rcv": grip_rcv,
        "grasp_src": grasp_src, "grasp_rcv": grasp_rcv, "grasp_both": grasp_both,
        "curl_far_pen": curl_far_pen, "air_pinch_pen": air_pinch_pen,
        "rim_hook_pen": rim_hook_pen, "dorsal_pen": dorsal_pen,
        "lift_src": lift_src, "lift_rcv": lift_rcv,
        "align": align, "tilt": tilt_r,
        "pour_delta": pour_delta, "spill_delta": spill_delta, "success": success,
        "pre_tilt_pen": pre_tilt_pen, "rcv_upright_pen": rcv_upright_pen,
        "disturb_pen": disturb_pen, "drop_pen": drop_pen, "cup_speed_pen": cup_speed_pen,
        "nested_pen": nested_pen, "cup_collision_pen": cup_collision_pen,
        "hand_foreign_pen": hand_foreign_pen, "palm_push_pen": palm_push_pen,
        "action_rate_pen": action_rate_pen, "arm_speed_pen": arm_speed_pen,
    }
    reward = zero
    for v in components.values():
        reward = reward + v
    return reward, components
```
