Observations for this round (previous reward function iter_04, corrected robot model, trained from scratch for 686 epochs). Items 1-6 are facts: training metrics, the environment's own grasp definition and a rollout video. Item 7 records operator decisions.

1. **The approach now works.** Both palms reach the pre-grasp region beside their cups with the thumb below the rim. The environment also logs an approach check that is independent of the reward: over envs whose palm is within 10 cm of the cup origin, how often the thumb tip is at or above rim height over the cup opening, and the mean thumb-tip height relative to the rim. Values are medians over epoch windows:

| metric | ep 0-10 | ep 225-235 | ep 400-410 | ep 590-600 | ep 676-686 |
|---|---|---|---|---|---|
| `task/src_palm_to_cup` [m] | 0.222 | 0.138 | 0.118 | 0.109 | 0.106 |
| `task/rcv_palm_to_cup` [m] | 0.241 | 0.086 | 0.083 | 0.088 | 0.082 |
| `task/src_near_rate` (palm < 10 cm) | 0.00 | 0.00 | 0.10 | 0.54 | 0.63 |
| `task/rcv_near_rate` | 0.00 | 0.88 | 0.91 | 0.92 | 0.93 |
| `task/src_thumb_over_rim_near` | – | – | 0.00 | 0.00 | 0.00 |
| `task/rcv_thumb_over_rim_near` | – | 0.10 | 0.02 | 0.14 | 0.03 |
| `task/src_thumb_above_rim_mm_near` [mm] | – | – | +0.2 | −5.3 | −11.8 |
| `task/rcv_thumb_above_rim_mm_near` [mm] | – | +3.0 | −11.1 | −11.3 | −15.0 |

2. **Contact grows on the receiver side only, and no grasp is ever registered.**

| metric | ep 0-10 | ep 225-235 | ep 400-410 | ep 590-600 | ep 676-686 |
|---|---|---|---|---|---|
| `contact/src_max` [N] | 0.32 | 0.011 | 0.023 | 0.002 | 0.001 |
| `contact/rcv_max` [N] | 0.45 | 0.38 | 0.96 | 1.64 | 2.25 |
| `task/src_grasped` / `task/rcv_grasped` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0.0015 | 0 / 0 |
| `task/src_closure` / `task/rcv_closure` | 0.13 / 0.13 | 0.55 / 0.57 | 0.62 / 0.66 | 0.59 / 0.57 | 0.59 / 0.55 |
| `task/src_cup_lift` / `task/rcv_cup_lift` [m] | 0.002 / 0.004 | 0.0001 / 0.0002 | 0.0001 / 0.0005 | 0.0002 / 0.0010 | 0.0002 / 0.0011 |

   - `contact/*_max` is the largest per-finger force on the hand's own cup, averaged over envs.
   - The environment's grasp flag (`task/*_grasped`, and `ctx.src_grasped` / `ctx.rcv_grasped`) is true only when the THUMB force is above 1 N AND at least one other finger's force is above 1 N at the same time.
   - So a receiver finger force of 2.25 N with the flag at 0 means that one side of the grasp (thumb, or all other fingers) stays at or below 1 N.

3. **Where the reward comes from.** Per-step medians over the last 30 epochs:
   - Approach, orient and tip terms: approach_src 1.23, approach_rcv 1.40 (max 1.5 each), orient_src 0.11, orient_rcv 0.18, tips_src 0.09, tips_rcv 0.20. Together 3.21 of `reward/total` 3.37.
   - Closing and contact terms: close_rcv 0.10, close_src 0.007, contact_rcv 0.148, contact_src 0.0002.
   - Terms at zero: grasp_src, grasp_rcv, grasp_both, lift_src (0.0000), lift_rcv (0.0002), align, tilt, pour_delta, success.
   - Penalties: rim_hook_pen −0.012, curl_far_pen −0.0004; all other penalties ≈ 0.
   - Most of the income is paid for being in the approach region, which the policy already holds.

4. **Rollout video** at the epoch-700 checkpoint (4 envs, default camera, env 0 shown, `our_source/pour_t2r_rh_i04_ep700_0916.mp4`, frame sheets in `pour_t2r_rh_i04_ep700_0916_frames/`):
   - From about step 100 until the reset at step 900, both hands stay low beside their cups. Their posture barely changes, no cup is lifted, and the arms do not move up.
   - The hand on the image left stands beside its cup with the four fingers curled downward and the thumb curled toward them. The fingers do not wrap the cup body, and the cup stands just outside the hand.
   - The hand at the image centre is closed, with fingers and thumb curled. Its cup is hidden behind or inside the hand, so thumb placement on the cup wall cannot be seen at this resolution.
   - Which image hand is source and which is receiver was not determined from the video. The contact metrics above suggest the centre hand is the one touching its cup.

5. **Physics stayed clean.** `ctrl/mimic_err_max` median was 0.12 rad over the last 10 epochs. `done/mimic_err_runaway`, `done/drop`, `task/cup_collision_rate` and `task/*_hand_foreign_rate` were all ≈ 0 after epoch 225.

6. **Previous rounds, for context.**
   - Round 5 (iter_03 on the corrected model): both palms stopped 16-17 cm away, with zero contact.
   - Round 4 (old model): the right hand held its cup with the thumb over the mouth, and the left hand never touched its cup.

7. **Operator decisions** (after reviewing items 1-6, the frame sheet and the zoomed hands):
   - This draft is approved as the feedback for this round, and a new reward is to be generated.
   - The approach stage is considered solved. Both palms reach the pre-grasp region beside the cup with the thumb below the rim, and that behaviour must be kept.
   - The next bottleneck is the grasp itself: the thumb and an opposing finger pressing the cup at the same time (both above 1 N), with the thumb on the wall opposite the fingers. After that comes lifting. Neither has ever happened for either hand.
   - The judgement from earlier rounds still stands: once both hands reliably grasp their cups, the later stages are expected to follow.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
