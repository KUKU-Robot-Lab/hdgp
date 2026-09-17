Observations for this round (previous reward function iter_05, 651 epochs from scratch, 1024 envs, 3.45 h). Items 1-6 are facts: training metrics, the previous reward function's own formulas, the environment's grasp definition and a rollout video. Item 7 records operator decisions.

1. **The pinch geometry started to form, on both hands, for the first time.** `pinch_geo` is the previous reward's own measure of "the cup axis lies between the thumb tip and the index or middle tip, both tips at wall height and outside the opening", scaled 0-1. Values are medians over epoch windows:

| metric | ep 0-10 | ep 150-160 | ep 300-310 | ep 450-460 | ep 641-651 |
|---|---|---|---|---|---|
| `reward/pinch_geo_src` (max 1.0) | 0.000 | 0.001 | 0.003 | 0.136 | 0.167 |
| `reward/pinch_geo_rcv` (max 1.0) | 0.000 | 0.001 | 0.001 | 0.003 | 0.092 |
| `reward/squeeze_src` (max 0.6) | 0.000 | 0.000 | 0.004 | 0.041 | 0.047 |
| `reward/squeeze_rcv` (max 0.6) | 0.000 | 0.000 | 0.000 | 0.003 | 0.026 |
| `task/src_thumb_above_rim_mm_near` [mm] | 0.0 | +13.6 | −2.6 | −18.0 | −16.7 |
| `task/rcv_thumb_above_rim_mm_near` [mm] | +1.7 | +29.9 | +18.0 | +12.0 | −3.2 |

   - The source hand reached the pinch posture around epoch 400-450; the receiver hand only in the last 100 epochs.
   - Both thumbs are now at or below the rim. The receiver thumb crossed below the rim only at the very end of the run.

2. **Contact never reached the grasp threshold, and no grasp was ever registered on either hand.**

| metric | ep 0-10 | ep 150-160 | ep 300-310 | ep 450-460 | ep 641-651 |
|---|---|---|---|---|---|
| `contact/src_max` [N] | 0.28 | 0.04 | 0.33 | 0.21 | 0.20 |
| `contact/rcv_max` [N] | 0.38 | 0.01 | 0.01 | 0.01 | 0.10 |
| `task/src_grasped` / `task/rcv_grasped` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `task/src_cup_lift` / `task/rcv_cup_lift` [m] | 0.002 / 0.009 | 0.0001 / 0.0001 | 0.0002 / 0 | 0.0006 / 0 | 0.0008 / 0.0003 |
| `task/src_closure` / `task/rcv_closure` | 0.14 / 0.13 | 0.45 / 0.49 | 0.48 / 0.49 | 0.42 / 0.46 | 0.38 / 0.42 |
| `task/episode_success` | 0 | 0 | 0 | 0 | 0 |

   - `contact/*_max` is the largest per-finger force on the hand's own cup, averaged over envs. It is a maximum over fingers, so 0.20 N is consistent with a single finger touching and the thumb at zero.
   - The environment's grasp flag (`task/*_grasped`) is true only when the THUMB force is above 1 N AND at least one other finger's force is above 1 N at the same time.
   - The previous reward's `pinch_force = (1−e^(−f_thumb/0.6))·(1−e^(−f_other/0.6))·quality` stayed at exactly 0.0000 for the whole run on both hands, although a single finger did reach 0.20 N. Both sides were therefore never pressing at the same time.
   - `reward/touch_src` (one-sided contact with the cup in the pinch) reached only 0.0115 of its 0.4 maximum.

3. **Where the reward comes from.** Per-step medians over the last 30 epochs, `reward/total` 3.71:
   - approach_rcv 1.376, approach_src 1.376 (max 1.5 each), orient_src 0.264, orient_rcv 0.231, tips_src 0.139, tips_rcv 0.118. Together 3.50 of 3.71.
   - The new pinch stages: pinch_geo_src 0.168, pinch_geo_rcv 0.092, squeeze_src 0.047, squeeze_rcv 0.026, touch_src 0.012, touch_rcv 0.003. Together 0.35.
   - Terms at exactly zero: pinch_force_src/rcv, grasp_src/rcv, grasp_both, lift_src/rcv, align, tilt, pour_delta, spill_delta, success.
   - Penalties: action_rate_pen −0.039, hand_foreign_pen −0.026, rim_hook_pen −0.023, arm_speed_pen −0.007, air_pinch_pen −0.0003; all others ≈ 0.
   - As in the previous round, most of the income is still paid for standing in the approach region, which the policy already holds.

4. **Approach regressed while the pinch improved.** Over envs whose palm is within 10 cm of the cup origin:

| metric | ep 0-10 | ep 150-160 | ep 300-310 | ep 450-460 | ep 641-651 |
|---|---|---|---|---|---|
| `task/src_palm_to_cup` [m] | 0.225 | 0.104 | 0.101 | 0.108 | 0.112 |
| `task/rcv_palm_to_cup` [m] | 0.271 | 0.113 | 0.100 | 0.107 | 0.112 |
| `task/src_near_rate` | 0.00 | 0.58 | 0.66 | 0.40 | 0.13 |
| `task/rcv_near_rate` | 0.00 | 0.36 | 0.67 | 0.46 | 0.21 |
| `task/src_thumb_over_rim_near` | 0.00 | 0.02 | 0.23 | 0.43 | 0.64 |
| `task/rcv_thumb_over_rim_near` | 0.67 | 0.16 | 0.01 | 0.03 | 0.31 |

   - The mean palm distance barely changed (10.1 → 11.2 cm) but the fraction of envs inside 10 cm fell from about two thirds to 13-21 %. The approach reward itself stayed near its cap (1.376 of 1.5).
   - `task/*_thumb_over_rim_near` counts the thumb tip inside a band from the rim down to 2 cm below it, over the cup footprint. It rose to 0.64 (source) while the mean thumb height went to −16.7 mm, so the thumb is passing through that band on its way down the wall rather than sitting on the mouth. The reward's own `rim_hook_pen` fell over the same epochs (−0.060 at ep 270 → −0.023 at the end).

5. **Rollout video** at the epoch-650 checkpoint (4 envs, 900 steps, `our_source/pour_t2r_rh_i05_ep650_0916.mp4`, frame sheet and 2.4x hand crops in `pour_t2r_rh_i05_ep650_0916_frames/`):
   - The hand in view reaches its cup within the first 100 steps and then keeps the same posture until the end of the episode. The arm does not rise at any point and no cup leaves the mat.
   - The hand stands beside the cup with the palm facing the cup wall. The four fingers are curled downward on the near side of the cup and the thumb is on the far side of the cup wall, below the rim. The cup body sits between them, which matches the `pinch_geo` values in item 1.
   - The curled fingers stop short of the wall: the cup stays upright and untouched-looking, and the fingertips do not close the last centimetres onto it. This matches contact staying at 0.20 N and `pinch_force` at exactly 0.
   - The second robot's hand and cup are only partly inside the frame, so the second hand's finger placement could not be judged from this video.

6. **Physics stayed clean.** `ctrl/mimic_err_max` median 0.24 rad over the last 10 epochs (max over the run 1.18 rad at epoch 0-10). `done/mimic_err_runaway`, `done/drop` and `task/cup_collision_rate` were 0 after epoch 100. `task/src_hand_foreign_rate` 0.007 and `task/rcv_hand_foreign_rate` 0.016 in the last window.

7. **Previous rounds, for context.**
   - Round 6 (iter_04): the approach was solved — both palms beside their cups with the thumb below the rim — but the fingers never closed on the cup; receiver contact reached 2.25 N on one finger, source 0, and no grasp fired.
   - Round 5 (iter_03 on the corrected model): both palms stopped 16-17 cm away, with zero contact.
   - Round 4 (old model): the right hand held its cup with the thumb over the mouth; the left hand never touched its cup.

8. **Operator decisions** (after reviewing items 1-7, the frame sheet and the zoomed hands):
   - This draft is approved as the feedback for this round, and a new reward is to be generated.
   - The pinch posture reached in this round must be kept: palm beside the cup, thumb below the rim on one wall, index/middle on the opposite side, cup between them.
   - The next bottleneck is closing the last centimetres: the fingertips must press the wall so that the THUMB and at least one opposing finger are both above 1 N at the same time (the environment's grasp flag). One-sided contact at 0.2 N is where the policy now rests.
   - After that comes lifting. The judgement from earlier rounds still stands: once both hands reliably grasp their cups, the later stages are expected to follow.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
