Observations for this round (previous reward function iter_06, trained from scratch, stopped early at epoch 565 by operator decision). Items 1-6 are facts: a rollout video the operator watched, two checks run against the reward code and the robot model, training metrics, and a new logging-only measurement added to the environment. Item 7 records operator decisions.

1. **The hand was never around the cup. The operator watched the previous round's video and found the parts in the wrong order.** Looking along the hand: index finger — THUMB — cup. The thumb sits between the fingers and the cup instead of on the far wall, so closing the fingers can never trap the cup. The tip sensors that have to meet (thumb tip against the index/middle/ring/pinky tips) never face each other.

2. **The previous reward pays for that unusable posture.** `pinch_geo` measures the distance from the cup axis to the segment joining the thumb tip and the index (or middle) tip, both taken in the plane normal to the cup axis. Evaluated directly on synthetic postures with a 57 mm cup:

| posture | cup axis to segment | `pinch_geo` |
|---|---|---|
| true opposition (thumb 0°, index 180°) | 0.0 mm | 1.00 |
| the posture in the video (index — thumb — cup, same side) | 34.5 mm | 0.19 |
| thumb and index side by side on the same wall | 34.0 mm | 0.26 |
| thumb and index 90° apart | 24.4 mm | 0.38 |

   The previous round (iter_05) ended with `pinch_geo` at 0.21, which is exactly this band. A term that reads as "one fifth achieved" was in fact paying for a posture that cannot grasp.
   The opposition factor `opp` exists inside the reward (`0.5 + 0.5·opp` multiplies `close`), but it is not logged, so no metric distinguished the two cases.
   `close` also has a floor: with `pinch_geo = 0` and `opp = 0` it still pays `0.35 × 0.5 = 0.175` of its weight 3.0 for tips merely near the wall.

3. **The hand itself can grasp this cup — the failure is not kinematic.** Forward kinematics on the robot's own URDF, sweeping the thumb opposition joint `r_hj_thumb_1` over its full range (0 to 2.09 rad):
   - open hand: thumb tip to index tip 95-107 mm; the profile's measured thumb-to-four-finger normal gap is 83.7 mm at 1.57 rad, its maximum.
   - grip pose (`thumb_1` 1.20, four fingers 1.08/0.85): thumb tip to index tip 43.8 mm, to middle tip 54.0 mm.
   - The cup is 57 mm across. It fits in the open pocket and the grip pose closes past it, so a real opposition grasp is reachable.

4. **Training metrics up to the stop (medians over epoch windows).** The source hand improved fast on the reward's own terms; the receiver hand never engaged.

| metric | ep 0-10 | ep 150-160 | ep 300-310 | ep 400-410 | ep 555-565 |
|---|---|---|---|---|---|
| `reward/close_src` (max 3.0) | 0.05 | 0.46 | 0.63 | 0.72 | 2.30 |
| `reward/close_rcv` (max 3.0) | 0.05 | 0.14 | 0.20 | 0.24 | 0.18 |
| `reward/pinch_geo_src` (max 0.8) | 0.00 | 0.126 | 0.166 | 0.200 | 0.619 |
| `reward/pinch_geo_rcv` (max 0.8) | 0.00 | 0.000 | 0.000 | 0.002 | 0.000 |
| `reward/grip_src` / `reward/grip_rcv` | 0 / 0 | 0 / 0 | 0.004 / 0 | 0.002 / 0 | 0 / 0 |
| `reward/touch_src` | 0.001 | 0.007 | 0.026 | 0.018 | 0.084 |
| `contact/src_max` [N] | 0.25 | 0.08 | 0.09 | 0.07 | 0.21 |
| `contact/rcv_max` [N] | 0.77 | 0.000 | 0.006 | 0.035 | 0.002 |
| `task/src_grasped` / `task/rcv_grasped` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `task/src_cup_lift` [m] | 0.013 | 0.0004 | 0.0003 | 0.0002 | 0.0007 |
| `task/src_palm_to_cup` / `task/rcv_palm_to_cup` [m] | 0.294 / 0.283 | 0.137 / 0.166 | 0.124 / 0.153 | 0.122 / 0.146 | 0.095 / 0.161 |
| `task/src_near_rate` / `task/rcv_near_rate` | 0 / 0.002 | 0 / 0 | 0.001 / 0 | 0.001 / 0 | 0.917 / 0 |
| `task/src_thumb_above_rim_mm_near` [mm] | – | – | – | −31.5 | −40.4 |
| `task/src_closure` / `task/rcv_closure` | 0.14 / 0.13 | 0.47 / 0.24 | 0.47 / 0.27 | 0.42 / 0.27 | 0.29 / 0.26 |

   - Reward split over the last 20 epochs, `reward/total` 4.44: close_src 2.30, pinch_geo_src 0.63, approach_src 0.48, squeeze_src 0.39, approach_rcv 0.30, close_rcv 0.16, orient_src 0.09, touch_src 0.08. Everything downstream of contact (grip, grasp, lift, align, tilt, pour, success) is 0.
   - So the source hand collected 3.4 of 4.4 from posture terms while its finger force stayed at 0.21 N, one fifth of the 1 N the environment's grasp flag needs, and `grip_src` (which needs BOTH the thumb and an opposing finger pressing) fell back to 0.
   - The receiver hand went backwards: it ended 16.1 cm from its cup with zero contact and `pinch_geo` 0.
   - Physics stayed clean: `ctrl/mimic_err_max` 0.25 rad, no runaway, no drop, no cup collision.

4b. **Rollout video at the stop point confirms the posture terms were not measuring a real grasp** (epoch-550 checkpoint, 4 envs, 900 steps, `our_source/pour_t2r_rh_i06_ep550_0916.mp4`, frame sheet and 2.4x crops in `pour_t2r_rh_i06_ep550_0916_frames/`):
   - The hand stands BEHIND its cup. The four fingers run down one side of the cup and the thumb is on the other side but lower, near the mat; the cup passes in front of the palm rather than between the fingertips and the thumb tip.
   - The fingertips point past the cup into open space, not at the cup wall. Nothing closes on the cup at any point in the episode, and the posture at step 500 and at step 880 is the same.
   - So `close` 2.30 of 3.0 and `pinch_geo` 0.62 of 0.8 were paid for a hand that is merely near the cup on both sides of it, which is the same failure the operator identified in the previous round's video. Contact stalling at 0.21 N follows from this: the tips graze past the cup instead of pressing its wall.

5. **New logging-only measurements were added to the environment for the next round** (not in the reward, not in the observations; fixed by a contract test). Over envs whose palm is within 12 cm of the cup:
   - `task/{src,rcv}_thumb_oppose_near` — how opposite the thumb tip is to the mean of the index/middle tips, in the plane normal to the cup axis. 1 = exactly opposite, 0 = same side. This is the quantity that was invisible in every round so far.
   - `task/{src,rcv}_tip_gap_mm_near` — thumb tip to nearest index/middle tip, in mm. Compare against the 57 mm cup.
   - `task/{src,rcv}_cup_in_pocket_near` — fraction of envs where the cup axis actually lies between the two tips and both tips are outside the cup's inner radius.

6. **Previous rounds, for context.**
   - Round 7 (iter_05): `pinch_geo` reached 0.21, contact stalled at 0.20 N, no grasp. The video showed the index — thumb — cup order described in item 1.
   - Round 6 (iter_04): the approach was solved (palms beside the cups, thumbs below the rim) but the fingers never closed on the cup.
   - Round 5 (iter_03): both palms stopped 16-17 cm away with zero contact.

7. **Operator decisions** (after reviewing items 1-6 and the previous round's video):
   - This draft is approved as the feedback for this round, and a new reward is to be generated. Training starts from scratch, because the reward's meaning changes.
   - The round was stopped early on purpose. The reason is item 1-2: the posture terms were rewarding a hand that cannot grasp, so letting the round finish would only have refined that posture.
   - What the next reward must fix: a posture only counts when the thumb is on the OPPOSITE wall from the index/middle tips, with the cup body between them. A thumb beside the fingers, a thumb between the fingers and the cup, or tips merely near the wall must not pay. The floor in `close` (0.175 of its weight with no opposition and no pinch) is part of this problem.
   - The goal after that is unchanged: the thumb and at least one opposing finger both above 1 N at the same time (the environment's grasp flag), then lifting.
   - The three new measurements in item 5 are for judging the next round without a video; they must stay out of the reward and the observations.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
