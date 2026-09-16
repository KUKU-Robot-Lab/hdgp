Observations for this round (previous reward function iter_07, trained from scratch for 680 epochs, 1024 envs, 3.47 h). Items 1-6 are facts: training metrics, the previous reward function's own terms, new environment measurements and a rollout video. Item 7 records operator decisions.

1. **The change made last round did what it was meant to do: the reward no longer pays for a posture that cannot grasp.** Last round's finding was that the posture term paid 0.19-0.38 for a hand with the thumb on the same side as the fingers. This round's reward gated everything on real opposition and removed the floors. The result is that the posture terms paid essentially nothing for the whole run:

| term | ep 0-10 | ep 200-210 | ep 400-410 | ep 670-680 | positive epochs in the last 150 |
|---|---|---|---|---|---|
| `reward/oppose_src` | 0.0066 | 0.0000 | 0.0000 | 0.0000 | 0 / 150 |
| `reward/oppose_rcv` | 0.0080 | 0.00002 | 0.0000 | 0.0000 | 6 / 150 (max 2.5e-05) |
| `reward/pose_src` / `pose_rcv` | 0.00006 / 0.0002 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 150 · 0 / 150 |
| `reward/close_src` / `close_rcv` | 0.00005 / 0.0002 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 150 · 0 / 150 |
| `reward/pocket_src` / `pocket_rcv` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 150 · 0 / 150 |
| `reward/grip_src` / `grip_rcv` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 150 · 0 / 150 |

2. **With no income past the approach, the policy settled on the approach term and stopped there.**
   - `reward/total` 0.64 at the end, of which `approach_src` 0.282 and `approach_rcv` 0.280 — 88 % of all income.
   - `task/src_palm_to_cup` 0.231 → 0.128 (ep 200) → 0.120 (ep 400) → 0.121 m; `task/rcv_palm_to_cup` 0.363 → 0.149 → 0.126 → 0.121 m. Both palms have been pinned at 12 cm since about epoch 400, which is where the approach term saturates.
   - `task/src_near_rate` (palm within 10 cm) stayed at 0-1 %, so the environment's rim/oppose measurements, which only count envs inside 12 cm, had almost no sample all round.

3. **The hand closed to narrower than the cup and stayed there. This is the mechanism that blocks the pocket.**

| metric | ep 0-10 | ep 200-210 | ep 400-410 | ep 670-680 |
|---|---|---|---|---|
| `task/src_tip_gap_mm_near` [mm] | 55.4 | 52.2 | 50.5 | 48.3 |
| `task/rcv_tip_gap_mm_near` [mm] | 66.6 | – | 48.9 | 49.5 |
| `task/src_closure` / `task/rcv_closure` | 0.15 / 0.14 | 0.26 / 0.36 | 0.32 / 0.44 | 0.36 / 0.29 |
| `task/{src,rcv}_cup_in_pocket_near` | 0 / 0.17 | 0 / 0 | 0 / 0 | 0 / 0 |

   - `tip_gap` is the distance from the thumb tip to the nearer of the index/middle tips. The cup is 57 mm across.
   - At reset the hand is 55-67 mm wide, which is wide enough. Over training it narrowed monotonically to 48-50 mm, i.e. the policy actively closed the hand to less than the cup's width and kept it there while standing 12 cm away.
   - Consequently the cup was never between the tips: `cup_in_pocket` was 0 in every one of the last 150 epochs.
   - Nothing in the reward asks the hand to be open wider than the cup near the cup. `curl_far` only penalises curling while FAR from the pre-grasp region, and every term that would pay for opening (`pocket`, `pose`, `close`) is gated behind an opposition that the closed hand can never reach.

4. **Contact, grasp and lift.**

| metric | ep 0-10 | ep 200-210 | ep 400-410 | ep 670-680 |
|---|---|---|---|---|
| `contact/src_max` / `contact/rcv_max` [N] | 0.16 / 0.66 | 0.02 / 0.04 | 0.06 / 0.08 | 0.12 / 0.09 |
| `task/src_grasped` / `task/rcv_grasped` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `task/src_cup_lift` / `task/rcv_cup_lift` [m] | 0.001 / 0.013 | 0.0001 / 0.0001 | 0.0001 / 0.0002 | 0.0003 / 0.0003 |
| `task/episode_success` | 0 | 0 | 0 | 0 |

   - 0.09-0.12 N with opposition at zero is a graze, not a press. The environment's grasp flag needs the thumb above 1 N AND another finger above 1 N at the same time.
   - The thumb height did come down: `task/src_thumb_above_rim_mm_near` reached −15.6 mm mid-round, and `task/{src,rcv}_thumb_over_rim_near` was 0 all round, so the old rim-hook failure is gone. Height is solved; the left-right placement is not.

5. **Rollout video** at the epoch-650 checkpoint (4 envs, 900 steps, `our_source/pour_t2r_rh_i07_ep650_0916.mp4`, frame sheet and 2.4x crops in `pour_t2r_rh_i07_ep650_0916_frames/`). The question asked of the video was whether the hand arrives already closed or opens and then closes again. The answer is the first:
   - At step 60, while the arm is still travelling, the four fingers are ALREADY curled downward. The hand passes beside the cup in that shape.
   - At step 500 the hand has settled next to the cup in the same shape and stays there to the end of the episode. The fingertips point down at the mat, not around the cup wall, and the cup stands to one side of the palm rather than between the fingers and the thumb.
   - **Operator observation (takes precedence over the frame reading above): both hands put the BACK of the index-to-pinky fingers against the cup and then hold still.** The dorsal side of the fingers is what meets the cup, not the pads. The finger force sensors report whatever touches the link, in any direction, so the 0.09-0.12 N in item 4 is contact on the wrong surface entirely — the palmar side never faces the cup.
   - At no point in the episode does the hand open wider and then close. So the narrow `tip_gap` in item 3 is not a closing motion that overshoots; the hand simply never opens, which matches a reward that contains no term asking it to.

5b. **The reward cannot tell the palm side from the back of the hand.** In the previous reward the orientation term is

```
face = |normal_xy · u_xy|        # absolute value
orient = near_pre * (0.5*face + 0.5*ahead)
```

   where `normal` is the palm-pad normal (`palm_axes[:, 0:3]`) and `u_xy` points from the palm to the cup. Because of the absolute value, a palm facing the cup and a palm facing exactly away from it score identically. Nothing else in the reward looks at pad direction either: `pose`, `pocket` and `close` use tip POSITIONS only, and `ctx.*_finger_force` is the net force on the sensor link, which is direction-agnostic. So the posture the operator saw — the back of the fingers laid against the cup — costs nothing and reads as legitimate contact. Any fix has to make the pad side facing the cup a signed requirement, and must not count force arriving on the dorsal side as progress.

6. **Physics stayed clean and was the best of any round.** `ctrl/mimic_err_max` 1.35 at start → 0.12 rad at the end. No mimic runaway, no drop, no cup collision.

7. **Operator decisions** (after reviewing items 1-6 and the video):
   - This draft is approved as the feedback for this round, and a new reward is to be generated. Training starts from scratch.
   - **Keep what worked.** The opposition gate with no floors (item 1) did its job: no posture that cannot grasp was paid anything, all round. Do not reintroduce a floor that pays for standing near the cup.
   - **Three things the next reward must add**, because the round showed each of them missing:
     1. *The pad side must face the cup, as a signed condition.* Replace the absolute value in `face` (item 5b). A palm turned away from the cup must score zero, not full marks.
     2. *Opening the hand near the cup must be rewarded before opposition is reached.* The thumb-to-finger gap must exceed the cup's 57 mm while the palm is near the cup. This is the step with no gradient today, and the reason the hand closed to 48-50 mm and stayed there (item 3).
     3. *Contact arriving on the back of the fingers must not count as progress.* Force on a finger whose pad does not face the cup is not a grasp in the making.
   - The goal after that is unchanged: the thumb and at least one opposing finger both above 1 N at the same time (the environment's grasp flag), then lifting.
   - The three measurements added in the previous round stay logging-only, and `task/{src,rcv}_tip_gap_mm_near` is the metric to watch for requirement 2.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
