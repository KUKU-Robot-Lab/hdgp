Operator notes for round 2 (iter_01). Every number below was measured in this simulation (same robot, same cups, same hand controller) or is a recorded observation of a trained policy. Nothing here is a hypothesis unless it says so.

## What happened to iter_00
- The iter_00 run (t2r_rh2_i00) was stopped by the operator after 35 epochs (about 2.3 M frames) to free the GPU for probes. reward/reach fell from 0.24 to 0.21 (source) and 0.23 to 0.20 (receiver), grasp stayed near 0. That is too short to judge the iter_00 reward. The changes asked for below come from the new hand measurement and from a sibling track, not from a failure of iter_00.

## New measurement: hand geometry in the palm frame (probe_rh_hand_geometry, 09.17)
Palm frame: origin = `*_palm_pos`; n = palm normal = `*_palm_axes[:, 0:3]`; f = finger-length direction = `*_palm_axes[:, 3:6]`; w = across-fingers direction = cross of those two (its sign is opposite on the two hands, so use it only through |w . cup_up| or through n and f).
- At reset both hands are already oriented for a side grasp: w is vertical (index finger on top, pinky at the bottom), n and f are horizontal. The cup origin sits at f = +138 mm, n = +129 mm, 106 mm below the palm origin (right hand) and f = +141 mm, n = +150 mm, 108 mm below (left hand). The grasp place is reached by TRANSLATION ONLY: about 70 mm along f, 85 to 100 mm along n, and about 100 mm down. No wrist rotation is needed: scripted trials at the reset orientation lifted at least as often (13 of 128) as any of the six +/-15 deg wrist variants (2 to 11 of 128 each).
- Open hand (mean closure 0.11 to 0.14): thumb tip at (f 43, n 72) mm, index tip at (f 100, n 10) mm. The opening between thumb tip and index tip, seen along the cup axis, is 84 mm (right) and 93 mm (left). The cup is 57 mm wide at the rim and about 51 mm at pinch height.
- The pinch centre (midpoint of the thumb tip and the mean of the other tips) of the open hand is at (f 71, n 40) mm right, (f 69, n 41) mm left. With the cup axis on that pinch centre, every open fingertip is 41 to 49 mm from the axis, i.e. 12 to 20 mm clear of the 28.6 mm rim radius all round.
- Fingertips stack along the cup axis about 17 to 20 mm apart: in scripted grasps with the thumb tip 53 to 60 mm above the cup origin, index was at +42 to +49, middle +23 to +26, ring +6 to +12, pinky -3 to -13 mm. Relative to the thumb tip the other tips are therefore about 10 mm (index), 32 mm (middle), 47 mm (ring) and 65 mm (pinky) lower. The cup bottom and the table are 59.9 mm below the cup origin, so with the thumb tip 5 mm above the cup origin the pinky tip is already at table height, and with the thumb at +25 mm the pinky is 20 mm above the table.
- While closing, the thumb tip and the index/middle tips reach a cup of radius 25 to 28.6 mm at the same time at mean closure 0.28 to 0.38, with the cup axis at (f 64 to 69, n 46 to 51) mm. Full closure is never needed to hold this cup.
- At mean closure 0.69 the thumb tip and index tip are 5 mm apart. A hand that is more than about 0.35 closed cannot admit the cup between thumb and fingers at all: it must ARRIVE OPEN and close only once the cup axis is at the pinch centre.
- At the reset height the lowest open fingertip (pinky) is 72 mm above the cup origin, 18 mm above the rim, so the open hand can move horizontally over the cup and then descend with the cup passing between thumb and fingers.

## New measurement: friction and restitution (probe_rh_grasp_lift, 6 conditions x 896 trials, 09.17) -- CORRECTION of an earlier draft of these notes, which wrongly said lifts did not change
Friction pairs combine by average. Hand-cup friction was about 0.97 in base / table_low / rest0 and 2.0 in fric_up / both / both_rest0; cup-table friction was 0.97 (base), 1.5 (fric_up), 0.3 (table_low, both). Counts are over the 753-781 trials per condition whose approach left the cup undisturbed.
- NOT changed by friction or restitution: closing moved the cup in 100 % of trials in all six conditions (largest single finger force at that instant 0.29 to 0.62 N, group medians); cups fallen over at the end of closing 486 to 556; cups upright with two-sided contact at the end of closing 73 (base), 86, 87, 90, 87, 60.
- CHANGED by hand-cup friction: held lifts (cup at least 2 cm up for the whole hold, peak tilt at most 30 deg, grasped at least half the hold, no termination) were 2 (base), 7 (table_low), 9 (rest0) against 41 (fric_up), 30 (both), 31 (both_rest0). Of the upright two-sided grasps, 1 of 73 became a held lift in base and 25 of 86 in fric_up. The strict success test passed 0 times in base and 19 / 18 / 9 times with hand-cup friction 2.0.
- The held lifts came mostly at the 1.0 N freeze force (fric_up 36 of 41, both 26 of 30, both_rest0 25 of 31), not at the training value of 4.0 N.
- Side effect of higher friction: cups thrown more than 15 cm up 30 (base) -> 85 / 78 / 111, and arm-runaway terminations 109 -> 201 / 203 / 247.
- Lower table friction alone (7 vs 2) and zero restitution alone (9 vs 2) are small counts from single runs; no repeat run exists to size the noise.
- THE TRAINING ENV FOR THIS ROUND IS STILL THE BASE CONDITION (hand-cup friction about 0.97, freeze 4.0 N) unless the operator says otherwise above.

## Observation of a trained policy on the sibling single-arm track (grasp_fj_rand iter_00, other hand, same reward generator, 2378 epochs)
- What worked there: a staged reward. First an approach pose with an OPEN hand (palm beside the cup, facing it, right height, not touching), then closing. Approach was reached in 0.94 of episodes by epoch 750 and the closed grasp in 0.76 by epoch 900, with the cup placed anywhere on the table.
- What failed there: holding the grasped cup on the table paid about 0.70 per step, lifting it added about 0.05 and brought tilt, table and arm-speed penalties. The policy closed its hand and stayed; in the recording the hand sank towards the table after the grasp instead of rising. The cup was airborne 0.021 of the time after 2378 epochs.
- What was decided there, and is asked for here too:
  1. Grasp terms must saturate once the grasp exists. They are the entry ticket to lifting, not the income.
  2. After the grasp the income must come from cup height above its own spawn height, dense from the first millimetres and rising steadily, and then from carrying. Upward motion of the held cup should pay immediately.
  3. Holding the cup and moving the hand or cup DOWN must pay clearly less than holding still, and holding still clearly less than rising. A policy that closes and never lifts must end up clearly worse off than one that lifts.
  4. Penalties that grow when the cup is lifted (tilt, arm speed, cup speed, table) must stay small against the lift income, so that a slow steady lift always beats no lift. They shape how to lift, they must not be the reason not to lift.
  5. The approach income must not drop once grasp income appears (there it fell from 0.94 to about 0.6).

## Carried over from round 1 (unchanged measurements)
(Round-1 notes, written for iter_00, kept verbatim.) Every number below was recomputed from the raw per-trial results of an open-loop scripted grasp-and-lift test in the same simulation (same robot, same cups, same hand controller). 1344 trials: 2 hands x 3 values of the env's per-finger contact-freeze force (0.3 / 1.0 / 4.0 N) x 224 palm offsets relative to the cup. The script moves the palm sideways onto the cup, closes the hand at the env's maximum closing rate, then raises the palm 10 cm. The training env uses a freeze force of 4.0 N: a finger stops closing once it feels that force. These are measurements of one scripted motion, not proof of what a learned policy can or cannot do.

History: rounds 1-12 of this track are archived. In the last one (reward iter_10, 677 epochs, 1024 envs) neither hand reached its cup: `task/src_grasped` was 0 in all 677 epochs, and `task/src_cup_lift` / `task/rcv_cup_lift` were 0.0001 / 0.007 in epochs 0-10 and 0 afterwards.

Cup: mass 0.134 kg, outer diameter 57 mm, rim at `cup_mouth_z` = +53.9 mm above the cup origin.

1. **The approach knocks the cup over when the palm comes in low.** Before closing started, the cup had already moved more than 10 mm or tilted more than 5 degrees in 120 of 336 trials with the grasp pocket 10 mm below nominal height, 59 of 336 at +10 mm, 13 of 336 at +27 mm and 0 of 336 at +45 mm. All 192 happened during the final sideways move onto the cup, with a median finger force of 1.8-2.5 N at a hand speed of 0.086-0.099 m/s. Items 2-5 use only the 1152 trials whose approach left the cup undisturbed.

2. **Closing the hand moves the cup almost every time, at a very small force.** The cup moved during closing in 94-100 % of trials in all six hand/threshold groups (1132 of 1152). At the instant it first moved, the largest single finger force was 0.30-0.43 N (group medians) and `hand_closure` was 0.31-0.33. Contact at that instant was one-sided in 84-125 trials per group and two-sided in 29-69. Thumb-only contact (62-84 per group) outnumbered fingers-only contact (18-48) by 1.6x to 3.7x.

3. **Which finger touches first goes with whether the cup survives closing.** Thumb first: 519 trials, 380 ended closing fallen over (tilt > 45 deg), 66 upright (tilt < 15 deg). Middle finger first: 573 trials, 303 fallen, 196 upright. Index first: 33 trials, 12 fallen, 13 upright.

4. **A higher freeze force leaves fewer cups standing.** Fallen at the end of closing, source hand: 69/180 (0.3 N), 118/181 (1.0 N), 144/180 (4.0 N). Receiver hand: 123/205, 118/206, 141/200. Upright with thumb AND an opposing finger both in contact (the condition behind `src_grasped` / `rcv_grasped`): source 70 / 37 / 7, receiver 49 / 47 / 8. At the training value of 4.0 N that is 15 of 380.

5. **`grasped` true and upright almost never meant the cup could be lifted.** 218 trials ended closing upright with the grasped condition true. Only 4 of them lifted the cup more than 2 cm when the palm rose 10 cm (the cup rose 6.9-8.6 cm). In the other 214 the cup rose at most 0.3 cm, tilted (median peak tilt 15 deg) and the hand slid off.
   - All 4 lifts were the receiver hand at the 1.0 N freeze force. There were 0 lifts with the source hand, 0 at 0.3 N and 0 at the training value of 4.0 N.
   - The 4 lifts had thumb force 2.08-2.52 N, index force 1.05-2.66 N, and three of them middle-finger force 0.97-1.63 N. Summed finger force 4.76-5.18 N. Palm force was 0 in all four.
   - Force size alone did not separate lifts from failures: the non-lifting trials at 1.0 N had a median summed force of 5.2 N, and those at 4.0 N had 6.1 N with 0 lifts. At 0.3 N (119 trials, median sum 2.2 N: thumb 0.87 N, other fingers about 0.4 N each) nothing lifted.
   - Thumb tip height (`tips_pos[:, 0]` z minus cup origin z) was the clearest split found, on small counts: thumb 0-25 mm above the cup origin, 3 lifts of 52; 25-50 mm, 1 lift of 53 (that thumb was at 49.5 mm); 50 mm or higher, i.e. at or above the rim, 0 lifts of 113, although that group had the largest thumb force (median 2.5 N). With summed force above 4 N: thumb below 50 mm, 4 lifts of 29; thumb at 50 mm or higher, 0 of 63.
   - Even in the best group found, 25 of 29 trials failed to lift. No tested condition lifted reliably.

6. **Items 1 and 5 pull against each other.** A low pocket gives the thumb height at which the lifts happened but hits the cup on the way in; a pocket at +45 mm arrives cleanly but puts the thumb near the rim.

7. **Contact explosions are common.** By the end of closing the cup was more than 1 m away horizontally in 330 of 1344 trials (more than 10 m in 177). In 41 trials the cup was more than 1 m above its rest height during the hold. A reward that pays for raw cup height or cup speed would pay for these. In training the env ends the episode when the cup drops.

Operator decisions: no reward audit on this track. The env is not changed for this round; only the reward function changes.
