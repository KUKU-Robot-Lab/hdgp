Operator notes for this reset (iter_00, no previous code). Every number below was recomputed from the raw per-trial results of an open-loop scripted grasp-and-lift test in the same simulation (same robot, same cups, same hand controller). 1344 trials: 2 hands x 3 values of the env's per-finger contact-freeze force (0.3 / 1.0 / 4.0 N) x 224 palm offsets relative to the cup. The script moves the palm sideways onto the cup, closes the hand at the env's maximum closing rate, then raises the palm 10 cm. The training env uses a freeze force of 4.0 N: a finger stops closing once it feels that force. These are measurements of one scripted motion, not proof of what a learned policy can or cannot do.

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
