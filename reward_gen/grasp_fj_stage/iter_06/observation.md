The round was stopped by the user at epoch 2041 of 3000 (3.35 h), before the time limit, because the recording showed the far-start approach being made in the wrong way and the user judged that both the reward and the environment settings need changing. The recording was made from the epoch-1500 checkpoint and shows two consecutive episodes of one environment (30 s); the user identified the first episode as a far start and the second as a near start.

What the robot does in the recording:
- First episode (far start): the hand does reach the cup and ends up gripping it, but it gets there by rotating the wrist rather than by carrying the hand to the cup's side in its start orientation. The hand arrives low and turned, so the cup is reached without the C-shaped pre-grasp pose the approach is supposed to hold.
- Second episode (near start): the hand is already beside the cup, closes on it immediately and lifts it. There is no approach phase to speak of.
- In both episodes the grasp itself is a real envelope: the palm is against the cup and the fingers wrap around it, and the cup is carried well clear of the table, up to about the height of the robot's column.
- The cup is visibly tilted while it is carried, by roughly thirty degrees in several frames. It is never dropped and never knocked over.
- The hand goes in low but does not sink into the table.

Metrics over the last 150 epochs, at epoch 2041:
- approach_done was set in 0.0014 of far-start episodes and 1.0000 of near-start episodes. Its highest value for far starts in the whole round was 0.669, reached at about epoch 530, after which it fell away and did not return.
- envelope_done was set in 0.0022 of far-start and 0.9288 of near-start episodes; lift in 0.0040 and 0.9350; success in 0.0000 and 0.7549. Essentially every success in this round came from a near start.
- Episode funnel: reach 0.45, grasp 0.42, envelope 0.40, lift 0.40, success 0.34. Successes per finished episode 1.71, with a maximum of 2.84 during the round.
- Of the six approach conditions, orientation 0.855 and no-touch 0.735 pass often, while gap 0.083, along 0.046, height 0.363 and pose 0.208 rarely do; all six must hold on the same step for approach_done to latch.
- The cup was raised 0.056 m on average, against 0.006 m in the previous round, and its distance to the goal fell to 0.210 m from 0.288 m. Mean cup tilt 3.7 degrees at the end, having peaked at 23.9 degrees mid-round; the cup was tipped over in 0.0002 of episodes.
- The hand stayed off the table for the whole round: the episode-ending floor condition never fired once, the deepest penetration was 0.006 m against 0.028 m in the previous round, and the lowest hand point stayed at 0.2145 while the table is at 0.205.
- The little finger touched the cup on 0.242 of steps, against 0.059 in the previous round.
- Both curricula tightened continuously: the success tolerance from 0.1125 m to 0.0664 m, and the wrap threshold that success also requires from 0.150 to 0.689, its ceiling being 0.85. The measured wrap fraction at the end was 0.189, far below that threshold.
- Total reward 1.071 at the end, with the lift terms now the largest positive parts: lift_height 0.394, lift_goal 0.212, lift_clear 0.180.
- The table measure used during this round did not include the palm. It was built from the finger and thumb links only, 28 of the hand's 30 bodies, leaving out the palm and the virtual palm point. The palm collision body measures 64 by 82 by 99 mm and its lowest extent sits 40 mm from the virtual palm point along one palm axis and up to 65 mm along another, so the palm can be on the table while that point is well above it. This is why the floor condition never fired and why the reward term meant to keep the palm up, which used the virtual point with a 20 mm threshold, paid exactly zero for the whole round.
- Per-joint arm action saturation does not single out the wrist: j1 0.554, j6 0.515, j4 0.487, j5 0.432, j7 0.396. No arm joint went past its limit at any point. The recording shows the wrist-led approach clearly, but the logged telemetry does not measure which joint closes the distance, so this is an observation from the video only.
