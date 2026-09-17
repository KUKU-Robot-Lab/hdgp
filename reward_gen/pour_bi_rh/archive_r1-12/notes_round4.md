Observations from watching the trained policy (rollout video at epoch 800, camera on the robot's right side, 4 envs) plus the training metrics. Items 1-6 are facts about what the policy does; item 7 records operator decisions about this round.

1. The RIGHT (source) hand now reaches its cup and holds it, but with the old rim hook: the four fingers wrap the upper part of the cup body from the robot side, while the thumb arches over the top and rests on the cup MOUTH (the rim of the opening) instead of pressing on the wall opposite the fingers. It keeps this hold for the whole episode and the cup never leaves the table; the arm does not move up.

2. The LEFT (receiver) hand never touches its cup. It hangs above and behind the cup with the palm facing down / back toward the robot and the fingers pointing down; late in the episode the fingertips come down near the rim, but the hand never wraps the cup body.

3. Grasp rose steadily while nothing after it moved (medians over the given epoch windows):

| metric | epoch 300 | epoch 520 | epoch 680 | last 30 (to 836) |
|---|---|---|---|---|
| `task/src_grasped` | 0.000 | 0.013 | 0.140 | 0.268 (max 0.31) |
| `contact/src_max` [N] | 0.14 | 2.72 | 4.53 | 5.82 |
| `task/src_palm_to_cup` [m] | 0.124 | 0.073 | 0.069 | 0.066 |
| `task/src_cup_lift` [m] | 0.0007 | 0.0005 | 0.0009 | 0.0012 |
| `reward/rim_hook_pen` | −0.022 | −0.047 | −0.236 | −0.212 |
| `task/rcv_palm_to_cup` [m] | 0.166 | 0.156 | 0.160 | 0.173 |
| `contact/rcv_max` [N] | 0.003 | 0.003 | 0.004 | 0.000 |
| `task/rcv_grasped` | 0 | 0 | 0 | 0 |

The rim-hook penalty grew together with the grasp rate: the grasps the policy found are mostly thumb-over-rim grasps.

4. Per-step reward over the last 30 epochs: approach_src 1.24, approach_rcv 0.60, contact_src 0.73, grasp_src 0.54, orient_src 0.20, orient_rcv 0.19, close_src 0.19, tips_src 0.17, tips_rcv 0.11, close_rcv 0.07, rim_hook_pen −0.21, cup_push_pen −0.08, curl_far_pen −0.09, lift_src 0.016, contact_rcv 0.000; `reward/total` 3.54. Everything that needs both hands holding or a lifted cup was 0 for the whole run. `reward/orient_src` fell from 0.29 (epoch 300) to about 0.20 once the hand moved in close.

5. In the previous function the grasp and contact terms have no position condition, and the thumb-below-rim quality only scales the lift income. With the cup not lifted, that quality factor never mattered, so a rim-hook grasp earned the full contact (0.73) and grasp (0.54) income. Meanwhile the receiver hand earned approach 0.60 and orient 0.19 by hovering about 17 cm from its cup, with no term pulling it the rest of the way.

6. Physics blow-ups of the underactuated hand (mimic coupling error above 10 rad) happened in 10 epochs; each was a single environment that the environment terminated. None occurred after epoch 737.

7. Operator decisions this round: the round was extended past 600 epochs because the source grasp was still rising; it was ended at epoch 836 by the operator's rule "if only the grasp keeps rising while the cup is not lifted and the left hand stays stuck, change the reward". The operator's judgement from the previous round still stands: once both hands reliably approach and grasp their cups, the later stages are expected to follow.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
