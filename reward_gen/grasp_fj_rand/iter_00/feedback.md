## I can see from the robot that
Round 1 of the random-placement track (`fj_rand_i00`) ran to epoch 2378 over 4.18 hours and ended at the hour cap. The reward is the unchanged round-9 reward from the fixed-placement track. The only change is the environment: the cup now appears anywhere in x 0.10-0.40 m, y -0.30-0.00 m every episode, and every episode starts far from the cup.

**Finding a cup that moves every episode was learned.** The approach latch went from zero at epoch 300 to 0.94 of far-start episodes by epoch 750. The palm-to-cup gap fell from 0.21 m to 0.04 m. Hand orientation passed 0.97 of the time and height 0.83, so turning the hand towards cups at the far corners is not what fails.

**The envelope grasp was learned on top of it.** The envelope latch jumped from zero to 0.76 between epochs 600 and 900. By the end, 3.2 digits and the palm (0.48 of the time) were in contact on average, 4.85 digits and the palm (0.41) at the moment of success, and all five digits were touching at success in almost every case. The recording shows the same: the hand arrives from the start pose, stops beside the cup with the palm facing it, and closes four fingers and the thumb around the lower body of the cup by about four seconds.

**Lifting barely happens, and after the grasp the hand moves down, not up.** This is the main failure. Over the last two hundred epochs the cup was airborne only 0.021 of the time, episodes that lifted at all were 0.07, mean lift height was 0.005 m, success was 0.0025 of episodes, and the tolerance never tightened from 0.1125. The lift fraction peaked at about 0.09 around epoch 1350 and drifted down over the following thousand epochs. In the recording the hand closes around the cup by about four seconds and then, instead of raising it, sinks steadily towards the table for the rest of the fifteen-second episode, pressing the wrapped cup down against the tabletop. The hand never starts upward. Consistent with this, the lowest point of the hand fell from 0.262 m to 0.249 m over the round, the table penalty grew from -0.028 to -0.033 and the palm-near-table penalty from -0.0003 to -0.0029.

**The reward explains why.** Over the last two hundred epochs, grasping the cup while it stands on the table paid about 0.70 per step. The largest terms were grasp pose 0.167, envelope quality 0.102, palm reach 0.098, grasp wrap 0.093, palm contact 0.090, finger and thumb grasp 0.130 combined, and finger curl 0.050. Lifting it added only lift hold 0.036, lift clearance 0.006 and goal closeness 0.008. Lifting also exposed the policy to more of the penalties that were already active: tilt -0.032, table -0.033 and arm speed -0.008 (the arm speed penalty grew from -0.003 at epoch 1000). Almost all of the income is available without ever lifting the cup.

**The approach latch settled at about 0.6 instead of 0.9.** After the envelope was learned, far-start episodes that fired the approach latch fell from 0.94 at epoch 750 to 0.59-0.66 from epoch 1050 onward, and the envelope latch fell with it to 0.55-0.60. The per-step condition logs cannot separate the cause, because most of every episode is spent already holding the cup. There is no per-placement breakdown. The one recorded episode completed the approach, so the recording does not show what the other forty percent do.

**Lifting motions are violent when they occur.** The 99th percentile of arm joint speed while the cup was airborne rose from 3.4 rad/s at epoch 750 to 6.0-7.1 rad/s at the end, and the action rate while airborne stayed around 0.21-0.30. Two separate windows (around epochs 1650 and 1950) show mean cup speed while airborne of 7.2 and 2.3 m/s against a normal 0.25 m/s, which means the cup was occasionally thrown. The cup tipped in only 0.0003 of steps, and mean cup tilt rose from 7.4 to 10.1 degrees over the round.

**Placement rejection behaved as designed.** 0.42 of resets resampled a placement to avoid the hand's shadow and the tabletop holes, and no reset ever failed to find a valid one.

The hand never rested on the table: the floor termination never fired, and palm-on-table stayed at 0.003.

## Feedback for improvement
Three things went right and must be kept.

- The policy finds the cup wherever it is placed. It approaches in the default open hand pose, with the palm facing the cup and the correct height, from a far start, and it turns the hand for cups at the far corners. The approach latch reached 0.94 of episodes under full placement randomisation.
- The grasp is a true envelope grasp: palm against the cup, all five digits in contact at the moment of success.
- The approach-then-envelope ordering held. The envelope was learned only after the approach, and both stayed on for fifteen hundred epochs.

The rest is what has to change.

**After the envelope, the hand must go up, and lifting must pay clearly more than holding the cup on the table.** This is the main failure. In the recording the hand closes around the cup and then sinks towards the table and presses the cup down instead of raising it. Holding the cup wrapped on the table earns about 0.70 per step, and lifting it adds about 0.05, while lifting also brings on tilt, table and arm-speed penalties. The policy therefore closes its hand and stays. Once the envelope is complete, the grasp terms should stop growing and should not by themselves be worth staying for. They should act as the entry ticket to lifting, not as the income. The income after the envelope latch should come from the cup's height above its starting position, rising steadily as it rises, and from carrying it towards the goal. Upward motion of the hand and cup right after the envelope completes should be rewarded immediately and densely, so that lifting is learned quickly. Moving the hand or cup downward towards the table while holding the cup should earn less than keeping it still, and clearly less than raising it. A policy that closes around the cup and never lifts it should end up clearly worse off than one that lifts it. The cup's starting height now differs every episode because the cup model differs, so lift should be measured relative to where the cup started, not to a fixed height.

**Lifting must be smooth, and smoothness must not be the reason not to lift.** When the cup is lifted, the arm moves at up to 6-7 rad/s and the cup is occasionally thrown. Penalise fast arm motion and cup speed while the cup is held in a way that shapes how it is lifted. The penalty must stay small against the lift income, so that a slow, steady lift is always better than no lift.

**The approach should keep firing near 0.9 after the grasp is learned.** It fell from 0.94 to about 0.6 once the envelope was learned, and the envelope latch fell with it. If some episodes are now closing on the cup without first completing the open-hand approach, that path should earn nothing from the grasp or lift terms. The approach income should also not become worthless once grasp income appears. Keep paying for reaching the approached pose at the same rate late in training as early.

One caution. The success count this round is too small (0.0025) to read anything from it, and the tolerance never moved, so do not loosen the success condition to make this round look better. The failure is that the cup is not lifted, not that success is too hard to reach once it is.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.00317, 0.00342, 0.385, 0.621, 0.666, 0.706, 0.672, 0.683, 0.694]  min 0 · mean 0.464 · max 0.725
contact/finger_index_at_success: [-1, -1, 0.783, 0.783, 0.756, 0.868, 0.838, 0.997, 1, 1]  min -1 · mean 0.605 · max 1
contact/finger_middle: [0.00146, 0.00342, 0.000244, 0.336, 0.594, 0.662, 0.692, 0.674, 0.696, 0.705]  min 0 · mean 0.46 · max 0.739
contact/finger_middle_at_success: [-1, -1, 0.955, 0.955, 0.912, 0.922, 0.74, 0.928, 0.885, 0.931]  min -1 · mean 0.63 · max 1
contact/finger_pinky: [0.00146, 0, 0, 0.252, 0.345, 0.474, 0.468, 0.477, 0.533, 0.515]  min 0 · mean 0.316 · max 0.608
contact/finger_pinky_at_success: [-1, -1, 0, 0, 0, 0.351, 0.77, 0.928, 0.93, 0.911]  min -1 · mean 0.297 · max 0.999
contact/finger_ring: [0.0083, 0.00171, 0.000244, 0.299, 0.529, 0.661, 0.652, 0.635, 0.63, 0.677]  min 0 · mean 0.432 · max 0.709
contact/finger_ring_at_success: [-1, -1, 1, 1, 0.952, 0.974, 0.848, 0.997, 1, 1]  min -1 · mean 0.681 · max 1
contact/finger_thumb: [0, 0.00635, 0.00439, 0.374, 0.615, 0.686, 0.727, 0.688, 0.708, 0.713]  min 0 · mean 0.476 · max 0.752
contact/finger_thumb_at_success: [-1, -1, 1, 1, 1, 1, 0.984, 0.948, 1, 1]  min -1 · mean 0.704 · max 1
contact/fingers_touching: [0.0112, 0.0146, 0.0083, 1.65, 2.7, 3.15, 3.25, 3.15, 3.25, 3.3]  min 0 · mean 2.15 · max 3.46
contact/fingers_touching_at_success: [-1, -1, 3.74, 3.74, 3.62, 4.12, 4.18, 4.8, 4.82, 4.84]  min -1 · mean 3.5 · max 5
contact/link_force_mean: [0.00415, 0.0113, 0.00236, 0.844, 1.57, 2.25, 2.26, 2.22, 2.23, 2.47]  min 0 · mean 1.49 · max 13.7
contact/links_touching: [0.0122, 0.0149, 0.00854, 1.9, 3.1, 3.83, 3.87, 3.76, 3.86, 3.94]  min 0 · mean 2.55 · max 4.21
contact/links_touching_at_success: [-1, -1, 3.74, 3.74, 3.62, 4.58, 4.92, 4.98, 5.04, 5.11]  min -1 · mean 3.79 · max 6.4
contact/palm_touching: [0, 0.000244, 0.203, 0.237, 0.319, 0.463, 0.493, 0.48, 0.488, 0.528]  min 0 · mean 0.314 · max 0.554
contact/palm_touching_at_success: [-1, -1, 0, 0, 0, 0.157, 0.342, 0.488, 0.383, 0.563]  min -1 · mean 0.117 · max 0.966
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0.000244, 0.00122, 0.00146, 0.00952, 0.000244, 0]  min 0 · mean 0.00191 · max 0.0359
done/abnormal: [0, 0, 0, 0.000488, 0.000732, 0.000977, 0.000488, 0.000977, 0.000732, 0]  min 0 · mean 0.000509 · max 0.00317
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.66e-07 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0.000244, 0]  min 0 · mean 2.15e-05 · max 0.000732
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.31e-07 · max 0.000244
done/out_xy: [0, 0.000244, 0, 0.000244, 0.000244, 0, 0.000244, 0, 0, 0.000732]  min 0 · mean 0.00022 · max 0.0022
done/tipped: [0, 0.000244, 0.000488, 0.00146, 0.000732, 0.000732, 0.000488, 0.000488, 0.000977, 0.000244]  min 0 · mean 0.000362 · max 0.00317
episode_lengths/step: [25.7, 854, 814, 575, 483, 412, 448, 419, 519, 486]  min 25.7 · mean 570 · max 899
reward/action_rate_pen: [-0.00266, -0.00223, -0.00124, -0.000845, -0.000694, -0.000649, -0.000499, -0.000483, -0.00045, -0.000452]  min -0.00269 · mean -0.00092 · max -0.000423
reward/approach_form: [0.0209, 0.0652, 0.0194, 0.0262, 0.0194, 0.0169, 0.0145, 0.0177, 0.0163, 0.0157]  min 0.0118 · mean 0.0226 · max 0.0879
reward/approach_keep: [0.00106, 0.00128, 0.003, 0.0074, 0.00367, 0.00222, 0.00192, 0.00267, 0.00283, 0.00261]  min 0.000103 · mean 0.00321 · max 0.0141
reward/approach_lock: [1.76e-14, 2.31e-08, 0.00373, 0.00498, 0.00284, 0.00143, 0.00166, 0.00212, 0.00186, 0.00182]  min 6.04e-22 · mean 0.00242 · max 0.0221
reward/approach_travel: [0.0147, 0.0208, 0.0131, 0.0279, 0.0161, 0.0111, 0.00966, 0.0123, 0.0124, 0.0117]  min 0.00781 · mean 0.0154 · max 0.053
reward/arm_vel_pen: [-0.000146, -0.000242, -0.0002, -0.000869, -0.00307, -0.00454, -0.00614, -0.00661, -0.00667, -0.00705]  min -0.00955 · mean -0.0039 · max -0.000136
reward/cup_disturb_pen: [-8.07e-06, -0.000466, -0.000133, -0.00289, -0.00717, -0.00408, -0.00532, -0.00344, -0.00423, -0.00315]  min -0.0117 · mean -0.00373 · max -1.84e-06
reward/cup_tilt_pen: [-0.00115, -0.00293, -0.00318, -0.018, -0.0285, -0.0299, -0.0261, -0.0321, -0.0368, -0.0211]  min -0.0444 · mean -0.0209 · max -0.000105
reward/early_contact_pen: [-0.00259, -0.0023, -0.000843, -0.00787, -0.0141, -0.00812, -0.0101, -0.0089, -0.0102, -0.00682]  min -0.0242 · mean -0.00847 · max 0
reward/envelope_quality: [0, 0, 0, 0.0377, 0.0597, 0.0985, 0.0977, 0.103, 0.104, 0.115]  min 0 · mean 0.0627 · max 0.125
reward/finger_curl: [0, 0, 0.00379, 0.0236, 0.04, 0.0468, 0.0466, 0.0499, 0.0509, 0.052]  min 0 · mean 0.0326 · max 0.0554
reward/goal_close: [0, 0, 0, 0.00749, 0.00624, 0.00844, 0.00974, 0.00855, 0.00758, 0.0116]  min 0 · mean 0.00591 · max 0.0125
reward/goal_in_tol: [0, 0, 0, 0, 0, 0, 0, 2.63e-09, 0, 8.83e-13]  min 0 · mean 4.37e-06 · max 0.000506
reward/goal_near: [0, 0, 0, 4.21e-05, 2.6e-05, 5.39e-05, 3.69e-05, 3.35e-05, 2.97e-05, 4.1e-05]  min 0 · mean 3.17e-05 · max 0.000595
reward/grasp_fingers: [0, 0, 3.05e-05, 0.0246, 0.0468, 0.0644, 0.0637, 0.0653, 0.0675, 0.0701]  min 0 · mean 0.0418 · max 0.0762
reward/grasp_pose: [0, 0, 0.139, 0.122, 0.154, 0.155, 0.155, 0.169, 0.172, 0.171]  min 0 · mean 0.134 · max 0.269
reward/grasp_thumb: [0, 0, 5.07e-05, 0.0249, 0.0459, 0.0619, 0.0623, 0.0636, 0.0656, 0.0675]  min 0 · mean 0.0407 · max 0.0729
reward/grasp_wrap: [0, 0, 2.48e-06, 0.0311, 0.0577, 0.0926, 0.0901, 0.0923, 0.0966, 0.1]  min 0 · mean 0.0579 · max 0.111
reward/grip_squeeze: [0, 0, 2.38e-06, 0.00729, 0.0129, 0.0175, 0.0167, 0.0179, 0.0186, 0.0192]  min 0 · mean 0.0113 · max 0.0205
reward/hold_still: [0, 0, 0, 1.39e-05, 7.42e-06, 8.49e-06, 9.39e-06, 7.95e-06, 6.82e-06, 1.12e-05]  min 0 · mean 6.7e-06 · max 3.9e-05
reward/lift_clear: [0, 0, 0, 0.00217, 0.00299, 0.00521, 0.00623, 0.00649, 0.00577, 0.0083]  min 0 · mean 0.00387 · max 0.01
reward/lift_hold: [0, 0, 0, 0.0183, 0.0261, 0.035, 0.0371, 0.0366, 0.0359, 0.0389]  min 0 · mean 0.0232 · max 0.0412
reward/light_contact: [0, 0, 2.08e-05, 0.00939, 0.0184, 0.0238, 0.0236, 0.0239, 0.0248, 0.0253]  min 0 · mean 0.0156 · max 0.0277
reward/palm_contact: [0, 0, 0.0135, 0.035, 0.0553, 0.0828, 0.0848, 0.0923, 0.0916, 0.101]  min 0 · mean 0.0555 · max 0.107
reward/palm_reach: [0, 0, 0.0801, 0.0665, 0.0883, 0.0899, 0.0914, 0.0992, 0.101, 0.1]  min 0 · mean 0.0769 · max 0.147
reward/palm_table_pen: [0, 0, 0, -8.23e-06, -0.000293, -0.00221, -0.00243, -0.00324, -0.00139, -0.00295]  min -0.00958 · mean -0.00121 · max 0
reward/pinky_join: [0, 0, 0, 0.00503, 0.00738, 0.0135, 0.0119, 0.0121, 0.0132, 0.0146]  min 0 · mean 0.00793 · max 0.0194
reward/pregrasp_shape: [0, 0, 0.000456, 0.00172, 0.00062, 0.000589, 0.00025, 0.000274, 0.000615, 0.000528]  min 0 · mean 0.000742 · max 0.00784
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.23e-05 · max 0.00732
reward/table_pen: [0, -0.000442, -0.0015, -0.015, -0.0316, -0.0298, -0.0458, -0.0385, -0.0323, -0.0424]  min -0.0499 · mean -0.0269 · max 0
reward/thumb_curl: [0, 0, 0.00493, 0.00968, 0.0183, 0.0181, 0.0188, 0.0209, 0.0207, 0.021]  min 0 · mean 0.0138 · max 0.0236
reward/total: [0.03, 0.0787, 0.274, 0.447, 0.596, 0.765, 0.748, 0.801, 0.818, 0.865]  min 0.0153 · mean 0.562 · max 0.925
rewards/step: [-1.31, 57.5, 191, 258, 244, 303, 286, 331, 432, 397]  min -6.84 · mean 279 · max 527
task/lifted_frac: [0, 0, 0, 0.00391, 0.00732, 0.0457, 0.0249, 0.0193, 0.0227, 0.0139]  min 0 · mean 0.014 · max 0.0562
task/successes_mean: [0, 0, 0, 0, 0, 0.000244, 0.000244, 0.0022, 0, 0]  min 0 · mean 0.000374 · max 0.00806
task/tilt_deg: [0.458, 0.548, 3.19, 5.45, 9.25, 9.92, 9.28, 10.1, 11.1, 8.37]  min 0.0796 · mean 7.01 · max 11.8
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112
