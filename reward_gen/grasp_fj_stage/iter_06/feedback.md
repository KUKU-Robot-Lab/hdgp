## I can see from the robot that
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

## Feedback for improvement
Three things went right and must be kept.
- The grasp and the lift both work now. The palm comes against the cup, the fingers wrap around it, and the cup is carried well clear of the table instead of rising a few millimetres. Whatever pays for that should keep paying for it.
- The hand stays off the table for the whole round. The episode-ending floor condition never fired once and the deepest penetration was a fifth of what it was before. This requirement is met and must not be given back.
- The little finger now joins the grasp instead of trailing beside it.

The rest is what has to change.

The approach from the raised start has been abandoned, and that is the main failure. Almost every success in this round came from episodes that begin beside the cup. Far-start episodes did learn the approach for a while in the middle of the round and then lost it, and by the end they almost never complete it. Starting beside the cup is a training aid, not the task; the task is to come from the raised pose outside the table and grasp the cup. Reaching the cup from the raised start has to be worth enough that it is never cheaper to give up on it, and progress made while travelling towards the cup from far away must keep paying all the way in rather than only once the hand is already there.

The approach that does happen is made in the wrong way. Instead of carrying the hand to the cup's side while holding its start orientation and its open default pose, the arm turns the wrist so the hand arrives low and rotated. That reaches the cup but skips the shape the approach is supposed to have, which is why the approach almost never registers as complete even in episodes that go on to grasp. Arriving at the cup's side in the start orientation with the hand still open should be what pays; getting within reach by rotating the wrist should not.

The cup is carried tilted. It is gripped and lifted, but it leans noticeably while being moved, at times by about thirty degrees. It is never dropped, so nothing corrects this. Carrying the cup upright should be worth clearly more than carrying it at an angle, from the moment it leaves the table rather than only at the goal.

The hand must not rest on the table, and this round could not see when it did. The measure the environment used for the table looked only at the finger and thumb links and left the palm out of it, and the palm is the largest flat part of the hand. The recording shows the hand skimming the table with the palm while the arm turned the wrist to bring the hand to the cup, and none of that registered. The environment now measures the palm itself and gives its clearance above the table to the reward directly, so that distance no longer has to be guessed from the virtual palm point, which is up to 0.065 m away from the palm surface and moves as the wrist turns. The episode-ending floor check still looks only at the fingers, so keeping the palm off the table is entirely the reward's job: resting or sliding the palm on the surface has to cost more than the approach it buys.

One caution about the middle of the round. The far-start approach reached its best value around epoch 530 and collapsed over the following two hundred epochs, exactly as near-start success took off. A reward that lets the easy start dominate will keep producing this, so the far start needs its own reason to improve that does not disappear once the near start is succeeding.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.0767, 0.313, 0.307, 0.134, 0.15, 0.164, 0.25, 0.232, 0.223]  min 0 · mean 0.222 · max 0.619
contact/finger_index_at_success: [-1, 0.226, 0.86, 0.894, 0.917, 0.921, 0.873, 0.833, 0.891, 0.887]  min -1 · mean 0.694 · max 0.995
contact/finger_middle: [0, 0.0581, 0.301, 0.327, 0.143, 0.152, 0.177, 0.278, 0.234, 0.252]  min 0 · mean 0.228 · max 0.638
contact/finger_middle_at_success: [-1, 1, 0.503, 0.922, 0.899, 0.889, 0.83, 0.909, 0.885, 0.951]  min -1 · mean 0.716 · max 1
contact/finger_pinky: [0, 0.02, 0.0259, 0.0334, 0.0603, 0.0752, 0.156, 0.284, 0.271, 0.25]  min 0 · mean 0.141 · max 0.46
contact/finger_pinky_at_success: [-1, 0.95, 0.116, 0.0175, 0.317, 0.443, 0.959, 0.989, 0.987, 0.987]  min -1 · mean 0.512 · max 1
contact/finger_ring: [0, 0.0886, 0.329, 0.329, 0.144, 0.161, 0.182, 0.296, 0.264, 0.26]  min 0 · mean 0.25 · max 0.628
contact/finger_ring_at_success: [-1, 1, 0.96, 0.936, 0.937, 0.962, 0.939, 0.946, 0.945, 0.968]  min -1 · mean 0.784 · max 1
contact/finger_thumb: [0, 0.109, 0.35, 0.358, 0.159, 0.176, 0.199, 0.325, 0.279, 0.266]  min 0 · mean 0.267 · max 0.693
contact/finger_thumb_at_success: [-1, 1, 1, 1, 0.995, 0.999, 0.992, 0.997, 1, 0.999]  min -1 · mean 0.854 · max 1
contact/fingers_touching: [0, 0.352, 1.32, 1.35, 0.639, 0.715, 0.878, 1.43, 1.28, 1.25]  min 0 · mean 1.11 · max 2.58
contact/fingers_touching_at_success: [-1, 4.18, 3.44, 3.77, 4.07, 4.21, 4.59, 4.67, 4.71, 4.79]  min -1 · mean 3.85 · max 4.84
contact/link_force_mean: [0, 0.291, 1.01, 0.665, 0.333, 0.343, 0.635, 0.893, 0.647, 0.874]  min 0 · mean 0.683 · max 1.83
contact/links_touching: [0, 0.45, 1.67, 1.74, 0.809, 0.897, 1.05, 1.61, 1.47, 1.47]  min 0 · mean 1.35 · max 3.27
contact/links_touching_at_success: [-1, 5.27, 4.69, 4.89, 5.25, 5.28, 5.41, 5.25, 5.39, 5.64]  min -1 · mean 4.74 · max 6.32
contact/palm_touching: [0, 0.0466, 0.145, 0.329, 0.134, 0.148, 0.178, 0.286, 0.248, 0.235]  min 0 · mean 0.21 · max 0.573
contact/palm_touching_at_success: [-1, 1, 0.541, 0.962, 0.918, 0.868, 0.928, 0.944, 0.934, 0.928]  min -1 · mean 0.736 · max 1
ctrl/prev_ep_successes_mean: [0, 0.00122, 0.0112, 0.816, 2.69, 2.49, 2.35, 2.09, 1.69, 1.81]  min 0 · mean 1.44 · max 2.84
done/abnormal: [0, 0.000244, 0, 0, 0, 0, 0.000244, 0, 0.000732, 0]  min 0 · mean 3.86e-05 · max 0.00122
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.44e-06 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.94e-05 · max 0.00122
done/max_goals: [0, 0, 0, 0.000244, 0.000488, 0.000488, 0.0022, 0.00122, 0, 0]  min 0 · mean 0.000567 · max 0.00391
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.42e-06 · max 0.000488
done/tipped: [0, 0.000244, 0.00122, 0, 0, 0, 0, 0.000488, 0.000244, 0]  min 0 · mean 0.000495 · max 0.00366
episode_lengths/step: [40, 518, 659, 668, 441, 500, 517, 404, 571, 621]  min 40 · mean 534 · max 899
reward/action_rate_pen: [-0.00537, -0.00439, -0.00354, -0.00318, -0.00308, -0.00297, -0.00269, -0.00219, -0.00248, -0.00224]  min -0.00539 · mean -0.00302 · max -0.00191
reward/approach_keep: [0.00191, 0.000859, 0.00248, 0.00141, 0.00102, 0.0014, 0.000556, 0.00549, 0.00107, 0.000974]  min 0.000421 · mean 0.00183 · max 0.00596
reward/approach_reach: [0.00573, 0.00415, 0.006, 0.00205, 0.00238, 0.00399, 0.00262, 0.00934, 0.00262, 0.00266]  min 0.00113 · mean 0.00437 · max 0.0105
reward/arm_vel_pen: [-0.000288, -0.000585, -0.000946, -0.00107, -0.00103, -0.00137, -0.00164, -0.00206, -0.00224, -0.00144]  min -0.0078 · mean -0.00141 · max -0.000274
reward/cup_disturb_pen: [0, -0.00623, -0.012, -0.000997, -0.00115, -0.00379, -0.000785, -0.0132, -0.00102, -0.000519]  min -0.0723 · mean -0.0059 · max 0
reward/cup_tilt_pen: [0, -0.0128, -0.0602, -0.0661, -0.0114, -0.0109, -0.0119, -0.0151, -0.0175, -0.00925]  min -0.156 · mean -0.0269 · max 0
reward/early_contact_pen: [0, -4.88e-05, -0.00245, 0, -0.000586, -0.00303, -0.000186, -0.0112, -0.000391, 0]  min -0.0586 · mean -0.00474 · max 0
reward/envelope_quality: [0, 0.00144, 0.00884, 0.0163, 0.0142, 0.0147, 0.0182, 0.0304, 0.0256, 0.0301]  min 0 · mean 0.0186 · max 0.0535
reward/finger_curl: [0, 0.00479, 0.00973, 0.00803, 0.00678, 0.00646, 0.00805, 0.0136, 0.0109, 0.0134]  min 0 · mean 0.00928 · max 0.0206
reward/grasp_fingers: [0, 0.0026, 0.0128, 0.0112, 0.00946, 0.00991, 0.012, 0.0194, 0.0172, 0.0194]  min 0 · mean 0.0131 · max 0.0339
reward/grasp_pose: [0, 0.0109, 0.0185, 0.0136, 0.0109, 0.0107, 0.0136, 0.022, 0.0174, 0.0201]  min 0 · mean 0.0157 · max 0.0295
reward/grasp_thumb: [0, 0.00362, 0.0135, 0.0118, 0.00961, 0.0103, 0.0119, 0.019, 0.017, 0.0188]  min 0 · mean 0.0133 · max 0.0324
reward/grasp_wrap: [0, 0.00244, 0.0134, 0.0129, 0.0109, 0.0112, 0.0137, 0.0228, 0.0198, 0.0232]  min 0 · mean 0.015 · max 0.0409
reward/grip_squeeze: [0, 0.000483, 0.0028, 0.00252, 0.00214, 0.00222, 0.00284, 0.00486, 0.00437, 0.00516]  min 0 · mean 0.00316 · max 0.00905
reward/hold_still: [0, 5.04e-06, 0.000104, 0.000837, 0.000942, 0.00109, 0.00158, 0.00266, 0.0037, 0.00414]  min 0 · mean 0.00193 · max 0.00692
reward/lift_clear: [0, 0.00747, 0.0667, 0.249, 0.101, 0.101, 0.135, 0.189, 0.188, 0.181]  min 0 · mean 0.146 · max 0.403
reward/lift_goal: [0, 0.000971, 0.0113, 0.0747, 0.0587, 0.0664, 0.111, 0.205, 0.193, 0.194]  min 0 · mean 0.104 · max 0.281
reward/lift_height: [0, 0.0055, 0.0732, 0.456, 0.207, 0.201, 0.297, 0.385, 0.397, 0.37]  min 0 · mean 0.288 · max 0.77
reward/lift_hold: [0, 0.00692, 0.0292, 0.0642, 0.0254, 0.0258, 0.0343, 0.0479, 0.0478, 0.0465]  min 0 · mean 0.0383 · max 0.103
reward/light_contact: [0, 0.00143, 0.00561, 0.00405, 0.00338, 0.00356, 0.0042, 0.00673, 0.00607, 0.00671]  min 0 · mean 0.00481 · max 0.0116
reward/palm_contact: [0, 0.00302, 0.0104, 0.0197, 0.0167, 0.0168, 0.0211, 0.0357, 0.0288, 0.0351]  min 0 · mean 0.0216 · max 0.0578
reward/palm_low_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/palm_reach: [0, 0.0119, 0.0265, 0.0201, 0.016, 0.0155, 0.02, 0.0327, 0.0261, 0.0311]  min 0 · mean 0.0225 · max 0.0461
reward/pinky_join: [0, 0.000185, 0.000178, 0.00063, 0.00127, 0.0016, 0.00322, 0.0058, 0.00517, 0.00556]  min 0 · mean 0.0027 · max 0.0106
reward/success_bonus: [0, 0, 0, 0.0439, 0.0513, 0.0806, 0.0549, 0.0842, 0.0366, 0.0256]  min 0 · mean 0.0424 · max 0.154
reward/table_pen: [0, -0.00556, -0.00191, -0.000828, -0.00153, -2.66e-05, -9.92e-05, -0.039, 0, -1.51e-06]  min -0.0569 · mean -0.00419 · max 0
reward/thumb_curl: [0, 0.00325, 0.00502, 0.00302, 0.00254, 0.00236, 0.00322, 0.00559, 0.00447, 0.00554]  min 0 · mean 0.00404 · max 0.00856
reward/total: [0.00198, 0.042, 0.235, 0.944, 0.532, 0.565, 0.752, 1.06, 1.03, 1.03]  min -0.0107 · mean 0.724 · max 1.83
rewards/step: [0.249, 10.1, 145, 896, 473, 303, 457, 363, 544, 657]  min -1.98 · mean 386 · max 1.11e+03
task/lifted_frac: [0, 0.00537, 0.0845, 0.301, 0.134, 0.143, 0.173, 0.274, 0.239, 0.235]  min 0 · mean 0.199 · max 0.515
task/successes_mean: [0, 0.000244, 0.00171, 0.133, 0.175, 0.199, 0.256, 0.248, 0.152, 0.131]  min 0 · mean 0.148 · max 0.413
task/tilt_deg: [0.00108, 2.75, 10.6, 10.9, 3.11, 3.28, 3.71, 5.86, 5.31, 3.89]  min 0.000912 · mean 6.12 · max 23.9
task/tol: [0.112, 0.112, 0.112, 0.112, 0.101, 0.0911, 0.082, 0.0738, 0.0664, 0.0664]  min 0.0664 · mean 0.0909 · max 0.112
