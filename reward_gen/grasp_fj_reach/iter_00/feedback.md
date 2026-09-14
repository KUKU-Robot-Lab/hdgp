## I can see from the robot that
- The arm leaves the raised start pose and reaches the cup within about 2-3 s; in almost every episode the palm reference point of that round (the palm link origin near the wrist, see the feedback) gets within 5 cm of the cup (98% of episodes).
- It does not approach the side of the cup with the palm turned toward it. During the approach (between about 1 s and 3 s) the hand rolls about the forearm: the fingers turn to point down and forward, and the palm ends up facing down and away from the cup instead of toward it.
- In this rolled pose the wrist end of the hand rests against the lower part of the cup on the side nearer the robot, and the straight fingers pass under and behind the cup, sticking out beyond its far side just above the table. The fingers never bend around the cup.
- It keeps this pose for the rest of the episode. At times it pushes the cup so that the cup tilts by roughly 20-25 degrees and then comes back upright; the cup is not knocked over. The cup is never lifted and no success is counted. After a reset the same rolled approach repeats.
- The thumb's placement cannot be seen clearly from this camera angle.
- Training metrics (training stopped at epoch 571): the contact sensor on the palm body registers cup contact in about 61% of steps (up from 21% over the last 200 epochs) even though the palm surface faces away from the cup; finger contact fell from about 0.09 to 0.03 fingers per step and only the thumb still touches occasionally; fewer than 1% of episodes ever had three or more fingers on the cup at once, and fewer than 1% had the palm and all five fingers on the cup together; the lifted fraction is 0. By component over the last 200 epochs: reach rose from about 0.56 to 0.78 per step, palm_contact from 0.13 to 0.41 and finger_wrap (finger links near the cup surface) from 0.14 to 0.26, while finger_contact and link_contact stayed near 0 and palm_orient stayed at about 0.10; the total reward rose from 0.96 to 1.53 while nothing beyond reaching improved.

## Feedback for improvement
- Environment correction for this round: `ctx.palm_pos` is now the centre of the palm (the palm_ee frame). In the previous round it was the palm link origin, which lies about 4 cm from the palm centre toward the wrist and about 3 cm behind the palm surface; `palm_normal` and `palm_side` are unchanged. The previous reward brought that point to the cup, which pulled the wrist end of the hand, not the palm centre, against the cup.
- Keep the fast move from the raised start pose to the cup.
- The approach must bring the palm surface to face the side of the cup with the fingers ready to go around the cup body, instead of rolling the hand so that the palm faces down and the fingers point under the cup. Being near the cup or touching it should only pay when the palm faces the cup; with the palm turned away, approaching and touching should earn little.
- Palm contact must mean the palm surface pressing on the cup. Right now any contact of the palm body counts, including the wrist end or the back of the hand while the palm faces away, and that is what the policy collects.
- Finger links merely being close to the cup surface must not pay on their own: straight fingers sliding along or behind the cup currently earn it. The fingers should be rewarded for bending around the cup body within the graspable band and actually touching it with their links — all five fingers, with the thumb on the opposite side of the cup.
- Rewards that can be collected just by resting next to the cup should not keep growing over the episode. Each next stage should be clearly worth more than staying at the previous one: a correctly oriented approach, then a closed wrap in contact, then lifting the cup upright, then holding it at the goal.
- Pushing or tilting the cup on the table before it is grasped should cost more than it does now; the resting hand pushes the cup and tilts it by 20-25 degrees.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.000244, 0.0035, 0.0161, 0.0262, 0.11, 0.045, 0.115, 0.0233, 0.000163]  min 0 · mean 0.0272 · max 0.12
contact/finger_index_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_middle: [0, 0.00122, 0.00317, 0.00716, 0.0447, 0.00765, 0.000488, 0.0212, 0.00244, 8.14e-05]  min 0 · mean 0.00571 · max 0.0458
contact/finger_middle_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_pinky: [0, 0.00675, 0.00106, 8.14e-05, 0.0179, 0.000244, 0, 0, 0, 0]  min 0 · mean 0.00552 · max 0.135
contact/finger_pinky_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_ring: [0, 0.0013, 0.00326, 0.00627, 0.0289, 0.00138, 0.000163, 0.00277, 0, 0]  min 0 · mean 0.00459 · max 0.0452
contact/finger_ring_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_thumb: [0, 0.000163, 0, 0.00057, 8.14e-05, 0.00448, 0.0013, 0.0955, 0.00171, 0.0333]  min 0 · mean 0.0106 · max 0.0971
contact/finger_thumb_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/fingers_touching: [0, 0.00968, 0.011, 0.0302, 0.118, 0.124, 0.047, 0.235, 0.0274, 0.0335]  min 0 · mean 0.0535 · max 0.236
contact/fingers_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/link_force_mean: [0, 0.0109, 0.00619, 0.015, 0.167, 0.103, 0.0152, 0.144, 0.0129, 0.00419]  min 0 · mean 0.0486 · max 3.73
contact/links_touching: [0, 0.0103, 0.0111, 0.0306, 0.138, 0.129, 0.0487, 0.242, 0.0285, 0.0335]  min 0 · mean 0.0565 · max 0.243
contact/links_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/palm_touching: [0, 8.14e-05, 0.0103, 0.0479, 0.08, 0.0388, 0.227, 0.107, 0.223, 0.487]  min 0 · mean 0.179 · max 0.623
contact/palm_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/abnormal: [0, 0, 0, 8.14e-05, 0, 0.000651, 0.000407, 0, 0.000407, 8.14e-05]  min 0 · mean 0.000211 · max 0.00244
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.54e-06 · max 0.000163
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.89e-07 · max 8.14e-05
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.39e-06 · max 0.000163
done/tipped: [0, 8.14e-05, 0.000407, 0.000977, 0.00057, 0.00122, 0.000326, 0.000651, 0.000244, 0.000407]  min 0 · mean 0.000475 · max 0.00317
episode_lengths/step: [46, 786, 757, 766, 744, 614, 614, 608, 584, 685]  min 46 · mean 643 · max 791
reward/action_rate_penalty: [-0.0103, -0.00941, -0.00809, -0.00778, -0.007, -0.00626, -0.00606, -0.00559, -0.00582, -0.00606]  min -0.0103 · mean -0.00708 · max -0.0053
reward/cup_push_penalty: [-0.000741, -0.00534, -0.0161, -0.0309, -0.0564, -0.0631, -0.0805, -0.0973, -0.0811, -0.0903]  min -0.113 · mean -0.0628 · max -0.00074
reward/finger_contact: [0, 0.000109, 0.000725, 0.00203, 0.00064, 0.00331, 0.00341, 0.0171, 0.00351, 0.0026]  min 0 · mean 0.0026 · max 0.0171
reward/finger_preshape: [0.177, 0.173, 0.181, 0.146, 0.139, 0.134, 0.112, 0.124, 0.11, 0.0773]  min 0.0731 · mean 0.119 · max 0.197
reward/finger_wrap: [2.33e-11, 0.00636, 0.0412, 0.0732, 0.0653, 0.158, 0.146, 0.184, 0.0955, 0.178]  min 1.39e-11 · mean 0.113 · max 0.261
reward/full_envelope: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/goal_coarse: [0, 0, 0, 0, 1.2e-06, 2.37e-06, 0, 3.43e-09, 0, 0]  min 0 · mean 4.83e-07 · max 1.18e-05
reward/goal_fine: [0, 0, 0, 0, 1.73e-10, 2.09e-10, 0, 1.02e-19, 0, 0]  min 0 · mean 1.68e-09 · max 4.72e-07
reward/hold_still: [0, 0, 0, 0, 9.18e-17, 4.02e-23, 0, 0, 0, 0]  min 0 · mean 3.76e-16 · max 1.2e-13
reward/in_tolerance: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.41e-07 · max 8.14e-05
reward/joint_vel_penalty: [-0.00468, -0.00405, -0.00371, -0.0033, -0.00356, -0.0038, -0.00403, -0.00544, -0.00383, -0.00315]  min -0.0296 · mean -0.00407 · max -0.00232
reward/lift: [5.02e-08, 0.000589, 0.00132, 0.00152, 0.0031, 0.00183, 0.00351, 0.00301, 0.00328, 0.00622]  min 4.73e-08 · mean 0.00354 · max 0.0102
reward/link_contact: [0, 1.82e-05, 0.000121, 0.000338, 0.000109, 0.000555, 0.000571, 0.00287, 0.000588, 0.000433]  min 0 · mean 0.000437 · max 0.00287
reward/palm_contact: [0, 0, 0.00602, 0.0321, 0.0481, 0.0202, 0.142, 0.0621, 0.141, 0.32]  min 0 · mean 0.113 · max 0.419
reward/palm_orient: [0.039, 0.063, 0.0801, 0.0975, 0.12, 0.16, 0.133, 0.09, 0.0838, 0.0875]  min 0.0337 · mean 0.105 · max 0.203
reward/reach: [0.0951, 0.18, 0.261, 0.365, 0.427, 0.493, 0.586, 0.589, 0.575, 0.742]  min 0.0925 · mean 0.51 · max 0.79
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/table_penalty: [0, 0, 0, 0, 0, -0.000332, -5.01e-06, -0.000425, -0.000108, 0]  min -0.000523 · mean -8.07e-05 · max 0
reward/thumb_opposition: [6.55e-16, 1.1e-06, 5.4e-05, 4.07e-05, 7.11e-05, 0.00039, 0.000598, 0.0228, 0.0042, 6.4e-05]  min 3.66e-16 · mean 0.00242 · max 0.0237
reward/total: [0.295, 0.405, 0.543, 0.676, 0.737, 0.897, 1.04, 0.985, 0.925, 1.31]  min 0.258 · mean 0.895 · max 1.54
rewards/step: [13, 272, 420, 453, 563, 525, 585, 607, 556, 683]  min 13 · mean 497 · max 883
task/lifted_frac: [0, 0, 0, 0, 0.000163, 0.000163, 0, 8.14e-05, 0, 0]  min 0 · mean 4.07e-05 · max 0.000407
task/successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/tilt_deg: [0.00104, 0.532, 1.3, 1.53, 3.08, 1.79, 2.62, 2.72, 2.48, 4.19]  min 0.000987 · mean 2.81 · max 6.71
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112
