## I can see from the robot that
- The hand approaches the cup from the side, places the palm against the side of the cup and closes the fingers around the cup body, so the hand does surround the cup. The fingers mostly press strongly on the cup with their fingertips rather than lying along it.
- While the cup is still on the table it stays upright inside this grasp.
- While lifting, the hand and the cup tilt: the cup is carried strongly tilted — roughly 45 degrees or more from upright, with its opening pointing sideways — and it is held tilted near the goal height. The arm does not rotate to bring the cup back upright. The same cycle (grasp, tilted lift, reset after about 4-5 s) repeats in every episode shown.
- The thumb's placement is not visible from this camera angle.
- Training metrics: successes per episode peaked at about 3.7 and have stayed between about 2.2 and 3.5 while the success tolerance tightened in steps from 0.1125 m to 0.074 m (latest value; recent successes about 2.8); the lifted fraction rose from 0.63 to 0.80; the mean cup tilt rose from about 24 to 27-30 degrees; on average 4.6 of 5 fingers and the palm (0.77 of the time) touch the cup, but only about 5 of the 15 measured links touch, i.e. about one link per finger; the hold_still component stayed at about 0.

## Feedback for improvement
- Keep the grasp that surrounds the cup with the palm and the fingers — that part works.
- The cup must stay upright while it is lifted and held. The goal pose is upright and goal_dist includes the cup's orientation, so a tilted carry stops producing successes as the tolerance tightens. The arm should rotate as needed to keep the cup upright during lifting and holding; make uprightness count throughout lifting and holding, strongly enough that carrying the cup tilted is not worth it.
- Near the goal the cup is almost never held still (hold_still stays about 0); successes need the upright cup held steady at the goal for consecutive steps.
- The fingers mainly press with their fingertips (about one measured link per finger touches); the task asks for the links of every finger to touch the cup, so wrapping with more of each finger should be worth more while holding.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0.641, 0.809, 0.881, 0.861, 0.887, 0.916, 0.921, 0.929, 0.931, 0.932]  min 0.362 · mean 0.884 · max 0.943
contact/finger_middle: [0.66, 0.758, 0.876, 0.905, 0.904, 0.912, 0.937, 0.941, 0.948, 0.947]  min 0.344 · mean 0.892 · max 0.955
contact/finger_pinky: [0.495, 0.711, 0.787, 0.785, 0.756, 0.813, 0.857, 0.866, 0.89, 0.885]  min 0.138 · mean 0.8 · max 0.904
contact/finger_ring: [0.786, 0.772, 0.867, 0.883, 0.887, 0.92, 0.914, 0.933, 0.935, 0.935]  min 0.458 · mean 0.886 · max 0.945
contact/finger_thumb: [0.812, 0.873, 0.926, 0.925, 0.912, 0.929, 0.937, 0.944, 0.949, 0.944]  min 0.696 · mean 0.92 · max 0.956
contact/fingers_touching: [3.39, 3.92, 4.34, 4.36, 4.35, 4.49, 4.57, 4.61, 4.65, 4.64]  min 2.01 · mean 4.38 · max 4.7
contact/link_force_mean: [1.52, 3.5, 4.47, 4.45, 4.53, 4.41, 5.15, 4.71, 4.57, 4.53]  min 1.1 · mean 4.83 · max 508
contact/links_touching: [3.78, 4.68, 5.13, 5.18, 5.11, 5.21, 5.22, 5.11, 5.08, 5.08]  min 2.28 · mean 5.03 · max 5.36
contact/palm_touching: [0, 0.33, 0.569, 0.548, 0.615, 0.661, 0.775, 0.747, 0.758, 0.778]  min 0 · mean 0.627 · max 0.825
ctrl/prev_ep_successes_mean: [0, 8.14e-05, 0.00277, 2.12, 3.48, 3.5, 3.04, 3.21, 2.73, 2.29]  min 0 · mean 2.18 · max 3.67
done/abnormal: [0, 0.000488, 0.000407, 0.000488, 0.000244, 0.000326, 0.000163, 8.14e-05, 0.000244, 8.14e-05]  min 0 · mean 0.00023 · max 0.00179
done/fell: [8.14e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.71e-06 · max 0.000163
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.63e-07 · max 8.14e-05
done/max_goals: [0, 0, 0, 0.000977, 0.00326, 0.00277, 0.00106, 0.00138, 0.000732, 0.00114]  min 0 · mean 0.00104 · max 0.00374
done/out_xy: [8.14e-05, 8.14e-05, 0, 0, 0, 0, 8.14e-05, 0, 0, 0]  min 0 · mean 3.16e-05 · max 0.00057
done/tipped: [0.00806, 0.00293, 0.00163, 0.0013, 0.00138, 0.000814, 0.000732, 0.000244, 0.00138, 0.000977]  min 8.14e-05 · mean 0.00144 · max 0.0128
episode_lengths/step: [21.7, 171, 252, 282, 260, 290, 329, 345, 403, 429]  min 21.7 · mean 301 · max 446
reward/approach: [0.239, 0.496, 0.625, 0.617, 0.63, 0.654, 0.683, 0.682, 0.681, 0.676]  min 0.239 · mean 0.627 · max 0.697
reward/arm_rate_penalty: [-0.145, -0.122, -0.112, -0.101, -0.0786, -0.0657, -0.0634, -0.0554, -0.0582, -0.0591]  min -0.145 · mean -0.0815 · max -0.0542
reward/contact: [0.825, 1.03, 1.2, 1.21, 1.21, 1.27, 1.31, 1.34, 1.34, 1.32]  min 0.483 · mean 1.23 · max 1.36
reward/envelope: [0, 0.509, 0.99, 0.973, 1.1, 1.26, 1.43, 1.45, 1.44, 1.44]  min 0 · mean 1.13 · max 1.56
reward/finger_reach: [0.668, 0.759, 0.782, 0.754, 0.774, 0.796, 0.794, 0.796, 0.794, 0.793]  min 0.565 · mean 0.779 · max 0.804
reward/goal: [0, 0.0175, 0.0603, 0.581, 0.617, 0.755, 0.827, 0.931, 0.934, 0.938]  min 0 · mean 0.616 · max 1.1
reward/hand_rate_penalty: [-0.0668, -0.0452, -0.0317, -0.0282, -0.0234, -0.0213, -0.0189, -0.0165, -0.0157, -0.0145]  min -0.0668 · mean -0.0256 · max -0.0137
reward/hold_still: [0, 8.76e-07, 6.01e-06, 0.00026, 0.000332, 0.00049, 0.000585, 0.000949, 0.00113, 0.00113]  min 0 · mean 0.000535 · max 0.00175
reward/lift: [0.0255, 0.0656, 0.241, 1.8, 1.68, 1.99, 2.4, 2.57, 2.64, 2.63]  min 0.0134 · mean 1.73 · max 2.83
reward/link_cover: [0.243, 0.302, 0.33, 0.337, 0.33, 0.34, 0.341, 0.338, 0.333, 0.33]  min 0.144 · mean 0.326 · max 0.349
reward/push_penalty: [-0.527, -0.591, -0.631, -0.149, -0.172, -0.141, -0.121, -0.0881, -0.0864, -0.0907]  min -0.797 · mean -0.238 · max -0.0792
reward/squeeze: [0.154, 0.19, 0.221, 0.213, 0.205, 0.215, 0.212, 0.214, 0.219, 0.213]  min 0.0799 · mean 0.208 · max 0.236
reward/success_bonus: [0, 0, 0.000883, 0.148, 0.266, 0.217, 0.156, 0.137, 0.124, 0.128]  min 0 · mean 0.126 · max 0.33
reward/table_penalty: [0, -0.000436, 0, 0, -5.55e-05, 0, 0, 0, 0, 0]  min -0.00234 · mean -6.31e-05 · max 0
reward/tilt_penalty: [-0.198, -0.296, -0.162, -0.388, -0.391, -0.468, -0.48, -0.584, -0.591, -0.487]  min -0.639 · mean -0.427 · max -0.151
reward/total: [1.32, 2.71, 4.02, 6.45, 6.64, 7.35, 8.04, 8.27, 8.3, 8.36]  min 0.567 · mean 6.5 · max 8.9
reward/wrap: [0.108, 0.393, 0.509, 0.481, 0.508, 0.542, 0.56, 0.556, 0.551, 0.545]  min 0.108 · mean 0.503 · max 0.568
rewards/step: [-2.16, 418, 899, 1.61e+03, 1.65e+03, 1.98e+03, 2.44e+03, 2.64e+03, 3.2e+03, 3.48e+03]  min -2.16 · mean 2.02e+03 · max 3.58e+03
task/lifted_frac: [0, 0.0801, 0.161, 0.676, 0.6, 0.685, 0.747, 0.814, 0.814, 0.796]  min 0 · mean 0.585 · max 0.831
task/successes_mean: [0, 0.000244, 0.0124, 0.496, 0.52, 0.74, 0.698, 0.748, 0.802, 0.838]  min 0 · mean 0.505 · max 0.894
task/tilt_deg: [14.5, 19.1, 13.7, 23.7, 23.5, 26.4, 27, 30.4, 30.3, 27.3]  min 12 · mean 24.4 · max 31.6
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.101, 0.0911, 0.082, 0.082, 0.082]  min 0.0738 · mean 0.0984 · max 0.112
