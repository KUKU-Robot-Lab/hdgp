## I can see from the robot that
- The video shows the last checkpoint (epoch 2418), which tilts the cup with the palm while pushing down on the table. The wrap-and-lift behaviour that appeared during training (around epochs 1820-2020 and 2330-2390) is not in this checkpoint; it is described below from the training metrics only.
- The arm moves from the start pose to the cup within about 3 s with the hand fully open: the four fingers straight and together, the thumb stretched out ahead of them. The fingers stay open for the whole approach; the early curl of the previous round is gone.
- From about 3.5 s the open hand is low beside the cup, and it does not really press on the cup. The palm leans against the lower side of the cup and tilts the cup away from the hand, while the straight fingers reach past the bottom of the cup and push down on the table. The hand is supported on the table rather than on the cup. The fingers never bend around the cup.
- The hand keeps this pose until the end of the episode, holding the cup tilted a few degrees away from the hand. The cup does not fall over and is never lifted.
- Training metrics (the round was ended at epoch 2418 by the stuck rule on the last point: reach 1.00 at both ends of the 600-epoch window, grasp 0.01; rechecked at epoch 2798 with the same verdict):
  - Two behaviours alternate during training:
    - Tilt the cup and push on the table (epochs 1778-1808, 2137-2267, and from about epoch 2400 to the end): reach 0.96-1.00 of episodes, grasp 0.003-0.013, envelope and lift 0.003 or less. The palm touches the cup in 63-68% of steps, and the mean cup tilt is 4.7-6.0 degrees. Reward per step 0.44-0.50, of which palm_contact 0.23-0.27 and lift 0.023; cup_tilt_penalty -0.003 to -0.021, cup_push_penalty about -0.006, table_penalty 0.0000.
    - Wrap and lift (epochs 1828-1928 and 2327-2387): grasp (three or more fingers on the cup) 0.15-0.32 of episodes, with a peak of 0.69 at epoch 2341; palm plus all five fingers 0.03-0.12, peak 0.28 at epoch 1858; lift 0.03-0.11, peak 0.28. The cup is off the table in 0.5-5% of steps, and successes occur in about 0.1% of episodes (peak 1.4% at epoch 1970). At success 4.8-4.9 fingers were on the cup and the palm in 13-36% of cases.
  - In the wrap-and-lift phases the cup is pushed and tilted much more: cup_push_penalty -0.05 to -0.12, cup_tilt_penalty -0.06 to -0.11, mean tilt 6-8 degrees. Reach drops to 0.57-0.71 and palm_contact to 0.15-0.24. The reward per step is 0.15-0.35, lower than tilting the cup and pushing on the table, and each time training went back to that pose.
  - The table penalty stayed at 0.0000 in every phase even though the fingers push on the table in the video, and no episode ended on the hand-below-table check.
  - The lift term pays about the same in every phase (0.022-0.023 per step), including the tilting phases where the cup is off the table in only 0.01-0.03% of steps.
  - The finger-wrapping terms stay tiny in both behaviours: wrap_fingers 0.001-0.003, finger_curl_onto_cup 0.0006 or less, envelope 0.0002 or less per step.
  - finger_open 0.05-0.06 per step: the fingers are held open during the approach, as intended.
  - Palm-centre-to-cup gap averaged over all steps, approach included: 0.024-0.049 m. Episodes ending by tipping, falling or abnormal states: 0.05% or fewer.

## Feedback for improvement
- Keep: the fast approach with the hand open until the palm is on the cup, the palm turned toward the side of the cup, and the cup not being knocked over.
- The hand must not support itself on the table. In this round the straight fingers push down on the table while the palm tilts the cup, and that costs nothing: the table penalty only looks at how low the hand's link origins go, and those stay above the table while the finger surfaces rest on it. Before the cup is lifted, fingers or the hand coming down onto the table should cost, judged by how close the fingertips and finger links get to the table surface.
- Tilting the cup with the palm must not pay. Palm contact and lift are both collected while the palm holds the cup tilted a few degrees: small tilts are free, and the few millimetres the cup centre rises when it tilts are paid as lift. Palm contact should count only while the cup stays upright. Tilting the cup before it is grasped should start costing at small angles. Lift should pay only when the cup has actually left the table in a grasp.
- Wrapping and lifting must earn clearly more than any resting pose. When the policy closes the fingers and starts to lift, the cup slides and tilts and the palm partly leaves the cup; the movement penalties plus the lost palm and approach rewards outweigh what wrapping and lifting pay. Training found the wrap, the lift and even some successes, and then went back to resting against the cup. Closing the fingers around the cup body while the palm is on it, and then lifting, must pay clearly more per step than resting, including during the transition when the cup moves a little. Resting against the cup with straight fingers should earn less and less the longer it lasts.
- The finger-wrapping rewards are too small to matter (thousandths of a unit per step). Fingers bending around the cup body and touching it while the palm is on the cup need to be a large share of the reward. More fingers touching should be worth clearly more, with the thumb on the opposite side of the cup from the other fingers.
- The cup-movement penalties must tell a grasped cup being lifted apart from a cup being shoved or tipped by the palm. While several fingers are wrapped on the cup, motion and tilt that come from lifting it should not cost as if it were being pushed. Tilting or shoving the cup without a grasp should cost.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0, 0, 0.00122, 0, 0.00708, 0.000244, 0.0522, 0.00317, 0.000488]  min 0 · mean 0.00872 · max 0.211
contact/finger_index_at_success: [-1, -1, -1, -1, -1, -1, -1, 0.989, 0.99, 0.993]  min -1 · mean -0.228 · max 1
contact/finger_middle: [0, 0.000732, 0, 0.000977, 0, 0.00635, 0.000244, 0.0447, 0.00317, 0.000244]  min 0 · mean 0.00782 · max 0.132
contact/finger_middle_at_success: [-1, -1, -1, -1, -1, -1, -1, 0.945, 0.903, 0.925]  min -1 · mean -0.252 · max 1
contact/finger_pinky: [0, 0.00171, 0, 0.00342, 0, 0.00244, 0.00146, 0.0186, 0.00146, 0.000244]  min 0 · mean 0.0105 · max 0.165
contact/finger_pinky_at_success: [-1, -1, -1, -1, -1, -1, -1, 0.97, 0.973, 0.979]  min -1 · mean -0.232 · max 1
contact/finger_ring: [0, 0.00269, 0, 0.00317, 0, 0.00366, 0.000488, 0.0251, 0.00293, 0.000244]  min 0 · mean 0.00679 · max 0.125
contact/finger_ring_at_success: [-1, -1, -1, -1, -1, -1, -1, 0.97, 0.973, 0.979]  min -1 · mean -0.232 · max 1
contact/finger_thumb: [0, 0, 0.0144, 0.154, 0.0859, 0.0217, 0.0535, 0.0984, 0.0574, 0.0383]  min 0 · mean 0.0913 · max 0.664
contact/finger_thumb_at_success: [-1, -1, -1, -1, -1, -1, -1, 1, 1, 1]  min -1 · mean -0.224 · max 1
contact/fingers_touching: [0, 0.00513, 0.0144, 0.163, 0.0859, 0.0413, 0.0559, 0.239, 0.0681, 0.0396]  min 0 · mean 0.125 · max 0.691
contact/fingers_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, 4.87, 4.84, 4.88]  min -1 · mean 1.28 · max 5
contact/link_force_mean: [0, 0.000506, 0.000944, 0.0425, 0.0369, 0.0157, 0.00805, 0.0927, 0.0117, 0.003]  min 0 · mean 0.0367 · max 0.352
contact/links_touching: [0, 0.00513, 0.0144, 0.165, 0.0869, 0.0449, 0.0562, 0.262, 0.0696, 0.0398]  min 0 · mean 0.131 · max 0.741
contact/links_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, 5.65, 5.54, 5.6]  min -1 · mean 1.58 · max 6
contact/palm_touching: [0, 0.0581, 0, 0.237, 0.218, 0.601, 0.621, 0.665, 0.712, 0.764]  min 0 · mean 0.429 · max 0.81
contact/palm_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, 0.0838, 0.173, 0.36]  min -1 · mean -0.525 · max 0.36
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0, 0, 0, 0.0022, 0, 0]  min 0 · mean 0.000159 · max 0.00366
done/abnormal: [0, 0, 0, 0, 0, 0, 0, 0.000244, 0, 0]  min 0 · mean 0.000123 · max 0.00293
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.68e-08 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.68e-07 · max 0.000244
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.65e-06 · max 0.000488
done/tipped: [0, 0, 0, 0, 0.000244, 0.000488, 0, 0.000244, 0, 0]  min 0 · mean 0.000128 · max 0.00317
episode_lengths/step: [59.7, 890, 843, 857, 852, 779, 885, 612, 820, 876]  min 59.7 · mean 790 · max 898
reward/action_rate_penalty: [-0.0103, -0.00573, -0.0047, -0.00406, -0.00389, -0.00325, -0.00299, -0.00275, -0.00311, -0.00271]  min -0.0104 · mean -0.00393 · max -0.0022
reward/cup_push_penalty: [0, -0.000768, -0.000968, -0.0322, -0.0178, -0.0143, -0.00112, -0.0438, -0.00618, -0.00172]  min -0.239 · mean -0.0187 · max 0
reward/cup_tilt_penalty: [0, -0.00137, -0.000658, -0.018, -0.0143, -0.0202, -0.00254, -0.0527, -0.00443, -0.00345]  min -0.233 · mean -0.0167 · max 0
reward/envelope: [0, 0, 0, 0, 0, 5.63e-05, 0, 4.08e-05, 0, 4.92e-06]  min 0 · mean 2.74e-05 · max 0.000944
reward/finger_curl_onto_cup: [0, 0, 0, 0.000149, 7.61e-05, 0.0003, 0.000302, 0.000545, 0.000771, 0.000955]  min 0 · mean 0.000325 · max 0.00172
reward/finger_open: [0.0182, 0.0793, 0.0867, 0.0823, 0.087, 0.069, 0.0652, 0.0553, 0.0548, 0.0574]  min -0.0316 · mean 0.0681 · max 0.0943
reward/goal_coarse: [0, 0, 4.32e-08, 2.67e-06, 0, 0, 0, 4.82e-06, 0, 2.82e-08]  min 0 · mean 4.1e-06 · max 0.000252
reward/goal_fine: [0, 0, 5.95e-14, 7.22e-12, 0, 0, 0, 1.1e-08, 0, 1.73e-13]  min 0 · mean 9.75e-09 · max 1.57e-06
reward/hold_still: [0, 0, 2.2e-31, 1.1e-25, 0, 0, 0, 5.32e-15, 0, 2.18e-25]  min 0 · mean 1.55e-10 · max 5.23e-08
reward/in_tolerance: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.95e-06 · max 0.000732
reward/joint_vel_penalty: [-0.00236, -0.000914, -0.000615, -0.000405, -0.00245, -0.000471, -0.000301, -0.000689, -0.000399, -0.000372]  min -0.0048 · mean -0.000616 · max -0.000234
reward/lift: [1.45e-07, 0.00234, 0.000688, 0.0138, 0.012, 0.0217, 0.0204, 0.0237, 0.0235, 0.0271]  min 1.43e-07 · mean 0.0168 · max 0.0322
reward/palm_contact: [0, 0, 0, 0.0761, 0.0491, 0.178, 0.231, 0.239, 0.306, 0.286]  min 0 · mean 0.147 · max 0.361
reward/palm_gap: [0, 0, 4.05e-06, 0.00686, 0.00876, 0.00886, 0.0127, 0.0148, 0.0113, 0.015]  min 0 · mean 0.00934 · max 0.0374
reward/palm_orient: [0.00491, 0.00464, 0.0134, 0.018, 0.0212, 0.0213, 0.0231, 0.0238, 0.0234, 0.0244]  min 0.00132 · mean 0.0191 · max 0.0287
reward/reach_coarse: [0.0337, 0.0545, 0.0686, 0.0803, 0.0932, 0.0937, 0.0951, 0.0973, 0.0971, 0.0993]  min 0.0226 · mean 0.0874 · max 0.103
reward/reach_oriented: [2.87e-07, 6.73e-10, 0.00448, 0.0231, 0.0244, 0.0254, 0.0359, 0.0383, 0.0355, 0.0399]  min 0 · mean 0.0263 · max 0.0719
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.46e-06 · max 0.00406
reward/table_penalty: [0, 0, -1.79e-05, 0, 0, 0, 0, 0, 0, 0]  min -0.0102 · mean -7.69e-05 · max 0
reward/total: [0.044, 0.132, 0.167, 0.246, 0.258, 0.381, 0.478, 0.394, 0.54, 0.542]  min -0.0362 · mean 0.335 · max 0.62
reward/wrap_fingers: [0, 0, 0, 0.000271, 0.000235, 0.000397, 0.000669, 0.00172, 0.000784, 0.000558]  min 0 · mean 0.000705 · max 0.00594
reward/wrap_links: [0, 0, 0, 0.000105, 8.57e-05, 9.13e-05, 0.000257, 0.000404, 0.00029, 0.000208]  min 0 · mean 0.000205 · max 0.00101
rewards/step: [2.49, 128, 143, 155, 276, 341, 422, 165, 398, 434]  min -6.04 · mean 259 · max 464
task/lifted_frac: [0, 0, 0.000244, 0.000244, 0, 0, 0, 0.00391, 0, 0.000244]  min 0 · mean 0.00267 · max 0.0806
task/successes_mean: [0, 0, 0, 0, 0, 0, 0, 0.00146, 0, 0]  min 0 · mean 6.36e-05 · max 0.00293
task/tilt_deg: [0.00108, 0.556, 0.147, 3.45, 3.25, 5.59, 4.11, 6.55, 4.69, 5.92]  min 0.000912 · mean 3.99 · max 9.69
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112
