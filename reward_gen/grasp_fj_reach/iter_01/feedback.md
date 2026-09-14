## I can see from the robot that
- The arm leaves the start pose outside the table edge and brings the hand next to the cup within about 2.5-3 s; the palm centre gets within 5 cm of the side of the cup in 99% of episodes.
- The fingers close long before the hand reaches the cup. They start to curl at about 0.5 s, while the hand is still travelling and clearly away from the cup, are fully curled by about 1 s, and never open again. The hand arrives beside the cup with its fingers already bent into a hook instead of open and ready to go around the cup.
- From about 3 s to 12.5 s the hand holds still beside the cup (on its left in the video). The palm does not touch the cup: there is a gap of a few centimetres between the palm and the cup, and bent finger links sit in that gap. The four fingers stay curled in front of the palm next to the lower half of the cup and do not go around the cup body. One digit near the wrist end of the hand rests against the side of the cup; by the contact sensors this is the thumb.
- The camera looks down at the hand from the front, so the hand appears to hang with its fingers pointing down. The orientation measures of this round's reward say otherwise: during the hold the palm faces the cup axis and the finger direction is close to horizontal, so the fingers most likely point horizontally toward the camera along the side of the cup. Which way the thumb points cannot be seen from this angle.
- In the last ~2 s of the episode the hand moves back away from the cup. The cup stays upright where it was placed; it is never lifted, pushed away or tipped.
- Training metrics (the round was ended at epoch 1256 by the stuck rule: reach at or above 0.9 of episodes for the last 600 epochs while grasp stayed near 0; rechecked at epoch 1419 with nothing changed; values are means of the last 150 epochs):
  - Episode funnel: reach 1.00; grasp (three or more fingers on the cup at the same time) 0.00, with a brief peak of 0.12 at epoch 633; palm together with all five fingers 0.00; lifted 0.00, with a brief peak of 0.06 at epoch 318; success 0.
  - Contacts per step: thumb 0.51 (0.41 six hundred epochs earlier); index, middle, ring and pinky 0.003 or less each; palm 0.000. Early in training (around epochs 200-300, while reaching was still unreliable) the palm touched the cup in up to 9% of steps and the index finger in about 4%; both have been zero since about epoch 400-500, when the oriented approach took over.
  - Palm-centre-to-cup gap averaged over all steps, approach included: 0.056 m, flat since epoch 800.
  - Reward components (per step): reach_oriented 0.18, reach_coarse 0.14, palm_orient 0.11, finger_close 0.075, finger_contact 0.034 (thumb only), link_contact 0.003, lift 0.004; palm_contact, envelope, thumb_opposition and all goal terms 0; cup_tilt_penalty -0.007, cup_push_penalty -0.008; total 0.52. reach_oriented and palm_orient have not changed since about epoch 800 and are close to the most that the episode-progress decay allows for a hand held still at the grasp spot.
  - Cup tilt 1.9 degrees on average; episodes ending by tipping, falling or abnormal states are 0.01% or fewer.

## Feedback for improvement
- Keep the fast move from the start pose to the cup, the palm turned toward the side of the cup, and the cup left upright and in place during the approach.
- Hovering beside the cup is currently the best the policy finds. With the palm a few centimetres from the cup and facing it, the approach and orientation rewards are already close to their maximum, and touching the cup adds little by comparison. Being oriented near the cup should earn only a small share; the grasp position should require the palm surface actually pressing on the side of the cup, and the palm touching the cup should be clearly worth more than hovering next to it.
- The fingers must stay open during the approach and arrive open. Right now they are already closed long before the hand gets near the cup, and closing them pays as soon as the palm is near the cup and oriented even when they touch nothing. The curled fingers then end up between the palm and the cup and can no longer go around it. While the palm is not yet on the cup, keeping the fingers extended and spread should be rewarded and closing them should cost. Closing should pay only once the palm is on the cup, and then for fingers closing onto the cup and touching it around the cup body, not for closing on air.
- A single thumb resting on the cup should not collect a meaningful contact reward. Contact should pay more the more fingers touch the cup together with the palm, with the thumb on the opposite side of the cup from the other fingers; the palm plus three or more fingers should be clearly worth more than one finger.
- Each stage should be clearly worth more than staying at the previous one: the palm pressed on the cup above hovering oriented, the palm plus several wrapped fingers above the palm alone, and the cup lifted upright above a wrapped cup still on the table.
- Pressing the palm on the cup and wrapping it will move the cup slightly on the table. That small movement should not cost so much that the policy avoids touching the cup; knocking the cup over or pushing it far away should still cost.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.000977, 0.00659, 0.00293, 0.00244, 0, 0, 0, 0, 0]  min 0 · mean 0.00479 · max 0.0796
contact/finger_index_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_middle: [0, 0.000488, 0.00171, 0.000732, 0.00122, 0, 0, 0.000244, 0, 0]  min 0 · mean 0.001 · max 0.0125
contact/finger_middle_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_pinky: [0, 0, 0.000488, 0, 0.0168, 0.00146, 0.000732, 0, 0.00195, 0.000732]  min 0 · mean 0.00209 · max 0.0481
contact/finger_pinky_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_ring: [0, 0, 0.000488, 0.000488, 0.00659, 0.000244, 0.000732, 0, 0.000488, 0.000244]  min 0 · mean 0.00169 · max 0.0232
contact/finger_ring_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_thumb: [0, 0.000488, 0.00415, 0.0142, 0.074, 0.135, 0.445, 0.592, 0.532, 0.544]  min 0 · mean 0.281 · max 0.656
contact/finger_thumb_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/fingers_touching: [0, 0.00195, 0.0134, 0.0183, 0.101, 0.137, 0.446, 0.592, 0.534, 0.545]  min 0 · mean 0.291 · max 0.661
contact/fingers_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/link_force_mean: [0, 0.000425, 0.00463, 0.00534, 0.0162, 0.0099, 0.0216, 0.0317, 0.0322, 0.0314]  min 0 · mean 0.0211 · max 0.187
contact/links_touching: [0, 0.0022, 0.0139, 0.0186, 0.103, 0.137, 0.447, 0.594, 0.536, 0.546]  min 0 · mean 0.292 · max 0.662
contact/links_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/palm_touching: [0, 0.000244, 0, 0, 0.000244, 0, 0, 0, 0, 0]  min 0 · mean 0.00121 · max 0.0906
contact/palm_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/abnormal: [0, 0.000244, 0, 0, 0.000488, 0, 0, 0, 0, 0]  min 0 · mean 6.13e-05 · max 0.00171
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.11e-07 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.21e-06 · max 0.000488
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.81e-07 · max 0.000244
done/tipped: [0, 0, 0, 0.000244, 0.000488, 0, 0, 0, 0, 0]  min 0 · mean 0.000209 · max 0.00757
episode_lengths/step: [48, 594, 715, 786, 798, 826, 871, 889, 881, 876]  min 48 · mean 797 · max 890
reward/action_rate_penalty: [-0.0103, -0.00907, -0.00874, -0.008, -0.00675, -0.00574, -0.0051, -0.00448, -0.0039, -0.00353]  min -0.0103 · mean -0.0062 · max -0.00334
reward/cup_push_penalty: [0, -0.0045, -0.00772, -0.00578, -0.022, -0.00593, -0.003, -0.0017, -0.00406, -0.00994]  min -0.133 · mean -0.0159 · max 0
reward/cup_tilt_penalty: [0, -0.00194, -0.00712, -0.0055, -0.0241, -0.00859, -0.00385, -0.0103, -0.00581, -0.0102]  min -0.303 · mean -0.0174 · max 0
reward/envelope: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/finger_close: [5.5e-34, 1.27e-05, 0.000202, 0.00444, 0.0311, 0.0449, 0.0621, 0.0715, 0.0757, 0.0757]  min 6.03e-44 · mean 0.0413 · max 0.0791
reward/finger_contact: [0, 1.63e-05, 6.4e-05, 0.000495, 0.00443, 0.00787, 0.0284, 0.0393, 0.0354, 0.0357]  min 0 · mean 0.0179 · max 0.0422
reward/goal_coarse: [0, 0, 0, 0, 3.18e-06, 0, 0, 0, 0, 0]  min 0 · mean 3.93e-08 · max 9.75e-06
reward/goal_fine: [0, 0, 0, 0, 1.22e-09, 0, 0, 0, 0, 0]  min 0 · mean 8.37e-11 · max 5.56e-08
reward/hold_still: [0, 0, 0, 0, 2.2e-43, 0, 0, 0, 0, 0]  min 0 · mean 7.23e-17 · max 6.45e-14
reward/in_tolerance: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/joint_vel_penalty: [-0.00236, -0.00196, -0.00239, -0.00172, -0.00376, -0.00205, -0.00085, -0.000968, -0.000958, -0.00103]  min -0.143 · mean -0.00171 · max -0.000834
reward/lift: [4.66e-08, 3.58e-05, 0.000225, 0.000278, 0.000943, 0.00112, 0.00285, 0.00529, 0.004, 0.00391]  min 4.66e-08 · mean 0.00245 · max 0.00686
reward/link_contact: [0, 1.63e-06, 6.4e-06, 4.95e-05, 0.000446, 0.000788, 0.00284, 0.00394, 0.00355, 0.00358]  min 0 · mean 0.0018 · max 0.00423
reward/palm_contact: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/palm_orient: [0.0271, 0.0456, 0.0371, 0.0688, 0.0898, 0.106, 0.115, 0.118, 0.11, 0.107]  min 0.00497 · mean 0.0875 · max 0.119
reward/reach_coarse: [0.0564, 0.0778, 0.0924, 0.125, 0.122, 0.134, 0.137, 0.14, 0.137, 0.137]  min 0.011 · mean 0.123 · max 0.149
reward/reach_oriented: [1.46e-07, 0.000641, 0.00162, 0.023, 0.0898, 0.134, 0.181, 0.19, 0.191, 0.182]  min 1.09e-08 · mean 0.112 · max 0.198
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/table_penalty: [0, -0.000147, -0.00225, 0, 0, -0.000209, 0, 0, 0, 0]  min -0.00434 · mean -0.000123 · max 0
reward/thumb_opposition: [0, 0, 0, 1.16e-08, 4.29e-07, 0, 5.45e-08, 0, 7.06e-08, 0]  min 0 · mean 2.13e-07 · max 9.86e-06
reward/total: [0.0709, 0.106, 0.103, 0.201, 0.282, 0.406, 0.516, 0.55, 0.541, 0.52]  min -0.293 · mean 0.344 · max 0.556
rewards/step: [3.06, -1.53, 63, 108, 224, 324, 416, 469, 466, 457]  min -13.1 · mean 275 · max 474
task/lifted_frac: [0, 0, 0, 0, 0.000244, 0, 0, 0, 0, 0]  min 0 · mean 1.09e-05 · max 0.000732
task/successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/tilt_deg: [0.00108, 0.0628, 0.235, 0.252, 0.985, 0.783, 1.57, 2.87, 2.19, 2.2]  min 0.000912 · mean 1.6 · max 7.38
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112
