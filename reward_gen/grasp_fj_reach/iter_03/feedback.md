## I can see from the robot that
- The video shows the last checkpoint (epoch 4423).
- The arm brings the hand toward the cup within about 3 s with the hand fully open: the four fingers straight and together, the thumb stretched out ahead of them. The hand ends up beside the upper half of the cup (on its left in the video), not low near the table.
- From about 3.5 s the hand is turned so that its thumb side faces the cup. From the camera the palm is seen edge-on and does not face the cup. The straight fingers hang down beside the cup, clear of the table, and the thumb is stretched out horizontally with its tip resting on the upper side of the cup near the rim. The palm stays a few centimetres away from the cup and never touches it, and the fingers never bend.
- The hand holds this pose, with only the thumb on the cup, until the end of the episode. The cup stays upright and in place and is never lifted. The hand does not touch the table.
- Training metrics (the round was ended at epoch 4423 by the stuck rule: reach at or above 0.9 across the last 600 epochs while grasp stayed at 0):
  - The round went through three phases:
    - Epochs 0-800: the hand stayed far from the cup (reach 0.00 of episodes, palm-centre-to-cup gap 0.25 m, reward 0.06 per step).
    - Epochs 800-2600: it hovered about 10-13 cm from the cup with the hand open (reach 0.20, gap 0.13 m, reward 0.14).
    - Epochs 2600-4423: it came closer (reach 0.81, gap 0.084 m, thumb contact in 5% of steps, reward 0.17).
  - At the end (training kept running while waiting for approval; rechecked at epoch 4973): reach 0.82-0.84 of episodes (1.00 at epoch 4423), grasp 0.00, palm plus all five fingers 0.00, lift 0.00, success 0. The thumb touches the cup in 36-50% of steps and is the only digit in contact; the palm 0.
  - Brief peaks during the round: grasp (three or more fingers on the cup) 0.24 at epoch 3138, palm plus all five fingers 0.03 at epoch 3481, lift 0.05 at epoch 4005; the cup was off the table in at most 0.3% of steps; no successes.
  - Around epoch 1000 the palm touched the cup in up to 12% of steps, but palm_contact never paid more than 0.001 per step during the whole round.
  - Reward components at the end (per step): finger_open 0.09 (its maximum is 0.1), reach_coarse 0.06, palm_orient 0.017, reach_oriented 0.011, palm_contact 0.0000, wrap_fingers 0.0000, lift 0; total 0.18. finger_open was more than half of the total reward from about epoch 800 to the end.
  - Penalties stayed near zero: table_penalty -0.0009 on average after epoch 2600 (its largest value was -0.10 at epoch 574), cup tilt 0.1-0.2 degrees, cup_tilt_penalty and cup_push_penalty -0.0007 or less.

## Feedback for improvement
- Keep: the hand open during the approach, the cup left upright and untouched during the approach, and the hand kept off the table.
- Hovering a few centimetres from the cup with an open hand is currently worth more than moving the palm onto the cup. The open-hand reward is paid only while the palm is away from the grasp position and does not fade over the episode, so it disappears exactly when the palm arrives. Meanwhile the approach rewards fade over the episode and require a strict orientation. Keeping the hand open must never be worth more than getting closer: that reward should not depend on staying away from the grasp position, and it should stay small compared with reaching the grasp position.
- Reaching the grasp position with the palm facing the cup must be clearly worth more than hovering at every point in the episode. Getting closer should keep paying as the palm approaches, even before the orientation is exact; the orientation had to be so exact that the palm-facing approach almost never paid.
- The palm has to face the cup. The hand turned so that only its thumb side faces the cup, with the palm sideways. Approaching thumb first, or resting only the thumb on the cup, should earn little.
- Palm contact must be reachable. When the palm did touch the cup early in training, the palm reward stayed at zero, because every gate had to hold at once: an exact orientation, a cup within a couple of degrees of upright, and the hand clear of the table. The palm on the side of the cup should pay once the palm faces the cup reasonably well and the cup is not being tipped; the strict conditions should shape the reward, not zero it.
- Keep the rest of the previous feedback, which this round could not test because the palm never reached the cup: fingers closing around the cup body with the palm on it must be worth much more than any resting pose, lift must pay only when the cup has left the table in a grasp, and the cup-movement penalties must be relaxed during a real grasp.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0, 0.0061, 0, 0, 0, 0, 0.000488, 0.000488, 0]  min 0 · mean 0.00171 · max 0.113
contact/finger_index_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_middle: [0, 0.000244, 0.000244, 0, 0, 0, 0, 0.000488, 0.000244, 0]  min 0 · mean 0.000402 · max 0.0256
contact/finger_middle_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_pinky: [0, 0, 0, 0.00635, 0.000488, 0.00195, 0, 0.000244, 0, 0]  min 0 · mean 0.00252 · max 0.428
contact/finger_pinky_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_ring: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000341 · max 0.0337
contact/finger_ring_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_thumb: [0, 0, 0, 0, 0, 0, 0.00635, 0.193, 0.0122, 0.335]  min 0 · mean 0.0591 · max 0.683
contact/finger_thumb_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/fingers_touching: [0, 0.000244, 0.00635, 0.00635, 0.000488, 0.00195, 0.00635, 0.194, 0.0129, 0.335]  min 0 · mean 0.0641 · max 0.683
contact/fingers_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/link_force_mean: [0, 1.25e-05, 0.000559, 0.000199, 1.64e-05, 4.89e-05, 0.000231, 0.0136, 0.000422, 0.0101]  min 0 · mean 0.00522 · max 0.893
contact/links_touching: [0, 0.000244, 0.00659, 0.00635, 0.000488, 0.00195, 0.00635, 0.199, 0.0129, 0.336]  min 0 · mean 0.0645 · max 0.685
contact/links_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/palm_touching: [0, 0, 0.0759, 0, 0, 0, 0, 0.000244, 0, 0.0137]  min 0 · mean 0.00132 · max 0.12
contact/palm_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/abnormal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000272 · max 0.0122
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.41e-07 · max 0.000244
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/tipped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.49e-05 · max 0.00684
episode_lengths/step: [57, 843, 880, 809, 890, 885, 884, 893, 898, 899]  min 57 · mean 820 · max 899
reward/action_rate_penalty: [-0.0103, -0.00421, -0.00245, -0.00276, -0.00152, -0.00122, -0.00118, -0.00109, -0.0008, -0.000709]  min -0.0104 · mean -0.00201 · max -0.000509
reward/cup_push_penalty: [0, -0.000191, -3.29e-05, 0, 0, 0, 0, -0.00827, -9.99e-05, 0]  min -0.441 · mean -0.00187 · max 0
reward/cup_tilt_penalty: [0, 0, -0.00233, -6.01e-05, 0, -2.8e-05, -0.000252, -0.00726, -0.00045, -0.000251]  min -0.627 · mean -0.00344 · max 0
reward/envelope: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.84e-08 · max 3.63e-05
reward/finger_curl_onto_cup: [7.93e-23, 3.8e-18, 3.12e-26, 1.07e-08, 6.4e-10, 9.02e-07, 1.89e-06, 4.78e-05, 3.06e-06, 1.09e-06]  min 0 · mean 5.44e-06 · max 0.000902
reward/finger_open: [0.0182, 0.0966, 0.0883, 0.097, 0.0962, 0.0965, 0.0956, 0.0731, 0.0888, 0.0951]  min -0.0866 · mean 0.0854 · max 0.0995
reward/goal_coarse: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.7e-08 · max 1.77e-05
reward/goal_fine: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.89e-12 · max 4.92e-09
reward/hold_still: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.77e-17 · max 6.58e-14
reward/in_tolerance: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/joint_vel_penalty: [-0.00236, -0.00034, -0.000256, -0.000232, -0.000167, -0.00017, -0.000169, -0.000145, -0.0001, -0.000101]  min -0.0104 · mean -0.00042 · max -8.08e-05
reward/lift: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.34e-07 · max 0.000453
reward/palm_contact: [0, 0, 0, 0, 0, 0, 0, 9.53e-06, 0, 1.37e-05]  min 0 · mean 2.91e-06 · max 0.000885
reward/palm_orient: [0.00483, 0.00308, 0.00395, 0.00871, 0.00818, 0.0115, 0.014, 0.0175, 0.0165, 0.0161]  min 0.000166 · mean 0.0105 · max 0.0181
reward/reach_coarse: [0.0272, 0.0243, 0.0475, 0.0502, 0.0539, 0.0571, 0.0585, 0.0599, 0.0601, 0.061]  min 0.00388 · mean 0.049 · max 0.0629
reward/reach_oriented: [0, 2.02e-11, 1.48e-10, 5.84e-09, 0, 1.75e-05, 0.00569, 0.0159, 0.0119, 0.00856]  min 0 · mean 0.00419 · max 0.0171
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/table_penalty: [0, 0, -0.000109, 0, -3.81e-05, 0, 0, -0.00661, 0, 0]  min -0.101 · mean -0.00093 · max 0
reward/total: [0.0375, 0.119, 0.135, 0.153, 0.157, 0.164, 0.172, 0.147, 0.176, 0.18]  min -0.986 · mean 0.141 · max 0.185
reward/wrap_fingers: [0, 0, 0, 0, 0, 0, 0, 7.05e-05, 0, 0]  min 0 · mean 6.64e-06 · max 0.0021
reward/wrap_links: [0, 0, 0, 0, 0, 5.58e-07, 1.12e-05, 0.00395, 0.00012, 0]  min 0 · mean 0.000353 · max 0.00942
rewards/step: [1.55, 87.8, 94.5, 118, 132, 121, 142, 140, 158, 160]  min -88.3 · mean 117 · max 162
task/lifted_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.67e-05 · max 0.00342
task/successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/tilt_deg: [0.00108, 0.00203, 0.151, 0.0112, 0.0031, 0.00636, 0.0125, 0.279, 0.0172, 0.303]  min 0.000912 · mean 0.129 · max 12.8
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112
