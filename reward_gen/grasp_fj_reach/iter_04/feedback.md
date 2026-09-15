## I can see from the robot that
- The video shows the checkpoint at epoch 4000. Training was extended from epoch 1302 to epoch 4000 on the user's decision, with the reward unchanged.
- The arm brings the open hand to the cup within about 3 s, with the palm turned toward the cup, and the fingers start to close around the cup at about 3 s.
- By about 3.5-4 s the hand closes on the cup, but only the thumb and the middle finger wrap it with their inner surfaces. The index and ring fingers do not wrap; they rest against the cup with the backs of the fingers. The pinky sticks out below the hand, away from the cup. The cup is still upright on the table.
- From about 4.5 s the hand lifts the cup and carries it up and to the side (to the right in the video, toward the other arm). By about 9 s the arm is stretched out almost horizontally, holding the cup upright high above the table and well to the side of where it started. The hand keeps it there until the end of the episode; the cup is neither dropped nor tipped.
- The goal is 12 cm straight above the cup's starting position. The cup is carried well past it, much higher and to the side, and does not stay near the goal.
- Training metrics (epoch 4000; round end after the extension; verdict advance(stuck:envelope)):
  - Episode funnel, mean of the last 150 epochs: reach 1.00, grasp (three or more fingers on the cup) 0.99, palm plus all five fingers 0.00, lift 0.89, success 0.06 (peak 0.29 at epoch 2366). Mean successes per episode 0.29, peak 0.76.
  - The cup is off the table in 45-50% of steps, up from 0.10 at epoch 2000.
  - The grasp formed between epochs 1800 and 2050: grasp went from 0.30 to 0.90 and lift from 0.01 to 0.56. Lift reached 0.95 by epoch 3050.
  - Contacts per step: thumb 0.75, index 0.72, middle 0.73, ring 0.68, pinky 0.00, palm 0.01. At a success, 3.9 fingers were on the cup and the palm in 2-12% of cases; grasp quality at success 0.34-0.42.
  - Reward per step, last 150 epochs: lift 2.24, wrap_fingers 1.16, finger_curl_onto_cup about 0.23, wrap_links about 0.23, palm_contact about 0.21 (credited while grasping), goal_coarse 0.08, goal_fine 0.0002, in_tolerance 0.005, success_bonus 0.008; total 4.3.
  - Penalties: cup_push_penalty -0.40, which grew from -0.09 at epoch 2000 to -0.41 as the lifted fraction rose; cup_tilt_penalty -0.03, mean tilt 3-4 degrees.
  - The success count swings strongly: mean successes 0.47 at epoch 3000, 0.07 at epoch 3250 and 0.28 at epoch 4000.
  - Rechecked at epoch 5231, with training still running while waiting for approval. Lift reached 0.99 of episodes and the cup is off the table in 67-72% of steps. Successes fell to 0.015-0.05 of episodes and 0.02-0.08 per episode on average, down from 0.29 at epoch 4000. cup_push_penalty grew to -0.62, the lift reward is 3.4 and the total 5.4. The cup is lifted more often and carried further away, and successes became rarer.

## Feedback for improvement
- Keep the approach and the lift. The policy now closes the hand on the cup and lifts it in almost every episode without dropping or tipping it; change as little of that as possible.
- The grasp is not a real envelope yet. Only the thumb and the middle finger wrap the cup with their inner surfaces. The index and ring fingers rest on the cup with the backs of the fingers, the pinky never joins, and the palm is off the cup (palm contact in 1% of steps). Finger contact currently counts whichever side of a finger touches the cup, so the back of a finger earns the same wrap reward as a real wrap. A finger should count as wrapping only when it curls around the cup with its inner surface, with the cup on the inside of the finger's bend. A finger pressing on the cup with its back should earn nothing for wrapping. A grasp with the palm and all five digits wrapped this way should be worth clearly more than the same grasp without them, including while lifting and holding at the goal.
- The cup must go to the goal and stay there, not be carried away. After lifting, the hand carries the cup far above and to the side of the goal and holds it there with the arm stretched out. Once the cup is off the table, being close to the goal must be worth clearly more than being lifted anywhere else. The lift reward should stop growing at the goal height, the goal reward should dominate it and rise sharply as the cup gets close, and moving the lifted cup away from the goal should cost more the further it goes.
- The penalty on sideways movement is the wrong signal for a carried cup. It stops growing after a short distance, so carrying the cup far costs no more than carrying it a little, and it says nothing about where the goal is. Steer the lifted cup by its distance to the goal instead. Keep punishing sliding or shoving the cup on the table before it is grasped.
- Success needs the cup held still at the goal. Once the cup is near the goal, staying still and upright there must pay much more than moving around; the successes seen so far come and go between checks.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0, 0.000244, 0.00146, 0.477, 0.653, 0.603, 0.688, 0.715, 0.756]  min 0 · mean 0.436 · max 0.784
contact/finger_index_at_success: [-1, -1, -1, -1, 0.902, 0.824, 0.833, 0.993, 0.918, 0.851]  min -1 · mean 0.197 · max 1
contact/finger_middle: [0, 0.00391, 0, 0.000732, 0.569, 0.677, 0.714, 0.707, 0.732, 0.762]  min 0 · mean 0.456 · max 0.792
contact/finger_middle_at_success: [-1, -1, -1, -1, 0.87, 0.975, 0.952, 0.959, 0.996, 0.902]  min -1 · mean 0.265 · max 1
contact/finger_pinky: [0, 0, 0, 0, 0, 0, 0.000244, 0, 0, 0]  min 0 · mean 0.00044 · max 0.0444
contact/finger_pinky_at_success: [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]  min -1 · mean -0.359 · max 0
contact/finger_ring: [0, 0.00439, 0.000488, 0.00464, 0.493, 0.647, 0.648, 0.669, 0.69, 0.731]  min 0 · mean 0.431 · max 0.769
contact/finger_ring_at_success: [-1, -1, -1, -1, 0.987, 0.739, 0.887, 0.897, 0.996, 0.838]  min -1 · mean 0.236 · max 1
contact/finger_thumb: [0, 0.349, 0.381, 0.0962, 0.604, 0.701, 0.741, 0.72, 0.741, 0.772]  min 0 · mean 0.513 · max 0.81
contact/finger_thumb_at_success: [-1, -1, -1, -1, 0.999, 1, 1, 1, 1, 1]  min -1 · mean 0.282 · max 1
contact/fingers_touching: [0, 0.357, 0.381, 0.103, 2.14, 2.68, 2.71, 2.78, 2.88, 3.02]  min 0 · mean 1.84 · max 3.13
contact/fingers_touching_at_success: [-1, -1, -1, -1, 3.76, 3.54, 3.67, 3.85, 3.91, 3.59]  min -1 · mean 2.06 · max 4
contact/link_force_mean: [0, 0.0394, 0.0182, 0.0119, 2.41, 2.31, 2.23, 2.14, 2.62, 2.9]  min 0 · mean 1.6 · max 4.52
contact/links_touching: [0, 0.363, 0.382, 0.104, 2.59, 3.4, 3.45, 3.25, 3.63, 4.05]  min 0 · mean 2.26 · max 4.15
contact/links_touching_at_success: [-1, -1, -1, -1, 4.26, 4.49, 4.29, 5.2, 5.54, 4.4]  min -1 · mean 2.79 · max 6.51
contact/palm_touching: [0, 0.0156, 0.151, 0.189, 0.0347, 0.00879, 0.00952, 0.00708, 0.0105, 0.0151]  min 0 · mean 0.0849 · max 0.622
contact/palm_touching_at_success: [-1, -1, -1, -1, 0.2, 0.000118, 0.0613, 0.0253, 0.468, 0.271]  min -1 · mean -0.273 · max 0.671
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0.159, 0.131, 0.117, 0.183, 0.149, 0]  min 0 · mean 0.111 · max 0.761
done/abnormal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.76e-06 · max 0.000732
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.3e-06 · max 0.000488
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.66e-08 · max 0.000244
done/max_goals: [0, 0, 0, 0, 0.000244, 0, 0, 0, 0, 0]  min 0 · mean 2.52e-05 · max 0.000977
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.91e-06 · max 0.000244
done/tipped: [0, 0, 0, 0, 0.000732, 0.000488, 0, 0, 0.000244, 0]  min 0 · mean 9.84e-05 · max 0.00806
episode_lengths/step: [41, 877, 891, 886, 711, 862, 879, 869, 879, 887]  min 41 · mean 859 · max 897
reward/action_rate_penalty: [-0.0103, -0.00449, -0.00487, -0.00446, -0.00373, -0.00302, -0.00241, -0.00177, -0.00142, -0.00122]  min -0.0103 · mean -0.00328 · max -0.00108
reward/cup_push_penalty: [0, -0.00572, -0.000576, -0.00444, -0.0772, -0.086, -0.475, -0.342, -0.564, -0.617]  min -0.677 · mean -0.242 · max 0
reward/cup_tilt_penalty: [0, -0.0371, -0.0167, -0.0126, -0.163, -0.0532, -0.0672, -0.0252, -0.0201, -0.0208]  min -0.421 · mean -0.0514 · max 0
reward/envelope: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.66e-06 · max 0.000431
reward/finger_curl_onto_cup: [7.12e-31, 0.00237, 0.0222, 0.02, 0.117, 0.216, 0.212, 0.247, 0.254, 0.263]  min 5.47e-32 · mean 0.152 · max 0.276
reward/finger_open: [0.00721, 0.027, 0.0234, 0.0248, 0.0139, 0.00962, 0.00769, 0.00775, 0.00725, 0.0061]  min -0.000309 · mean 0.0131 · max 0.0305
reward/goal_coarse: [0, 1.95e-05, 0, 0, 0.0202, 0.0358, 0.0614, 0.0842, 0.0538, 0.0529]  min 0 · mean 0.0357 · max 0.106
reward/goal_fine: [0, 9.52e-09, 0, 0, 4.77e-05, 8.34e-05, 9.96e-05, 0.000223, 5.09e-05, 2.02e-05]  min 0 · mean 7.23e-05 · max 0.000372
reward/hold_still: [0, 3.36e-12, 0, 0, 9.05e-06, 7.55e-06, 5.63e-06, 3.39e-05, 3.7e-06, 1.86e-07]  min 0 · mean 7.87e-06 · max 6.17e-05
reward/in_tolerance: [0, 0, 0, 0, 0.00221, 0.0013, 0.00137, 0.0074, 0.000708, 0]  min 0 · mean 0.00203 · max 0.0219
reward/joint_vel_penalty: [-0.00236, -0.000561, -0.00084, -0.000829, -0.00128, -0.000927, -0.00104, -0.000855, -0.000921, -0.000969]  min -0.0883 · mean -0.00103 · max -0.000506
reward/lift: [0, 0.000438, 0, 0, 0.354, 0.65, 1.67, 2.19, 2.78, 2.91]  min 0 · mean 1.22 · max 3.73
reward/palm_contact: [0, 0.00273, 0.0341, 0.0461, 0.105, 0.201, 0.182, 0.236, 0.247, 0.255]  min 0 · mean 0.156 · max 0.281
reward/palm_orient: [0.00777, 0.0357, 0.039, 0.0369, 0.0424, 0.0554, 0.055, 0.0573, 0.0587, 0.0607]  min 0.00228 · mean 0.0472 · max 0.0633
reward/reach_facing: [2.21e-08, 0.103, 0.177, 0.142, 0.168, 0.243, 0.249, 0.261, 0.268, 0.288]  min 9.3e-09 · mean 0.204 · max 0.297
reward/reach_far: [0.0613, 0.121, 0.122, 0.129, 0.139, 0.163, 0.16, 0.162, 0.166, 0.171]  min 0.0349 · mean 0.145 · max 0.174
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0.03, 0.00595, 0]  min 0 · mean 0.00291 · max 0.0504
reward/table_penalty: [0, -0.000474, -0.00192, -0.000546, -0.0157, -0.0182, -0.00192, -0.00326, -0.000362, -0.00207]  min -0.0913 · mean -0.00511 · max 0
reward/total: [0.0636, 0.252, 0.408, 0.38, 1.21, 2.46, 3.07, 4.24, 4.65, 4.8]  min -0.446 · mean 2.43 · max 5.8
reward/wrap_fingers: [0, 0.000424, 0, 3.75e-05, 0.39, 0.848, 0.808, 1.1, 1.16, 1.17]  min 0 · mean 0.618 · max 1.34
reward/wrap_links: [0, 0.00829, 0.0144, 0.00376, 0.118, 0.2, 0.212, 0.223, 0.24, 0.261]  min 0 · mean 0.139 · max 0.266
rewards/step: [2.04, 124, 362, 294, 890, 2.06e+03, 2.92e+03, 3.63e+03, 4.05e+03, 4.33e+03]  min -46.6 · mean 2.06e+03 · max 4.79e+03
task/lifted_frac: [0, 0.000488, 0, 0, 0.0745, 0.122, 0.533, 0.419, 0.577, 0.548]  min 0 · mean 0.271 · max 0.732
task/successes_mean: [0, 0, 0, 0, 0.0254, 0.00757, 0.00903, 0.0232, 0.00464, 0.000732]  min 0 · mean 0.0102 · max 0.104
task/tilt_deg: [0.00108, 1.24, 1.2, 0.79, 5.02, 4.01, 4.45, 3.24, 3.06, 2.79]  min 0.000912 · mean 3 · max 9.31
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112
