## I can see from the robot that
The round was ended by the user at about epoch 1750 of 3000 (2.7 h), before the time limit, because the policy had settled into pressing the hand into the table - a behaviour that cannot be used on the real robot. The recording available for this round was made from the epoch-1290 checkpoint, about 460 epochs before the end; the behaviour it shows matches the metrics measured at the end of the round.

What the robot does in the recording:
- The hand reaches the cup and closes on it. The palm comes against the cup's side and all four fingers curl around it. This is a real envelope grasp and it is the first time this track has produced one; in the previous round the fingers stayed extended and only grazed the cup's surface.
- The hand travels in low, along the table surface, and presses down into it while approaching and while grasping.
- The grasp is taken low on the cup, near its base, with the hand coming in from underneath rather than against the middle of the cup's body.
- The little finger stays out of the grasp. The thumb and the other three fingers close on the cup.
- The cup stays upright at its spawn position. It is cradled at a slight angle at times but is never knocked over and is never lifted clear of the table.

Metrics over the last 150 epochs of the round:
- approach_done was set in 0.68 of far-start episodes and 1.00 of near-start episodes; envelope_done in 0.59 of far-start and 0.70 of near-start episodes.
- Episode funnel: reach 0.81, grasp 0.76, envelope 0.48, lift 0.11, success 0.0001.
- The palm touched the cup on 0.55 of steps and the mean palm-to-cup gap was 0.036 m. In the previous round palm contact never left the range 0.001 to 0.013.
- On an average step 2.98 of the five digits touched the cup: thumb 0.76, index 0.73, middle 0.72, ring 0.70, little finger 0.059. Mean contact force 4.1 N.
- The cup rose 0.006 m on average. Its distance to the goal stayed at 0.288 m and was never inside the success tolerance; near_goal was 0.0000 and there were no successes in the last 600 epochs.
- Mean cup tilt 8.8 degrees; the cup was tipped over in 0.0003 of episodes and never fell.
- The hand penetrated the table. The maximum penetration depth grew through the round to 0.028 m, and the lowest hand point reached 0.187 while the table surface is at 0.205. The reward's table term paid -0.0002, so this cost almost nothing.
- Total reward was 1.91 and still rising at the end of the round.
- Around epoch 950 there was a short period with the best lifting of the round: lift in 0.296 of episodes, cup rise 0.019 m, successes 0.0080. In that same period table penetration was at its lowest of the whole round, 0.0012 m. As penetration climbed back to 0.028 m, lifting fell to about a third of that peak and never recovered, while the total reward went on rising.
- Success in this round required only that the cup's keypoints be within 0.1125 m of the goal for ten steps. It did not require that the cup be grasped, lifted, upright, or that the hand stay off the table.

## Feedback for improvement
The big thing went right and must be preserved. The hand now reaches the cup, brings its palm against the cup's side and curls the fingers around it. That envelope grasp is what the previous rounds could never produce, and whatever pays for it should keep paying for it.

Everything below is about what happens before and after that grasp.

The hand must not touch the table. A policy that presses into the table cannot be used on the real robot, so this is a hard requirement and not a preference. At the moment the hand travels in low along the surface and pushes down into it, the penetration grew steadily through the round, and what the reward charged for it was so small it may as well have been nothing. Table contact has to be expensive from the first step of the episode - expensive enough that approaching over the table and settling onto it is never the cheapest way to reach the cup. It should cost more than any amount of progress that pressing down buys.

There is direct evidence that the table contact is what blocks the lift. The best lifting of the whole round happened in the one period when the hand was barely touching the table; as the hand sank back onto the surface, lifting fell away and never recovered even though the grasp itself kept improving. Getting the hand off the table is likely to unblock the lift by itself.

The grasp is taken too low. The hand comes in beneath the cup and grips near its base, which both drives the hand into the table and leaves the arm stretched out with nowhere to go when it should lift. The palm should meet the cup on the middle of its body, within about two centimetres above or below the cup's centre, and the approach should arrive at that height rather than climbing up from the table.

The lift is the blocked stage. The cup rises a few millimetres and then stays there, a quarter of a metre from the goal, for the rest of the episode. Nothing in the round ever carried the cup. Once the envelope grasp is held, raising the cup clear of the table and moving it toward the goal has to become the most valuable thing available, and holding a good grasp in place while going nowhere must not be able to pay more than actually carrying it.

The little finger never joins the grasp. The other four digits close and it stays out. If a five-finger wrap is wanted it needs its own reason to close; if four digits are considered enough, nothing has to change here.

Finally, the same warning as last round. Total reward rose all the way to the end while lifting and success went backwards. No term should be able to keep growing while the stage it belongs to regresses; progress on the grasp must not be able to substitute for progress on the lift.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.0908, 0.268, 0.549, 0.433, 0.681, 0.421, 0.696, 0.728, 0.747]  min 0 · mean 0.497 · max 0.783
contact/finger_index_at_success: [-1, -1, -1, 0.0929, 0.0929, 0.884, 0.913, 0.646, 0.712, 0.736]  min -1 · mean 0.227 · max 0.992
contact/finger_middle: [0, 0.0381, 0.293, 0.589, 0.406, 0.663, 0.411, 0.708, 0.711, 0.753]  min 0 · mean 0.496 · max 0.769
contact/finger_middle_at_success: [-1, -1, -1, 0.181, 0.181, 0.89, 0.924, 0.914, 0.882, 0.909]  min -1 · mean 0.306 · max 0.98
contact/finger_pinky: [0, 0.0417, 0.041, 0.148, 0.24, 0.0964, 0.0718, 0.0669, 0.0601, 0.0669]  min 0 · mean 0.0861 · max 0.302
contact/finger_pinky_at_success: [-1, -1, -1, 0, 0, 0.129, 0.0926, 0.0526, 0.0429, 0.0332]  min -1 · mean -0.178 · max 0.244
contact/finger_ring: [0, 0.0679, 0.221, 0.536, 0.44, 0.705, 0.4, 0.698, 0.698, 0.741]  min 0 · mean 0.487 · max 0.755
contact/finger_ring_at_success: [-1, -1, -1, 0, 0, 0.935, 0.996, 0.998, 0.998, 0.953]  min -1 · mean 0.294 · max 0.998
contact/finger_thumb: [0, 0.146, 0.298, 0.649, 0.476, 0.729, 0.443, 0.741, 0.759, 0.776]  min 0 · mean 0.543 · max 0.804
contact/finger_thumb_at_success: [-1, -1, -1, 0.0929, 0.0929, 0.943, 0.978, 0.988, 0.99, 0.947]  min -1 · mean 0.317 · max 0.995
contact/fingers_touching: [0, 0.384, 1.12, 2.47, 2, 2.88, 1.75, 2.91, 2.96, 3.08]  min 0 · mean 2.11 · max 3.18
contact/fingers_touching_at_success: [-1, -1, -1, 0.367, 0.367, 3.78, 3.9, 3.6, 3.63, 3.58]  min -1 · mean 1.82 · max 4.1
contact/link_force_mean: [0, 0.231, 0.747, 1.78, 2.73, 4.1, 2.01, 3.96, 4.07, 4.51]  min 0 · mean 2.63 · max 71
contact/links_touching: [0, 0.433, 1.31, 3.09, 2.38, 3.5, 2.06, 3.45, 3.5, 3.66]  min 0 · mean 2.51 · max 3.74
contact/links_touching_at_success: [-1, -1, -1, 0.367, 0.367, 4.65, 4.86, 4.32, 4.49, 4.39]  min -1 · mean 2.27 · max 5.09
contact/palm_touching: [0, 0.00195, 0.063, 0.359, 0.238, 0.611, 0.28, 0.585, 0.517, 0.628]  min 0 · mean 0.371 · max 0.647
contact/palm_touching_at_success: [-1, -1, -1, 0, 0, 0.859, 0.602, 0.774, 0.723, 0.65]  min -1 · mean 0.141 · max 0.975
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0, 0.033, 0.000244, 0.00122, 0.000244, 0.000732]  min 0 · mean 0.00225 · max 0.0405
done/abnormal: [0, 0, 0.000977, 0.000732, 0, 0, 0.000732, 0.000732, 0.000488, 0.000244]  min 0 · mean 0.000513 · max 0.00342
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.08e-06 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.74e-06 · max 0.000488
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.69e-07 · max 0.000244
done/out_xy: [0, 0, 0, 0.000977, 0, 0.000244, 0, 0.000244, 0, 0.000244]  min 0 · mean 0.000143 · max 0.0022
done/tipped: [0, 0.00171, 0.000977, 0.000244, 0.000977, 0, 0, 0.000244, 0.000244, 0.000488]  min 0 · mean 0.000535 · max 0.00781
episode_lengths/step: [48, 496, 420, 436, 670, 580, 736, 472, 612, 538]  min 48 · mean 517 · max 899
reward/action_rate_pen: [-0.00537, -0.0032, -0.00317, -0.00319, -0.0033, -0.00269, -0.00253, -0.00225, -0.00204, -0.00178]  min -0.00537 · mean -0.00278 · max -0.00174
reward/approach_keep: [0.00382, 0.0139, 0.011, 0.0042, 0.0034, 0.00217, 0.000564, 0.00533, 0.006, 0.00473]  min 0.000321 · mean 0.00493 · max 0.0167
reward/approach_reach: [0.0138, 0.027, 0.0217, 0.0103, 0.0106, 0.00746, 0.00444, 0.0105, 0.0114, 0.00905]  min 0.00374 · mean 0.0119 · max 0.0291
reward/arm_vel_pen: [-0.000288, -0.00057, -0.00136, -0.00552, -0.00278, -0.00726, -0.00505, -0.00719, -0.00481, -0.00633]  min -0.0118 · mean -0.00447 · max -0.000274
reward/cup_disturb_pen: [0, -0.00478, -0.0598, -0.0192, -0.0146, -0.0204, -0.00692, -0.0179, -0.023, -0.0165]  min -0.0645 · mean -0.0199 · max 0
reward/cup_tilt_pen: [0, -0.0122, -0.0153, -0.0229, -0.00929, -0.0185, -0.00764, -0.0185, -0.0146, -0.0153]  min -0.0356 · mean -0.0155 · max 0
reward/early_contact_pen: [0, -0.00615, -0.00664, -0.00924, -0.00437, -0.0141, 0, -0.0106, -0.00815, -0.00601]  min -0.0313 · mean -0.00839 · max 0
reward/envelope_quality: [0, 1.4e-05, 0.0107, 0.103, 0.0892, 0.203, 0.116, 0.216, 0.204, 0.25]  min 0 · mean 0.135 · max 0.259
reward/finger_curl: [0, 0.016, 0.0266, 0.0446, 0.0429, 0.0542, 0.0423, 0.0611, 0.0661, 0.0627]  min 0 · mean 0.0451 · max 0.0764
reward/grasp_fingers: [0, 0.00792, 0.0355, 0.102, 0.0914, 0.139, 0.0984, 0.157, 0.162, 0.171]  min 0 · mean 0.104 · max 0.178
reward/grasp_pose: [0, 0.0553, 0.0759, 0.0927, 0.0839, 0.104, 0.0897, 0.138, 0.148, 0.147]  min 0 · mean 0.101 · max 0.158
reward/grasp_thumb: [0, 0.0118, 0.0342, 0.104, 0.0921, 0.139, 0.0979, 0.154, 0.161, 0.167]  min 0 · mean 0.105 · max 0.174
reward/grasp_wrap: [0, 0.00447, 0.0365, 0.119, 0.103, 0.163, 0.113, 0.181, 0.178, 0.185]  min 0 · mean 0.118 · max 0.199
reward/grip_squeeze: [0, 0.00132, 0.00583, 0.0211, 0.0189, 0.0268, 0.0203, 0.0288, 0.0292, 0.0285]  min 0 · mean 0.0199 · max 0.0332
reward/hold_still: [0, 0, 3.69e-07, 2.7e-06, 1.34e-06, 2.19e-06, 2.73e-06, 6.55e-06, 5.92e-06, 5.74e-06]  min 0 · mean 3.2e-06 · max 8.23e-06
reward/lift_goal: [0, 0, 0.000754, 0.00524, 0.00325, 0.0081, 0.00441, 0.00931, 0.00854, 0.0096]  min 0 · mean 0.00549 · max 0.0111
reward/lift_height: [0, 0, 0.00116, 0.00704, 0.00455, 0.0485, 0.00612, 0.0168, 0.012, 0.032]  min 0 · mean 0.012 · max 0.0503
reward/lift_hold: [0, 0, 0.0276, 0.177, 0.136, 0.281, 0.155, 0.276, 0.262, 0.312]  min 0 · mean 0.182 · max 0.322
reward/light_contact: [0, 0.00635, 0.0175, 0.0404, 0.0358, 0.0487, 0.0371, 0.0559, 0.0596, 0.06]  min 0 · mean 0.0387 · max 0.0642
reward/palm_contact: [0, 0.000633, 0.0159, 0.133, 0.119, 0.264, 0.158, 0.296, 0.279, 0.339]  min 0 · mean 0.181 · max 0.349
reward/palm_reach: [0, 0.0436, 0.119, 0.166, 0.152, 0.198, 0.168, 0.261, 0.283, 0.284]  min 0 · mean 0.181 · max 0.303
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.84e-06 · max 0.00146
reward/table_pen: [0, 0, -6.88e-05, -0.000734, -0.000252, -2.02e-05, -6.61e-05, -0.000271, -7.57e-06, -8.22e-05]  min -0.00158 · mean -0.000136 · max 0
reward/thumb_curl: [0, 0.0149, 0.0291, 0.023, 0.0157, 0.0264, 0.0213, 0.0279, 0.0328, 0.0312]  min 0 · mean 0.0235 · max 0.0349
reward/total: [0.012, 0.176, 0.382, 1.09, 0.967, 1.65, 1.11, 1.84, 1.85, 2.05]  min 0.00756 · mean 1.22 · max 2.12
rewards/step: [-2.78, 37.7, 144, 412, 657, 930, 830, 860, 1.17e+03, 1.09e+03]  min -2.78 · mean 648 · max 1.26e+03
task/lifted_frac: [0, 0.000488, 0.00122, 0.0217, 0.0166, 0.149, 0.022, 0.0808, 0.0427, 0.146]  min 0 · mean 0.0482 · max 0.225
task/successes_mean: [0, 0, 0, 0, 0, 0.00659, 0, 0.00122, 0, 0]  min 0 · mean 0.000538 · max 0.0129
task/tilt_deg: [0.00108, 5.04, 5.98, 10.6, 5.84, 10, 4.69, 9.72, 8.63, 9.29]  min 0.000912 · mean 7.57 · max 12.5
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112
