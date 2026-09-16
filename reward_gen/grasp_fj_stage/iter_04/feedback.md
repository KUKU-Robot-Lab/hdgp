## I can see from the robot that
The round was judged over at the four-hour limit, at epoch 2788 of 3000. The playback uses a snapshot of the checkpoint at that point and shows two consecutive episodes of one environment (30 s). The run itself was left training while the recording was made and has since reached epoch 2989; the figures below are given for both points where they differ.

What the robot does in the recording:
- In both episodes the hand is at the cup's side within the first three seconds and stays there for the rest of the episode. It does not drift away and it does not turn away from its start orientation: the palm keeps facing the cup's side and the fingers keep pointing forward across it.
- The fingers stay extended for the whole recording. They reach across the near side of the cup and touch it with their tips and middle segments, but they never curl around it. At no point in either episode does a fingertip pass the far side of the cup, and the thumb hangs beside and below the cup instead of opposing the fingers around it.
- A gap between the palm and the cup's side is visible in every frame. The palm never rests against the cup.
- The cup stands upright at its spawn position in every frame of both episodes. It is never tipped, never pushed noticeably and never lifted off the table.
- Which of the two episodes starts beside the cup and which starts from the raised pose cannot be told apart in the recording; after the first three seconds both look the same.

Metrics (mean over the last 150 epochs, at epoch 2788):
- approach_done was set in 0.97 of far-start episodes and in 1.00 of near-start episodes. envelope_done was set in 0.0097 of far-start and 0.0033 of near-start episodes, and the episode funnel value for envelope is 0.0000.
- Episode funnel: reach 1.00, grasp 1.00 (at least three digits touched at some step), envelope 0.0000, lift 0.0017, success 0.0000.
- On an average step 3.2 to 3.3 of the five digits touch the cup (thumb 0.85, middle 0.85, ring 0.82, index 0.81, little finger 0.0003), while the palm touches on between 0.001 and 0.013 of steps depending on the window (0.0013 at epoch 2788, 0.013 at epoch 2989).
- The mean palm-to-cup gap is 0.046 m.
- The total reward is 0.958. Its largest parts are grasp_stay 0.330, grasp_fingers 0.210, grasp_palm_close 0.155, grasp_thumb 0.155, grasp_wrap 0.077 and grip_squeeze 0.026. envelope_quality is 0.0001 and grasp_palm_touch is 0.0002; every lift term and the success bonus are 0.
- cup_disturb_pen is -0.002 and cup_tilt_pen is -0.005; no episode ended with the cup tipped.
- There were no successes in the last 600 epochs. The contact values recorded at the moment of success stopped changing after about epoch 2300 and still carry the values from the successes before that: palm touching 0.80 and 2.11 fingers touching. Those were the only grasps in the whole round in which the palm touched the cup, and they stopped.
- The total reward rose through the entire round, from 0.73 at epoch 728 to 0.87 at 1757 and 0.90 to 0.96 at the end, while over the same period palm contact stayed inside a band between about 0.001 and 0.013 of steps with no sustained direction, and the envelope was never completed in a measurable share of episodes.

## Feedback for improvement
Three things are working and should be kept.
- The approach is solved. From both starting states the hand reaches the cup's side in about three seconds, keeps its start orientation and holds that position for the rest of the episode. It no longer drifts away or turns away, which were the failures of the previous rounds.
- The hand closes on the cup. Three or four digits touch it on an average step, and that has been rising steadily.
- The cup is handled gently. It stays upright at its spawn position and is never knocked over, so the penalties for disturbing it are doing their job without suppressing contact.

The rest is what has to change.

The grasp that was learned is a fingertip grasp, not an envelope. The fingers reach across the near side of the cup and touch it with their tips while staying extended, the thumb hangs beside the cup instead of opposing the fingers, and the palm stays a few centimetres away. The reward currently pays its largest amounts for exactly this shape: holding position beside the cup and touching it with the fingers. Touching the cup with extended fingers while the palm stays away must be worth clearly less than a grasp in which the palm is against the cup's side and the fingers are curled around it.

Palm contact is the one missing condition, and it is the whole blockage. Everything else the envelope needs is already there on most steps: the thumb touches, and three or more fingers touch. The palm touched on somewhere between one step in a thousand and one step in a hundred, drifting up and down inside that band all round without ever climbing out of it, and as a result the envelope was completed in far less than one episode in a hundred. The distance still to cover is a few centimetres of palm travel. Those last centimetres, and the palm contact at the end of them, have to become the most valuable thing available in this stage, worth more than any amount of time spent holding position or touching with the fingertips.

The clearest evidence that the present shape is a dead end: the total reward rose for the entire round while palm contact never left a band far below what the envelope needs. Terms that pay for staying beside the cup and for finger contact can be driven up indefinitely without the grasp ever being completed. No term should be able to keep growing while the stage it belongs to goes backwards.

The policy did find the right grasp once. In the middle of the round there was a short period with rare successes in which the palm was touching and two fingers were on the cup - the only palm-contact grasps of the round. They disappeared and never returned, which means what the reward offered for them was too small, or too easily matched by the fingertip shape, for the policy to hold on to. A completed envelope, and the lift that follows it, must be worth far more than the fingertip grasp can ever accumulate, so that finding it once is enough to keep it.

Two smaller points. The little finger never touches the cup in any episode; if a five-finger wrap is intended it needs its own reason to close, and if it is not intended nothing has to change. And nothing has been learned about lifting, because the envelope was never completed - lifting feedback will only become meaningful once the envelope happens.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.000244, 0.196, 0.718, 0.726, 0.719, 0.665, 0.809, 0.756, 0.839]  min 0 · mean 0.59 · max 0.841
contact/finger_index_at_success: [-1, -1, -1, 0, 0.258, 0.245, 0.245, 0.18, 0.0407, 0.0407]  min -1 · mean -0.113 · max 0.258
contact/finger_middle: [0, 0.000488, 0.421, 0.689, 0.737, 0.803, 0.831, 0.827, 0.815, 0.866]  min 0 · mean 0.653 · max 0.873
contact/finger_middle_at_success: [-1, -1, -1, 0, 0.326, 0.31, 0.31, 0.454, 0.877, 0.877]  min -1 · mean 0.152 · max 0.877
contact/finger_pinky: [0, 0.00146, 0.00146, 0.000977, 0.0022, 0.000488, 0.00146, 0.000488, 0, 0]  min 0 · mean 0.00362 · max 0.17
contact/finger_pinky_at_success: [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0]  min -1 · mean -0.207 · max 0
contact/finger_ring: [0, 0.00757, 0.333, 0.756, 0.774, 0.781, 0.759, 0.817, 0.765, 0.854]  min 0 · mean 0.638 · max 0.858
contact/finger_ring_at_success: [-1, -1, -1, 0, 0.224, 0.213, 0.213, 0.157, 0.355, 0.355]  min -1 · mean -0.0408 · max 0.355
contact/finger_thumb: [0, 0.01, 0.533, 0.714, 0.786, 0.834, 0.834, 0.839, 0.811, 0.874]  min 0 · mean 0.684 · max 0.879
contact/finger_thumb_at_success: [-1, -1, -1, 0, 0.431, 0.41, 0.41, 0.527, 0.839, 0.839]  min -1 · mean 0.176 · max 0.839
contact/fingers_touching: [0, 0.0198, 1.48, 2.88, 3.03, 3.14, 3.09, 3.29, 3.15, 3.43]  min 0 · mean 2.57 · max 3.45
contact/fingers_touching_at_success: [-1, -1, -1, 0, 1.24, 1.18, 1.18, 1.32, 2.11, 2.11]  min -1 · mean 0.797 · max 2.11
contact/link_force_mean: [0, 0.0186, 0.553, 0.486, 0.658, 0.828, 0.864, 0.795, 0.832, 0.927]  min 0 · mean 0.701 · max 43.6
contact/links_touching: [0, 0.0225, 1.63, 3.05, 3.27, 3.42, 3.44, 3.67, 3.51, 3.88]  min 0 · mean 2.85 · max 3.9
contact/links_touching_at_success: [-1, -1, -1, 0, 1.36, 1.29, 1.29, 1.4, 2.46, 2.46]  min -1 · mean 0.93 · max 2.46
contact/palm_touching: [0, 0.000732, 0.0344, 0.000488, 0.000977, 0.00903, 0.011, 0.000488, 0.00269, 0.000977]  min 0 · mean 0.00843 · max 0.122
contact/palm_touching_at_success: [-1, -1, -1, 0, 0, 0, 0, 0.181, 0.803, 0.803]  min -1 · mean 0.0235 · max 0.803
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0, 0.000244, 0, 0, 0, 0]  min 0 · mean 0.000228 · max 0.00732
done/abnormal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.19e-05 · max 0.000977
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.32e-07 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.69e-07 · max 0.000244
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.76e-06 · max 0.000244
done/tipped: [0, 0.000244, 0.000244, 0, 0, 0.000244, 0, 0, 0.000244, 0]  min 0 · mean 0.000166 · max 0.00659
episode_lengths/step: [34.5, 803, 776, 843, 869, 873, 872, 885, 858, 885]  min 34.5 · mean 820 · max 886
reward/action_rate_pen: [-0.00537, -0.0029, -0.00264, -0.00189, -0.00175, -0.00158, -0.0015, -0.0013, -0.00126, -0.00112]  min -0.00537 · mean -0.00188 · max -0.00104
reward/approach_keep: [0.00382, 0.00742, 0.0104, 0.00658, 0.00512, 0.0049, 0.00432, 0.00432, 0.00472, 0.0038]  min 0.00175 · mean 0.00604 · max 0.0271
reward/approach_reach: [0.023, 0.0239, 0.0336, 0.0203, 0.0156, 0.0143, 0.0142, 0.0126, 0.0154, 0.0109]  min 0.0105 · mean 0.0186 · max 0.0862
reward/arm_vel_pen: [-0.000719, -0.000999, -0.00177, -0.00108, -0.000991, -0.0011, -0.00121, -0.00151, -0.000951, -0.000875]  min -0.00393 · mean -0.00118 · max -0.000719
reward/cup_disturb_pen: [0, -0.00114, -0.017, -0.00282, -0.00578, -0.00412, -0.00332, -0.00139, -0.00338, -0.00186]  min -0.0298 · mean -0.00515 · max 0
reward/cup_tilt_pen: [0, -0.00198, -0.0135, -0.00605, -0.00577, -0.00296, -0.00413, -0.00347, -0.00752, -0.00402]  min -0.0283 · mean -0.00613 · max 0
reward/early_contact_pen: [0, -0.00195, -0.0112, -0.00403, -0.00269, -0.00189, -0.00537, -0.000879, -0.00493, -0.00186]  min -0.0255 · mean -0.00361 · max 0
reward/envelope_quality: [0, 0, 0.000402, 0, 1.31e-05, 0.000518, 0.000675, 2.56e-05, 0.000121, 8.3e-05]  min 0 · mean 0.000285 · max 0.00537
reward/grasp_fingers: [0, 0.000427, 0.0581, 0.145, 0.17, 0.187, 0.179, 0.209, 0.189, 0.22]  min 0 · mean 0.148 · max 0.222
reward/grasp_palm_close: [0, 0.081, 0.0702, 0.116, 0.134, 0.145, 0.141, 0.153, 0.141, 0.16]  min 0 · mean 0.124 · max 0.162
reward/grasp_palm_touch: [0, 5.61e-05, 0.0036, 1.32e-05, 0.000147, 0.00143, 0.00181, 8.08e-05, 0.000324, 0.000154]  min 0 · mean 0.00127 · max 0.0206
reward/grasp_stay: [0, 0.256, 0.206, 0.294, 0.314, 0.322, 0.316, 0.331, 0.313, 0.338]  min 0 · mean 0.29 · max 0.341
reward/grasp_thumb: [0, 0.000253, 0.0657, 0.109, 0.13, 0.145, 0.142, 0.153, 0.141, 0.16]  min 0 · mean 0.114 · max 0.162
reward/grasp_wrap: [0, 0.0195, 0.0339, 0.0581, 0.0676, 0.0713, 0.0691, 0.0754, 0.0671, 0.0794]  min 0 · mean 0.0594 · max 0.08
reward/grip_squeeze: [0, 1.25e-05, 0.00821, 0.0164, 0.0191, 0.0223, 0.0221, 0.0246, 0.0232, 0.0269]  min 0 · mean 0.0179 · max 0.0278
reward/hold_still: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.95e-08]  min 0 · mean 8.5e-09 · max 2.14e-07
reward/lift_goal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.23e-05]  min 0 · mean 1.33e-05 · max 0.00039
reward/lift_grip_hold: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000146]  min 0 · mean 0.000268 · max 0.00952
reward/lift_height: [0, 0, 0, 0, 0, 0, 0, 0, 0, 8.03e-06]  min 0 · mean 1.86e-05 · max 0.000725
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/table_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.000504 · mean -3.7e-06 · max 0
reward/total: [0.0208, 0.38, 0.445, 0.749, 0.838, 0.902, 0.874, 0.954, 0.877, 0.99]  min 0.0146 · mean 0.762 · max 0.997
rewards/step: [0.73, 306, 414, 659, 717, 777, 783, 827, 748, 830]  min 0.0362 · mean 636 · max 860
task/lifted_frac: [0, 0, 0.000244, 0, 0.000244, 0.00146, 0, 0.000488, 0.000732, 0]  min 0 · mean 0.000281 · max 0.00806
task/successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.8e-06 · max 0.00195
task/tilt_deg: [0.00108, 0.625, 6.02, 4.98, 5.4, 3.85, 4.41, 4.37, 5.66, 4.9]  min 0.000912 · mean 4.66 · max 9.62
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112
