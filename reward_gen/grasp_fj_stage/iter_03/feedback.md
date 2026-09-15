## I can see from the robot that
The round was ended early, at about epoch 500 of 3000, because the hand had stopped interacting with the cup; the playback uses the latest checkpoint and shows three consecutive episodes of one environment.
Episodes 1 and 2 start from the raised pose. The hand moves down beside the robot column near the table edge closest to the robot, turns away from its start orientation with the fingers spread, drifts around there and rises again. It never moves towards the cup and never touches it.
Episode 3 starts beside the cup, with approach_done already set: the fingers point forward along +x, the palm faces the cup's side and the cup is just in front of the hand. Within 0.25 s the fingers are still open and the hand is already pulling back; by 0.7 s it is clearly away from the cup with the fingers spread; later it rises high and moves to the table edge. It never closes the fingers and never touches the cup.
Metrics (up to epoch 498):
- approach_done was set in every near-start episode (as the environment does) and in no far-start episode; envelope_done was never set in either group.
- In near-start episodes, the share in which at least three digits touched the cup at some step rose to 0.60 at epoch 120, was 0.51 at epoch 338 and fell to 0.00 at epoch 498.
- The mean palm-to-band distance grew from 0.19 m at epoch 60 to 0.41 m at epoch 498.
- The total reward was 0.182 at epoch 498, of which grasp_stage_base was 0.178. grasp_palm was 0 at every logged epoch; grasp_contacts fell from 0.016 (epoch 226) to 0.000, grasp_wrap from 0.012 to 0.000 and grip_squeeze stayed at about 0.
- cup_disturb_pen went from -0.018 at epoch 60 to -0.001 at epoch 498; early_contact_pen stayed at about 0.
- approach_reach stayed at about 0.006; the share of steps with the hand in its start orientation fell from 1.00 to 0.29.

## Feedback for improvement
The policy learned to stay away from the cup: the reward pays almost all of its value for the stage flags alone.
- After approach_done is set, the stage-2 base amount is paid on every step whatever the hand does, and it was nearly the whole reward. Staying beside the cup, closing the hand and moving away all earn that same amount, so the policy pulled the hand back at the start of near-start episodes and kept the reward. Stage progress must only pay while the hand keeps doing that stage's work: after the approach, the reward has to fall clearly when the palm leaves the grasp position beside the cup, and rise as the palm reaches the cup's side and the fingers and thumb close on it.
- Early in training the hand did touch the cup with several digits in near-start episodes, but those touches earned very little while nudging the cup was penalised, and the touching disappeared. Closing the hand around the cup, palm contact and thumb and finger contacts must be worth clearly more than the small cup movements that happen while closing; only knocking the cup over or pushing it away should cost more than grasping gains.
- The palm never touched the cup. In near-start episodes the cup already sits between the thumb and the four fingers in front of the palm; the palm should first move the last few centimetres to the cup's side, then the fingers and thumb close.
- From the raised start the hand never approached, and the approach reward was too small to matter next to what near-start episodes paid. Moving the palm towards the cup's -y side in the start orientation has to pay clearly more than drifting.
- The hand turned away from its start orientation in both kinds of episodes; keeping the orientation should stay part of the approach and of the grasp.
- Stage 3 was never reached, so there is no feedback on lifting yet.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.000488, 0, 0.00757, 0, 0, 0.000732, 0, 0, 0.00122]  min 0 · mean 0.00185 · max 0.116
contact/finger_index_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_middle: [0, 0.00146, 0, 0.00195, 0, 0, 0.000732, 0, 0, 0]  min 0 · mean 0.000671 · max 0.0718
contact/finger_middle_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_pinky: [0, 0.00366, 0, 0.0479, 0.0256, 0.00171, 0.0681, 0.0134, 0.108, 0.0171]  min 0 · mean 0.0162 · max 0.216
contact/finger_pinky_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_ring: [0, 0.00293, 0, 0.00537, 0, 0, 0.00269, 0, 0, 0]  min 0 · mean 0.00168 · max 0.118
contact/finger_ring_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_thumb: [0, 0.000488, 0.0808, 0.34, 0.276, 0.443, 0.356, 0.251, 0.344, 0.167]  min 0 · mean 0.208 · max 0.6
contact/finger_thumb_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/fingers_touching: [0, 0.00903, 0.0808, 0.403, 0.302, 0.444, 0.429, 0.265, 0.451, 0.185]  min 0 · mean 0.228 · max 0.656
contact/fingers_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/link_force_mean: [0, 0.013, 0.00291, 0.0317, 0.0229, 0.0376, 0.0365, 0.0141, 0.0402, 0.0244]  min 0 · mean 0.0204 · max 0.128
contact/links_touching: [0, 0.0103, 0.0813, 0.406, 0.304, 0.461, 0.445, 0.268, 0.454, 0.186]  min 0 · mean 0.233 · max 0.67
contact/links_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/palm_touching: [0, 0.000488, 0, 0.0164, 0.242, 0.0249, 0.1, 0.258, 0.371, 0.461]  min 0 · mean 0.159 · max 0.666
contact/palm_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/abnormal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.24e-05 · max 0.00146
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2e-07 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.86e-06 · max 0.000732
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.51e-07 · max 0.000244
done/tipped: [0, 0, 0, 0.000732, 0, 0.000244, 0, 0, 0.000244, 0]  min 0 · mean 0.000114 · max 0.00439
episode_lengths/step: [46, 847, 868, 821, 882, 829, 844, 884, 832, 847]  min 46 · mean 838 · max 898
reward/action_rate_pen: [-0.00537, -0.00233, -0.00101, -0.00117, -0.000986, -0.000887, -0.000823, -0.000695, -0.000594, -0.000521]  min -0.00537 · mean -0.00117 · max -0.000467
reward/approach_keep: [0.0268, 0.00523, 0.0234, 0.0107, 0.00704, 0.0116, 0.00698, 0.00681, 0.00908, 0.00577]  min 0.00091 · mean 0.00941 · max 0.0268
reward/approach_reach: [0.0127, 0.00523, 0.0261, 0.0144, 0.0081, 0.0132, 0.00902, 0.00812, 0.011, 0.00678]  min 0.00115 · mean 0.0104 · max 0.028
reward/arm_vel_pen: [-0.000719, -0.00138, -0.000648, -0.00193, -0.000982, -0.00093, -0.000488, -0.000501, -0.00444, -0.00294]  min -0.00546 · mean -0.000996 · max -0.000369
reward/cup_disturb_pen: [0, -0.00183, -0.00374, -0.0264, -0.022, -0.0124, -0.00899, -0.00934, -0.0397, -0.0224]  min -0.0779 · mean -0.0122 · max 0
reward/cup_tilt_pen: [-1.19e-09, -0.000306, -0.000313, -0.00241, -0.000998, -0.00214, -0.000754, -0.000687, -0.00403, -0.00136]  min -0.0083 · mean -0.00114 · max -1.05e-09
reward/early_contact_pen: [0, -0.00116, 0, -0.00851, -0.000536, -0.00691, -0.00211, -0.000891, -0.0104, -0.000384]  min -0.0167 · mean -0.00153 · max 0
reward/grasp_contacts: [0, 0, 0.0042, 0.0159, 0.0176, 0.0246, 0.0246, 0.0165, 0.0178, 0.0109]  min 0 · mean 0.0127 · max 0.0393
reward/grasp_palm: [0, 3.38e-05, 0.000196, 0.00716, 0.0203, 0.00244, 0.0118, 0.0251, 0.0308, 0.035]  min 0 · mean 0.0145 · max 0.0519
reward/grasp_stage_base: [0, 0.176, 0.168, 0.265, 0.283, 0.237, 0.291, 0.287, 0.266, 0.282]  min 0 · mean 0.249 · max 0.314
reward/grasp_wrap: [0, 0.000141, 0.00611, 0.0216, 0.0256, 0.0197, 0.0253, 0.0286, 0.027, 0.0262]  min 0 · mean 0.0183 · max 0.0326
reward/grip_squeeze: [0, 0, 1.31e-06, 3.29e-06, 8.36e-06, 3.75e-07, 1.96e-06, 1.45e-06, 3.03e-07, 6.29e-07]  min 0 · mean 9.41e-06 · max 0.00151
reward/hold_contacts: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.12e-06 · max 0.000323
reward/hold_still: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.21e-11 · max 4.94e-09
reward/lift_goal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.53e-07 · max 2.33e-05
reward/lift_height: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.01e-06 · max 0.000122
reward/lift_stage_base: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.26e-06 · max 0.00122
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/table_pen: [0, -9.39e-05, -6.9e-06, 0, -9.21e-06, 0, 0, -7.15e-07, -6.45e-05, 0]  min -0.00872 · mean -0.000148 · max 0
reward/total: [0.0334, 0.18, 0.222, 0.295, 0.336, 0.285, 0.356, 0.36, 0.302, 0.339]  min 0.0279 · mean 0.298 · max 0.401
rewards/step: [-2.07, 148, 186, 216, 304, 267, 290, 335, 288, 283]  min -2.07 · mean 249 · max 335
task/lifted_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.4e-05 · max 0.00146
task/successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/tilt_deg: [0.00108, 0.216, 0.686, 2.97, 2.65, 3.42, 2.11, 2.09, 6.02, 3.25]  min 0.000912 · mean 2.28 · max 7.22
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112
