## I can see from the robot that
Round 8 (`fj_stage_i07`) ran to epoch 1712 over 3.14 hours and was stopped by the user. It is the round that fixed the failure of the previous one.

**The raised start learned to approach.** This was the single goal of the round and it succeeded. The fraction of far-start episodes that complete the approach was 0.000 for the first three hundred epochs, crossed zero around epoch 330, and reached 0.742 by epoch 800. It settled at 0.625. In the previous round this figure peaked at 0.669 around epoch 530 and then collapsed to 0.001; here it did not collapse.

**The whole chain now works from the raised start.** By the end of the round, far-start episodes reached the cup 0.943 of the time, closed a grasp 0.889, completed the envelope 0.769, and lifted 0.780. The previous round produced no far-start successes at all; this one peaked at 0.370.

**The grasp is a real envelope grasp.** At the moment of success the palm was on the cup 0.914 of the time and 4.35 digits were in contact. Grasp quality at lift was 0.648. The recording confirms it: the palm lies against the side of the cup with four fingers wrapped around the body and the thumb opposing them. This is not a fingertip pinch.

**The cup is carried much higher than before.** Lift height reached 0.208 m against 0.056 m in the previous round, and 0.62 of episodes had the cup airborne.

**The hand stayed off the table for the whole round.** Palm clearance ended at 0.276 m, the palm-on-table fraction was zero, and the episode-ending floor condition never fired.

Three things went wrong, and they are the substance of the next round.

**The round peaked at epoch 1000-1200 and declined for the remaining five hundred epochs.** Far-start success went 0.370, then 0.215, then 0.066. The mean episode success count went 1.257, then 0.655, then 0.200. Meanwhile the total reward kept climbing the whole time — 1.455, 1.387, 1.705, its maximum at the very end. Reward rose while the task went backwards. This is the same signature as round 5, where the total climbed from 0.73 to 0.96 while palm contact fell away.

The cause is visible in the individual terms. Income for raising the cup rose from 0.489 to 0.622 and income for clearing the table from 0.195 to 0.234, while the success bonus fell from 0.0140 to 0.0014 — a tenfold drop — and the term that pays for holding still at the goal never left 0.0001 in more than a thousand epochs. Income for arriving at the goal also fell, from 0.0290 to 0.0139. The policy found that holding the cup up pays better than carrying it to the goal and stopping there, and moved its income accordingly.

**The cup is carried lying over.** The recording shows it held at roughly thirty to forty-five degrees from upright through most of the carry. The logged mean tilt of 9.6 degrees hides this, because it averages every environment and every step including the long stretches before the lift. The reason tilting is free is arithmetic: the tilt penalty was 0.0146 against 0.622 of lift income, a ratio of one to forty-three, and the upright term is added alongside the lift income rather than gating it, so a cup raised on its side still collects the raising income in full.

**The approach posture is abandoned the instant the latch fires.** The approach gate is a latch — six conditions true in a single step, and it stays on for the rest of the episode. The per-step condition rates show what happens after that: not touching the cup fell from 0.968 to 0.240, the along-axis window from 0.217 to 0.046, the plane gap from 0.292 to 0.169. The penalty for touching the cup during the approach grew fifteenfold, from 0.0017 to 0.0250, and the policy paid it willingly. The episode-level latch stayed at 0.625 through all of this, so the funnel never showed it. In the recording the approach itself still looks reasonable — the hand comes in with the fingers extended — but the shape is held for an instant rather than maintained into the grasp.

Two notes for the record. The wrap ratchet never moved: the mean episode success count peaked at 1.697 and never reached the threshold of 2.0, so the tolerance stayed at its starting value of 0.150 for the entire round and the new ceiling of 0.35 introduced this round is still untested. And a maximum hand joint tracking error of 1051 and later 341 appeared in two buckets; the arm velocity maximum spiked with it and the runaway termination fired, so these were isolated diverging environments that were terminated, not a control fault — the value returned to 2.9 afterwards.

## Feedback for improvement
Four things went right and must be kept.

- The approach from the raised start works now, and it does not collapse. This was the failure of the previous round and it is fixed. Whatever pays for travelling in from far away and arriving in the pre-grasp shape must keep paying exactly as it does.
- The grasp is a true envelope grasp: the palm against the side of the cup, four fingers wrapped around the body, the thumb opposing. At the moment of success the palm is in contact 0.91 of the time. Keep whatever produces this.
- The cup is lifted well clear of the table — 0.208 m, almost four times the previous round.
- The hand never rests on the table. Palm clearance ends at 0.276 m and the floor condition never fires. This requirement is met and must not be given back.

The rest is what has to change.

**Carrying the cup upward must stop being a substitute for finishing the task.** This is the main failure. Income for raising the cup grew to 0.622 and income for clearing the table to 0.234, while the bonus for succeeding fell to 0.0014 and the term that pays for settling at the goal stayed at 0.0001 for more than a thousand epochs. Success peaked at epoch 1000-1200 and then fell by a factor of five, and the total reward rose the whole time it was falling. Holding the cup in the air is currently the most profitable thing the policy can do, and it is not the task. Raising the cup should be worth something only as a step towards the goal: the income for height should saturate quickly once the cup is clear of the table, and the great majority of the remaining value should sit at the goal — arriving there, slowing down, and staying. A policy that lifts the cup and hovers must earn clearly less than one that carries it to the goal and stops.

**Carrying the cup on its side must not collect the carrying income.** The cup is held at thirty to forty-five degrees from upright through most of the carry. The tilt penalty is 0.0146 against 0.622 of lift income — one part in forty-three — and being upright is paid as a separate additive term, so a cup raised sideways still earns the raising income in full. Being upright should multiply the carrying and goal income rather than sit beside it, so that a cup carried lying over earns close to nothing for being carried, from the moment it leaves the table rather than only at the goal.

**Holding the approach shape must pay until the grasp actually begins.** The approach gate latches on a single step and then stays on, and the policy has learned to satisfy it for an instant and immediately drive into the cup. Across the round, not touching the cup during the approach fell from 0.968 to 0.240 and the penalty for touching it early grew fifteenfold while being paid willingly. The episode-level latch never showed this because it only needs one step. Maintaining the pre-grasp shape — the hand open, the palm facing the cup, no digit contact — from the moment the approach completes until the palm is actually against the cup should be worth enough that breaking it early is not profitable. The intended order is approach, then close from the approached pose; right now it is approach for one step, then dive.

One caution about the shape of this round. Everything good happened between epoch 300 and epoch 1200, and the last five hundred epochs undid a large part of it while the total reward reached its maximum. A reward that lets a mid-task behaviour pay more than finishing will keep producing this, so the value has to be concentrated at the end of the task rather than along the way to it.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.00537, 0.0105, 0.144, 0.461, 0.655, 0.58, 0.616, 0.694, 0.625]  min 0 · mean 0.428 · max 0.734
contact/finger_index_at_success: [-1, -1, -1, 0.06, 0.855, 0.911, 0.917, 0.794, 0.906, 0.812]  min -1 · mean 0.298 · max 0.986
contact/finger_middle: [0, 0.00952, 0.00464, 0.0759, 0.365, 0.71, 0.583, 0.658, 0.753, 0.773]  min 0 · mean 0.448 · max 0.794
contact/finger_middle_at_success: [-1, -1, -1, 0.107, 0.676, 0.867, 0.766, 0.919, 0.989, 0.961]  min -1 · mean 0.298 · max 1
contact/finger_pinky: [0, 0.000244, 0.000732, 0.0645, 0.266, 0.704, 0.462, 0.623, 0.704, 0.756]  min 0 · mean 0.402 · max 0.758
contact/finger_pinky_at_success: [-1, -1, -1, 0.107, 0.382, 0.857, 0.499, 0.819, 0.965, 0.967]  min -1 · mean 0.25 · max 1
contact/finger_ring: [0, 0.00293, 0.00439, 0.133, 0.361, 0.696, 0.557, 0.6, 0.678, 0.68]  min 0 · mean 0.42 · max 0.724
contact/finger_ring_at_success: [-1, -1, -1, 0.181, 0.83, 0.916, 0.858, 0.636, 0.917, 0.899]  min -1 · mean 0.298 · max 0.994
contact/finger_thumb: [0, 0.00635, 0.00659, 0.215, 0.501, 0.76, 0.64, 0.695, 0.764, 0.778]  min 0 · mean 0.492 · max 0.8
contact/finger_thumb_at_success: [-1, -1, -1, 1, 0.993, 0.987, 0.997, 1, 1, 0.979]  min -1 · mean 0.437 · max 1
contact/fingers_touching: [0, 0.0244, 0.0269, 0.632, 1.95, 3.52, 2.82, 3.19, 3.59, 3.61]  min 0 · mean 2.19 · max 3.74
contact/fingers_touching_at_success: [-1, -1, -1, 1.45, 3.74, 4.54, 4.04, 4.17, 4.78, 4.62]  min -1 · mean 2.69 · max 4.96
contact/link_force_mean: [0, 0.00461, 0.00585, 0.441, 1.01, 2.48, 1.76, 2.21, 2.55, 3]  min 0 · mean 2.19 · max 1.08e+03
contact/links_touching: [0, 0.0249, 0.0283, 0.745, 2.36, 4.66, 3.42, 3.88, 4.61, 4.83]  min 0 · mean 2.75 · max 4.86
contact/links_touching_at_success: [-1, -1, -1, 1.66, 4.6, 5.88, 4.85, 4.99, 6.04, 5.91]  min -1 · mean 3.49 · max 6.85
contact/palm_touching: [0, 0, 0, 0.052, 0.206, 0.503, 0.297, 0.486, 0.412, 0.567]  min 0 · mean 0.289 · max 0.625
contact/palm_touching_at_success: [-1, -1, -1, 0.797, 0.795, 0.621, 0.186, 0.75, 0.76, 0.95]  min -1 · mean 0.203 · max 1
ctrl/prev_ep_successes_mean: [0, 0, 0, 0.00293, 0.04, 0.238, 0.958, 0.854, 0.929, 0.133]  min 0 · mean 0.363 · max 1.7
done/abnormal: [0, 0, 0, 0, 0.000488, 0, 0, 0, 0, 0.000488]  min 0 · mean 6.22e-05 · max 0.00122
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.71e-06 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.57e-06 · max 0.000244
done/max_goals: [0, 0, 0, 0, 0, 0, 0.000732, 0.000244, 0, 0]  min 0 · mean 7.44e-05 · max 0.00122
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.01e-05 · max 0.00195
done/tipped: [0, 0.000732, 0.000732, 0.000977, 0.000732, 0.000488, 0.00122, 0, 0.000488, 0]  min 0 · mean 0.000695 · max 0.00977
episode_lengths/step: [29, 573, 642, 427, 524, 723, 652, 730, 754, 809]  min 29 · mean 632 · max 894
reward/action_rate_pen: [-0.00268, -0.00241, -0.00193, -0.00151, -0.00143, -0.00152, -0.00117, -0.00117, -0.000975, -0.00094]  min -0.00269 · mean -0.00149 · max -0.000887
reward/approach_form: [0.0177, 0.0515, 0.0578, 0.0443, 0.0242, 0.0186, 0.0233, 0.02, 0.0197, 0.0176]  min 0.014 · mean 0.0283 · max 0.0658
reward/approach_keep: [0.000838, 0.000981, 0.00186, 0.00716, 0.00484, 0.00246, 0.00311, 0.00161, 0.00135, 0.00174]  min 0.000375 · mean 0.00278 · max 0.00964
reward/approach_lock: [0, 5.74e-09, 9.97e-05, 0.00437, 0.00179, 0.00101, 0.00171, 0.00133, 0.00122, 0.00157]  min 0 · mean 0.00125 · max 0.00703
reward/approach_travel: [0.0135, 0.0158, 0.021, 0.031, 0.0202, 0.0126, 0.016, 0.0102, 0.00967, 0.0101]  min 0.00831 · mean 0.0163 · max 0.0373
reward/arm_vel_pen: [-0.000144, -0.000188, -0.000353, -0.000527, -0.000794, -0.00135, -0.000997, -0.00158, -0.000996, -0.00106]  min -0.00222 · mean -0.00097 · max -0.000135
reward/carry_upright: [0, 7.85e-07, 0, 0.00141, 0.0085, 0.00797, 0.0883, 0.127, 0.062, 0.155]  min 0 · mean 0.0583 · max 0.162
reward/cup_disturb_pen: [0, -0.000457, -0.000311, -0.00785, -0.0115, -0.0171, -0.0151, -0.017, -0.0191, -0.019]  min -0.0455 · mean -0.0155 · max 0
reward/cup_tilt_pen: [0, -0.00232, -0.00141, -0.0153, -0.021, -0.076, -0.0143, -0.00892, -0.0169, -0.0101]  min -0.076 · mean -0.0132 · max 0
reward/early_contact_pen: [0, -0.0018, -0.000826, -0.00949, -0.0125, -0.023, -0.0138, -0.0183, -0.0238, -0.0195]  min -0.0592 · mean -0.016 · max 0
reward/envelope_quality: [0, 0, 0, 0.00498, 0.0286, 0.0349, 0.0614, 0.0935, 0.0662, 0.102]  min 0 · mean 0.0472 · max 0.116
reward/finger_curl: [0, 0.00113, 0.00455, 0.0196, 0.0415, 0.0215, 0.0333, 0.0401, 0.0362, 0.0476]  min 0 · mean 0.0289 · max 0.0527
reward/grasp_fingers: [0, 3.19e-05, 8.92e-05, 0.00465, 0.0253, 0.026, 0.0423, 0.0579, 0.047, 0.0641]  min 0 · mean 0.0317 · max 0.0652
reward/grasp_pose: [0, 0.00444, 0.013, 0.0731, 0.127, 0.0698, 0.112, 0.133, 0.12, 0.152]  min 0 · mean 0.0939 · max 0.176
reward/grasp_thumb: [0, 5.4e-05, 8e-05, 0.00563, 0.0271, 0.0254, 0.0412, 0.0565, 0.046, 0.0641]  min 0 · mean 0.0315 · max 0.0656
reward/grasp_wrap: [0, 2.25e-05, 4.67e-05, 0.00568, 0.0319, 0.0371, 0.0595, 0.0829, 0.0632, 0.0844]  min 0 · mean 0.0431 · max 0.0884
reward/grip_squeeze: [0, 6.98e-06, 1.74e-05, 0.000946, 0.00568, 0.00623, 0.00947, 0.0132, 0.0102, 0.013]  min 0 · mean 0.007 · max 0.014
reward/hold_still: [0, 9.97e-09, 0, 2.89e-06, 3.21e-05, 1.64e-05, 0.000239, 0.000129, 0.000122, 6.59e-05]  min 0 · mean 7.54e-05 · max 0.000413
reward/lift_clear: [0, 8.83e-07, 0, 0.00237, 0.0131, 0.013, 0.163, 0.236, 0.114, 0.29]  min 0 · mean 0.108 · max 0.303
reward/lift_goal: [0, 1.28e-06, 0, 0.000471, 0.00344, 0.00253, 0.0273, 0.0248, 0.0164, 0.0163]  min 0 · mean 0.0119 · max 0.0511
reward/lift_height: [0, 4.77e-07, 0, 0.00492, 0.0185, 0.0125, 0.371, 0.626, 0.253, 0.787]  min 0 · mean 0.262 · max 0.853
reward/lift_hold: [0, 5.49e-06, 0, 0.00519, 0.0279, 0.0636, 0.044, 0.0659, 0.0587, 0.0763]  min 0 · mean 0.0376 · max 0.0781
reward/light_contact: [0, 0.000136, 0.000226, 0.00254, 0.0117, 0.0103, 0.0161, 0.0212, 0.0179, 0.0235]  min 0 · mean 0.0121 · max 0.0237
reward/palm_contact: [0, 0, 0, 0.005, 0.0255, 0.0287, 0.0521, 0.0783, 0.0562, 0.0875]  min 0 · mean 0.0409 · max 0.102
reward/palm_reach: [0, 0.000304, 0.00206, 0.027, 0.0562, 0.0388, 0.0638, 0.0767, 0.0691, 0.088]  min 0 · mean 0.0501 · max 0.102
reward/palm_table_pen: [0, 0, -1.12e-06, -5e-06, -0.00276, -0.00255, 0, 0, 0, 0]  min -0.0646 · mean -0.000577 · max 0
reward/pinky_join: [0, 1.85e-06, 5.86e-06, 0.000808, 0.00603, 0.00843, 0.0129, 0.0175, 0.0148, 0.0191]  min 0 · mean 0.00941 · max 0.0193
reward/success_bonus: [0, 0, 0, 0, 0, 0.00732, 0.011, 0.011, 0.00732, 0.00366]  min 0 · mean 0.00423 · max 0.0623
reward/table_pen: [0, -0.00169, -0.000489, -0.0516, -0.00946, -0.0012, -4.72e-05, -0.00338, -0.00224, -0.000675]  min -0.0639 · mean -0.0057 · max 0
reward/thumb_curl: [0, 0.000452, 0.00195, 0.0096, 0.0164, 0.00874, 0.0143, 0.0167, 0.0145, 0.0182]  min 0 · mean 0.0119 · max 0.0216
reward/total: [0.0293, 0.066, 0.0976, 0.174, 0.466, 0.335, 1.22, 1.76, 1.04, 2.07]  min -0.0625 · mean 0.884 · max 2.18
rewards/step: [0.412, 14.7, 62.9, 106, 206, 607, 755, 1.11e+03, 1.15e+03, 1.5e+03]  min 0.412 · mean 587 · max 1.58e+03
task/lifted_frac: [0, 0, 0, 0.0603, 0.0525, 0.448, 0.497, 0.618, 0.627, 0.691]  min 0 · mean 0.348 · max 0.721
task/successes_mean: [0, 0, 0, 0.000244, 0.00879, 0.137, 0.212, 0.21, 0.193, 0.0461]  min 0 · mean 0.0857 · max 0.345
task/tilt_deg: [0.00108, 1.09, 0.779, 5.69, 9.88, 22.3, 8.87, 8.18, 11.5, 8.5]  min 0.00096 · mean 7.61 · max 22.3
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112
