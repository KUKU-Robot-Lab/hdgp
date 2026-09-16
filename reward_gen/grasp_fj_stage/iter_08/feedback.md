## I can see from the robot that
Round 9 (`fj_stage_i08`) ran to epoch 4048 over 6.47 hours and was stopped by the user. It is the round in which the task, as the user defined it, started working from the raised start.

**The whole chain works from the raised start.** Over the last two hundred epochs, far-start episodes completed the approach 0.882 of the time, closed the envelope 0.919, lifted 0.934, and succeeded 0.764. The previous round ended with far-start success at 0.042 after collapsing from a peak of 0.370. This round did not collapse; success rose from zero at epoch 800 to 0.84 by epoch 2900 and held between 0.75 and 0.89 for the remaining twelve hundred epochs.

**The cup is now carried upright.** This was the clearest failure of the previous round, where the recording showed the cup held at thirty to forty-five degrees through most of the carry. Mean tilt ended at 5.45 degrees against 10.44 in the previous round, and the tilt penalty shrank from 0.046 to 0.007 as the policy stopped incurring it. The recording confirms it: in the frames where the cup is off the table it stands close to vertical in the hand, with the palm against its side and four fingers wrapped around the body.

**The grasp is a true envelope grasp and it improved.** At the moment of success the palm was in contact 0.767 of the time and 4.70 digits were touching. Grasp quality rose through the round to 0.439. Lift height reached 0.169 m.

**The reward moved to the end of the task, which is what the round was for.** The terms that pay for arriving at the goal and settling there went from nothing to real income: goal proximity 0.611, goal closeness 0.387, and the settling term 0.068. In the previous round that settling term never left 0.0001 across 1712 epochs; here it is 678 times larger. The success bonus ended at 0.077 against 0.0009.

**The hand never touched the table.** Palm clearance ended at 0.253 m, the palm-on-table fraction was zero from epoch 1000 onward, and the floor termination never fired.

**The wrap ratchet was tested and held.** Mean episode success crossed the threshold of 2.0 for the first time in this track, the tolerance ratchet ran 0.150 → 0.192 → 0.288 → 0.350, and it stopped exactly at the ceiling of 0.350 and stayed there for the last two thousand epochs. Two rounds ago the same ratchet ran to 0.689 and locked success out.

Four things are wrong, and they are the substance of the next round.

**The cup is set back down and picked up again within one episode.** The recording shows the cup airborne and upright in several frames and back on the table in others, with the hand beside or above it, across what appears to be a single episode window. The metrics are consistent with this: the airborne fraction ended at 0.692 while success is 0.764 and the settling term is only 0.068, so episodes are spending a large part of their time holding or re-approaching rather than finishing. Carrying the cup to the goal once and stopping should be worth more than a sequence of lifts.

**The approach posture is still abandoned the moment the latch fires, and worse than before.** The approach gate is satisfied for a single step and then the policy drives into the cup. Not touching the cup during the approach fell from 0.982 to 0.258 over the round, and the open-pose condition fell from a mid-round 0.371 to 0.063 by the end. Both are lower than the previous round finished at. The episode-level latch stays at 0.880 through all of this because it only needs one step, so the funnel never shows it. The new pre-grasp term introduced this round pays only 0.0009, which is too little to hold the shape against the grasp income waiting on the other side.

**Stage-1 income decays as the round goes on.** Travel income fell from 0.022 at epoch 1800 to 0.007 at the end, and the form term from 0.022 to 0.012. The approach still latches, so this is not yet breaking anything, but the trend is the same one that preceded the previous round's approach collapse.

**The round could not end by itself.** The two exit conditions — mean success at or above 4.0 and tolerance at or below 0.03 — moved in opposite directions: as the tolerance curriculum tightened, success fell, and the pair was never satisfied at the same time. The round ran 6.47 hours against a 4.0 hour cap because the success branch in the round policy returns before the hour check is reached. This is a loop-policy problem, not a reward problem, but it cost two hours of GPU time and needs to be recorded.

One note for the record. Tolerance tightened from 0.1125 to 0.0273 over the round, a 76 percent reduction, and the far-start success rate rose across the same span. Falls in the success count late in the round track the tightening criterion, not a regression in behaviour: grasp quality, lift height, airborne fraction and palm contact at success all rose while the success count fell.

## Feedback for improvement
Five things went right and must be kept exactly as they are.

- The approach from the raised start works and does not collapse. Far-start episodes complete it 0.88 of the time and have done so for two thousand epochs.
- The cup is carried upright. Mean tilt is 5.45 degrees against 10.44 in the previous round, and the recording shows the cup close to vertical in the hand. Making uprightness multiply the carrying income instead of sitting beside it is what did this. Keep it.
- The grasp is a true envelope grasp: palm against the side of the cup, four fingers wrapped, thumb opposing, palm in contact 0.77 of the time at the moment of success.
- The value now sits at the end of the task. Goal proximity, goal closeness and the settling term all earn real income where previously the settling term never left 0.0001. This was the whole point of the last revision and it worked.
- The hand never rests on the table, and the wrap tolerance stopped at its ceiling instead of running away.

The rest is what has to change.

**Finishing once must be worth more than lifting repeatedly.** This is the main remaining failure. The recording shows the cup lifted, set back down, and picked up again inside a single episode. The numbers agree: the cup is airborne 0.69 of the time but the settling term earns only 0.068, so episodes spend their length holding and re-approaching rather than completing. Carrying the cup to the goal and coming to rest there should be worth clearly more than the same time spent lifting it over and over. Income that accrues simply from having the cup off the table should saturate early and stop growing; what remains should be paid for arriving, slowing, and staying until the episode ends. Putting the cup down after having reached the goal should not be a way to start earning the lift income again.

**Holding the pre-grasp shape must pay enough to survive the latch.** The approach gate is satisfied for one step and then abandoned: not touching the cup during the approach fell from 0.98 to 0.26 across the round and the open-pose condition ended at 0.06, both worse than the previous round. The term added last round to hold that shape earns 0.0009, which is nothing against the grasp income waiting on the other side, so the policy pays the early-contact penalty and dives. Maintaining the open hand, the palm facing the cup, and no digit contact from the moment the approach completes until the palm is actually against the cup has to be worth enough that breaking it early is a loss. The intended order is approach, then close from the approached pose.

**Travelling in from far away must not get cheaper as the policy improves.** Income for travelling fell from 0.022 to 0.007 and the form term from 0.022 to 0.012 over the round. Nothing is broken yet because the latch still fires, but this is the same decay that preceded the approach collapse two rounds ago. Whatever pays for closing the distance in the correct shape should keep paying at the same rate late in training as early.

One caution about reading this round. The tolerance tightened by 76 percent while the round ran, so the success count falling late in the round is the criterion moving, not the behaviour getting worse — grasp quality, lift height, airborne fraction and palm contact at success all rose over the same span. Do not respond to that fall by making success easier to obtain.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.02, 0.399, 0.595, 0.508, 0.597, 0.618, 0.673, 0.584, 0.67]  min 0 · mean 0.477 · max 0.807
contact/finger_index_at_success: [-1, 0.54, 0.873, 0.818, 0.98, 0.985, 1, 0.999, 0.997, 0.993]  min -1 · mean 0.824 · max 1
contact/finger_middle: [0, 0.019, 0.385, 0.599, 0.449, 0.492, 0.502, 0.518, 0.4, 0.489]  min 0 · mean 0.378 · max 0.683
contact/finger_middle_at_success: [-1, 0.926, 0.87, 0.842, 0.864, 0.815, 0.863, 0.802, 0.802, 0.833]  min -1 · mean 0.739 · max 1
contact/finger_pinky: [0, 0.0232, 0.169, 0.642, 0.497, 0.585, 0.534, 0.598, 0.548, 0.645]  min 0 · mean 0.433 · max 0.799
contact/finger_pinky_at_success: [-1, 0.224, 0.188, 0.861, 0.979, 0.989, 0.974, 0.996, 0.974, 0.994]  min -1 · mean 0.704 · max 1
contact/finger_ring: [0, 0.0195, 0.362, 0.655, 0.478, 0.602, 0.607, 0.666, 0.585, 0.668]  min 0 · mean 0.476 · max 0.811
contact/finger_ring_at_success: [-1, 0.828, 0.913, 0.922, 0.929, 0.977, 0.992, 0.999, 0.998, 0.998]  min -1 · mean 0.863 · max 1
contact/finger_thumb: [0, 0.0212, 0.467, 0.696, 0.527, 0.63, 0.627, 0.68, 0.6, 0.684]  min 0 · mean 0.504 · max 0.82
contact/finger_thumb_at_success: [-1, 1, 0.968, 0.977, 0.995, 0.998, 1, 1, 1, 1]  min -1 · mean 0.91 · max 1
contact/fingers_touching: [0, 0.103, 1.78, 3.19, 2.46, 2.91, 2.89, 3.13, 2.72, 3.16]  min 0 · mean 2.27 · max 3.8
contact/fingers_touching_at_success: [-1, 3.52, 3.81, 4.42, 4.75, 4.76, 4.83, 4.79, 4.77, 4.82]  min -1 · mean 4.21 · max 4.88
contact/link_force_mean: [0, 0.0809, 1.46, 2.26, 1.65, 2.12, 2.17, 2.24, 1.9, 2.26]  min 0 · mean 3.08 · max 5.85e+03
contact/links_touching: [0, 0.118, 2.11, 4.13, 3.1, 3.69, 3.58, 3.91, 3.4, 4.01]  min 0 · mean 2.88 · max 4.95
contact/links_touching_at_success: [-1, 4.64, 4.78, 5.87, 5.9, 6.2, 6.04, 5.92, 6.02, 6.12]  min -1 · mean 5.42 · max 6.58
contact/palm_touching: [0, 0.00537, 0.309, 0.473, 0.428, 0.413, 0.425, 0.418, 0.456, 0.502]  min 0 · mean 0.351 · max 0.586
contact/palm_touching_at_success: [-1, 0.367, 0.874, 0.776, 0.758, 0.65, 0.752, 0.582, 0.76, 0.722]  min -1 · mean 0.611 · max 0.999
ctrl/prev_ep_successes_mean: [0, 0.000244, 0.00269, 0.696, 3.69, 3.44, 3.99, 4, 4.35, 3.95]  min 0 · mean 2.58 · max 4.55
done/abnormal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.3e-05 · max 0.00195
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.53e-06 · max 0.000488
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.65e-06 · max 0.000488
done/max_goals: [0, 0, 0, 0.000488, 0.000977, 0.00195, 0.00146, 0.00122, 0.00171, 0.000732]  min 0 · mean 0.000976 · max 0.00415
done/out_xy: [0, 0, 0.000244, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.56e-05 · max 0.000977
done/tipped: [0, 0, 0.000244, 0.000977, 0.000244, 0.000244, 0.000488, 0, 0.000244, 0.000244]  min 0 · mean 0.00037 · max 0.00464
episode_lengths/step: [43.5, 654, 552, 700, 467, 517, 473, 477, 471, 617]  min 43.5 · mean 551 · max 899
reward/action_rate_pen: [-0.00268, -0.00204, -0.0016, -0.00138, -0.000995, -0.000844, -0.000694, -0.000634, -0.000515, -0.000483]  min -0.00268 · mean -0.00109 · max -0.000462
reward/approach_form: [0.0177, 0.0315, 0.0205, 0.0176, 0.024, 0.0203, 0.0225, 0.0205, 0.0171, 0.0155]  min 0.00725 · mean 0.0218 · max 0.0682
reward/approach_keep: [0.000838, 0.000651, 0.00563, 0.00451, 0.00638, 0.00417, 0.00327, 0.00165, 0.00161, 0.00173]  min 1.67e-05 · mean 0.00304 · max 0.0151
reward/approach_lock: [0, 1.69e-13, 0.00209, 0.00128, 0.00227, 0.00141, 0.00218, 0.00153, 0.00154, 0.00107]  min 0 · mean 0.0014 · max 0.00624
reward/approach_travel: [0.0135, 0.0127, 0.0221, 0.0178, 0.0249, 0.0189, 0.0162, 0.011, 0.01, 0.00982]  min 0.00403 · mean 0.0155 · max 0.0521
reward/arm_vel_pen: [-0.000144, -0.000257, -0.00162, -0.00147, -0.00113, -0.000996, -0.001, -0.000903, -0.000884, -0.00105]  min -0.00251 · mean -0.000995 · max -0.000136
reward/cup_disturb_pen: [0, -0.00116, -0.00682, -0.0103, -0.00435, -0.00591, -0.0119, -0.00947, -0.00203, -0.00375]  min -0.0259 · mean -0.00498 · max 0
reward/cup_tilt_pen: [0, -0.00578, -0.0353, -0.0551, -0.0154, -0.0249, -0.0209, -0.0223, -0.00811, -0.00994]  min -0.067 · mean -0.0172 · max 0
reward/early_contact_pen: [0, -0.00549, -0.00969, -0.021, -0.00808, -0.0197, -0.0247, -0.0292, -0.00708, -0.0144]  min -0.0489 · mean -0.0118 · max 0
reward/envelope_quality: [0, 0.000406, 0.0351, 0.0647, 0.0917, 0.0856, 0.0859, 0.0878, 0.0951, 0.105]  min 0 · mean 0.0706 · max 0.134
reward/finger_curl: [0, 0.00178, 0.024, 0.0351, 0.0442, 0.0377, 0.0404, 0.045, 0.0423, 0.0508]  min 0 · mean 0.0349 · max 0.0671
reward/goal_close: [0, 2.9e-05, 0.00443, 0.0164, 0.0763, 0.0938, 0.113, 0.172, 0.262, 0.328]  min 0 · mean 0.13 · max 0.448
reward/goal_in_tol: [0, 0, 0.000241, 0.00219, 0.0473, 0.0395, 0.0462, 0.0574, 0.0948, 0.0873]  min 0 · mean 0.0436 · max 0.124
reward/goal_near: [0, 8.6e-08, 0.00029, 0.0038, 0.0627, 0.093, 0.132, 0.247, 0.394, 0.511]  min 0 · mean 0.182 · max 0.712
reward/grasp_fingers: [0, 0.000633, 0.0243, 0.0431, 0.0532, 0.0551, 0.0493, 0.0562, 0.0556, 0.0639]  min 0 · mean 0.0438 · max 0.0838
reward/grasp_pose: [0, 0.01, 0.109, 0.114, 0.15, 0.12, 0.135, 0.148, 0.147, 0.165]  min 0 · mean 0.118 · max 0.207
reward/grasp_thumb: [0, 0.000554, 0.0258, 0.0429, 0.0533, 0.0553, 0.0495, 0.0563, 0.0561, 0.0641]  min 0 · mean 0.0441 · max 0.0836
reward/grasp_wrap: [0, 0.000657, 0.0346, 0.0621, 0.0756, 0.0736, 0.0635, 0.0686, 0.0705, 0.0853]  min 0 · mean 0.0586 · max 0.115
reward/grip_squeeze: [0, 0.000105, 0.00548, 0.0098, 0.0115, 0.0125, 0.0113, 0.0129, 0.0125, 0.0147]  min 0 · mean 0.01 · max 0.0193
reward/hold_still: [0, 2.13e-08, 1.1e-05, 0.00027, 0.00453, 0.00897, 0.013, 0.0262, 0.0359, 0.0543]  min 0 · mean 0.0181 · max 0.0831
reward/lift_clear: [0, 1.56e-05, 0.00729, 0.0275, 0.0652, 0.0599, 0.0618, 0.0692, 0.0953, 0.105]  min 0 · mean 0.0567 · max 0.132
reward/lift_hold: [0, 0.000387, 0.0187, 0.0322, 0.034, 0.0323, 0.033, 0.0339, 0.0396, 0.0422]  min 0 · mean 0.0283 · max 0.0505
reward/light_contact: [0, 0.000595, 0.00958, 0.0161, 0.0189, 0.0204, 0.0175, 0.0208, 0.02, 0.0232]  min 0 · mean 0.0161 · max 0.0316
reward/palm_contact: [0, 0.000493, 0.0324, 0.0568, 0.0833, 0.0689, 0.0772, 0.0778, 0.0834, 0.0945]  min 0 · mean 0.0621 · max 0.12
reward/palm_reach: [0, 0.00296, 0.0567, 0.0635, 0.0845, 0.0683, 0.0753, 0.084, 0.0828, 0.0946]  min 0 · mean 0.0659 · max 0.119
reward/palm_table_pen: [0, -3.85e-08, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0404 · mean -0.000242 · max 0
reward/pinky_join: [0, 0.000149, 0.00439, 0.0126, 0.0158, 0.0148, 0.0133, 0.0146, 0.0153, 0.0175]  min 0 · mean 0.0118 · max 0.0238
reward/pregrasp_shape: [0, 0.000446, 0.00285, 0.00177, 0.00223, 0.00138, 0.00171, 0.00126, 0.00147, 0.00092]  min 0 · mean 0.00141 · max 0.0138
reward/success_bonus: [0, 0, 0, 0.0256, 0.114, 0.0696, 0.0952, 0.0916, 0.11, 0.0769]  min 0 · mean 0.0717 · max 0.209
reward/table_pen: [0, -0.00524, -0.0195, -0.000512, -0.00141, -3.66e-05, -0.000171, -0.000789, -2.93e-05, -0.000131]  min -0.0898 · mean -0.00256 · max 0
reward/thumb_curl: [0, 0.000568, 0.0109, 0.0129, 0.0162, 0.014, 0.0149, 0.0172, 0.0165, 0.0194]  min 0 · mean 0.0133 · max 0.0254
reward/total: [0.0293, 0.0446, 0.382, 0.595, 1.13, 1.02, 1.11, 1.36, 1.74, 2]  min -0.0698 · mean 1.08 · max 2.61
rewards/step: [1.36, 10.3, 168, 333, 525, 594, 469, 681, 847, 1.2e+03]  min -11 · mean 564 · max 1.82e+03
task/lifted_frac: [0, 0.000732, 0.0945, 0.469, 0.413, 0.529, 0.543, 0.618, 0.524, 0.63]  min 0 · mean 0.391 · max 0.787
task/successes_mean: [0, 0, 0.00171, 0.183, 0.513, 0.599, 0.576, 0.439, 0.446, 0.657]  min 0 · mean 0.381 · max 1.01
task/tilt_deg: [0.00108, 1.39, 8.9, 13.2, 6.33, 8.32, 7.74, 8.18, 5.34, 6.06]  min 0.000831 · mean 6.22 · max 15.6
task/tol: [0.112, 0.112, 0.112, 0.112, 0.101, 0.082, 0.0664, 0.0484, 0.0392, 0.0318]  min 0.0257 · mean 0.0774 · max 0.112
