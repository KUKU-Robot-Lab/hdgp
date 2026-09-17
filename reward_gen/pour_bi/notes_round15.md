Round 15 (iter_14, t2r_i14, 4096 env, warm-started from the round-14 checkpoint, 557 epochs / 4.1 h, ADR level 0) — measured facts only, environment unchanged. episode_success 0.0 and bead/in_target 0.0 for the whole round; bead/spill 0.017; reward/pour_delta, reward/pour_pose, reward/hold_src, reward/hold_rcv, reward/success were 0.0.

Grasp/lift: task/src_grasped 0.80, task/rcv_grasped 0.81 (round 14 ended at 0.83/0.82), src cup lift 0.193 m, rcv cup lift 0.081 m, task/rcv_tilt_deg 6.8.

Approach and tilt (env mean, epoch: task/pour_lip_dist / task/cups_center_dist / task/src_tilt_deg): first epoch 0.211 / 0.279 / 9.2 deg; 288: 0.060 / 0.193 / 26.2 deg; 557: 0.053 / 0.184 / 32.2 deg (max of the logged mean 37.5 deg). task/aim_dist 0.149 m. Unlike round 14, the lip stayed inside the 0.081 m latch radius on average and did not drift back out. task/tilt_limit_deg averaged 60.5, so iter_14's theta_pre (limit - 10 deg) is about 50 deg; the mean source tilt stayed about 18 deg below it for the whole round.

Tilt terms: reward/pretilt 0 -> 2.07 (max 2.65, weight 4.0), reward/tilt 0.004 (max 0.006, weight 4.0). In iter_14 code `tilt` pays only for theta_eff above theta_pre and multiplies over * lip_above * z_ok * stack_gate; `pretilt` pays linearly 0 -> theta_pre.

Contact and other indicators: task/cup_collision_rate 0.064 (round 14 end 0.037), hand_foreign src/rcv 0.184/0.093 (round 14 end 0.132/0.081), task/nested_rate 0.015, task/premature_tilt_rate 0.008 (max 0.037), task/tilt_far_deg 22.7, task/rcv_palm_speed 0.39 m/s, bead/fill_level 0.75, dr/bead_active_hi 12. Penalty terms at the end: cup_contact -0.056, hand_foreign_src -0.115, hand_foreign_rcv -0.060, rcv_still -0.158, rot_speed -0.053.

Best so far remains iter_05 (round-3 environment, not comparable).
