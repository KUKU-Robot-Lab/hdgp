## I can see from the robot that
Training was stopped early, after about 180 of 20,000 epochs, because the hand was not approaching the cup; the playback uses the best checkpoint of that run.
In the viewed environment the hand never moves towards the cup. Within the first 1.5 s the arm lowers the hand beside the robot column, next to the table edge closest to the robot, and turns the hand away from its start orientation: the fingers no longer point forward over the table (+x) but down towards the table, and the wrist keeps rotating there for the rest of the episode. At the end of the episode the hand is still next to the robot, far from the cup.
The fingers and the thumb curl in from the first second, so the hand does not keep its default pose during what should be the approach.
The cup is never touched and stays upright.
Metrics:
- The approach flag (ctx.approach_done) was set in 0 of the finished episodes up to epoch 110.
- The palm-to-band distance averaged 0.26 m at epoch 1, 0.28 m at epoch 17, 0.30 m at epoch 55 and 0.23 m at epoch 110. In 56 % of the finished episodes the palm centre came within 5 cm of the band at some step, but none of them met the approach flag.
- approach_face rose from 0.19 at epoch 1 to 0.36 at epoch 17, while approach_reach fell from 0.111 to 0.098 over the same epochs; at epoch 110 they were 0.37 and 0.16.
- approach_pose_keep was 0.000 at every logged epoch.
- In the start pose the palm normal points along +y and the fingers along +x; the horizontal direction from the palm centre to the cup axis is about 68 degrees away from the palm normal there, so approach_face paid about 0.19 before the hand moved.

## Feedback for improvement
The approach went wrong from the first steps: the policy turned the hand instead of moving it to the cup.
- The hand already starts with the orientation it needs for the grasp: palm normal along +y towards the cup's side and fingers along +x. The approach should be a movement of the palm to the cup's -y side while this start orientation is kept the whole way. The approach reward asked the palm to face the cup axis from wherever the hand was; from the start pose that means turning the wrist by about 68 degrees, and the policy learned that turn first while the hand moved away from the cup. Reward keeping the start orientation (ctx.palm_normal along +y, ctx.palm_finger_dir along +x) during the approach rather than turning the palm towards the cup axis from a distance.
- The approach target was the nearest point of the cup's band, which from the start pose lies on a diagonal side of the cup. The target should be the cup's -y side of the band. The environment's approach flag now also requires the palm on the cup's -y side and the start orientation of the hand.
- The default hand pose was never kept: approach_pose_keep gave no reward at any point of training, and the fingers curled from the first second. Keeping the default pose needs a signal that changes over the finger deviations the policy actually produces, instead of one that is already at zero as soon as any finger joint, actual or commanded, moves a little.
- Getting closer to the cup earned less than turning the palm (approach_reach 0.10 to 0.16 against approach_face 0.19 to 0.37), so moving the palm to the cup's side must be the part of the approach that pays most.
- Stages 2 and 3 were never reached, so there is no feedback on them yet; keep their order and their dependence on the stage flags.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.28e-05 · max 0.000488
contact/finger_index_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_middle: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000244]  min 0 · mean 6.42e-06 · max 0.000244
contact/finger_middle_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_pinky: [0, 0, 0, 0, 0, 0, 0, 0.0479, 0.00122, 0.00244]  min 0 · mean 0.00509 · max 0.062
contact/finger_pinky_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_ring: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000488]  min 0 · mean 0.000108 · max 0.00171
contact/finger_ring_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/finger_thumb: [0, 0, 0, 0, 0, 0, 0, 0.000244, 0, 0.00122]  min 0 · mean 0.000278 · max 0.00391
contact/finger_thumb_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/fingers_touching: [0, 0, 0, 0, 0, 0, 0, 0.0481, 0.00122, 0.00439]  min 0 · mean 0.0055 · max 0.0625
contact/fingers_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/link_force_mean: [0, 0, 0, 0, 0, 0, 0, 0.0165, 0.002, 0.025]  min 0 · mean 0.00248 · max 0.0276
contact/links_touching: [0, 0, 0, 0, 0, 0, 0, 0.0491, 0.00122, 0.00464]  min 0 · mean 0.00567 · max 0.064
contact/links_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
contact/palm_touching: [0, 0, 0, 0, 0, 0, 0, 0.00146, 0, 0]  min 0 · mean 0.000149 · max 0.00391
contact/palm_touching_at_success: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  min -1 · mean -1 · max -1
ctrl/prev_ep_successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/abnormal: [0, 0.000244, 0.000488, 0, 0, 0, 0, 0, 0.000488, 0.000488]  min 0 · mean 0.000315 · max 0.00317
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/hand_floor: [0, 0, 0, 0.000732, 0, 0, 0.000244, 0, 0.000244, 0]  min 0 · mean 0.000104 · max 0.00122
done/max_goals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.61e-06 · max 0.000244
done/tipped: [0, 0, 0, 0, 0, 0, 0, 0.00586, 0, 0.000244]  min 0 · mean 0.000397 · max 0.00659
episode_lengths/step: [61, 184, 294, 530, 899, 890, 878, 839, 817, 707]  min 61 · mean 627 · max 899
reward/action_rate: [-0.111, -0.109, -0.102, -0.0991, -0.105, -0.0901, -0.0922, -0.0863, -0.0829, -0.0802]  min -0.112 · mean -0.0936 · max -0.075
reward/approach_face: [0.189, 0.34, 0.487, 0.228, 0.23, 0.472, 0.278, 0.269, 0.478, 0.432]  min 0.0597 · mean 0.349 · max 0.491
reward/approach_finger_contact: [0, 0, 0, 0, 0, 0, 0, -0.0175, -0.000453, -0.00145]  min -0.0222 · mean -0.00197 · max 0
reward/approach_pose_keep: [4.17e-08, 2.09e-07, 1.14e-08, 0.000117, 6.37e-09, 2.72e-05, 5.35e-05, 0.00041, 0.000114, 0.000146]  min 7.28e-12 · mean 8.49e-05 · max 0.000717
reward/approach_reach: [0.111, 0.1, 0.0609, 0.0627, 0.0937, 0.0611, 0.136, 0.227, 0.158, 0.216]  min 0.0549 · mean 0.136 · max 0.324
reward/cup_disturb: [0, 0, 0, 0, 0, -9.52e-05, -7.79e-05, -0.0429, -0.000682, -0.00334]  min -0.0565 · mean -0.00453 · max 0
reward/cup_tilt: [0, 0, 0, 0, 0, -0.000232, -0.000248, -0.113, -0.000871, -0.00671]  min -0.158 · mean -0.0107 · max 0
reward/envelope_closure: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_digit_contact: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_face: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_opposition: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_palm_contact: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_stay: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/envelope_wrap: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_goal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_grip_hold: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_hold_still: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_path_dev: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/stage_base: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/table_clearance: [0, 0, 0, -0.0358, 0, -0.000233, -0.00476, -0.000563, -0.00222, -0.00215]  min -0.0476 · mean -0.00491 · max 0
reward/total: [0.189, 0.331, 0.446, 0.156, 0.219, 0.443, 0.316, 0.236, 0.549, 0.554]  min 0.0293 · mean 0.369 · max 0.605
rewards/step: [7.69, 37.8, 77.5, 157, 251, 248, 246, 243, 251, 235]  min 7.69 · mean 188 · max 260
task/lifted_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.12e-05 · max 0.000488
task/successes_mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/tilt_deg: [0.00108, 0.00106, 0.00118, 0.00108, 0.00111, 0.0094, 0.00753, 4.1, 0.029, 0.25]  min 0.000912 · mean 0.377 · max 5.45
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]  min 0.112 · mean 0.112 · max 0.112
