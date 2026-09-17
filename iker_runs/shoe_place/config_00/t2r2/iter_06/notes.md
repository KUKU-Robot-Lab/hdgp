All numbers below were measured, not estimated. Sources: the training log of the policy trained by the previous
reward (600 epochs, finished), and deterministic probes of that policy (64 environments, noise off, first episode of
each environment). Training for this round CONTINUES from that policy's weights. The environment did not change.

WHAT THE PREVIOUS REWARD ACHIEVED

Deterministic success (20 of the last 30 steps with all five conditions) by checkpoint of the previous run:

  epoch 200   0.531
  epoch 400   0.563
  epoch 600   0.578

For reference, the policy it started from scored 0.531 under the same probe. Training-log success rose from 0.37 to
0.41 (100-epoch averages) and was flat over the last 300 epochs.

At epoch 600: every episode in which the shoe was ever placed within 0.03 m went on to succeed except 3 of 64
(ever placed 0.625, success 0.578). The scripted return works: final palm distance to home 0.0005 m median.
Carrying, seating, opening the hand and triggering the takeover are learned. Do not rebuild them.

HOW THE REMAINING EPISODES FAIL (epoch 600, 27 of 64 episodes; state at the end of the episode)

  shoe resting on the rack, hand released, but keypoint_dist > 0.03 m      0.3125 of all episodes
      final keypoint_dist 0.031 / 0.040 / 0.081 m (10th / 50th / 90th percentile)
      smallest keypoint_dist during the episode 0.030 / 0.038 / 0.063 m
      these episodes run to the time limit (199 steps); the shoe stays where it was set down
  released, shoe not resting on the rack                                  0.031
  dropped below the table                                                 0.031
  still holding the shoe at the time limit                                0.047

At epoch 400 the same probe gave 0.375 for the first class with a median final error of 0.054 m, and 92 % of those
episodes never had keypoint_dist within 0.03 m at any step. So the dominant failure is: the shoe is set down a few
centimetres off the target and is never moved again.

STRUCTURAL FACTS ABOUT THAT STATE

  - The takeover requires `ctx.placed`. While keypoint_dist > 0.03 m the policy keeps control of the arm and the
    grip for the rest of the episode, so it can go back to the shoe and push or re-grasp it.
  - In the previous reward, `settle` (1.2 x calm) and `seat` (0.5) are paid whenever keypoint_dist <= 0.09 m and the
    shoe rests on the rack, whether or not it is placed. Over the last 50 epochs `settle` averaged 0.614 per step,
    the largest term after `success`; `seat` averaged 0.189. A shoe left motionless 0.04 m from the target keeps
    collecting both terms every step; touching it again makes it move, which lowers `settle`.

WHAT THE PERSON RUNNING THIS WANTS

After the shoe is set down, if it is not on the target, the policy should touch the shoe and move it onto the target
(adjust only the shoe it placed). Reaching the target with the first set-down is equally fine.
