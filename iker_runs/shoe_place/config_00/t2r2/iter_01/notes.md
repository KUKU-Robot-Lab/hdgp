All numbers below were measured, not estimated: a deterministic rollout (observation and action noise off)
of the trained policy from the previous reward, 64 environments, first episode of each, 173 steps on average.
The four predicate flags and the grip state were read directly out of the environment on every step.

Gate occupancy, as a fraction of all rollout steps (and the fraction of environments that reached it at least
once during their episode):

  placed   (keypoint_dist <= 0.05)                 0.330 of steps   0.719 of environments
  resting  (|shoe_bottom_z - rack_top_z| <= 0.01)  0.486 of steps   0.984 of environments
  placed AND resting                               0.299 of steps   0.719 of environments
  released (palm_shoe_dist > 0.15)                 0.043 of steps   0.219 of environments
  still    (lin. and ang. speed < 0.05)            0.017 of steps   0.047 of environments
  all four at once                                 0.000 of steps   0.000 of environments

The consecutive-success counter never left zero: its maximum over the whole rollout was 0 in every one of the
64 environments, against the 20 consecutive steps the task requires.

The grip axis, measured as open_frac = clamp((grip_norm + 1) / 2, 0, 1), where 0 is the grasp-bank grip pose
and 1 is the open pose:

  mean over all steps                              0.014
  mean over the steps where placed AND resting     0.000
  per-environment maximum, mean over environments  0.130
  per-environment maximum, 90th percentile         0.598

So the state in which the previous reward paid for opening the hand was reached on 30% of all steps and by 72%
of environments, and in those states the policy's grip stayed shut. The policy did not fail to reach the
release condition; it reached it constantly and did not open.

Two structural facts that follow from the definitions and hold regardless of the policy:

  1. `released` is palm-to-shoe distance above 0.15 m, so it cannot become true while the hand still holds the
     shoe. It is a consequence of opening the grip, not an independent condition the policy can satisfy.
  2. `still` requires the shoe's linear and angular speed to be below 0.05. A shoe held in a hand that is
     still moving is almost never still: this is why it sits at 0.017 while `resting` sits at 0.486.

Consequently the four-way conjunction has no path to becoming true until the grip opens, and the 20-step
counter cannot start. Over training, the components that pay without ever opening the hand (align and seat)
rose from 0.13 and 0.001 to 0.58 and 0.25, while release fell to 0.0001 and withdraw stayed at exactly 0.
Policy entropy fell from 9.87 to 1.83 over the same run.
