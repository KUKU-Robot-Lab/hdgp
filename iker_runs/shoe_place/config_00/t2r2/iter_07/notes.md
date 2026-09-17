All numbers below were measured, not estimated. Sources: training logs, and deterministic probes (64 environments,
noise off, first episode of each environment). The environment did not change.

HISTORY OF THE LAST TWO REWARDS

The reward before the previous one (call it R5) was trained 600 epochs. Its final policy (P5) is the best so far.
The previous reward (shown above as the previous code, R6) added `off_target`, `push` and `reach` and scaled `seat`
and `settle` by closeness. It was trained from P5 and STOPPED at epoch 200 because the deterministic probe got worse:

                                               P5 (epoch 600)    R6 policy (epoch 200)
  success                                       0.52 - 0.58       0.297
  failure: resting on rack, released, off target    0.36          0.39
  failure: still holding the shoe at the time limit 0.09          0.16
  failure: released, shoe not resting on the rack   0.03          0.13
  failure: dropped below the table                  0.00          0.03

Probe repeatability: the same P5 checkpoint probed twice gave 0.578 and 0.516, so differences under about 0.07 are
not meaningful. 0.297 is far outside that.

During the 200 epochs of R6 the training-log averages of `push` stayed at 0.006 - 0.009 and `reach` at 0.002 per
step, flat from the first epoch; `off_target` stayed at -0.38. The penalty was paid, but no behaviour that reduces it
was ever found.

WHY `reach` AND `push` NEVER FIRED (measured state of the off-target failures at the end of the episode)

  distance from the palm to the nearest point of the shoe (`ctx.palm_gap`), 10th / 50th / 90th percentile:
      P5:              0.238 / 0.297 / 0.347 m
      R6 policy:       0.159 / 0.255 / 0.409 m
  keypoint_dist of the same episodes: P5 0.033 / 0.043 / 0.078 m; R6 policy 0.034 / 0.056 / 0.084 m

After letting go of a shoe that is not placed, the policy moves its hand 0.25 - 0.30 m away from the shoe and keeps
it there. At that distance R6's `reach` = 0.5 * (1 - tanh(palm_gap / 0.05)) is below 0.0001, and `push` needs the
shoe to move, which needs the hand on it. So neither term gave any signal in the states the policy actually visits.

WHAT STAYS TRUE FROM BEFORE

  - Episodes in which the shoe is ever placed within 0.03 m almost all succeed; the scripted return works (final palm
    distance to home 0.0005 m median). Carrying, seating, opening the hand and triggering the takeover are learned.
  - The takeover requires `ctx.placed`. While the shoe is not placed the policy keeps control of the arm and grip for
    the rest of the episode.

TRAINING FOR THIS ROUND STARTS FROM P5's WEIGHTS (not from the R6 policy).

WHAT THE PERSON RUNNING THIS WANTS

After the shoe is set down, if it is not on the target, the policy should touch the shoe and move it onto the target
(adjust only the shoe it placed). Reaching the target with the first set-down is equally fine.
