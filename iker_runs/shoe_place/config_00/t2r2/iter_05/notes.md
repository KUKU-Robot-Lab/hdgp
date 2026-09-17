All numbers below were measured, not estimated. Sources: the training logs of the policy trained by the previous
reward (661 epochs, stopped on purpose), deterministic probes of that policy (64 environments, noise off), and a
person watching a video of it. Training for this round CONTINUES from that policy's weights.

WHAT THE PREVIOUS REWARD ACHIEVED

Logged at the last epoch: sustained success 0.277 (max 0.367), placed 0.212, resting 0.664, still 0.812, dropped
0.111. A deterministic probe of the epoch-500 policy: final keypoint error median 0.028 m (10th percentile 0.018,
90th percentile 0.100); the shoe ends 0.150 m (median) from the other shoe against a target of 0.136 m.
Carrying the shoe, setting it on the target and opening the hand are learned. Do not rebuild them.

WHY THE ENVIRONMENT CHANGED

The previous rounds judged "home" by the palm POSITION only. The trained policy put its palm at that position with
the arm in a completely different shape: with the palm within 0.05 m of home, the arm joints differed from the rest
posture by 34, 5, 54, 40, 25, 25 and 62 degrees (medians per joint) and the palm orientation by 126 degrees. A
person watching the video saw the arm splayed out in an unnatural pose. The policy cannot fix this: its action
commands the palm pose, and the 7-joint arm has a free redundant direction it does not control.

So the return is no longer learned. As the task description above states, the environment now takes over the arm
the first step the shoe is placed, resting and still with the grip open at least 0.9, opens the hand fully and
moves the arm joints to the rest posture over 20 steps; `ctx.home` now means every arm joint within 0.035 rad of
the rest posture; `ctx.retracting` is True from the takeover on, and the policy's actions have no effect then.

Consequences, all structural facts:
  - The previous reward's `home_return` and `home_hold` terms paid for moving the palm toward home. From now on the
    only way the arm gets home is the takeover; nothing the policy does after the takeover changes any reward.
  - The previous reward read `ctx.home_radius`, which no longer exists (it is now `ctx.home_joint_tol`).
  - The one thing the policy must do to trigger the return is: shoe placed (keypoint_dist <= 0.03 m), resting,
    still, AND grip open fraction (`(ctx.grip_norm + 1) / 2`) >= 0.9, all on the same step.

MEASURED UNDER THE NEW ENVIRONMENT (the previous policy, never trained with the takeover, zero reward, 64 envs)

  success (20 of the last 30 steps, all five conditions)   0.531 of episodes
  arm reached home at least once                            0.531 of episodes (every one of them succeeded)
  final palm distance to home                               0.0006 m median
  shoe placed on at least one step                          0.578 of episodes
  open fraction on steps where placed and resting held      0.985
  final keypoint error                                      0.019 / 0.030 / 0.109 m (10th / 50th / 90th percentile)

A separate probe of the takeover itself (64 envs): the shoe moved at most 0.0009 m while the arm withdrew, the palm
stayed at least 0.08 m from the shoe, and the arm ended within 0.13 degrees of the rest posture. The takeover does
not disturb a correctly placed shoe.

So the remaining failures are episodes where the shoe was never placed within 0.03 m and still with the hand open:
42% of episodes never reached `placed` at all.
