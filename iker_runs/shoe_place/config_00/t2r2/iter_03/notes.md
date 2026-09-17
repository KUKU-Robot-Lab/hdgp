All numbers below were measured, not estimated. Sources: the training logs of the policy trained by the
previous reward (349 epochs; the run ended there), a video of that policy watched by a person, and a
deterministic probe of the arm's rest posture.

WHAT THE PREVIOUS REWARD ACHIEVED (final logged epoch)

  sustained success 0.235     success within 5 cm 0.410     keypoint distance 0.157 m
  dropped 0.105               placed 0.045 (at the 0.03 m tolerance)
  resting 0.617               still 0.803                   released 0.882

Carrying the shoe to the rack, setting it down and letting go are learned. Do not rebuild them.

WHAT WENT WRONG: THE ARM IS LIFTED AFTER LETTING GO

A person watching the trained policy saw that, once the shoe is set down and released, the arm is raised
upward into a "hands up" pose instead of being brought back.

  - The previous reward had no term that depended on where the hand goes after letting go. Nothing in it
    read the hand's position relative to any rest location, and the context did not offer one.
  - `released` only asks that the palm be more than 0.15 m from the shoe, in any direction.
  - place/retreated (the hand ending within 0.15 m of where it started the episode) fell from 0.79 at the
    first epoch to 0.054 at the last.

WHAT CHANGED IN THE ENVIRONMENT SINCE THAT RUN

A fifth success condition, `ctx.home`, has been added (see the task description above). The hand must be back
at `ctx.home_palm_pos` — where the palm sits when the arm is in its default rest posture — for the steps to count.
The previous policy's final behaviour (arm raised) therefore now earns no success at all.

Measured facts about that rest position:

  home_palm_pos (env-local)                              (0.290, 0.380, 0.418) m, identical in every env
  distance from home_palm_pos to the target shoe centre  0.447 m   (so `released` and `home` can both hold)
  distance from the episode-start palm to home_palm_pos  0.225 m (median)
  driving the palm toward home_palm_pos with a simple proportional action from the episode start:
      within 0.05 m after 9 steps (median); final position error 0.010 m median, 0.032 m 90th percentile

Only the palm POSITION is judged. The arm's joint angles do not return to the rest posture when the palm does
(maximum joint error 1.01 rad median in the same probe), because the action commands the palm pose and the
7-joint arm has a free redundant direction. A reward term on joint angles would pay for something the policy
cannot control.

THE SUCCESS TERM IN THE PREVIOUS REWARD

It paid 40 + 12 per remaining step when success fired, and by the last epoch it was the largest component
(3.24 of a total of 4.89). With the home condition added, success can only fire after the hand has travelled
back about 0.45 m from the shoe, so it arrives later in the episode than before.
