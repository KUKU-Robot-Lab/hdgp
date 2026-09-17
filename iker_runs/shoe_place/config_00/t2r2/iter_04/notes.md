All numbers below were measured or computed directly from the previous reward's code, not estimated.
The previous reward was trained for 218 epochs and stopped there on purpose. Training for this round
CONTINUES from that policy's weights, so everything it learned is kept; only the reward changes.

WHAT THE PREVIOUS REWARD ACHIEVED (last logged epoch; max over training in brackets)

  placed 0.378 (0.393)    success within 5 cm 0.564 (0.633)    keypoint distance 0.142 m
  resting 0.793           still 0.866                          dropped 0.093
  release component 0.299 (0.310), still rising

Carrying the shoe, setting it on the target and opening the hand are learned, faster than in any previous round.
Do not rebuild them.

WHAT DID NOT HAPPEN: THE HAND NEVER GOES HOME

  place/home          0.0001 last, 0.002 max over the whole run
  home_hold           0 every epoch
  success             0 every epoch
  home_return         0.0040 last, 0.0041 max — flat from epoch 150 on, while the release component that uses
                      a similar gate grew from 0.03 to 0.30 over the same epochs

WHY: THE RETURN TERM IS NEARLY FLAT WHERE THE HAND ACTUALLY IS

Geometry (measured): home_palm_pos is 0.447 m from the target shoe centre, and `released` requires the palm to be
more than 0.15 m from the shoe. So at the moment the return should begin, palm_home_dist is roughly 0.3 to 0.6 m.

The previous home_return shape was 0.40*(1 - tanh(d/0.25)) + 0.30*(1 - tanh(d/0.10)) + 0.30*exp(-(d/0.05)^2),
multiplied by 2.5 and by its gate. Its value, and how much it grows when the palm moves 0.02 m closer (the
largest move one step allows):

  palm_home_dist 0.60 m   value 0.016   gain for a 0.02 m step 0.003
  palm_home_dist 0.45 m   value 0.053   gain for a 0.02 m step 0.009
  palm_home_dist 0.30 m   value 0.170   gain for a 0.02 m step 0.028
  palm_home_dist 0.10 m   value 0.813   gain for a 0.02 m step 0.188

The gain is 20 to 60 times smaller over the first 0.3 m of the return than over the last 0.1 m — and the policy
never got close enough to feel the large part. Its gate also multiplied by a placement-quality factor and by the
grip being open past half, both below 1 in most states.

Measured reachability (unchanged): driving the palm toward home_palm_pos with a simple proportional action reaches
within 0.05 m in 9 steps (median); only the palm position is judged, not the joint angles.
