All numbers below were measured, not estimated. Sources: the training log of the policy trained by the previous reward
(3000 epochs, 8192 environments, 30 % of episodes starting with the shoe already in the hand), and deterministic probes
of its final checkpoint (256 environments, training noise off, first episode of every environment), run once with every
episode starting from the table and once with every episode starting with the shoe in the hand. Training for this round
CONTINUES from that policy's weights. The environment is unchanged except for ONE new context field, `ctx.open_frac`
(see below).

RESULT OF THE PREVIOUS REWARD: NO EPISODE EVER SUCCEEDED, FROM EITHER START. Two separate reasons.

1. THE SHOE IS NEVER PICKED UP FROM THE TABLE.

   Episodes starting from the table (256 of 256):
     the shoe never rose 1 cm above the table in any episode                       256 of 256
     finger links touching the shoe at the same time, most over the episode        1 / 2 / 3 (10th / 50th / 90th percentile) of 15
     closest the shoe came to the target (keypoint_dist)                           0.19 / 0.22 / 0.27 m
   Training log, whole run: the fraction of steps with `ctx.carried` true rose from 0.0004 to 0.0043 and no higher;
   `ctx.held` stayed at 0.0005.
   Frames of a recorded video of this policy show the hand going to the shoe and resting its fingertips on the TOP of the shoe, staying
   there; it does not reach around or under it.
   For comparison, when every episode started with the shoe already in the hand, the same policy's hand touched it with
   6 / 8 / 10 links (10th / 50th / 90th percentile) — that is what a real grasp of this shoe looks like in this hand.

   The previous reward's `grasp` term paid, whenever the palm was within 0.12 m of the shoe: 0.50 x the mean over fingers
   of tanh(force / 2), 0.30 x the thumb's closing (clamped at 0.6 rad), and 0.20 x the mean finger closing times a
   closeness factor. Over the last 100 epochs it averaged 0.546 per step, the second largest term, while nothing was
   ever lifted: closing the fingers on top of the shoe collects most of it.

2. THE HAND NEVER OPENS AFTER THE SHOE IS SET DOWN, SO THE SCRIPTED RETURN NEVER STARTS.

   Episodes starting with the shoe in the hand (256 of 256):
     episodes in which the shoe was placed (keypoint_dist <= 0.03 m) at least once  0.57
     closest the shoe came to the target in those episodes                        0.026 m median
   On the 15 452 steps where the shoe was placed:
     resting on the rack                                                          0.998 of those steps
     still                                                                        0.032
     open fraction as the takeover measures it                                    0.33 median, never >= 0.9
     released (palm_shoe_dist > release_radius)                                   0.005
   `ctx.retracting` was never true in the whole run, so `ctx.home` and `ctx.success` could never become true.

   Why: the takeover starts only when the hand's open fraction reaches 0.9. The previous reward measured openness with
   its own formula built from `hand_target_norm` and a per-joint direction guess; the quantity the takeover actually
   compares was not in the context. It now is: `ctx.open_frac` is EXACTLY that quantity, 0 at the grip pose, 1 at the
   open hand of a reset.

WHAT IS LEARNED AND MUST NOT BE LOST: approaching the shoe from the rest posture, and — once the shoe is in the hand —
carrying it to the rack and setting it within 3 cm of the target on the rack top (0.57 of such episodes).
