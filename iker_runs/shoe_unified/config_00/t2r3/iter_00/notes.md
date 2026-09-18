All numbers below were measured on this environment's two predecessors, not estimated. Sources: deterministic probes
(256 environments, training noise off, first episode of every environment) of the best stage-1 grasp policy and the best
stage-2 placement policy, and the training logs of those runs. This is the FIRST reward for the unified task: one policy
now does both halves, so neither of those policies can be reused and there is no previous reward to improve.

WHAT THE TWO SEPARATE POLICIES ACHIEVED, AND WHERE THEY FAILED

Stage 1 (grasp only, 120-step episodes, started from a pre-grasp pose 8-12 cm above the shoe with the hand open):

  success (the shoe lifted clear and held 20 steps)                       0.344
  the shoe was lifted and the hold conditions held at least once,
      but the 20-step hold never completed before the time limit          0.410
  the hold latched and the shoe then fell back to the table               0.207
  the shoe fell off the table                                             0.016
  lifted, but the hold conditions never held                              0.020
  the palm never came near the shoe                                       0.004

So reaching the shoe is solved and closing on it is solved; KEEPING it is not. 62 % of episodes reach a proper hold at
least once and only 34 % sustain it. In the unified task the shoe must stay in the hand for the whole carry to the rack,
which is far longer than 20 steps, so a grasp that merely touches and lifts is not enough.

Stage 2 (placement only, started with the shoe already in the hand):

  success from the states its own training used                           0.563
  success from the states the stage-1 policy actually hands over          0.285
  final keypoint error when it failed by leaving the shoe on the rack
      off target                                                          0.032 / 0.045 / 0.075 m (10th / 50th / 90th percentile)

The gap between 0.563 and 0.285 is the handover: the states its training bank contained had passed a re-verification of
the grasp, and the ones a real chain produces had not. Chained end to end the two policies finish about 0.12 of episodes.

Two more measured facts about the placement half:
  - Once the shoe is set down within 3 cm of the target and the hand opens, the environment's scripted return of the arm
    works: the arm ends 0.13 degrees from the rest posture and the shoe moves at most 0.9 mm while it withdraws.
  - The placement policy never learned to touch a shoe it had set down off target. Two generated rewards paid for going
    back to it and the terms stayed flat for 200 epochs each, because after letting go the policy parks the hand
    0.25-0.30 m away from the shoe and never returns.

WHAT THIS MEANS FOR THIS REWARD (facts, not instructions)

  - The policy holds the shoe for most of the episode. `ctx.dz_free` only measures a free lift near the pick spot, so it
    reads 0 as soon as the shoe is carried away; `ctx.held`, `ctx.hold_count` and `ctx.latched` are built on it and
    therefore describe the pick-up, not the carry. During the carry the facts available are the contact forces
    (`ctx.link_shoe_force`, `ctx.palm_shoe_force`), `ctx.palm_gap`, `ctx.palm_shoe_dist`, `ctx.slip_speed` and the shoe's
    height.
  - The shoe's position on the table is drawn fresh every episode, so the approach cannot be memorised.
  - A fraction of episodes starts from a recorded later state (shoe already held, or already set down off target);
    `ctx.start_held` marks them. They see the same reward.
