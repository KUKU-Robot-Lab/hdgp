All numbers below were measured, not estimated. Source: deterministic rollouts (256 environments, first episode
of each) of the policy trained by the previous reward, plus the training logs of that run.

WHAT THE PREVIOUS REWARD ACHIEVED

The release chain worked. The policy now opens the hand, and it opens it in the right state:

  open_frac on the steps where placed AND resting held      0.97   (the reward before that one: 0.00)
  placed AND resting                                        0.446 of steps, 0.606 of environments
  released                                                  0.700 of steps
  all four conditions at once                               0.197 of steps, 0.539 of environments

So setting the shoe down, letting go, and backing off are all learned behaviours now. Do not rebuild them.

WHERE IT STILL FAILS

1. `still` is the least satisfied of the four conditions, and it is the one that gates the rest:

     placed   0.447 of steps        resting  0.711 of steps
     released 0.700 of steps        still    0.491 of steps

   `still` also fell over training while the other three rose. It was 0.811 at epoch 100 and 0.614 at epoch 250.

2. Final placement is short of the goal. The target sits 0.139 m (centre to centre) from the other shoe on the
   rack. Where the shoe actually came to rest, over 256 episodes:

     10th percentile  0.117 m      median  0.168 m      90th percentile  0.328 m

   The median is 0.029 m farther from the other shoe than the target is. Final keypoint error, same episodes:

     10th percentile  0.020 m      median  0.046 m      90th percentile  0.254 m

   The placement tolerance has since been tightened to 0.03 m, so that median of 0.046 m would now fail. Closing
   the last three centimetres is the main thing this reward has to buy that the previous one did not.

3. Reward mass concentrated away from finishing. In the previous reward's final epoch, the components paid after
   the hand lets go (release, handsfree, withdraw, settle, stability) summed to 5.17 of a total of 8.67 — 59.7%
   — while the success component averaged 0.020, or 0.23% of the total. Those components rose monotonically to
   the end of training while the task's own success metric did not.

4. One component contradicted its own name: `settle` rose from 0.126 to 0.699 over the same epochs in which the
   measured `still` occupancy fell from 0.811 to 0.614. Whatever it was gated on, it was not the shoe coming to
   rest.

WHAT CHANGED IN THE ENVIRONMENT SINCE THAT RUN

The success rule and the tolerance changed; both are already stated in the task description above, and the
context fields carry the new values. Two consequences worth stating plainly:

  - A single bad step no longer resets progress toward success, so the policy can set the shoe down, push it
    to correct its position, and let go again, all inside the same window.
  - 0.03 m is close to the 0.028 m gap the target leaves between the two shoes, so "succeeded" and "looks
    correctly placed" now mean nearly the same thing.
