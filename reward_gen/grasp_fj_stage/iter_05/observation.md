The round was ended by the user at about epoch 1750 of 3000 (2.7 h), before the time limit, because the policy had settled into pressing the hand into the table - a behaviour that cannot be used on the real robot. The recording available for this round was made from the epoch-1290 checkpoint, about 460 epochs before the end; the behaviour it shows matches the metrics measured at the end of the round.

What the robot does in the recording:
- The hand reaches the cup and closes on it. The palm comes against the cup's side and all four fingers curl around it. This is a real envelope grasp and it is the first time this track has produced one; in the previous round the fingers stayed extended and only grazed the cup's surface.
- The hand travels in low, along the table surface, and presses down into it while approaching and while grasping.
- The grasp is taken low on the cup, near its base, with the hand coming in from underneath rather than against the middle of the cup's body.
- The little finger stays out of the grasp. The thumb and the other three fingers close on the cup.
- The cup stays upright at its spawn position. It is cradled at a slight angle at times but is never knocked over and is never lifted clear of the table.

Metrics over the last 150 epochs of the round:
- approach_done was set in 0.68 of far-start episodes and 1.00 of near-start episodes; envelope_done in 0.59 of far-start and 0.70 of near-start episodes.
- Episode funnel: reach 0.81, grasp 0.76, envelope 0.48, lift 0.11, success 0.0001.
- The palm touched the cup on 0.55 of steps and the mean palm-to-cup gap was 0.036 m. In the previous round palm contact never left the range 0.001 to 0.013.
- On an average step 2.98 of the five digits touched the cup: thumb 0.76, index 0.73, middle 0.72, ring 0.70, little finger 0.059. Mean contact force 4.1 N.
- The cup rose 0.006 m on average. Its distance to the goal stayed at 0.288 m and was never inside the success tolerance; near_goal was 0.0000 and there were no successes in the last 600 epochs.
- Mean cup tilt 8.8 degrees; the cup was tipped over in 0.0003 of episodes and never fell.
- The hand penetrated the table. The maximum penetration depth grew through the round to 0.028 m, and the lowest hand point reached 0.187 while the table surface is at 0.205. The reward's table term paid -0.0002, so this cost almost nothing.
- Total reward was 1.91 and still rising at the end of the round.
- Around epoch 950 there was a short period with the best lifting of the round: lift in 0.296 of episodes, cup rise 0.019 m, successes 0.0080. In that same period table penetration was at its lowest of the whole round, 0.0012 m. As penetration climbed back to 0.028 m, lifting fell to about a third of that peak and never recovered, while the total reward went on rising.
- Success in this round required only that the cup's keypoints be within 0.1125 m of the goal for ten steps. It did not require that the cup be grasped, lifted, upright, or that the hand stay off the table.
