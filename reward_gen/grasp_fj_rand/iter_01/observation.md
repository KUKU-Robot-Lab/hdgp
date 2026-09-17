Round 2 of the random-placement track (`fj_rand_i01`) resumed the weights of round 1 at epoch 800, when the approach was learned but the hand-sinking habit had not yet formed. It ran to epoch 5577 over 9.02 hours. The environment was unchanged. The cup appears anywhere in x 0.10-0.40 m, y -0.30-0.00 m every episode, and every episode starts far from the cup.

**The whole task works under random placement.** Across the last two hundred epochs, far-start episodes completed the approach 0.91 of the time, closed the envelope 0.89, lifted 0.98 and succeeded 0.79. Success first passed 0.9 at epoch 2000 and stayed between 0.90 and 0.94 for more than two thousand epochs before the final tightening.

**The success criterion was tightened to its floor.** Mean episode success crossed 2.0, and the tolerance ran from 0.1125 through 0.054 and 0.026 down to 0.015, which is the floor. Far-start success held at 0.90 down to a tolerance of 0.026. At 0.015 it settled at 0.79-0.86, and mean episode success at 3.67-3.84, just below the loop's exit threshold of 4.0. The fall follows the criterion getting stricter, not a change in behaviour: over the same span lift height rose from 0.122 to 0.174 m, tilt stayed at 4.3 degrees and palm contact at success rose to 0.94.

**The recording shows the intended sequence.** The hand leaves the start pose open, travels to the cup and stops beside it with the palm facing it. By about four seconds it has closed four fingers and the thumb around the cup body with the palm against the side. It then raises the cup well clear of the table, about the height of the arm's shoulder link in the frame, and holds it upright while making small moves for the rest of the episode. The cup is never put back down and never tips. The next episode starts with a new placement.

**The hand-sinking failure from round 1 is gone.** The sink penalty has been zero since epoch 1400. The hand's lowest point rose from 0.25 m to 0.45 m.

**The motion is smooth.** While the cup is held, the 99th percentile of arm joint speed fell from 4.8 rad/s to about 1.1 rad/s, cup speed is 0.15 m/s, the carry penalties are about 0.003 in total, and tipping is 0.0001.

**Stage-by-stage, the approach dipped once and recovered.** When lifting was being learned (epochs 1200-1400), the approach latch fell to 0.39 and recovered to above 0.9 by epoch 2000. It has not dipped since.

Placement rejection behaved as designed: 0.42 of resets resampled, and none failed.

Two things are not ideal but are not failures. The palm-to-cup gap sits at 0.04-0.08 m during carrying. The round could not end by itself because success fell under 4.0 once the tolerance reached its floor. The hour cap is bypassed while success is high, which is the same loop-policy issue recorded for the fixed-placement track.
