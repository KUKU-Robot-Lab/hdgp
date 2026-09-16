Round 8 (`fj_stage_i07`) ran to epoch 1712 over 3.14 hours and was stopped by the user. It is the round that fixed the failure of the previous one.

**The raised start learned to approach.** This was the single goal of the round and it succeeded. The fraction of far-start episodes that complete the approach was 0.000 for the first three hundred epochs, crossed zero around epoch 330, and reached 0.742 by epoch 800. It settled at 0.625. In the previous round this figure peaked at 0.669 around epoch 530 and then collapsed to 0.001; here it did not collapse.

**The whole chain now works from the raised start.** By the end of the round, far-start episodes reached the cup 0.943 of the time, closed a grasp 0.889, completed the envelope 0.769, and lifted 0.780. The previous round produced no far-start successes at all; this one peaked at 0.370.

**The grasp is a real envelope grasp.** At the moment of success the palm was on the cup 0.914 of the time and 4.35 digits were in contact. Grasp quality at lift was 0.648. The recording confirms it: the palm lies against the side of the cup with four fingers wrapped around the body and the thumb opposing them. This is not a fingertip pinch.

**The cup is carried much higher than before.** Lift height reached 0.208 m against 0.056 m in the previous round, and 0.62 of episodes had the cup airborne.

**The hand stayed off the table for the whole round.** Palm clearance ended at 0.276 m, the palm-on-table fraction was zero, and the episode-ending floor condition never fired.

Three things went wrong, and they are the substance of the next round.

**The round peaked at epoch 1000-1200 and declined for the remaining five hundred epochs.** Far-start success went 0.370, then 0.215, then 0.066. The mean episode success count went 1.257, then 0.655, then 0.200. Meanwhile the total reward kept climbing the whole time — 1.455, 1.387, 1.705, its maximum at the very end. Reward rose while the task went backwards. This is the same signature as round 5, where the total climbed from 0.73 to 0.96 while palm contact fell away.

The cause is visible in the individual terms. Income for raising the cup rose from 0.489 to 0.622 and income for clearing the table from 0.195 to 0.234, while the success bonus fell from 0.0140 to 0.0014 — a tenfold drop — and the term that pays for holding still at the goal never left 0.0001 in more than a thousand epochs. Income for arriving at the goal also fell, from 0.0290 to 0.0139. The policy found that holding the cup up pays better than carrying it to the goal and stopping there, and moved its income accordingly.

**The cup is carried lying over.** The recording shows it held at roughly thirty to forty-five degrees from upright through most of the carry. The logged mean tilt of 9.6 degrees hides this, because it averages every environment and every step including the long stretches before the lift. The reason tilting is free is arithmetic: the tilt penalty was 0.0146 against 0.622 of lift income, a ratio of one to forty-three, and the upright term is added alongside the lift income rather than gating it, so a cup raised on its side still collects the raising income in full.

**The approach posture is abandoned the instant the latch fires.** The approach gate is a latch — six conditions true in a single step, and it stays on for the rest of the episode. The per-step condition rates show what happens after that: not touching the cup fell from 0.968 to 0.240, the along-axis window from 0.217 to 0.046, the plane gap from 0.292 to 0.169. The penalty for touching the cup during the approach grew fifteenfold, from 0.0017 to 0.0250, and the policy paid it willingly. The episode-level latch stayed at 0.625 through all of this, so the funnel never showed it. In the recording the approach itself still looks reasonable — the hand comes in with the fingers extended — but the shape is held for an instant rather than maintained into the grasp.

Two notes for the record. The wrap ratchet never moved: the mean episode success count peaked at 1.697 and never reached the threshold of 2.0, so the tolerance stayed at its starting value of 0.150 for the entire round and the new ceiling of 0.35 introduced this round is still untested. And a maximum hand joint tracking error of 1051 and later 341 appeared in two buckets; the arm velocity maximum spiked with it and the runaway termination fired, so these were isolated diverging environments that were terminated, not a control fault — the value returned to 2.9 afterwards.
