Round 9 (`fj_stage_i08`) ran to epoch 4048 over 6.47 hours and was stopped by the user. It is the round in which the task, as the user defined it, started working from the raised start.

**The whole chain works from the raised start.** Over the last two hundred epochs, far-start episodes completed the approach 0.882 of the time, closed the envelope 0.919, lifted 0.934, and succeeded 0.764. The previous round ended with far-start success at 0.042 after collapsing from a peak of 0.370. This round did not collapse; success rose from zero at epoch 800 to 0.84 by epoch 2900 and held between 0.75 and 0.89 for the remaining twelve hundred epochs.

**The cup is now carried upright.** This was the clearest failure of the previous round, where the recording showed the cup held at thirty to forty-five degrees through most of the carry. Mean tilt ended at 5.45 degrees against 10.44 in the previous round, and the tilt penalty shrank from 0.046 to 0.007 as the policy stopped incurring it. The recording confirms it: in the frames where the cup is off the table it stands close to vertical in the hand, with the palm against its side and four fingers wrapped around the body.

**The grasp is a true envelope grasp and it improved.** At the moment of success the palm was in contact 0.767 of the time and 4.70 digits were touching. Grasp quality rose through the round to 0.439. Lift height reached 0.169 m.

**The reward moved to the end of the task, which is what the round was for.** The terms that pay for arriving at the goal and settling there went from nothing to real income: goal proximity 0.611, goal closeness 0.387, and the settling term 0.068. In the previous round that settling term never left 0.0001 across 1712 epochs; here it is 678 times larger. The success bonus ended at 0.077 against 0.0009.

**The hand never touched the table.** Palm clearance ended at 0.253 m, the palm-on-table fraction was zero from epoch 1000 onward, and the floor termination never fired.

**The wrap ratchet was tested and held.** Mean episode success crossed the threshold of 2.0 for the first time in this track, the tolerance ratchet ran 0.150 → 0.192 → 0.288 → 0.350, and it stopped exactly at the ceiling of 0.350 and stayed there for the last two thousand epochs. Two rounds ago the same ratchet ran to 0.689 and locked success out.

Four things are wrong, and they are the substance of the next round.

**The cup is set back down and picked up again within one episode.** The recording shows the cup airborne and upright in several frames and back on the table in others, with the hand beside or above it, across what appears to be a single episode window. The metrics are consistent with this: the airborne fraction ended at 0.692 while success is 0.764 and the settling term is only 0.068, so episodes are spending a large part of their time holding or re-approaching rather than finishing. Carrying the cup to the goal once and stopping should be worth more than a sequence of lifts.

**The approach posture is still abandoned the moment the latch fires, and worse than before.** The approach gate is satisfied for a single step and then the policy drives into the cup. Not touching the cup during the approach fell from 0.982 to 0.258 over the round, and the open-pose condition fell from a mid-round 0.371 to 0.063 by the end. Both are lower than the previous round finished at. The episode-level latch stays at 0.880 through all of this because it only needs one step, so the funnel never shows it. The new pre-grasp term introduced this round pays only 0.0009, which is too little to hold the shape against the grasp income waiting on the other side.

**Stage-1 income decays as the round goes on.** Travel income fell from 0.022 at epoch 1800 to 0.007 at the end, and the form term from 0.022 to 0.012. The approach still latches, so this is not yet breaking anything, but the trend is the same one that preceded the previous round's approach collapse.

**The round could not end by itself.** The two exit conditions — mean success at or above 4.0 and tolerance at or below 0.03 — moved in opposite directions: as the tolerance curriculum tightened, success fell, and the pair was never satisfied at the same time. The round ran 6.47 hours against a 4.0 hour cap because the success branch in the round policy returns before the hour check is reached. This is a loop-policy problem, not a reward problem, but it cost two hours of GPU time and needs to be recorded.

One note for the record. Tolerance tightened from 0.1125 to 0.0273 over the round, a 76 percent reduction, and the far-start success rate rose across the same span. Falls in the success count late in the round track the tightening criterion, not a regression in behaviour: grasp quality, lift height, airborne fraction and palm contact at success all rose while the success count fell.
