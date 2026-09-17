Keep what this round fixed.

- The approach from the raised start is the best it has been: 0.93 of far-start episodes complete it, the open pose and the no-contact condition both hold through the approach, and travel income no longer decays over training. Whatever produced that should be kept.
- Round 9's working structure after the grasp was not tested this round, because nothing was lifted. Do not treat its absence of signal as evidence against it. Round 9 carried the cup upright, reached the goal and held there with far-start success around 0.76; that downstream structure is the reference.

What has to change.

**Hovering open next to the cup must stop paying more than closing on it.** This is the only failure of this round and it blocked everything after it. The term that pays for holding the pre-grasp shape pays only while the palm has not arrived, and it grew to 0.136, the second-largest income. The grasp-pose and palm-reach terms added another 0.275 without needing any contact. Together that is 84 percent of the total, all earned by standing about 6 cm from the cup with the hand open. Everything that requires touching the cup earned under 0.03. The policy found that staying just short of the cup is the best place to be, and stayed there for 1800 epochs.

The pre-grasp shape should be worth holding only on the way in. Once the approach has been completed it should stop paying altogether, not keep paying for as long as the palm stays away. Income that does not require contact — hand pose and palm position relative to the cup — should be a small fraction of what closing on the cup earns, so that the move from hovering to touching is always a clear step up. The first real contact of the palm on the cup, with the hand still in shape, should be the largest single jump in reward available at that stage, and curling the digits around the cup from there should add more on top.

**Do not give back the approach to get the grasp.** Last round the approach shape was abandoned the moment the latch fired and the hand dived in; this round it was held but never let go of. The target is between the two: hold the shape until the palm reaches the cup, then close. A penalty for touching early is fine, but it must only apply before the palm has arrived, and the reward for arriving must outweigh everything the hand can earn by waiting.
