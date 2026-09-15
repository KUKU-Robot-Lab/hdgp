The round was ended early, at about epoch 500 of 3000, because the hand had stopped interacting with the cup; the playback uses the latest checkpoint and shows three consecutive episodes of one environment.
Episodes 1 and 2 start from the raised pose. The hand moves down beside the robot column near the table edge closest to the robot, turns away from its start orientation with the fingers spread, drifts around there and rises again. It never moves towards the cup and never touches it.
Episode 3 starts beside the cup, with approach_done already set: the fingers point forward along +x, the palm faces the cup's side and the cup is just in front of the hand. Within 0.25 s the fingers are still open and the hand is already pulling back; by 0.7 s it is clearly away from the cup with the fingers spread; later it rises high and moves to the table edge. It never closes the fingers and never touches the cup.
Metrics (up to epoch 498):
- approach_done was set in every near-start episode (as the environment does) and in no far-start episode; envelope_done was never set in either group.
- In near-start episodes, the share in which at least three digits touched the cup at some step rose to 0.60 at epoch 120, was 0.51 at epoch 338 and fell to 0.00 at epoch 498.
- The mean palm-to-band distance grew from 0.19 m at epoch 60 to 0.41 m at epoch 498.
- The total reward was 0.182 at epoch 498, of which grasp_stage_base was 0.178. grasp_palm was 0 at every logged epoch; grasp_contacts fell from 0.016 (epoch 226) to 0.000, grasp_wrap from 0.012 to 0.000 and grip_squeeze stayed at about 0.
- cup_disturb_pen went from -0.018 at epoch 60 to -0.001 at epoch 498; early_contact_pen stayed at about 0.
- approach_reach stayed at about 0.006; the share of steps with the hand in its start orientation fell from 1.00 to 0.29.
