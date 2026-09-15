Training was stopped early, after about 180 of 20,000 epochs, because the hand was not approaching the cup; the playback uses the best checkpoint of that run.
In the viewed environment the hand never moves towards the cup. Within the first 1.5 s the arm lowers the hand beside the robot column, next to the table edge closest to the robot, and turns the hand away from its start orientation: the fingers no longer point forward over the table (+x) but down towards the table, and the wrist keeps rotating there for the rest of the episode. At the end of the episode the hand is still next to the robot, far from the cup.
The fingers and the thumb curl in from the first second, so the hand does not keep its default pose during what should be the approach.
The cup is never touched and stays upright.
Metrics:
- The approach flag (ctx.approach_done) was set in 0 of the finished episodes up to epoch 110.
- The palm-to-band distance averaged 0.26 m at epoch 1, 0.28 m at epoch 17, 0.30 m at epoch 55 and 0.23 m at epoch 110. In 56 % of the finished episodes the palm centre came within 5 cm of the band at some step, but none of them met the approach flag.
- approach_face rose from 0.19 at epoch 1 to 0.36 at epoch 17, while approach_reach fell from 0.111 to 0.098 over the same epochs; at epoch 110 they were 0.37 and 0.16.
- approach_pose_keep was 0.000 at every logged epoch.
- In the start pose the palm normal points along +y and the fingers along +x; the horizontal direction from the palm centre to the cup axis is about 68 degrees away from the palm normal there, so approach_face paid about 0.19 before the hand moved.
